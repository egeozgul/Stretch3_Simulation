#!/usr/bin/env python3
"""ROS 2 node for Stretch 3 MuJoCo simulation."""

import os
import time
import threading
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64MultiArray, String, Bool
import mujoco
import mujoco.viewer
import numpy as np
import cv2
from ik import IKSolver
from navigation import NavigationController
from anchor_utils import load_anchors_from_xml

# Joint configuration
JOINT_NAMES = [
    'joint_lift', 'joint_arm_l0', 'joint_arm_l1', 'joint_arm_l2', 'joint_arm_l3',
    'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll', 'joint_head_pan', 'joint_head_tilt'
]

JOINT_QPOS_MAP = {
    'joint_lift': 7, 'joint_arm_l0': 8, 'joint_arm_l1': 9,
    'joint_arm_l2': 10, 'joint_arm_l3': 11
}

JOINT_LIMITS = {
    'lift': (0.0, 1.1), 'arm_extend': (0.0, 0.52), 'wrist_yaw': (-1.39, 4.42),
    'wrist_pitch': (-1.57, 0.56), 'wrist_roll': (-3.14, 3.14),
    'gripper': (-0.02, 0.04), 'head_pan': (-4.04, 1.73), 'head_tilt': (-1.53, 0.79)
}

ACTUATOR_NAMES = ['left_wheel_vel', 'right_wheel_vel', 'lift', 'arm', 'wrist_yaw',
                  'wrist_pitch', 'wrist_roll', 'gripper', 'head_pan', 'head_tilt']

JOINT_COMMAND_MAP = [
    ('lift', 'lift'), ('arm_extend', 'arm'), ('wrist_yaw', 'wrist_yaw'),
    ('wrist_pitch', 'wrist_pitch'), ('wrist_roll', 'wrist_roll'),
    ('gripper', 'gripper'), ('head_pan', 'head_pan'), ('head_tilt', 'head_tilt')
]

RESET_POSITIONS = {
    'lift': 0.6,
    'arm_extend': 0.0,
    'wrist_yaw': 0.0,
    'wrist_pitch': 0.0,
    'wrist_roll': 0.0,
    'gripper': 0.04
}

# Timing constants
PUB_RATE = 30.0  # Hz
RENDER_RATE = 20.0  # Hz
RESET_SPEED = 0.005  # Reduced for smoother movement
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
DEFAULT_SPEED = 50.0
JOINT_TOLERANCE = 0.001
ZERO_PLACEHOLDER_THRESHOLD = 0.05
SMOOTHING_FACTOR = 0.15  # Exponential smoothing factor (0-1, lower = smoother)


class StretchSimNode(Node):
    """ROS 2 node that runs MuJoCo simulation and handles ROS 2 communication."""
    
    def __init__(self):
        super().__init__('stretch_sim')
        
        xml_path = os.path.join(os.path.dirname(__file__), 'table_world.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._alignment_cancel = False
        self._alignment_active = False 
        self.anchors = self._load_anchors(xml_path)
        self._init_robot_state()
        self._init_actuators()
        self._init_camera()
        self._setup_ros2()
        
        self.nav_controller = NavigationController()
        self.manual_control = False
        self.running = True
        self.sim_thread = None
        self._nav_log_counter = 0
        self._resetting_arm = False
        self._reset_targets = {}
        self._joint_targets = {}
        self._joint_speed_percent = {}
        self._base_joint_speed = 0.005  # Reduced for smoother movement
        self._joint_velocities = {}  # Track velocities for smooth acceleration
        self.ik_solver=IKSolver(self.model,self.data,logger=self.get_logger())
        self.get_logger().info('Stretch 3 ROS 2 Simulation Node started')
    
    def _load_anchors(self, xml_path):
        """Load anchors from XML file."""
        try:
            anchors = load_anchors_from_xml(xml_path)
            self.get_logger().info(f'Loaded {len(anchors)} anchors: {sorted(anchors.keys())}')
            for letter, data in sorted(anchors.items()):
                self.get_logger().info(f'  {letter}: {data}')
            return anchors
        except Exception as e:
            self.get_logger().error(f'Failed to load anchors: {e}')
            return {}
    
    def _init_robot_state(self):
        """Initialize robot to default state."""
        self.data.qpos[7] = 0.0  # Lift at bottom
        self.data.qpos[8:12] = 0.0  # Arm retracted
        mujoco.mj_forward(self.model, self.data)
    
    def _init_actuators(self):
        """Initialize actuator and joint mappings."""
        self.actuator_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ACTUATOR_NAMES
        }
        self.ctrl_state = {name: 0.0 for name in ACTUATOR_NAMES}
        
        self.joint_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
        }
        
        self.base_link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    
    def _init_camera(self):
        """Initialize camera if available."""
        # Try Stretch 3 cameras in order of preference
        camera_names = ['d435i_camera_rgb', 'd405_rgb', 'nav_camera_rgb']
        self.camera_id = None
        self.camera_name = None
        
        for cam_name in camera_names:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id >= 0:
                self.camera_id = cam_id
                self.camera_name = cam_name
                self.get_logger().info(f'Camera "{cam_name}" found (ID: {self.camera_id})')
                break
        
        if self.camera_id is None:
            self.get_logger().warn('No Stretch 3 camera found, camera display disabled')
            self.camera_renderer = None
            self.camera_obj = None
        else:
            self.camera_renderer = None
            self.camera_obj = None
    
    def _setup_ros2(self):
        """Setup ROS 2 publishers and subscribers."""
        self.create_subscription(Twist, '/stretch/cmd_vel', self._cmd_vel_callback, 10)
        self.create_subscription(Float64MultiArray, '/stretch/joint_command', 
                                self._joint_command_callback, 10)
        self.create_subscription(Float64MultiArray, '/stretch/joint_commands', 
                                self._joint_commands_callback, 10)  # Legacy
        self.create_subscription(String, '/stretch/navigate_to_anchor', 
                                self._navigate_to_anchor_callback, 10)
        self.create_subscription(String, '/stretch/turn_towards_anchor', 
                                self._turn_towards_anchor_callback, 10)
        self.create_subscription(String, '/stretch/reset_arm', 
                                self._reset_arm_callback, 10)
        self.create_subscription(Float64MultiArray, '/stretch/navigate_to_position', 
                                self._navigate_to_position_callback, 10)
        self.create_subscription(String, '/stretch/align_with_target',
                            self._align_with_target_callback, 10)
        self.create_subscription(String, '/stretch/compute_ik',  # NEW
                            self._compute_ik_callback, 10)
        self.joint_state_pub = self.create_publisher(JointState, '/stretch/joint_states', 10)
        self.nav_status_pub = self.create_publisher(Bool, '/stretch/navigation_active', 10)
        self.camera_pub = self.create_publisher(Image, '/stretch/camera/image_raw', 10)
        self.ik_result_pub = self.create_publisher(Float64MultiArray, '/stretch/ik_result', 10)
    @staticmethod
    def _clamp_speed(speed_percent):
        """Clamp speed percentage to valid range."""
        return max(0.0, min(100.0, float(speed_percent)))
    
    def _set_joint_target(self, actuator_name, value, speed_percent):
        """Set target for a joint actuator."""
        if actuator_name not in self.ctrl_state:
            return False
        
        # Find corresponding cmd_name for limits
        cmd_name = next((cmd for cmd, act in JOINT_COMMAND_MAP if act == actuator_name), None)
        if cmd_name:
            min_val, max_val = JOINT_LIMITS[cmd_name]
            value = np.clip(value, min_val, max_val)
        
        self._joint_targets[actuator_name] = value
        self._joint_speed_percent[actuator_name] = speed_percent
        return True
    #
    def _compute_ik_callback(self, msg):
        """Handle IK computation request for a target object."""
        target_name = msg.data.strip()
        
        self.get_logger().info(f'Computing IK for target: {target_name}')
        
        # Get target object position from site
        try:
            target_site_id = mujoco.mj_name2id(
                self.model, 
                mujoco.mjtObj.mjOBJ_SITE, 
                f'{target_name}_site'
            )
            
            if target_site_id < 0:
                self.get_logger().error(f'Target site {target_name}_site not found')
                # Publish failure
                result_msg = Float64MultiArray()
                result_msg.data = [0.0, 0.0, 0.0, 0.0]  # 0.0 = failure
                self.ik_result_pub.publish(result_msg)
                return
            
            # Get target position from site
            target_pos = self.data.site_xpos[target_site_id].copy()
            
            self.get_logger().info(f'Target position: {target_pos}')
            
            # Compute IK using IKSolver
            success, joint_solution = self.ik_solver.compute_ik(target_pos)
            
            if success:
                # Extract joint values
                # joint_solution = [lift, arm_l3, arm_l2, arm_l1, arm_l0, wrist_yaw]
                q_lift = joint_solution[0]
                q_arm = sum(joint_solution[1:5])  # Sum of all arm segments
                q_wrist = joint_solution[5]
                
                self.get_logger().info(
                    f'IK solution: lift={q_lift:.3f}, arm={q_arm:.3f}, wrist={q_wrist:.3f}'
                )
                
                # Publish result: [lift, arm_extend, wrist_yaw, success_flag]
                result_msg = Float64MultiArray()
                result_msg.data = [float(q_lift), float(q_arm), float(q_wrist), 1.0]
                self.ik_result_pub.publish(result_msg)
            else:
                self.get_logger().error('IK computation failed')
                # Publish failure
                result_msg = Float64MultiArray()
                result_msg.data = [0.0, 0.0, 0.0, 0.0]
                self.ik_result_pub.publish(result_msg)
                
        except Exception as e:
            self.get_logger().error(f'IK computation error: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            # Publish failure
            result_msg = Float64MultiArray()
            result_msg.data = [0.0, 0.0, 0.0, 0.0]
            self.ik_result_pub.publish(result_msg)
    #
    def _joint_command_callback(self, msg):
        """Handle single joint position command: [joint_index, value, speed_percent]."""
        if self._resetting_arm or len(msg.data) < 2:
            return
        
        joint_index = int(msg.data[0])
        value = float(msg.data[1])
        speed_percent = self._clamp_speed(msg.data[2] if len(msg.data) > 2 else DEFAULT_SPEED)
        
        if not (0 <= joint_index < len(JOINT_COMMAND_MAP)):
            self.get_logger().warn(f'Invalid joint index: {joint_index}')
            return
        
        _, actuator_name = JOINT_COMMAND_MAP[joint_index]
        self._set_joint_target(actuator_name, value, speed_percent)
    
    def _joint_commands_callback(self, msg):
        """Handle multiple joint commands (legacy): [lift, arm_extend, ..., speed_percent]."""
        if self._resetting_arm or len(msg.data) < len(JOINT_COMMAND_MAP):
            return
        
        speed_percent = self._clamp_speed(
            msg.data[-1] if len(msg.data) > len(JOINT_COMMAND_MAP) else DEFAULT_SPEED
        )
        
        for i, ((cmd_name, actuator_name), value) in enumerate(
            zip(JOINT_COMMAND_MAP, msg.data[:len(JOINT_COMMAND_MAP)])
        ):
            if actuator_name not in self.ctrl_state:
                continue
            
            min_val, max_val = JOINT_LIMITS[cmd_name]
            target_value = np.clip(value, min_val, max_val)
            current_value = self.ctrl_state[actuator_name]
            
            # Skip zero placeholders for joints with negative minimum
            if (target_value == 0.0 and abs(current_value) > ZERO_PLACEHOLDER_THRESHOLD 
                and min_val < 0.0):
                continue
            
            self._set_joint_target(actuator_name, target_value, speed_percent)
    
    def _reset_arm_callback(self, msg):
        """Handle arm reset command: "reset" or "reset:speed_percent"."""
        data = msg.data.strip()
        if not data.lower().startswith('reset'):
            return
        
        speed_percent = DEFAULT_SPEED
        if ':' in data:
            try:
                speed_percent = self._clamp_speed(data.split(':', 1)[1])
            except ValueError:
                self.get_logger().warn(f'Invalid speed in reset command: {data}')
        
        self.get_logger().info(f'Starting arm reset (speed: {speed_percent}%)')
        self._resetting_arm = True
        self._reset_speed_percent = speed_percent
        self._reset_targets = {
            'lift': RESET_POSITIONS['lift'],
            'arm': RESET_POSITIONS['arm_extend'],
            'wrist_yaw': RESET_POSITIONS['wrist_yaw'],
            'wrist_pitch': RESET_POSITIONS['wrist_pitch'],
            'wrist_roll': RESET_POSITIONS['wrist_roll'],
            'gripper': RESET_POSITIONS['gripper']
        }
    
    def _update_arm_reset(self):
        """Smoothly move arm to reset positions with improved interpolation."""
        if not self._resetting_arm:
            return
        
        speed_multiplier = getattr(self, '_reset_speed_percent', DEFAULT_SPEED) / DEFAULT_SPEED
        dt = self.model.opt.timestep
        max_speed = RESET_SPEED * speed_multiplier / dt  # Convert to velocity per timestep
        all_reached = True
        
        for actuator_name, target in self._reset_targets.items():
            if actuator_name not in self.ctrl_state:
                continue
            
            current = self.ctrl_state[actuator_name]
            diff = target - current
            
            if abs(diff) < JOINT_TOLERANCE:
                self.ctrl_state[actuator_name] = target
                continue
            
            # Use smooth velocity-based movement
            desired_velocity = np.clip(diff * 2.0, -max_speed, max_speed)
            
            # Initialize velocity if needed
            if actuator_name not in self._joint_velocities:
                self._joint_velocities[actuator_name] = 0.0
            
            # Smooth velocity changes
            current_velocity = self._joint_velocities[actuator_name]
            smoothing = SMOOTHING_FACTOR * speed_multiplier
            new_velocity = current_velocity + smoothing * (desired_velocity - current_velocity)
            self._joint_velocities[actuator_name] = new_velocity
            
            # Apply velocity
            new_position = current + new_velocity * dt
            
            # Check if reached target
            if (diff > 0 and new_position >= target) or (diff < 0 and new_position <= target):
                self.ctrl_state[actuator_name] = target
                self._joint_velocities.pop(actuator_name, None)
            else:
                self.ctrl_state[actuator_name] = new_position
                all_reached = False
        
        if all_reached:
            self._resetting_arm = False
            self.get_logger().info('✓ Arm reset complete')
        #
    def _align_with_target_callback(self, msg):
        """Handle alignment with target object command."""
        data = msg.data.strip()
        delta_angle = 5.0
        
        if ':' in data:
            parts = data.split(':')
            target_name = parts[0].strip()
            try:
                if len(parts) >= 2:
                    delta_angle = float(parts[1].strip())
            except (ValueError, IndexError):
                self.get_logger().warn(f'Invalid delta_angle in align command: {data}, using default')
        else:
            target_name = data.strip()
        
        # Cancel any previous alignment
        self._alignment_cancel = True
        time.sleep(0.1)
        
        # Start new alignment
        self._alignment_cancel = False
        self._alignment_active = True
        threading.Thread(
            target=self._do_alignment,
            args=(target_name, delta_angle),
            daemon=True
        ).start()

    def _do_alignment(self, target_name, delta_angle):
        """Do the actual alignment (runs in separate thread)."""
        try:
            self.get_logger().info(f'Aligning with target: {target_name}')
            
            max_iterations = 10  # ← ADD THIS SAFETY LIMIT
            
            for iteration in range(max_iterations):  # ← CHANGE TO FOR LOOP
                if self._alignment_cancel:
                    self.get_logger().info('Alignment cancelled')
                    return
                
                pos, quat = self._get_robot_pose()
                
                try:
                    yaw_diff, current_yaw, desired_yaw = self.ik_solver.align_with_target(
                        pos, quat, tomato_name=target_name
                    )
                    
                    yaw_diff_deg = abs(math.degrees(yaw_diff))
                    
                    # Check if aligned
                    if yaw_diff_deg <= delta_angle:
                        self.get_logger().info(f'✓ Aligned! Error: {yaw_diff_deg:.2f}° (iteration {iteration+1})')
                        return
                    
                    self.get_logger().info(f'Iteration {iteration+1}: Error {yaw_diff_deg:.1f}°, correcting...')
                    
                    # Rotate to correct orientation
                    self.nav_controller.set_target(
                        pos.copy(),
                        desired_yaw + 1.5708,
                        position_tolerance=0.01,
                        direction_tolerance=math.radians(delta_angle)
                    )
                    self.manual_control = False
                    
                    # Wait for rotation to complete
                    timeout = 10.0
                    start_time = time.time()
                    while self.nav_controller.is_active() and not self._alignment_cancel:
                        if time.time() - start_time > timeout:
                            self.get_logger().warn('Alignment timeout')
                            return
                        time.sleep(0.05)
                    
                    if self._alignment_cancel:
                        self.get_logger().info('Alignment cancelled')
                        return
                    
                    # Small delay for stabilization
                    time.sleep(0.2)
                    
                except Exception as e:
                    self.get_logger().error(f'Alignment failed: {e}')
                    return
            
            # Max iterations reached
            pos, quat = self._get_robot_pose()
            yaw_diff, _, _ = self.ik_solver.align_with_target(pos, quat, tomato_name=target_name)
            final_error = abs(math.degrees(yaw_diff))
            self.get_logger().warn(f'Max iterations ({max_iterations}) reached. Final error: {final_error:.2f}°')
        
        finally:
            self._alignment_active = False
            self.get_logger().info('Alignment thread finished')
    def _update_joint_movements(self):
        """Gradually move joints towards their target positions with smooth interpolation."""
        if not self._joint_targets:
            return
        
        # Get simulation timestep for smooth movement
        dt = self.model.opt.timestep
        
        for actuator_name, target in list(self._joint_targets.items()):
            if actuator_name not in self.ctrl_state:
                continue
            
            current = self.ctrl_state[actuator_name]
            diff = target - current
            
            if abs(diff) < JOINT_TOLERANCE:
                self.ctrl_state[actuator_name] = target
                self._joint_targets.pop(actuator_name, None)
                self._joint_speed_percent.pop(actuator_name, None)
                self._joint_velocities.pop(actuator_name, None)
                continue
            
            # Get speed multiplier
            speed_multiplier = self._joint_speed_percent.get(actuator_name, DEFAULT_SPEED) / DEFAULT_SPEED
            
            # Calculate desired velocity based on distance and max speed
            max_speed = self._base_joint_speed * speed_multiplier / dt  # Convert to velocity per timestep
            
            # Use exponential smoothing for velocity to avoid sudden changes
            if actuator_name not in self._joint_velocities:
                self._joint_velocities[actuator_name] = 0.0
            
            # Calculate desired velocity (proportional to distance, capped at max_speed)
            desired_velocity = np.clip(diff * 2.0, -max_speed, max_speed)
            
            # Smooth velocity changes using exponential smoothing
            current_velocity = self._joint_velocities[actuator_name]
            smoothing = SMOOTHING_FACTOR * speed_multiplier  # Adjust smoothing based on speed
            new_velocity = current_velocity + smoothing * (desired_velocity - current_velocity)
            self._joint_velocities[actuator_name] = new_velocity
            
            # Apply velocity to position
            new_position = current + new_velocity * dt
            
            # Check if we've reached or overshot the target
            if (diff > 0 and new_position >= target) or (diff < 0 and new_position <= target):
                self.ctrl_state[actuator_name] = target
                self._joint_targets.pop(actuator_name, None)
                self._joint_speed_percent.pop(actuator_name, None)
                self._joint_velocities.pop(actuator_name, None)
            else:
                self.ctrl_state[actuator_name] = new_position
    
    def _handle_anchor_command(self, anchor_key, turn_only=False, delta_angle=None, 
                               position_tolerance=None, target_angle_degrees=None):
        """
        Handle anchor navigation command.
        
        Args:
            anchor_key: Anchor identifier (e.g., "ORIGIN", "A", "B") or None if target_angle_degrees is used
            turn_only: If True, only turn towards anchor/angle without moving
            delta_angle: Optional delta angle in degrees for turn-only mode (default: 5.0)
            position_tolerance: Optional position tolerance in meters (default: 0.15)
            target_angle_degrees: Optional absolute target angle in degrees (0-360) for turn_towards
        """
        if turn_only:
            if target_angle_degrees is not None:
                # Absolute angle mode
                self.nav_controller.set_turn_only_target(None, delta_angle, target_angle_degrees)
                delta_value = delta_angle if delta_angle is not None else 5.0
                self.get_logger().info(f'Turning to absolute angle {target_angle_degrees:.1f}° (delta_angle={delta_value:.1f}°)')
            else:
                # Position-based mode
                anchor_key = anchor_key.strip().upper()
                if anchor_key not in self.anchors:
                    available = ', '.join(sorted(self.anchors.keys())) if self.anchors else 'none'
                    self.get_logger().warn(f'Unknown anchor: {anchor_key}. Available: {available}')
                    return
                anchor_data = self.anchors[anchor_key]
                target_pos = anchor_data['pos']
                current_pos, _ = self._get_robot_pose()
                distance = np.linalg.norm(np.array(target_pos[:2]) - current_pos[:2])
                self.nav_controller.set_turn_only_target(target_pos, delta_angle)
                delta_value = delta_angle if delta_angle is not None else 5.0
                self.get_logger().info(f'Turning towards anchor {anchor_key} (distance: {distance:.2f}m, delta_angle={delta_value:.1f}°)')
        else:
            # go_to_anchor: no direction alignment
            anchor_key = anchor_key.strip().upper()
            if anchor_key not in self.anchors:
                available = ', '.join(sorted(self.anchors.keys())) if self.anchors else 'none'
                self.get_logger().warn(f'Unknown anchor: {anchor_key}. Available: {available}')
                return
            anchor_data = self.anchors[anchor_key]
            target_pos = anchor_data['pos']
            current_pos, _ = self._get_robot_pose()
            distance = np.linalg.norm(np.array(target_pos[:2]) - current_pos[:2])
            # go_to_anchor no longer does direction alignment
            self.nav_controller.set_target(target_pos, None, position_tolerance, None)
            pos_tol_info = f", pos_tol={position_tolerance:.2f}m" if position_tolerance else ""
            self.get_logger().info(f'Navigating to anchor {anchor_key} (distance: {distance:.2f}m{pos_tol_info})')
        
        self.manual_control = False
    
    def _navigate_to_anchor_callback(self, msg):
        """Handle navigate to anchor command. Format: "ANCHOR" or "ANCHOR:pos_tol"."""
        data = msg.data.strip()
        position_tolerance = None
        
        if ':' in data:
            parts = data.split(':')
            anchor_key = parts[0].strip().upper()
            try:
                if len(parts) >= 2:
                    position_tolerance = float(parts[1].strip())
            except (ValueError, IndexError):
                self.get_logger().warn(f'Invalid position_tolerance in go_to_anchor command: {data}, using default')
        else:
            anchor_key = data.strip().upper()
        
        self._handle_anchor_command(anchor_key, turn_only=False, 
                                   position_tolerance=position_tolerance)
    
    def _turn_towards_anchor_callback(self, msg):
        """Handle turn towards anchor/angle command. Format: "ANCHOR:delta_angle" or "degrees:target_angle:delta_angle"."""
        data = msg.data.strip()
        delta_angle = None
        target_angle_degrees = None
        anchor_key = None
        
        if ':' in data:
            parts = data.split(':')
            first_part = parts[0].strip().upper()
            
            if first_part == "DEGREES" or first_part.replace('.', '').replace('-', '').isdigit():
                # Absolute angle mode: "degrees:target_angle:delta_angle" or "target_angle:delta_angle"
                try:
                    if first_part == "DEGREES":
                        if len(parts) >= 2:
                            target_angle_degrees = float(parts[1].strip())
                        if len(parts) >= 3:
                            delta_angle = float(parts[2].strip())
                    else:
                        # Assume format: "target_angle:delta_angle"
                        target_angle_degrees = float(parts[0].strip())
                        if len(parts) >= 2:
                            delta_angle = float(parts[1].strip())
                except (ValueError, IndexError):
                    self.get_logger().warn(f'Invalid angle values in turn_towards command: {data}, using defaults')
            else:
                # Position-based mode: "ANCHOR:delta_angle"
                anchor_key = first_part
                try:
                    if len(parts) >= 2:
                        delta_angle = float(parts[1].strip())
                except (ValueError, IndexError):
                    self.get_logger().warn(f'Invalid delta_angle in turn_towards command: {data}, using default')
        else:
            # Simple format: just anchor name
            anchor_key = data.strip().upper()
        
        self._handle_anchor_command(anchor_key, turn_only=True, delta_angle=delta_angle,
                                   target_angle_degrees=target_angle_degrees)
    
    def _navigate_to_position_callback(self, msg):
        """Handle navigate to position command: [x, y, direction?, speed_percent?]."""
        if len(msg.data) < 2:
            self.get_logger().warn('Position command requires at least x and y coordinates')
            return
        
        x, y = float(msg.data[0]), float(msg.data[1])
        target_pos = [x, y, 0.0]
        target_direction = float(msg.data[2]) if len(msg.data) > 2 else None
        speed_percent = self._clamp_speed(msg.data[3] if len(msg.data) > 3 else DEFAULT_SPEED)
        
        current_pos, _ = self._get_robot_pose()
        distance = np.linalg.norm(np.array(target_pos[:2]) - current_pos[:2])
        
        self.nav_controller.set_target(target_pos, target_direction)
        self.manual_control = False
        
        direction_info = f", direction={math.degrees(target_direction):.1f}°" if target_direction else ""
        speed_info = f", speed={speed_percent}%" if speed_percent != DEFAULT_SPEED else ""
        self.get_logger().info(f'Navigating to ({x}, {y}) (distance: {distance:.2f}m{direction_info}{speed_info})')
    
    def _cmd_vel_callback(self, msg):
        """Handle base velocity commands."""
        has_manual_input = abs(msg.linear.x) > 0.05 or abs(msg.angular.z) > 0.05
        
        if has_manual_input and self.nav_controller.is_active():
            self.manual_control = True
            self.nav_controller.cancel()
            self.get_logger().info('Manual control override')
        
        # Convert linear and angular velocities to left/right wheel angular velocities
        # For differential drive: 
        #   v_left_linear = v - (ω * wheel_base/2)
        #   v_right_linear = v + (ω * wheel_base/2)
        # Then convert to wheel angular velocities: ω_wheel = v_linear / wheel_radius
        # MuJoCo velocity actuators: control value directly sets joint angular velocity (rad/s)
        # ctrlrange="-6 6" means control values should be in [-6, 6] rad/s range
        
        wheel_base = 0.3407  # meters (distance between wheels)
        wheel_radius = 0.05  # meters (from XML: size=".05 .0125")
        
        linear_x = msg.linear.x
        angular_z = msg.angular.z
        
        # Calculate linear velocities for each wheel (m/s)
        v_left_linear = linear_x - (angular_z * wheel_base / 2.0)
        v_right_linear = linear_x + (angular_z * wheel_base / 2.0)
        
        # Convert to wheel angular velocities (rad/s)
        # Clamp to ctrlrange [-6, 6] rad/s to avoid actuator saturation
        omega_left = np.clip(v_left_linear / wheel_radius, -6.0, 6.0)
        omega_right = np.clip(v_right_linear / wheel_radius, -6.0, 6.0)
        
        self.ctrl_state['left_wheel_vel'] = omega_left
        self.ctrl_state['right_wheel_vel'] = omega_right
    
    def _get_robot_pose(self):
        """Get current robot position and orientation."""
        return self.data.qpos[0:3].copy(), self.data.qpos[3:7].copy()
    
    def _log_navigation_status(self, pos, quat, linear_vel, angular_vel):
        """Log navigation status for debugging."""
        target = self.nav_controller.target_pos
        if target is None:
            return
        
        distance = np.linalg.norm(target - pos[:2])
        diff = target - pos[:2]
        desired_angle = np.arctan2(diff[1], diff[0])
        
        w, x, y, z = quat
        current_yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        angle_error = np.arctan2(
            np.sin(desired_angle - current_yaw),
            np.cos(desired_angle - current_yaw)
        )
        angle_error_deg = np.degrees(abs(angle_error))
        aligned = angle_error_deg <= 10.0
        
        # Calculate average wheel velocity for logging
        avg_wheel_vel = (self.ctrl_state.get("left_wheel_vel", 0.0) + 
                        self.ctrl_state.get("right_wheel_vel", 0.0)) / 2.0
        self.get_logger().info(
            f'Nav: dist={distance:.2f}m, lin={linear_vel:.2f} m/s, '
            f'ctrl={avg_wheel_vel:.3f}, ang={angular_vel:.2f}, '
            f'err={angle_error_deg:.1f}°, aligned={aligned}'
        )
    
    def _update_navigation(self):
        """Update navigation controller and apply its commands."""
        nav_status = Bool()
        nav_status.data = (self.nav_controller.is_active() or self._alignment_active) and not self.manual_control
        self.nav_status_pub.publish(nav_status)
        
        if not self.nav_controller.is_active() or self.manual_control:
            return
        
        pos, quat = self._get_robot_pose()
        linear_vel, angular_vel = self.nav_controller.get_control(pos, quat)
        
        self._nav_log_counter += 1
        if self._nav_log_counter % (10 if abs(linear_vel) > 0.01 else 50) == 0:
            self._log_navigation_status(pos, quat, linear_vel, angular_vel)
        
        if abs(linear_vel) < 0.001 and abs(angular_vel) > 0.001:
            linear_vel = 0.0
        
        # Convert linear and angular velocities to left/right wheel angular velocities
        # For differential drive: 
        #   v_left_linear = v - (ω * wheel_base/2)
        #   v_right_linear = v + (ω * wheel_base/2)
        # Then convert to wheel angular velocities: ω_wheel = v_linear / wheel_radius
        # MuJoCo velocity actuators: control value directly sets joint angular velocity (rad/s)
        # ctrlrange="-6 6" means control values should be in [-6, 6] rad/s range
        # gear="3" in XML applies to the actuator transmission, control is still in rad/s
        
        wheel_base = 0.3407  # meters (distance between wheels)
        wheel_radius = 0.05  # meters (from XML: size=".05 .0125")
        
        # Calculate linear velocities for each wheel (m/s)
        v_left_linear = linear_vel - (angular_vel * wheel_base / 2.0)
        v_right_linear = linear_vel + (angular_vel * wheel_base / 2.0)
        
        # Convert to wheel angular velocities (rad/s)
        # Clamp to ctrlrange [-6, 6] rad/s to avoid actuator saturation
        omega_left = np.clip(v_left_linear / wheel_radius, -6.0, 6.0)
        omega_right = np.clip(v_right_linear / wheel_radius, -6.0, 6.0)
        
        self.ctrl_state['left_wheel_vel'] = omega_left
        self.ctrl_state['right_wheel_vel'] = omega_right
        
        if self.nav_controller.has_reached():
            self.get_logger().info('✓ Reached target position!')
            self.ctrl_state['left_wheel_vel'] = 0.0
            self.ctrl_state['right_wheel_vel'] = 0.0
    
    def _get_joint_state(self, joint_name):
        """Get position and velocity for a joint."""
        if joint_name in JOINT_QPOS_MAP:
            idx = JOINT_QPOS_MAP[joint_name]
            if idx < len(self.data.qpos):
                return (
                    float(self.data.qpos[idx]),
                    float(self.data.qvel[idx]) if idx < len(self.data.qvel) else 0.0
                )
        
        if joint_name in self.joint_ids:
            try:
                joint_id = self.joint_ids[joint_name]
                qpos_addr = self.model.jnt_qposadr[joint_id]
                qvel_addr = self.model.jnt_dofadr[joint_id]
                pos = self.data.qpos[qpos_addr] if 0 <= qpos_addr < len(self.data.qpos) else 0.0
                vel = self.data.qvel[qvel_addr] if 0 <= qvel_addr < len(self.data.qvel) else 0.0
                return (float(pos), float(vel))
            except (AttributeError, IndexError):
                pass
        
        return (0.0, 0.0)
    
    def publish_joint_states(self):
        """Publish current joint states."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        for joint_name in JOINT_NAMES:
            pos, vel = self._get_joint_state(joint_name)
            msg.name.append(joint_name)
            msg.position.append(pos)
            msg.velocity.append(vel)
            msg.effort.append(0.0)
        
        self.joint_state_pub.publish(msg)
    
    def _render_camera(self):
        """Render camera view and publish it."""
        if self.camera_id is None or self.camera_renderer is None:
            return
        
        try:
            self.camera_obj.fixedcamid = self.camera_id
            self.camera_obj.type = mujoco.mjtCamera.mjCAMERA_FIXED
            
            self.camera_renderer.update_scene(self.data, camera=self.camera_obj)
            camera_rgb = self.camera_renderer.render()
            camera_bgr = cv2.cvtColor(camera_rgb, cv2.COLOR_RGB2BGR)
            camera_bgr_rotated = cv2.rotate(camera_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            cv2.imshow('Robot Camera', camera_bgr_rotated)
            cv2.waitKey(1)
            
            img_msg = Image()
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = self.camera_name if self.camera_name else 'camera_rgb'
            img_msg.height = CAMERA_HEIGHT
            img_msg.width = CAMERA_WIDTH
            img_msg.encoding = 'rgb8'
            img_msg.is_bigendian = False
            img_msg.step = CAMERA_WIDTH * 3
            img_msg.data = camera_rgb.tobytes()
            self.camera_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f'Camera rendering error: {e}')
    
    def _init_camera_rendering(self):
        """Initialize camera rendering components."""
        if self.camera_id is None:
            return
        
        try:
            self.camera_renderer = mujoco.Renderer(
                self.model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH
            )
            self.camera_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
            self.camera_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
            self.camera_obj = mujoco.MjvCamera()
            self.camera_obj.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.camera_obj.fixedcamid = self.camera_id
            cv2.namedWindow('Robot Camera', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Robot Camera', CAMERA_HEIGHT, CAMERA_WIDTH)
            self.get_logger().info('Camera rendering initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize camera rendering: {e}')
            self.camera_id = None
    
    def run_simulation(self):
        """Run the MuJoCo simulation loop."""
        # Hide lidar sites by setting their size to 0 (sites are already modified in XML)
        # This prevents the yellow lidar rays from being displayed
        lidar_site_ids = []
        for i in range(self.model.nsite):
            site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
            if site_name and 'lidar' in site_name.lower():
                lidar_site_ids.append(i)
                # Set site size to 0 to hide visualization
                self.model.site_size[i] = 0.0
        
        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            viewer.cam.lookat[:] = [0, 3, 1]
            viewer.cam.distance = 5
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 90
            
            self._init_camera_rendering()
            
            prev_render_time = prev_pub_time = prev_camera_time = time.time()
            pub_interval = 1.0 / PUB_RATE
            render_interval = 1.0 / RENDER_RATE
            camera_interval = 1.0 / PUB_RATE
            
            while self.running and viewer.is_running():
                step_start = time.time()
                
                self._update_navigation()
                self._update_arm_reset()
                self._update_joint_movements()
                
                for name, actuator_id in self.actuator_ids.items():
                    self.data.ctrl[actuator_id] = self.ctrl_state[name]
                
                mujoco.mj_step(self.model, self.data)
                
                now = time.time()
                if now - prev_pub_time > pub_interval:
                    self.publish_joint_states()
                    prev_pub_time = now
                
                if now - prev_camera_time > camera_interval:
                    self._render_camera()
                    prev_camera_time = now
                
                if now - prev_render_time > render_interval:
                    viewer.sync()
                    prev_render_time = now
                
                # Maintain consistent simulation rate
                elapsed = time.time() - step_start
                sleep_time = self.model.opt.timestep - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # Don't sleep if we're running behind - just continue
            
            if self.camera_id is not None:
                cv2.destroyAllWindows()
    
    def start_simulation(self):
        """Start simulation in a separate thread."""
        self.sim_thread = threading.Thread(target=self.run_simulation, daemon=True)
        self.sim_thread.start()
    
    def stop(self):
        """Stop the simulation."""
        self.running = False
        if self.sim_thread:
            self.sim_thread.join(timeout=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = StretchSimNode()
    node.start_simulation()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
