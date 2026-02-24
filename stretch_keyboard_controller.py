#!/usr/bin/env python3
"""Keyboard controller for Stretch 3 robot via ROS 2."""

import time
import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float64MultiArray
from pynput import keyboard
from anchor_utils import get_anchor_list


class StretchKeyboardController(Node):
    """Keyboard controller that sends ROS 2 commands based on key presses."""
    
    JOINT_LIMITS = {
        'lift': (0.0, 1.1),
        'arm_extend': (0.0, 0.52),
        'wrist_yaw': (-1.39, 4.42),
        'wrist_pitch': (-1.57, 0.56),
        'wrist_roll': (-3.14, 3.14),
        'gripper': (-0.02, 0.04),
        'head_pan': (-4.04, 1.73),
        'head_tilt': (-1.53, 0.79)
    }
    
    JOINT_ORDER = ['lift', 'arm_extend', 'wrist_yaw', 'wrist_pitch', 'wrist_roll', 'gripper', 'head_pan', 'head_tilt']
    
    JOINT_CONTROLS = {
        'Q': ('lift', 0.05), 'E': ('lift', -0.05),
        'R': ('arm_extend', 0.05), 'F': ('arm_extend', -0.05),
        'T': ('wrist_yaw', 0.1), 'G': ('wrist_yaw', -0.1),
        'Y': ('wrist_pitch', 0.1), 'H': ('wrist_pitch', -0.1),
        'U': ('wrist_roll', 0.1), 'J': ('wrist_roll', -0.1),
        'Z': ('gripper', 0.01), 'X': ('gripper', -0.01)
    }
    
    HEAD_CONTROLS = {
        keyboard.Key.left: ('head_pan', 0.1),
        keyboard.Key.right: ('head_pan', -0.1),
        keyboard.Key.up: ('head_tilt', 0.1),
        keyboard.Key.down: ('head_tilt', -0.1)
    }
    
    BASE_VELOCITY = {
        'W': {'linear_x': -1.0}, 'S': {'linear_x': 1.0},
        'A': {'angular_z': -1.0}, 'D': {'angular_z': 1.0}
    }
    
    BASE_STOP_KEYS = {'W': 'linear_x', 'S': 'linear_x', 'A': 'angular_z', 'D': 'angular_z'}
    
    def __init__(self):
        super().__init__('stretch_keyboard_controller')
        
        self.anchor_map = self._load_anchors()
        self._setup_publishers()
        self._init_state()
        self._start_keyboard_listener()
    
    def _load_anchors(self):
        """Load anchor mapping from XML or use defaults."""
        try:
            anchor_list = get_anchor_list()
            return {str(i): letter for i, letter in 
                   enumerate(sorted(anchor_list)[:5], start=1)}
        except Exception as e:
            self.get_logger().warn(f'Failed to load anchors: {e}, using default')
            return {'1': 'A', '2': 'B', '3': 'C', '4': 'D', '5': 'E'}
    
    def _setup_publishers(self):
        """Initialize ROS 2 publishers."""
        self.cmd_vel_pub = self.create_publisher(Twist, '/stretch/cmd_vel', 10)
        self.anchor_pub = self.create_publisher(String, '/stretch/navigate_to_anchor', 10)
        self.joint_cmd_pub = self.create_publisher(Float64MultiArray, '/stretch/joint_commands', 10)
        self.reset_pub = self.create_publisher(String, '/stretch/reset_arm', 10)
    
    def _init_state(self):
        """Initialize control state."""
        self.running = True
        self.base_vel = {'linear_x': 0.0, 'angular_z': 0.0}
        self.joint_state = {key: 0.0 for key in self.JOINT_LIMITS.keys()}
    
    def _start_keyboard_listener(self):
        """Start keyboard listener and base velocity timer."""
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        self.create_timer(0.1, self._publish_base_velocity)
    
    def _on_press(self, key):
        """Handle key press events."""
        try:
            if key == keyboard.Key.esc:
                self.running = False
                return False
            
            if key in self.HEAD_CONTROLS:
                self._update_joint(*self.HEAD_CONTROLS[key])
                return
            
            if not hasattr(key, 'char') or not key.char:
                return
            
            key_char = key.char.upper()
            
            if key_char == '0':
                self._reset_arm()
            elif key_char in self.anchor_map:
                self._navigate_to_anchor(self.anchor_map[key_char])
            elif key_char in self.BASE_VELOCITY:
                self.base_vel.update(self.BASE_VELOCITY[key_char])
            elif key_char in self.JOINT_CONTROLS:
                self._update_joint(*self.JOINT_CONTROLS[key_char])
        except AttributeError:
            pass
    
    def _on_release(self, key):
        """Handle key release events."""
        try:
            if hasattr(key, 'char') and key.char:
                key_char = key.char.upper()
                if key_char in self.BASE_STOP_KEYS:
                    self.base_vel[self.BASE_STOP_KEYS[key_char]] = 0.0
                    self._publish_base_velocity()
        except AttributeError:
            pass
    
    def _update_joint(self, joint_name, delta):
        """Update joint position and publish command."""
        if joint_name not in self.joint_state:
            self.get_logger().warn(f'Unknown joint: {joint_name}')
            return
        
        min_val, max_val = self.JOINT_LIMITS[joint_name]
        self.joint_state[joint_name] = np.clip(
            self.joint_state[joint_name] + delta, min_val, max_val
        )
        self._publish_joint_commands()
    
    def _reset_arm(self):
        """Send arm reset command."""
        msg = String()
        msg.data = 'reset'
        self.reset_pub.publish(msg)
        self.get_logger().info('Resetting arm to default position')
    
    def _navigate_to_anchor(self, anchor_key):
        """Send navigation command to specified anchor."""
        msg = String()
        msg.data = anchor_key
        self.anchor_pub.publish(msg)
        self.get_logger().info(f'Navigating to anchor {anchor_key}')
    
    def _publish_base_velocity(self):
        """Publish current base velocity."""
        msg = Twist()
        msg.linear.x = self.base_vel['linear_x']
        msg.angular.z = self.base_vel['angular_z']
        self.cmd_vel_pub.publish(msg)
    
    def _publish_joint_commands(self):
        """Publish current joint commands."""
        msg = Float64MultiArray()
        msg.data = [self.joint_state.get(key, 0.0) for key in self.JOINT_ORDER]
        self.joint_cmd_pub.publish(msg)
    
    def stop(self):
        """Stop the controller."""
        self.running = False
        self.base_vel = {'linear_x': 0.0, 'angular_z': 0.0}
        self._publish_base_velocity()
        if self.listener:
            self.listener.stop()


def main(args=None):
    rclpy.init(args=args)
    controller = StretchKeyboardController()
    
    anchor_str = '/'.join([f'{k}→{v}' for k, v in sorted(controller.anchor_map.items())])
    
    print("\n" + "="*50)
    print("Stretch 3 Keyboard Controller")
    print("="*50)
    print("Base Movement:")
    print(f"  {anchor_str} - Navigate to anchors (1-5)")
    print("  W/S - Move forward/backward")
    print("  A/D - Turn left/right")
    print("\nArm & Gripper:")
    print("  Q/E - Lift up/down")
    print("  R/F - Arm extend/retract")
    print("  T/G - Wrist yaw rotate")
    print("  Y/H - Wrist pitch (Stretch 3)")
    print("  U/J - Wrist roll (Stretch 3)")
    print("  Z/X - Gripper open/close")
    print("  0 - Reset arm to default position")
    print("\nCamera/Head:")
    print("  Arrow Keys - Pan/tilt head")
    print("\n  ESC - Exit")
    print("="*50 + "\n")
    
    try:
        while controller.running:
            rclpy.spin_once(controller, timeout_sec=0.1)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
