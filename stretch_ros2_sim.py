#!/usr/bin/env python3
"""ROS 2 node for Stretch 3 MuJoCo simulation."""

import os
import json
import argparse
import time
import threading
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry, Path
from nav2_msgs.action import ComputePathToPose, FollowPath, NavigateToPose
from sensor_msgs.msg import JointState, Image, LaserScan
from std_msgs.msg import Float64MultiArray, String, Bool
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
import mujoco
import mujoco.viewer
import numpy as np
import cv2
from ik import IKSolver
from navigation import NavigationController
from anchor_utils import load_anchors_from_xml
from stretch_marl_observation import (
    DEFAULT_OBS_RADIUS as DEFAULT_MARL_OBS_RADIUS,
    OBS_OBJECTS,
    build_observation as build_marl_observation,
    get_observation_layout as get_marl_observation_layout,
)

# Joint configuration
JOINT_NAMES = [
    'joint_lift', 'joint_arm_l0', 'joint_arm_l1', 'joint_arm_l2', 'joint_arm_l3',
    'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll', 'joint_head_pan', 'joint_head_tilt'
]

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
STEPS_PER_CONTROL = 10
RESET_SPEED = 0.25  # m/rad per second at 50% speed
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
DEFAULT_SPEED = 50.0
JOINT_TOLERANCE = 0.001
ZERO_PLACEHOLDER_THRESHOLD = 0.05
SMOOTHING_FACTOR = 0.6  # Exponential smoothing factor (0-1, lower = smoother)
JOINT_POSITION_GAIN = 8.0
ROBOT1_DEFAULT_LIFT = 0.20
ROBOT2_DEFAULT_LIFT = 0.35
LIDAR_FRAME_ID = 'stretch/laser'
BASE_FRAME_ID = 'stretch/base_link'
ODOM_FRAME_ID = 'stretch/odom'
R2_LIDAR_FRAME_ID = 'stretch2/laser'
R2_BASE_FRAME_ID = 'stretch2/base_link'
R2_ODOM_FRAME_ID = 'stretch2/odom'
LIDAR_RATE = 10.0
LIDAR_RANGE_MIN = 0.05
LIDAR_RANGE_MAX = 8.0
LIDAR_ANGLE_MIN = -math.pi
LIDAR_ANGLE_MAX = math.pi
LIDAR_NUM_RAYS = 720
DYNAMIC_ROBOT_OBSTACLE_RADIUS = 0.35
NAV2_COLLISION_RADIUS = 0.65
NAV2_COLLISION_RELEASE_RADIUS = 0.85
GRASPABLE_OBJECTS = ('tomato1', 'lettuce1', 'onion1')  # kept for MARL obs helpers
GRASP_CLOSE_WIDTH = 0.012
GRASP_RELEASE_WIDTH = 0.026
VEGGIE_MATERIALS = {
    'tomato1': 'tomato_mat',
    'lettuce1': 'lettuce_mat',
    'onion1': 'onion_mat',
}
CHOPPED_VISUAL_RGBA = {
    'tomato1': np.array([1.0, 0.35, 0.25, 1.0]),
    'lettuce1': np.array([0.55, 1.0, 0.35, 1.0]),
    'onion1': np.array([1.0, 0.85, 0.45, 1.0]),
}
# Cartographer's map frame starts at the robot pose used during mapping.
# In this MuJoCo world the robot starts at anchor_ORIGIN, not at world (0, 0),
# so map coordinates are world coordinates translated by -anchor_ORIGIN.
MAP_TO_ODOM_X = float(os.environ.get('MAP_TO_ODOM_X', '-0.533'))
MAP_TO_ODOM_Y = float(os.environ.get('MAP_TO_ODOM_Y', '-2.317'))
MAP_TO_ODOM_YAW = math.radians(float(os.environ.get('MAP_TO_ODOM_YAW_DEG', '0.0')))
R2_MAP_TO_ODOM_X = float(os.environ.get('R2_MAP_TO_ODOM_X', str(MAP_TO_ODOM_X)))
R2_MAP_TO_ODOM_Y = float(os.environ.get('R2_MAP_TO_ODOM_Y', str(MAP_TO_ODOM_Y)))
R2_MAP_TO_ODOM_YAW = math.radians(
    float(os.environ.get('R2_MAP_TO_ODOM_YAW_DEG', str(math.degrees(MAP_TO_ODOM_YAW))))
)
GLOBAL_PATH_WAYPOINT_SPACING = 0.15
GLOBAL_PATH_WAYPOINT_TOLERANCE = 0.08
NAV2_ANCHOR_STANDOFFS = {
    # These are reachable navigation poses near the task anchors. The task
    # anchors can sit close to table surfaces, which puts them inside inflated
    # Nav2 costmap cells even when the robot should stand nearby.
    'A': [-0.65, 1.00, 0.0],
    'B': [0.65, 1.00, 0.0],
    'C': [-0.65, 3.50, 0.0],
    'D': [0.65, 3.50, 0.0],
    'E': [1.55, 2.90, 0.0],
    'F': [1.55, 1.60, 0.0],
}
class StretchSimNode(Node):
    """ROS 2 node that runs MuJoCo simulation and handles ROS 2 communication."""
    
    def __init__(
        self,
        world_xml='table_world.xml',
        enable_camera=True,
        enable_nav2=False,
        global_plan_only=False,
        local_plan=False,
        single_robot=False,
    ):
        super().__init__('stretch_sim')
        
        xml_path = world_xml
        if not os.path.isabs(xml_path):
            xml_path = os.path.join(os.path.dirname(__file__), xml_path)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.enable_camera = enable_camera
        self.enable_nav2 = enable_nav2
        self.global_plan_only = global_plan_only
        self.local_plan = local_plan
        self.single_robot = single_robot
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
        self._last_global_path = None
        self._last_r2_global_path = None
        self._last_global_path_marker = None
        self._last_r2_global_path_marker = None
        self._last_global_endpoint_markers = []
        self._last_r2_global_endpoint_markers = []
        self._last_global_marker_array = None
        self._last_r2_global_marker_array = None
        self._global_follow_waypoints = []
        self._global_follow_idx = 0
        self._global_follow_active = False
        self._r2_global_follow_waypoints = []
        self._r2_global_follow_idx = 0
        self._r2_global_follow_active = False
        self._active_global_goal_keys = {}
        self._pending_global_anchor_fallbacks = {}
        self._active_local_replan_target = None
        self._active_r2_local_replan_target = None
        self._local_replan_attempts = 0
        self._r2_local_replan_attempts = 0
        self.marl_chopped_status = {
            'tomato1': 0.0,
            'lettuce1': 0.0,
            'onion1': 0.0,
        }
        self._veggie_material_ids = {}
        self._original_veggie_material_rgba = {}
        self._init_chopped_visuals()
        self._adhesion_ids = {}
        self._follow_path_goal_handle = None
        self._follow_path_goal_seq = 0
        self._follow_path_goal_pending = False
        self._r2_follow_path_goal_handle = None
        self._r2_follow_path_goal_seq = 0
        self._r2_follow_path_goal_pending = False
        self._nav_log_counter = 0
        self._resetting_arm = False
        self._reset_targets = {}
        self._joint_targets = {}
        self._joint_speed_percent = {}
        self._base_joint_speed = 0.25  # m/rad per second at 50% speed
        self._joint_velocities = {}  # Track velocities for smooth acceleration
        self._control_dt = self.model.opt.timestep * STEPS_PER_CONTROL
        self.ik_solver=IKSolver(self.model,self.data,logger=self.get_logger())
        self.r2_ik_solver = IKSolver(self.model, self.data, logger=self.get_logger(), name_prefix='r2_') if self.robot2_enabled else None
        self.nav2_client = ActionClient(self, NavigateToPose, '/stretch/navigate_to_pose') if self.enable_nav2 else None
        self.global_path_client = (
            ActionClient(self, ComputePathToPose, '/stretch/compute_path_to_pose')
            if self.global_plan_only else None
        )
        self.r2_global_path_client = (
            ActionClient(self, ComputePathToPose, '/stretch2/compute_path_to_pose')
            if self.global_plan_only and self.robot2_enabled else None
        )
        self.follow_path_client = (
            ActionClient(self, FollowPath, '/stretch/follow_path')
            if self.local_plan else None
        )
        self.r2_follow_path_client = (
            ActionClient(self, FollowPath, '/stretch2/follow_path')
            if self.local_plan and self.robot2_enabled else None
        )
        self.r2_nav2_client = (
            ActionClient(self, NavigateToPose, '/stretch2/navigate_to_pose')
            if self.enable_nav2 and self.robot2_enabled and not self.global_plan_only else None
        )
        self.get_logger().info('Stretch 3 ROS 2 Simulation Node started')
        if self.global_plan_only or self.enable_nav2:
            self.get_logger().info(
                'Using map<-odom alignment: '
                f'x={MAP_TO_ODOM_X:.3f}, y={MAP_TO_ODOM_Y:.3f}, '
                f'yaw={math.degrees(MAP_TO_ODOM_YAW):.1f}deg'
            )
            if self.robot2_enabled:
                self.get_logger().info(
                    'Using stretch2 map<-odom alignment: '
                    f'x={R2_MAP_TO_ODOM_X:.3f}, y={R2_MAP_TO_ODOM_Y:.3f}, '
                    f'yaw={math.degrees(R2_MAP_TO_ODOM_YAW):.1f}deg'
                )
    
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
        self._set_joint_qpos('joint_lift', ROBOT1_DEFAULT_LIFT)
        for joint_name in ('joint_arm_l0', 'joint_arm_l1', 'joint_arm_l2', 'joint_arm_l3'):
            self._set_joint_qpos(joint_name, 0.0)
        self._set_joint_qpos('r2_joint_lift', ROBOT2_DEFAULT_LIFT)
        for joint_name in ('r2_joint_arm_l0', 'r2_joint_arm_l1', 'r2_joint_arm_l2', 'r2_joint_arm_l3'):
            self._set_joint_qpos(joint_name, 0.0)
        mujoco.mj_forward(self.model, self.data)

    def _set_joint_qpos(self, joint_name, value):
        """Set a scalar joint position by name if it exists in the model."""
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            return False

        qpos_addr = self.model.jnt_qposadr[joint_id]
        if 0 <= qpos_addr < len(self.data.qpos):
            self.data.qpos[qpos_addr] = value
            return True
        return False

    def _init_actuators(self):
        """Initialize actuator and joint mappings."""
        self.actuator_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ACTUATOR_NAMES
        }
        self.ctrl_state = {name: 0.0 for name in ACTUATOR_NAMES}
        self.ctrl_state['lift'] = ROBOT1_DEFAULT_LIFT
        
        self.joint_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
        }
        
        self.base_link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        self.base_drive_dofs = {
            name: self._get_joint_dof(name)
            for name in ('base_x', 'base_y', 'base_yaw')
        }
        self.use_planar_base_drive = all(dof is not None for dof in self.base_drive_dofs.values())
        self.base_freejoint_dof = self._get_body_freejoint_dof(self.base_link_id)
        self.base_velocity_cmd = {'linear_x': 0.0, 'angular_z': 0.0}
        if self.use_planar_base_drive:
            self.get_logger().info('Using direct planar base drive (base_x/base_y/base_yaw qvel)')
        elif self.base_freejoint_dof is not None:
            self.get_logger().info('Using direct freejoint base drive for /stretch')
        
        self.robot2_enabled = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'r2_base_link') >= 0
            and not self.single_robot
        )
        if self.robot2_enabled:
            self.r2_actuator_ids = {
                name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'r2_{name}')
                for name in ACTUATOR_NAMES
            }
            self.r2_ctrl_state = {name: 0.0 for name in ACTUATOR_NAMES}
            self.r2_ctrl_state['lift'] = ROBOT2_DEFAULT_LIFT
            self.r2_joint_ids = {
                name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'r2_{name}')
                for name in JOINT_NAMES
                if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'r2_{name}') >= 0
            }
            self.r2_base_link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'r2_base_link')
            self.r2_base_drive_dofs = {
                name: self._get_joint_dof(f'r2_{name}')
                for name in ('base_x', 'base_y', 'base_yaw')
            }
            self.r2_use_planar_base_drive = all(dof is not None for dof in self.r2_base_drive_dofs.values())
            self.r2_base_freejoint_dof = self._get_body_freejoint_dof(self.r2_base_link_id)
            self.r2_base_velocity_cmd = {'linear_x': 0.0, 'angular_z': 0.0}
            self.r2_nav_controller = NavigationController()
            self.r2_manual_control = False
            self.r2_resetting_arm = False
            self.r2_reset_targets = {}
            self.r2_joint_targets = {}
            self.r2_joint_speed_percent = {}
            self.r2_joint_velocities = {}
            self.r2_alignment_cancel = False
            self.r2_alignment_active = False
            self.get_logger().info('Second Stretch robot enabled on /stretch2 topics')

        self._adhesion_ids = {
            key: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for key, name in (
                ('r1_left',  'adh_r1_left'),
                ('r1_right', 'adh_r1_right'),
                ('r2_left',  'adh_r2_left'),
                ('r2_right', 'adh_r2_right'),
            )
        }

    def _get_joint_dof(self, joint_name):
        """Return the qvel address for a scalar joint, or None if missing."""
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            return None
        dof_addr = int(self.model.jnt_dofadr[joint_id])
        return dof_addr if 0 <= dof_addr < self.model.nv else None

    def _get_body_freejoint_dof(self, body_id):
        """Return qvel address for a body's freejoint, or None when it has no freejoint."""
        if body_id < 0:
            return None
        joint_start = int(self.model.body_jntadr[body_id])
        joint_count = int(self.model.body_jntnum[body_id])
        for joint_id in range(joint_start, joint_start + joint_count):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                dof_addr = int(self.model.jnt_dofadr[joint_id])
                return dof_addr if 0 <= dof_addr + 5 < self.model.nv else None
        return None

    def _get_body_freejoint_qpos(self, body_id):
        """Return qpos address for a body's freejoint, or None when it has no freejoint."""
        if body_id < 0:
            return None
        joint_start = int(self.model.body_jntadr[body_id])
        joint_count = int(self.model.body_jntnum[body_id])
        for joint_id in range(joint_start, joint_start + joint_count):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                qpos_addr = int(self.model.jnt_qposadr[joint_id])
                return qpos_addr if 0 <= qpos_addr + 6 < self.model.nq else None
        return None
    
    def _init_camera(self):
        """Initialize camera if available."""
        self.camera_id = None
        self.camera_name = None
        self.camera_renderer = None
        self.camera_obj = None
        if not self.enable_camera:
            self.get_logger().info('Camera display disabled')
            return

        # Try Stretch 3 cameras in order of preference
        camera_names = ['d435i_camera_rgb', 'd405_rgb', 'nav_camera_rgb']

        for cam_name in camera_names:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id >= 0:
                self.camera_id = cam_id
                self.camera_name = cam_name
                self.get_logger().info(f'Camera "{cam_name}" found (ID: {self.camera_id})')
                break
        
        if self.camera_id is None:
            self.get_logger().warn('No Stretch 3 camera found, camera display disabled')
    
    def _setup_ros2(self):
        """Setup ROS 2 publishers and subscribers."""
        self.create_subscription(Twist, '/stretch/cmd_vel', self._cmd_vel_callback, 10)
        self.create_subscription(Twist, '/stretch/cmd_vel_nav', self._nav2_cmd_vel_callback, 10)
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
        self.create_subscription(PoseStamped, '/goal_pose',
                                self._rviz_goal_pose_callback, 10)
        self.create_subscription(PointStamped, '/clicked_point',
                                self._rviz_clicked_point_callback, 10)
        self.create_subscription(String, '/stretch/align_with_target',
                            self._align_with_target_callback, 10)
        self.create_subscription(String, '/stretch/compute_ik',  # NEW
                            self._compute_ik_callback, 10)
        self.create_subscription(String, '/stretch/marl_chopped_status',
                            self._marl_chopped_status_callback, 10)
        self.joint_state_pub = self.create_publisher(JointState, '/stretch/joint_states', 10)
        self.nav_status_pub = self.create_publisher(Bool, '/stretch/navigation_active', 10)
        global_path_qos = QoSProfile(depth=1)
        global_path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        global_path_qos.reliability = ReliabilityPolicy.RELIABLE
        self.global_path_pub = self.create_publisher(Path, '/stretch/global_path', global_path_qos)
        self.r2_global_path_pub = self.create_publisher(Path, '/stretch2/global_path', global_path_qos)
        self.global_path_marker_pub = self.create_publisher(
            Marker, '/stretch/global_path_marker', global_path_qos
        )
        self.r2_global_path_marker_pub = self.create_publisher(
            Marker, '/stretch2/global_path_marker', global_path_qos
        )
        self.global_endpoint_marker_pub = self.create_publisher(
            Marker, '/stretch/global_endpoint_markers', global_path_qos
        )
        self.r2_global_endpoint_marker_pub = self.create_publisher(
            Marker, '/stretch2/global_endpoint_markers', global_path_qos
        )
        self.global_visualization_pub = self.create_publisher(
            MarkerArray, '/stretch/global_plan_markers', global_path_qos
        )
        self.r2_global_visualization_pub = self.create_publisher(
            MarkerArray, '/stretch2/global_plan_markers', global_path_qos
        )
        self.anchor_marker_pub = self.create_publisher(
            MarkerArray, '/stretch/anchor_markers', global_path_qos
        )
        self.anchor_pose_pub = self.create_publisher(
            PoseArray, '/stretch/anchor_poses', global_path_qos
        )
        self.anchor_path_pub = self.create_publisher(
            Path, '/stretch/anchor_path_debug', global_path_qos
        )
        self.global_path_timer = self.create_timer(1.0, self._republish_global_path)
        self.camera_pub = self.create_publisher(Image, '/stretch/camera/image_raw', 10)
        self.ik_result_pub = self.create_publisher(Float64MultiArray, '/stretch/ik_result', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/stretch/scan', 10)
        self.odom_pub = self.create_publisher(Odometry, '/stretch/odom', 10)
        self.tf_pub = self.create_publisher(TFMessage, '/stretch/tf', 10)
        self.object_positions_pub = self.create_publisher(String, '/sim/object_positions', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        if getattr(self, 'robot2_enabled', False):
            self.create_subscription(Twist, '/stretch2/cmd_vel', self._r2_cmd_vel_callback, 10)
            self.create_subscription(Twist, '/stretch2/cmd_vel_nav', self._r2_nav2_cmd_vel_callback, 10)
            self.create_subscription(Float64MultiArray, '/stretch2/joint_command', self._r2_joint_command_callback, 10)
            self.create_subscription(Float64MultiArray, '/stretch2/joint_commands', self._r2_joint_commands_callback, 10)
            self.create_subscription(String, '/stretch2/navigate_to_anchor', self._r2_navigate_to_anchor_callback, 10)
            self.create_subscription(String, '/stretch2/turn_towards_anchor', self._r2_turn_towards_anchor_callback, 10)
            self.create_subscription(String, '/stretch2/reset_arm', self._r2_reset_arm_callback, 10)
            self.create_subscription(Float64MultiArray, '/stretch2/navigate_to_position', self._r2_navigate_to_position_callback, 10)
            self.create_subscription(String, '/stretch2/align_with_target', self._r2_align_with_target_callback, 10)
            self.create_subscription(String, '/stretch2/compute_ik', self._r2_compute_ik_callback, 10)
            self.create_subscription(String, '/stretch2/marl_chopped_status', self._marl_chopped_status_callback, 10)
            self.r2_joint_state_pub = self.create_publisher(JointState, '/stretch2/joint_states', 10)
            self.r2_nav_status_pub = self.create_publisher(Bool, '/stretch2/navigation_active', 10)
            self.r2_ik_result_pub = self.create_publisher(Float64MultiArray, '/stretch2/ik_result', 10)
            self.r2_scan_pub = self.create_publisher(LaserScan, '/stretch2/scan', 10)
            self.r2_odom_pub = self.create_publisher(Odometry, '/stretch2/odom', 10)
            self.r2_tf_pub = self.create_publisher(TFMessage, '/stretch2/tf', 10)
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
    
    def _set_joint_target_for(self, ctrl_state, joint_targets, joint_speed_percent,
                              actuator_name, value, speed_percent):
        """Set target for a robot-specific joint actuator state."""
        if actuator_name not in ctrl_state:
            return False
        
        cmd_name = next((cmd for cmd, act in JOINT_COMMAND_MAP if act == actuator_name), None)
        if cmd_name:
            min_val, max_val = JOINT_LIMITS[cmd_name]
            value = np.clip(value, min_val, max_val)
        
        joint_targets[actuator_name] = value
        joint_speed_percent[actuator_name] = speed_percent
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
    def _r2_compute_ik_callback(self, msg):
        """Handle robot 2 IK computation request for a target object."""
        target_name = msg.data.strip()
        self.get_logger().info(f'/stretch2 computing IK for target: {target_name}')
        result_msg = Float64MultiArray()
        try:
            target_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, f'{target_name}_site'
            )
            if target_site_id < 0:
                self.get_logger().error(f'/stretch2 target site {target_name}_site not found')
                result_msg.data = [0.0, 0.0, 0.0, 0.0]
                self.r2_ik_result_pub.publish(result_msg)
                return
            target_pos = self.data.site_xpos[target_site_id].copy()
            success, joint_solution = self.r2_ik_solver.compute_ik(target_pos)
            if success:
                q_lift = joint_solution[0]
                q_arm = sum(joint_solution[1:5])
                q_wrist = joint_solution[5]
                result_msg.data = [float(q_lift), float(q_arm), float(q_wrist), 1.0]
            else:
                result_msg.data = [0.0, 0.0, 0.0, 0.0]
            self.r2_ik_result_pub.publish(result_msg)
        except Exception as e:
            self.get_logger().error(f'/stretch2 IK computation error: {e}')
            result_msg.data = [0.0, 0.0, 0.0, 0.0]
            self.r2_ik_result_pub.publish(result_msg)

    def _init_chopped_visuals(self):
        for object_name, material_name in VEGGIE_MATERIALS.items():
            material_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, material_name)
            if material_id < 0:
                continue
            self._veggie_material_ids[object_name] = material_id
            self._original_veggie_material_rgba[object_name] = self.model.mat_rgba[material_id].copy()

    def _apply_chopped_visual(self, object_name):
        material_id = self._veggie_material_ids.get(object_name)
        if material_id is None:
            return

        if self.marl_chopped_status.get(object_name, 0.0) > 0.5:
            self.model.mat_rgba[material_id] = CHOPPED_VISUAL_RGBA[object_name]
        else:
            self.model.mat_rgba[material_id] = self._original_veggie_material_rgba[object_name]

    def _marl_chopped_status_callback(self, msg):
        """Update veggie chopped status used by MARL observations."""
        data = msg.data.strip()
        if ':' in data:
            object_name, value_text = data.split(':', 1)
        else:
            object_name, value_text = data, '1.0'

        object_name = object_name.strip()
        if object_name not in self.marl_chopped_status:
            self.get_logger().warn(f'Unknown MARL chopped object: {object_name}')
            return

        try:
            value = float(value_text.strip())
        except ValueError:
            self.get_logger().warn(f'Invalid MARL chopped status command: {data}')
            return

        self.marl_chopped_status[object_name] = 1.0 if value > 0.5 else 0.0
        self._apply_chopped_visual(object_name)
        self.get_logger().info(
            f'MARL chopped status updated: {object_name}={self.marl_chopped_status[object_name]:.1f}'
        )
    
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
    
    def _r2_joint_command_callback(self, msg):
        """Handle robot 2 single joint position command."""
        if len(msg.data) < 2:
            return
        joint_index = int(msg.data[0])
        value = float(msg.data[1])
        speed_percent = self._clamp_speed(msg.data[2] if len(msg.data) > 2 else DEFAULT_SPEED)
        if not (0 <= joint_index < len(JOINT_COMMAND_MAP)):
            self.get_logger().warn(f'Invalid /stretch2 joint index: {joint_index}')
            return
        _, actuator_name = JOINT_COMMAND_MAP[joint_index]
        self._set_joint_target_for(
            self.r2_ctrl_state, self.r2_joint_targets, self.r2_joint_speed_percent,
            actuator_name, value, speed_percent
        )
    
    def _r2_joint_commands_callback(self, msg):
        """Handle robot 2 multiple joint commands."""
        if self.r2_resetting_arm or len(msg.data) < len(JOINT_COMMAND_MAP):
            return
        speed_percent = self._clamp_speed(
            msg.data[-1] if len(msg.data) > len(JOINT_COMMAND_MAP) else DEFAULT_SPEED
        )
        for (cmd_name, actuator_name), value in zip(JOINT_COMMAND_MAP, msg.data[:len(JOINT_COMMAND_MAP)]):
            if actuator_name not in self.r2_ctrl_state:
                continue
            min_val, max_val = JOINT_LIMITS[cmd_name]
            target_value = np.clip(value, min_val, max_val)
            current_value = self.r2_ctrl_state[actuator_name]
            if (target_value == 0.0 and abs(current_value) > ZERO_PLACEHOLDER_THRESHOLD and min_val < 0.0):
                continue
            self._set_joint_target_for(
                self.r2_ctrl_state, self.r2_joint_targets, self.r2_joint_speed_percent,
                actuator_name, target_value, speed_percent
            )
    
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
    
    def _r2_reset_arm_callback(self, msg):
        """Handle robot 2 arm reset command."""
        data = msg.data.strip()
        if not data.lower().startswith('reset'):
            return
        speed_percent = DEFAULT_SPEED
        if ':' in data:
            try:
                speed_percent = self._clamp_speed(data.split(':', 1)[1])
            except ValueError:
                self.get_logger().warn(f'Invalid /stretch2 reset speed: {data}')
        self.get_logger().info(f'Starting /stretch2 arm reset (speed: {speed_percent}%)')
        self.r2_resetting_arm = True
        self.r2_reset_speed_percent = speed_percent
        self.r2_reset_targets = {
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
        dt = self._control_dt
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
            desired_velocity = np.clip(diff * JOINT_POSITION_GAIN, -max_speed, max_speed)
            
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
    def _update_r2_arm_reset(self):
        """Smoothly move robot 2 arm to reset positions."""
        if not self.robot2_enabled or not self.r2_resetting_arm:
            return
        
        speed_multiplier = getattr(self, 'r2_reset_speed_percent', DEFAULT_SPEED) / DEFAULT_SPEED
        dt = self._control_dt
        max_speed = RESET_SPEED * speed_multiplier / dt
        all_reached = True
        
        for actuator_name, target in self.r2_reset_targets.items():
            if actuator_name not in self.r2_ctrl_state:
                continue
            current = self.r2_ctrl_state[actuator_name]
            diff = target - current
            if abs(diff) < JOINT_TOLERANCE:
                self.r2_ctrl_state[actuator_name] = target
                continue
            desired_velocity = np.clip(diff * JOINT_POSITION_GAIN, -max_speed, max_speed)
            if actuator_name not in self.r2_joint_velocities:
                self.r2_joint_velocities[actuator_name] = 0.0
            current_velocity = self.r2_joint_velocities[actuator_name]
            smoothing = SMOOTHING_FACTOR * speed_multiplier
            new_velocity = current_velocity + smoothing * (desired_velocity - current_velocity)
            self.r2_joint_velocities[actuator_name] = new_velocity
            new_position = current + new_velocity * dt
            if (diff > 0 and new_position >= target) or (diff < 0 and new_position <= target):
                self.r2_ctrl_state[actuator_name] = target
                self.r2_joint_velocities.pop(actuator_name, None)
            else:
                self.r2_ctrl_state[actuator_name] = new_position
                all_reached = False
        
        if all_reached:
            self.r2_resetting_arm = False
            self.get_logger().info('✓ /stretch2 arm reset complete')
    
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
    
    def _r2_align_with_target_callback(self, msg):
        """Handle robot 2 alignment with target object command."""
        data = msg.data.strip()
        delta_angle = 5.0
        if ':' in data:
            parts = data.split(':')
            target_name = parts[0].strip()
            try:
                if len(parts) >= 2:
                    delta_angle = float(parts[1].strip())
            except (ValueError, IndexError):
                self.get_logger().warn(f'Invalid /stretch2 align command: {data}, using default')
        else:
            target_name = data.strip()
        
        self.r2_alignment_cancel = True
        time.sleep(0.1)
        self.r2_alignment_cancel = False
        self.r2_alignment_active = True
        threading.Thread(
            target=self._do_r2_alignment,
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
                    
                    # Rotate to correct orientation. NavigationController expects
                    # direction_tolerance in degrees and converts it internally.
                    target_heading = desired_yaw + (math.pi / 2.0)
                    self.nav_controller.set_target(
                        pos.copy(),
                        target_heading,
                        position_tolerance=0.01,
                        direction_tolerance=delta_angle
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
    
    def _do_r2_alignment(self, target_name, delta_angle):
        """Do robot 2 alignment in a background thread."""
        try:
            self.get_logger().info(f'/stretch2 aligning with target: {target_name}')
            for iteration in range(10):
                if self.r2_alignment_cancel:
                    self.get_logger().info('/stretch2 alignment cancelled')
                    return
                pos, quat = self._get_r2_robot_pose()
                yaw_diff, _, desired_yaw = self.r2_ik_solver.align_with_target(
                    pos, quat, tomato_name=target_name
                )
                yaw_diff_deg = abs(math.degrees(yaw_diff))
                if yaw_diff_deg <= delta_angle:
                    self.get_logger().info(f'✓ /stretch2 aligned! Error: {yaw_diff_deg:.2f}°')
                    return
                target_heading = desired_yaw + (math.pi / 2.0)
                self.r2_nav_controller.set_target(
                    pos.copy(), target_heading,
                    position_tolerance=0.01,
                    direction_tolerance=delta_angle
                )
                self.r2_manual_control = False
                start_time = time.time()
                while self.r2_nav_controller.is_active() and not self.r2_alignment_cancel:
                    if time.time() - start_time > 10.0:
                        self.get_logger().warn('/stretch2 alignment timeout')
                        return
                    time.sleep(0.05)
                time.sleep(0.2)
            pos, quat = self._get_r2_robot_pose()
            yaw_diff, _, _ = self.r2_ik_solver.align_with_target(pos, quat, tomato_name=target_name)
            self.get_logger().warn(f'/stretch2 max alignment iterations reached. Final error: {abs(math.degrees(yaw_diff)):.2f}°')
        finally:
            self.r2_alignment_active = False
            self.get_logger().info('/stretch2 alignment thread finished')
    def _update_joint_movements(self):
        """Gradually move joints towards their target positions with smooth interpolation."""
        if not self._joint_targets:
            return
        
        # Get simulation timestep for smooth movement
        dt = self._control_dt
        
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
            desired_velocity = np.clip(diff * JOINT_POSITION_GAIN, -max_speed, max_speed)
            
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
    
    def _update_r2_joint_movements(self):
        """Gradually move robot 2 joints towards their target positions."""
        if not self.robot2_enabled or not self.r2_joint_targets:
            return
        
        dt = self._control_dt
        for actuator_name, target in list(self.r2_joint_targets.items()):
            if actuator_name not in self.r2_ctrl_state:
                continue
            current = self.r2_ctrl_state[actuator_name]
            diff = target - current
            if abs(diff) < JOINT_TOLERANCE:
                self.r2_ctrl_state[actuator_name] = target
                self.r2_joint_targets.pop(actuator_name, None)
                self.r2_joint_speed_percent.pop(actuator_name, None)
                self.r2_joint_velocities.pop(actuator_name, None)
                continue
            speed_multiplier = self.r2_joint_speed_percent.get(actuator_name, DEFAULT_SPEED) / DEFAULT_SPEED
            max_speed = self._base_joint_speed * speed_multiplier / dt
            if actuator_name not in self.r2_joint_velocities:
                self.r2_joint_velocities[actuator_name] = 0.0
            desired_velocity = np.clip(diff * JOINT_POSITION_GAIN, -max_speed, max_speed)
            current_velocity = self.r2_joint_velocities[actuator_name]
            smoothing = SMOOTHING_FACTOR * speed_multiplier
            new_velocity = current_velocity + smoothing * (desired_velocity - current_velocity)
            self.r2_joint_velocities[actuator_name] = new_velocity
            new_position = current + new_velocity * dt
            if (diff > 0 and new_position >= target) or (diff < 0 and new_position <= target):
                self.r2_ctrl_state[actuator_name] = target
                self.r2_joint_targets.pop(actuator_name, None)
                self.r2_joint_speed_percent.pop(actuator_name, None)
                self.r2_joint_velocities.pop(actuator_name, None)
            else:
                self.r2_ctrl_state[actuator_name] = new_position
    
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

    def _handle_r2_anchor_command(self, anchor_key, turn_only=False, delta_angle=None,
                                  position_tolerance=None, target_angle_degrees=None):
        """Handle /stretch2 anchor navigation or turn-only commands."""
        if not self.robot2_enabled:
            return

        if turn_only:
            if target_angle_degrees is not None:
                self.r2_nav_controller.set_turn_only_target(None, delta_angle, target_angle_degrees)
                delta_value = delta_angle if delta_angle is not None else 5.0
                self.get_logger().info(
                    f'/stretch2 turning to absolute angle {target_angle_degrees:.1f}° '
                    f'(delta_angle={delta_value:.1f}°)'
                )
            else:
                anchor_key = anchor_key.strip().upper()
                if anchor_key not in self.anchors:
                    available = ', '.join(sorted(self.anchors.keys())) if self.anchors else 'none'
                    self.get_logger().warn(f'/stretch2 unknown anchor: {anchor_key}. Available: {available}')
                    return
                anchor_data = self.anchors[anchor_key]
                target_pos = anchor_data['pos']
                current_pos, _ = self._get_r2_robot_pose()
                distance = np.linalg.norm(np.array(target_pos[:2]) - current_pos[:2])
                self.r2_nav_controller.set_turn_only_target(target_pos, delta_angle)
                delta_value = delta_angle if delta_angle is not None else 5.0
                self.get_logger().info(
                    f'/stretch2 turning towards anchor {anchor_key} '
                    f'(distance: {distance:.2f}m, delta_angle={delta_value:.1f}°)'
                )
        else:
            anchor_key = anchor_key.strip().upper()
            if anchor_key not in self.anchors:
                available = ', '.join(sorted(self.anchors.keys())) if self.anchors else 'none'
                self.get_logger().warn(f'/stretch2 unknown anchor: {anchor_key}. Available: {available}')
                return
            anchor_data = self.anchors[anchor_key]
            target_pos = anchor_data['pos']
            current_pos, _ = self._get_r2_robot_pose()
            distance = np.linalg.norm(np.array(target_pos[:2]) - current_pos[:2])
            self.r2_nav_controller.set_target(target_pos, None, position_tolerance, None)
            pos_tol_info = f", pos_tol={position_tolerance:.2f}m" if position_tolerance else ""
            self.get_logger().info(
                f'/stretch2 navigating to anchor {anchor_key} '
                f'(distance: {distance:.2f}m{pos_tol_info})'
            )

        self.r2_manual_control = False

    def _set_anchor_target(self, robot_label, nav_controller, get_pose_fn, anchor_key,
                           position_tolerance=None):
        """Set an anchor target directly."""
        anchor_key = anchor_key.strip().upper()
        if anchor_key not in self.anchors:
            available = ', '.join(sorted(self.anchors.keys())) if self.anchors else 'none'
            self.get_logger().warn(f'{robot_label} unknown anchor: {anchor_key}. Available: {available}')
            return False

        target_pos = self.anchors[anchor_key]['pos']
        current_pos, _ = get_pose_fn()
        distance = np.linalg.norm(np.array(target_pos[:2]) - current_pos[:2])
        nav_controller.set_target(target_pos, None, position_tolerance, None)
        self.get_logger().info(f'{robot_label} navigating to anchor {anchor_key} (distance: {distance:.2f}m)')
        return True

    def _send_nav2_goal(self, robot_label, client, target_pos, target_direction=None):
        """Send a Nav2 NavigateToPose goal for a robot namespace."""
        if client is None:
            self.get_logger().warn(f'{robot_label} Nav2 client is not available')
            return False

        if not client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn(f'{robot_label} Nav2 navigate_to_pose server is not available yet')
            return False

        goal = NavigateToPose.Goal()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = 'map'
        map_target = self._world_to_map_pos(target_pos, robot_label=robot_label)
        goal.pose.pose.position.x = float(map_target[0])
        goal.pose.pose.position.y = float(map_target[1])
        goal.pose.pose.position.z = 0.0
        yaw = self._world_to_map_yaw(
            float(target_direction) if target_direction is not None else 0.0,
            robot_label=robot_label,
        )
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        client.send_goal_async(goal)
        self.get_logger().info(
            f'{robot_label} sent Nav2 goal map=({map_target[0]:.2f}, {map_target[1]:.2f}, '
            f'yaw={math.degrees(yaw):.1f}deg), world=({target_pos[0]:.2f}, {target_pos[1]:.2f})'
        )
        return True

    def _send_global_path_goal(self, target_pos, target_direction=None, robot_label='/stretch', is_replan=False):
        """Ask Nav2 planner_server for a global path without running a local controller."""
        client = self.r2_global_path_client if robot_label == '/stretch2' else self.global_path_client
        if client is None:
            self.get_logger().warn(f'{robot_label} global planner client is not available')
            return False

        if not client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn(f'{robot_label}/compute_path_to_pose server is not available yet')
            return False

        goal = ComputePathToPose.Goal()
        goal.goal.header.stamp = self.get_clock().now().to_msg()
        goal.goal.header.frame_id = 'map'
        current_pos, current_quat = (
            self._get_r2_robot_pose() if robot_label == '/stretch2' else self._get_robot_pose()
        )
        map_start = self._world_to_map_pos(current_pos, robot_label=robot_label)
        start_yaw = self._world_to_map_yaw(self._yaw_from_quat(current_quat), robot_label=robot_label)
        goal.start.header = goal.goal.header
        goal.start.pose.position.x = float(map_start[0])
        goal.start.pose.position.y = float(map_start[1])
        goal.start.pose.position.z = 0.0
        goal.start.pose.orientation.z = math.sin(start_yaw / 2.0)
        goal.start.pose.orientation.w = math.cos(start_yaw / 2.0)
        map_target = self._world_to_map_pos(target_pos, robot_label=robot_label)
        goal.goal.pose.position.x = float(map_target[0])
        goal.goal.pose.position.y = float(map_target[1])
        goal.goal.pose.position.z = 0.0
        yaw = self._world_to_map_yaw(
            float(target_direction) if target_direction is not None else 0.0,
            robot_label=robot_label,
        )
        goal.goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.goal.pose.orientation.w = math.cos(yaw / 2.0)
        goal.planner_id = 'GridBased'
        goal.use_start = True

        future = client.send_goal_async(goal)
        future.add_done_callback(lambda fut: self._global_path_goal_response_callback(fut, robot_label))
        if self.local_plan and robot_label in ('/stretch', '/stretch2'):
            if not is_replan:
                if robot_label == '/stretch2':
                    self._r2_local_replan_attempts = 0
                else:
                    self._local_replan_attempts = 0
            target_state = {
                'mode': 'world',
                'target': list(target_pos),
                'direction': target_direction,
            }
            if robot_label == '/stretch2':
                self._active_r2_local_replan_target = target_state
            else:
                self._active_local_replan_target = target_state
        self.get_logger().info(
            f'{robot_label} sent global path request from map=({map_start[0]:.2f}, {map_start[1]:.2f}) '
            f'to map=({map_target[0]:.2f}, {map_target[1]:.2f}, yaw={math.degrees(yaw):.1f}deg), '
            f'world=({target_pos[0]:.2f}, {target_pos[1]:.2f})'
        )
        return True

    def _global_goal_is_active(self, robot_label, goal_key):
        active = self._r2_global_follow_active if robot_label == '/stretch2' else self._global_follow_active
        if active and self._active_global_goal_keys.get(robot_label) == goal_key:
            self.get_logger().info(f'{robot_label} already following {goal_key}; ignoring duplicate request')
            return True
        self._active_global_goal_keys[robot_label] = goal_key
        return False

    @staticmethod
    def _world_to_map_pos(pos, robot_label='/stretch'):
        x = float(pos[0])
        y = float(pos[1])
        offset_x, offset_y, yaw = (
            (R2_MAP_TO_ODOM_X, R2_MAP_TO_ODOM_Y, R2_MAP_TO_ODOM_YAW)
            if robot_label == '/stretch2'
            else (MAP_TO_ODOM_X, MAP_TO_ODOM_Y, MAP_TO_ODOM_YAW)
        )
        c = math.cos(yaw)
        s = math.sin(yaw)
        return [
            offset_x + c * x - s * y,
            offset_y + s * x + c * y,
            float(pos[2]) if len(pos) > 2 else 0.0,
        ]

    @staticmethod
    def _world_to_map_yaw(yaw, robot_label='/stretch'):
        return yaw + (R2_MAP_TO_ODOM_YAW if robot_label == '/stretch2' else MAP_TO_ODOM_YAW)

    @staticmethod
    def _map_to_world_pos(pos, robot_label='/stretch'):
        offset_x, offset_y, yaw = (
            (R2_MAP_TO_ODOM_X, R2_MAP_TO_ODOM_Y, R2_MAP_TO_ODOM_YAW)
            if robot_label == '/stretch2'
            else (MAP_TO_ODOM_X, MAP_TO_ODOM_Y, MAP_TO_ODOM_YAW)
        )
        x = float(pos[0]) - offset_x
        y = float(pos[1]) - offset_y
        c = math.cos(yaw)
        s = math.sin(yaw)
        return [
            c * x + s * y,
            -s * x + c * y,
            float(pos[2]) if len(pos) > 2 else 0.0,
        ]

    @staticmethod
    def _map_to_world_yaw(yaw):
        return yaw - MAP_TO_ODOM_YAW

    def _send_global_path_goal_map(self, map_target, target_yaw=0.0, is_replan=False):
        """Ask Nav2 for a global path to a target already expressed in map frame."""
        client = self.global_path_client
        if client is None:
            self.get_logger().warn('Global planner client is not available')
            return False

        if not client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('/stretch/compute_path_to_pose server is not available yet')
            return False

        goal = ComputePathToPose.Goal()
        goal.goal.header.stamp = self.get_clock().now().to_msg()
        goal.goal.header.frame_id = 'map'
        current_pos, current_quat = self._get_robot_pose()
        map_start = self._world_to_map_pos(current_pos)
        start_yaw = self._world_to_map_yaw(self._yaw_from_quat(current_quat))
        goal.start.header = goal.goal.header
        goal.start.pose.position.x = float(map_start[0])
        goal.start.pose.position.y = float(map_start[1])
        goal.start.pose.position.z = 0.0
        goal.start.pose.orientation.z = math.sin(start_yaw / 2.0)
        goal.start.pose.orientation.w = math.cos(start_yaw / 2.0)
        goal.goal.pose.position.x = float(map_target[0])
        goal.goal.pose.position.y = float(map_target[1])
        goal.goal.pose.position.z = 0.0
        goal.goal.pose.orientation.z = math.sin(target_yaw / 2.0)
        goal.goal.pose.orientation.w = math.cos(target_yaw / 2.0)
        goal.planner_id = 'GridBased'
        goal.use_start = True

        future = client.send_goal_async(goal)
        future.add_done_callback(lambda fut: self._global_path_goal_response_callback(fut, '/stretch'))
        if self.local_plan:
            if not is_replan:
                self._local_replan_attempts = 0
            self._active_local_replan_target = {
                'mode': 'map',
                'target': list(map_target),
                'yaw': target_yaw,
            }
        self.get_logger().info(
            f'Sent RViz global path request from map=({map_start[0]:.2f}, {map_start[1]:.2f}) '
            f'to map=({map_target[0]:.2f}, {map_target[1]:.2f}, yaw={math.degrees(target_yaw):.1f}deg)'
        )
        return True

    def _global_path_goal_response_callback(self, future, robot_label='/stretch'):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f'{robot_label} global path request was rejected')
            self._retry_global_anchor_fallback(robot_label, 'request rejected')
            return
        goal_handle.get_result_async().add_done_callback(
            lambda fut: self._global_path_result_callback(fut, robot_label)
        )

    def _global_path_result_callback(self, future, robot_label='/stretch'):
        result = future.result().result
        path = result.path
        if not path.poses:
            self.get_logger().warn(f'{robot_label} global planner returned an empty path')
            if self._retry_global_anchor_fallback(robot_label, 'empty path'):
                return
            return
        self._pending_global_anchor_fallbacks.pop(robot_label, None)
        marker = self._path_to_marker(path, robot_label=robot_label)
        endpoint_markers = self._path_endpoint_markers(path, robot_label=robot_label)
        marker_array = MarkerArray()
        marker_array.markers = [marker] + endpoint_markers
        if robot_label == '/stretch2':
            self._last_r2_global_path = path
            self._last_r2_global_path_marker = marker
            self._last_r2_global_endpoint_markers = endpoint_markers
            self._last_r2_global_marker_array = marker_array
            self.r2_global_path_pub.publish(path)
            self.r2_global_path_marker_pub.publish(marker)
            for endpoint_marker in endpoint_markers:
                self.r2_global_endpoint_marker_pub.publish(endpoint_marker)
            self.r2_global_visualization_pub.publish(marker_array)
        else:
            self._last_global_path = path
            self._last_global_path_marker = marker
            self._last_global_endpoint_markers = endpoint_markers
            self._last_global_marker_array = marker_array
            self.global_path_pub.publish(path)
            self.global_path_marker_pub.publish(marker)
            for endpoint_marker in endpoint_markers:
                self.global_endpoint_marker_pub.publish(endpoint_marker)
            self.global_visualization_pub.publish(marker_array)
        self._publish_combined_global_plan_markers()
        if self.local_plan and robot_label == '/stretch':
            self._send_follow_path_goal(path)
        elif self.local_plan and robot_label == '/stretch2':
            self._send_r2_follow_path_goal(path)
        else:
            self._start_global_path_following(path, robot_label=robot_label)
        self.get_logger().info(
            f'{robot_label} global path ready: {len(path.poses)} poses, '
            f'length={self._path_length(path):.2f}m; published on {robot_label}/global_path '
            f'plus path/start/target markers'
        )

    def _send_follow_path_goal(self, path):
        """Send a computed global path to Nav2 controller_server FollowPath."""
        if self.follow_path_client is None:
            self.get_logger().warn('/stretch follow_path client is not available')
            self._active_global_goal_keys.pop('/stretch', None)
            return False

        if not self.follow_path_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('/stretch/follow_path server is not available yet')
            self._active_global_goal_keys.pop('/stretch', None)
            return False

        if self._follow_path_goal_handle is not None:
            try:
                self._follow_path_goal_handle.cancel_goal_async()
                self.get_logger().info('/stretch canceled previous FollowPath goal before sending a new path')
            except Exception as exc:
                self.get_logger().warn(f'/stretch could not cancel previous FollowPath goal: {exc}')
            self._follow_path_goal_handle = None

        self._follow_path_goal_seq += 1
        goal_seq = self._follow_path_goal_seq
        self._follow_path_goal_pending = True
        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = 'FollowPath'
        goal.goal_checker_id = ''
        future = self.follow_path_client.send_goal_async(goal)
        future.add_done_callback(lambda fut: self._follow_path_goal_response_callback(fut, goal_seq))
        self.manual_control = False
        self.get_logger().info(
            f'/stretch sent {len(path.poses)}-pose path to Nav2 FollowPath controller '
            f'(goal #{goal_seq})'
        )
        return True

    def _send_r2_follow_path_goal(self, path):
        """Send a computed global path to Stretch2 Nav2 FollowPath."""
        if self.r2_follow_path_client is None:
            self.get_logger().warn('/stretch2 follow_path client is not available')
            self._active_global_goal_keys.pop('/stretch2', None)
            return False

        if not self.r2_follow_path_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('/stretch2/follow_path server is not available yet')
            self._active_global_goal_keys.pop('/stretch2', None)
            return False

        if self._r2_follow_path_goal_handle is not None:
            try:
                self._r2_follow_path_goal_handle.cancel_goal_async()
                self.get_logger().info('/stretch2 canceled previous FollowPath goal before sending a new path')
            except Exception as exc:
                self.get_logger().warn(f'/stretch2 could not cancel previous FollowPath goal: {exc}')
            self._r2_follow_path_goal_handle = None

        self._r2_follow_path_goal_seq += 1
        goal_seq = self._r2_follow_path_goal_seq
        self._r2_follow_path_goal_pending = True
        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = 'FollowPath'
        goal.goal_checker_id = ''
        future = self.r2_follow_path_client.send_goal_async(goal)
        future.add_done_callback(lambda fut: self._r2_follow_path_goal_response_callback(fut, goal_seq))
        self.r2_manual_control = False
        self.get_logger().info(
            f'/stretch2 sent {len(path.poses)}-pose path to Nav2 FollowPath controller '
            f'(goal #{goal_seq})'
        )
        return True

    def _r2_follow_path_goal_response_callback(self, future, goal_seq):
        goal_handle = future.result()
        if goal_seq != self._r2_follow_path_goal_seq:
            self.get_logger().info(f'/stretch2 ignoring stale FollowPath goal response #{goal_seq}')
            return
        self._r2_follow_path_goal_pending = False
        if not goal_handle.accepted:
            self.get_logger().warn('/stretch2 FollowPath goal was rejected')
            self._active_global_goal_keys.pop('/stretch2', None)
            return
        self._r2_follow_path_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(
            lambda fut: self._r2_follow_path_result_callback(fut, goal_seq)
        )

    def _r2_follow_path_result_callback(self, future, goal_seq):
        if goal_seq != self._r2_follow_path_goal_seq:
            self.get_logger().info(f'/stretch2 ignoring stale FollowPath result #{goal_seq}')
            return
        try:
            status = future.result().status
        except Exception as exc:
            self.get_logger().warn(f'/stretch2 FollowPath result failed: {exc}')
            self._active_global_goal_keys.pop('/stretch2', None)
            self._r2_follow_path_goal_handle = None
            return

        self._set_r2_base_velocity(0.0, 0.0)
        if status != GoalStatus.STATUS_SUCCEEDED:
            if self._retry_r2_local_follow_replan(status):
                return
            self.get_logger().warn(f'/stretch2 FollowPath finished with status {status} and no replan was possible')

        self._active_global_goal_keys.pop('/stretch2', None)
        self._r2_follow_path_goal_handle = None
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._active_r2_local_replan_target = None
            self._r2_local_replan_attempts = 0
        self.get_logger().info(f'/stretch2 FollowPath finished with status {status}')

    def _follow_path_goal_response_callback(self, future, goal_seq):
        goal_handle = future.result()
        if goal_seq != self._follow_path_goal_seq:
            self.get_logger().info(f'/stretch ignoring stale FollowPath goal response #{goal_seq}')
            return
        self._follow_path_goal_pending = False
        if not goal_handle.accepted:
            self.get_logger().warn('/stretch FollowPath goal was rejected')
            self._active_global_goal_keys.pop('/stretch', None)
            return
        self._follow_path_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(
            lambda fut: self._follow_path_result_callback(fut, goal_seq)
        )

    def _follow_path_result_callback(self, future, goal_seq):
        if goal_seq != self._follow_path_goal_seq:
            self.get_logger().info(f'/stretch ignoring stale FollowPath result #{goal_seq}')
            return
        try:
            status = future.result().status
        except Exception as exc:
            self.get_logger().warn(f'/stretch FollowPath result failed: {exc}')
            self._active_global_goal_keys.pop('/stretch', None)
            self._follow_path_goal_handle = None
            return

        self._set_base_velocity(0.0, 0.0)
        if status != GoalStatus.STATUS_SUCCEEDED:
            if self._retry_local_follow_replan(status):
                return
            self.get_logger().warn(f'/stretch FollowPath finished with status {status} and no replan was possible')

        self._active_global_goal_keys.pop('/stretch', None)
        self._follow_path_goal_handle = None
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._active_local_replan_target = None
            self._local_replan_attempts = 0
        self.get_logger().info(f'/stretch FollowPath finished with status {status}')

    def _retry_local_follow_replan(self, status):
        """Recompute the global path when FollowPath aborts on a blocked route."""
        if not self.local_plan:
            return False
        if self._active_local_replan_target is None:
            self.get_logger().warn(
                f'/stretch FollowPath failed with status {status}, but no saved target exists for replan'
            )
            return False
        if self._local_replan_attempts >= 3:
            self.get_logger().warn(f'/stretch FollowPath failed with status {status}; no replans left')
            return False

        self._local_replan_attempts += 1
        target = self._active_local_replan_target
        self.get_logger().warn(
            f'/stretch FollowPath failed with status {status}; '
            f'replanning around current obstacles ({self._local_replan_attempts}/3)'
        )
        time.sleep(0.5)
        if target['mode'] == 'map':
            return self._send_global_path_goal_map(
                target['target'],
                target.get('yaw', 0.0),
                is_replan=True,
            )
        return self._send_global_path_goal(
            target['target'],
            target.get('direction'),
            robot_label='/stretch',
            is_replan=True,
        )

    def _retry_r2_local_follow_replan(self, status):
        """Recompute /stretch2 global path when its FollowPath goal aborts."""
        if not self.local_plan:
            return False
        if self._active_r2_local_replan_target is None:
            self.get_logger().warn(
                f'/stretch2 FollowPath failed with status {status}, but no saved target exists for replan'
            )
            return False
        if self._r2_local_replan_attempts >= 3:
            self.get_logger().warn(f'/stretch2 FollowPath failed with status {status}; no replans left')
            return False

        self._r2_local_replan_attempts += 1
        target = self._active_r2_local_replan_target
        self.get_logger().warn(
            f'/stretch2 FollowPath failed with status {status}; '
            f'replanning around current obstacles ({self._r2_local_replan_attempts}/3)'
        )
        time.sleep(0.5)
        return self._send_global_path_goal(
            target['target'],
            target.get('direction'),
            robot_label='/stretch2',
            is_replan=True,
        )

    def _retry_global_anchor_fallback(self, robot_label, reason):
        fallback = self._pending_global_anchor_fallbacks.get(robot_label)
        if fallback is None or fallback.get('used'):
            self._pending_global_anchor_fallbacks.pop(robot_label, None)
            self._active_global_goal_keys.pop(robot_label, None)
            return False

        fallback['used'] = True
        target = fallback['target']
        anchor_key = fallback['anchor_key']
        self.get_logger().warn(
            f'{robot_label} exact anchor {anchor_key} produced {reason}; '
            f'retrying reachable fallback at ({target[0]:.2f}, {target[1]:.2f})'
        )
        if not self._send_global_path_goal(target, fallback.get('direction'), robot_label=robot_label):
            self._pending_global_anchor_fallbacks.pop(robot_label, None)
            self._active_global_goal_keys.pop(robot_label, None)
            return False
        return True

    def _start_global_path_following(self, path, robot_label='/stretch'):
        waypoints = []
        last_world = None
        for pose_stamped in path.poses:
            p = pose_stamped.pose.position
            world = self._map_to_world_pos([p.x, p.y, p.z], robot_label=robot_label)
            if last_world is None:
                waypoints.append(world)
                last_world = world
                continue
            if math.hypot(world[0] - last_world[0], world[1] - last_world[1]) >= GLOBAL_PATH_WAYPOINT_SPACING:
                waypoints.append(world)
                last_world = world

        final_pose = path.poses[-1].pose.position
        final_world = self._map_to_world_pos([final_pose.x, final_pose.y, final_pose.z], robot_label=robot_label)
        if not waypoints or math.hypot(final_world[0] - waypoints[-1][0], final_world[1] - waypoints[-1][1]) > 1e-6:
            waypoints.append(final_world)

        # Drop waypoints already very close to the current robot pose.
        current_pos, _ = self._get_r2_robot_pose() if robot_label == '/stretch2' else self._get_robot_pose()
        current_xy = np.array(current_pos[:2])
        waypoints = [
            wp for wp in waypoints
            if np.linalg.norm(np.array(wp[:2]) - current_xy) > GLOBAL_PATH_WAYPOINT_TOLERANCE
        ]

        if robot_label == '/stretch2':
            self._r2_global_follow_waypoints = waypoints
            self._r2_global_follow_idx = 0
            self._r2_global_follow_active = bool(waypoints)
            self.r2_manual_control = False
            self.r2_nav_controller.cancel()
        else:
            self._global_follow_waypoints = waypoints
            self._global_follow_idx = 0
            self._global_follow_active = bool(waypoints)
            self.manual_control = False
            self.nav_controller.cancel()

        active = self._r2_global_follow_active if robot_label == '/stretch2' else self._global_follow_active
        if active:
            self._advance_global_path_waypoint(robot_label=robot_label)
            self.get_logger().info(f'{robot_label} following global path with {len(waypoints)} world-frame waypoints')
        else:
            self.get_logger().info(f'{robot_label} global path target is already within tolerance')
            self._active_global_goal_keys.pop(robot_label, None)

    def _advance_global_path_waypoint(self, robot_label='/stretch'):
        if robot_label == '/stretch2':
            active = self._r2_global_follow_active
            waypoints = self._r2_global_follow_waypoints
            idx = self._r2_global_follow_idx
            controller = self.r2_nav_controller
            stop = self._set_r2_base_velocity
        else:
            active = self._global_follow_active
            waypoints = self._global_follow_waypoints
            idx = self._global_follow_idx
            controller = self.nav_controller
            stop = self._set_base_velocity
        if not active:
            return
        if idx >= len(waypoints):
            if robot_label == '/stretch2':
                self._r2_global_follow_active = False
            else:
                self._global_follow_active = False
            self._active_global_goal_keys.pop(robot_label, None)
            controller.cancel()
            stop(0.0, 0.0)
            self.get_logger().info(f'✓ {robot_label} finished global path')
            return

        waypoint = waypoints[idx]
        if robot_label == '/stretch2':
            self._r2_global_follow_idx += 1
            idx = self._r2_global_follow_idx
        else:
            self._global_follow_idx += 1
            idx = self._global_follow_idx
        controller.set_target(
            waypoint,
            None,
            position_tolerance=GLOBAL_PATH_WAYPOINT_TOLERANCE,
            direction_tolerance=None,
        )
        self.get_logger().info(
            f'{robot_label} global path waypoint {idx}/{len(waypoints)}: '
            f'world=({waypoint[0]:.2f}, {waypoint[1]:.2f})'
        )

    def _republish_global_path(self):
        self.anchor_marker_pub.publish(self._anchor_markers())
        self.anchor_pose_pub.publish(self._anchor_pose_array())
        self.anchor_path_pub.publish(self._anchor_path_debug())
        if self._last_global_path is not None:
            self.global_path_pub.publish(self._last_global_path)
        if self._last_r2_global_path is not None:
            self.r2_global_path_pub.publish(self._last_r2_global_path)
        if self._last_global_path_marker is not None:
            self.global_path_marker_pub.publish(self._last_global_path_marker)
        if self._last_r2_global_path_marker is not None:
            self.r2_global_path_marker_pub.publish(self._last_r2_global_path_marker)
        for endpoint_marker in self._last_global_endpoint_markers:
            self.global_endpoint_marker_pub.publish(endpoint_marker)
        for endpoint_marker in self._last_r2_global_endpoint_markers:
            self.r2_global_endpoint_marker_pub.publish(endpoint_marker)
        if getattr(self, '_last_global_marker_array', None) is not None:
            self.global_visualization_pub.publish(self._last_global_marker_array)
        if getattr(self, '_last_r2_global_marker_array', None) is not None:
            self.r2_global_visualization_pub.publish(self._last_r2_global_marker_array)
        self._publish_combined_global_plan_markers()

    def _publish_combined_global_plan_markers(self):
        marker_array = MarkerArray()
        if getattr(self, '_last_global_marker_array', None) is not None:
            marker_array.markers.extend(self._last_global_marker_array.markers)
        if getattr(self, '_last_r2_global_marker_array', None) is not None:
            marker_array.markers.extend(self._last_r2_global_marker_array.markers)
        if marker_array.markers:
            self.global_visualization_pub.publish(marker_array)

    def _anchor_markers(self):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for idx, (anchor_key, anchor_data) in enumerate(sorted(self.anchors.items())):
            pos = self._world_to_map_pos(anchor_data['pos'])

            point_marker = Marker()
            point_marker.header.stamp = stamp
            point_marker.header.frame_id = 'map'
            point_marker.ns = 'stretch_anchors'
            point_marker.id = idx * 2
            point_marker.type = Marker.SPHERE
            point_marker.action = Marker.ADD
            point_marker.pose.position.x = float(pos[0])
            point_marker.pose.position.y = float(pos[1])
            point_marker.pose.position.z = 0.18
            point_marker.pose.orientation.w = 1.0
            point_marker.scale.x = 0.18
            point_marker.scale.y = 0.18
            point_marker.scale.z = 0.18
            point_marker.color.r = 0.2
            point_marker.color.g = 0.6
            point_marker.color.b = 1.0
            point_marker.color.a = 1.0
            marker_array.markers.append(point_marker)

            text_marker = Marker()
            text_marker.header.stamp = stamp
            text_marker.header.frame_id = 'map'
            text_marker.ns = 'stretch_anchor_labels'
            text_marker.id = idx * 2 + 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(pos[0])
            text_marker.pose.position.y = float(pos[1])
            text_marker.pose.position.z = 0.42
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.22
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = anchor_key
            marker_array.markers.append(text_marker)

        return marker_array

    def _anchor_pose_array(self):
        poses = PoseArray()
        poses.header.stamp = self.get_clock().now().to_msg()
        poses.header.frame_id = 'map'

        for _, anchor_data in sorted(self.anchors.items()):
            map_pos = self._world_to_map_pos(anchor_data['pos'])
            pose = poses.poses.add() if hasattr(poses.poses, 'add') else None
            if pose is None:
                from geometry_msgs.msg import Pose
                pose = Pose()
                poses.poses.append(pose)
            pose.position.x = float(map_pos[0])
            pose.position.y = float(map_pos[1])
            pose.position.z = 0.05
            pose.orientation.w = 1.0

        return poses

    def _anchor_path_debug(self):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'

        from geometry_msgs.msg import PoseStamped
        for _, anchor_data in sorted(self.anchors.items()):
            map_pos = self._world_to_map_pos(anchor_data['pos'])
            pose_stamped = PoseStamped()
            pose_stamped.header = path.header
            pose_stamped.pose.position.x = float(map_pos[0])
            pose_stamped.pose.position.y = float(map_pos[1])
            pose_stamped.pose.position.z = 0.08
            pose_stamped.pose.orientation.w = 1.0
            path.poses.append(pose_stamped)

        return path

    @staticmethod
    def _path_length(path):
        if len(path.poses) < 2:
            return 0.0
        total = 0.0
        last = path.poses[0].pose.position
        for pose_stamped in path.poses[1:]:
            current = pose_stamped.pose.position
            total += math.hypot(current.x - last.x, current.y - last.y)
            last = current
        return total

    def _path_to_marker(self, path, robot_label='/stretch'):
        marker = Marker()
        marker.header = path.header
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'
        marker.ns = 'stretch2_global_path' if robot_label == '/stretch2' else 'stretch_global_path'
        marker.id = 10 if robot_label == '/stretch2' else 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.07
        if robot_label == '/stretch2':
            marker.color.r = 0.0
            marker.color.g = 0.55
            marker.color.b = 1.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        marker.color.a = 1.0
        marker.points = [pose_stamped.pose.position for pose_stamped in path.poses]
        return marker

    def _path_endpoint_markers(self, path, robot_label='/stretch'):
        if not path.poses:
            return []

        markers = []
        stamp = self.get_clock().now().to_msg()
        start = path.poses[0].pose.position
        target = path.poses[-1].pose.position
        marker_ns = (
            'stretch2_global_path_endpoints'
            if robot_label == '/stretch2'
            else 'stretch_global_path_endpoints'
        )
        marker_id_offset = 10 if robot_label == '/stretch2' else 0

        start_marker = Marker()
        start_marker.header = path.header
        start_marker.header.stamp = stamp
        start_marker.header.frame_id = 'map'
        start_marker.ns = marker_ns
        start_marker.id = marker_id_offset + 1
        start_marker.type = Marker.SPHERE
        start_marker.action = Marker.ADD
        start_marker.pose.position.x = start.x
        start_marker.pose.position.y = start.y
        start_marker.pose.position.z = 0.12
        start_marker.pose.orientation.w = 1.0
        start_marker.scale.x = 0.18
        start_marker.scale.y = 0.18
        start_marker.scale.z = 0.18
        start_marker.color.r = 0.0
        start_marker.color.g = 0.9
        start_marker.color.b = 0.2
        start_marker.color.a = 1.0
        markers.append(start_marker)

        target_marker = Marker()
        target_marker.header = path.header
        target_marker.header.stamp = stamp
        target_marker.header.frame_id = 'map'
        target_marker.ns = marker_ns
        target_marker.id = marker_id_offset + 2
        target_marker.type = Marker.CUBE
        target_marker.action = Marker.ADD
        target_marker.pose.position.x = target.x
        target_marker.pose.position.y = target.y
        target_marker.pose.position.z = 0.12
        target_marker.pose.orientation.w = 1.0
        target_marker.scale.x = 0.2
        target_marker.scale.y = 0.2
        target_marker.scale.z = 0.2
        if robot_label == '/stretch2':
            target_marker.color.r = 0.0
            target_marker.color.g = 0.85
            target_marker.color.b = 1.0
        else:
            target_marker.color.r = 1.0
            target_marker.color.g = 0.55
            target_marker.color.b = 0.0
        target_marker.color.a = 1.0
        markers.append(target_marker)

        return markers

    def _set_nav2_anchor_target(self, robot_label, client, get_pose_fn, anchor_key):
        anchor_key = anchor_key.strip().upper()
        if anchor_key not in self.anchors:
            available = ', '.join(sorted(self.anchors.keys())) if self.anchors else 'none'
            self.get_logger().warn(f'{robot_label} unknown anchor: {anchor_key}. Available: {available}')
            return False
        anchor_data = self.anchors[anchor_key]
        nav_target = NAV2_ANCHOR_STANDOFFS.get(anchor_key, anchor_data['pos'])
        current_pos, _ = get_pose_fn()
        distance = np.linalg.norm(np.array(nav_target[:2]) - current_pos[:2])
        if nav_target is anchor_data['pos']:
            self.get_logger().info(f'{robot_label} Nav2 anchor {anchor_key} requested (distance: {distance:.2f}m)')
        else:
            self.get_logger().info(
                f'{robot_label} Nav2 anchor {anchor_key} using standoff '
                f'({nav_target[0]:.2f}, {nav_target[1]:.2f}); distance={distance:.2f}m'
            )
        return self._send_nav2_goal(robot_label, client, nav_target, anchor_data.get('direction'))

    def _set_global_path_anchor_target(self, anchor_key):
        anchor_key = anchor_key.strip().upper()
        if anchor_key not in self.anchors:
            available = ', '.join(sorted(self.anchors.keys())) if self.anchors else 'none'
            self.get_logger().warn(f'Unknown anchor: {anchor_key}. Available: {available}')
            return False
        anchor_data = self.anchors[anchor_key]
        nav_target = NAV2_ANCHOR_STANDOFFS.get(anchor_key, anchor_data['pos']) if self.local_plan else anchor_data['pos']
        goal_key = f'anchor:{anchor_key}'
        if self._global_goal_is_active('/stretch', goal_key):
            return True
        fallback_target = NAV2_ANCHOR_STANDOFFS.get(anchor_key)
        if not self.local_plan and fallback_target is not None and fallback_target != nav_target:
            self._pending_global_anchor_fallbacks['/stretch'] = {
                'anchor_key': anchor_key,
                'target': fallback_target,
                'direction': anchor_data.get('direction'),
                'goal_key': goal_key,
                'used': False,
            }
        else:
            self._pending_global_anchor_fallbacks.pop('/stretch', None)
        self.get_logger().info(
            f'Global planner anchor {anchor_key} requested at '
            f'({nav_target[0]:.2f}, {nav_target[1]:.2f})'
        )
        if not self._send_global_path_goal(nav_target, anchor_data.get('direction'), robot_label='/stretch'):
            self._pending_global_anchor_fallbacks.pop('/stretch', None)
            self._active_global_goal_keys.pop('/stretch', None)
            return False
        return True

    def _set_r2_global_path_anchor_target(self, anchor_key):
        anchor_key = anchor_key.strip().upper()
        if anchor_key not in self.anchors:
            available = ', '.join(sorted(self.anchors.keys())) if self.anchors else 'none'
            self.get_logger().warn(f'/stretch2 unknown anchor: {anchor_key}. Available: {available}')
            return False
        anchor_data = self.anchors[anchor_key]
        nav_target = NAV2_ANCHOR_STANDOFFS.get(anchor_key, anchor_data['pos']) if self.local_plan else anchor_data['pos']
        goal_key = f'anchor:{anchor_key}'
        if self._global_goal_is_active('/stretch2', goal_key):
            return True
        fallback_target = NAV2_ANCHOR_STANDOFFS.get(anchor_key)
        if not self.local_plan and fallback_target is not None and fallback_target != nav_target:
            self._pending_global_anchor_fallbacks['/stretch2'] = {
                'anchor_key': anchor_key,
                'target': fallback_target,
                'direction': anchor_data.get('direction'),
                'goal_key': goal_key,
                'used': False,
            }
        else:
            self._pending_global_anchor_fallbacks.pop('/stretch2', None)
        self.get_logger().info(
            f'/stretch2 global planner anchor {anchor_key} requested at '
            f'({nav_target[0]:.2f}, {nav_target[1]:.2f})'
        )
        if not self._send_global_path_goal(
            nav_target,
            anchor_data.get('direction'),
            robot_label='/stretch2',
        ):
            self._pending_global_anchor_fallbacks.pop('/stretch2', None)
            self._active_global_goal_keys.pop('/stretch2', None)
            return False
        return True
    
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
        
        if self.global_plan_only:
            self._set_global_path_anchor_target(anchor_key)
            return

        if self.enable_nav2:
            self._set_nav2_anchor_target('/stretch', self.nav2_client, self._get_robot_pose, anchor_key)
            return

        if self._set_anchor_target(
            '/stretch', self.nav_controller, self._get_robot_pose,
            anchor_key, position_tolerance=position_tolerance,
        ):
            self.manual_control = False
    
    def _r2_navigate_to_anchor_callback(self, msg):
        """Handle robot 2 navigate to anchor command."""
        data = msg.data.strip()
        self.get_logger().info(f'/stretch2 navigate_to_anchor command received: {data}')
        position_tolerance = None
        if ':' in data:
            parts = data.split(':')
            anchor_key = parts[0].strip().upper()
            try:
                if len(parts) >= 2:
                    position_tolerance = float(parts[1].strip())
            except (ValueError, IndexError):
                self.get_logger().warn(f'Invalid /stretch2 position_tolerance: {data}')
        else:
            anchor_key = data.strip().upper()
        
        if self.global_plan_only:
            self._set_r2_global_path_anchor_target(anchor_key)
            return

        if self.enable_nav2:
            self._set_nav2_anchor_target('/stretch2', self.r2_nav2_client, self._get_r2_robot_pose, anchor_key)
            return

        if self._set_anchor_target(
            '/stretch2', self.r2_nav_controller, self._get_r2_robot_pose,
            anchor_key, position_tolerance=position_tolerance,
        ):
            self.r2_manual_control = False
    
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

    def _r2_turn_towards_anchor_callback(self, msg):
        """Handle /stretch2 turn towards anchor/angle command."""
        data = msg.data.strip()
        delta_angle = None
        target_angle_degrees = None
        anchor_key = None

        if ':' in data:
            parts = data.split(':')
            first_part = parts[0].strip().upper()

            if first_part == "DEGREES" or first_part.replace('.', '').replace('-', '').isdigit():
                try:
                    if first_part == "DEGREES":
                        if len(parts) >= 2:
                            target_angle_degrees = float(parts[1].strip())
                        if len(parts) >= 3:
                            delta_angle = float(parts[2].strip())
                    else:
                        target_angle_degrees = float(parts[0].strip())
                        if len(parts) >= 2:
                            delta_angle = float(parts[1].strip())
                except (ValueError, IndexError):
                    self.get_logger().warn(f'Invalid /stretch2 angle values in turn_towards command: {data}')
            else:
                anchor_key = first_part
                try:
                    if len(parts) >= 2:
                        delta_angle = float(parts[1].strip())
                except (ValueError, IndexError):
                    self.get_logger().warn(f'Invalid /stretch2 delta_angle in turn_towards command: {data}')
        else:
            anchor_key = data.strip().upper()

        self._handle_r2_anchor_command(
            anchor_key,
            turn_only=True,
            delta_angle=delta_angle,
            target_angle_degrees=target_angle_degrees,
        )
    
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

        if self.global_plan_only:
            self._send_global_path_goal(target_pos, target_direction)
            return

        if self.enable_nav2:
            self._send_nav2_goal('/stretch', self.nav2_client, target_pos, target_direction)
            return
        
        self.nav_controller.set_target(target_pos, target_direction)
        self.manual_control = False
        
        direction_info = f", direction={math.degrees(target_direction):.1f}°" if target_direction else ""
        speed_info = f", speed={speed_percent}%" if speed_percent != DEFAULT_SPEED else ""
        self.get_logger().info(f'Navigating to ({x}, {y}) (distance: {distance:.2f}m{direction_info}{speed_info})')

    def _rviz_goal_pose_callback(self, msg):
        """Plan and follow to an RViz 2D Goal Pose in the map frame."""
        if not self.global_plan_only:
            self.get_logger().warn('RViz goal_pose requires --global-plan-only in this setup')
            return
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._send_global_path_goal_map(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            yaw,
        )

    def _rviz_clicked_point_callback(self, msg):
        """Plan and follow to an RViz Publish Point click in the map frame."""
        if not self.global_plan_only:
            self.get_logger().warn('RViz clicked_point requires --global-plan-only in this setup')
            return
        self._send_global_path_goal_map(
            [msg.point.x, msg.point.y, msg.point.z],
            0.0,
        )
    
    def _r2_navigate_to_position_callback(self, msg):
        """Handle robot 2 navigate to position command."""
        if len(msg.data) < 2:
            self.get_logger().warn('/stretch2 position command requires at least x and y')
            return
        x, y = float(msg.data[0]), float(msg.data[1])
        target_pos = [x, y, 0.0]
        target_direction = float(msg.data[2]) if len(msg.data) > 2 else None
        current_pos, _ = self._get_r2_robot_pose()
        distance = np.linalg.norm(np.array(target_pos[:2]) - current_pos[:2])

        if self.global_plan_only:
            key = f'pos:{x:.3f}:{y:.3f}:{target_direction if target_direction is not None else 0.0:.3f}'
            if not self._global_goal_is_active('/stretch2', key):
                self._send_global_path_goal(target_pos, target_direction, robot_label='/stretch2')
            return

        if self.enable_nav2:
            self._send_nav2_goal('/stretch2', self.r2_nav2_client, target_pos, target_direction)
            return

        self.r2_nav_controller.set_target(target_pos, target_direction)
        self.r2_manual_control = False
        self.get_logger().info(f'/stretch2 navigating to ({x}, {y}) (distance: {distance:.2f}m)')
    
    def _cmd_vel_callback(self, msg):
        """Handle base velocity commands."""
        has_manual_input = abs(msg.linear.x) > 0.05 or abs(msg.angular.z) > 0.05
        
        if has_manual_input and self.nav_controller.is_active():
            self.manual_control = True
            self.nav_controller.cancel()
            self.get_logger().info('Manual control override')
        
        angular_z = -msg.angular.z if self.enable_nav2 else msg.angular.z
        self._set_base_velocity(msg.linear.x, angular_z)

    def _nav2_cmd_vel_callback(self, msg):
        """Handle Nav2 velocity commands using standard ROS angular convention."""
        self._set_base_velocity(msg.linear.x, -msg.angular.z)
    
    def _r2_cmd_vel_callback(self, msg):
        """Handle robot 2 base velocity commands."""
        has_manual_input = abs(msg.linear.x) > 0.05 or abs(msg.angular.z) > 0.05
        if has_manual_input and self.r2_nav_controller.is_active():
            self.r2_manual_control = True
            self.r2_nav_controller.cancel()
            self.get_logger().info('/stretch2 manual control override')
        self._set_r2_base_velocity(msg.linear.x, -msg.angular.z)

    def _r2_nav2_cmd_vel_callback(self, msg):
        """Handle robot 2 Nav2 velocity commands using standard ROS angular convention."""
        if not hasattr(self, '_logged_r2_nav2_cmd_vel'):
            self._logged_r2_nav2_cmd_vel = True
            self.get_logger().info('/stretch2 received first cmd_vel_nav from Nav2 controller')
        self._set_r2_base_velocity(msg.linear.x, -msg.angular.z)
    
    def _set_base_velocity(self, linear_x, angular_z):
        """Set base velocity using planar qvel drive when available, else wheel actuators."""
        self.base_velocity_cmd['linear_x'] = float(linear_x)
        self.base_velocity_cmd['angular_z'] = float(angular_z)
        
        if self.use_planar_base_drive or self.base_freejoint_dof is not None:
            self.ctrl_state['left_wheel_vel'] = 0.0
            self.ctrl_state['right_wheel_vel'] = 0.0
            return
        
        # Convert linear and angular velocities to left/right wheel angular velocities
        # For differential drive: 
        #   v_left_linear = v - (ω * wheel_base/2)
        #   v_right_linear = v + (ω * wheel_base/2)
        # Then convert to wheel angular velocities: ω_wheel = v_linear / wheel_radius
        # MuJoCo velocity actuators: control value directly sets joint angular velocity (rad/s)
        # ctrlrange="-6 6" means control values should be in [-6, 6] rad/s range
        
        wheel_base = 0.3407  # meters (distance between wheels)
        wheel_radius = 0.05  # meters (from XML: size=".05 .0125")
        
        linear_x = self.base_velocity_cmd['linear_x']
        angular_z = self.base_velocity_cmd['angular_z']
        
        # Calculate linear velocities for each wheel (m/s)
        v_left_linear = linear_x - (angular_z * wheel_base / 2.0)
        v_right_linear = linear_x + (angular_z * wheel_base / 2.0)
        
        # Convert to wheel angular velocities (rad/s)
        # Clamp to ctrlrange [-6, 6] rad/s to avoid actuator saturation
        omega_left = np.clip(v_left_linear / wheel_radius, -6.0, 6.0)
        omega_right = np.clip(v_right_linear / wheel_radius, -6.0, 6.0)

        # Use the same wheel sign convention as the navigation controller.
        self.ctrl_state['left_wheel_vel'] = -omega_left
        self.ctrl_state['right_wheel_vel'] = omega_right
    
    def _set_r2_base_velocity(self, linear_x, angular_z):
        """Set robot 2 base velocity."""
        self.r2_base_velocity_cmd['linear_x'] = float(linear_x)
        self.r2_base_velocity_cmd['angular_z'] = float(angular_z)
        
        if self.r2_use_planar_base_drive or self.r2_base_freejoint_dof is not None:
            self.r2_ctrl_state['left_wheel_vel'] = 0.0
            self.r2_ctrl_state['right_wheel_vel'] = 0.0
            return
        
        wheel_base = 0.3407
        wheel_radius = 0.05
        v_left_linear = linear_x - (angular_z * wheel_base / 2.0)
        v_right_linear = linear_x + (angular_z * wheel_base / 2.0)
        omega_left = np.clip(v_left_linear / wheel_radius, -6.0, 6.0)
        omega_right = np.clip(v_right_linear / wheel_radius, -6.0, 6.0)
        self.r2_ctrl_state['left_wheel_vel'] = -omega_left
        self.r2_ctrl_state['right_wheel_vel'] = omega_right
    
    def _apply_planar_base_velocity(self):
        """Apply the stored planar base velocity before each MuJoCo step."""
        if not self.use_planar_base_drive and self.base_freejoint_dof is None:
            return
        
        linear_x = self.base_velocity_cmd['linear_x']
        angular_z = self.base_velocity_cmd['angular_z']
        _, quat = self._get_robot_pose()
        yaw = self._yaw_from_quat(quat)

        if self.use_planar_base_drive:
            self.data.qvel[self.base_drive_dofs['base_x']] = linear_x * math.cos(yaw)
            self.data.qvel[self.base_drive_dofs['base_y']] = linear_x * math.sin(yaw)
            self.data.qvel[self.base_drive_dofs['base_yaw']] = -angular_z
        else:
            dof = self.base_freejoint_dof
            self.data.qvel[dof + 0] = linear_x * math.cos(yaw)
            self.data.qvel[dof + 1] = linear_x * math.sin(yaw)
            self.data.qvel[dof + 5] = -angular_z
    
    def _apply_r2_planar_base_velocity(self):
        """Apply robot 2 stored planar base velocity before each MuJoCo step."""
        if (
            not self.robot2_enabled
            or (not self.r2_use_planar_base_drive and self.r2_base_freejoint_dof is None)
        ):
            return
        
        linear_x = self.r2_base_velocity_cmd['linear_x']
        angular_z = self.r2_base_velocity_cmd['angular_z']
        _, quat = self._get_r2_robot_pose()
        yaw = self._yaw_from_quat(quat)
        if self.r2_use_planar_base_drive:
            self.data.qvel[self.r2_base_drive_dofs['base_x']] = linear_x * math.cos(yaw)
            self.data.qvel[self.r2_base_drive_dofs['base_y']] = linear_x * math.sin(yaw)
            self.data.qvel[self.r2_base_drive_dofs['base_yaw']] = -angular_z
        else:
            dof = self.r2_base_freejoint_dof
            self.data.qvel[dof + 0] = linear_x * math.cos(yaw)
            self.data.qvel[dof + 1] = linear_x * math.sin(yaw)
            self.data.qvel[dof + 5] = -angular_z

    @staticmethod
    def _yaw_from_quat(quat):
        """Extract yaw angle from a MuJoCo [w, x, y, z] quaternion."""
        w, x, y, z = quat
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _normalize_angle(angle):
        """Normalize an angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def _get_robot_pose(self):
        """Get current robot position and orientation."""
        if self.base_link_id >= 0:
            return self.data.xpos[self.base_link_id].copy(), self.data.xquat[self.base_link_id].copy()
        return self.data.qpos[0:3].copy(), np.array([1.0, 0.0, 0.0, 0.0])
    
    def _get_r2_robot_pose(self):
        """Get robot 2 current position and orientation."""
        if self.r2_base_link_id >= 0:
            return self.data.xpos[self.r2_base_link_id].copy(), self.data.xquat[self.r2_base_link_id].copy()
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    def _update_adhesion(self):
        r1_width = float(self.ctrl_state.get('gripper', RESET_POSITIONS['gripper']))
        r1_on = 1.0 if r1_width < GRASP_RELEASE_WIDTH else 0.0
        for key in ('r1_left', 'r1_right'):
            aid = self._adhesion_ids.get(key, -1)
            if aid >= 0:
                self.data.ctrl[aid] = r1_on

        if self.robot2_enabled:
            r2_width = float(self.r2_ctrl_state.get('gripper', RESET_POSITIONS['gripper']))
            r2_on = 1.0 if r2_width < GRASP_RELEASE_WIDTH else 0.0
            for key in ('r2_left', 'r2_right'):
                aid = self._adhesion_ids.get(key, -1)
                if aid >= 0:
                    self.data.ctrl[aid] = r2_on

    def get_marl_observations(self, obs_radius=DEFAULT_MARL_OBS_RADIUS):
        """Return fixed-size MARL observations for all enabled robots."""
        observations = [build_marl_observation(self, '/stretch', obs_radius)]
        if self.robot2_enabled:
            observations.append(build_marl_observation(self, '/stretch2', obs_radius))
        return observations

    def get_marl_obs_size(self):
        """Return per-agent MARL observation sizes."""
        size = len(self.get_marl_observation_layout())
        return [size] * (2 if self.robot2_enabled else 1)

    @staticmethod
    def get_marl_observation_layout():
        return get_marl_observation_layout()
    
    def _log_navigation_status(self, pos, quat, linear_vel, angular_vel):
        """Log navigation status for debugging."""
        target = self.nav_controller.target_pos
        if target is None:
            return
        
        distance = np.linalg.norm(target - pos[:2])
        diff = target - pos[:2]
        desired_angle = np.arctan2(diff[1], diff[0])
        
        current_yaw = self._yaw_from_quat(quat)
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
        if self.enable_nav2:
            nav_status = Bool()
            nav_status.data = False
            self.nav_status_pub.publish(nav_status)
            return

        nav_status = Bool()
        nav_status.data = (
            self.nav_controller.is_active()
            or self._alignment_active
            or self._global_follow_active
            or self._follow_path_goal_pending
            or self._follow_path_goal_handle is not None
        ) and not self.manual_control
        self.nav_status_pub.publish(nav_status)

        if self.global_plan_only and self._global_follow_active and not self.nav_controller.is_active():
            self._advance_global_path_waypoint()
        
        if not self.nav_controller.is_active() or self.manual_control:
            return
        pos, quat = self._get_robot_pose()
        linear_vel, angular_vel = self.nav_controller.get_control(pos, quat)
        
        self._nav_log_counter += 1
        if self._nav_log_counter % (10 if abs(linear_vel) > 0.01 else 50) == 0:
            self._log_navigation_status(pos, quat, linear_vel, angular_vel)
        
        if abs(linear_vel) < 0.001 and abs(angular_vel) > 0.001:
            linear_vel = 0.0
        
        self._set_base_velocity(linear_vel, angular_vel)
        
        if self.nav_controller.has_reached():
            self._set_base_velocity(0.0, 0.0)
            if self.global_plan_only and self._global_follow_active:
                self._advance_global_path_waypoint()
            else:
                self.get_logger().info('✓ Reached target position!')
    
    def _update_r2_navigation(self):
        """Update robot 2 navigation controller and apply its commands."""
        if not self.robot2_enabled:
            return
        if self.enable_nav2:
            nav_status = Bool()
            nav_status.data = False
            self.r2_nav_status_pub.publish(nav_status)
            return
        nav_status = Bool()
        nav_status.data = (
            self.r2_nav_controller.is_active()
            or self.r2_alignment_active
            or self._r2_global_follow_active
            or self._r2_follow_path_goal_pending
            or self._r2_follow_path_goal_handle is not None
        ) and not self.r2_manual_control
        self.r2_nav_status_pub.publish(nav_status)
        if self.global_plan_only and self._r2_global_follow_active and not self.r2_nav_controller.is_active():
            self._advance_global_path_waypoint(robot_label='/stretch2')
        if not self.r2_nav_controller.is_active() or self.r2_manual_control:
            return
        pos, quat = self._get_r2_robot_pose()
        linear_vel, angular_vel = self.r2_nav_controller.get_control(pos, quat)
        if abs(linear_vel) < 0.001 and abs(angular_vel) > 0.001:
            linear_vel = 0.0
        self._set_r2_base_velocity(linear_vel, angular_vel)
        if self.r2_nav_controller.has_reached():
            self._set_r2_base_velocity(0.0, 0.0)
            if self.global_plan_only and self._r2_global_follow_active:
                self._advance_global_path_waypoint(robot_label='/stretch2')
            else:
                self.get_logger().info('✓ /stretch2 reached target position!')
    
    def _get_joint_state(self, joint_name):
        """Get position and velocity for a joint."""
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
    
    def _get_r2_joint_state(self, joint_name):
        """Get robot 2 position and velocity for a joint."""
        if joint_name in self.r2_joint_ids:
            try:
                joint_id = self.r2_joint_ids[joint_name]
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
    
    def publish_r2_joint_states(self):
        """Publish robot 2 current joint states."""
        if not self.robot2_enabled:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for joint_name in JOINT_NAMES:
            pos, vel = self._get_r2_joint_state(joint_name)
            msg.name.append(joint_name)
            msg.position.append(pos)
            msg.velocity.append(vel)
            msg.effort.append(0.0)
        self.r2_joint_state_pub.publish(msg)

    def publish_object_positions(self):
        positions = {}
        for object_label, body_name, _ in OBS_OBJECTS:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                pos = self.data.xpos[body_id]
                positions[object_label] = [float(pos[0]), float(pos[1]), float(pos[2])]
        msg = String()
        msg.data = json.dumps(positions)
        self.object_positions_pub.publish(msg)

    def publish_mapping_sensors(self):
        """Publish odometry, TF, and synthetic 2D lidar scans for navigation/SLAM."""
        stamp = self.get_clock().now().to_msg()
        self._publish_robot_nav_sensors(
            stamp, self._get_robot_pose, self.odom_pub, self.scan_pub,
            self.tf_pub, ODOM_FRAME_ID, BASE_FRAME_ID, LIDAR_FRAME_ID, self.base_link_id
        )
        if self.robot2_enabled:
            self._publish_robot_nav_sensors(
                stamp, self._get_r2_robot_pose, self.r2_odom_pub, self.r2_scan_pub,
                self.r2_tf_pub, R2_ODOM_FRAME_ID, R2_BASE_FRAME_ID, R2_LIDAR_FRAME_ID,
                self.r2_base_link_id
            )

    def _publish_robot_nav_sensors(self, stamp, get_pose_fn, odom_pub, scan_pub, tf_pub,
                                   odom_frame, base_frame, lidar_frame, base_body_id):
        pos, quat = get_pose_fn()
        yaw = self._yaw_from_quat(quat)

        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = odom_frame
        odom_msg.child_frame_id = base_frame
        odom_msg.pose.pose.position.x = float(pos[0])
        odom_msg.pose.pose.position.y = float(pos[1])
        odom_msg.pose.pose.position.z = float(pos[2])
        odom_msg.pose.pose.orientation.w = float(quat[0])
        odom_msg.pose.pose.orientation.x = float(quat[1])
        odom_msg.pose.pose.orientation.y = float(quat[2])
        odom_msg.pose.pose.orientation.z = float(quat[3])
        odom_pub.publish(odom_msg)

        odom_tf = TransformStamped()
        odom_tf.header.stamp = stamp
        odom_tf.header.frame_id = odom_frame
        odom_tf.child_frame_id = base_frame
        odom_tf.transform.translation.x = float(pos[0])
        odom_tf.transform.translation.y = float(pos[1])
        odom_tf.transform.translation.z = float(pos[2])
        odom_tf.transform.rotation.w = float(quat[0])
        odom_tf.transform.rotation.x = float(quat[1])
        odom_tf.transform.rotation.y = float(quat[2])
        odom_tf.transform.rotation.z = float(quat[3])

        laser_pos = pos + np.array([0.004 * math.cos(yaw), 0.004 * math.sin(yaw), 0.1664])
        laser_tf = TransformStamped()
        laser_tf.header.stamp = stamp
        laser_tf.header.frame_id = base_frame
        laser_tf.child_frame_id = lidar_frame
        laser_tf.transform.translation.x = 0.004
        laser_tf.transform.translation.y = 0.0
        laser_tf.transform.translation.z = 0.1664
        laser_tf.transform.rotation.w = 1.0
        laser_tf.transform.rotation.x = 0.0
        laser_tf.transform.rotation.y = 0.0
        laser_tf.transform.rotation.z = 0.0
        self.tf_broadcaster.sendTransform([odom_tf, laser_tf])
        tf_pub.publish(TFMessage(transforms=[odom_tf, laser_tf]))

        scan_msg = LaserScan()
        scan_msg.header.stamp = stamp
        scan_msg.header.frame_id = lidar_frame
        scan_msg.angle_min = LIDAR_ANGLE_MIN
        scan_msg.angle_max = LIDAR_ANGLE_MAX
        scan_msg.angle_increment = (LIDAR_ANGLE_MAX - LIDAR_ANGLE_MIN) / float(LIDAR_NUM_RAYS - 1)
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 1.0 / LIDAR_RATE
        scan_msg.range_min = LIDAR_RANGE_MIN
        scan_msg.range_max = LIDAR_RANGE_MAX
        ranges = []
        dynamic_robot_hit_count = 0
        geomgroup = np.ones(6, dtype=np.uint8)
        geomgroup[2] = 0  # Ignore robot visual meshes.
        geomgroup[3] = 0  # Ignore robot collision meshes.
        bodyexclude = base_body_id if base_body_id >= 0 else -1
        other_robot_pos = self._other_robot_position_for_scan(base_frame)
        for idx in range(LIDAR_NUM_RAYS):
            angle = LIDAR_ANGLE_MIN + idx * scan_msg.angle_increment
            world_angle = yaw + angle
            ray_vec = np.array([math.cos(world_angle), math.sin(world_angle), 0.0], dtype=np.float64)
            geom_id = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                laser_pos.astype(np.float64),
                ray_vec,
                geomgroup,
                1,
                bodyexclude,
                geom_id,
            )
            if other_robot_pos is not None:
                robot_hit = self._ray_circle_intersection(
                    laser_pos[:2],
                    ray_vec[:2],
                    other_robot_pos[:2],
                    DYNAMIC_ROBOT_OBSTACLE_RADIUS,
                )
                if robot_hit is not None and LIDAR_RANGE_MIN <= robot_hit <= LIDAR_RANGE_MAX:
                    distance = robot_hit if distance < LIDAR_RANGE_MIN else min(distance, robot_hit)
                    dynamic_robot_hit_count += 1
            if distance < LIDAR_RANGE_MIN or distance > LIDAR_RANGE_MAX:
                ranges.append(float('inf'))
            else:
                ranges.append(float(distance))
        scan_msg.ranges = ranges
        scan_pub.publish(scan_msg)
        if dynamic_robot_hit_count > 0 and not hasattr(self, '_logged_dynamic_robot_scan'):
            self._logged_dynamic_robot_scan = set()
        if dynamic_robot_hit_count > 0 and base_frame not in self._logged_dynamic_robot_scan:
            self._logged_dynamic_robot_scan.add(base_frame)
            self.get_logger().info(
                f'{base_frame} scan sees other robot as dynamic obstacle '
                f'({dynamic_robot_hit_count} rays)'
            )

    def _other_robot_position_for_scan(self, base_frame):
        if not self.robot2_enabled:
            return None
        if base_frame == BASE_FRAME_ID:
            pos, _ = self._get_r2_robot_pose()
            return pos
        if base_frame == R2_BASE_FRAME_ID:
            pos, _ = self._get_robot_pose()
            return pos
        return None

    @staticmethod
    def _ray_circle_intersection(origin_xy, direction_xy, center_xy, radius):
        origin = np.array(origin_xy, dtype=np.float64)
        direction = np.array(direction_xy, dtype=np.float64)
        center = np.array(center_xy, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return None
        direction = direction / norm
        center_delta = center - origin
        projection = float(np.dot(center_delta, direction))
        if projection <= 0.0:
            return None
        closest_sq = float(np.dot(center_delta, center_delta) - projection * projection)
        radius_sq = radius * radius
        if closest_sq > radius_sq:
            return None
        half_chord = math.sqrt(max(0.0, radius_sq - closest_sq))
        hit_distance = projection - half_chord
        return hit_distance if hit_distance > 0.0 else projection + half_chord
    
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
        if not self.enable_camera:
            return
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
        for i in range(self.model.nsite):
            site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
            if site_name and 'lidar' in site_name.lower():
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
            pub_interval = 1.0 / PUB_RATE        # 30Hz
            render_interval = 1.0 / RENDER_RATE  # 20Hz
            camera_interval = 1.0 / 10.0         # 10Hz (was 30Hz — expensive)
            
            while self.running and viewer.is_running():
                loop_start = time.time()
                
                # Update controllers once per render frame
                self._update_navigation()
                self._update_r2_navigation()
                self._update_arm_reset()
                self._update_r2_arm_reset()
                self._update_joint_movements()
                self._update_r2_joint_movements()
                
                # Apply controls
                for name, actuator_id in self.actuator_ids.items():
                    self.data.ctrl[actuator_id] = self.ctrl_state[name]
                if self.robot2_enabled:
                    for name, actuator_id in self.r2_actuator_ids.items():
                        self.data.ctrl[actuator_id] = self.r2_ctrl_state[name]
                
                # Step multiple times per render frame
                for _ in range(STEPS_PER_CONTROL):
                    self._apply_planar_base_velocity()
                    self._apply_r2_planar_base_velocity()
                    mujoco.mj_step(self.model, self.data)
                self._update_adhesion()
                now = time.time()
                
                if rclpy.ok() and now - prev_pub_time > pub_interval:
                    self.publish_joint_states()
                    self.publish_r2_joint_states()
                    self.publish_mapping_sensors()
                    self.publish_object_positions()
                    prev_pub_time = now
                
                if now - prev_camera_time > camera_interval:
                    self._render_camera()
                    prev_camera_time = now
                
                if now - prev_render_time > render_interval:
                    viewer.sync()
                    prev_render_time = now
                
                # Only sleep if we have spare time (don't force sleep every step)
                elapsed = time.time() - loop_start
                sleep_time = (render_interval * 0.8) - elapsed  # Target 80% of render budget
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
            
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
    parser = argparse.ArgumentParser(description='Stretch MuJoCo ROS 2 simulation')
    parser.add_argument('--world', default='table_world.xml', help='MuJoCo world XML file')
    parser.add_argument('--no-camera', action='store_true', help='Disable RGB camera rendering/publishing')
    parser.add_argument('--nav2', action='store_true', help='Forward anchor/position goals to Nav2 action servers')
    parser.add_argument(
        '--global-plan-only',
        action='store_true',
        help='Use Nav2 planner_server only: compute and publish global paths, but do not run local control',
    )
    parser.add_argument(
        '--local-plan',
        action='store_true',
        default=True,
        help='Compute a global path, then send it to Nav2 controller_server FollowPath for local control (default)',
    )
    parser.add_argument(
        '--no-local-plan',
        dest='local_plan',
        action='store_false',
        help='Use the built-in direct navigation instead of Nav2 global+local planning',
    )
    parser.add_argument('--single-robot', action='store_true', help='Disable second-robot ROS interfaces even if the world contains one')
    parsed, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    node = StretchSimNode(
        world_xml=parsed.world,
        enable_camera=not parsed.no_camera,
        enable_nav2=parsed.nav2 and not parsed.global_plan_only and not parsed.local_plan,
        global_plan_only=parsed.global_plan_only or parsed.local_plan,
        local_plan=parsed.local_plan,
        single_robot=parsed.single_robot,
    )
    node.start_simulation()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
