#!/usr/bin/env python3
"""Interactive command-line controller for Stretch 3 robot via ROS 2."""

import sys
import os
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float64MultiArray, Bool
from navigation import NavigationController
import time
import threading

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install it with: pip install pyyaml")
    sys.exit(1)

# Try to import readline for command history (works on Linux/Mac)
READLINE_AVAILABLE = False
try:
    import readline
    READLINE_AVAILABLE = True
except ImportError:
    pass  # readline not available (e.g., on Windows)


class InteractiveController(Node):
    """Interactive command-line controller that sends ROS 2 commands."""
    
    # Constants
    DEFAULT_SPEED = 50.0
    NAV_TIMEOUT = 30.0
    ARM_TIMEOUT = 10.0
    POSITION_TOLERANCE = 0.05
    CHECK_INTERVAL = 0.1
    
    # Parameter ranges for normalization (min, max)
    PARAM_RANGES = {
        'lift': (0.0, 1.1),
        'arm_extend': (0.0, 0.52),
        'wrist_yaw': (-1.39, 4.42),
        'wrist_pitch': (-1.57, 0.56),
        'wrist_roll': (-3.14, 3.14),
        'gripper': (-0.02, 0.04),
        'x': (-1.0, 2.0),
        'y': (1.0, 4.0),
        'direction': (0.0, 2.0 * math.pi),
        'speed': (0.0, 100.0)
    }
    
    # Joint name mappings
    JOINT_NAME_MAP = {
        'joint_lift': 'lift',
        'joint_wrist_yaw': 'wrist_yaw',
        'joint_wrist_pitch': 'wrist_pitch',
        'joint_wrist_roll': 'wrist_roll',
        'joint_head_pan': 'head_pan',
        'joint_head_tilt': 'head_tilt',
    }
    ARM_JOINTS = ['joint_arm_l0', 'joint_arm_l1', 'joint_arm_l2', 'joint_arm_l3']
    JOINT_ORDER = ['lift', 'arm_extend', 'wrist_yaw', 'wrist_pitch', 'wrist_roll', 'gripper', 'head_pan', 'head_tilt']
    
    def __init__(self, actions_file='actions.yaml'):
        super().__init__('interactive_controller')

    
        # Storage for computed IK values
        self._ik_computed_values = {
            'wrist_yaw': None,
            'wrist_pitch': None,
            'wrist_roll': None,
            'arm_extend': None,
            'lift': None
        }
        
        # Storage for alignment target
        self._alignment_target = None

        self.actions_file = os.path.join(os.path.dirname(__file__), actions_file)
        self.micro_actions = {}
        self.macro_actions = {}
        self._load_actions()
        
        self._setup_publishers()
        self._setup_subscribers()
        self._init_joint_state()
        self._init_wait_state()
        self._init_readline()
        self._print_welcome()
    
    @staticmethod
    def _normalize_to_range(value, min_val, max_val):
        """Normalize a value from 0-1 range to actual range [min_val, max_val]."""
        normalized = max(0.0, min(1.0, float(value)))
        return min_val + normalized * (max_val - min_val)
    
    @staticmethod
    def _normalize_speed(value):
        """Normalize speed from 0-1 to 0-100 percentage."""
        normalized = max(0.0, min(1.0, float(value)))
        return normalized * 100.0
    
    @staticmethod
    def _clamp(value, min_val=0.0, max_val=1.0):
        """Clamp value to range."""
        return max(min_val, min(max_val, float(value)))
    
    def _format_speed_str(self, speed_percent):
        """Format speed string for display."""
        return f" (speed: {speed_percent:.0f}%)" if speed_percent != self.DEFAULT_SPEED else ""
    
    def _get_speed(self, params, default=None):
        """Extract and normalize speed from parameters."""
        default = default if default is not None else 0.5
        speed_normalized = self._clamp(params.get('speed', default))
        return self._normalize_speed(speed_normalized)
    
    def _require_param(self, params, param_name, error_msg):
        """Require a parameter and return it, or print error and return None."""
        value = params.get(param_name)
        if value is None:
            print(f"Error: {error_msg}")
            return None
        return value
    
    def _init_readline(self):
        """Initialize readline for command history."""
        if not READLINE_AVAILABLE:
            return
        
        histfile = os.path.join(os.path.expanduser("~"), ".stretch_controller_history")
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            pass
        
        readline.set_history_length(1000)
        readline.parse_and_bind("tab: complete")
        
        def completer(text, state):
            options = [name for name in list(self.micro_actions.keys()) + list(self.macro_actions.keys()) 
                      if name.startswith(text)]
            return options[state] if state < len(options) else None
        
        readline.set_completer(completer)
    
    def _load_actions(self):
        """Load micro and macro actions from YAML file."""
        try:
            with open(self.actions_file, 'r') as f:
                data = yaml.safe_load(f)
            
            self.micro_actions = {action['name']: action for action in data.get('micro_actions', [])}
            self.macro_actions = {action['name']: action for action in data.get('macro_actions', [])}
            
            self.get_logger().info(
                f'Loaded {len(self.micro_actions)} micro actions and '
                f'{len(self.macro_actions)} macro actions'
            )
        except FileNotFoundError:
            self.get_logger().error(f'Actions file not found: {self.actions_file}')
            sys.exit(1)
        except yaml.YAMLError as e:
            self.get_logger().error(f'Error parsing YAML file: {e}')
            sys.exit(1)
        except Exception as e:
            self.get_logger().error(f'Error loading actions: {e}')
            sys.exit(1)
    
    def _setup_publishers(self):
        """Initialize ROS 2 publishers."""
        self.cmd_vel_pub = self.create_publisher(Twist, '/stretch/cmd_vel', 10)
        self.anchor_pub = self.create_publisher(String, '/stretch/navigate_to_anchor', 10)
        self.turn_towards_pub = self.create_publisher(String, '/stretch/turn_towards_anchor', 10)
        self.joint_cmd_pub = self.create_publisher(Float64MultiArray, '/stretch/joint_command', 10)
        self.reset_pub = self.create_publisher(String, '/stretch/reset_arm', 10)
        self.position_pub = self.create_publisher(Float64MultiArray, '/stretch/navigate_to_position', 10)
        #
        self.align_target_pub = self.create_publisher(String, '/stretch/align_with_target', 10)
        self.compute_ik_pub = self.create_publisher(String, '/stretch/compute_ik', 10)
        #
    def _setup_subscribers(self):
        """Initialize ROS 2 subscribers."""
        self.joint_state_sub = self.create_subscription(
            JointState, '/stretch/joint_states', self._joint_state_callback, 10
        )
        self.nav_status_sub = self.create_subscription(
            Bool, '/stretch/navigation_active', self._navigation_status_callback, 10
        )
        self.ik_result_sub = self.create_subscription(
            Float64MultiArray, '/stretch/ik_result', self._ik_result_callback, 10
        )
        self.current_joint_states = {}
        self.navigation_active = False
    
    def _joint_state_callback(self, msg):
        """Update current joint states from robot."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_states[name] = msg.position[i]
    
    def _navigation_status_callback(self, msg):
        """Update navigation status from robot."""
        self.navigation_active = msg.data
    
    def _init_joint_state(self):
        """Initialize joint state tracker."""
        self.joint_state = {joint: 0.0 for joint in self.JOINT_ORDER}
        self.target_joint_states = {}
    
    def _init_wait_state(self):
        """Initialize wait state tracking."""
        self.current_joint_states = {}
        self.wait_event = threading.Event()
    
    def _sync_joint_state_from_robot(self):
        """Update self.joint_state from actual robot joint states."""
        # Update mapped joints
        for robot_joint_name, controller_key in self.JOINT_NAME_MAP.items():
            if robot_joint_name in self.current_joint_states:
                self.joint_state[controller_key] = self.current_joint_states[robot_joint_name]
        
        # Handle arm_extend (sum of all arm joint segments)
        arm_total = sum(self.current_joint_states.get(j, 0.0) for j in self.ARM_JOINTS)
        if any(j in self.current_joint_states for j in self.ARM_JOINTS):
            self.joint_state['arm_extend'] = arm_total
        
        # Handle gripper
        if 'joint_gripper_slide' in self.current_joint_states:
            self.joint_state['gripper'] = self.current_joint_states['joint_gripper_slide']
        
        # Initialize wrist_pitch and wrist_roll if not already set
        if 'wrist_pitch' not in self.joint_state:
            self.joint_state['wrist_pitch'] = 0.0
        if 'wrist_roll' not in self.joint_state:
            self.joint_state['wrist_roll'] = 0.0
    
    def _publish_single_joint_command(self, joint_name, value, speed_percent=DEFAULT_SPEED):
        """Publish a single joint command.
        
        Args:
            joint_name: Name of the joint (e.g., 'lift', 'arm_extend')
            value: Target value for the joint
            speed_percent: Speed percentage (0-100)
        """
        if joint_name not in self.JOINT_ORDER:
            self.get_logger().warn(f'Unknown joint name: {joint_name}')
            return
        
        joint_index = self.JOINT_ORDER.index(joint_name)
        msg = Float64MultiArray()
        # Format: [joint_index, value, speed_percent]
        msg.data = [float(joint_index), float(value), float(speed_percent)]
        self.joint_cmd_pub.publish(msg)
    
    def _publish_multiple_joint_commands(self, joint_commands, speed_percent=DEFAULT_SPEED):
        """Publish multiple joint commands as separate messages.
        
        Args:
            joint_commands: Dictionary mapping joint names to target values
                          (e.g., {'lift': 0.3, 'arm_extend': 0.2})
            speed_percent: Speed percentage (0-100) to apply to all joints
        """
        for joint_name, value in joint_commands.items():
            self._publish_single_joint_command(joint_name, value, speed_percent)
    
    def _print_table(self, title, headers, rows, width=120):
        """Print a formatted table."""
        print(f"\n{title.center(width)}")
        print("=" * width)
        print(" ".join(f"{h:<{width//len(headers)-1}}" for h in headers))
        print("=" * width)
        for row in rows:
            print(" ".join(f"{str(cell)[:width//len(headers)-2]:<{width//len(headers)-1}}" for cell in row))
        print("=" * width)
    
    def _truncate(self, text, max_len):
        """Truncate text with ellipsis if too long."""
        return text[:max_len-3] + "..." if len(text) > max_len else text
    
    def _extract_usage(self, description):
        """Extract usage from description (text in parentheses)."""
        if '(' in description and 'Usage:' in description:
            start = description.find('Usage:') + 6
            end = description.find(')', start)
            if end > start:
                return description[start:end].strip()
        return ""
    
    def _print_welcome(self):
        """Print welcome message and available actions in table format."""
        print("\n" + "="*120)
        print("Stretch 3 Interactive Controller".center(120))
        print("="*120)
        
        # Micro Actions table
        micro_rows = []
        for name, action in sorted(self.micro_actions.items()):
            desc = action.get('description', 'No description')
            desc_clean = desc.split('(Usage:')[0].strip() if '(Usage:' in desc else desc
            usage = self._extract_usage(desc)
            micro_rows.append([
                self._truncate(name, 18),
                self._truncate(usage, 50),
                self._truncate(desc_clean, 52)
            ])
        self._print_table("MICRO ACTIONS (Primitive Commands)", 
                         ['Action Name', 'Usage Example', 'Description'], micro_rows)
        
        # Macro Actions table
        macro_rows = []
        for name, action in sorted(self.macro_actions.items()):
            desc = action.get('description', 'No description')
            sequence = action.get('sequence', [])
            status = "✓ Implemented" if sequence else "○ Not implemented"
            macro_rows.append([
                self._truncate(name, 22),
                self._truncate(desc, 70),
                status
            ])
        self._print_table("MACRO ACTIONS (Complex Behaviors)",
                         ['Action Name', 'Description', 'Status'], macro_rows)
        
        # Commands table
        commands = [
            ("help", "Show this help message"),
            ("help <action_name>", "Show detailed help for a specific action"),
            ("list", "List all available actions"),
            ("exit / quit", "Exit the controller")
        ]
        self._print_table("CONTROLLER COMMANDS", ['Command', 'Description'],
                         [[cmd, desc] for cmd, desc in commands])
        
        print("\n" + "="*120)
        print("Usage: Type an action name followed by parameters (if needed)")
        print("All parameters use normalized 0-1 range: 0=minimum, 0.5=middle/default, 1=maximum")
        print("Example: go_to_anchor anchor=A speed=0.5")
        print("="*120 + "\n")
    
    def _print_help(self, action_name=None):
        """Print help information in table format."""
        if action_name is None:
            self._print_welcome()
            return
        
        # Check micro actions
        if action_name in self.micro_actions:
            action = self.micro_actions[action_name]
            print("\n" + "="*80)
            print(f"Action: {action_name}".center(80))
            print("="*80)
            
            rows = [
                ['Type', action.get('type', 'unknown')],
                ['Description', action.get('description', 'No description')]
            ]
            self._print_table("", ['Property', 'Value'], rows, width=80)
            
            params = action.get('parameters', {})
            param_rows = [[p, t] for p, t in params.items()] if params else [['None', 'N/A']]
            self._print_table("", ['Parameter', 'Type'], param_rows, width=80)
            print()
            return
        
        # Check macro actions
        if action_name in self.macro_actions:
            action = self.macro_actions[action_name]
            print("\n" + "="*80)
            print(f"Action: {action_name}".center(80))
            print("="*80)
            
            rows = [
                ['Type', 'Macro Action'],
                ['Description', action.get('description', 'No description')]
            ]
            self._print_table("", ['Property', 'Value'], rows, width=80)
            
            sequence = action.get('sequence', [])
            if sequence:
                step_rows = []
                for i, step in enumerate(sequence, 1):
                    step_action = step.get('action', 'unknown')
                    step_params = step.get('parameters', {})
                    params_str = ", ".join(f"{k}={v}" for k, v in step_params.items()) if step_params else "None"
                    step_rows.append([i, step_action, self._truncate(params_str, 50)])
                self._print_table("", ['Step', 'Action', 'Parameters'], step_rows, width=80)
            else:
                self._print_table("", ['Step', 'Action', 'Parameters'], 
                                [['N/A', 'Not yet implemented', 'N/A']], width=80)
            print()
            return
        
        print(f"\nAction '{action_name}' not found.\n")
        print("Available actions:")
        print("  Micro:", ", ".join(sorted(self.micro_actions.keys())))
        print("  Macro:", ", ".join(sorted(self.macro_actions.keys())))
        print()
    
    def _list_actions(self):
        """List all available actions in table format."""
        micro_list = sorted(self.micro_actions.keys())
        macro_list = sorted(self.macro_actions.keys())
        max_len = max(len(micro_list), len(macro_list))
        
        rows = []
        for i in range(max_len):
            rows.append([
                micro_list[i] if i < len(micro_list) else "",
                macro_list[i] if i < len(macro_list) else ""
            ])
        
        self._print_table("Available Actions", ['Micro Actions', 'Macro Actions'], rows, width=80)
    
    def _parse_command(self, command_line):
        """Parse command line input."""
        parts = command_line.strip().split()
        if not parts:
            return None, {}
        
        action_name = parts[0]
        params = {}
        
        # Parse parameters (key=value format)
        for part in parts[1:]:
            if '=' in part:
                key, value = part.split('=', 1)
                try:
                    params[key] = float(value) if '.' in value else int(value)
                except ValueError:
                    params[key] = value
            else:
                # Backward compatibility: positional parameters
                if 'anchor' not in params:
                    params['anchor'] = part
                elif 'x' not in params:
                    params['x'] = float(part)
                elif 'y' not in params:
                    params['y'] = float(part)
        
        return action_name, params
    
    # Navigation action handlers
    def _handle_go_to_anchor(self, params):
        """Handle go_to_anchor action."""
        anchor = params.get('anchor', '')
        if not anchor:
            print("Error: 'anchor' parameter required (e.g., go_to_anchor anchor=A)")
            return False
        speed_percent = self._get_speed(params)
        position_tolerance = params.get('position_tolerance', 0.15)  # Default 0.15 meters
        self._go_to_anchor(anchor.upper(), speed_percent, position_tolerance)
        return True
    
    def _handle_turn_towards(self, params):
        """Handle turn_towards action."""
        anchor = params.get('anchor', '')
        degrees = params.get('degrees')
        speed_percent = self._get_speed(params)
        delta_angle = params.get('delta_angle', 5.0)  # Default 5.0 degrees
        
        if not anchor and degrees is None:
            print("Error: Either 'anchor' or 'degrees' parameter required (e.g., turn_towards anchor=ORIGIN OR turn_towards degrees=90.0)")
            return False
        
        if anchor and degrees is not None:
            print("Error: 'anchor' and 'degrees' are mutually exclusive. Use one or the other.")
            return False
        
        if degrees is not None:
            self._turn_towards(None, speed_percent, delta_angle, degrees)
        else:
            self._turn_towards(anchor.upper(), speed_percent, delta_angle, None)
        return True
    
    def _handle_go_to_position(self, params):
        """Handle go_to_position action."""
        x_normalized = self._require_param(params, 'x', 
            "'x' and 'y' parameters required (0-1 range, e.g., go_to_position x=0.5 y=0.5)")
        y_normalized = self._require_param(params, 'y', 
            "'x' and 'y' parameters required (0-1 range, e.g., go_to_position x=0.5 y=0.5)")
        if x_normalized is None or y_normalized is None:
            return False
        
        x = self._normalize_to_range(x_normalized, *self.PARAM_RANGES['x'])
        y = self._normalize_to_range(y_normalized, *self.PARAM_RANGES['y'])
        direction_normalized = params.get('direction')
        direction = None
        if direction_normalized is not None:
            direction = self._normalize_to_range(direction_normalized, *self.PARAM_RANGES['direction'])
        
        speed_percent = self._get_speed(params)
        self._go_to_position(x, y, direction, speed_percent)
        return True
    
    # Arm control action handlers
    def _handle_reset_arm(self, params):
        """Handle reset_arm action."""
        speed_percent = self._get_speed(params)
        self._reset_arm(speed_percent)
        return True
    

    #
    def _handle_align_with_target(self, params):
        """Handle align_with_target action."""
        target = params.get('target', '')
        if not target:
            print("Error: 'target' parameter required (e.g., align_with_target target=tomato1)")
            return False
            
        
        speed_percent = self._get_speed(params)
        delta_angle = params.get('delta_angle', 5.0)
        
        # Store target name for use by simulation node
        self._alignment_target = target
        
        # Publish alignment request to simulation node
        self._align_with_target(target, speed_percent, delta_angle)
        return True
    
    def _handle_compute_ik(self, params):
        """Handle compute_ik action - requests IK computation from simulation node."""
        target = params.get('target', '')
        if not target:
            print("Error: 'target' parameter required (e.g., compute_ik target=tomato3)")
            return False
        
        speed_percent = self._get_speed(params)
        
        # Publish IK computation request to simulation node
        self._compute_ik(target, speed_percent)
        return True
    def _ik_result_callback(self, msg):
        """Receive and store IK computation results."""
        if len(msg.data) < 4:
            return
        
        success = msg.data[3] > 0.5
        
        if success:
            # Results are: [lift_raw, arm_raw, wrist_yaw_raw, success_flag]
            # Note: IK solver may not compute wrist_pitch and wrist_roll, so they remain None
            # Store RAW values directly (no normalization)
            self._ik_computed_values['lift'] = float(msg.data[0])
            self._ik_computed_values['arm_extend'] = float(msg.data[1])
            self._ik_computed_values['wrist_yaw'] = float(msg.data[2])
            # wrist_pitch and wrist_roll are not computed by IK, so leave as None
            
            print(f"  ✓ IK computed: wrist_yaw={msg.data[2]:.3f}, arm={msg.data[1]:.3f}, lift={msg.data[0]:.3f}")
        else:
            print("  ✗ IK computation failed - target unreachable")
            # Clear stored values
            self._ik_computed_values = {
                'wrist_yaw': None, 
                'wrist_pitch': None,
                'wrist_roll': None,
                'arm_extend': None, 
                'lift': None
            }
    
    def _handle_extend_arm(self, params):
        """Handle extend_arm action - supports both manual and IK-computed values."""
        
        use_ik = params.get('use_ik', 0)
        
        if use_ik:
            if self._ik_computed_values['arm_extend'] is None:
                print("Error: No IK arm value available. Call compute_ik first.")
                return False
            
            length = self._ik_computed_values['arm_extend']  # Already in meters - use directly
            print(f"  → Using IK arm value: {length:.3f}m")
        else:
            length_normalized = self._require_param(params, 'length',
                "'length' parameter required (0-1 range, e.g., extend_arm length=0.5)")
            if length_normalized is None:
                return False
            length = self._normalize_to_range(length_normalized, *self.PARAM_RANGES['arm_extend'])
        
        speed_percent = self._get_speed(params)
        self._extend_arm(length, speed_percent)
        return True
    def _handle_rotate_wrist(self, params):
        """Handle rotate_wrist action - supports both manual and IK-computed values."""
        
        use_ik = params.get('use_ik', 0)
        
        if use_ik:
            if self._ik_computed_values['wrist_yaw'] is None:
                print("Error: No IK wrist value available. Call compute_ik first.")
                return False
            
            angle = self._ik_computed_values['wrist_yaw']  # Already in radians - use directly
            print(f"  → Using IK wrist yaw value: {angle:.3f} rad")
        else:
            angle_normalized = self._require_param(params, 'angle',
                "'angle' parameter required (0-1 range, e.g., rotate_wrist angle=0.5)")
            if angle_normalized is None:
                return False
            angle = self._normalize_to_range(angle_normalized, *self.PARAM_RANGES['wrist_yaw'])
        
        speed_percent = self._get_speed(params)
        self._rotate_wrist(angle, speed_percent)
        return True
    
    def _handle_pitch_wrist(self, params):
        """Handle pitch_wrist action."""
        angle_normalized = self._require_param(params, 'angle',
            "'angle' parameter required (0-1 range, e.g., pitch_wrist angle=0.5)")
        if angle_normalized is None:
            return False
        angle = self._normalize_to_range(angle_normalized, *self.PARAM_RANGES['wrist_pitch'])
        speed_percent = self._get_speed(params)
        self._pitch_wrist(angle, speed_percent)
        return True
    
    def _handle_roll_wrist(self, params):
        """Handle roll_wrist action."""
        angle_normalized = self._require_param(params, 'angle',
            "'angle' parameter required (0-1 range, e.g., roll_wrist angle=0.5)")
        if angle_normalized is None:
            return False
        angle = self._normalize_to_range(angle_normalized, *self.PARAM_RANGES['wrist_roll'])
        speed_percent = self._get_speed(params)
        self._roll_wrist(angle, speed_percent)
        return True
    def _handle_elevate_arm(self, params):
        """Handle elevate_arm action - supports both manual and IK-computed values."""
        
        use_ik = params.get('use_ik', 0)
        
        if use_ik:
            if self._ik_computed_values['lift'] is None:
                print("Error: No IK lift value available. Call compute_ik first.")
                return False
            
            height = self._ik_computed_values['lift']  # Already in meters - use directly
            print(f"  → Using IK lift value: {height:.3f}m")
        else:
            height_normalized = self._require_param(params, 'height',
                "'height' parameter required (0-1 range, e.g., elevate_arm height=0.5)")
            if height_normalized is None:
                return False
            height = self._normalize_to_range(height_normalized, *self.PARAM_RANGES['lift'])
        
        speed_percent = self._get_speed(params)
        self._elevate_arm(height, speed_percent)
        return True
    def _handle_open_gripper(self, params):
        """Handle open_gripper action."""
        speed_percent = self._get_speed(params)
        self._set_gripper(self.PARAM_RANGES['gripper'][1], speed_percent)
        return True
    
    def _handle_close_gripper(self, params):
        """Handle close_gripper action."""
        speed_percent = self._get_speed(params)
        self._set_gripper(self.PARAM_RANGES['gripper'][0], speed_percent)
        return True
    
    def _handle_set_gripper(self, params):
        """Handle set_gripper action."""
        width_normalized = self._require_param(params, 'width',
            "'width' parameter required (0-1 range, e.g., set_gripper width=0.5)")
        if width_normalized is None:
            return False
        width = self._normalize_to_range(width_normalized, *self.PARAM_RANGES['gripper'])
        speed_percent = self._get_speed(params)
        self._set_gripper(width, speed_percent)
        return True
    
    # Utility action handlers
    def _handle_wait(self, params):
        """Handle wait action."""
        duration = params.get('duration', 1.0)
        self._wait(duration)
        return True
    
    def _handle_wait_for_arm(self, params):
        """Handle wait_for_arm action."""
        timeout = params.get('timeout', self.ARM_TIMEOUT)
        return self._wait_for_arm(timeout)
    
    # Action dispatch dictionary
    NAVIGATION_HANDLERS = {
        'go_to_anchor': '_handle_go_to_anchor',
        'turn_towards': '_handle_turn_towards',
        'go_to_position': '_handle_go_to_position',
        #
        'align_with_target': '_handle_align_with_target',
        #
    }
    
    ARM_HANDLERS = {
        'reset_arm': '_handle_reset_arm',
        'elevate_arm': '_handle_elevate_arm',
        'extend_arm': '_handle_extend_arm',
        'rotate_wrist': '_handle_rotate_wrist',
        'pitch_wrist': '_handle_pitch_wrist',
        'roll_wrist': '_handle_roll_wrist',
        'open_gripper': '_handle_open_gripper',
        'close_gripper': '_handle_close_gripper',
        'set_gripper': '_handle_set_gripper',
    }
    
    UTILITY_HANDLERS = {
        'wait': '_handle_wait',
        'wait_for_arm': '_handle_wait_for_arm',
        'compute_ik': '_handle_compute_ik',
    }
    
    def _execute_micro_action(self, action_name, params):
        """Execute a micro action."""
        if action_name not in self.micro_actions:
            print(f"Error: Micro action '{action_name}' not found")
            return False
        
        action = self.micro_actions[action_name]
        action_type = action.get('type', '')
        
        try:
            # Dispatch to appropriate handler
            handler_map = {
                'navigation': self.NAVIGATION_HANDLERS,
                'arm_control': self.ARM_HANDLERS,
                'utility': self.UTILITY_HANDLERS,
            }
            
            handlers = handler_map.get(action_type, {})
            handler_name = handlers.get(action_name)
            
            if handler_name:
                handler = getattr(self, handler_name)
                return handler(params)
            else:
                print(f"Error: Unknown action type '{action_type}' or handler not found")
                return False
        
        except Exception as e:
            print(f"Error executing action: {e}")
            return False
    
    def _execute_macro_action(self, action_name, params):
        """Execute a macro action."""
        if action_name not in self.macro_actions:
            print(f"Error: Macro action '{action_name}' not found")
            return False
        
        action = self.macro_actions[action_name]
        sequence = action.get('sequence', [])
        
        if not sequence:
            print(f"Error: Macro action '{action_name}' is not yet implemented")
            return False
        
        print(f"Executing macro action: {action_name}")
        
        for i, step in enumerate(sequence, 1):
            step_action = step.get('action')
            step_params = step.get('parameters', {})
            step_params.update(params)  # Merge macro params with step params
            
            print(f"  Step {i}: {step_action}")
            
            if step_action in self.micro_actions:
                if not self._execute_micro_action(step_action, step_params):
                    print(f"Error: Failed at step {i}")
                    return False
            elif step_action in self.macro_actions:
                if not self._execute_macro_action(step_action, step_params):
                    print(f"Error: Failed at step {i}")
                    return False
            else:
                print(f"Error: Unknown action '{step_action}' in sequence")
                return False
        
        print(f"✓ Macro action '{action_name}' completed\n")
        return True
    
    # Micro action implementations
    def _go_to_anchor(self, anchor, speed_percent=DEFAULT_SPEED, position_tolerance=0.15):
        """Navigate to an anchor point (no direction alignment)."""
        msg = String()
        # Include position_tolerance in message only if not default (for backward compatibility)
        if abs(position_tolerance - 0.15) > 0.001:
            msg.data = f"{anchor}:{position_tolerance:.3f}"
        else:
            msg.data = anchor
        
        self.anchor_pub.publish(msg)
        pos_tol_str = f", pos_tol={position_tolerance:.3f}m" if abs(position_tolerance - 0.15) > 0.001 else ""
        print(f"→ Navigating to anchor {anchor}{self._format_speed_str(speed_percent)}{pos_tol_str}")
        self._wait_for_navigation(timeout=self.NAV_TIMEOUT)
    
    def _turn_towards(self, anchor, speed_percent=DEFAULT_SPEED, delta_angle=5.0, degrees=None):
        """Turn robot towards an anchor point or absolute angle without moving."""
        msg = String()
        
        if degrees is not None:
            # Absolute angle mode: format "degrees:target_angle:delta_angle"
            if abs(delta_angle - 5.0) > 0.01:
                msg.data = f"degrees:{degrees:.1f}:{delta_angle:.1f}"
            else:
                msg.data = f"degrees:{degrees:.1f}"
            delta_str = f", delta_angle={delta_angle:.1f}°" if abs(delta_angle - 5.0) > 0.01 else ""
            print(f"→ Turning to absolute angle {degrees:.1f}°{self._format_speed_str(speed_percent)}{delta_str}")
        else:
            # Position-based mode: format "ANCHOR:delta_angle"
            msg.data = f"{anchor}:{delta_angle:.1f}" if abs(delta_angle - 5.0) > 0.01 else anchor
            delta_str = f", delta_angle={delta_angle:.1f}°" if abs(delta_angle - 5.0) > 0.01 else ""
            print(f"→ Turning towards anchor {anchor}{self._format_speed_str(speed_percent)}{delta_str}")
        
        self.turn_towards_pub.publish(msg)
        self._wait_for_navigation(timeout=self.NAV_TIMEOUT)
    
    def _wait_for_navigation(self, timeout=NAV_TIMEOUT):
        """Wait until navigation completes. First waits for start, then for finish."""
        start_time = time.time()
        start_timeout = 2.0
        
        # Wait for navigation to start
        while time.time() - start_time < start_timeout:
            if self.navigation_active:
                break
            rclpy.spin_once(self, timeout_sec=self.CHECK_INTERVAL)
            time.sleep(self.CHECK_INTERVAL)
        
        if not self.navigation_active:
            print(f"  ⚠ Navigation did not start within {start_timeout}s")
            return False
        
        # Wait for navigation to complete
        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=self.CHECK_INTERVAL)
            if not self.navigation_active:
                return True
            time.sleep(self.CHECK_INTERVAL)
        
        print(f"  ⚠ Navigation timeout after {timeout}s")
        return False
    
    def _go_to_position(self, x, y, direction=None, speed_percent=DEFAULT_SPEED):
        """Navigate to a specific position."""
        msg = Float64MultiArray()
        msg.data = [float(x), float(y)]
        if direction is not None:
            msg.data.append(float(direction))
        msg.data.append(float(speed_percent))
        self.position_pub.publish(msg)
        
        direction_str = f", direction={direction:.2f}" if direction is not None else ""
        speed_str = f", speed={speed_percent:.0f}%" if speed_percent != self.DEFAULT_SPEED else ""
        print(f"→ Navigating to position ({x:.2f}, {y:.2f}{direction_str}{speed_str})")
        self._wait_for_navigation(timeout=self.NAV_TIMEOUT)
    
    def _reset_arm(self, speed_percent=DEFAULT_SPEED):
        """Reset arm to default position with speed percentage."""
        msg = String()
        msg.data = f'reset:{speed_percent}'
        self.reset_pub.publish(msg)
        print(f"→ Resetting arm to default position{self._format_speed_str(speed_percent)}")
    
    def _elevate_arm(self, height, speed_percent=DEFAULT_SPEED):
        """Set lift height with speed control."""
        self.joint_state['lift'] = height
        self._publish_single_joint_command('lift', height, speed_percent)
        print(f"→ Elevating arm to height {height:.3f}{self._format_speed_str(speed_percent)}")
    
    def _extend_arm(self, length, speed_percent=DEFAULT_SPEED):
        """Set arm extension with speed control."""
        self.joint_state['arm_extend'] = length
        self._publish_single_joint_command('arm_extend', length, speed_percent)
        print(f"→ Extending arm to length {length:.3f}{self._format_speed_str(speed_percent)}")
    
    def _rotate_wrist(self, angle, speed_percent=DEFAULT_SPEED):
        """Set wrist yaw angle with speed control."""
        self.joint_state['wrist_yaw'] = angle
        self._publish_single_joint_command('wrist_yaw', angle, speed_percent)
        print(f"→ Rotating wrist yaw to angle {angle:.3f}{self._format_speed_str(speed_percent)}")
    
    def _pitch_wrist(self, angle, speed_percent=DEFAULT_SPEED):
        """Set wrist pitch angle with speed control."""
        self.joint_state['wrist_pitch'] = angle
        self._publish_single_joint_command('wrist_pitch', angle, speed_percent)
        print(f"→ Pitching wrist to angle {angle:.3f}{self._format_speed_str(speed_percent)}")
    
    def _roll_wrist(self, angle, speed_percent=DEFAULT_SPEED):
        """Set wrist roll angle with speed control."""
        self.joint_state['wrist_roll'] = angle
        self._publish_single_joint_command('wrist_roll', angle, speed_percent)
        print(f"→ Rolling wrist to angle {angle:.3f}{self._format_speed_str(speed_percent)}")
    
    def _set_gripper(self, width, speed_percent=DEFAULT_SPEED):
        """Set gripper opening width with speed control."""
        self.joint_state['gripper'] = width
        self._publish_single_joint_command('gripper', width, speed_percent)
        print(f"→ Setting gripper to width {width:.3f}{self._format_speed_str(speed_percent)}")
    #
    def _align_with_target(self, target_name, speed_percent=DEFAULT_SPEED, delta_angle=5.0):
        """Request robot to align with target object."""
        msg = String()
        msg.data = f"{target_name}:{delta_angle:.1f}"
        self.align_target_pub.publish(msg)
        
        delta_str = f", delta_angle={delta_angle:.1f}°" if abs(delta_angle - 5.0) > 0.01 else ""
        print(f"→ Aligning with target '{target_name}'{self._format_speed_str(speed_percent)}{delta_str}")
        self._wait_for_navigation(timeout=self.NAV_TIMEOUT)
    def _compute_ik(self, target_name, speed_percent=DEFAULT_SPEED):
        """Request IK computation for target object."""
        # Clear previous IK values
        self._ik_computed_values = {
            'wrist_yaw': None,
            'wrist_pitch': None,
            'wrist_roll': None,
            'arm_extend': None,
            'lift': None
        }
        
        # Publish IK request
        msg = String()
        msg.data = target_name
        self.compute_ik_pub.publish(msg)
        
        print(f"→ Computing IK for target '{target_name}'...")
        
        # Wait for IK result with timeout
        start_time = time.time()
        timeout = 5.0  # 5 second timeout
        
        while time.time() - start_time < timeout:
            # Spin to process incoming messages
            rclpy.spin_once(self, timeout_sec=0.1)
            
            # Check if IK result arrived
            if self._ik_computed_values['wrist_yaw'] is not None:
                print(f"  ✓ IK received and stored")
                return True
            
            time.sleep(0.1)
        
        # Timeout - IK never arrived
        print(f"  ⚠ IK computation timeout after {timeout}s")
        return False
    #
    def _wait(self, duration):
        """Wait for specified duration."""
        print(f"→ Waiting {duration} seconds...")
        time.sleep(duration)
    
    def _wait_for_arm(self, timeout=ARM_TIMEOUT):
        """Wait until arm reaches target positions."""
        print(f"→ Waiting for arm to reach target positions (timeout: {timeout}s)...")
        
        targets = {
            'joint_lift': self.joint_state.get('lift'),
            'joint_arm_l0': self.joint_state.get('arm_extend') / 4.0,
            'joint_wrist_yaw': self.joint_state.get('wrist_yaw'),
            'joint_wrist_pitch': self.joint_state.get('wrist_pitch'),
            'joint_wrist_roll': self.joint_state.get('wrist_roll'),
        }
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_reached = True
            
            for joint_name, target in targets.items():
                if target is None:
                    continue
                
                current = self.current_joint_states.get(joint_name)
                if current is None or abs(current - target) > self.POSITION_TOLERANCE:
                    all_reached = False
                    break
            
            if all_reached:
                print("  ✓ Arm reached target positions")
                return True
            
            rclpy.spin_once(self, timeout_sec=self.CHECK_INTERVAL)
            time.sleep(self.CHECK_INTERVAL)
        
        print(f"  ⚠ Timeout reached ({timeout}s)")
        return False
    
    def run(self):
        """Run the interactive command loop."""
        print("Type 'help' for available commands, 'exit' to quit")
        if READLINE_AVAILABLE:
            print("Use ↑/↓ arrow keys to navigate command history\n")
        else:
            print("(Command history not available on this system)\n")
        
        try:
            while rclpy.ok():
                try:
                    command = input("stretch> ").strip()
                    
                    if not command:
                        continue
                    
                    if command.lower() in ['exit', 'quit']:
                        print("Exiting...")
                        break
                    
                    if command.lower() == 'help':
                        self._print_help()
                        continue
                    
                    if command.lower().startswith('help '):
                        action_name = command[5:].strip()
                        self._print_help(action_name)
                        continue
                    
                    if command.lower() == 'list':
                        self._list_actions()
                        continue
                    
                    # Parse and execute command
                    action_name, params = self._parse_command(command)
                    
                    if action_name in self.micro_actions:
                        self._execute_micro_action(action_name, params)
                    elif action_name in self.macro_actions:
                        self._execute_macro_action(action_name, params)
                    else:
                        print(f"Error: Unknown action '{action_name}'")
                        print("Type 'list' to see available actions or 'help' for more information")
                
                except EOFError:
                    print("\nExiting...")
                    break
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error: {e}")
        
        finally:
            # Save history before exiting
            if READLINE_AVAILABLE:
                histfile = os.path.join(os.path.expanduser("~"), ".stretch_controller_history")
                try:
                    readline.write_history_file(histfile)
                except Exception:
                    pass
            
            self.destroy_node()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    controller = InteractiveController()
    
    try:
        controller.run()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
