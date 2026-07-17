#!/usr/bin/env python3
"""Terminal keyboard controller for the second Stretch robot via ROS 2."""

import select
import sys
import termios
import time
import tty

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from anchor_utils import get_anchor_list


class Stretch2KeyboardController(Node):
    """Terminal keyboard controller that publishes to /stretch2 topics."""

    JOINT_LIMITS = {
        'lift': (0.0, 1.1),
        'arm_extend': (0.0, 0.52),
        'wrist_yaw': (-1.39, 4.42),
        'wrist_pitch': (-1.57, 0.56),
        'wrist_roll': (-3.14, 3.14),
        'gripper': (-0.02, 0.04),
        'head_pan': (-4.04, 1.73),
        'head_tilt': (-1.53, 0.79),
    }

    JOINT_ORDER = [
        'lift', 'arm_extend', 'wrist_yaw', 'wrist_pitch',
        'wrist_roll', 'gripper', 'head_pan', 'head_tilt',
    ]

    BASE_KEYS = {
        'w': (0.45, 0.0),
        's': (-0.35, 0.0),
        'a': (0.0, -1.8),
        'd': (0.0, 1.8),
    }

    JOINT_KEYS = {
        'q': ('lift', 0.05), 'e': ('lift', -0.05),
        'r': ('arm_extend', 0.05), 'f': ('arm_extend', -0.05),
        't': ('wrist_yaw', 0.1), 'g': ('wrist_yaw', -0.1),
        'y': ('wrist_pitch', 0.1), 'h': ('wrist_pitch', -0.1),
        'u': ('wrist_roll', 0.1), 'j': ('wrist_roll', -0.1),
        'z': ('gripper', 0.01), 'x': ('gripper', -0.01),
    }

    ARROW_KEYS = {
        'D': ('head_pan', 0.1),
        'C': ('head_pan', -0.1),
        'A': ('head_tilt', 0.1),
        'B': ('head_tilt', -0.1),
    }

    def __init__(self):
        super().__init__('stretch2_keyboard_controller')
        self.anchor_map = self._load_anchors()
        self.cmd_vel_pub = self.create_publisher(Twist, '/stretch2/cmd_vel', 10)
        self.anchor_pub = self.create_publisher(String, '/stretch2/navigate_to_anchor', 10)
        self.joint_cmd_pub = self.create_publisher(Float64MultiArray, '/stretch2/joint_commands', 10)
        self.reset_pub = self.create_publisher(String, '/stretch2/reset_arm', 10)

        self.running = True
        self.base_vel = {'linear_x': 0.0, 'angular_z': 0.0}
        self.last_base_key_time = 0.0
        self.joint_state = {key: 0.0 for key in self.JOINT_LIMITS}
        self.create_timer(0.05, self._publish_base_velocity)

    def _load_anchors(self):
        try:
            anchor_list = get_anchor_list()
            return {str(i): letter for i, letter in enumerate(sorted(anchor_list)[:5], start=1)}
        except Exception as exc:
            self.get_logger().warn(f'Failed to load anchors: {exc}, using default')
            return {'1': 'A', '2': 'B', '3': 'C', '4': 'D', '5': 'E'}

    def handle_key(self, key):
        if key in ('\x03', '\x1b'):
            self.running = False
            self.stop()
            return

        key = key.lower()
        if key == ' ':
            self.stop_base()
        elif key == '0':
            self.reset_arm()
        elif key in self.anchor_map:
            self.navigate_to_anchor(self.anchor_map[key])
        elif key in self.BASE_KEYS:
            linear_x, angular_z = self.BASE_KEYS[key]
            self.base_vel['linear_x'] = linear_x
            self.base_vel['angular_z'] = angular_z
            self.last_base_key_time = time.monotonic()
        elif key in self.JOINT_KEYS:
            self.update_joint(*self.JOINT_KEYS[key])

    def handle_arrow_key(self, code):
        if code in self.ARROW_KEYS:
            self.update_joint(*self.ARROW_KEYS[code])

    def update_joint(self, joint_name, delta):
        min_val, max_val = self.JOINT_LIMITS[joint_name]
        self.joint_state[joint_name] = float(np.clip(
            self.joint_state[joint_name] + delta,
            min_val,
            max_val,
        ))
        msg = Float64MultiArray()
        msg.data = [self.joint_state.get(key, 0.0) for key in self.JOINT_ORDER]
        self.joint_cmd_pub.publish(msg)

    def reset_arm(self):
        msg = String()
        msg.data = 'reset'
        self.reset_pub.publish(msg)
        self.get_logger().info('Resetting /stretch2 arm')

    def navigate_to_anchor(self, anchor_key):
        msg = String()
        msg.data = anchor_key
        self.anchor_pub.publish(msg)
        self.get_logger().info(f'/stretch2 navigating to anchor {anchor_key}')

    def stop_base(self):
        self.base_vel['linear_x'] = 0.0
        self.base_vel['angular_z'] = 0.0
        self._publish_base_velocity()

    def _publish_base_velocity(self):
        if time.monotonic() - self.last_base_key_time > 0.25:
            self.base_vel['linear_x'] = 0.0
            self.base_vel['angular_z'] = 0.0

        msg = Twist()
        msg.linear.x = self.base_vel['linear_x']
        msg.angular.z = self.base_vel['angular_z']
        self.cmd_vel_pub.publish(msg)

    def stop(self):
        self.stop_base()


def read_key(timeout=0.05):
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return None

    ch = sys.stdin.read(1)
    if ch == '\x1b' and select.select([sys.stdin], [], [], 0.001)[0]:
        bracket = sys.stdin.read(1)
        code = sys.stdin.read(1) if bracket == '[' else ''
        return ('arrow', code)
    return ('key', ch)


def print_help(anchor_map):
    anchor_str = '/'.join([f'{key}->{value}' for key, value in sorted(anchor_map.items())])
    print("\n" + "=" * 56)
    print("Stretch2 Terminal Keyboard Controller")
    print("=" * 56)
    print(f"  {anchor_str} - Navigate to anchors")
    print("  W/S - Move forward/backward")
    print("  A/D - Turn left/right")
    print("  SPACE - Stop base")
    print("  Q/E - Lift up/down")
    print("  R/F - Arm extend/retract")
    print("  T/G, Y/H, U/J - Wrist")
    print("  Z/X - Gripper")
    print("  Arrow keys - Head pan/tilt")
    print("  0 - Reset arm")
    print("  ESC or Ctrl-C - Exit")
    print("=" * 56 + "\n")


def main(args=None):
    rclpy.init(args=args)
    controller = Stretch2KeyboardController()
    old_settings = termios.tcgetattr(sys.stdin)

    print_help(controller.anchor_map)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while controller.running:
            event = read_key()
            if event:
                kind, value = event
                if kind == 'arrow':
                    controller.handle_arrow_key(value)
                else:
                    controller.handle_key(value)
            rclpy.spin_once(controller, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
