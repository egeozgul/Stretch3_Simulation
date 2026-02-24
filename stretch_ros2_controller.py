#!/usr/bin/env python3
"""Simple ROS 2 controller for sending commands to Stretch 3 simulation."""

import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray


class StretchController(Node):
    """Simple controller node for sending commands to Stretch simulation."""
    
    def __init__(self):
        super().__init__('stretch_controller')
        self.cmd_vel_pub = self.create_publisher(Twist, '/stretch/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(Float64MultiArray, '/stretch/joint_commands', 10)
    
    def send_base_velocity(self, linear_x=0.0, angular_z=0.0):
        """Send base velocity command."""
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(msg)
        self.get_logger().info(f'Sent velocity: linear={linear_x}, angular={angular_z}')
    
    def send_joint_commands(self, lift=0.0, arm_extend=0.0, wrist_yaw=0.0,
                           wrist_pitch=0.0, wrist_roll=0.0, gripper=0.0, 
                           head_pan=0.0, head_tilt=0.0):
        """Send joint position commands."""
        msg = Float64MultiArray()
        msg.data = [float(lift), float(arm_extend), float(wrist_yaw),
                   float(wrist_pitch), float(wrist_roll), float(gripper),
                   float(head_pan), float(head_tilt)]
        self.joint_cmd_pub.publish(msg)
        self.get_logger().info(f'Sent joints: lift={lift}, arm={arm_extend}, wrist_yaw={wrist_yaw}, wrist_pitch={wrist_pitch}, wrist_roll={wrist_roll}')


def main(args=None):
    rclpy.init(args=args)
    controller = StretchController()
    
    commands = {
        'forward': lambda: controller.send_base_velocity(linear_x=1.0),
        'backward': lambda: controller.send_base_velocity(linear_x=-1.0),
        'turn_left': lambda: controller.send_base_velocity(angular_z=1.0),
        'turn_right': lambda: controller.send_base_velocity(angular_z=-1.0),
        'stop': lambda: controller.send_base_velocity(0.0, 0.0),
        'lift_up': lambda: controller.send_joint_commands(lift=0.3),
        'arm_extend': lambda: controller.send_joint_commands(arm_extend=0.2)
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in commands:
        commands[sys.argv[1]]()
    else:
        print("\nAvailable commands:", ', '.join(commands.keys()))
        print("Usage: python stretch_ros2_controller.py <command>")
        print("\nOr use ROS 2 CLI:")
        print("  ros2 topic pub /stretch/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 1.0}}'")
    
    try:
        rclpy.spin_once(controller, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
