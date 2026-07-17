#!/usr/bin/env python3
"""Send two Stretch robots to anchor goals through global path planning."""

import argparse
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from std_msgs.msg import String


class TwoRobotAnchorTest(Node):
    def __init__(self):
        super().__init__('two_robot_global_anchor_test')
        self.r1_pub = self.create_publisher(String, '/stretch/navigate_to_anchor', 10)
        self.r2_pub = self.create_publisher(String, '/stretch2/navigate_to_anchor', 10)
        self.r1_path_count = 0
        self.r2_path_count = 0
        self.create_subscription(Path, '/stretch/global_path', self._r1_path_cb, 10)
        self.create_subscription(Path, '/stretch2/global_path', self._r2_path_cb, 10)

    def _r1_path_cb(self, msg):
        self.r1_path_count += 1
        self.get_logger().info(f'Received /stretch/global_path with {len(msg.poses)} poses')

    def _r2_path_cb(self, msg):
        self.r2_path_count += 1
        self.get_logger().info(f'Received /stretch2/global_path with {len(msg.poses)} poses')

    def send(self, robot1_anchor, robot2_anchor, duration):
        msg1 = String()
        msg1.data = robot1_anchor.upper()
        msg2 = String()
        msg2.data = robot2_anchor.upper()

        # Give discovery a moment when the script starts right after launch.
        deadline = time.time() + 3.0
        while time.time() < deadline and (
            self.r1_pub.get_subscription_count() == 0
            or self.r2_pub.get_subscription_count() == 0
        ):
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info(
            f'Subscriber counts: /stretch={self.r1_pub.get_subscription_count()}, '
            f'/stretch2={self.r2_pub.get_subscription_count()}'
        )

        end_time = time.time() + duration
        while time.time() < end_time:
            self.r1_pub.publish(msg1)
            self.r2_pub.publish(msg2)
            self.get_logger().info(f'Sent /stretch -> {msg1.data}, /stretch2 -> {msg2.data}')
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.4)

        wait_until = time.time() + 5.0
        while time.time() < wait_until and (self.r1_path_count == 0 or self.r2_path_count == 0):
            rclpy.spin_once(self, timeout_sec=0.2)

        self.get_logger().info(
            f'Path result counts: /stretch={self.r1_path_count}, /stretch2={self.r2_path_count}'
        )


def main():
    parser = argparse.ArgumentParser(description='Test two-robot global planner anchor goals.')
    parser.add_argument('--stretch', default='A', help='Anchor for /stretch')
    parser.add_argument('--stretch2', default='D', help='Anchor for /stretch2')
    parser.add_argument('--duration', type=float, default=1.0, help='Seconds to repeatedly publish goals')
    args = parser.parse_args()

    rclpy.init()
    node = TwoRobotAnchorTest()
    try:
        node.send(args.stretch, args.stretch2, args.duration)
        time.sleep(1.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
