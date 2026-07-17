#!/usr/bin/env python3
"""Test a two-robot local-planner swap: /stretch C->B and /stretch2 B->C.

The script only sends the swap command after both robots have reached the setup
anchors: /stretch at C and /stretch2 at B.
"""

import argparse
import math
import time

import rclpy
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import String


NAV2_ANCHOR_STANDOFFS = {
    "A": [-0.65, 1.00, 0.0],
    "B": [0.65, 1.00, 0.0],
    "C": [-0.65, 3.50, 0.0],
    "D": [0.65, 3.50, 0.0],
    "E": [1.55, 2.90, 0.0],
    "F": [1.55, 1.60, 0.0],
}


class TwoRobotCBSwapTest(Node):
    def __init__(self, tolerance):
        super().__init__("two_robot_c_b_swap_test")
        self.tolerance = tolerance
        self.odom = {
            "/stretch": None,
            "/stretch2": None,
        }
        self.path_time = {
            "/stretch": 0.0,
            "/stretch2": 0.0,
        }
        self.path_count = {
            "/stretch": 0,
            "/stretch2": 0,
        }
        self.pubs = {
            "/stretch": self.create_publisher(String, "/stretch/navigate_to_anchor", 10),
            "/stretch2": self.create_publisher(String, "/stretch2/navigate_to_anchor", 10),
        }
        self.create_subscription(Odometry, "/stretch/odom", self._odom_cb("/stretch"), 10)
        self.create_subscription(Odometry, "/stretch2/odom", self._odom_cb("/stretch2"), 10)
        self.create_subscription(Path, "/stretch/global_path", self._path_cb("/stretch"), 10)
        self.create_subscription(Path, "/stretch2/global_path", self._path_cb("/stretch2"), 10)

    def _odom_cb(self, namespace):
        def callback(msg):
            pos = msg.pose.pose.position
            self.odom[namespace] = (float(pos.x), float(pos.y))

        return callback

    def _path_cb(self, namespace):
        def callback(msg):
            self.path_time[namespace] = time.monotonic()
            self.path_count[namespace] = len(msg.poses)

        return callback

    def wait_for_odom(self, timeout):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(value is not None for value in self.odom.values()):
                return True
        return False

    def _distance(self, namespace, anchor):
        if self.odom[namespace] is None:
            return float("inf")
        target = NAV2_ANCHOR_STANDOFFS[anchor]
        return math.hypot(self.odom[namespace][0] - target[0], self.odom[namespace][1] - target[1])

    def _publish_goal(self, namespace, anchor):
        msg = String()
        msg.data = anchor
        self.pubs[namespace].publish(msg)

    def send_pair(self, r1_anchor, r2_anchor):
        self._publish_goal("/stretch", r1_anchor)
        self._publish_goal("/stretch2", r2_anchor)

    def wait_until_pair_reached(self, r1_anchor, r2_anchor, timeout):
        start = time.monotonic()
        goals = {
            "/stretch": r1_anchor,
            "/stretch2": r2_anchor,
        }
        saw_path = {namespace: False for namespace in goals}
        best = {namespace: float("inf") for namespace in goals}

        while time.monotonic() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            reached = True
            for namespace, anchor in goals.items():
                dist = self._distance(namespace, anchor)
                best[namespace] = min(best[namespace], dist)
                if self.path_time[namespace] >= start and self.path_count[namespace] > 0:
                    saw_path[namespace] = True
                if not saw_path[namespace] or dist > self.tolerance:
                    reached = False
            if reached:
                return True, time.monotonic() - start, best
        return False, time.monotonic() - start, best


def main(args=None):
    parser = argparse.ArgumentParser(description="Test /stretch C->B and /stretch2 B->C.")
    parser.add_argument("--timeout", type=float, default=75.0)
    parser.add_argument("--tolerance", type=float, default=0.18)
    parser.add_argument("--settle", type=float, default=1.0)
    parsed, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    node = TwoRobotCBSwapTest(parsed.tolerance)
    try:
        if not node.wait_for_odom(timeout=5.0):
            raise SystemExit("No odom from both robots. Start two-robot sim first.")

        print("setup: /stretch -> C, /stretch2 -> B")
        node.send_pair("C", "B")
        ok, elapsed, best = node.wait_until_pair_reached("C", "B", parsed.timeout)
        print(
            f"setup {'OK' if ok else 'TIMEOUT'} in {elapsed:.1f}s; "
            f"best /stretch={best['/stretch']:.3f}m /stretch2={best['/stretch2']:.3f}m"
        )
        if not ok:
            raise SystemExit(1)

        end_settle = time.monotonic() + parsed.settle
        while time.monotonic() < end_settle:
            rclpy.spin_once(node, timeout_sec=0.05)

        print("swap: /stretch C->B, /stretch2 B->C")
        node.send_pair("B", "C")
        ok, elapsed, best = node.wait_until_pair_reached("B", "C", parsed.timeout)
        print(
            f"swap {'OK' if ok else 'TIMEOUT'} in {elapsed:.1f}s; "
            f"best /stretch={best['/stretch']:.3f}m /stretch2={best['/stretch2']:.3f}m"
        )
        if not ok:
            raise SystemExit(1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
