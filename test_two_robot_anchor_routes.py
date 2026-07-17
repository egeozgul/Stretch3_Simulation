#!/usr/bin/env python3
"""Run paired two-robot anchor route tests with local planners for both robots."""

import argparse
import math
import os
import time

import rclpy
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import String

from anchor_utils import load_anchors_from_xml


DEFAULT_SCENARIOS = [
    ("top_to_bottom", "A", "C", "B", "D"),
    ("bottom_to_top", "C", "A", "D", "B"),
    ("top_diagonal_cross", "A", "D", "B", "C"),
    ("bottom_diagonal_cross", "C", "B", "D", "A"),
    ("side_swap", "E", "F", "F", "E"),
    ("top_to_side_cross", "A", "E", "B", "F"),
    ("side_to_bottom_cross", "E", "C", "F", "D"),
    ("long_cross_1", "A", "F", "D", "E"),
    ("long_cross_2", "B", "E", "C", "F"),
]

NAV2_ANCHOR_STANDOFFS = {
    "A": [-0.65, 1.00, 0.0],
    "B": [0.65, 1.00, 0.0],
    "C": [-0.65, 3.50, 0.0],
    "D": [0.65, 3.50, 0.0],
    "E": [1.55, 2.90, 0.0],
    "F": [1.55, 1.60, 0.0],
}


class TwoRobotAnchorRouteTester(Node):
    def __init__(self, anchors, tolerance):
        super().__init__("two_robot_anchor_route_tester")
        self.anchors = anchors
        self.tolerance = tolerance
        self.odom = {
            "/stretch": None,
            "/stretch2": None,
        }
        self.path_counts = {
            "/stretch": 0,
            "/stretch2": 0,
        }
        self.path_times = {
            "/stretch": 0.0,
            "/stretch2": 0.0,
        }
        self.anchor_publishers = {
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
            self.path_counts[namespace] = len(msg.poses)
            self.path_times[namespace] = time.monotonic()

        return callback

    def _publish_anchor(self, namespace, anchor):
        msg = String()
        msg.data = anchor
        self.anchor_publishers[namespace].publish(msg)

    def _spin_for(self, seconds):
        end_time = time.monotonic() + seconds
        while time.monotonic() < end_time:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _target_for_anchor(self, anchor):
        if anchor in NAV2_ANCHOR_STANDOFFS:
            return NAV2_ANCHOR_STANDOFFS[anchor]
        return self.anchors[anchor]["pos"]

    def _distance_to_anchor(self, namespace, anchor):
        if self.odom[namespace] is None:
            return float("inf")
        target = self._target_for_anchor(anchor)
        return math.hypot(self.odom[namespace][0] - target[0], self.odom[namespace][1] - target[1])

    def wait_for_odom(self, timeout=5.0):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(value is not None for value in self.odom.values()):
                return True
        return False

    def wait_until_reached(self, goals, timeout):
        start = time.monotonic()
        saw_path = {namespace: False for namespace in goals}
        best = {namespace: float("inf") for namespace in goals}

        while time.monotonic() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            all_reached = True
            for namespace, anchor in goals.items():
                distance = self._distance_to_anchor(namespace, anchor)
                best[namespace] = min(best[namespace], distance)
                if self.path_times[namespace] >= start and self.path_counts[namespace] > 0:
                    saw_path[namespace] = True
                if not saw_path[namespace] or distance > self.tolerance:
                    all_reached = False
            if all_reached:
                return True, time.monotonic() - start, best

        return False, time.monotonic() - start, best

    def run_scenario(self, name, r1_start, r1_goal, r2_start, r2_goal, timeout, settle):
        print(f"\n=== {name} ===")

        setup_goals = {"/stretch": r1_start, "/stretch2": r2_start}
        print(f"setup: /stretch->{r1_start}, /stretch2->{r2_start}")
        self._publish_anchor("/stretch", r1_start)
        self._publish_anchor("/stretch2", r2_start)
        ok, elapsed, best = self.wait_until_reached(setup_goals, timeout)
        print(
            f"setup {'OK' if ok else 'TIMEOUT'} in {elapsed:.1f}s; "
            f"best /stretch={best['/stretch']:.3f}m /stretch2={best['/stretch2']:.3f}m"
        )
        self._spin_for(settle)
        if not ok:
            return False

        test_goals = {"/stretch": r1_goal, "/stretch2": r2_goal}
        print(f"test:  /stretch {r1_start}->{r1_goal}, /stretch2 {r2_start}->{r2_goal}")
        self._publish_anchor("/stretch", r1_goal)
        self._publish_anchor("/stretch2", r2_goal)
        ok, elapsed, best = self.wait_until_reached(test_goals, timeout)
        print(
            f"test  {'OK' if ok else 'TIMEOUT'} in {elapsed:.1f}s; "
            f"best /stretch={best['/stretch']:.3f}m /stretch2={best['/stretch2']:.3f}m"
        )
        self._spin_for(settle)
        return ok


def parse_scenario(text):
    parts = [part.strip().upper() for part in text.split(":")]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("scenario must be name:r1_start:r1_goal:r2_start:r2_goal")
    name = parts[0].lower()
    return (name, parts[1], parts[2], parts[3], parts[4])


def main(args=None):
    parser = argparse.ArgumentParser(description="Run two-robot local-planner anchor route tests.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Seconds per setup/test leg.")
    parser.add_argument("--settle", type=float, default=1.0, help="Seconds to wait between legs.")
    parser.add_argument("--tolerance", type=float, default=0.18, help="Anchor completion tolerance in meters.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat all scenarios this many times.")
    parser.add_argument(
        "--scenario",
        action="append",
        type=parse_scenario,
        help="Custom scenario: name:r1_start:r1_goal:r2_start:r2_goal. Can be repeated.",
    )
    parsed, ros_args = parser.parse_known_args(args)

    xml_path = os.path.join(os.path.dirname(__file__), "table_world.xml")
    anchors = load_anchors_from_xml(xml_path)
    scenarios = parsed.scenario or DEFAULT_SCENARIOS
    missing = sorted({anchor for scenario in scenarios for anchor in scenario[1:] if anchor not in anchors})
    if missing:
        raise SystemExit(f"Missing anchors in table_world.xml: {', '.join(missing)}")

    rclpy.init(args=ros_args)
    node = TwoRobotAnchorRouteTester(anchors, parsed.tolerance)
    try:
        if not node.wait_for_odom():
            raise SystemExit("No odom from both robots. Start the two-robot local planner sim first.")

        total = 0
        failures = []
        for run_idx in range(parsed.repeat):
            print(f"\n######## Run {run_idx + 1}/{parsed.repeat} ########")
            for scenario in scenarios:
                total += 1
                if not node.run_scenario(*scenario, timeout=parsed.timeout, settle=parsed.settle):
                    failures.append((run_idx + 1, scenario[0]))

        print("\n=== Summary ===")
        print(f"scenarios run: {total}")
        if failures:
            for run_idx, name in failures:
                print(f"FAILED/TIMEOUT: run {run_idx} {name}")
            raise SystemExit(1)
        print("all scenarios completed")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
