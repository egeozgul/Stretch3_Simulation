#!/usr/bin/env python3
"""
Set the Stretch gripper openness. Runs on the robot Linux without ROS.

Argument: openness in [0, 1] (float).
  0 = fully closed, 1 = fully open.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Set Stretch gripper openness (0=closed, 1=open). No ROS."
    )
    parser.add_argument(
        "openness",
        type=float,
        help="Gripper openness from 0 (closed) to 1 (open)",
    )
    args = parser.parse_args()

    # Clamp to [0, 1]
    openness = max(0.0, min(1.0, args.openness))
    if args.openness != openness:
        print(f"Warning: openness clamped to {openness}", file=sys.stderr)

    # Stretch API: -100 = closed, 100 = open
    gripper_pos = -100.0 + 200.0 * openness

    try:
        import stretch_body.robot
    except ImportError:
        print(
            "Error: stretch_body not found. Install with: pip install hello-robot-stretch-body",
            file=sys.stderr,
        )
        sys.exit(1)

    r = stretch_body.robot.Robot()
    if not r.startup():
        print("Error: failed to connect to robot. Is another process using it?", file=sys.stderr)
        sys.exit(1)

    try:
        if "stretch_gripper" in r.end_of_arm.joints:
            r.end_of_arm.move_to("stretch_gripper", gripper_pos)
            r.wait_command()
            print(f"Gripper set to {openness:.2f} (raw {gripper_pos:.0f})")
        else:
            print("Error: stretch_gripper not found in end_of_arm joints", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        r.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
