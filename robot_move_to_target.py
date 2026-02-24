#!/usr/bin/env python3
"""
Standalone script to move the Stretch robot end-effector to a target position
relative to the robot base. Runs on the robot Linux without ROS.

Uses stretch_body SDK and a simple geometric IK.
Parameters: speed (0-1), target (x, y, z) in meters in base frame.
Base frame: X forward, Y left, Z up (typical Stretch convention).
"""

import argparse
import math
import sys


def clamp(value, low, high):
    return max(low, min(high, value))


def geometric_ik(x, y, z, arm_range=(0.0, 0.52), lift_range=(0.0, 1.1),
                 wrist_yaw_range=(-1.39, 4.42), wrist_yaw_rad=None):
    """
    Simple geometric IK for Stretch: target (x,y,z) in base frame (m).
    Returns (lift_m, arm_m, wrist_yaw_rad), clamped to joint limits.
    If wrist_yaw_rad is not None, use it instead of atan2(y,x) so the gripper
    stays at a fixed orientation when only x/z change.
    """
    lift_m = clamp(z, lift_range[0], lift_range[1])
    horizontal = math.sqrt(x * x + y * y)
    arm_m = clamp(horizontal, arm_range[0], arm_range[1])
    if wrist_yaw_rad is None:
        wrist_yaw_rad = math.atan2(y, x)
    wrist_yaw_rad = clamp(wrist_yaw_rad, wrist_yaw_range[0], wrist_yaw_range[1])
    return lift_m, arm_m, wrist_yaw_rad


def main():
    parser = argparse.ArgumentParser(
        description="Move Stretch end-effector to target position (no ROS)."
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="Speed factor from 0 to 1 (default: 0.5)",
    )
    parser.add_argument(
        "--x",
        type=float,
        required=True,
        help="Target X position in meters (forward from base)",
    )
    parser.add_argument(
        "--y",
        type=float,
        required=True,
        help="Target Y position in meters (left from base)",
    )
    parser.add_argument(
        "--z",
        type=float,
        required=True,
        help="Target Z position in meters (up from base)",
    )
    parser.add_argument(
        "--wrist-yaw",
        type=float,
        default=None,
        metavar="RAD",
        help="Fix wrist yaw in radians (e.g. 0 = forward). If not set, wrist points toward (x,y) so it may turn when x/y change.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute and print IK solution, do not move the robot",
    )
    args = parser.parse_args()

    speed = clamp(args.speed, 0.0, 1.0)
    if args.speed != speed:
        print(f"Warning: speed clamped to {speed}", file=sys.stderr)

    # Geometric IK (no MuJoCo); result is clamped to joint limits
    lift_m, arm_m, wrist_yaw_rad = geometric_ik(
        args.x, args.y, args.z, wrist_yaw_rad=args.wrist_yaw
    )
    print(f"Target (base frame): x={args.x:.3f} y={args.y:.3f} z={args.z:.3f}")
    print(f"IK solution: lift={lift_m:.3f} m, arm={arm_m:.3f} m, wrist_yaw={wrist_yaw_rad:.3f} rad")
    if args.dry_run:
        print("Dry-run: not moving robot.")
        return 0

    try:
        import stretch_body.robot
    except ImportError:
        print("Error: stretch_body not found. Install with: pip install hello-robot-stretch-body", file=sys.stderr)
        sys.exit(1)

    r = stretch_body.robot.Robot()
    if not r.startup():
        print("Error: failed to connect to robot. Is another process using it?", file=sys.stderr)
        sys.exit(1)

    if not r.is_homed():
        print("Error: robot is not homed. Run homing first.", file=sys.stderr)
        r.stop()
        sys.exit(1)

    try:
        # Scale velocity by speed (0-1). Use a reasonable max for arm/lift (m/s).
        vel_scale = 0.05 + 0.15 * speed  # e.g. 0.05–0.20 m/s
        r.arm.set_velocity(vel_scale)
        r.lift.set_velocity(vel_scale)

        r.lift.move_to(lift_m)
        r.arm.move_to(arm_m)
        r.push_command()
        r.wait_command()

        # Wrist executes immediately (Dynamixel)
        if "wrist_yaw" in r.end_of_arm.joints:
            r.end_of_arm.move_to("wrist_yaw", wrist_yaw_rad)
            r.wait_command()

        print("Move completed.")
    except Exception as e:
        print(f"Error during motion: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        r.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
