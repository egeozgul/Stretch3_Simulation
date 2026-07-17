#!/usr/bin/env python3
"""Compute map-to-world alignment from matched MuJoCo and RViz map points.

The simulator uses:

    map_xy = R(yaw) * world_xy + [x, y]

This script estimates x, y, and yaw from matching point pairs.

Example:
  python3 calibrate_map_alignment.py \
    --pair -0.65 0.90 -0.42 1.15 \
    --pair 0.65 3.60 0.88 3.85 \
    --pair 0.533 2.317 0.76 2.56

Each --pair is:
    world_x world_y map_x map_y
"""

import argparse
import math

import numpy as np


def estimate_transform(pairs):
    world = np.array([[p[0], p[1]] for p in pairs], dtype=float)
    map_pts = np.array([[p[2], p[3]] for p in pairs], dtype=float)

    world_centroid = world.mean(axis=0)
    map_centroid = map_pts.mean(axis=0)
    world_centered = world - world_centroid
    map_centered = map_pts - map_centroid

    h = world_centered.T @ map_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    t = map_centroid - r @ world_centroid
    yaw = math.atan2(r[1, 0], r[0, 0])
    predicted = (world @ r.T) + t
    residuals = np.linalg.norm(predicted - map_pts, axis=1)
    return t, yaw, predicted, residuals


def main():
    parser = argparse.ArgumentParser(description="Estimate MAP_TO_ODOM alignment from point pairs.")
    parser.add_argument(
        "--pair",
        nargs=4,
        type=float,
        action="append",
        metavar=("WORLD_X", "WORLD_Y", "MAP_X", "MAP_Y"),
        required=True,
        help="Matching MuJoCo/world point and RViz/map point. Provide at least two.",
    )
    args = parser.parse_args()

    if len(args.pair) < 2:
        raise SystemExit("Need at least two --pair values.")

    translation, yaw, predicted, residuals = estimate_transform(args.pair)
    yaw_deg = math.degrees(yaw)

    print("Estimated alignment:")
    print(f"  MAP_TO_ODOM_X={translation[0]:.6f}")
    print(f"  MAP_TO_ODOM_Y={translation[1]:.6f}")
    print(f"  MAP_TO_ODOM_YAW_DEG={yaw_deg:.6f}")
    print()
    print("Run sim:")
    print(
        f"  MAP_TO_ODOM_X={translation[0]:.6f} "
        f"MAP_TO_ODOM_Y={translation[1]:.6f} "
        f"MAP_TO_ODOM_YAW_DEG={yaw_deg:.6f} "
        "./start_global_planner_sim.sh"
    )
    print()
    print("Run planner/RViz:")
    print(
        "  ros2 launch ./single_robot_global_planner.launch.py "
        f"map_to_odom_x:={translation[0]:.6f} "
        f"map_to_odom_y:={translation[1]:.6f} "
        f"map_to_odom_yaw:={yaw:.6f}"
    )
    print()
    print("Fit residuals:")
    for idx, (pred, residual) in enumerate(zip(predicted, residuals), start=1):
        print(f"  pair {idx}: predicted_map=({pred[0]:.4f}, {pred[1]:.4f}), error={residual:.4f} m")
    print(f"  mean error={residuals.mean():.4f} m")


if __name__ == "__main__":
    main()
