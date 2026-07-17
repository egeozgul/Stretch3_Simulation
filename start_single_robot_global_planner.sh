#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
ros2 launch ./single_robot_global_planner.launch.py "$@"
