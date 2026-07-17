#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
ros2 launch ./two_robot_nav2.launch.py "$@"
