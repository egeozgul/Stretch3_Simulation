#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
ros2 launch ./stretch_cartographer_mapping.launch.py
