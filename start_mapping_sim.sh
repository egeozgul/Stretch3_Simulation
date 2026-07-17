#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
python3 stretch_ros2_sim.py --world table_world_mapping.xml --no-camera
