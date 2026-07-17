#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
python3 stretch_ros2_sim.py --world table_world.xml --global-plan-only --single-robot --no-camera
