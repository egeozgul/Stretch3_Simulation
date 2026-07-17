#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
mkdir -p maps

map_name="${1:-maps/stretch_map}"
ros2 run nav2_map_server map_saver_cli -f "$map_name"
