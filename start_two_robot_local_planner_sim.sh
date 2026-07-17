#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

sim_pid=""
planner_pid=""

cleanup() {
  if [ -n "$planner_pid" ] && kill -0 "$planner_pid" 2>/dev/null; then
    kill "$planner_pid" 2>/dev/null || true
  fi
  if [ -n "$sim_pid" ] && kill -0 "$sim_pid" 2>/dev/null; then
    kill "$sim_pid" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

python3 stretch_ros2_sim.py &
sim_pid=$!

sleep 2

ros2 launch ./two_robot_local_planner.launch.py "$@" &
planner_pid=$!

wait "$sim_pid" "$planner_pid"
