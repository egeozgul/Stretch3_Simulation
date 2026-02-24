#!/bin/bash
# Wrapper to run ROS 2 simulation with correct environment
export MUJOCO_GL=egl
export __GL_SYNC_TO_VBLANK=0
export XDG_SESSION_TYPE=x11
# Activate ROS 2 environment
eval "$(conda shell.bash hook)"
conda activate simenv_ros2

# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Run simulation
python stretch_ros2_sim.py
