#!/bin/bash
# Quick start script for ROS 2 Stretch simulation

# Check if ROS 2 is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "ROS 2 not sourced. Attempting to source..."
    if [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
        echo "Sourced ROS 2 Humble"
    elif [ -f "/opt/ros/jazzy/setup.bash" ]; then
        source /opt/ros/jazzy/setup.bash
        echo "Sourced ROS 2 Jazzy"
    else
        echo "ERROR: ROS 2 not found. Please install ROS 2 first."
        echo "See: https://docs.ros.org/en/humble/Installation.html"
        exit 1
    fi
fi

# Activate conda environment if available
if command -v conda &> /dev/null; then
    if conda env list | grep -q "simenv"; then
        echo "Activating conda environment: simenv"
        eval "$(conda shell.bash hook)"
        conda activate simenv
    fi
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Starting Stretch 3 ROS 2 Simulation"
echo "=========================================="
echo "ROS_DISTRO: $ROS_DISTRO"
echo "Working directory: $SCRIPT_DIR"
echo ""
echo "This will start the MuJoCo simulation with ROS 2 communication."
echo "In another terminal, you can:"
echo "  - Run: python stretch_keyboard_controller.py"
echo "  - Or use: ros2 topic pub /stretch/cmd_vel ..."
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

python stretch_ros2_sim.py

