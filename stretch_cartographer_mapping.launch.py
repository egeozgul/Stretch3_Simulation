#!/usr/bin/env python3
"""Launch Cartographer for mapping the Stretch MuJoCo environment."""

import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_dir = os.path.dirname(os.path.abspath(__file__))

    return LaunchDescription([
        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            arguments=[
                '-configuration_directory', config_dir,
                '-configuration_basename', 'cartographer_stretch.lua',
            ],
            remappings=[
                ('scan', '/stretch/scan'),
                ('odom', '/stretch/odom'),
            ],
        ),
        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            output='screen',
            arguments=['-resolution', '0.05', '-publish_period_sec', '1.0'],
        ),
    ])
