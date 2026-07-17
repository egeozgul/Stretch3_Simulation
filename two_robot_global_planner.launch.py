#!/usr/bin/env python3
"""Launch Nav2 global planners for both Stretch robots on the same map."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _planner_nodes(namespace, params_file, x_arg, y_arg, yaw_arg):
    return [
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'{namespace}_map_to_odom',
            namespace=namespace,
            arguments=[
                LaunchConfiguration(x_arg),
                LaunchConfiguration(y_arg),
                '0',
                LaunchConfiguration(yaw_arg),
                '0',
                '0',
                'map',
                f'{namespace}/odom',
            ],
            remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'{namespace}_map_to_odom_rviz',
            arguments=[
                LaunchConfiguration(x_arg),
                LaunchConfiguration(y_arg),
                '0',
                LaunchConfiguration(yaw_arg),
                '0',
                '0',
                'map',
                f'{namespace}/odom',
            ],
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            namespace=namespace,
            output='screen',
            parameters=[params_file],
            remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_planner',
            namespace=namespace,
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'autostart': True,
                'node_names': ['planner_server'],
            }],
        ),
    ]


def generate_launch_description():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    default_map_file = os.path.join(repo_dir, 'maps', 'careful_map.yaml')
    stretch_params = os.path.join(repo_dir, 'nav2_global_planner_params.yaml')
    stretch2_params = os.path.join(repo_dir, 'nav2_global_planner_params_stretch2.yaml')
    rviz_config = os.path.join(repo_dir, 'global_planner.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('map', default_value=default_map_file),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('map_to_odom_x', default_value='-0.533'),
        DeclareLaunchArgument('map_to_odom_y', default_value='-2.317'),
        DeclareLaunchArgument('map_to_odom_yaw', default_value='0.0'),
        DeclareLaunchArgument('r2_map_to_odom_x', default_value='-0.533'),
        DeclareLaunchArgument('r2_map_to_odom_y', default_value='-2.317'),
        DeclareLaunchArgument('r2_map_to_odom_yaw', default_value='0.0'),
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'yaml_filename': LaunchConfiguration('map'),
            }],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'autostart': True,
                'node_names': ['map_server'],
            }],
        ),
        *_planner_nodes('stretch', stretch_params, 'map_to_odom_x', 'map_to_odom_y', 'map_to_odom_yaw'),
        *_planner_nodes('stretch2', stretch2_params, 'r2_map_to_odom_x', 'r2_map_to_odom_y', 'r2_map_to_odom_yaw'),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_global_planner',
            output='screen',
            arguments=['-d', rviz_config],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])
