#!/usr/bin/env python3
"""Launch Nav2 global planner plus FollowPath local controller for Stretch1."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    default_map_file = os.path.join(repo_dir, 'maps', 'careful_map.yaml')
    params_file = os.path.join(repo_dir, 'nav2_local_planner_params.yaml')
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
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='stretch_map_to_odom',
            namespace='stretch',
            arguments=[
                LaunchConfiguration('map_to_odom_x'),
                LaunchConfiguration('map_to_odom_y'),
                '0',
                LaunchConfiguration('map_to_odom_yaw'),
                '0',
                '0',
                'map',
                'stretch/odom',
            ],
            remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='stretch_map_to_odom_rviz',
            arguments=[
                LaunchConfiguration('map_to_odom_x'),
                LaunchConfiguration('map_to_odom_y'),
                '0',
                LaunchConfiguration('map_to_odom_yaw'),
                '0',
                '0',
                'map',
                'stretch/odom',
            ],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='stretch2_map_to_odom',
            namespace='stretch2',
            arguments=[
                LaunchConfiguration('r2_map_to_odom_x'),
                LaunchConfiguration('r2_map_to_odom_y'),
                '0',
                LaunchConfiguration('r2_map_to_odom_yaw'),
                '0',
                '0',
                'map',
                'stretch2/odom',
            ],
            remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='stretch2_map_to_odom_rviz',
            arguments=[
                LaunchConfiguration('r2_map_to_odom_x'),
                LaunchConfiguration('r2_map_to_odom_y'),
                '0',
                LaunchConfiguration('r2_map_to_odom_yaw'),
                '0',
                '0',
                'map',
                'stretch2/odom',
            ],
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            namespace='stretch',
            output='screen',
            parameters=[params_file],
            remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
        ),
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            namespace='stretch',
            output='screen',
            parameters=[params_file],
            remappings=[
                ('/tf', 'tf'),
                ('/tf_static', 'tf_static'),
                ('cmd_vel', 'cmd_vel_nav'),
            ],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_planner_controller',
            namespace='stretch',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'autostart': True,
                'node_names': ['planner_server', 'controller_server'],
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_local_planner',
            output='screen',
            arguments=['-d', rviz_config],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])
