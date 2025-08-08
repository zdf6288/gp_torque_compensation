#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='py_controllers',
            executable='effort_pd',
            name='effort_pd',
            output='screen',
            parameters=[{
                # 可以在这里添加参数
                'kp': 100.0,
                'kd': 10.0,
            }]
        )
    ]) 