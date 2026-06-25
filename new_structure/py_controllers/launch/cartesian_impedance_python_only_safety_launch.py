#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Safety dry-run only. Cartesian startup targets may still be computed, but
    # effort_output_mode=disabled blocks every /effort_command publication.
    return LaunchDescription([
        Node(
            package='py_controllers',
            executable='cartesian_impedance',
            name='cartesian_impedance',
            output='screen',
            parameters=[{
                'effort_output_mode': 'disabled',
                'gp_prediction_enabled': False,
                'gp_online_update_enabled': False,
                'gp_compensation_enabled': False,
                'timing_logging_enabled': False,
                'reference_mode': 'cartesian',
            }],
        ),
    ])
