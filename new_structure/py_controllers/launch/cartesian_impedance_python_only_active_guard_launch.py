#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    # Guarded short active validation only. Distance is logging/warning-only
    # unless startup_distance_refuse_enabled is explicitly enabled. The normal
    # current pose -> fixed start interpolation remains active. This launch
    # intentionally starts no robot bringup, relay, trajectory publisher, or GP
    # server, and is not for formal GP data collection.
    return LaunchDescription([
        LogInfo(
            msg=(
                'GUARDED SHORT ACTIVE VALIDATION ONLY: fixed-start distance '
                'guard, startup torque clip, and torque rate limit are enabled. '
                'No trajectory or GP process is started by this launch.'
            )
        ),
        Node(
            package='py_controllers',
            executable='cartesian_impedance',
            name='cartesian_impedance',
            output='screen',
            parameters=[{
                'effort_output_mode': 'active',
                'reference_mode': 'cartesian',
                'gp_prediction_enabled': False,
                'gp_online_update_enabled': False,
                'gp_compensation_enabled': False,
                'startup_distance_guard_enabled': True,
                'startup_distance_warn_m': 0.10,
                'startup_distance_refuse_m': 0.30,
                'startup_distance_refuse_enabled': False,
                'startup_linear_speed': 0.01,
                'startup_torque_clip_nm': 10.0,
                'startup_torque_rate_limit_from_zero': True,
                'torque_rate_limit_enabled': True,
                'torque_rate_limit_nm_per_s': 20.0,
                'timing_logging_enabled': False,
            }],
        ),
    ])
