#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    mock_rate_parameter_name = 'mock_state_rate_hz'
    trajectory_mode_parameter_name = 'trajectory_mode'
    circle_frequency_parameter_name = 'circle_frequency'
    transition_duration_parameter_name = 'transition_duration'
    trajectory_start_delay_parameter_name = 'trajectory_start_delay_sec'
    cartesian_start_delay_parameter_name = 'cartesian_start_delay_sec'

    mock_state_rate_hz = LaunchConfiguration(mock_rate_parameter_name)
    trajectory_mode = LaunchConfiguration(trajectory_mode_parameter_name)
    circle_frequency = LaunchConfiguration(circle_frequency_parameter_name)
    transition_duration = LaunchConfiguration(transition_duration_parameter_name)
    trajectory_start_delay_sec = LaunchConfiguration(trajectory_start_delay_parameter_name)
    cartesian_start_delay_sec = LaunchConfiguration(cartesian_start_delay_parameter_name)

    mock_state_node = Node(
        package='py_controllers',
        executable='goal1_mock_state_parameter_publisher',
        name='goal1_mock_state_parameter_publisher',
        output='screen',
        parameters=[{
            'publish_rate_hz': ParameterValue(mock_state_rate_hz, value_type=float),
        }],
    )

    trajectory_node = Node(
        package='py_controllers',
        executable='trajectory_publisher',
        name='trajectory_publisher',
        output='screen',
        parameters=[{
            trajectory_mode_parameter_name: ParameterValue(trajectory_mode, value_type=str),
            'z_amplitude': 0.0,
            circle_frequency_parameter_name: ParameterValue(circle_frequency, value_type=float),
            transition_duration_parameter_name: ParameterValue(
                transition_duration,
                value_type=float,
            ),
        }],
    )

    cartesian_node = Node(
        package='py_controllers',
        executable='cartesian_impedance',
        name='cartesian_impedance',
        output='screen',
        parameters=[{
            'start_x': 0.35,
            'start_y': 0.0,
            'start_z': 0.65,
            'gp_prediction_enabled': False,
            'gp_online_update_enabled': False,
            'gp_compensation_enabled': False,
            'gp_compensation_source': 'local',
            'gp_compensation_scale': 0.1,
            'gp_compensation_clip_nm': 0.5,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            mock_rate_parameter_name,
            default_value='50.0',
            description='Mock /state_parameter publish rate in Hz for offline/no-motion checks.',
        ),
        DeclareLaunchArgument(
            trajectory_mode_parameter_name,
            default_value='planar_circle',
            description='Offline trajectory mode; default keeps planar_circle.',
        ),
        DeclareLaunchArgument(
            circle_frequency_parameter_name,
            default_value='0.1',
            description='Offline trajectory publisher circle frequency in Hz.',
        ),
        DeclareLaunchArgument(
            transition_duration_parameter_name,
            default_value='3.0',
            description='Offline trajectory transition duration in seconds.',
        ),
        DeclareLaunchArgument(
            trajectory_start_delay_parameter_name,
            default_value='1.0',
            description='Delay before starting trajectory_publisher.',
        ),
        DeclareLaunchArgument(
            cartesian_start_delay_parameter_name,
            default_value='2.0',
            description='Delay before starting cartesian_impedance.',
        ),
        mock_state_node,
        TimerAction(
            period=trajectory_start_delay_sec,
            actions=[trajectory_node],
        ),
        TimerAction(
            period=cartesian_start_delay_sec,
            actions=[cartesian_node],
        ),
    ])
