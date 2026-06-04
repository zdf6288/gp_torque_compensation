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
    z_amplitude_parameter_name = 'z_amplitude'
    z_frequency_multiplier_parameter_name = 'z_frequency_multiplier'
    transition_duration_parameter_name = 'transition_duration'
    trajectory_start_delay_parameter_name = 'trajectory_start_delay_sec'
    cartesian_start_delay_parameter_name = 'cartesian_start_delay_sec'
    gp_prediction_enabled_parameter_name = 'gp_prediction_enabled'
    gp_model_dir_parameter_name = 'gp_model_dir'
    gp_compensation_enabled_parameter_name = 'gp_compensation_enabled'
    gp_shadow_logging_enabled_parameter_name = 'gp_shadow_paper_fusion_logging_enabled'
    gp_historical_shadow_enabled_parameter_name = 'gp_historical_shadow_enabled'
    gp_historical_source_mode_parameter_name = 'gp_historical_source_mode'
    gp_historical_min_points_parameter_name = 'gp_historical_shadow_min_points'
    gp_historical_k_parameter_name = 'gp_historical_shadow_k'

    mock_state_rate_hz = LaunchConfiguration(mock_rate_parameter_name)
    trajectory_mode = LaunchConfiguration(trajectory_mode_parameter_name)
    circle_frequency = LaunchConfiguration(circle_frequency_parameter_name)
    z_amplitude = LaunchConfiguration(z_amplitude_parameter_name)
    z_frequency_multiplier = LaunchConfiguration(z_frequency_multiplier_parameter_name)
    transition_duration = LaunchConfiguration(transition_duration_parameter_name)
    trajectory_start_delay_sec = LaunchConfiguration(trajectory_start_delay_parameter_name)
    cartesian_start_delay_sec = LaunchConfiguration(cartesian_start_delay_parameter_name)
    gp_prediction_enabled = LaunchConfiguration(gp_prediction_enabled_parameter_name)
    gp_model_dir = LaunchConfiguration(gp_model_dir_parameter_name)
    gp_compensation_enabled = LaunchConfiguration(gp_compensation_enabled_parameter_name)
    gp_shadow_logging_enabled = LaunchConfiguration(gp_shadow_logging_enabled_parameter_name)
    gp_historical_shadow_enabled = LaunchConfiguration(gp_historical_shadow_enabled_parameter_name)
    gp_historical_source_mode = LaunchConfiguration(gp_historical_source_mode_parameter_name)
    gp_historical_min_points = LaunchConfiguration(gp_historical_min_points_parameter_name)
    gp_historical_k = LaunchConfiguration(gp_historical_k_parameter_name)

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
            z_amplitude_parameter_name: ParameterValue(z_amplitude, value_type=float),
            z_frequency_multiplier_parameter_name: ParameterValue(
                z_frequency_multiplier,
                value_type=float,
            ),
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
            gp_prediction_enabled_parameter_name: ParameterValue(
                gp_prediction_enabled,
                value_type=bool,
            ),
            'gp_online_update_enabled': False,
            gp_model_dir_parameter_name: ParameterValue(gp_model_dir, value_type=str),
            gp_compensation_enabled_parameter_name: ParameterValue(
                gp_compensation_enabled,
                value_type=bool,
            ),
            'gp_compensation_source': 'local',
            'gp_compensation_scale': 0.1,
            'gp_compensation_clip_nm': 0.5,
            gp_shadow_logging_enabled_parameter_name: ParameterValue(
                gp_shadow_logging_enabled,
                value_type=bool,
            ),
            gp_historical_shadow_enabled_parameter_name: ParameterValue(
                gp_historical_shadow_enabled,
                value_type=bool,
            ),
            gp_historical_source_mode_parameter_name: ParameterValue(
                gp_historical_source_mode,
                value_type=str,
            ),
            gp_historical_min_points_parameter_name: ParameterValue(
                gp_historical_min_points,
                value_type=int,
            ),
            gp_historical_k_parameter_name: ParameterValue(
                gp_historical_k,
                value_type=int,
            ),
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
            z_amplitude_parameter_name,
            default_value='0.02',
            description='Fake/no-motion z modulation amplitude in meters.',
        ),
        DeclareLaunchArgument(
            z_frequency_multiplier_parameter_name,
            default_value='0.5',
            description='Fake/no-motion z modulation frequency multiplier.',
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
        DeclareLaunchArgument(
            gp_prediction_enabled_parameter_name,
            default_value='false',
            description='Enable GP prediction only for explicit fake/no-motion validation.',
        ),
        DeclareLaunchArgument(
            gp_model_dir_parameter_name,
            default_value='./new_structure/gp/gp_models',
            description='GP model directory for explicit fake/no-motion validation.',
        ),
        DeclareLaunchArgument(
            gp_compensation_enabled_parameter_name,
            default_value='false',
            description='Keep GP torque compensation disabled in fake/no-motion validation.',
        ),
        DeclareLaunchArgument(
            gp_shadow_logging_enabled_parameter_name,
            default_value='false',
            description='Enable paper fusion shadow logging only for explicit fake validation.',
        ),
        DeclareLaunchArgument(
            gp_historical_shadow_enabled_parameter_name,
            default_value='false',
            description='Enable historical shadow source only for explicit fake validation.',
        ),
        DeclareLaunchArgument(
            gp_historical_source_mode_parameter_name,
            default_value='none',
            description='Historical shadow source mode for explicit fake validation.',
        ),
        DeclareLaunchArgument(
            gp_historical_min_points_parameter_name,
            default_value='10',
            description='Historical shadow minimum pool size for explicit fake validation.',
        ),
        DeclareLaunchArgument(
            gp_historical_k_parameter_name,
            default_value='5',
            description='Historical shadow top-k size for explicit fake validation.',
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
