#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    mock_rate_parameter_name = 'mock_state_rate_hz'
    control_frequency_parameter_name = 'control_frequency'
    delay_steps_parameter_name = 'delay_steps'
    run_name_parameter_name = 'run_name'
    data_output_dir_parameter_name = 'data_output_dir'
    timing_logging_enabled_parameter_name = 'timing_logging_enabled'
    timing_log_stride_parameter_name = 'timing_log_stride'
    timing_output_dir_parameter_name = 'timing_output_dir'
    deadline_ratio_warn_threshold_parameter_name = 'deadline_ratio_warn_threshold'
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
    gp_compensation_disable_joint7_parameter_name = 'gp_compensation_disable_joint7'
    gp_shadow_logging_enabled_parameter_name = 'gp_shadow_paper_fusion_logging_enabled'
    gp_historical_shadow_enabled_parameter_name = 'gp_historical_shadow_enabled'
    gp_historical_source_mode_parameter_name = 'gp_historical_source_mode'
    gp_historical_min_points_parameter_name = 'gp_historical_shadow_min_points'
    gp_historical_k_parameter_name = 'gp_historical_shadow_k'
    gp_historical_db_enabled_parameter_name = 'gp_historical_db_enabled'
    gp_historical_db_path_parameter_name = 'gp_historical_db_path'
    gp_historical_db_k_parameter_name = 'gp_historical_db_k'
    gp_historical_db_q_scale_parameter_name = 'gp_historical_db_q_scale'
    gp_historical_db_dq_scale_parameter_name = 'gp_historical_db_dq_scale'
    gp_historical_db_max_distance_parameter_name = 'gp_historical_db_max_distance'
    gp_historical_db_disable_online_parameter_name = (
        'gp_historical_db_disable_when_online_update'
    )
    gp_historical_db_fallback_source_parameter_name = 'gp_historical_db_fallback_source'
    gp_historical_soft_shadow_enabled_parameter_name = (
        'gp_historical_soft_shadow_enabled'
    )
    gp_historical_soft_alpha_parameter_name = 'gp_historical_soft_alpha'
    gp_historical_soft_distance_threshold_parameter_name = (
        'gp_historical_soft_distance_threshold'
    )
    gp_historical_soft_online_scale_parameter_name = (
        'gp_historical_soft_online_scale'
    )
    gp_historical_soft_non_online_scale_parameter_name = (
        'gp_historical_soft_non_online_scale'
    )

    mock_state_rate_hz = LaunchConfiguration(mock_rate_parameter_name)
    control_frequency = LaunchConfiguration(control_frequency_parameter_name)
    delay_steps = LaunchConfiguration(delay_steps_parameter_name)
    run_name = LaunchConfiguration(run_name_parameter_name)
    data_output_dir = LaunchConfiguration(data_output_dir_parameter_name)
    timing_logging_enabled = LaunchConfiguration(timing_logging_enabled_parameter_name)
    timing_log_stride = LaunchConfiguration(timing_log_stride_parameter_name)
    timing_output_dir = LaunchConfiguration(timing_output_dir_parameter_name)
    deadline_ratio_warn_threshold = LaunchConfiguration(
        deadline_ratio_warn_threshold_parameter_name
    )
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
    gp_compensation_disable_joint7 = LaunchConfiguration(
        gp_compensation_disable_joint7_parameter_name
    )
    gp_shadow_logging_enabled = LaunchConfiguration(gp_shadow_logging_enabled_parameter_name)
    gp_historical_shadow_enabled = LaunchConfiguration(gp_historical_shadow_enabled_parameter_name)
    gp_historical_source_mode = LaunchConfiguration(gp_historical_source_mode_parameter_name)
    gp_historical_min_points = LaunchConfiguration(gp_historical_min_points_parameter_name)
    gp_historical_k = LaunchConfiguration(gp_historical_k_parameter_name)
    gp_historical_db_enabled = LaunchConfiguration(gp_historical_db_enabled_parameter_name)
    gp_historical_db_path = LaunchConfiguration(gp_historical_db_path_parameter_name)
    gp_historical_db_k = LaunchConfiguration(gp_historical_db_k_parameter_name)
    gp_historical_db_q_scale = LaunchConfiguration(gp_historical_db_q_scale_parameter_name)
    gp_historical_db_dq_scale = LaunchConfiguration(gp_historical_db_dq_scale_parameter_name)
    gp_historical_db_max_distance = LaunchConfiguration(
        gp_historical_db_max_distance_parameter_name
    )
    gp_historical_db_disable_online = LaunchConfiguration(
        gp_historical_db_disable_online_parameter_name
    )
    gp_historical_db_fallback_source = LaunchConfiguration(
        gp_historical_db_fallback_source_parameter_name
    )
    gp_historical_soft_shadow_enabled = LaunchConfiguration(
        gp_historical_soft_shadow_enabled_parameter_name
    )
    gp_historical_soft_alpha = LaunchConfiguration(
        gp_historical_soft_alpha_parameter_name
    )
    gp_historical_soft_distance_threshold = LaunchConfiguration(
        gp_historical_soft_distance_threshold_parameter_name
    )
    gp_historical_soft_online_scale = LaunchConfiguration(
        gp_historical_soft_online_scale_parameter_name
    )
    gp_historical_soft_non_online_scale = LaunchConfiguration(
        gp_historical_soft_non_online_scale_parameter_name
    )

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
            control_frequency_parameter_name: ParameterValue(
                control_frequency,
                value_type=float,
            ),
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
            control_frequency_parameter_name: ParameterValue(
                control_frequency,
                value_type=float,
            ),
            delay_steps_parameter_name: ParameterValue(
                delay_steps,
                value_type=int,
            ),
            run_name_parameter_name: ParameterValue(run_name, value_type=str),
            data_output_dir_parameter_name: ParameterValue(data_output_dir, value_type=str),
            timing_logging_enabled_parameter_name: ParameterValue(
                timing_logging_enabled,
                value_type=bool,
            ),
            timing_log_stride_parameter_name: ParameterValue(
                timing_log_stride,
                value_type=int,
            ),
            timing_output_dir_parameter_name: ParameterValue(
                timing_output_dir,
                value_type=str,
            ),
            deadline_ratio_warn_threshold_parameter_name: ParameterValue(
                deadline_ratio_warn_threshold,
                value_type=float,
            ),
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
            gp_compensation_disable_joint7_parameter_name: ParameterValue(
                gp_compensation_disable_joint7,
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
            gp_historical_db_enabled_parameter_name: ParameterValue(
                gp_historical_db_enabled,
                value_type=bool,
            ),
            gp_historical_db_path_parameter_name: ParameterValue(
                gp_historical_db_path,
                value_type=str,
            ),
            gp_historical_db_k_parameter_name: ParameterValue(
                gp_historical_db_k,
                value_type=int,
            ),
            gp_historical_db_q_scale_parameter_name: ParameterValue(
                gp_historical_db_q_scale,
                value_type=float,
            ),
            gp_historical_db_dq_scale_parameter_name: ParameterValue(
                gp_historical_db_dq_scale,
                value_type=float,
            ),
            gp_historical_db_max_distance_parameter_name: ParameterValue(
                gp_historical_db_max_distance,
                value_type=float,
            ),
            gp_historical_db_disable_online_parameter_name: ParameterValue(
                gp_historical_db_disable_online,
                value_type=bool,
            ),
            gp_historical_db_fallback_source_parameter_name: ParameterValue(
                gp_historical_db_fallback_source,
                value_type=str,
            ),
            gp_historical_soft_shadow_enabled_parameter_name: ParameterValue(
                gp_historical_soft_shadow_enabled,
                value_type=bool,
            ),
            gp_historical_soft_alpha_parameter_name: ParameterValue(
                gp_historical_soft_alpha,
                value_type=float,
            ),
            gp_historical_soft_distance_threshold_parameter_name: ParameterValue(
                gp_historical_soft_distance_threshold,
                value_type=float,
            ),
            gp_historical_soft_online_scale_parameter_name: ParameterValue(
                gp_historical_soft_online_scale,
                value_type=float,
            ),
            gp_historical_soft_non_online_scale_parameter_name: ParameterValue(
                gp_historical_soft_non_online_scale,
                value_type=float,
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
            control_frequency_parameter_name,
            default_value='50',
            description='Trajectory and controller metadata frequency in Hz for fake validation.',
        ),
        DeclareLaunchArgument(
            delay_steps_parameter_name,
            default_value='0',
            description='Cloud-like control-step delay for fake validation.',
        ),
        DeclareLaunchArgument(
            run_name_parameter_name,
            default_value='',
            description='Optional fake validation run name written to controller CSV metadata.',
        ),
        DeclareLaunchArgument(
            data_output_dir_parameter_name,
            default_value='.',
            description='Directory for fake validation controller data CSV output.',
        ),
        DeclareLaunchArgument(
            timing_logging_enabled_parameter_name,
            default_value='false',
            description='Enable fake validation controller timing CSV logging.',
        ),
        DeclareLaunchArgument(
            timing_log_stride_parameter_name,
            default_value='1',
            description='Record one fake validation timing row every N callbacks.',
        ),
        DeclareLaunchArgument(
            timing_output_dir_parameter_name,
            default_value='outputs/goal12_controller_timing',
            description='Directory for fake validation controller timing CSV output.',
        ),
        DeclareLaunchArgument(
            deadline_ratio_warn_threshold_parameter_name,
            default_value='0.8',
            description='Warn when fake validation max callback deadline ratio reaches this threshold.',
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
            gp_compensation_disable_joint7_parameter_name,
            default_value='false',
            description='Disable active GP applied torque on joint7 only when explicitly true.',
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
        DeclareLaunchArgument(
            gp_historical_db_enabled_parameter_name,
            default_value='false',
            description='Enable persistent residual DB shadow query only for explicit fake validation.',
        ),
        DeclareLaunchArgument(
            gp_historical_db_path_parameter_name,
            default_value='',
            description='Persistent residual DB .npz path; empty keeps the DB unavailable.',
        ),
        DeclareLaunchArgument(
            gp_historical_db_k_parameter_name,
            default_value='25',
            description='Persistent residual DB shadow top-k size.',
        ),
        DeclareLaunchArgument(
            gp_historical_db_q_scale_parameter_name,
            default_value='0.1',
            description='Persistent residual DB joint-position distance scale.',
        ),
        DeclareLaunchArgument(
            gp_historical_db_dq_scale_parameter_name,
            default_value='0.1',
            description='Persistent residual DB joint-velocity distance scale.',
        ),
        DeclareLaunchArgument(
            gp_historical_db_max_distance_parameter_name,
            default_value='1.0',
            description='Persistent residual DB shadow nearest-distance gate.',
        ),
        DeclareLaunchArgument(
            gp_historical_db_disable_online_parameter_name,
            default_value='true',
            description='Keep persistent residual DB unavailable while GP online update is enabled.',
        ),
        DeclareLaunchArgument(
            gp_historical_db_fallback_source_parameter_name,
            default_value='cloud',
            description='Shadow-only fallback source: none, local, cloud, or combined.',
        ),
        DeclareLaunchArgument(
            gp_historical_soft_shadow_enabled_parameter_name,
            default_value='false',
            description=(
                'Enable GOAL1 historical soft-weight CSV shadow logging only for '
                'explicit fake/no-robot validation; no active torque compensation.'
            ),
        ),
        DeclareLaunchArgument(
            gp_historical_soft_alpha_parameter_name,
            default_value='1.0',
            description=(
                'GOAL1 historical soft-shadow alpha for fake/no-robot validation '
                'logging only; no active torque compensation.'
            ),
        ),
        DeclareLaunchArgument(
            gp_historical_soft_distance_threshold_parameter_name,
            default_value='0.2',
            description=(
                'GOAL1 historical soft-shadow nearest-distance threshold for '
                'fake/no-robot validation logging only; no active torque compensation.'
            ),
        ),
        DeclareLaunchArgument(
            gp_historical_soft_online_scale_parameter_name,
            default_value='0.02',
            description=(
                'GOAL1 historical soft-shadow online-mode historical scale for '
                'fake/no-robot validation logging only; no active torque compensation.'
            ),
        ),
        DeclareLaunchArgument(
            gp_historical_soft_non_online_scale_parameter_name,
            default_value='1.0',
            description=(
                'GOAL1 historical soft-shadow non-online historical scale for '
                'fake/no-robot validation logging only; no active torque compensation.'
            ),
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
