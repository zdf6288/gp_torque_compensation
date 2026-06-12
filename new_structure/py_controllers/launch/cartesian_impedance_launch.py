#!/usr/bin/env python3

import math
import tempfile

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _positive_float_or_raise(value_text, parameter_name):
    try:
        value = float(value_text)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f'{parameter_name} must be positive and finite; got {value_text!r}.'
        ) from exc

    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(
            f'{parameter_name} must be positive and finite; got {value_text!r}.'
        )
    return value


def _bool_or_raise(value_text, parameter_name):
    normalized = str(value_text).strip().lower()
    if normalized in ('true', '1', 'yes', 'on'):
        return True
    if normalized in ('false', '0', 'no', 'off'):
        return False
    raise RuntimeError(
        f'{parameter_name} must be a boolean value; got {value_text!r}.'
    )


def _guard_frequency_config(
        context,
        control_frequency,
        ros2_control_update_rate,
        trajectory_publish_rate,
        state_parameter_publish_rate,
        allow_high_ros2_control_rate,
        use_fake_hardware,
        spawn_cpp_relayer,
        spawn_update_rate_diagnostic):
    control_rate = _positive_float_or_raise(
        control_frequency.perform(context), 'control_frequency')
    ros2_control_rate = _positive_float_or_raise(
        ros2_control_update_rate.perform(context), 'ros2_control_update_rate')
    trajectory_rate = _positive_float_or_raise(
        trajectory_publish_rate.perform(context), 'trajectory_publish_rate')
    state_rate = _positive_float_or_raise(
        state_parameter_publish_rate.perform(context), 'state_parameter_publish_rate')
    allow_high_rate = _bool_or_raise(
        allow_high_ros2_control_rate.perform(context), 'allow_high_ros2_control_rate')
    fake_hardware = _bool_or_raise(
        use_fake_hardware.perform(context), 'use_fake_hardware')
    spawn_cpp_relayer_enabled = _bool_or_raise(
        spawn_cpp_relayer.perform(context), 'spawn_cpp_relayer')
    spawn_update_rate_diagnostic_enabled = _bool_or_raise(
        spawn_update_rate_diagnostic.perform(context), 'spawn_update_rate_diagnostic')

    actions = []

    if spawn_update_rate_diagnostic_enabled and not fake_hardware:
        raise RuntimeError(
            'update_rate_diagnostic is fake-only and requires '
            'use_fake_hardware:=true.'
        )

    if spawn_update_rate_diagnostic_enabled and spawn_cpp_relayer_enabled:
        raise RuntimeError(
            'update_rate_diagnostic fake validation requires '
            'spawn_cpp_relayer:=false so cpp_relayer is not activated.'
        )

    if spawn_update_rate_diagnostic_enabled:
        actions.append(
            LogInfo(
                msg=(
                    'Fake-only update_rate_diagnostic requested. It declares no '
                    'command interfaces, no state interfaces, and does not validate '
                    'real cpp_relayer or real robot safety.'
                )
            )
        )

    if ros2_control_rate > control_rate and not allow_high_rate:
        raise RuntimeError(
            'High-rate communication mode blocked: '
            'ros2_control_update_rate > control_frequency requires '
            'allow_high_ros2_control_rate:=true '
            f'(control_frequency={control_rate:.3f}, '
            f'ros2_control_update_rate={ros2_control_rate:.3f}).'
        )

    if ros2_control_rate > control_rate:
        actions.append(
            LogInfo(
                msg=(
                    'WARNING: high-rate ros2_control communication is experimental. '
                    f'ros2_control_update_rate={ros2_control_rate:.3f} Hz, '
                    f'control_frequency={control_rate:.3f} Hz, '
                    f'Python command update remains at '
                    f'state_parameter_publish_rate={state_rate:.3f} Hz, '
                    f'trajectory update remains at '
                    f'trajectory_publish_rate={trajectory_rate:.3f} Hz. '
                    'First validation must use gp_prediction_enabled:=false, '
                    'gp_online_update_enabled:=false, '
                    'gp_compensation_enabled:=false.'
                )
            )
        )
        return actions

    actions.append(
        LogInfo(
            msg=(
                'High-rate communication mode disabled or inactive; '
                f'legacy-safe guard passed with control_frequency={control_rate:.3f} Hz, '
                f'ros2_control_update_rate={ros2_control_rate:.3f} Hz, '
                f'trajectory_publish_rate={trajectory_rate:.3f} Hz, '
                f'state_parameter_publish_rate={state_rate:.3f} Hz, '
                f'allow_high_ros2_control_rate={str(allow_high_rate).lower()}.'
            )
        )
    )
    return actions


def _write_update_rate_diagnostic_params(expected_update_rate_hz, diagnostics_log_period_sec):
    with tempfile.NamedTemporaryFile(
        mode='w',
        prefix='update_rate_diagnostic_',
        suffix='.yaml',
        delete=False,
    ) as param_file:
        param_file.write(
            'update_rate_diagnostic:\n'
            '  ros__parameters:\n'
            '    diagnostics_enabled: true\n'
            f'    diagnostics_log_period_sec: {diagnostics_log_period_sec:.6f}\n'
            f'    expected_update_rate_hz: {expected_update_rate_hz:.6f}\n'
            '    expected_publish_rate_hz: 50.0\n'
            '    warn_ratio_low: 0.8\n'
            '    warn_ratio_high: 1.2\n'
        )
        return param_file.name


def _make_update_rate_diagnostic_spawner(
        context,
        update_rate_diagnostic_expected_rate,
        update_rate_diagnostic_log_period_sec):
    expected_update_rate_hz = _positive_float_or_raise(
        update_rate_diagnostic_expected_rate.perform(context),
        'update_rate_diagnostic_expected_rate')
    diagnostics_log_period_sec = _positive_float_or_raise(
        update_rate_diagnostic_log_period_sec.perform(context),
        'update_rate_diagnostic_log_period_sec')
    param_file = _write_update_rate_diagnostic_params(
        expected_update_rate_hz,
        diagnostics_log_period_sec)

    return [
        LogInfo(
            msg=(
                'Spawning fake-only update_rate_diagnostic with '
                f'expected_update_rate_hz={expected_update_rate_hz:.3f}, '
                f'diagnostics_log_period_sec={diagnostics_log_period_sec:.3f}.'
            )
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                '--param-file',
                param_file,
                'update_rate_diagnostic',
            ],
            output='screen',
        ),
    ]


def generate_launch_description():
    robot_ip_parameter_name = 'robot_ip'
    load_gripper_parameter_name = 'load_gripper'
    use_fake_hardware_parameter_name = 'use_fake_hardware'
    fake_sensor_commands_parameter_name = 'fake_sensor_commands'
    use_rviz_parameter_name = 'use_rviz'
    spawn_gp_server_parameter_name = 'spawn_gp_server'
    spawn_fake_state_parameter_publisher_parameter_name = 'spawn_fake_state_parameter_publisher'
    spawn_cpp_relayer_parameter_name = 'spawn_cpp_relayer'
    spawn_update_rate_diagnostic_parameter_name = 'spawn_update_rate_diagnostic'
    update_rate_diagnostic_expected_rate_parameter_name = 'update_rate_diagnostic_expected_rate'
    update_rate_diagnostic_log_period_sec_parameter_name = 'update_rate_diagnostic_log_period_sec'
    control_frequency_parameter_name = 'control_frequency'
    allow_high_ros2_control_rate_parameter_name = 'allow_high_ros2_control_rate'
    ros2_control_update_rate_parameter_name = 'ros2_control_update_rate'
    trajectory_publish_rate_parameter_name = 'trajectory_publish_rate'
    state_parameter_publish_rate_parameter_name = 'state_parameter_publish_rate'
    run_name_parameter_name = 'run_name'
    data_output_dir_parameter_name = 'data_output_dir'
    csv_output_profile_parameter_name = 'csv_output_profile'
    reference_mode_parameter_name = 'reference_mode'
    joint_space_command_topic_parameter_name = 'joint_space_command_topic'
    # Stage 1: frozen GP / compensation experiment 参数，默认值保持安全。
    gp_prediction_enabled_parameter_name = 'gp_prediction_enabled'
    gp_prediction_stride_parameter_name = 'gp_prediction_stride'
    gp_output_timeout_sec_parameter_name = 'gp_output_timeout_sec'
    future_trajectory_request_stride_parameter_name = 'future_trajectory_request_stride'
    gp_online_update_enabled_parameter_name = 'gp_online_update_enabled'
    gp_model_dir_parameter_name = 'gp_model_dir'
    gp_compensation_enabled_parameter_name = 'gp_compensation_enabled'
    gp_compensation_source_parameter_name = 'gp_compensation_source'
    gp_compensation_scale_parameter_name = 'gp_compensation_scale'
    gp_compensation_clip_nm_parameter_name = 'gp_compensation_clip_nm'
    gp_compensation_disable_joint7_parameter_name = 'gp_compensation_disable_joint7'
    torque_rate_limit_enabled_parameter_name = 'torque_rate_limit_enabled'
    torque_rate_limit_nm_per_s_parameter_name = 'torque_rate_limit_nm_per_s'
    torque_rate_limit_log_first_n_parameter_name = 'torque_rate_limit_log_first_n'
    torque_rate_limit_reset_on_first_command_parameter_name = (
        'torque_rate_limit_reset_on_first_command'
    )
    delay_steps_parameter_name = 'delay_steps'
    timing_logging_enabled_parameter_name = 'timing_logging_enabled'
    timing_log_stride_parameter_name = 'timing_log_stride'
    timing_output_dir_parameter_name = 'timing_output_dir'
    deadline_ratio_warn_threshold_parameter_name = 'deadline_ratio_warn_threshold'
    effort_gap_diagnostics_enabled_parameter_name = 'effort_gap_diagnostics_enabled'
    effort_gap_log_stride_parameter_name = 'effort_gap_log_stride'
    effort_gap_warn_sec_parameter_name = 'effort_gap_warn_sec'
    callback_wall_warn_sec_parameter_name = 'callback_wall_warn_sec'
    gp_historical_db_enabled_parameter_name = 'gp_historical_db_enabled'
    gp_historical_db_path_parameter_name = 'gp_historical_db_path'
    gp_historical_db_k_parameter_name = 'gp_historical_db_k'
    gp_historical_db_q_scale_parameter_name = 'gp_historical_db_q_scale'
    gp_historical_db_dq_scale_parameter_name = 'gp_historical_db_dq_scale'
    gp_historical_db_max_distance_parameter_name = 'gp_historical_db_max_distance'
    gp_historical_db_query_stride_parameter_name = 'gp_historical_db_query_stride'
    gp_historical_db_disable_online_parameter_name = (
        'gp_historical_db_disable_when_online_update'
    )
    gp_historical_db_fallback_source_parameter_name = 'gp_historical_db_fallback_source'
    gp_historical_db_preflight_enabled_parameter_name = (
        'gp_historical_db_preflight_enabled'
    )
    gp_historical_db_preflight_required_parameter_name = (
        'gp_historical_db_preflight_required'
    )
    gp_historical_db_preflight_mode_parameter_name = (
        'gp_historical_db_preflight_mode'
    )
    gp_historical_db_preflight_duration_sec_parameter_name = (
        'gp_historical_db_preflight_duration_sec'
    )
    gp_historical_db_preflight_min_samples_parameter_name = (
        'gp_historical_db_preflight_min_samples'
    )
    gp_historical_db_preflight_min_pass_ratio_parameter_name = (
        'gp_historical_db_preflight_min_pass_ratio'
    )
    gp_historical_db_preflight_p95_max_distance_parameter_name = (
        'gp_historical_db_preflight_p95_max_distance'
    )
    gp_historical_db_preflight_max_distance_parameter_name = (
        'gp_historical_db_preflight_max_distance'
    )
    gp_historical_db_preflight_log_first_n_parameter_name = (
        'gp_historical_db_preflight_log_first_n'
    )
    gp_disable_silent_hist_fallback_parameter_name = 'gp_disable_silent_hist_fallback'
    gp_triple_weight_mode_parameter_name = 'gp_triple_weight_mode'
    gp_triple_weight_local_parameter_name = 'gp_triple_weight_local'
    gp_triple_weight_cloud_parameter_name = 'gp_triple_weight_cloud'
    gp_triple_weight_hist_parameter_name = 'gp_triple_weight_hist'
    gp_triple_weight_normalize_parameter_name = 'gp_triple_weight_normalize'
    gp_triple_rmse_local_parameter_name = 'gp_triple_rmse_local'
    gp_triple_rmse_cloud_parameter_name = 'gp_triple_rmse_cloud'
    gp_triple_rmse_hist_parameter_name = 'gp_triple_rmse_hist'
    gp_triple_inverse_rmse_eps_parameter_name = 'gp_triple_inverse_rmse_eps'
    gp_triple_hist_distance_scale_parameter_name = 'gp_triple_hist_distance_scale'
    gp_triple_hist_distance_power_parameter_name = 'gp_triple_hist_distance_power'
    gp_triple_hist_weight_cap_parameter_name = 'gp_triple_hist_weight_cap'
    gp_triple_hist_min_weight_parameter_name = 'gp_triple_hist_min_weight'
    gp_triple_dynamic_eps_parameter_name = 'gp_triple_dynamic_eps'
    gp_triple_min_weight_local_parameter_name = 'gp_triple_min_weight_local'
    gp_triple_min_weight_cloud_parameter_name = 'gp_triple_min_weight_cloud'
    gp_triple_require_hist_available_parameter_name = 'gp_triple_require_hist_available'
    gp_triple_fallback_source_parameter_name = 'gp_triple_fallback_source'
    gp_triple_debug_safety_log_enabled_parameter_name = (
        'gp_triple_debug_safety_log_enabled'
    )
    gp_triple_debug_safety_log_first_n_parameter_name = (
        'gp_triple_debug_safety_log_first_n'
    )
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
    # Stage 3A trajectory 参数默认保持 planar_circle，只有显式传参才启用 z modulation。
    trajectory_mode_parameter_name = 'trajectory_mode'
    z_amplitude_parameter_name = 'z_amplitude'
    z_frequency_multiplier_parameter_name = 'z_frequency_multiplier'
    circle_frequency_parameter_name = 'circle_frequency'
    circle_center_x_parameter_name = 'circle_center_x'
    circle_center_y_parameter_name = 'circle_center_y'
    circle_center_z_parameter_name = 'circle_center_z'
    anchor_trajectory_start_to_current_pose_parameter_name = (
        'anchor_trajectory_start_to_current_pose'
    )
    transition_duration_parameter_name = 'transition_duration'
    trajectory_start_distance_warn_m_parameter_name = 'trajectory_start_distance_warn_m'
    trajectory_start_distance_refuse_m_parameter_name = 'trajectory_start_distance_refuse_m'
    trajectory_start_distance_guard_enabled_parameter_name = (
        'trajectory_start_distance_guard_enabled'
    )
    trajectory_max_cartesian_step_m_parameter_name = 'trajectory_max_cartesian_step_m'

    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)
    use_rviz = LaunchConfiguration(use_rviz_parameter_name)
    spawn_gp_server = LaunchConfiguration(spawn_gp_server_parameter_name)
    spawn_cpp_relayer = LaunchConfiguration(spawn_cpp_relayer_parameter_name)
    spawn_update_rate_diagnostic = LaunchConfiguration(
        spawn_update_rate_diagnostic_parameter_name)
    update_rate_diagnostic_expected_rate = LaunchConfiguration(
        update_rate_diagnostic_expected_rate_parameter_name)
    update_rate_diagnostic_log_period_sec = LaunchConfiguration(
        update_rate_diagnostic_log_period_sec_parameter_name)
    control_frequency = LaunchConfiguration(control_frequency_parameter_name)
    allow_high_ros2_control_rate = LaunchConfiguration(
        allow_high_ros2_control_rate_parameter_name)
    ros2_control_update_rate = LaunchConfiguration(ros2_control_update_rate_parameter_name)
    trajectory_publish_rate = LaunchConfiguration(trajectory_publish_rate_parameter_name)
    state_parameter_publish_rate = LaunchConfiguration(state_parameter_publish_rate_parameter_name)
    run_name = LaunchConfiguration(run_name_parameter_name)
    data_output_dir = LaunchConfiguration(data_output_dir_parameter_name)
    csv_output_profile = LaunchConfiguration(csv_output_profile_parameter_name)
    reference_mode = LaunchConfiguration(reference_mode_parameter_name)
    joint_space_command_topic = LaunchConfiguration(joint_space_command_topic_parameter_name)
    gp_prediction_enabled = LaunchConfiguration(gp_prediction_enabled_parameter_name)
    gp_prediction_stride = LaunchConfiguration(gp_prediction_stride_parameter_name)
    gp_output_timeout_sec = LaunchConfiguration(gp_output_timeout_sec_parameter_name)
    future_trajectory_request_stride = LaunchConfiguration(future_trajectory_request_stride_parameter_name)
    gp_online_update_enabled = LaunchConfiguration(gp_online_update_enabled_parameter_name)
    gp_model_dir = LaunchConfiguration(gp_model_dir_parameter_name)
    gp_compensation_enabled = LaunchConfiguration(gp_compensation_enabled_parameter_name)
    gp_compensation_source = LaunchConfiguration(gp_compensation_source_parameter_name)
    gp_compensation_scale = LaunchConfiguration(gp_compensation_scale_parameter_name)
    gp_compensation_clip_nm = LaunchConfiguration(gp_compensation_clip_nm_parameter_name)
    gp_compensation_disable_joint7 = LaunchConfiguration(
        gp_compensation_disable_joint7_parameter_name
    )
    torque_rate_limit_enabled = LaunchConfiguration(
        torque_rate_limit_enabled_parameter_name
    )
    torque_rate_limit_nm_per_s = LaunchConfiguration(
        torque_rate_limit_nm_per_s_parameter_name
    )
    torque_rate_limit_log_first_n = LaunchConfiguration(
        torque_rate_limit_log_first_n_parameter_name
    )
    torque_rate_limit_reset_on_first_command = LaunchConfiguration(
        torque_rate_limit_reset_on_first_command_parameter_name
    )
    delay_steps = LaunchConfiguration(delay_steps_parameter_name)
    timing_logging_enabled = LaunchConfiguration(timing_logging_enabled_parameter_name)
    timing_log_stride = LaunchConfiguration(timing_log_stride_parameter_name)
    timing_output_dir = LaunchConfiguration(timing_output_dir_parameter_name)
    deadline_ratio_warn_threshold = LaunchConfiguration(
        deadline_ratio_warn_threshold_parameter_name
    )
    effort_gap_diagnostics_enabled = LaunchConfiguration(
        effort_gap_diagnostics_enabled_parameter_name
    )
    effort_gap_log_stride = LaunchConfiguration(effort_gap_log_stride_parameter_name)
    effort_gap_warn_sec = LaunchConfiguration(effort_gap_warn_sec_parameter_name)
    callback_wall_warn_sec = LaunchConfiguration(callback_wall_warn_sec_parameter_name)
    gp_historical_db_enabled = LaunchConfiguration(gp_historical_db_enabled_parameter_name)
    gp_historical_db_path = LaunchConfiguration(gp_historical_db_path_parameter_name)
    gp_historical_db_k = LaunchConfiguration(gp_historical_db_k_parameter_name)
    gp_historical_db_q_scale = LaunchConfiguration(gp_historical_db_q_scale_parameter_name)
    gp_historical_db_dq_scale = LaunchConfiguration(gp_historical_db_dq_scale_parameter_name)
    gp_historical_db_max_distance = LaunchConfiguration(
        gp_historical_db_max_distance_parameter_name
    )
    gp_historical_db_query_stride = LaunchConfiguration(
        gp_historical_db_query_stride_parameter_name
    )
    gp_historical_db_disable_online = LaunchConfiguration(
        gp_historical_db_disable_online_parameter_name
    )
    gp_historical_db_fallback_source = LaunchConfiguration(
        gp_historical_db_fallback_source_parameter_name
    )
    gp_historical_db_preflight_enabled = LaunchConfiguration(
        gp_historical_db_preflight_enabled_parameter_name
    )
    gp_historical_db_preflight_required = LaunchConfiguration(
        gp_historical_db_preflight_required_parameter_name
    )
    gp_historical_db_preflight_mode = LaunchConfiguration(
        gp_historical_db_preflight_mode_parameter_name
    )
    gp_historical_db_preflight_duration_sec = LaunchConfiguration(
        gp_historical_db_preflight_duration_sec_parameter_name
    )
    gp_historical_db_preflight_min_samples = LaunchConfiguration(
        gp_historical_db_preflight_min_samples_parameter_name
    )
    gp_historical_db_preflight_min_pass_ratio = LaunchConfiguration(
        gp_historical_db_preflight_min_pass_ratio_parameter_name
    )
    gp_historical_db_preflight_p95_max_distance = LaunchConfiguration(
        gp_historical_db_preflight_p95_max_distance_parameter_name
    )
    gp_historical_db_preflight_max_distance = LaunchConfiguration(
        gp_historical_db_preflight_max_distance_parameter_name
    )
    gp_historical_db_preflight_log_first_n = LaunchConfiguration(
        gp_historical_db_preflight_log_first_n_parameter_name
    )
    gp_disable_silent_hist_fallback = LaunchConfiguration(
        gp_disable_silent_hist_fallback_parameter_name
    )
    gp_triple_weight_mode = LaunchConfiguration(gp_triple_weight_mode_parameter_name)
    gp_triple_weight_local = LaunchConfiguration(gp_triple_weight_local_parameter_name)
    gp_triple_weight_cloud = LaunchConfiguration(gp_triple_weight_cloud_parameter_name)
    gp_triple_weight_hist = LaunchConfiguration(gp_triple_weight_hist_parameter_name)
    gp_triple_weight_normalize = LaunchConfiguration(
        gp_triple_weight_normalize_parameter_name
    )
    gp_triple_rmse_local = LaunchConfiguration(gp_triple_rmse_local_parameter_name)
    gp_triple_rmse_cloud = LaunchConfiguration(gp_triple_rmse_cloud_parameter_name)
    gp_triple_rmse_hist = LaunchConfiguration(gp_triple_rmse_hist_parameter_name)
    gp_triple_inverse_rmse_eps = LaunchConfiguration(
        gp_triple_inverse_rmse_eps_parameter_name
    )
    gp_triple_hist_distance_scale = LaunchConfiguration(
        gp_triple_hist_distance_scale_parameter_name
    )
    gp_triple_hist_distance_power = LaunchConfiguration(
        gp_triple_hist_distance_power_parameter_name
    )
    gp_triple_hist_weight_cap = LaunchConfiguration(
        gp_triple_hist_weight_cap_parameter_name
    )
    gp_triple_hist_min_weight = LaunchConfiguration(
        gp_triple_hist_min_weight_parameter_name
    )
    gp_triple_dynamic_eps = LaunchConfiguration(gp_triple_dynamic_eps_parameter_name)
    gp_triple_min_weight_local = LaunchConfiguration(
        gp_triple_min_weight_local_parameter_name
    )
    gp_triple_min_weight_cloud = LaunchConfiguration(
        gp_triple_min_weight_cloud_parameter_name
    )
    gp_triple_require_hist_available = LaunchConfiguration(
        gp_triple_require_hist_available_parameter_name
    )
    gp_triple_fallback_source = LaunchConfiguration(
        gp_triple_fallback_source_parameter_name
    )
    gp_triple_debug_safety_log_enabled = LaunchConfiguration(
        gp_triple_debug_safety_log_enabled_parameter_name
    )
    gp_triple_debug_safety_log_first_n = LaunchConfiguration(
        gp_triple_debug_safety_log_first_n_parameter_name
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
    trajectory_mode = LaunchConfiguration(trajectory_mode_parameter_name)
    z_amplitude = LaunchConfiguration(z_amplitude_parameter_name)
    z_frequency_multiplier = LaunchConfiguration(z_frequency_multiplier_parameter_name)
    circle_frequency = LaunchConfiguration(circle_frequency_parameter_name)
    circle_center_x = LaunchConfiguration(circle_center_x_parameter_name)
    circle_center_y = LaunchConfiguration(circle_center_y_parameter_name)
    circle_center_z = LaunchConfiguration(circle_center_z_parameter_name)
    anchor_trajectory_start_to_current_pose = LaunchConfiguration(
        anchor_trajectory_start_to_current_pose_parameter_name
    )
    transition_duration = LaunchConfiguration(transition_duration_parameter_name)
    trajectory_start_distance_warn_m = LaunchConfiguration(
        trajectory_start_distance_warn_m_parameter_name
    )
    trajectory_start_distance_refuse_m = LaunchConfiguration(
        trajectory_start_distance_refuse_m_parameter_name
    )
    trajectory_start_distance_guard_enabled = LaunchConfiguration(
        trajectory_start_distance_guard_enabled_parameter_name
    )
    trajectory_max_cartesian_step_m = LaunchConfiguration(
        trajectory_max_cartesian_step_m_parameter_name
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            robot_ip_parameter_name,
            description='Hostname or IP address of the robot.'),
        DeclareLaunchArgument(
            use_rviz_parameter_name,
            default_value='false',
            description='Visualize the robot in Rviz'),
        DeclareLaunchArgument(
            use_fake_hardware_parameter_name,
            default_value='false',
            description='Use fake hardware'),
        DeclareLaunchArgument(
            fake_sensor_commands_parameter_name,
            default_value='false',
            description="Fake sensor commands. Only valid when '{}' is true".format(
                use_fake_hardware_parameter_name)),
        DeclareLaunchArgument(
            spawn_gp_server_parameter_name,
            default_value='false',
            description=(
                'Start standalone gp_server only when explicitly requested; '
                'default false for GOAL1 real shadow validation.'
            )),
        DeclareLaunchArgument(
            spawn_cpp_relayer_parameter_name,
            default_value='true',
            description='Spawn cpp_relayer controller unless explicitly disabled.'),
        DeclareLaunchArgument(
            spawn_update_rate_diagnostic_parameter_name,
            default_value='false',
            description=(
                'Spawn fake-only update_rate_diagnostic controller. Requires '
                'use_fake_hardware:=true and spawn_cpp_relayer:=false.'
            )),
        DeclareLaunchArgument(
            update_rate_diagnostic_expected_rate_parameter_name,
            default_value='1000.0',
            description='Expected controller_manager update rate for update_rate_diagnostic.'),
        DeclareLaunchArgument(
            update_rate_diagnostic_log_period_sec_parameter_name,
            default_value='5.0',
            description='Log period for update_rate_diagnostic summary output.'),
        DeclareLaunchArgument(
            spawn_fake_state_parameter_publisher_parameter_name,
            default_value='false',
            description=(
                'Reserved for GOAL2 fake/sim smoke; current branch has no '
                'goal2 fake state publisher executable, so this launch does '
                'not start an extra node.'
            )),
        DeclareLaunchArgument(
            control_frequency_parameter_name,
            default_value='50',
            description=(
                'Legacy umbrella frequency in Hz; new rate-specific arguments '
                'inherit this value unless explicitly set.'
            )),
        DeclareLaunchArgument(
            allow_high_ros2_control_rate_parameter_name,
            default_value='false',
            description=(
                'Hard opt-in for ros2_control update rates above control_frequency.'
            )),
        DeclareLaunchArgument(
            ros2_control_update_rate_parameter_name,
            default_value=control_frequency,
            description='Controller manager / ros2_control update_rate in Hz.'),
        DeclareLaunchArgument(
            trajectory_publish_rate_parameter_name,
            default_value=control_frequency,
            description='trajectory_publisher timer rate in Hz.'),
        DeclareLaunchArgument(
            state_parameter_publish_rate_parameter_name,
            default_value=control_frequency,
            description='cpp_relayer /state_parameter publish rate in Hz.'),
        LogInfo(
            msg=[
                'Frequency config: control_frequency=',
                control_frequency,
                ', allow_high_ros2_control_rate=',
                allow_high_ros2_control_rate,
                ', ros2_control_update_rate=',
                ros2_control_update_rate,
                ', trajectory_publish_rate=',
                trajectory_publish_rate,
                ', state_parameter_publish_rate=',
                state_parameter_publish_rate,
                ' Hz',
            ]
        ),
        OpaqueFunction(
            function=_guard_frequency_config,
            args=[
                control_frequency,
                ros2_control_update_rate,
                trajectory_publish_rate,
                state_parameter_publish_rate,
                allow_high_ros2_control_rate,
                use_fake_hardware,
                spawn_cpp_relayer,
                spawn_update_rate_diagnostic,
            ],
        ),
        DeclareLaunchArgument(
            run_name_parameter_name,
            default_value='',
            description='Optional run name written to controller CSV metadata and filename.'),
        DeclareLaunchArgument(
            data_output_dir_parameter_name,
            default_value='.',
            description='Directory for controller data CSV output.'),
        DeclareLaunchArgument(
            csv_output_profile_parameter_name,
            default_value='full',
            description='Controller CSV output profile: full or final.'),
        DeclareLaunchArgument(
            load_gripper_parameter_name,
            default_value='true',
            description='Use Franka Gripper as an end-effector, otherwise, the robot is loaded '
                        'without an end-effector.'),
        DeclareLaunchArgument(
            reference_mode_parameter_name,
            default_value='cartesian',
            description='Controller reference mode: cartesian or joint.'),
        DeclareLaunchArgument(
            joint_space_command_topic_parameter_name,
            default_value='/joint_space_command',
            description='JointSpaceCommand topic used only when reference_mode:=joint.'),
        DeclareLaunchArgument(
            gp_prediction_enabled_parameter_name,
            default_value='true',
            description='Enable GP prediction path in the controller.'),
        DeclareLaunchArgument(
            gp_prediction_stride_parameter_name,
            default_value='5',
            description='Run GP predict/update once every N controller callbacks.'),
        DeclareLaunchArgument(
            gp_output_timeout_sec_parameter_name,
            default_value='0.5',
            description='Maximum local/cloud GP output age before active compensation fails closed.'),
        DeclareLaunchArgument(
            future_trajectory_request_stride_parameter_name,
            default_value='5',
            description='Request /future_task_space once every N controller callbacks.'),
        DeclareLaunchArgument(
            gp_online_update_enabled_parameter_name,
            default_value='true',
            description='Enable online GP model updates in the controller.'),
        DeclareLaunchArgument(
            gp_model_dir_parameter_name,
            default_value='./new_structure/gp/gp_models',
            description='Directory containing offline GP model pickle files.'),
        DeclareLaunchArgument(
            gp_compensation_enabled_parameter_name,
            default_value='false',
            description='Enable GP torque compensation.'),
        DeclareLaunchArgument(
            gp_compensation_source_parameter_name,
            default_value='local',
            description=(
                'GP compensation source: local, cloud, combined, hist_db, '
                'triple, or triple_dynamic.'
            )),
        DeclareLaunchArgument(
            gp_compensation_scale_parameter_name,
            default_value='0.1',
            description='Scale applied to GP torque compensation before clipping.'),
        DeclareLaunchArgument(
            gp_compensation_clip_nm_parameter_name,
            default_value='0.5',
            description='Per-joint GP compensation clip in Nm.'),
        DeclareLaunchArgument(
            gp_compensation_disable_joint7_parameter_name,
            default_value='false',
            description='Disable active GP applied torque on joint7 only when explicitly true.'),
        DeclareLaunchArgument(
            torque_rate_limit_enabled_parameter_name,
            default_value='false',
            description='Enable optional per-joint torque slew-rate limiting before /effort_command publish.'),
        DeclareLaunchArgument(
            torque_rate_limit_nm_per_s_parameter_name,
            default_value='80.0',
            description='Scalar per-joint torque slew-rate limit in Nm/s.'),
        DeclareLaunchArgument(
            torque_rate_limit_log_first_n_parameter_name,
            default_value='5',
            description='Log only the first N torque rate-limit clipping events.'),
        DeclareLaunchArgument(
            torque_rate_limit_reset_on_first_command_parameter_name,
            default_value='true',
            description='Initialize limiter state from the first command instead of slewing from zero.'),
        DeclareLaunchArgument(
            delay_steps_parameter_name,
            default_value='0',
            description='Cloud-like control-step delay; not real network cloud latency.'),
        DeclareLaunchArgument(
            timing_logging_enabled_parameter_name,
            default_value='false',
            description='Enable controller timing CSV logging.'),
        DeclareLaunchArgument(
            timing_log_stride_parameter_name,
            default_value='1',
            description='Record one controller timing row every N callbacks.'),
        DeclareLaunchArgument(
            timing_output_dir_parameter_name,
            default_value='outputs/goal12_controller_timing',
            description='Directory for controller timing CSV output.'),
        DeclareLaunchArgument(
            deadline_ratio_warn_threshold_parameter_name,
            default_value='0.8',
            description='Warn in timing summary when max callback deadline ratio reaches this threshold.'),
        DeclareLaunchArgument(
            effort_gap_diagnostics_enabled_parameter_name,
            default_value='false',
            description='Enable low-frequency Python effort publish gap diagnostics.'),
        DeclareLaunchArgument(
            effort_gap_log_stride_parameter_name,
            default_value='100',
            description='Log one Python effort gap diagnostics summary every N callbacks.'),
        DeclareLaunchArgument(
            effort_gap_warn_sec_parameter_name,
            default_value='0.2',
            description='Warn when Python effort publish gap exceeds this duration in seconds.'),
        DeclareLaunchArgument(
            callback_wall_warn_sec_parameter_name,
            default_value='0.02',
            description='Warn when Python stateParameterCallback wall time exceeds this duration.'),
        DeclareLaunchArgument(
            gp_historical_db_enabled_parameter_name,
            default_value='false',
            description=(
                'Enable persistent historical DB CSV query only when explicitly '
                'requested; active torque uses it only with explicit hist_db, '
                'triple, or triple_dynamic source.'
            )),
        DeclareLaunchArgument(
            gp_historical_db_path_parameter_name,
            default_value='',
            description=(
                'Persistent historical DB .npz path for shadow logging or explicit '
                'hist_db/triple/triple_dynamic source; empty keeps the DB unavailable.'
            )),
        DeclareLaunchArgument(
            gp_historical_db_k_parameter_name,
            default_value='25',
            description='Persistent historical DB top-k size for validation.'),
        DeclareLaunchArgument(
            gp_historical_db_q_scale_parameter_name,
            default_value='0.1',
            description='Persistent historical DB joint-position distance scale.'),
        DeclareLaunchArgument(
            gp_historical_db_dq_scale_parameter_name,
            default_value='0.1',
            description='Persistent historical DB joint-velocity distance scale.'),
        DeclareLaunchArgument(
            gp_historical_db_max_distance_parameter_name,
            default_value='1.0',
            description='Persistent historical DB nearest-distance hard gate.'),
        DeclareLaunchArgument(
            gp_historical_db_query_stride_parameter_name,
            default_value='1',
            description=(
                'Query stride for persistent historical DB KNN lookup. '
                '1 preserves per-callback queries; larger values reuse the previous query result '
                'between lookup callbacks to reduce load.'
            ),
        ),
        DeclareLaunchArgument(
            gp_historical_db_disable_online_parameter_name,
            default_value='true',
            description=(
                'Keep persistent historical DB unavailable while GP online update is '
                'enabled; hist_db active source therefore uses zero fallback in that mode.'
            )),
        DeclareLaunchArgument(
            gp_historical_db_fallback_source_parameter_name,
            default_value='cloud',
            description='Shadow-only fallback source: none, local, cloud, or combined.'),
        DeclareLaunchArgument(
            gp_historical_db_preflight_enabled_parameter_name,
            default_value='false',
            description='Enable hist_db source preflight diagnostics only when explicitly requested.'),
        DeclareLaunchArgument(
            gp_historical_db_preflight_required_parameter_name,
            default_value='false',
            description='Require passing hist_db preflight before active hist_db torque compensation.'),
        DeclareLaunchArgument(
            gp_historical_db_preflight_mode_parameter_name,
            default_value='segment',
            description='Hist DB preflight mode: single, segment, or single_and_segment.'),
        DeclareLaunchArgument(
            gp_historical_db_preflight_duration_sec_parameter_name,
            default_value='5.0',
            description='Hist DB segment preflight duration in seconds.'),
        DeclareLaunchArgument(
            gp_historical_db_preflight_min_samples_parameter_name,
            default_value='50',
            description='Minimum hist DB preflight query samples before pass/fail.'),
        DeclareLaunchArgument(
            gp_historical_db_preflight_min_pass_ratio_parameter_name,
            default_value='0.95',
            description='Minimum hist DB preflight pass ratio.'),
        DeclareLaunchArgument(
            gp_historical_db_preflight_p95_max_distance_parameter_name,
            default_value='1.5',
            description='Hist DB preflight nearest-distance p95 threshold.'),
        DeclareLaunchArgument(
            gp_historical_db_preflight_max_distance_parameter_name,
            default_value='2.0',
            description='Hist DB preflight nearest-distance max threshold.'),
        DeclareLaunchArgument(
            gp_historical_db_preflight_log_first_n_parameter_name,
            default_value='5',
            description='Number of initial hist DB preflight probe logs.'),
        DeclareLaunchArgument(
            gp_disable_silent_hist_fallback_parameter_name,
            default_value='false',
            description='Fail closed instead of silently using hist_db fallback when active hist_db is invalid.'),
        DeclareLaunchArgument(
            gp_triple_weight_mode_parameter_name,
            default_value='inverse_rmse',
            description='Triple fusion weight mode: fixed or inverse_rmse.'),
        DeclareLaunchArgument(
            gp_triple_weight_local_parameter_name,
            default_value='0.10',
            description='Fixed-mode triple fusion local weight.'),
        DeclareLaunchArgument(
            gp_triple_weight_cloud_parameter_name,
            default_value='0.20',
            description='Fixed-mode triple fusion cloud-like weight.'),
        DeclareLaunchArgument(
            gp_triple_weight_hist_parameter_name,
            default_value='0.70',
            description='Fixed-mode triple fusion historical DB weight.'),
        DeclareLaunchArgument(
            gp_triple_weight_normalize_parameter_name,
            default_value='true',
            description='Normalize fixed triple fusion weights before safety constraints.'),
        DeclareLaunchArgument(
            gp_triple_rmse_local_parameter_name,
            default_value='0.330269',
            description='Inverse-RMSE triple fusion local RMSE.'),
        DeclareLaunchArgument(
            gp_triple_rmse_cloud_parameter_name,
            default_value='0.330278',
            description='Inverse-RMSE triple fusion cloud-like RMSE.'),
        DeclareLaunchArgument(
            gp_triple_rmse_hist_parameter_name,
            default_value='0.093071',
            description='Inverse-RMSE triple fusion historical DB RMSE.'),
        DeclareLaunchArgument(
            gp_triple_inverse_rmse_eps_parameter_name,
            default_value='1e-9',
            description='Positive epsilon used by inverse-RMSE squared triple weights.'),
        DeclareLaunchArgument(
            gp_triple_hist_distance_scale_parameter_name,
            default_value='2.0',
            description='Distance scale for triple_dynamic historical confidence penalty.'),
        DeclareLaunchArgument(
            gp_triple_hist_distance_power_parameter_name,
            default_value='2.0',
            description='Distance power for triple_dynamic historical confidence penalty.'),
        DeclareLaunchArgument(
            gp_triple_hist_weight_cap_parameter_name,
            default_value='0.70',
            description='Fractional cap for active triple historical DB weight.'),
        DeclareLaunchArgument(
            gp_triple_hist_min_weight_parameter_name,
            default_value='0.0',
            description='Optional minimum historical DB weight for triple_dynamic.'),
        DeclareLaunchArgument(
            gp_triple_dynamic_eps_parameter_name,
            default_value='1e-9',
            description='Positive epsilon used by triple_dynamic precision weights.'),
        DeclareLaunchArgument(
            gp_triple_min_weight_local_parameter_name,
            default_value='0.05',
            description='Minimum local weight for triple fusion when feasible.'),
        DeclareLaunchArgument(
            gp_triple_min_weight_cloud_parameter_name,
            default_value='0.05',
            description='Minimum cloud-like weight for triple fusion when feasible.'),
        DeclareLaunchArgument(
            gp_triple_require_hist_available_parameter_name,
            default_value='true',
            description='Use triple fallback unless gated historical DB prediction is available.'),
        DeclareLaunchArgument(
            gp_triple_fallback_source_parameter_name,
            default_value='combined',
            description='Triple fallback source: none, local, cloud, combined, or hist_db.'),
        DeclareLaunchArgument(
            gp_triple_debug_safety_log_enabled_parameter_name,
            default_value='true',
            description='Log first-N triple compensation safety values when active.'),
        DeclareLaunchArgument(
            gp_triple_debug_safety_log_first_n_parameter_name,
            default_value='5',
            description='Number of initial triple compensation callbacks to safety-log.'),
        DeclareLaunchArgument(
            gp_historical_soft_shadow_enabled_parameter_name,
            default_value='false',
            description=(
                'Enable GOAL1 historical soft-weight CSV shadow logging only when '
                'explicitly requested for validation; no active torque compensation.'
            )),
        DeclareLaunchArgument(
            gp_historical_soft_alpha_parameter_name,
            default_value='1.0',
            description=(
                'GOAL1 historical soft-shadow alpha for validation logging only; '
                'no active torque compensation.'
            )),
        DeclareLaunchArgument(
            gp_historical_soft_distance_threshold_parameter_name,
            default_value='0.2',
            description=(
                'GOAL1 historical soft-shadow nearest-distance threshold for '
                'validation logging only; no active torque compensation.'
            )),
        DeclareLaunchArgument(
            gp_historical_soft_online_scale_parameter_name,
            default_value='0.02',
            description=(
                'GOAL1 historical soft-shadow online-mode historical scale for '
                'validation logging only; no active torque compensation.'
            )),
        DeclareLaunchArgument(
            gp_historical_soft_non_online_scale_parameter_name,
            default_value='1.0',
            description=(
                'GOAL1 historical soft-shadow non-online historical scale for '
                'validation logging only; no active torque compensation.'
            )),
        # Stage 3A launch defaults 保持 Stage 1 / Stage 2A 的平面圆轨迹行为。
        DeclareLaunchArgument(
            trajectory_mode_parameter_name,
            default_value='planar_circle',
            description='Trajectory mode: planar_circle, z_modulated_circle, or goal1_spatial_multisine.'),
        DeclareLaunchArgument(
            z_amplitude_parameter_name,
            default_value='0.0',
            description='Z modulation amplitude in meters for z_modulated_circle.'),
        DeclareLaunchArgument(
            z_frequency_multiplier_parameter_name,
            default_value='0.5',
            description='Z modulation frequency multiplier relative to circle omega.'),
        DeclareLaunchArgument(
            circle_frequency_parameter_name,
            default_value='0.1',
            description='Circle trajectory frequency in Hz.'),
        DeclareLaunchArgument(
            circle_center_x_parameter_name,
            default_value='0.3',
            description='Cartesian trajectory center x [m].'),
        DeclareLaunchArgument(
            circle_center_y_parameter_name,
            default_value='0.0',
            description='Cartesian trajectory center y [m].'),
        DeclareLaunchArgument(
            circle_center_z_parameter_name,
            default_value='0.65',
            description='Cartesian trajectory center z [m].'),
        DeclareLaunchArgument(
            anchor_trajectory_start_to_current_pose_parameter_name,
            default_value='false',
            description=(
                'If true, shift the GOAL1 spatial multisine center at runtime so '
                'the trajectory start point matches the measured current EE pose.'
            )),
        DeclareLaunchArgument(
            transition_duration_parameter_name,
            default_value='3.0',
            description='Smooth transition duration before trajectory recording starts.'),
        DeclareLaunchArgument(
            trajectory_start_distance_warn_m_parameter_name,
            default_value='0.03',
            description='Warn if current EE pose is farther than this from trajectory start.'),
        DeclareLaunchArgument(
            trajectory_start_distance_refuse_m_parameter_name,
            default_value='0.12',
            description='Refuse trajectory start if current EE pose is farther than this.'),
        DeclareLaunchArgument(
            trajectory_start_distance_guard_enabled_parameter_name,
            default_value='true',
            description='Enable trajectory start distance warn/refuse guard.'),
        DeclareLaunchArgument(
            trajectory_max_cartesian_step_m_parameter_name,
            default_value='0.0',
            description='Optional transition-only Cartesian step clamp in meters; 0 disables.'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([PathJoinSubstitution(
                [FindPackageShare('new_bringup'), 'launch', 'franka.launch.py'])]),
            launch_arguments={robot_ip_parameter_name: robot_ip,
                              load_gripper_parameter_name: load_gripper,
                              use_fake_hardware_parameter_name: use_fake_hardware,
                              fake_sensor_commands_parameter_name: fake_sensor_commands,
                              use_rviz_parameter_name: use_rviz,
                              control_frequency_parameter_name: control_frequency,
                              allow_high_ros2_control_rate_parameter_name: allow_high_ros2_control_rate,
                              ros2_control_update_rate_parameter_name: ros2_control_update_rate
                              }.items(),
        ),

        LogInfo(
            msg=(
                'cpp_relayer safety parameters are loaded from stable '
                'new_bringup/config/controllers.yaml; no runtime spawner '
                'param-file override is used. state_parameter_publish_rate is '
                'currently fixed by controllers.yaml for L0 safety recovery.'
            )
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['cpp_relayer'],
            output='screen',
            condition=IfCondition(spawn_cpp_relayer),
        ),
        OpaqueFunction(
            function=_make_update_rate_diagnostic_spawner,
            args=[
                update_rate_diagnostic_expected_rate,
                update_rate_diagnostic_log_period_sec,
            ],
            condition=IfCondition(spawn_update_rate_diagnostic),
        ),
        Node(
            package='py_controllers',
            executable='gp_server',          # 和 setup.py 里 entry_points 名字一致
            name='gp_server',
            output='screen',
            condition=IfCondition(spawn_gp_server),
        ),

        Node(
            package='py_controllers',
            executable='cartesian_impedance',
            name='cartesian_impedance',
            output='screen',
            parameters=[{
                reference_mode_parameter_name: ParameterValue(reference_mode, value_type=str),
                joint_space_command_topic_parameter_name: joint_space_command_topic,
                gp_prediction_enabled_parameter_name: gp_prediction_enabled,
                gp_prediction_stride_parameter_name: ParameterValue(gp_prediction_stride, value_type=int),
                gp_output_timeout_sec_parameter_name: ParameterValue(
                    gp_output_timeout_sec, value_type=float),
                future_trajectory_request_stride_parameter_name: ParameterValue(future_trajectory_request_stride, value_type=int),
                gp_online_update_enabled_parameter_name: gp_online_update_enabled,
                gp_model_dir_parameter_name: gp_model_dir,
                gp_compensation_enabled_parameter_name: gp_compensation_enabled,
                gp_compensation_source_parameter_name: gp_compensation_source,
                gp_compensation_scale_parameter_name: gp_compensation_scale,
                gp_compensation_clip_nm_parameter_name: gp_compensation_clip_nm,
                gp_compensation_disable_joint7_parameter_name: gp_compensation_disable_joint7,
                torque_rate_limit_enabled_parameter_name: ParameterValue(
                    torque_rate_limit_enabled, value_type=bool),
                torque_rate_limit_nm_per_s_parameter_name: ParameterValue(
                    torque_rate_limit_nm_per_s, value_type=float),
                torque_rate_limit_log_first_n_parameter_name: ParameterValue(
                    torque_rate_limit_log_first_n, value_type=int),
                torque_rate_limit_reset_on_first_command_parameter_name: ParameterValue(
                    torque_rate_limit_reset_on_first_command, value_type=bool),
                delay_steps_parameter_name: ParameterValue(
                    delay_steps, value_type=int),
                control_frequency_parameter_name: ParameterValue(
                    control_frequency, value_type=float),
                ros2_control_update_rate_parameter_name: ParameterValue(
                    ros2_control_update_rate, value_type=float),
                trajectory_publish_rate_parameter_name: ParameterValue(
                    trajectory_publish_rate, value_type=float),
                state_parameter_publish_rate_parameter_name: ParameterValue(
                    state_parameter_publish_rate, value_type=float),
                run_name_parameter_name: ParameterValue(run_name, value_type=str),
                data_output_dir_parameter_name: ParameterValue(data_output_dir, value_type=str),
                csv_output_profile_parameter_name: ParameterValue(
                    csv_output_profile, value_type=str),
                trajectory_mode_parameter_name: ParameterValue(
                    trajectory_mode, value_type=str),
                circle_frequency_parameter_name: ParameterValue(
                    circle_frequency, value_type=float),
                transition_duration_parameter_name: ParameterValue(
                    transition_duration, value_type=float),
                timing_logging_enabled_parameter_name: ParameterValue(
                    timing_logging_enabled, value_type=bool),
                timing_log_stride_parameter_name: ParameterValue(
                    timing_log_stride, value_type=int),
                timing_output_dir_parameter_name: ParameterValue(
                    timing_output_dir, value_type=str),
                deadline_ratio_warn_threshold_parameter_name: ParameterValue(
                    deadline_ratio_warn_threshold, value_type=float),
                effort_gap_diagnostics_enabled_parameter_name: ParameterValue(
                    effort_gap_diagnostics_enabled, value_type=bool),
                effort_gap_log_stride_parameter_name: ParameterValue(
                    effort_gap_log_stride, value_type=int),
                effort_gap_warn_sec_parameter_name: ParameterValue(
                    effort_gap_warn_sec, value_type=float),
                callback_wall_warn_sec_parameter_name: ParameterValue(
                    callback_wall_warn_sec, value_type=float),
                gp_historical_db_enabled_parameter_name: ParameterValue(
                    gp_historical_db_enabled,
                    value_type=bool),
                gp_historical_db_path_parameter_name: ParameterValue(
                    gp_historical_db_path,
                    value_type=str),
                gp_historical_db_k_parameter_name: ParameterValue(
                    gp_historical_db_k,
                    value_type=int),
                gp_historical_db_q_scale_parameter_name: ParameterValue(
                    gp_historical_db_q_scale,
                    value_type=float),
                gp_historical_db_dq_scale_parameter_name: ParameterValue(
                    gp_historical_db_dq_scale,
                    value_type=float),
                gp_historical_db_max_distance_parameter_name: ParameterValue(
                    gp_historical_db_max_distance,
                    value_type=float),
                gp_historical_db_query_stride_parameter_name: ParameterValue(
                    gp_historical_db_query_stride,
                    value_type=int),
                gp_historical_db_disable_online_parameter_name: ParameterValue(
                    gp_historical_db_disable_online,
                    value_type=bool),
                gp_historical_db_fallback_source_parameter_name: ParameterValue(
                    gp_historical_db_fallback_source,
                    value_type=str),
                gp_historical_db_preflight_enabled_parameter_name: ParameterValue(
                    gp_historical_db_preflight_enabled,
                    value_type=bool),
                gp_historical_db_preflight_required_parameter_name: ParameterValue(
                    gp_historical_db_preflight_required,
                    value_type=bool),
                gp_historical_db_preflight_mode_parameter_name: ParameterValue(
                    gp_historical_db_preflight_mode,
                    value_type=str),
                gp_historical_db_preflight_duration_sec_parameter_name: ParameterValue(
                    gp_historical_db_preflight_duration_sec,
                    value_type=float),
                gp_historical_db_preflight_min_samples_parameter_name: ParameterValue(
                    gp_historical_db_preflight_min_samples,
                    value_type=int),
                gp_historical_db_preflight_min_pass_ratio_parameter_name: ParameterValue(
                    gp_historical_db_preflight_min_pass_ratio,
                    value_type=float),
                gp_historical_db_preflight_p95_max_distance_parameter_name: ParameterValue(
                    gp_historical_db_preflight_p95_max_distance,
                    value_type=float),
                gp_historical_db_preflight_max_distance_parameter_name: ParameterValue(
                    gp_historical_db_preflight_max_distance,
                    value_type=float),
                gp_historical_db_preflight_log_first_n_parameter_name: ParameterValue(
                    gp_historical_db_preflight_log_first_n,
                    value_type=int),
                gp_disable_silent_hist_fallback_parameter_name: ParameterValue(
                    gp_disable_silent_hist_fallback,
                    value_type=bool),
                gp_triple_weight_mode_parameter_name: ParameterValue(
                    gp_triple_weight_mode,
                    value_type=str),
                gp_triple_weight_local_parameter_name: ParameterValue(
                    gp_triple_weight_local,
                    value_type=float),
                gp_triple_weight_cloud_parameter_name: ParameterValue(
                    gp_triple_weight_cloud,
                    value_type=float),
                gp_triple_weight_hist_parameter_name: ParameterValue(
                    gp_triple_weight_hist,
                    value_type=float),
                gp_triple_weight_normalize_parameter_name: ParameterValue(
                    gp_triple_weight_normalize,
                    value_type=bool),
                gp_triple_rmse_local_parameter_name: ParameterValue(
                    gp_triple_rmse_local,
                    value_type=float),
                gp_triple_rmse_cloud_parameter_name: ParameterValue(
                    gp_triple_rmse_cloud,
                    value_type=float),
                gp_triple_rmse_hist_parameter_name: ParameterValue(
                    gp_triple_rmse_hist,
                    value_type=float),
                gp_triple_inverse_rmse_eps_parameter_name: ParameterValue(
                    gp_triple_inverse_rmse_eps,
                    value_type=float),
                gp_triple_hist_distance_scale_parameter_name: ParameterValue(
                    gp_triple_hist_distance_scale,
                    value_type=float),
                gp_triple_hist_distance_power_parameter_name: ParameterValue(
                    gp_triple_hist_distance_power,
                    value_type=float),
                gp_triple_hist_weight_cap_parameter_name: ParameterValue(
                    gp_triple_hist_weight_cap,
                    value_type=float),
                gp_triple_hist_min_weight_parameter_name: ParameterValue(
                    gp_triple_hist_min_weight,
                    value_type=float),
                gp_triple_dynamic_eps_parameter_name: ParameterValue(
                    gp_triple_dynamic_eps,
                    value_type=float),
                gp_triple_min_weight_local_parameter_name: ParameterValue(
                    gp_triple_min_weight_local,
                    value_type=float),
                gp_triple_min_weight_cloud_parameter_name: ParameterValue(
                    gp_triple_min_weight_cloud,
                    value_type=float),
                gp_triple_require_hist_available_parameter_name: ParameterValue(
                    gp_triple_require_hist_available,
                    value_type=bool),
                gp_triple_fallback_source_parameter_name: ParameterValue(
                    gp_triple_fallback_source,
                    value_type=str),
                gp_triple_debug_safety_log_enabled_parameter_name: ParameterValue(
                    gp_triple_debug_safety_log_enabled,
                    value_type=bool),
                gp_triple_debug_safety_log_first_n_parameter_name: ParameterValue(
                    gp_triple_debug_safety_log_first_n,
                    value_type=int),
                gp_historical_soft_shadow_enabled_parameter_name: ParameterValue(
                    gp_historical_soft_shadow_enabled,
                    value_type=bool),
                gp_historical_soft_alpha_parameter_name: ParameterValue(
                    gp_historical_soft_alpha,
                    value_type=float),
                gp_historical_soft_distance_threshold_parameter_name: ParameterValue(
                    gp_historical_soft_distance_threshold,
                    value_type=float),
                gp_historical_soft_online_scale_parameter_name: ParameterValue(
                    gp_historical_soft_online_scale,
                    value_type=float),
                gp_historical_soft_non_online_scale_parameter_name: ParameterValue(
                    gp_historical_soft_non_online_scale,
                    value_type=float),
            }]
        ),
        Node(
            package='py_controllers',
            executable='trajectory_publisher',
            name='trajectory_publisher',
            output='screen',
            parameters=[{
                control_frequency_parameter_name: ParameterValue(
                    control_frequency, value_type=float),
                trajectory_publish_rate_parameter_name: ParameterValue(
                    trajectory_publish_rate, value_type=float),
                trajectory_mode_parameter_name: ParameterValue(trajectory_mode, value_type=str),
                z_amplitude_parameter_name: ParameterValue(z_amplitude, value_type=float),
                z_frequency_multiplier_parameter_name: ParameterValue(
                    z_frequency_multiplier, value_type=float),
                circle_frequency_parameter_name: ParameterValue(circle_frequency, value_type=float),
                circle_center_x_parameter_name: ParameterValue(circle_center_x, value_type=float),
                circle_center_y_parameter_name: ParameterValue(circle_center_y, value_type=float),
                circle_center_z_parameter_name: ParameterValue(circle_center_z, value_type=float),
                anchor_trajectory_start_to_current_pose_parameter_name: ParameterValue(
                    anchor_trajectory_start_to_current_pose, value_type=bool),
                transition_duration_parameter_name: ParameterValue(
                    transition_duration, value_type=float),
                trajectory_start_distance_warn_m_parameter_name: ParameterValue(
                    trajectory_start_distance_warn_m, value_type=float),
                trajectory_start_distance_refuse_m_parameter_name: ParameterValue(
                    trajectory_start_distance_refuse_m, value_type=float),
                trajectory_start_distance_guard_enabled_parameter_name: ParameterValue(
                    trajectory_start_distance_guard_enabled, value_type=bool),
                trajectory_max_cartesian_step_m_parameter_name: ParameterValue(
                    trajectory_max_cartesian_step_m, value_type=float),
            }]
        ),
        # Node(
        #     package='py_controllers',
        #     executable='trajectory_eclipse_publisher',
        #     name='trajectory_eclipse_publisher',
        #     output='screen',
        #     # parameters=[{
        #     #     'circle_radius': 0.2,
        #     #     'circle_frequency': 0.5,
        #     #     'circle_center_x': 0.5,
        #     #     'circle_center_y': 0.0,
        #     #     'circle_center_z': 0.3,
        #     #     'bank_angle': 10.0,
        #     # }]
        # )
    ])
