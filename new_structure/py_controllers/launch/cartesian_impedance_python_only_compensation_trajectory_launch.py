#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    run_name = LaunchConfiguration('run_name')
    data_output_dir = LaunchConfiguration('data_output_dir')
    csv_output_profile = LaunchConfiguration('csv_output_profile')
    trajectory_mode = LaunchConfiguration('trajectory_mode')
    circle_frequency = LaunchConfiguration('circle_frequency')
    rounds_per_mode = LaunchConfiguration('rounds_per_mode')
    control_frequency = LaunchConfiguration('control_frequency')
    trajectory_publish_rate = LaunchConfiguration('trajectory_publish_rate')
    state_parameter_publish_rate = LaunchConfiguration(
        'state_parameter_publish_rate'
    )
    transition_duration = LaunchConfiguration('transition_duration')
    torque_rate_limit_nm_per_s = LaunchConfiguration(
        'torque_rate_limit_nm_per_s'
    )
    startup_linear_speed = LaunchConfiguration('startup_linear_speed')
    startup_distance_warn_m = LaunchConfiguration('startup_distance_warn_m')
    startup_distance_refuse_m = LaunchConfiguration(
        'startup_distance_refuse_m'
    )
    startup_distance_refuse_enabled = LaunchConfiguration(
        'startup_distance_refuse_enabled'
    )
    session_home_mode = LaunchConfiguration('session_home_mode')
    session_home_path = LaunchConfiguration('session_home_path')
    session_home_capture_enabled = LaunchConfiguration(
        'session_home_capture_enabled'
    )
    session_home_capture_max_distance_from_nominal_m = LaunchConfiguration(
        'session_home_capture_max_distance_from_nominal_m'
    )
    session_home_capture_requires_stable_state = LaunchConfiguration(
        'session_home_capture_requires_stable_state'
    )
    session_home_capture_stability_samples = LaunchConfiguration(
        'session_home_capture_stability_samples'
    )
    session_home_capture_stability_position_std_m = LaunchConfiguration(
        'session_home_capture_stability_position_std_m'
    )
    session_home_capture_min_z = LaunchConfiguration(
        'session_home_capture_min_z'
    )
    session_home_capture_max_z = LaunchConfiguration(
        'session_home_capture_max_z'
    )
    session_home_joint_check_enabled = LaunchConfiguration(
        'session_home_joint_check_enabled'
    )
    session_home_joint_check_required_for_hist = LaunchConfiguration(
        'session_home_joint_check_required_for_hist'
    )
    session_home_joint_max_abs_warn_rad = LaunchConfiguration(
        'session_home_joint_max_abs_warn_rad'
    )
    session_home_joint_max_abs_refuse_rad = LaunchConfiguration(
        'session_home_joint_max_abs_refuse_rad'
    )
    session_home_joint_l2_warn_rad = LaunchConfiguration(
        'session_home_joint_l2_warn_rad'
    )
    session_home_joint_l2_refuse_rad = LaunchConfiguration(
        'session_home_joint_l2_refuse_rad'
    )
    session_home_dq_stillness_warn_rad_s = LaunchConfiguration(
        'session_home_dq_stillness_warn_rad_s'
    )
    session_home_dq_stillness_refuse_rad_s = LaunchConfiguration(
        'session_home_dq_stillness_refuse_rad_s'
    )
    trajectory_reference_mode = LaunchConfiguration(
        'trajectory_reference_mode'
    )
    session_relative_capture_enabled = LaunchConfiguration(
        'session_relative_capture_enabled'
    )
    session_relative_max_anchor_delta_m = LaunchConfiguration(
        'session_relative_max_anchor_delta_m'
    )
    session_relative_warn_anchor_delta_m = LaunchConfiguration(
        'session_relative_warn_anchor_delta_m'
    )
    session_relative_anchor_delta_limit_mode = LaunchConfiguration(
        'session_relative_anchor_delta_limit_mode'
    )
    session_relative_min_z = LaunchConfiguration('session_relative_min_z')
    session_relative_max_z = LaunchConfiguration('session_relative_max_z')
    session_relative_requires_stable_state = LaunchConfiguration(
        'session_relative_requires_stable_state'
    )
    session_relative_stability_samples = LaunchConfiguration(
        'session_relative_stability_samples'
    )
    session_relative_stability_position_std_m = LaunchConfiguration(
        'session_relative_stability_position_std_m'
    )
    session_relative_apply_to_startup_and_return = LaunchConfiguration(
        'session_relative_apply_to_startup_and_return'
    )
    session_relative_apply_to_trajectory_center = LaunchConfiguration(
        'session_relative_apply_to_trajectory_center'
    )
    session_relative_nominal_trajectory_start_xyz = LaunchConfiguration(
        'session_relative_nominal_trajectory_start_xyz'
    )
    session_relative_nominal_circle_center_xyz = LaunchConfiguration(
        'session_relative_nominal_circle_center_xyz'
    )
    normal_run_start_gate_enabled = LaunchConfiguration(
        'normal_run_start_gate_enabled'
    )
    normal_run_start_warn_m = LaunchConfiguration('normal_run_start_warn_m')
    normal_run_start_refuse_m = LaunchConfiguration(
        'normal_run_start_refuse_m'
    )
    emergency_return_start_refuse_m = LaunchConfiguration(
        'emergency_return_start_refuse_m'
    )
    return_only_if_too_far_enabled = LaunchConfiguration(
        'return_only_if_too_far_enabled'
    )
    post_run_return_to_session_home_enabled = LaunchConfiguration(
        'post_run_return_to_session_home_enabled'
    )
    post_run_return_linear_speed = LaunchConfiguration(
        'post_run_return_linear_speed'
    )
    post_run_return_timeout_sec = LaunchConfiguration(
        'post_run_return_timeout_sec'
    )
    post_run_return_hold_sec = LaunchConfiguration(
        'post_run_return_hold_sec'
    )
    post_run_return_tolerance_m = LaunchConfiguration(
        'post_run_return_tolerance_m'
    )
    post_run_return_disable_gp_compensation = LaunchConfiguration(
        'post_run_return_disable_gp_compensation'
    )
    post_run_return_disable_online_update = LaunchConfiguration(
        'post_run_return_disable_online_update'
    )
    post_run_return_wait_timeout_sec = LaunchConfiguration(
        'post_run_return_wait_timeout_sec'
    )
    gp_model_dir = LaunchConfiguration('gp_model_dir')
    gp_prediction_enabled = LaunchConfiguration('gp_prediction_enabled')
    gp_online_update_enabled = LaunchConfiguration('gp_online_update_enabled')
    gp_compensation_enabled = LaunchConfiguration('gp_compensation_enabled')
    gp_compensation_source = LaunchConfiguration('gp_compensation_source')
    gp_compensation_scale = LaunchConfiguration('gp_compensation_scale')
    gp_compensation_clip_nm = LaunchConfiguration('gp_compensation_clip_nm')
    gp_compensation_disable_joint7 = LaunchConfiguration(
        'gp_compensation_disable_joint7'
    )
    delay_steps = LaunchConfiguration('delay_steps')
    gp_prediction_stride = LaunchConfiguration('gp_prediction_stride')
    future_trajectory_request_stride = LaunchConfiguration(
        'future_trajectory_request_stride'
    )
    gp_output_timeout_sec = LaunchConfiguration('gp_output_timeout_sec')
    gp_historical_db_enabled = LaunchConfiguration('gp_historical_db_enabled')
    gp_historical_db_path = LaunchConfiguration('gp_historical_db_path')
    gp_historical_db_k = LaunchConfiguration('gp_historical_db_k')
    gp_historical_db_q_scale = LaunchConfiguration('gp_historical_db_q_scale')
    gp_historical_db_dq_scale = LaunchConfiguration('gp_historical_db_dq_scale')
    gp_historical_db_max_distance = LaunchConfiguration(
        'gp_historical_db_max_distance'
    )
    gp_historical_db_require_distance_pass_for_active = LaunchConfiguration(
        'gp_historical_db_require_distance_pass_for_active'
    )
    gp_historical_db_distance_contribution_logging = LaunchConfiguration(
        'gp_historical_db_distance_contribution_logging'
    )
    gp_historical_db_metadata_path = LaunchConfiguration(
        'gp_historical_db_metadata_path'
    )
    gp_historical_db_metadata_enforcement_enabled = LaunchConfiguration(
        'gp_historical_db_metadata_enforcement_enabled'
    )
    gp_historical_db_query_stride = LaunchConfiguration(
        'gp_historical_db_query_stride'
    )
    gp_historical_db_disable_when_online_update = LaunchConfiguration(
        'gp_historical_db_disable_when_online_update'
    )
    gp_historical_db_fallback_source = LaunchConfiguration(
        'gp_historical_db_fallback_source'
    )
    gp_historical_db_preflight_enabled = LaunchConfiguration(
        'gp_historical_db_preflight_enabled'
    )
    gp_historical_db_preflight_required = LaunchConfiguration(
        'gp_historical_db_preflight_required'
    )
    gp_historical_db_preflight_mode = LaunchConfiguration(
        'gp_historical_db_preflight_mode'
    )
    gp_historical_db_preflight_duration_sec = LaunchConfiguration(
        'gp_historical_db_preflight_duration_sec'
    )
    gp_historical_db_preflight_min_samples = LaunchConfiguration(
        'gp_historical_db_preflight_min_samples'
    )
    gp_historical_db_preflight_min_pass_ratio = LaunchConfiguration(
        'gp_historical_db_preflight_min_pass_ratio'
    )
    gp_historical_db_preflight_p95_max_distance = LaunchConfiguration(
        'gp_historical_db_preflight_p95_max_distance'
    )
    gp_historical_db_preflight_max_distance = LaunchConfiguration(
        'gp_historical_db_preflight_max_distance'
    )
    gp_disable_silent_hist_fallback = LaunchConfiguration(
        'gp_disable_silent_hist_fallback'
    )
    gp_triple_weight_mode = LaunchConfiguration('gp_triple_weight_mode')
    gp_triple_weight_local = LaunchConfiguration('gp_triple_weight_local')
    gp_triple_weight_cloud = LaunchConfiguration('gp_triple_weight_cloud')
    gp_triple_weight_hist = LaunchConfiguration('gp_triple_weight_hist')
    gp_triple_weight_normalize = LaunchConfiguration(
        'gp_triple_weight_normalize'
    )
    gp_triple_rmse_local = LaunchConfiguration('gp_triple_rmse_local')
    gp_triple_rmse_cloud = LaunchConfiguration('gp_triple_rmse_cloud')
    gp_triple_rmse_hist = LaunchConfiguration('gp_triple_rmse_hist')
    gp_triple_hist_distance_scale = LaunchConfiguration(
        'gp_triple_hist_distance_scale'
    )
    gp_triple_hist_distance_power = LaunchConfiguration(
        'gp_triple_hist_distance_power'
    )
    gp_triple_hist_weight_cap = LaunchConfiguration(
        'gp_triple_hist_weight_cap'
    )
    gp_triple_hist_min_weight = LaunchConfiguration(
        'gp_triple_hist_min_weight'
    )
    gp_triple_min_weight_local = LaunchConfiguration(
        'gp_triple_min_weight_local'
    )
    gp_triple_min_weight_cloud = LaunchConfiguration(
        'gp_triple_min_weight_cloud'
    )
    gp_triple_require_hist_available = LaunchConfiguration(
        'gp_triple_require_hist_available'
    )
    gp_triple_fallback_source = LaunchConfiguration(
        'gp_triple_fallback_source'
    )
    gp_triple_debug_safety_log_enabled = LaunchConfiguration(
        'gp_triple_debug_safety_log_enabled'
    )
    gp_triple_debug_safety_log_first_n = LaunchConfiguration(
        'gp_triple_debug_safety_log_first_n'
    )
    gp_triple_combined_base_shadow_enabled = LaunchConfiguration(
        'gp_triple_combined_base_shadow_enabled'
    )
    gp_triple_combined_base_hist_weight_cap = LaunchConfiguration(
        'gp_triple_combined_base_hist_weight_cap'
    )
    gp_triple_combined_base_hist_weight_ramp_sec = LaunchConfiguration(
        'gp_triple_combined_base_hist_weight_ramp_sec'
    )
    gp_triple_gated_hist_cap_f50 = LaunchConfiguration(
        'gp_triple_gated_hist_cap_f50'
    )
    gp_triple_gated_hist_cap_f100 = LaunchConfiguration(
        'gp_triple_gated_hist_cap_f100'
    )
    gp_triple_gated_hist_cap_f200 = LaunchConfiguration(
        'gp_triple_gated_hist_cap_f200'
    )
    gp_triple_gated_disagreement_ref_norm = LaunchConfiguration(
        'gp_triple_gated_disagreement_ref_norm'
    )
    gp_triple_gated_disagreement_hard_max_norm = LaunchConfiguration(
        'gp_triple_gated_disagreement_hard_max_norm'
    )
    gp_triple_gated_correction_clip_norm = LaunchConfiguration(
        'gp_triple_gated_correction_clip_norm'
    )
    gp_triple_gated_use_distance_gate = LaunchConfiguration(
        'gp_triple_gated_use_distance_gate'
    )
    gp_historical_shadow_enabled = LaunchConfiguration(
        'gp_historical_shadow_enabled'
    )
    gp_historical_source_mode = LaunchConfiguration(
        'gp_historical_source_mode'
    )
    gp_historical_soft_shadow_enabled = LaunchConfiguration(
        'gp_historical_soft_shadow_enabled'
    )
    gp_historical_soft_alpha = LaunchConfiguration(
        'gp_historical_soft_alpha'
    )
    gp_historical_soft_distance_threshold = LaunchConfiguration(
        'gp_historical_soft_distance_threshold'
    )
    gp_historical_soft_online_scale = LaunchConfiguration(
        'gp_historical_soft_online_scale'
    )
    gp_historical_soft_non_online_scale = LaunchConfiguration(
        'gp_historical_soft_non_online_scale'
    )

    controller_node = Node(
        package='py_controllers',
        executable='cartesian_impedance',
        name='cartesian_impedance',
        output='screen',
        on_exit=Shutdown(
            reason=(
                'cartesian_impedance exited; stopping Python-only GP '
                'compensation launch'
            )
        ),
        parameters=[{
            'effort_output_mode': 'active',
            'reference_mode': 'cartesian',
            'gp_prediction_enabled': ParameterValue(
                gp_prediction_enabled, value_type=bool
            ),
            'gp_online_update_enabled': ParameterValue(
                gp_online_update_enabled, value_type=bool
            ),
            'gp_compensation_enabled': ParameterValue(
                gp_compensation_enabled, value_type=bool
            ),
            'gp_compensation_source': ParameterValue(
                gp_compensation_source, value_type=str
            ),
            'gp_compensation_scale': ParameterValue(
                gp_compensation_scale, value_type=float
            ),
            'gp_compensation_clip_nm': ParameterValue(
                gp_compensation_clip_nm, value_type=float
            ),
            'gp_compensation_disable_joint7': ParameterValue(
                gp_compensation_disable_joint7, value_type=bool
            ),
            'gp_model_dir': ParameterValue(gp_model_dir, value_type=str),
            'delay_steps': ParameterValue(delay_steps, value_type=int),
            'gp_prediction_stride': ParameterValue(
                gp_prediction_stride, value_type=int
            ),
            'future_trajectory_request_stride': ParameterValue(
                future_trajectory_request_stride, value_type=int
            ),
            'gp_output_timeout_sec': ParameterValue(
                gp_output_timeout_sec, value_type=float
            ),
            'gp_historical_shadow_enabled': ParameterValue(
                gp_historical_shadow_enabled, value_type=bool
            ),
            'gp_historical_source_mode': ParameterValue(
                gp_historical_source_mode, value_type=str
            ),
            'gp_historical_db_enabled': ParameterValue(
                gp_historical_db_enabled, value_type=bool
            ),
            'gp_historical_db_path': ParameterValue(
                gp_historical_db_path, value_type=str
            ),
            'gp_historical_db_k': ParameterValue(
                gp_historical_db_k, value_type=int
            ),
            'gp_historical_db_q_scale': ParameterValue(
                gp_historical_db_q_scale, value_type=float
            ),
            'gp_historical_db_dq_scale': ParameterValue(
                gp_historical_db_dq_scale, value_type=float
            ),
            'gp_historical_db_max_distance': ParameterValue(
                gp_historical_db_max_distance, value_type=float
            ),
            'gp_historical_db_require_distance_pass_for_active': ParameterValue(
                gp_historical_db_require_distance_pass_for_active,
                value_type=bool,
            ),
            'gp_historical_db_distance_contribution_logging': ParameterValue(
                gp_historical_db_distance_contribution_logging,
                value_type=bool,
            ),
            'gp_historical_db_metadata_path': ParameterValue(
                gp_historical_db_metadata_path, value_type=str
            ),
            'gp_historical_db_metadata_enforcement_enabled': ParameterValue(
                gp_historical_db_metadata_enforcement_enabled,
                value_type=bool,
            ),
            'gp_historical_db_query_stride': ParameterValue(
                gp_historical_db_query_stride, value_type=int
            ),
            'gp_historical_db_disable_when_online_update': ParameterValue(
                gp_historical_db_disable_when_online_update, value_type=bool
            ),
            'gp_historical_db_fallback_source': ParameterValue(
                gp_historical_db_fallback_source, value_type=str
            ),
            'gp_historical_db_preflight_enabled': ParameterValue(
                gp_historical_db_preflight_enabled, value_type=bool
            ),
            'gp_historical_db_preflight_required': ParameterValue(
                gp_historical_db_preflight_required, value_type=bool
            ),
            'gp_historical_db_preflight_mode': ParameterValue(
                gp_historical_db_preflight_mode, value_type=str
            ),
            'gp_historical_db_preflight_duration_sec': ParameterValue(
                gp_historical_db_preflight_duration_sec, value_type=float
            ),
            'gp_historical_db_preflight_min_samples': ParameterValue(
                gp_historical_db_preflight_min_samples, value_type=int
            ),
            'gp_historical_db_preflight_min_pass_ratio': ParameterValue(
                gp_historical_db_preflight_min_pass_ratio, value_type=float
            ),
            'gp_historical_db_preflight_p95_max_distance': ParameterValue(
                gp_historical_db_preflight_p95_max_distance, value_type=float
            ),
            'gp_historical_db_preflight_max_distance': ParameterValue(
                gp_historical_db_preflight_max_distance, value_type=float
            ),
            'gp_disable_silent_hist_fallback': ParameterValue(
                gp_disable_silent_hist_fallback, value_type=bool
            ),
            'gp_triple_weight_mode': ParameterValue(
                gp_triple_weight_mode, value_type=str
            ),
            'gp_triple_weight_local': ParameterValue(
                gp_triple_weight_local, value_type=float
            ),
            'gp_triple_weight_cloud': ParameterValue(
                gp_triple_weight_cloud, value_type=float
            ),
            'gp_triple_weight_hist': ParameterValue(
                gp_triple_weight_hist, value_type=float
            ),
            'gp_triple_weight_normalize': ParameterValue(
                gp_triple_weight_normalize, value_type=bool
            ),
            'gp_triple_rmse_local': ParameterValue(
                gp_triple_rmse_local, value_type=float
            ),
            'gp_triple_rmse_cloud': ParameterValue(
                gp_triple_rmse_cloud, value_type=float
            ),
            'gp_triple_rmse_hist': ParameterValue(
                gp_triple_rmse_hist, value_type=float
            ),
            'gp_triple_hist_distance_scale': ParameterValue(
                gp_triple_hist_distance_scale, value_type=float
            ),
            'gp_triple_hist_distance_power': ParameterValue(
                gp_triple_hist_distance_power, value_type=float
            ),
            'gp_triple_hist_weight_cap': ParameterValue(
                gp_triple_hist_weight_cap, value_type=float
            ),
            'gp_triple_hist_min_weight': ParameterValue(
                gp_triple_hist_min_weight, value_type=float
            ),
            'gp_triple_min_weight_local': ParameterValue(
                gp_triple_min_weight_local, value_type=float
            ),
            'gp_triple_min_weight_cloud': ParameterValue(
                gp_triple_min_weight_cloud, value_type=float
            ),
            'gp_triple_require_hist_available': ParameterValue(
                gp_triple_require_hist_available, value_type=bool
            ),
            'gp_triple_fallback_source': ParameterValue(
                gp_triple_fallback_source, value_type=str
            ),
            'gp_triple_debug_safety_log_enabled': ParameterValue(
                gp_triple_debug_safety_log_enabled, value_type=bool
            ),
            'gp_triple_debug_safety_log_first_n': ParameterValue(
                gp_triple_debug_safety_log_first_n, value_type=int
            ),
            'gp_triple_combined_base_shadow_enabled': ParameterValue(
                gp_triple_combined_base_shadow_enabled, value_type=bool
            ),
            'gp_triple_combined_base_hist_weight_cap': ParameterValue(
                gp_triple_combined_base_hist_weight_cap, value_type=float
            ),
            'gp_triple_combined_base_hist_weight_ramp_sec': ParameterValue(
                gp_triple_combined_base_hist_weight_ramp_sec, value_type=float
            ),
            'gp_triple_gated_hist_cap_f50': ParameterValue(
                gp_triple_gated_hist_cap_f50, value_type=float
            ),
            'gp_triple_gated_hist_cap_f100': ParameterValue(
                gp_triple_gated_hist_cap_f100, value_type=float
            ),
            'gp_triple_gated_hist_cap_f200': ParameterValue(
                gp_triple_gated_hist_cap_f200, value_type=float
            ),
            'gp_triple_gated_disagreement_ref_norm': ParameterValue(
                gp_triple_gated_disagreement_ref_norm, value_type=float
            ),
            'gp_triple_gated_disagreement_hard_max_norm': ParameterValue(
                gp_triple_gated_disagreement_hard_max_norm, value_type=float
            ),
            'gp_triple_gated_correction_clip_norm': ParameterValue(
                gp_triple_gated_correction_clip_norm, value_type=float
            ),
            'gp_triple_gated_use_distance_gate': ParameterValue(
                gp_triple_gated_use_distance_gate, value_type=bool
            ),
            'gp_historical_soft_shadow_enabled': ParameterValue(
                gp_historical_soft_shadow_enabled, value_type=bool
            ),
            'gp_historical_soft_alpha': ParameterValue(
                gp_historical_soft_alpha, value_type=float
            ),
            'gp_historical_soft_distance_threshold': ParameterValue(
                gp_historical_soft_distance_threshold, value_type=float
            ),
            'gp_historical_soft_online_scale': ParameterValue(
                gp_historical_soft_online_scale, value_type=float
            ),
            'gp_historical_soft_non_online_scale': ParameterValue(
                gp_historical_soft_non_online_scale, value_type=float
            ),
            'timing_logging_enabled': False,
            'control_frequency': ParameterValue(
                control_frequency, value_type=float
            ),
            'trajectory_publish_rate': ParameterValue(
                trajectory_publish_rate, value_type=float
            ),
            'state_parameter_publish_rate': ParameterValue(
                state_parameter_publish_rate, value_type=float
            ),
            'trajectory_mode': ParameterValue(
                trajectory_mode, value_type=str
            ),
            'circle_frequency': ParameterValue(
                circle_frequency, value_type=float
            ),
            'transition_duration': ParameterValue(
                transition_duration, value_type=float
            ),
            'run_name': ParameterValue(run_name, value_type=str),
            'data_output_dir': ParameterValue(
                data_output_dir, value_type=str
            ),
            'csv_output_profile': ParameterValue(
                csv_output_profile, value_type=str
            ),
            'startup_linear_speed': ParameterValue(
                startup_linear_speed, value_type=float
            ),
            'startup_distance_warn_m': ParameterValue(
                startup_distance_warn_m, value_type=float
            ),
            'startup_distance_refuse_m': ParameterValue(
                startup_distance_refuse_m, value_type=float
            ),
            'startup_distance_refuse_enabled': ParameterValue(
                startup_distance_refuse_enabled, value_type=bool
            ),
            'session_home_mode': ParameterValue(
                session_home_mode, value_type=str
            ),
            'session_home_path': ParameterValue(
                session_home_path, value_type=str
            ),
            'session_home_capture_enabled': ParameterValue(
                session_home_capture_enabled, value_type=bool
            ),
            'session_home_capture_max_distance_from_nominal_m': ParameterValue(
                session_home_capture_max_distance_from_nominal_m,
                value_type=float,
            ),
            'session_home_capture_requires_stable_state': ParameterValue(
                session_home_capture_requires_stable_state, value_type=bool
            ),
            'session_home_capture_stability_samples': ParameterValue(
                session_home_capture_stability_samples, value_type=int
            ),
            'session_home_capture_stability_position_std_m': ParameterValue(
                session_home_capture_stability_position_std_m,
                value_type=float,
            ),
            'session_home_capture_min_z': ParameterValue(
                session_home_capture_min_z, value_type=float
            ),
            'session_home_capture_max_z': ParameterValue(
                session_home_capture_max_z, value_type=float
            ),
            'session_home_joint_check_enabled': ParameterValue(
                session_home_joint_check_enabled, value_type=bool
            ),
            'session_home_joint_check_required_for_hist': ParameterValue(
                session_home_joint_check_required_for_hist, value_type=bool
            ),
            'session_home_joint_max_abs_warn_rad': ParameterValue(
                session_home_joint_max_abs_warn_rad, value_type=float
            ),
            'session_home_joint_max_abs_refuse_rad': ParameterValue(
                session_home_joint_max_abs_refuse_rad, value_type=float
            ),
            'session_home_joint_l2_warn_rad': ParameterValue(
                session_home_joint_l2_warn_rad, value_type=float
            ),
            'session_home_joint_l2_refuse_rad': ParameterValue(
                session_home_joint_l2_refuse_rad, value_type=float
            ),
            'session_home_dq_stillness_warn_rad_s': ParameterValue(
                session_home_dq_stillness_warn_rad_s, value_type=float
            ),
            'session_home_dq_stillness_refuse_rad_s': ParameterValue(
                session_home_dq_stillness_refuse_rad_s, value_type=float
            ),
            'trajectory_reference_mode': ParameterValue(
                trajectory_reference_mode, value_type=str
            ),
            'session_relative_capture_enabled': ParameterValue(
                session_relative_capture_enabled, value_type=bool
            ),
            'session_relative_max_anchor_delta_m': ParameterValue(
                session_relative_max_anchor_delta_m, value_type=float
            ),
            'session_relative_warn_anchor_delta_m': ParameterValue(
                session_relative_warn_anchor_delta_m, value_type=float
            ),
            'session_relative_anchor_delta_limit_mode': ParameterValue(
                session_relative_anchor_delta_limit_mode, value_type=str
            ),
            'session_relative_min_z': ParameterValue(
                session_relative_min_z, value_type=float
            ),
            'session_relative_max_z': ParameterValue(
                session_relative_max_z, value_type=float
            ),
            'session_relative_requires_stable_state': ParameterValue(
                session_relative_requires_stable_state, value_type=bool
            ),
            'session_relative_stability_samples': ParameterValue(
                session_relative_stability_samples, value_type=int
            ),
            'session_relative_stability_position_std_m': ParameterValue(
                session_relative_stability_position_std_m, value_type=float
            ),
            'session_relative_apply_to_startup_and_return': ParameterValue(
                session_relative_apply_to_startup_and_return, value_type=bool
            ),
            'session_relative_nominal_trajectory_start_xyz': ParameterValue(
                session_relative_nominal_trajectory_start_xyz, value_type=str
            ),
            'session_relative_nominal_circle_center_xyz': ParameterValue(
                session_relative_nominal_circle_center_xyz, value_type=str
            ),
            'normal_run_start_gate_enabled': ParameterValue(
                normal_run_start_gate_enabled, value_type=bool
            ),
            'normal_run_start_warn_m': ParameterValue(
                normal_run_start_warn_m, value_type=float
            ),
            'normal_run_start_refuse_m': ParameterValue(
                normal_run_start_refuse_m, value_type=float
            ),
            'emergency_return_start_refuse_m': ParameterValue(
                emergency_return_start_refuse_m, value_type=float
            ),
            'return_only_if_too_far_enabled': ParameterValue(
                return_only_if_too_far_enabled, value_type=bool
            ),
            'post_run_return_to_session_home_enabled': ParameterValue(
                post_run_return_to_session_home_enabled, value_type=bool
            ),
            'post_run_return_linear_speed': ParameterValue(
                post_run_return_linear_speed, value_type=float
            ),
            'post_run_return_timeout_sec': ParameterValue(
                post_run_return_timeout_sec, value_type=float
            ),
            'post_run_return_hold_sec': ParameterValue(
                post_run_return_hold_sec, value_type=float
            ),
            'post_run_return_tolerance_m': ParameterValue(
                post_run_return_tolerance_m, value_type=float
            ),
            'post_run_return_disable_gp_compensation': ParameterValue(
                post_run_return_disable_gp_compensation, value_type=bool
            ),
            'post_run_return_disable_online_update': ParameterValue(
                post_run_return_disable_online_update, value_type=bool
            ),
            'startup_torque_clip_nm': 10.0,
            'startup_torque_rate_limit_from_zero': True,
            'torque_rate_limit_enabled': True,
            'torque_rate_limit_nm_per_s': ParameterValue(
                torque_rate_limit_nm_per_s, value_type=float
            ),
        }],
    )

    trajectory_node = Node(
        package='py_controllers',
        executable='trajectory_publisher',
        name='trajectory_publisher',
        output='screen',
        on_exit=Shutdown(
            reason=(
                'trajectory_publisher exited; stopping Python-only GP '
                'compensation launch'
            )
        ),
        parameters=[{
            'control_frequency': ParameterValue(
                control_frequency, value_type=float
            ),
            'trajectory_publish_rate': ParameterValue(
                trajectory_publish_rate, value_type=float
            ),
            'trajectory_mode': ParameterValue(
                trajectory_mode, value_type=str
            ),
            'circle_frequency': ParameterValue(
                circle_frequency, value_type=float
            ),
            # One launch runs this many full trajectory rounds before the
            # publisher stops recording and triggers the single post-run
            # return; the runner wires --repeat to this value.
            'rounds_per_mode': ParameterValue(
                rounds_per_mode, value_type=int
            ),
            'transition_duration': ParameterValue(
                transition_duration, value_type=float
            ),
            # Preserve the formal fixed-start comparison. Enabling this for
            # goal1_spatial_multisine would shift the center to the measured
            # pose after /joint_position_adjust is called.
            'anchor_trajectory_start_to_current_pose': False,
            'trajectory_reference_mode': ParameterValue(
                trajectory_reference_mode, value_type=str
            ),
            # The controller calls this session_home_path; trajectory_publisher
            # calls the same JSON file session_anchor_path.
            'session_anchor_path': ParameterValue(
                session_home_path, value_type=str
            ),
            'session_relative_apply_to_trajectory_center': ParameterValue(
                session_relative_apply_to_trajectory_center, value_type=bool
            ),
            'session_relative_max_anchor_delta_m': ParameterValue(
                session_relative_max_anchor_delta_m, value_type=float
            ),
            'session_relative_anchor_delta_limit_mode': ParameterValue(
                session_relative_anchor_delta_limit_mode, value_type=str
            ),
            'session_relative_nominal_trajectory_start_xyz': ParameterValue(
                session_relative_nominal_trajectory_start_xyz, value_type=str
            ),
            'session_relative_nominal_circle_center_xyz': ParameterValue(
                session_relative_nominal_circle_center_xyz, value_type=str
            ),
            # Keep this node alive after the final round while the controller
            # returns to the session home; same flag as the controller return.
            'post_run_return_wait_enabled': ParameterValue(
                post_run_return_to_session_home_enabled, value_type=bool
            ),
            'post_run_return_wait_timeout_sec': ParameterValue(
                post_run_return_wait_timeout_sec, value_type=float
            ),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'run_name',
            default_value='MANUAL_F50_GP_COMPENSATION_STARTUP_TRAJ_TEST',
            description='Controller CSV run name.',
        ),
        DeclareLaunchArgument(
            'data_output_dir',
            default_value='.',
            description='Directory for controller CSV output.',
        ),
        DeclareLaunchArgument(
            'csv_output_profile',
            default_value='full',
            description='Controller CSV output profile: full or final.',
        ),
        DeclareLaunchArgument(
            'trajectory_mode',
            default_value='goal1_spatial_multisine',
            description='Trajectory mode passed to both Python nodes.',
        ),
        DeclareLaunchArgument(
            'circle_frequency',
            default_value='0.05',
            description='Base trajectory frequency in Hz.',
        ),
        DeclareLaunchArgument(
            'rounds_per_mode',
            default_value='6',
            description=(
                'Full trajectory rounds executed inside one launch before the '
                'trajectory publisher stops and post-run return starts. '
                'Default 6 matches the previous trajectory_publisher default.'
            ),
        ),
        DeclareLaunchArgument(
            'control_frequency',
            default_value='50',
            description='Python controller frequency metadata in Hz.',
        ),
        DeclareLaunchArgument(
            'trajectory_publish_rate',
            default_value='50',
            description='trajectory_publisher timer rate in Hz.',
        ),
        DeclareLaunchArgument(
            'state_parameter_publish_rate',
            default_value='50',
            description='Expected external cpp_relayer state publish rate in Hz.',
        ),
        DeclareLaunchArgument(
            'transition_duration',
            default_value='10.0',
            description='Fixed-start to trajectory-start transition duration in seconds.',
        ),
        DeclareLaunchArgument(
            'torque_rate_limit_nm_per_s',
            default_value='20.0',
            description='Per-joint torque slew-rate limit in Nm/s.',
        ),
        DeclareLaunchArgument(
            'startup_linear_speed',
            default_value='0.005',
            description='Startup move-to-fixed-start linear speed in m/s.',
        ),
        DeclareLaunchArgument(
            'startup_distance_warn_m',
            default_value='0.100',
            description='Warn threshold for distance to fixed start in m.',
        ),
        DeclareLaunchArgument(
            'startup_distance_refuse_m',
            default_value='0.300',
            description='Hard-refuse threshold for distance to fixed start in m.',
        ),
        DeclareLaunchArgument(
            'startup_distance_refuse_enabled',
            default_value='true',
            description='Refuse startup when distance exceeds startup_distance_refuse_m.',
        ),
        DeclareLaunchArgument(
            'session_home_mode',
            default_value='fixed',
            description='Session home mode: fixed, capture_first, or load.',
        ),
        DeclareLaunchArgument(
            'session_home_path',
            default_value='',
            description='JSON path for captured/loaded session home.',
        ),
        DeclareLaunchArgument(
            'session_home_capture_enabled',
            default_value='false',
            description='Allow capture_first session home capture.',
        ),
        DeclareLaunchArgument(
            'session_home_capture_max_distance_from_nominal_m',
            default_value='0.250',
            description='Max session home distance from nominal fixed start in m.',
        ),
        DeclareLaunchArgument(
            'session_home_capture_requires_stable_state',
            default_value='true',
            description='Require position stability before session home capture.',
        ),
        DeclareLaunchArgument(
            'session_home_capture_stability_samples',
            default_value='10',
            description='State samples used for session home capture stability.',
        ),
        DeclareLaunchArgument(
            'session_home_capture_stability_position_std_m',
            default_value='0.003',
            description='Max per-axis position std for session home capture in m.',
        ),
        DeclareLaunchArgument(
            'session_home_capture_min_z',
            default_value='0.45',
            description='Minimum z for a valid session home in m.',
        ),
        DeclareLaunchArgument(
            'session_home_capture_max_z',
            default_value='0.85',
            description='Maximum z for a valid session home in m.',
        ),
        DeclareLaunchArgument(
            'session_home_joint_check_enabled',
            default_value='false',
            description='Enable read-only q/dq gate against q_at_capture.',
        ),
        DeclareLaunchArgument(
            'session_home_joint_check_required_for_hist',
            default_value='true',
            description='Require q_at_capture joint gate for active hist sources.',
        ),
        DeclareLaunchArgument(
            'session_home_joint_max_abs_warn_rad',
            default_value='0.10',
        ),
        DeclareLaunchArgument(
            'session_home_joint_max_abs_refuse_rad',
            default_value='0.30',
        ),
        DeclareLaunchArgument(
            'session_home_joint_l2_warn_rad',
            default_value='0.20',
        ),
        DeclareLaunchArgument(
            'session_home_joint_l2_refuse_rad',
            default_value='0.50',
        ),
        DeclareLaunchArgument(
            'session_home_dq_stillness_warn_rad_s',
            default_value='0.02',
        ),
        DeclareLaunchArgument(
            'session_home_dq_stillness_refuse_rad_s',
            default_value='0.05',
        ),
        DeclareLaunchArgument(
            'trajectory_reference_mode',
            default_value='fixed_absolute',
            description='Trajectory reference mode: fixed_absolute or session_relative.',
        ),
        DeclareLaunchArgument(
            'session_relative_capture_enabled',
            default_value='false',
            description='Allow capture_first to save a session-relative trajectory anchor.',
        ),
        DeclareLaunchArgument(
            'session_relative_max_anchor_delta_m',
            default_value='0.250',
            description='Maximum allowed session-relative anchor_delta norm in m.',
        ),
        DeclareLaunchArgument(
            'session_relative_warn_anchor_delta_m',
            default_value='0.150',
            description='Warn threshold for session-relative anchor_delta norm in m.',
        ),
        DeclareLaunchArgument(
            'session_relative_anchor_delta_limit_mode',
            default_value='refuse',
            description=(
                'Policy when anchor_delta norm exceeds '
                'session_relative_max_anchor_delta_m: refuse keeps legacy '
                'behavior; warn allows floating-anchor session_relative runs '
                'from the current pose while warning when the shift exceeds '
                'the nominal-start limit; off disables this norm check.'
            ),
        ),
        DeclareLaunchArgument(
            'session_relative_min_z',
            default_value='0.45',
            description='Minimum z for a valid session-relative trajectory start in m.',
        ),
        DeclareLaunchArgument(
            'session_relative_max_z',
            default_value='0.85',
            description='Maximum z for a valid session-relative trajectory start in m.',
        ),
        DeclareLaunchArgument(
            'session_relative_requires_stable_state',
            default_value='true',
            description='Require EE position stability before session-relative capture.',
        ),
        DeclareLaunchArgument(
            'session_relative_stability_samples',
            default_value='10',
            description='State samples used for session-relative capture stability.',
        ),
        DeclareLaunchArgument(
            'session_relative_stability_position_std_m',
            default_value='0.003',
            description='Max per-axis position std for session-relative capture in m.',
        ),
        DeclareLaunchArgument(
            'session_relative_apply_to_startup_and_return',
            default_value='true',
            description='Use session_trajectory_start for startup and post-run return.',
        ),
        DeclareLaunchArgument(
            'session_relative_apply_to_trajectory_center',
            default_value='true',
            description='Shift trajectory center by anchor_delta in session_relative mode.',
        ),
        DeclareLaunchArgument(
            'session_relative_nominal_trajectory_start_xyz',
            default_value='[0.3077306122468523, 0.043799833015107294, 0.6648721535244662]',
            description='Nominal unshifted trajectory start xyz used to compute anchor_delta.',
        ),
        DeclareLaunchArgument(
            'session_relative_nominal_circle_center_xyz',
            default_value='[0.3, 0.0, 0.65]',
            description='Nominal unshifted circle center xyz used to compute shifted center.',
        ),
        DeclareLaunchArgument(
            'normal_run_start_gate_enabled',
            default_value='false',
            description='Enable stricter three-tier run-start gate against session home.',
        ),
        DeclareLaunchArgument(
            'normal_run_start_warn_m',
            default_value='0.100',
            description='Warn threshold for run-start distance to session home in m.',
        ),
        DeclareLaunchArgument(
            'normal_run_start_refuse_m',
            default_value='0.150',
            description='Refuse official GP run above this distance to session home in m.',
        ),
        DeclareLaunchArgument(
            'emergency_return_start_refuse_m',
            default_value='0.300',
            description='Refuse all automatic motion above this distance to session home in m.',
        ),
        DeclareLaunchArgument(
            'return_only_if_too_far_enabled',
            default_value='false',
            description='Allow no-GP return-only cleanup between normal and emergency thresholds.',
        ),
        DeclareLaunchArgument(
            'post_run_return_to_session_home_enabled',
            default_value='false',
            description='Slowly return to session home after the final round before exit.',
        ),
        DeclareLaunchArgument(
            'post_run_return_linear_speed',
            default_value='0.005',
            description='Post-run return linear speed in m/s.',
        ),
        DeclareLaunchArgument(
            'post_run_return_timeout_sec',
            default_value='60.0',
            description='Post-run return timeout in seconds.',
        ),
        DeclareLaunchArgument(
            'post_run_return_hold_sec',
            default_value='2.0',
            description='Hold time at session home before zero torque in seconds.',
        ),
        DeclareLaunchArgument(
            'post_run_return_tolerance_m',
            default_value='0.015',
            description='Position tolerance for post-run return completion in m.',
        ),
        DeclareLaunchArgument(
            'post_run_return_disable_gp_compensation',
            default_value='true',
            description='Disable GP compensation during post-run return.',
        ),
        DeclareLaunchArgument(
            'post_run_return_disable_online_update',
            default_value='true',
            description='Disable GP online update during post-run return.',
        ),
        DeclareLaunchArgument(
            'post_run_return_wait_timeout_sec',
            default_value='90.0',
            description='trajectory_publisher wait timeout for post-run return in seconds.',
        ),
        DeclareLaunchArgument(
            'gp_model_dir',
            default_value='./new_structure/gp/gp_models',
            description='Directory containing offline GP model pickle files.',
        ),
        DeclareLaunchArgument(
            'gp_prediction_enabled',
            default_value='true',
            description=(
                'Master GP prediction switch. false gives true GP-off (no '
                'prediction, no online update, no compensation) for clean '
                'historical-DB raw runs. Default true keeps the previous '
                'hardcoded launch behavior.'
            ),
        ),
        DeclareLaunchArgument(
            'gp_online_update_enabled',
            default_value='true',
            description=(
                'Enable GP online update. Default true keeps the previous '
                'hardcoded launch behavior.'
            ),
        ),
        DeclareLaunchArgument(
            'gp_compensation_enabled',
            default_value='true',
            description=(
                'Enable active GP compensation torque. Default true keeps '
                'the previous hardcoded launch behavior.'
            ),
        ),
        DeclareLaunchArgument(
            'gp_compensation_source',
            default_value='local',
            description='Active GP compensation source.',
        ),
        DeclareLaunchArgument(
            'gp_compensation_scale',
            default_value='0.25',
            description='Scale applied to selected GP compensation torque.',
        ),
        DeclareLaunchArgument(
            'gp_compensation_clip_nm',
            default_value='0.5',
            description='Per-joint absolute clip for GP compensation torque in Nm.',
        ),
        DeclareLaunchArgument(
            'gp_compensation_disable_joint7',
            default_value='true',
            description='Disable active GP compensation on joint 7.',
        ),
        DeclareLaunchArgument(
            'delay_steps',
            default_value='0',
            description='Cloud-like GP delay in controller callbacks.',
        ),
        DeclareLaunchArgument(
            'gp_prediction_stride',
            default_value='5',
            description='Run GP predict/update once every N controller callbacks.',
        ),
        DeclareLaunchArgument(
            'future_trajectory_request_stride',
            default_value='5',
            description='Request /future_task_space once every N controller callbacks.',
        ),
        DeclareLaunchArgument(
            'gp_output_timeout_sec',
            default_value='0.5',
            description='Maximum local/cloud GP output age before active compensation fails closed.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_enabled',
            default_value='false',
            description='Load and query persistent historical residual DB.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_path',
            default_value='',
            description='Path to persistent historical residual DB .npz.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_k',
            default_value='25',
            description='KNN neighbor count for historical residual DB.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_q_scale',
            default_value='0.1',
            description='Joint position scale for historical DB KNN.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_dq_scale',
            default_value='0.1',
            description='Joint velocity scale for historical DB KNN.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_max_distance',
            default_value='1.0',
            description='Maximum scaled nearest-neighbor distance for hist DB availability.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_require_distance_pass_for_active',
            default_value='false',
            description='Require distance_pass before active hist correction.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_distance_contribution_logging',
            default_value='false',
            description='Log per-dimension scaled nearest-distance contributions.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_metadata_path',
            default_value='',
            description='Optional hist DB metadata sidecar JSON path.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_metadata_enforcement_enabled',
            default_value='false',
            description='Require DB/session metadata identity match.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_query_stride',
            default_value='1',
            description='Query historical DB once every N controller callbacks.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_disable_when_online_update',
            default_value='true',
            description='Disable hist DB availability while online GP update is enabled.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_fallback_source',
            default_value='cloud',
            description='hist_db fallback source: none, local, cloud, or combined.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_preflight_enabled',
            default_value='false',
            description='Enable historical DB distance preflight before active hist compensation.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_preflight_required',
            default_value='false',
            description='Require hist DB preflight pass before active hist compensation.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_preflight_mode',
            default_value='segment',
            description='Hist DB preflight mode: single, segment, or single_and_segment.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_preflight_duration_sec',
            default_value='5.0',
            description='Hist DB segment preflight duration.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_preflight_min_samples',
            default_value='50',
            description='Minimum hist DB preflight samples.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_preflight_min_pass_ratio',
            default_value='0.95',
            description='Minimum hist DB preflight distance-pass ratio.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_preflight_p95_max_distance',
            default_value='1.5',
            description='Maximum p95 nearest distance for hist DB preflight.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_preflight_max_distance',
            default_value='2.0',
            description='Maximum nearest distance for hist DB preflight.',
        ),
        DeclareLaunchArgument(
            'gp_disable_silent_hist_fallback',
            default_value='false',
            description='Fail closed instead of silently using hist DB fallback.',
        ),
        DeclareLaunchArgument(
            'gp_triple_weight_mode',
            default_value='inverse_rmse',
            description='Triple fusion weight mode: fixed or inverse_rmse.',
        ),
        DeclareLaunchArgument(
            'gp_triple_weight_local',
            default_value='0.10',
            description='Fixed local weight for triple fusion.',
        ),
        DeclareLaunchArgument(
            'gp_triple_weight_cloud',
            default_value='0.20',
            description='Fixed cloud weight for triple fusion.',
        ),
        DeclareLaunchArgument(
            'gp_triple_weight_hist',
            default_value='0.70',
            description='Fixed historical DB weight for triple fusion.',
        ),
        DeclareLaunchArgument(
            'gp_triple_weight_normalize',
            default_value='true',
            description='Normalize fixed triple fusion weights.',
        ),
        DeclareLaunchArgument(
            'gp_triple_rmse_local',
            default_value='0.330269',
            description='Local RMSE prior for inverse-RMSE triple weights.',
        ),
        DeclareLaunchArgument(
            'gp_triple_rmse_cloud',
            default_value='0.330278',
            description='Cloud RMSE prior for inverse-RMSE triple weights.',
        ),
        DeclareLaunchArgument(
            'gp_triple_rmse_hist',
            default_value='0.093071',
            description='Historical DB RMSE prior for inverse-RMSE triple weights.',
        ),
        DeclareLaunchArgument(
            'gp_triple_hist_distance_scale',
            default_value='2.0',
            description='Distance scale for dynamic triple hist penalty.',
        ),
        DeclareLaunchArgument(
            'gp_triple_hist_distance_power',
            default_value='2.0',
            description='Distance power for dynamic triple hist penalty.',
        ),
        DeclareLaunchArgument(
            'gp_triple_hist_weight_cap',
            default_value='0.70',
            description='Maximum hist weight in triple fusion.',
        ),
        DeclareLaunchArgument(
            'gp_triple_hist_min_weight',
            default_value='0.0',
            description='Minimum hist weight in triple fusion.',
        ),
        DeclareLaunchArgument(
            'gp_triple_min_weight_local',
            default_value='0.05',
            description='Minimum local weight in triple dynamic fusion.',
        ),
        DeclareLaunchArgument(
            'gp_triple_min_weight_cloud',
            default_value='0.05',
            description='Minimum cloud weight in triple dynamic fusion.',
        ),
        DeclareLaunchArgument(
            'gp_triple_require_hist_available',
            default_value='true',
            description='Require hist DB availability before normal triple fusion.',
        ),
        DeclareLaunchArgument(
            'gp_triple_fallback_source',
            default_value='combined',
            description='Triple fallback source: none, local, cloud, combined, or hist_db.',
        ),
        DeclareLaunchArgument(
            'gp_triple_debug_safety_log_enabled',
            default_value='true',
            description='Log first triple active-safety diagnostics.',
        ),
        DeclareLaunchArgument(
            'gp_triple_debug_safety_log_first_n',
            default_value='5',
            description='Number of triple active-safety diagnostics to log.',
        ),
        DeclareLaunchArgument(
            'gp_triple_combined_base_shadow_enabled',
            default_value='false',
            description='Enable combined-base plus hist shadow logging only.',
        ),
        DeclareLaunchArgument(
            'gp_triple_combined_base_hist_weight_cap',
            default_value='0.50',
            description='Hist weight cap for combined-base shadow.',
        ),
        DeclareLaunchArgument(
            'gp_triple_combined_base_hist_weight_ramp_sec',
            default_value='0.0',
            description='Ramp time for combined-base shadow hist weight.',
        ),
        DeclareLaunchArgument(
            'gp_triple_gated_hist_cap_f50',
            default_value='0.25',
            description='Hist residual weight cap for F50 triple_dynamic_gated.',
        ),
        DeclareLaunchArgument(
            'gp_triple_gated_hist_cap_f100',
            default_value='0.10',
            description='Hist residual weight cap for F100 triple_dynamic_gated.',
        ),
        DeclareLaunchArgument(
            'gp_triple_gated_hist_cap_f200',
            default_value='0.0',
            description='Hist residual weight cap for F200 triple_dynamic_gated.',
        ),
        DeclareLaunchArgument(
            'gp_triple_gated_disagreement_ref_norm',
            default_value='0.80',
            description='Full-weight disagreement norm reference for triple_dynamic_gated.',
        ),
        DeclareLaunchArgument(
            'gp_triple_gated_disagreement_hard_max_norm',
            default_value='1.50',
            description='Hard-zero disagreement norm for triple_dynamic_gated.',
        ),
        DeclareLaunchArgument(
            'gp_triple_gated_correction_clip_norm',
            default_value='0.30',
            description='Norm clip for hist residual correction in triple_dynamic_gated.',
        ),
        DeclareLaunchArgument(
            'gp_triple_gated_use_distance_gate',
            default_value='true',
            description='Enable soft hist distance gate in triple_dynamic_gated.',
        ),
        DeclareLaunchArgument(
            'gp_historical_shadow_enabled',
            default_value='false',
            description='Enable runtime historical prediction-pool shadow logging.',
        ),
        DeclareLaunchArgument(
            'gp_historical_source_mode',
            default_value='none',
            description='Historical shadow source mode: none or local_prediction_pool.',
        ),
        DeclareLaunchArgument(
            'gp_historical_soft_shadow_enabled',
            default_value='false',
            description='Enable historical soft-fusion shadow logging only.',
        ),
        DeclareLaunchArgument(
            'gp_historical_soft_alpha',
            default_value='1.0',
            description='Historical soft-fusion distance decay alpha.',
        ),
        DeclareLaunchArgument(
            'gp_historical_soft_distance_threshold',
            default_value='0.2',
            description='Historical soft-fusion distance threshold.',
        ),
        DeclareLaunchArgument(
            'gp_historical_soft_online_scale',
            default_value='0.02',
            description='Historical soft-fusion hist scale during online update.',
        ),
        DeclareLaunchArgument(
            'gp_historical_soft_non_online_scale',
            default_value='1.0',
            description='Historical soft-fusion hist scale without online update.',
        ),
        LogInfo(
            msg=(
                'PYTHON-ONLY GP COMPENSATION TRAJECTORY: expects external '
                'IMPL bringup and external cpp_relayer to be active already. '
                'Starts only cartesian_impedance and trajectory_publisher. '
                'GP prediction, online update, and torque compensation are '
                'enabled with conservative local defaults; no bringup, '
                'controller manager, or relayer activation is started here.'
            )
        ),
        controller_node,
        trajectory_node,
    ])
