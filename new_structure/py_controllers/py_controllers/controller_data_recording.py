"""Pure CSV schema, projection, and output-path helpers."""

from pathlib import Path


def controller_csv_path(data_output_dir, run_name):
    """Preserve the controller's output directory and filename semantics."""
    output_dir = Path(data_output_dir).expanduser()
    run_name_stem = Path(run_name).name if run_name else ""
    filename_stem = (
        f"{run_name_stem}_cartesian_impedance_controller_data.csv"
        if run_name_stem
        else "cartesian_impedance_controller_data.csv"
    )
    return output_dir, output_dir / filename_stem


def final_csv_extra_header():
    return [
        "ros2_control_update_rate",
        "trajectory_publish_rate",
        "state_parameter_publish_rate",
        "trajectory_mode",
        "circle_frequency",
        "transition_duration",
        "gp_compensation_disable_joint7",
    ]


def final_csv_column_names():
    columns = [
        "Time(s)", "PredTime(s)", "run_name", "control_frequency",
        "ros2_control_update_rate", "trajectory_publish_rate",
        "state_parameter_publish_rate", "trajectory_mode",
        "circle_frequency", "transition_duration", "delay_steps",
    ]
    columns.extend([f"joint_pos_{i+1}" for i in range(7)])
    columns.extend([f"joint_vel_{i+1}" for i in range(7)])
    columns.extend(["x_actual", "y_actual", "z_actual"])
    columns.extend(["x_desired", "y_desired", "z_desired"])
    columns.extend(["dx_actual", "dy_actual", "dz_actual"])
    columns.extend(["dx_desired", "dy_desired", "dz_desired"])
    for prefix in (
        "tau_final", "tau_final_raw", "tau_rate_limited",
    ):
        columns.extend([f"{prefix}_{i+1}" for i in range(7)])
    columns.extend([
        "gp_prediction_enabled", "gp_online_update_enabled",
        "gp_compensation_enabled", "gp_compensation_source_code",
        "gp_compensation_scale", "gp_compensation_clip_nm",
        "gp_compensation_disable_joint7",
    ])
    for prefix in (
        "tau_residual", "y_hat", "y_hat_local", "y_hat_cloud",
        "hist_db_pred", "hist_db_gated_pred",
        "gp_shadow_combined_paper_raw",
    ):
        columns.extend([f"{prefix}_{i+1}" for i in range(7)])
    columns.extend([
        "gp_triple_combined_base_shadow_enabled",
        "gp_triple_combined_base_shadow_available",
        "gp_triple_combined_base_shadow_used_fallback",
        "gp_triple_combined_base_shadow_w_hist",
        "gp_triple_combined_base_shadow_hist_weight_cap",
        "gp_triple_combined_base_shadow_norm",
        "gp_triple_combined_base_shadow_delta_from_combined_norm",
        "gp_triple_combined_base_shadow_delta_from_legacy_triple_norm",
        "gp_triple_gated_active", "gp_triple_gated_available",
        "gp_triple_gated_fallback_to_combined",
        "gp_triple_gated_hist_weight_eff", "gp_triple_gated_hist_cap",
        "gp_triple_gated_distance_gate",
        "gp_triple_gated_disagreement_gate",
        "gp_triple_gated_disagreement_norm",
        "gp_triple_gated_correction_norm",
        "gp_triple_gated_delta_raw_norm",
        "gp_triple_gated_distance_ratio",
    ])
    columns.extend([
        f"gp_triple_combined_base_shadow_raw_{i+1}" for i in range(7)
    ])
    columns.extend([f"gp_applied_{i+1}" for i in range(7)])
    columns.extend([f"gp_clip_active_{i+1}" for i in range(7)])
    columns.extend([
        "torque_rate_limit_enabled", "torque_rate_limit_nm_per_s",
        "torque_rate_limit_active", "torque_rate_limit_max_delta",
        "torque_rate_limit_dt",
    ])
    return columns


def _extend_control_and_gp_header(header, joint_count):
    header.extend([f"tau_{i+1}" for i in range(joint_count)])
    header.extend(["x_actual", "y_actual", "z_actual"])
    header.extend(["x_desired", "y_desired", "z_desired"])
    header.extend(["dx_actual", "dy_actual", "dz_actual"])
    header.extend(["dx_desired", "dy_desired", "dz_desired"])
    for prefix in ("tau_measured", "gravity"):
        header.extend([f"{prefix}_{i+1}" for i in range(joint_count)])
    for prefix in (
        "joint_pos", "joint_vel", "dq_des_joint", "ddq_des_joint",
        "y_hat", "y_hat_local", "y_hat_cloud", "y_hat_mem",
        "tau_residual", "tau_residual_raw", "q_pred", "dq_pred",
        "q_future_actual", "dq_future_actual", "q_pred_err", "dq_pred_err",
    ):
        header.extend([f"{prefix}_{i+1}" for i in range(7)])
    header.extend([
        "gp_prediction_enabled", "gp_online_update_enabled",
        "gp_compensation_enabled", "gp_compensation_source_code",
        "gp_compensation_scale", "gp_compensation_clip_nm",
        "gp_model_local_loaded_count", "gp_model_cloud_loaded_count",
        "gp_model_cloud_fallback_count", "gp_model_empty_or_prior_count",
        "gp_model_cloud_uses_cloud_pkl", "gp_model_cloud_uses_local_fallback",
        "gp_prediction_stride", "gp_prediction_updated_this_tick",
        "gp_prediction_age_sec", "gp_output_fresh",
        "future_trajectory_request_stride",
        "future_trajectory_updated_this_tick",
    ])
    for prefix in (
        "tau_nominal", "tau_final_raw", "tau_final", "tau_rate_limited",
    ):
        header.extend([f"{prefix}_{i+1}" for i in range(7)])
    header.extend([
        "torque_rate_limit_enabled", "torque_rate_limit_nm_per_s",
        "torque_rate_limit_active", "torque_rate_limit_max_delta",
        "torque_rate_limit_dt",
    ])
    for prefix in (
        "gp_selected_raw", "gp_scaled", "gp_applied", "gp_clip_active",
        "gp_triple_raw",
    ):
        header.extend([f"{prefix}_{i+1}" for i in range(7)])


def _extend_triple_and_shadow_header(header):
    header.extend([
        "gp_triple_weight_local", "gp_triple_weight_cloud",
        "gp_triple_weight_hist", "gp_triple_available",
        "gp_triple_used_fallback", "gp_triple_fallback_source_code",
        "gp_triple_weight_mode_code", "gp_triple_hist_weight_cap",
        "gp_triple_rmse_local", "gp_triple_rmse_cloud", "gp_triple_rmse_hist",
        "gp_triple_dynamic_distance_ratio", "gp_triple_dynamic_hist_penalty",
        "gp_triple_dynamic_mode_code",
        "gp_triple_combined_base_shadow_enabled",
        "gp_triple_combined_base_shadow_available",
        "gp_triple_combined_base_shadow_used_fallback",
        "gp_triple_combined_base_shadow_w_hist",
        "gp_triple_combined_base_shadow_hist_weight_cap",
        "gp_triple_combined_base_shadow_ramp_factor",
        "gp_triple_combined_base_shadow_distance_ratio",
        "gp_triple_combined_base_shadow_hist_penalty",
        "gp_triple_combined_base_shadow_norm",
        "gp_triple_combined_base_shadow_delta_from_combined_norm",
        "gp_triple_combined_base_shadow_delta_from_legacy_triple_norm",
        "gp_triple_gated_active", "gp_triple_gated_available",
        "gp_triple_gated_fallback_to_combined",
        "gp_triple_gated_hist_weight_eff", "gp_triple_gated_hist_cap",
        "gp_triple_gated_distance_gate", "gp_triple_gated_disagreement_gate",
        "gp_triple_gated_disagreement_norm",
        "gp_triple_gated_correction_norm", "gp_triple_gated_delta_raw_norm",
        "gp_triple_gated_distance_ratio",
    ])
    header.extend([
        f"gp_triple_combined_base_shadow_raw_{i+1}" for i in range(7)
    ])
    header.extend([
        "gp_shadow_paper_fusion_logging_enabled",
        "gp_historical_shadow_enabled", "gp_historical_source_mode_code",
        "gp_shadow_paper_formula_available",
        "gp_shadow_historical_available", "gp_shadow_variance_eps",
        "gp_shadow_hist_fallback_variance",
    ])
    for prefix in (
        "gp_shadow_local_raw", "gp_shadow_cloud_raw", "gp_shadow_hist_raw",
        "gp_shadow_combined_paper_raw", "gp_shadow_var_local",
        "gp_shadow_var_cloud", "gp_shadow_var_hist", "gp_shadow_weight_local",
        "gp_shadow_weight_cloud", "gp_shadow_weight_hist",
        "gp_shadow_precision_local", "gp_shadow_precision_cloud",
        "gp_shadow_precision_hist", "gp_shadow_paper_scaled",
        "gp_shadow_paper_clip_proxy_applied",
        "gp_shadow_paper_clip_proxy_active",
    ):
        header.extend([f"{prefix}_{i+1}" for i in range(7)])
    header.extend([
        "gp_shadow_hist_pool_size", "gp_shadow_hist_k_used",
        "gp_shadow_hist_nearest_distance",
        "gp_shadow_hist_mean_distance_topk",
    ])


def _extend_historical_header(header):
    header.extend([
        "hist_db_loaded", "hist_db_query_valid", "hist_db_available",
        "hist_db_online_disabled", "hist_db_distance_pass", "hist_db_k_used",
        "hist_db_nearest_distance", "hist_db_mean_topk_distance",
        "hist_db_q_scale", "hist_db_dq_scale", "hist_db_max_distance",
        "hist_db_fallback_source_code", "hist_db_gated_source_code",
    ])
    header.extend([f"hist_db_pred_{i+1}" for i in range(7)])
    header.extend([f"hist_db_gated_pred_{i+1}" for i in range(7)])
    header.extend([
        "hist_db_query_stride", "hist_db_query_updated_this_tick",
        "hist_db_query_reused", "hist_db_query_counter",
        "hist_db_preflight_enabled", "hist_db_preflight_required",
        "hist_db_preflight_mode", "hist_db_preflight_phase",
        "hist_db_preflight_pass", "hist_db_preflight_active_allowed",
        "hist_db_preflight_sample_count", "hist_db_preflight_pass_ratio",
        "hist_db_preflight_nearest_mean", "hist_db_preflight_nearest_p95",
        "hist_db_preflight_nearest_max", "hist_db_runtime_fallback_used",
        "hist_soft_enabled", "hist_soft_valid", "hist_soft_online_mode",
        "hist_soft_alpha", "hist_soft_distance_threshold",
        "hist_soft_online_scale", "hist_soft_non_online_scale",
        "hist_soft_nearest_distance", "hist_soft_raw_w_hist",
        "hist_soft_norm_w_local", "hist_soft_norm_w_cloud",
        "hist_soft_norm_w_hist",
    ])
    header.extend([f"hist_soft_pred_{i+1}" for i in range(7)])
    header.extend([
        f"hist_soft_delta_vs_local_cloud_{i+1}" for i in range(7)
    ])
    header.extend([
        "run_name", "control_frequency", "delay_steps", "data_output_dir",
    ])


def build_full_csv_header(joint_count):
    header = ["Time(s)", "PredTime(s)"]
    _extend_control_and_gp_header(header, joint_count)
    _extend_triple_and_shadow_header(header)
    _extend_historical_header(header)
    return header


def requested_column_indices(header, requested_columns):
    """Return legacy ordered indices and names absent from the source header."""
    header_index = {name: index for index, name in enumerate(header)}
    missing = [name for name in requested_columns if name not in header_index]
    indices = [
        header_index[name] for name in requested_columns if name in header_index
    ]
    return indices, missing


def project_row(row, indices):
    """Project a row without numeric conversion or mutation."""
    return [row[index] for index in indices]
