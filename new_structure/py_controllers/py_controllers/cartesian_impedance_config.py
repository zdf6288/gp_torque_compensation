"""Pure parameter specifications for GOAL12 home-support configuration."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from py_controllers.session_home_feasibility import (
    validate_joint_home_thresholds,
)


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    default: object
    value_type: type


HOME_SUPPORT_PARAMETER_SPECS = (
    ParameterSpec("session_home_joint_check_enabled", False, bool),
    ParameterSpec("session_home_joint_check_required_for_hist", True, bool),
    ParameterSpec("session_home_joint_max_abs_warn_rad", 0.10, float),
    ParameterSpec("session_home_joint_max_abs_refuse_rad", 0.30, float),
    ParameterSpec("session_home_joint_l2_warn_rad", 0.20, float),
    ParameterSpec("session_home_joint_l2_refuse_rad", 0.50, float),
    ParameterSpec("session_home_dq_stillness_warn_rad_s", 0.02, float),
    ParameterSpec("session_home_dq_stillness_refuse_rad_s", 0.05, float),
)

HIST_SUPPORT_PARAMETER_SPECS = (
    ParameterSpec(
        "gp_historical_db_require_distance_pass_for_active", False, bool
    ),
    ParameterSpec(
        "gp_historical_db_distance_contribution_logging", False, bool
    ),
    ParameterSpec("gp_historical_db_metadata_path", "", str),
    ParameterSpec(
        "gp_historical_db_metadata_enforcement_enabled", False, bool
    ),
)


GP_PARAMETER_SPECS = (
    ParameterSpec('gp_prediction_enabled', True, bool),
    ParameterSpec('gp_online_update_enabled', True, bool),
    ParameterSpec('gp_model_dir', './new_structure/gp/gp_models', str),
    ParameterSpec('gp_compensation_enabled', False, bool),
    ParameterSpec('gp_compensation_source', 'local', str),
    ParameterSpec('gp_compensation_scale', 0.1, float),
    ParameterSpec('gp_compensation_clip_nm', 0.5, float),
    ParameterSpec('gp_compensation_disable_joint7', False, bool),
    ParameterSpec('gp_shadow_paper_fusion_logging_enabled', False, bool),
    ParameterSpec('gp_historical_shadow_enabled', False, bool),
    ParameterSpec('gp_historical_source_mode', 'none', str),
    ParameterSpec('gp_shadow_variance_eps', 1e-09, float),
    ParameterSpec('gp_shadow_hist_fallback_variance', 1000000.0, float),
    ParameterSpec('gp_historical_shadow_max_points', 2000, int),
    ParameterSpec('gp_historical_shadow_min_points', 10, int),
    ParameterSpec('gp_historical_shadow_k', 5, int),
    ParameterSpec('gp_historical_shadow_max_distance', 1000000.0, float),
    ParameterSpec('gp_historical_shadow_variance_floor', 1e-08, float),
    ParameterSpec('gp_historical_shadow_distance_eps', 1e-09, float),
    ParameterSpec('gp_historical_db_enabled', False, bool),
    ParameterSpec('gp_historical_db_path', '', str),
    ParameterSpec('gp_historical_db_k', 25, int),
    ParameterSpec('gp_historical_db_q_scale', 0.1, float),
    ParameterSpec('gp_historical_db_dq_scale', 0.1, float),
    ParameterSpec('gp_historical_db_max_distance', 1.0, float),
    ParameterSpec('gp_historical_db_require_distance_pass_for_active', False, bool),
    ParameterSpec('gp_historical_db_distance_contribution_logging', False, bool),
    ParameterSpec('gp_historical_db_metadata_path', '', str),
    ParameterSpec('gp_historical_db_metadata_enforcement_enabled', False, bool),
    ParameterSpec('gp_historical_db_query_stride', 1, int),
    ParameterSpec('gp_historical_db_disable_when_online_update', True, bool),
    ParameterSpec('gp_historical_db_fallback_source', 'cloud', str),
    ParameterSpec('gp_historical_db_preflight_enabled', False, bool),
    ParameterSpec('gp_historical_db_preflight_required', False, bool),
    ParameterSpec('gp_disable_silent_hist_fallback', False, bool),
    ParameterSpec('gp_historical_db_preflight_mode', 'segment', str),
    ParameterSpec('gp_historical_db_preflight_duration_sec', 5.0, float),
    ParameterSpec('gp_historical_db_preflight_min_samples', 50, int),
    ParameterSpec('gp_historical_db_preflight_min_pass_ratio', 0.95, float),
    ParameterSpec('gp_historical_db_preflight_p95_max_distance', 1.5, float),
    ParameterSpec('gp_historical_db_preflight_max_distance', 2.0, float),
    ParameterSpec('gp_historical_db_preflight_log_first_n', 5, int),
    ParameterSpec('gp_triple_weight_mode', 'inverse_rmse', str),
    ParameterSpec('gp_triple_weight_local', 0.1, float),
    ParameterSpec('gp_triple_weight_cloud', 0.2, float),
    ParameterSpec('gp_triple_weight_hist', 0.7, float),
    ParameterSpec('gp_triple_weight_normalize', True, bool),
    ParameterSpec('gp_triple_rmse_local', 0.330269, float),
    ParameterSpec('gp_triple_rmse_cloud', 0.330278, float),
    ParameterSpec('gp_triple_rmse_hist', 0.093071, float),
    ParameterSpec('gp_triple_inverse_rmse_eps', 1e-09, float),
    ParameterSpec('gp_triple_hist_distance_scale', 2.0, float),
    ParameterSpec('gp_triple_hist_distance_power', 2.0, float),
    ParameterSpec('gp_triple_hist_weight_cap', 0.7, float),
    ParameterSpec('gp_triple_hist_min_weight', 0.0, float),
    ParameterSpec('gp_triple_dynamic_eps', 1e-09, float),
    ParameterSpec('gp_triple_min_weight_local', 0.05, float),
    ParameterSpec('gp_triple_min_weight_cloud', 0.05, float),
    ParameterSpec('gp_triple_require_hist_available', True, bool),
    ParameterSpec('gp_triple_fallback_source', 'combined', str),
    ParameterSpec('gp_triple_debug_safety_log_enabled', True, bool),
    ParameterSpec('gp_triple_debug_safety_log_first_n', 5, int),
    ParameterSpec('gp_triple_combined_base_shadow_enabled', False, bool),
    ParameterSpec('gp_triple_combined_base_hist_weight_cap', 0.5, float),
    ParameterSpec('gp_triple_combined_base_hist_weight_ramp_sec', 0.0, float),
    ParameterSpec('gp_triple_gated_hist_cap_f50', 0.25, float),
    ParameterSpec('gp_triple_gated_hist_cap_f100', 0.1, float),
    ParameterSpec('gp_triple_gated_hist_cap_f200', 0.0, float),
    ParameterSpec('gp_triple_gated_disagreement_ref_norm', 0.8, float),
    ParameterSpec('gp_triple_gated_disagreement_hard_max_norm', 1.5, float),
    ParameterSpec('gp_triple_gated_correction_clip_norm', 0.3, float),
    ParameterSpec('gp_triple_gated_use_distance_gate', True, bool),
    ParameterSpec('gp_historical_soft_shadow_enabled', False, bool),
    ParameterSpec('gp_historical_soft_alpha', 1.0, float),
    ParameterSpec('gp_historical_soft_distance_threshold', 0.2, float),
    ParameterSpec('gp_historical_soft_online_scale', 0.02, float),
    ParameterSpec('gp_historical_soft_non_online_scale', 1.0, float),
    ParameterSpec('csv_output_profile', 'full', str),
    ParameterSpec('run_name', '', str),
    ParameterSpec('data_output_dir', '.', str),
    ParameterSpec('control_frequency', 50.0, float),
    ParameterSpec('trajectory_mode', '', str),
    ParameterSpec('circle_frequency', 0.0, float),
    ParameterSpec('transition_duration', 0.0, float),
    ParameterSpec('torque_rate_limit_enabled', False, bool),
    ParameterSpec('torque_rate_limit_nm_per_s', 80.0, float),
    ParameterSpec('torque_rate_limit_log_first_n', 5, int),
    ParameterSpec('torque_rate_limit_reset_on_first_command', True, bool),
    ParameterSpec('timing_logging_enabled', False, bool),
    ParameterSpec('timing_log_stride', 1, int),
    ParameterSpec('timing_output_dir', 'outputs/goal12_controller_timing', str),
    ParameterSpec('deadline_ratio_warn_threshold', 0.8, float),
    ParameterSpec('effort_gap_diagnostics_enabled', False, bool),
    ParameterSpec('effort_gap_log_stride', 100, int),
    ParameterSpec('effort_gap_warn_sec', 0.2, float),
    ParameterSpec('callback_wall_warn_sec', 0.02, float),
    ParameterSpec('gp_prediction_stride', 5, int),
    ParameterSpec('gp_output_timeout_sec', 0.5, float),
    ParameterSpec('future_trajectory_request_stride', 5, int),
)


@dataclass(frozen=True)
class HomeSupportConfig:
    joint_check_enabled: bool
    joint_check_required_for_hist: bool
    joint_thresholds: Mapping[str, float]


@dataclass(frozen=True)
class HistoricalSupportConfig:
    require_distance_pass_for_active: bool
    distance_contribution_logging: bool
    metadata_path: str
    metadata_enforcement_enabled: bool


def declare_parameter_specs(declare_parameter, specs):
    """Declare ordered specs through the controller-owned ROS API."""
    for spec in specs:
        declare_parameter(spec.name, spec.default)


def read_home_support_config(get_bool, get_nonnegative_float):
    """Resolve home-support values with the controller's existing getters."""
    thresholds = validate_joint_home_thresholds({
        "max_abs_warn_rad": get_nonnegative_float(
            "session_home_joint_max_abs_warn_rad", 0.10
        ),
        "max_abs_refuse_rad": get_nonnegative_float(
            "session_home_joint_max_abs_refuse_rad", 0.30
        ),
        "l2_warn_rad": get_nonnegative_float(
            "session_home_joint_l2_warn_rad", 0.20
        ),
        "l2_refuse_rad": get_nonnegative_float(
            "session_home_joint_l2_refuse_rad", 0.50
        ),
        "dq_warn_rad_s": get_nonnegative_float(
            "session_home_dq_stillness_warn_rad_s", 0.02
        ),
        "dq_refuse_rad_s": get_nonnegative_float(
            "session_home_dq_stillness_refuse_rad_s", 0.05
        ),
    })
    return HomeSupportConfig(
        joint_check_enabled=get_bool("session_home_joint_check_enabled"),
        joint_check_required_for_hist=get_bool(
            "session_home_joint_check_required_for_hist"
        ),
        joint_thresholds=MappingProxyType(thresholds),
    )


def read_historical_support_config(get_bool, get_string):
    """Resolve hist-support gate values without retaining parameter access."""
    return HistoricalSupportConfig(
        require_distance_pass_for_active=get_bool(
            "gp_historical_db_require_distance_pass_for_active"
        ),
        distance_contribution_logging=get_bool(
            "gp_historical_db_distance_contribution_logging"
        ),
        metadata_path=get_string("gp_historical_db_metadata_path").strip(),
        metadata_enforcement_enabled=get_bool(
            "gp_historical_db_metadata_enforcement_enabled"
        ),
    )
