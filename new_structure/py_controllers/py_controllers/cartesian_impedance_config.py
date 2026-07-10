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
