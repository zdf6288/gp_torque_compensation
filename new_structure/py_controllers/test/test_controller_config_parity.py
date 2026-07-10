"""Parity tests for extracted GOAL12 controller parameter configuration."""

import ast
from pathlib import Path
import sys
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from py_controllers.cartesian_impedance_config import (  # noqa: E402
    HIST_SUPPORT_PARAMETER_SPECS,
    HOME_SUPPORT_PARAMETER_SPECS,
    declare_parameter_specs,
    read_historical_support_config,
    read_home_support_config,
)


EXPECTED_SPECS = (
    ("session_home_joint_check_enabled", False, bool),
    ("session_home_joint_check_required_for_hist", True, bool),
    ("session_home_joint_max_abs_warn_rad", 0.10, float),
    ("session_home_joint_max_abs_refuse_rad", 0.30, float),
    ("session_home_joint_l2_warn_rad", 0.20, float),
    ("session_home_joint_l2_refuse_rad", 0.50, float),
    ("session_home_dq_stillness_warn_rad_s", 0.02, float),
    ("session_home_dq_stillness_refuse_rad_s", 0.05, float),
    ("gp_historical_db_require_distance_pass_for_active", False, bool),
    ("gp_historical_db_distance_contribution_logging", False, bool),
    ("gp_historical_db_metadata_path", "", str),
    ("gp_historical_db_metadata_enforcement_enabled", False, bool),
)


class ControllerConfigParityTest(unittest.TestCase):
    def test_parameter_names_defaults_types_and_declaration_order(self):
        specs = HOME_SUPPORT_PARAMETER_SPECS + HIST_SUPPORT_PARAMETER_SPECS
        actual = tuple(
            (spec.name, spec.default, spec.value_type) for spec in specs
        )
        self.assertEqual(actual, EXPECTED_SPECS)

        declarations = []
        declare_parameter_specs(
            lambda name, default: declarations.append((name, default)), specs
        )
        self.assertEqual(
            declarations,
            [(name, default) for name, default, _ in EXPECTED_SPECS],
        )

    def test_resolved_values_preserve_controller_getter_behavior(self):
        values = {name: default for name, default, _ in EXPECTED_SPECS}
        values.update({
            "session_home_joint_check_enabled": True,
            "session_home_joint_l2_refuse_rad": 0.75,
            "gp_historical_db_require_distance_pass_for_active": True,
            "gp_historical_db_metadata_path": "  /tmp/db_metadata.json  ",
        })
        home = read_home_support_config(
            lambda name: bool(values[name]),
            lambda name, default: float(values.get(name, default)),
        )
        hist = read_historical_support_config(
            lambda name: bool(values[name]),
            lambda name: str(values[name]),
        )
        self.assertTrue(home.joint_check_enabled)
        self.assertTrue(home.joint_check_required_for_hist)
        self.assertEqual(home.joint_thresholds["l2_refuse_rad"], 0.75)
        self.assertTrue(hist.require_distance_pass_for_active)
        self.assertEqual(hist.metadata_path, "/tmp/db_metadata.json")

    def test_invalid_threshold_order_still_fails(self):
        values = {name: default for name, default, _ in EXPECTED_SPECS}
        values["session_home_joint_max_abs_warn_rad"] = 0.4
        values["session_home_joint_max_abs_refuse_rad"] = 0.3
        with self.assertRaisesRegex(ValueError, "must be >="):
            read_home_support_config(
                lambda name: bool(values[name]),
                lambda name, default: float(values.get(name, default)),
            )

    def test_python_only_launch_defaults_match_specs(self):
        launch_path = PACKAGE_ROOT / "launch" / (
            "cartesian_impedance_python_only_compensation_trajectory_launch.py"
        )
        tree = ast.parse(launch_path.read_text(encoding="utf-8"))
        launch_defaults = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "DeclareLaunchArgument" or not node.args:
                continue
            if not isinstance(node.args[0], ast.Constant):
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg == "default_value"
                    and isinstance(keyword.value, ast.Constant)
                ):
                    launch_defaults[node.args[0].value] = keyword.value.value
        for name, default, value_type in EXPECTED_SPECS:
            self.assertIn(name, launch_defaults)
            text = launch_defaults[name]
            if value_type is bool:
                self.assertEqual(text, str(default).lower(), name)
            elif value_type is float:
                self.assertEqual(float(text), default, name)
            else:
                self.assertEqual(text, str(default), name)


if __name__ == "__main__":
    unittest.main()
