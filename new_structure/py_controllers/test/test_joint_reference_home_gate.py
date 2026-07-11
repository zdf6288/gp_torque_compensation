"""Offline regression tests for the joint-reference home-support gate."""

import ast
from pathlib import Path
import sys
import unittest

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from py_controllers.session_home_feasibility import (  # noqa: E402
    classify_joint_home,
    compute_joint_home_metrics,
    format_joint_home_report,
)


CONTROLLER_PATH = (
    PACKAGE_ROOT / "py_controllers" / "cartesian_impedance.py"
)


def load_controller_method(name):
    """Compile one controller method without importing ROS modules."""
    tree = ast.parse(CONTROLLER_PATH.read_text(encoding="utf-8"))
    controller = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "CartesianImpedanceController"
    )
    method = next(
        (
            node for node in controller.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )
    if method is None:
        raise AssertionError(f"controller method not found: {name}")
    namespace = {
        "np": np,
        "classify_joint_home": classify_joint_home,
        "compute_joint_home_metrics": compute_joint_home_metrics,
        "format_joint_home_report": format_joint_home_report,
    }
    ast.fix_missing_locations(method)
    exec(compile(ast.Module([method], []), str(CONTROLLER_PATH), "exec"), namespace)
    return namespace[name]


class RecordingLogger:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []
        self.error_messages = []

    def info(self, message):
        self.info_messages.append(message)

    def warn(self, message):
        self.warning_messages.append(message)

    def error(self, message):
        self.error_messages.append(message)


class SyntheticController:
    _historical_db_source_requested = load_controller_method(
        "_historical_db_source_requested"
    )
    _evaluate_session_home_joint_gate = load_controller_method(
        "_evaluate_session_home_joint_gate"
    )
    _joint_reference_home_gate_allows_effort = load_controller_method(
        "_joint_reference_home_gate_allows_effort"
    )
    _handle_joint_reference_control = load_controller_method(
        "_handle_joint_reference_control"
    )

    def __init__(self, source="hist_db", q_at_capture=None):
        self.gp_compensation_enabled = True
        self.gp_prediction_enabled = True
        self.gp_compensation_source = source
        self.session_home_joint_check_enabled = False
        self.session_home_joint_check_required_for_hist = True
        self.session_home_joint_thresholds = {
            "max_abs_warn_rad": 0.10,
            "max_abs_refuse_rad": 0.30,
            "l2_warn_rad": 0.20,
            "l2_refuse_rad": 0.50,
            "dq_warn_rad_s": 0.02,
            "dq_refuse_rad_s": 0.05,
        }
        self.session_home_q_at_capture = q_at_capture
        self.session_home = np.zeros(3)
        self._last_ee_pose = np.zeros(3)
        self._session_home_joint_gate_decision = None
        self._session_home_joint_gate_context = None
        self.logger = RecordingLogger()
        self.skip_reasons = []
        self.gp_apply_count = 0
        self.publish_count = 0

        self.joint_command_received = True
        self.joint_command_enabled = True
        self.joint_command_time = FakeTime(0.0)
        self.joint_space_command_timeout_sec = 1.0
        self.joint_reference_kp = np.zeros(7)
        self.joint_reference_kd = np.zeros(7)
        self.q_des_joint = np.zeros(7)
        self.dq_des_joint = np.zeros(7)
        self.ddq_des_joint = np.zeros(7)
        self.joint_reference_torque_clip_nm = 1.0
        self.joint_reference_last_tau = np.zeros(7)
        self.joint_reference_last_tau_time = None
        self.joint_reference_torque_rate_limit_nm_per_s = 1.0
        self.data_recording_enabled = False
        self.tau_residual_filtered = np.zeros(7)
        self.state_buffer = []
        self.use_gp = False
        self.effort_msg = type("Effort", (), {})()

    def get_logger(self):
        return self.logger

    def _mark_effort_publish_skipped(self, reason):
        self.skip_reasons.append(reason)

    def _reset_historical_residual_db_shadow_state(self):
        pass

    def _update_gp_shadow_logging_state(self, q, dq):
        pass

    def _apply_gp_compensation(self, tau):
        self.gp_apply_count += 1
        return tau

    def _apply_torque_rate_limit(self, tau, t_now):
        return tau

    def _publish_effort(self, message):
        self.publish_count += 1


class FakeDuration:
    def __init__(self, seconds):
        self.nanoseconds = int(seconds * 1e9)


class FakeTime:
    def __init__(self, seconds):
        self.seconds = float(seconds)

    def __sub__(self, other):
        return FakeDuration(self.seconds - other.seconds)


def dispatch(controller, q, dq):
    return controller._joint_reference_home_gate_allows_effort(
        np.asarray(q, dtype=float), np.asarray(dq, dtype=float)
    )


def run_joint_control(controller, q, dq):
    controller._handle_joint_reference_control(
        FakeTime(0.1), 0.1,
        np.asarray(q, dtype=float), np.asarray(dq, dtype=float),
        0.02, np.zeros(7), np.zeros(7), np.zeros(7),
        timing_row=None,
    )


class JointReferenceHomeGateTest(unittest.TestCase):
    def test_matching_hist_home_reaches_joint_control(self):
        controller = SyntheticController(q_at_capture=np.zeros(7))
        self.assertTrue(dispatch(controller, np.zeros(7), np.zeros(7)))
        self.assertEqual(controller.skip_reasons, [])

    def test_far_hist_home_refuses_before_joint_control(self):
        controller = SyntheticController(q_at_capture=np.zeros(7))
        q_far = np.zeros(7)
        q_far[6] = 1.0
        run_joint_control(controller, q_far, np.zeros(7))
        self.assertEqual(controller.gp_apply_count, 0)
        self.assertEqual(controller.publish_count, 0)
        self.assertEqual(
            controller.skip_reasons,
            ["joint_session_home_gate_refused"],
        )
        self.assertTrue(
            any(
                "SESSION_HOME_JOINT_GATE_REFUSE" in message
                for message in controller.logger.error_messages
            )
        )

    def test_missing_q_at_capture_refuses_active_hist(self):
        controller = SyntheticController(q_at_capture=None)
        run_joint_control(controller, np.zeros(7), np.zeros(7))
        self.assertEqual(controller.gp_apply_count, 0)
        self.assertEqual(controller.publish_count, 0)
        self.assertTrue(
            any(
                "SESSION_HOME_JOINT_GATE_NO_Q_AT_CAPTURE" in message
                for message in controller.logger.error_messages
            )
        )

    def test_non_hist_sources_preserve_default_joint_behavior(self):
        for source in ("local", "cloud", "combined"):
            with self.subTest(source=source):
                controller = SyntheticController(
                    source=source, q_at_capture=None
                )
                self.assertTrue(
                    dispatch(controller, np.ones(7), np.ones(7))
                )

    def test_explicit_joint_check_still_applies_to_non_hist_source(self):
        controller = SyntheticController(
            source="local", q_at_capture=np.zeros(7)
        )
        controller.session_home_joint_check_enabled = True
        q_far = np.zeros(7)
        q_far[0] = 1.0
        self.assertFalse(dispatch(controller, q_far, np.zeros(7)))

    def test_warn_only_preserves_existing_allow_policy(self):
        controller = SyntheticController(q_at_capture=np.zeros(7))
        q_warn = np.zeros(7)
        q_warn[0] = 0.11
        self.assertTrue(dispatch(controller, q_warn, np.zeros(7)))
        self.assertTrue(
            any(
                "SESSION_HOME_JOINT_GATE_WARN" in message
                for message in controller.logger.warning_messages
            )
        )

    def test_switching_from_non_hist_to_hist_rechecks_cached_decision(self):
        controller = SyntheticController(source="local", q_at_capture=None)
        self.assertTrue(dispatch(controller, np.zeros(7), np.zeros(7)))
        controller.gp_compensation_source = "hist_db"
        self.assertFalse(dispatch(controller, np.zeros(7), np.zeros(7)))

    def test_all_accepted_historical_sources_require_support(self):
        for source in (
            "hist_db",
            "triple",
            "triple_dynamic",
            "triple_dynamic_gated",
        ):
            with self.subTest(source=source):
                controller = SyntheticController(
                    source=source, q_at_capture=None
                )
                self.assertTrue(
                    controller._historical_db_source_requested()
                )
                self.assertFalse(
                    dispatch(controller, np.zeros(7), np.zeros(7))
                )

    def test_cartesian_start_gate_call_remains_present(self):
        source = CONTROLLER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        controller = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "CartesianImpedanceController"
        )
        callback = next(
            node for node in controller.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "stateParameterCallback"
        )
        callback_calls = {
            node.func.attr
            for node in ast.walk(callback)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("_handle_joint_reference_control", callback_calls)
        self.assertIn("_evaluate_normal_run_start_gate", callback_calls)

        joint_handler = next(
            node for node in controller.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_handle_joint_reference_control"
        )
        call_lines = {
            node.func.attr: node.lineno
            for node in ast.walk(joint_handler)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in (
                "_joint_reference_home_gate_allows_effort",
                "_apply_gp_compensation",
                "_publish_effort",
            )
        }
        self.assertLess(
            call_lines["_joint_reference_home_gate_allows_effort"],
            call_lines["_apply_gp_compensation"],
        )
        self.assertLess(
            call_lines["_joint_reference_home_gate_allows_effort"],
            call_lines["_publish_effort"],
        )

    def test_adopting_new_session_home_invalidates_cached_gate(self):
        tree = ast.parse(CONTROLLER_PATH.read_text(encoding="utf-8"))
        controller = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "CartesianImpedanceController"
        )
        adopt = next(
            node for node in controller.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_adopt_session_home"
        )
        reset_names = {
            target.attr
            for node in ast.walk(adopt)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
            for target in node.targets
            if isinstance(target, ast.Attribute)
        }
        self.assertIn("_session_home_joint_gate_decision", reset_names)
        self.assertIn("_session_home_joint_gate_context", reset_names)


if __name__ == "__main__":
    unittest.main()
