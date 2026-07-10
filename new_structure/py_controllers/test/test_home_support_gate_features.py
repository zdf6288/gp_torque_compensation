"""Focused tests for GOAL12 M-HomeSupportGate-1 feature behavior."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from py_controllers.historical_db_metadata import (  # noqa: E402
    create_historical_db_metadata,
    validate_historical_db_metadata,
)
from py_controllers.historical_db_support import (  # noqa: E402
    DEFAULT_FEATURE_NAMES,
    compute_scaled_delta_contributions,
    select_active_gated_prediction,
)
from py_controllers.session_home_feasibility import (  # noqa: E402
    classify_joint_home,
    compute_joint_home_metrics,
    validate_joint_home_thresholds,
)


class SessionHomeFeasibilityTest(unittest.TestCase):
    def classify(self, q, dq, q_home, require=True):
        metrics = compute_joint_home_metrics(q, dq, q_home)
        return classify_joint_home(
            metrics, enabled=True, require_q_home=require
        )

    def test_allow_warn_and_refuse(self):
        q_home = np.zeros(7)
        self.assertEqual(
            self.classify(q_home, np.zeros(7), q_home)["decision"], "ALLOW"
        )
        q_warn = q_home.copy()
        q_warn[2] = 0.11
        self.assertEqual(
            self.classify(q_warn, np.zeros(7), q_home)["decision"],
            "WARN_ONLY",
        )
        q_far = q_home.copy()
        q_far[6] = 1.2
        decision = self.classify(q_far, np.zeros(7), q_home)
        self.assertEqual(decision["decision"], "REFUSE")
        self.assertFalse(decision["allowed"])

    def test_missing_data_decisions(self):
        q = np.zeros(7)
        self.assertEqual(
            self.classify(q, q, None)["decision"], "NO_Q_AT_CAPTURE"
        )
        self.assertEqual(
            self.classify(None, q, q)["decision"], "NO_CURRENT_Q"
        )
        self.assertEqual(
            self.classify(q, None, q)["decision"], "NO_CURRENT_DQ"
        )

    def test_threshold_order_is_validated(self):
        with self.assertRaisesRegex(ValueError, "must be >="):
            validate_joint_home_thresholds({
                "max_abs_warn_rad": 0.4,
                "max_abs_refuse_rad": 0.3,
            })


class HistoricalSupportContractTest(unittest.TestCase):
    def test_strict_distance_contract_preserves_fallback(self):
        prediction = np.arange(7, dtype=float)
        fallback = np.full(7, -2.0)
        available, gated, source = select_active_gated_prediction(
            prediction, fallback, 3,
            loaded=1, query_valid=1, prediction_valid=1,
            online_disabled=0, distance_pass=0,
            require_distance_pass=True,
        )
        self.assertEqual(available, 0)
        self.assertEqual(source, 3)
        np.testing.assert_array_equal(gated, fallback)

    def test_legacy_distance_behavior_remains_opt_in(self):
        prediction = np.arange(7, dtype=float)
        available, gated, source = select_active_gated_prediction(
            prediction, np.zeros(7), 3,
            loaded=1, query_valid=1, prediction_valid=1,
            online_disabled=0, distance_pass=0,
            require_distance_pass=False,
        )
        self.assertEqual(available, 1)
        self.assertEqual(source, 4)
        np.testing.assert_array_equal(gated, prediction)

    def test_q7_and_dq7_contributions(self):
        nearest = np.zeros(14)
        query = np.zeros(14)
        query[6] = 0.2
        query[13] = 0.3
        scale = np.array([0.1] * 14)
        result = compute_scaled_delta_contributions(nearest, query, scale)
        self.assertAlmostEqual(result["contribution"][6], 4.0)
        self.assertAlmostEqual(result["contribution"][13], 9.0)
        self.assertAlmostEqual(result["total_distance"], np.sqrt(13.0))


class HistoricalMetadataBindingTest(unittest.TestCase):
    def test_db_and_session_hash_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "db.npz"
            db_path.write_bytes(b"synthetic-db")
            session_path = root / "session_home.json"
            session_path.write_text(json.dumps({
                "q_at_capture": [0.0] * 7,
                "ee_pose_xyz": [0.3, 0.0, 0.6],
            }))
            metadata = create_historical_db_metadata(
                db_path, [root / "source.csv"], DEFAULT_FEATURE_NAMES,
                [f"tau_residual_{index}" for index in range(1, 8)],
                session_home_path=session_path, q_scale=0.1, dq_scale=0.1,
            )
            validation = validate_historical_db_metadata(
                metadata, db_path, session_home_path=session_path,
                expected_feature_schema=DEFAULT_FEATURE_NAMES,
                q_scale=0.1, dq_scale=0.1,
                require_metadata=True, require_session_binding=True,
            )
            self.assertTrue(validation["valid"], validation["errors"])

            malformed = dict(metadata)
            malformed["feature_schema"] = None
            malformed["q_scale"] = "invalid"
            malformed_result = validate_historical_db_metadata(
                malformed, db_path, session_home_path=session_path,
                expected_feature_schema=DEFAULT_FEATURE_NAMES,
                q_scale=0.1, dq_scale=0.1,
                require_metadata=False, require_session_binding=False,
            )
            self.assertFalse(malformed_result["valid"])
            self.assertIn(
                "feature_schema must be a list", malformed_result["errors"]
            )

            missing_runtime_home = validate_historical_db_metadata(
                metadata, db_path,
                expected_feature_schema=DEFAULT_FEATURE_NAMES,
                q_scale=0.1, dq_scale=0.1,
                require_metadata=True, require_session_binding=True,
            )
            self.assertFalse(missing_runtime_home["valid"])
            self.assertIn(
                "runtime session_home_path is required for binding",
                missing_runtime_home["errors"],
            )

            session_path.write_text(json.dumps({"q_at_capture": [1.0] * 7}))
            mismatch = validate_historical_db_metadata(
                metadata, db_path, session_home_path=session_path,
                expected_feature_schema=DEFAULT_FEATURE_NAMES,
                q_scale=0.1, dq_scale=0.1,
                require_metadata=True, require_session_binding=True,
            )
            self.assertFalse(mismatch["valid"])
            self.assertIn(
                "session_home_sha256 does not match runtime session home",
                mismatch["errors"],
            )


if __name__ == "__main__":
    unittest.main()
