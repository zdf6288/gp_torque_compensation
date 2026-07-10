"""Parity tests for GOAL12 pure helper extraction."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from py_controllers.historical_db_support import (  # noqa: E402
    build_joint_feature,
    query_scaled_nearest_support,
    scale_feature,
    scale_feature_matrix,
    select_legacy_gated_prediction,
)
from py_controllers.session_anchor_utils import (  # noqa: E402
    load_session_home_payload,
    read_optional_q_at_capture,
)


def legacy_query(
    x_db_scaled, y_db, q, dq, scale, k, max_distance, online_disabled,
):
    result = {
        "query_valid": 0,
        "available": 0,
        "distance_pass": 0,
        "k_used": 0,
        "nearest_distance": 0.0,
        "mean_topk_distance": 0.0,
        "prediction": np.zeros(7),
        "gated_prediction": np.full(7, -3.0),
        "gated_source_code": 2,
    }
    q_arr = np.asarray(q, dtype=float)
    dq_arr = np.asarray(dq, dtype=float)
    if q_arr.shape != (7,) or dq_arr.shape != (7,):
        return result
    if not np.all(np.isfinite(q_arr)) or not np.all(np.isfinite(dq_arr)):
        return result
    x_query_scaled = np.ascontiguousarray(
        np.concatenate([q_arr, dq_arr]) / scale, dtype=float
    )
    if not np.all(np.isfinite(x_query_scaled)):
        return result
    result["query_valid"] = 1
    delta = x_db_scaled - x_query_scaled.reshape(1, -1)
    distance_sq = np.einsum("ij,ij->i", delta, delta)
    k_used = min(k, len(x_db_scaled))
    nearest = np.argpartition(distance_sq, kth=k_used - 1)[:k_used]
    nearest = nearest[np.argsort(distance_sq[nearest])]
    distances = np.sqrt(distance_sq[nearest])
    prediction = np.mean(y_db[nearest], axis=0)
    result.update({
        "k_used": int(k_used),
        "nearest_distance": float(distances[0]),
        "mean_topk_distance": float(np.mean(distances)),
        "distance_pass": int(distances[0] <= max_distance),
        "prediction": prediction.copy(),
        "available": int(not online_disabled),
    })
    if result["available"]:
        result["gated_prediction"] = prediction.copy()
        result["gated_source_code"] = 4
    return result


class HistoricalSupportParityTest(unittest.TestCase):
    def test_query_feature_scaling_distance_and_fallback_parity(self):
        rng = np.random.default_rng(1201)
        x_db = rng.normal(size=(31, 14))
        y_db = rng.normal(size=(31, 7))
        q = rng.normal(size=7)
        dq = rng.normal(size=7)
        scale = np.array([0.1] * 7 + [0.2] * 7)
        legacy_scaled = np.ascontiguousarray(x_db / scale.reshape(1, -1))
        helper_scaled = scale_feature_matrix(x_db, scale)
        np.testing.assert_allclose(
            helper_scaled, legacy_scaled, rtol=0.0, atol=0.0
        )

        feature = build_joint_feature(q, dq)
        np.testing.assert_allclose(
            feature, np.concatenate([q, dq]), rtol=0.0, atol=0.0
        )
        query_scaled = scale_feature(feature, scale)
        legacy_query_scaled = np.ascontiguousarray(
            feature / scale, dtype=float
        )
        np.testing.assert_allclose(
            query_scaled, legacy_query_scaled, rtol=0.0, atol=0.0
        )

        for max_distance in (0.01, 100.0):
            for online_disabled in (0, 1):
                expected = legacy_query(
                    legacy_scaled, y_db, q, dq, scale, 5,
                    max_distance, online_disabled,
                )
                support = query_scaled_nearest_support(
                    helper_scaled, y_db, query_scaled, 5, max_distance
                )
                available, gated, source_code = select_legacy_gated_prediction(
                    support["prediction"], np.full(7, -3.0), 2,
                    1, 1, support["valid"], online_disabled,
                )
                self.assertEqual(support["k_used"], expected["k_used"])
                self.assertEqual(
                    support["distance_pass"], expected["distance_pass"]
                )
                self.assertEqual(available, expected["available"])
                self.assertEqual(source_code, expected["gated_source_code"])
                self.assertAlmostEqual(
                    support["nearest_distance"],
                    expected["nearest_distance"], places=15,
                )
                self.assertAlmostEqual(
                    support["mean_topk_distance"],
                    expected["mean_topk_distance"], places=15,
                )
                np.testing.assert_allclose(
                    support["prediction"], expected["prediction"],
                    rtol=0.0, atol=0.0,
                )
                np.testing.assert_allclose(
                    gated, expected["gated_prediction"], rtol=0.0, atol=0.0
                )

    def test_invalid_joint_feature_matches_legacy_non_query(self):
        self.assertIsNone(build_joint_feature([0.0] * 6, [0.0] * 7))
        self.assertIsNone(build_joint_feature([0.0] * 7, [0.0] * 6))
        self.assertIsNone(
            build_joint_feature([float("nan")] * 7, [0.0] * 7)
        )


class SessionPayloadParityTest(unittest.TestCase):
    def test_json_object_and_invalid_payload_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.json"
            valid.write_text(json.dumps({"q_at_capture": [0.0] * 7}))
            self.assertEqual(
                load_session_home_payload(valid), {"q_at_capture": [0.0] * 7}
            )

            non_object = root / "non_object.json"
            non_object.write_text("[]")
            with self.assertRaisesRegex(ValueError, "is not an object"):
                load_session_home_payload(non_object)

            malformed = root / "malformed.json"
            malformed.write_text("{")
            with self.assertRaisesRegex(
                ValueError, "failed to parse session home JSON"
            ):
                load_session_home_payload(malformed)

    def test_optional_q_schema_and_legacy_errors(self):
        prefix = "[SessionAnchor] 'fixture.json': "
        self.assertIsNone(read_optional_q_at_capture({}, prefix))
        np.testing.assert_array_equal(
            read_optional_q_at_capture({"q_at_capture": [0.0] * 7}, prefix),
            np.zeros(7),
        )
        with self.assertRaisesRegex(
            ValueError, "must be null or 7 finite values"
        ):
            read_optional_q_at_capture({"q_at_capture": [0.0] * 6}, prefix)


if __name__ == "__main__":
    unittest.main()
