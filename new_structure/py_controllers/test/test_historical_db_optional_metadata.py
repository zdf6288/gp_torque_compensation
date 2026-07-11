"""Offline DB-load regressions for optional historical metadata sidecars."""

import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from py_controllers.historical_db_metadata import (  # noqa: E402
    create_historical_db_metadata,
    default_metadata_path,
    load_metadata_sidecar,
    validate_historical_db_metadata,
)
from py_controllers.historical_db_support import (  # noqa: E402
    DEFAULT_FEATURE_NAMES,
    scale_feature_matrix,
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
        node for node in controller.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {
        "DEFAULT_FEATURE_NAMES": DEFAULT_FEATURE_NAMES,
        "default_metadata_path": default_metadata_path,
        "load_metadata_sidecar": load_metadata_sidecar,
        "np": np,
        "os": os,
        "scale_feature_matrix": scale_feature_matrix,
        "validate_historical_db_metadata": validate_historical_db_metadata,
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


class SyntheticDbLoader:
    _load_historical_residual_db = load_controller_method(
        "_load_historical_residual_db"
    )

    def __init__(
        self, db_path, enforcement=False, session_home_path="",
        embedded_metadata=None,
    ):
        self.gp_historical_db_enabled = True
        self.gp_historical_db_path = str(db_path)
        self.gp_historical_db_metadata_path = ""
        self.gp_historical_db_metadata_enforcement_enabled = enforcement
        self.session_home_path = str(session_home_path)
        self.gp_historical_db_q_scale = 0.1
        self.gp_historical_db_dq_scale = 0.1
        self.gp_historical_db_feature_scale = np.full(14, 0.1)
        self.embedded_metadata = dict(embedded_metadata or {})
        self.logger = RecordingLogger()

    def get_logger(self):
        return self.logger

    def _read_historical_db_metadata(self, db):
        return dict(self.embedded_metadata)

    def _log_historical_db_metadata_summary(self, db_path):
        pass


class HistoricalDbOptionalMetadataTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.db_path = self.root / "historical_db.npz"
        np.savez_compressed(
            self.db_path,
            X=np.zeros((3, 14)),
            Y_residual=np.zeros((3, 7)),
        )
        self.session_home_path = self.root / "session_home.json"
        self.session_home_path.write_text(
            json.dumps({
                "q_at_capture": [0.0] * 7,
                "ee_pose_xyz": [0.3, 0.0, 0.65],
            }),
            encoding="utf-8",
        )
        self.sidecar_path = default_metadata_path(self.db_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def write_valid_sidecar(self):
        metadata = create_historical_db_metadata(
            self.db_path,
            source_csvs=[self.root / "source.csv"],
            feature_schema=DEFAULT_FEATURE_NAMES,
            target_schema=[
                f"tau_residual_{index}" for index in range(1, 8)
            ],
            session_home_path=self.session_home_path,
            q_scale=0.1,
            dq_scale=0.1,
        )
        self.sidecar_path.write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    def test_valid_sidecar_loads_with_enforcement_disabled(self):
        self.write_valid_sidecar()
        loader = SyntheticDbLoader(self.db_path, enforcement=False)
        loader._load_historical_residual_db()
        self.assertTrue(loader.gp_historical_db_loaded)
        self.assertTrue(loader.gp_historical_db_metadata_validation["valid"])

    def test_valid_bound_sidecar_loads_with_enforcement_enabled(self):
        self.write_valid_sidecar()
        loader = SyntheticDbLoader(
            self.db_path,
            enforcement=True,
            session_home_path=self.session_home_path,
        )
        loader._load_historical_residual_db()
        self.assertTrue(loader.gp_historical_db_loaded)
        self.assertTrue(loader.gp_historical_db_metadata_validation["valid"])

    def test_missing_sidecar_preserves_legacy_embedded_metadata(self):
        loader = SyntheticDbLoader(
            self.db_path,
            enforcement=False,
            embedded_metadata={"legacy_marker": "preserved"},
        )
        loader._load_historical_residual_db()
        self.assertTrue(loader.gp_historical_db_loaded)
        self.assertEqual(
            loader.gp_historical_db_metadata["legacy_marker"],
            "preserved",
        )

    def test_malformed_optional_sidecar_warns_and_db_remains_loaded(self):
        original = b"{malformed-json"
        self.sidecar_path.write_bytes(original)
        loader = SyntheticDbLoader(self.db_path, enforcement=False)
        loader._load_historical_residual_db()
        self.assertTrue(loader.gp_historical_db_loaded)
        self.assertEqual(self.sidecar_path.read_bytes(), original)
        warnings = [
            message for message in loader.logger.warning_messages
            if "Ignoring malformed optional metadata sidecar" in message
        ]
        self.assertEqual(len(warnings), 1)

    def test_malformed_sidecar_fails_closed_with_enforcement_enabled(self):
        self.sidecar_path.write_text("{malformed-json", encoding="utf-8")
        loader = SyntheticDbLoader(
            self.db_path,
            enforcement=True,
            session_home_path=self.session_home_path,
        )
        loader._load_historical_residual_db()
        self.assertFalse(loader.gp_historical_db_loaded)
        self.assertTrue(
            any(
                "failed to parse metadata sidecar" in message
                for message in loader.logger.error_messages
            )
        )

    def test_invalid_npz_still_fails_when_enforcement_is_disabled(self):
        invalid_db = self.root / "invalid.npz"
        invalid_db.write_bytes(b"not-an-npz")
        loader = SyntheticDbLoader(invalid_db, enforcement=False)
        loader._load_historical_residual_db()
        self.assertFalse(loader.gp_historical_db_loaded)
        self.assertEqual(loader.logger.warning_messages, [])
        self.assertEqual(len(loader.logger.error_messages), 1)

    def test_sidecar_loading_has_no_query_tick_call_site(self):
        tree = ast.parse(CONTROLLER_PATH.read_text(encoding="utf-8"))
        controller = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "CartesianImpedanceController"
        )
        callers = {
            method.name
            for method in controller.body
            if isinstance(method, ast.FunctionDef)
            and any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "load_metadata_sidecar"
                for node in ast.walk(method)
            )
        }
        self.assertEqual(callers, {"_load_historical_residual_db"})


if __name__ == "__main__":
    unittest.main()
