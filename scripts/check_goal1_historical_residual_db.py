#!/usr/bin/env python3
"""Check GOAL1 historical residual database schema and numeric health.

Offline-only:
- no ROS
- no robot
- no controller modification
- no active compensation
- no tau_final modification
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1] / "new_structure" / "py_controllers"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from py_controllers.historical_db_metadata import (  # noqa: E402
    load_metadata_sidecar,
    validate_historical_db_metadata,
)
from py_controllers.historical_db_support import (  # noqa: E402
    DEFAULT_FEATURE_NAMES,
    build_joint_feature,
    compute_scaled_delta_contributions,
    format_distance_contribution_report,
    query_scaled_nearest_support,
    scale_feature,
    scale_feature_matrix,
)


REQUIRED_ARRAYS = [
    "X",
    "Y_residual",
    "Y_local",
    "Y_cloud",
    "source_run",
    "source_row",
    "feature_names",
    "residual_names",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check GOAL1 historical residual DB .npz.")
    p.add_argument(
        "--db",
        default="outputs/goal1_historical_residual_db_20260604/goal1_historical_residual_db.npz",
        help="Path to historical residual DB .npz.",
    )
    p.add_argument(
        "--output-json",
        default="",
        help="Optional output JSON report path.",
    )
    p.add_argument(
        "--metadata", default="", help="Optional metadata sidecar path."
    )
    p.add_argument("--require-metadata", action="store_true")
    p.add_argument("--session-home", default="")
    p.add_argument("--query-q", default="", help="Comma-separated q1..q7.")
    p.add_argument("--query-dq", default="", help="Comma-separated dq1..dq7.")
    p.add_argument("--q-scale", type=float, default=0.1)
    p.add_argument("--dq-scale", type=float, default=0.1)
    p.add_argument("--max-distance", type=float, default=1.0)
    return p.parse_args()


def parse_vector(text: str, name: str) -> np.ndarray:
    try:
        vector = np.asarray([float(value) for value in text.split(",")])
    except ValueError as exc:
        raise ValueError(
            f"{name} must contain comma-separated numbers"
        ) from exc
    if vector.shape != (7,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain 7 finite values")
    return vector


def add_metadata_report(report, args, db_path):
    try:
        metadata, metadata_path = load_metadata_sidecar(db_path, args.metadata)
        validation = validate_historical_db_metadata(
            metadata,
            db_path,
            session_home_path=args.session_home,
            expected_feature_schema=DEFAULT_FEATURE_NAMES,
            q_scale=args.q_scale,
            dq_scale=args.dq_scale,
            require_metadata=args.require_metadata,
            require_session_binding=bool(
                args.require_metadata and args.session_home
            ),
        )
        report["metadata_path"] = str(metadata_path)
        report["metadata"] = metadata
        report["metadata_validation"] = validation
        report["errors"].extend(validation["errors"])
        report["warnings"].extend(validation["warnings"])
    except Exception as exc:
        report["errors"].append(f"metadata check failed: {exc}")


def add_query_report(report, args, data):
    if not args.query_q and not args.query_dq:
        return
    if not args.query_q or not args.query_dq:
        report["errors"].append(
            "--query-q and --query-dq must be used together"
        )
        return
    try:
        q = parse_vector(args.query_q, "--query-q")
        dq = parse_vector(args.query_dq, "--query-dq")
        feature = build_joint_feature(q, dq)
        scale = np.array([args.q_scale] * 7 + [args.dq_scale] * 7)
        x_scaled = scale_feature_matrix(data["X"], scale)
        query_scaled = scale_feature(feature, scale)
        support = query_scaled_nearest_support(
            x_scaled,
            data["Y_residual"],
            query_scaled,
            1,
            args.max_distance,
        )
        if not support["valid"]:
            raise ValueError("nearest-support query failed")
        nearest = np.asarray(data["X"][support["nearest_index"]], dtype=float)
        contributions = compute_scaled_delta_contributions(
            nearest, feature, scale
        )
        report["query_support"] = {
            "nearest_index": support["nearest_index"],
            "nearest_distance": support["nearest_distance"],
            "distance_pass": support["distance_pass"],
            "feature_names": list(contributions["feature_names"]),
            "scaled_delta": contributions["scaled_delta"].tolist(),
            "contribution": contributions["contribution"].tolist(),
            "diagnostic": format_distance_contribution_report(contributions),
        }
    except Exception as exc:
        report["errors"].append(f"query support check failed: {exc}")


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)

    report = {
        "db_path": str(db_path),
        "exists": db_path.exists(),
        "status": "UNKNOWN",
        "errors": [],
        "warnings": [],
    }

    if not db_path.exists():
        report["status"] = "FAIL"
        report["errors"].append("DB file does not exist.")
        print(json.dumps(report, indent=2))
        raise SystemExit(1)

    data = np.load(db_path, allow_pickle=True)
    arrays = list(data.files)
    report["arrays"] = arrays

    missing = [k for k in REQUIRED_ARRAYS if k not in arrays]
    if missing:
        report["errors"].append(f"Missing arrays: {missing}")

    if not missing:
        X = data["X"]
        Y_residual = data["Y_residual"]
        Y_local = data["Y_local"]
        Y_cloud = data["Y_cloud"]
        source_run = data["source_run"]
        source_row = data["source_row"]

        report["rows"] = int(len(X))
        report["feature_dim"] = int(X.shape[1]) if X.ndim == 2 else None
        report["target_dim"] = int(Y_residual.shape[1]) if Y_residual.ndim == 2 else None

        if X.ndim != 2 or X.shape[1] != 14:
            report["errors"].append(f"X must be shape (N,14), got {X.shape}.")
        if Y_residual.ndim != 2 or Y_residual.shape[1] != 7:
            report["errors"].append(f"Y_residual must be shape (N,7), got {Y_residual.shape}.")
        if Y_local.shape != Y_residual.shape:
            report["errors"].append(f"Y_local shape {Y_local.shape} != Y_residual shape {Y_residual.shape}.")
        if Y_cloud.shape != Y_residual.shape:
            report["errors"].append(f"Y_cloud shape {Y_cloud.shape} != Y_residual shape {Y_residual.shape}.")
        if len(source_run) != len(X):
            report["errors"].append("source_run length mismatch.")
        if len(source_row) != len(X):
            report["errors"].append("source_row length mismatch.")

        for name in ["X", "Y_residual", "Y_local", "Y_cloud"]:
            arr = data[name]
            finite = bool(np.isfinite(arr).all())
            report[f"{name}_finite"] = finite
            report[f"{name}_max_abs"] = float(np.max(np.abs(arr))) if arr.size else 0.0
            if not finite:
                report["errors"].append(f"{name} contains non-finite values.")

        if len(X) == 0:
            report["errors"].append("DB has zero rows.")

    add_metadata_report(report, args, db_path)
    add_query_report(report, args, data)

    report["status"] = "PASS" if not report["errors"] else "FAIL"

    text = json.dumps(report, indent=2)
    print(text)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print("wrote:", out)

    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
