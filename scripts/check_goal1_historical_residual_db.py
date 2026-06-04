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

import numpy as np


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
    return p.parse_args()


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
