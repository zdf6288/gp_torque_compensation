#!/usr/bin/env python3
"""Build a GOAL1 historical residual database from existing controller CSV files.

Offline-only:
- no ROS
- no robot
- no controller modification
- no active compensation
- no tau_final modification

The database stores:
- X: [joint_pos_1..7, joint_vel_1..7]
- Y_residual: [tau_residual_1..7]
- Y_local: [y_hat_local_1..7]
- Y_cloud: [y_hat_cloud_1..7]
- source/run metadata per row

This is intended as a persistent offline historical database candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1] / "new_structure" / "py_controllers"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from py_controllers.historical_db_metadata import (  # noqa: E402
    create_historical_db_metadata,
)


DEFAULT_INPUTS: Dict[str, str] = {
    # 优先用真实 no-GP spatial multisine 数据作为 historical residual database。
    # 原因：no-GP residual 更接近 tau_residual target，不混入 active compensation feedback。
    "nogp_20260603": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_nogp_3000_20260603/cartesian_impedance_controller_data.csv",
    "nogp_repeat_20260603": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_nogp_repeat_end_3000_20260603/cartesian_impedance_controller_data.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build offline GOAL1 historical residual database from controller CSV files."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/goal1_historical_residual_db_20260604",
        help="Output directory for .npz database and metadata files.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help=(
            "Optional input in name=path form. Can be repeated. "
            "If omitted, uses default GOAL1 no-GP spatial multisine runs."
        ),
    )
    parser.add_argument(
        "--max-abs-q",
        type=float,
        default=10.0,
        help="Reject rows with abs(joint_pos) above this value.",
    )
    parser.add_argument(
        "--max-abs-dq",
        type=float,
        default=20.0,
        help="Reject rows with abs(joint_vel) above this value.",
    )
    parser.add_argument(
        "--max-abs-residual",
        type=float,
        default=100.0,
        help="Reject rows with abs(tau_residual) above this value.",
    )
    parser.add_argument(
        "--session-home",
        default="",
        help="Optional session_home.json to bind by SHA-256.",
    )
    parser.add_argument("--trajectory-id", default="")
    parser.add_argument("--frequency-hz", type=float, default=None)
    parser.add_argument("--q-scale", type=float, default=0.1)
    parser.add_argument("--dq-scale", type=float, default=0.1)
    parser.add_argument(
        "--notes",
        action="append",
        default=[],
        help="Optional metadata note; may be repeated.",
    )
    return parser.parse_args()


def parse_inputs(items: List[str]) -> Dict[str, str]:
    if not items:
        return dict(DEFAULT_INPUTS)

    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--input must be name=path, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --input: {item}")
        out[name] = path
    return out


def columns() -> dict:
    return {
        "q": [f"joint_pos_{j}" for j in range(1, 8)],
        "dq": [f"joint_vel_{j}" for j in range(1, 8)],
        "residual": [f"tau_residual_{j}" for j in range(1, 8)],
        "local": [f"y_hat_local_{j}" for j in range(1, 8)],
        "cloud": [f"y_hat_cloud_{j}" for j in range(1, 8)],
    }


def load_one(name: str, path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    col = columns()
    required = col["q"] + col["dq"] + col["residual"] + col["local"] + col["cloud"]

    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_csv(path)
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}")

    numeric = raw[required].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy()).all(axis=1)

    q_ok = numeric[col["q"]].abs().max(axis=1) <= args.max_abs_q
    dq_ok = numeric[col["dq"]].abs().max(axis=1) <= args.max_abs_dq
    residual_ok = numeric[col["residual"]].abs().max(axis=1) <= args.max_abs_residual

    mask = finite & q_ok & dq_ok & residual_ok
    clean = numeric.loc[mask].reset_index(drop=True)

    meta = {
        "run_name": name,
        "path": str(path),
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(clean)),
        "rows_rejected": int(len(raw) - len(clean)),
        "finite_reject_count": int((~finite).sum()),
        "q_reject_count": int((~q_ok).sum()),
        "dq_reject_count": int((~dq_ok).sum()),
        "residual_reject_count": int((~residual_ok).sum()),
    }

    clean.insert(0, "source_run", name)
    clean.insert(1, "source_row", np.nonzero(mask.to_numpy())[0])
    return clean, meta


def main() -> None:
    args = parse_args()
    if (
        not np.isfinite(args.q_scale)
        or args.q_scale <= 0.0
        or not np.isfinite(args.dq_scale)
        or args.dq_scale <= 0.0
    ):
        raise SystemExit(
            "--q-scale and --dq-scale must be finite and positive"
        )
    input_map = parse_inputs(args.input)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = []
    metas = []

    for name, path_text in input_map.items():
        frame, meta = load_one(name, Path(path_text), args)
        frames.append(frame)
        metas.append(meta)

    if not frames:
        raise SystemExit("No input frames loaded.")

    db = pd.concat(frames, ignore_index=True)
    col = columns()

    X = db[col["q"] + col["dq"]].to_numpy(float)
    Y_residual = db[col["residual"]].to_numpy(float)
    Y_local = db[col["local"]].to_numpy(float)
    Y_cloud = db[col["cloud"]].to_numpy(float)
    source_run = db["source_run"].astype(str).to_numpy()
    source_row = db["source_row"].to_numpy(int)

    feature_names = np.array(col["q"] + col["dq"], dtype=object)
    residual_names = np.array(col["residual"], dtype=object)

    npz_path = outdir / "goal1_historical_residual_db.npz"
    np.savez_compressed(
        npz_path,
        X=X,
        Y_residual=Y_residual,
        Y_local=Y_local,
        Y_cloud=Y_cloud,
        source_run=source_run,
        source_row=source_row,
        feature_names=feature_names,
        residual_names=residual_names,
    )

    summary = create_historical_db_metadata(
        npz_path,
        source_csvs=list(input_map.values()),
        feature_schema=feature_names.tolist(),
        target_schema=residual_names.tolist(),
        session_home_path=args.session_home,
        trajectory_id=args.trajectory_id,
        frequency_hz=args.frequency_hz,
        q_scale=args.q_scale,
        dq_scale=args.dq_scale,
        notes=args.notes,
    )
    summary.update({
        "offline_only": True,
        "active_compensation": False,
        "description": "Persistent offline historical residual database candidate built from real GOAL1 controller CSV files.",
        "rows_total": int(len(db)),
        "inputs": metas,
        "quality_thresholds": {
            "max_abs_q": args.max_abs_q,
            "max_abs_dq": args.max_abs_dq,
            "max_abs_residual": args.max_abs_residual,
        },
        "safety_notes": [
            "This file is not loaded by the controller.",
            "This database does not enter tau_final.",
            "Use only for offline retrieval analysis until separate safety review.",
        ],
    })

    json_path = outdir / "goal1_historical_residual_db_metadata.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pd.DataFrame(metas).to_csv(outdir / "goal1_historical_residual_db_input_summary.csv", index=False)

    print("===== GOAL1 historical residual DB built =====")
    print("rows_total:", len(db))
    print("feature_dim:", X.shape[1])
    print("target_dim:", Y_residual.shape[1])
    print("npz:", npz_path, npz_path.stat().st_size, "bytes")
    print("metadata:", json_path, json_path.stat().st_size, "bytes")
    print("summary:", outdir / "goal1_historical_residual_db_input_summary.csv")


if __name__ == "__main__":
    main()
