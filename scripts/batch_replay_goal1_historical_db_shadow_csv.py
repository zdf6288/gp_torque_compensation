#!/usr/bin/env python3
"""Batch replay GOAL1 historical DB shadow columns onto multiple controller CSVs.

Offline-only:
- no ROS
- no robot
- no controller modification
- no active compensation
- no tau_final modification

This script reuses replay_goal1_historical_db_shadow_csv.py logic and writes
ignored/generated shadow CSV outputs for multiple real GOAL1 runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

# 允许从同一 scripts/ 目录 import 已提交的 replay 工具函数。
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import replay_goal1_historical_db_shadow_csv as replay


DEFAULT_DB = "outputs/goal1_historical_residual_db_20260604/goal1_historical_residual_db.npz"

DEFAULT_INPUTS: Dict[str, str] = {
    "local_scale10": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_local_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
    "cloud_scale10": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_cloud_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
    "combined_scale10": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_combined_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
    "online_local_scale10": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_online_local_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
    "online_cloud_scale10": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_online_cloud_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
    "online_combined_scale10": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_online_combined_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch offline replay of GOAL1 historical DB shadow CSVs.")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--output-dir", default="outputs/goal1_historical_db_shadow_replay_batch_20260604")
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--q-scale", type=float, default=0.1)
    p.add_argument("--dq-scale", type=float, default=0.1)
    p.add_argument("--max-distance", type=float, default=1.0)
    p.add_argument("--chunk-size", type=int, default=250)
    p.add_argument(
        "--disable-historical-for-online",
        action="store_true",
        default=True,
        help="Disable historical gate for input names/paths containing online.",
    )
    return p.parse_args()


def replay_one(
    *,
    db_path: Path,
    input_name: str,
    input_csv: Path,
    output_csv: Path,
    k: int,
    q_scale: float,
    dq_scale: float,
    max_distance: float,
    chunk_size: int,
    disable_historical_for_online: bool,
) -> dict:
    X_db, Y_db = replay.load_db(db_path)
    df = replay.pd.read_csv(input_csv)

    x_cols, local_cols, cloud_cols = replay.columns()
    required = x_cols + local_cols + cloud_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{input_csv} missing columns: {missing}")

    numeric = df[required].apply(replay.pd.to_numeric, errors="coerce")
    finite = replay.np.isfinite(numeric.to_numpy()).all(axis=1)

    Xq = numeric.loc[finite, x_cols].to_numpy(float)
    scale = replay.np.array([q_scale] * 7 + [dq_scale] * 7, dtype=float)

    hist_pred = replay.np.full((len(df), 7), replay.np.nan)
    nearest = replay.np.full(len(df), replay.np.nan)
    mean_topk = replay.np.full(len(df), replay.np.nan)

    pred_clean, nearest_clean, mean_topk_clean = replay.knn_query(
        X_db,
        Y_db,
        Xq,
        k,
        scale,
        chunk_size,
    )

    finite_idx = replay.np.nonzero(finite)[0]
    hist_pred[finite_idx] = pred_clean
    nearest[finite_idx] = nearest_clean
    mean_topk[finite_idx] = mean_topk_clean

    is_online = "online" in input_name.lower() or "online" in str(input_csv).lower()
    available = replay.np.isfinite(nearest) & (nearest <= max_distance)
    if disable_historical_for_online and is_online:
        available[:] = False

    local = df[local_cols].apply(replay.pd.to_numeric, errors="coerce").to_numpy(float)
    cloud = df[cloud_cols].apply(replay.pd.to_numeric, errors="coerce").to_numpy(float)
    gated = replay.np.where(available[:, None], hist_pred, cloud)

    out = df.copy()
    for j in range(1, 8):
        out[f"hist_db_pred_{j}"] = hist_pred[:, j - 1]
        out[f"hist_db_gated_pred_{j}"] = gated[:, j - 1]

    out["hist_db_nearest_distance"] = nearest
    out["hist_db_mean_topk_distance"] = mean_topk
    out["hist_db_available"] = available.astype(int)
    out["hist_db_gated_source_code"] = replay.np.where(available, 2, 1)
    out["hist_db_k"] = k
    out["hist_db_max_distance"] = max_distance
    out["hist_db_online_disabled"] = int(disable_historical_for_online and is_online)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    return {
        "input_name": input_name,
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows": int(len(out)),
        "hist_available_count": int(available.sum()),
        "hist_available_ratio": float(available.mean()),
        "nearest_median": float(replay.np.nanmedian(nearest)),
        "nearest_max": float(replay.np.nanmax(nearest)),
        "online_detected": bool(is_online),
        "online_disabled": bool(disable_historical_for_online and is_online),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, path_text in DEFAULT_INPUTS.items():
        output_csv = output_dir / f"{name}_historical_shadow.csv"
        result = replay_one(
            db_path=Path(args.db),
            input_name=name,
            input_csv=Path(path_text),
            output_csv=output_csv,
            k=args.k,
            q_scale=args.q_scale,
            dq_scale=args.dq_scale,
            max_distance=args.max_distance,
            chunk_size=args.chunk_size,
            disable_historical_for_online=args.disable_historical_for_online,
        )
        results.append(result)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "offline_only": True,
        "active_compensation": False,
        "db": args.db,
        "output_dir": str(output_dir),
        "k": args.k,
        "q_scale": args.q_scale,
        "dq_scale": args.dq_scale,
        "max_distance": args.max_distance,
        "disable_historical_for_online": args.disable_historical_for_online,
        "results": results,
    }

    manifest_path = output_dir / "batch_shadow_replay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("===== GOAL1 historical DB batch shadow replay complete =====")
    for r in results:
        print(
            r["input_name"],
            "rows=", r["rows"],
            "available=", r["hist_available_count"],
            "ratio=", f'{r["hist_available_ratio"]:.6f}',
            "online_disabled=", r["online_disabled"],
        )
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
