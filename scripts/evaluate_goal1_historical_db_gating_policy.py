#!/usr/bin/env python3
"""Evaluate simple gating policies for GOAL1 historical residual DB.

Offline-only:
- no ROS
- no robot
- no controller modification
- no active compensation
- no tau_final modification

This script tests distance-threshold and mode-aware gating policies:
- use historical DB KNN only when nearest distance is below threshold
- otherwise fallback to cloud or local_cloud_equal
- optionally disable historical for online-update test runs

The goal is to identify safe offline gating directions before any controller work.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_DB = "outputs/goal1_historical_residual_db_20260604/goal1_historical_residual_db.npz"

DEFAULT_TEST_FILES: Dict[str, str] = {
    "local_scale10": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_local_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
    "cloud_scale10": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_cloud_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
    "combined_scale10": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_combined_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
    "online_local_scale10": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_online_local_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
    "online_cloud_scale10": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_online_cloud_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
    "online_combined_scale10": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_online_combined_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline GOAL1 historical DB gating policy evaluator.")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--output-dir", default="outputs/goal1_historical_db_gating_policy_20260604")
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--q-scale", type=float, default=0.1)
    p.add_argument("--dq-scale", type=float, default=0.1)
    p.add_argument("--thresholds", default="0.1,0.2,0.3,0.5,0.8,1.0,1.5,2.0,3.0")
    p.add_argument("--chunk-size", type=int, default=250)
    return p.parse_args()


def cols() -> Tuple[List[str], List[str], List[str], List[str]]:
    x = [f"joint_pos_{j}" for j in range(1, 8)] + [f"joint_vel_{j}" for j in range(1, 8)]
    r = [f"tau_residual_{j}" for j in range(1, 8)]
    local = [f"y_hat_local_{j}" for j in range(1, 8)]
    cloud = [f"y_hat_cloud_{j}" for j in range(1, 8)]
    return x, r, local, cloud


def load_test(path: Path, required: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path} missing columns: {missing}")
    numeric = df[list(required)].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(numeric.to_numpy()).all(axis=1)
    return df.loc[mask].reset_index(drop=True)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(((pred - target) ** 2).mean()))


def knn_query(
    x_db: np.ndarray,
    y_db: np.ndarray,
    x_query: np.ndarray,
    k: int,
    scale: np.ndarray,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xdb = x_db / scale
    xq = x_query / scale
    pred = np.zeros((len(xq), y_db.shape[1]))
    nearest = np.zeros(len(xq))

    for start in range(0, len(xq), chunk_size):
        end = min(start + chunk_size, len(xq))
        d = ((xq[start:end, None, :] - xdb[None, :, :]) ** 2).sum(axis=2)
        kk = min(k, d.shape[1])
        idx = np.argpartition(d, kth=kk - 1, axis=1)[:, :kk]
        dsel = np.take_along_axis(d, idx, axis=1)
        order = np.argsort(dsel, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        dsel = np.take_along_axis(dsel, order, axis=1)
        pred[start:end] = y_db[idx].mean(axis=1)
        nearest[start:end] = np.sqrt(dsel[:, 0])

    return pred, nearest


def write_md(summary: pd.DataFrame, path: Path) -> None:
    best = summary.sort_values("overall_rmse_tau").groupby("test_run", as_index=False).first()
    lines = [
        "# GOAL1 Historical DB Gating Policy Evaluation",
        "",
        "Offline-only. This does not modify controller behavior.",
        "",
        "## Best policy per test run",
        "",
        "| test_run | policy | threshold | overall_rmse_tau | hist_used_ratio |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for _, r in best.iterrows():
        lines.append(
            f"| {r['test_run']} | {r['policy']} | {r['threshold']} | "
            f"{r['overall_rmse_tau']:.6g} | {r['hist_used_ratio']:.6g} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- `mode_aware_cloud_else_hist` disables historical for `online_*` runs.",
        "- This is only an offline feasibility test.",
        "- Active historical compensation remains future work.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    db = np.load(args.db, allow_pickle=True)
    x_db = db["X"].astype(float)
    y_db = db["Y_residual"].astype(float)

    x_cols, res_cols, local_cols, cloud_cols = cols()
    required = x_cols + res_cols + local_cols + cloud_cols
    scale = np.array([args.q_scale] * 7 + [args.dq_scale] * 7, dtype=float)

    rows = []

    for test_name, path in DEFAULT_TEST_FILES.items():
        df = load_test(Path(path), required)
        x = df[x_cols].to_numpy(float)
        y = df[res_cols].to_numpy(float)
        local = df[local_cols].to_numpy(float)
        cloud = df[cloud_cols].to_numpy(float)
        local_cloud = 0.5 * (local + cloud)

        hist, nearest = knn_query(x_db, y_db, x, args.k, scale, args.chunk_size)

        base = {
            "local": local,
            "cloud": cloud,
            "local_cloud_equal": local_cloud,
            "historical_always": hist,
        }
        for policy, pred in base.items():
            rows.append({
                "test_run": test_name,
                "policy": policy,
                "threshold": "",
                "overall_rmse_tau": rmse(pred, y),
                "hist_used_ratio": 1.0 if policy == "historical_always" else 0.0,
                "nearest_median": float(np.median(nearest)),
            })

        for th in thresholds:
            use_hist = nearest <= th
            pred_cloud = np.where(use_hist[:, None], hist, cloud)
            pred_lc = np.where(use_hist[:, None], hist, local_cloud)

            rows.append({
                "test_run": test_name,
                "policy": "hist_if_close_else_cloud",
                "threshold": th,
                "overall_rmse_tau": rmse(pred_cloud, y),
                "hist_used_ratio": float(use_hist.mean()),
                "nearest_median": float(np.median(nearest)),
            })
            rows.append({
                "test_run": test_name,
                "policy": "hist_if_close_else_local_cloud",
                "threshold": th,
                "overall_rmse_tau": rmse(pred_lc, y),
                "hist_used_ratio": float(use_hist.mean()),
                "nearest_median": float(np.median(nearest)),
            })

            if test_name.startswith("online_"):
                pred_mode = cloud
                used = np.zeros_like(use_hist, dtype=bool)
            else:
                pred_mode = np.where(use_hist[:, None], hist, cloud)
                used = use_hist

            rows.append({
                "test_run": test_name,
                "policy": "mode_aware_cloud_else_hist",
                "threshold": th,
                "overall_rmse_tau": rmse(pred_mode, y),
                "hist_used_ratio": float(used.mean()),
                "nearest_median": float(np.median(nearest)),
            })

    summary = pd.DataFrame(rows)
    summary_path = outdir / "historical_db_gating_policy_summary.csv"
    report_path = outdir / "historical_db_gating_policy_report.md"
    summary.to_csv(summary_path, index=False)
    write_md(summary, report_path)

    best = summary.sort_values("overall_rmse_tau").groupby("test_run", as_index=False).first()
    print("===== best policy per test run =====")
    print(best[["test_run", "policy", "threshold", "overall_rmse_tau", "hist_used_ratio", "nearest_median"]].to_string(index=False))
    print()
    print("outputs:")
    print(summary_path, summary_path.stat().st_size, "bytes")
    print(report_path, report_path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
