#!/usr/bin/env python3
"""Evaluate a persistent GOAL1 historical residual database on held-out CSV files.

Offline-only:
- no ROS
- no robot
- no controller modification
- no active compensation
- no tau_final modification

This script explicitly loads a prebuilt .npz historical residual database and
queries it with [joint_pos_1..7, joint_vel_1..7] from held-out controller CSVs.
It compares:
- local GP prediction
- cloud GP prediction
- equal local/cloud fusion
- historical DB KNN residual retrieval
- equal local/cloud/historical fusion

The output is an offline feasibility report only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_DB = "outputs/goal1_historical_residual_db_20260604/goal1_historical_residual_db.npz"

DEFAULT_TEST_FILES: Dict[str, str] = {
    "nogp_20260603": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_nogp_3000_20260603/cartesian_impedance_controller_data.csv",
    "nogp_repeat_20260603": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_nogp_repeat_end_3000_20260603/cartesian_impedance_controller_data.csv",
    "local_scale10": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_local_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
    "cloud_scale10": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_cloud_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
    "combined_scale10": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_combined_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
    "online_local_scale10": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_online_local_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
    "online_cloud_scale10": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_online_cloud_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
    "online_combined_scale10": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_online_combined_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline evaluation of a GOAL1 historical residual database."
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to historical residual DB .npz.")
    parser.add_argument(
        "--output-dir",
        default="outputs/goal1_historical_residual_db_eval_20260604",
        help="Output directory.",
    )
    parser.add_argument("--k", default="1,3,5,10,25,50", help="Comma-separated K values.")
    parser.add_argument("--q-scale", type=float, default=0.1, help="Joint position scale in rad.")
    parser.add_argument("--dq-scale", type=float, default=0.1, help="Joint velocity scale in rad/s.")
    parser.add_argument("--chunk-size", type=int, default=250, help="KNN chunk size.")
    parser.add_argument(
        "--test",
        action="append",
        default=[],
        help="Optional test input in name=path form. Can be repeated.",
    )
    return parser.parse_args()


def parse_k_values(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"Invalid --k: {text}")
    return values


def parse_named_paths(items: List[str], default: Dict[str, str]) -> Dict[str, str]:
    if not items:
        return dict(default)

    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected name=path, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid name=path item: {item}")
        out[name] = path
    return out


def column_sets() -> Tuple[List[str], List[str], List[str], List[str]]:
    x_cols = [f"joint_pos_{j}" for j in range(1, 8)] + [f"joint_vel_{j}" for j in range(1, 8)]
    residual_cols = [f"tau_residual_{j}" for j in range(1, 8)]
    local_cols = [f"y_hat_local_{j}" for j in range(1, 8)]
    cloud_cols = [f"y_hat_cloud_{j}" for j in range(1, 8)]
    return x_cols, residual_cols, local_cols, cloud_cols


def load_db(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)
    required = ["X", "Y_residual", "Y_local", "Y_cloud", "source_run", "source_row"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise RuntimeError(f"DB missing arrays: {missing}")

    db = {k: data[k] for k in data.files}
    if db["X"].shape[1] != 14:
        raise RuntimeError(f"Expected DB feature_dim 14, got {db['X'].shape}")
    if db["Y_residual"].shape[1] != 7:
        raise RuntimeError(f"Expected DB target_dim 7, got {db['Y_residual'].shape}")
    if len(db["X"]) != len(db["Y_residual"]):
        raise RuntimeError("DB X/Y length mismatch")

    return db


def load_test_csv(path: Path, required_cols: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path} missing columns: {missing}")

    numeric = df[list(required_cols)].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(numeric.to_numpy()).all(axis=1)
    return df.loc[mask].reset_index(drop=True)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(((pred - target) ** 2).mean()))


def joint_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(((pred - target) ** 2).mean(axis=0))


def knn_query(
    x_db: np.ndarray,
    y_db: np.ndarray,
    x_query: np.ndarray,
    k: int,
    scale: np.ndarray,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xdb = x_db / scale
    xq = x_query / scale

    pred = np.zeros((len(xq), y_db.shape[1]))
    nearest = np.zeros(len(xq))
    mean_topk = np.zeros(len(xq))

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
        mean_topk[start:end] = np.sqrt(dsel).mean(axis=1)

    return pred, nearest, mean_topk


def add_row(
    rows: List[dict],
    *,
    test_run: str,
    method: str,
    k: str | int,
    pred: np.ndarray,
    target: np.ndarray,
    rows_db: int,
    rows_test: int,
    nearest: np.ndarray | None = None,
    mean_topk: np.ndarray | None = None,
) -> None:
    jr = joint_rmse(pred, target)
    row = {
        "test_run": test_run,
        "method": method,
        "k": k,
        "rows_db": rows_db,
        "rows_test": rows_test,
        "overall_rmse_tau": rmse(pred, target),
        "nearest_median": "" if nearest is None else float(np.median(nearest)),
        "nearest_max": "" if nearest is None else float(np.max(nearest)),
        "mean_topk_median": "" if mean_topk is None else float(np.median(mean_topk)),
    }
    for j, value in enumerate(jr, 1):
        row[f"rmse_j{j}"] = float(value)
    rows.append(row)


def simple_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append("" if np.isnan(v) else f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, path: Path, db_path: Path, rows_db: int) -> None:
    best = summary.sort_values("overall_rmse_tau").groupby("test_run", as_index=False).first()

    compact = summary[
        summary["method"].isin(["local", "cloud", "local_cloud_equal"])
        | ((summary["method"].isin(["historical_db_knn", "local_cloud_hist_equal"])) & (summary["k"].astype(str) == "25"))
    ].copy()
    compact = compact.sort_values(["test_run", "overall_rmse_tau"])

    lines = [
        "# GOAL1 Historical Residual DB Evaluation",
        "",
        "Offline-only evaluation. This does not enable active compensation.",
        "",
        f"- db: `{db_path}`",
        f"- rows_db: `{rows_db}`",
        "",
        "## Best method per test run",
        "",
        simple_markdown_table(best[["test_run", "method", "k", "overall_rmse_tau", "nearest_median", "mean_topk_median"]]),
        "",
        "## Compact comparison at k=25",
        "",
        simple_markdown_table(compact[["test_run", "method", "k", "overall_rmse_tau", "nearest_median"]]),
        "",
        "## Safety notes",
        "",
        "- This script reads CSV/NPZ files only.",
        "- This script does not import rclpy.",
        "- This script does not run ROS launch.",
        "- This script does not connect to a robot.",
        "- This script does not modify tau_final or any controller code.",
        "- Active historical compensation remains future work.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    k_values = parse_k_values(args.k)

    x_cols, residual_cols, local_cols, cloud_cols = column_sets()
    required = x_cols + residual_cols + local_cols + cloud_cols

    db_path = Path(args.db)
    db = load_db(db_path)

    x_db = db["X"].astype(float)
    y_db = db["Y_residual"].astype(float)

    scale = np.array([args.q_scale] * 7 + [args.dq_scale] * 7, dtype=float)

    test_files = parse_named_paths(args.test, DEFAULT_TEST_FILES)
    rows: List[dict] = []

    for test_name, test_path_text in test_files.items():
        df = load_test_csv(Path(test_path_text), required)

        x_test = df[x_cols].to_numpy(float)
        y_test = df[residual_cols].to_numpy(float)
        y_local = df[local_cols].to_numpy(float)
        y_cloud = df[cloud_cols].to_numpy(float)

        add_row(
            rows,
            test_run=test_name,
            method="local",
            k="",
            pred=y_local,
            target=y_test,
            rows_db=len(x_db),
            rows_test=len(df),
        )
        add_row(
            rows,
            test_run=test_name,
            method="cloud",
            k="",
            pred=y_cloud,
            target=y_test,
            rows_db=len(x_db),
            rows_test=len(df),
        )
        add_row(
            rows,
            test_run=test_name,
            method="local_cloud_equal",
            k="",
            pred=0.5 * (y_local + y_cloud),
            target=y_test,
            rows_db=len(x_db),
            rows_test=len(df),
        )

        for k in k_values:
            hist, nearest, mean_topk = knn_query(
                x_db,
                y_db,
                x_test,
                k,
                scale,
                args.chunk_size,
            )
            add_row(
                rows,
                test_run=test_name,
                method="historical_db_knn",
                k=k,
                pred=hist,
                target=y_test,
                rows_db=len(x_db),
                rows_test=len(df),
                nearest=nearest,
                mean_topk=mean_topk,
            )
            add_row(
                rows,
                test_run=test_name,
                method="local_cloud_hist_equal",
                k=k,
                pred=(y_local + y_cloud + hist) / 3.0,
                target=y_test,
                rows_db=len(x_db),
                rows_test=len(df),
                nearest=nearest,
                mean_topk=mean_topk,
            )

    summary = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "historical_residual_db_eval_summary.csv"
    report_path = output_dir / "historical_residual_db_eval_report.md"

    summary.to_csv(summary_path, index=False)
    write_report(summary, report_path, db_path, len(x_db))

    best = summary.sort_values("overall_rmse_tau").groupby("test_run", as_index=False).first()

    print("===== historical residual DB evaluation =====")
    print("db:", db_path)
    print("rows_db:", len(x_db))
    print()
    print("===== best method per test run =====")
    print(best[["test_run", "method", "k", "overall_rmse_tau", "nearest_median", "mean_topk_median"]].to_string(index=False))
    print()
    print("outputs:")
    print(summary_path, summary_path.stat().st_size, "bytes")
    print(report_path, report_path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
