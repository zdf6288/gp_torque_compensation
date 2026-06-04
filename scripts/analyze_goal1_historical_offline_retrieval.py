#!/usr/bin/env python3
"""GOAL1 historical retrieval offline analysis.

This script is offline-only. It reads existing controller CSV files and tests
whether a simple historical nearest-neighbor residual database can predict
held-out residual torque better than local/cloud GP predictions.

It does not run ROS, does not connect to a robot, and does not modify any
controller/torque path.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_TRAIN_FILES: Dict[str, str] = {
    "nogp_20260603": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_nogp_3000_20260603/cartesian_impedance_controller_data.csv",
    "nogp_repeat_20260603": "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_nogp_repeat_end_3000_20260603/cartesian_impedance_controller_data.csv",
}

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
        description="Offline GOAL1 historical KNN residual retrieval analysis."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/goal1_historical_offline_retrieval_probe",
        help="Directory for summary CSV/Markdown outputs.",
    )
    parser.add_argument(
        "--k",
        default="1,3,5,10,25,50",
        help="Comma-separated K values for historical KNN.",
    )
    parser.add_argument(
        "--q-scale",
        type=float,
        default=0.1,
        help="Manual scale for joint position features in rad.",
    )
    parser.add_argument(
        "--dq-scale",
        type=float,
        default=0.1,
        help="Manual scale for joint velocity features in rad/s.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help="Chunk size for distance computation.",
    )
    return parser.parse_args()


def parse_k_values(text: str) -> List[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"k must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one k value is required.")
    return values


def required_columns() -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    q_cols = [f"joint_pos_{j}" for j in range(1, 8)]
    dq_cols = [f"joint_vel_{j}" for j in range(1, 8)]
    x_cols = q_cols + dq_cols
    residual_cols = [f"tau_residual_{j}" for j in range(1, 8)]
    local_cols = [f"y_hat_local_{j}" for j in range(1, 8)]
    cloud_cols = [f"y_hat_cloud_{j}" for j in range(1, 8)]
    return x_cols, residual_cols, local_cols, cloud_cols, q_cols + dq_cols + residual_cols + local_cols + cloud_cols


def load_clean_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}")

    numeric = df[list(required)].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy()).all(axis=1)
    return df.loc[finite_mask].reset_index(drop=True)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(((pred - target) ** 2).mean()))


def joint_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(((pred - target) ** 2).mean(axis=0))


def knn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    k: int,
    scale: np.ndarray,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return KNN residual prediction and distance diagnostics.

    这里使用固定物理尺度，而不是 train std 的 z-score。
    原因是某些 joint 几乎不动，例如 q7 的 std 非常小，z-score 会把微小偏移放大成异常大距离。
    """

    xtr = x_train / scale
    xte = x_test / scale

    pred = np.zeros((len(xte), y_train.shape[1]))
    nearest = np.zeros(len(xte))
    mean_topk = np.zeros(len(xte))

    for start in range(0, len(xte), chunk_size):
        end = min(start + chunk_size, len(xte))
        distances = ((xte[start:end, None, :] - xtr[None, :, :]) ** 2).sum(axis=2)

        kk = min(k, distances.shape[1])
        idx = np.argpartition(distances, kth=kk - 1, axis=1)[:, :kk]
        selected_distances = np.take_along_axis(distances, idx, axis=1)

        order = np.argsort(selected_distances, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        selected_distances = np.take_along_axis(selected_distances, order, axis=1)

        pred[start:end] = y_train[idx].mean(axis=1)
        nearest[start:end] = np.sqrt(selected_distances[:, 0])
        mean_topk[start:end] = np.sqrt(selected_distances).mean(axis=1)

    return pred, nearest, mean_topk


def append_prediction_row(
    rows: List[dict],
    *,
    train_pool: str,
    test_run: str,
    method: str,
    k_value: str | int,
    rows_train: int,
    rows_test: int,
    pred: np.ndarray,
    target: np.ndarray,
    nearest: np.ndarray | None = None,
    mean_topk: np.ndarray | None = None,
) -> None:
    jr = joint_rmse(pred, target)
    row = {
        "train_pool": train_pool,
        "test_run": test_run,
        "method": method,
        "k": k_value,
        "rows_train": rows_train,
        "rows_test": rows_test,
        "overall_rmse_tau": rmse(pred, target),
        "nearest_median": "" if nearest is None else float(np.median(nearest)),
        "nearest_max": "" if nearest is None else float(np.max(nearest)),
        "mean_topk_median": "" if mean_topk is None else float(np.median(mean_topk)),
    }
    for j, value in enumerate(jr, 1):
        row[f"rmse_j{j}"] = float(value)
    rows.append(row)



def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table without optional tabulate dependency."""
    if df.empty:
        return "_empty_"

    columns = [str(c) for c in df.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in df.iterrows():
        values = []
        for c in df.columns:
            value = row[c]
            if isinstance(value, float):
                if math.isnan(value):
                    values.append("")
                else:
                    values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows)


def write_markdown_report(summary: pd.DataFrame, output_path: Path) -> None:
    best = (
        summary.sort_values("overall_rmse_tau")
        .groupby(["train_pool", "test_run"], as_index=False)
        .first()
    )

    compact = summary[
        (summary["method"].isin(["local", "cloud", "local_cloud_equal"]))
        | (
            (summary["method"].isin(["historical_raw_knn", "local_cloud_hist_equal"]))
            & (summary["k"].astype(str) == "25")
        )
    ].copy()
    compact = compact.sort_values(["train_pool", "test_run", "overall_rmse_tau"])

    best_cols = [
        "train_pool",
        "test_run",
        "method",
        "k",
        "overall_rmse_tau",
        "nearest_median",
        "mean_topk_median",
    ]
    compact_cols = [
        "train_pool",
        "test_run",
        "method",
        "k",
        "overall_rmse_tau",
        "nearest_median",
    ]

    lines = []
    lines.append("# GOAL1 Historical Offline Retrieval Probe")
    lines.append("")
    lines.append("This report is offline-only. It does not imply active torque compensation.")
    lines.append("")
    lines.append("## Best method per train/test")
    lines.append("")
    lines.append(dataframe_to_markdown(best[best_cols]))
    lines.append("")
    lines.append("## Compact comparison at k=25")
    lines.append("")
    lines.append(dataframe_to_markdown(compact[compact_cols]))
    lines.append("")
    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- `historical_raw_knn` is an offline residual retrieval baseline, not an active controller.")
    lines.append("- Lower RMSE suggests historical retrieval may be useful under similar trajectory/state distributions.")
    lines.append("- If online-update runs prefer local/cloud over historical, historical should be gated rather than always fused.")
    lines.append("- Active historical compensation remains future work and requires separate safety review.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    args = parse_args()
    k_values = parse_k_values(args.k)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_cols, residual_cols, local_cols, cloud_cols, required = required_columns()
    feature_scale = np.array([args.q_scale] * 7 + [args.dq_scale] * 7, dtype=float)

    rows: List[dict] = []

    train_data = {
        name: load_clean_csv(Path(path), required)
        for name, path in DEFAULT_TRAIN_FILES.items()
    }
    test_data = {
        name: load_clean_csv(Path(path), required)
        for name, path in DEFAULT_TEST_FILES.items()
    }

    for train_name, train_df in train_data.items():
        x_train = train_df[x_cols].to_numpy(float)
        y_train = train_df[residual_cols].to_numpy(float)

        for test_name, test_df in test_data.items():
            x_test = test_df[x_cols].to_numpy(float)
            y_test = test_df[residual_cols].to_numpy(float)
            y_local = test_df[local_cols].to_numpy(float)
            y_cloud = test_df[cloud_cols].to_numpy(float)

            append_prediction_row(
                rows,
                train_pool=train_name,
                test_run=test_name,
                method="local",
                k_value="",
                rows_train=len(train_df),
                rows_test=len(test_df),
                pred=y_local,
                target=y_test,
            )
            append_prediction_row(
                rows,
                train_pool=train_name,
                test_run=test_name,
                method="cloud",
                k_value="",
                rows_train=len(train_df),
                rows_test=len(test_df),
                pred=y_cloud,
                target=y_test,
            )
            append_prediction_row(
                rows,
                train_pool=train_name,
                test_run=test_name,
                method="local_cloud_equal",
                k_value="",
                rows_train=len(train_df),
                rows_test=len(test_df),
                pred=0.5 * (y_local + y_cloud),
                target=y_test,
            )

            for k in k_values:
                hist_pred, nearest, mean_topk = knn_predict(
                    x_train,
                    y_train,
                    x_test,
                    k,
                    feature_scale,
                    args.chunk_size,
                )

                append_prediction_row(
                    rows,
                    train_pool=train_name,
                    test_run=test_name,
                    method="historical_raw_knn",
                    k_value=k,
                    rows_train=len(train_df),
                    rows_test=len(test_df),
                    pred=hist_pred,
                    target=y_test,
                    nearest=nearest,
                    mean_topk=mean_topk,
                )
                append_prediction_row(
                    rows,
                    train_pool=train_name,
                    test_run=test_name,
                    method="local_cloud_hist_equal",
                    k_value=k,
                    rows_train=len(train_df),
                    rows_test=len(test_df),
                    pred=(y_local + y_cloud + hist_pred) / 3.0,
                    target=y_test,
                    nearest=nearest,
                    mean_topk=mean_topk,
                )

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "historical_raw_weighted_knn_matrix_summary.csv"
    report_path = output_dir / "historical_raw_weighted_knn_matrix_report.md"

    summary.to_csv(summary_path, index=False)
    write_markdown_report(summary, report_path)

    best = (
        summary.sort_values("overall_rmse_tau")
        .groupby(["train_pool", "test_run"], as_index=False)
        .first()
    )

    print("===== best method per train/test =====")
    print(best[
        [
            "train_pool",
            "test_run",
            "method",
            "k",
            "overall_rmse_tau",
            "nearest_median",
            "mean_topk_median",
        ]
    ].to_string(index=False))

    print("\n===== outputs =====")
    for path in [summary_path, report_path]:
        print(path, path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
