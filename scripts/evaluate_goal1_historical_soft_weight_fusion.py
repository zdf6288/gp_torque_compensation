#!/usr/bin/env python3
"""Evaluate GOAL1 historical soft-weight fusion using existing offline data.

Offline-only:
- no ROS import
- no robot connection
- no controller or launch change
- no active historical compensation
- no tau_final change

This script queries a persistent historical residual DB with raw 14D Euclidean
distance, computes inverse-distance weighted KNN predictions, and evaluates a
distance-gated soft fusion of local, cloud, and historical predictions.
"""

from __future__ import annotations

import argparse
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_DB = "outputs/goal1_historical_residual_db_20260604/goal1_historical_residual_db.npz"
DEFAULT_CSVS = [
    "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_nogp_3000_20260603/cartesian_impedance_controller_data.csv",
    "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_local_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
    "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_cloud_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
    "/home/mirmi_ros2_2/gp_torque_data_backups/goal1_spatial_multisine_combined_scale10_clip05_3000_20260603/cartesian_impedance_controller_data.csv",
]
KNN_CHUNK_SIZE = 500
INVERSE_DISTANCE_EPS = 1e-12


def default_out_dir() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"outputs/goal1_historical_soft_weight_fusion_eval_{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline GOAL1 historical soft-weight fusion evaluator."
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="Historical residual DB .npz path.")
    parser.add_argument(
        "--csv",
        action="append",
        default=[],
        help="Controller CSV path. Can be repeated; defaults to four GOAL1 real CSVs.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: timestamped outputs/goal1_historical_soft_weight_fusion_eval_*.",
    )
    parser.add_argument("--k", type=int, default=5, help="Historical KNN neighbor count.")
    parser.add_argument(
        "--distance-thresholds",
        default="0.02,0.05,0.1,0.2,0.5",
        help="Comma-separated historical distance gates.",
    )
    parser.add_argument(
        "--alpha-values",
        default="0.1,0.2,0.5,1.0,2.0",
        help="Comma-separated exponential distance-decay alpha values.",
    )
    parser.add_argument(
        "--online-hist-scales",
        default="0,0.02,0.05,0.1,0.2,1.0",
        help="Comma-separated historical weight scales applied only to online runs.",
    )
    parser.add_argument(
        "--clip-nm",
        type=float,
        default=0.5,
        help="Offline analysis clip magnitude in Nm.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=3000,
        help="Maximum rows read from each CSV.",
    )
    parser.add_argument(
        "--row-sample-limit",
        type=int,
        default=1000,
        help="Maximum cleaned rows saved per CSV for its best soft-fusion combination.",
    )
    return parser.parse_args()


def parse_float_values(text: str, name: str, *, nonnegative: bool = True) -> List[float]:
    try:
        values = [float(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {text}") from exc

    if not values:
        raise ValueError(f"{name} requires at least one value.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} values must be finite: {text}")
    if nonnegative and any(value < 0.0 for value in values):
        raise ValueError(f"{name} values must be nonnegative: {text}")
    return values


def validate_args(args: argparse.Namespace) -> None:
    if args.k <= 0:
        raise ValueError(f"--k must be positive, got {args.k}")
    if not math.isfinite(args.clip_nm) or args.clip_nm <= 0.0:
        raise ValueError(f"--clip-nm must be finite and positive, got {args.clip_nm}")
    if args.max_rows <= 0:
        raise ValueError(f"--max-rows must be positive, got {args.max_rows}")
    if args.row_sample_limit < 0:
        raise ValueError(
            f"--row-sample-limit must be nonnegative, got {args.row_sample_limit}"
        )


def load_db(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)
    required = [
        "X",
        "Y_residual",
        "Y_local",
        "Y_cloud",
        "feature_names",
        "residual_names",
    ]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise RuntimeError(f"DB missing required arrays: {missing}")

    db = {key: data[key] for key in data.files}
    x = np.asarray(db["X"], dtype=float)
    y_residual = np.asarray(db["Y_residual"], dtype=float)
    y_local = np.asarray(db["Y_local"], dtype=float)
    y_cloud = np.asarray(db["Y_cloud"], dtype=float)
    feature_names = [str(value) for value in db["feature_names"].tolist()]
    residual_names = [str(value) for value in db["residual_names"].tolist()]

    if x.ndim != 2 or x.shape[1] != 14:
        raise RuntimeError(f"Expected DB X shape (N, 14), got {x.shape}")
    for name, array in [
        ("Y_residual", y_residual),
        ("Y_local", y_local),
        ("Y_cloud", y_cloud),
    ]:
        if array.ndim != 2 or array.shape != (len(x), 7):
            raise RuntimeError(f"Expected DB {name} shape ({len(x)}, 7), got {array.shape}")
    if len(feature_names) != x.shape[1] or len(set(feature_names)) != len(feature_names):
        raise RuntimeError(f"Invalid DB feature_names: {feature_names}")
    if len(residual_names) != y_residual.shape[1] or len(set(residual_names)) != len(
        residual_names
    ):
        raise RuntimeError(f"Invalid DB residual_names: {residual_names}")
    for name, array in [
        ("X", x),
        ("Y_residual", y_residual),
        ("Y_local", y_local),
        ("Y_cloud", y_cloud),
    ]:
        if not np.isfinite(array).all():
            raise RuntimeError(f"DB {name} contains non-finite values.")

    db["X"] = x
    db["Y_residual"] = y_residual
    db["Y_local"] = y_local
    db["Y_cloud"] = y_cloud
    db["feature_names"] = np.asarray(feature_names, dtype=object)
    db["residual_names"] = np.asarray(residual_names, dtype=object)
    return db


def prediction_columns(prefix: str, count: int = 7) -> List[str]:
    return [f"{prefix}_{joint}" for joint in range(1, count + 1)]


def load_csv(
    path: Path,
    feature_names: Sequence[str],
    residual_names: Sequence[str],
    max_rows: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path, nrows=max_rows)
    local_cols = prediction_columns("y_hat_local")
    cloud_cols = prediction_columns("y_hat_cloud")

    missing_features = [name for name in feature_names if name not in frame.columns]
    if missing_features:
        raise RuntimeError(f"{path} missing feature columns: {missing_features}")

    required_other = list(residual_names) + local_cols + cloud_cols
    missing_other = [name for name in required_other if name not in frame.columns]
    if missing_other:
        raise RuntimeError(f"{path} missing required prediction/target columns: {missing_other}")

    required = list(feature_names) + required_other
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy()).all(axis=1)
    cleaned = frame.loc[finite_mask].copy()
    cleaned.loc[:, required] = numeric.loc[finite_mask]
    cleaned.insert(0, "_source_row_index", frame.index[finite_mask].to_numpy())
    cleaned.reset_index(drop=True, inplace=True)

    if cleaned.empty:
        raise RuntimeError(f"{path} has no finite rows for required columns.")

    column_info: Dict[str, object] = {
        "csv_path": str(path),
        "rows_read": int(len(frame)),
        "rows_used": int(len(cleaned)),
        "rows_dropped_nonfinite": int(len(frame) - len(cleaned)),
        "feature_columns": list(feature_names),
        "local_columns": local_cols,
        "cloud_columns": cloud_cols,
        "target_columns": list(residual_names),
        "online_column_present": "gp_online_update_enabled" in frame.columns,
    }
    return cleaned, column_info


def detect_online_run(frame: pd.DataFrame, path: Path) -> Tuple[bool, str]:
    column = "gp_online_update_enabled"
    if column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        online = bool((finite_values > 0.5).any())
        reason = (
            f"{column} present; any finite used-row value > 0.5 is "
            f"{str(online).lower()} ({len(finite_values)} finite values)"
        )
        return online, reason

    path_text = str(path).lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", path_text) if token]
    matched = next(
        (
            marker
            for marker in ("online", "update_on", "ou")
            if marker in tokens or (marker != "ou" and marker in path_text)
        ),
        None,
    )
    online = matched is not None
    if online:
        return True, f"gp_online_update_enabled absent; path matched '{matched}'"
    return False, "gp_online_update_enabled absent; path did not match online/update_on/ou"


def require_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise RuntimeError(f"{name} contains non-finite values.")


def knn_inverse_distance_predict(
    x_db: np.ndarray,
    y_db: np.ndarray,
    x_query: np.ndarray,
    k: int,
    *,
    chunk_size: int = KNN_CHUNK_SIZE,
    epsilon: float = INVERSE_DISTANCE_EPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return raw-Euclidean inverse-distance weighted KNN prediction."""

    k_used = min(k, len(x_db))
    if k_used <= 0:
        raise RuntimeError("Historical DB is empty.")

    predictions = np.empty((len(x_query), y_db.shape[1]), dtype=float)
    nearest_distance = np.empty(len(x_query), dtype=float)
    db_norm_sq = np.einsum("ij,ij->i", x_db, x_db)

    for start in range(0, len(x_query), chunk_size):
        end = min(start + chunk_size, len(x_query))
        query_chunk = x_query[start:end]
        query_norm_sq = np.einsum("ij,ij->i", query_chunk, query_chunk)
        distance_sq = (
            query_norm_sq[:, None] + db_norm_sq[None, :] - 2.0 * query_chunk @ x_db.T
        )
        np.maximum(distance_sq, 0.0, out=distance_sq)

        indices = np.argpartition(distance_sq, kth=k_used - 1, axis=1)[:, :k_used]
        selected_sq = np.take_along_axis(distance_sq, indices, axis=1)
        order = np.argsort(selected_sq, axis=1)
        indices = np.take_along_axis(indices, order, axis=1)
        selected_distance = np.sqrt(np.take_along_axis(selected_sq, order, axis=1))

        weights = 1.0 / (selected_distance + epsilon)
        weights /= weights.sum(axis=1, keepdims=True)
        predictions[start:end] = np.einsum("nk,nkj->nj", weights, y_db[indices])
        nearest_distance[start:end] = selected_distance[:, 0]

    require_finite("historical KNN prediction", predictions)
    require_finite("historical nearest distance", nearest_distance)
    return predictions, nearest_distance


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def percentile(array: np.ndarray, value: float) -> float:
    return float(np.percentile(array, value))


def max_abs(array: np.ndarray) -> float:
    return float(np.max(np.abs(array)))


def clipped(array: np.ndarray, clip_nm: float) -> np.ndarray:
    result = np.clip(array, -clip_nm, clip_nm)
    require_finite("offline clipped prediction", result)
    return result


def compute_soft_fusion(
    local: np.ndarray,
    cloud: np.ndarray,
    historical: np.ndarray,
    nearest_distance: np.ndarray,
    *,
    distance_threshold: float,
    alpha: float,
    online_hist_scale: float,
    online_detected: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_w_hist = np.exp(-alpha * nearest_distance)
    raw_w_hist = np.where(nearest_distance <= distance_threshold, raw_w_hist, 0.0)
    if online_detected:
        raw_w_hist = raw_w_hist * online_hist_scale

    sum_w = 1.0 + raw_w_hist
    w_local = 0.5 / sum_w
    w_cloud = 0.5 / sum_w
    norm_w_hist = raw_w_hist / sum_w
    prediction = (
        w_local[:, None] * local
        + w_cloud[:, None] * cloud
        + norm_w_hist[:, None] * historical
    )

    require_finite("raw historical soft weight", raw_w_hist)
    require_finite("normalized historical soft weight", norm_w_hist)
    require_finite("soft fusion prediction", prediction)
    return prediction, raw_w_hist, norm_w_hist


def simple_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("" if math.isnan(float(value)) else f"{float(value):.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_row_sample(
    *,
    csv_path: Path,
    frame: pd.DataFrame,
    local: np.ndarray,
    cloud: np.ndarray,
    historical: np.ndarray,
    target: np.ndarray,
    nearest_distance: np.ndarray,
    best_row: pd.Series,
    clip_nm: float,
    row_sample_limit: int,
) -> pd.DataFrame:
    limit = min(row_sample_limit, len(frame))
    if limit == 0:
        return pd.DataFrame()

    soft, raw_w_hist, norm_w_hist = compute_soft_fusion(
        local,
        cloud,
        historical,
        nearest_distance,
        distance_threshold=float(best_row["distance_threshold"]),
        alpha=float(best_row["alpha"]),
        online_hist_scale=float(best_row["online_hist_scale"]),
        online_detected=bool(best_row["online_detected"]),
    )
    local_clip = clipped(local, clip_nm)
    cloud_clip = clipped(cloud, clip_nm)
    local_cloud = 0.5 * (local + cloud)
    local_cloud_clip = clipped(local_cloud, clip_nm)
    historical_clip = clipped(historical, clip_nm)
    soft_clip = clipped(soft, clip_nm)

    sample = pd.DataFrame(
        {
            "source_csv_basename": [csv_path.name] * limit,
            "csv_path": [str(csv_path)] * limit,
            "row_index": frame["_source_row_index"].to_numpy()[:limit],
            "distance_threshold": [float(best_row["distance_threshold"])] * limit,
            "alpha": [float(best_row["alpha"])] * limit,
            "online_hist_scale": [float(best_row["online_hist_scale"])] * limit,
            "online_detected": [bool(best_row["online_detected"])] * limit,
            "nearest_distance": nearest_distance[:limit],
            "raw_w_hist": raw_w_hist[:limit],
            "norm_w_hist": norm_w_hist[:limit],
        }
    )

    arrays = {
        "local_pred": local,
        "cloud_pred": cloud,
        "local_cloud_pred": local_cloud,
        "hist_knn_pred": historical,
        "soft_pred": soft,
        "local_pred_clipped": local_clip,
        "cloud_pred_clipped": cloud_clip,
        "local_cloud_pred_clipped": local_cloud_clip,
        "hist_knn_pred_clipped": historical_clip,
        "soft_pred_clipped": soft_clip,
        "target_tau_residual": target,
    }
    for prefix, array in arrays.items():
        for joint in range(array.shape[1]):
            sample[f"{prefix}_{joint + 1}"] = array[:limit, joint]
    return sample


def write_report(
    *,
    path: Path,
    db_path: Path,
    db: Dict[str, np.ndarray],
    column_info: List[Dict[str, object]],
    baseline_rows: List[Dict[str, object]],
    summary: pd.DataFrame,
    clip_nm: float,
) -> None:
    result_sort_columns = [
        "soft_fusion_rmse",
        "distance_threshold",
        "alpha",
        "online_hist_scale",
    ]
    best_per_csv = (
        summary.sort_values(["csv_path", *result_sort_columns], kind="mergesort")
        .groupby("csv_path", as_index=False)
        .first()
    )
    top_results = summary.sort_values(result_sort_columns, kind="mergesort").head(20).copy()
    for frame in (best_per_csv, top_results):
        frame["better_than_local_cloud"] = frame["soft_fusion_rmse"] < frame["local_cloud_rmse"]
        frame["better_than_historical"] = (
            frame["soft_fusion_rmse"] < frame["historical_knn_rmse"]
        )

    db_shapes = pd.DataFrame(
        [
            {
                "key": key,
                "shape": str(tuple(np.asarray(db[key]).shape)),
                "dtype": str(np.asarray(db[key]).dtype),
            }
            for key in db
        ]
    )
    recognition = pd.DataFrame(
        [
            {
                "csv_path": info["csv_path"],
                "rows_read": info["rows_read"],
                "rows_used": info["rows_used"],
                "rows_dropped_nonfinite": info["rows_dropped_nonfinite"],
                "feature_columns_found": len(info["feature_columns"]),
                "local_columns_found": len(info["local_columns"]),
                "cloud_columns_found": len(info["cloud_columns"]),
                "target_columns_found": len(info["target_columns"]),
                "online_column_present": info["online_column_present"],
            }
            for info in column_info
        ]
    )
    baseline = pd.DataFrame(baseline_rows)
    offline_csv_count = int((~baseline["online_detected"].astype(bool)).sum())
    total_csv_count = int(len(baseline))

    result_columns = [
        "csv_path",
        "distance_threshold",
        "alpha",
        "online_hist_scale",
        "online_detected",
        "soft_fusion_rmse",
        "local_cloud_rmse",
        "historical_knn_rmse",
        "delta_soft_vs_local_cloud",
        "delta_soft_vs_historical",
        "norm_w_hist_mean",
        "better_than_local_cloud",
        "better_than_historical",
    ]
    baseline_columns = [
        "csv_path",
        "online_detected",
        "online_detection_reason",
        "local_rmse",
        "cloud_rmse",
        "local_cloud_rmse",
        "historical_knn_rmse",
    ]

    lines = [
        "# GOAL1 Historical Soft-Weight Fusion Evaluation",
        "",
        "This is an offline-only residual prediction analysis.",
        "",
        "## Input DB",
        "",
        f"- db: `{db_path}`",
        f"- feature_names: `{list(db['feature_names'])}`",
        f"- residual_names: `{list(db['residual_names'])}`",
        "",
        simple_markdown_table(db_shapes),
        "",
        "## CSV Column Recognition",
        "",
        simple_markdown_table(recognition),
        "",
        "All query features were read from each CSV using the DB `feature_names`. "
        "Local/cloud baselines were read from CSV `y_hat_local_*` and `y_hat_cloud_*` columns.",
        "",
        "## Online Detection And Baseline RMSE",
        "",
        simple_markdown_table(baseline[baseline_columns]),
        "",
        "## Best Soft Fusion Per CSV",
        "",
        simple_markdown_table(best_per_csv[result_columns]),
        "",
        "## Top Soft Fusion Results",
        "",
        "Sorted deterministically by `soft_fusion_rmse`, `distance_threshold`, `alpha`, "
        "and `online_hist_scale`, all ascending.",
        "",
        simple_markdown_table(top_results[result_columns]),
        "",
        "## Interpretation caveats",
        "",
        "- This evaluator is offline-only and does not modify controller behavior.",
        "- Very low historical KNN RMSE, especially for the no-GP CSV, may reflect DB/CSV overlap because the persistent DB was built from related GOAL1 data.",
        "- The global best soft-fusion result should not be interpreted as held-out generalization.",
        "- These results mainly support that in-support historical residual retrieval / DB reuse can be predictive.",
        "- They do not prove real-robot active compensation stability, repeated robustness, or safety.",
        "- Active use would require separate shadow validation, distance/confidence gating, clip/scale preservation, and real-robot safety review.",
        "",
        "## Tie-break and online_hist_scale interpretation",
        "",
        "- Best rows use deterministic sorting by `soft_fusion_rmse`, `distance_threshold`, `alpha`, and `online_hist_scale`, all ascending.",
        "- When multiple parameter combinations have identical or nearly identical RMSE, the reported best parameters are a deterministic selection and should not be over-interpreted.",
        f"- This evaluation detected `{offline_csv_count}` of `{total_csv_count}` CSV files as offline (`online_detected=False`).",
        "- For rows where `online_detected=False`, `online_hist_scale` does not affect historical weights or RMSE.",
        "- Therefore, an offline best row with `online_hist_scale=0.0` is only a tie-break selection, not evidence that zero online historical scale is physically optimal.",
        "- Online historical scale requires separate evaluation on online CSV files or online-designated replay.",
        "",
        "## Clip Interpretation",
        "",
        f"- `clip_nm={clip_nm:.6g}` is an offline analysis clip only.",
        "- Raw RMSE columns remain the primary comparison; `_clipped` columns show the offline-clipped analysis version.",
        "- This script did not modify the controller clip.",
        "- This script did not enable active compensation.",
        "",
        "## Safety Notes",
        "",
        "- Offline-only.",
        "- No ROS import.",
        "- No controller change.",
        "- No launch change.",
        "- No active historical compensation.",
        "- No tau_final change.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    distance_thresholds = parse_float_values(
        args.distance_thresholds, "--distance-thresholds"
    )
    alpha_values = parse_float_values(args.alpha_values, "--alpha-values")
    online_hist_scales = parse_float_values(
        args.online_hist_scales, "--online-hist-scales"
    )

    db_path = Path(args.db)
    db = load_db(db_path)
    x_db = db["X"]
    y_db = db["Y_residual"]
    feature_names = [str(value) for value in db["feature_names"].tolist()]
    residual_names = [str(value) for value in db["residual_names"].tolist()]
    local_cols = prediction_columns("y_hat_local")
    cloud_cols = prediction_columns("y_hat_cloud")

    csv_paths = [Path(value) for value in (args.csv or DEFAULT_CSVS)]
    out_dir = Path(args.out_dir or default_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    baseline_rows: List[Dict[str, object]] = []
    column_info: List[Dict[str, object]] = []
    sample_inputs: List[Dict[str, object]] = []

    for csv_path in csv_paths:
        frame, info = load_csv(csv_path, feature_names, residual_names, args.max_rows)
        column_info.append(info)
        online_detected, online_reason = detect_online_run(frame, csv_path)

        query = frame[feature_names].to_numpy(dtype=float)
        target = frame[residual_names].to_numpy(dtype=float)
        local = frame[local_cols].to_numpy(dtype=float)
        cloud = frame[cloud_cols].to_numpy(dtype=float)
        local_cloud = 0.5 * (local + cloud)
        historical, nearest_distance = knn_inverse_distance_predict(
            x_db, y_db, query, args.k
        )

        for name, array in [
            ("query features", query),
            ("target residual", target),
            ("local prediction", local),
            ("cloud prediction", cloud),
            ("local/cloud prediction", local_cloud),
            ("historical prediction", historical),
        ]:
            require_finite(name, array)

        local_clip = clipped(local, args.clip_nm)
        cloud_clip = clipped(cloud, args.clip_nm)
        local_cloud_clip = clipped(local_cloud, args.clip_nm)
        historical_clip = clipped(historical, args.clip_nm)

        baseline = {
            "csv_path": str(csv_path),
            "online_detected": online_detected,
            "online_detection_reason": online_reason,
            "local_rmse": rmse(local, target),
            "cloud_rmse": rmse(cloud, target),
            "local_cloud_rmse": rmse(local_cloud, target),
            "historical_knn_rmse": rmse(historical, target),
            "local_clipped_rmse": rmse(local_clip, target),
            "cloud_clipped_rmse": rmse(cloud_clip, target),
            "local_cloud_clipped_rmse": rmse(local_cloud_clip, target),
            "historical_knn_clipped_rmse": rmse(historical_clip, target),
        }
        baseline_rows.append(baseline)

        nearest_stats = {
            "nearest_distance_mean": float(np.mean(nearest_distance)),
            "nearest_distance_p50": percentile(nearest_distance, 50),
            "nearest_distance_p95": percentile(nearest_distance, 95),
            "nearest_distance_max": float(np.max(nearest_distance)),
        }

        for distance_threshold in distance_thresholds:
            distance_pass = nearest_distance <= distance_threshold
            for alpha in alpha_values:
                for online_hist_scale in online_hist_scales:
                    soft, raw_w_hist, norm_w_hist = compute_soft_fusion(
                        local,
                        cloud,
                        historical,
                        nearest_distance,
                        distance_threshold=distance_threshold,
                        alpha=alpha,
                        online_hist_scale=online_hist_scale,
                        online_detected=online_detected,
                    )
                    soft_clip = clipped(soft, args.clip_nm)
                    soft_rmse = rmse(soft, target)
                    historical_rmse = float(baseline["historical_knn_rmse"])
                    local_cloud_rmse = float(baseline["local_cloud_rmse"])

                    row: Dict[str, object] = {
                        "csv_path": str(csv_path),
                        "db_path": str(db_path),
                        "rows_used": len(frame),
                        "db_rows": len(x_db),
                        "k": min(args.k, len(x_db)),
                        "distance_threshold": distance_threshold,
                        "alpha": alpha,
                        "online_hist_scale": online_hist_scale,
                        "online_detected": online_detected,
                        "online_detection_reason": online_reason,
                        **nearest_stats,
                        "hist_distance_pass_ratio": float(np.mean(distance_pass)),
                        "raw_w_hist_mean": float(np.mean(raw_w_hist)),
                        "raw_w_hist_p95": percentile(raw_w_hist, 95),
                        "norm_w_hist_mean": float(np.mean(norm_w_hist)),
                        "norm_w_hist_p95": percentile(norm_w_hist, 95),
                        "local_rmse": baseline["local_rmse"],
                        "cloud_rmse": baseline["cloud_rmse"],
                        "local_cloud_rmse": local_cloud_rmse,
                        "historical_knn_rmse": historical_rmse,
                        "soft_fusion_rmse": soft_rmse,
                        "delta_soft_vs_local_cloud": soft_rmse - local_cloud_rmse,
                        "delta_soft_vs_historical": soft_rmse - historical_rmse,
                        "max_abs_local": max_abs(local),
                        "max_abs_cloud": max_abs(cloud),
                        "max_abs_hist_knn": max_abs(historical),
                        "max_abs_soft_fusion": max_abs(soft),
                        "clip_nm": args.clip_nm,
                        "local_clipped_rmse": baseline["local_clipped_rmse"],
                        "cloud_clipped_rmse": baseline["cloud_clipped_rmse"],
                        "local_cloud_clipped_rmse": baseline["local_cloud_clipped_rmse"],
                        "historical_knn_clipped_rmse": baseline[
                            "historical_knn_clipped_rmse"
                        ],
                        "soft_fusion_clipped_rmse": rmse(soft_clip, target),
                        "max_abs_local_clipped": max_abs(local_clip),
                        "max_abs_cloud_clipped": max_abs(cloud_clip),
                        "max_abs_local_cloud_clipped": max_abs(local_cloud_clip),
                        "max_abs_hist_knn_clipped": max_abs(historical_clip),
                        "max_abs_soft_fusion_clipped": max_abs(soft_clip),
                    }
                    summary_rows.append(row)

        sample_inputs.append(
            {
                "csv_path": csv_path,
                "frame": frame,
                "local": local,
                "cloud": cloud,
                "historical": historical,
                "target": target,
                "nearest_distance": nearest_distance,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.sort_values(
        ["csv_path", "soft_fusion_rmse", "distance_threshold", "alpha", "online_hist_scale"],
        inplace=True,
    )
    summary.reset_index(drop=True, inplace=True)

    row_samples = []
    for sample_input in sample_inputs:
        csv_path = sample_input["csv_path"]
        best_row = summary.loc[summary["csv_path"] == str(csv_path)].iloc[0]
        row_samples.append(
            make_row_sample(
                **sample_input,
                best_row=best_row,
                clip_nm=args.clip_nm,
                row_sample_limit=args.row_sample_limit,
            )
        )
    row_sample = pd.concat(row_samples, ignore_index=True) if row_samples else pd.DataFrame()

    summary_path = out_dir / "goal1_historical_soft_weight_fusion_summary.csv"
    report_path = out_dir / "goal1_historical_soft_weight_fusion_report.md"
    row_sample_path = out_dir / "goal1_historical_soft_weight_fusion_row_sample.csv"
    summary.to_csv(summary_path, index=False)
    row_sample.to_csv(row_sample_path, index=False)
    write_report(
        path=report_path,
        db_path=db_path,
        db=db,
        column_info=column_info,
        baseline_rows=baseline_rows,
        summary=summary,
        clip_nm=args.clip_nm,
    )

    best = summary.sort_values(
        ["soft_fusion_rmse", "distance_threshold", "alpha", "online_hist_scale"],
        kind="mergesort",
    ).iloc[0]
    print("===== GOAL1 historical soft-weight fusion offline evaluation =====")
    print("db:", db_path)
    print("csv_count:", len(csv_paths))
    print("db_rows:", len(x_db))
    print("summary_rows:", len(summary))
    print("offline_analysis_clip_nm:", args.clip_nm)
    print()
    print("===== best raw soft fusion result =====")
    print(
        best[
            [
                "csv_path",
                "soft_fusion_rmse",
                "local_cloud_rmse",
                "historical_knn_rmse",
                "distance_threshold",
                "alpha",
                "online_hist_scale",
                "online_detected",
            ]
        ].to_string()
    )
    print()
    print("outputs:")
    for output_path in [summary_path, report_path, row_sample_path]:
        print(output_path, output_path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
