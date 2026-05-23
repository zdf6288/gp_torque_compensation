#!/usr/bin/env python3
"""Validate Stage 4 frozen GP model support against formal CSV inputs.

This script is offline-only. It loads restored frozen local GP pickles, checks
whether formal runtime inputs are covered by the stored model support, compares
formal CSV feature distribution with the training CSV, and probes whether model
predictions vary on formal samples versus stored training-support samples.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable


try:
    import numpy as np
except ModuleNotFoundError as exc:
    print("Missing Python dependency: numpy", file=sys.stderr)
    print("Use an environment that already has project dependencies installed.", file=sys.stderr)
    raise SystemExit(1) from exc


JOINTS = range(1, 8)
NEAR_ZERO_XS_EPS = 1e-5
OUTSIDE_MARGIN = 5.0
SEVERE_STD_ABS = 100.0
PREDICTION_DIFF_EPS = 1e-9
PREDICTION_SPAN_EPS = 1e-9
PREDICTION_STD_EPS = 1e-9
DEFAULT_OUT_DIR = Path("outputs/stage4_gp_model_validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline validator for Stage 4 restored frozen local GP model support.",
    )
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing joint*_local.pkl files.")
    parser.add_argument("--formal-csv", type=Path, required=True, help="Formal Stage 4 CSV to validate.")
    parser.add_argument("--train-csv", type=Path, required=True, help="Training CSV used for this model family.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help=f"Default: {DEFAULT_OUT_DIR}")
    parser.add_argument("--mode-name", required=True, help="Mode label written into reports.")
    parser.add_argument("--sample-a", type=int, default=0, help="Formal sample index A. Negative indexing allowed.")
    parser.add_argument("--sample-b", type=int, default=-1, help="Formal sample index B. Negative indexing allowed.")
    parser.add_argument(
        "--max-prediction-rows",
        type=int,
        default=0,
        help="Max formal rows for full-run prediction span check. 0 means all rows.",
    )
    parser.add_argument(
        "--gp-online-update-enabled",
        type=parse_optional_bool,
        default=None,
        help="Optional runtime parameter value used for frozen formal safety preflight.",
    )
    parser.add_argument(
        "--gp-compensation-scale",
        type=float,
        default=None,
        help="Optional runtime parameter value used for frozen formal safety preflight.",
    )
    parser.add_argument(
        "--gp-compensation-clip-nm",
        type=float,
        default=None,
        help="Optional runtime parameter value used for frozen formal safety preflight.",
    )
    parser.add_argument(
        "--feature-source",
        choices=("joint_vel", "dq_des_joint"),
        default="joint_vel",
        help="Velocity feature source for X = joint_pos_1..7 + velocity_1..7. Default: joint_vel.",
    )
    return parser.parse_args()


def parse_optional_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def ensure_skygp_import() -> None:
    sys.dont_write_bytecode = True
    repo_root = Path(__file__).resolve().parents[1]
    gp_dir = repo_root / "new_structure" / "gp"
    if str(gp_dir) not in sys.path:
        sys.path.insert(0, str(gp_dir))
    try:
        import skygp  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Cannot import skygp from {gp_dir}") from exc


def parse_float(value: str | None) -> float:
    if value is None:
        return math.nan
    stripped = value.strip()
    if not stripped:
        return math.nan
    try:
        return float(stripped)
    except ValueError:
        return math.nan


def load_csv_numeric(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no CSV header found")
        columns = list(reader.fieldnames)
        data = {column: [] for column in columns}
        for row in reader:
            for column in columns:
                data[column].append(parse_float(row.get(column)))

    arrays = {column: np.asarray(values, dtype=float) for column, values in data.items()}
    rows = len(next(iter(arrays.values()))) if arrays else 0
    return {"path": path, "columns": columns, "data": arrays, "rows": rows}


def feature_columns(feature_source: str) -> list[str]:
    return [f"joint_pos_{joint}" for joint in JOINTS] + [f"{feature_source}_{joint}" for joint in JOINTS]


def feature_names(feature_source: str) -> list[str]:
    velocity_names = [f"dq{joint}" for joint in JOINTS]
    if feature_source == "dq_des_joint":
        velocity_names = [f"dq_des{joint}" for joint in JOINTS]
    return [f"q{joint}" for joint in JOINTS] + velocity_names


def require_columns(dataset: dict[str, Any], required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in dataset["columns"]]
    if missing:
        raise KeyError(f"{label} is missing required columns: {', '.join(missing)}")


def build_feature_matrix(dataset: dict[str, Any], feature_source: str, label: str) -> np.ndarray:
    columns = feature_columns(feature_source)
    require_columns(dataset, columns, label)
    data = dataset["data"]
    matrix = np.stack([data[column] for column in columns], axis=1).astype(np.float32)
    if matrix.shape[1] != 14:
        raise ValueError(f"{label}: expected 14 feature columns, got {matrix.shape[1]}")
    return matrix


def format_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        if math.isnan(number):
            return "nan"
        return "inf" if number > 0 else "-inf"
    return f"{number:.9g}"


def format_bool(value: bool) -> str:
    return "true" if bool(value) else "false"


def format_array(values: Any, max_items: int | None = None) -> str:
    array = np.asarray(values).reshape(-1)
    if max_items is not None:
        array = array[:max_items]
    return ";".join(format_float(value) for value in array)


def format_list(values: Iterable[Any]) -> str:
    return ";".join(str(value) for value in values)


def safe_len(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def safe_shape(value: Any) -> str:
    try:
        return "x".join(str(dim) for dim in np.asarray(value).shape)
    except Exception:
        return ""


def first_scalar(values: Any, default: float = math.nan) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return default
    return float(array[0])


def load_pickle(path: Path) -> Any:
    ensure_skygp_import()
    with path.open("rb") as handle:
        return pickle.load(handle)


def unpack_model_pack(pack: Any) -> dict[str, Any]:
    is_dict = isinstance(pack, dict)
    model = pack.get("model") if is_dict else pack
    stats = pack.get("stats") if is_dict else getattr(model, "stats", None)
    hps_std = pack.get("hps_std") if is_dict else getattr(model, "hps_std", None)
    pack_type = pack.get("type") if is_dict else getattr(model, "type", "")

    if stats is None:
        attrs = []
        for name in ("Xm", "Xs", "Ym", "Ys"):
            attrs.append(getattr(model, name, None))
        if all(item is not None for item in attrs):
            stats = tuple(attrs)

    return {
        "pack": pack,
        "model": model,
        "stats": stats,
        "hps_std": hps_std,
        "pack_type": pack_type,
        "is_dict": is_dict,
    }


def stats_arrays(stats: Any, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if stats is None:
        raise ValueError(f"{path}: model pack does not contain stats or Xm/Xs/Ym/Ys attributes")
    if len(stats) != 4:
        raise ValueError(f"{path}: stats must contain (Xm, Xs, Ym, Ys), got length {len(stats)}")
    Xm, Xs, Ym, Ys = stats
    Xm = np.asarray(Xm, dtype=float).reshape(-1)
    Xs = np.asarray(Xs, dtype=float).reshape(-1)
    Ym = np.asarray(Ym, dtype=float).reshape(-1)
    Ys = np.asarray(Ys, dtype=float).reshape(-1)
    if Xm.size == 0 or Xs.size == 0:
        raise ValueError(f"{path}: empty Xm/Xs stats")
    if Xm.size != Xs.size:
        raise ValueError(f"{path}: Xm and Xs dimensions differ ({Xm.size} vs {Xs.size})")
    return Xm, Xs, Ym, Ys


def get_attr(model: Any, name: str, default: Any = None) -> Any:
    return getattr(model, name, default)


def counts_array(model: Any) -> np.ndarray:
    counts = get_attr(model, "localCount", [])
    try:
        return np.asarray(counts, dtype=int).reshape(-1)
    except Exception:
        return np.asarray([], dtype=int)


def model_training_x_matrix(model: Any, x_dim: int) -> np.ndarray:
    x_list = get_attr(model, "X_list", [])
    counts = counts_array(model)
    chunks = []

    for index, item in enumerate(x_list):
        array = np.asarray(item, dtype=float)
        if array.size == 0:
            continue
        count = int(counts[index]) if index < counts.size else None
        if count is not None and count <= 0:
            continue

        if array.ndim == 1:
            if array.size == x_dim:
                chunks.append(array.reshape(1, x_dim))
            continue

        if array.ndim != 2:
            continue

        if array.shape[0] == x_dim:
            usable = array[:, :count] if count is not None else array
            chunks.append(usable.T)
        elif array.shape[1] == x_dim:
            usable = array[:count, :] if count is not None else array
            chunks.append(usable)

    if not chunks:
        return np.empty((0, x_dim), dtype=float)
    return np.vstack(chunks).astype(float)


def finite_min(values: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmin(np.where(np.isfinite(values), values, np.nan), axis=axis)


def finite_max(values: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmax(np.where(np.isfinite(values), values, np.nan), axis=axis)


def finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmean(np.where(np.isfinite(values), values, np.nan), axis=axis)


def finite_std(values: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanstd(np.where(np.isfinite(values), values, np.nan), axis=axis)


def resolve_sample_index(index: int, rows: int, label: str) -> int:
    if rows <= 0:
        raise ValueError(f"{label}: no rows available")
    resolved = index + rows if index < 0 else index
    if resolved < 0 or resolved >= rows:
        raise IndexError(f"{label}: sample index {index} resolves to {resolved}, outside 0..{rows - 1}")
    return resolved


def standardize_features(x_matrix: np.ndarray, Xm: np.ndarray, Xs: np.ndarray, x_dim: int) -> np.ndarray:
    if x_dim > x_matrix.shape[1]:
        raise ValueError(
            f"Model expects x_dim={x_dim}, but feature matrix has only {x_matrix.shape[1]} columns. "
            "Use the matching --feature-source or regenerate matched data."
        )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return (x_matrix[:, :x_dim] - Xm[:x_dim]) / Xs[:x_dim]


def reset_prediction_state(model: Any) -> None:
    for name, value in (
        ("last_sorted_experts", None),
        ("last_prediction_cache", {}),
        ("last_x", None),
        ("last_expert_idx", None),
    ):
        if hasattr(model, name):
            setattr(model, name, copy.deepcopy(value))


def predict_once(model: Any, x_std: np.ndarray, Ym: float, Ys: float) -> dict[str, Any]:
    try:
        model_copy = copy.deepcopy(model)
    except Exception as exc:
        return {"ok": False, "error": f"deepcopy failed: {exc}"}

    try:
        reset_prediction_state(model_copy)
        mu_std, var_std = model_copy.predict(np.asarray(x_std, dtype=np.float32))
        raw = first_scalar(mu_std)
        var = first_scalar(var_std)
        return {
            "ok": True,
            "raw": raw,
            "var_raw": var,
            "destd": raw * Ys + Ym,
            "error": "",
        }
    except Exception as exc:
        return {"ok": False, "error": f"predict failed: {exc}"}


def prediction_row_indices(total_rows: int, max_rows: int) -> np.ndarray:
    if total_rows <= 0:
        return np.asarray([], dtype=int)
    if max_rows <= 0 or max_rows >= total_rows:
        return np.arange(total_rows, dtype=int)
    return np.unique(np.rint(np.linspace(0, total_rows - 1, max_rows)).astype(int))


def format_prediction_indices(indices: np.ndarray, total_rows: int) -> str:
    if indices.size == 0:
        return ""
    if indices.size == total_rows and int(indices[0]) == 0 and int(indices[-1]) == total_rows - 1:
        return "all"
    if indices.size > 20:
        return f"{int(indices[0])};...;{int(indices[-1])} ({indices.size} evenly sampled rows)"
    return format_array(indices)


def formal_prediction_span_row(
    mode_name: str,
    joint: int,
    model: Any,
    Xm: np.ndarray,
    Xs: np.ndarray,
    Ym_array: np.ndarray,
    Ys_array: np.ndarray,
    x_formal: np.ndarray,
    selected_indices: np.ndarray,
    max_prediction_rows: int,
) -> dict[str, Any]:
    x_dim = int(Xm.size)
    Ym = first_scalar(Ym_array, default=0.0)
    Ys = first_scalar(Ys_array, default=1.0)
    if Ys == 0.0:
        Ys = 1.0

    try:
        model_copy = copy.deepcopy(model)
    except Exception as exc:
        error = f"deepcopy failed: {exc}"
        return {
            "mode_name": mode_name,
            "joint": joint,
            "formal_rows_total": int(x_formal.shape[0]),
            "formal_rows_selected": int(selected_indices.size),
            "formal_rows_predicted": 0,
            "max_prediction_rows": max_prediction_rows,
            "selected_row_indices": format_prediction_indices(selected_indices, int(x_formal.shape[0])),
            "raw_prediction_min": "nan",
            "raw_prediction_max": "nan",
            "raw_prediction_span": "nan",
            "raw_prediction_std": "nan",
            "destd_prediction_min": "nan",
            "destd_prediction_max": "nan",
            "destd_prediction_span": "nan",
            "destd_prediction_std": "nan",
            "formal_prediction_constant_fullrun": "false",
            "formal_prediction_complete": "false",
            "prediction_error": "true",
            "prediction_error_message": error,
            "prediction_errors": error,
        }

    raw_values = []
    errors = []
    formal_std = standardize_features(x_formal, Xm, Xs, x_dim)
    reset_prediction_state(model_copy)
    for row_index in selected_indices:
        try:
            mu_std, _ = model_copy.predict(np.asarray(formal_std[row_index], dtype=np.float32))
            raw_values.append(first_scalar(mu_std))
        except Exception as exc:
            errors.append(f"row {int(row_index)}: {exc}")

    raw_array = np.asarray(raw_values, dtype=float)
    destd_array = raw_array * Ys + Ym
    if raw_array.size:
        raw_min = float(np.nanmin(raw_array))
        raw_max = float(np.nanmax(raw_array))
        raw_span = raw_max - raw_min
        raw_std = float(np.nanstd(raw_array))
        destd_min = float(np.nanmin(destd_array))
        destd_max = float(np.nanmax(destd_array))
        destd_span = destd_max - destd_min
        destd_std = float(np.nanstd(destd_array))
        is_constant = (
            math.isfinite(raw_span)
            and math.isfinite(raw_std)
            and raw_span <= PREDICTION_SPAN_EPS
            and raw_std <= PREDICTION_STD_EPS
        )
    else:
        raw_min = raw_max = raw_span = raw_std = math.nan
        destd_min = destd_max = destd_span = destd_std = math.nan
        is_constant = False

    prediction_complete = raw_array.size == selected_indices.size and not errors
    prediction_error = bool(errors)
    prediction_error_message = " | ".join(errors)

    return {
        "mode_name": mode_name,
        "joint": joint,
        "formal_rows_total": int(x_formal.shape[0]),
        "formal_rows_selected": int(selected_indices.size),
        "formal_rows_predicted": int(raw_array.size),
        "max_prediction_rows": max_prediction_rows,
        "selected_row_indices": format_prediction_indices(selected_indices, int(x_formal.shape[0])),
        "raw_prediction_min": format_float(raw_min),
        "raw_prediction_max": format_float(raw_max),
        "raw_prediction_span": format_float(raw_span),
        "raw_prediction_std": format_float(raw_std),
        "destd_prediction_min": format_float(destd_min),
        "destd_prediction_max": format_float(destd_max),
        "destd_prediction_span": format_float(destd_span),
        "destd_prediction_std": format_float(destd_std),
        "formal_prediction_constant_fullrun": format_bool(is_constant),
        "formal_prediction_complete": format_bool(prediction_complete),
        "prediction_error": format_bool(prediction_error),
        "prediction_error_message": prediction_error_message,
        "prediction_errors": prediction_error_message,
    }


def model_state_row(
    mode_name: str,
    joint: int,
    path: Path,
    pack_info: dict[str, Any],
    Xm: np.ndarray,
    Xs: np.ndarray,
    Ym: np.ndarray,
    Ys: np.ndarray,
    feature_name_list: list[str],
) -> tuple[dict[str, Any], np.ndarray, bool, bool]:
    model = pack_info["model"]
    x_dim = int(Xm.size)
    train_x = model_training_x_matrix(model, x_dim)
    counts = counts_array(model)
    x_list_len = safe_len(get_attr(model, "X_list", []))
    y_list_len = safe_len(get_attr(model, "Y_list", []))
    centers_len = safe_len(get_attr(model, "expert_centers", []))
    l_all = get_attr(model, "L_all", [])
    alpha_all = get_attr(model, "alpha_all", [])
    alpha_shapes = [safe_shape(item) for item in alpha_all]
    l_shapes = [safe_shape(item) for item in l_all]
    total_samples = int(np.sum(counts)) if counts.size else int(train_x.shape[0])
    appears_empty = total_samples <= 0 or x_list_len == 0
    appears_trained = (not appears_empty) and centers_len > 0 and safe_len(alpha_all) > 0
    near_zero_mask = np.abs(Xs[:x_dim]) < NEAR_ZERO_XS_EPS
    near_zero_indices = np.flatnonzero(near_zero_mask)
    near_zero_names = [feature_name_list[index] for index in near_zero_indices if index < len(feature_name_list)]

    row = {
        "mode_name": mode_name,
        "joint": joint,
        "model_path": str(path),
        "pack_object_type": type(pack_info["pack"]).__name__,
        "model_object_type": type(model).__name__,
        "is_dict_pack": format_bool(pack_info["is_dict"]),
        "has_model_key": format_bool(pack_info["is_dict"] and "model" in pack_info["pack"]),
        "has_stats_key": format_bool(pack_info["is_dict"] and "stats" in pack_info["pack"]),
        "has_hps_std_key": format_bool(pack_info["is_dict"] and "hps_std" in pack_info["pack"]),
        "has_type_key": format_bool(pack_info["is_dict"] and "type" in pack_info["pack"]),
        "pack_type": str(pack_info["pack_type"]),
        "Xm": format_array(Xm),
        "Xs": format_array(Xs),
        "Ym": format_array(Ym),
        "Ys": format_array(Ys),
        "input_dim": x_dim,
        "X_list_length": x_list_len,
        "total_sample_count": total_samples,
        "Y_list_length": y_list_len,
        "localCount": format_array(counts),
        "expert_centers_count": centers_len,
        "L_all_count": safe_len(l_all),
        "L_all_shapes": format_list(l_shapes),
        "alpha_all_count": safe_len(alpha_all),
        "alpha_all_shapes": format_list(alpha_shapes),
        "appears_empty": format_bool(appears_empty),
        "appears_trained": format_bool(appears_trained),
        "near_zero_Xs_count": int(near_zero_indices.size),
        "near_zero_Xs_dims": format_list(near_zero_names),
        "near_zero_Xs_values": format_array(Xs[near_zero_indices]),
    }
    return row, train_x, appears_empty, bool(near_zero_indices.size)


def support_check_row(
    mode_name: str,
    joint: int,
    Xm: np.ndarray,
    Xs: np.ndarray,
    x_formal: np.ndarray,
    x_train_model_std: np.ndarray,
    feature_name_list: list[str],
) -> dict[str, Any]:
    x_dim = int(Xm.size)
    formal_std = standardize_features(x_formal, Xm, Xs, x_dim)
    if x_train_model_std.shape[0] == 0:
        model_min = np.full(x_dim, math.nan)
        model_max = np.full(x_dim, math.nan)
    else:
        model_min = finite_min(x_train_model_std[:, :x_dim], axis=0)
        model_max = finite_max(x_train_model_std[:, :x_dim], axis=0)

    formal_min = finite_min(formal_std, axis=0)
    formal_max = finite_max(formal_std, axis=0)
    formal_abs_max = np.maximum(np.abs(formal_min), np.abs(formal_max))
    outside_low = np.maximum(model_min - formal_min, 0.0)
    outside_high = np.maximum(formal_max - model_max, 0.0)
    outside_distance = np.maximum(outside_low, outside_high)
    outside_beyond_margin = np.maximum(outside_distance - OUTSIDE_MARGIN, 0.0)
    outside_mask = outside_distance > OUTSIDE_MARGIN
    severe_abs_mask = formal_abs_max > SEVERE_STD_ABS
    near_zero_mask = np.abs(Xs[:x_dim]) < NEAR_ZERO_XS_EPS
    suspicious_mask = near_zero_mask | outside_mask | severe_abs_mask

    score = np.nan_to_num(formal_abs_max, nan=-1.0, posinf=float("inf"))
    score = np.maximum(score, np.nan_to_num(outside_distance, nan=-1.0, posinf=float("inf")))
    if np.any(suspicious_mask):
        score = np.where(suspicious_mask, score + 1e6, score)
    worst_dim_index = int(np.argmax(score)) if score.size else -1
    worst_dim_name = feature_name_list[worst_dim_index] if 0 <= worst_dim_index < len(feature_name_list) else ""
    likely_out = bool(np.any(suspicious_mask))

    return {
        "mode_name": mode_name,
        "joint": joint,
        "input_dim": x_dim,
        "formal_rows": int(x_formal.shape[0]),
        "model_training_support_samples": int(x_train_model_std.shape[0]),
        "formal_std_min": format_array(formal_min),
        "formal_std_max": format_array(formal_max),
        "model_training_std_min": format_array(model_min),
        "model_training_std_max": format_array(model_max),
        "max_abs_standardized_formal_value": format_float(np.nanmax(formal_abs_max)),
        "max_outside_training_range_distance": format_float(np.nanmax(outside_distance)),
        "max_outside_margin_distance": format_float(np.nanmax(outside_beyond_margin)),
        "dimensions_outside_training_range": int(np.sum(outside_mask)),
        "dimensions_with_severe_abs_std": int(np.sum(severe_abs_mask)),
        "near_zero_Xs_dimensions": int(np.sum(near_zero_mask)),
        "suspicious_dimensions": int(np.sum(suspicious_mask)),
        "worst_dimension_index": worst_dim_index,
        "worst_dimension_name": worst_dim_name,
        "worst_dimension_Xs": format_float(Xs[worst_dim_index] if worst_dim_index >= 0 else math.nan),
        "worst_dimension_formal_std_min": format_float(formal_min[worst_dim_index] if worst_dim_index >= 0 else math.nan),
        "worst_dimension_formal_std_max": format_float(formal_max[worst_dim_index] if worst_dim_index >= 0 else math.nan),
        "worst_dimension_model_std_min": format_float(model_min[worst_dim_index] if worst_dim_index >= 0 else math.nan),
        "worst_dimension_model_std_max": format_float(model_max[worst_dim_index] if worst_dim_index >= 0 else math.nan),
        "worst_dimension_outside_distance": format_float(outside_distance[worst_dim_index] if worst_dim_index >= 0 else math.nan),
        "formal_likely_out_of_support": format_bool(likely_out),
    }


def distribution_rows(
    mode_name: str,
    train_csv: Path,
    formal_csv: Path,
    x_train: np.ndarray,
    x_formal: np.ndarray,
    feature_name_list: list[str],
) -> list[dict[str, Any]]:
    train_mean = finite_mean(x_train, axis=0)
    train_std = finite_std(x_train, axis=0)
    train_min = finite_min(x_train, axis=0)
    train_max = finite_max(x_train, axis=0)
    train_span = train_max - train_min
    formal_mean = finite_mean(x_formal, axis=0)
    formal_std = finite_std(x_formal, axis=0)
    formal_min = finite_min(x_formal, axis=0)
    formal_max = finite_max(x_formal, axis=0)
    formal_span = formal_max - formal_min

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        mean_offset_std = (formal_mean - train_mean) / train_std

    rows = []
    for index, name in enumerate(feature_name_list):
        near_zero = bool(abs(train_std[index]) < NEAR_ZERO_XS_EPS)
        outside = bool(formal_min[index] < train_min[index] or formal_max[index] > train_max[index])
        rows.append(
            {
                "mode_name": mode_name,
                "train_csv": str(train_csv),
                "formal_csv": str(formal_csv),
                "feature_dim": index,
                "feature_name": name,
                "train_mean": format_float(train_mean[index]),
                "train_std": format_float(train_std[index]),
                "train_min": format_float(train_min[index]),
                "train_max": format_float(train_max[index]),
                "train_span": format_float(train_span[index]),
                "formal_mean": format_float(formal_mean[index]),
                "formal_std": format_float(formal_std[index]),
                "formal_min": format_float(formal_min[index]),
                "formal_max": format_float(formal_max[index]),
                "formal_span": format_float(formal_span[index]),
                "formal_vs_train_mean_offset_std": format_float(mean_offset_std[index]),
                "near_zero_train_std": format_bool(near_zero),
                "formal_outside_train_minmax": format_bool(outside),
            }
        )
    return rows


def prediction_sanity_row(
    mode_name: str,
    joint: int,
    model: Any,
    Xm: np.ndarray,
    Xs: np.ndarray,
    Ym_array: np.ndarray,
    Ys_array: np.ndarray,
    x_formal: np.ndarray,
    x_train_model_std: np.ndarray,
    sample_a: int,
    sample_b: int,
) -> dict[str, Any]:
    x_dim = int(Xm.size)
    Ym = first_scalar(Ym_array, default=0.0)
    Ys = first_scalar(Ys_array, default=1.0)
    if Ys == 0.0:
        Ys = 1.0

    formal_std = standardize_features(x_formal, Xm, Xs, x_dim)
    pred_fa = predict_once(model, formal_std[sample_a], Ym, Ys)
    pred_fb = predict_once(model, formal_std[sample_b], Ym, Ys)

    if x_train_model_std.shape[0] >= 2:
        train_a_index = 0
        train_b_index = int(x_train_model_std.shape[0] - 1)
    elif x_train_model_std.shape[0] == 1:
        train_a_index = 0
        train_b_index = 0
    else:
        train_a_index = -1
        train_b_index = -1

    if train_a_index >= 0:
        pred_ta = predict_once(model, x_train_model_std[train_a_index, :x_dim], Ym, Ys)
        pred_tb = predict_once(model, x_train_model_std[train_b_index, :x_dim], Ym, Ys)
    else:
        pred_ta = {"ok": False, "error": "no model training support samples"}
        pred_tb = {"ok": False, "error": "no model training support samples"}

    formal_raw_diff = (
        abs(pred_fa["raw"] - pred_fb["raw"]) if pred_fa.get("ok") and pred_fb.get("ok") else math.nan
    )
    formal_destd_diff = (
        abs(pred_fa["destd"] - pred_fb["destd"]) if pred_fa.get("ok") and pred_fb.get("ok") else math.nan
    )
    train_raw_diff = (
        abs(pred_ta["raw"] - pred_tb["raw"]) if pred_ta.get("ok") and pred_tb.get("ok") else math.nan
    )
    train_destd_diff = (
        abs(pred_ta["destd"] - pred_tb["destd"]) if pred_ta.get("ok") and pred_tb.get("ok") else math.nan
    )
    formal_dependent = bool(
        np.isfinite(formal_raw_diff)
        and np.isfinite(formal_destd_diff)
        and (formal_raw_diff > PREDICTION_DIFF_EPS or formal_destd_diff > PREDICTION_DIFF_EPS)
    )
    train_dependent = bool(
        np.isfinite(train_raw_diff)
        and np.isfinite(train_destd_diff)
        and (train_raw_diff > PREDICTION_DIFF_EPS or train_destd_diff > PREDICTION_DIFF_EPS)
    )

    errors = [item.get("error", "") for item in (pred_fa, pred_fb, pred_ta, pred_tb) if item.get("error")]
    return {
        "mode_name": mode_name,
        "joint": joint,
        "formal_sample_a": sample_a,
        "formal_sample_b": sample_b,
        "support_sample_a": train_a_index,
        "support_sample_b": train_b_index,
        "formal_raw_prediction_a": format_float(pred_fa.get("raw", math.nan)),
        "formal_raw_prediction_b": format_float(pred_fb.get("raw", math.nan)),
        "formal_raw_abs_diff": format_float(formal_raw_diff),
        "formal_destd_prediction_a": format_float(pred_fa.get("destd", math.nan)),
        "formal_destd_prediction_b": format_float(pred_fb.get("destd", math.nan)),
        "formal_destd_abs_diff": format_float(formal_destd_diff),
        "formal_prediction_input_dependent": format_bool(formal_dependent),
        "support_raw_prediction_a": format_float(pred_ta.get("raw", math.nan)),
        "support_raw_prediction_b": format_float(pred_tb.get("raw", math.nan)),
        "support_raw_abs_diff": format_float(train_raw_diff),
        "support_destd_prediction_a": format_float(pred_ta.get("destd", math.nan)),
        "support_destd_prediction_b": format_float(pred_tb.get("destd", math.nan)),
        "support_destd_abs_diff": format_float(train_destd_diff),
        "support_prediction_input_dependent": format_bool(train_dependent),
        "prediction_errors": " | ".join(errors),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def bool_from_row(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key, "")).lower() == "true"


def top_distribution_rows(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    def score(row: dict[str, Any]) -> float:
        value = row.get("formal_vs_train_mean_offset_std", "nan")
        try:
            parsed = abs(float(value))
        except ValueError:
            parsed = math.inf if value in ("inf", "-inf") else -1.0
        if bool_from_row(row, "near_zero_train_std") or bool_from_row(row, "formal_outside_train_minmax"):
            parsed += 1e6
        return parsed

    return sorted(rows, key=score, reverse=True)[:limit]


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def classify_statuses(
    model_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    distribution_check_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    formal_span_rows: list[dict[str, Any]],
) -> list[str]:
    statuses = []
    if any(bool_from_row(row, "appears_empty") for row in model_rows):
        statuses.append("fail_empty_model")
    if any(bool_from_row(row, "formal_likely_out_of_support") for row in support_rows):
        statuses.append("fail_formal_out_of_support")
    if prediction_rows and all(not bool_from_row(row, "formal_prediction_input_dependent") for row in prediction_rows):
        statuses.append("fail_constant_formal_prediction")
    if formal_span_rows and all(bool_from_row(row, "formal_prediction_constant_fullrun") for row in formal_span_rows):
        statuses.append("fail_constant_formal_prediction_fullrun")
    if any(int(row.get("near_zero_Xs_count", 0)) > 0 for row in model_rows):
        statuses.append("warning_near_zero_scaler")
    if any(
        bool_from_row(row, "near_zero_train_std") or bool_from_row(row, "formal_outside_train_minmax")
        for row in distribution_check_rows
    ):
        statuses.append("warning_train_formal_distribution_shift")
    if (
        "fail_empty_model" not in statuses
        and "fail_formal_out_of_support" not in statuses
        and prediction_rows
        and all(bool_from_row(row, "formal_prediction_input_dependent") for row in prediction_rows)
    ):
        statuses.append("pass_input_dependent")
    if not statuses:
        statuses.append("unknown")
    return statuses


def row_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def row_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def optional_bool_text(value: bool | None) -> str:
    if value is None:
        return ""
    return format_bool(value)


def safety_parameter_check(args: argparse.Namespace) -> dict[str, str]:
    warnings = []
    blocking = []

    if args.gp_online_update_enabled is None:
        warnings.append("gp_online_update_enabled not provided")
    elif args.gp_online_update_enabled:
        blocking.append("gp_online_update_enabled should be false for frozen formal tests")

    if args.gp_compensation_scale is None:
        warnings.append("gp_compensation_scale not provided")
    elif not math.isfinite(args.gp_compensation_scale):
        blocking.append("gp_compensation_scale must be finite")
    elif args.gp_compensation_scale >= 1.0:
        blocking.append("gp_compensation_scale >= 1.0 is not conservative")
    elif args.gp_compensation_scale < 0.0:
        warnings.append("gp_compensation_scale is negative")

    if args.gp_compensation_clip_nm is None:
        warnings.append("gp_compensation_clip_nm not provided")
    elif not math.isfinite(args.gp_compensation_clip_nm):
        blocking.append("gp_compensation_clip_nm must be finite")
    elif args.gp_compensation_clip_nm < 0.0:
        blocking.append("gp_compensation_clip_nm must not be negative")
    elif args.gp_compensation_clip_nm == 0.0:
        warnings.append("gp_compensation_clip_nm=0 may clip compensation to zero; it is not treated as no-clip")
    elif args.gp_compensation_clip_nm >= 1.0e6:
        blocking.append("gp_compensation_clip_nm is effectively no clip")

    status = "pass" if not warnings and not blocking else "warning"
    if blocking:
        status = "fail"
    return {
        "safety_parameter_status": status,
        "safety_warnings": " | ".join(warnings),
        "safety_blocking_reasons": " | ".join(blocking),
    }


def preflight_gate_summary_row(
    args: argparse.Namespace,
    model_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    distribution_check_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    formal_span_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    num_joints = len(model_rows)
    num_empty = sum(bool_from_row(row, "appears_empty") for row in model_rows)
    num_trained = sum(bool_from_row(row, "appears_trained") for row in model_rows)
    num_near_zero = sum(row_int(row, "near_zero_Xs_count") > 0 for row in model_rows)
    num_out_of_support = sum(bool_from_row(row, "formal_likely_out_of_support") for row in support_rows)
    num_constant_full = sum(bool_from_row(row, "formal_prediction_constant_fullrun") for row in formal_span_rows)
    num_support_dependent = sum(bool_from_row(row, "support_prediction_input_dependent") for row in prediction_rows)
    num_prediction_error = sum(bool_from_row(row, "prediction_error") for row in formal_span_rows)
    num_incomplete_prediction = sum(not bool_from_row(row, "formal_prediction_complete") for row in formal_span_rows)
    num_distribution_shift = sum(
        bool_from_row(row, "near_zero_train_std") or bool_from_row(row, "formal_outside_train_minmax")
        for row in distribution_check_rows
    )

    worst_support_row = max(
        support_rows,
        key=lambda row: row_float(row, "max_abs_standardized_formal_value", default=-1.0),
        default={},
    )
    worst_dimension = str(worst_support_row.get("worst_dimension_name", ""))
    worst_formal_abs_std = row_float(worst_support_row, "max_abs_standardized_formal_value")

    if num_empty:
        overall_status = "fail_model_empty"
    elif num_out_of_support:
        overall_status = "fail_formal_out_of_support"
    elif num_constant_full:
        overall_status = "fail_constant_formal_prediction"
    elif num_prediction_error:
        overall_status = "fail_prediction_error"
    elif num_incomplete_prediction:
        overall_status = "fail_incomplete_formal_prediction"
    elif num_near_zero:
        overall_status = "fail_near_zero_scaler"
    elif num_trained == num_joints and num_joints > 0:
        overall_status = "pass_ready_for_conservative_robot_validation"
    else:
        overall_status = "unknown"

    safety = safety_parameter_check(args)
    safety_passed = safety["safety_parameter_status"] == "pass"
    blocking_reasons = []
    risk_reasons = []
    if num_empty:
        blocking_reasons.append("model_empty")
    if num_out_of_support:
        blocking_reasons.append("formal_out_of_support")
    if num_constant_full:
        blocking_reasons.append("constant_formal_prediction_fullrun")
    if num_prediction_error:
        blocking_reasons.append("prediction_error")
    if num_incomplete_prediction:
        blocking_reasons.append("incomplete_formal_prediction")
    if not safety_passed:
        blocking_reasons.append("safety_parameter_failure")
    if num_near_zero:
        risk_reasons.append("near_zero_scaler")
    if num_distribution_shift:
        risk_reasons.append("train_formal_distribution_shift")

    gate_pass = (
        num_empty == 0
        and num_out_of_support == 0
        and num_constant_full == 0
        and num_prediction_error == 0
        and num_incomplete_prediction == 0
        and safety_passed
        and overall_status == "pass_ready_for_conservative_robot_validation"
    )

    if gate_pass:
        recommended_action = (
            "ready only for conservative read-only-reviewed frozen robot validation; keep online update off, "
            "keep compensation clipped, and do not bypass safety gating"
        )
    else:
        recommended_action = (
            "do not run real-robot scale sweep yet; regenerate or retrain matched frozen models; rerun validator; "
            "treat current GP-on runs as fixed-bias compensation observations"
        )

    return {
        "mode_name": args.mode_name,
        "overall_status": overall_status,
        "gate_pass": format_bool(gate_pass),
        "num_joints": num_joints,
        "num_empty_models": int(num_empty),
        "num_trained_models": int(num_trained),
        "num_near_zero_scaler_joints": int(num_near_zero),
        "num_out_of_support_joints": int(num_out_of_support),
        "num_constant_formal_prediction_joints": int(num_constant_full),
        "num_prediction_error_joints": int(num_prediction_error),
        "num_incomplete_formal_prediction_joints": int(num_incomplete_prediction),
        "num_training_support_input_dependent_joints": int(num_support_dependent),
        "num_train_formal_distribution_shift_dimensions": int(num_distribution_shift),
        "worst_dimension": worst_dimension,
        "worst_formal_abs_std": format_float(worst_formal_abs_std),
        "gp_online_update_enabled": optional_bool_text(args.gp_online_update_enabled),
        "gp_compensation_scale": "" if args.gp_compensation_scale is None else format_float(args.gp_compensation_scale),
        "gp_compensation_clip_nm": ""
        if args.gp_compensation_clip_nm is None
        else format_float(args.gp_compensation_clip_nm),
        "safety_parameter_status": safety["safety_parameter_status"],
        "safety_warnings": safety["safety_warnings"],
        "safety_blocking_reasons": safety["safety_blocking_reasons"],
        "blocking_reasons": format_list(blocking_reasons),
        "risk_reasons": format_list(risk_reasons),
        "recommended_action": recommended_action,
    }


def write_summary(
    path: Path,
    args: argparse.Namespace,
    feature_name_list: list[str],
    sample_a: int,
    sample_b: int,
    model_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    distribution_check_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    formal_span_rows: list[dict[str, Any]],
    preflight_row: dict[str, Any],
    statuses: list[str],
) -> None:
    out_of_support = any(bool_from_row(row, "formal_likely_out_of_support") for row in support_rows)
    constant_formal = prediction_rows and all(
        not bool_from_row(row, "formal_prediction_input_dependent") for row in prediction_rows
    )
    constant_formal_fullrun = formal_span_rows and all(
        bool_from_row(row, "formal_prediction_constant_fullrun") for row in formal_span_rows
    )
    support_varies = prediction_rows and any(
        bool_from_row(row, "support_prediction_input_dependent") for row in prediction_rows
    )
    distribution_shift = any(
        bool_from_row(row, "near_zero_train_std") or bool_from_row(row, "formal_outside_train_minmax")
        for row in distribution_check_rows
    )
    near_zero_scaler = any(int(row.get("near_zero_Xs_count", 0)) > 0 for row in model_rows)

    inventory_rows = [
        [
            row["joint"],
            row["input_dim"],
            row["total_sample_count"],
            row["X_list_length"],
            row["expert_centers_count"],
            row["appears_trained"],
            row["near_zero_Xs_dims"] or "-",
        ]
        for row in model_rows
    ]
    support_table_rows = [
        [
            row["joint"],
            row["worst_dimension_name"],
            row["worst_dimension_Xs"],
            row["max_abs_standardized_formal_value"],
            row["max_outside_training_range_distance"],
            row["dimensions_outside_training_range"],
            row["formal_likely_out_of_support"],
        ]
        for row in support_rows
    ]
    distribution_table_rows = [
        [
            row["feature_name"],
            row["train_std"],
            row["train_min"],
            row["train_max"],
            row["formal_min"],
            row["formal_max"],
            row["formal_vs_train_mean_offset_std"],
            row["near_zero_train_std"],
            row["formal_outside_train_minmax"],
        ]
        for row in top_distribution_rows(distribution_check_rows)
    ]
    prediction_table_rows = [
        [
            row["joint"],
            row["formal_raw_abs_diff"],
            row["formal_destd_abs_diff"],
            row["formal_prediction_input_dependent"],
            row["support_raw_abs_diff"],
            row["support_destd_abs_diff"],
            row["support_prediction_input_dependent"],
        ]
        for row in prediction_rows
    ]
    full_prediction_table_rows = [
        [
            row["joint"],
            row["formal_rows_selected"],
            row["formal_rows_predicted"],
            row["raw_prediction_span"],
            row["raw_prediction_std"],
            row["destd_prediction_span"],
            row["destd_prediction_std"],
            row["formal_prediction_constant_fullrun"],
            row["formal_prediction_complete"],
            row["prediction_error"],
        ]
        for row in formal_span_rows
    ]

    gate_pass = bool_from_row(preflight_row, "gate_pass")
    blocking_reasons = str(preflight_row.get("blocking_reasons", ""))
    risk_reasons = str(preflight_row.get("risk_reasons", ""))
    if not blocking_reasons and not gate_pass:
        blocking_reasons = str(preflight_row.get("overall_status", "unknown"))
    near_zero_warning_text = "present" if near_zero_scaler else "not observed"
    safety_text = preflight_row.get("safety_parameter_status", "unknown")
    if preflight_row.get("safety_warnings"):
        safety_text += f" ({preflight_row['safety_warnings']})"

    if out_of_support and constant_formal:
        recommendation = (
            "Retrain or regenerate matched frozen local GP models before a real-robot scale sweep. "
            "Add this support validation as a read-only gate, and treat current Stage 4 GP-on runs as "
            "fixed-bias compensation observations rather than dynamic state-dependent GP generalization."
        )
    elif "pass_input_dependent" in statuses:
        recommendation = "Models look input-dependent on formal inputs; check runtime loading and logging paths next."
    else:
        recommendation = "Keep interpretation conservative and inspect the CSV/model mismatch before changing robot behavior."

    lines = [
        f"# Frozen GP Support Validation: {args.mode_name}",
        "",
        "## Inputs",
        "",
        f"- model_dir: `{args.model_dir}`",
        f"- formal_csv: `{args.formal_csv}`",
        f"- train_csv: `{args.train_csv}`",
        f"- feature_source: `{args.feature_source}`",
        f"- feature_definition: `[{', '.join(feature_name_list)}]`",
        f"- formal_samples: `{sample_a}` and `{sample_b}`",
        f"- max_prediction_rows: `{args.max_prediction_rows}`",
        f"- gp_online_update_enabled: `{optional_bool_text(args.gp_online_update_enabled) or 'not provided'}`",
        f"- gp_compensation_scale: `{preflight_row['gp_compensation_scale'] or 'not provided'}`",
        f"- gp_compensation_clip_nm: `{preflight_row['gp_compensation_clip_nm'] or 'not provided'}`",
        "",
        "## Status",
        "",
        f"- classifications: `{', '.join(statuses)}`",
        f"- near_zero_Xs_warning: `{format_bool(near_zero_scaler)}`",
        f"- formal_vs_model_support_out_of_support: `{format_bool(out_of_support)}`",
        f"- formal_vs_training_csv_distribution_shift: `{format_bool(distribution_shift)}`",
        f"- formal_prediction_constant: `{format_bool(bool(constant_formal))}`",
        f"- training_support_prediction_varies: `{format_bool(bool(support_varies))}`",
        "",
        "## Offline Preflight Gate",
        "",
        f"- result: `{'PASS' if gate_pass else 'FAIL'}`",
        f"- overall_status: `{preflight_row['overall_status']}`",
        f"- gate_pass: `{preflight_row['gate_pass']}`",
        f"- blocking_reasons: `{blocking_reasons or 'none'}`",
        f"- risk_reasons: `{risk_reasons or 'none'}`",
        f"- near_zero_scaler_warnings: `{near_zero_warning_text}`",
        f"- formal_out_of_support: `{format_bool(out_of_support)}`",
        f"- full_formal_prediction_constant: `{format_bool(bool(constant_formal_fullrun))}`",
        f"- prediction_error_joints: `{preflight_row['num_prediction_error_joints']}`",
        f"- incomplete_formal_prediction_joints: `{preflight_row['num_incomplete_formal_prediction_joints']}`",
        f"- train_formal_distribution_shift_dimensions: `{preflight_row['num_train_formal_distribution_shift_dimensions']}`",
        f"- worst_dimension: `{preflight_row['worst_dimension']}`",
        f"- training_support_predictions_vary: `{format_bool(bool(support_varies))}`",
        f"- safety_parameter_check: `{safety_text}`",
        f"- recommended_next_step: {preflight_row['recommended_action']}",
        "",
        "## Model Inventory",
        "",
        markdown_table(
            ["joint", "input_dim", "samples", "X_list", "centers", "trained", "near_zero_Xs_dims"],
            inventory_rows,
        ),
        "",
        "## Formal vs Model Support",
        "",
        markdown_table(
            [
                "joint",
                "worst_dim",
                "worst_Xs",
                "max_abs_formal_std",
                "max_outside_range",
                "outside_dims",
                "out_of_support",
            ],
            support_table_rows,
        ),
        "",
        "## Formal vs Training CSV Distribution",
        "",
        markdown_table(
            [
                "feature",
                "train_std",
                "train_min",
                "train_max",
                "formal_min",
                "formal_max",
                "mean_offset_std",
                "near_zero_train_std",
                "formal_outside_minmax",
            ],
            distribution_table_rows,
        ),
        "",
        "## Prediction Sanity",
        "",
        markdown_table(
            [
                "joint",
                "formal_raw_diff",
                "formal_destd_diff",
                "formal_input_dependent",
                "support_raw_diff",
                "support_destd_diff",
                "support_input_dependent",
            ],
            prediction_table_rows,
        ),
        "",
        "## Full Formal Prediction Span",
        "",
        markdown_table(
            [
                "joint",
                "selected",
                "predicted",
                "raw_span",
                "raw_std",
                "destd_span",
                "destd_std",
                "constant_fullrun",
                "complete",
                "prediction_error",
            ],
            full_prediction_table_rows,
        ),
        "",
        "## Interpretation",
        "",
        f"- Recommended next step: {recommendation}",
        "- Safety boundary: this read-only validator does not recommend no-clip compensation, unlimited GP compensation, direct scale=1.0 robot runs, bypassing safety gating, or modifying real-robot torque behavior.",
        "- Output files are CSV/Markdown only; no model, raw CSV, controller, launch, or config files are written.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_required_paths(args: argparse.Namespace) -> None:
    missing = []
    if not args.model_dir.is_dir():
        missing.append(str(args.model_dir))
    for joint in JOINTS:
        path = args.model_dir / f"joint{joint}_local.pkl"
        if not path.is_file():
            missing.append(str(path))
    for path in (args.formal_csv, args.train_csv):
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Required validation inputs are missing:\n  - " + "\n  - ".join(missing))


def main() -> int:
    args = parse_args()
    validate_required_paths(args)

    feature_name_list = feature_names(args.feature_source)
    formal_dataset = load_csv_numeric(args.formal_csv)
    train_dataset = load_csv_numeric(args.train_csv)
    x_formal = build_feature_matrix(formal_dataset, args.feature_source, str(args.formal_csv))
    x_train_csv = build_feature_matrix(train_dataset, args.feature_source, str(args.train_csv))
    sample_a = resolve_sample_index(args.sample_a, x_formal.shape[0], "formal CSV")
    sample_b = resolve_sample_index(args.sample_b, x_formal.shape[0], "formal CSV")
    selected_prediction_indices = prediction_row_indices(x_formal.shape[0], args.max_prediction_rows)

    model_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    formal_span_rows: list[dict[str, Any]] = []

    for joint in JOINTS:
        path = args.model_dir / f"joint{joint}_local.pkl"
        pack = load_pickle(path)
        pack_info = unpack_model_pack(pack)
        Xm, Xs, Ym, Ys = stats_arrays(pack_info["stats"], path)
        model_row, train_x_model_std, _, _ = model_state_row(
            args.mode_name,
            joint,
            path,
            pack_info,
            Xm,
            Xs,
            Ym,
            Ys,
            feature_name_list,
        )
        model_rows.append(model_row)
        support_rows.append(
            support_check_row(args.mode_name, joint, Xm, Xs, x_formal, train_x_model_std, feature_name_list)
        )
        prediction_rows.append(
            prediction_sanity_row(
                args.mode_name,
                joint,
                pack_info["model"],
                Xm,
                Xs,
                Ym,
                Ys,
                x_formal,
                train_x_model_std,
                sample_a,
                sample_b,
            )
        )
        formal_span_rows.append(
            formal_prediction_span_row(
                args.mode_name,
                joint,
                pack_info["model"],
                Xm,
                Xs,
                Ym,
                Ys,
                x_formal,
                selected_prediction_indices,
                args.max_prediction_rows,
            )
        )

    distribution_check_rows = distribution_rows(
        args.mode_name,
        args.train_csv,
        args.formal_csv,
        x_train_csv,
        x_formal,
        feature_name_list,
    )
    statuses = classify_statuses(model_rows, support_rows, distribution_check_rows, prediction_rows, formal_span_rows)
    preflight_row = preflight_gate_summary_row(
        args,
        model_rows,
        support_rows,
        distribution_check_rows,
        prediction_rows,
        formal_span_rows,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "model_state_summary.csv", model_rows, list(model_rows[0].keys()))
    write_csv(args.out_dir / "support_check_per_joint.csv", support_rows, list(support_rows[0].keys()))
    write_csv(
        args.out_dir / "training_distribution_check.csv",
        distribution_check_rows,
        list(distribution_check_rows[0].keys()),
    )
    write_csv(args.out_dir / "prediction_sanity_per_joint.csv", prediction_rows, list(prediction_rows[0].keys()))
    write_csv(
        args.out_dir / "formal_prediction_span_per_joint.csv",
        formal_span_rows,
        list(formal_span_rows[0].keys()),
    )
    write_csv(args.out_dir / "preflight_gate_summary.csv", [preflight_row], list(preflight_row.keys()))
    write_summary(
        args.out_dir / "summary.md",
        args,
        feature_name_list,
        sample_a,
        sample_b,
        model_rows,
        support_rows,
        distribution_check_rows,
        prediction_rows,
        formal_span_rows,
        preflight_row,
        statuses,
    )

    print(f"mode_name: {args.mode_name}")
    print(f"classifications: {', '.join(statuses)}")
    print(f"preflight_gate_pass: {preflight_row['gate_pass']}")
    print(f"overall_status: {preflight_row['overall_status']}")
    print(f"wrote: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
