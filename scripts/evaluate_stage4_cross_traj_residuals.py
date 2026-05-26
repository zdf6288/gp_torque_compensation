#!/usr/bin/env python3
"""Offline Stage 4 cross-trajectory residual GP evaluation.

This script builds combined no-GP residual datasets, trains existing frozen GP
model directories through scripts/train_stage4_matched_frozen_gp.py, and
evaluates GP_A/GP_B on the same held-out C residual dataset.

It is offline-only. It does not modify controller, launch, config, trajectory,
or torque command logic.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


sys.dont_write_bytecode = True

try:
    import numpy as np
except ModuleNotFoundError as exc:
    print("Missing Python dependency: numpy", file=sys.stderr)
    print("Use an environment that already has project dependencies installed.", file=sys.stderr)
    raise SystemExit(1) from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    print("Missing Python dependency: matplotlib", file=sys.stderr)
    print("Use an environment that already has project dependencies installed.", file=sys.stderr)
    raise SystemExit(1) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
GP_DIR = REPO_ROOT / "new_structure" / "gp"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_stage4_matched_frozen_gp.py"
DEFAULT_BASE_DIR = Path("data/stage4/cross_traj")
DEFAULT_OUT_DIR = Path("outputs/stage4_cross_traj_residual_eval")
JOINTS = range(1, 8)
SCRIPT_VERSION = "2026-05-26-stage4-cross-traj-residual-eval-v1"
FEATURE_SOURCE = "joint_vel"
TARGET_KIND = "tau_residual_raw"
PREDICTION_SPAN_EPS = 1e-9
CORR_EPS = 1e-12

DEFAULT_A_CSVS = [
    DEFAULT_BASE_DIR / "A_no_gp_planar/usable_runs/A_no_gp_planar_3000pts_20260526_205804.csv",
    DEFAULT_BASE_DIR / "A_no_gp_planar/usable_runs/A_no_gp_planar_3000pts_20260526_210945.csv",
]
DEFAULT_B_CSVS = [
    DEFAULT_BASE_DIR / "B_no_gp_zmod/usable_runs/B_no_gp_zmod_3001pts_20260526_210139.csv",
    DEFAULT_BASE_DIR / "B_no_gp_zmod/usable_runs/B_no_gp_zmod_3000pts_20260526_212242.csv",
]
DEFAULT_C_CSVS = [
    DEFAULT_BASE_DIR / "C_no_gp_zmod_heldout/usable_runs/C_no_gp_zmod_heldout_3000pts_20260526_211759.csv",
    DEFAULT_BASE_DIR / "C_no_gp_zmod_heldout/usable_runs/C_no_gp_zmod_heldout_3000pts_20260526_212556.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline cross-trajectory Stage 4 residual GP evaluation.",
    )
    parser.add_argument("--run-all", action="store_true", help="Build datasets, train GP_A/GP_B, and evaluate on C.")
    parser.add_argument("--a-csv", action="append", type=Path, default=[], help="A planar no-GP training CSV.")
    parser.add_argument("--b-csv", action="append", type=Path, default=[], help="B zmod no-GP training CSV.")
    parser.add_argument("--c-csv", action="append", type=Path, default=[], help="C zmod held-out evaluation CSV.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help=f"Default: {DEFAULT_OUT_DIR}")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_BASE_DIR / "datasets")
    parser.add_argument("--model-base-dir", type=Path, default=DEFAULT_BASE_DIR / "models")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite this script's generated datasets, models, and output files.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Reuse existing model directories and only rebuild datasets/evaluate.",
    )
    parser.add_argument(
        "--drop-nonfinite",
        action="store_true",
        help="Drop rows containing NaN/Inf in X/Y. Default fails instead of dropping.",
    )
    parser.add_argument(
        "--max-prediction-rows",
        type=int,
        default=0,
        help="Optional evenly sampled C rows for prediction. 0 means all rows.",
    )
    parser.add_argument(
        "--train-max-samples",
        type=int,
        default=None,
        help="Optional existing-trainer random subset size for GP_A/GP_B training.",
    )
    return parser.parse_args()


def prefixed_joint_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{joint}" for joint in JOINTS]


def feature_columns() -> list[str]:
    return prefixed_joint_columns("joint_pos") + prefixed_joint_columns(FEATURE_SOURCE)


def target_columns() -> list[str]:
    return prefixed_joint_columns(TARGET_KIND)


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
    if not path.is_file():
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


def require_columns(dataset: dict[str, Any], columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in dataset["columns"]]
    if missing:
        raise KeyError(f"{label} is missing required columns: {', '.join(missing)}")


def stack_columns(dataset: dict[str, Any], columns: list[str], label: str) -> np.ndarray:
    require_columns(dataset, columns, label)
    return np.stack([dataset["data"][column] for column in columns], axis=1).astype(np.float32)


def finite_counts(*arrays: np.ndarray) -> dict[str, int]:
    nan_count = 0
    inf_count = 0
    for array in arrays:
        nan_count += int(np.isnan(array).sum())
        inf_count += int(np.isinf(array).sum())
    return {"nan_count": nan_count, "inf_count": inf_count}


def rows_with_finite_xy(x_matrix: np.ndarray, y_matrix: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(x_matrix), axis=1) & np.all(np.isfinite(y_matrix), axis=1)


def stats_arrays(matrix: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": np.mean(matrix, axis=0).astype(np.float64),
        "std": np.std(matrix, axis=0).astype(np.float64),
        "min": np.min(matrix, axis=0).astype(np.float64),
        "max": np.max(matrix, axis=0).astype(np.float64),
    }


def format_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.9g}"


def format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def as_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_csvs(provided: list[Path], defaults: list[Path]) -> list[Path]:
    return provided if provided else list(defaults)


def selected_row_indices(total_rows: int, max_rows: int) -> np.ndarray:
    if total_rows <= 0:
        return np.asarray([], dtype=int)
    if max_rows <= 0 or max_rows >= total_rows:
        return np.arange(total_rows, dtype=int)
    return np.unique(np.rint(np.linspace(0, total_rows - 1, max_rows)).astype(int))


def ensure_no_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to regenerate it")


def build_combined_dataset(
    csv_paths: list[Path],
    out_npz: Path,
    dataset_role: str,
    trajectory_label: str,
    mode_name: str,
    expected_rows: int | None,
    drop_nonfinite: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if not csv_paths:
        raise ValueError(f"{mode_name}: at least one CSV is required")
    ensure_no_overwrite(out_npz, overwrite)

    features = feature_columns()
    targets = target_columns()
    x_parts = []
    y_parts = []
    source_rows = []
    rows_dropped_per_csv = []

    for csv_path in csv_paths:
        dataset = load_csv_numeric(csv_path)
        x_matrix = stack_columns(dataset, features, f"{csv_path} feature matrix")
        y_matrix = stack_columns(dataset, targets, f"{csv_path} target matrix")
        if x_matrix.shape[1] != 14:
            raise ValueError(f"{csv_path}: expected X second dimension 14, got {x_matrix.shape[1]}")
        if y_matrix.shape[1] != 7:
            raise ValueError(f"{csv_path}: expected Y second dimension 7, got {y_matrix.shape[1]}")
        if x_matrix.shape[0] != y_matrix.shape[0]:
            raise ValueError(f"{csv_path}: X/Y row mismatch {x_matrix.shape[0]} vs {y_matrix.shape[0]}")

        keep = rows_with_finite_xy(x_matrix, y_matrix)
        dropped = int(keep.size - np.sum(keep))
        if dropped and not drop_nonfinite:
            raise ValueError(f"{csv_path}: found {dropped} rows with NaN/Inf; pass --drop-nonfinite to drop them")
        if dropped:
            x_matrix = x_matrix[keep]
            y_matrix = y_matrix[keep]

        source_rows.append(int(dataset["rows"]))
        rows_dropped_per_csv.append(dropped)
        x_parts.append(x_matrix)
        y_parts.append(y_matrix)

    x_all = np.vstack(x_parts).astype(np.float32)
    y_all = np.vstack(y_parts).astype(np.float32)
    counts = finite_counts(x_all, y_all)
    if counts["nan_count"] or counts["inf_count"]:
        raise ValueError(f"{mode_name}: output contains NaN/Inf counts {counts}")
    if x_all.shape[1] != 14 or y_all.shape[1] != 7:
        raise ValueError(f"{mode_name}: unexpected output shapes X={x_all.shape}, Y={y_all.shape}")
    if expected_rows is not None and x_all.shape[0] != expected_rows:
        raise ValueError(f"{mode_name}: expected {expected_rows} rows, got {x_all.shape[0]}")

    feature_stats = stats_arrays(x_all)
    target_stats = stats_arrays(y_all)
    created_utc = datetime.now(timezone.utc).isoformat()
    metadata = {
        "script": as_repo_relative(Path(__file__)),
        "script_version": SCRIPT_VERSION,
        "created_utc": created_utc,
        "out_npz": as_repo_relative(out_npz),
        "mode_name": mode_name,
        "dataset_role": dataset_role,
        "trajectory_label": trajectory_label,
        "source_csvs": [as_repo_relative(path) for path in csv_paths],
        "source_rows_per_csv": source_rows,
        "source_rows_total": int(sum(source_rows)),
        "rows_written": int(x_all.shape[0]),
        "rows_dropped_nonfinite": int(sum(rows_dropped_per_csv)),
        "rows_dropped_per_csv": rows_dropped_per_csv,
        "drop_nonfinite": bool(drop_nonfinite),
        "feature_source": FEATURE_SOURCE,
        "feature_definition": "X = [joint_pos_1..7, joint_vel_1..7]",
        "feature_columns": features,
        "target_kind": TARGET_KIND,
        "target_definition": "Y = tau_residual_raw_1..7",
        "target_columns": targets,
        "x_shape": list(x_all.shape),
        "y_shape": list(y_all.shape),
        "output_nan_count": counts["nan_count"],
        "output_inf_count": counts["inf_count"],
        "caveat": (
            "No-GP residual dataset for offline frozen residual prediction only. "
            "C held-out data must not be used for training."
        ),
    }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        X=x_all,
        Y=y_all,
        **{f"X{joint}": x_all for joint in JOINTS},
        **{f"Y{joint}": y_all[:, joint - 1 : joint] for joint in JOINTS},
        feature_columns=np.asarray(features, dtype=object),
        target_columns=np.asarray(targets, dtype=object),
        source_csv=np.asarray(";".join(str(path) for path in csv_paths)),
        source_csvs=np.asarray([str(path) for path in csv_paths], dtype=object),
        mode_name=np.asarray(mode_name),
        dataset_role=np.asarray(dataset_role),
        trajectory_label=np.asarray(trajectory_label),
        target_kind=np.asarray(TARGET_KIND),
        feature_source=np.asarray(FEATURE_SOURCE),
        metadata_json=np.asarray(json.dumps(metadata, indent=2, sort_keys=True)),
        feature_mean=feature_stats["mean"],
        feature_std=feature_stats["std"],
        feature_min=feature_stats["min"],
        feature_max=feature_stats["max"],
        target_mean=target_stats["mean"],
        target_std=target_stats["std"],
        target_min=target_stats["min"],
        target_max=target_stats["max"],
        source_rows_per_csv=np.asarray(source_rows, dtype=np.int64),
        rows_dropped_per_csv=np.asarray(rows_dropped_per_csv, dtype=np.int64),
        meta=np.asarray(metadata, dtype=object),
    )
    return metadata


def run_training(
    dataset: Path,
    output_dir: Path,
    model_name: str,
    overwrite: bool,
    train_max_samples: int | None,
) -> None:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset",
        str(dataset),
        "--output-dir",
        str(output_dir),
        "--model-name",
        model_name,
    ]
    if overwrite:
        command.append("--overwrite")
    if train_max_samples is not None:
        command.extend(["--max-samples", str(train_max_samples)])
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def ensure_skygp_import() -> None:
    if str(GP_DIR) not in sys.path:
        sys.path.insert(0, str(GP_DIR))
    try:
        import skygp  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Cannot import skygp from {GP_DIR}") from exc


def load_npz_dataset(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        x_matrix = np.asarray(data["X"], dtype=np.float32)
        y_matrix = np.asarray(data["Y"], dtype=np.float32)
        feature_names = [str(item) for item in np.asarray(data["feature_columns"]).reshape(-1)]
        target_names = [str(item) for item in np.asarray(data["target_columns"]).reshape(-1)]
        metadata_json = str(data["metadata_json"].item()) if "metadata_json" in data.files else "{}"
    return {
        "path": path,
        "X": x_matrix,
        "Y": y_matrix,
        "feature_columns": feature_names,
        "target_columns": target_names,
        "metadata": json.loads(metadata_json),
    }


def load_model_pack(path: Path) -> dict[str, Any]:
    ensure_skygp_import()
    with path.open("rb") as handle:
        pack = pickle.load(handle)
    if not isinstance(pack, dict):
        raise ValueError(f"{path}: expected dict model pack")
    if "model" not in pack or "stats" not in pack:
        raise ValueError(f"{path}: expected model and stats keys")
    stats = pack["stats"]
    if len(stats) != 4:
        raise ValueError(f"{path}: stats must be (Xm, Xs, Ym, Ys)")
    Xm, Xs, Ym, Ys = stats
    return {
        "model": pack["model"],
        "Xm": np.asarray(Xm, dtype=float).reshape(-1),
        "Xs": np.asarray(Xs, dtype=float).reshape(-1),
        "Ym": np.asarray(Ym, dtype=float).reshape(-1),
        "Ys": np.asarray(Ys, dtype=float).reshape(-1),
    }


def reset_prediction_state(model: Any) -> None:
    for name, value in (
        ("last_sorted_experts", None),
        ("last_prediction_cache", {}),
        ("last_x", None),
        ("last_expert_idx", None),
    ):
        if hasattr(model, name):
            setattr(model, name, copy.deepcopy(value))


def first_scalar(values: Any, default: float = math.nan) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return default
    return float(array[0])


def fast_single_expert_mean_std(model: Any, x_std: np.ndarray) -> np.ndarray | None:
    if len(getattr(model, "expert_centers", [])) != 1:
        return None
    if int(getattr(model, "nearest_k", 0)) != 1:
        return None
    local_count = int(model.localCount[0])
    if local_count <= 0:
        return None

    params = model.model_params[model.expert_creation_order[0]]
    sigma_f = np.exp(params["log_sigma_f"][0])
    lengthscale = np.exp(params["log_lengthscale"])
    if np.ndim(lengthscale) == 2:
        lengthscale = lengthscale[:, 0]
    x_snapshot = model.X_list[0][:, :local_count]
    alpha = model.alpha_all[0][:local_count, 0]

    # Same mean formula as SkyGP_rBCM.predict for the single-expert frozen
    # local model, batched for offline evaluation. Variance is not needed here.
    k_star = model.kernel_np(x_snapshot, x_std.T, lengthscale, sigma_f)
    return np.asarray(k_star.T @ alpha, dtype=float).reshape(-1)


def predict_model_on_x(model_dir: Path, x_matrix: np.ndarray, max_prediction_rows: int) -> tuple[np.ndarray, np.ndarray]:
    indices = selected_row_indices(x_matrix.shape[0], max_prediction_rows)
    predictions = np.empty((indices.size, 7), dtype=np.float64)
    for joint in JOINTS:
        pack = load_model_pack(model_dir / f"joint{joint}_local.pkl")
        x_dim = pack["Xm"].size
        if x_dim > x_matrix.shape[1]:
            raise ValueError(f"{model_dir}: joint {joint} expects x_dim={x_dim}, got {x_matrix.shape[1]}")
        model = copy.deepcopy(pack["model"])
        reset_prediction_state(model)
        x_std = (x_matrix[indices, :x_dim] - pack["Xm"][:x_dim]) / pack["Xs"][:x_dim]
        Ym = first_scalar(pack["Ym"], default=0.0)
        Ys = first_scalar(pack["Ys"], default=1.0)
        if Ys == 0.0:
            Ys = 1.0
        mu_std_values = fast_single_expert_mean_std(model, x_std)
        if mu_std_values is None:
            joint_values = []
            for row in x_std:
                mu_std, _ = model.predict(np.asarray(row, dtype=np.float32))
                joint_values.append(first_scalar(mu_std))
            mu_std_values = np.asarray(joint_values, dtype=float)
        predictions[:, joint - 1] = mu_std_values * Ys + Ym
        print(f"predicted {model_dir.name} joint {joint}: rows={indices.size}", flush=True)
    return predictions, indices


def safe_corr(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size < 2:
        return math.nan
    if not (np.std(pred) > CORR_EPS and np.std(target) > CORR_EPS):
        return math.nan
    return float(np.corrcoef(pred, target)[0, 1])


def evaluate_predictions(
    model_label: str,
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    error = prediction - target
    per_joint = []
    for joint in JOINTS:
        idx = joint - 1
        pred_j = prediction[:, idx]
        target_j = target[:, idx]
        err_j = error[:, idx]
        pred_span = float(np.max(pred_j) - np.min(pred_j))
        target_span = float(np.max(target_j) - np.min(target_j))
        pred_std = float(np.std(pred_j))
        target_std = float(np.std(target_j))
        per_joint.append(
            {
                "model": model_label,
                "joint": joint,
                "rmse_tau_j": float(np.sqrt(np.mean(err_j**2))),
                "mae_tau_j": float(np.mean(np.abs(err_j))),
                "bias_tau_j": float(np.mean(err_j)),
                "std_error_tau_j": float(np.std(err_j)),
                "target_std_tau_j": target_std,
                "prediction_std_tau_j": pred_std,
                "prediction_span_tau_j": pred_span,
                "target_span_tau_j": target_span,
                "corr_pred_target_j": safe_corr(pred_j, target_j),
                "constant_prediction_flag_j": pred_span <= PREDICTION_SPAN_EPS and pred_std <= PREDICTION_SPAN_EPS,
            }
        )

    rmse_by_joint = np.asarray([row["rmse_tau_j"] for row in per_joint], dtype=float)
    worst_index = int(np.argmax(rmse_by_joint))
    overall = {
        "model": model_label,
        "overall_rmse_tau": float(np.sqrt(np.mean(error**2))),
        "overall_mae_tau": float(np.mean(np.abs(error))),
        "mean_per_joint_rmse_tau": float(np.mean(rmse_by_joint)),
        "median_per_joint_rmse_tau": float(np.median(rmse_by_joint)),
        "worst_joint_by_rmse": int(per_joint[worst_index]["joint"]),
        "worst_joint_rmse_tau": float(rmse_by_joint[worst_index]),
        "non_constant_joint_count": int(
            sum(not bool(row["constant_prediction_flag_j"]) for row in per_joint)
        ),
        "constant_joint_count": int(sum(bool(row["constant_prediction_flag_j"]) for row in per_joint)),
    }
    return overall, per_joint


def support_metrics(train_dataset: dict[str, Any], eval_dataset: dict[str, Any], model_label: str) -> list[dict[str, Any]]:
    x_train = train_dataset["X"].astype(float)
    x_eval = eval_dataset["X"].astype(float)
    names = train_dataset["feature_columns"]
    train_mean = np.mean(x_train, axis=0)
    train_std = np.std(x_train, axis=0)
    train_std_safe = train_std.copy()
    train_std_safe[train_std_safe < 1e-12] = 1.0
    train_min = np.min(x_train, axis=0)
    train_max = np.max(x_train, axis=0)
    eval_min = np.min(x_eval, axis=0)
    eval_max = np.max(x_eval, axis=0)
    z_abs = np.abs((x_eval - train_mean) / train_std_safe)

    rows = []
    for idx, name in enumerate(names):
        outside_low = int(np.sum(x_eval[:, idx] < train_min[idx]))
        outside_high = int(np.sum(x_eval[:, idx] > train_max[idx]))
        rows.append(
            {
                "model": model_label,
                "dimension": idx,
                "feature": name,
                "train_mean": float(train_mean[idx]),
                "train_std": float(train_std[idx]),
                "train_min": float(train_min[idx]),
                "train_max": float(train_max[idx]),
                "eval_min": float(eval_min[idx]),
                "eval_max": float(eval_max[idx]),
                "outside_train_minmax_count": outside_low + outside_high,
                "outside_train_minmax_fraction": float((outside_low + outside_high) / x_eval.shape[0]),
                "mean_abs_standardized_distance": float(np.mean(z_abs[:, idx])),
                "p95_abs_standardized_distance": float(np.percentile(z_abs[:, idx], 95)),
                "max_abs_standardized_distance": float(np.max(z_abs[:, idx])),
                "is_q7_or_dq7": name in ("joint_pos_7", "joint_vel_7"),
            }
        )
    return rows


def model_inventory(model_dir: Path, model_label: str) -> dict[str, Any]:
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    local_files = sorted(model_dir.glob("joint*_local.pkl"))
    cloud_files = sorted(model_dir.glob("joint*_cloud.pkl"))
    return {
        "model": model_label,
        "model_dir": as_repo_relative(model_dir),
        "local_pickle_count": len(local_files),
        "cloud_pickle_count": len(cloud_files),
        "metadata_json_exists": metadata_path.is_file(),
        "dataset": metadata.get("dataset", ""),
        "X_shape": metadata.get("X_shape", ""),
        "Y_shape": metadata.get("Y_shape", ""),
        "max_samples": metadata.get("max_samples", ""),
        "fit_hparams": metadata.get("fit_hparams", ""),
        "x_std_floor": metadata.get("x_std_floor", ""),
        "caveat": metadata.get("caveat", ""),
    }


def make_plots(
    out_dir: Path,
    per_joint_rows: list[dict[str, Any]],
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    support_rows: list[dict[str, Any]],
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    joints = np.arange(1, 8)
    a_rmse = [row["rmse_tau_j"] for row in per_joint_rows if row["model"] == "GP_A_planar_train"]
    b_rmse = [row["rmse_tau_j"] for row in per_joint_rows if row["model"] == "GP_B_zmod_train"]
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(joints - width / 2, a_rmse, width, label="GP_A -> C")
    ax.bar(joints + width / 2, b_rmse, width, label="GP_B -> C")
    ax.set_xlabel("joint")
    ax.set_ylabel("RMSE tau residual")
    ax.set_title("Held-out C residual prediction RMSE")
    ax.set_xticks(joints)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "rmse_tau_per_joint_A_vs_B_on_C.png", dpi=160)
    plt.close(fig)

    for joint in (1, 4, 7):
        idx = joint - 1
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(target[:, idx], label="target", linewidth=1.0)
        ax.plot(predictions["GP_A_planar_train"][:, idx], label="GP_A prediction", linewidth=1.0)
        ax.plot(predictions["GP_B_zmod_train"][:, idx], label="GP_B prediction", linewidth=1.0)
        ax.set_xlabel("held-out C row")
        ax.set_ylabel(f"tau residual joint {joint}")
        ax.set_title(f"Prediction vs target joint {joint}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"prediction_vs_target_joint{joint}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, pred in predictions.items():
        errors = (pred - target).reshape(-1)
        ax.hist(errors, bins=60, alpha=0.55, label=label)
    ax.set_xlabel("prediction error")
    ax.set_ylabel("count")
    ax.set_title("Held-out C residual prediction error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "error_hist_A_vs_B.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["GP_A_planar_train", "GP_B_zmod_train"]
    max_vals = [
        max(row["max_abs_standardized_distance"] for row in support_rows if row["model"] == label)
        for label in labels
    ]
    p95_vals = [
        max(row["p95_abs_standardized_distance"] for row in support_rows if row["model"] == label)
        for label in labels
    ]
    x = np.arange(len(labels))
    ax.bar(x - width / 2, p95_vals, width, label="max dimension p95")
    ax.bar(x + width / 2, max_vals, width, label="max dimension max")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("absolute standardized distance")
    ax.set_title("C input distance from training support")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "support_distance_summary.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    a_span = [row["prediction_span_tau_j"] for row in per_joint_rows if row["model"] == "GP_A_planar_train"]
    b_span = [row["prediction_span_tau_j"] for row in per_joint_rows if row["model"] == "GP_B_zmod_train"]
    target_span = [row["target_span_tau_j"] for row in per_joint_rows if row["model"] == "GP_A_planar_train"]
    ax.bar(joints - width, a_span, width, label="GP_A prediction")
    ax.bar(joints, b_span, width, label="GP_B prediction")
    ax.bar(joints + width, target_span, width, label="target")
    ax.set_xlabel("joint")
    ax.set_ylabel("span")
    ax.set_title("Prediction and target span on held-out C")
    ax.set_xticks(joints)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "prediction_span_per_joint.png", dpi=160)
    plt.close(fig)


def summarize_support(support_rows: list[dict[str, Any]], model_label: str) -> dict[str, Any]:
    rows = [row for row in support_rows if row["model"] == model_label]
    worst = max(rows, key=lambda row: row["max_abs_standardized_distance"])
    outside = [row for row in rows if row["outside_train_minmax_count"] > 0]
    q7_rows = [row for row in rows if row["feature"] in ("joint_pos_7", "joint_vel_7")]
    return {
        "model": model_label,
        "worst_feature": worst["feature"],
        "worst_max_abs_standardized_distance": worst["max_abs_standardized_distance"],
        "max_p95_abs_standardized_distance": max(row["p95_abs_standardized_distance"] for row in rows),
        "mean_abs_standardized_distance": float(np.mean([row["mean_abs_standardized_distance"] for row in rows])),
        "outside_dimension_count": len(outside),
        "outside_features": ";".join(row["feature"] for row in outside),
        "q7_summary": "; ".join(
            f"{row['feature']}: max_z={format_float(row['max_abs_standardized_distance'])}, "
            f"outside={row['outside_train_minmax_count']}"
            for row in q7_rows
        ),
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(format_float(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_summary(
    out_dir: Path,
    dataset_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    overall_rows: list[dict[str, Any]],
    per_joint_rows: list[dict[str, Any]],
    comparison_row: dict[str, Any],
    support_summary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Stage 4 Cross-Trajectory Offline Residual Evaluation",
        "",
        "## Safety Boundary",
        "",
        "- This is offline residual prediction evaluation only.",
        "- It is not GP-on tracking validation and does not prove real-robot tracking improvement.",
        "- C held-out data was used only for evaluation, not training.",
        "- Any future GP-on real-robot step requires a separate conservative gate.",
        "",
        "## Dataset Inventory",
        "",
        *markdown_table(
            dataset_rows,
            [
                "dataset",
                "dataset_role",
                "trajectory_label",
                "rows_written",
                "rows_dropped_nonfinite",
                "nan_count",
                "inf_count",
            ],
        ),
        "",
        "## Training Setup",
        "",
        *markdown_table(
            model_rows,
            ["model", "model_dir", "local_pickle_count", "dataset", "max_samples", "fit_hparams", "x_std_floor"],
        ),
        "",
        "## Evaluation Setup",
        "",
        "- `GP_A_planar_train`: trained on A1 + A2 planar no-GP residual data.",
        "- `GP_B_zmod_train`: trained on B1 + B2 zmod no-GP residual data.",
        "- Both models were evaluated on the same held-out C1 + C2 zmod no-GP residual dataset.",
        "- Feature definition: `X = joint_pos_1..7 + joint_vel_1..7`.",
        "- Target definition: `Y = tau_residual_raw_1..7`.",
        "",
        "## Main Comparison",
        "",
        *markdown_table(
            overall_rows,
            [
                "model",
                "overall_rmse_tau",
                "overall_mae_tau",
                "mean_per_joint_rmse_tau",
                "median_per_joint_rmse_tau",
                "worst_joint_by_rmse",
                "worst_joint_rmse_tau",
                "non_constant_joint_count",
            ],
        ),
        "",
        *markdown_table(
            [comparison_row],
            ["RMSE_tau_A_to_C", "RMSE_tau_B_to_C", "delta_rmse_A_minus_B", "better_model_on_C"],
        ),
        "",
        "## Per-Joint RMSE",
        "",
        *markdown_table(
            per_joint_rows,
            [
                "model",
                "joint",
                "rmse_tau_j",
                "mae_tau_j",
                "bias_tau_j",
                "target_std_tau_j",
                "prediction_std_tau_j",
                "prediction_span_tau_j",
                "target_span_tau_j",
                "corr_pred_target_j",
                "constant_prediction_flag_j",
            ],
        ),
        "",
        "## Support Coverage Summary",
        "",
        *markdown_table(
            support_summary_rows,
            [
                "model",
                "worst_feature",
                "worst_max_abs_standardized_distance",
                "max_p95_abs_standardized_distance",
                "mean_abs_standardized_distance",
                "outside_dimension_count",
                "outside_features",
                "q7_summary",
            ],
        ),
        "",
        "## Prediction Non-Constant Check",
        "",
    ]

    for row in overall_rows:
        lines.append(
            f"- `{row['model']}`: non_constant_joint_count=`{row['non_constant_joint_count']}`, "
            f"constant_joint_count=`{row['constant_joint_count']}`."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- `delta_rmse = RMSE_A_on_C - RMSE_B_on_C = {format_float(comparison_row['delta_rmse_A_minus_B'])}`.",
            f"- Positive delta means `GP_B_zmod_train` is better on C. Result: `{comparison_row['better_model_on_C']}`.",
            "- Support metrics report C input coverage relative to each training dataset using absolute standardized distance and train min/max checks.",
            "- Near-constant prediction flags are warnings for residual models that may not be input-dependent on C.",
            "",
            "## Caveats",
            "",
            "- This evaluation uses no-GP residual CSVs and frozen offline models.",
            "- There is no online update in this evaluation.",
            "- C was held out from training.",
            "- This output must not be described as real-robot GP-on tracking improvement proof.",
            "- Future real-robot GP-on work needs a separate conservative gate and safety review.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    a_csvs = resolve_csvs(args.a_csv, DEFAULT_A_CSVS)
    b_csvs = resolve_csvs(args.b_csv, DEFAULT_B_CSVS)
    c_csvs = resolve_csvs(args.c_csv, DEFAULT_C_CSVS)

    dataset_paths = {
        "GP_A_planar_train": args.dataset_dir / "GP_A_planar_train.npz",
        "GP_B_zmod_train": args.dataset_dir / "GP_B_zmod_train.npz",
        "GP_C_zmod_heldout_eval": args.dataset_dir / "GP_C_zmod_heldout_eval.npz",
    }
    model_dirs = {
        "GP_A_planar_train": args.model_base_dir / "GP_A_planar_train",
        "GP_B_zmod_train": args.model_base_dir / "GP_B_zmod_train",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_meta = {
        "GP_A_planar_train": build_combined_dataset(
            a_csvs,
            dataset_paths["GP_A_planar_train"],
            "train",
            "A_no_gp_planar",
            "GP_A_planar_train",
            6000,
            args.drop_nonfinite,
            args.overwrite,
        ),
        "GP_B_zmod_train": build_combined_dataset(
            b_csvs,
            dataset_paths["GP_B_zmod_train"],
            "train",
            "B_no_gp_zmod",
            "GP_B_zmod_train",
            6001,
            args.drop_nonfinite,
            args.overwrite,
        ),
        "GP_C_zmod_heldout_eval": build_combined_dataset(
            c_csvs,
            dataset_paths["GP_C_zmod_heldout_eval"],
            "heldout_eval",
            "C_no_gp_zmod_heldout",
            "GP_C_zmod_heldout_eval",
            6000,
            args.drop_nonfinite,
            args.overwrite,
        ),
    }

    if not args.skip_training:
        for model_label in ("GP_A_planar_train", "GP_B_zmod_train"):
            run_training(
                dataset_paths[model_label],
                model_dirs[model_label],
                model_label,
                args.overwrite,
                args.train_max_samples,
            )

    datasets = {label: load_npz_dataset(path) for label, path in dataset_paths.items()}
    c_dataset = datasets["GP_C_zmod_heldout_eval"]
    predictions = {}
    prediction_indices: np.ndarray | None = None
    for model_label in ("GP_A_planar_train", "GP_B_zmod_train"):
        prediction, indices = predict_model_on_x(model_dirs[model_label], c_dataset["X"], args.max_prediction_rows)
        predictions[model_label] = prediction
        if prediction_indices is None:
            prediction_indices = indices
        elif not np.array_equal(prediction_indices, indices):
            raise RuntimeError("Prediction index mismatch between models")

    assert prediction_indices is not None
    target_eval = c_dataset["Y"][prediction_indices].astype(float)

    overall_rows = []
    per_joint_rows = []
    for model_label in ("GP_A_planar_train", "GP_B_zmod_train"):
        overall, per_joint = evaluate_predictions(model_label, predictions[model_label], target_eval)
        overall_rows.append(overall)
        per_joint_rows.extend(per_joint)

    rmse_a = next(row["overall_rmse_tau"] for row in overall_rows if row["model"] == "GP_A_planar_train")
    rmse_b = next(row["overall_rmse_tau"] for row in overall_rows if row["model"] == "GP_B_zmod_train")
    delta = rmse_a - rmse_b
    better = "GP_B_zmod_train" if delta > 0 else "GP_A_planar_train" if delta < 0 else "tie"
    comparison_row = {
        "RMSE_tau_A_to_C": rmse_a,
        "RMSE_tau_B_to_C": rmse_b,
        "delta_rmse_A_minus_B": delta,
        "better_model_on_C": better,
    }

    support_rows = []
    support_rows.extend(support_metrics(datasets["GP_A_planar_train"], c_dataset, "GP_A_planar_train"))
    support_rows.extend(support_metrics(datasets["GP_B_zmod_train"], c_dataset, "GP_B_zmod_train"))
    support_summary_rows = [
        summarize_support(support_rows, "GP_A_planar_train"),
        summarize_support(support_rows, "GP_B_zmod_train"),
    ]

    dataset_rows = []
    for label, metadata in dataset_meta.items():
        dataset_rows.append(
            {
                "dataset": label,
                "dataset_role": metadata["dataset_role"],
                "trajectory_label": metadata["trajectory_label"],
                "source_csvs": ";".join(metadata["source_csvs"]),
                "source_rows_total": metadata["source_rows_total"],
                "rows_written": metadata["rows_written"],
                "rows_dropped_nonfinite": metadata["rows_dropped_nonfinite"],
                "nan_count": metadata["output_nan_count"],
                "inf_count": metadata["output_inf_count"],
                "npz": metadata["out_npz"],
            }
        )

    model_rows = [
        model_inventory(model_dirs["GP_A_planar_train"], "GP_A_planar_train"),
        model_inventory(model_dirs["GP_B_zmod_train"], "GP_B_zmod_train"),
    ]

    write_csv(
        args.out_dir / "dataset_inventory.csv",
        dataset_rows,
        [
            "dataset",
            "dataset_role",
            "trajectory_label",
            "source_rows_total",
            "rows_written",
            "rows_dropped_nonfinite",
            "nan_count",
            "inf_count",
            "source_csvs",
            "npz",
        ],
    )
    write_csv(
        args.out_dir / "model_inventory.csv",
        model_rows,
        [
            "model",
            "model_dir",
            "local_pickle_count",
            "cloud_pickle_count",
            "metadata_json_exists",
            "dataset",
            "X_shape",
            "Y_shape",
            "max_samples",
            "fit_hparams",
            "x_std_floor",
            "caveat",
        ],
    )
    write_csv(
        args.out_dir / "metrics_overall.csv",
        overall_rows,
        [
            "model",
            "overall_rmse_tau",
            "overall_mae_tau",
            "mean_per_joint_rmse_tau",
            "median_per_joint_rmse_tau",
            "worst_joint_by_rmse",
            "worst_joint_rmse_tau",
            "non_constant_joint_count",
            "constant_joint_count",
        ],
    )
    write_csv(
        args.out_dir / "metrics_per_joint.csv",
        per_joint_rows,
        [
            "model",
            "joint",
            "rmse_tau_j",
            "mae_tau_j",
            "bias_tau_j",
            "std_error_tau_j",
            "target_std_tau_j",
            "prediction_std_tau_j",
            "prediction_span_tau_j",
            "target_span_tau_j",
            "corr_pred_target_j",
            "constant_prediction_flag_j",
        ],
    )
    write_csv(
        args.out_dir / "support_metrics.csv",
        support_rows,
        [
            "model",
            "dimension",
            "feature",
            "train_mean",
            "train_std",
            "train_min",
            "train_max",
            "eval_min",
            "eval_max",
            "outside_train_minmax_count",
            "outside_train_minmax_fraction",
            "mean_abs_standardized_distance",
            "p95_abs_standardized_distance",
            "max_abs_standardized_distance",
            "is_q7_or_dq7",
        ],
    )
    write_csv(
        args.out_dir / "comparison_A_vs_B_on_C.csv",
        [comparison_row],
        ["RMSE_tau_A_to_C", "RMSE_tau_B_to_C", "delta_rmse_A_minus_B", "better_model_on_C"],
    )
    make_plots(args.out_dir, per_joint_rows, predictions, target_eval, support_rows)
    write_summary(
        args.out_dir,
        dataset_rows,
        model_rows,
        overall_rows,
        per_joint_rows,
        comparison_row,
        support_summary_rows,
    )

    return {
        "dataset_rows": dataset_rows,
        "model_rows": model_rows,
        "overall_rows": overall_rows,
        "per_joint_rows": per_joint_rows,
        "comparison_row": comparison_row,
        "support_summary_rows": support_summary_rows,
    }


def main() -> int:
    args = parse_args()
    if not args.run_all:
        print("Nothing to do. Pass --run-all to build datasets, train models, and evaluate.")
        return 0
    if args.max_prediction_rows < 0:
        raise ValueError("--max-prediction-rows must be >= 0")
    if args.train_max_samples is not None and args.train_max_samples < 1:
        raise ValueError("--train-max-samples must be >= 1 when provided")
    result = run_pipeline(args)
    comparison = result["comparison_row"]
    print(f"wrote_outputs: {args.out_dir}")
    print(f"RMSE_tau_A_to_C: {format_float(comparison['RMSE_tau_A_to_C'])}")
    print(f"RMSE_tau_B_to_C: {format_float(comparison['RMSE_tau_B_to_C'])}")
    print(f"delta_rmse_A_minus_B: {format_float(comparison['delta_rmse_A_minus_B'])}")
    print(f"better_model_on_C: {comparison['better_model_on_C']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
