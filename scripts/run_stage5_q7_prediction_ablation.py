#!/usr/bin/env python3
"""Offline Stage 5A q7/dq7 residual prediction ablation.

This script retrains per-joint local GP models with either the normal 14D
input or a 12D input that excludes joint_pos_7 and joint_vel_7, then evaluates
tau1..tau6 residual prediction on held-out CSVs. It is offline-only: it does
not import ROS, connect to Franka, launch controllers, or modify runtime
control behavior.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import dataclass
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

import train_stage4_matched_frozen_gp as stage4_train


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("outputs/stage5_q7_prediction_ablation")
DEFAULT_STAGE4_BASE = Path("data/stage4/cross_traj")
DEFAULT_MODEL_DIRS = {
    "GP_A_planar_train": DEFAULT_STAGE4_BASE / "models/GP_A_planar_train",
    "GP_B_zmod_train": DEFAULT_STAGE4_BASE / "models/GP_B_zmod_train",
}
DEFAULT_C_GLOB = DEFAULT_STAGE4_BASE / "C_no_gp_zmod_heldout/usable_runs/*.csv"
SUMMARY_CSV = "stage5_q7_prediction_ablation.csv"
SUMMARY_MD = "stage5_q7_prediction_ablation.md"
METADATA_JSON = "metadata.json"
SCRIPT_VERSION = "2026-05-29-stage5a-q7-prediction-ablation-v1"
PREDICTION_STD_EPS = 1e-9
JOINTS_ALL = range(1, 8)


@dataclass(frozen=True)
class TrainSpec:
    label: str
    csv_paths: tuple[Path, ...]
    source: str


@dataclass(frozen=True)
class EvalSpec:
    label: str
    csv_paths: tuple[Path, ...]


class Stage5PredictionAblationParser(argparse.ArgumentParser):
    """Use exit code 1 for invalid CLI usage."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def parse_args() -> argparse.Namespace:
    parser = Stage5PredictionAblationParser(
        description="Offline Stage 5A q7/dq7 residual prediction ablation.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite this script's generated outputs.")
    parser.add_argument("--train-max-samples", type=int, default=2000, help="Default: 2000")
    parser.add_argument("--random-seed", type=int, default=0, help="Default: 0")
    parser.add_argument("--auto-stage4", action="store_true", help="Use GP_A/GP_B Stage 4 metadata and C held-out CSVs.")
    parser.add_argument("--candidate-csv", action="append", type=Path, default=[], help="Evaluation CSV. May repeat.")
    parser.add_argument("--eval-csv", action="append", type=Path, default=[], help="Alias for --candidate-csv.")
    parser.add_argument("--eval-label", default="", help="Optional label for a single manual eval CSV or aggregate.")
    parser.add_argument("--target-joints", type=int, default=6, help="Evaluate tau1..tauN. Default: 6")
    parser.add_argument(
        "--include-raw-residual",
        action="store_true",
        help="Also evaluate tau_residual_raw columns when tau_residual columns are available.",
    )
    parser.add_argument("--mode", choices=("prediction",), default="prediction", help="Default: prediction")
    parser.add_argument("--force", action="store_true", help="Alias for --overwrite.")
    parser.add_argument(
        "--reference-model-dir",
        action="append",
        type=Path,
        default=[],
        help="Manual model dir with metadata.json used only to resolve source_csv. May repeat.",
    )
    parser.add_argument(
        "--reference-label",
        action="append",
        default=[],
        help="Label for manual --reference-model-dir or --train-csv group. May repeat.",
    )
    parser.add_argument("--train-csv", action="append", type=Path, default=[], help="Manual train CSV. May repeat.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_csv_numeric(path: Path) -> dict[str, Any]:
    csv_path = resolve_path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(path)
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no CSV header found")
        columns = list(reader.fieldnames)
        data: dict[str, list[float]] = {column: [] for column in columns}
        for row in reader:
            for column in columns:
                data[column].append(parse_float(row.get(column)))
    rows = len(next(iter(data.values()))) if data else 0
    if rows == 0:
        raise ValueError(f"{path}: no data rows found")
    arrays = {column: np.asarray(values, dtype=np.float32) for column, values in data.items()}
    return {"path": csv_path, "columns": columns, "data": arrays, "rows": rows}


def require_columns(dataset: dict[str, Any], columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in dataset["columns"]]
    if missing:
        raise KeyError(f"{label} missing required columns: {', '.join(missing)}")


def stack_columns(dataset: dict[str, Any], columns: list[str], label: str) -> np.ndarray:
    require_columns(dataset, columns, label)
    return np.stack([dataset["data"][column] for column in columns], axis=1).astype(np.float32)


def feature_columns(input_mode: str) -> list[str]:
    if input_mode == "14d":
        return [f"joint_pos_{joint}" for joint in JOINTS_ALL] + [f"joint_vel_{joint}" for joint in JOINTS_ALL]
    if input_mode == "12d_without_q7_dq7":
        joints = range(1, 7)
        return [f"joint_pos_{joint}" for joint in joints] + [f"joint_vel_{joint}" for joint in joints]
    raise ValueError(f"unknown input mode: {input_mode}")


def target_columns_for_kind(kind: str, target_joints: int) -> list[str]:
    return [f"{kind}_{joint}" for joint in range(1, target_joints + 1)]


def available_target_kinds(datasets: list[dict[str, Any]], target_joints: int, include_raw: bool) -> list[str]:
    preferred = "tau_residual"
    raw = "tau_residual_raw"
    preferred_cols = target_columns_for_kind(preferred, target_joints)
    raw_cols = target_columns_for_kind(raw, target_joints)
    have_preferred = all(all(column in dataset["columns"] for column in preferred_cols) for dataset in datasets)
    have_raw = all(all(column in dataset["columns"] for column in raw_cols) for dataset in datasets)
    if have_preferred:
        kinds = [preferred]
        if include_raw and have_raw:
            kinds.append(raw)
        return kinds
    if have_raw:
        return [raw]
    raise KeyError(
        f"missing target columns: expected {preferred_cols[0]}.. or {raw_cols[0]}.. through joint {target_joints}"
    )


def rows_with_finite_xy(x_matrix: np.ndarray, y_matrix: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(x_matrix), axis=1) & np.all(np.isfinite(y_matrix), axis=1)


def combine_csvs(
    csv_paths: tuple[Path, ...],
    input_mode: str,
    target_kind: str,
    target_joints: int,
    role: str,
) -> dict[str, Any]:
    features = feature_columns(input_mode)
    targets = target_columns_for_kind(target_kind, target_joints)
    datasets = [load_csv_numeric(path) for path in csv_paths]
    x_parts = []
    y_parts = []
    source_rows = []
    dropped = []
    for dataset in datasets:
        label = repo_relative(dataset["path"])
        x_matrix = stack_columns(dataset, features, f"{label} features")
        y_matrix = stack_columns(dataset, targets, f"{label} targets")
        keep = rows_with_finite_xy(x_matrix, y_matrix)
        x_parts.append(x_matrix[keep])
        y_parts.append(y_matrix[keep])
        source_rows.append(int(dataset["rows"]))
        dropped.append(int(keep.size - np.sum(keep)))
    x_all = np.vstack(x_parts).astype(np.float32)
    y_all = np.vstack(y_parts).astype(np.float32)
    if x_all.shape[0] == 0:
        raise ValueError(f"{role}: no finite rows after loading {len(csv_paths)} CSV(s)")
    return {
        "X": x_all,
        "Y": y_all,
        "feature_columns": features,
        "target_columns": targets,
        "target_kind": target_kind,
        "source_rows": source_rows,
        "rows_dropped_nonfinite": dropped,
        "csv_paths": csv_paths,
    }


def subset_training(
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    max_samples: int | None,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if max_samples is None or max_samples <= 0 or max_samples >= x_matrix.shape[0]:
        return x_matrix, y_matrix, None
    rng = np.random.default_rng(random_seed)
    indices = np.sort(rng.choice(x_matrix.shape[0], size=max_samples, replace=False))
    return x_matrix[indices], y_matrix[indices], indices


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
    k_star = model.kernel_np(x_snapshot, x_std.T, lengthscale, sigma_f)
    return np.asarray(k_star.T @ alpha, dtype=float).reshape(-1)


def train_models(
    trainer: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    target_joints: int,
) -> dict[int, dict[str, Any]]:
    models = {}
    for joint in range(1, target_joints + 1):
        print(f"training joint {joint}: X={x_train.shape} y={y_train[:, joint - 1].shape}", flush=True)
        models[joint] = stage4_train.train_joint(
            trainer,
            x_train,
            y_train[:, joint - 1],
            joint,
            fit_hparams=False,
            x_std_floor=stage4_train.DEFAULT_X_STD_FLOOR,
        )
    return models


def predict_models(models: dict[int, dict[str, Any]], x_eval: np.ndarray, target_joints: int) -> np.ndarray:
    predictions = np.empty((x_eval.shape[0], target_joints), dtype=np.float64)
    for joint in range(1, target_joints + 1):
        result = models[joint]
        model = copy.deepcopy(result["local_model"])
        reset_prediction_state(model)
        x_mean, x_std, y_mean, y_std = result["stats"]
        x_mean = np.asarray(x_mean, dtype=float).reshape(-1)
        x_std = np.asarray(x_std, dtype=float).reshape(-1)
        x_std[x_std == 0.0] = 1.0
        x_norm = (x_eval - x_mean) / x_std
        y_mean_scalar = first_scalar(y_mean, 0.0)
        y_std_scalar = first_scalar(y_std, 1.0)
        if y_std_scalar == 0.0:
            y_std_scalar = 1.0
        mu_std_values = fast_single_expert_mean_std(model, x_norm)
        if mu_std_values is None:
            values = []
            for row in x_norm:
                mu_std, _ = model.predict(np.asarray(row, dtype=np.float32))
                values.append(first_scalar(mu_std))
            mu_std_values = np.asarray(values, dtype=float)
        predictions[:, joint - 1] = mu_std_values * y_std_scalar + y_mean_scalar
        print(f"predicted joint {joint}: rows={x_eval.shape[0]}", flush=True)
    return predictions


def evaluate_prediction(
    prediction: np.ndarray,
    target: np.ndarray,
    target_joints: int,
) -> dict[str, Any]:
    error = prediction - target
    row: dict[str, Any] = {
        "overall_rmse_tau1_to_tau6": float(np.sqrt(np.mean(error**2))),
    }
    constant_joints = []
    for joint in range(1, target_joints + 1):
        idx = joint - 1
        pred_j = prediction[:, idx]
        err_j = error[:, idx]
        pred_std = float(np.std(pred_j))
        row[f"rmse_tau{joint}"] = float(np.sqrt(np.mean(err_j**2)))
        row[f"prediction_std_tau{joint}"] = pred_std
        if pred_std <= PREDICTION_STD_EPS:
            constant_joints.append(f"tau{joint}")
    for joint in range(target_joints + 1, 7):
        row[f"rmse_tau{joint}"] = ""
        row[f"prediction_std_tau{joint}"] = ""
    row["constant_prediction_joints"] = ";".join(constant_joints)
    return row


def read_metadata_source_csvs(model_dir: Path) -> tuple[str, tuple[Path, ...]]:
    metadata_path = resolve_path(model_dir) / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.json not found: {model_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_csv = str(metadata.get("source_csv", "")).strip()
    if not source_csv:
        raise ValueError(f"{metadata_path}: missing source_csv")
    paths = tuple(Path(part.strip()) for part in source_csv.split(";") if part.strip())
    label = str(metadata.get("model_name") or metadata.get("source_mode_name") or resolve_path(model_dir).name)
    return label, paths


def csv_sort_key(path: Path) -> tuple[str, str]:
    name = path.name
    digits = "".join(ch for ch in name if ch.isdigit())
    return (digits, name)


def discover_c_csvs() -> list[Path]:
    paths = [path.relative_to(REPO_ROOT) for path in REPO_ROOT.glob(str(DEFAULT_C_GLOB)) if path.is_file()]
    return sorted(paths, key=csv_sort_key)


def manual_label(labels: list[str], index: int, fallback: str) -> str:
    if index < len(labels) and labels[index].strip():
        return labels[index].strip()
    return fallback


def discover_train_specs(args: argparse.Namespace) -> list[TrainSpec]:
    specs: list[TrainSpec] = []
    if args.auto_stage4:
        for fallback_label, model_dir in DEFAULT_MODEL_DIRS.items():
            label, source_csvs = read_metadata_source_csvs(model_dir)
            specs.append(TrainSpec(label=label or fallback_label, csv_paths=source_csvs, source=repo_relative(resolve_path(model_dir))))
    for index, model_dir in enumerate(args.reference_model_dir):
        label, source_csvs = read_metadata_source_csvs(model_dir)
        label = manual_label(args.reference_label, index, label)
        specs.append(TrainSpec(label=label, csv_paths=source_csvs, source=repo_relative(resolve_path(model_dir))))
    if args.train_csv:
        label = manual_label(args.reference_label, len(args.reference_model_dir), "manual_train")
        specs.append(TrainSpec(label=label, csv_paths=tuple(args.train_csv), source="manual_train_csv"))
    return specs


def discover_eval_specs(args: argparse.Namespace) -> list[EvalSpec]:
    provided = list(args.candidate_csv) + list(args.eval_csv)
    specs: list[EvalSpec] = []
    if args.auto_stage4 and not provided:
        c_csvs = discover_c_csvs()
        for index, csv_path in enumerate(c_csvs, start=1):
            specs.append(EvalSpec(label=f"C{index}_heldout_zmod", csv_paths=(csv_path,)))
        if len(c_csvs) > 1:
            specs.append(EvalSpec(label="C_all_heldout_zmod", csv_paths=tuple(c_csvs)))
    elif provided:
        for index, csv_path in enumerate(provided, start=1):
            label = args.eval_label if args.eval_label and len(provided) == 1 else f"eval{index}_{csv_path.stem}"
            specs.append(EvalSpec(label=label, csv_paths=(csv_path,)))
        if len(provided) > 1:
            specs.append(EvalSpec(label=args.eval_label or "eval_all", csv_paths=tuple(provided)))
    return specs


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    path = resolve_path(path)
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(path)
    csv_path = path / SUMMARY_CSV
    md_path = path / SUMMARY_MD
    metadata_path = path / METADATA_JSON
    if not overwrite and any(item.exists() for item in (csv_path, md_path, metadata_path)):
        raise FileExistsError(f"{path} already has generated outputs; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def add_drop_effects(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (str(row["train_label"]), str(row["eval_label"]), str(row["target_kind"]))
        groups.setdefault(key, {})[str(row["input_mode"])] = row
    for modes in groups.values():
        row_14 = modes.get("14d")
        row_12 = modes.get("12d_without_q7_dq7")
        if not row_14 or not row_12:
            continue
        delta = float(row_12["overall_rmse_tau1_to_tau6"]) - float(row_14["overall_rmse_tau1_to_tau6"])
        if abs(delta) < 1e-6:
            interpretation = "near_zero"
        elif delta > 0.0:
            interpretation = "14d_better"
        else:
            interpretation = "12d_better"
        for row in (row_14, row_12):
            row["delta_rmse_12d_minus_14d"] = delta
            row["drop_effect_interpretation"] = interpretation


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


def main_findings(rows: list[dict[str, Any]]) -> list[str]:
    aggregate = [
        row for row in rows
        if row.get("status") == "ok" and str(row.get("eval_label", "")).endswith("all_heldout_zmod")
    ]
    if not aggregate:
        aggregate = [row for row in rows if row.get("status") == "ok"]
    findings = []
    seen: set[tuple[str, str]] = set()
    for row in aggregate:
        key = (str(row["train_label"]), str(row["target_kind"]))
        if key in seen or row.get("input_mode") != "14d":
            continue
        seen.add(key)
        delta = row.get("delta_rmse_12d_minus_14d", "")
        interpretation = row.get("drop_effect_interpretation", "")
        findings.append(
            f"- `{row['train_label']}` `{row['target_kind']}`: "
            f"`delta_rmse_12d_minus_14d={format_float(delta)}` ({interpretation})."
        )
    deltas = {
        str(row["train_label"]): float(row["delta_rmse_12d_minus_14d"])
        for row in aggregate
        if row.get("status") == "ok"
        and row.get("input_mode") == "14d"
        and row.get("delta_rmse_12d_minus_14d") not in ("", None)
    }
    if "GP_A_planar_train" in deltas and "GP_B_zmod_train" in deltas:
        delta_a = deltas["GP_A_planar_train"]
        delta_b = deltas["GP_B_zmod_train"]
        if delta_a > 0.0 and delta_b > 0.0 and delta_b > delta_a:
            findings.append(
                "- In this offline setup, q7/dq7 appears more useful for the zmod-trained model than the planar-trained model."
            )
        elif delta_a > 0.0 and delta_b > 0.0 and delta_a > delta_b:
            findings.append(
                "- In this offline setup, q7/dq7 appears more useful for the planar-trained model than the zmod-trained model."
            )
        elif delta_b < 0.0 <= delta_a:
            findings.append(
                "- In this offline setup, dropping q7/dq7 improves the zmod-trained model and does not change the planar-trained model; this is not evidence that q7/dq7 is useful for zmod prediction here."
            )
        elif delta_a < 0.0 <= delta_b:
            findings.append(
                "- In this offline setup, dropping q7/dq7 improves the planar-trained model and does not improve the zmod-trained model."
            )
        elif delta_a < 0.0 and delta_b < 0.0:
            findings.append(
                "- In this offline setup, dropping q7/dq7 improves both aggregate models; compare delta magnitude rather than treating q7/dq7 as a useful feature."
            )
        else:
            findings.append("- The planar and zmod drop effects are equal within the recorded precision.")
    constant = [row for row in aggregate if row.get("constant_prediction_joints")]
    if constant:
        findings.append("- Constant prediction warnings were detected; inspect `constant_prediction_joints` before relying on RMSE.")
    else:
        findings.append("- No constant prediction joints were detected with the configured prediction-std threshold.")
    return findings or ["- No completed prediction rows were available for interpretation."]


def write_markdown(
    output_dir: Path,
    rows: list[dict[str, Any]],
    train_specs: list[TrainSpec],
    eval_specs: list[EvalSpec],
    target_kinds: list[str],
    target_joints: int,
) -> None:
    lines = [
        "# Stage 5A q7/dq7 Prediction Ablation Report",
        "",
        "## 1. Purpose",
        "",
        "This is an offline residual prediction ablation. It compares 14D input against a 12D input that removes `joint_pos_7` and `joint_vel_7` for `tau1..tau6` residual prediction.",
        "",
        "It is not GP-on validation and does not authorize any controller change.",
        "",
        "## 2. Inputs",
        "",
        f"- target_joints: `{target_joints}`",
        f"- target_kinds: `{';'.join(target_kinds)}`",
        "- input modes: `14d`, `12d_without_q7_dq7`",
        "",
        "### Train CSVs",
        "",
    ]
    for spec in train_specs:
        lines.append(f"- `{spec.label}` from `{spec.source}`")
        for path in spec.csv_paths:
            lines.append(f"  - `{repo_relative(resolve_path(path))}`")
    lines.extend(["", "### Eval CSVs", ""])
    for spec in eval_specs:
        lines.append(f"- `{spec.label}`")
        for path in spec.csv_paths:
            lines.append(f"  - `{repo_relative(resolve_path(path))}`")
    lines.extend(
        [
            "",
            "## 3. Summary Table",
            "",
            *markdown_table(
                rows,
                [
                    "experiment_id",
                    "train_label",
                    "input_mode",
                    "eval_label",
                    "target_kind",
                    "train_rows",
                    "eval_rows",
                    "overall_rmse_tau1_to_tau6",
                    "delta_rmse_12d_minus_14d",
                    "drop_effect_interpretation",
                    "constant_prediction_joints",
                    "status",
                ],
            ),
            "",
            "## 4. Main Findings",
            "",
            *main_findings(rows),
            "",
            "Positive `delta_rmse_12d_minus_14d` means 14D is better; near zero means the q7/dq7 contribution is unclear; negative means 12D is better in this offline metric.",
            "",
            "## 5. Safety Notes",
            "",
            "- This does not authorize GP-on.",
            "- This does not modify controller.",
            "- A 12D model would require separate training and review.",
            "- Do not use these results to directly remove q7 from runtime GP without review.",
            "- Do not enable no-clip, high-scale, or online-update behavior from this report.",
            "- RMSE improvement here is not real-robot tracking improvement proof.",
            "",
        ]
    )
    (output_dir / SUMMARY_MD).write_text("\n".join(lines), encoding="utf-8")


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    train_specs: list[TrainSpec],
    eval_specs: list[EvalSpec],
    target_kinds: list[str],
) -> None:
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": repo_relative(Path(__file__)),
        "script_version": SCRIPT_VERSION,
        "mode": args.mode,
        "train_max_samples": args.train_max_samples,
        "random_seed": args.random_seed,
        "auto_stage4": bool(args.auto_stage4),
        "target_joints": args.target_joints,
        "target_kinds": target_kinds,
        "input_modes": ["14d", "12d_without_q7_dq7"],
        "train_specs": [
            {
                "label": spec.label,
                "source": spec.source,
                "csv_paths": [repo_relative(resolve_path(path)) for path in spec.csv_paths],
            }
            for spec in train_specs
        ],
        "eval_specs": [
            {
                "label": spec.label,
                "csv_paths": [repo_relative(resolve_path(path)) for path in spec.csv_paths],
            }
            for spec in eval_specs
        ],
        "safety_boundary": "offline residual prediction only; no controller, launch, config, torque path, ROS launch, or Franka connection",
    }
    (output_dir / METADATA_JSON).write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.train_max_samples is not None and args.train_max_samples < 1:
        raise ValueError("--train-max-samples must be >= 1")
    if args.target_joints < 1 or args.target_joints > 6:
        raise ValueError("--target-joints must be in [1, 6] for this Stage 5A task")

    overwrite = bool(args.overwrite or args.force)
    output_dir = resolve_path(args.output_dir)
    ensure_output_dir(output_dir, overwrite)

    train_specs = discover_train_specs(args)
    eval_specs = discover_eval_specs(args)
    if not train_specs:
        raise ValueError("no train specs found; pass --auto-stage4, --reference-model-dir, or --train-csv")
    if not eval_specs:
        raise ValueError("no eval specs found; pass --auto-stage4, --candidate-csv, or --eval-csv")

    target_probe_datasets = [load_csv_numeric(path) for spec in train_specs + eval_specs for path in spec.csv_paths]
    target_kinds = available_target_kinds(target_probe_datasets, args.target_joints, args.include_raw_residual)

    trainer = stage4_train.ensure_training_imports()
    rows: list[dict[str, Any]] = []
    input_modes = ("14d", "12d_without_q7_dq7")
    for target_kind in target_kinds:
        for train_spec in train_specs:
            for input_mode in input_modes:
                train_data = combine_csvs(train_spec.csv_paths, input_mode, target_kind, args.target_joints, "train")
                x_train, y_train, subset_indices = subset_training(
                    train_data["X"],
                    train_data["Y"],
                    args.train_max_samples,
                    args.random_seed,
                )
                print(
                    f"training {train_spec.label} {input_mode} {target_kind}: "
                    f"rows={x_train.shape[0]} features={x_train.shape[1]}",
                    flush=True,
                )
                models = train_models(trainer, x_train, y_train, args.target_joints)
                for eval_spec in eval_specs:
                    experiment_id = f"{train_spec.label}_{input_mode}_to_{eval_spec.label}_{target_kind}"
                    try:
                        eval_data = combine_csvs(eval_spec.csv_paths, input_mode, target_kind, args.target_joints, "eval")
                        prediction = predict_models(models, eval_data["X"], args.target_joints)
                        metrics = evaluate_prediction(prediction, eval_data["Y"].astype(float), args.target_joints)
                        row = {
                            "experiment_id": experiment_id,
                            "train_label": train_spec.label,
                            "input_mode": input_mode,
                            "eval_label": eval_spec.label,
                            "target_kind": target_kind,
                            "target_joints": args.target_joints,
                            "train_rows": int(x_train.shape[0]),
                            "eval_rows": int(eval_data["X"].shape[0]),
                            "train_source_rows": int(train_data["X"].shape[0]),
                            "eval_source_rows": int(eval_data["X"].shape[0]),
                            "train_csvs": ";".join(repo_relative(resolve_path(path)) for path in train_spec.csv_paths),
                            "eval_csvs": ";".join(repo_relative(resolve_path(path)) for path in eval_spec.csv_paths),
                            "feature_columns": ";".join(train_data["feature_columns"]),
                            "target_columns": ";".join(train_data["target_columns"]),
                            "subset_indices_count": 0 if subset_indices is None else int(subset_indices.size),
                            "status": "ok",
                            "notes": "offline_gp_prediction_ablation",
                            **metrics,
                        }
                    except Exception as exc:  # noqa: BLE001 - report per experiment and keep other rows.
                        row = {
                            "experiment_id": experiment_id,
                            "train_label": train_spec.label,
                            "input_mode": input_mode,
                            "eval_label": eval_spec.label,
                            "target_kind": target_kind,
                            "target_joints": args.target_joints,
                            "train_rows": int(x_train.shape[0]),
                            "eval_rows": "",
                            "status": "skipped",
                            "notes": str(exc),
                        }
                    rows.append(row)

    add_drop_effects(rows)
    fieldnames = [
        "experiment_id",
        "train_label",
        "input_mode",
        "eval_label",
        "target_kind",
        "target_joints",
        "train_rows",
        "eval_rows",
        "overall_rmse_tau1_to_tau6",
        "rmse_tau1",
        "rmse_tau2",
        "rmse_tau3",
        "rmse_tau4",
        "rmse_tau5",
        "rmse_tau6",
        "prediction_std_tau1",
        "prediction_std_tau2",
        "prediction_std_tau3",
        "prediction_std_tau4",
        "prediction_std_tau5",
        "prediction_std_tau6",
        "constant_prediction_joints",
        "delta_rmse_12d_minus_14d",
        "drop_effect_interpretation",
        "status",
        "notes",
        "train_source_rows",
        "eval_source_rows",
        "subset_indices_count",
        "train_csvs",
        "eval_csvs",
        "feature_columns",
        "target_columns",
    ]
    write_csv(output_dir / SUMMARY_CSV, rows, fieldnames)
    write_markdown(output_dir, rows, train_specs, eval_specs, target_kinds, args.target_joints)
    write_metadata(output_dir, args, train_specs, eval_specs, target_kinds)
    return rows


def main() -> int:
    args = parse_args()
    rows = run(args)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    print(f"wrote_csv: {args.output_dir / SUMMARY_CSV}")
    print(f"wrote_markdown: {args.output_dir / SUMMARY_MD}")
    print(f"rows_total: {len(rows)}")
    print(f"rows_ok: {len(ok_rows)}")
    for row in ok_rows:
        if row.get("eval_label") == "C_all_heldout_zmod":
            print(
                f"{row['train_label']} {row['input_mode']} "
                f"rmse={format_float(row['overall_rmse_tau1_to_tau6'])} "
                f"delta={format_float(row.get('delta_rmse_12d_minus_14d', ''))}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
