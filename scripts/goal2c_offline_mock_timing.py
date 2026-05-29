#!/usr/bin/env python3
"""GOAL2 C offline/mock GP timing benchmark.

This script is offline-only. It does not import ROS, start nodes, launch
controllers, connect to Franka, or instantiate CartesianImpedanceController.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as exc:
    print("Missing Python dependency: numpy", file=sys.stderr)
    print("Use an environment that already has project dependencies installed.", file=sys.stderr)
    raise SystemExit(1) from exc

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    pd = None
    PANDAS_IMPORT_ERROR = str(exc)
else:
    PANDAS_IMPORT_ERROR = ""


REPO_ROOT = Path(__file__).resolve().parents[1]
JOINTS = range(1, 8)
DEFAULT_MODEL_DIR = Path("new_structure/gp/gp_models")
DEFAULT_OUTPUT_DIR = Path("outputs/goal2c_offline_mock_timing")
RECORDS_CSV = "goal2c_timing_records.csv"
SUMMARY_CSV = "goal2c_timing_summary.csv"
SUMMARY_MD = "goal2c_timing_summary.md"
RECORD_FIELDS = [
    "timestamp",
    "benchmark",
    "joint",
    "sample_idx",
    "input_source",
    "model_kind",
    "operation",
    "duration_ms",
    "success",
    "skipped",
    "skip_reason",
    "model_path",
    "feature_dim",
    "num_samples",
    "warmup",
    "notes",
]


@dataclass
class ModelEntry:
    joint: int
    model_kind: str
    model: Any
    stats: tuple[Any, Any, Any, Any]
    x_dim: int
    path: Path
    fallback_from: str = ""


class Goal2CTimingError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GOAL2 C offline/mock GP timing benchmark. No ROS, no launch, no robot commands.",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help=f"Default: {DEFAULT_MODEL_DIR}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--num-samples", type=positive_int, default=200, help="Measured prediction samples.")
    parser.add_argument("--warmup", type=nonnegative_int, default=20, help="Unrecorded warmup prediction samples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-csv", type=Path, default=None, help="Optional numeric feature CSV.")
    parser.add_argument("--input-npz", type=Path, default=None, help="Optional NPZ with X/features array.")
    parser.add_argument("--include-add-point", action="store_true", help="Benchmark add_point on copied models only.")
    parser.add_argument("--add-point-samples", type=nonnegative_int, default=20)
    parser.add_argument("--mock-cloud", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mock-cloud-sleep-ms", type=nonnegative_float, default=0.0)
    parser.add_argument("--fail-on-missing-cloud", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def run_readonly(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        return f"unavailable: {exc}"
    output = completed.stdout.strip()
    if completed.returncode != 0:
        detail = completed.stderr.strip() or output
        return f"unavailable: {detail}"
    return output


def ensure_skygp_import() -> None:
    # pickle 反序列化需要能找到训练时的 skygp module；这里只加 sys.path，不 import ROS。
    gp_dir = REPO_ROOT / "new_structure" / "gp"
    if str(gp_dir) not in sys.path:
        sys.path.insert(0, str(gp_dir))
    try:
        import skygp  # noqa: F401
    except ModuleNotFoundError as exc:
        raise Goal2CTimingError(f"Cannot import skygp from {gp_dir}") from exc


def make_record(
    *,
    benchmark: str,
    joint: str,
    sample_idx: int | str,
    input_source: str,
    model_kind: str,
    operation: str,
    duration_ms: float | None,
    success: bool,
    skipped: bool = False,
    skip_reason: str = "",
    model_path: Path | str = "",
    feature_dim: int | str = "",
    num_samples: int = 0,
    warmup: int = 0,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "timestamp": utc_now(),
        "benchmark": benchmark,
        "joint": joint,
        "sample_idx": sample_idx,
        "input_source": input_source,
        "model_kind": model_kind,
        "operation": operation,
        "duration_ms": duration_ms,
        "success": bool(success),
        "skipped": bool(skipped),
        "skip_reason": skip_reason,
        "model_path": str(model_path),
        "feature_dim": feature_dim,
        "num_samples": num_samples,
        "warmup": warmup,
        "notes": notes,
    }


def infer_pack(pack: Any, path: Path) -> tuple[Any, tuple[Any, Any, Any, Any], int]:
    if not isinstance(pack, dict):
        raise Goal2CTimingError(f"{path}: expected pickle dict with 'model' and 'stats'")
    if "model" not in pack or "stats" not in pack:
        raise Goal2CTimingError(f"{path}: missing required 'model' or 'stats' key")
    stats = pack["stats"]
    if not isinstance(stats, (tuple, list)) or len(stats) < 4:
        raise Goal2CTimingError(f"{path}: stats must contain (Xm, Xs, Ym, Ys)")
    model = pack["model"]
    Xm, Xs, Ym, Ys = stats[:4]
    x_dim = len(np.asarray(Xm).reshape(-1))
    model_x_dim = getattr(model, "x_dim", None)
    if model_x_dim is not None and int(model_x_dim) != x_dim:
        raise Goal2CTimingError(f"{path}: stats x_dim={x_dim} but model.x_dim={model_x_dim}")
    if len(np.asarray(Xs).reshape(-1)) < x_dim:
        raise Goal2CTimingError(f"{path}: Xs length is smaller than x_dim={x_dim}")
    return model, (Xm, Xs, Ym, Ys), x_dim


def load_one_model(
    path: Path,
    *,
    joint: int,
    model_kind: str,
    fallback_from: str,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
) -> ModelEntry | None:
    if not path.is_file():
        records.append(
            make_record(
                benchmark="model_load",
                joint=f"joint{joint}",
                sample_idx="",
                input_source="none",
                model_kind=model_kind,
                operation="pickle_load",
                duration_ms=None,
                success=False,
                skipped=True,
                skip_reason="model file not found",
                model_path=repo_relative(path),
                num_samples=args.num_samples,
                warmup=args.warmup,
                notes=fallback_from,
            )
        )
        return None

    try:
        ensure_skygp_import()
        start = time.perf_counter()
        with path.open("rb") as handle:
            pack = pickle.load(handle)
        duration_ms = (time.perf_counter() - start) * 1000.0
        model, stats, x_dim = infer_pack(pack, path)
    except Exception as exc:
        records.append(
            make_record(
                benchmark="model_load",
                joint=f"joint{joint}",
                sample_idx="",
                input_source="none",
                model_kind=model_kind,
                operation="pickle_load",
                duration_ms=None,
                success=False,
                skipped=False,
                skip_reason=str(exc),
                model_path=repo_relative(path),
                num_samples=args.num_samples,
                warmup=args.warmup,
                notes=fallback_from,
            )
        )
        return None

    records.append(
        make_record(
            benchmark="model_load",
            joint=f"joint{joint}",
            sample_idx="",
            input_source="none",
            model_kind=model_kind,
            operation="pickle_load",
            duration_ms=duration_ms,
            success=True,
            model_path=repo_relative(path),
            feature_dim=x_dim,
            num_samples=args.num_samples,
            warmup=args.warmup,
            notes=fallback_from,
        )
    )
    return ModelEntry(
        joint=joint,
        model_kind=model_kind,
        model=model,
        stats=stats,
        x_dim=x_dim,
        path=path,
        fallback_from=fallback_from,
    )


def load_models(args: argparse.Namespace, records: list[dict[str, Any]]) -> tuple[dict[int, ModelEntry], dict[int, ModelEntry], list[str], list[str]]:
    model_dir = resolve_repo_path(args.model_dir)
    local_models: dict[int, ModelEntry] = {}
    cloud_models: dict[int, ModelEntry] = {}
    missing: list[str] = []
    fallbacks: list[str] = []

    total_start = time.perf_counter()
    for joint in JOINTS:
        local_path = model_dir / f"joint{joint}_local.pkl"
        local_entry = load_one_model(
            local_path,
            joint=joint,
            model_kind="local",
            fallback_from="",
            args=args,
            records=records,
        )
        if local_entry is None:
            missing.append(repo_relative(local_path))
        else:
            local_models[joint] = local_entry

    for joint in JOINTS:
        cloud_path = model_dir / f"joint{joint}_cloud.pkl"
        fallback_note = ""
        load_path = cloud_path
        if not cloud_path.is_file():
            missing.append(repo_relative(cloud_path))
            local_path = model_dir / f"joint{joint}_local.pkl"
            if args.fail_on_missing_cloud or not local_path.is_file():
                load_one_model(
                    cloud_path,
                    joint=joint,
                    model_kind="cloud_like",
                    fallback_from="",
                    args=args,
                    records=records,
                )
                continue
            load_path = local_path
            fallback_note = "fallback_to_local_model; not real cloud communication"
            fallbacks.append(f"joint{joint}: {repo_relative(cloud_path)} -> {repo_relative(local_path)}")

        cloud_entry = load_one_model(
            load_path,
            joint=joint,
            model_kind="cloud_like",
            fallback_from=fallback_note,
            args=args,
            records=records,
        )
        if cloud_entry is not None:
            cloud_entry.fallback_from = fallback_note
            cloud_models[joint] = cloud_entry

    total_duration_ms = (time.perf_counter() - total_start) * 1000.0
    records.append(
        make_record(
            benchmark="model_load",
            joint="all",
            sample_idx="",
            input_source="none",
            model_kind="all",
            operation="total_model_load",
            duration_ms=total_duration_ms,
            success=bool(local_models or cloud_models),
            skipped=False,
            skip_reason="" if (local_models or cloud_models) else "no loadable GP models found",
            model_path=repo_relative(model_dir),
            num_samples=args.num_samples,
            warmup=args.warmup,
            notes=f"local_loaded={len(local_models)}; cloud_like_loaded={len(cloud_models)}",
        )
    )
    return local_models, cloud_models, missing, fallbacks


def load_input_matrix(args: argparse.Namespace) -> tuple[np.ndarray | None, str, list[str]]:
    if args.input_csv and args.input_npz:
        raise Goal2CTimingError("Use only one of --input-csv or --input-npz")
    if args.input_csv:
        path = resolve_repo_path(args.input_csv)
        if not path.is_file():
            raise Goal2CTimingError(f"input CSV not found: {path}")
        if pd is not None:
            frame = pd.read_csv(path)
            numeric = frame.select_dtypes(include=[np.number])
            if numeric.empty:
                raise Goal2CTimingError(f"input CSV has no numeric columns: {path}")
            return numeric.to_numpy(dtype=np.float32), f"csv:{repo_relative(path)}", []
        return load_input_csv_without_pandas(path), f"csv:{repo_relative(path)}", [
            "pandas is unavailable; input CSV was parsed with stdlib fallback.",
        ]
    if args.input_npz:
        path = resolve_repo_path(args.input_npz)
        if not path.is_file():
            raise Goal2CTimingError(f"input NPZ not found: {path}")
        data = np.load(path, allow_pickle=False)
        candidates = [name for name in ("X", "x", "features", "input", "inputs") if name in data]
        candidates.extend(name for name in data.files if name not in candidates)
        for name in candidates:
            array = np.asarray(data[name])
            if array.ndim == 2 and np.issubdtype(array.dtype, np.number):
                return array.astype(np.float32), f"npz:{repo_relative(path)}:{name}", []
        raise Goal2CTimingError(f"input NPZ has no 2D numeric feature array: {path}")
    return None, "synthetic_deterministic", [
        "Synthetic input is used for timing only.",
        "Synthetic input must not be used for accuracy conclusions.",
    ]


def load_input_csv_without_pandas(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        first_row = next(reader, None)
        if first_row is None:
            raise Goal2CTimingError(f"input CSV is empty: {path}")
        has_header = any(not is_float(value) for value in first_row)
        data_rows = reader if has_header else [first_row, *reader]
        for row in data_rows:
            numeric = [float(value) for value in row if is_float(value)]
            if numeric:
                rows.append(numeric)
    if not rows:
        raise Goal2CTimingError(f"input CSV has no numeric values: {path}")
    min_width = min(len(row) for row in rows)
    return np.asarray([row[:min_width] for row in rows], dtype=np.float32)


def is_float(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def build_standard_noise(max_rows: int, max_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.25, size=(max_rows, max_dim)).astype(np.float32)


def get_raw_rows(
    entry: ModelEntry,
    *,
    input_matrix: np.ndarray | None,
    standard_noise: np.ndarray,
    total_rows: int,
) -> np.ndarray:
    Xm, Xs, _, _ = entry.stats
    x_dim = entry.x_dim
    if input_matrix is not None:
        if input_matrix.shape[1] < x_dim:
            raise Goal2CTimingError(
                f"{repo_relative(entry.path)} requires feature_dim={x_dim}, "
                f"but input only has {input_matrix.shape[1]} columns"
            )
        indices = np.arange(total_rows) % input_matrix.shape[0]
        return input_matrix[indices, :x_dim].astype(np.float32)

    # 合成 raw feature，再按 controller 相同方式标准化；只用于 timing，不代表真实轨迹分布。
    xm = np.asarray(Xm, dtype=np.float32).reshape(-1)[:x_dim]
    xs = np.asarray(Xs, dtype=np.float32).reshape(-1)[:x_dim]
    safe_xs = np.where(np.abs(xs) < 1e-9, 1.0, xs)
    return (xm[None, :] + standard_noise[:total_rows, :x_dim] * safe_xs[None, :]).astype(np.float32)


def standardize_row(entry: ModelEntry, raw_row: np.ndarray) -> np.ndarray:
    Xm, Xs, _, _ = entry.stats
    x_dim = entry.x_dim
    xm = np.asarray(Xm, dtype=np.float32).reshape(-1)[:x_dim]
    xs = np.asarray(Xs, dtype=np.float32).reshape(-1)[:x_dim]
    safe_xs = np.where(np.abs(xs) < 1e-9, 1.0, xs)
    return ((raw_row[:x_dim] - xm) / safe_xs).astype(np.float32)


def predict_one(entry: ModelEntry, raw_row: np.ndarray) -> tuple[float, float]:
    x_std = standardize_row(entry, raw_row)
    mu_std, var_std = entry.model.predict(x_std)
    mu = float(np.asarray(mu_std).reshape(-1)[0])
    var = float(np.asarray(var_std).reshape(-1)[0])
    if not (math.isfinite(mu) and math.isfinite(var)):
        raise Goal2CTimingError(f"non-finite prediction from {entry.model_kind} joint{entry.joint}")
    return mu, var


def benchmark_prediction_kind(
    models: dict[int, ModelEntry],
    *,
    model_kind: str,
    args: argparse.Namespace,
    input_matrix: np.ndarray | None,
    input_source: str,
    standard_noise: np.ndarray,
    records: list[dict[str, Any]],
) -> None:
    if not models:
        records.append(
            make_record(
                benchmark="prediction",
                joint="all",
                sample_idx="",
                input_source=input_source,
                model_kind=model_kind,
                operation="predict_7joint_total",
                duration_ms=None,
                success=False,
                skipped=True,
                skip_reason=f"no {model_kind} models loaded",
                num_samples=args.num_samples,
                warmup=args.warmup,
            )
        )
        return

    total_rows = args.warmup + args.num_samples
    rows_by_joint = {
        joint: get_raw_rows(entry, input_matrix=input_matrix, standard_noise=standard_noise, total_rows=total_rows)
        for joint, entry in models.items()
    }

    for sample_idx in range(args.warmup):
        for joint, entry in models.items():
            predict_one(entry, rows_by_joint[joint][sample_idx])

    for measured_idx in range(args.num_samples):
        row_idx = args.warmup + measured_idx
        total_start = time.perf_counter()
        total_success = True
        total_reason = ""
        for joint, entry in models.items():
            start = time.perf_counter()
            try:
                predict_one(entry, rows_by_joint[joint][row_idx])
                duration_ms = (time.perf_counter() - start) * 1000.0
                success = True
                reason = ""
            except Exception as exc:
                duration_ms = (time.perf_counter() - start) * 1000.0
                success = False
                reason = str(exc)
                total_success = False
                total_reason = reason
            records.append(
                make_record(
                    benchmark="prediction",
                    joint=f"joint{joint}",
                    sample_idx=measured_idx,
                    input_source=input_source,
                    model_kind=model_kind,
                    operation="predict_per_joint",
                    duration_ms=duration_ms,
                    success=success,
                    skip_reason=reason,
                    model_path=repo_relative(entry.path),
                    feature_dim=entry.x_dim,
                    num_samples=args.num_samples,
                    warmup=args.warmup,
                    notes=entry.fallback_from,
                )
            )
        total_duration_ms = (time.perf_counter() - total_start) * 1000.0
        records.append(
            make_record(
                benchmark="prediction",
                joint="all",
                sample_idx=measured_idx,
                input_source=input_source,
                model_kind=model_kind,
                operation="predict_7joint_total",
                duration_ms=total_duration_ms,
                success=total_success and len(models) == 7,
                skipped=False,
                skip_reason=total_reason if total_reason else ("" if len(models) == 7 else "partial joint model set"),
                feature_dim=common_feature_dim(models),
                num_samples=args.num_samples,
                warmup=args.warmup,
                notes=f"loaded_joints={len(models)}",
            )
        )


def benchmark_combined_prediction(
    local_models: dict[int, ModelEntry],
    cloud_models: dict[int, ModelEntry],
    *,
    args: argparse.Namespace,
    input_matrix: np.ndarray | None,
    input_source: str,
    standard_noise: np.ndarray,
    records: list[dict[str, Any]],
) -> None:
    common_joints = sorted(set(local_models) & set(cloud_models))
    if not common_joints:
        records.append(
            make_record(
                benchmark="prediction",
                joint="all",
                sample_idx="",
                input_source=input_source,
                model_kind="combined",
                operation="predict_local_plus_cloud_7joint_total",
                duration_ms=None,
                success=False,
                skipped=True,
                skip_reason="no common local/cloud-like joints loaded",
                num_samples=args.num_samples,
                warmup=args.warmup,
            )
        )
        return

    total_rows = args.warmup + args.num_samples
    local_rows = {
        joint: get_raw_rows(local_models[joint], input_matrix=input_matrix, standard_noise=standard_noise, total_rows=total_rows)
        for joint in common_joints
    }
    cloud_rows = {
        joint: get_raw_rows(cloud_models[joint], input_matrix=input_matrix, standard_noise=standard_noise, total_rows=total_rows)
        for joint in common_joints
    }

    for sample_idx in range(args.warmup):
        for joint in common_joints:
            predict_one(local_models[joint], local_rows[joint][sample_idx])
            predict_one(cloud_models[joint], cloud_rows[joint][sample_idx])

    for measured_idx in range(args.num_samples):
        row_idx = args.warmup + measured_idx
        start = time.perf_counter()
        success = True
        reason = ""
        try:
            for joint in common_joints:
                predict_one(local_models[joint], local_rows[joint][row_idx])
                predict_one(cloud_models[joint], cloud_rows[joint][row_idx])
        except Exception as exc:
            success = False
            reason = str(exc)
        duration_ms = (time.perf_counter() - start) * 1000.0
        records.append(
            make_record(
                benchmark="prediction",
                joint="all",
                sample_idx=measured_idx,
                input_source=input_source,
                model_kind="combined",
                operation="predict_local_plus_cloud_7joint_total",
                duration_ms=duration_ms,
                success=success and len(common_joints) == 7,
                skip_reason=reason if reason else ("" if len(common_joints) == 7 else "partial joint model set"),
                feature_dim=common_feature_dim(local_models),
                num_samples=args.num_samples,
                warmup=args.warmup,
                notes=f"common_joints={len(common_joints)}; not real cloud communication",
            )
        )


def benchmark_add_point(
    models: dict[int, ModelEntry],
    *,
    model_kind: str,
    args: argparse.Namespace,
    input_matrix: np.ndarray | None,
    input_source: str,
    standard_noise: np.ndarray,
    records: list[dict[str, Any]],
) -> None:
    if not args.include_add_point:
        records.append(
            make_record(
                benchmark="add_point",
                joint="all",
                sample_idx="",
                input_source=input_source,
                model_kind=model_kind,
                operation="add_point",
                duration_ms=None,
                success=True,
                skipped=True,
                skip_reason="--include-add-point not set",
                num_samples=args.add_point_samples,
                warmup=0,
            )
        )
        return
    if not models:
        records.append(
            make_record(
                benchmark="add_point",
                joint="all",
                sample_idx="",
                input_source=input_source,
                model_kind=model_kind,
                operation="add_point_total",
                duration_ms=None,
                success=False,
                skipped=True,
                skip_reason=f"no {model_kind} models loaded",
                num_samples=args.add_point_samples,
                warmup=0,
            )
        )
        return

    copied: dict[int, ModelEntry] = {}
    for joint, entry in models.items():
        if not hasattr(entry.model, "add_point"):
            records.append(
                make_record(
                    benchmark="add_point",
                    joint=f"joint{joint}",
                    sample_idx="",
                    input_source=input_source,
                    model_kind=model_kind,
                    operation="copy_model_for_add_point",
                    duration_ms=None,
                    success=False,
                    skipped=True,
                    skip_reason="model has no add_point method",
                    model_path=repo_relative(entry.path),
                    feature_dim=entry.x_dim,
                    num_samples=args.add_point_samples,
                    warmup=0,
                )
            )
            continue
        start = time.perf_counter()
        try:
            copied_model = copy.deepcopy(entry.model)
            duration_ms = (time.perf_counter() - start) * 1000.0
            copied[joint] = ModelEntry(
                joint=entry.joint,
                model_kind=entry.model_kind,
                model=copied_model,
                stats=entry.stats,
                x_dim=entry.x_dim,
                path=entry.path,
                fallback_from=entry.fallback_from,
            )
            success = True
            reason = ""
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            success = False
            reason = str(exc)
        records.append(
            make_record(
                benchmark="add_point",
                joint=f"joint{joint}",
                sample_idx="",
                input_source=input_source,
                model_kind=model_kind,
                operation="copy_model_for_add_point",
                duration_ms=duration_ms,
                success=success,
                skipped=not success,
                skip_reason=reason,
                model_path=repo_relative(entry.path),
                feature_dim=entry.x_dim,
                num_samples=args.add_point_samples,
                warmup=0,
                notes="original pickle/model is not modified",
            )
        )

    if not copied:
        return

    total_rows = max(args.add_point_samples, 1)
    rows_by_joint = {
        joint: get_raw_rows(entry, input_matrix=input_matrix, standard_noise=standard_noise, total_rows=total_rows)
        for joint, entry in copied.items()
    }
    for sample_idx in range(args.add_point_samples):
        total_start = time.perf_counter()
        total_success = True
        total_reason = ""
        for joint, entry in copied.items():
            x_std = standardize_row(entry, rows_by_joint[joint][sample_idx])
            y_std = np.array([0.0], dtype=np.float32)
            start = time.perf_counter()
            try:
                entry.model.add_point(x_std, y_std)
                duration_ms = (time.perf_counter() - start) * 1000.0
                success = True
                reason = ""
            except Exception as exc:
                duration_ms = (time.perf_counter() - start) * 1000.0
                success = False
                reason = str(exc)
                total_success = False
                total_reason = reason
            records.append(
                make_record(
                    benchmark="add_point",
                    joint=f"joint{joint}",
                    sample_idx=sample_idx,
                    input_source=input_source,
                    model_kind=model_kind,
                    operation="add_point_per_joint",
                    duration_ms=duration_ms,
                    success=success,
                    skip_reason=reason,
                    model_path=repo_relative(entry.path),
                    feature_dim=entry.x_dim,
                    num_samples=args.add_point_samples,
                    warmup=0,
                    notes="copied model only; original pickle/model is not modified",
                )
            )
        records.append(
            make_record(
                benchmark="add_point",
                joint="all",
                sample_idx=sample_idx,
                input_source=input_source,
                model_kind=model_kind,
                operation="add_point_total",
                duration_ms=(time.perf_counter() - total_start) * 1000.0,
                success=total_success and len(copied) == 7,
                skip_reason=total_reason if total_reason else ("" if len(copied) == 7 else "partial joint model set"),
                feature_dim=common_feature_dim(copied),
                num_samples=args.add_point_samples,
                warmup=0,
                notes="copied models only; original pickle/model is not modified",
            )
        )


def benchmark_mock_cloud(
    args: argparse.Namespace,
    *,
    input_source: str,
    max_feature_dim: int,
    standard_noise: np.ndarray,
    records: list[dict[str, Any]],
) -> None:
    if not args.mock_cloud:
        records.append(
            make_record(
                benchmark="mock_cloud",
                joint="all",
                sample_idx="",
                input_source=input_source,
                model_kind="mock_cloud",
                operation="mock_roundtrip",
                duration_ms=None,
                success=True,
                skipped=True,
                skip_reason="--no-mock-cloud set",
                num_samples=args.num_samples,
                warmup=args.warmup,
                notes="not real cloud communication; not ROS service timing",
            )
        )
        return

    feature_dim = max(max_feature_dim, 1)
    rows = standard_noise[: args.num_samples, :feature_dim]
    for sample_idx in range(args.num_samples):
        payload = {
            "sample_idx": sample_idx,
            "feature_dim": feature_dim,
            "x": rows[sample_idx].astype(float).tolist(),
            "request_kind": "goal2c_mock_cloud",
        }
        roundtrip_start = time.perf_counter()
        start = time.perf_counter()
        request_body = json.dumps(payload, separators=(",", ":"))
        serialize_ms = (time.perf_counter() - start) * 1000.0
        records.append(
            make_record(
                benchmark="mock_cloud",
                joint="all",
                sample_idx=sample_idx,
                input_source=input_source,
                model_kind="mock_cloud",
                operation="mock_request_serialize",
                duration_ms=serialize_ms,
                success=True,
                feature_dim=feature_dim,
                num_samples=args.num_samples,
                warmup=args.warmup,
                notes="mock only; not real cloud communication; not ROS service timing",
            )
        )

        if args.mock_cloud_sleep_ms > 0.0:
            start = time.perf_counter()
            time.sleep(args.mock_cloud_sleep_ms / 1000.0)
            records.append(
                make_record(
                    benchmark="mock_cloud",
                    joint="all",
                    sample_idx=sample_idx,
                    input_source=input_source,
                    model_kind="mock_cloud",
                    operation="mock_artificial_sleep",
                    duration_ms=(time.perf_counter() - start) * 1000.0,
                    success=True,
                    feature_dim=feature_dim,
                    num_samples=args.num_samples,
                    warmup=args.warmup,
                    notes=f"requested_sleep_ms={args.mock_cloud_sleep_ms}",
                )
            )

        response_body = '{"status":"ok","y_hat":[0,0,0,0,0,0,0]}'
        start = time.perf_counter()
        response = json.loads(response_body)
        parse_ms = (time.perf_counter() - start) * 1000.0
        response_ok = response.get("status") == "ok" and bool(request_body)
        records.append(
            make_record(
                benchmark="mock_cloud",
                joint="all",
                sample_idx=sample_idx,
                input_source=input_source,
                model_kind="mock_cloud",
                operation="mock_response_parse",
                duration_ms=parse_ms,
                success=response_ok,
                feature_dim=feature_dim,
                num_samples=args.num_samples,
                warmup=args.warmup,
                notes="mock only; not real cloud communication; not ROS service timing",
            )
        )
        records.append(
            make_record(
                benchmark="mock_cloud",
                joint="all",
                sample_idx=sample_idx,
                input_source=input_source,
                model_kind="mock_cloud",
                operation="mock_roundtrip",
                duration_ms=(time.perf_counter() - roundtrip_start) * 1000.0,
                success=response_ok,
                feature_dim=feature_dim,
                num_samples=args.num_samples,
                warmup=args.warmup,
                notes="mock only; not real cloud communication; not ROS service timing",
            )
        )


def common_feature_dim(models: dict[int, ModelEntry]) -> str:
    dims = sorted({entry.x_dim for entry in models.values()})
    if not dims:
        return ""
    return str(dims[0]) if len(dims) == 1 else ";".join(str(dim) for dim in dims)


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            str(record["benchmark"]),
            str(record["model_kind"]),
            str(record["operation"]),
            str(record["joint"]),
        )
        groups.setdefault(key, []).append(record)

    summary: list[dict[str, Any]] = []
    for (benchmark, model_kind, operation, joint), items in sorted(groups.items()):
        durations = np.asarray(
            [float(item["duration_ms"]) for item in items if is_finite_number(item.get("duration_ms"))],
            dtype=float,
        )
        if durations.size:
            mean_ms = float(np.mean(durations))
            std_ms = float(np.std(durations, ddof=1)) if durations.size > 1 else 0.0
            p50_ms = float(np.percentile(durations, 50))
            p95_ms = float(np.percentile(durations, 95))
            p99_ms = float(np.percentile(durations, 99))
            max_ms = float(np.max(durations))
            min_ms = float(np.min(durations))
        else:
            mean_ms = std_ms = p50_ms = p95_ms = p99_ms = max_ms = min_ms = math.nan
        summary.append(
            {
                "benchmark": benchmark,
                "model_kind": model_kind,
                "operation": operation,
                "joint": joint,
                "count": len(items),
                "success_count": sum(1 for item in items if bool(item.get("success"))),
                "skip_count": sum(1 for item in items if bool(item.get("skipped"))),
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "p50_ms": p50_ms,
                "p95_ms": p95_ms,
                "p99_ms": p99_ms,
                "max_ms": max_ms,
                "min_ms": min_ms,
            }
        )
    return summary


def is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def format_ms(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.6f}"


def summary_fields() -> list[str]:
    return [
        "benchmark",
        "model_kind",
        "operation",
        "joint",
        "count",
        "success_count",
        "skip_count",
        "mean_ms",
        "std_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "min_ms",
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    # pandas 可用时按用户要求优先使用 pandas；不可用时保守 fallback，避免 offline smoke 被环境依赖完全挡住。
    if pd is not None:
        pd.DataFrame(rows, columns=fields).to_csv(path, index=False)
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def markdown_table(
    summary: list[dict[str, Any]],
    benchmark: str,
    operations: set[str] | None = None,
    model_kinds: set[str] | None = None,
) -> str:
    if not summary:
        return "_No records._"
    subset = [
        row
        for row in summary
        if row["benchmark"] == benchmark
        and (operations is None or row["operation"] in operations)
        and (model_kinds is None or row["model_kind"] in model_kinds)
    ]
    if not subset:
        return "_No records._"
    lines = [
        "| model_kind | operation | joint | count | success | skipped | mean_ms | p50_ms | p95_ms | p99_ms | max_ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(subset, key=lambda item: (item["model_kind"], item["operation"], item["joint"])):
        lines.append(
            "| {model_kind} | {operation} | {joint} | {count} | {success_count} | {skip_count} | {mean_ms} | {p50_ms} | {p95_ms} | {p99_ms} | {max_ms} |".format(
                model_kind=row["model_kind"],
                operation=row["operation"],
                joint=row["joint"],
                count=int(row["count"]),
                success_count=int(row["success_count"]),
                skip_count=int(row["skip_count"]),
                mean_ms=format_ms(row["mean_ms"]),
                p50_ms=format_ms(row["p50_ms"]),
                p95_ms=format_ms(row["p95_ms"]),
                p99_ms=format_ms(row["p99_ms"]),
                max_ms=format_ms(row["max_ms"]),
            )
        )
    return "\n".join(lines)


def write_outputs(
    *,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    input_source: str,
    input_notes: list[str],
    missing: list[str],
    fallbacks: list[str],
    status: str,
    error_message: str,
) -> None:
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / RECORDS_CSV
    summary_path = output_dir / SUMMARY_CSV
    md_path = output_dir / SUMMARY_MD

    write_csv(records_path, records, RECORD_FIELDS)
    write_csv(summary_path, summary, summary_fields())
    md_path.write_text(
        build_markdown_summary(
            args=args,
            summary=summary,
            input_source=input_source,
            input_notes=input_notes,
            missing=missing,
            fallbacks=fallbacks,
            status=status,
            error_message=error_message,
            records_path=records_path,
            summary_path=summary_path,
        ),
        encoding="utf-8",
    )


def build_markdown_summary(
    *,
    args: argparse.Namespace,
    summary: list[dict[str, Any]],
    input_source: str,
    input_notes: list[str],
    missing: list[str],
    fallbacks: list[str],
    status: str,
    error_message: str,
    records_path: Path,
    summary_path: Path,
) -> str:
    command = " ".join(shlex.quote(part) for part in sys.argv)
    model_dir = resolve_repo_path(args.model_dir)
    output_dir = resolve_repo_path(args.output_dir)
    found_local = sorted(model_dir.glob("joint*_local.pkl"))
    found_cloud = sorted(model_dir.glob("joint*_cloud.pkl"))
    env_lines = [
        f"- cwd: `{Path.cwd()}`",
        f"- repo_root: `{REPO_ROOT}`",
        f"- git_branch: `{run_readonly(['git', 'branch', '--show-current'])}`",
        f"- git_status_short: `{run_readonly(['git', 'status', '--short']) or 'clean'}`",
        f"- python: `{sys.version.split()[0]}`",
        f"- platform: `{platform.platform()}`",
        f"- pandas_available: `{'yes' if pd is not None else 'no'}`",
    ]
    if PANDAS_IMPORT_ERROR:
        env_lines.append(f"- pandas_note: `pandas unavailable ({PANDAS_IMPORT_ERROR}); CSV outputs used stdlib fallback`")
    caveats = [
        "This is offline/mock timing only.",
        "It does not run ROS, launch, fake hardware, controller, trajectory publisher, or robot commands.",
        "Mock cloud timing is local JSON serialization/parse plus optional sleep, not real cloud communication and not ROS service timing.",
        "Offline/mock timing is not fake hardware timing and is not real robot safety proof.",
    ]
    missing_lines = [f"- `{item}`" for item in missing] if missing else ["- none"]
    fallback_lines = [f"- {item}" for item in fallbacks] if fallbacks else ["- none"]
    input_note_lines = [f"- {note}" for note in input_notes] if input_notes else ["- none"]
    lines = [
        "# GOAL2 C Offline / Mock Timing Benchmark Summary",
        "",
        f"- status: `{status}`",
        f"- command: `{command}`",
        f"- model_dir: `{repo_relative(model_dir)}`",
        f"- output_dir: `{repo_relative(output_dir)}`",
        f"- records_csv: `{repo_relative(records_path)}`",
        f"- summary_csv: `{repo_relative(summary_path)}`",
        f"- input_source: `{input_source}`",
        f"- synthetic_input: `{'yes' if input_source == 'synthetic_deterministic' else 'no'}`",
        f"- num_samples: `{args.num_samples}`",
        f"- warmup: `{args.warmup}`",
        f"- include_add_point: `{args.include_add_point}`",
        f"- add_point_samples: `{args.add_point_samples}`",
        f"- mock_cloud: `{args.mock_cloud}`",
        f"- mock_cloud_sleep_ms: `{args.mock_cloud_sleep_ms}`",
        "",
        "## Environment",
        *env_lines,
        "",
        "## Model Files Found",
        f"- local files found: `{len(found_local)}`",
        f"- cloud-like files found: `{len(found_cloud)}`",
        f"- local file list: `{'; '.join(repo_relative(path) for path in found_local) or 'none'}`",
        f"- cloud-like file list: `{'; '.join(repo_relative(path) for path in found_cloud) or 'none'}`",
        "",
        "## Missing Models",
        *missing_lines,
        "",
        "## Fallback Behavior",
        *fallback_lines,
        "",
        "## Local GP Timing Summary",
        markdown_table(summary, "prediction", {"predict_per_joint", "predict_7joint_total"}, model_kinds={"local"}),
        "",
        "## Cloud-like GP Timing Summary",
        markdown_table(summary, "prediction", model_kinds={"cloud_like"}),
        "",
        "## Combined Timing Summary",
        markdown_table(summary, "prediction", model_kinds={"combined"}),
        "",
        "## add_point Timing Summary",
        markdown_table(summary, "add_point"),
        "",
        "## Mock Cloud Timing Summary",
        markdown_table(summary, "mock_cloud"),
        "",
        "## Input Notes",
        *input_note_lines,
        "",
        "## Caveats",
        *[f"- {item}" for item in caveats],
        "",
        "## Recommended Next Step",
    ]
    if status == "success":
        lines.append("- Run GOAL2 C read-only self-review before any further code changes.")
    else:
        lines.append("- Fix the script input/model availability issue, then rerun the offline benchmark.")
    if error_message:
        lines.extend(["", "## Error", f"- {error_message}"])
    lines.append("")
    return "\n".join(lines)


def run_benchmark(args: argparse.Namespace) -> int:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    fallbacks: list[str] = []
    input_source = "unknown"
    input_notes: list[str] = []
    status = "success"
    error_message = ""

    try:
        input_matrix, input_source, input_notes = load_input_matrix(args)
        local_models, cloud_models, missing, fallbacks = load_models(args, records)
        if not local_models:
            raise Goal2CTimingError(
                f"No local GP models could be loaded from {resolve_repo_path(args.model_dir)}. "
                "Expected joint1_local.pkl ... joint7_local.pkl."
            )
        if not cloud_models:
            raise Goal2CTimingError(
                f"No cloud-like GP models could be loaded from {resolve_repo_path(args.model_dir)}. "
                "Expected joint*_cloud.pkl or fallback joint*_local.pkl when --fail-on-missing-cloud is not set."
            )

        max_dim = max(entry.x_dim for entry in list(local_models.values()) + list(cloud_models.values()))
        total_rows = max(args.warmup + args.num_samples, args.add_point_samples, args.num_samples, 1)
        standard_noise = build_standard_noise(total_rows, max_dim, args.seed)

        benchmark_prediction_kind(
            local_models,
            model_kind="local",
            args=args,
            input_matrix=input_matrix,
            input_source=input_source,
            standard_noise=standard_noise,
            records=records,
        )
        benchmark_prediction_kind(
            cloud_models,
            model_kind="cloud_like",
            args=args,
            input_matrix=input_matrix,
            input_source=input_source,
            standard_noise=standard_noise,
            records=records,
        )
        benchmark_combined_prediction(
            local_models,
            cloud_models,
            args=args,
            input_matrix=input_matrix,
            input_source=input_source,
            standard_noise=standard_noise,
            records=records,
        )
        benchmark_add_point(
            local_models,
            model_kind="local",
            args=args,
            input_matrix=input_matrix,
            input_source=input_source,
            standard_noise=standard_noise,
            records=records,
        )
        benchmark_add_point(
            cloud_models,
            model_kind="cloud_like",
            args=args,
            input_matrix=input_matrix,
            input_source=input_source,
            standard_noise=standard_noise,
            records=records,
        )
        benchmark_mock_cloud(
            args,
            input_source=input_source,
            max_feature_dim=max_dim,
            standard_noise=standard_noise,
            records=records,
        )
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        records.append(
            make_record(
                benchmark="run_status",
                joint="all",
                sample_idx="",
                input_source=input_source,
                model_kind="all",
                operation="benchmark_run",
                duration_ms=None,
                success=False,
                skipped=False,
                skip_reason=error_message,
                model_path=repo_relative(resolve_repo_path(args.model_dir)),
                num_samples=args.num_samples,
                warmup=args.warmup,
            )
        )
        if args.verbose:
            print(f"GOAL2 C benchmark failed: {error_message}", file=sys.stderr)

    summary = summarize_records(records)
    write_outputs(
        args=args,
        records=records,
        summary=summary,
        input_source=input_source,
        input_notes=input_notes,
        missing=missing,
        fallbacks=fallbacks,
        status=status,
        error_message=error_message,
    )
    if status != "success":
        print(f"GOAL2 C offline/mock benchmark failed: {error_message}", file=sys.stderr)
        print(f"Wrote failure summary to: {resolve_repo_path(args.output_dir) / SUMMARY_MD}", file=sys.stderr)
        return 2
    print(f"GOAL2 C offline/mock benchmark completed: {resolve_repo_path(args.output_dir)}")
    return 0


def main() -> int:
    args = parse_args()
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
