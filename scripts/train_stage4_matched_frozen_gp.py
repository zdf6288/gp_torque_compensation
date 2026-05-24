#!/usr/bin/env python3
"""Train a Stage 4 matched frozen local GP model directory offline.

This wrapper trains controller/validator-compatible per-joint SkyGP pickles
from a matched Stage 4 .npz dataset. It is intended only for an engineering
sanity check: the default dataset comes from the `strict_no_gp` formal run, so
evaluating the resulting model on the same formal trajectory has train/test
leakage and is not paper-level generalization evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True

try:
    import numpy as np
except ModuleNotFoundError as exc:
    print("Missing Python dependency: numpy", file=sys.stderr)
    print("Use an environment that already has project dependencies installed.", file=sys.stderr)
    raise SystemExit(1) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
GP_DIR = REPO_ROOT / "new_structure" / "gp"
DEFAULT_DATASET = Path("data/stage4/datasets/GP_matched_strict_no_gp_zmod_20260523_154902.npz")
DEFAULT_OUTPUT_DIR = Path("data/stage4/models/GP_matched_strict_no_gp_zmod")
DEFAULT_MODEL_NAME = "GP_matched_strict_no_gp_zmod"
JOINTS = range(1, 8)
SCRIPT_VERSION = "2026-05-24-stage4-matched-frozen-gp-v2"
DEFAULT_X_STD_FLOOR = 1e-5
Y_STD_FLOOR = 1e-9
CAVEAT = (
    "Matched model trained from strict_no_gp formal run data. Use only for "
    "engineering sanity checks. Evaluating on the same or near-identical formal "
    "trajectory introduces train/test leakage and is not paper-level "
    "generalization proof."
)


def ensure_training_imports() -> Any:
    if str(GP_DIR) not in sys.path:
        sys.path.insert(0, str(GP_DIR))
    try:
        import train_gp_hdimensional as trainer
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Cannot import train_gp_hdimensional from {GP_DIR}") from exc
    return trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train offline Stage 4 matched frozen local GP models from a .npz dataset.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help=f"Default: {DEFAULT_DATASET}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help=f"Default: {DEFAULT_MODEL_NAME}")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting files in an existing output dir.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional random subset size for quick debug.")
    parser.add_argument("--random-seed", type=int, default=0, help="Default: 0")
    parser.add_argument(
        "--fit-hparams",
        action="store_true",
        help="Run the heavier project GPyTorch hparam fit. Default uses the existing standardized fallback hparams.",
    )
    parser.add_argument(
        "--x-std-floor",
        type=float,
        default=DEFAULT_X_STD_FLOOR,
        help=f"Feature std values below this are stored as 1.0. Default: {DEFAULT_X_STD_FLOOR}",
    )
    return parser.parse_args()


def as_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def dataset_text(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data.files:
        return default
    value = data[key]
    try:
        return str(value.item())
    except ValueError:
        return str(value)


def dataset_string_list(data: np.lib.npyio.NpzFile, key: str, fallback: list[str]) -> list[str]:
    if key not in data.files:
        return fallback
    return [str(item) for item in np.asarray(data[key]).reshape(-1)]


def load_and_validate_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        if "X" not in data.files or "Y" not in data.files:
            raise KeyError(f"{path}: expected keys X and Y")
        x_matrix = np.asarray(data["X"], dtype=np.float32)
        y_matrix = np.asarray(data["Y"], dtype=np.float32)
        feature_names = dataset_string_list(
            data,
            "feature_columns",
            [f"joint_pos_{joint}" for joint in JOINTS] + [f"joint_vel_{joint}" for joint in JOINTS],
        )
        target_names = dataset_string_list(data, "target_columns", [f"tau_residual_raw_{joint}" for joint in JOINTS])
        source_csv = dataset_text(data, "source_csv")
        mode_name = dataset_text(data, "mode_name")
        target_kind = dataset_text(data, "target_kind")
        feature_source = dataset_text(data, "feature_source")

    if x_matrix.ndim != 2:
        raise ValueError(f"X.ndim must be 2, got {x_matrix.ndim}")
    if y_matrix.ndim != 2:
        raise ValueError(f"Y.ndim must be 2, got {y_matrix.ndim}")
    if x_matrix.shape[0] != y_matrix.shape[0]:
        raise ValueError(f"X/Y row mismatch: {x_matrix.shape[0]} vs {y_matrix.shape[0]}")
    if x_matrix.shape[1] != 14:
        raise ValueError(f"X.shape[1] must be 14, got {x_matrix.shape[1]}")
    if y_matrix.shape[1] != 7:
        raise ValueError(f"Y.shape[1] must be 7, got {y_matrix.shape[1]}")
    if not np.all(np.isfinite(x_matrix)):
        raise ValueError("X contains NaN or Inf")
    if not np.all(np.isfinite(y_matrix)):
        raise ValueError("Y contains NaN or Inf")

    metadata = {
        "source_csv": source_csv,
        "source_mode_name": mode_name,
        "target_kind": target_kind,
        "feature_source": feature_source,
        "feature_names": feature_names,
        "target_names": target_names,
    }
    return x_matrix, y_matrix, metadata


def maybe_subset(
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    max_samples: int | None,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if max_samples is None:
        return x_matrix, y_matrix, None
    if max_samples < 1:
        raise ValueError("--max-samples must be >= 1 when provided")
    if max_samples >= x_matrix.shape[0]:
        return x_matrix, y_matrix, None

    rng = np.random.default_rng(random_seed)
    indices = np.sort(rng.choice(x_matrix.shape[0], size=max_samples, replace=False))
    return x_matrix[indices], y_matrix[indices], indices


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(path)
    if path.is_dir() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} already exists and is not empty; pass --overwrite to replace expected files")
    path.mkdir(parents=True, exist_ok=True)


def reset_prediction_state(trainer: Any, model: Any) -> None:
    trainer.reset_prediction_state(model)


def fallback_hparams(x_norm: np.ndarray, y_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_std = float(np.std(y_norm))
    outputscale = np.array([max(y_std, 1e-6)], dtype=float)
    noise = np.array([max(1e-4, 0.05 * y_std)], dtype=float)
    lengthscale = np.ones(x_norm.shape[1], dtype=float)
    return outputscale, noise, lengthscale


def standardize_for_validator_gate(
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    x_std_floor: float,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], list[int]]:
    if not math.isfinite(x_std_floor) or x_std_floor <= 0.0:
        raise ValueError("--x-std-floor must be finite and > 0")

    x_mean = x_matrix.mean(0)
    x_std = x_matrix.std(0)
    floored_indices = np.flatnonzero(x_std < x_std_floor).astype(int).tolist()
    x_std = x_std.copy()
    x_std[x_std < x_std_floor] = 1.0

    y_mean = y_matrix.mean(0)
    y_std = y_matrix.std(0)
    y_std = y_std.copy()
    y_std[y_std < Y_STD_FLOOR] = 1.0

    x_norm = (x_matrix - x_mean) / x_std
    y_norm = (y_matrix - y_mean) / y_std
    stats = (x_mean, x_std, y_mean, y_std)
    return x_norm.astype(np.float32), y_norm.astype(np.float32), stats, floored_indices


def train_joint(
    trainer: Any,
    x_matrix: np.ndarray,
    y_column: np.ndarray,
    joint: int,
    fit_hparams: bool,
    x_std_floor: float,
) -> dict[str, Any]:
    y_matrix = y_column.reshape(-1, 1).astype(np.float32)
    x_norm, y_norm, stats, x_std_floored_indices = standardize_for_validator_gate(
        x_matrix,
        y_matrix,
        x_std_floor,
    )
    if fit_hparams:
        hps = trainer.fit_global_hparams(
            x_norm,
            y_norm,
            max_pts_hparam=3000,
            iters=600,
            lr=0.04,
            print_every=100,
        )
        hparam_source = "fit_global_hparams"
    else:
        hps = fallback_hparams(x_norm, y_norm)
        hparam_source = "standardized_fallback"

    frozen_mde = max(64, x_norm.shape[0])
    local_model = trainer.build_rBCM(
        x_dim=x_matrix.shape[1],
        hps=hps,
        max_data_per_expert=frozen_mde,
        nearest_k=1,
        max_experts=1,
        timescale=0.03,
    )
    local_model.offline_pretrain(
        x_norm,
        y_norm,
        show_progress=True,
        optimize_hparams=False,
    )
    trainer.validate_frozen_local_model(local_model, joint)
    reset_prediction_state(trainer, local_model)
    mu_check, var_check = local_model.predict(x_norm[0])
    if not (np.all(np.isfinite(mu_check)) and np.all(np.isfinite(var_check))):
        raise RuntimeError(f"joint {joint}: frozen local model predict sanity check failed")
    reset_prediction_state(trainer, local_model)

    cloud_model = trainer.build_rBCM(
        x_dim=x_matrix.shape[1],
        hps=hps,
        max_data_per_expert=64,
        nearest_k=4,
        max_experts=64,
        timescale=0.03,
    )
    return {
        "stats": stats,
        "hps": hps,
        "local_model": local_model,
        "cloud_model": cloud_model,
        "local_samples": int(np.sum(local_model.localCount)),
        "local_experts": int(len(local_model.X_list)),
        "hparam_source": hparam_source,
        "x_std_floored_indices": x_std_floored_indices,
    }


def save_joint(output_dir: Path, joint: int, result: dict[str, Any]) -> None:
    payload_common = {
        "stats": result["stats"],
        "hps_std": result["hps"],
    }
    with (output_dir / f"joint{joint}_local.pkl").open("wb") as handle:
        pickle.dump({"model": result["local_model"], "type": "local", **payload_common}, handle)
    with (output_dir / f"joint{joint}_cloud.pkl").open("wb") as handle:
        pickle.dump({"model": result["cloud_model"], "type": "cloud", **payload_common}, handle)


def scalar_float(value: Any) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return math.nan
    return float(array[0])


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    source_metadata: dict[str, Any],
    subset_indices: np.ndarray | None,
    joint_summaries: list[dict[str, Any]],
) -> None:
    created_utc = datetime.now(timezone.utc).isoformat()
    metadata = {
        "model_name": args.model_name,
        "created_utc": created_utc,
        "script": as_repo_relative(Path(__file__)),
        "script_version": SCRIPT_VERSION,
        "dataset": as_repo_relative(args.dataset),
        "output_dir": as_repo_relative(output_dir),
        "X_shape": list(x_matrix.shape),
        "Y_shape": list(y_matrix.shape),
        "max_samples": args.max_samples,
        "random_seed": args.random_seed,
        "fit_hparams": bool(args.fit_hparams),
        "x_std_floor": args.x_std_floor,
        "subset_indices": subset_indices.tolist() if subset_indices is not None else None,
        "feature_names": source_metadata["feature_names"],
        "target_names": source_metadata["target_names"],
        "source_csv": source_metadata["source_csv"],
        "source_mode_name": source_metadata["source_mode_name"],
        "target_kind": source_metadata["target_kind"],
        "feature_source": source_metadata["feature_source"],
        "model_file_layout": "joint1_local.pkl..joint7_local.pkl plus joint1_cloud.pkl..joint7_cloud.pkl",
        "caveat": CAVEAT,
        "joint_summaries": joint_summaries,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        f"# {args.model_name}",
        "",
        "## Purpose",
        "",
        "Offline Stage 4 matched frozen local GP model directory for engineering sanity checks.",
        "",
        "## Caveat",
        "",
        CAVEAT,
        "",
        "## Source",
        "",
        f"- dataset: `{metadata['dataset']}`",
        f"- source_csv: `{metadata['source_csv']}`",
        f"- X_shape: `{metadata['X_shape']}`",
        f"- Y_shape: `{metadata['Y_shape']}`",
        "",
        "## Layout",
        "",
        "- `jointN_local.pkl`: frozen local model used by controller/validator.",
        "- `jointN_cloud.pkl`: compatible cloud-format model with shared stats/hparams and no stored samples.",
        "- `metadata.json`: training metadata and caveat.",
        "",
        "## Safety Boundary",
        "",
        "This training wrapper is offline-only. It does not modify controller, launch, config, or torque command logic.",
        f"Feature std values below `{args.x_std_floor}` are stored as `1.0` to avoid near-zero scaler gate failures.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    trainer = ensure_training_imports()
    x_all, y_all, source_metadata = load_and_validate_dataset(args.dataset)
    x_train, y_train, subset_indices = maybe_subset(x_all, y_all, args.max_samples, args.random_seed)
    ensure_output_dir(args.output_dir, args.overwrite)

    print("Stage 4 matched frozen GP training", flush=True)
    print(f"dataset: {args.dataset}", flush=True)
    print(f"output_dir: {args.output_dir}", flush=True)
    print(f"model_name: {args.model_name}", flush=True)
    print(f"X_shape: {x_train.shape}", flush=True)
    print(f"Y_shape: {y_train.shape}", flush=True)
    print(f"fit_hparams: {args.fit_hparams}", flush=True)
    print(f"x_std_floor: {args.x_std_floor}", flush=True)
    print(f"caveat: {CAVEAT}", flush=True)

    joint_summaries = []
    for joint in JOINTS:
        print(f"\n========== Training joint {joint} ==========", flush=True)
        result = train_joint(trainer, x_train, y_train[:, joint - 1], joint, args.fit_hparams, args.x_std_floor)
        save_joint(args.output_dir, joint, result)
        hps = result["hps"]
        summary = {
            "joint": joint,
            "local_samples": result["local_samples"],
            "local_experts": result["local_experts"],
            "outputscale": scalar_float(hps[0]),
            "noise": scalar_float(hps[1]),
            "lengthscale_shape": list(np.asarray(hps[2]).shape),
            "hparam_source": result["hparam_source"],
            "x_std_floored_indices": result["x_std_floored_indices"],
            "local_path": str(args.output_dir / f"joint{joint}_local.pkl"),
            "cloud_path": str(args.output_dir / f"joint{joint}_cloud.pkl"),
        }
        joint_summaries.append(summary)
        print(
            f"saved joint {joint}: local_samples={summary['local_samples']} "
            f"local_experts={summary['local_experts']} "
            f"hparam_source={summary['hparam_source']} "
            f"x_std_floored_indices={summary['x_std_floored_indices']}",
            flush=True,
        )

    write_metadata(args.output_dir, args, x_train, y_train, source_metadata, subset_indices, joint_summaries)

    print("\nTraining complete", flush=True)
    print(f"wrote_model_dir: {args.output_dir}", flush=True)
    print(f"wrote_metadata: {args.output_dir / 'metadata.json'}", flush=True)
    print(f"wrote_readme: {args.output_dir / 'README.md'}", flush=True)
    print("next_validator: python3 scripts/validate_frozen_gp_support.py --help", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
