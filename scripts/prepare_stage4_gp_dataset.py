#!/usr/bin/env python3
"""Prepare matched Stage 4 GP training datasets.

This script is offline-only. It does not modify source CSV files and does not
depend on ROS2 runtime state.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


np = None
pd = None

JOINTS = range(1, 8)
PLANAR_OUT_NAME = "GP_planar_matched.npz"
SPATIAL_OUT_NAME = "GP_spatial_matched.npz"
MANIFEST_NAME = "stage4_dataset_manifest.json"

BUILDER_MODES = {
    "runtime-real-dq": {
        "feature_velocity_prefix": "joint_vel",
        "feature_definition": "X = [joint_pos_1..7, joint_vel_1..7]",
        "target_definition": "Y_j = tau_residual_j",
        "note": (
            "Default Stage 4 mode. It mirrors build_dataset_real_dq.py's q + joint_vel "
            "feature choice and better matches the current controller runtime call, "
            "where measured dq is passed into _gp_predict_and_update()."
        ),
    },
    "no-filter-dq-des": {
        "feature_velocity_prefix": "dq_des_joint",
        "feature_definition": "X = [joint_pos_1..7, dq_des_joint_1..7]",
        "target_definition": "Y_j = tau_residual_j",
        "note": (
            "Compatibility mode for the build_dataset_no_filter.py 14D q + dq_des_joint "
            "feature convention. Use only if Stage 4 intentionally keeps that older "
            "training feature definition."
        ),
    },
}


def import_dependencies() -> None:
    global np, pd

    missing = []
    try:
        import numpy as numpy_module
    except ModuleNotFoundError:
        missing.append("numpy")
    try:
        import pandas as pandas_module
    except ModuleNotFoundError:
        missing.append("pandas")

    if missing:
        print("Missing Python dependencies: " + ", ".join(missing), file=sys.stderr)
        print("Use an environment that already has them; this script does not install packages.", file=sys.stderr)
        raise SystemExit(1)

    np = numpy_module
    pd = pandas_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare sample-count-matched Stage 4 GP datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Dry-run:
    python3 scripts/prepare_stage4_gp_dataset.py --planar-pattern "data/stage4/train/planar_circle/*.csv" --spatial-pattern "data/stage4/train/spatial_tilted/*.csv" --out-dir data/stage4/datasets --dry-run

  Real run:
    python3 scripts/prepare_stage4_gp_dataset.py --planar-pattern "data/stage4/train/planar_circle/*.csv" --spatial-pattern "data/stage4/train/spatial_tilted/*.csv" --out-dir data/stage4/datasets

  Train matched models:
    python3 new_structure/gp/train_gp_hdimensional.py --data data/stage4/datasets/GP_planar_matched.npz --out-dir data/stage4/models/GP_planar --joint all
    python3 new_structure/gp/train_gp_hdimensional.py --data data/stage4/datasets/GP_spatial_matched.npz --out-dir data/stage4/models/GP_spatial --joint all
""",
    )
    parser.add_argument("--planar-pattern", required=True, help="Glob for planar training CSV files.")
    parser.add_argument("--spatial-pattern", required=True, help="Glob for spatial / tilted training CSV files.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for matched npz files and manifest.")
    parser.add_argument(
        "--builder-mode",
        choices=sorted(BUILDER_MODES),
        default="runtime-real-dq",
        help="Feature/target convention to use. Default: runtime-real-dq.",
    )
    parser.add_argument("--dt", type=float, default=0.001, help="Original sample period in seconds. Default: 0.001.")
    parser.add_argument("--decimate", type=int, default=5, help="Decimation factor applied before smoothing. Default: 5.")
    parser.add_argument("--smooth", type=int, default=10, help="Centered rolling mean window. Default: 10.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional upper limit after matching both datasets.")
    parser.add_argument(
        "--crop-mode",
        choices=("head", "center"),
        default="head",
        help="Deterministic crop position. Default: head.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print statistics without writing output files.")
    return parser.parse_args()


def expanded_csv_paths(pattern: str) -> list[Path]:
    paths = sorted(Path(path) for path in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV files matched pattern: {pattern}")
    non_csv = [str(path) for path in paths if path.suffix.lower() != ".csv"]
    if non_csv:
        raise ValueError("Matched non-CSV files: " + ", ".join(non_csv))
    return paths


def prefixed_joint_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{joint}" for joint in JOINTS]


def required_columns(builder_mode: str) -> dict[str, list[str]]:
    mode = BUILDER_MODES[builder_mode]
    velocity_prefix = mode["feature_velocity_prefix"]
    return {
        "joint_pos_*": prefixed_joint_columns("joint_pos"),
        f"{velocity_prefix}_*": prefixed_joint_columns(velocity_prefix),
        "tau_residual_*": prefixed_joint_columns("tau_residual"),
    }


def validate_required_columns(df: Any, label: str, builder_mode: str) -> None:
    missing_by_group = {}
    for group, columns in required_columns(builder_mode).items():
        missing = [column for column in columns if column not in df.columns]
        if missing:
            missing_by_group[group] = missing

    if not missing_by_group:
        return

    lines = [f"{label} CSV data is missing required columns for builder-mode '{builder_mode}':"]
    for group, missing in missing_by_group.items():
        lines.append(f"  - {group}: {', '.join(missing)}")
    if builder_mode == "runtime-real-dq":
        lines.append("  - Velocity requirement: expected joint_vel_1..7 for q + measured dq features.")
    else:
        lines.append("  - Velocity requirement: expected dq_des_joint_1..7 for q + dq_des_joint features.")
    raise KeyError("\n".join(lines))


def load_csv_group(pattern: str, label: str, builder_mode: str) -> dict[str, Any]:
    paths = expanded_csv_paths(pattern)
    frames = []
    file_rows = []

    for path in paths:
        df = pd.read_csv(path)
        validate_required_columns(df, str(path), builder_mode)
        frames.append(df)
        file_rows.append({"path": str(path), "rows": int(len(df))})

    combined = pd.concat(frames, ignore_index=True)
    raw_rows = int(len(combined))
    combined = combined.dropna().reset_index(drop=True)
    return {
        "label": label,
        "paths": paths,
        "file_rows": file_rows,
        "raw_rows": raw_rows,
        "rows_after_dropna": int(len(combined)),
        "df": combined,
    }


def apply_decimate_and_smooth(df: Any, decimate: int, smooth: int) -> Any:
    if decimate < 1:
        raise ValueError("--decimate must be >= 1")
    if smooth < 1:
        raise ValueError("--smooth must be >= 1")

    result = df
    if decimate > 1:
        result = result.iloc[::decimate, :].reset_index(drop=True)
    if smooth > 1:
        num_cols = result.select_dtypes(include=["number"]).columns
        result.loc[:, num_cols] = result.loc[:, num_cols].rolling(window=smooth, center=True).mean()
        result = result.dropna().reset_index(drop=True)
    return result


def build_xy(df: Any, builder_mode: str) -> tuple[list[Any], list[Any]]:
    mode = BUILDER_MODES[builder_mode]
    velocity_prefix = mode["feature_velocity_prefix"]

    q_mat = np.stack([df[f"joint_pos_{joint}"].to_numpy(dtype=float) for joint in JOINTS], axis=1)
    dq_mat = np.stack([df[f"{velocity_prefix}_{joint}"].to_numpy(dtype=float) for joint in JOINTS], axis=1)
    y_mat = np.stack([df[f"tau_residual_{joint}"].to_numpy(dtype=float) for joint in JOINTS], axis=1)

    x_full = np.concatenate([q_mat, dq_mat], axis=1).astype(np.float32)
    y_full = y_mat.astype(np.float32)

    x_list = [x_full for _ in JOINTS]
    y_list = [y_full[:, index][:, None] for index in range(7)]
    return x_list, y_list


def crop_array(array: Any, selected_samples: int, crop_mode: str) -> Any:
    total = int(array.shape[0])
    if selected_samples > total:
        raise ValueError(f"Cannot crop {total} samples to {selected_samples}")
    if crop_mode == "head":
        start = 0
    elif crop_mode == "center":
        start = (total - selected_samples) // 2
    else:
        raise ValueError(f"Unsupported crop mode: {crop_mode}")
    end = start + selected_samples
    return array[start:end]


def crop_dataset(x_list: list[Any], y_list: list[Any], selected_samples: int, crop_mode: str) -> tuple[list[Any], list[Any]]:
    return (
        [crop_array(array, selected_samples, crop_mode) for array in x_list],
        [crop_array(array, selected_samples, crop_mode) for array in y_list],
    )


def npz_payload(
    x_list: list[Any],
    y_list: list[Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    payload = {}
    payload.update({f"X{joint}": x_list[joint - 1] for joint in JOINTS})
    payload.update({f"Y{joint}": y_list[joint - 1] for joint in JOINTS})
    payload["meta"] = np.array(meta, dtype=object)
    return payload


def dataset_summary(label: str, group: dict[str, Any], raw_samples: int, selected_samples: int) -> dict[str, Any]:
    return {
        "label": label,
        "source_csv_files": group["file_rows"],
        "raw_csv_rows": group["raw_rows"],
        "rows_after_dropna": group["rows_after_dropna"],
        "raw_dataset_samples": raw_samples,
        "selected_samples": selected_samples,
    }


def print_summary(manifest: dict[str, Any]) -> None:
    print("Stage 4 matched dataset preparation")
    print(f"builder_mode: {manifest['builder_mode']}")
    print(f"feature_definition: {manifest['feature_definition']}")
    print(f"target_definition: {manifest['target_definition']}")
    print(f"decimate: {manifest['decimate']}")
    print(f"smooth: {manifest['smooth']}")
    print(f"crop_mode: {manifest['crop_mode']}")
    print(f"selected_samples: {manifest['selected_samples']}")
    for key in ("planar", "spatial"):
        item = manifest["datasets"][key]
        print(
            f"{key}: raw_csv_rows={item['raw_csv_rows']}, "
            f"rows_after_dropna={item['rows_after_dropna']}, "
            f"raw_dataset_samples={item['raw_dataset_samples']}, "
            f"selected_samples={item['selected_samples']}"
        )
        for source in item["source_csv_files"]:
            print(f"  - {source['path']} ({source['rows']} rows)")
    if manifest["dry_run"]:
        print("dry-run: no output files written")
    else:
        print("outputs:")
        for path in manifest["output_npz_paths"].values():
            print(f"  - {path}")
        print(f"  - {manifest['manifest_path']}")


def main() -> int:
    args = parse_args()
    import_dependencies()

    planar = load_csv_group(args.planar_pattern, "planar", args.builder_mode)
    spatial = load_csv_group(args.spatial_pattern, "spatial", args.builder_mode)

    planar_df = apply_decimate_and_smooth(planar["df"], args.decimate, args.smooth)
    spatial_df = apply_decimate_and_smooth(spatial["df"], args.decimate, args.smooth)

    validate_required_columns(planar_df, "planar preprocessed", args.builder_mode)
    validate_required_columns(spatial_df, "spatial preprocessed", args.builder_mode)

    planar_x, planar_y = build_xy(planar_df, args.builder_mode)
    spatial_x, spatial_y = build_xy(spatial_df, args.builder_mode)

    planar_samples = int(planar_x[0].shape[0])
    spatial_samples = int(spatial_x[0].shape[0])
    selected_samples = min(planar_samples, spatial_samples)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be >= 1")
        selected_samples = min(selected_samples, args.max_samples)
    if selected_samples < 1:
        raise ValueError("No samples remain after preprocessing; reduce --decimate/--smooth or check CSV data.")

    planar_x, planar_y = crop_dataset(planar_x, planar_y, selected_samples, args.crop_mode)
    spatial_x, spatial_y = crop_dataset(spatial_x, spatial_y, selected_samples, args.crop_mode)

    out_dir = args.out_dir
    planar_npz = out_dir / PLANAR_OUT_NAME
    spatial_npz = out_dir / SPATIAL_OUT_NAME
    manifest_path = out_dir / MANIFEST_NAME
    mode = BUILDER_MODES[args.builder_mode]
    eff_dt = args.dt * args.decimate

    common_meta = {
        "builder_mode": args.builder_mode,
        "feature_definition": mode["feature_definition"],
        "target_definition": mode["target_definition"],
        "input_dim": int(planar_x[0].shape[1]),
        "decimate": args.decimate,
        "smooth": args.smooth,
        "dt": args.dt,
        "eff_dt": eff_dt,
        "crop_mode": args.crop_mode,
        "selected_samples": selected_samples,
    }

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__)),
        "builder_mode": args.builder_mode,
        "builder_mode_note": mode["note"],
        "feature_definition": mode["feature_definition"],
        "target_definition": mode["target_definition"],
        "runtime_feature_caveat": (
            "Current controller runtime should be kept consistent with the training feature definition. "
            "This script defaults to q + joint_vel because the main GP prediction call passes measured dq. "
            "If Stage 4 intentionally uses q + dq_des_joint instead, regenerate both datasets with "
            "--builder-mode no-filter-dq-des and train both models from that matched pair."
        ),
        "dt": args.dt,
        "decimate": args.decimate,
        "smooth": args.smooth,
        "eff_dt": eff_dt,
        "crop_mode": args.crop_mode,
        "max_samples": args.max_samples,
        "selected_samples": selected_samples,
        "dry_run": bool(args.dry_run),
        "datasets": {
            "planar": dataset_summary("planar", planar, planar_samples, selected_samples),
            "spatial": dataset_summary("spatial", spatial, spatial_samples, selected_samples),
        },
        "output_npz_paths": {
            "planar": str(planar_npz),
            "spatial": str(spatial_npz),
        },
        "manifest_path": str(manifest_path),
        "training_commands": [
            f"python3 new_structure/gp/train_gp_hdimensional.py --data {planar_npz} --out-dir data/stage4/models/GP_planar --joint all",
            f"python3 new_structure/gp/train_gp_hdimensional.py --data {spatial_npz} --out-dir data/stage4/models/GP_spatial --joint all",
        ],
    }

    print_summary(manifest)

    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(planar_npz, **npz_payload(planar_x, planar_y, {**common_meta, "dataset_label": "planar"}))
    np.savez(spatial_npz, **npz_payload(spatial_x, spatial_y, {**common_meta, "dataset_label": "spatial"}))
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
