#!/usr/bin/env python3
"""Build a matched Stage 4 GP dataset from one formal CSV.

This script is offline-only. It reads a source CSV and writes generated dataset
artifacts; it does not modify controller, launch, config, model, or raw CSV
files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


try:
    import numpy as np
except ModuleNotFoundError as exc:
    print("Missing Python dependency: numpy", file=sys.stderr)
    print("Use an environment that already has project dependencies installed.", file=sys.stderr)
    raise SystemExit(1) from exc


JOINTS = range(1, 8)
SCRIPT_VERSION = "2026-05-24-stage4-matched-dataset-v1"
RECONSTRUCT_DIFF_WARN_NM = 1e-6
NEAR_ZERO_STD_WARN = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an offline Stage 4 matched GP training dataset from a strict_no_gp formal CSV.",
    )
    parser.add_argument("--source-csv", type=Path, required=True, help="Input formal CSV path.")
    parser.add_argument("--out-npz", type=Path, required=True, help="Output .npz path.")
    parser.add_argument("--out-summary", type=Path, required=True, help="Output Markdown summary path.")
    parser.add_argument("--mode-name", required=True, help="Mode label stored in metadata.")
    parser.add_argument(
        "--target-kind",
        choices=("tau_residual", "tau_residual_raw", "reconstructed_raw"),
        default="tau_residual_raw",
        help="Target definition. Default: tau_residual_raw.",
    )
    parser.add_argument(
        "--feature-source",
        choices=("joint_vel", "dq_des_joint"),
        default="joint_vel",
        help="Velocity feature source for X = joint_pos_1..7 + velocity_1..7. Default: joint_vel.",
    )
    parser.add_argument("--drop-nan", action="store_true", help="Drop rows containing NaN or Inf in X/Y.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap after any NaN/Inf drop.")
    return parser.parse_args()


def prefixed_joint_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{joint}" for joint in JOINTS]


def feature_columns(feature_source: str) -> list[str]:
    return prefixed_joint_columns("joint_pos") + prefixed_joint_columns(feature_source)


def target_columns(target_kind: str) -> list[str]:
    if target_kind == "tau_residual":
        return prefixed_joint_columns("tau_residual")
    if target_kind == "tau_residual_raw":
        return prefixed_joint_columns("tau_residual_raw")
    if target_kind == "reconstructed_raw":
        return [f"tau_measured_{joint} - gravity_{joint} - tau_{joint}" for joint in JOINTS]
    raise ValueError(f"Unsupported target kind: {target_kind}")


def reconstructed_required_columns() -> list[str]:
    return (
        prefixed_joint_columns("tau_measured")
        + prefixed_joint_columns("gravity")
        + prefixed_joint_columns("tau")
    )


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


def require_columns(dataset: dict[str, Any], columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in dataset["columns"]]
    if missing:
        raise KeyError(f"{label} is missing required columns: {', '.join(missing)}")


def stack_columns(dataset: dict[str, Any], columns: list[str], label: str) -> np.ndarray:
    require_columns(dataset, columns, label)
    data = dataset["data"]
    return np.stack([data[column] for column in columns], axis=1).astype(np.float32)


def build_target_matrix(dataset: dict[str, Any], kind: str) -> tuple[np.ndarray, list[str]]:
    if kind in ("tau_residual", "tau_residual_raw"):
        columns = target_columns(kind)
        return stack_columns(dataset, columns, f"{kind} target"), columns

    require_columns(dataset, reconstructed_required_columns(), "reconstructed_raw target")
    data = dataset["data"]
    matrix = np.stack(
        [
            data[f"tau_measured_{joint}"] - data[f"gravity_{joint}"] - data[f"tau_{joint}"]
            for joint in JOINTS
        ],
        axis=1,
    ).astype(np.float32)
    return matrix, target_columns(kind)


def nan_inf_counts(*arrays: np.ndarray) -> dict[str, int]:
    nan_count = 0
    inf_count = 0
    for array in arrays:
        nan_count += int(np.isnan(array).sum())
        inf_count += int(np.isinf(array).sum())
    return {"nan_count": nan_count, "inf_count": inf_count}


def finite_stats(matrix: np.ndarray, columns: list[str]) -> list[dict[str, Any]]:
    rows = []
    for index, column in enumerate(columns):
        values = matrix[:, index].astype(float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            stats = {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "span": math.nan}
        else:
            minimum = float(np.min(finite))
            maximum = float(np.max(finite))
            stats = {
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "min": minimum,
                "max": maximum,
                "span": maximum - minimum,
            }
        rows.append({"column": column, **stats})
    return rows


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


def stats_markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| column | mean | std | min | max | span |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            "| {column} | {mean} | {std} | {min} | {max} | {span} |".format(
                column=row["column"],
                mean=format_float(row["mean"]),
                std=format_float(row["std"]),
                min=format_float(row["min"]),
                max=format_float(row["max"]),
                span=format_float(row["span"]),
            )
        )
    return "\n".join(lines)


def stats_to_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "mean": np.asarray([row["mean"] for row in rows], dtype=np.float64),
        "std": np.asarray([row["std"] for row in rows], dtype=np.float64),
        "min": np.asarray([row["min"] for row in rows], dtype=np.float64),
        "max": np.asarray([row["max"] for row in rows], dtype=np.float64),
        "span": np.asarray([row["span"] for row in rows], dtype=np.float64),
    }


def raw_reconstruction_diff(dataset: dict[str, Any]) -> tuple[np.ndarray | None, list[str]]:
    raw_columns = prefixed_joint_columns("tau_residual_raw")
    needed = raw_columns + reconstructed_required_columns()
    missing = [column for column in needed if column not in dataset["columns"]]
    if missing:
        return None, missing

    data = dataset["data"]
    diffs = []
    for joint in JOINTS:
        reconstructed = data[f"tau_measured_{joint}"] - data[f"gravity_{joint}"] - data[f"tau_{joint}"]
        raw = data[f"tau_residual_raw_{joint}"]
        diffs.append(float(np.nanmax(np.abs(raw - reconstructed))))
    return np.asarray(diffs, dtype=float), []


def rows_with_finite_xy(x_matrix: np.ndarray, y_matrix: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(x_matrix), axis=1) & np.all(np.isfinite(y_matrix), axis=1)


def write_summary(
    path: Path,
    metadata: dict[str, Any],
    feature_stats: list[dict[str, Any]],
    target_stats: list[dict[str, Any]],
    q7_stats: dict[str, Any],
    near_zero_features: list[dict[str, Any]],
    reconstruction_diff: np.ndarray | None,
    reconstruction_missing: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Stage 4 Matched GP Dataset: {metadata['mode_name']}",
        "",
        "## Dataset Source",
        "",
        f"- source_csv: `{metadata['source_csv']}`",
        f"- out_npz: `{metadata['out_npz']}`",
        f"- created_utc: `{metadata['created_utc']}`",
        f"- script_version: `{metadata['script_version']}`",
        "",
        "## Feature Definition",
        "",
        f"- feature_source: `{metadata['feature_source']}`",
        "- X shape: `{}`".format(metadata["x_shape"]),
        "- X columns: `" + ", ".join(metadata["feature_columns"]) + "`",
        "",
        "## Target Definition",
        "",
        f"- target_kind: `{metadata['target_kind']}`",
        "- Y shape: `{}`".format(metadata["y_shape"]),
        "- Y columns: `" + ", ".join(metadata["target_columns"]) + "`",
        "",
        "## Row Count",
        "",
        f"- source_rows: `{metadata['source_rows']}`",
        f"- rows_written: `{metadata['rows_written']}`",
        f"- rows_dropped_nonfinite: `{metadata['rows_dropped_nonfinite']}`",
        f"- max_rows: `{metadata['max_rows']}`",
        "",
        "## NaN / Inf Check",
        "",
        f"- pre_filter_nan_count: `{metadata['pre_filter_nan_count']}`",
        f"- pre_filter_inf_count: `{metadata['pre_filter_inf_count']}`",
        f"- output_nan_count: `{metadata['output_nan_count']}`",
        f"- output_inf_count: `{metadata['output_inf_count']}`",
        "",
        "## Feature Distribution",
        "",
        stats_markdown_table(feature_stats),
        "",
        "## Target Distribution",
        "",
        stats_markdown_table(target_stats),
        "",
        "## Near-Zero Feature Std Warnings",
        "",
    ]

    if near_zero_features:
        lines.extend(
            f"- `{row['column']}` std=`{format_float(row['std'])}`" for row in near_zero_features
        )
    else:
        lines.append(f"- none with threshold `{NEAR_ZERO_STD_WARN}`")

    lines.extend(
        [
            "",
            "## q7 Stats",
            "",
            f"- column: `{q7_stats['column']}`",
            f"- mean: `{format_float(q7_stats['mean'])}`",
            f"- std: `{format_float(q7_stats['std'])}`",
            f"- min: `{format_float(q7_stats['min'])}`",
            f"- max: `{format_float(q7_stats['max'])}`",
            f"- span: `{format_float(q7_stats['span'])}`",
            "",
            "## tau_residual_raw Reconstruction Cross-Check",
            "",
        ]
    )

    if reconstruction_diff is None:
        lines.append("- skipped; missing columns: `" + ", ".join(reconstruction_missing) + "`")
    else:
        max_diff = float(np.nanmax(reconstruction_diff))
        status = "WARN" if max_diff > RECONSTRUCT_DIFF_WARN_NM else "OK"
        lines.append(f"- status: `{status}`")
        lines.append(f"- warning_threshold_nm: `{RECONSTRUCT_DIFF_WARN_NM}`")
        lines.append("- max_abs_diff_per_joint_nm: `" + ", ".join(format_float(v) for v in reconstruction_diff) + "`")
        lines.append(f"- max_abs_diff_nm: `{format_float(max_diff)}`")

    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "- Using `strict_no_gp` formal CSV as a matched training source is acceptable for an engineering sanity check.",
            "- It introduces train/test leakage if evaluated on the same or near-identical formal trajectory.",
            "- It is not sufficient for a paper-level generalization claim.",
            "",
            "## Next Step",
            "",
            "- Train frozen local GP using this dataset.",
            "- Run `scripts/validate_frozen_gp_support.py` before any robot run.",
            "- Do not run the robot before the validator passes.",
            "- Keep conservative GP compensation gating, scale, and clip; do not use unlimited compensation or direct `scale=1.0` robot runs.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be >= 1 when provided")

    dataset = load_csv_numeric(args.source_csv)
    features = feature_columns(args.feature_source)
    x_matrix = stack_columns(dataset, features, "feature matrix")
    y_matrix, targets = build_target_matrix(dataset, args.target_kind)
    if x_matrix.shape[1] != 14:
        raise ValueError(f"Expected X shape (N, 14), got {x_matrix.shape}")
    if y_matrix.shape[1] != 7:
        raise ValueError(f"Expected Y shape (N, 7), got {y_matrix.shape}")
    if x_matrix.shape[0] != y_matrix.shape[0]:
        raise ValueError(f"X/Y row mismatch: {x_matrix.shape[0]} vs {y_matrix.shape[0]}")

    pre_counts = nan_inf_counts(x_matrix, y_matrix)
    rows_dropped = 0
    if args.drop_nan:
        keep = rows_with_finite_xy(x_matrix, y_matrix)
        rows_dropped = int(keep.size - np.sum(keep))
        x_matrix = x_matrix[keep]
        y_matrix = y_matrix[keep]

    if args.max_rows is not None:
        x_matrix = x_matrix[: args.max_rows]
        y_matrix = y_matrix[: args.max_rows]

    output_counts = nan_inf_counts(x_matrix, y_matrix)
    feature_stats = finite_stats(x_matrix, features)
    target_stats = finite_stats(y_matrix, targets)
    feature_arrays = stats_to_arrays(feature_stats)
    target_arrays = stats_to_arrays(target_stats)
    near_zero_features = [
        row for row in feature_stats if math.isfinite(float(row["std"])) and abs(float(row["std"])) < NEAR_ZERO_STD_WARN
    ]
    q7_stats = next(row for row in feature_stats if row["column"] == "joint_pos_7")
    reconstruction_diff, reconstruction_missing = raw_reconstruction_diff(dataset)

    created_utc = datetime.now(timezone.utc).isoformat()
    metadata = {
        "script": str(Path(__file__)),
        "script_version": SCRIPT_VERSION,
        "created_utc": created_utc,
        "source_csv": str(args.source_csv),
        "out_npz": str(args.out_npz),
        "out_summary": str(args.out_summary),
        "mode_name": args.mode_name,
        "feature_source": args.feature_source,
        "feature_definition": "X = [joint_pos_1..7, {}_1..7]".format(args.feature_source),
        "feature_columns": features,
        "target_kind": args.target_kind,
        "target_columns": targets,
        "target_definition": (
            "Y = tau_measured_j - gravity_j - tau_j"
            if args.target_kind == "reconstructed_raw"
            else f"Y = {args.target_kind}_1..7"
        ),
        "source_rows": int(dataset["rows"]),
        "rows_written": int(x_matrix.shape[0]),
        "rows_dropped_nonfinite": rows_dropped,
        "drop_nan": bool(args.drop_nan),
        "max_rows": args.max_rows,
        "x_shape": list(x_matrix.shape),
        "y_shape": list(y_matrix.shape),
        "pre_filter_nan_count": pre_counts["nan_count"],
        "pre_filter_inf_count": pre_counts["inf_count"],
        "output_nan_count": output_counts["nan_count"],
        "output_inf_count": output_counts["inf_count"],
        "near_zero_std_threshold": NEAR_ZERO_STD_WARN,
        "near_zero_feature_columns": [row["column"] for row in near_zero_features],
        "reconstruct_diff_warn_nm": RECONSTRUCT_DIFF_WARN_NM,
        "reconstruct_raw_max_abs_diff_per_joint_nm": (
            reconstruction_diff.tolist() if reconstruction_diff is not None else None
        ),
        "engineering_sanity_caveat": (
            "Matched strict_no_gp data is useful for engineering sanity checks, but evaluating on the same "
            "or near-identical formal trajectory introduces train/test leakage and is not a paper-level "
            "generalization proof."
        ),
    }

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_npz,
        X=x_matrix,
        Y=y_matrix,
        **{f"X{joint}": x_matrix for joint in JOINTS},
        **{f"Y{joint}": y_matrix[:, joint - 1 : joint] for joint in JOINTS},
        feature_columns=np.asarray(features, dtype=object),
        target_columns=np.asarray(targets, dtype=object),
        source_csv=np.asarray(str(args.source_csv)),
        mode_name=np.asarray(args.mode_name),
        target_kind=np.asarray(args.target_kind),
        feature_source=np.asarray(args.feature_source),
        metadata_json=np.asarray(json.dumps(metadata, indent=2, sort_keys=True)),
        feature_mean=feature_arrays["mean"],
        feature_std=feature_arrays["std"],
        feature_min=feature_arrays["min"],
        feature_max=feature_arrays["max"],
        feature_span=feature_arrays["span"],
        target_mean=target_arrays["mean"],
        target_std=target_arrays["std"],
        target_min=target_arrays["min"],
        target_max=target_arrays["max"],
        target_span=target_arrays["span"],
        meta=np.asarray(metadata, dtype=object),
    )

    write_summary(
        args.out_summary,
        metadata,
        feature_stats,
        target_stats,
        q7_stats,
        near_zero_features,
        reconstruction_diff,
        reconstruction_missing,
    )

    print(f"wrote_npz: {args.out_npz}")
    print(f"wrote_summary: {args.out_summary}")
    print(f"X_shape: {x_matrix.shape}")
    print(f"Y_shape: {y_matrix.shape}")
    print(f"output_nan_count: {output_counts['nan_count']}")
    print(f"output_inf_count: {output_counts['inf_count']}")
    if near_zero_features:
        print("near_zero_feature_std: " + ", ".join(row["column"] for row in near_zero_features))
    else:
        print("near_zero_feature_std: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
