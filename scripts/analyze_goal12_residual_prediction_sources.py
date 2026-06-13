#!/usr/bin/env python3
"""Offline GOAL12 residual prediction source RMSE analysis.

This script has no ROS dependency. It reads one or more controller CSV logs,
detects residual target and prediction columns, and compares each prediction
source against the residual torque target.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


JOINT_COUNT = 7
DEFAULT_OUT_DIR = Path("analysis_runs/goal12_residual_prediction_source_rmse")

TARGET_CANDIDATES = [
    ("y_real", "y_real"),
    ("y_target", "y_target"),
    ("tau_residual", "tau_residual"),
    ("residual_tau", "residual_tau"),
    ("y", "y"),
]

SOURCE_CANDIDATES = [
    ("local", "y_hat_local"),
    ("cloud", "y_hat_cloud"),
    ("combined", "y_hat"),
    ("shadow_combined_paper", "gp_shadow_combined_paper_raw"),
    ("triple_combined_base_shadow", "gp_triple_combined_base_shadow_raw"),
    ("hist_db_pred", "hist_db_pred"),
    ("hist_db_gated_pred", "hist_db_gated_pred"),
    ("gp_hist_db_pred", "gp_hist_db_pred"),
    ("gp_historical_pred", "gp_historical_pred"),
]

TARGET_HINTS = ("y", "tau", "residual", "target", "measured")
SUMMARY_BY_SOURCE_COLUMNS = [
    "source",
    "file_count",
    "rows_used_total",
    "rmse_tau_all_joints_mean",
    "rmse_tau_all_joints_median",
    "rmse_tau_all_joints_min",
    "rmse_tau_all_joints_max",
    "mae_tau_all_joints_mean",
    "max_abs_error_all_joints_max",
    "pred_norm_mean_mean",
    "error_norm_mean_mean",
]


def joint_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{joint}" for joint in range(1, JOINT_COUNT + 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze GOAL12 controller CSV residual torque prediction RMSE for "
            "local/cloud/combined/historical/shadow sources."
        )
    )
    parser.add_argument(
        "--csv",
        dest="csv_paths",
        action="append",
        type=Path,
        default=[],
        help="Controller CSV path. May be repeated.",
    )
    parser.add_argument(
        "--glob",
        dest="glob_patterns",
        action="append",
        default=[],
        help="Glob pattern for controller CSV paths. May be repeated.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument("--tag", default="", help="Optional analysis tag.")
    parser.add_argument("--min-rows", type=int, default=100, help="Minimum rows after trimming.")
    parser.add_argument(
        "--skip-start-rows",
        type=int,
        default=0,
        help="Rows to drop from the beginning of each CSV before analysis.",
    )
    parser.add_argument(
        "--skip-end-rows",
        type=int,
        default=0,
        help="Rows to drop from the end of each CSV before analysis.",
    )
    parser.add_argument(
        "--write-per-file",
        dest="write_per_file",
        action="store_true",
        default=True,
        help="Write one per-input source metrics CSV under per_file_details/ (default).",
    )
    parser.add_argument(
        "--no-write-per-file",
        dest="write_per_file",
        action="store_false",
        help="Skip per-input source metrics CSV files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Treat --glob directory matches as recursive CSV searches.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.min_rows < 0:
        raise ValueError(f"--min-rows must be nonnegative, got {args.min_rows}")
    if args.skip_start_rows < 0:
        raise ValueError(
            f"--skip-start-rows must be nonnegative, got {args.skip_start_rows}"
        )
    if args.skip_end_rows < 0:
        raise ValueError(f"--skip-end-rows must be nonnegative, got {args.skip_end_rows}")


def expand_input_paths(args: argparse.Namespace) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    warnings: list[str] = []

    for path in args.csv_paths:
        paths.append(path)

    for pattern in args.glob_patterns:
        matches = [Path(match) for match in glob.glob(pattern, recursive=args.recursive)]
        if not matches:
            warnings.append(f"glob matched no paths: {pattern}")
            continue
        for match in matches:
            if match.is_dir() and args.recursive:
                paths.extend(sorted(match.rglob("*.csv")))
            else:
                paths.append(match)

    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.expanduser().resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)

    return unique_paths, warnings


def find_exact_group(columns: set[str], prefix: str) -> list[str] | None:
    group = joint_columns(prefix)
    if all(column in columns for column in group):
        return group
    return None


def candidate_target_columns(columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if any(hint in column.lower() for hint in TARGET_HINTS)
    ]


def detect_target_columns(columns: list[str]) -> tuple[str, list[str]]:
    column_set = set(columns)
    for source_name, prefix in TARGET_CANDIDATES:
        group = find_exact_group(column_set, prefix)
        if group is None:
            continue
        overlap = set(group).intersection(
            column
            for _, prediction_prefix in SOURCE_CANDIDATES
            for column in joint_columns(prediction_prefix)
        )
        if overlap:
            continue
        return source_name, group

    candidates = candidate_target_columns(columns)
    candidate_text = ", ".join(candidates[:80])
    if len(candidates) > 80:
        candidate_text += f", ... ({len(candidates)} candidate columns total)"
    if not candidate_text:
        candidate_text = "<none>"
    raise ValueError(
        "could not detect residual target 7D columns. Tried prefixes: "
        + ", ".join(prefix for _, prefix in TARGET_CANDIDATES)
        + f". Candidate columns containing y/tau/residual/target/measured: {candidate_text}"
    )


def detect_prediction_sources(columns: list[str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    column_set = set(columns)
    sources: dict[str, list[str]] = {}
    missing: dict[str, str] = {}

    for source_name, prefix in SOURCE_CANDIDATES:
        group = joint_columns(prefix)
        missing_columns = [column for column in group if column not in column_set]
        if missing_columns:
            missing[source_name] = ";".join(missing_columns)
            continue
        sources[source_name] = group

    return sources, missing


def trim_frame(frame: pd.DataFrame, skip_start_rows: int, skip_end_rows: int) -> pd.DataFrame:
    start = skip_start_rows
    end = len(frame) - skip_end_rows if skip_end_rows else len(frame)
    if end < start:
        end = start
    return frame.iloc[start:end].copy()


def numeric_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return numeric.to_numpy(dtype=float)


def safe_float(value: float) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def compute_source_metrics(
    *,
    csv_path: Path,
    rows_total: int,
    rows_after_trim: int,
    target_name: str,
    target_columns: list[str],
    source_name: str,
    source_columns: list[str],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    target = numeric_matrix(frame, target_columns)
    pred = numeric_matrix(frame, source_columns)

    combined = np.concatenate([target, pred], axis=1)
    finite_mask = np.isfinite(combined).all(axis=1)
    finite_row_count = int(np.count_nonzero(finite_mask))
    nonfinite_row_count = int(len(finite_mask) - finite_row_count)

    row: dict[str, Any] = {
        "csv_path": str(csv_path),
        "source": source_name,
        "target_source": target_name,
        "target_columns": ";".join(target_columns),
        "prediction_columns": ";".join(source_columns),
        "rows_total": int(rows_total),
        "rows_after_trim": int(rows_after_trim),
        "rows_used": finite_row_count,
        "finite_row_count": finite_row_count,
        "nonfinite_row_count": nonfinite_row_count,
    }

    if finite_row_count <= 0:
        row.update(
            {
                "rmse_tau_all_joints": math.nan,
                "rmse_tau_vector_norm": math.nan,
                "mae_tau_all_joints": math.nan,
                "max_abs_error_all_joints": math.nan,
                "pred_norm_mean": math.nan,
                "error_norm_mean": math.nan,
            }
        )
        for joint in range(1, JOINT_COUNT + 1):
            row[f"rmse_tau_j{joint}"] = math.nan
        return row

    target_finite = target[finite_mask]
    pred_finite = pred[finite_mask]
    error = pred_finite - target_finite

    row["rmse_tau_all_joints"] = float(np.sqrt(np.mean(error**2)))
    row["rmse_tau_vector_norm"] = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
    row["mae_tau_all_joints"] = float(np.mean(np.abs(error)))
    row["max_abs_error_all_joints"] = float(np.max(np.abs(error)))
    row["pred_norm_mean"] = float(np.mean(np.linalg.norm(pred_finite, axis=1)))
    row["error_norm_mean"] = float(np.mean(np.linalg.norm(error, axis=1)))
    for joint_index in range(JOINT_COUNT):
        row[f"rmse_tau_j{joint_index + 1}"] = float(
            np.sqrt(np.mean(error[:, joint_index] ** 2))
        )

    return row


def metric_value(metrics_by_source: dict[str, dict[str, Any]], source: str) -> float | None:
    row = metrics_by_source.get(source)
    if row is None:
        return None
    return safe_float(row.get("rmse_tau_all_joints"))


def make_comparison_row(
    csv_path: Path,
    target_name: str,
    rows_total: int,
    rows_after_trim: int,
    metrics_by_source: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    local = metric_value(metrics_by_source, "local")
    cloud = metric_value(metrics_by_source, "cloud")
    combined = metric_value(metrics_by_source, "combined")
    shadow = metric_value(metrics_by_source, "shadow_combined_paper")
    triple_shadow = metric_value(metrics_by_source, "triple_combined_base_shadow")

    local_cloud = {
        source: value
        for source, value in (("local", local), ("cloud", cloud))
        if value is not None
    }
    best_local_cloud_source = ""
    best_local_cloud = None
    if local_cloud:
        best_local_cloud_source, best_local_cloud = min(
            local_cloud.items(), key=lambda item: item[1]
        )

    all_rmse = {
        source: safe_float(row.get("rmse_tau_all_joints"))
        for source, row in metrics_by_source.items()
    }
    finite_rmse = {source: value for source, value in all_rmse.items() if value is not None}
    best_source_by_rmse = ""
    if finite_rmse:
        best_source_by_rmse = min(finite_rmse.items(), key=lambda item: item[1])[0]

    row: dict[str, Any] = {
        "csv_path": str(csv_path),
        "target_source": target_name,
        "rows_total": int(rows_total),
        "rows_after_trim": int(rows_after_trim),
        "best_source_by_rmse": best_source_by_rmse,
        "best_local_cloud_source": best_local_cloud_source,
        "local_rmse": local,
        "cloud_rmse": cloud,
        "combined_rmse": combined,
        "shadow_combined_paper_rmse": shadow,
        "triple_combined_base_shadow_rmse": triple_shadow,
        "combined_minus_local_rmse": None,
        "combined_minus_cloud_rmse": None,
        "combined_minus_best_local_cloud_rmse": None,
        "combined_better_than_local": "",
        "combined_better_than_cloud": "",
        "combined_better_than_best_local_cloud": "",
        "shadow_minus_combined_rmse": None,
        "shadow_minus_cloud_rmse": None,
        "shadow_better_than_combined": "",
        "shadow_better_than_cloud": "",
        "triple_shadow_minus_combined_rmse": None,
        "triple_shadow_minus_cloud_rmse": None,
        "triple_shadow_better_than_combined": "",
        "triple_shadow_better_than_cloud": "",
    }

    if combined is not None and local is not None:
        row["combined_minus_local_rmse"] = combined - local
        row["combined_better_than_local"] = int(combined < local)
    if combined is not None and cloud is not None:
        row["combined_minus_cloud_rmse"] = combined - cloud
        row["combined_better_than_cloud"] = int(combined < cloud)
    if combined is not None and best_local_cloud is not None:
        row["combined_minus_best_local_cloud_rmse"] = combined - best_local_cloud
        row["combined_better_than_best_local_cloud"] = int(combined < best_local_cloud)
    if shadow is not None and combined is not None:
        row["shadow_minus_combined_rmse"] = shadow - combined
        row["shadow_better_than_combined"] = int(shadow < combined)
    if shadow is not None and cloud is not None:
        row["shadow_minus_cloud_rmse"] = shadow - cloud
        row["shadow_better_than_cloud"] = int(shadow < cloud)
    if triple_shadow is not None and combined is not None:
        row["triple_shadow_minus_combined_rmse"] = triple_shadow - combined
        row["triple_shadow_better_than_combined"] = int(triple_shadow < combined)
    if triple_shadow is not None and cloud is not None:
        row["triple_shadow_minus_cloud_rmse"] = triple_shadow - cloud
        row["triple_shadow_better_than_cloud"] = int(triple_shadow < cloud)

    return row


def missing_columns_rows(csv_path: Path, missing: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "csv_path": str(csv_path),
            "source": source,
            "missing_columns": columns,
        }
        for source, columns in sorted(missing.items())
    ]


def analyze_one_csv(
    path: Path,
    *,
    min_rows: int,
    skip_start_rows: int,
    skip_end_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if not path.exists():
        raise FileNotFoundError(f"input CSV does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"input path is not a file: {path}")

    frame = pd.read_csv(path)
    rows_total = int(len(frame))
    if rows_total <= 0:
        raise ValueError(f"CSV has no data rows: {path}")

    columns = [str(column) for column in frame.columns]
    target_name, target_columns = detect_target_columns(columns)
    sources, missing = detect_prediction_sources(columns)
    if not sources:
        raise ValueError(
            f"no prediction source columns found in {path}; missing sources: {missing}"
        )

    trimmed = trim_frame(frame, skip_start_rows, skip_end_rows)
    rows_after_trim = int(len(trimmed))
    if rows_after_trim < min_rows:
        warnings.append(
            f"{path}: rows after trim {rows_after_trim} < --min-rows {min_rows}"
        )

    metrics: list[dict[str, Any]] = []
    metrics_by_source: dict[str, dict[str, Any]] = {}
    for source_name, source_columns in sources.items():
        row = compute_source_metrics(
            csv_path=path,
            rows_total=rows_total,
            rows_after_trim=rows_after_trim,
            target_name=target_name,
            target_columns=target_columns,
            source_name=source_name,
            source_columns=source_columns,
            frame=trimmed,
        )
        metrics.append(row)
        metrics_by_source[source_name] = row
        if row["rows_used"] < min_rows:
            warnings.append(
                f"{path}: source={source_name} finite rows {row['rows_used']} "
                f"< --min-rows {min_rows}"
            )

    comparison = make_comparison_row(
        path,
        target_name,
        rows_total,
        rows_after_trim,
        metrics_by_source,
    )
    comparison["detected_sources"] = ";".join(sorted(sources))
    comparison["missing_sources"] = ";".join(sorted(missing))
    comparison["target_columns"] = ";".join(target_columns)
    return metrics, comparison, missing_columns_rows(path, missing), warnings


def summarize_by_source(long_frame: pd.DataFrame) -> pd.DataFrame:
    if long_frame.empty:
        return pd.DataFrame(columns=SUMMARY_BY_SOURCE_COLUMNS)

    rows: list[dict[str, Any]] = []
    for source, group in long_frame.groupby("source", sort=True):
        rmse = pd.to_numeric(group["rmse_tau_all_joints"], errors="coerce").dropna()
        mae = pd.to_numeric(group["mae_tau_all_joints"], errors="coerce").dropna()
        max_abs = pd.to_numeric(group["max_abs_error_all_joints"], errors="coerce").dropna()
        pred_norm = pd.to_numeric(group["pred_norm_mean"], errors="coerce").dropna()
        error_norm = pd.to_numeric(group["error_norm_mean"], errors="coerce").dropna()
        rows_used = pd.to_numeric(group["rows_used"], errors="coerce").fillna(0)

        rows.append(
            {
                "source": source,
                "file_count": int(group["csv_path"].nunique()),
                "rows_used_total": int(rows_used.sum()),
                "rmse_tau_all_joints_mean": rmse.mean() if not rmse.empty else math.nan,
                "rmse_tau_all_joints_median": rmse.median() if not rmse.empty else math.nan,
                "rmse_tau_all_joints_min": rmse.min() if not rmse.empty else math.nan,
                "rmse_tau_all_joints_max": rmse.max() if not rmse.empty else math.nan,
                "mae_tau_all_joints_mean": mae.mean() if not mae.empty else math.nan,
                "max_abs_error_all_joints_max": max_abs.max() if not max_abs.empty else math.nan,
                "pred_norm_mean_mean": pred_norm.mean() if not pred_norm.empty else math.nan,
                "error_norm_mean_mean": error_norm.mean() if not error_norm.empty else math.nan,
            }
        )

    return pd.DataFrame(rows, columns=SUMMARY_BY_SOURCE_COLUMNS)


def write_optional_plots(out_dir: Path, summary_by_source: pd.DataFrame, comparison: pd.DataFrame) -> list[str]:
    written: list[str] = []
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/gp_torque_matplotlib_cache")
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return written

    if not summary_by_source.empty and "rmse_tau_all_joints_mean" in summary_by_source:
        plot_frame = summary_by_source.dropna(subset=["rmse_tau_all_joints_mean"])
        if not plot_frame.empty:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.bar(plot_frame["source"], plot_frame["rmse_tau_all_joints_mean"])
            ax.set_ylabel("RMSE tau all joints")
            ax.set_title("Residual Prediction Source RMSE")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            path = out_dir / "source_rmse_bar.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(str(path))

    if not comparison.empty and "combined_minus_best_local_cloud_rmse" in comparison:
        deltas = pd.to_numeric(
            comparison["combined_minus_best_local_cloud_rmse"], errors="coerce"
        )
        plot_frame = comparison.loc[deltas.notna()].copy()
        if not plot_frame.empty:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            labels = [Path(path).stem for path in plot_frame["csv_path"]]
            ax.bar(labels, deltas[deltas.notna()])
            ax.axhline(0.0, color="black", linewidth=1)
            ax.set_ylabel("combined - best(local, cloud) RMSE")
            ax.set_title("Combined RMSE Delta")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            path = out_dir / "combined_delta_bar.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(str(path))

    return written


def write_outputs(
    *,
    args: argparse.Namespace,
    input_paths: list[Path],
    input_warnings: list[str],
    long_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
    file_errors: list[dict[str, Any]],
    analysis_warnings: list[str],
) -> dict[str, Any]:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    long_frame = pd.DataFrame(long_rows)
    comparison_frame = pd.DataFrame(comparison_rows)
    missing_frame = pd.DataFrame(missing_rows)
    errors_frame = pd.DataFrame(file_errors)
    summary_by_source = summarize_by_source(long_frame)

    per_file_path = out_dir / "residual_prediction_source_rmse_per_file.csv"
    long_path = out_dir / "residual_prediction_source_rmse_long.csv"
    summary_path = out_dir / "residual_prediction_source_rmse_summary_by_source.csv"
    comparison_path = out_dir / "combined_vs_local_cloud_summary.csv"
    manifest_path = out_dir / "analysis_manifest.json"

    per_file_frame = long_frame.copy()
    if not missing_frame.empty:
        missing_summary = (
            missing_frame.groupby("csv_path")["source"]
            .apply(lambda values: ";".join(sorted(values)))
            .rename("missing_columns")
            .reset_index()
        )
        per_file_frame = per_file_frame.merge(missing_summary, on="csv_path", how="left")
    else:
        per_file_frame["missing_columns"] = ""

    per_file_frame.to_csv(per_file_path, index=False)
    long_frame.to_csv(long_path, index=False)
    summary_by_source.to_csv(summary_path, index=False)
    comparison_frame.to_csv(comparison_path, index=False)

    extra_outputs: list[str] = []
    if not missing_frame.empty:
        missing_path = out_dir / "missing_prediction_source_columns.csv"
        missing_frame.to_csv(missing_path, index=False)
        extra_outputs.append(str(missing_path))
    if not errors_frame.empty:
        errors_path = out_dir / "file_errors.csv"
        errors_frame.to_csv(errors_path, index=False)
        extra_outputs.append(str(errors_path))

    if args.write_per_file and long_rows:
        details_dir = out_dir / "per_file_details"
        details_dir.mkdir(exist_ok=True)
        for csv_path, group in long_frame.groupby("csv_path", sort=False):
            safe_name = Path(str(csv_path)).stem or "csv"
            detail_path = details_dir / f"{safe_name}_source_metrics.csv"
            suffix = 1
            while detail_path.exists():
                detail_path = details_dir / f"{safe_name}_{suffix}_source_metrics.csv"
                suffix += 1
            group.to_csv(detail_path, index=False)
            extra_outputs.append(str(detail_path))

    plot_outputs = write_optional_plots(out_dir, summary_by_source, comparison_frame)
    extra_outputs.extend(plot_outputs)

    manifest = {
        "script": Path(__file__).name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "inputs_requested": [str(path) for path in input_paths],
        "input_count": len(input_paths),
        "files_analyzed": int(comparison_frame["csv_path"].nunique())
        if not comparison_frame.empty
        else 0,
        "file_error_count": len(file_errors),
        "min_rows": args.min_rows,
        "skip_start_rows": args.skip_start_rows,
        "skip_end_rows": args.skip_end_rows,
        "target_candidates": [prefix for _, prefix in TARGET_CANDIDATES],
        "source_candidates": {
            source_name: joint_columns(prefix)
            for source_name, prefix in SOURCE_CANDIDATES
        },
        "outputs": [
            str(per_file_path),
            str(long_path),
            str(summary_path),
            str(comparison_path),
            str(manifest_path),
        ]
        + extra_outputs,
        "warnings": input_warnings + analysis_warnings,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    input_paths, input_warnings = expand_input_paths(args)
    if not input_paths:
        print("ERROR: no input CSVs. Use --csv and/or --glob.", file=sys.stderr)
        return 2

    for warning in input_warnings:
        print(f"WARNING: {warning}")

    long_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    file_errors: list[dict[str, Any]] = []
    analysis_warnings: list[str] = []

    for path in input_paths:
        try:
            metrics, comparison, missing, warnings = analyze_one_csv(
                path,
                min_rows=args.min_rows,
                skip_start_rows=args.skip_start_rows,
                skip_end_rows=args.skip_end_rows,
            )
            long_rows.extend(metrics)
            comparison_rows.append(comparison)
            missing_rows.extend(missing)
            analysis_warnings.extend(warnings)
            for warning in warnings:
                print(f"WARNING: {warning}")
            print(f"OK: analyzed {path} with {len(metrics)} source(s)")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"WARNING: skipping {path}: {message}")
            file_errors.append({"csv_path": str(path), "error": message})

    manifest = write_outputs(
        args=args,
        input_paths=input_paths,
        input_warnings=input_warnings,
        long_rows=long_rows,
        comparison_rows=comparison_rows,
        missing_rows=missing_rows,
        file_errors=file_errors,
        analysis_warnings=analysis_warnings,
    )

    print(f"Wrote outputs to {args.out_dir}")
    print(
        "Analyzed "
        f"{manifest['files_analyzed']} file(s); "
        f"{manifest['file_error_count']} file(s) had errors."
    )
    return 0 if manifest["files_analyzed"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
