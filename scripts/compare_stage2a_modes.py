#!/usr/bin/env python3
"""Compare Stage 2A offline CSV modes.

This script has no ROS2 dependency. It loads saved Stage 2A CSV logs, detects
common column groups, writes machine-readable summaries, and saves quick-look
PNG plots for offline comparison.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import math
import os
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT_DIR = Path("data/stage2a/csv")
DEFAULT_OUTPUT_DIR = Path("outputs/stage2b_comparison")
EPS = 1e-9

np = None
plt = None


TIMING_FIELDS = [
    "mode_name",
    "csv_file",
    "rows",
    "columns",
    "time_col",
    "duration_s",
    "median_dt_s",
    "mean_dt_s",
    "std_dt_s",
    "min_dt_s",
    "max_dt_s",
    "estimated_hz",
    "nan_count",
    "inf_count",
]

CARTESIAN_FIELDS = [
    "mode_name",
    "csv_file",
    "actual_columns",
    "desired_columns",
    "rmse_x",
    "rmse_y",
    "rmse_z",
    "rmse_norm",
    "max_abs_x",
    "max_abs_y",
    "max_abs_z",
    "p95_norm",
]

TAU_LIKE_FIELDS = [
    "mode_name",
    "csv_file",
    "joint",
    "column",
    "mean",
    "std",
    "rms",
    "abs_mean",
    "p95_abs",
    "min",
    "max",
]

GP_PREDICTION_FIELDS = [
    "mode_name",
    "csv_file",
    "joint",
    "column",
    "mean",
    "std",
    "min",
    "max",
    "range",
    "rms",
    "p95_abs",
    "variation_ratio",
]

CLIP_PROXY_FIELDS = [
    "mode_name",
    "csv_file",
    "joint",
    "column",
    "scaled_y_hat_abs_max",
    "scaled_y_hat_abs_p95",
    "clip_proxy_count",
    "clip_proxy_ratio",
]

RESIDUAL_COMPARISON_FIELDS = [
    "comparison",
    "joint",
    "rms_no_gp",
    "rms_gp_on",
    "rms_change_percent",
]


def import_dependencies(output_dir: Path) -> bool:
    global np, plt

    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = output_dir / ".matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    missing = []
    try:
        import numpy as numpy_module
    except ModuleNotFoundError:
        missing.append("numpy")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot_module
    except ModuleNotFoundError:
        missing.append("matplotlib")

    if missing:
        print("Missing Python dependencies: " + ", ".join(sorted(set(missing))), file=sys.stderr)
        print("Use a project environment with these packages installed; do not install globally by default.", file=sys.stderr)
        return False

    np = numpy_module
    plt = pyplot_module
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Stage 2A CSV modes offline without ROS2.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Filename/stem pattern to include. May be repeated. If omitted, all CSVs are included.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Filename/stem pattern to exclude. May be repeated.",
    )
    parser.add_argument(
        "--gp-scale",
        type=float,
        default=0.1,
        help="Scale used for clip proxy calculation. Default: 0.1",
    )
    parser.add_argument(
        "--gp-clip-nm",
        type=float,
        default=0.5,
        help="Clip threshold in Nm used for clip proxy calculation. Default: 0.5",
    )
    return parser.parse_args()


def normalized_name(name: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in name]
    return "_".join(part for part in "".join(chars).split("_") if part)


def matches_pattern(path: Path, pattern: str) -> bool:
    pattern_lower = pattern.lower()
    candidates = [path.name.lower(), path.stem.lower()]
    return any(
        fnmatch.fnmatch(candidate, pattern_lower) or pattern_lower in candidate
        for candidate in candidates
    )


def find_csv_files(input_dir: Path, include: list[str], exclude: list[str]) -> list[Path]:
    if not input_dir.exists():
        return []

    paths = sorted(input_dir.glob("*.csv"))
    if include:
        paths = [path for path in paths if any(matches_pattern(path, pattern) for pattern in include)]
    if exclude:
        paths = [path for path in paths if not any(matches_pattern(path, pattern) for pattern in exclude)]
    return paths


def infer_mode_name(path: Path) -> str:
    stem = normalized_name(path.stem)
    if "pure_no_gp" in stem:
        return "pure_no_gp"
    if "gpon" in stem or "gp_on" in stem:
        return "gp_on_conservative"
    if "compute" in stem:
        return "compute_only"
    if "stage1" in stem:
        return "stage1_baseline"
    return path.stem


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


def load_csv(path: Path) -> dict[str, object] | None:
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                print(f"WARNING: {path}: no CSV header found")
                return None

            columns = list(reader.fieldnames)
            data = {column: [] for column in columns}
            for row in reader:
                for column in columns:
                    data[column].append(parse_float(row.get(column)))
    except Exception as exc:  # pragma: no cover - defensive CLI behavior
        print(f"WARNING: failed to read {path}: {exc}")
        return None

    arrays = {column: np.asarray(values, dtype=float) for column, values in data.items()}
    rows = len(next(iter(arrays.values()))) if arrays else 0
    mode_name = infer_mode_name(path)
    print(f"Loaded {path.name}: mode={mode_name}, rows={rows}, columns={len(columns)}")
    return {
        "path": path,
        "columns": columns,
        "data": arrays,
        "rows": rows,
        "mode_name": mode_name,
    }


def load_datasets(csv_files: list[Path]) -> list[dict[str, object]]:
    datasets = []
    for path in csv_files:
        dataset = load_csv(path)
        if dataset is not None:
            datasets.append(dataset)
    return datasets


def detect_time_column(columns: Iterable[str]) -> str | None:
    candidates = []
    for column in columns:
        lowered = normalized_name(column)
        if any(pattern in lowered for pattern in ("time", "timestamp", "elapsed")):
            candidates.append(column)
        elif lowered == "t":
            candidates.append(column)

    if not candidates:
        return None
    for preferred in ("time_s", "time", "elapsed", "timestamp"):
        for column in candidates:
            if preferred in normalized_name(column):
                return column
    return candidates[0]


def finite_values(values: "np.ndarray") -> "np.ndarray":
    return values[np.isfinite(values)]


def rms(values: "np.ndarray") -> float:
    if len(values) == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.square(values))))


def p95_abs(values: "np.ndarray") -> float:
    if len(values) == 0:
        return math.nan
    return float(np.percentile(np.abs(values), 95))


def summarize_time(values: "np.ndarray" | None) -> dict[str, float]:
    result = {
        "duration_s": math.nan,
        "median_dt_s": math.nan,
        "mean_dt_s": math.nan,
        "std_dt_s": math.nan,
        "min_dt_s": math.nan,
        "max_dt_s": math.nan,
        "estimated_hz": math.nan,
    }
    if values is None:
        return result

    series = finite_values(values)
    if len(series) < 2:
        return result

    result["duration_s"] = float(series[-1] - series[0])
    diffs = np.diff(series)
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(positive_diffs) == 0:
        return result

    result["median_dt_s"] = float(np.median(positive_diffs))
    result["mean_dt_s"] = float(np.mean(positive_diffs))
    result["std_dt_s"] = float(np.std(positive_diffs))
    result["min_dt_s"] = float(np.min(positive_diffs))
    result["max_dt_s"] = float(np.max(positive_diffs))
    if result["median_dt_s"] > 0:
        result["estimated_hz"] = float(1.0 / result["median_dt_s"])
    return result


def make_timing_summary(dataset: dict[str, object]) -> dict[str, object]:
    columns = dataset["columns"]
    data = dataset["data"]
    path = dataset["path"]
    time_col = detect_time_column(columns)
    all_values = np.concatenate(list(data.values())) if data else np.asarray([], dtype=float)
    row = {
        "mode_name": dataset["mode_name"],
        "csv_file": path.name,
        "rows": dataset["rows"],
        "columns": len(columns),
        "time_col": time_col or "",
        "nan_count": int(np.isnan(all_values).sum()),
        "inf_count": int(np.isinf(all_values).sum()),
    }
    row.update(summarize_time(data.get(time_col) if time_col else None))
    return row


def first_matching_triplet(
    normalized_to_original: dict[str, str],
    patterns: list[tuple[str, str, str]],
) -> list[str] | None:
    for pattern in patterns:
        normalized_pattern = tuple(normalized_name(item) for item in pattern)
        if all(item in normalized_to_original for item in normalized_pattern):
            return [normalized_to_original[item] for item in normalized_pattern]
    return None


def detect_cartesian_position_columns(columns: Iterable[str]) -> tuple[list[str] | None, list[str] | None]:
    normalized_to_original = {normalized_name(column): column for column in columns}
    actual_patterns = [
        ("x_actual", "y_actual", "z_actual"),
        ("actual_x", "actual_y", "actual_z"),
        ("position_actual_x", "position_actual_y", "position_actual_z"),
    ]
    desired_patterns = [
        ("x_desired", "y_desired", "z_desired"),
        ("desired_x", "desired_y", "desired_z"),
        ("position_desired_x", "position_desired_y", "position_desired_z"),
    ]
    return (
        first_matching_triplet(normalized_to_original, actual_patterns),
        first_matching_triplet(normalized_to_original, desired_patterns),
    )


def make_cartesian_tracking_summary(
    datasets: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, tuple[list[str], list[str]]]]:
    rows = []
    detected: dict[str, tuple[list[str], list[str]]] = {}

    for dataset in datasets:
        path = dataset["path"]
        columns = dataset["columns"]
        data = dataset["data"]
        actual, desired = detect_cartesian_position_columns(columns)
        if not actual or not desired:
            print(f"WARNING: {path.name}: Cartesian actual/desired position columns not found; skipping tracking metrics.")
            continue

        errors = []
        row = {
            "mode_name": dataset["mode_name"],
            "csv_file": path.name,
            "actual_columns": ",".join(actual),
            "desired_columns": ",".join(desired),
        }
        for axis, actual_col, desired_col in zip(("x", "y", "z"), actual, desired):
            error = data[actual_col] - data[desired_col]
            finite = finite_values(error)
            row[f"rmse_{axis}"] = rms(finite)
            row[f"max_abs_{axis}"] = float(np.max(np.abs(finite))) if len(finite) else math.nan
            errors.append(error)

        error_matrix = np.vstack(errors).T
        finite_rows = error_matrix[np.all(np.isfinite(error_matrix), axis=1)]
        if len(finite_rows):
            norms = np.linalg.norm(finite_rows, axis=1)
            row["rmse_norm"] = rms(norms)
            row["p95_norm"] = float(np.percentile(norms, 95))
        else:
            row["rmse_norm"] = math.nan
            row["p95_norm"] = math.nan

        rows.append(row)
        detected[path.name] = (actual, desired)

    return rows, detected


def joint_candidate_names(prefixes: Iterable[str], suffix: int, style: str) -> list[str]:
    names = []
    for prefix in prefixes:
        if style == "plain":
            names.append(f"{prefix}_{suffix}")
        elif style == "j":
            names.append(f"{prefix}_j{suffix}")
    return names


def detect_joint_columns(columns: Iterable[str], prefixes: Iterable[str]) -> list[tuple[int, str]]:
    normalized_to_original = {normalized_name(column): column for column in columns}
    normalized_prefixes = [normalized_name(prefix) for prefix in prefixes]

    for base in (1, 0):
        detected = []
        for offset in range(7):
            suffix = base + offset
            candidates = []
            candidates.extend(joint_candidate_names(normalized_prefixes, suffix, "plain"))
            candidates.extend(joint_candidate_names(normalized_prefixes, suffix, "j"))
            match = next((normalized_to_original[name] for name in candidates if name in normalized_to_original), None)
            if match is not None:
                detected.append((offset + 1, match))
        if len(detected) == 7:
            return detected

    detected = []
    seen_columns = set()
    for base in (1, 0):
        for offset in range(7):
            suffix = base + offset
            candidates = []
            candidates.extend(joint_candidate_names(normalized_prefixes, suffix, "plain"))
            candidates.extend(joint_candidate_names(normalized_prefixes, suffix, "j"))
            match = next((normalized_to_original[name] for name in candidates if name in normalized_to_original), None)
            if match is not None and match not in seen_columns:
                detected.append((offset + 1, match))
                seen_columns.add(match)
    return sorted(detected, key=lambda item: item[0])


def make_tau_like_row(dataset: dict[str, object], joint: int, column: str) -> dict[str, object]:
    values = finite_values(dataset["data"][column])
    return {
        "mode_name": dataset["mode_name"],
        "csv_file": dataset["path"].name,
        "joint": joint,
        "column": column,
        "mean": float(np.mean(values)) if len(values) else math.nan,
        "std": float(np.std(values)) if len(values) else math.nan,
        "rms": rms(values),
        "abs_mean": float(np.mean(np.abs(values))) if len(values) else math.nan,
        "p95_abs": p95_abs(values),
        "min": float(np.min(values)) if len(values) else math.nan,
        "max": float(np.max(values)) if len(values) else math.nan,
    }


def make_prediction_row(dataset: dict[str, object], joint: int, column: str) -> dict[str, object]:
    values = finite_values(dataset["data"][column])
    mean = float(np.mean(values)) if len(values) else math.nan
    std = float(np.std(values)) if len(values) else math.nan
    min_value = float(np.min(values)) if len(values) else math.nan
    max_value = float(np.max(values)) if len(values) else math.nan
    return {
        "mode_name": dataset["mode_name"],
        "csv_file": dataset["path"].name,
        "joint": joint,
        "column": column,
        "mean": mean,
        "std": std,
        "min": min_value,
        "max": max_value,
        "range": max_value - min_value if len(values) else math.nan,
        "rms": rms(values),
        "p95_abs": p95_abs(values),
        "variation_ratio": std / (abs(mean) + EPS) if len(values) else math.nan,
    }


def make_joint_metric_rows(
    datasets: list[dict[str, object]],
    prefixes: Iterable[str],
    group_label: str,
    metrics: str,
) -> tuple[list[dict[str, object]], dict[str, list[tuple[int, str]]]]:
    rows = []
    detected: dict[str, list[tuple[int, str]]] = {}
    for dataset in datasets:
        path = dataset["path"]
        joint_columns = detect_joint_columns(dataset["columns"], prefixes)
        if not joint_columns:
            print(f"WARNING: {path.name}: {group_label} columns not found; skipping this metric group for this CSV.")
            continue

        detected[path.name] = joint_columns
        for joint, column in joint_columns:
            if metrics == "tau_like":
                rows.append(make_tau_like_row(dataset, joint, column))
            elif metrics == "prediction":
                rows.append(make_prediction_row(dataset, joint, column))
    return rows, detected


def make_clip_proxy_rows(
    datasets: list[dict[str, object]],
    y_hat_detected: dict[str, list[tuple[int, str]]],
    gp_scale: float,
    gp_clip_nm: float,
) -> list[dict[str, object]]:
    rows = []
    threshold = gp_clip_nm * 0.98
    for dataset in datasets:
        path = dataset["path"]
        joint_columns = y_hat_detected.get(path.name, [])
        for joint, column in joint_columns:
            values = finite_values(dataset["data"][column])
            scaled_abs = np.abs(gp_scale * values)
            clip_proxy = scaled_abs >= threshold
            rows.append({
                "mode_name": dataset["mode_name"],
                "csv_file": path.name,
                "joint": joint,
                "column": column,
                "scaled_y_hat_abs_max": float(np.max(scaled_abs)) if len(scaled_abs) else math.nan,
                "scaled_y_hat_abs_p95": float(np.percentile(scaled_abs, 95)) if len(scaled_abs) else math.nan,
                "clip_proxy_count": int(np.sum(clip_proxy)) if len(scaled_abs) else 0,
                "clip_proxy_ratio": float(np.mean(clip_proxy)) if len(scaled_abs) else math.nan,
            })
    return rows


def make_tau_residual_comparison(residual_metrics: list[dict[str, object]]) -> list[dict[str, object]]:
    by_mode_joint = {
        (row["mode_name"], int(row["joint"])): float(row["rms"])
        for row in residual_metrics
    }
    modes = {row["mode_name"] for row in residual_metrics}
    if not {"pure_no_gp", "gp_on_conservative"}.issubset(modes):
        print("WARNING: pure_no_gp and gp_on_conservative were not both detected; skipping residual comparison.")
        return []

    rows = []
    for joint in range(1, 8):
        no_gp_rms = by_mode_joint.get(("pure_no_gp", joint), math.nan)
        gp_on_rms = by_mode_joint.get(("gp_on_conservative", joint), math.nan)
        if not np.isfinite(no_gp_rms) or not np.isfinite(gp_on_rms):
            continue
        rows.append({
            "comparison": "gp_on_conservative_vs_pure_no_gp",
            "joint": joint,
            "rms_no_gp": no_gp_rms,
            "rms_gp_on": gp_on_rms,
            "rms_change_percent": 100.0 * (gp_on_rms - no_gp_rms) / no_gp_rms if no_gp_rms else math.nan,
        })
    return rows


def choose_x(dataset: dict[str, object]) -> tuple["np.ndarray", str]:
    time_col = detect_time_column(dataset["columns"])
    if time_col and time_col in dataset["data"]:
        return dataset["data"][time_col], time_col
    return np.arange(dataset["rows"]), "sample"


def save_cartesian_error_overlay(
    datasets: list[dict[str, object]],
    detected: dict[str, tuple[list[str], list[str]]],
    output_path: Path,
) -> bool:
    if not detected:
        return False

    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = False
    for dataset in datasets:
        path = dataset["path"]
        if path.name not in detected:
            continue
        actual, desired = detected[path.name]
        errors = [dataset["data"][actual_col] - dataset["data"][desired_col] for actual_col, desired_col in zip(actual, desired)]
        norms = np.linalg.norm(np.vstack(errors).T, axis=1)
        x, x_label = choose_x(dataset)
        ax.plot(x, norms, label=dataset["mode_name"], linewidth=0.9)
        ax.set_xlabel(x_label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_title("Cartesian error norm overlay")
    ax.set_ylabel("error norm")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True


def save_actual_vs_desired_plots(
    datasets: list[dict[str, object]],
    detected: dict[str, tuple[list[str], list[str]]],
    output_dir: Path,
) -> int:
    count = 0
    for dataset in datasets:
        path = dataset["path"]
        if path.name not in detected:
            continue
        actual, desired = detected[path.name]
        x, x_label = choose_x(dataset)
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for ax, axis, actual_col, desired_col in zip(axes, ("x", "y", "z"), actual, desired):
            ax.plot(x, dataset["data"][actual_col], label=actual_col, linewidth=0.9)
            ax.plot(x, dataset["data"][desired_col], label=desired_col, linewidth=0.9)
            ax.set_ylabel(axis)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize="small")
        axes[-1].set_xlabel(x_label)
        fig.suptitle(f"Cartesian actual vs desired: {dataset['mode_name']}")
        fig.tight_layout()
        output_path = output_dir / f"cartesian_actual_vs_desired_{path.stem}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=140)
        plt.close(fig)
        count += 1
    return count


def save_joint_bar_plot(
    rows: list[dict[str, object]],
    value_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> bool:
    if not rows:
        return False

    modes = sorted({str(row["mode_name"]) for row in rows})
    joints = sorted({int(row["joint"]) for row in rows})
    if not modes or not joints:
        return False

    values = {
        (str(row["mode_name"]), int(row["joint"])): float(row[value_column])
        for row in rows
        if value_column in row
    }
    x = np.arange(len(joints))
    width = 0.8 / max(len(modes), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for mode_index, mode in enumerate(modes):
        offsets = x - 0.4 + width / 2 + mode_index * width
        y = [values.get((mode, joint), math.nan) for joint in joints]
        ax.bar(offsets, y, width, label=mode)
    ax.set_title(title)
    ax.set_xlabel("joint")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(joint) for joint in joints])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {path}")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not import_dependencies(args.output_dir):
        return 2

    csv_files = find_csv_files(args.input_dir, args.include, args.exclude)
    if not csv_files:
        print(f"WARNING: no CSV files found in {args.input_dir}")
        return 1

    print("Detected CSV files:")
    for path in csv_files:
        print(f"  - {path}")

    datasets = load_datasets(csv_files)
    if not datasets:
        print("WARNING: no CSV files could be loaded.")
        return 1

    timing_summary = [make_timing_summary(dataset) for dataset in datasets]
    write_rows(args.output_dir / "stage2b_timing_summary.csv", timing_summary, TIMING_FIELDS)

    cartesian_summary, cartesian_detected = make_cartesian_tracking_summary(datasets)
    write_rows(args.output_dir / "stage2b_cartesian_tracking_summary.csv", cartesian_summary, CARTESIAN_FIELDS)

    residual_metrics, residual_detected = make_joint_metric_rows(
        datasets,
        prefixes=("tau_residual", "residual_tau"),
        group_label="tau residual",
        metrics="tau_like",
    )
    write_rows(args.output_dir / "stage2b_tau_residual_metrics.csv", residual_metrics, TAU_LIKE_FIELDS)

    residual_comparison = make_tau_residual_comparison(residual_metrics)
    write_rows(args.output_dir / "stage2b_tau_residual_comparison.csv", residual_comparison, RESIDUAL_COMPARISON_FIELDS)

    gp_prediction_stats, y_hat_detected = make_joint_metric_rows(
        datasets,
        prefixes=("y_hat", "y_hat_local", "gp_prediction"),
        group_label="GP prediction",
        metrics="prediction",
    )
    write_rows(args.output_dir / "stage2b_gp_prediction_stats.csv", gp_prediction_stats, GP_PREDICTION_FIELDS)

    clip_proxy = make_clip_proxy_rows(datasets, y_hat_detected, args.gp_scale, args.gp_clip_nm)
    if not clip_proxy:
        print("WARNING: no y_hat-like columns found; clip proxy summary is empty.")
    write_rows(args.output_dir / "stage2b_clip_proxy.csv", clip_proxy, CLIP_PROXY_FIELDS)

    tau_metrics, tau_detected = make_joint_metric_rows(
        datasets,
        prefixes=("tau", "tau_d", "commanded_tau"),
        group_label="torque",
        metrics="tau_like",
    )
    write_rows(args.output_dir / "stage2b_tau_metrics.csv", tau_metrics, TAU_LIKE_FIELDS)

    plot_dir = args.output_dir / "plots"
    plots_written = []
    if save_cartesian_error_overlay(
        datasets,
        cartesian_detected,
        plot_dir / "cartesian_error_norm_overlay.png",
    ):
        plots_written.append("cartesian_error_norm_overlay.png")
    actual_vs_desired_count = save_actual_vs_desired_plots(datasets, cartesian_detected, plot_dir)
    if actual_vs_desired_count:
        plots_written.append(f"{actual_vs_desired_count} cartesian_actual_vs_desired plots")
    if save_joint_bar_plot(
        residual_metrics,
        "rms",
        "Tau residual RMS by joint",
        "RMS",
        plot_dir / "tau_residual_rms_by_joint.png",
    ):
        plots_written.append("tau_residual_rms_by_joint.png")
    if save_joint_bar_plot(
        gp_prediction_stats,
        "std",
        "y_hat std by joint",
        "std",
        plot_dir / "y_hat_std_by_joint.png",
    ):
        plots_written.append("y_hat_std_by_joint.png")
    if save_joint_bar_plot(
        gp_prediction_stats,
        "range",
        "y_hat range by joint",
        "range",
        plot_dir / "y_hat_range_by_joint.png",
    ):
        plots_written.append("y_hat_range_by_joint.png")
    if save_joint_bar_plot(
        clip_proxy,
        "clip_proxy_ratio",
        "Clip proxy ratio by joint",
        "ratio",
        plot_dir / "clip_proxy_ratio_by_joint.png",
    ):
        plots_written.append("clip_proxy_ratio_by_joint.png")
    if save_joint_bar_plot(
        tau_metrics,
        "rms",
        "Tau RMS by joint",
        "RMS",
        plot_dir / "tau_rms_by_joint.png",
    ):
        plots_written.append("tau_rms_by_joint.png")

    print("\nMetric groups:")
    print(f"  timing: {len(timing_summary)} rows")
    print(f"  cartesian tracking: {len(cartesian_summary)} rows")
    print(f"  tau residual: {len(residual_metrics)} rows")
    print(f"  tau residual comparison: {len(residual_comparison)} rows")
    print(f"  GP prediction: {len(gp_prediction_stats)} rows")
    print(f"  clip proxy: {len(clip_proxy)} rows")
    print(f"  torque: {len(tau_metrics)} rows")
    print(f"  plots: {', '.join(plots_written) if plots_written else 'none'}")

    if not residual_detected:
        print("WARNING: tau residual metric group was unavailable for all CSVs.")
    if not y_hat_detected:
        print("WARNING: GP prediction metric group was unavailable for all CSVs.")
    if not tau_detected:
        print("WARNING: torque metric group was unavailable for all CSVs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
