#!/usr/bin/env python3
"""Analyze Stage 4 formal frozen-GP test results offline.

This script intentionally has no ROS2 dependency. It follows the Stage 3A
offline comparison style: CSV-only inputs, numpy metrics, matplotlib Agg plots,
and explicit caveats for one-run formal evidence.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_STRICT_CSV = Path(
    "data/stage4/test/strict_no_gp/usable_runs/strict_no_gp_zmod_3000pts_20260523_154902.csv"
)
DEFAULT_PLANAR_CSV = Path(
    "data/stage4/test/gp_planar_scale03/usable_runs/gp_planar_scale03_zmod_2999pts_20260523_161222.csv"
)
DEFAULT_SPATIAL_CSV = Path(
    "data/stage4/test/gp_spatial_scale03/usable_runs/gp_spatial_scale03_zmod_3000pts_20260523_163907.csv"
)
DEFAULT_OUT_DIR = Path("outputs/stage4_formal_analysis")
EPS = 1e-12
JOINTS = range(1, 8)
TRACKING_ACTUAL = ("x_actual", "y_actual", "z_actual")
TRACKING_DESIRED = ("x_desired", "y_desired", "z_desired")

np = None
plt = None


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
        print("Use the project .venv with these packages installed; this script does not install packages.", file=sys.stderr)
        return False

    np = numpy_module
    plt = pyplot_module
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Stage 4 formal strict-no-GP, planar-GP, and spatial-GP CSV runs offline.",
    )
    parser.add_argument("--strict-csv", type=Path, default=DEFAULT_STRICT_CSV, help=f"Default: {DEFAULT_STRICT_CSV}")
    parser.add_argument("--planar-csv", type=Path, default=DEFAULT_PLANAR_CSV, help=f"Default: {DEFAULT_PLANAR_CSV}")
    parser.add_argument("--spatial-csv", type=Path, default=DEFAULT_SPATIAL_CSV, help=f"Default: {DEFAULT_SPATIAL_CSV}")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help=f"Default: {DEFAULT_OUT_DIR}")
    parser.add_argument("--scale", type=float, default=0.3, help="GP compensation proxy scale. Default: 0.3")
    parser.add_argument("--clip-nm", type=float, default=0.5, help="GP compensation proxy clip in Nm. Default: 0.5")
    return parser.parse_args()


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


def normalized_name(name: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in name]
    return "_".join(part for part in "".join(chars).split("_") if part)


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
    finite = finite_values(values)
    if len(finite) == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.square(finite))))


def max_abs(values: "np.ndarray") -> float:
    finite = finite_values(values)
    if len(finite) == 0:
        return math.nan
    return float(np.max(np.abs(finite)))


def relative_change_percent(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or abs(reference) < EPS:
        return math.nan
    return float(100.0 * (value - reference) / reference)


def improvement_percent(value: float, baseline: float) -> float:
    if not np.isfinite(value) or not np.isfinite(baseline) or abs(baseline) < EPS:
        return math.nan
    return float(100.0 * (baseline - value) / baseline)


def required_columns() -> list[str]:
    columns = list(TRACKING_ACTUAL) + list(TRACKING_DESIRED)
    columns.extend(f"tau_residual_{joint}" for joint in JOINTS)
    columns.extend(f"y_hat_local_{joint}" for joint in JOINTS)
    columns.extend(f"tau_{joint}" for joint in JOINTS)
    return columns


def load_csv(path: Path, mode_name: str) -> dict[str, object]:
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
    print(f"Loaded {path.name}: mode={mode_name}, rows={rows}, columns={len(columns)}")
    return {
        "mode_name": mode_name,
        "path": path,
        "columns": columns,
        "data": arrays,
        "rows": rows,
    }


def summarize_time(dataset: dict[str, object]) -> dict[str, object]:
    columns = dataset["columns"]
    data = dataset["data"]
    time_col = detect_time_column(columns)
    result = {
        "time_col": time_col or "",
        "time_span_s": math.nan,
        "median_dt_s": math.nan,
        "mean_dt_s": math.nan,
        "estimated_hz": math.nan,
    }
    if not time_col:
        return result

    values = finite_values(data[time_col])
    if len(values) < 2:
        return result

    result["time_span_s"] = float(values[-1] - values[0])
    diffs = np.diff(values)
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(positive_diffs) == 0:
        return result

    result["median_dt_s"] = float(np.median(positive_diffs))
    result["mean_dt_s"] = float(np.mean(positive_diffs))
    if result["median_dt_s"] > 0:
        result["estimated_hz"] = float(1.0 / result["median_dt_s"])
    return result


def all_numeric_values(dataset: dict[str, object]) -> "np.ndarray":
    data = dataset["data"]
    if not data:
        return np.asarray([], dtype=float)
    return np.concatenate(list(data.values()))


def make_data_quality_rows(datasets: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    required = required_columns()
    for dataset in datasets:
        values = all_numeric_values(dataset)
        missing = [column for column in required if column not in dataset["columns"]]
        row = {
            "mode_name": dataset["mode_name"],
            "csv_file": dataset["path"].name,
            "csv_path": str(dataset["path"]),
            "rows": dataset["rows"],
            "columns": len(dataset["columns"]),
            "nan_count": int(np.isnan(values).sum()),
            "inf_count": int(np.isinf(values).sum()),
            "required_columns_present": not missing,
            "missing_required_columns": ",".join(missing),
        }
        row.update(summarize_time(dataset))
        rows.append(row)
    return rows


def tracking_arrays(dataset: dict[str, object]) -> tuple["np.ndarray", "np.ndarray"]:
    data = dataset["data"]
    errors = []
    for actual_col, desired_col in zip(TRACKING_ACTUAL, TRACKING_DESIRED):
        errors.append(data[actual_col] - data[desired_col])
    matrix = np.vstack(errors).T
    norms = np.linalg.norm(matrix, axis=1)
    return matrix, norms


def make_tracking_rows(datasets: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for dataset in datasets:
        error_matrix, norms = tracking_arrays(dataset)
        finite_rows = error_matrix[np.all(np.isfinite(error_matrix), axis=1)]
        finite_norms = norms[np.isfinite(norms)]
        row = {
            "mode_name": dataset["mode_name"],
            "csv_file": dataset["path"].name,
            "rows": dataset["rows"],
        }
        for index, axis in enumerate(("x", "y", "z")):
            axis_values = finite_rows[:, index] if len(finite_rows) else np.asarray([], dtype=float)
            row[f"rms_{axis}_m"] = rms(axis_values)
            row[f"rms_{axis}_mm"] = row[f"rms_{axis}_m"] * 1000.0
            row[f"max_abs_{axis}_m"] = max_abs(axis_values)
            row[f"max_abs_{axis}_mm"] = row[f"max_abs_{axis}_m"] * 1000.0

        row["mean_3d_error_m"] = float(np.mean(finite_norms)) if len(finite_norms) else math.nan
        row["mean_3d_error_mm"] = row["mean_3d_error_m"] * 1000.0
        row["rms_3d_error_m"] = rms(finite_norms)
        row["rms_3d_error_mm"] = row["rms_3d_error_m"] * 1000.0
        row["max_3d_error_m"] = float(np.max(finite_norms)) if len(finite_norms) else math.nan
        row["max_3d_error_mm"] = row["max_3d_error_m"] * 1000.0
        rows.append(row)

    by_mode = {row["mode_name"]: row for row in rows}
    strict = by_mode.get("strict_no_gp", {})
    baseline = float(strict.get("rms_3d_error_m", math.nan))
    planar = float(by_mode.get("gp_planar_scale03", {}).get("rms_3d_error_m", math.nan))
    spatial = float(by_mode.get("gp_spatial_scale03", {}).get("rms_3d_error_m", math.nan))
    for row in rows:
        current = float(row["rms_3d_error_m"])
        row["tracking_rms_improvement_vs_strict_percent"] = (
            0.0 if row["mode_name"] == "strict_no_gp" else improvement_percent(current, baseline)
        )
        row["tracking_rms_relative_diff_vs_planar_percent"] = (
            relative_change_percent(spatial, planar) if row["mode_name"] == "gp_spatial_scale03" else math.nan
        )
    return rows


def joint_columns(prefix: str) -> list[tuple[int, str]]:
    return [(joint, f"{prefix}_{joint}") for joint in JOINTS]


def make_joint_metric_rows(
    datasets: list[dict[str, object]],
    prefix: str,
    metric_group: str,
) -> list[dict[str, object]]:
    rows = []
    for dataset in datasets:
        all_values = []
        for joint, column in joint_columns(prefix):
            values = finite_values(dataset["data"][column])
            all_values.append(values)
            rows.append({
                "mode_name": dataset["mode_name"],
                "csv_file": dataset["path"].name,
                "metric_group": metric_group,
                "joint": joint,
                "column": column,
                "rms": rms(values),
                "max_abs": max_abs(values),
            })

        combined = np.concatenate(all_values) if all_values else np.asarray([], dtype=float)
        rows.append({
            "mode_name": dataset["mode_name"],
            "csv_file": dataset["path"].name,
            "metric_group": metric_group,
            "joint": "all",
            "column": f"{prefix}_1..7",
            "rms": rms(combined),
            "max_abs": max_abs(combined),
        })
    return rows


def make_compensation_proxy_rows(
    datasets: list[dict[str, object]],
    scale: float,
    clip_nm: float,
) -> list[dict[str, object]]:
    rows = []
    for dataset in datasets:
        all_proxy = []
        total_clip_hits = 0
        total_count = 0
        for joint, column in joint_columns("y_hat_local"):
            values = finite_values(dataset["data"][column])
            proxy = np.clip(scale * values, -clip_nm, clip_nm)
            clip_hits = np.abs(scale * values) >= clip_nm if len(values) else np.asarray([], dtype=bool)
            total_clip_hits += int(np.sum(clip_hits))
            total_count += int(len(values))
            all_proxy.append(proxy)
            rows.append({
                "mode_name": dataset["mode_name"],
                "csv_file": dataset["path"].name,
                "joint": joint,
                "source_column": column,
                "scale": scale,
                "clip_nm": clip_nm,
                "rms": rms(proxy),
                "max_abs": max_abs(proxy),
                "clip_hit_count": int(np.sum(clip_hits)),
                "clip_hit_ratio": float(np.mean(clip_hits)) if len(clip_hits) else math.nan,
            })

        combined = np.concatenate(all_proxy) if all_proxy else np.asarray([], dtype=float)
        rows.append({
            "mode_name": dataset["mode_name"],
            "csv_file": dataset["path"].name,
            "joint": "all",
            "source_column": "y_hat_local_1..7",
            "scale": scale,
            "clip_nm": clip_nm,
            "rms": rms(combined),
            "max_abs": max_abs(combined),
            "clip_hit_count": total_clip_hits,
            "clip_hit_ratio": float(total_clip_hits / total_count) if total_count else math.nan,
        })
    return rows


def format_float(value: object, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {path}")


def metric_lookup(rows: list[dict[str, object]], mode_name: str, joint: object = "all") -> dict[str, object]:
    for row in rows:
        if row["mode_name"] == mode_name and row.get("joint") == joint:
            return row
    return {}


def make_summary_rows(
    datasets: list[dict[str, object]],
    tracking_rows: list[dict[str, object]],
    tau_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    proxy_rows: list[dict[str, object]],
    quality_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summary = []
    quality_by_mode = {row["mode_name"]: row for row in quality_rows}
    tracking_by_mode = {row["mode_name"]: row for row in tracking_rows}
    for dataset in datasets:
        mode = dataset["mode_name"]
        tracking = tracking_by_mode[mode]
        tau = metric_lookup(tau_rows, mode)
        pred = metric_lookup(prediction_rows, mode)
        proxy = metric_lookup(proxy_rows, mode)
        quality = quality_by_mode[mode]
        summary.append({
            "mode_name": mode,
            "csv_file": dataset["path"].name,
            "rows": dataset["rows"],
            "nan_count": quality["nan_count"],
            "inf_count": quality["inf_count"],
            "tracking_rms_3d_m": tracking["rms_3d_error_m"],
            "tracking_rms_3d_mm": tracking["rms_3d_error_mm"],
            "tracking_mean_3d_m": tracking["mean_3d_error_m"],
            "tracking_max_3d_m": tracking["max_3d_error_m"],
            "tracking_rms_improvement_vs_strict_percent": tracking["tracking_rms_improvement_vs_strict_percent"],
            "tracking_rms_relative_diff_vs_planar_percent": tracking["tracking_rms_relative_diff_vs_planar_percent"],
            "tau_residual_all_rms": tau.get("rms", math.nan),
            "tau_residual_all_max_abs": tau.get("max_abs", math.nan),
            "y_hat_local_all_rms": pred.get("rms", math.nan),
            "y_hat_local_all_max_abs": pred.get("max_abs", math.nan),
            "comp_proxy_all_rms": proxy.get("rms", math.nan),
            "comp_proxy_all_max_abs": proxy.get("max_abs", math.nan),
            "comp_proxy_clip_hit_count": proxy.get("clip_hit_count", math.nan),
            "comp_proxy_clip_hit_ratio": proxy.get("clip_hit_ratio", math.nan),
        })
    return summary


def plot_tracking_3d_error(datasets: list[dict[str, object]], output_path: Path) -> None:
    min_rows = min(int(dataset["rows"]) for dataset in datasets)
    fig, ax = plt.subplots(figsize=(12, 6))
    for dataset in datasets:
        _errors, norms = tracking_arrays(dataset)
        ax.plot(np.arange(min_rows), norms[:min_rows] * 1000.0, label=dataset["mode_name"], linewidth=0.9)
    ax.set_title(f"Stage 4 3D tracking error norm (cropped to {min_rows} samples)")
    ax.set_xlabel("sample")
    ax.set_ylabel("3D error norm (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def save_grouped_bar(
    rows: list[dict[str, object]],
    value_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    filtered = [row for row in rows if row.get("joint") != "all"]
    modes = ["strict_no_gp", "gp_planar_scale03", "gp_spatial_scale03"]
    joints = list(JOINTS)
    values = {
        (row["mode_name"], int(row["joint"])): float(row[value_column])
        for row in filtered
        if value_column in row
    }
    x = np.arange(len(joints))
    width = 0.8 / len(modes)

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


def plot_tracking_axis_rms(tracking_rows: list[dict[str, object]], output_path: Path) -> None:
    modes = ["strict_no_gp", "gp_planar_scale03", "gp_spatial_scale03"]
    axes = ["x", "y", "z"]
    rows_by_mode = {row["mode_name"]: row for row in tracking_rows}
    x = np.arange(len(axes))
    width = 0.8 / len(modes)

    fig, ax = plt.subplots(figsize=(10, 6))
    for mode_index, mode in enumerate(modes):
        offsets = x - 0.4 + width / 2 + mode_index * width
        y = [float(rows_by_mode[mode][f"rms_{axis}_mm"]) for axis in axes]
        ax.bar(offsets, y, width, label=mode)
    ax.set_title("Stage 4 tracking axis RMS error")
    ax.set_xlabel("axis")
    ax.set_ylabel("RMS error (mm)")
    ax.set_xticks(x)
    ax.set_xticklabels(axes)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_tracking_3d_rms(tracking_rows: list[dict[str, object]], output_path: Path) -> None:
    modes = ["strict_no_gp", "gp_planar_scale03", "gp_spatial_scale03"]
    rows_by_mode = {row["mode_name"]: row for row in tracking_rows}
    values = [float(rows_by_mode[mode]["rms_3d_error_mm"]) for mode in modes]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(modes, values)
    ax.set_title("Stage 4 3D tracking RMS error")
    ax.set_xlabel("mode")
    ax.set_ylabel("3D RMS error (mm)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def choose_x(dataset: dict[str, object]) -> tuple["np.ndarray", str]:
    time_col = detect_time_column(dataset["columns"])
    if time_col and time_col in dataset["data"]:
        return dataset["data"][time_col], time_col
    return np.arange(dataset["rows"]), "sample"


def plot_actual_vs_desired(dataset: dict[str, object], output_dir: Path) -> None:
    x, x_label = choose_x(dataset)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, axis, actual_col, desired_col in zip(axes, ("x", "y", "z"), TRACKING_ACTUAL, TRACKING_DESIRED):
        ax.plot(x, dataset["data"][actual_col], label=actual_col, linewidth=0.9)
        ax.plot(x, dataset["data"][desired_col], label=desired_col, linewidth=0.9)
        ax.set_ylabel(f"{axis} (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize="small")
    axes[-1].set_xlabel(x_label)
    fig.suptitle(f"Stage 4 actual vs desired XYZ: {dataset['mode_name']}")
    fig.tight_layout()
    output_path = output_dir / f"actual_vs_desired_xyz_timeseries_{dataset['mode_name']}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def write_plots(
    datasets: list[dict[str, object]],
    tracking_rows: list[dict[str, object]],
    tau_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    proxy_rows: list[dict[str, object]],
    out_dir: Path,
) -> None:
    plot_tracking_3d_error(datasets, out_dir / "tracking_3d_error_timeseries.png")
    plot_tracking_axis_rms(tracking_rows, out_dir / "tracking_axis_error_rms_bar.png")
    plot_tracking_3d_rms(tracking_rows, out_dir / "tracking_3d_rms_bar.png")
    save_grouped_bar(tau_rows, "rms", "Stage 4 tau residual RMS per joint", "RMS (Nm)", out_dir / "tau_residual_rms_per_joint.png")
    save_grouped_bar(prediction_rows, "rms", "Stage 4 y_hat_local RMS per joint", "RMS (Nm)", out_dir / "y_hat_local_rms_per_joint.png")
    save_grouped_bar(proxy_rows, "rms", "Stage 4 compensation proxy RMS per joint", "RMS (Nm)", out_dir / "compensation_proxy_rms_per_joint.png")
    save_grouped_bar(
        proxy_rows,
        "clip_hit_ratio",
        "Stage 4 compensation proxy clip hit ratio per joint",
        "clip hit ratio",
        out_dir / "compensation_clip_ratio_per_joint.png",
    )
    for dataset in datasets:
        plot_actual_vs_desired(dataset, out_dir)


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_markdown_summary(
    path: Path,
    args: argparse.Namespace,
    datasets: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    tracking_rows: list[dict[str, object]],
    tau_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    proxy_rows: list[dict[str, object]],
    quality_rows: list[dict[str, object]],
) -> None:
    summary_by_mode = {row["mode_name"]: row for row in summary_rows}
    tracking_by_mode = {row["mode_name"]: row for row in tracking_rows}
    strict = tracking_by_mode["strict_no_gp"]["rms_3d_error_m"]
    planar = tracking_by_mode["gp_planar_scale03"]["rms_3d_error_m"]
    spatial = tracking_by_mode["gp_spatial_scale03"]["rms_3d_error_m"]
    planar_improvement = improvement_percent(planar, strict)
    spatial_improvement = improvement_percent(spatial, strict)
    spatial_vs_planar = relative_change_percent(spatial, planar)

    input_rows = [[dataset["mode_name"], dataset["rows"], str(dataset["path"])] for dataset in datasets]
    tracking_table = []
    for row in tracking_rows:
        tracking_table.append([
            row["mode_name"],
            format_float(row["rms_x_mm"], 3),
            format_float(row["rms_y_mm"], 3),
            format_float(row["rms_z_mm"], 3),
            format_float(row["rms_3d_error_mm"], 3),
            format_float(row["mean_3d_error_mm"], 3),
            format_float(row["max_3d_error_mm"], 3),
            format_float(row["tracking_rms_improvement_vs_strict_percent"], 2),
        ])

    residual_table = []
    prediction_table = []
    proxy_table = []
    for mode in ("strict_no_gp", "gp_planar_scale03", "gp_spatial_scale03"):
        tau = metric_lookup(tau_rows, mode)
        pred = metric_lookup(prediction_rows, mode)
        proxy = metric_lookup(proxy_rows, mode)
        residual_table.append([mode, format_float(tau.get("rms"), 6), format_float(tau.get("max_abs"), 6)])
        prediction_table.append([mode, format_float(pred.get("rms"), 6), format_float(pred.get("max_abs"), 6)])
        proxy_table.append([
            mode,
            format_float(proxy.get("rms"), 6),
            format_float(proxy.get("max_abs"), 6),
            proxy.get("clip_hit_count", "nan"),
            format_float(proxy.get("clip_hit_ratio"), 6),
        ])

    quality_table = [
        [
            row["mode_name"],
            row["rows"],
            row["columns"],
            row["nan_count"],
            row["inf_count"],
            row["required_columns_present"],
            format_float(row["time_span_s"], 3),
            format_float(row["estimated_hz"], 2),
        ]
        for row in quality_rows
    ]

    conclusion = (
        "In this single Stage 4 formal run set, the spatial-trained frozen GP has the lowest 3D tracking RMS "
        "among the three tested modes."
    )
    if spatial > planar:
        conclusion = (
            "In this single Stage 4 formal run set, the spatial-trained frozen GP does not improve 3D tracking RMS "
            "relative to the planar-trained frozen GP."
        )
    if spatial > strict:
        conclusion = (
            "In this single Stage 4 formal run set, the spatial-trained frozen GP has higher 3D tracking RMS than "
            "the strict no-GP baseline, so the result should be reported as worse tracking for this run."
        )

    content = f"""# Stage 4 Formal Offline Analysis

## Experiment setup

- Branch context: `frozen_gp_spatial_trajectory`
- Formal test trajectory: `trajectory_mode=z_modulated_circle`, `z_amplitude=0.03`, `z_frequency_multiplier=0.5`
- Comparison modes: `strict_no_gp`, `gp_planar_scale03`, `gp_spatial_scale03`
- Frozen GP test settings: `gp_online_update_enabled=false`, `gp_compensation_source=local`
- Compensation proxy settings used offline: `scale={args.scale}`, `clip={args.clip_nm} Nm`
- Row-count handling: metrics use each CSV's full length; overlay timeseries plots are cropped to the shortest run length.

## Input files

{md_table(["mode", "rows", "csv"], input_rows)}

## Data quality

{md_table(["mode", "rows", "columns", "nan", "inf", "required columns", "time span s", "estimated Hz"], quality_table)}

## Tracking comparison

{md_table(["mode", "RMS x mm", "RMS y mm", "RMS z mm", "3D RMS mm", "3D mean mm", "3D max mm", "improvement vs strict %"], tracking_table)}

- `gp_planar_scale03` tracking RMS improvement vs `strict_no_gp`: `{format_float(planar_improvement, 2)}%`
- `gp_spatial_scale03` tracking RMS improvement vs `strict_no_gp`: `{format_float(spatial_improvement, 2)}%`
- `gp_spatial_scale03` relative difference vs `gp_planar_scale03`: `{format_float(spatial_vs_planar, 2)}%` (`negative` means lower 3D RMS than planar)

## Tau residual comparison

{md_table(["mode", "all-joint RMS", "all-joint max abs"], residual_table)}

## GP prediction / compensation proxy comparison

{md_table(["mode", "y_hat_local all-joint RMS", "y_hat_local all-joint max abs"], prediction_table)}

{md_table(["mode", "comp proxy all-joint RMS", "comp proxy all-joint max abs", "clip hit count", "clip hit ratio"], proxy_table)}

## Caveats

- 三组 formal CSV 都有效，`nan=0` 且 `inf=0`。
- `gp_planar_scale03` 和 `gp_spatial_scale03` 都是 frozen local GP，`gp_online_update_enabled=false`。
- Offline compensation proxy 使用 `scale={args.scale}` 和 `clip={args.clip_nm} Nm`；这是根据 `y_hat_local_*` 重建的 proxy，不替代控制器内部完整安全判断。
- 数据来自一次 fullrun，不等同于 robust repeated validation。
- 真机日志中可能存在 post-save `communication_constraints_violation` / shutdown caveat，但不阻止本次离线分析。
- 本 summary 不声称 fully stable，也不声称 robust repeated validation。
- 三组 row count 不完全一致：`strict_no_gp=3000`，`gp_planar_scale03=2999`，`gp_spatial_scale03=3000`；数值指标按各自完整长度计算，timeseries overlay 裁剪到最短长度。

## Conclusion wording for thesis notes

{conclusion} The result should be described as evidence from one formal run set, not as a robust repeated validation claim. The formal comparison is still useful because all three runs used the same test trajectory and the two GP modes used the same conservative compensation scale and clip setting.
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path}")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not import_dependencies(args.out_dir):
        return 2

    datasets = [
        load_csv(args.strict_csv, "strict_no_gp"),
        load_csv(args.planar_csv, "gp_planar_scale03"),
        load_csv(args.spatial_csv, "gp_spatial_scale03"),
    ]

    missing_by_mode = {
        dataset["mode_name"]: [column for column in required_columns() if column not in dataset["columns"]]
        for dataset in datasets
    }
    missing = {mode: columns for mode, columns in missing_by_mode.items() if columns}
    if missing:
        for mode, columns in missing.items():
            print(f"ERROR: {mode} missing required columns: {', '.join(columns)}", file=sys.stderr)
        return 1

    quality_rows = make_data_quality_rows(datasets)
    tracking_rows = make_tracking_rows(datasets)
    tau_rows = make_joint_metric_rows(datasets, "tau_residual", "tau_residual")
    prediction_rows = make_joint_metric_rows(datasets, "y_hat_local", "y_hat_local")
    proxy_rows = make_compensation_proxy_rows(datasets, args.scale, args.clip_nm)
    summary_rows = make_summary_rows(datasets, tracking_rows, tau_rows, prediction_rows, proxy_rows, quality_rows)

    write_rows(args.out_dir / "stage4_formal_summary.csv", summary_rows, [
        "mode_name",
        "csv_file",
        "rows",
        "nan_count",
        "inf_count",
        "tracking_rms_3d_m",
        "tracking_rms_3d_mm",
        "tracking_mean_3d_m",
        "tracking_max_3d_m",
        "tracking_rms_improvement_vs_strict_percent",
        "tracking_rms_relative_diff_vs_planar_percent",
        "tau_residual_all_rms",
        "tau_residual_all_max_abs",
        "y_hat_local_all_rms",
        "y_hat_local_all_max_abs",
        "comp_proxy_all_rms",
        "comp_proxy_all_max_abs",
        "comp_proxy_clip_hit_count",
        "comp_proxy_clip_hit_ratio",
    ])
    write_rows(args.out_dir / "stage4_tracking_metrics.csv", tracking_rows, [
        "mode_name",
        "csv_file",
        "rows",
        "rms_x_m",
        "rms_x_mm",
        "rms_y_m",
        "rms_y_mm",
        "rms_z_m",
        "rms_z_mm",
        "mean_3d_error_m",
        "mean_3d_error_mm",
        "rms_3d_error_m",
        "rms_3d_error_mm",
        "max_3d_error_m",
        "max_3d_error_mm",
        "max_abs_x_m",
        "max_abs_x_mm",
        "max_abs_y_m",
        "max_abs_y_mm",
        "max_abs_z_m",
        "max_abs_z_mm",
        "tracking_rms_improvement_vs_strict_percent",
        "tracking_rms_relative_diff_vs_planar_percent",
    ])
    write_rows(args.out_dir / "stage4_tau_residual_metrics.csv", tau_rows, [
        "mode_name",
        "csv_file",
        "metric_group",
        "joint",
        "column",
        "rms",
        "max_abs",
    ])
    write_rows(args.out_dir / "stage4_gp_prediction_metrics.csv", prediction_rows, [
        "mode_name",
        "csv_file",
        "metric_group",
        "joint",
        "column",
        "rms",
        "max_abs",
    ])
    write_rows(args.out_dir / "stage4_compensation_proxy_metrics.csv", proxy_rows, [
        "mode_name",
        "csv_file",
        "joint",
        "source_column",
        "scale",
        "clip_nm",
        "rms",
        "max_abs",
        "clip_hit_count",
        "clip_hit_ratio",
    ])
    write_rows(args.out_dir / "stage4_data_quality.csv", quality_rows, [
        "mode_name",
        "csv_file",
        "csv_path",
        "rows",
        "columns",
        "nan_count",
        "inf_count",
        "required_columns_present",
        "missing_required_columns",
        "time_col",
        "time_span_s",
        "median_dt_s",
        "mean_dt_s",
        "estimated_hz",
    ])

    write_markdown_summary(
        args.out_dir / "stage4_formal_summary.md",
        args,
        datasets,
        summary_rows,
        tracking_rows,
        tau_rows,
        prediction_rows,
        proxy_rows,
        quality_rows,
    )
    write_plots(datasets, tracking_rows, tau_rows, prediction_rows, proxy_rows, args.out_dir)

    print("\nStage 4 formal analysis complete.")
    print(f"Output directory: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
