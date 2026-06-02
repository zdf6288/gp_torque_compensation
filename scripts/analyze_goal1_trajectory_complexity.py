#!/usr/bin/env python3
"""Compare GOAL1 real trajectory variants and a MuJoCo EE path.

This is an offline CSV analysis script. It has no ROS2 dependency and does not
touch robot/controller code.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path


np = None
pd = None
plt = None


REAL_ACTUAL_COLUMNS = ("x_actual", "y_actual", "z_actual")
REAL_DESIRED_COLUMNS = ("x_desired", "y_desired", "z_desired")
MUJOCO_EE_COLUMNS = ("ee_x", "ee_y", "ee_z")
JOINT_COLUMNS = tuple(f"joint_pos_{idx}" for idx in range(1, 8))
TAU_COLUMNS = tuple(f"tau_{idx}" for idx in range(1, 8))
TAU_RESIDUAL_COLUMNS = tuple(f"tau_residual_{idx}" for idx in range(1, 8))


def import_dependencies(out_dir: Path) -> bool:
    global np, pd, plt

    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = out_dir / ".matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    missing = []
    try:
        import numpy as numpy_module
    except ModuleNotFoundError:
        missing.append("numpy")
    try:
        import pandas as pandas_module
    except ModuleNotFoundError:
        missing.append("pandas")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot_module
    except ModuleNotFoundError:
        missing.append("matplotlib")

    if missing:
        print("Missing Python dependencies: " + ", ".join(sorted(set(missing))), file=sys.stderr)
        return False

    np = numpy_module
    pd = pandas_module
    plt = pyplot_module
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline GOAL1 EE trajectory complexity comparison.",
    )
    parser.add_argument("--planar-real-csv", required=True, type=Path)
    parser.add_argument("--complex-real-csv", required=True, type=Path)
    parser.add_argument("--multisine-real-csv", type=Path)
    parser.add_argument("--mujoco-ee-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def require_columns(df: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def numeric_array(df: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def valid_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 3)
    return points[np.isfinite(points).all(axis=1)]


def time_array(df: pd.DataFrame, time_column: str) -> np.ndarray:
    values = pd.to_numeric(df[time_column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def summarize_time(times: np.ndarray, sample_count: int) -> dict[str, float]:
    if len(times) >= 2:
        dt = np.diff(times)
        dt = dt[np.isfinite(dt)]
        positive_dt = dt[dt > 0.0]
        duration = float(times[-1] - times[0])
        dt_mean = float(np.mean(positive_dt)) if len(positive_dt) else float("nan")
        dt_std = float(np.std(positive_dt)) if len(positive_dt) else float("nan")
    else:
        duration = float("nan")
        dt_mean = float("nan")
        dt_std = float("nan")

    if not np.isfinite(duration) and sample_count > 1 and np.isfinite(dt_mean):
        duration = float((sample_count - 1) * dt_mean)

    return {
        "duration_sec": duration,
        "dt_mean": dt_mean,
        "dt_std": dt_std,
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def angle_change_metrics(points: np.ndarray) -> tuple[float, float]:
    if len(points) < 3:
        return float("nan"), float("nan")

    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 1e-12
    segments = segments[valid]
    lengths = lengths[valid]
    if len(segments) < 2:
        return float("nan"), float("nan")

    unit_segments = segments / lengths[:, None]
    dots = np.sum(unit_segments[:-1] * unit_segments[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    if len(angles) == 0:
        return float("nan"), float("nan")

    path_length = float(np.sum(lengths))
    curvature_proxy = float(np.sum(np.abs(angles)) / path_length) if path_length > 1e-12 else float("nan")
    velocity_direction_change_proxy = float(np.mean(np.abs(angles)))
    return curvature_proxy, velocity_direction_change_proxy


def trajectory_metrics(
    label: str,
    df: pd.DataFrame,
    time_column: str,
    point_columns: tuple[str, ...],
    point_kind: str,
) -> dict[str, float | str]:
    require_columns(df, point_columns, label)
    raw_points = numeric_array(df, point_columns)
    points = valid_points(raw_points)
    times = time_array(df, time_column)

    sample_count = int(len(points))
    metrics: dict[str, float | str] = {
        "dataset": label,
        "point_kind": point_kind,
        "sample_count": sample_count,
    }
    metrics.update(summarize_time(times, sample_count))

    if sample_count == 0:
        for key in (
            "x_range",
            "y_range",
            "z_range",
            "xy_range",
            "z_range_over_xy_range",
            "path_length",
            "straight_line_distance",
            "path_length_over_displacement",
            "mean_step_length",
            "curvature_proxy",
            "velocity_direction_change_proxy",
        ):
            metrics[key] = float("nan")
        return metrics

    ranges = np.nanmax(points, axis=0) - np.nanmin(points, axis=0)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1) if sample_count >= 2 else np.array([])
    path_length = float(np.sum(steps)) if len(steps) else 0.0
    displacement = float(np.linalg.norm(points[-1] - points[0])) if sample_count >= 2 else 0.0
    curvature_proxy, velocity_direction_change_proxy = angle_change_metrics(points)

    x_range = float(ranges[0])
    y_range = float(ranges[1])
    z_range = float(ranges[2])
    xy_range = float(math.hypot(x_range, y_range))

    metrics.update(
        {
            "x_range": x_range,
            "y_range": y_range,
            "z_range": z_range,
            "xy_range": xy_range,
            "z_range_over_xy_range": safe_ratio(z_range, xy_range),
            "path_length": path_length,
            "straight_line_distance": displacement,
            "path_length_over_displacement": safe_ratio(path_length, displacement),
            "mean_step_length": float(np.mean(steps)) if len(steps) else 0.0,
            "curvature_proxy": curvature_proxy,
            "velocity_direction_change_proxy": velocity_direction_change_proxy,
        }
    )
    return metrics


def add_tracking_metrics(metrics: dict[str, float | str], df: pd.DataFrame, label: str) -> None:
    require_columns(df, REAL_ACTUAL_COLUMNS, label)
    require_columns(df, REAL_DESIRED_COLUMNS, label)
    actual = numeric_array(df, REAL_ACTUAL_COLUMNS)
    desired = numeric_array(df, REAL_DESIRED_COLUMNS)
    valid = np.isfinite(actual).all(axis=1) & np.isfinite(desired).all(axis=1)
    error_m = actual[valid] - desired[valid]

    if len(error_m) == 0:
        values = {
            "tracking_rmse_x_mm": float("nan"),
            "tracking_rmse_y_mm": float("nan"),
            "tracking_rmse_z_mm": float("nan"),
            "tracking_rmse_3d_mm": float("nan"),
            "tracking_p95_3d_mm": float("nan"),
            "tracking_max_3d_mm": float("nan"),
            "tracking_p95_mm": float("nan"),
            "tracking_max_mm": float("nan"),
        }
    else:
        error_mm = error_m * 1000.0
        norm_mm = np.linalg.norm(error_mm, axis=1)
        values = {
            "tracking_rmse_x_mm": float(np.sqrt(np.mean(error_mm[:, 0] ** 2))),
            "tracking_rmse_y_mm": float(np.sqrt(np.mean(error_mm[:, 1] ** 2))),
            "tracking_rmse_z_mm": float(np.sqrt(np.mean(error_mm[:, 2] ** 2))),
            "tracking_rmse_3d_mm": float(np.sqrt(np.mean(norm_mm**2))),
            "tracking_p95_3d_mm": float(np.percentile(norm_mm, 95)),
            "tracking_max_3d_mm": float(np.max(norm_mm)),
            "tracking_p95_mm": float(np.percentile(norm_mm, 95)),
            "tracking_max_mm": float(np.max(norm_mm)),
        }
    metrics.update(values)


def add_range_metrics(
    metrics: dict[str, float | str],
    df: pd.DataFrame,
    columns: tuple[str, ...],
    output_prefix: str,
    convert_to_deg: bool = False,
) -> None:
    for column in columns:
        key = f"{output_prefix}_{column.split('_')[-1]}"
        if column not in df.columns:
            metrics[key] = float("nan")
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            metrics[key] = float("nan")
            continue
        value_range = float(np.max(values) - np.min(values))
        if convert_to_deg:
            value_range = float(np.degrees(value_range))
        metrics[key] = value_range


def real_actual_metrics(label: str, df: pd.DataFrame) -> dict[str, float | str]:
    metrics = trajectory_metrics(label, df, "Time(s)", REAL_ACTUAL_COLUMNS, "actual")
    add_tracking_metrics(metrics, df, label)
    add_range_metrics(metrics, df, JOINT_COLUMNS, "joint_range_deg_q", convert_to_deg=True)
    add_range_metrics(metrics, df, TAU_COLUMNS, "tau_range_nm_tau")
    add_range_metrics(metrics, df, TAU_RESIDUAL_COLUMNS, "tau_residual_range_nm_tau")
    return metrics


def build_metrics(real_datasets: list[tuple[str, pd.DataFrame]], mujoco_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for label, df in real_datasets:
        rows.append(real_actual_metrics(label, df))

    mujoco = trajectory_metrics("mujoco_old_spatial_rich_ee", mujoco_df, "time", MUJOCO_EE_COLUMNS, "ee_path")
    add_range_metrics(mujoco, mujoco_df, JOINT_COLUMNS, "joint_range_deg_q", convert_to_deg=True)
    rows.append(mujoco)

    return pd.DataFrame(rows)


def get_points(df: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return valid_points(numeric_array(df, columns))


def get_time_for_plot(df: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(values).any():
        return values
    return np.arange(len(df), dtype=float)


def set_axes_equal_3d(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    centers = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    if not np.isfinite(radius) or radius <= 0:
        return
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def save_trajectory_3d_overlay(
    real_datasets: list[tuple[str, pd.DataFrame]],
    mujoco_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    styles = {
        "planar_real_actual": ("planar real actual", "#1f77b4", "-"),
        "spatial_rich_real_actual": ("old spatial-rich real actual", "#2ca02c", "-"),
        "spatial_multisine_real_actual": ("new spatial-multisine real actual", "#ff7f0e", "-"),
    }
    series = []
    for label, df in real_datasets:
        display, color, linestyle = styles.get(label, (label, None, "-"))
        series.append((display, get_points(df, REAL_ACTUAL_COLUMNS), color, linestyle))
    series.append(("MuJoCo old spatial-rich EE", get_points(mujoco_df, MUJOCO_EE_COLUMNS), "#d62728", ":"))
    for label, points, color, linestyle in series:
        if len(points):
            ax.plot(points[:, 0], points[:, 1], points[:, 2], label=label, color=color, linestyle=linestyle, linewidth=1.5)

    ax.set_title("GOAL1 EE trajectory 3D overlay")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="best")
    set_axes_equal_3d(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_xyz(axs, times: np.ndarray, points: np.ndarray, label: str, linestyle: str) -> None:
    if len(points) == 0:
        return
    count = min(len(times), len(points))
    if count == 0:
        times = np.arange(len(points), dtype=float)
        count = len(points)
    axis_labels = ("x [m]", "y [m]", "z [m]")
    for idx, ax in enumerate(axs):
        ax.plot(times[:count], points[:count, idx], label=label, linestyle=linestyle, linewidth=1.2)
        ax.set_ylabel(axis_labels[idx])
        ax.grid(True, alpha=0.3)


def save_xyz_over_time_overlay(
    real_datasets: list[tuple[str, pd.DataFrame]],
    mujoco_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=False)
    display_names = {
        "planar_real_actual": "planar real actual",
        "spatial_rich_real_actual": "old spatial-rich real actual",
        "spatial_multisine_real_actual": "new spatial-multisine real actual",
    }
    for label, df in real_datasets:
        plot_xyz(
            axs,
            get_time_for_plot(df, "Time(s)"),
            get_points(df, REAL_ACTUAL_COLUMNS),
            display_names.get(label, label),
            "-",
        )
    plot_xyz(
        axs,
        get_time_for_plot(mujoco_df, "time"),
        get_points(mujoco_df, MUJOCO_EE_COLUMNS),
        "MuJoCo old spatial-rich EE",
        ":",
    )
    axs[-1].set_xlabel("time [s]")
    axs[0].set_title("GOAL1 xyz over time overlay")
    axs[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_tracking_error_comparison_plot(real_datasets: list[tuple[str, pd.DataFrame]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    display_names = {
        "spatial_rich_real_actual": "old spatial-rich real 3D error",
        "spatial_multisine_real_actual": "new spatial-multisine real 3D error",
    }
    for label, df in real_datasets:
        if label not in display_names:
            continue
        time = get_time_for_plot(df, "Time(s)")
        actual = numeric_array(df, REAL_ACTUAL_COLUMNS)
        desired = numeric_array(df, REAL_DESIRED_COLUMNS)
        valid = np.isfinite(actual).all(axis=1) & np.isfinite(desired).all(axis=1)
        error_mm = (actual[valid] - desired[valid]) * 1000.0
        time = time[valid] if len(time) == len(valid) else np.arange(len(error_mm), dtype=float)
        norm_mm = np.linalg.norm(error_mm, axis=1) if len(error_mm) else np.array([])
        if len(norm_mm):
            ax.plot(time, norm_mm, label=display_names[label], linewidth=1.2)
    ax.set_title("Real trajectory tracking error comparison")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("3D error norm [mm]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def values_by_dataset(metrics_df: pd.DataFrame, prefix: str) -> tuple[list[str], dict[str, list[float]]]:
    columns = [f"{prefix}_{idx}" for idx in range(1, 8)]
    labels = [f"q{idx}" if prefix.startswith("joint") else f"j{idx}" for idx in range(1, 8)]
    values = {}
    for _, row in metrics_df.iterrows():
        values[str(row["dataset"])] = [float(row.get(column, np.nan)) for column in columns]
    return labels, values


def save_grouped_bar_plot(
    labels: list[str],
    values: dict[str, list[float]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels), dtype=float)
    datasets = list(values.keys())
    width = min(0.18, 0.8 / max(len(datasets), 1))

    for idx, dataset in enumerate(datasets):
        offset = (idx - (len(datasets) - 1) / 2.0) * width
        ax.bar(x + offset, values[dataset], width=width, label=dataset)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_joint_range_plot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    labels, values = values_by_dataset(metrics_df, "joint_range_deg_q")
    save_grouped_bar_plot(labels, values, "Joint position range comparison", "range [deg]", output_path)


def save_tau_range_plot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    labels, values = values_by_dataset(metrics_df[metrics_df["dataset"].str.contains("real")], "tau_range_nm_tau")
    save_grouped_bar_plot(labels, values, "Commanded tau range comparison", "range [Nm]", output_path)


def fmt(value: float | str, digits: int = 4) -> str:
    if isinstance(value, str):
        return value
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value_float):
        return "nan"
    return f"{value_float:.{digits}g}"


def row_by_dataset(metrics_df: pd.DataFrame, dataset: str) -> pd.Series:
    matches = metrics_df[metrics_df["dataset"] == dataset]
    if matches.empty:
        raise ValueError(f"missing metrics row: {dataset}")
    return matches.iloc[0]


def relative_delta(new_value: float, old_value: float) -> float:
    return safe_ratio(new_value - old_value, old_value)


def make_complexity_statement(planar: pd.Series, complex_actual: pd.Series) -> str:
    comparisons = {
        "z_range": safe_ratio(float(complex_actual["z_range"]), float(planar["z_range"])),
        "path_length": safe_ratio(float(complex_actual["path_length"]), float(planar["path_length"])),
        "curvature_proxy": safe_ratio(float(complex_actual["curvature_proxy"]), float(planar["curvature_proxy"])),
    }
    wins = sum(1 for value in comparisons.values() if np.isfinite(value) and value > 1.2)
    if wins >= 2:
        return "spatial-rich real is clearly more complex than planar real for this usable-data comparison."
    if wins == 1:
        return "spatial-rich real shows partial complexity increase over planar real, but the evidence is mixed."
    return "spatial-rich real is not clearly more complex than planar real by these EE metrics."


def make_mujoco_statement(complex_actual: pd.Series, mujoco: pd.Series) -> str:
    ratios = {
        "xy_range": safe_ratio(float(complex_actual["xy_range"]), float(mujoco["xy_range"])),
        "z_range": safe_ratio(float(complex_actual["z_range"]), float(mujoco["z_range"])),
        "path_length": safe_ratio(float(complex_actual["path_length"]), float(mujoco["path_length"])),
        "curvature_proxy": safe_ratio(float(complex_actual["curvature_proxy"]), float(mujoco["curvature_proxy"])),
    }
    close_count = sum(1 for value in ratios.values() if np.isfinite(value) and 0.5 <= value <= 1.5)
    if close_count >= 3:
        return "spatial-rich real is broadly close to the old MuJoCo EE path in coverage/path/curvature scale."
    if close_count >= 2:
        return "spatial-rich real is partly close to the old MuJoCo EE path, with some metric-scale gaps."
    return "spatial-rich real is not close to the old MuJoCo EE path by these metric scales."


def make_multisine_improvement_statement(old_actual: pd.Series, multisine_actual: pd.Series) -> str:
    ratios = {
        "xy_range": safe_ratio(float(multisine_actual["xy_range"]), float(old_actual["xy_range"])),
        "z_range": safe_ratio(float(multisine_actual["z_range"]), float(old_actual["z_range"])),
        "path_length": safe_ratio(float(multisine_actual["path_length"]), float(old_actual["path_length"])),
        "curvature_proxy": safe_ratio(float(multisine_actual["curvature_proxy"]), float(old_actual["curvature_proxy"])),
    }
    coverage_wins = sum(1 for key in ("xy_range", "z_range", "path_length") if np.isfinite(ratios[key]) and ratios[key] > 1.1)
    if coverage_wins >= 2:
        return (
            "spatial-multisine real improves absolute EE coverage over old spatial-rich real "
            f"(xy {ratios['xy_range']:.3g}x, z {ratios['z_range']:.3g}x, path {ratios['path_length']:.3g}x)."
        )
    if coverage_wins == 1:
        return (
            "spatial-multisine real improves one main coverage metric over old spatial-rich real, "
            f"but the coverage gain is mixed (xy {ratios['xy_range']:.3g}x, z {ratios['z_range']:.3g}x, path {ratios['path_length']:.3g}x)."
        )
    return (
        "spatial-multisine real does not clearly improve absolute EE coverage over old spatial-rich real "
        f"(xy {ratios['xy_range']:.3g}x, z {ratios['z_range']:.3g}x, path {ratios['path_length']:.3g}x)."
    )


def make_multisine_mujoco_statement(old_actual: pd.Series, multisine_actual: pd.Series, mujoco: pd.Series) -> str:
    metric_names = ("xy_range", "z_range", "path_length", "curvature_proxy")
    old_errors = {
        metric: abs(math.log(safe_ratio(float(old_actual[metric]), float(mujoco[metric]))))
        for metric in metric_names
        if np.isfinite(safe_ratio(float(old_actual[metric]), float(mujoco[metric])))
    }
    multisine_errors = {
        metric: abs(math.log(safe_ratio(float(multisine_actual[metric]), float(mujoco[metric]))))
        for metric in metric_names
        if np.isfinite(safe_ratio(float(multisine_actual[metric]), float(mujoco[metric])))
    }
    improved = [
        metric
        for metric in metric_names
        if metric in old_errors and metric in multisine_errors and multisine_errors[metric] < old_errors[metric]
    ]
    ratio_text = ", ".join(
        f"{metric} {safe_ratio(float(multisine_actual[metric]), float(mujoco[metric])):.3g}x"
        for metric in metric_names
    )
    if len(improved) >= 3:
        return f"spatial-multisine real is closer to MuJoCo than old spatial-rich real on {len(improved)}/4 metrics; ratios to MuJoCo: {ratio_text}."
    if len(improved) >= 2:
        return f"spatial-multisine real is partly closer to MuJoCo on {len(improved)}/4 metrics; ratios to MuJoCo: {ratio_text}."
    return f"spatial-multisine real is not materially closer to MuJoCo than old spatial-rich real; ratios to MuJoCo: {ratio_text}."


def make_tracking_statement(complex_actual: pd.Series) -> str:
    rmse = float(complex_actual["tracking_rmse_3d_mm"])
    p95 = float(complex_actual["tracking_p95_3d_mm"])
    max_error = float(complex_actual["tracking_max_3d_mm"])
    if rmse <= 15.0 and p95 <= 30.0:
        return f"tracking looks acceptable for offline usable-data comparison: 3D RMSE {rmse:.2f} mm, p95 {p95:.2f} mm, max {max_error:.2f} mm."
    if rmse <= 30.0 and p95 <= 60.0:
        return f"tracking is usable but should be treated cautiously: 3D RMSE {rmse:.2f} mm, p95 {p95:.2f} mm, max {max_error:.2f} mm."
    return f"tracking looks weak for clean comparison: 3D RMSE {rmse:.2f} mm, p95 {p95:.2f} mm, max {max_error:.2f} mm."


def make_q7_statement(metrics_df: pd.DataFrame) -> str:
    q7_values = []
    for _, row in metrics_df.iterrows():
        value = float(row.get("joint_range_deg_q_7", np.nan))
        if np.isfinite(value):
            q7_values.append((str(row["dataset"]), value))
    if not q7_values:
        return "q7 range is unavailable; this is a caveat, not a blocker for EE trajectory complexity comparison."
    min_label, min_value = min(q7_values, key=lambda item: item[1])
    return (
        f"q7 minimum range is {min_value:.3f} deg in {min_label}; q7 being static or weakly excited is a caveat, "
        "not a blocker, because this analysis focuses on end-effector trajectory complexity."
    )


def write_markdown_summary(metrics_df: pd.DataFrame, output_path: Path) -> None:
    planar = row_by_dataset(metrics_df, "planar_real_actual")
    complex_actual = row_by_dataset(metrics_df, "spatial_rich_real_actual")
    multisine_actual = None
    if "spatial_multisine_real_actual" in set(metrics_df["dataset"]):
        multisine_actual = row_by_dataset(metrics_df, "spatial_multisine_real_actual")
    mujoco = row_by_dataset(metrics_df, "mujoco_old_spatial_rich_ee")

    lines = [
        "# GOAL1 Trajectory Complexity Comparison",
        "",
        "This is an offline usable-data comparison of end-effector trajectory complexity. It is not robust repeated validation.",
        "",
        "## Key Metrics",
        "",
        "| dataset | samples | duration_s | x_range_m | y_range_m | z_range_m | xy_range_m | z/xy | path_length_m | path/disp | curvature_proxy | dir_change_rad | tracking_rmse_3d_mm | tracking_p95_3d_mm | tracking_max_3d_mm | q7_range_deg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    fmt(row["sample_count"], 0),
                    fmt(row["duration_sec"], 4),
                    fmt(row["x_range"], 4),
                    fmt(row["y_range"], 4),
                    fmt(row["z_range"], 4),
                    fmt(row["xy_range"], 4),
                    fmt(row["z_range_over_xy_range"], 4),
                    fmt(row["path_length"], 4),
                    fmt(row["path_length_over_displacement"], 4),
                    fmt(row["curvature_proxy"], 4),
                    fmt(row["velocity_direction_change_proxy"], 4),
                    fmt(row.get("tracking_rmse_3d_mm", np.nan), 4),
                    fmt(row.get("tracking_p95_3d_mm", np.nan), 4),
                    fmt(row.get("tracking_max_3d_mm", np.nan), 4),
                    fmt(row.get("joint_range_deg_q_7", np.nan), 4),
                ]
            )
            + " |"
        )

    complexity_statement = make_complexity_statement(planar, complex_actual)
    mujoco_statement = make_mujoco_statement(complex_actual, mujoco)
    old_tracking_statement = make_tracking_statement(complex_actual)
    multisine_tracking_statement = make_tracking_statement(multisine_actual) if multisine_actual is not None else None
    q7_statement = make_q7_statement(metrics_df)

    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            f"- Planar vs spatial-rich: {complexity_statement}",
            f"- Spatial-rich real vs old MuJoCo EE path: {mujoco_statement}",
            f"- Tracking old spatial-rich: {old_tracking_statement}",
            f"- q7 caveat: {q7_statement}",
        ]
    )
    if multisine_actual is not None:
        lines.extend(
            [
                f"- Spatial-multisine vs old spatial-rich: {make_multisine_improvement_statement(complex_actual, multisine_actual)}",
                f"- Spatial-multisine vs MuJoCo: {make_multisine_mujoco_statement(complex_actual, multisine_actual, mujoco)}",
                f"- Tracking spatial-multisine: {multisine_tracking_statement}",
            ]
        )

    lines.extend(
        [
            "",
            "## Ratio Notes",
            "",
            (
                "- spatial-rich actual / planar actual: "
                f"z_range {safe_ratio(float(complex_actual['z_range']), float(planar['z_range'])):.3g}, "
                f"path_length {safe_ratio(float(complex_actual['path_length']), float(planar['path_length'])):.3g}, "
                f"curvature_proxy {safe_ratio(float(complex_actual['curvature_proxy']), float(planar['curvature_proxy'])):.3g}."
            ),
            (
                "- spatial-rich actual / MuJoCo EE: "
                f"xy_range {safe_ratio(float(complex_actual['xy_range']), float(mujoco['xy_range'])):.3g}, "
                f"z_range {safe_ratio(float(complex_actual['z_range']), float(mujoco['z_range'])):.3g}, "
                f"path_length {safe_ratio(float(complex_actual['path_length']), float(mujoco['path_length'])):.3g}, "
                f"curvature_proxy {safe_ratio(float(complex_actual['curvature_proxy']), float(mujoco['curvature_proxy'])):.3g}."
            ),
        ]
    )
    if multisine_actual is not None:
        lines.extend(
            [
                (
                    "- spatial-multisine actual / old spatial-rich actual: "
                    f"xy_range {safe_ratio(float(multisine_actual['xy_range']), float(complex_actual['xy_range'])):.3g}, "
                    f"z_range {safe_ratio(float(multisine_actual['z_range']), float(complex_actual['z_range'])):.3g}, "
                    f"path_length {safe_ratio(float(multisine_actual['path_length']), float(complex_actual['path_length'])):.3g}, "
                    f"curvature_proxy {safe_ratio(float(multisine_actual['curvature_proxy']), float(complex_actual['curvature_proxy'])):.3g}."
                ),
                (
                    "- spatial-multisine actual / MuJoCo EE: "
                    f"xy_range {safe_ratio(float(multisine_actual['xy_range']), float(mujoco['xy_range'])):.3g}, "
                    f"z_range {safe_ratio(float(multisine_actual['z_range']), float(mujoco['z_range'])):.3g}, "
                    f"path_length {safe_ratio(float(multisine_actual['path_length']), float(mujoco['path_length'])):.3g}, "
                    f"curvature_proxy {safe_ratio(float(multisine_actual['curvature_proxy']), float(mujoco['curvature_proxy'])):.3g}."
                ),
            ]
        )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(metrics_df: pd.DataFrame, out_dir: Path, real_datasets: list[tuple[str, pd.DataFrame]], mujoco_df: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_dir / "trajectory_metrics_summary.csv", index=False)
    write_markdown_summary(metrics_df, out_dir / "trajectory_metrics_summary.md")
    save_trajectory_3d_overlay(real_datasets, mujoco_df, out_dir / "trajectory_3d_overlay.png")
    save_xyz_over_time_overlay(real_datasets, mujoco_df, out_dir / "xyz_over_time_overlay.png")
    save_tracking_error_comparison_plot(real_datasets, out_dir / "tracking_error_comparison_real.png")
    save_joint_range_plot(metrics_df, out_dir / "joint_range_comparison.png")

    tau_columns = [column for column in TAU_COLUMNS if any(column in df.columns for _, df in real_datasets)]
    if tau_columns:
        save_tau_range_plot(metrics_df, out_dir / "tau_range_comparison.png")


def main() -> int:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    if not import_dependencies(out_dir):
        return 2

    planar_path = resolve_path(args.planar_real_csv)
    complex_path = resolve_path(args.complex_real_csv)
    multisine_path = resolve_path(args.multisine_real_csv) if args.multisine_real_csv else None
    mujoco_path = resolve_path(args.mujoco_ee_csv)

    input_paths = [planar_path, complex_path, mujoco_path]
    if multisine_path is not None:
        input_paths.append(multisine_path)
    for path in input_paths:
        if not path.exists():
            print(f"Input CSV not found: {path}", file=sys.stderr)
            return 1

    planar_df = pd.read_csv(planar_path)
    complex_df = pd.read_csv(complex_path)
    multisine_df = pd.read_csv(multisine_path) if multisine_path is not None else None
    mujoco_df = pd.read_csv(mujoco_path)

    require_columns(planar_df, REAL_ACTUAL_COLUMNS + REAL_DESIRED_COLUMNS, "planar_real")
    require_columns(complex_df, REAL_ACTUAL_COLUMNS + REAL_DESIRED_COLUMNS, "spatial_rich_real")
    if multisine_df is not None:
        require_columns(multisine_df, REAL_ACTUAL_COLUMNS + REAL_DESIRED_COLUMNS, "spatial_multisine_real")
    require_columns(mujoco_df, MUJOCO_EE_COLUMNS, "mujoco_ee")
    if "Time(s)" not in planar_df.columns:
        raise ValueError("planar_real missing required column: Time(s)")
    if "Time(s)" not in complex_df.columns:
        raise ValueError("spatial_rich_real missing required column: Time(s)")
    if multisine_df is not None and "Time(s)" not in multisine_df.columns:
        raise ValueError("spatial_multisine_real missing required column: Time(s)")
    if "time" not in mujoco_df.columns:
        raise ValueError("mujoco_ee missing required column: time")

    real_datasets = [
        ("planar_real_actual", planar_df),
        ("spatial_rich_real_actual", complex_df),
    ]
    if multisine_df is not None:
        real_datasets.append(("spatial_multisine_real_actual", multisine_df))

    metrics_df = build_metrics(real_datasets, mujoco_df)
    write_outputs(metrics_df, out_dir, real_datasets, mujoco_df)

    print(f"Wrote outputs to: {out_dir}")
    print(metrics_df[["dataset", "sample_count", "duration_sec", "z_range", "xy_range", "path_length", "curvature_proxy"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
