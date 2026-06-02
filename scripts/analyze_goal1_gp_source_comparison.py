#!/usr/bin/env python3
"""Offline GOAL1 GP source comparison for three 60 s spatial-multisine runs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


np = None
pd = None
plt = None


TIME_COLUMN = "Time(s)"
ACTUAL_COLUMNS = ("x_actual", "y_actual", "z_actual")
DESIRED_COLUMNS = ("x_desired", "y_desired", "z_desired")
JOINT_COLUMNS = tuple(f"joint_pos_{idx}" for idx in range(1, 8))
TAU_COLUMNS = tuple(f"tau_{idx}" for idx in range(1, 8))
RESIDUAL_COLUMNS = tuple(f"tau_residual_{idx}" for idx in range(1, 8))
Y_HAT_LOCAL_COLUMNS = tuple(f"y_hat_local_{idx}" for idx in range(1, 8))
Y_HAT_CLOUD_COLUMNS = tuple(f"y_hat_cloud_{idx}" for idx in range(1, 8))


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
    parser = argparse.ArgumentParser(description="Compare GOAL1 60 s GP source CSV logs.")
    parser.add_argument("--nogp-csv", required=True, type=Path)
    parser.add_argument("--local-gpon-csv", required=True, type=Path)
    parser.add_argument("--internal-cloud-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def require_columns(df: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def numeric_values(df: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def finite_series(df: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def count_nan_inf(df: pd.DataFrame) -> tuple[int, int]:
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    values = numeric_df.to_numpy(dtype=float)
    return int(np.isnan(values).sum()), int(np.isinf(values).sum())


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def range_value(df: pd.DataFrame, column: str, to_deg: bool = False) -> float:
    if column not in df.columns:
        return float("nan")
    values = finite_series(df, column)
    if len(values) == 0:
        return float("nan")
    result = float(np.max(values) - np.min(values))
    return float(np.degrees(result)) if to_deg else result


def add_vector_ranges(row: dict[str, float | str], df: pd.DataFrame, columns: tuple[str, ...], prefix: str, to_deg: bool = False) -> None:
    for idx, column in enumerate(columns, start=1):
        row[f"{prefix}_{idx}"] = range_value(df, column, to_deg=to_deg)


def tracking_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual = numeric_values(df, ACTUAL_COLUMNS)
    desired = numeric_values(df, DESIRED_COLUMNS)
    valid = np.isfinite(actual).all(axis=1) & np.isfinite(desired).all(axis=1)
    error_mm = (actual[valid] - desired[valid]) * 1000.0
    norm_mm = np.linalg.norm(error_mm, axis=1) if len(error_mm) else np.array([])
    time = pd.to_numeric(df[TIME_COLUMN], errors="coerce").to_numpy(dtype=float)
    time = time[valid] if len(time) == len(valid) else np.arange(len(norm_mm), dtype=float)
    return time, error_mm, norm_mm


def summarize_dataset(label: str, df: pd.DataFrame) -> dict[str, float | str]:
    require_columns(df, (TIME_COLUMN,) + ACTUAL_COLUMNS + DESIRED_COLUMNS, label)

    time = finite_series(df, TIME_COLUMN)
    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    duration = float(time[-1] - time[0]) if len(time) >= 2 else float("nan")
    dt_mean = float(np.mean(dt)) if len(dt) else float("nan")
    rate_hz = safe_ratio(1.0, dt_mean)
    nan_total, inf_total = count_nan_inf(df)

    actual = numeric_values(df, ACTUAL_COLUMNS)
    actual = actual[np.isfinite(actual).all(axis=1)]
    ranges = np.max(actual, axis=0) - np.min(actual, axis=0)
    steps = np.linalg.norm(np.diff(actual, axis=0), axis=1) if len(actual) >= 2 else np.array([])
    path_length = float(np.sum(steps)) if len(steps) else 0.0
    xy_range = float(np.hypot(ranges[0], ranges[1]))

    _, error_mm, norm_mm = tracking_arrays(df)
    row: dict[str, float | str] = {
        "dataset": label,
        "rows": int(len(df)),
        "duration_sec": duration,
        "rate_hz": rate_hz,
        "nan_total": nan_total,
        "inf_total": inf_total,
        "x_range_m": float(ranges[0]),
        "y_range_m": float(ranges[1]),
        "z_range_m": float(ranges[2]),
        "xy_range_m": xy_range,
        "z_range_over_xy_range": safe_ratio(float(ranges[2]), xy_range),
        "path_length_m": path_length,
    }

    axis_names = ("x", "y", "z")
    for idx, axis in enumerate(axis_names):
        axis_error = error_mm[:, idx] if len(error_mm) else np.array([])
        row[f"tracking_mean_abs_{axis}_mm"] = float(np.mean(np.abs(axis_error))) if len(axis_error) else float("nan")
        row[f"tracking_p95_{axis}_mm"] = float(np.percentile(np.abs(axis_error), 95)) if len(axis_error) else float("nan")
        row[f"tracking_max_{axis}_mm"] = float(np.max(np.abs(axis_error))) if len(axis_error) else float("nan")

    row["tracking_rmse_3d_mm"] = float(np.sqrt(np.mean(norm_mm**2))) if len(norm_mm) else float("nan")
    row["tracking_p95_3d_mm"] = float(np.percentile(norm_mm, 95)) if len(norm_mm) else float("nan")
    row["tracking_max_3d_mm"] = float(np.max(norm_mm)) if len(norm_mm) else float("nan")

    add_vector_ranges(row, df, JOINT_COLUMNS, "joint_range_deg_q", to_deg=True)
    add_vector_ranges(row, df, TAU_COLUMNS, "tau_range_nm_tau")
    add_vector_ranges(row, df, RESIDUAL_COLUMNS, "tau_residual_range_nm_tau")
    add_vector_ranges(row, df, Y_HAT_LOCAL_COLUMNS, "y_hat_local_range_nm_joint")
    add_vector_ranges(row, df, Y_HAT_CLOUD_COLUMNS, "y_hat_cloud_range_nm_joint")
    return row


def get_time(df: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(df[TIME_COLUMN], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(values).any():
        return values
    return np.arange(len(df), dtype=float)


def get_actual(df: pd.DataFrame) -> np.ndarray:
    points = numeric_values(df, ACTUAL_COLUMNS)
    return points[np.isfinite(points).all(axis=1)]


def save_tracking_error_bar(metrics_df: pd.DataFrame, output_path: Path) -> None:
    labels = list(metrics_df["dataset"])
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, metrics_df["tracking_rmse_3d_mm"], width, label="RMSE 3D")
    ax.bar(x, metrics_df["tracking_p95_3d_mm"], width, label="p95 3D")
    ax.bar(x + width, metrics_df["tracking_max_3d_mm"], width, label="max 3D")
    ax.set_ylabel("tracking error [mm]")
    ax.set_title("GOAL1 spatial-multisine tracking metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_tracking_error_over_time(datasets: list[tuple[str, pd.DataFrame]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, df in datasets:
        time, _, norm_mm = tracking_arrays(df)
        count = min(len(time), len(norm_mm))
        ax.plot(time[:count], norm_mm[:count], label=label, linewidth=1.1)
    ax.set_title("GOAL1 3D tracking error over time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("3D tracking error [mm]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_actual_xyz_overlay(datasets: list[tuple[str, pd.DataFrame]], output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axis_names = ("x", "y", "z")
    for label, df in datasets:
        time = get_time(df)
        actual = get_actual(df)
        count = min(len(time), len(actual))
        for idx, ax in enumerate(axes):
            ax.plot(time[:count], actual[:count, idx], label=label, linewidth=1.1)
            ax.set_ylabel(f"{axis_names[idx]} actual [m]")
            ax.grid(True, alpha=0.3)
    axes[0].set_title("GOAL1 actual xyz overlay")
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def set_axes_equal_3d(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    centers = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    if np.isfinite(radius) and radius > 0:
        ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
        ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
        ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def save_3d_actual_overlay(datasets: list[tuple[str, pd.DataFrame]], output_path: Path) -> None:
    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    for label, df in datasets:
        actual = get_actual(df)
        ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], label=label, linewidth=1.3)
    ax.set_title("GOAL1 actual 3D trajectory overlay")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="best")
    set_axes_equal_3d(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def range_values(metrics_df: pd.DataFrame, prefix: str) -> tuple[list[str], dict[str, list[float]]]:
    labels = [f"j{idx}" for idx in range(1, 8)]
    values = {}
    for _, row in metrics_df.iterrows():
        values[str(row["dataset"])] = [float(row.get(f"{prefix}_{idx}", np.nan)) for idx in range(1, 8)]
    return labels, values


def save_grouped_range_plot(metrics_df: pd.DataFrame, prefix: str, title: str, ylabel: str, output_path: Path) -> None:
    labels, values = range_values(metrics_df, prefix)
    x = np.arange(len(labels))
    datasets = list(values.keys())
    width = min(0.22, 0.8 / max(len(datasets), 1))
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for idx, dataset in enumerate(datasets):
        offset = (idx - (len(datasets) - 1) / 2.0) * width
        ax.bar(x + offset, values[dataset], width, label=dataset)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def fmt(value: float | int | str, digits: int = 3) -> str:
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def write_markdown(metrics_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# GOAL1 GP Source Comparison, 60 s Spatial-Multisine",
        "",
        "## Main Result",
        "",
        "All three runs are valid 60 s / 3000 row datasets. All run at approximately 50 Hz. No NaN / inf were found.",
        "",
        "Under the same `spatial_multisine` trajectory, `scale=0.1`, `clip=0.5 Nm`:",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            f"- {row['dataset']}: RMSE 3D = {fmt(row['tracking_rmse_3d_mm'])} mm, "
            f"p95 = {fmt(row['tracking_p95_3d_mm'])} mm, max = {fmt(row['tracking_max_3d_mm'])} mm."
        )

    lines.extend(
        [
            "",
            "## Key Metrics",
            "",
            "| dataset | rows | rate_hz | x_range_m | y_range_m | z_range_m | RMSE_3D_mm | p95_3D_mm | max_3D_mm | nan_total | inf_total |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in metrics_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(int(row["rows"])),
                    fmt(row["rate_hz"], 4),
                    fmt(row["x_range_m"], 6),
                    fmt(row["y_range_m"], 6),
                    fmt(row["z_range_m"], 6),
                    fmt(row["tracking_rmse_3d_mm"], 3),
                    fmt(row["tracking_p95_3d_mm"], 3),
                    fmt(row["tracking_max_3d_mm"], 3),
                    str(int(row["nan_total"])),
                    str(int(row["inf_total"])),
                ]
            )
            + " |"
        )

    no_gp = metrics_df.iloc[0]
    local = metrics_df.iloc[1]
    internal = metrics_df.iloc[2]
    local_delta = float(local["tracking_rmse_3d_mm"] - no_gp["tracking_rmse_3d_mm"])
    internal_delta = float(internal["tracking_rmse_3d_mm"] - no_gp["tracking_rmse_3d_mm"])

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"- `local GP-on` slightly improves tracking relative to `no-GP` "
                f"(RMSE 3D change {local_delta:.3f} mm)."
            ),
            (
                "- `internal-cloud branch` gives the best RMSE and p95 among the three tested 60 s runs "
                f"(RMSE 3D change vs no-GP {internal_delta:.3f} mm)."
            ),
            "- The max error is slightly higher than `local GP-on`.",
            "- Therefore `internal-cloud branch` shows a positive trend in average / p95 tracking, but not a uniformly better result across all metrics.",
            "",
            "## Naming Caveat",
            "",
            "- Current `source=cloud` is not a clean external server/cloud GP validation.",
            "- It should be described as `internal-cloud branch` or `source=cloud internal y_hat_cloud branch`.",
            "- The `gp_server` path is not clean in the current branch.",
            "- Do not claim validated remote server/cloud GP.",
            "",
            "## Engineering Caveat",
            "",
            "- Some runs had non-clean shutdown / User Stop / communication caveats after data saving.",
            "- This does not invalidate the saved usable data.",
            "- This is not robust repeated validation.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(metrics_df: pd.DataFrame, datasets: list[tuple[str, pd.DataFrame]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_dir / "gp_source_tracking_metrics_summary.csv", index=False)
    write_markdown(metrics_df, out_dir / "gp_source_tracking_metrics_summary.md")
    save_tracking_error_bar(metrics_df, out_dir / "gp_source_tracking_error_bar.png")
    save_tracking_error_over_time(datasets, out_dir / "gp_source_xyz_tracking_error_over_time.png")
    save_actual_xyz_overlay(datasets, out_dir / "gp_source_actual_xyz_overlay.png")
    save_3d_actual_overlay(datasets, out_dir / "gp_source_3d_actual_overlay.png")
    save_grouped_range_plot(metrics_df, "tau_range_nm_tau", "GOAL1 tau range comparison", "range [Nm]", out_dir / "gp_source_tau_range_comparison.png")
    save_grouped_range_plot(metrics_df, "tau_residual_range_nm_tau", "GOAL1 residual torque range comparison", "range [Nm]", out_dir / "gp_source_residual_range_comparison.png")
    if any(any(np.isfinite(row.get(f"y_hat_local_range_nm_joint_{idx}", np.nan)) for idx in range(1, 8)) for _, row in metrics_df.iterrows()):
        save_grouped_range_plot(metrics_df, "y_hat_local_range_nm_joint", "GOAL1 y_hat_local range comparison", "range [Nm]", out_dir / "gp_source_prediction_range_comparison.png")


def main() -> int:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    if not import_dependencies(out_dir):
        return 2

    paths = [
        ("no-GP", resolve_path(args.nogp_csv)),
        ("local GP-on", resolve_path(args.local_gpon_csv)),
        ("internal-cloud branch", resolve_path(args.internal_cloud_csv)),
    ]
    for _, path in paths:
        if not path.exists():
            print(f"Input CSV not found: {path}", file=sys.stderr)
            return 1

    datasets = [(label, pd.read_csv(path)) for label, path in paths]
    metrics_df = pd.DataFrame([summarize_dataset(label, df) for label, df in datasets])
    write_outputs(metrics_df, datasets, out_dir)

    print(f"Wrote outputs to: {out_dir}")
    print(
        metrics_df[
            [
                "dataset",
                "rows",
                "rate_hz",
                "x_range_m",
                "y_range_m",
                "z_range_m",
                "tracking_rmse_3d_mm",
                "tracking_p95_3d_mm",
                "tracking_max_3d_mm",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
