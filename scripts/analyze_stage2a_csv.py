#!/usr/bin/env python3
"""Offline Stage 2A CSV inventory and quick-look plots.

This script intentionally has no ROS2 dependency. It reads saved CSV logs,
summarizes structure and basic data quality, and writes small diagnostic plots.
"""

from __future__ import annotations

import argparse
import os
import sys
import math
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT_DIR = Path("data/stage2a/csv")
DEFAULT_OUTPUT_DIR = Path("outputs/stage2a_analysis")
EXPECTED_SAMPLES = 3000
EXPECTED_ROUNDS = 6
MAX_PLOT_COLUMNS = 12
HUGE_VALUE_THRESHOLD = 1e6


GROUP_PATTERNS = {
    "q": ("joint_pos", "q_", "q"),
    "dq": ("joint_vel", "dq", "qdot", "q_dot"),
    "tau": ("tau",),
    "gp": ("y_hat", "yhat", "gp", "prediction"),
    "error": ("residual", "error", "err"),
    "cartesian": ("x_actual", "y_actual", "z_actual", "x_desired", "y_desired", "z_desired"),
}

np = None
pd = None
plt = None


def import_dependencies(output_dir: Path) -> bool:
    global np, pd, plt

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
        print("Install them in a project .venv or use an existing environment; do not install globally by default.", file=sys.stderr)
        return False

    np = numpy_module
    pd = pandas_module
    plt = pyplot_module
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Stage 2A offline CSV logs without ROS2.",
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        type=Path,
        help="Specific CSV files to analyze. If omitted, all CSVs in --input-dir are used.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory used when no CSV paths are provided. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=EXPECTED_SAMPLES,
        help=f"Expected sample count for a complete Stage 2A run. Default: {EXPECTED_SAMPLES}",
    )
    parser.add_argument(
        "--expected-rounds",
        type=int,
        default=EXPECTED_ROUNDS,
        help=f"Expected trajectory rounds. Default: {EXPECTED_ROUNDS}",
    )
    return parser.parse_args()


def find_csv_files(args: argparse.Namespace) -> list[Path]:
    if args.csv_paths:
        return sorted(path for path in args.csv_paths if path.suffix.lower() == ".csv")
    if not args.input_dir.exists():
        return []
    return sorted(args.input_dir.glob("*.csv"))


def normalized_name(name: str) -> str:
    return name.lower().replace("(", "_").replace(")", "_").replace("-", "_")


def detect_time_column(columns: Iterable[str]) -> str | None:
    candidates = []
    for column in columns:
        lowered = normalized_name(column)
        if any(pattern in lowered for pattern in ("time", "timestamp", "elapsed")):
            candidates.append(column)
        elif lowered.strip("_") == "t":
            candidates.append(column)
    if not candidates:
        return None
    for preferred in ("time", "elapsed", "timestamp"):
        for column in candidates:
            if preferred in normalized_name(column):
                return column
    return candidates[0]


def likely_columns(columns: Iterable[str], patterns: Iterable[str]) -> list[str]:
    result = []
    for column in columns:
        lowered = normalized_name(column)
        if any(pattern in lowered for pattern in patterns):
            result.append(column)
    return result


def strict_joint_position_columns(columns: Iterable[str]) -> list[str]:
    result = []
    for column in columns:
        lowered = normalized_name(column)
        if lowered.startswith("joint_pos") or lowered.startswith("q_"):
            result.append(column)
    return result


def summarize_time(df: pd.DataFrame, time_column: str | None) -> tuple[float | None, float | None, float | None, list[str]]:
    notes = []
    if time_column is None:
        notes.append("no time-like column detected")
        return None, None, None, notes

    series = pd.to_numeric(df[time_column], errors="coerce").dropna()
    if len(series) < 2:
        notes.append(f"time column {time_column} has fewer than two numeric values")
        return None, None, None, notes

    diffs = series.diff().dropna()
    diffs = diffs[np.isfinite(diffs)]
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.empty:
        notes.append(f"time column {time_column} is not strictly increasing enough for dt estimate")
        duration = float(series.iloc[-1] - series.iloc[0])
        return duration, None, None, notes

    duration = float(series.iloc[-1] - series.iloc[0])
    median_dt = float(positive_diffs.median())
    estimated_hz = 1.0 / median_dt if median_dt > 0 else None
    return duration, median_dt, estimated_hz, notes


def count_inf(df: pd.DataFrame, numeric_columns: list[str]) -> int:
    if not numeric_columns:
        return 0
    values = df[numeric_columns].to_numpy(dtype=float, copy=True)
    return int(np.isinf(values).sum())


def count_huge_values(df: pd.DataFrame, numeric_columns: list[str]) -> int:
    if not numeric_columns:
        return 0
    values = df[numeric_columns].to_numpy(dtype=float, copy=True)
    finite_values = values[np.isfinite(values)]
    return int((np.abs(finite_values) > HUGE_VALUE_THRESHOLD).sum())


def choose_x(df: pd.DataFrame, time_column: str | None) -> tuple[pd.Series, str]:
    if time_column and time_column in df.columns:
        return pd.to_numeric(df[time_column], errors="coerce"), time_column
    return pd.Series(np.arange(len(df)), name="sample"), "sample"


def limited_columns(columns: list[str], limit: int = MAX_PLOT_COLUMNS) -> tuple[list[str], bool]:
    return columns[:limit], len(columns) > limit


def save_plot(
    df: pd.DataFrame,
    columns: list[str],
    time_column: str | None,
    title: str,
    output_path: Path,
) -> bool:
    selected, truncated = limited_columns(columns)
    if not selected:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x, x_label = choose_x(df, time_column)

    fig, ax = plt.subplots(figsize=(12, 6))
    for column in selected:
        ax.plot(x, pd.to_numeric(df[column], errors="coerce"), label=column, linewidth=0.9)
    suffix = " (truncated)" if truncated else ""
    ax.set_title(f"{title}{suffix}")
    ax.set_xlabel(x_label)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return truncated


def analyze_csv(path: Path, output_dir: Path, expected_samples: int, expected_rounds: int) -> dict[str, object]:
    print(f"\n=== {path.name} ===")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive CLI behavior
        print(f"WARNING: failed to read {path}: {exc}")
        return {
            "filename": path.name,
            "rows": 0,
            "columns": 0,
            "detected_time_column": "",
            "estimated_duration": math.nan,
            "median_dt": math.nan,
            "estimated_hz": math.nan,
            "nan_count_total": math.nan,
            "inf_count_total": math.nan,
            "likely_q_columns_count": 0,
            "likely_dq_columns_count": 0,
            "likely_tau_columns_count": 0,
            "likely_gp_columns_count": 0,
            "likely_error_columns_count": 0,
            "notes": f"read failed: {exc}",
        }

    print(f"rows: {len(df)}")
    print(f"columns: {len(df.columns)}")
    print("column names:")
    for column in df.columns:
        print(f"  - {column}")

    numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
    non_numeric_columns = [column for column in df.columns if column not in numeric_columns]
    time_column = detect_time_column(df.columns)
    duration, median_dt, estimated_hz, time_notes = summarize_time(df, time_column)

    nan_count = int(df.isna().sum().sum())
    inf_count = count_inf(df, numeric_columns)
    huge_count = count_huge_values(df, numeric_columns)

    q_columns = strict_joint_position_columns(df.columns)
    dq_columns = likely_columns(df.columns, GROUP_PATTERNS["dq"])
    tau_columns = likely_columns(df.columns, GROUP_PATTERNS["tau"])
    gp_columns = likely_columns(df.columns, GROUP_PATTERNS["gp"])
    error_columns = likely_columns(df.columns, GROUP_PATTERNS["error"])
    cartesian_columns = likely_columns(df.columns, GROUP_PATTERNS["cartesian"])

    notes = []
    notes.extend(time_notes)
    if len(df) == expected_samples:
        notes.append(f"sample count matches expected {expected_rounds} rounds / {expected_samples} points")
    else:
        notes.append(f"sample count differs from expected {expected_samples}")
    if non_numeric_columns:
        notes.append(f"{len(non_numeric_columns)} non-numeric columns")
    if huge_count:
        notes.append(f"{huge_count} finite numeric values exceed abs threshold {HUGE_VALUE_THRESHOLD:g}")
    if inf_count:
        notes.append(f"{inf_count} inf values detected")
    if nan_count:
        notes.append(f"{nan_count} NaN values detected")

    print(f"numeric columns: {len(numeric_columns)}")
    print(f"NaN total: {nan_count}")
    print(f"inf total: {inf_count}")
    print(f"huge finite values > {HUGE_VALUE_THRESHOLD:g}: {huge_count}")
    print(f"time column: {time_column or 'not detected'}")
    if median_dt is not None and estimated_hz is not None:
        print(f"median dt: {median_dt:.6g} s")
        print(f"estimated hz: {estimated_hz:.3f}")
    elif duration is not None and len(df) > 1:
        inferred_hz = (len(df) - 1) / duration if duration > 0 else math.nan
        print(f"duration: {duration:.6g} s")
        print(f"estimated hz from duration only: {inferred_hz:.3f}")
    else:
        print("sampling estimate: unavailable")

    print("detected groups:")
    print(f"  q: {len(q_columns)}")
    print(f"  dq: {len(dq_columns)}")
    print(f"  tau: {len(tau_columns)}")
    print(f"  gp: {len(gp_columns)}")
    print(f"  residual/error: {len(error_columns)}")
    print(f"  cartesian obvious: {len(cartesian_columns)}")

    plot_dir = output_dir / "plots" / path.stem
    plot_notes = []
    overview_columns = numeric_columns[:MAX_PLOT_COLUMNS]
    plot_specs = [
        ("numeric_overview", overview_columns, "Numeric overview"),
        ("tau_columns", tau_columns, "Tau-related columns"),
        ("q_dq_columns", q_columns + dq_columns, "Joint position / velocity columns"),
        ("gp_residual_error_columns", gp_columns + error_columns, "GP / residual / error columns"),
        ("cartesian_columns", cartesian_columns, "Cartesian actual / desired columns"),
    ]
    for filename, columns, title in plot_specs:
        truncated = save_plot(df, columns, time_column, title, plot_dir / f"{filename}.png")
        if truncated:
            plot_notes.append(f"{filename} plot truncated to {MAX_PLOT_COLUMNS} columns")
    notes.extend(plot_notes)

    return {
        "filename": path.name,
        "rows": len(df),
        "columns": len(df.columns),
        "detected_time_column": time_column or "",
        "estimated_duration": duration if duration is not None else math.nan,
        "median_dt": median_dt if median_dt is not None else math.nan,
        "estimated_hz": estimated_hz if estimated_hz is not None else math.nan,
        "nan_count_total": nan_count,
        "inf_count_total": inf_count,
        "likely_q_columns_count": len(q_columns),
        "likely_dq_columns_count": len(dq_columns),
        "likely_tau_columns_count": len(tau_columns),
        "likely_gp_columns_count": len(gp_columns),
        "likely_error_columns_count": len(error_columns),
        "notes": "; ".join(notes),
    }


def print_final_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        print("\nNo CSV files were analyzed.")
        return

    columns = [
        "filename",
        "rows",
        "columns",
        "detected_time_column",
        "median_dt",
        "estimated_hz",
        "nan_count_total",
        "inf_count_total",
    ]
    print("\nFinal summary:")
    print(summary[columns].to_string(index=False))

    comparable_columns = summary["columns"].nunique() == 1
    comparable_rows = summary["rows"].nunique() == 1
    print(f"\nStructural comparability: columns_match={comparable_columns}, rows_match={comparable_rows}")


def main() -> int:
    args = parse_args()
    csv_files = find_csv_files(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not import_dependencies(args.output_dir):
        return 2

    if not csv_files:
        print(f"WARNING: no CSV files found in {args.input_dir}")
        return 1

    rows = [
        analyze_csv(path, args.output_dir, args.expected_samples, args.expected_rounds)
        for path in csv_files
    ]
    summary = pd.DataFrame(rows)
    summary_path = args.output_dir / "stage2a_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote summary CSV: {summary_path}")
    print_final_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
