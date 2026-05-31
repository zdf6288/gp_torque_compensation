#!/usr/bin/env python3
"""Offline GOAL1 joint-space tracking error analysis.

This script intentionally has no ROS2 dependency. It reads one or two saved CSV
logs, computes tracking metrics, and writes summaries plus optional plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


JOINT_COUNT = 7
CARTESIAN_DIM = 6
DEFAULT_TIME_COL = "time"

# Extend this table if future robot-side logs use different column names.
ALIASES: dict[str, tuple[str, ...]] = {
    "time": ("timestamp", "elapsed_time", "t"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze GOAL1 offline tracking error CSV logs without ROS2.",
    )
    parser.add_argument("--no-gp-csv", type=Path, help="CSV log for the without-GP run.")
    parser.add_argument("--gp-on-csv", type=Path, help="CSV log for the with-GP run.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for analysis outputs.")
    parser.add_argument("--label-no-gp", default="without_gp", help="Label for --no-gp-csv.")
    parser.add_argument("--label-gp-on", default="with_gp", help="Label for --gp-on-csv.")
    parser.add_argument("--time-col", default=DEFAULT_TIME_COL, help="Time column name.")
    return parser.parse_args()


def canonical_joint_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{joint}" for joint in range(1, JOINT_COUNT + 1)]


def canonical_cartesian_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{idx}" for idx in range(1, CARTESIAN_DIM + 1)]


def resolve_column(columns: set[str], canonical: str) -> str | None:
    if canonical in columns:
        return canonical
    for alias in ALIASES.get(canonical, ()):
        if alias in columns:
            return alias
    return None


def resolve_group(columns: set[str], names: list[str]) -> list[str] | None:
    resolved = []
    for name in names:
        column = resolve_column(columns, name)
        if column is None:
            return None
        resolved.append(column)
    return resolved


def require_group(columns: set[str], names: list[str], description: str) -> list[str]:
    resolved = resolve_group(columns, names)
    if resolved is not None:
        return resolved
    missing = [name for name in names if resolve_column(columns, name) is None]
    raise ValueError(f"missing required {description} columns: {', '.join(missing)}")


def optional_group(columns: set[str], names: list[str]) -> list[str] | None:
    return resolve_group(columns, names)


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"input CSV does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"input path is not a file: {path}")

    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows, list(reader.fieldnames)


def parse_float(value: str, column: str, row_number: int) -> float:
    if value is None or value == "":
        raise ValueError(f"empty numeric value in column {column} row {row_number}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"non-numeric value in column {column} row {row_number}: {value!r}") from exc


def matrix_from_rows(rows: list[dict[str, str]], columns: list[str]) -> list[list[float]]:
    matrix = []
    for row_index, row in enumerate(rows, start=2):
        matrix.append([parse_float(row[column], column, row_index) for column in columns])
    return matrix


def vector_from_rows(rows: list[dict[str, str]], column: str) -> list[float]:
    return [parse_float(row[column], column, row_index) for row_index, row in enumerate(rows, start=2)]


def rmse(values: list[float]) -> float:
    if not values:
        return math.nan
    return math.sqrt(sum(value * value for value in values) / len(values))


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def flatten_abs(matrix: list[list[float]]) -> list[float]:
    return [abs(value) for row in matrix for value in row]


def diff_matrix(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[left - right for left, right in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def column_values(matrix: list[list[float]], index: int) -> list[float]:
    return [row[index] for row in matrix]


def row_norms(matrix: list[list[float]]) -> list[float]:
    return [math.sqrt(sum(value * value for value in row)) for row in matrix]


def summarize_time(rows: list[dict[str, str]], time_column: str) -> tuple[dict[str, Any], list[str]]:
    time_values = vector_from_rows(rows, time_column)
    notes = []
    decreasing_count = 0
    duplicate_count = 0
    positive_dts = []

    for prev, current in zip(time_values, time_values[1:]):
        dt = current - prev
        if dt < 0.0:
            decreasing_count += 1
        elif dt == 0.0:
            duplicate_count += 1
        elif dt > 0.0:
            positive_dts.append(dt)

    if decreasing_count:
        raise ValueError(f"time column {time_column} is not monotonic nondecreasing")
    if duplicate_count:
        notes.append(f"time column {time_column} has {duplicate_count} duplicate timestamp step(s)")

    duration = time_values[-1] - time_values[0] if len(time_values) >= 2 else 0.0
    mean_dt = sum(positive_dts) / len(positive_dts) if positive_dts else math.nan
    sample_rate_hz = 1.0 / mean_dt if mean_dt > 0.0 else math.nan

    return {
        "start_time": time_values[0],
        "end_time": time_values[-1],
        "duration_s": duration,
        "sample_count": len(time_values),
        "mean_dt_s": mean_dt,
        "approx_sample_rate_hz": sample_rate_hz,
        "duplicate_time_steps": duplicate_count,
    }, notes


def summarize_matrix_abs(matrix: list[list[float]]) -> dict[str, float]:
    values = sorted(flatten_abs(matrix))
    return {
        "mean_abs": sum(values) / len(values) if values else math.nan,
        "p95_abs": percentile(values, 0.95),
        "max_abs": values[-1] if values else math.nan,
    }


def summarize_optional_effort(rows: list[dict[str, str]], columns: set[str], prefix: str) -> dict[str, Any] | None:
    group = optional_group(columns, canonical_joint_columns(prefix))
    if group is None:
        return None
    matrix = matrix_from_rows(rows, group)
    summary = summarize_matrix_abs(matrix)
    summary["columns"] = group
    return summary


def summarize_tau_clip(rows: list[dict[str, str]], columns: set[str]) -> dict[str, Any] | None:
    group = optional_group(columns, canonical_joint_columns("tau_clip"))
    if group is None:
        return None
    matrix = matrix_from_rows(rows, group)
    flat = flatten_abs(matrix)
    clipped = sum(1 for value in flat if value > 0.0)
    return {
        "columns": group,
        "clip_count": clipped,
        "clip_ratio": clipped / len(flat) if flat else math.nan,
        "max_abs": max(flat) if flat else math.nan,
    }


def summarize_gp_enabled(rows: list[dict[str, str]], columns: set[str]) -> dict[str, Any] | None:
    column = resolve_column(columns, "gp_compensation_enabled")
    if column is None:
        return None

    raw_values = [row[column].strip().lower() for row in rows]
    true_values = {"1", "1.0", "true", "yes", "on"}
    false_values = {"0", "0.0", "false", "no", "off"}
    parsed = []
    unknown = []
    for value in raw_values:
        if value in true_values:
            parsed.append(True)
        elif value in false_values:
            parsed.append(False)
        else:
            unknown.append(value)

    true_count = sum(1 for value in parsed if value)
    false_count = sum(1 for value in parsed if not value)
    return {
        "column": column,
        "true_count": true_count,
        "false_count": false_count,
        "unknown_count": len(unknown),
        "consistent": (true_count == 0 or false_count == 0) and not unknown,
    }


def analyze_run(path: Path, label: str, time_col: str) -> dict[str, Any]:
    rows, headers = read_csv_rows(path)
    columns = set(headers)

    time_column = resolve_column(columns, time_col)
    if time_column is None:
        raise ValueError(f"missing required time column: {time_col}")

    q_des_cols = require_group(columns, canonical_joint_columns("q_des"), "q_des")
    q_meas_cols = require_group(columns, canonical_joint_columns("q_meas"), "q_meas")
    q_des = matrix_from_rows(rows, q_des_cols)
    q_meas = matrix_from_rows(rows, q_meas_cols)
    q_error = diff_matrix(q_des, q_meas)
    q_norms = row_norms(q_error)

    time_summary, notes = summarize_time(rows, time_column)
    per_joint_rmse = [rmse(column_values(q_error, idx)) for idx in range(JOINT_COUNT)]
    per_joint_max_abs = [max(abs(value) for value in column_values(q_error, idx)) for idx in range(JOINT_COUNT)]

    result: dict[str, Any] = {
        "label": label,
        "csv_path": str(path),
        "time_column": time_column,
        "row_count": len(rows),
        "columns": headers,
        "notes": notes,
        "time": time_summary,
        "joint_tracking": {
            "q_des_columns": q_des_cols,
            "q_meas_columns": q_meas_cols,
            "per_joint_rmse": per_joint_rmse,
            "rmse_q_all": rmse(flatten_abs(q_error)),
            "per_joint_max_abs_error": per_joint_max_abs,
            "max_norm_error": max(q_norms),
        },
        "optional": {},
        "_series": {
            "time": vector_from_rows(rows, time_column),
            "q_des": q_des,
            "q_meas": q_meas,
            "q_error": q_error,
        },
    }

    dq_des_cols = optional_group(columns, canonical_joint_columns("dq_des"))
    dq_meas_cols = optional_group(columns, canonical_joint_columns("dq_meas"))
    if dq_des_cols is not None and dq_meas_cols is not None:
        dq_error = diff_matrix(matrix_from_rows(rows, dq_des_cols), matrix_from_rows(rows, dq_meas_cols))
        result["optional"]["dq_tracking"] = {
            "dq_des_columns": dq_des_cols,
            "dq_meas_columns": dq_meas_cols,
            "per_joint_rmse": [rmse(column_values(dq_error, idx)) for idx in range(JOINT_COUNT)],
            "rmse_dq_all": rmse(flatten_abs(dq_error)),
        }

    tau_cmd_summary = summarize_optional_effort(rows, columns, "tau_cmd")
    if tau_cmd_summary is not None:
        result["optional"]["tau_cmd"] = tau_cmd_summary
        result["_series"]["tau_cmd"] = matrix_from_rows(rows, tau_cmd_summary["columns"])

    tau_clip_summary = summarize_tau_clip(rows, columns)
    if tau_clip_summary is not None:
        result["optional"]["tau_clip"] = tau_clip_summary

    gp_y_hat_summary = summarize_optional_effort(rows, columns, "gp_y_hat")
    if gp_y_hat_summary is not None:
        result["optional"]["gp_y_hat"] = gp_y_hat_summary

    gp_comp_summary = summarize_optional_effort(rows, columns, "gp_compensation")
    if gp_comp_summary is not None:
        result["optional"]["gp_compensation"] = gp_comp_summary
        result["_series"]["gp_compensation"] = matrix_from_rows(rows, gp_comp_summary["columns"])

    gp_enabled_summary = summarize_gp_enabled(rows, columns)
    if gp_enabled_summary is not None:
        result["optional"]["gp_compensation_enabled"] = gp_enabled_summary

    x_des_cols = optional_group(columns, canonical_cartesian_columns("x_des"))
    x_meas_cols = optional_group(columns, canonical_cartesian_columns("x_meas"))
    if x_des_cols is not None and x_meas_cols is not None:
        x_error = diff_matrix(matrix_from_rows(rows, x_des_cols), matrix_from_rows(rows, x_meas_cols))
        result["optional"]["cartesian_tracking"] = {
            "x_des_columns": x_des_cols,
            "x_meas_columns": x_meas_cols,
            "rmse_x": rmse(flatten_abs(x_error)),
            "max_norm_error": max(row_norms(x_error)),
        }
        result["_series"]["x_error_norm"] = row_norms(x_error)

    return result


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_for_json(item) for key, item in value.items() if key != "_series"}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def metric_rows_for_run(run: dict[str, Any]) -> list[dict[str, Any]]:
    label = run["label"]
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: Any, unit: str = "") -> None:
        rows.append({"label": label, "metric": metric, "value": value, "unit": unit})

    add("row_count", run["row_count"], "rows")
    add("duration_s", run["time"]["duration_s"], "s")
    add("approx_sample_rate_hz", run["time"]["approx_sample_rate_hz"], "Hz")
    add("RMSE_q_all", run["joint_tracking"]["rmse_q_all"], "rad")
    add("e_q_max_norm", run["joint_tracking"]["max_norm_error"], "rad")

    for idx, value in enumerate(run["joint_tracking"]["per_joint_rmse"], start=1):
        add(f"RMSE_q_{idx}", value, "rad")
    for idx, value in enumerate(run["joint_tracking"]["per_joint_max_abs_error"], start=1):
        add(f"max_abs_q_error_{idx}", value, "rad")

    optional = run["optional"]
    if "dq_tracking" in optional:
        add("RMSE_dq_all", optional["dq_tracking"]["rmse_dq_all"], "rad/s")
        for idx, value in enumerate(optional["dq_tracking"]["per_joint_rmse"], start=1):
            add(f"RMSE_dq_{idx}", value, "rad/s")
    if "tau_cmd" in optional:
        add("tau_cmd_mean_abs", optional["tau_cmd"]["mean_abs"], "Nm")
        add("tau_cmd_p95_abs", optional["tau_cmd"]["p95_abs"], "Nm")
        add("tau_cmd_max_abs", optional["tau_cmd"]["max_abs"], "Nm")
    if "tau_clip" in optional:
        add("tau_clip_count", optional["tau_clip"]["clip_count"], "samples")
        add("tau_clip_ratio", optional["tau_clip"]["clip_ratio"], "ratio")
    if "gp_compensation" in optional:
        add("gp_compensation_mean_abs", optional["gp_compensation"]["mean_abs"], "Nm")
        add("gp_compensation_p95_abs", optional["gp_compensation"]["p95_abs"], "Nm")
        add("gp_compensation_max_abs", optional["gp_compensation"]["max_abs"], "Nm")
    if "gp_compensation_enabled" in optional:
        add("gp_compensation_enabled_consistent", optional["gp_compensation_enabled"]["consistent"], "bool")
    if "cartesian_tracking" in optional:
        add("RMSE_x", optional["cartesian_tracking"]["rmse_x"], "m_or_rad")
        add("e_x_max_norm", optional["cartesian_tracking"]["max_norm_error"], "m_or_rad")

    return rows


def write_summary_csv(runs: list[dict[str, Any]], out_dir: Path) -> None:
    rows = [row for run in runs for row in metric_rows_for_run(run)]
    path = out_dir / "goal1_tracking_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "metric", "value", "unit"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(runs: list[dict[str, Any]], out_dir: Path) -> None:
    path = out_dir / "goal1_tracking_summary.md"
    lines = [
        "# GOAL1 Tracking Error Summary",
        "",
        "| label | rows | duration_s | sample_rate_hz | RMSE_q_all | e_q_max_norm |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        lines.append(
            "| {label} | {rows} | {duration:.6g} | {rate:.6g} | {rmse:.6g} | {max_norm:.6g} |".format(
                label=run["label"],
                rows=run["row_count"],
                duration=run["time"]["duration_s"],
                rate=run["time"]["approx_sample_rate_hz"],
                rmse=run["joint_tracking"]["rmse_q_all"],
                max_norm=run["joint_tracking"]["max_norm_error"],
            )
        )

    lines.extend(["", "## Per-Joint RMSE_q", "", "| label | q1 | q2 | q3 | q4 | q5 | q6 | q7 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for run in runs:
        values = " | ".join(f"{value:.6g}" for value in run["joint_tracking"]["per_joint_rmse"])
        lines.append(f"| {run['label']} | {values} |")

    lines.extend(["", "## Notes", ""])
    for run in runs:
        notes = run["notes"] or ["none"]
        lines.append(f"- {run['label']}: " + "; ".join(notes))

    path.write_text("\n".join(lines) + "\n")


def write_metrics_json(runs: list[dict[str, Any]], out_dir: Path) -> None:
    path = out_dir / "goal1_tracking_metrics.json"
    payload = {"runs": [clean_for_json(run) for run in runs]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def import_matplotlib(out_dir: Path) -> Any | None:
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = out_dir / ".matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("warning: matplotlib is not available; skipping plot generation", file=sys.stderr)
        return None
    return plt


def plot_line(plt: Any, series: list[tuple[str, list[float], list[float]]], title: str, y_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for label, x_values, values in series:
        ax.plot(x_values, values, label=label, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_bar(plt: Any, labels: list[str], values_by_name: list[tuple[str, list[float]]], title: str, y_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x_positions = list(range(len(labels)))
    width = 0.8 / max(1, len(values_by_name))
    offset_start = -0.4 + width / 2.0
    for idx, (name, values) in enumerate(values_by_name):
        offsets = [pos + offset_start + idx * width for pos in x_positions]
        ax.bar(offsets, values, width=width, label=name)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_plots(runs: list[dict[str, Any]], out_dir: Path) -> None:
    plt = import_matplotlib(out_dir)
    if plt is None:
        return

    joint_labels = [f"q{idx}" for idx in range(1, JOINT_COUNT + 1)]
    for idx in range(JOINT_COUNT):
        q_series = []
        error_series = []
        for run in runs:
            label = run["label"]
            time_values = run["_series"]["time"]
            q_series.append((f"{label} q_des_{idx + 1}", time_values, column_values(run["_series"]["q_des"], idx)))
            q_series.append((f"{label} q_meas_{idx + 1}", time_values, column_values(run["_series"]["q_meas"], idx)))
            error_series.append((f"{label} q_error_{idx + 1}", time_values, column_values(run["_series"]["q_error"], idx)))
        plot_line(
            plt,
            q_series,
            f"q_des vs q_meas joint {idx + 1}",
            "rad",
            out_dir / f"q_des_vs_q_meas_joint_{idx + 1}.png",
        )
        plot_line(
            plt,
            error_series,
            f"q error joint {idx + 1}",
            "rad",
            out_dir / f"q_error_joint_{idx + 1}.png",
        )

    plot_bar(
        plt,
        joint_labels,
        [(run["label"], run["joint_tracking"]["per_joint_rmse"]) for run in runs],
        "RMSE_q per joint",
        "rad",
        out_dir / "rmse_q_per_joint.png",
    )
    plot_bar(
        plt,
        joint_labels,
        [(run["label"], run["joint_tracking"]["per_joint_max_abs_error"]) for run in runs],
        "Max abs q error per joint",
        "rad",
        out_dir / "max_abs_q_error_per_joint.png",
    )

    tau_runs = [run for run in runs if "tau_cmd" in run["_series"]]
    if tau_runs:
        series = [
            (f"{run['label']} tau_cmd_{idx + 1}", run["_series"]["time"], column_values(run["_series"]["tau_cmd"], idx))
            for run in tau_runs
            for idx in range(JOINT_COUNT)
        ]
        plot_line(plt, series, "tau_cmd per joint", "Nm", out_dir / "tau_cmd_per_joint.png")

    gp_runs = [run for run in runs if "gp_compensation" in run["_series"]]
    if gp_runs:
        series = [
            (f"{run['label']} gp_compensation_{idx + 1}", run["_series"]["time"], column_values(run["_series"]["gp_compensation"], idx))
            for run in gp_runs
            for idx in range(JOINT_COUNT)
        ]
        plot_line(plt, series, "GP compensation per joint", "Nm", out_dir / "gp_compensation_per_joint.png")

    cartesian_runs = [run for run in runs if "x_error_norm" in run["_series"]]
    if cartesian_runs:
        series = [(run["label"], run["_series"]["time"], run["_series"]["x_error_norm"]) for run in cartesian_runs]
        plot_line(plt, series, "Cartesian error norm", "m_or_rad", out_dir / "cartesian_error_norm.png")

    if len(runs) == 2:
        plot_bar(
            plt,
            joint_labels,
            [(run["label"], run["joint_tracking"]["per_joint_rmse"]) for run in runs],
            "Comparison RMSE_q per joint",
            "rad",
            out_dir / "comparison_rmse_q_per_joint.png",
        )
        plot_bar(
            plt,
            ["RMSE_q_all", "e_q_max_norm"],
            [
                (
                    run["label"],
                    [run["joint_tracking"]["rmse_q_all"], run["joint_tracking"]["max_norm_error"]],
                )
                for run in runs
            ],
            "Comparison overall metrics",
            "rad",
            out_dir / "comparison_overall_metrics.png",
        )


def validate_inputs(args: argparse.Namespace) -> list[tuple[Path, str]]:
    inputs = []
    if args.no_gp_csv is not None:
        inputs.append((args.no_gp_csv, args.label_no_gp))
    if args.gp_on_csv is not None:
        inputs.append((args.gp_on_csv, args.label_gp_on))
    if not inputs:
        raise ValueError("provide at least one of --no-gp-csv or --gp-on-csv")
    return inputs


def main() -> int:
    args = parse_args()
    try:
        inputs = validate_inputs(args)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        runs = [analyze_run(path, label, args.time_col) for path, label in inputs]
        write_summary_csv(runs, args.out_dir)
        write_summary_md(runs, args.out_dir)
        write_metrics_json(runs, args.out_dir)
        write_plots(runs, args.out_dir)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote GOAL1 tracking analysis to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
