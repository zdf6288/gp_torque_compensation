#!/usr/bin/env python3
"""Offline sanity checker for GOAL2 timing CSV files.

This script is intentionally read-only for input data and does not import ROS,
Franka packages, or project controller modules.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Any


EXPECTED_COLUMNS = [
    "callback_wall_ms",
    "callback_deadline_ms",
    "callback_deadline_ratio",
    "gp_total_ms",
    "gp_add_point_ms",
    "delay_steps",
    "control_frequency",
    "gp_prediction_enabled",
    "gp_online_update_enabled",
    "gp_compensation_enabled",
    "gp_compensation_source",
]

TIMING_COLUMNS = [
    "callback_wall_ms",
    "callback_deadline_ms",
    "callback_deadline_ratio",
    "gp_total_ms",
    "gp_add_point_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read GOAL2 timing CSV files or directories and print offline sanity summaries.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Timing CSV file(s) or directory/directories scanned recursively for *.csv.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="Warn when a CSV has fewer rows than this value. Default: 1.",
    )
    return parser.parse_args()


def collect_csv_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_file():
            paths.append(input_path)
        elif input_path.is_dir():
            paths.extend(sorted(path for path in input_path.rglob("*.csv") if path.is_file()))
        else:
            print(f"[missing] {input_path}")
    return sorted(dict.fromkeys(paths))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)
    except OSError as exc:
        print(f"[error] cannot read {path}: {exc}")
        return [], []
    except csv.Error as exc:
        print(f"[error] cannot parse {path}: {exc}")
        return [], []


def parse_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_numeric(rows: list[dict[str, str]], column: str) -> dict[str, float | int | None]:
    values = [number for row in rows if (number := parse_finite_float(row.get(column))) is not None]
    missing_or_nonfinite = len(rows) - len(values)
    if not values:
        return {
            "finite": 0,
            "missing_or_nonfinite": missing_or_nonfinite,
            "mean": None,
            "p95": None,
            "max": None,
        }
    return {
        "finite": len(values),
        "missing_or_nonfinite": missing_or_nonfinite,
        "mean": statistics.fmean(values),
        "p95": percentile(values, 95.0),
        "max": max(values),
    }


def format_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def unique_values(rows: list[dict[str, str]], column: str, limit: int = 8) -> list[str]:
    values = sorted({str(row.get(column, "")).strip() for row in rows if str(row.get(column, "")).strip()})
    if len(values) > limit:
        return values[:limit] + [f"...({len(values) - limit} more)"]
    return values


def print_csv_summary(path: Path, min_rows: int) -> int:
    fieldnames, rows = read_csv(path)
    status = 0
    print()
    print(f"== {path} ==")
    print(f"rows: {len(rows)}")

    if not fieldnames:
        print("columns: none")
        print("status: WARN empty or unreadable CSV")
        return 1

    if len(rows) < min_rows:
        print(f"warning: rows {len(rows)} < min_rows {min_rows}")
        status = 1

    missing = [column for column in EXPECTED_COLUMNS if column not in fieldnames]
    present = [column for column in EXPECTED_COLUMNS if column in fieldnames]
    print("present expected columns: " + (", ".join(present) if present else "none"))
    print("missing optional expected columns: " + (", ".join(missing) if missing else "none"))

    for column in TIMING_COLUMNS:
        if column not in fieldnames:
            continue
        summary = summarize_numeric(rows, column)
        if summary["missing_or_nonfinite"]:
            status = 1
        print(
            f"{column}: finite={summary['finite']} "
            f"missing_or_nonfinite={summary['missing_or_nonfinite']} "
            f"mean={format_number(summary['mean'])} "
            f"p95={format_number(summary['p95'])} "
            f"max={format_number(summary['max'])}"
        )

    for column in [
        "control_frequency",
        "delay_steps",
        "gp_prediction_enabled",
        "gp_online_update_enabled",
        "gp_compensation_enabled",
        "gp_compensation_source",
    ]:
        if column in fieldnames:
            print(f"{column}: {', '.join(unique_values(rows, column)) or 'none'}")

    print("status: " + ("WARN" if status else "OK"))
    return status


def main() -> int:
    args = parse_args()
    if args.min_rows < 0:
        print("--min-rows must be >= 0", file=sys.stderr)
        return 2

    csv_paths = collect_csv_paths(args.inputs)
    if not csv_paths:
        print("No CSV files found.")
        return 1

    overall_status = 0
    for path in csv_paths:
        overall_status = max(overall_status, print_csv_summary(path, args.min_rows))

    print()
    print(f"checked_files: {len(csv_paths)}")
    print("overall_status: " + ("WARN" if overall_status else "OK"))
    return overall_status


if __name__ == "__main__":
    raise SystemExit(main())
