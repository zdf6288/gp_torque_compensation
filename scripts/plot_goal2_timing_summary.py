#!/usr/bin/env python3
"""Create offline GOAL2 timing summaries and optional plots.

The script accepts timing CSV files or directories, writes a compact summary
CSV into an explicit output directory, and writes PNG plots when matplotlib is
available. It never imports ROS or robot runtime modules.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any


METRIC_COLUMNS = [
    "callback_wall_ms",
    "callback_deadline_ratio",
    "gp_total_ms",
]

SUMMARY_ALIASES = {
    "callback_wall": {
        "mean": ["callback_wall_mean", "callback_wall_mean_ms"],
        "p95": ["callback_wall_p95", "callback_wall_p95_ms"],
        "max": ["callback_wall_max", "callback_wall_max_ms"],
    },
    "callback_deadline_ratio": {
        "mean": ["callback_deadline_ratio_mean", "deadline_ratio_mean"],
        "p95": ["callback_deadline_ratio_p95", "deadline_ratio_p95"],
        "max": ["callback_deadline_ratio_max", "deadline_ratio_max"],
    },
    "gp_total": {
        "mean": ["gp_total_mean", "gp_total_mean_ms"],
        "p95": ["gp_total_p95", "gp_total_p95_ms"],
        "max": ["gp_total_max", "gp_total_max_ms"],
    },
}

GROUP_COLUMNS = [
    "control_frequency",
    "delay_steps",
    "gp_compensation_source",
    "gp_online_update_enabled",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize GOAL2 timing CSVs and optionally plot p95/max timing metrics.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Timing CSV file(s), summary CSV file(s), or directories scanned recursively for *.csv.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for goal2_timing_summary.csv and optional PNG plots.",
    )
    parser.add_argument(
        "--group-by",
        choices=GROUP_COLUMNS,
        default=None,
        help="Optional grouping column for summary rows and plots.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write the summary CSV; skip matplotlib plotting.",
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
    except (OSError, csv.Error) as exc:
        print(f"[skip] {path}: {exc}")
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


def summarize_values(rows: list[dict[str, str]], column: str) -> dict[str, float | int | None]:
    values = [number for row in rows if (number := parse_finite_float(row.get(column))) is not None]
    if not values:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p95": percentile(values, 95.0),
        "max": max(values),
    }


def first_present(fieldnames: list[str], aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in fieldnames:
            return alias
    return None


def summarize_metric(
    fieldnames: list[str],
    rows: list[dict[str, str]],
    raw_column: str,
    metric_key: str,
) -> tuple[dict[str, float | int | None], list[str]]:
    if raw_column in fieldnames:
        return summarize_values(rows, raw_column), []

    aliases = SUMMARY_ALIASES[metric_key]
    p95_column = first_present(fieldnames, aliases["p95"])
    max_column = first_present(fieldnames, aliases["max"])
    mean_column = first_present(fieldnames, aliases["mean"])
    if not p95_column and not max_column and not mean_column:
        return {"count": 0, "mean": None, "p95": None, "max": None}, [raw_column]

    p95_values = [number for row in rows if p95_column and (number := parse_finite_float(row.get(p95_column))) is not None]
    max_values = [number for row in rows if max_column and (number := parse_finite_float(row.get(max_column))) is not None]
    mean_values = [number for row in rows if mean_column and (number := parse_finite_float(row.get(mean_column))) is not None]
    return {
        "count": max(len(p95_values), len(max_values), len(mean_values)),
        "mean": statistics.fmean(mean_values) if mean_values else None,
        "p95": max(p95_values) if p95_values else None,
        "max": max(max_values) if max_values else None,
    }, []


def format_csv_value(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.10g}"


def make_summary_rows(
    csv_paths: list[Path],
    group_by: str | None,
) -> tuple[list[dict[str, str]], set[str]]:
    summary_rows: list[dict[str, str]] = []
    missing_columns: set[str] = set()

    for path in csv_paths:
        fieldnames, rows = read_csv(path)
        if not fieldnames:
            continue

        if group_by and group_by not in fieldnames:
            missing_columns.add(group_by)

        groups: dict[str, list[dict[str, str]]] = {"all": rows}
        if group_by and group_by in fieldnames:
            groups = {}
            for row in rows:
                key = str(row.get(group_by, "")).strip() or "(blank)"
                groups.setdefault(key, []).append(row)

        for group_value, group_rows in sorted(groups.items()):
            summary: dict[str, str] = {
                "source_csv": str(path),
                "group_by": group_by or "",
                "group_value": group_value if group_by else "",
                "rows": str(len(group_rows)),
            }
            for column, metric_key in [
                ("callback_wall_ms", "callback_wall"),
                ("callback_deadline_ratio", "callback_deadline_ratio"),
                ("gp_total_ms", "gp_total"),
            ]:
                values, missing = summarize_metric(fieldnames, group_rows, column, metric_key)
                missing_columns.update(missing)
                prefix = metric_key
                summary[f"{prefix}_count"] = format_csv_value(values.get("count"))
                summary[f"{prefix}_mean"] = format_csv_value(values.get("mean"))
                summary[f"{prefix}_p95"] = format_csv_value(values.get("p95"))
                summary[f"{prefix}_max"] = format_csv_value(values.get("max"))
            summary_rows.append(summary)

    return summary_rows, missing_columns


def write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "source_csv",
        "group_by",
        "group_value",
        "rows",
        "callback_wall_count",
        "callback_wall_mean",
        "callback_wall_p95",
        "callback_wall_max",
        "callback_deadline_ratio_count",
        "callback_deadline_ratio_mean",
        "callback_deadline_ratio_p95",
        "callback_deadline_ratio_max",
        "gp_total_count",
        "gp_total_mean",
        "gp_total_p95",
        "gp_total_max",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: list[dict[str, str]], out_dir: Path, metric_key: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not available; summary CSV was still written.")
        return

    labels = [Path(row["source_csv"]).stem for row in rows]
    if any(row["group_value"] for row in rows):
        labels = [
            f"{Path(row['source_csv']).stem}\n{row['group_value']}" if row["group_value"] else Path(row["source_csv"]).stem
            for row in rows
        ]
    p95_values = [parse_finite_float(row.get(f"{metric_key}_p95")) or math.nan for row in rows]
    max_values = [parse_finite_float(row.get(f"{metric_key}_max")) or math.nan for row in rows]

    if all(math.isnan(value) for value in p95_values + max_values):
        print(f"[plots] skip {metric_key}: no finite values")
        return

    x_values = list(range(len(rows)))
    plt.figure(figsize=(max(8, len(rows) * 0.8), 4.8))
    plt.plot(x_values, p95_values, marker="o", label="p95")
    plt.plot(x_values, max_values, marker="o", label="max")
    plt.xticks(x_values, labels, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(metric_key)
    plt.legend()
    plt.tight_layout()
    output_path = out_dir / f"{metric_key}_p95_max.png"
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"[plots] wrote {output_path}")


def main() -> int:
    args = parse_args()
    csv_paths = collect_csv_paths(args.inputs)
    if not csv_paths:
        print("No CSV files found.")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, missing_columns = make_summary_rows(csv_paths, args.group_by)
    summary_path = args.out_dir / "goal2_timing_summary.csv"
    write_summary_csv(summary_path, rows)
    print(f"Wrote summary CSV: {summary_path}")

    if missing_columns:
        print("Missing optional columns encountered: " + ", ".join(sorted(missing_columns)))

    if not args.no_plots and rows:
        plot_metric(rows, args.out_dir, "callback_wall", "GOAL2 callback wall timing")
        plot_metric(rows, args.out_dir, "callback_deadline_ratio", "GOAL2 callback deadline ratio")
        plot_metric(rows, args.out_dir, "gp_total", "GOAL2 GP total timing")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
