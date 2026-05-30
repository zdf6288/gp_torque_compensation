#!/usr/bin/env python3
"""Offline sanity checker for GOAL2 D controller timing CSV files.

This script is intentionally offline-only: it does not import ROS, Franka
packages, or project controller modules, and it never modifies input CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COLUMNS = [
    "callback_wall_ms",
    "callback_period_ms",
    "callback_deadline_ms",
    "callback_deadline_ratio",
    "callback_deadline_miss",
    "gp_total_ms",
    "gp_local_predict_ms",
    "gp_cloud_like_predict_ms",
    "gp_add_point_ms",
    "future_request_ms",
    "csv_append_ms",
    "csv_save_ms",
    "exception_flag",
]

KEY_COLUMNS = [
    "callback_wall_ms",
    "callback_deadline_ms",
    "callback_deadline_ratio",
    "callback_deadline_miss",
    "exception_flag",
]

OPTIONAL_EXPECTED_COLUMNS = [column for column in EXPECTED_COLUMNS if column not in KEY_COLUMNS]
GROUP_COLUMN = "delay_steps"
REPO_OUTPUT_DIR = "outputs"
SOURCE_LIKE_PARTS = {
    "config",
    "controller",
    "controllers",
    "custom_msgs",
    "franka_hardware",
    "franka_ros2",
    "hardware",
    "launch",
    "new_structure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GOAL2 offline timing CSV sanity checker. No ROS, no Franka, read-only CSV analysis.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="Single timing CSV file.")
    source.add_argument("--dir", type=Path, help="Directory to scan recursively for CSV files.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional path for summary JSON output.")
    parser.add_argument("--min-rows", type=positive_int, default=100, help="Minimum expected data rows. Default: 100.")
    parser.add_argument(
        "--deadline-ratio-warn",
        type=positive_float,
        default=0.8,
        help="Warn threshold for callback_deadline_ratio. Default: 0.8.",
    )
    parser.add_argument(
        "--deadline-ratio-fail",
        type=positive_float,
        default=1.0,
        help="Fail threshold for callback_deadline_ratio. Default: 1.0.",
    )
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def collect_csv_paths(args: argparse.Namespace) -> list[Path]:
    if args.csv is not None:
        return [args.csv]
    return sorted(path for path in args.dir.rglob("*.csv") if path.is_file())


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def is_safe_output_json_path(path: Path, repo_root: Path) -> tuple[bool, str]:
    # offline checker 只应写 analysis artifact，避免误写 source/config/controller 路径。
    if path.suffix.lower() != ".json":
        return False, "--out-json must end with .json"

    resolved = path.expanduser().resolve()
    resolved_repo_root = repo_root.resolve()
    if is_relative_to(resolved, resolved_repo_root):
        relative = resolved.relative_to(resolved_repo_root)
        if relative.parts and relative.parts[0] == REPO_OUTPUT_DIR:
            return True, ""
        return False, "repo-internal --out-json is only allowed under outputs/"

    source_like_parts = {part.lower() for part in resolved.parts}
    blocked_parts = sorted(source_like_parts & SOURCE_LIKE_PARTS)
    if blocked_parts:
        return False, "outside-repo --out-json path looks source-like: " + ", ".join(blocked_parts)
    return True, ""


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


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
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_summary(rows: list[dict[str, str]], column: str) -> dict[str, Any]:
    values = [parsed for row in rows if (parsed := parse_number(row.get(column))) is not None]
    if not values:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p95": percentile(values, 95.0),
        "max": max(values),
    }


def empty_metrics() -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for column in EXPECTED_COLUMNS:
        if column in {"callback_deadline_miss", "exception_flag"}:
            metrics[column] = {"count": 0, "ratio": None}
        else:
            metrics[column] = {"count": 0, "mean": None, "p95": None, "max": None}
    return metrics


def count_truthy(rows: list[dict[str, str]], column: str) -> int:
    count = 0
    for row in rows:
        value = row.get(column)
        parsed = parse_number(value)
        if parsed is not None:
            count += int(parsed != 0.0)
            continue
        text = str(value or "").strip().lower()
        if text in {"true", "yes", "y", "on"}:
            count += 1
    return count


def summarize_group(rows: list[dict[str, str]], delay_steps: str) -> dict[str, Any]:
    # `delay_steps` grouping 用来快速对比不同 cloud-like step-delay 的 deadline 压力。
    deadline_miss_count = count_truthy(rows, "callback_deadline_miss")
    exception_count = count_truthy(rows, "exception_flag")
    ratio = numeric_summary(rows, "callback_deadline_ratio")
    wall = numeric_summary(rows, "callback_wall_ms")
    return {
        "delay_steps": delay_steps,
        "rows": len(rows),
        "callback_deadline_ratio": ratio,
        "callback_wall_ms": wall,
        "callback_deadline_miss": {
            "count": deadline_miss_count,
            "ratio": deadline_miss_count / len(rows) if rows else None,
        },
        "exception_flag": {"count": exception_count},
    }


def classify(summary: dict[str, Any], args: argparse.Namespace) -> tuple[str, list[str]]:
    reasons: list[str] = []
    rows = summary["rows"]
    missing_key = [column for column in summary["missing_expected_columns"] if column in KEY_COLUMNS]
    missing_optional = [column for column in summary["missing_expected_columns"] if column in OPTIONAL_EXPECTED_COLUMNS]
    deadline_ratio = summary["metrics"].get("callback_deadline_ratio", {})
    deadline_ratio_p95 = deadline_ratio.get("p95")
    deadline_ratio_max = deadline_ratio.get("max")
    deadline_miss_count = summary["metrics"]["callback_deadline_miss"]["count"]
    exception_count = summary["metrics"]["exception_flag"]["count"]

    if rows < args.min_rows:
        reasons.append(f"rows {rows} < min_rows {args.min_rows}")
    if missing_key:
        reasons.append("missing key columns: " + ", ".join(missing_key))
    if deadline_ratio_max is not None and deadline_ratio_max >= args.deadline_ratio_fail:
        reasons.append(f"deadline ratio max {deadline_ratio_max:.3f} >= fail {args.deadline_ratio_fail:.3f}")
    if deadline_miss_count > 0:
        reasons.append(f"deadline miss count {deadline_miss_count} > 0")
    if exception_count > 0:
        reasons.append(f"exception_flag count {exception_count} > 0")

    if reasons:
        return "FAIL", reasons

    warn_reasons: list[str] = []
    if missing_optional:
        warn_reasons.append("missing optional columns: " + ", ".join(missing_optional))
    if deadline_ratio_p95 is not None and deadline_ratio_p95 >= args.deadline_ratio_warn:
        warn_reasons.append(f"deadline ratio p95 {deadline_ratio_p95:.3f} >= warn {args.deadline_ratio_warn:.3f}")
    if deadline_ratio_max is not None and deadline_ratio_max >= args.deadline_ratio_warn:
        warn_reasons.append(f"deadline ratio max {deadline_ratio_max:.3f} >= warn {args.deadline_ratio_warn:.3f}")

    if warn_reasons:
        return "WARN", warn_reasons
    return "OK", ["rows sufficient, no deadline miss, no exception, deadline ratio below warn threshold"]


def analyze_csv(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    columns, rows = read_csv(path)
    missing = [column for column in EXPECTED_COLUMNS if column not in columns]

    # deadline ratio 接近 1.0 代表 callback wall time 接近控制周期预算。
    # deadline miss 是更直接的硬门槛：callback 已经超过 deadline。
    # exception_flag 非零说明 controller callback 里发生异常，不能当作正常 timing 数据。
    metrics: dict[str, Any] = {}
    for column in EXPECTED_COLUMNS:
        if column in {"callback_deadline_miss", "exception_flag"}:
            count = count_truthy(rows, column)
            metrics[column] = {
                "count": count,
                "ratio": count / len(rows) if rows else None,
            }
        else:
            metrics[column] = numeric_summary(rows, column)

    grouping: list[dict[str, Any]] = []
    if GROUP_COLUMN in columns:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            key = str(row.get(GROUP_COLUMN, "")).strip() or "(blank)"
            grouped[key].append(row)
        grouping = [summarize_group(group_rows, key) for key, group_rows in sorted(grouped.items())]

    summary = {
        "path": str(path),
        "rows": len(rows),
        "detected_columns": columns,
        "missing_expected_columns": missing,
        "metrics": metrics,
        "delay_steps_grouping": grouping,
    }
    status, reasons = classify(summary, args)
    summary["status"] = status
    summary["status_reasons"] = reasons
    return summary


def fmt_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_metric_line(name: str, metric: dict[str, Any]) -> None:
    if "mean" in metric:
        print(
            f"  {name}: count={metric['count']} "
            f"mean={fmt_number(metric['mean'])} p95={fmt_number(metric['p95'])} max={fmt_number(metric['max'])}"
        )
    else:
        print(f"  {name}: count={metric['count']} ratio={fmt_number(metric.get('ratio'))}")


def print_summary(summary: dict[str, Any]) -> None:
    print(f"\n== {summary['path']} ==")
    print(f"status: {summary['status']}")
    for reason in summary["status_reasons"]:
        print(f"  - {reason}")
    print(f"rows: {summary['rows']}")
    print("detected columns: " + ", ".join(summary["detected_columns"]))
    missing = summary["missing_expected_columns"]
    print("missing expected columns: " + (", ".join(missing) if missing else "none"))
    print("metrics:")
    for column in EXPECTED_COLUMNS:
        print_metric_line(column, summary["metrics"][column])

    if summary["delay_steps_grouping"]:
        print("delay_steps grouping:")
        for group in summary["delay_steps_grouping"]:
            ratio = group["callback_deadline_ratio"]
            miss = group["callback_deadline_miss"]
            exc = group["exception_flag"]
            print(
                f"  delay_steps={group['delay_steps']} rows={group['rows']} "
                f"ratio_p95={fmt_number(ratio['p95'])} ratio_max={fmt_number(ratio['max'])} "
                f"deadline_miss={miss['count']} exception_flag={exc['count']}"
            )


def overall_status(file_summaries: list[dict[str, Any]]) -> str:
    statuses = {summary["status"] for summary in file_summaries}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "OK"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    if args.out_json is not None:
        is_safe, reason = is_safe_output_json_path(args.out_json, REPO_ROOT)
        if not is_safe:
            print(f"Unsafe --out-json path '{args.out_json}': {reason}", file=sys.stderr)
            return 2

    paths = collect_csv_paths(args)
    if not paths:
        print("No CSV files found.", file=sys.stderr)
        return 2

    summaries: list[dict[str, Any]] = []
    for path in paths:
        try:
            summary = analyze_csv(path, args)
        except (csv.Error, OSError, UnicodeDecodeError, ValueError) as exc:
            summary = {
                "path": str(path),
                "rows": 0,
                "detected_columns": [],
                "missing_expected_columns": EXPECTED_COLUMNS,
                "metrics": empty_metrics(),
                "delay_steps_grouping": [],
                "status": "FAIL",
                "status_reasons": [f"could not read CSV ({type(exc).__name__}): {exc}"],
            }
        summaries.append(summary)
        print_summary(summary)

    payload = {
        "generated_at_utc": utc_now(),
        "overall_status": overall_status(summaries),
        "min_rows": args.min_rows,
        "deadline_ratio_warn": args.deadline_ratio_warn,
        "deadline_ratio_fail": args.deadline_ratio_fail,
        "files": summaries,
    }

    print(f"\noverall_status: {payload['overall_status']}")
    if args.out_json is not None:
        try:
            write_json(args.out_json, payload)
        except OSError as exc:
            print(f"Could not write --out-json '{args.out_json}' ({type(exc).__name__}): {exc}", file=sys.stderr)
            return 2
        print(f"wrote JSON summary: {args.out_json}")

    return 1 if payload["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
