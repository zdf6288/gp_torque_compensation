#!/usr/bin/env python3
"""Offline Stage 5 q7-focused support preflight validator.

This script compares reference/training CSV support with a candidate/live/test
CSV. It is offline-only: it does not import ROS, connect to Franka, launch
controllers, or modify any runtime behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("outputs/stage5_q7_support_preflight")
SUMMARY_JSON = "stage5_q7_support_summary.json"
SUMMARY_MD = "stage5_q7_support_summary.md"
JOINTS = range(1, 8)
SAFETY_NOTE = (
    "Offline/preflight support check only. This does not prove real-robot GP-on "
    "tracking improvement and does not authorize GP-on by itself."
)

Q7_CANDIDATE_GROUPS = [
    ["joint_pos_7", "q7", "joint7", "joint_7", "position_7", "q_7"],
    ["q[6]", "q6"],
]
DQ7_CANDIDATE_GROUPS = [
    ["joint_vel_7", "dq7", "joint7_velocity", "joint_7_velocity", "velocity_7", "dq_7"],
    ["dq[6]", "dq6"],
]

POSITION_COLUMN_SETS = [
    [f"joint_pos_{joint}" for joint in JOINTS],
    [f"q{joint}" for joint in JOINTS],
    [f"q_{joint}" for joint in JOINTS],
    [f"joint{joint}" for joint in JOINTS],
    [f"joint_{joint}" for joint in JOINTS],
    [f"position_{joint}" for joint in JOINTS],
    [f"q[{idx}]" for idx in range(7)],
    [f"q{idx}" for idx in range(7)],
]
VELOCITY_COLUMN_SETS = [
    [f"joint_vel_{joint}" for joint in JOINTS],
    [f"dq{joint}" for joint in JOINTS],
    [f"dq_{joint}" for joint in JOINTS],
    [f"joint{joint}_velocity" for joint in JOINTS],
    [f"joint_{joint}_velocity" for joint in JOINTS],
    [f"velocity_{joint}" for joint in JOINTS],
    [f"dq[{idx}]" for idx in range(7)],
    [f"dq{idx}" for idx in range(7)],
]


class ValidationError(Exception):
    """Expected validation failure with a reportable status."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class Stage5ArgumentParser(argparse.ArgumentParser):
    """Use exit code 1 for invalid CLI usage."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def parse_args() -> argparse.Namespace:
    parser = Stage5ArgumentParser(
        description="Offline Stage 5 q7-focused support preflight validator.",
    )
    parser.add_argument(
        "--reference-csv",
        action="append",
        type=Path,
        default=[],
        help="Reference/train CSV. May be repeated to combine multiple training runs.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optional model directory with metadata.json. Used to find source_csv when --reference-csv is omitted.",
    )
    parser.add_argument("--candidate-csv", type=Path, required=True, help="Candidate/live/test CSV to validate.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--label-reference", default="reference", help="Reference label written into reports.")
    parser.add_argument("--label-candidate", default="candidate", help="Candidate label written into reports.")
    parser.add_argument("--q7-margin", type=float, default=0.0, help="Allowed q7 support margin in rad.")
    parser.add_argument("--support-margin", type=float, default=0.0, help="Allowed 14D support margin in column units.")
    parser.add_argument("--q7-column", default=None, help="Explicit q7 position column name.")
    parser.add_argument("--dq7-column", default=None, help="Explicit dq7 velocity column name.")
    parser.add_argument("--strict", action="store_true", help="Fail when complete 14D joint support columns are missing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing summary JSON/Markdown outputs.")
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


def load_csv_numeric(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationError("fail_invalid_input", f"CSV not found: {path}")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError("fail_invalid_input", f"{path}: no CSV header found")
        columns = list(reader.fieldnames)
        data: dict[str, list[float]] = {column: [] for column in columns}
        for row in reader:
            for column in columns:
                data[column].append(parse_float(row.get(column)))

    rows = len(next(iter(data.values()))) if data else 0
    if rows == 0:
        raise ValidationError("fail_invalid_input", f"{path}: no data rows found")
    return {"path": path, "columns": columns, "data": data, "rows": rows}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_metadata_reference_csvs(model_dir: Path | None) -> list[Path]:
    if model_dir is None:
        return []
    metadata_path = resolve_path(model_dir) / "metadata.json"
    if not metadata_path.is_file():
        raise ValidationError("fail_invalid_input", f"metadata.json not found in model dir: {model_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_csv = str(metadata.get("source_csv", "")).strip()
    if not source_csv:
        raise ValidationError("fail_invalid_input", f"{metadata_path}: missing source_csv")
    return [resolve_path(Path(part.strip())) for part in source_csv.split(";") if part.strip()]


def normalized_column_map(columns: Iterable[str]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for column in columns:
        normalized = column.strip().lower()
        mapping.setdefault(normalized, []).append(column)
    return mapping


def matching_columns(columns: Iterable[str], candidate_names: Iterable[str]) -> list[str]:
    mapping = normalized_column_map(columns)
    matches: list[str] = []
    for name in candidate_names:
        matches.extend(mapping.get(name.lower(), []))
    return sorted(set(matches))


def choose_column(
    datasets: list[dict[str, Any]],
    candidate_groups: list[list[str]],
    explicit: str | None,
    label: str,
) -> tuple[str | None, list[str], str | None]:
    columns = datasets[0]["columns"]
    if explicit:
        missing = [repo_relative(dataset["path"]) for dataset in datasets if explicit not in dataset["columns"]]
        if missing:
            return None, [explicit], f"{label} column {explicit!r} missing in: {', '.join(missing)}"
        return explicit, [explicit], None

    all_candidates = sorted({name for group in candidate_groups for name in group})
    for group in candidate_groups:
        matches = matching_columns(columns, group)
        if len(matches) == 1:
            column = matches[0]
            missing = [repo_relative(dataset["path"]) for dataset in datasets if column not in dataset["columns"]]
            if missing:
                return None, matches, f"{label} column {column!r} missing in: {', '.join(missing)}"
            return column, matches, None
        if len(matches) > 1:
            return None, matches, f"ambiguous {label} columns: {', '.join(matches)}"
    return None, all_candidates, f"missing {label} column"


def choose_column_set(datasets: list[dict[str, Any]], column_sets: list[list[str]]) -> list[str] | None:
    for columns in column_sets:
        if all(all(column in dataset["columns"] for column in columns) for dataset in datasets):
            return columns
    return None


def finite_values(datasets: list[dict[str, Any]], column: str) -> list[float]:
    values: list[float] = []
    for dataset in datasets:
        values.extend(value for value in dataset["data"][column] if math.isfinite(value))
    if not values:
        raise ValidationError("fail_invalid_input", f"column {column!r} has no finite values")
    return values


def stats(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return {
        "count": count,
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": math.sqrt(variance),
    }


def support_row(
    name: str,
    reference_values: list[float],
    candidate_values: list[float],
    margin: float,
) -> dict[str, Any]:
    reference = stats(reference_values)
    candidate = stats(candidate_values)
    lower_limit = float(reference["min"]) - margin
    upper_limit = float(reference["max"]) + margin
    below = max(0.0, lower_limit - float(candidate["min"]))
    above = max(0.0, float(candidate["max"]) - upper_limit)
    if below > 0.0 and above > 0.0:
        side = "both"
    elif below > 0.0:
        side = "below"
    elif above > 0.0:
        side = "above"
    else:
        side = "none"
    return {
        "dimension": name,
        "reference_min": reference["min"],
        "reference_max": reference["max"],
        "candidate_min": candidate["min"],
        "candidate_max": candidate["max"],
        "candidate_mean": candidate["mean"],
        "candidate_std": candidate["std"],
        "margin": margin,
        "pass": side == "none",
        "violation_side": side,
        "max_violation_amount": max(below, above),
    }


def worst_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("max_violation_amount", 0.0)))


def determine_status(
    q7_pass: bool,
    joint_space_14d_available: bool,
    joint_space_14d_pass: bool | None,
    strict: bool,
) -> tuple[str, str]:
    if not q7_pass:
        return "fail_q7_out_of_support", "candidate q7 is outside reference q7 support"
    if not joint_space_14d_available:
        if strict:
            return "fail_strict_missing_14d", "strict mode requires complete q1..q7 + dq1..dq7 support columns"
        return (
            "pass_q7_only_preflight",
            "q7 support passed, but complete 14D joint-space support was not available",
        )
    if joint_space_14d_pass is False:
        return "fail_14d_out_of_support", "candidate has at least one 14D joint-space dimension outside support"
    return "pass_14d_support_preflight", "q7 and complete 14D joint-space support passed"


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No complete 14D joint-space support table is available."]
    lines = [
        "| dimension | reference_min | reference_max | candidate_min | candidate_max | candidate_mean | candidate_std | pass | violation_side | max_violation_amount |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {dimension} | {reference_min:.9g} | {reference_max:.9g} | {candidate_min:.9g} | "
            "{candidate_max:.9g} | {candidate_mean:.9g} | {candidate_std:.9g} | {pass_value} | "
            "{violation_side} | {max_violation_amount:.9g} |".format(
                pass_value=str(bool(row["pass"])).lower(),
                **row,
            )
        )
    return lines


def write_reports(summary: dict[str, Any], output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / SUMMARY_JSON
    md_path = output_dir / SUMMARY_MD
    if not overwrite:
        existing = [path for path in (json_path, md_path) if path.exists()]
        if existing:
            names = ", ".join(repo_relative(path) for path in existing)
            raise ValidationError("fail_invalid_input", f"output exists; pass --overwrite: {names}")

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    return json_path, md_path


def render_markdown(summary: dict[str, Any]) -> str:
    q7 = summary["q7_result"]
    joint_rows = summary.get("joint_space_14d_results", [])
    worst = summary.get("worst_dimension") or {}
    lines = [
        "# Stage 5 q7 Support Preflight Summary",
        "",
        "## Inputs",
        "",
        f"- reference_label: `{summary['labels']['reference']}`",
        f"- candidate_label: `{summary['labels']['candidate']}`",
        f"- reference_csv: `{';'.join(summary['input_paths']['reference_csv'])}`",
        f"- candidate_csv: `{summary['input_paths']['candidate_csv']}`",
        f"- q7_column_used: `{summary.get('q7_column_used') or ''}`",
        f"- dq7_column_used: `{summary.get('dq7_column_used') or ''}`",
        f"- q7_margin: `{summary['parameters']['q7_margin']}`",
        f"- support_margin: `{summary['parameters']['support_margin']}`",
        "",
        "## q7 Support Result",
        "",
        f"- q7_support_pass: `{str(summary['q7_support_pass']).lower()}`",
        f"- reference_min: `{q7['reference_min']:.9g}`",
        f"- reference_max: `{q7['reference_max']:.9g}`",
        f"- candidate_min: `{q7['candidate_min']:.9g}`",
        f"- candidate_max: `{q7['candidate_max']:.9g}`",
        f"- candidate_mean: `{q7['candidate_mean']:.9g}`",
        f"- candidate_std: `{q7['candidate_std']:.9g}`",
        f"- violation_side: `{q7['violation_side']}`",
        f"- max_violation_amount: `{q7['max_violation_amount']:.9g}`",
        "",
        "## Joint-Space Support Result",
        "",
        f"- joint_space_14d_available: `{str(summary['joint_space_14d_available']).lower()}`",
        f"- joint_space_14d_pass: `{str(summary.get('joint_space_14d_pass')).lower()}`",
        f"- limited_dimension_check: `{str(summary['limited_dimension_check']).lower()}`",
        f"- worst_dimension: `{worst.get('dimension', '')}`",
        f"- worst_violation_side: `{worst.get('violation_side', '')}`",
        f"- worst_max_violation_amount: `{worst.get('max_violation_amount', 0.0):.9g}`",
        "",
        *markdown_table(joint_rows),
        "",
        "## Overall Recommendation",
        "",
        f"- overall_status: `{summary['overall_status']}`",
        f"- blocking_reason: `{summary['blocking_reason']}`",
        "",
        "This is an offline/preflight support check. A pass result can only support further review; it cannot approve GP-on by itself.",
        "",
        "## Safety Notes",
        "",
        "- This does not prove real-robot GP-on tracking improvement.",
        "- This does not authorize GP-on by itself.",
        "- GP-on still requires conservative scale, clip, no online update, and live support gate.",
        "- Do not proceed if q7 or 14D support fails.",
        "",
    ]
    return "\n".join(lines)


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    reference_paths = [resolve_path(path) for path in args.reference_csv]
    if not reference_paths:
        reference_paths = read_metadata_reference_csvs(args.model_dir)
    if not reference_paths:
        raise ValidationError("fail_invalid_input", "provide --reference-csv or --model-dir with metadata source_csv")

    reference_datasets = [load_csv_numeric(path) for path in reference_paths]
    candidate_path = resolve_path(args.candidate_csv)
    candidate_datasets = [load_csv_numeric(candidate_path)]
    all_for_q7 = reference_datasets + candidate_datasets

    q7_column, q7_candidates, q7_error = choose_column(all_for_q7, Q7_CANDIDATE_GROUPS, args.q7_column, "q7")
    if q7_error or q7_column is None:
        raise ValidationError("fail_missing_q7_column", q7_error or "missing q7 column")

    dq7_column, dq7_candidates, _dq7_error = choose_column(all_for_q7, DQ7_CANDIDATE_GROUPS, args.dq7_column, "dq7")
    position_columns = choose_column_set(all_for_q7, POSITION_COLUMN_SETS)
    velocity_columns = choose_column_set(all_for_q7, VELOCITY_COLUMN_SETS)
    joint_space_14d_available = position_columns is not None and velocity_columns is not None

    q7_result = support_row(
        q7_column,
        finite_values(reference_datasets, q7_column),
        finite_values(candidate_datasets, q7_column),
        args.q7_margin,
    )
    joint_rows: list[dict[str, Any]] = []
    if joint_space_14d_available:
        assert position_columns is not None
        assert velocity_columns is not None
        for column in position_columns + velocity_columns:
            margin = args.q7_margin if column == q7_column else args.support_margin
            joint_rows.append(
                support_row(
                    column,
                    finite_values(reference_datasets, column),
                    finite_values(candidate_datasets, column),
                    margin,
                )
            )

    joint_space_14d_pass = all(bool(row["pass"]) for row in joint_rows) if joint_space_14d_available else None
    overall_status, blocking_reason = determine_status(
        bool(q7_result["pass"]),
        joint_space_14d_available,
        joint_space_14d_pass,
        args.strict,
    )
    worst = worst_row(joint_rows if joint_rows else [q7_result])

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths": {
            "reference_csv": [repo_relative(path) for path in reference_paths],
            "candidate_csv": repo_relative(candidate_path),
            "model_dir": repo_relative(resolve_path(args.model_dir)) if args.model_dir else None,
        },
        "labels": {
            "reference": args.label_reference,
            "candidate": args.label_candidate,
        },
        "parameters": {
            "q7_margin": args.q7_margin,
            "support_margin": args.support_margin,
            "strict": bool(args.strict),
        },
        "q7_column_used": q7_column,
        "dq7_column_used": dq7_column,
        "q7_candidate_columns": q7_candidates,
        "dq7_candidate_columns": dq7_candidates,
        "reference_stats": {
            "rows": sum(int(dataset["rows"]) for dataset in reference_datasets),
            "q7": stats(finite_values(reference_datasets, q7_column)),
        },
        "candidate_stats": {
            "rows": int(candidate_datasets[0]["rows"]),
            "q7": stats(finite_values(candidate_datasets, q7_column)),
        },
        "q7_result": q7_result,
        "q7_support_pass": bool(q7_result["pass"]),
        "joint_space_14d_available": joint_space_14d_available,
        "joint_space_14d_pass": joint_space_14d_pass,
        "joint_space_14d_columns": (position_columns + velocity_columns) if joint_space_14d_available else [],
        "joint_space_14d_results": joint_rows,
        "overall_status": overall_status,
        "blocking_reason": blocking_reason,
        "worst_dimension": worst,
        "limited_dimension_check": not joint_space_14d_available,
        "safety_note": SAFETY_NOTE,
    }


def exit_code_for_status(status: str) -> int:
    if status in ("pass_q7_only_preflight", "pass_14d_support_preflight"):
        return 0
    if status in ("fail_invalid_input", "fail_missing_q7_column"):
        return 1
    if status in ("fail_q7_out_of_support", "fail_14d_out_of_support", "fail_strict_missing_14d"):
        return 2
    return 1


def main() -> int:
    args = parse_args()
    try:
        summary = build_summary(args)
        json_path, md_path = write_reports(summary, resolve_path(args.output_dir), args.overwrite)
    except ValidationError as exc:
        print(f"{exc.status}: {exc.message}", file=sys.stderr)
        return exit_code_for_status(exc.status)
    print(f"overall_status: {summary['overall_status']}")
    print(f"blocking_reason: {summary['blocking_reason']}")
    print(f"summary_json: {repo_relative(json_path)}")
    print(f"summary_md: {repo_relative(md_path)}")
    return exit_code_for_status(str(summary["overall_status"]))


if __name__ == "__main__":
    raise SystemExit(main())
