#!/usr/bin/env python3
"""Offline Stage 5 q7 / joint7 ablation utilities.

First implemented mode: compare full 14D support gates against a 12D ablation
that excludes joint_pos_7 and joint_vel_7. This script is offline-only. It does
not import ROS, connect to Franka, launch controllers, or modify runtime
control behavior.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True

import run_stage5_support_matrix as matrix_runner
import validate_stage5_q7_support as validator


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("outputs/stage5_q7_ablation")
SUPPORT_GATE_CSV = "stage5_q7_ablation_support_gate.csv"
SUPPORT_GATE_MD = "stage5_q7_ablation_support_gate.md"
EXCLUDED_Q7_DQ7 = ("joint_pos_7", "joint_vel_7")

CSV_FIELDS = [
    "pair_id",
    "row_status",
    "reference_label",
    "candidate_label",
    "reference_source",
    "candidate_csv",
    "gate_mode",
    "support_pass",
    "overall_status",
    "worst_dimension",
    "blocking_reason",
    "num_dimensions_checked",
    "excluded_dimensions",
    "q7_reference_mean",
    "q7_candidate_mean",
    "q7_reference_range",
    "q7_candidate_range",
]

SAFETY_NOTE = (
    "Offline support-gate ablation only. A 12D pass does not authorize GP-on, "
    "does not justify using a 14D-trained GP outside 14D support, and does not "
    "replace real-robot validation."
)


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    reference_label: str
    candidate_label: str
    candidate_csv: Path
    reference_csv: tuple[Path, ...] = ()
    model_dir: Path | None = None


class Stage5Q7AblationParser(argparse.ArgumentParser):
    """Use exit code 1 for invalid CLI usage."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def parse_args() -> argparse.Namespace:
    parser = Stage5Q7AblationParser(
        description="Offline Stage 5 q7 / joint7 ablation runner.",
    )
    parser.add_argument(
        "--mode",
        choices=("support-gate",),
        default="support-gate",
        help="Ablation mode to run. Default: support-gate.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite this script's generated outputs.")
    parser.add_argument("--auto-stage4", action="store_true", help="Discover existing Stage 4 reference/candidate pairs.")
    parser.add_argument("--support-margin", type=float, default=0.0, help="Support margin for non-q7 dimensions.")
    parser.add_argument("--q7-margin", type=float, default=0.0, help="Support margin for q7.")
    parser.add_argument(
        "--matrix-csv",
        type=Path,
        default=None,
        help="Optional existing support matrix CSV to reuse pair definitions.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.9g}"
    return str(value)


def range_text(stats: dict[str, Any]) -> str:
    return f"{fmt(stats.get('min'))}..{fmt(stats.get('max'))}"


def reference_source(pair: PairSpec) -> str:
    if pair.model_dir is not None:
        return repo_relative(resolve_path(pair.model_dir))
    return ";".join(repo_relative(resolve_path(path)) for path in pair.reference_csv)


def convert_matrix_pair(pair: matrix_runner.PairSpec) -> PairSpec:
    return PairSpec(
        pair_id=pair.pair_id,
        reference_label=pair.reference_label,
        candidate_label=pair.candidate_label,
        candidate_csv=pair.candidate_csv,
        reference_csv=tuple(pair.reference_csv),
        model_dir=pair.model_dir,
    )


def unique_pair_id(base_id: str, used_ids: set[str]) -> str:
    candidate = matrix_runner.safe_id(base_id)
    if candidate not in used_ids:
        used_ids.add(candidate)
        return candidate
    index = 2
    while f"{candidate}_{index}" in used_ids:
        index += 1
    unique = f"{candidate}_{index}"
    used_ids.add(unique)
    return unique


def pair_from_matrix_row(row: dict[str, str], used_ids: set[str]) -> PairSpec:
    pair_id = unique_pair_id(row.get("pair_id", "pair"), used_ids)
    reference_source_value = row.get("reference_source", "").strip()
    candidate_csv = Path(row.get("candidate_csv", "").strip())
    if not candidate_csv:
        raise ValueError(f"{pair_id}: missing candidate_csv")

    reference_csv: tuple[Path, ...] = ()
    model_dir: Path | None = None
    source_path = resolve_path(Path(reference_source_value)) if reference_source_value else None
    if ";" in reference_source_value:
        reference_csv = tuple(Path(part.strip()) for part in reference_source_value.split(";") if part.strip())
    elif source_path is not None and (source_path / "metadata.json").is_file():
        model_dir = Path(reference_source_value)
    elif reference_source_value:
        reference_csv = (Path(reference_source_value),)
    else:
        raise ValueError(f"{pair_id}: missing reference_source")

    return PairSpec(
        pair_id=pair_id,
        reference_label=row.get("reference_label", "").strip() or "reference",
        candidate_label=row.get("candidate_label", "").strip() or candidate_csv.stem,
        candidate_csv=candidate_csv,
        reference_csv=reference_csv,
        model_dir=model_dir,
    )


def pairs_from_matrix_csv(path: Path, used_ids: set[str]) -> list[PairSpec]:
    csv_path = resolve_path(path)
    if not csv_path.is_file():
        raise ValueError(f"matrix CSV not found: {path}")
    pairs: list[PairSpec] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no CSV header found")
        for row in reader:
            if row.get("row_status") not in ("", None, "evaluated"):
                continue
            pairs.append(pair_from_matrix_row(row, used_ids))
    return pairs


def discover_pairs(args: argparse.Namespace) -> list[PairSpec]:
    pairs: list[PairSpec] = []
    used_ids: set[str] = set()
    if args.auto_stage4:
        for pair in matrix_runner.auto_stage4_pairs():
            pairs.append(convert_matrix_pair(pair))
            used_ids.add(pair.pair_id)
    if args.matrix_csv is not None:
        pairs.extend(pairs_from_matrix_csv(args.matrix_csv, used_ids))
    return pairs


def validator_namespace(pair: PairSpec, args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        reference_csv=list(pair.reference_csv),
        model_dir=pair.model_dir,
        candidate_csv=pair.candidate_csv,
        output_dir=Path(".unused_stage5_q7_ablation_pair_output"),
        label_reference=pair.reference_label,
        label_candidate=pair.candidate_label,
        q7_margin=args.q7_margin,
        support_margin=args.support_margin,
        q7_column=None,
        dq7_column=None,
        strict=False,
        overwrite=True,
    )


def worst_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    failing = [row for row in rows if not bool(row.get("pass"))]
    if failing:
        return max(failing, key=lambda row: float(row.get("max_violation_amount", 0.0)))
    return max(rows, key=lambda row: float(row.get("max_violation_amount", 0.0)))


def gate_status(gate_mode: str, rows: list[dict[str, Any]]) -> tuple[bool, str, str, str]:
    support_pass = bool(rows) and all(bool(row.get("pass")) for row in rows)
    worst = worst_row(rows)
    worst_dimension = str(worst.get("dimension", "")) if worst else ""
    if support_pass:
        return True, f"pass_{gate_mode}_support_preflight", "", "all checked dimensions within reference support"
    return (
        False,
        f"fail_{gate_mode}_out_of_support",
        worst_dimension,
        f"candidate dimension {worst_dimension} is outside reference support",
    )


def row_for_gate(pair: PairSpec, summary: dict[str, Any], gate_mode: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    q7_ref = summary["reference_stats"]["q7"]
    q7_candidate = summary["candidate_stats"]["q7"]
    support_pass, overall_status, worst_dimension, blocking_reason = gate_status(gate_mode, rows)
    excluded = "" if gate_mode == "14d" else ";".join(EXCLUDED_Q7_DQ7)
    return {
        "pair_id": pair.pair_id,
        "row_status": "evaluated",
        "reference_label": pair.reference_label,
        "candidate_label": pair.candidate_label,
        "reference_source": reference_source(pair),
        "candidate_csv": repo_relative(resolve_path(pair.candidate_csv)),
        "gate_mode": gate_mode,
        "support_pass": support_pass,
        "overall_status": overall_status,
        "worst_dimension": worst_dimension,
        "blocking_reason": blocking_reason,
        "num_dimensions_checked": len(rows),
        "excluded_dimensions": excluded,
        "q7_reference_mean": q7_ref["mean"],
        "q7_candidate_mean": q7_candidate["mean"],
        "q7_reference_range": range_text(q7_ref),
        "q7_candidate_range": range_text(q7_candidate),
    }


def skipped_rows(pair: PairSpec, status: str, message: str) -> list[dict[str, Any]]:
    rows = []
    for gate_mode, count, excluded in (
        ("14d", 0, ""),
        ("12d_without_q7_dq7", 0, ";".join(EXCLUDED_Q7_DQ7)),
    ):
        rows.append(
            {
                "pair_id": pair.pair_id,
                "row_status": "skipped",
                "reference_label": pair.reference_label,
                "candidate_label": pair.candidate_label,
                "reference_source": reference_source(pair),
                "candidate_csv": repo_relative(resolve_path(pair.candidate_csv)),
                "gate_mode": gate_mode,
                "support_pass": "",
                "overall_status": status,
                "worst_dimension": "",
                "blocking_reason": message,
                "num_dimensions_checked": count,
                "excluded_dimensions": excluded,
                "q7_reference_mean": "",
                "q7_candidate_mean": "",
                "q7_reference_range": "",
                "q7_candidate_range": "",
            }
        )
    return rows


def run_pair(pair: PairSpec, args: argparse.Namespace) -> list[dict[str, Any]]:
    try:
        summary = validator.build_summary(validator_namespace(pair, args))
    except validator.ValidationError as exc:
        return skipped_rows(pair, exc.status, exc.message)

    joint_rows = list(summary.get("joint_space_14d_results") or [])
    if not joint_rows:
        return skipped_rows(pair, "fail_missing_14d_columns", "complete 14D joint-space support rows are unavailable")

    excluded = set(EXCLUDED_Q7_DQ7)
    rows_12d = [row for row in joint_rows if row.get("dimension") not in excluded]
    return [
        row_for_gate(pair, summary, "14d", joint_rows),
        row_for_gate(pair, summary, "12d_without_q7_dq7", rows_12d),
    ]


def write_csv_report(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ValueError(f"output exists; pass --overwrite: {repo_relative(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in CSV_FIELDS})


def rows_by_pair(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["pair_id"]), {})[str(row["gate_mode"])] = row
    return grouped


def pair_label(row: dict[str, Any]) -> str:
    return f"`{row['pair_id']}` ({row['reference_label']} -> {row['candidate_label']})"


def render_markdown(rows: list[dict[str, Any]], pairs: list[PairSpec], args: argparse.Namespace) -> str:
    evaluated = [row for row in rows if row["row_status"] == "evaluated"]
    skipped = [row for row in rows if row["row_status"] == "skipped"]
    grouped = rows_by_pair(evaluated)
    fail_14d_pass_12d = []
    fail_both = []
    pass_both = []
    worst_changes = []

    for gate_rows in grouped.values():
        row_14d = gate_rows.get("14d")
        row_12d = gate_rows.get("12d_without_q7_dq7")
        if row_14d is None or row_12d is None:
            continue
        if row_14d["support_pass"] is False and row_12d["support_pass"] is True:
            fail_14d_pass_12d.append((row_14d, row_12d))
        elif row_14d["support_pass"] is False and row_12d["support_pass"] is False:
            fail_both.append((row_14d, row_12d))
        elif row_14d["support_pass"] is True and row_12d["support_pass"] is True:
            pass_both.append((row_14d, row_12d))
        if row_14d.get("worst_dimension") != row_12d.get("worst_dimension"):
            worst_changes.append((row_14d, row_12d))

    worst_12d_counts = Counter(
        str(row["worst_dimension"]) for row in evaluated
        if row["gate_mode"] == "12d_without_q7_dq7" and row.get("worst_dimension")
    )

    lines = [
        "# Stage 5 q7 / joint7 Support-Gate Ablation",
        "",
        f"- generated_utc: `{datetime.now(timezone.utc).isoformat()}`",
        f"- mode: `{args.mode}`",
        f"- support_margin: `{args.support_margin}`",
        f"- q7_margin: `{args.q7_margin}`",
        "",
        "## Purpose",
        "",
        "Compare the existing full 14D joint-space support gate with an offline 12D ablation gate that excludes `joint_pos_7` and `joint_vel_7`.",
        "This does not retrain GP models and does not authorize GP-on.",
        "",
        "## Inputs",
        "",
        f"- pairs_discovered: `{len(pairs)}`",
        f"- evaluated_rows: `{len(evaluated)}`",
        f"- skipped_rows: `{len(skipped)}`",
        "",
        "## 14D vs 12D Results",
        "",
        f"- 14D fail but 12D pass pairs: `{len(fail_14d_pass_12d)}`.",
        f"- 14D fail and 12D still fail pairs: `{len(fail_both)}`.",
        f"- 14D pass and 12D pass pairs: `{len(pass_both)}`.",
        "",
    ]

    if fail_14d_pass_12d:
        lines.append("### 14D Fail, 12D Pass")
        lines.append("")
        for row_14d, row_12d in fail_14d_pass_12d:
            lines.append(
                f"- {pair_label(row_14d)}: 14D worst `{row_14d['worst_dimension']}`, "
                f"12D status `{row_12d['overall_status']}`."
            )
        lines.append("")
    else:
        lines.extend(["### 14D Fail, 12D Pass", "", "- None.", ""])

    if fail_both:
        lines.append("### 14D Fail, 12D Still Fail")
        lines.append("")
        for row_14d, row_12d in fail_both:
            lines.append(
                f"- {pair_label(row_14d)}: 14D worst `{row_14d['worst_dimension']}`, "
                f"12D worst `{row_12d['worst_dimension']}`."
            )
        lines.append("")
    else:
        lines.extend(["### 14D Fail, 12D Still Fail", "", "- None.", ""])

    lines.extend(
        [
            "## Blocker Interpretation",
            "",
            "- If a pair is 14D fail but 12D pass, q7/dq7 materially affected this support-gate decision.",
            "- If a pair is 14D fail and 12D still fail, q7/dq7 is not the only blocker for that pair.",
            "- A 12D pass is only an ablation observation; it is not permission to ignore q7/dq7 for a 14D-trained GP.",
            "",
        ]
    )

    if worst_12d_counts:
        common = ", ".join(f"`{name}` ({count})" for name, count in worst_12d_counts.most_common())
        lines.append(f"- 12D worst-dimension distribution: {common}.")
    else:
        lines.append("- No 12D worst-dimension distribution is available.")
    lines.append("")

    lines.extend(
        [
            "## Worst-Dimension Changes",
            "",
            "| pair | 14D worst | 12D worst | 14D status | 12D status |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    if worst_changes:
        for row_14d, row_12d in worst_changes:
            lines.append(
                f"| `{row_14d['pair_id']}` | `{row_14d['worst_dimension']}` | "
                f"`{row_12d['worst_dimension']}` | `{row_14d['overall_status']}` | "
                f"`{row_12d['overall_status']}` |"
            )
    else:
        lines.append("| | | | | No worst-dimension changes. |")

    lines.extend(
        [
            "",
            "## Pair Table",
            "",
            "| pair | gate | pass | worst dimension | status | q7 reference range | q7 candidate range |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in evaluated:
        lines.append(
            f"| `{row['pair_id']}` | `{row['gate_mode']}` | `{fmt(row['support_pass'])}` | "
            f"`{row['worst_dimension']}` | `{row['overall_status']}` | "
            f"`{row['q7_reference_range']}` | `{row['q7_candidate_range']}` |"
        )
    if not evaluated:
        lines.append("| | | | | No evaluated rows. | | |")

    if skipped:
        lines.extend(["", "## Skipped Rows", ""])
        for row in skipped:
            lines.append(f"- `{row['pair_id']}` `{row['gate_mode']}`: {row['blocking_reason']}")

    lines.extend(
        [
            "",
            "## Safety Notes",
            "",
            f"- {SAFETY_NOTE}",
            "- This does not prove real-robot GP-on tracking improvement.",
            "- This does not authorize GP-on.",
            "- If the original GP input is 14D, real GP support should still respect 14D unless the model is retrained without q7/dq7.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(path: Path, rows: list[dict[str, Any]], pairs: list[PairSpec], args: argparse.Namespace) -> None:
    if path.exists() and not args.overwrite:
        raise ValueError(f"output exists; pass --overwrite: {repo_relative(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(rows, pairs, args), encoding="utf-8")


def run_support_gate(args: argparse.Namespace) -> int:
    pairs = discover_pairs(args)
    if not pairs:
        print("fail_invalid_input: provide --auto-stage4 and/or --matrix-csv", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    for pair in pairs:
        rows.extend(run_pair(pair, args))

    output_dir = resolve_path(args.output_dir)
    csv_path = output_dir / SUPPORT_GATE_CSV
    md_path = output_dir / SUPPORT_GATE_MD
    try:
        write_csv_report(csv_path, rows, args.overwrite)
        write_markdown_report(md_path, rows, pairs, args)
    except ValueError as exc:
        print(f"fail_invalid_input: {exc}", file=sys.stderr)
        return 1

    evaluated = sum(1 for row in rows if row["row_status"] == "evaluated")
    skipped = sum(1 for row in rows if row["row_status"] == "skipped")
    print(f"pairs_total: {len(pairs)}")
    print(f"rows_evaluated: {evaluated}")
    print(f"rows_skipped: {skipped}")
    print(f"summary_csv: {repo_relative(csv_path)}")
    print(f"summary_md: {repo_relative(md_path)}")
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "support-gate":
        return run_support_gate(args)
    print(f"fail_invalid_input: unsupported mode {args.mode}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
