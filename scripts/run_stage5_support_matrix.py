#!/usr/bin/env python3
"""Offline Stage 5A support matrix report.

This script batches scripts/validate_stage5_q7_support.py over reference /
candidate pairs and writes CSV + Markdown summaries. It is offline-only: it
does not import ROS, connect to Franka, launch controllers, or modify runtime
control behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import validate_stage5_q7_support as validator


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("outputs/stage5_support_matrix")
MATRIX_CSV = "stage5_support_matrix.csv"
MATRIX_MD = "stage5_support_matrix.md"
EXIT_CODE_SEMANTICS = "0=preflight pass; 1=invalid input/usage error; 2=valid input but support/preflight fail"

DEFAULT_STAGE4_BASE = Path("data/stage4/cross_traj")
DEFAULT_CROSS_MODEL_DIRS = [
    DEFAULT_STAGE4_BASE / "models/GP_A_planar_train",
    DEFAULT_STAGE4_BASE / "models/GP_B_zmod_train",
]
DEFAULT_LEGACY_MODEL_DIRS = [
    Path("data/stage4/models/GP_matched_strict_no_gp_zmod"),
]
DEFAULT_A_CSV_GLOB = DEFAULT_STAGE4_BASE / "A_no_gp_planar/usable_runs/*.csv"
DEFAULT_B_CSV_GLOB = DEFAULT_STAGE4_BASE / "B_no_gp_zmod/usable_runs/*.csv"
DEFAULT_C_CSV_GLOB = DEFAULT_STAGE4_BASE / "C_no_gp_zmod_heldout/usable_runs/*.csv"

CSV_FIELDS = [
    "pair_id",
    "row_status",
    "reference_label",
    "candidate_label",
    "reference_source",
    "candidate_csv",
    "q7_support_pass",
    "joint_space_14d_available",
    "joint_space_14d_pass",
    "overall_status",
    "blocking_reason",
    "worst_dimension",
    "q7_reference_min",
    "q7_reference_max",
    "q7_reference_mean",
    "q7_reference_std",
    "q7_candidate_min",
    "q7_candidate_max",
    "q7_candidate_mean",
    "q7_candidate_std",
    "validator_exit_code",
    "exit_code_semantics",
    "pair_summary_json",
    "pair_summary_md",
]


class MatrixArgumentParser(argparse.ArgumentParser):
    """Use exit code 1 for invalid CLI usage."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    reference_label: str
    candidate_label: str
    candidate_csv: Path
    reference_csv: tuple[Path, ...] = ()
    model_dir: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = MatrixArgumentParser(
        description=(
            "Offline Stage 5A support matrix report. JSON config support is "
            "limited to a top-level {'pairs': [...]} list with reference_csv "
            "or model_dir plus candidate_csv per pair."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite this script's matrix outputs.")
    parser.add_argument("--strict", action="store_true", help="Pass strict mode through to the q7 support validator.")
    parser.add_argument("--support-margin", type=float, default=0.0, help="14D support margin passed to validator.")
    parser.add_argument("--q7-margin", type=float, default=0.0, help="q7 support margin passed to validator.")
    parser.add_argument(
        "--keep-pair-summaries",
        action="store_true",
        help="Keep per-pair validator JSON/Markdown under output-dir/pairs/<pair_id>/.",
    )
    parser.add_argument("--auto-stage4", action="store_true", help="Discover Stage 4 model dirs and CSV pairs.")
    parser.add_argument(
        "--matrix-config",
        type=Path,
        default=None,
        help=(
            "Optional JSON config with {'pairs': [{'reference_label': str, "
            "'reference_csv': str|list[str] or 'model_dir': str, "
            "'candidate_label': str, 'candidate_csv': str, optional 'pair_id': str}]}."
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    normalized = normalized.strip("_.-")
    return normalized or "pair"


def unique_pair_id(base_id: str, used_ids: set[str]) -> str:
    candidate = safe_id(base_id)
    if candidate not in used_ids:
        used_ids.add(candidate)
        return candidate
    index = 2
    while f"{candidate}_{index}" in used_ids:
        index += 1
    unique = f"{candidate}_{index}"
    used_ids.add(unique)
    return unique


def csv_sort_key(path: Path) -> tuple[str, str]:
    match = re.search(r"(\d{8})_(\d{6})", path.name)
    if match:
        return (match.group(1) + match.group(2), path.name)
    return ("", path.name)


def glob_existing(pattern: Path) -> list[Path]:
    paths = [path.relative_to(REPO_ROOT) for path in REPO_ROOT.glob(str(pattern)) if path.is_file()]
    return sorted(paths, key=csv_sort_key)


def read_model_label(model_dir: Path) -> str:
    metadata_path = resolve_path(model_dir) / "metadata.json"
    if not metadata_path.is_file():
        return model_dir.name
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return model_dir.name
    return str(metadata.get("model_name") or metadata.get("source_mode_name") or model_dir.name)


def discover_model_dirs() -> list[Path]:
    candidates = DEFAULT_CROSS_MODEL_DIRS + DEFAULT_LEGACY_MODEL_DIRS
    discovered = []
    for metadata_path in sorted((REPO_ROOT / "data/stage4").glob("**/metadata.json")):
        discovered.append(metadata_path.parent.relative_to(REPO_ROOT))
    ordered: list[Path] = []
    seen: set[str] = set()
    for path in candidates + discovered:
        resolved = resolve_path(path)
        key = str(resolved)
        if key not in seen and (resolved / "metadata.json").is_file():
            ordered.append(path)
            seen.add(key)
    return ordered


def add_pair(
    pairs: list[PairSpec],
    used_ids: set[str],
    base_id: str,
    reference_label: str,
    candidate_label: str,
    candidate_csv: Path,
    reference_csv: list[Path] | None = None,
    model_dir: Path | None = None,
) -> None:
    pairs.append(
        PairSpec(
            pair_id=unique_pair_id(base_id, used_ids),
            reference_label=reference_label,
            candidate_label=candidate_label,
            candidate_csv=candidate_csv,
            reference_csv=tuple(reference_csv or []),
            model_dir=model_dir,
        )
    )


def auto_stage4_pairs() -> list[PairSpec]:
    pairs: list[PairSpec] = []
    used_ids: set[str] = set()
    model_dirs = discover_model_dirs()
    a_csvs = glob_existing(DEFAULT_A_CSV_GLOB)
    b_csvs = glob_existing(DEFAULT_B_CSV_GLOB)
    c_csvs = glob_existing(DEFAULT_C_CSV_GLOB)

    for model_dir in model_dirs:
        label = read_model_label(model_dir)
        for index, candidate in enumerate(c_csvs, start=1):
            add_pair(
                pairs,
                used_ids,
                f"{label}_vs_C{index}",
                label,
                f"C{index}_heldout_zmod",
                candidate,
                model_dir=model_dir,
            )

    if len(b_csvs) >= 2:
        add_pair(
            pairs,
            used_ids,
            "B1_reference_vs_B2_candidate",
            "B1_zmod_reference",
            "B2_zmod_candidate",
            b_csvs[1],
            reference_csv=[b_csvs[0]],
        )

    if len(b_csvs) >= 2 and c_csvs:
        add_pair(
            pairs,
            used_ids,
            "B2_reference_vs_C1_candidate",
            "B2_zmod_reference",
            "C1_heldout_zmod",
            c_csvs[0],
            reference_csv=[b_csvs[1]],
        )

    if b_csvs and c_csvs:
        for index, candidate in enumerate(c_csvs, start=1):
            add_pair(
                pairs,
                used_ids,
                f"B1_B2_reference_vs_C{index}",
                "B1+B2_zmod_reference",
                f"C{index}_heldout_zmod",
                candidate,
                reference_csv=b_csvs,
            )

    if a_csvs and c_csvs:
        for index, candidate in enumerate(c_csvs, start=1):
            add_pair(
                pairs,
                used_ids,
                f"A1_A2_reference_vs_C{index}",
                "A1+A2_planar_reference",
                f"C{index}_heldout_zmod",
                candidate,
                reference_csv=a_csvs,
            )

    return pairs


def parse_reference_csv(value: Any) -> tuple[Path, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (Path(value),)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(Path(item) for item in value)
    raise ValueError("reference_csv must be a string or list of strings")


def config_pairs(config_path: Path, used_ids: set[str]) -> list[PairSpec]:
    path = resolve_path(config_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"matrix config not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid matrix config JSON: {config_path}: {exc}") from exc

    raw_pairs = payload.get("pairs")
    if not isinstance(raw_pairs, list):
        raise ValueError("matrix config must contain a top-level 'pairs' list")

    pairs: list[PairSpec] = []
    for index, raw_pair in enumerate(raw_pairs, start=1):
        if not isinstance(raw_pair, dict):
            raise ValueError(f"pairs[{index}] must be an object")
        candidate_csv = raw_pair.get("candidate_csv")
        if not isinstance(candidate_csv, str) or not candidate_csv.strip():
            raise ValueError(f"pairs[{index}] requires candidate_csv")
        reference_csv = parse_reference_csv(raw_pair.get("reference_csv"))
        model_dir_value = raw_pair.get("model_dir")
        model_dir = Path(model_dir_value) if isinstance(model_dir_value, str) and model_dir_value.strip() else None
        if not reference_csv and model_dir is None:
            raise ValueError(f"pairs[{index}] requires reference_csv or model_dir")
        default_reference_label = model_dir.name if model_dir else f"reference_{index}"
        reference_label = str(raw_pair.get("reference_label") or default_reference_label)
        candidate_label = str(raw_pair.get("candidate_label") or Path(candidate_csv).stem)
        base_id = str(raw_pair.get("pair_id") or f"{reference_label}_vs_{candidate_label}")
        pairs.append(
            PairSpec(
                pair_id=unique_pair_id(base_id, used_ids),
                reference_label=reference_label,
                candidate_label=candidate_label,
                candidate_csv=Path(candidate_csv),
                reference_csv=reference_csv,
                model_dir=model_dir,
            )
        )
    return pairs


def validator_namespace(pair: PairSpec, args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        reference_csv=list(pair.reference_csv),
        model_dir=pair.model_dir,
        candidate_csv=pair.candidate_csv,
        output_dir=output_dir,
        label_reference=pair.reference_label,
        label_candidate=pair.candidate_label,
        q7_margin=args.q7_margin,
        support_margin=args.support_margin,
        q7_column=None,
        dq7_column=None,
        strict=bool(args.strict),
        overwrite=bool(args.overwrite),
    )


def reference_source(pair: PairSpec) -> str:
    if pair.model_dir is not None:
        return repo_relative(resolve_path(pair.model_dir))
    return ";".join(repo_relative(resolve_path(path)) for path in pair.reference_csv)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def row_from_summary(pair: PairSpec, summary: dict[str, Any], pair_dir: Path | None) -> dict[str, Any]:
    q7_ref = summary["reference_stats"]["q7"]
    q7_candidate = summary["candidate_stats"]["q7"]
    worst = summary.get("worst_dimension") or {}
    overall_status = str(summary["overall_status"])
    row = {
        "pair_id": pair.pair_id,
        "row_status": "evaluated",
        "reference_label": pair.reference_label,
        "candidate_label": pair.candidate_label,
        "reference_source": reference_source(pair),
        "candidate_csv": repo_relative(resolve_path(pair.candidate_csv)),
        "q7_support_pass": summary["q7_support_pass"],
        "joint_space_14d_available": summary["joint_space_14d_available"],
        "joint_space_14d_pass": summary["joint_space_14d_pass"],
        "overall_status": overall_status,
        "blocking_reason": summary["blocking_reason"],
        "worst_dimension": worst.get("dimension", ""),
        "q7_reference_min": q7_ref["min"],
        "q7_reference_max": q7_ref["max"],
        "q7_reference_mean": q7_ref["mean"],
        "q7_reference_std": q7_ref["std"],
        "q7_candidate_min": q7_candidate["min"],
        "q7_candidate_max": q7_candidate["max"],
        "q7_candidate_mean": q7_candidate["mean"],
        "q7_candidate_std": q7_candidate["std"],
        "validator_exit_code": validator.exit_code_for_status(overall_status),
        "exit_code_semantics": EXIT_CODE_SEMANTICS,
        "pair_summary_json": "",
        "pair_summary_md": "",
    }
    if pair_dir is not None:
        row["pair_summary_json"] = repo_relative(pair_dir / validator.SUMMARY_JSON)
        row["pair_summary_md"] = repo_relative(pair_dir / validator.SUMMARY_MD)
    return row


def skipped_row(pair: PairSpec, status: str, message: str, exit_code: int) -> dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "row_status": "skipped",
        "reference_label": pair.reference_label,
        "candidate_label": pair.candidate_label,
        "reference_source": reference_source(pair),
        "candidate_csv": repo_relative(resolve_path(pair.candidate_csv)),
        "q7_support_pass": "",
        "joint_space_14d_available": "",
        "joint_space_14d_pass": "",
        "overall_status": status,
        "blocking_reason": message,
        "worst_dimension": "",
        "q7_reference_min": "",
        "q7_reference_max": "",
        "q7_reference_mean": "",
        "q7_reference_std": "",
        "q7_candidate_min": "",
        "q7_candidate_max": "",
        "q7_candidate_mean": "",
        "q7_candidate_std": "",
        "validator_exit_code": exit_code,
        "exit_code_semantics": EXIT_CODE_SEMANTICS,
        "pair_summary_json": "",
        "pair_summary_md": "",
    }


def write_csv_matrix(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ValueError(f"output exists; pass --overwrite: {repo_relative(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in CSV_FIELDS})


def markdown_bool(value: Any) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def render_inputs(pairs: list[PairSpec]) -> list[str]:
    references = sorted({pair.reference_label + " -> " + reference_source(pair) for pair in pairs})
    candidates = sorted({pair.candidate_label + " -> " + repo_relative(resolve_path(pair.candidate_csv)) for pair in pairs})
    lines = ["## 2. Inputs", ""]
    if not pairs:
        return lines + ["No pairs were discovered or configured.", ""]
    lines.extend(["### References", ""])
    lines.extend(f"- `{item}`" for item in references)
    lines.extend(["", "### Candidates", ""])
    lines.extend(f"- `{item}`" for item in candidates)
    lines.append("")
    return lines


def render_matrix_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## 3. Matrix Summary",
        "",
        "| pair | reference | candidate | q7 pass | 14D pass | worst dimension | overall status | blocking reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| `{pair_id}` | `{reference_label}` | `{candidate_label}` | `{q7}` | `{d14}` | `{worst}` | `{status}` | {reason} |".format(
                pair_id=row["pair_id"],
                reference_label=row["reference_label"],
                candidate_label=row["candidate_label"],
                q7=markdown_bool(row["q7_support_pass"]),
                d14=markdown_bool(row["joint_space_14d_pass"]),
                worst=row.get("worst_dimension", ""),
                status=row["overall_status"],
                reason=str(row["blocking_reason"]).replace("|", "\\|"),
            )
        )
    if not rows:
        lines.append("| | | | | | | | No pairs evaluated. |")
    lines.append("")
    return lines


def render_observations(rows: list[dict[str, Any]]) -> list[str]:
    evaluated = [row for row in rows if row["row_status"] == "evaluated"]
    skipped = [row for row in rows if row["row_status"] == "skipped"]
    q7_pass_14d_fail = [
        row
        for row in evaluated
        if row["q7_support_pass"] is True and row["joint_space_14d_pass"] is False
    ]
    worst_counts = Counter(str(row.get("worst_dimension", "")) for row in evaluated if row.get("worst_dimension"))
    by_reference: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        by_reference[str(row["reference_label"])].append(row)

    lines = ["## 4. Key Observations", ""]
    lines.append(f"- Evaluated pairs: `{len(evaluated)}`; skipped pairs: `{len(skipped)}`.")
    if q7_pass_14d_fail:
        pair_list = ", ".join(f"`{row['pair_id']}`" for row in q7_pass_14d_fail)
        lines.append(f"- q7 passed but 14D support failed in {len(q7_pass_14d_fail)} pair(s): {pair_list}.")
        lines.append("- This supports the Stage 5A caution that q7 alone is insufficient for support gating.")
    else:
        lines.append("- No evaluated pair showed q7 pass with 14D fail in this matrix.")

    if worst_counts:
        common = ", ".join(f"`{name}` ({count})" for name, count in worst_counts.most_common(5))
        lines.append(f"- Most common worst dimensions: {common}.")
    else:
        lines.append("- No worst-dimension distribution is available.")

    coverage_parts = []
    for reference_label, reference_rows in sorted(by_reference.items()):
        pass_count = sum(1 for row in reference_rows if row["joint_space_14d_pass"] is True)
        coverage_parts.append(f"`{reference_label}` {pass_count}/{len(reference_rows)} 14D pass")
    if coverage_parts:
        lines.append("- Reference coverage by 14D pass count: " + "; ".join(coverage_parts) + ".")

    gp_b_rows = [
        row
        for row in evaluated
        if "GP_B_zmod_train" in str(row["reference_label"]) and "heldout" in str(row["candidate_label"]).lower()
    ]
    if gp_b_rows:
        statuses = ", ".join(f"`{row['pair_id']}` -> `{row['overall_status']}`" for row in gp_b_rows)
        lines.append(f"- GP_B_zmod_train vs C held-out status: {statuses}.")

    if skipped:
        skipped_ids = ", ".join(f"`{row['pair_id']}`" for row in skipped)
        lines.append(f"- Skipped pair(s) were reported without fabricated results: {skipped_ids}.")

    lines.append("- These observations are support diagnostics only; they are not GP-on tracking evidence.")
    lines.append("")
    return lines


def render_safety_notes() -> list[str]:
    return [
        "## 5. Safety Notes",
        "",
        "- This is offline support analysis only.",
        "- This does not prove real-robot GP-on tracking improvement.",
        "- This does not authorize GP-on.",
        "- GP-on still requires conservative scale, clip, no online update, and live support gate.",
        "- Do not proceed when q7 or 14D support fails.",
        "",
    ]


def write_markdown_report(path: Path, rows: list[dict[str, Any]], pairs: list[PairSpec], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ValueError(f"output exists; pass --overwrite: {repo_relative(path)}")
    lines = [
        "# Stage 5A Offline Support Matrix Report",
        "",
        f"- generated_utc: `{datetime.now(timezone.utc).isoformat()}`",
        f"- exit_code_semantics: `{EXIT_CODE_SEMANTICS}`",
        "",
        "## 1. Purpose",
        "",
        "This report batches offline q7 and 14D joint-space support diagnostics across reference/candidate pairs.",
        "It is an offline support diagnostic, not GP-on proof.",
        "",
    ]
    lines.extend(render_inputs(pairs))
    lines.extend(render_matrix_table(rows))
    lines.extend(render_observations(rows))
    lines.extend(render_safety_notes())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pair(pair: PairSpec, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    pair_dir = output_dir / "pairs" / pair.pair_id if args.keep_pair_summaries else None
    pair_output_dir = pair_dir or output_dir / ".pair_summaries_disabled"
    try:
        summary = validator.build_summary(validator_namespace(pair, args, pair_output_dir))
        if pair_dir is not None:
            validator.write_reports(summary, pair_dir, args.overwrite)
        return row_from_summary(pair, summary, pair_dir)
    except validator.ValidationError as exc:
        return skipped_row(pair, exc.status, exc.message, validator.exit_code_for_status(exc.status))


def main() -> int:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    used_ids: set[str] = set()
    pairs: list[PairSpec] = []

    if args.auto_stage4:
        for pair in auto_stage4_pairs():
            pairs.append(pair)
            used_ids.add(pair.pair_id)

    if args.matrix_config is not None:
        try:
            pairs.extend(config_pairs(args.matrix_config, used_ids))
        except ValueError as exc:
            print(f"fail_invalid_input: {exc}", file=sys.stderr)
            return 1

    if not pairs:
        print("fail_invalid_input: provide --auto-stage4 and/or --matrix-config", file=sys.stderr)
        return 1

    matrix_csv = output_dir / MATRIX_CSV
    matrix_md = output_dir / MATRIX_MD
    try:
        if not args.overwrite:
            existing = [path for path in (matrix_csv, matrix_md) if path.exists()]
            if existing:
                names = ", ".join(repo_relative(path) for path in existing)
                raise ValueError(f"output exists; pass --overwrite: {names}")
        rows = [run_pair(pair, args, output_dir) for pair in pairs]
        write_csv_matrix(matrix_csv, rows, args.overwrite)
        write_markdown_report(matrix_md, rows, pairs, args.overwrite)
    except ValueError as exc:
        print(f"fail_invalid_input: {exc}", file=sys.stderr)
        return 1

    evaluated = sum(1 for row in rows if row["row_status"] == "evaluated")
    skipped = len(rows) - evaluated
    print(f"matrix_csv: {repo_relative(matrix_csv)}")
    print(f"matrix_md: {repo_relative(matrix_md)}")
    print(f"pairs_total: {len(rows)}")
    print(f"pairs_evaluated: {evaluated}")
    print(f"pairs_skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
