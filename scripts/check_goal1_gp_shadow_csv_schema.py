#!/usr/bin/env python3
"""Offline GOAL1 GP shadow CSV schema and relation checker.

This script has no ROS dependency. It reads saved
cartesian_impedance_controller_data.csv logs, checks Phase 0 debug columns,
checks Phase 1 paper-fusion shadow columns when present, and writes summary
reports for offline validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tarfile
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JOINTS = range(1, 8)
TOLERANCE = 1e-9
DEFAULT_OUTPUT_DIR = Path("outputs/goal1_gp_shadow_csv_schema_check")
CSV_BASENAME = "cartesian_impedance_controller_data.csv"
EXPECTED_SOURCE_CODES = {0, 1, 2, 3}
EXPECTED_BOOL_CODES = {0, 1}


PHASE0_SCALAR_COLUMNS = [
    "gp_prediction_enabled",
    "gp_online_update_enabled",
    "gp_compensation_enabled",
    "gp_compensation_source_code",
    "gp_compensation_scale",
    "gp_compensation_clip_nm",
]

PHASE0_JOINT_PREFIXES = [
    "tau_nominal",
    "tau_final",
    "gp_selected_raw",
    "gp_scaled",
    "gp_applied",
    "gp_clip_active",
]

PHASE1_SCALAR_COLUMNS = [
    "gp_shadow_paper_fusion_logging_enabled",
    "gp_historical_shadow_enabled",
    "gp_historical_source_mode_code",
    "gp_shadow_paper_formula_available",
    "gp_shadow_historical_available",
    "gp_shadow_variance_eps",
    "gp_shadow_hist_fallback_variance",
]

PHASE1_JOINT_PREFIXES = [
    "gp_shadow_local_raw",
    "gp_shadow_cloud_raw",
    "gp_shadow_hist_raw",
    "gp_shadow_combined_paper_raw",
    "gp_shadow_var_local",
    "gp_shadow_var_cloud",
    "gp_shadow_var_hist",
    "gp_shadow_weight_local",
    "gp_shadow_weight_cloud",
    "gp_shadow_weight_hist",
    "gp_shadow_paper_scaled",
    "gp_shadow_paper_clip_proxy_applied",
    "gp_shadow_paper_clip_proxy_active",
]

OPTIONAL_PRECISION_PREFIXES = [
    "gp_shadow_precision_local",
    "gp_shadow_precision_cloud",
    "gp_shadow_precision_hist",
]

SUMMARY_FIELDS = [
    "csv_path",
    "run_name",
    "rows",
    "columns",
    "row_length_mismatch_count",
    "duplicate_header_count",
    "phase0_schema_ok",
    "phase1_shadow_schema_ok",
    "finite_ok",
    "phase0_numeric_ok",
    "phase1_shadow_numeric_ok",
    "overall_status",
    "missing_phase0_columns",
    "missing_phase1_columns",
    "missing_optional_precision_columns",
    "nonfinite_counts_by_column",
    "unique_gp_compensation_source_code",
    "unique_gp_compensation_enabled",
    "unique_gp_prediction_enabled",
    "unique_gp_online_update_enabled",
    "max_abs_tau_relation_error",
    "max_abs_scale_relation_error",
    "max_abs_clip_relation_error",
    "max_abs_clip_active_flag_error",
    "clip_active_count_total",
    "clip_active_count_by_joint",
    "compensation_disabled_row_count",
    "max_abs_disabled_selected_raw",
    "max_abs_disabled_scaled",
    "max_abs_disabled_applied",
    "max_abs_disabled_clip_active",
    "max_abs_disabled_tau_relation_error",
    "max_abs_shadow_weight_sum_error",
    "shadow_weight_min",
    "shadow_weight_max",
    "shadow_negative_weight_count",
    "shadow_weight_gt_one_count",
    "shadow_disabled_zero_weight_sum_count",
    "shadow_disabled_nonzero_weight_sum_count",
    "max_abs_hist_weight_when_unavailable",
    "max_abs_hist_raw_when_unavailable",
    "max_abs_local_cloud_weight_sum_when_hist_unavailable",
    "max_abs_shadow_formula_error",
    "max_abs_shadow_scale_error",
    "max_abs_shadow_clip_proxy_error",
    "max_abs_shadow_clip_proxy_active_flag_error",
    "shadow_clip_proxy_active_count_total",
    "shadow_clip_proxy_active_count_by_joint",
    "max_abs_tau_vs_actual_gp_applied_error",
    "max_abs_tau_vs_shadow_proxy_error",
    "shadow_proxy_equals_gp_applied_count",
    "shadow_proxy_equals_gp_applied_ratio",
    "warnings",
    "failures",
    "notes",
]


@dataclass(frozen=True)
class InputCsv:
    path: Path
    label: str


@dataclass
class LoadedCsv:
    input_csv: InputCsv
    columns: list[str]
    rows: list[dict[str, str]]
    row_length_mismatch_count: int
    duplicate_header_count: int
    duplicate_headers: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline checker for GOAL1 Phase 0 and Phase 1 GP shadow CSV logs.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_paths",
        action="append",
        type=Path,
        default=[],
        help="Path to one cartesian_impedance_controller_data.csv file. May be repeated.",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dirs",
        action="append",
        type=Path,
        default=[],
        help="Directory to recursively search for cartesian_impedance_controller_data.csv. May be repeated.",
    )
    parser.add_argument(
        "--archive",
        dest="archives",
        action="append",
        type=Path,
        default=[],
        help="Read-only .tar.gz/.tgz archive to extract into a temporary directory. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when missing columns or numeric relation failures are found.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the Markdown summary only; do not write output files.",
    )
    return parser.parse_args()


def joint_columns(prefixes: list[str]) -> list[str]:
    return [f"{prefix}_{joint}" for prefix in prefixes for joint in JOINTS]


def phase0_required_columns() -> list[str]:
    return PHASE0_SCALAR_COLUMNS + joint_columns(PHASE0_JOINT_PREFIXES)


def phase1_required_columns() -> list[str]:
    return PHASE1_SCALAR_COLUMNS + joint_columns(PHASE1_JOINT_PREFIXES)


def optional_precision_columns() -> list[str]:
    return joint_columns(OPTIONAL_PRECISION_PREFIXES)


def find_csv_files(root: Path) -> list[InputCsv]:
    if not root.exists():
        raise FileNotFoundError(f"data directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"data-dir is not a directory: {root}")
    return [InputCsv(path=path, label=str(path)) for path in sorted(root.rglob(CSV_BASENAME))]


def safe_extract_archive(archive_path: Path, target_dir: Path) -> None:
    if not archive_path.exists():
        raise FileNotFoundError(f"archive does not exist: {archive_path}")
    if not archive_path.is_file():
        raise ValueError(f"archive path is not a file: {archive_path}")
    if not tarfile.is_tarfile(archive_path):
        raise ValueError(f"archive is not a tar file: {archive_path}")

    target_root = target_dir.resolve()
    with tarfile.open(archive_path, "r:*") as tar:
        members = tar.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise ValueError(f"archive contains a link member, refusing to extract: {member.name}")
            destination = (target_root / member.name).resolve()
            if not destination.is_relative_to(target_root):
                raise ValueError(f"archive member escapes extraction directory: {member.name}")
        tar.extractall(target_root, members=members)


def find_archive_csv_files(archive_path: Path, extract_dir: Path) -> list[InputCsv]:
    safe_extract_archive(archive_path, extract_dir)
    result = []
    for path in sorted(extract_dir.rglob(CSV_BASENAME)):
        label = f"{archive_path}::{path.relative_to(extract_dir)}"
        result.append(InputCsv(path=path, label=label))
    return result


def duplicate_names(columns: list[str]) -> list[str]:
    seen = set()
    duplicates = []
    for column in columns:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)
    return duplicates


def load_csv(input_csv: InputCsv) -> LoadedCsv:
    if not input_csv.path.exists():
        raise FileNotFoundError(f"CSV does not exist: {input_csv.path}")
    if not input_csv.path.is_file():
        raise ValueError(f"CSV path is not a file: {input_csv.path}")

    with input_csv.path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV has no header: {input_csv.label}") from exc

        rows = []
        row_length_mismatch_count = 0
        for raw_row in reader:
            if len(raw_row) != len(columns):
                row_length_mismatch_count += 1
            padded = raw_row + [""] * max(0, len(columns) - len(raw_row))
            rows.append(dict(zip(columns, padded[: len(columns)])))

    if not rows:
        raise ValueError(f"CSV has no data rows: {input_csv.label}")

    duplicates = duplicate_names(columns)
    return LoadedCsv(
        input_csv=input_csv,
        columns=columns,
        rows=rows,
        row_length_mismatch_count=row_length_mismatch_count,
        duplicate_header_count=len(duplicates),
        duplicate_headers=duplicates,
    )


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


def finite(value: float) -> bool:
    return math.isfinite(value)


def boolish_is_one(value: float) -> bool:
    return finite(value) and abs(value - 1.0) <= TOLERANCE


def boolish_is_zero(value: float) -> bool:
    return finite(value) and abs(value) <= TOLERANCE


def clip(value: float, limit: float) -> float:
    if not finite(value) or not finite(limit):
        return math.nan
    abs_limit = abs(limit)
    return min(max(value, -abs_limit), abs_limit)


def max_abs(current: float | None, value: float) -> float:
    if not finite(value):
        return current if current is not None else math.nan
    if current is None or not finite(current):
        return abs(value)
    return max(current, abs(value))


def format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.12g}"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def csv_cell(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json_dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format_float(value)
    if value is None:
        return ""
    return str(value)


def int_like_uniques(rows: list[dict[str, str]], column: str) -> list[int | str]:
    values: set[int | str] = set()
    for row in rows:
        value = parse_float(row.get(column))
        if not finite(value):
            values.add("nonfinite")
            continue
        rounded = int(round(value))
        if abs(value - rounded) <= TOLERANCE:
            values.add(rounded)
        else:
            values.add(format_float(value))
    return sorted(values, key=lambda item: str(item))


def check_required_columns(columns: list[str]) -> dict[str, Any]:
    column_set = set(columns)
    phase0_missing = [column for column in phase0_required_columns() if column not in column_set]
    phase1_missing = [column for column in phase1_required_columns() if column not in column_set]
    optional_missing = [column for column in optional_precision_columns() if column not in column_set]
    phase1_present_count = len(phase1_required_columns()) - len(phase1_missing)
    optional_present = [column for column in optional_precision_columns() if column in column_set]
    return {
        "phase0_missing": phase0_missing,
        "phase1_missing": phase1_missing,
        "optional_missing": optional_missing,
        "optional_present": optional_present,
        "phase0_schema_ok": not phase0_missing,
        "phase1_shadow_schema_ok": not phase1_missing,
        "phase1_present_count": phase1_present_count,
    }


def finite_check(rows: list[dict[str, str]], columns: list[str]) -> tuple[bool, dict[str, int]]:
    nonfinite_counts = {}
    for column in columns:
        bad_count = 0
        for row in rows:
            if not finite(parse_float(row.get(column))):
                bad_count += 1
        if bad_count:
            nonfinite_counts[column] = bad_count
    return not nonfinite_counts, nonfinite_counts


def valid_relation_values(*values: float) -> bool:
    return all(finite(value) for value in values)


def check_source_codes(rows: list[dict[str, str]], strict: bool) -> tuple[dict[str, list[int | str]], list[str], list[str]]:
    unique_values = {
        "gp_compensation_source_code": int_like_uniques(rows, "gp_compensation_source_code"),
        "gp_compensation_enabled": int_like_uniques(rows, "gp_compensation_enabled"),
        "gp_prediction_enabled": int_like_uniques(rows, "gp_prediction_enabled"),
        "gp_online_update_enabled": int_like_uniques(rows, "gp_online_update_enabled"),
    }
    warnings = []
    failures = []

    source_values = [value for value in unique_values["gp_compensation_source_code"] if isinstance(value, int)]
    unknown_source_codes = sorted(set(source_values) - EXPECTED_SOURCE_CODES)
    if unknown_source_codes:
        message = f"unexpected gp_compensation_source_code values: {unknown_source_codes}"
        (failures if strict else warnings).append(message)

    for column in ("gp_compensation_enabled", "gp_prediction_enabled", "gp_online_update_enabled"):
        bool_values = [value for value in unique_values[column] if isinstance(value, int)]
        unexpected = sorted(set(bool_values) - EXPECTED_BOOL_CODES)
        if unexpected or any(not isinstance(value, int) for value in unique_values[column]):
            message = f"unexpected boolean-like values in {column}: {unique_values[column]}"
            (failures if strict else warnings).append(message)

    return unique_values, warnings, failures


def check_phase0_numeric_relations(rows: list[dict[str, str]], strict: bool) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "max_abs_tau_relation_error": math.nan,
        "max_abs_scale_relation_error": math.nan,
        "max_abs_clip_relation_error": math.nan,
        "max_abs_clip_active_flag_error": math.nan,
        "clip_active_count_total": 0,
        "clip_active_count_by_joint": {joint: 0 for joint in JOINTS},
        "compensation_disabled_row_count": 0,
        "max_abs_disabled_selected_raw": math.nan,
        "max_abs_disabled_scaled": math.nan,
        "max_abs_disabled_applied": math.nan,
        "max_abs_disabled_clip_active": math.nan,
        "max_abs_disabled_tau_relation_error": math.nan,
        "phase0_numeric_ok": True,
        "warnings": [],
        "failures": [],
    }

    source_uniques, source_warnings, source_failures = check_source_codes(rows, strict)
    metrics["unique_values"] = source_uniques
    metrics["warnings"].extend(source_warnings)
    metrics["failures"].extend(source_failures)

    disabled_seen_rows: set[int] = set()
    for row_index, row in enumerate(rows):
        scale = parse_float(row.get("gp_compensation_scale"))
        clip_nm = parse_float(row.get("gp_compensation_clip_nm"))
        compensation_enabled = parse_float(row.get("gp_compensation_enabled"))
        disabled = boolish_is_zero(compensation_enabled)
        if disabled:
            disabled_seen_rows.add(row_index)

        for joint in JOINTS:
            tau_nominal = parse_float(row.get(f"tau_nominal_{joint}"))
            tau_final = parse_float(row.get(f"tau_final_{joint}"))
            selected_raw = parse_float(row.get(f"gp_selected_raw_{joint}"))
            scaled = parse_float(row.get(f"gp_scaled_{joint}"))
            applied = parse_float(row.get(f"gp_applied_{joint}"))
            clip_active = parse_float(row.get(f"gp_clip_active_{joint}"))

            if valid_relation_values(tau_nominal, tau_final, applied):
                error = tau_final - (tau_nominal - applied)
                metrics["max_abs_tau_relation_error"] = max_abs(metrics["max_abs_tau_relation_error"], error)

            if valid_relation_values(scaled, scale, selected_raw):
                error = scaled - scale * selected_raw
                metrics["max_abs_scale_relation_error"] = max_abs(metrics["max_abs_scale_relation_error"], error)

            if valid_relation_values(applied, scaled, clip_nm):
                clipped = clip(scaled, clip_nm)
                error = applied - clipped
                metrics["max_abs_clip_relation_error"] = max_abs(metrics["max_abs_clip_relation_error"], error)
                expected_active = 1.0 if abs(clipped - scaled) > TOLERANCE else 0.0
                if valid_relation_values(clip_active):
                    active_error = clip_active - expected_active
                    metrics["max_abs_clip_active_flag_error"] = max_abs(
                        metrics["max_abs_clip_active_flag_error"],
                        active_error,
                    )
                    if boolish_is_one(clip_active):
                        metrics["clip_active_count_total"] += 1
                        metrics["clip_active_count_by_joint"][joint] += 1

            if disabled:
                if finite(selected_raw):
                    metrics["max_abs_disabled_selected_raw"] = max_abs(
                        metrics["max_abs_disabled_selected_raw"],
                        selected_raw,
                    )
                if finite(scaled):
                    metrics["max_abs_disabled_scaled"] = max_abs(metrics["max_abs_disabled_scaled"], scaled)
                if finite(applied):
                    metrics["max_abs_disabled_applied"] = max_abs(metrics["max_abs_disabled_applied"], applied)
                if finite(clip_active):
                    metrics["max_abs_disabled_clip_active"] = max_abs(
                        metrics["max_abs_disabled_clip_active"],
                        clip_active,
                    )
                if valid_relation_values(tau_nominal, tau_final):
                    metrics["max_abs_disabled_tau_relation_error"] = max_abs(
                        metrics["max_abs_disabled_tau_relation_error"],
                        tau_final - tau_nominal,
                    )

    metrics["compensation_disabled_row_count"] = len(disabled_seen_rows)

    relation_limits = [
        "max_abs_tau_relation_error",
        "max_abs_scale_relation_error",
        "max_abs_clip_relation_error",
        "max_abs_clip_active_flag_error",
        "max_abs_disabled_selected_raw",
        "max_abs_disabled_scaled",
        "max_abs_disabled_applied",
        "max_abs_disabled_clip_active",
        "max_abs_disabled_tau_relation_error",
    ]
    for key in relation_limits:
        value = metrics[key]
        if finite(value) and value > TOLERANCE:
            metrics["failures"].append(f"{key} exceeds tolerance: {format_float(value)}")

    if metrics["failures"]:
        metrics["phase0_numeric_ok"] = False
    return metrics


def check_phase1_shadow_relations(rows: list[dict[str, str]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "max_abs_shadow_weight_sum_error": math.nan,
        "shadow_weight_min": math.nan,
        "shadow_weight_max": math.nan,
        "shadow_negative_weight_count": 0,
        "shadow_weight_gt_one_count": 0,
        "shadow_disabled_zero_weight_sum_count": 0,
        "shadow_disabled_nonzero_weight_sum_count": 0,
        "max_abs_hist_weight_when_unavailable": math.nan,
        "max_abs_hist_raw_when_unavailable": math.nan,
        "max_abs_local_cloud_weight_sum_when_hist_unavailable": math.nan,
        "max_abs_shadow_formula_error": math.nan,
        "max_abs_shadow_scale_error": math.nan,
        "max_abs_shadow_clip_proxy_error": math.nan,
        "max_abs_shadow_clip_proxy_active_flag_error": math.nan,
        "shadow_clip_proxy_active_count_total": 0,
        "shadow_clip_proxy_active_count_by_joint": {joint: 0 for joint in JOINTS},
        "max_abs_tau_vs_actual_gp_applied_error": math.nan,
        "max_abs_tau_vs_shadow_proxy_error": math.nan,
        "shadow_proxy_equals_gp_applied_count": 0,
        "shadow_proxy_equals_gp_applied_ratio": math.nan,
        "phase1_shadow_numeric_ok": True,
        "warnings": [],
        "failures": [],
        "notes": [],
    }

    proxy_compare_total = 0
    for row in rows:
        scale = parse_float(row.get("gp_compensation_scale"))
        clip_nm = parse_float(row.get("gp_compensation_clip_nm"))
        shadow_enabled = boolish_is_one(parse_float(row.get("gp_shadow_paper_fusion_logging_enabled")))
        hist_available = boolish_is_one(parse_float(row.get("gp_shadow_historical_available")))

        for joint in JOINTS:
            local_raw = parse_float(row.get(f"gp_shadow_local_raw_{joint}"))
            cloud_raw = parse_float(row.get(f"gp_shadow_cloud_raw_{joint}"))
            hist_raw = parse_float(row.get(f"gp_shadow_hist_raw_{joint}"))
            combined_raw = parse_float(row.get(f"gp_shadow_combined_paper_raw_{joint}"))
            weight_local = parse_float(row.get(f"gp_shadow_weight_local_{joint}"))
            weight_cloud = parse_float(row.get(f"gp_shadow_weight_cloud_{joint}"))
            weight_hist = parse_float(row.get(f"gp_shadow_weight_hist_{joint}"))
            shadow_scaled = parse_float(row.get(f"gp_shadow_paper_scaled_{joint}"))
            proxy_applied = parse_float(row.get(f"gp_shadow_paper_clip_proxy_applied_{joint}"))
            proxy_active = parse_float(row.get(f"gp_shadow_paper_clip_proxy_active_{joint}"))
            tau_nominal = parse_float(row.get(f"tau_nominal_{joint}"))
            tau_final = parse_float(row.get(f"tau_final_{joint}"))
            gp_applied = parse_float(row.get(f"gp_applied_{joint}"))

            weights = [weight_local, weight_cloud, weight_hist]
            finite_weights = [weight for weight in weights if finite(weight)]
            if finite_weights:
                current_min = min(finite_weights)
                current_max = max(finite_weights)
                metrics["shadow_weight_min"] = current_min if not finite(metrics["shadow_weight_min"]) else min(
                    metrics["shadow_weight_min"],
                    current_min,
                )
                metrics["shadow_weight_max"] = current_max if not finite(metrics["shadow_weight_max"]) else max(
                    metrics["shadow_weight_max"],
                    current_max,
                )
                metrics["shadow_negative_weight_count"] += sum(1 for weight in finite_weights if weight < -TOLERANCE)
                metrics["shadow_weight_gt_one_count"] += sum(
                    1 for weight in finite_weights if weight > 1.0 + TOLERANCE
                )

            if valid_relation_values(weight_local, weight_cloud, weight_hist):
                weight_sum = weight_local + weight_cloud + weight_hist
                if shadow_enabled:
                    metrics["max_abs_shadow_weight_sum_error"] = max_abs(
                        metrics["max_abs_shadow_weight_sum_error"],
                        weight_sum - 1.0,
                    )
                elif abs(weight_sum) <= TOLERANCE:
                    metrics["shadow_disabled_zero_weight_sum_count"] += 1
                else:
                    metrics["shadow_disabled_nonzero_weight_sum_count"] += 1

                if not hist_available:
                    metrics["max_abs_hist_weight_when_unavailable"] = max_abs(
                        metrics["max_abs_hist_weight_when_unavailable"],
                        weight_hist,
                    )
                    if shadow_enabled:
                        metrics["max_abs_local_cloud_weight_sum_when_hist_unavailable"] = max_abs(
                            metrics["max_abs_local_cloud_weight_sum_when_hist_unavailable"],
                            weight_local + weight_cloud - 1.0,
                        )

            if not hist_available and finite(hist_raw):
                metrics["max_abs_hist_raw_when_unavailable"] = max_abs(
                    metrics["max_abs_hist_raw_when_unavailable"],
                    hist_raw,
                )

            if valid_relation_values(combined_raw, weight_local, local_raw, weight_cloud, cloud_raw, weight_hist, hist_raw):
                expected = weight_local * local_raw + weight_cloud * cloud_raw + weight_hist * hist_raw
                metrics["max_abs_shadow_formula_error"] = max_abs(
                    metrics["max_abs_shadow_formula_error"],
                    combined_raw - expected,
                )

            if valid_relation_values(shadow_scaled, scale, combined_raw):
                metrics["max_abs_shadow_scale_error"] = max_abs(
                    metrics["max_abs_shadow_scale_error"],
                    shadow_scaled - scale * combined_raw,
                )

            if valid_relation_values(proxy_applied, shadow_scaled, clip_nm):
                proxy_expected = clip(shadow_scaled, clip_nm)
                metrics["max_abs_shadow_clip_proxy_error"] = max_abs(
                    metrics["max_abs_shadow_clip_proxy_error"],
                    proxy_applied - proxy_expected,
                )
                expected_active = 1.0 if abs(proxy_expected - shadow_scaled) > TOLERANCE else 0.0
                if finite(proxy_active):
                    metrics["max_abs_shadow_clip_proxy_active_flag_error"] = max_abs(
                        metrics["max_abs_shadow_clip_proxy_active_flag_error"],
                        proxy_active - expected_active,
                    )
                    if boolish_is_one(proxy_active):
                        metrics["shadow_clip_proxy_active_count_total"] += 1
                        metrics["shadow_clip_proxy_active_count_by_joint"][joint] += 1

            if valid_relation_values(tau_nominal, tau_final, gp_applied):
                metrics["max_abs_tau_vs_actual_gp_applied_error"] = max_abs(
                    metrics["max_abs_tau_vs_actual_gp_applied_error"],
                    tau_final - (tau_nominal - gp_applied),
                )
            if valid_relation_values(tau_nominal, tau_final, proxy_applied):
                metrics["max_abs_tau_vs_shadow_proxy_error"] = max_abs(
                    metrics["max_abs_tau_vs_shadow_proxy_error"],
                    tau_final - (tau_nominal - proxy_applied),
                )
            if valid_relation_values(proxy_applied, gp_applied):
                proxy_compare_total += 1
                if abs(proxy_applied - gp_applied) <= TOLERANCE:
                    metrics["shadow_proxy_equals_gp_applied_count"] += 1

    if proxy_compare_total:
        metrics["shadow_proxy_equals_gp_applied_ratio"] = (
            metrics["shadow_proxy_equals_gp_applied_count"] / proxy_compare_total
        )
        if metrics["shadow_proxy_equals_gp_applied_ratio"] >= 0.9:
            metrics["notes"].append(
                "shadow proxy equals gp_applied in at least 90% of comparable samples; this is informational only",
            )

    relation_limits = [
        "max_abs_shadow_weight_sum_error",
        "max_abs_hist_weight_when_unavailable",
        "max_abs_hist_raw_when_unavailable",
        "max_abs_local_cloud_weight_sum_when_hist_unavailable",
        "max_abs_shadow_formula_error",
        "max_abs_shadow_scale_error",
        "max_abs_shadow_clip_proxy_error",
        "max_abs_shadow_clip_proxy_active_flag_error",
        "max_abs_tau_vs_actual_gp_applied_error",
    ]
    for key in relation_limits:
        value = metrics[key]
        if finite(value) and value > TOLERANCE:
            metrics["failures"].append(f"{key} exceeds tolerance: {format_float(value)}")
    if metrics["shadow_negative_weight_count"]:
        metrics["failures"].append(f"negative shadow weights: {metrics['shadow_negative_weight_count']}")
    if metrics["shadow_weight_gt_one_count"]:
        metrics["failures"].append(f"shadow weights > 1+tolerance: {metrics['shadow_weight_gt_one_count']}")

    if metrics["failures"]:
        metrics["phase1_shadow_numeric_ok"] = False
    return metrics


def base_summary(loaded: LoadedCsv) -> dict[str, Any]:
    return {
        "csv_path": loaded.input_csv.label,
        "run_name": loaded.input_csv.path.parent.name,
        "rows": len(loaded.rows),
        "columns": len(loaded.columns),
        "row_length_mismatch_count": loaded.row_length_mismatch_count,
        "duplicate_header_count": loaded.duplicate_header_count,
        "warnings": [],
        "failures": [],
        "notes": [],
    }


def check_loaded_csv(loaded: LoadedCsv, strict: bool) -> dict[str, Any]:
    summary = base_summary(loaded)
    column_checks = check_required_columns(loaded.columns)
    summary.update(
        {
            "phase0_schema_ok": column_checks["phase0_schema_ok"],
            "phase1_shadow_schema_ok": column_checks["phase1_shadow_schema_ok"],
            "missing_phase0_columns": column_checks["phase0_missing"],
            "missing_phase1_columns": column_checks["phase1_missing"],
            "missing_optional_precision_columns": column_checks["optional_missing"],
        }
    )

    if loaded.row_length_mismatch_count:
        summary["warnings"].append(f"row length mismatch count: {loaded.row_length_mismatch_count}")
    if loaded.duplicate_header_count:
        summary["failures"].append(f"duplicate CSV header names: {loaded.duplicate_headers}")

    if column_checks["phase0_missing"]:
        summary["failures"].append(f"missing Phase 0 columns: {column_checks['phase0_missing']}")

    if column_checks["phase1_missing"]:
        if column_checks["phase1_present_count"] == 0:
            summary["warnings"].append("missing Phase 1 shadow columns; compatible with old Phase 0 archive CSV")
        else:
            summary["failures"].append(f"partial Phase 1 shadow columns missing: {column_checks['phase1_missing']}")

    optional_present = column_checks["optional_present"]
    if optional_present and column_checks["optional_missing"]:
        summary["warnings"].append("optional precision columns are partially present; present columns will be validated")

    numeric_columns = []
    if column_checks["phase0_schema_ok"]:
        numeric_columns.extend(phase0_required_columns())
    if column_checks["phase1_shadow_schema_ok"]:
        numeric_columns.extend(phase1_required_columns())
    numeric_columns.extend(optional_present)
    finite_ok, nonfinite_counts = finite_check(loaded.rows, numeric_columns)
    summary["finite_ok"] = finite_ok
    summary["nonfinite_counts_by_column"] = nonfinite_counts
    if nonfinite_counts:
        summary["failures"].append(f"non-finite numeric values found: {nonfinite_counts}")

    summary["phase0_numeric_ok"] = False
    if column_checks["phase0_schema_ok"]:
        phase0 = check_phase0_numeric_relations(loaded.rows, strict)
        summary.update(
            {
                "phase0_numeric_ok": phase0["phase0_numeric_ok"],
                "unique_gp_compensation_source_code": phase0["unique_values"]["gp_compensation_source_code"],
                "unique_gp_compensation_enabled": phase0["unique_values"]["gp_compensation_enabled"],
                "unique_gp_prediction_enabled": phase0["unique_values"]["gp_prediction_enabled"],
                "unique_gp_online_update_enabled": phase0["unique_values"]["gp_online_update_enabled"],
                "max_abs_tau_relation_error": phase0["max_abs_tau_relation_error"],
                "max_abs_scale_relation_error": phase0["max_abs_scale_relation_error"],
                "max_abs_clip_relation_error": phase0["max_abs_clip_relation_error"],
                "max_abs_clip_active_flag_error": phase0["max_abs_clip_active_flag_error"],
                "clip_active_count_total": phase0["clip_active_count_total"],
                "clip_active_count_by_joint": phase0["clip_active_count_by_joint"],
                "compensation_disabled_row_count": phase0["compensation_disabled_row_count"],
                "max_abs_disabled_selected_raw": phase0["max_abs_disabled_selected_raw"],
                "max_abs_disabled_scaled": phase0["max_abs_disabled_scaled"],
                "max_abs_disabled_applied": phase0["max_abs_disabled_applied"],
                "max_abs_disabled_clip_active": phase0["max_abs_disabled_clip_active"],
                "max_abs_disabled_tau_relation_error": phase0["max_abs_disabled_tau_relation_error"],
            }
        )
        summary["warnings"].extend(phase0["warnings"])
        summary["failures"].extend(phase0["failures"])
    else:
        summary["warnings"].append("Phase 0 numeric checks skipped because required columns are missing")

    summary["phase1_shadow_numeric_ok"] = False
    if column_checks["phase1_shadow_schema_ok"]:
        phase1 = check_phase1_shadow_relations(loaded.rows)
        summary.update(
            {
                "phase1_shadow_numeric_ok": phase1["phase1_shadow_numeric_ok"],
                "max_abs_shadow_weight_sum_error": phase1["max_abs_shadow_weight_sum_error"],
                "shadow_weight_min": phase1["shadow_weight_min"],
                "shadow_weight_max": phase1["shadow_weight_max"],
                "shadow_negative_weight_count": phase1["shadow_negative_weight_count"],
                "shadow_weight_gt_one_count": phase1["shadow_weight_gt_one_count"],
                "shadow_disabled_zero_weight_sum_count": phase1["shadow_disabled_zero_weight_sum_count"],
                "shadow_disabled_nonzero_weight_sum_count": phase1["shadow_disabled_nonzero_weight_sum_count"],
                "max_abs_hist_weight_when_unavailable": phase1["max_abs_hist_weight_when_unavailable"],
                "max_abs_hist_raw_when_unavailable": phase1["max_abs_hist_raw_when_unavailable"],
                "max_abs_local_cloud_weight_sum_when_hist_unavailable": phase1[
                    "max_abs_local_cloud_weight_sum_when_hist_unavailable"
                ],
                "max_abs_shadow_formula_error": phase1["max_abs_shadow_formula_error"],
                "max_abs_shadow_scale_error": phase1["max_abs_shadow_scale_error"],
                "max_abs_shadow_clip_proxy_error": phase1["max_abs_shadow_clip_proxy_error"],
                "max_abs_shadow_clip_proxy_active_flag_error": phase1[
                    "max_abs_shadow_clip_proxy_active_flag_error"
                ],
                "shadow_clip_proxy_active_count_total": phase1["shadow_clip_proxy_active_count_total"],
                "shadow_clip_proxy_active_count_by_joint": phase1["shadow_clip_proxy_active_count_by_joint"],
                "max_abs_tau_vs_actual_gp_applied_error": phase1["max_abs_tau_vs_actual_gp_applied_error"],
                "max_abs_tau_vs_shadow_proxy_error": phase1["max_abs_tau_vs_shadow_proxy_error"],
                "shadow_proxy_equals_gp_applied_count": phase1["shadow_proxy_equals_gp_applied_count"],
                "shadow_proxy_equals_gp_applied_ratio": phase1["shadow_proxy_equals_gp_applied_ratio"],
            }
        )
        summary["warnings"].extend(phase1["warnings"])
        summary["failures"].extend(phase1["failures"])
        summary["notes"].extend(phase1["notes"])
    else:
        summary["warnings"].append("Phase 1 shadow numeric checks skipped")

    if summary["failures"]:
        summary["overall_status"] = "FAIL"
    elif summary["warnings"]:
        summary["overall_status"] = "WARN"
    else:
        summary["overall_status"] = "PASS"

    return summary


def error_summary(input_csv: InputCsv, error: Exception) -> dict[str, Any]:
    summary = {field: "" for field in SUMMARY_FIELDS}
    summary.update(
        {
            "csv_path": input_csv.label,
            "run_name": input_csv.path.parent.name,
            "rows": 0,
            "columns": 0,
            "phase0_schema_ok": False,
            "phase1_shadow_schema_ok": False,
            "finite_ok": False,
            "phase0_numeric_ok": False,
            "phase1_shadow_numeric_ok": False,
            "overall_status": "ERROR",
            "warnings": [],
            "failures": [str(error)],
            "notes": [],
        }
    )
    return summary


def render_markdown_report(summaries: list[dict[str, Any]], strict: bool) -> str:
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "ERROR": 0}
    for summary in summaries:
        counts[summary.get("overall_status", "ERROR")] = counts.get(summary.get("overall_status", "ERROR"), 0) + 1

    lines = [
        "# GOAL1 GP shadow CSV schema check",
        "",
        f"- generated_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- strict: {str(strict).lower()}",
        f"- csv_count: {len(summaries)}",
        f"- pass: {counts.get('PASS', 0)}",
        f"- warn: {counts.get('WARN', 0)}",
        f"- fail: {counts.get('FAIL', 0)}",
        f"- error: {counts.get('ERROR', 0)}",
        "",
        "| status | run_name | rows | phase0_schema | phase1_shadow_schema | finite | phase0_numeric | phase1_numeric | csv_path |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        lines.append(
            "| {status} | {run} | {rows} | {p0} | {p1} | {finite_ok} | {p0n} | {p1n} | {path} |".format(
                status=summary.get("overall_status", ""),
                run=summary.get("run_name", ""),
                rows=summary.get("rows", ""),
                p0=summary.get("phase0_schema_ok", ""),
                p1=summary.get("phase1_shadow_schema_ok", ""),
                finite_ok=summary.get("finite_ok", ""),
                p0n=summary.get("phase0_numeric_ok", ""),
                p1n=summary.get("phase1_shadow_numeric_ok", ""),
                path=summary.get("csv_path", ""),
            )
        )

    for summary in summaries:
        warnings = summary.get("warnings") or []
        failures = summary.get("failures") or []
        notes = summary.get("notes") or []
        if not warnings and not failures and not notes:
            continue
        lines.extend(["", f"## {summary.get('run_name', summary.get('csv_path', 'csv'))}"])
        lines.append(f"- csv_path: {summary.get('csv_path', '')}")
        if failures:
            lines.append(f"- failures: {json_dumps(failures)}")
        if warnings:
            lines.append(f"- warnings: {json_dumps(warnings)}")
        if notes:
            lines.append(f"- notes: {json_dumps(notes)}")

    return "\n".join(lines) + "\n"


def write_outputs(summaries: list[dict[str, Any]], output_dir: Path, strict: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "schema_check_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: csv_cell(summary.get(field, "")) for field in SUMMARY_FIELDS})

    json_path = output_dir / "schema_check_summary.json"
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "strict": strict,
        "summaries": summaries,
    }
    with json_path.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    report_path = output_dir / "schema_check_report.md"
    report_path.write_text(render_markdown_report(summaries, strict))


def collect_inputs(args: argparse.Namespace, stack: ExitStack) -> list[InputCsv]:
    inputs: list[InputCsv] = []
    for csv_path in args.csv_paths:
        inputs.append(InputCsv(path=csv_path, label=str(csv_path)))
    for data_dir in args.data_dirs:
        inputs.extend(find_csv_files(data_dir))
    for archive_path in args.archives:
        temp_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="goal1_gp_shadow_csv_")))
        inputs.extend(find_archive_csv_files(archive_path, temp_dir))

    deduped: list[InputCsv] = []
    seen = set()
    for input_csv in inputs:
        key = input_csv.label
        if key in seen:
            continue
        seen.add(key)
        deduped.append(input_csv)
    return deduped


def main() -> int:
    args = parse_args()
    if not args.csv_paths and not args.data_dirs and not args.archives:
        print("error: provide at least one of --csv, --data-dir, or --archive", file=sys.stderr)
        return 2

    with ExitStack() as stack:
        try:
            inputs = collect_inputs(args, stack)
        except Exception as exc:
            print(f"error while collecting inputs: {exc}", file=sys.stderr)
            return 2

        if not inputs:
            print(f"error: no {CSV_BASENAME} files found", file=sys.stderr)
            return 2

        summaries = []
        for input_csv in inputs:
            try:
                loaded = load_csv(input_csv)
                summaries.append(check_loaded_csv(loaded, args.strict))
            except Exception as exc:
                summaries.append(error_summary(input_csv, exc))

        report = render_markdown_report(summaries, args.strict)
        print(report, end="")

        if not args.no_write:
            write_outputs(summaries, args.output_dir, args.strict)
            print(f"Wrote outputs to {args.output_dir}")

    has_error = any(summary.get("overall_status") == "ERROR" for summary in summaries)
    has_failure = any(summary.get("overall_status") == "FAIL" for summary in summaries)
    if has_error or (args.strict and has_failure):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
