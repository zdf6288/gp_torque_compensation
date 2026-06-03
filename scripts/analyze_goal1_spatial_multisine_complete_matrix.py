#!/usr/bin/env python3
"""Offline GOAL1 spatial multisine complete matrix analysis.

This script has no ROS2 dependency. It reads saved controller CSV logs from an
archive or extracted directory, computes sanity, tracking, GP compensation, and
clip summaries, then writes generated evidence tables and plots under one output
directory.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any


JOINT_COUNT = 7
DEFAULT_OUTPUT_DIR = Path("outputs/goal1_spatial_multisine_complete_matrix_20260603")
CSV_NAME = "cartesian_impedance_controller_data.csv"

EXPECTED_RUNS = [
    "goal1_spatial_multisine_combined_scale01_clip05_short_20260603",
    "goal1_spatial_multisine_nogp_3000_20260603",
    "goal1_spatial_multisine_local_scale01_clip05_3000_20260603",
    "goal1_spatial_multisine_cloud_scale01_clip05_3000_20260603",
    "goal1_spatial_multisine_combined_scale01_clip05_3000_20260603",
    "goal1_spatial_multisine_combined_scale10_clip05_short_20260603",
    "goal1_spatial_multisine_local_scale10_clip05_3000_20260603",
    "goal1_spatial_multisine_cloud_scale10_clip05_3000_20260603",
    "goal1_spatial_multisine_combined_scale10_clip05_3000_20260603",
    "goal1_spatial_multisine_nogp_repeat_end_3000_20260603",
    "goal1_spatial_multisine_online_local_scale01_clip05_3000_20260603",
    "goal1_spatial_multisine_online_cloud_scale01_clip05_3000_20260603",
    "goal1_spatial_multisine_online_combined_scale01_clip05_3000_20260603",
    "goal1_spatial_multisine_online_local_scale10_clip05_3000_20260603",
    "goal1_spatial_multisine_online_cloud_scale10_clip05_3000_20260603",
    "goal1_spatial_multisine_online_combined_scale10_clip05_3000_20260603",
]

SOURCE_CODE_TO_NAME = {
    0: "none",
    1: "local",
    2: "cloud",
    3: "combined",
}

SUMMARY_FILES = [
    "run_manifest.csv",
    "sanity_summary.csv",
    "tracking_summary.csv",
    "gp_compensation_summary.csv",
    "clip_summary.csv",
    "analysis_summary.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the GOAL1 spatial multisine frozen and online matrix.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--archive", type=Path, help="Path to a .tar.gz, .tgz, .tar, or .zip archive.")
    input_group.add_argument("--data-dir", type=Path, help="Path to an already extracted data directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated outputs.")
    parser.add_argument("--extract-dir", type=Path, help="Extraction directory for --archive.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def joint_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{idx}" for idx in range(1, JOINT_COUNT + 1)]


def parse_float(value: str) -> float:
    if value is None or value == "":
        raise ValueError("empty numeric value")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return number


def read_csv_numeric(path: Path) -> tuple[list[str], dict[str, list[float]], int, bool, list[str]]:
    warnings: list[str] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        headers = list(reader.fieldnames)
        values = {column: [] for column in headers}
        finite_numeric_ok = True
        rows = 0
        for row_number, row in enumerate(reader, start=2):
            rows += 1
            for column in headers:
                try:
                    values[column].append(parse_float(row.get(column, "")))
                except ValueError as exc:
                    finite_numeric_ok = False
                    values[column].append(math.nan)
                    if len(warnings) < 10:
                        warnings.append(f"{column} row {row_number}: {exc}")

    if rows == 0:
        raise ValueError(f"CSV has no data rows: {path}")
    return headers, values, rows, finite_numeric_ok, warnings


def require_columns(headers: list[str], columns: list[str]) -> bool:
    available = set(headers)
    return all(column in available for column in columns)


def unique_values(values: dict[str, list[float]], column: str) -> list[float]:
    if column not in values:
        return []
    unique = sorted({round(value, 12) for value in values[column] if math.isfinite(value)})
    return unique


def unique_string(values: list[float]) -> str:
    return ";".join(f"{value:g}" for value in values)


def stable_unique(values: dict[str, list[float]], column: str) -> tuple[float | None, str]:
    unique = unique_values(values, column)
    if len(unique) == 1:
        return unique[0], unique_string(unique)
    return None, unique_string(unique)


def bool_from_number(value: float | None) -> str:
    if value is None:
        return ""
    return "true" if abs(value) > 0.5 else "false"


def source_from_code(code: float | None) -> str:
    if code is None:
        return ""
    return SOURCE_CODE_TO_NAME.get(int(round(code)), f"unknown_{code:g}")


def infer_category(run_name: str) -> str:
    if "_online_" in run_name:
        return "online_diagnostic"
    if "_short_" in run_name:
        return "frozen_short"
    return "frozen_main"


def infer_source_from_name(run_name: str) -> str:
    if "_nogp" in run_name:
        return "none"
    for source in ("local", "cloud", "combined"):
        if f"_{source}_" in run_name:
            return source
    return ""


def infer_scale_from_name(run_name: str) -> float | None:
    if "_nogp" in run_name:
        return 0.0
    if "_scale01_" in run_name:
        return 0.1
    if "_scale10_" in run_name:
        return 1.0
    return None


def infer_clip_from_name(run_name: str) -> float | None:
    if "_clip05" in run_name:
        return 0.5
    return None


def matrix(values: dict[str, list[float]], columns: list[str]) -> list[list[float]]:
    row_count = len(values[columns[0]]) if columns else 0
    return [[values[column][row_idx] for column in columns] for row_idx in range(row_count)]


def column_abs_max(values: list[float]) -> float:
    finite = [abs(value) for value in values if math.isfinite(value)]
    return max(finite) if finite else math.nan


def flatten(matrix_values: list[list[float]]) -> list[float]:
    return [value for row in matrix_values for value in row if math.isfinite(value)]


def rms(numbers: list[float]) -> float:
    finite = [value for value in numbers if math.isfinite(value)]
    if not finite:
        return math.nan
    return math.sqrt(sum(value * value for value in finite) / len(finite))


def max_abs_matrix(matrix_values: list[list[float]]) -> float:
    finite = [abs(value) for value in flatten(matrix_values)]
    return max(finite) if finite else math.nan


def per_joint_max_abs(matrix_values: list[list[float]]) -> list[float]:
    result = []
    for idx in range(JOINT_COUNT):
        result.append(column_abs_max([row[idx] for row in matrix_values]))
    return result


def per_joint_rms(matrix_values: list[list[float]]) -> list[float]:
    result = []
    for idx in range(JOINT_COUNT):
        result.append(rms([row[idx] for row in matrix_values]))
    return result


def count_clip_active(values: dict[str, list[float]], headers: list[str]) -> tuple[int, list[int]]:
    columns = joint_columns("gp_clip_active")
    if not require_columns(headers, columns):
        return 0, [0] * JOINT_COUNT
    per_joint = [sum(1 for value in values[column] if math.isfinite(value) and abs(value) > 0.5) for column in columns]
    return sum(per_joint), per_joint


def format_per_joint(values: list[Any]) -> str:
    return ";".join(f"j{idx}={value}" for idx, value in enumerate(values, start=1))


def format_float(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.12g}"


def first_or_empty(value: str | None) -> str:
    return value if value is not None else ""


def detect_cartesian_columns(headers: list[str]) -> tuple[list[str] | None, list[str] | None, list[str]]:
    candidates = [
        (["x_desired", "y_desired", "z_desired"], ["x_actual", "y_actual", "z_actual"]),
        (["x_des", "y_des", "z_des"], ["x_meas", "y_meas", "z_meas"]),
        (["x_des_1", "x_des_2", "x_des_3"], ["x_meas_1", "x_meas_2", "x_meas_3"]),
        (["x_des_1", "x_des_2", "x_des_3"], ["x_actual_1", "x_actual_2", "x_actual_3"]),
    ]
    for desired, actual in candidates:
        if require_columns(headers, desired) and require_columns(headers, actual):
            return desired, actual, []

    lowered = {column: column.lower() for column in headers}
    position_like = [
        column
        for column, lower in lowered.items()
        if any(token in lower for token in ("x_", "y_", "z_", "_x", "_y", "_z", "actual", "desired"))
    ]
    return None, None, position_like


def tracking_summary(values: dict[str, list[float]], headers: list[str]) -> dict[str, Any]:
    desired_cols, actual_cols, position_like = detect_cartesian_columns(headers)
    if desired_cols is None or actual_cols is None:
        return {
            "tracking_available": False,
            "desired_position_columns": "",
            "actual_position_columns": "",
            "position_like_columns": ";".join(position_like),
        }

    desired = matrix(values, desired_cols)
    actual = matrix(values, actual_cols)
    errors_m = [[actual_row[idx] - desired_row[idx] for idx in range(3)] for desired_row, actual_row in zip(desired, actual)]
    errors_mm = [[value * 1000.0 for value in row] for row in errors_m]
    norms_mm = [math.sqrt(sum(value * value for value in row)) for row in errors_mm]

    return {
        "tracking_available": True,
        "desired_position_columns": ";".join(desired_cols),
        "actual_position_columns": ";".join(actual_cols),
        "position_like_columns": "",
        "rmse_x_mm": rms([row[0] for row in errors_mm]),
        "rmse_y_mm": rms([row[1] for row in errors_mm]),
        "rmse_z_mm": rms([row[2] for row in errors_mm]),
        "rmse_3d_mm": rms(norms_mm),
        "mean_3d_error_mm": sum(norms_mm) / len(norms_mm) if norms_mm else math.nan,
        "max_3d_error_mm": max(norms_mm) if norms_mm else math.nan,
    }


def torque_relation_errors(values: dict[str, list[float]], headers: list[str]) -> tuple[float, float]:
    nominal_cols = joint_columns("tau_nominal")
    final_cols = joint_columns("tau_final")
    applied_cols = joint_columns("gp_applied")
    tau_cols = joint_columns("tau")

    relation_error = math.nan
    if require_columns(headers, nominal_cols + final_cols + applied_cols):
        relation_error = 0.0
        row_count = len(values[final_cols[0]])
        for row_idx in range(row_count):
            for joint_idx in range(JOINT_COUNT):
                expected = values[nominal_cols[joint_idx]][row_idx] - values[applied_cols[joint_idx]][row_idx]
                actual = values[final_cols[joint_idx]][row_idx]
                relation_error = max(relation_error, abs(actual - expected))

    tau_cmd_error = math.nan
    if require_columns(headers, tau_cols + final_cols):
        tau_cmd_error = 0.0
        row_count = len(values[final_cols[0]])
        for row_idx in range(row_count):
            for joint_idx in range(JOINT_COUNT):
                tau_cmd_error = max(tau_cmd_error, abs(values[tau_cols[joint_idx]][row_idx] - values[final_cols[joint_idx]][row_idx]))

    return relation_error, tau_cmd_error


def run_status(warnings: list[str], missing_required: list[str], finite_numeric_ok: bool, relation_error: float, tau_cmd_error: float) -> str:
    if missing_required or not finite_numeric_ok:
        return "fail"
    if warnings:
        return "warning"
    if math.isfinite(relation_error) and relation_error > 1e-9:
        return "warning"
    if math.isfinite(tau_cmd_error) and tau_cmd_error > 1e-9:
        return "warning"
    return "ok"


def analyze_run(run_name: str, csv_path: Path) -> dict[str, Any]:
    headers, values, rows, finite_numeric_ok, numeric_warnings = read_csv_numeric(csv_path)
    warnings = list(numeric_warnings)
    missing_required: list[str] = []

    required_debug_columns = [
        "gp_prediction_enabled",
        "gp_online_update_enabled",
        "gp_compensation_enabled",
        "gp_compensation_source_code",
        "gp_compensation_scale",
        "gp_compensation_clip_nm",
    ]
    required_debug_columns += joint_columns("tau_nominal")
    required_debug_columns += joint_columns("tau_final")
    required_debug_columns += joint_columns("gp_selected_raw")
    required_debug_columns += joint_columns("gp_scaled")
    required_debug_columns += joint_columns("gp_applied")
    required_debug_columns += joint_columns("gp_clip_active")
    for column in required_debug_columns:
        if column not in headers:
            missing_required.append(column)

    category = infer_category(run_name)
    name_source = infer_source_from_name(run_name)
    name_scale = infer_scale_from_name(run_name)
    name_clip = infer_clip_from_name(run_name)

    prediction_value, prediction_unique = stable_unique(values, "gp_prediction_enabled")
    online_value, online_unique = stable_unique(values, "gp_online_update_enabled")
    compensation_value, compensation_unique = stable_unique(values, "gp_compensation_enabled")
    source_code_value, source_code_unique = stable_unique(values, "gp_compensation_source_code")
    scale_value, scale_unique = stable_unique(values, "gp_compensation_scale")
    clip_value, clip_unique = stable_unique(values, "gp_compensation_clip_nm")

    source = source_from_code(source_code_value) if source_code_value is not None else name_source
    scale = scale_value if scale_value is not None else name_scale
    clip = clip_value if clip_value is not None else name_clip

    if source_code_value is not None and name_source and source != name_source:
        warnings.append(f"source from CSV ({source}) differs from run name ({name_source})")
    if scale_value is not None and name_scale is not None and abs(scale_value - name_scale) > 1e-9:
        warnings.append(f"scale from CSV ({scale_value:g}) differs from run name ({name_scale:g})")
    if clip_value is not None and name_clip is not None and abs(clip_value - name_clip) > 1e-9:
        warnings.append(f"clip from CSV ({clip_value:g}) differs from run name ({name_clip:g})")

    total_clip_count, per_joint_clip_count = count_clip_active(values, headers)
    relation_error, tau_cmd_error = torque_relation_errors(values, headers)
    tracking = tracking_summary(values, headers)

    gp_matrices: dict[str, list[list[float]]] = {}
    for prefix in ("gp_selected_raw", "gp_scaled", "gp_applied"):
        columns = joint_columns(prefix)
        if require_columns(headers, columns):
            gp_matrices[prefix] = matrix(values, columns)
        else:
            gp_matrices[prefix] = []

    status = run_status(warnings, missing_required, finite_numeric_ok, relation_error, tau_cmd_error)
    if missing_required:
        warnings.append("missing required debug columns: " + ";".join(missing_required))
    if not tracking["tracking_available"]:
        warnings.append("Cartesian tracking columns were not auto-identified")

    return {
        "run_name": run_name,
        "csv_path": str(csv_path),
        "category": category,
        "source": source,
        "scale": scale,
        "clip_nm": clip,
        "online_update_enabled": bool_from_number(online_value),
        "prediction_enabled": bool_from_number(prediction_value),
        "compensation_enabled": bool_from_number(compensation_value),
        "rows": rows,
        "columns": len(headers),
        "finite_numeric_ok": finite_numeric_ok,
        "source_code_unique": source_code_unique,
        "scale_unique": scale_unique,
        "clip_unique": clip_unique,
        "prediction_unique": prediction_unique,
        "online_unique": online_unique,
        "compensation_unique": compensation_unique,
        "max_abs_gp_selected_raw": max_abs_matrix(gp_matrices["gp_selected_raw"]),
        "max_abs_gp_scaled": max_abs_matrix(gp_matrices["gp_scaled"]),
        "max_abs_gp_applied": max_abs_matrix(gp_matrices["gp_applied"]),
        "rms_gp_selected_raw": rms(flatten(gp_matrices["gp_selected_raw"])),
        "rms_gp_scaled": rms(flatten(gp_matrices["gp_scaled"])),
        "rms_gp_applied": rms(flatten(gp_matrices["gp_applied"])),
        "per_joint_max_abs_gp_selected_raw": per_joint_max_abs(gp_matrices["gp_selected_raw"]) if gp_matrices["gp_selected_raw"] else [math.nan] * JOINT_COUNT,
        "per_joint_rms_gp_selected_raw": per_joint_rms(gp_matrices["gp_selected_raw"]) if gp_matrices["gp_selected_raw"] else [math.nan] * JOINT_COUNT,
        "per_joint_max_abs_gp_scaled": per_joint_max_abs(gp_matrices["gp_scaled"]) if gp_matrices["gp_scaled"] else [math.nan] * JOINT_COUNT,
        "per_joint_rms_gp_scaled": per_joint_rms(gp_matrices["gp_scaled"]) if gp_matrices["gp_scaled"] else [math.nan] * JOINT_COUNT,
        "per_joint_max_abs_gp_applied": per_joint_max_abs(gp_matrices["gp_applied"]) if gp_matrices["gp_applied"] else [math.nan] * JOINT_COUNT,
        "per_joint_rms_gp_applied": per_joint_rms(gp_matrices["gp_applied"]) if gp_matrices["gp_applied"] else [math.nan] * JOINT_COUNT,
        "total_clip_active_count": total_clip_count,
        "per_joint_clip_active_count": per_joint_clip_count,
        "max_tau_relation_error": relation_error,
        "max_tau_cmd_error": tau_cmd_error,
        "status": status,
        "warnings": warnings,
        "tracking": tracking,
    }


def safe_target(base: Path, name: str) -> Path:
    target = (base / name).resolve()
    base_resolved = base.resolve()
    if target != base_resolved and base_resolved not in target.parents:
        raise ValueError(f"archive member would extract outside target directory: {name}")
    return target


def extract_tar_archive(archive: Path, extract_dir: Path) -> None:
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            safe_target(extract_dir, member.name)
        handle.extractall(extract_dir)


def extract_zip_archive(archive: Path, extract_dir: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            safe_target(extract_dir, name)
        handle.extractall(extract_dir)


def resolve_data_root(args: argparse.Namespace) -> Path:
    if args.data_dir is not None:
        if not args.data_dir.exists():
            raise FileNotFoundError(f"data directory does not exist: {args.data_dir}")
        if not args.data_dir.is_dir():
            raise ValueError(f"--data-dir is not a directory: {args.data_dir}")
        return args.data_dir

    archive = args.archive
    if archive is None:
        raise ValueError("provide --archive or --data-dir")
    if not archive.exists():
        raise FileNotFoundError(f"archive does not exist: {archive}")
    if not archive.is_file():
        raise ValueError(f"--archive is not a file: {archive}")

    extract_dir = args.extract_dir if args.extract_dir is not None else args.output_dir / "extracted_archive"
    extract_dir.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive.suffixes).lower()
    if suffixes.endswith(".zip"):
        extract_zip_archive(archive, extract_dir)
    elif suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz") or suffixes.endswith(".tar"):
        extract_tar_archive(archive, extract_dir)
    else:
        raise ValueError(f"unsupported archive format: {archive}")
    return extract_dir


def find_csv_paths(data_root: Path) -> dict[str, Path]:
    csv_paths: dict[str, Path] = {}
    for csv_path in sorted(data_root.rglob(CSV_NAME)):
        run_name = csv_path.parent.name
        csv_paths[run_name] = csv_path
    return csv_paths


def ordered_runs(csv_paths: dict[str, Path]) -> list[tuple[str, Path]]:
    result = []
    for run_name in EXPECTED_RUNS:
        if run_name in csv_paths:
            result.append((run_name, csv_paths[run_name]))
    extras = sorted(name for name in csv_paths if name not in EXPECTED_RUNS)
    for run_name in extras:
        result.append((run_name, csv_paths[run_name]))
    return result


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def base_summary_row(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": run["run_name"],
        "category": run["category"],
        "source": run["source"],
        "scale": format_float(run["scale"]),
        "clip_nm": format_float(run["clip_nm"]),
        "online_update_enabled": run["online_update_enabled"],
        "prediction_enabled": run["prediction_enabled"],
        "compensation_enabled": run["compensation_enabled"],
    }


def write_run_manifest(runs: list[dict[str, Any]], output_dir: Path, missing_expected: list[str]) -> None:
    rows = []
    for run in runs:
        row = base_summary_row(run)
        row.update(
            {
                "csv_path": run["csv_path"],
                "rows": run["rows"],
                "columns": run["columns"],
                "status": run["status"],
                "warnings": "; ".join(run["warnings"]),
            }
        )
        rows.append(row)
    for run_name in missing_expected:
        rows.append(
            {
                "run_name": run_name,
                "category": infer_category(run_name),
                "source": infer_source_from_name(run_name),
                "scale": format_float(infer_scale_from_name(run_name)),
                "clip_nm": format_float(infer_clip_from_name(run_name)),
                "status": "fail",
                "warnings": "expected run directory or CSV missing",
            }
        )

    write_csv(
        output_dir / "run_manifest.csv",
        [
            "run_name",
            "category",
            "source",
            "scale",
            "clip_nm",
            "online_update_enabled",
            "prediction_enabled",
            "compensation_enabled",
            "csv_path",
            "rows",
            "columns",
            "status",
            "warnings",
        ],
        rows,
    )


def write_sanity_summary(runs: list[dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for run in runs:
        row = base_summary_row(run)
        row.update(
            {
                "rows": run["rows"],
                "columns": run["columns"],
                "finite_numeric_ok": run["finite_numeric_ok"],
                "source_code_unique": run["source_code_unique"],
                "scale_unique": run["scale_unique"],
                "clip_unique": run["clip_unique"],
                "max_abs_gp_selected_raw": format_float(run["max_abs_gp_selected_raw"]),
                "max_abs_gp_scaled": format_float(run["max_abs_gp_scaled"]),
                "max_abs_gp_applied": format_float(run["max_abs_gp_applied"]),
                "total_clip_active_count": run["total_clip_active_count"],
                "per_joint_clip_active_count": format_per_joint(run["per_joint_clip_active_count"]),
                "max_tau_relation_error": format_float(run["max_tau_relation_error"]),
                "max_tau_cmd_error": format_float(run["max_tau_cmd_error"]),
                "status": run["status"],
                "warnings": "; ".join(run["warnings"]),
            }
        )
        rows.append(row)

    write_csv(
        output_dir / "sanity_summary.csv",
        [
            "run_name",
            "category",
            "source",
            "scale",
            "clip_nm",
            "online_update_enabled",
            "prediction_enabled",
            "compensation_enabled",
            "rows",
            "columns",
            "finite_numeric_ok",
            "source_code_unique",
            "scale_unique",
            "clip_unique",
            "max_abs_gp_selected_raw",
            "max_abs_gp_scaled",
            "max_abs_gp_applied",
            "total_clip_active_count",
            "per_joint_clip_active_count",
            "max_tau_relation_error",
            "max_tau_cmd_error",
            "status",
            "warnings",
        ],
        rows,
    )


def write_tracking_summary(runs: list[dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for run in runs:
        row = base_summary_row(run)
        tracking = run["tracking"]
        row.update(
            {
                "tracking_available": tracking["tracking_available"],
                "desired_position_columns": first_or_empty(tracking.get("desired_position_columns")),
                "actual_position_columns": first_or_empty(tracking.get("actual_position_columns")),
                "rmse_x_mm": format_float(tracking.get("rmse_x_mm")),
                "rmse_y_mm": format_float(tracking.get("rmse_y_mm")),
                "rmse_z_mm": format_float(tracking.get("rmse_z_mm")),
                "rmse_3d_mm": format_float(tracking.get("rmse_3d_mm")),
                "mean_3d_error_mm": format_float(tracking.get("mean_3d_error_mm")),
                "max_3d_error_mm": format_float(tracking.get("max_3d_error_mm")),
                "position_like_columns": first_or_empty(tracking.get("position_like_columns")),
                "status": run["status"],
                "warnings": "; ".join(run["warnings"]),
            }
        )
        rows.append(row)

    write_csv(
        output_dir / "tracking_summary.csv",
        [
            "run_name",
            "category",
            "source",
            "scale",
            "clip_nm",
            "online_update_enabled",
            "prediction_enabled",
            "compensation_enabled",
            "tracking_available",
            "desired_position_columns",
            "actual_position_columns",
            "rmse_x_mm",
            "rmse_y_mm",
            "rmse_z_mm",
            "rmse_3d_mm",
            "mean_3d_error_mm",
            "max_3d_error_mm",
            "position_like_columns",
            "status",
            "warnings",
        ],
        rows,
    )


def add_per_joint_fields(row: dict[str, Any], prefix: str, values: list[float]) -> None:
    for idx, value in enumerate(values, start=1):
        row[f"{prefix}_{idx}"] = format_float(value)


def write_gp_compensation_summary(runs: list[dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for run in runs:
        row = base_summary_row(run)
        row.update(
            {
                "max_abs_gp_selected_raw": format_float(run["max_abs_gp_selected_raw"]),
                "rms_gp_selected_raw": format_float(run["rms_gp_selected_raw"]),
                "max_abs_gp_scaled": format_float(run["max_abs_gp_scaled"]),
                "rms_gp_scaled": format_float(run["rms_gp_scaled"]),
                "max_abs_gp_applied": format_float(run["max_abs_gp_applied"]),
                "rms_gp_applied": format_float(run["rms_gp_applied"]),
                "status": run["status"],
                "warnings": "; ".join(run["warnings"]),
            }
        )
        add_per_joint_fields(row, "max_abs_gp_selected_raw_j", run["per_joint_max_abs_gp_selected_raw"])
        add_per_joint_fields(row, "rms_gp_selected_raw_j", run["per_joint_rms_gp_selected_raw"])
        add_per_joint_fields(row, "max_abs_gp_scaled_j", run["per_joint_max_abs_gp_scaled"])
        add_per_joint_fields(row, "rms_gp_scaled_j", run["per_joint_rms_gp_scaled"])
        add_per_joint_fields(row, "max_abs_gp_applied_j", run["per_joint_max_abs_gp_applied"])
        add_per_joint_fields(row, "rms_gp_applied_j", run["per_joint_rms_gp_applied"])
        rows.append(row)

    fields = [
        "run_name",
        "category",
        "source",
        "scale",
        "clip_nm",
        "online_update_enabled",
        "prediction_enabled",
        "compensation_enabled",
        "max_abs_gp_selected_raw",
        "rms_gp_selected_raw",
        "max_abs_gp_scaled",
        "rms_gp_scaled",
        "max_abs_gp_applied",
        "rms_gp_applied",
    ]
    for metric in (
        "max_abs_gp_selected_raw_j",
        "rms_gp_selected_raw_j",
        "max_abs_gp_scaled_j",
        "rms_gp_scaled_j",
        "max_abs_gp_applied_j",
        "rms_gp_applied_j",
    ):
        fields += [f"{metric}_{idx}" for idx in range(1, JOINT_COUNT + 1)]
    fields += ["status", "warnings"]
    write_csv(output_dir / "gp_compensation_summary.csv", fields, rows)


def write_clip_summary(runs: list[dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for run in runs:
        total_cells = run["rows"] * JOINT_COUNT
        row = base_summary_row(run)
        row.update(
            {
                "total_clip_active_count": run["total_clip_active_count"],
                "clip_active_ratio": format_float(run["total_clip_active_count"] / total_cells if total_cells else math.nan),
                "max_abs_gp_applied": format_float(run["max_abs_gp_applied"]),
                "status": run["status"],
                "warnings": "; ".join(run["warnings"]),
            }
        )
        for idx, count in enumerate(run["per_joint_clip_active_count"], start=1):
            row[f"gp_clip_active_j{idx}"] = count
        rows.append(row)

    fields = [
        "run_name",
        "category",
        "source",
        "scale",
        "clip_nm",
        "online_update_enabled",
        "prediction_enabled",
        "compensation_enabled",
        "total_clip_active_count",
        "clip_active_ratio",
        "max_abs_gp_applied",
    ]
    fields += [f"gp_clip_active_j{idx}" for idx in range(1, JOINT_COUNT + 1)]
    fields += ["status", "warnings"]
    write_csv(output_dir / "clip_summary.csv", fields, rows)


def short_label(run: dict[str, Any]) -> str:
    category = "online" if run["category"] == "online_diagnostic" else "frozen"
    source = run["source"]
    scale = format_float(run["scale"])
    if source == "none":
        if "repeat_end" in run["run_name"]:
            return "noGP end"
        return "noGP begin"
    if "_short_" in run["run_name"]:
        return f"{source} s{scale} short"
    return f"{category} {source} s{scale}"


def import_matplotlib(output_dir: Path) -> Any | None:
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = output_dir / ".matplotlib"
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


def plot_bar(plt: Any, labels: list[str], values: list[float], title: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.65), 4.8))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_plots(runs: list[dict[str, Any]], output_dir: Path) -> None:
    plt = import_matplotlib(output_dir)
    if plt is None:
        return

    frozen = [run for run in runs if run["category"] in ("frozen_main", "frozen_short") and run["tracking"]["tracking_available"]]
    online = [run for run in runs if run["category"] == "online_diagnostic" and run["tracking"]["tracking_available"]]
    if frozen:
        plot_bar(
            plt,
            [short_label(run) for run in frozen],
            [run["tracking"]["rmse_3d_mm"] for run in frozen],
            "Frozen matrix 3D tracking RMSE",
            "RMSE 3D position error (mm)",
            output_dir / "frozen_tracking_rmse_comparison.png",
        )
    if online:
        plot_bar(
            plt,
            [short_label(run) for run in online],
            [run["tracking"]["rmse_3d_mm"] for run in online],
            "Online diagnostic 3D tracking RMSE",
            "RMSE 3D position error (mm)",
            output_dir / "online_tracking_rmse_comparison.png",
        )

    gp_on = [run for run in runs if run["source"] != "none"]
    if gp_on:
        plot_bar(
            plt,
            [short_label(run) for run in gp_on],
            [run["max_abs_gp_applied"] for run in gp_on],
            "Max abs GP applied by run",
            "max |gp_applied| (Nm)",
            output_dir / "max_abs_gp_applied_by_run.png",
        )
        plot_bar(
            plt,
            [short_label(run) for run in gp_on],
            [run["total_clip_active_count"] for run in gp_on],
            "Clip active count by run",
            "clip active samples across joints",
            output_dir / "clip_active_count_by_run.png",
        )

    joint_counts = [sum(run["per_joint_clip_active_count"][idx] for run in runs) for idx in range(JOINT_COUNT)]
    plot_bar(
        plt,
        [f"j{idx}" for idx in range(1, JOINT_COUNT + 1)],
        joint_counts,
        "Clip active count by joint",
        "clip active samples",
        output_dir / "clip_active_count_by_joint.png",
    )

    begin = next((run for run in runs if run["run_name"] == "goal1_spatial_multisine_nogp_3000_20260603"), None)
    end = next((run for run in runs if run["run_name"] == "goal1_spatial_multisine_nogp_repeat_end_3000_20260603"), None)
    if begin is not None and end is not None and begin["tracking"]["tracking_available"] and end["tracking"]["tracking_available"]:
        plot_bar(
            plt,
            ["noGP begin", "noGP repeat end"],
            [begin["tracking"]["rmse_3d_mm"], end["tracking"]["rmse_3d_mm"]],
            "No-GP begin vs repeat-end drift",
            "RMSE 3D position error (mm)",
            output_dir / "nogp_begin_vs_end_drift.png",
        )


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> list[str]:
    selected = rows if limit is None else rows[:limit]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines


def write_analysis_summary(runs: list[dict[str, Any]], output_dir: Path, missing_expected: list[str], plots_enabled: bool) -> None:
    tracking_rows = []
    clip_rows = []
    for run in runs:
        tracking = run["tracking"]
        tracking_rows.append(
            {
                "run": short_label(run),
                "category": run["category"],
                "source": run["source"],
                "scale": format_float(run["scale"]),
                "rmse_3d_mm": format_float(tracking.get("rmse_3d_mm")),
                "max_3d_mm": format_float(tracking.get("max_3d_error_mm")),
                "status": run["status"],
            }
        )
        clip_rows.append(
            {
                "run": short_label(run),
                "category": run["category"],
                "source": run["source"],
                "scale": format_float(run["scale"]),
                "max_abs_gp_applied": format_float(run["max_abs_gp_applied"]),
                "clip_count": run["total_clip_active_count"],
                "per_joint": format_per_joint(run["per_joint_clip_active_count"]),
            }
        )

    frozen_rows = [row for row in tracking_rows if row["category"] in ("frozen_main", "frozen_short")]
    online_rows = [row for row in tracking_rows if row["category"] == "online_diagnostic"]
    warning_runs = [run for run in runs if run["status"] != "ok" or run["warnings"]]

    lines = [
        "# GOAL1 Spatial Multisine Complete Matrix Analysis",
        "",
        "This summary was generated from CSV-derived values. It does not assume tracking improvement from GP-on configuration.",
        "",
        "## Generated outputs",
        "",
    ]
    for filename in SUMMARY_FILES:
        lines.append(f"- `{filename}`")
    if plots_enabled:
        lines.extend(
            [
                "- `frozen_tracking_rmse_comparison.png`",
                "- `online_tracking_rmse_comparison.png`",
                "- `max_abs_gp_applied_by_run.png`",
                "- `clip_active_count_by_run.png`",
                "- `clip_active_count_by_joint.png`",
                "- `nogp_begin_vs_end_drift.png`",
            ]
        )

    lines.extend(
        [
            "",
            "## Main frozen GP matrix",
            "",
            "The frozen matrix is the primary controlled comparison. `gp_online_update_enabled=false` keeps the GP model fixed during each run, so no-GP, local, cloud, combined, scale 0.1, and scale 1.0 runs can be compared without model-state changes during the run.",
            "",
        ]
    )
    lines.extend(markdown_table(frozen_rows, ["run", "category", "source", "scale", "rmse_3d_mm", "max_3d_mm", "status"]))

    lines.extend(
        [
            "",
            "## Legacy online-update diagnostic matrix",
            "",
            "The online-update matrix is supplementary. It checks compatibility with legacy behavior where online update may be enabled by default. It should not replace the frozen matrix as the main controlled comparison because model state changes during the run.",
            "",
        ]
    )
    lines.extend(markdown_table(online_rows, ["run", "category", "source", "scale", "rmse_3d_mm", "max_3d_mm", "status"]))

    lines.extend(
        [
            "",
            "## Clip interpretation",
            "",
            "`clip=0.5` is kept as the safety bound. If frozen scale 1.0 runs show no clip activation, that supports not increasing the clip for this matrix. If online scale 1.0 runs activate the clip, that indicates online update can push GP output closer to the safety bound and supports keeping `clip=0.5`.",
            "",
        ]
    )
    lines.extend(markdown_table(clip_rows, ["run", "category", "source", "scale", "max_abs_gp_applied", "clip_count", "per_joint"]))

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Several real-robot runs may save valid CSV/plots but end with `User Stop`, `communication_constraints_violation`, or `rclpy.shutdown()` traceback. Treat these as engineering caveats, not automatic data invalidation.",
            "- Use wording such as usable real-robot data, complete evidence matrix for offline comparison, and post-run shutdown caveats.",
            "- Avoid claiming the system is fully stable, robustly validated, or that communication constraints are solved.",
            "- Tracking improvement should be concluded only from `tracking_summary.csv`, not from GP-on configuration.",
            "",
            "## Warnings",
            "",
        ]
    )
    if missing_expected:
        lines.append("- Missing expected runs: " + "; ".join(missing_expected))
    if warning_runs:
        for run in warning_runs:
            warnings = "; ".join(run["warnings"]) if run["warnings"] else run["status"]
            lines.append(f"- `{run['run_name']}`: {warnings}")
    if not missing_expected and not warning_runs:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Conclusion wording",
            "",
            "The real-robot dataset provides a complete evidence matrix for offline comparison. The frozen GP matrix is the primary controlled comparison. The online-update matrix is treated as a legacy diagnostic. The clip summary confirms whether the safety bound was active and where.",
        ]
    )
    output_dir.joinpath("analysis_summary.md").write_text("\n".join(lines) + "\n")


def write_all_outputs(runs: list[dict[str, Any]], output_dir: Path, missing_expected: list[str], no_plots: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(runs, output_dir, missing_expected)
    write_sanity_summary(runs, output_dir)
    write_tracking_summary(runs, output_dir)
    write_gp_compensation_summary(runs, output_dir)
    write_clip_summary(runs, output_dir)
    if not no_plots:
        write_plots(runs, output_dir)
    write_analysis_summary(runs, output_dir, missing_expected, not no_plots)


def main() -> int:
    args = parse_args()
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        data_root = resolve_data_root(args)
        csv_paths = find_csv_paths(data_root)
        if not csv_paths:
            raise ValueError(f"no {CSV_NAME} files found under {data_root}")

        missing_expected = [run_name for run_name in EXPECTED_RUNS if run_name not in csv_paths]
        runs = [analyze_run(run_name, csv_path) for run_name, csv_path in ordered_runs(csv_paths)]
        write_all_outputs(runs, args.output_dir, missing_expected, args.no_plots)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote GOAL1 complete matrix analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
