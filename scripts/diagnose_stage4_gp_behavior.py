#!/usr/bin/env python3
"""Diagnose Stage 4 frozen-GP behavior from formal CSV logs.

This script is intentionally offline-only: no ROS2 dependency, no robot command,
and no writes outside the diagnostic output directory. It follows the existing
Stage 4 formal-analysis style with csv/numpy/matplotlib Agg outputs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_STRICT_CSV = Path(
    "data/stage4/test/strict_no_gp/usable_runs/strict_no_gp_zmod_3000pts_20260523_154902.csv"
)
DEFAULT_PLANAR_CSV = Path(
    "data/stage4/test/gp_planar_scale03/usable_runs/gp_planar_scale03_zmod_2999pts_20260523_161222.csv"
)
DEFAULT_SPATIAL_CSV = Path(
    "data/stage4/test/gp_spatial_scale03/usable_runs/gp_spatial_scale03_zmod_3000pts_20260523_163907.csv"
)
DEFAULT_PLANAR_PARTIAL_GLOB = "data/stage4/test/gp_planar_scale03/partial_runs/*.csv"
DEFAULT_SPATIAL_PARTIAL_GLOB = "data/stage4/test/gp_spatial_scale03/partial_runs/*.csv"
DEFAULT_OUT_DIR = Path("outputs/stage4_gp_diagnostic")
DEFAULT_SCALES = "0.1,0.3,0.5,0.7,1.0"

EPS = 1e-12
ZERO_VARIANCE_EPS = 1e-12
JOINTS = range(1, 8)
TRACKING_ACTUAL = ("x_actual", "y_actual", "z_actual")
TRACKING_DESIRED = ("x_desired", "y_desired", "z_desired")
MODES = ("strict_no_gp", "gp_planar_scale03", "gp_spatial_scale03")
GP_MODES = ("gp_planar_scale03", "gp_spatial_scale03")
MODE_LABELS = {
    "strict_no_gp": "strict no-GP",
    "gp_planar_scale03": "GP planar scale03",
    "gp_spatial_scale03": "GP spatial scale03",
}

np = None
plt = None


def import_dependencies(output_dir: Path) -> bool:
    global np, plt

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
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot_module
    except ModuleNotFoundError:
        missing.append("matplotlib")

    if missing:
        print("Missing Python dependencies: " + ", ".join(sorted(set(missing))), file=sys.stderr)
        print("Use the project .venv with these packages installed; this script does not install packages.", file=sys.stderr)
        return False

    np = numpy_module
    plt = pyplot_module
    return True


def parse_scales(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise argparse.ArgumentTypeError("--scales must contain at least one float")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline diagnostic for Stage 4 frozen-GP formal results.",
    )
    parser.add_argument("--strict-csv", type=Path, default=DEFAULT_STRICT_CSV, help=f"Default: {DEFAULT_STRICT_CSV}")
    parser.add_argument("--planar-csv", type=Path, default=DEFAULT_PLANAR_CSV, help=f"Default: {DEFAULT_PLANAR_CSV}")
    parser.add_argument("--spatial-csv", type=Path, default=DEFAULT_SPATIAL_CSV, help=f"Default: {DEFAULT_SPATIAL_CSV}")
    parser.add_argument(
        "--planar-partial-glob",
        default=DEFAULT_PLANAR_PARTIAL_GLOB,
        help=f"Default: {DEFAULT_PLANAR_PARTIAL_GLOB}",
    )
    parser.add_argument(
        "--spatial-partial-glob",
        default=DEFAULT_SPATIAL_PARTIAL_GLOB,
        help=f"Default: {DEFAULT_SPATIAL_PARTIAL_GLOB}",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help=f"Default: {DEFAULT_OUT_DIR}")
    parser.add_argument("--scales", type=parse_scales, default=parse_scales(DEFAULT_SCALES), help=f"Default: {DEFAULT_SCALES}")
    parser.add_argument("--clip-nm", type=float, default=0.5, help="Compensation proxy clip in Nm. Default: 0.5")
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


def finite_values(values: "np.ndarray") -> "np.ndarray":
    return values[np.isfinite(values)]


def finite_pair(a: "np.ndarray", b: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def rms(values: "np.ndarray") -> float:
    finite = finite_values(values)
    if len(finite) == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.square(finite))))


def max_abs(values: "np.ndarray") -> float:
    finite = finite_values(values)
    if len(finite) == 0:
        return math.nan
    return float(np.max(np.abs(finite)))


def p95_abs(values: "np.ndarray") -> float:
    finite = finite_values(values)
    if len(finite) == 0:
        return math.nan
    return float(np.percentile(np.abs(finite), 95.0))


def mean(values: "np.ndarray") -> float:
    finite = finite_values(values)
    if len(finite) == 0:
        return math.nan
    return float(np.mean(finite))


def std(values: "np.ndarray") -> float:
    finite = finite_values(values)
    if len(finite) == 0:
        return math.nan
    return float(np.std(finite))


def span(values: "np.ndarray") -> float:
    finite = finite_values(values)
    if len(finite) == 0:
        return math.nan
    return float(np.max(finite) - np.min(finite))


def relative_change_percent(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or abs(reference) < EPS:
        return math.nan
    return float(100.0 * (value - reference) / reference)


def improvement_percent(value: float, baseline: float) -> float:
    if not np.isfinite(value) or not np.isfinite(baseline) or abs(baseline) < EPS:
        return math.nan
    return float(100.0 * (baseline - value) / baseline)


def normalized_name(name: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in name]
    return "_".join(part for part in "".join(chars).split("_") if part)


def detect_time_column(columns: Iterable[str]) -> str | None:
    candidates = []
    for column in columns:
        lowered = normalized_name(column)
        if any(pattern in lowered for pattern in ("time", "timestamp", "elapsed")):
            candidates.append(column)
        elif lowered == "t":
            candidates.append(column)
    if not candidates:
        return None
    for preferred in ("time_s", "time", "elapsed", "timestamp"):
        for column in candidates:
            if preferred in normalized_name(column):
                return column
    return candidates[0]


def required_columns() -> list[str]:
    columns = list(TRACKING_ACTUAL) + list(TRACKING_DESIRED)
    columns.extend(f"tau_residual_{joint}" for joint in JOINTS)
    columns.extend(f"y_hat_local_{joint}" for joint in JOINTS)
    return columns


def load_csv(path: Path, mode_name: str, run_kind: str = "fullrun") -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no CSV header found")
        columns = list(reader.fieldnames)
        data = {column: [] for column in columns}
        for row in reader:
            for column in columns:
                data[column].append(parse_float(row.get(column)))

    arrays = {column: np.asarray(values, dtype=float) for column, values in data.items()}
    rows = len(next(iter(arrays.values()))) if arrays else 0
    missing = [column for column in required_columns() if column not in columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
    print(f"Loaded {path}: mode={mode_name}, kind={run_kind}, rows={rows}, columns={len(columns)}")
    return {
        "mode_name": mode_name,
        "run_kind": run_kind,
        "path": path,
        "columns": columns,
        "data": arrays,
        "rows": rows,
    }


def resolve_partial_runs(pattern: str, mode_name: str) -> tuple[list[dict[str, object]], str]:
    paths = sorted(Path(path) for path in glob.glob(pattern))
    if not paths:
        return [], f"missing: no files matched {pattern}"
    datasets = [load_csv(path, mode_name, "partial") for path in paths]
    return datasets, f"found {len(paths)} file(s)"


def tracking_arrays(dataset: dict[str, object], length: int | None = None) -> tuple["np.ndarray", "np.ndarray"]:
    data = dataset["data"]
    errors = []
    for actual_col, desired_col in zip(TRACKING_ACTUAL, TRACKING_DESIRED):
        values = data[actual_col] - data[desired_col]
        errors.append(values[:length] if length is not None else values)
    matrix = np.vstack(errors).T
    norms = np.linalg.norm(matrix, axis=1)
    return matrix, norms


def joint_values(dataset: dict[str, object], prefix: str, joint: int, length: int | None = None) -> "np.ndarray":
    values = dataset["data"][f"{prefix}_{joint}"]
    return values[:length] if length is not None else values


def all_joint_values(dataset: dict[str, object], prefix: str, length: int | None = None) -> "np.ndarray":
    arrays = [joint_values(dataset, prefix, joint, length) for joint in JOINTS]
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=float)


def pearson_corr(a: "np.ndarray", b: "np.ndarray") -> float:
    x, y = finite_pair(a, b)
    if len(x) < 2:
        return math.nan
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered)))
    if denom < EPS:
        return math.nan
    return float(np.sum(x_centered * y_centered) / denom)


def rankdata_average_ties(values: "np.ndarray") -> "np.ndarray":
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_corr(a: "np.ndarray", b: "np.ndarray") -> float:
    x, y = finite_pair(a, b)
    if len(x) < 2:
        return math.nan
    return pearson_corr(rankdata_average_ties(x), rankdata_average_ties(y))


def cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    x, y = finite_pair(a, b)
    if len(x) == 0:
        return math.nan
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < EPS:
        return math.nan
    return float(np.dot(x, y) / denom)


def sign_counts(values: "np.ndarray") -> tuple[int, int, int]:
    finite = finite_values(values)
    return int(np.sum(finite > 0)), int(np.sum(finite < 0)), int(np.sum(finite == 0))


def sign_agreement(a: "np.ndarray", b: "np.ndarray") -> tuple[float, int, int, int, int, int, int]:
    x, y = finite_pair(a, b)
    if len(x) == 0:
        return math.nan, 0, 0, 0, 0, 0, 0
    sx = np.sign(x)
    sy = np.sign(y)
    agreement = sx == sy
    tau_pos, tau_neg, tau_zero = sign_counts(x)
    y_pos, y_neg, y_zero = sign_counts(y)
    return float(np.mean(agreement)), int(np.sum(agreement)), len(x), tau_pos, tau_neg, tau_zero, y_pos + y_neg + y_zero


def sign_detail(values: "np.ndarray") -> dict[str, int]:
    positive, negative, zero = sign_counts(values)
    return {"positive_count": positive, "negative_count": negative, "zero_count": zero}


def is_constant_signal(values: "np.ndarray") -> bool:
    value_std = std(values)
    value_span = span(values)
    return (
        np.isfinite(value_std)
        and np.isfinite(value_span)
        and (abs(value_std) <= ZERO_VARIANCE_EPS or abs(value_span) <= ZERO_VARIANCE_EPS)
    )


def make_suspicious_score(
    sign_agreement_ratio: float,
    mean_product: float,
    cosine: float,
    pearson: float,
    pearson_valid: bool,
    y_hat_is_constant: bool,
) -> tuple[float, str]:
    score = 0.0
    reasons = []

    if np.isfinite(cosine) and cosine < 0.0:
        score += 100.0 * (-cosine)
        reasons.append("negative_cosine")

    if np.isfinite(sign_agreement_ratio) and sign_agreement_ratio < 0.55:
        score += 20.0 * (0.55 - sign_agreement_ratio)
        reasons.append("low_sign_agreement")

    if np.isfinite(mean_product) and mean_product < 0.0:
        score += min(10.0, 1000.0 * (-mean_product))
        reasons.append("negative_mean_product")

    if y_hat_is_constant:
        reasons.append("pearson_undefined_constant_y_hat")
    elif pearson_valid and np.isfinite(pearson) and pearson < 0.0:
        score += 5.0 * (-pearson)
        reasons.append("finite_negative_pearson")
    elif not pearson_valid:
        reasons.append("pearson_invalid_zero_variance")

    if not reasons:
        reasons.append("no_major_alignment_warning")
    return score, ",".join(reasons)


def make_residual_alignment_rows(datasets: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_joint = []
    summary = []
    for dataset in datasets:
        if dataset["mode_name"] not in GP_MODES:
            continue
        mode = dataset["mode_name"]
        products = []
        correlations = []
        weighted_product_num = 0.0
        weighted_product_den = 0
        total_agree = 0
        total_count = 0
        all_tau = []
        all_yhat = []
        for joint in JOINTS:
            tau = joint_values(dataset, "tau_residual", joint)
            yhat = joint_values(dataset, "y_hat_local", joint)
            tau_finite, yhat_finite = finite_pair(tau, yhat)
            agreement_ratio, agree_count, count, _tau_pos, _tau_neg, _tau_zero, _y_total = sign_agreement(tau, yhat)
            tau_sign = sign_detail(tau_finite)
            yhat_sign = sign_detail(yhat_finite)
            product = tau_finite * yhat_finite
            mean_product = float(np.mean(product)) if len(product) else math.nan
            y_hat_std = std(yhat_finite)
            y_hat_span = span(yhat_finite)
            y_hat_is_constant = is_constant_signal(yhat_finite)
            tau_is_constant = is_constant_signal(tau_finite)
            pearson_valid = not y_hat_is_constant and not tau_is_constant
            pearson = pearson_corr(tau, yhat)
            spearman = spearman_corr(tau, yhat)
            cosine = cosine_similarity(tau, yhat)
            suspicious_score, suspicious_reason = make_suspicious_score(
                agreement_ratio,
                mean_product,
                cosine,
                pearson,
                pearson_valid,
                y_hat_is_constant,
            )
            products.append(mean_product)
            if np.isfinite(pearson):
                correlations.append(pearson)
            if np.isfinite(mean_product):
                weighted_product_num += float(np.sum(product))
                weighted_product_den += int(len(product))
            total_agree += agree_count
            total_count += count
            all_tau.append(tau_finite)
            all_yhat.append(yhat_finite)
            per_joint.append({
                "mode_name": mode,
                "csv_file": dataset["path"].name,
                "joint": joint,
                "samples": count,
                "sign_agreement_ratio": agreement_ratio,
                "sign_agreement_count": agree_count,
                "tau_positive_count": tau_sign["positive_count"],
                "tau_negative_count": tau_sign["negative_count"],
                "tau_zero_count": tau_sign["zero_count"],
                "y_hat_positive_count": yhat_sign["positive_count"],
                "y_hat_negative_count": yhat_sign["negative_count"],
                "y_hat_zero_count": yhat_sign["zero_count"],
                "y_hat_std": y_hat_std,
                "y_hat_span": y_hat_span,
                "y_hat_is_constant": y_hat_is_constant,
                "pearson_valid": pearson_valid,
                "pearson_correlation": pearson,
                "spearman_correlation": spearman,
                "mean_product": mean_product,
                "cosine_similarity": cosine,
                "suspicious_score": suspicious_score,
                "suspicious_reason": suspicious_reason,
            })

        all_tau_values = np.concatenate(all_tau) if all_tau else np.asarray([], dtype=float)
        all_yhat_values = np.concatenate(all_yhat) if all_yhat else np.asarray([], dtype=float)
        summary.append({
            "mode_name": mode,
            "csv_file": dataset["path"].name,
            "joint": "all",
            "samples": total_count,
            "sign_agreement_ratio": float(total_agree / total_count) if total_count else math.nan,
            "mean_pearson_correlation": float(np.mean(correlations)) if correlations else math.nan,
            "all_joint_pearson_correlation": pearson_corr(all_tau_values, all_yhat_values),
            "all_joint_spearman_correlation": spearman_corr(all_tau_values, all_yhat_values),
            "weighted_mean_product": float(weighted_product_num / weighted_product_den) if weighted_product_den else math.nan,
            "all_joint_cosine_similarity": cosine_similarity(all_tau_values, all_yhat_values),
            "mean_product_unweighted": float(np.nanmean(products)) if products else math.nan,
        })
    return per_joint, summary


def make_y_hat_magnitude_rows(datasets: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    per_mode_joint = {}
    for dataset in datasets:
        if dataset["mode_name"] not in GP_MODES:
            continue
        mode = dataset["mode_name"]
        for joint in JOINTS:
            values = joint_values(dataset, "y_hat_local", joint)
            finite = finite_values(values)
            diffs = np.diff(finite) if len(finite) >= 2 else np.asarray([], dtype=float)
            row = {
                "mode_name": mode,
                "csv_file": dataset["path"].name,
                "joint": joint,
                "samples": len(finite),
                "rms": rms(finite),
                "mean": mean(finite),
                "std": std(finite),
                "max_abs": max_abs(finite),
                "p95_abs": p95_abs(finite),
                "roughness_rms_first_diff": rms(diffs),
                "planar_to_spatial_rms_ratio": math.nan,
                "spatial_to_planar_rms_ratio": math.nan,
                "planar_to_spatial_roughness_ratio": math.nan,
                "spatial_to_planar_roughness_ratio": math.nan,
            }
            per_mode_joint[(mode, joint)] = row
            rows.append(row)

    for joint in JOINTS:
        planar = per_mode_joint.get(("gp_planar_scale03", joint))
        spatial = per_mode_joint.get(("gp_spatial_scale03", joint))
        if not planar or not spatial:
            continue
        planar_rms = float(planar["rms"])
        spatial_rms = float(spatial["rms"])
        planar_rough = float(planar["roughness_rms_first_diff"])
        spatial_rough = float(spatial["roughness_rms_first_diff"])
        if abs(spatial_rms) > EPS:
            planar["planar_to_spatial_rms_ratio"] = planar_rms / spatial_rms
        if abs(planar_rms) > EPS:
            spatial["spatial_to_planar_rms_ratio"] = spatial_rms / planar_rms
        if abs(spatial_rough) > EPS:
            planar["planar_to_spatial_roughness_ratio"] = planar_rough / spatial_rough
        if abs(planar_rough) > EPS:
            spatial["spatial_to_planar_roughness_ratio"] = spatial_rough / planar_rough

    return rows


def make_tau_residual_comparison_rows(datasets: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    metrics = {}
    for dataset in datasets:
        mode = dataset["mode_name"]
        for joint in JOINTS:
            values = joint_values(dataset, "tau_residual", joint)
            metrics[(mode, joint)] = {
                "rms": rms(values),
                "max_abs": max_abs(values),
            }
        all_values = all_joint_values(dataset, "tau_residual")
        metrics[(mode, "all")] = {
            "rms": rms(all_values),
            "max_abs": max_abs(all_values),
        }

    for joint in list(JOINTS) + ["all"]:
        strict = metrics.get(("strict_no_gp", joint), {})
        planar = metrics.get(("gp_planar_scale03", joint), {})
        spatial = metrics.get(("gp_spatial_scale03", joint), {})
        for mode in MODES:
            current = metrics.get((mode, joint), {})
            rows.append({
                "mode_name": mode,
                "joint": joint,
                "tau_residual_rms": current.get("rms", math.nan),
                "tau_residual_max_abs": current.get("max_abs", math.nan),
                "rms_change_vs_strict_percent": relative_change_percent(current.get("rms", math.nan), strict.get("rms", math.nan)),
                "rms_improvement_vs_strict_percent": improvement_percent(current.get("rms", math.nan), strict.get("rms", math.nan)),
                "rms_change_spatial_vs_planar_percent": (
                    relative_change_percent(spatial.get("rms", math.nan), planar.get("rms", math.nan))
                    if mode == "gp_spatial_scale03"
                    else math.nan
                ),
            })
    return rows


def tracking_rms_3d(dataset: dict[str, object], length: int | None = None) -> float:
    _matrix, norms = tracking_arrays(dataset, length)
    return rms(norms)


def make_tracking_window_rows(datasets: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    min_rows = min(int(dataset["rows"]) for dataset in datasets)
    for dataset in datasets:
        mode = dataset["mode_name"]
        full_rows = int(dataset["rows"])
        for analysis_type, total_rows in (("full_length", full_rows), ("aligned_min_length", min_rows)):
            windows = [
                ("full", 0, total_rows),
                ("early_third", 0, total_rows // 3),
                ("middle_third", total_rows // 3, 2 * total_rows // 3),
                ("late_third", 2 * total_rows // 3, total_rows),
            ]
            for window_name, start, end in windows:
                length = end - start
                if length <= 0:
                    continue
                _error_matrix, norms = tracking_arrays(dataset)
                tau_values = np.concatenate([joint_values(dataset, "tau_residual", joint)[start:end] for joint in JOINTS])
                yhat_values = np.concatenate([joint_values(dataset, "y_hat_local", joint)[start:end] for joint in JOINTS])
                rows.append({
                    "mode_name": mode,
                    "csv_file": dataset["path"].name,
                    "analysis_type": analysis_type,
                    "window": window_name,
                    "start_index": start,
                    "end_index_exclusive": end,
                    "rows": length,
                    "tracking_3d_rms_m": rms(norms[start:end]),
                    "tracking_3d_rms_mm": rms(norms[start:end]) * 1000.0,
                    "tau_residual_all_rms": rms(tau_values),
                    "y_hat_local_all_rms": rms(yhat_values),
                })
    return rows


def make_scale_sweep_rows(datasets: list[dict[str, object]], scales: list[float], clip_nm: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_joint = []
    summary = []
    for dataset in datasets:
        if dataset["mode_name"] not in GP_MODES:
            continue
        mode = dataset["mode_name"]
        for scale in scales:
            all_proxy = []
            total_clip_hits = 0
            total_count = 0
            for joint in JOINTS:
                yhat = joint_values(dataset, "y_hat_local", joint)
                finite = finite_values(yhat)
                scaled = scale * finite
                proxy = np.clip(scaled, -clip_nm, clip_nm)
                clip_hits = np.abs(scaled) >= clip_nm
                all_proxy.append(proxy)
                total_clip_hits += int(np.sum(clip_hits))
                total_count += int(len(proxy))
                per_joint.append({
                    "mode_name": mode,
                    "csv_file": dataset["path"].name,
                    "scale": scale,
                    "clip_nm": clip_nm,
                    "joint": joint,
                    "proxy_rms": rms(proxy),
                    "proxy_max_abs": max_abs(proxy),
                    "proxy_p95_abs": p95_abs(proxy),
                    "clip_hit_count": int(np.sum(clip_hits)),
                    "clip_hit_ratio": float(np.mean(clip_hits)) if len(clip_hits) else math.nan,
                })
            combined = np.concatenate(all_proxy) if all_proxy else np.asarray([], dtype=float)
            summary.append({
                "mode_name": mode,
                "csv_file": dataset["path"].name,
                "scale": scale,
                "clip_nm": clip_nm,
                "joint": "all",
                "proxy_rms": rms(combined),
                "proxy_max_abs": max_abs(combined),
                "proxy_p95_abs": p95_abs(combined),
                "clip_hit_count": total_clip_hits,
                "clip_hit_ratio": float(total_clip_hits / total_count) if total_count else math.nan,
            })
    return per_joint, summary


def fullrun_reference_by_mode(datasets: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    refs = {}
    for dataset in datasets:
        if dataset["mode_name"] not in GP_MODES:
            continue
        refs[dataset["mode_name"]] = {
            "tracking_3d_rms_m": tracking_rms_3d(dataset),
            "tau_residual_all_rms": rms(all_joint_values(dataset, "tau_residual")),
            "y_hat_local_all_rms": rms(all_joint_values(dataset, "y_hat_local")),
            "max_abs_y_hat_per_joint": {
                joint: max_abs(joint_values(dataset, "y_hat_local", joint)) for joint in JOINTS
            },
        }
    return refs


def make_partial_run_rows(
    partial_datasets: list[dict[str, object]],
    fullrun_refs: dict[str, dict[str, object]],
    clip_nm: float,
) -> list[dict[str, object]]:
    rows = []
    for dataset in partial_datasets:
        mode = dataset["mode_name"]
        rows_count = int(dataset["rows"])
        final_start = int(math.floor(rows_count * 0.9))
        yhat_all = all_joint_values(dataset, "y_hat_local")
        proxy03 = np.clip(0.3 * finite_values(yhat_all), -clip_nm, clip_nm)
        ref = fullrun_refs.get(mode, {})
        full_joint_max = ref.get("max_abs_y_hat_per_joint", {})
        partial_joint_max = {joint: max_abs(joint_values(dataset, "y_hat_local", joint)) for joint in JOINTS}
        spike_joints = []
        for joint, partial_max in partial_joint_max.items():
            full_max = float(full_joint_max.get(joint, math.nan))
            if np.isfinite(partial_max) and np.isfinite(full_max) and partial_max > max(2.0 * full_max, full_max + 0.05):
                spike_joints.append(str(joint))
        _errors, norms = tracking_arrays(dataset)
        final_tau = np.concatenate([joint_values(dataset, "tau_residual", joint)[final_start:] for joint in JOINTS])
        final_yhat = np.concatenate([joint_values(dataset, "y_hat_local", joint)[final_start:] for joint in JOINTS])
        row = {
            "mode_name": mode,
            "csv_file": dataset["path"].name,
            "csv_path": str(dataset["path"]),
            "rows": rows_count,
            "tracking_3d_rms_m": rms(norms),
            "tracking_3d_rms_mm": rms(norms) * 1000.0,
            "tau_residual_all_rms": rms(all_joint_values(dataset, "tau_residual")),
            "y_hat_local_all_rms": rms(yhat_all),
            "comp_proxy_scale03_rms": rms(proxy03),
            "final_10_percent_start_index": final_start,
            "final_10_percent_tracking_3d_rms_m": rms(norms[final_start:]),
            "final_10_percent_tracking_3d_rms_mm": rms(norms[final_start:]) * 1000.0,
            "final_10_percent_tau_residual_all_rms": rms(final_tau),
            "final_10_percent_y_hat_local_all_rms": rms(final_yhat),
            "spike_joint_list_vs_fullrun": ",".join(spike_joints),
            "has_obvious_y_hat_spike_vs_fullrun": bool(spike_joints),
        }
        for joint in JOINTS:
            row[f"max_abs_y_hat_joint{joint}"] = partial_joint_max[joint]
        rows.append(row)
    return rows


def fmt(value: object, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {path}")


def row_lookup(rows: list[dict[str, object]], **criteria: object) -> dict[str, object]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    return {}


def save_grouped_joint_bar(
    rows: list[dict[str, object]],
    modes: Iterable[str],
    value_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    joints = list(JOINTS)
    values = {
        (row["mode_name"], int(row["joint"])): float(row[value_column])
        for row in rows
        if row.get("joint") != "all" and value_column in row
    }
    modes = list(modes)
    x = np.arange(len(joints))
    width = 0.8 / len(modes)
    fig, ax = plt.subplots(figsize=(12, 6))
    for mode_index, mode in enumerate(modes):
        offsets = x - 0.4 + width / 2 + mode_index * width
        y = [values.get((mode, joint), math.nan) for joint in joints]
        ax.bar(offsets, y, width, label=mode)
    ax.set_title(title)
    ax.set_xlabel("joint")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(joint) for joint in joints])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_tracking_window(rows: list[dict[str, object]], output_path: Path) -> None:
    filtered = [row for row in rows if row["analysis_type"] == "aligned_min_length" and row["window"] != "full"]
    windows = ["early_third", "middle_third", "late_third"]
    x = np.arange(len(windows))
    width = 0.8 / len(MODES)
    fig, ax = plt.subplots(figsize=(11, 6))
    for mode_index, mode in enumerate(MODES):
        offsets = x - 0.4 + width / 2 + mode_index * width
        y = [
            float(row_lookup(filtered, mode_name=mode, window=window).get("tracking_3d_rms_mm", math.nan))
            for window in windows
        ]
        ax.bar(offsets, y, width, label=mode)
    ax.set_title("Stage 4 aligned window 3D tracking RMS")
    ax.set_xlabel("window")
    ax.set_ylabel("3D RMS error (mm)")
    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_scale_sweep(rows: list[dict[str, object]], value_column: str, title: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in GP_MODES:
        mode_rows = sorted([row for row in rows if row["mode_name"] == mode], key=lambda row: float(row["scale"]))
        x = [float(row["scale"]) for row in mode_rows]
        y = [float(row[value_column]) for row in mode_rows]
        ax.plot(x, y, marker="o", label=mode)
    ax.set_title(title)
    ax.set_xlabel("scale")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_partial_runs(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    labels = [f"{row['mode_name']}\n{row['csv_file'][:28]}" for row in rows]
    x = np.arange(len(rows))
    tracking = [float(row["tracking_3d_rms_mm"]) for row in rows]
    yhat = [float(row["y_hat_local_all_rms"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(x, tracking)
    axes[0].set_ylabel("3D RMS (mm)")
    axes[0].set_title("Partial run tracking RMS")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(x, yhat)
    axes[1].set_ylabel("y_hat RMS (Nm)")
    axes[1].set_title("Partial run y_hat RMS")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_scatter_for_suspicious(
    datasets: list[dict[str, object]],
    alignment_rows: list[dict[str, object]],
    output_dir: Path,
) -> None:
    by_mode_joint = {
        (row["mode_name"], int(row["joint"])): row
        for row in alignment_rows
        if row.get("joint") != "all"
    }
    for dataset in datasets:
        mode = dataset["mode_name"]
        if mode not in GP_MODES:
            continue
        candidates = []
        for joint in JOINTS:
            row = by_mode_joint[(mode, joint)]
            score = float(row["suspicious_score"])
            candidates.append((score, joint))
        selected = [joint for _score, joint in sorted(candidates, reverse=True)[:3]]
        fig, axes = plt.subplots(1, len(selected), figsize=(15, 4))
        if len(selected) == 1:
            axes = [axes]
        for ax, joint in zip(axes, selected):
            tau = joint_values(dataset, "tau_residual", joint)
            yhat = joint_values(dataset, "y_hat_local", joint)
            tau_finite, yhat_finite = finite_pair(tau, yhat)
            stride = max(1, len(tau_finite) // 900)
            ax.scatter(tau_finite[::stride], yhat_finite[::stride], s=6, alpha=0.35)
            ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
            ax.axvline(0.0, color="black", linewidth=0.7, alpha=0.5)
            row = by_mode_joint[(mode, joint)]
            score = row["suspicious_score"]
            ax.set_title(f"{mode} joint {joint}, score={fmt(score, 1)}")
            ax.set_xlabel("tau_residual (Nm)")
            ax.set_ylabel("y_hat_local (Nm)")
            ax.grid(True, alpha=0.25)
        fig.suptitle(f"tau_residual vs y_hat_local suspicious joints: {mode} (constant y_hat can form horizontal lines)")
        fig.tight_layout()
        output_path = output_dir / f"scatter_tau_residual_vs_y_hat_suspicious_{mode}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=140)
        plt.close(fig)


def write_plots(
    datasets: list[dict[str, object]],
    alignment_rows: list[dict[str, object]],
    yhat_rows: list[dict[str, object]],
    tau_rows: list[dict[str, object]],
    tracking_rows: list[dict[str, object]],
    scale_summary_rows: list[dict[str, object]],
    partial_rows: list[dict[str, object]],
    out_dir: Path,
) -> None:
    save_grouped_joint_bar(
        alignment_rows,
        GP_MODES,
        "pearson_correlation",
        "Residual/y_hat Pearson correlation per joint",
        "Pearson r",
        out_dir / "residual_yhat_correlation_bar.png",
    )
    save_grouped_joint_bar(
        alignment_rows,
        GP_MODES,
        "sign_agreement_ratio",
        "Residual/y_hat sign agreement per joint",
        "sign agreement ratio",
        out_dir / "sign_agreement_per_joint.png",
    )
    save_grouped_joint_bar(
        yhat_rows,
        GP_MODES,
        "rms",
        "y_hat_local RMS per joint: planar vs spatial",
        "RMS (Nm)",
        out_dir / "y_hat_rms_per_joint_planar_vs_spatial.png",
    )
    save_grouped_joint_bar(
        yhat_rows,
        GP_MODES,
        "roughness_rms_first_diff",
        "y_hat_local temporal roughness per joint",
        "RMS first difference (Nm/sample)",
        out_dir / "y_hat_roughness_per_joint.png",
    )
    save_grouped_joint_bar(
        tau_rows,
        MODES,
        "tau_residual_rms",
        "tau_residual RMS per joint comparison",
        "RMS (Nm)",
        out_dir / "tau_residual_rms_per_joint_comparison.png",
    )
    plot_tracking_window(tracking_rows, out_dir / "tracking_window_3d_rms.png")
    plot_scale_sweep(
        scale_summary_rows,
        "proxy_rms",
        "Offline compensation proxy RMS scale sweep",
        "all-joint proxy RMS (Nm)",
        out_dir / "scale_sweep_proxy_rms.png",
    )
    plot_scale_sweep(
        scale_summary_rows,
        "clip_hit_ratio",
        "Offline compensation proxy clip ratio scale sweep",
        "clip hit ratio",
        out_dir / "scale_sweep_clip_ratio.png",
    )
    plot_partial_runs(partial_rows, out_dir / "partial_run_rows_and_rms.png")
    plot_scatter_for_suspicious(datasets, alignment_rows, out_dir)


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def make_decision_summary(
    tracking_rows: list[dict[str, object]],
    alignment_summary: list[dict[str, object]],
    scale_summary: list[dict[str, object]],
    partial_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    aligned_full = [row for row in tracking_rows if row["analysis_type"] == "aligned_min_length" and row["window"] == "full"]
    strict = row_lookup(aligned_full, mode_name="strict_no_gp")
    planar = row_lookup(aligned_full, mode_name="gp_planar_scale03")
    spatial = row_lookup(aligned_full, mode_name="gp_spatial_scale03")
    spatial_tracking_improvement = improvement_percent(
        float(spatial.get("tracking_3d_rms_m", math.nan)),
        float(strict.get("tracking_3d_rms_m", math.nan)),
    )
    spatial_vs_planar = relative_change_percent(
        float(spatial.get("tracking_3d_rms_m", math.nan)),
        float(planar.get("tracking_3d_rms_m", math.nan)),
    )
    spatial_alignment = row_lookup(alignment_summary, mode_name="gp_spatial_scale03")
    spatial_corr = float(spatial_alignment.get("mean_pearson_correlation", math.nan))
    spatial_sign = float(spatial_alignment.get("sign_agreement_ratio", math.nan))
    scale05 = row_lookup(scale_summary, mode_name="gp_spatial_scale03", scale=0.5)
    scale10 = row_lookup(scale_summary, mode_name="gp_spatial_scale03", scale=1.0)
    clip05 = float(scale05.get("clip_hit_ratio", math.nan))
    clip10 = float(scale10.get("clip_hit_ratio", math.nan))

    if spatial_tracking_improvement < 0 or spatial_vs_planar > 0:
        primary = "No immediate real-robot rerun; analyze model quality first"
    elif spatial_corr > 0.2 and spatial_sign > 0.55:
        primary = "Try GP_spatial scale01 or scale02 diagnostic only"
    else:
        primary = "Retrain hparams before next real-robot run"

    scale05_text = "Try GP_spatial scale05 only if alignment is positive and clip ratio remains low"
    if not (spatial_corr > 0.2 and spatial_sign > 0.55 and np.isfinite(clip05) and clip05 < 0.01):
        scale05_text = "Do not prioritize GP_spatial scale05 before alignment/model-quality checks"

    pearson_caveat = ""
    if not np.isfinite(spatial_corr):
        pearson_caveat = "; per-joint Pearson is undefined because y_hat_local_j is constant in these logs"

    return [
        {
            "recommendation_rank": 1,
            "recommendation": primary,
            "reason": (
                f"aligned spatial tracking improvement vs strict={fmt(spatial_tracking_improvement, 2)}%, "
                f"spatial vs planar tracking change={fmt(spatial_vs_planar, 2)}%, "
                f"mean Pearson={fmt(spatial_corr, 3)}, sign agreement={fmt(spatial_sign, 3)}"
                f"{pearson_caveat}"
            ),
        },
        {
            "recommendation_rank": 2,
            "recommendation": scale05_text,
            "reason": f"offline scale05 clip ratio={fmt(clip05, 6)}; this is amplitude-only evidence, not tracking evidence",
        },
        {
            "recommendation_rank": 3,
            "recommendation": "Do not try scale1.0 directly",
            "reason": f"offline scale1.0 clip ratio={fmt(clip10, 6)} does not prove real-robot tracking or stability",
        },
        {
            "recommendation_rank": 4,
            "recommendation": "Retrain hparams before next real-robot run",
            "reason": "current frozen GP models use fallback standardized hparams; model quality remains a plausible limiting factor",
        },
        {
            "recommendation_rank": 5,
            "recommendation": "Partial-run evidence should stay diagnostic only",
            "reason": f"partial runs analyzed={len(partial_rows)}; failed/partial logs are not comparable formal fullruns",
        },
    ]


def write_markdown_summary(
    path: Path,
    args: argparse.Namespace,
    datasets: list[dict[str, object]],
    partial_status: dict[str, str],
    alignment_rows: list[dict[str, object]],
    alignment_summary: list[dict[str, object]],
    yhat_rows: list[dict[str, object]],
    tau_rows: list[dict[str, object]],
    tracking_rows: list[dict[str, object]],
    scale_summary: list[dict[str, object]],
    partial_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
) -> None:
    aligned_full = [row for row in tracking_rows if row["analysis_type"] == "aligned_min_length" and row["window"] == "full"]
    scale03_summary = [row for row in scale_summary if abs(float(row["scale"]) - 0.3) < EPS]
    scale10_summary = [row for row in scale_summary if abs(float(row["scale"]) - 1.0) < EPS]
    suspicious_rows = sorted(alignment_rows, key=lambda row: float(row["suspicious_score"]), reverse=True)[:8]

    lines = [
        "# Stage 4 GP Diagnostic Summary",
        "",
        "## Diagnosis Scope",
        "",
        "This offline diagnostic examines the Stage 4 formal frozen-GP CSV logs to compare residual alignment, prediction magnitude, time-window behavior, and compensation proxy envelopes. It does not run ROS2 and does not command the robot.",
        "",
        "## Input Files",
        "",
        md_table(
            ["mode", "csv", "rows"],
            [[dataset["mode_name"], str(dataset["path"]), dataset["rows"]] for dataset in datasets],
        ),
        "",
        "Partial run search:",
        "",
        md_table(
            ["mode", "glob", "status"],
            [
                ["gp_planar_scale03", args.planar_partial_glob, partial_status["gp_planar_scale03"]],
                ["gp_spatial_scale03", args.spatial_partial_glob, partial_status["gp_spatial_scale03"]],
            ],
        ),
        "",
        "## Residual Alignment Result",
        "",
        md_table(
            ["mode", "sign agreement", "mean Pearson", "all-joint Pearson", "cosine", "weighted mean product"],
            [
                [
                    row["mode_name"],
                    fmt(row["sign_agreement_ratio"], 4),
                    fmt(row["mean_pearson_correlation"], 4),
                    fmt(row["all_joint_pearson_correlation"], 4),
                    fmt(row["all_joint_cosine_similarity"], 4),
                    fmt(row["weighted_mean_product"], 6),
                ]
                for row in alignment_summary
            ],
        ),
        "",
        "低 sign agreement、负 correlation 或负 mean product 只说明离线方向一致性有风险信号；这里不声称 causality。",
        "",
        "Per-joint Pearson / Spearman is undefined when `y_hat_local_j` has zero variance. In these logs, `y_hat_local_j` is constant per joint, so Pearson is not used as the primary suspicious-joint ranking criterion.",
        "",
        "Suspicious joints are ranked by negative cosine similarity, low sign agreement, negative mean product, and finite negative Pearson only when Pearson is valid. Scatter plots may look like horizontal lines because `y_hat_local_j` is constant.",
        "",
        "Suspicious joints by sign/cosine/mean-product alignment:",
        "",
        md_table(
            ["mode", "joint", "score", "reason", "cosine", "sign agreement", "mean product", "Pearson valid"],
            [
                [
                    row["mode_name"],
                    row["joint"],
                    fmt(row["suspicious_score"], 3),
                    row["suspicious_reason"],
                    fmt(row["cosine_similarity"], 4),
                    fmt(row["sign_agreement_ratio"], 4),
                    fmt(row["mean_product"], 6),
                    row["pearson_valid"],
                ]
                for row in suspicious_rows
            ],
        ),
        "",
        "## GP_planar vs GP_spatial Comparison",
        "",
        "Tracking aligned full-length comparison:",
        "",
        md_table(
            ["mode", "aligned rows", "3D RMS mm", "tau residual RMS", "y_hat RMS"],
            [
                [
                    row["mode_name"],
                    row["rows"],
                    fmt(row["tracking_3d_rms_mm"], 3),
                    fmt(row["tau_residual_all_rms"], 6),
                    fmt(row["y_hat_local_all_rms"], 6),
                ]
                for row in aligned_full
            ],
        ),
        "",
        "Prediction magnitude summary:",
        "",
        md_table(
            ["mode", "joint", "y_hat RMS", "std", "max abs", "p95 abs", "roughness"],
            [
                [
                    row["mode_name"],
                    row["joint"],
                    fmt(row["rms"], 6),
                    fmt(row["std"], 6),
                    fmt(row["max_abs"], 6),
                    fmt(row["p95_abs"], 6),
                    fmt(row["roughness_rms_first_diff"], 6),
                ]
                for row in yhat_rows
            ],
        ),
        "",
        "Tau residual all-joint comparison:",
        "",
        md_table(
            ["mode", "joint", "tau RMS", "change vs strict %", "improvement vs strict %"],
            [
                [
                    row["mode_name"],
                    row["joint"],
                    fmt(row["tau_residual_rms"], 6),
                    fmt(row["rms_change_vs_strict_percent"], 2),
                    fmt(row["rms_improvement_vs_strict_percent"], 2),
                ]
                for row in tau_rows
                if row["joint"] == "all"
            ],
        ),
        "",
        "## Scale Sweep Proxy Result",
        "",
        md_table(
            ["mode", "scale", "proxy RMS", "proxy max abs", "p95 abs", "clip ratio"],
            [
                [
                    row["mode_name"],
                    fmt(row["scale"], 1),
                    fmt(row["proxy_rms"], 6),
                    fmt(row["proxy_max_abs"], 6),
                    fmt(row["proxy_p95_abs"], 6),
                    fmt(row["clip_hit_ratio"], 6),
                ]
                for row in scale_summary
            ],
        ),
        "",
        f"The formal robot run used `scale=0.3` and `clip_nm={fmt(args.clip_nm, 3)}`. Scale sweep here is only an offline `clip(scale * y_hat_local, -clip_nm, clip_nm)` proxy.",
        "",
        "## Partial Run Diagnostic",
        "",
    ]

    if partial_rows:
        lines.extend([
            md_table(
                ["mode", "csv", "rows", "3D RMS mm", "tau RMS", "y_hat RMS", "proxy03 RMS", "spike joints"],
                [
                    [
                        row["mode_name"],
                        row["csv_file"],
                        row["rows"],
                        fmt(row["tracking_3d_rms_mm"], 3),
                        fmt(row["tau_residual_all_rms"], 6),
                        fmt(row["y_hat_local_all_rms"], 6),
                        fmt(row["comp_proxy_scale03_rms"], 6),
                        row["spike_joint_list_vs_fullrun"] or "none",
                    ]
                    for row in partial_rows
                ],
            ),
            "",
        ])
    else:
        lines.extend(["No partial runs were found by the configured globs.", ""])

    lines.extend([
        "## Interpretation",
        "",
        "- `GP_planar` 和 `GP_spatial` 的 offline alignment / magnitude 差异可以帮助解释本次 formal tracking 差异，但不能单独证明控制效果因果关系。",
        "- 如果某些 joint 的 `tau_residual_j` 与 `y_hat_local_j` correlation 低、sign agreement 接近随机或 mean product 为负，这些 joint 可能存在反向或弱相关 compensation 风险。",
        "- 如果 `GP_spatial` 的 `y_hat_local` RMS 更小，可能说明 compensation amplitude 不足；如果 roughness 更高，可能说明 prediction 更 noisy。",
        "- Aligned window metrics 用最短长度裁剪到共同 sample 数，避免 2999/3000 row 差异影响模式比较。",
        "",
        "## Next Real-Robot Recommendation",
        "",
        md_table(
            ["rank", "recommendation", "reason"],
            [[row["recommendation_rank"], row["recommendation"], row["reason"]] for row in decision_rows],
        ),
        "",
        "Conservative reading: do not directly escalate to `scale=1.0`. Consider `GP_spatial scale01` or `scale02` only as a diagnostic if alignment is not negative and the robot session has the usual safety workflow ready. Otherwise prioritize model-quality checks and hparam retraining before another real-robot compensation run.",
        "",
        "## Caveats",
        "",
        "- 当前结果来自 single formal fullrun，不是 robust repeated validation。",
        "- Scale sweep 是 offline compensation proxy，不代表真机 tracking。",
        "- 当前真机 formal run 使用 `scale=0.3`, `clip=0.5`。",
        "- 不建议直接真机跑 `scale=1.0`。",
        "- No-clip / unlimited GP compensation 不建议。",
        "- `clip_nm=0` 不是 no clip，通常会把 compensation clip 到 0。",
        "- 当前 frozen GP models 使用 fallback standardized hparams，这可能影响 GP quality。",
        "- Post-save `communication_constraints_violation` / shutdown caveat 不阻止离线分析，但不能声称 fully stable。",
        "",
        "## Output Files",
        "",
        "- `residual_alignment_per_joint.csv`",
        "- `residual_alignment_summary.csv`",
        "- `y_hat_magnitude_per_joint.csv`",
        "- `tau_residual_comparison_per_joint.csv`",
        "- `tracking_window_metrics.csv`",
        "- `scale_sweep_proxy_per_joint.csv`",
        "- `scale_sweep_proxy_summary.csv`",
        "- `partial_run_diagnostics.csv`",
        "- `diagnostic_decision_summary.csv`",
        "",
        "Related plots include residual correlation/sign agreement bars, y_hat magnitude/roughness bars, tau residual comparison, tracking window RMS, scale sweep proxy plots, partial run plot when available, and suspicious-joint scatter plots.",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def write_all_outputs(
    args: argparse.Namespace,
    datasets: list[dict[str, object]],
    partial_datasets: list[dict[str, object]],
    partial_status: dict[str, str],
) -> None:
    out_dir = args.out_dir
    alignment_rows, alignment_summary = make_residual_alignment_rows(datasets)
    yhat_rows = make_y_hat_magnitude_rows(datasets)
    tau_rows = make_tau_residual_comparison_rows(datasets)
    tracking_rows = make_tracking_window_rows(datasets)
    scale_per_joint, scale_summary = make_scale_sweep_rows(datasets, args.scales, args.clip_nm)
    partial_rows = make_partial_run_rows(partial_datasets, fullrun_reference_by_mode(datasets), args.clip_nm)
    decision_rows = make_decision_summary(tracking_rows, alignment_summary, scale_summary, partial_rows)

    write_rows(
        out_dir / "residual_alignment_per_joint.csv",
        alignment_rows,
        [
            "mode_name",
            "csv_file",
            "joint",
            "samples",
            "sign_agreement_ratio",
            "sign_agreement_count",
            "tau_positive_count",
            "tau_negative_count",
            "tau_zero_count",
            "y_hat_positive_count",
            "y_hat_negative_count",
            "y_hat_zero_count",
            "y_hat_std",
            "y_hat_span",
            "y_hat_is_constant",
            "pearson_valid",
            "pearson_correlation",
            "spearman_correlation",
            "mean_product",
            "cosine_similarity",
            "suspicious_score",
            "suspicious_reason",
        ],
    )
    write_rows(
        out_dir / "residual_alignment_summary.csv",
        alignment_summary,
        [
            "mode_name",
            "csv_file",
            "joint",
            "samples",
            "sign_agreement_ratio",
            "mean_pearson_correlation",
            "all_joint_pearson_correlation",
            "all_joint_spearman_correlation",
            "weighted_mean_product",
            "all_joint_cosine_similarity",
            "mean_product_unweighted",
        ],
    )
    write_rows(
        out_dir / "y_hat_magnitude_per_joint.csv",
        yhat_rows,
        [
            "mode_name",
            "csv_file",
            "joint",
            "samples",
            "rms",
            "mean",
            "std",
            "max_abs",
            "p95_abs",
            "roughness_rms_first_diff",
            "planar_to_spatial_rms_ratio",
            "spatial_to_planar_rms_ratio",
            "planar_to_spatial_roughness_ratio",
            "spatial_to_planar_roughness_ratio",
        ],
    )
    write_rows(
        out_dir / "tau_residual_comparison_per_joint.csv",
        tau_rows,
        [
            "mode_name",
            "joint",
            "tau_residual_rms",
            "tau_residual_max_abs",
            "rms_change_vs_strict_percent",
            "rms_improvement_vs_strict_percent",
            "rms_change_spatial_vs_planar_percent",
        ],
    )
    write_rows(
        out_dir / "tracking_window_metrics.csv",
        tracking_rows,
        [
            "mode_name",
            "csv_file",
            "analysis_type",
            "window",
            "start_index",
            "end_index_exclusive",
            "rows",
            "tracking_3d_rms_m",
            "tracking_3d_rms_mm",
            "tau_residual_all_rms",
            "y_hat_local_all_rms",
        ],
    )
    write_rows(
        out_dir / "scale_sweep_proxy_per_joint.csv",
        scale_per_joint,
        [
            "mode_name",
            "csv_file",
            "scale",
            "clip_nm",
            "joint",
            "proxy_rms",
            "proxy_max_abs",
            "proxy_p95_abs",
            "clip_hit_count",
            "clip_hit_ratio",
        ],
    )
    write_rows(
        out_dir / "scale_sweep_proxy_summary.csv",
        scale_summary,
        [
            "mode_name",
            "csv_file",
            "scale",
            "clip_nm",
            "joint",
            "proxy_rms",
            "proxy_max_abs",
            "proxy_p95_abs",
            "clip_hit_count",
            "clip_hit_ratio",
        ],
    )
    write_rows(
        out_dir / "partial_run_diagnostics.csv",
        partial_rows,
        [
            "mode_name",
            "csv_file",
            "csv_path",
            "rows",
            "tracking_3d_rms_m",
            "tracking_3d_rms_mm",
            "tau_residual_all_rms",
            "y_hat_local_all_rms",
            "comp_proxy_scale03_rms",
            "final_10_percent_start_index",
            "final_10_percent_tracking_3d_rms_m",
            "final_10_percent_tracking_3d_rms_mm",
            "final_10_percent_tau_residual_all_rms",
            "final_10_percent_y_hat_local_all_rms",
            "max_abs_y_hat_joint1",
            "max_abs_y_hat_joint2",
            "max_abs_y_hat_joint3",
            "max_abs_y_hat_joint4",
            "max_abs_y_hat_joint5",
            "max_abs_y_hat_joint6",
            "max_abs_y_hat_joint7",
            "spike_joint_list_vs_fullrun",
            "has_obvious_y_hat_spike_vs_fullrun",
        ],
    )
    write_rows(
        out_dir / "diagnostic_decision_summary.csv",
        decision_rows,
        ["recommendation_rank", "recommendation", "reason"],
    )
    write_markdown_summary(
        out_dir / "stage4_gp_diagnostic_summary.md",
        args,
        datasets,
        partial_status,
        alignment_rows,
        alignment_summary,
        yhat_rows,
        tau_rows,
        tracking_rows,
        scale_summary,
        partial_rows,
        decision_rows,
    )
    write_plots(
        datasets,
        alignment_rows,
        yhat_rows,
        tau_rows,
        tracking_rows,
        scale_summary,
        partial_rows,
        out_dir,
    )


def main() -> int:
    args = parse_args()
    if not import_dependencies(args.out_dir):
        return 2

    if args.clip_nm < 0:
        print("--clip-nm must be non-negative", file=sys.stderr)
        return 2
    if args.clip_nm == 0:
        print("Warning: --clip-nm=0 clips the compensation proxy to zero; it is not no-clip.", file=sys.stderr)

    datasets = [
        load_csv(args.strict_csv, "strict_no_gp"),
        load_csv(args.planar_csv, "gp_planar_scale03"),
        load_csv(args.spatial_csv, "gp_spatial_scale03"),
    ]
    planar_partials, planar_status = resolve_partial_runs(args.planar_partial_glob, "gp_planar_scale03")
    spatial_partials, spatial_status = resolve_partial_runs(args.spatial_partial_glob, "gp_spatial_scale03")
    partial_status = {
        "gp_planar_scale03": planar_status,
        "gp_spatial_scale03": spatial_status,
    }
    partial_datasets = planar_partials + spatial_partials
    write_all_outputs(args, datasets, partial_datasets, partial_status)
    print(f"Diagnostic complete: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
