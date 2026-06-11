#!/usr/bin/env python3
"""Offline sanity check for frozen local/cloud-like GP model predictions."""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


JOINTS = range(1, 8)
DEFAULT_MAX_SAMPLES = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether joint*_local.pkl and joint*_cloud.pkl produce "
            "dynamic predictions on q/dq samples from a controller CSV."
        )
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--constant-range-tol", type=float, default=1e-9)
    parser.add_argument("--constant-unique-max", type=int, default=3)
    return parser.parse_args()


def ensure_skygp_import() -> None:
    sys.dont_write_bytecode = True
    repo_root = Path(__file__).resolve().parents[1]
    gp_dir = repo_root / "new_structure" / "gp"
    if str(gp_dir) not in sys.path:
        sys.path.insert(0, str(gp_dir))
    try:
        import skygp  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Cannot import skygp from {gp_dir}") from exc


def require_columns(fieldnames: list[str], prefix: str) -> list[str]:
    cols = [f"{prefix}_{joint}" for joint in JOINTS]
    missing = [col for col in cols if col not in fieldnames]
    if missing:
        raise ValueError(f"Missing required CSV columns for {prefix}: {missing}")
    return cols


def optional_columns(fieldnames: list[str], prefix: str) -> list[str] | None:
    cols = [f"{prefix}_{joint}" for joint in JOINTS]
    return cols if all(col in fieldnames for col in cols) else None


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


def read_vector(row: dict[str, str], cols: list[str]) -> np.ndarray:
    values = np.array([parse_float(row.get(col)) for col in cols], dtype=float)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError
    return values


def load_feature_rows(csv_path: Path, max_samples: int) -> tuple[np.ndarray, str, bool]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        q_cols = require_columns(fieldnames, "joint_pos")
        dq_cols = optional_columns(fieldnames, "dq_des_joint")
        dq_source = "dq_des_joint"
        if dq_cols is None:
            dq_cols = require_columns(fieldnames, "joint_vel")
            dq_source = "joint_vel"
        ddq_cols = optional_columns(fieldnames, "ddq_des_joint")

        rows = []
        for row in reader:
            try:
                q = read_vector(row, q_cols)
                dq = read_vector(row, dq_cols)
                ddq = read_vector(row, ddq_cols) if ddq_cols is not None else np.zeros(7)
            except ValueError:
                continue
            rows.append(np.concatenate([q, dq, ddq]).astype(np.float32))

    if not rows:
        raise ValueError(f"No finite q/dq feature rows found in {csv_path}")

    if max_samples > 0 and len(rows) > max_samples:
        idx = np.linspace(0, len(rows) - 1, max_samples, dtype=int)
        rows = [rows[i] for i in idx]

    return np.vstack(rows), dq_source, ddq_cols is not None


def count_numeric_samples(value: Any) -> int | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return 0
        if arr.dtype.kind in ("b", "i", "u", "f"):
            return int(np.sum(arr))
    except Exception:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def model_sample_summary(model: Any) -> tuple[int | None, int | None, str]:
    details: dict[str, Any] = {}
    for attr in ("X_list", "Y_list", "y_list", "experts", "local_experts"):
        value = getattr(model, attr, None)
        if value is not None:
            try:
                details[f"{attr}_len"] = len(value)
            except TypeError:
                details[f"{attr}_len"] = "NA"

    sample_count = None
    for attr in ("localCount", "num_points", "N"):
        count = count_numeric_samples(getattr(model, attr, None))
        if count is not None:
            details[attr] = count
            sample_count = count
            break

    expert_count = None
    for attr in ("expert_centers", "experts", "local_experts"):
        value = getattr(model, attr, None)
        if value is None:
            continue
        try:
            expert_count = len(value)
            break
        except TypeError:
            continue

    detail_text = ",".join(f"{key}={value}" for key, value in sorted(details.items()))
    return sample_count, expert_count, detail_text or "no_sample_attrs"


def load_pack(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        pack = pickle.load(f)
    if not isinstance(pack, dict) or "model" not in pack or "stats" not in pack:
        raise ValueError(f"Unexpected model pack format: {path}")
    return pack


def scalar_first(value: Any, default: float) -> float:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 0 or not np.isfinite(arr[0]):
            return default
        return float(arr[0])
    except (TypeError, ValueError):
        return default


def predict_range(pack: dict[str, Any], feature_rows: np.ndarray) -> dict[str, Any]:
    model = pack["model"]
    Xm, Xs, Ym, Ys = pack["stats"]
    Xm = np.asarray(Xm, dtype=np.float32).reshape(-1)
    Xs = np.asarray(Xs, dtype=np.float32).reshape(-1)
    x_dim = int(len(Xm))
    Ym_scalar = scalar_first(Ym, 0.0)
    Ys_scalar = scalar_first(Ys, 1.0)
    if abs(Ys_scalar) <= 1e-12:
        Ys_scalar = 1.0

    x_safe = np.where(np.abs(Xs[:x_dim]) > 1e-12, Xs[:x_dim], 1.0)
    preds = []
    for x_full in feature_rows:
        # frozen GP 如果没有在线样本/experts，predict 常会只返回 prior mean。
        # 这里用真实 CSV 的 q/dq 输入扫 prediction range，确认输出是否真的动态。
        if x_full.shape[0] < x_dim:
            padded = np.zeros(x_dim, dtype=np.float32)
            padded[:x_full.shape[0]] = x_full
            x_query = padded
        else:
            x_query = x_full[:x_dim]
        x_std = (x_query - Xm[:x_dim]) / x_safe
        mu_std, _ = model.predict(x_std.astype(np.float32))
        mu = scalar_first(mu_std, 0.0) * Ys_scalar + Ym_scalar
        preds.append(mu)

    pred_arr = np.asarray(preds, dtype=float)
    rounded_unique = len(np.unique(np.round(pred_arr, decimals=9)))
    return {
        "x_dim": x_dim,
        "Ym": Ym_scalar,
        "Ys": Ys_scalar,
        "pred_min": float(np.min(pred_arr)),
        "pred_max": float(np.max(pred_arr)),
        "pred_range": float(np.ptp(pred_arr)),
        "unique_rounded_1e-9": rounded_unique,
    }


def print_row(values: list[Any]) -> None:
    print("\t".join(str(value) for value in values))


def main() -> int:
    args = parse_args()
    ensure_skygp_import()
    feature_rows, dq_source, has_ddq = load_feature_rows(args.csv, args.max_samples)

    print(f"model_dir={args.model_dir}")
    print(f"csv={args.csv}")
    print(f"samples={len(feature_rows)} dq_source={dq_source} ddq_available={int(has_ddq)}")
    print_row([
        "joint",
        "kind",
        "file",
        "x_dim",
        "Ym",
        "Ys",
        "sample_count",
        "expert_count",
        "pred_min",
        "pred_max",
        "pred_range",
        "unique_rounded_1e-9",
        "constant",
        "model_details",
    ])

    for joint in JOINTS:
        for kind in ("local", "cloud"):
            path = args.model_dir / f"joint{joint}_{kind}.pkl"
            if not path.exists():
                print_row([joint, kind, path.name, "missing", "", "", "", "", "", "", "", "", "", ""])
                continue
            try:
                pack = load_pack(path)
                model = pack["model"]
                sample_count, expert_count, details = model_sample_summary(model)
                pred = predict_range(pack, feature_rows)
                is_constant = (
                    pred["pred_range"] <= args.constant_range_tol
                    or pred["unique_rounded_1e-9"] <= args.constant_unique_max
                )
                print_row([
                    joint,
                    kind,
                    path.name,
                    pred["x_dim"],
                    f"{pred['Ym']:.9g}",
                    f"{pred['Ys']:.9g}",
                    sample_count if sample_count is not None else "unknown",
                    expert_count if expert_count is not None else "unknown",
                    f"{pred['pred_min']:.9g}",
                    f"{pred['pred_max']:.9g}",
                    f"{pred['pred_range']:.9g}",
                    pred["unique_rounded_1e-9"],
                    int(is_constant),
                    details,
                ])
            except Exception as exc:
                print_row([joint, kind, path.name, "error", "", "", "", "", "", "", "", "", "", exc])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
