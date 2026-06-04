#!/usr/bin/env python3
"""Replay GOAL1 historical residual DB as offline shadow columns on a controller CSV.

Offline-only:
- no ROS
- no robot
- no controller modification
- no active compensation
- no tau_final modification

Input:
- historical residual DB .npz
- one controller CSV with joint_pos/joint_vel and local/cloud predictions

Output:
- a new CSV with appended historical shadow diagnostics:
  - hist_db_pred_1..7
  - hist_db_nearest_distance
  - hist_db_mean_topk_distance
  - hist_db_available
  - hist_db_gated_pred_1..7
  - hist_db_gated_source_code
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DB = "outputs/goal1_historical_residual_db_20260604/goal1_historical_residual_db.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline replay historical DB shadow columns onto a controller CSV.")
    p.add_argument("--db", default=DEFAULT_DB, help="Historical residual DB .npz path.")
    p.add_argument("--input-csv", required=True, help="Input controller CSV.")
    p.add_argument("--output-csv", required=True, help="Output shadow CSV.")
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--q-scale", type=float, default=0.1)
    p.add_argument("--dq-scale", type=float, default=0.1)
    p.add_argument("--max-distance", type=float, default=1.0)
    p.add_argument(
        "--disable-historical-for-online",
        action="store_true",
        help="Disable historical gate if input path/name contains 'online'.",
    )
    p.add_argument("--chunk-size", type=int, default=250)
    return p.parse_args()


def columns():
    x_cols = [f"joint_pos_{j}" for j in range(1, 8)] + [f"joint_vel_{j}" for j in range(1, 8)]
    local_cols = [f"y_hat_local_{j}" for j in range(1, 8)]
    cloud_cols = [f"y_hat_cloud_{j}" for j in range(1, 8)]
    return x_cols, local_cols, cloud_cols


def load_db(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    db = np.load(path, allow_pickle=True)
    required = ["X", "Y_residual"]
    missing = [k for k in required if k not in db.files]
    if missing:
        raise RuntimeError(f"DB missing arrays: {missing}")

    X = db["X"].astype(float)
    Y = db["Y_residual"].astype(float)

    if X.ndim != 2 or X.shape[1] != 14:
        raise RuntimeError(f"Expected X shape (N,14), got {X.shape}")
    if Y.ndim != 2 or Y.shape[1] != 7:
        raise RuntimeError(f"Expected Y_residual shape (N,7), got {Y.shape}")
    if len(X) != len(Y):
        raise RuntimeError("DB X/Y length mismatch")

    return X, Y


def knn_query(X_db, Y_db, X_query, k, scale, chunk_size):
    Xd = X_db / scale
    Xq = X_query / scale

    pred = np.zeros((len(Xq), 7))
    nearest = np.zeros(len(Xq))
    mean_topk = np.zeros(len(Xq))

    for start in range(0, len(Xq), chunk_size):
        end = min(start + chunk_size, len(Xq))
        D = ((Xq[start:end, None, :] - Xd[None, :, :]) ** 2).sum(axis=2)
        kk = min(k, D.shape[1])

        idx = np.argpartition(D, kth=kk - 1, axis=1)[:, :kk]
        dsel = np.take_along_axis(D, idx, axis=1)
        order = np.argsort(dsel, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        dsel = np.take_along_axis(dsel, order, axis=1)

        pred[start:end] = Y_db[idx].mean(axis=1)
        nearest[start:end] = np.sqrt(dsel[:, 0])
        mean_topk[start:end] = np.sqrt(dsel).mean(axis=1)

    return pred, nearest, mean_topk


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    X_db, Y_db = load_db(Path(args.db))
    df = pd.read_csv(input_csv)

    x_cols, local_cols, cloud_cols = columns()
    required = x_cols + local_cols + cloud_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Input CSV missing columns: {missing}")

    numeric = df[required].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy()).all(axis=1)

    Xq = numeric.loc[finite, x_cols].to_numpy(float)
    scale = np.array([args.q_scale] * 7 + [args.dq_scale] * 7, dtype=float)

    hist_pred = np.full((len(df), 7), np.nan)
    nearest = np.full(len(df), np.nan)
    mean_topk = np.full(len(df), np.nan)

    pred_clean, nearest_clean, mean_topk_clean = knn_query(
        X_db,
        Y_db,
        Xq,
        args.k,
        scale,
        args.chunk_size,
    )

    finite_idx = np.nonzero(finite)[0]
    hist_pred[finite_idx] = pred_clean
    nearest[finite_idx] = nearest_clean
    mean_topk[finite_idx] = mean_topk_clean

    is_online = "online" in str(input_csv).lower()
    available = np.isfinite(nearest) & (nearest <= args.max_distance)
    if args.disable_historical_for_online and is_online:
        available[:] = False

    local = df[local_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    cloud = df[cloud_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    fallback = cloud

    gated = np.where(available[:, None], hist_pred, fallback)

    out = df.copy()
    for j in range(1, 8):
        out[f"hist_db_pred_{j}"] = hist_pred[:, j - 1]
        out[f"hist_db_gated_pred_{j}"] = gated[:, j - 1]

    out["hist_db_nearest_distance"] = nearest
    out["hist_db_mean_topk_distance"] = mean_topk
    out["hist_db_available"] = available.astype(int)
    out["hist_db_gated_source_code"] = np.where(available, 2, 1)
    out["hist_db_k"] = args.k
    out["hist_db_max_distance"] = args.max_distance
    out["hist_db_online_disabled"] = int(args.disable_historical_for_online and is_online)

    out.to_csv(output_csv, index=False)

    print("===== GOAL1 historical DB shadow CSV written =====")
    print("input:", input_csv)
    print("output:", output_csv)
    print("rows:", len(out))
    print("hist_available_count:", int(available.sum()))
    print("hist_available_ratio:", float(available.mean()))
    print("nearest_median:", float(np.nanmedian(nearest)))
    print("nearest_max:", float(np.nanmax(nearest)))
    print("online_detected:", is_online)
    print("online_disabled:", bool(args.disable_historical_for_online and is_online))


if __name__ == "__main__":
    main()
