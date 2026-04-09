#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import copy
import pickle
import argparse
import importlib.util
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Utility
# ============================================================

def lowpass_vector(x_raw, x_prev, dt, cutoff_hz):
    if cutoff_hz <= 0.0:
        return x_raw.copy()

    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    alpha = dt / (tau + dt)
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha * x_raw + (1.0 - alpha) * x_prev


def sample_rollout_times_uniform(Td, n, span):
    if n <= 1:
        return np.array([Td], dtype=float)

    t_min = max(0.0, Td - 0.5 * span)
    t_max = max(t_min, Td + 0.5 * span)
    return np.linspace(t_min, t_max, n, dtype=float)


def ensure_skygp_import(skygp_path):
    if not os.path.isfile(skygp_path):
        raise FileNotFoundError(f"skygp.py not found: {skygp_path}")

    if "skygp" in sys.modules:
        return

    spec = importlib.util.spec_from_file_location("skygp", skygp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["skygp"] = mod
    spec.loader.exec_module(mod)


def load_gp_models(model_dir, skygp_path, kind="small"):
    """
    kind:
        - "small": 对应 local/small GP 配置
        - "big":   对应 cloud/big GP 配置
    """
    ensure_skygp_import(skygp_path)

    if kind == "small":
        per_joint_cfg = {
            "default": dict(
                max_data_per_expert=50,
                nearest_k=1,
                max_experts=1,
                timescale=0.03,
            ),
            6: dict(
                max_data_per_expert=50,
                nearest_k=1,
                max_experts=1,
                timescale=0.05,
            ),
        }
    elif kind == "big":
        per_joint_cfg = {
            "default": dict(
                max_data_per_expert=50,
                nearest_k=2,
                max_experts=50,
                timescale=0.03,
            ),
            6: dict(
                max_data_per_expert=50,
                nearest_k=2,
                max_experts=50,
                timescale=0.05,
            ),
        }
    else:
        raise ValueError(f"Unknown kind: {kind}")

    models = {}
    loaded = 0

    for j in range(1, 8):
        p = os.path.join(model_dir, f"joint{j}_local.pkl")
        if not os.path.isfile(p):
            print(f"[WARN] model not found: {p}")
            continue

        with open(p, "rb") as f:
            pack = pickle.load(f)

        model = pack["model"]
        stats = pack["stats"]
        Xm, Xs, Ym, Ys = stats
        x_dim = int(len(Xm))

        cfg = per_joint_cfg.get(j, per_joint_cfg["default"])
        if hasattr(model, "max_data_per_expert"):
            model.max_data_per_expert = int(cfg["max_data_per_expert"])
        if hasattr(model, "nearest_k"):
            model.nearest_k = int(cfg["nearest_k"])
        if hasattr(model, "max_experts"):
            model.max_experts = int(cfg["max_experts"])
        if hasattr(model, "timescale"):
            model.timescale = float(cfg["timescale"])

        models[j] = {
            "model": model,
            "stats": stats,
            "x_dim": x_dim,
        }
        loaded += 1
        print(f"[INFO] loaded {kind} joint{j}, x_dim={x_dim}")

    if loaded == 0:
        raise RuntimeError(f"No {kind} GP models loaded from {model_dir}")

    return models


def gp_predict_and_update(q, dq_feature, ddq_feature, tau_residual, models, use_gp=True, update=True):
    """
    对齐你当前控制器的输入方式：
        x_full = [q, dq_feature]
    当前脚本默认还是 14 维输入。
    """
    if not use_gp:
        return np.zeros(7, dtype=float), np.ones(7, dtype=float) * 1e6

    y_hat = np.zeros(7, dtype=float)
    y_var = np.ones(7, dtype=float) * 1e6

    x_full = np.concatenate([q, dq_feature]).astype(np.float32)

    for j in range(1, 8):
        pack = models.get(j)
        if pack is None:
            continue

        model = pack["model"]
        Xm, Xs, Ym, Ys = pack["stats"]
        x_dim = pack["x_dim"]

        Xm = np.asarray(Xm, dtype=np.float32)
        Xs = np.asarray(Xs, dtype=np.float32)
        Xs = np.where(np.abs(Xs) < 1e-8, 1.0, Xs)

        Ym = float(Ym[0])
        Ys = float(Ys[0]) if float(Ys[0]) != 0.0 else 1.0

        x_std = (x_full[:x_dim] - Xm[:x_dim]) / Xs[:x_dim]

        mu_std, var_std = model.predict(x_std.astype(np.float32))
        mu_std = float(mu_std[0])
        var_std = float(var_std[0])

        y_hat[j - 1] = mu_std * Ys + Ym
        y_var[j - 1] = max(var_std * (Ys ** 2), 1e-8)

        if update:
            y_real = float(tau_residual[j - 1])
            y_std = (y_real - Ym) / Ys
            if np.isfinite(y_std):
                model.add_point(
                    x_std.astype(np.float32),
                    np.array([y_std], dtype=np.float32)
                )

    return y_hat, y_var


def compute_metrics(y_pred, y_true, prefix=""):
    err = y_pred - y_true
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    out = {}
    for j in range(len(mae)):
        out[f"{prefix}mae_j{j+1}"] = mae[j]
        out[f"{prefix}rmse_j{j+1}"] = rmse[j]
    out[f"{prefix}mae_mean"] = float(np.mean(mae))
    out[f"{prefix}rmse_mean"] = float(np.mean(rmse))
    return out

def plot_joint_comparison_grid(times, y_true, y_local, y_cloud, y_fused,
                               save_path="offline_gp_plots/all_joints_grid.png",
                               num_joints=6):
    fig, axes = plt.subplots(num_joints, 1, figsize=(14, 3 * num_joints), sharex=True)

    if num_joints == 1:
        axes = [axes]

    for j in range(num_joints):
        ax = axes[j]
        ax.plot(times, y_true[:, j], label="true")
        ax.plot(times, y_local[:, j], label="local")
        ax.plot(times, y_cloud[:, j], label="cloud")
        ax.set_ylabel(f"J{j+1}")
        ax.grid(True)

        if j == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Offline GP Replay Comparison", fontsize=14)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] saved grid plot to: {save_path}")
# ============================================================
# Main replay
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="controller 保存的 csv")
    parser.add_argument("--model_dir", required=True, help="joint*_local.pkl 所在目录")
    parser.add_argument("--skygp", required=True, help="skygp.py 路径")
    parser.add_argument("--out_csv", default="offline_gp_replay_out.csv")
    parser.add_argument("--delay_steps", type=int, default=1)
    parser.add_argument("--cloud_rollout_n", type=int, default=5)
    parser.add_argument("--cloud_rollout_span", type=float, default=0.02)
    parser.add_argument("--dq_lpf_hz", type=float, default=30.0)
    parser.add_argument("--ddq_lpf_hz", type=float, default=15.0)
    parser.add_argument("--var_floor", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    # 两套模型分开加载，避免互相 add_point 污染
    gp_small = load_gp_models(args.model_dir, args.skygp, kind="small")
    gp_big = load_gp_models(args.model_dir, args.skygp, kind="big")

    df = pd.read_csv(args.csv)
    n = len(df)
    print(f"[INFO] loaded csv rows = {n}")

    q_cols = [f"joint_pos_{i}" for i in range(1, 8)]
    dq_cols = [f"joint_vel_{i}" for i in range(1, 8)]
    tau_res_cols = [f"tau_residual_{i}" for i in range(1, 8)]

    time_col = "Time(s)"
    if time_col not in df.columns:
        raise KeyError(f"{time_col} not found in CSV")

    for c in q_cols + dq_cols + tau_res_cols:
        if c not in df.columns:
            raise KeyError(f"{c} not found in CSV")

    times = df[time_col].to_numpy(dtype=float)
    qs = df[q_cols].to_numpy(dtype=float)
    dqs_raw = df[dq_cols].to_numpy(dtype=float)
    tau_res_all = df[tau_res_cols].to_numpy(dtype=float)

    # 输出缓存
    y_local_hist = []
    y_cloud_hist = []
    y_fused_hist = []
    var_local_hist = []
    var_cloud_hist = []

    # 状态缓存：和线上 controller 一样，用于取 base_state
    state_buffer = deque(maxlen=5000)

    # 速度/加速度滤波状态
    dq_filt_prev = dqs_raw[0].copy()
    dq_filt_initialized = False

    dq_prev = None
    ddq_est_prev = np.zeros(7, dtype=float)
    ddq_est_initialized = False

    for i in range(n):
        t = times[i]
        q = qs[i].copy()
        dq_raw = dqs_raw[i].copy()
        tau_res = tau_res_all[i].copy()

        if i == 0:
            dt = 1e-3
        else:
            dt = max(times[i] - times[i - 1], 1e-6)

        # dq low-pass
        if not dq_filt_initialized:
            dq = dq_raw.copy()
            dq_filt_prev = dq.copy()
            dq_filt_initialized = True
        else:
            dq = lowpass_vector(dq_raw, dq_filt_prev, dt, args.dq_lpf_hz)
            dq_filt_prev = dq.copy()

        # ddq estimate
        if dq_prev is None:
            ddq_est = np.zeros(7, dtype=float)
            dq_prev = dq.copy()
            ddq_est_prev = ddq_est.copy()
            ddq_est_initialized = True
        else:
            ddq_raw = (dq - dq_prev) / dt
            dq_prev = dq.copy()

            if not ddq_est_initialized:
                ddq_est = ddq_raw.copy()
                ddq_est_initialized = True
            else:
                ddq_est = lowpass_vector(ddq_raw, ddq_est_prev, dt, args.ddq_lpf_hz)
            ddq_est_prev = ddq_est.copy()

        # 先存当前帧，和你线上代码一致
        state_buffer.append({
            "t": t,
            "q": q.copy(),
            "dq": dq.copy(),
            "ddq_est": ddq_est.copy(),
            "tau_res": tau_res.copy(),
        })

        # 预热前几帧不做复杂 rollout
        if i < args.warmup:
            y_local = np.zeros(7, dtype=float)
            var_local = np.ones(7, dtype=float) * 1e6
            y_cloud = np.zeros(7, dtype=float)
            var_cloud = np.ones(7, dtype=float) * 1e6
            y_fused = np.zeros(7, dtype=float)
        else:
            # -------------------------
            # 1) small/local GP
            # -------------------------
            y_local, var_local = gp_predict_and_update(
                q, dq, ddq_est, tau_res,
                gp_small,
                use_gp=True,
                update=True
            )

            # -------------------------
            # 2) big/cloud GP
            # -------------------------
            delay_steps = max(1, int(args.delay_steps))

            if len(state_buffer) > delay_steps:
                base_state = state_buffer[-(delay_steps + 1)]
            else:
                base_state = state_buffer[0]

            q_base = base_state["q"].copy()
            dq_base = base_state["dq"].copy()
            ddq_base = base_state["ddq_est"].copy()
            tau_base = base_state["tau_res"].copy()

            # big GP 先用基准帧 update
            gp_predict_and_update(
                q_base, dq_base, ddq_base, tau_base,
                gp_big,
                use_gp=True,
                update=True
            )

            # 在 Td 附近做 rollout 采样
            Td_center = delay_steps * dt
            Td_samples = sample_rollout_times_uniform(
                Td_center,
                args.cloud_rollout_n,
                args.cloud_rollout_span
            )

            y_list = []
            var_list = []

            for Td_i in Td_samples:
                q_roll = q_base + dq_base * Td_i + 0.5 * ddq_base * (Td_i ** 2)
                dq_roll = dq_base + ddq_base * Td_i
                ddq_roll = ddq_base.copy()

                y_i, v_i = gp_predict_and_update(
                    q_roll, dq_roll, ddq_roll, tau_base,
                    gp_big,
                    use_gp=True,
                    update=False
                )
                y_list.append(y_i.copy())
                var_list.append(v_i.copy())

            y_arr = np.asarray(y_list, dtype=float)      # (N, 7)
            var_arr = np.asarray(var_list, dtype=float)  # (N, 7)

            # 方差加权融合
            prec_arr = 1.0 / np.maximum(var_arr, args.var_floor)
            w_arr = prec_arr / np.sum(prec_arr, axis=0, keepdims=True)

            y_cloud = np.sum(y_arr * w_arr, axis=0)
            var_cloud = 1.0 / np.maximum(np.sum(prec_arr, axis=0), 1e-8)

            # -------------------------
            # 3) local/cloud 再融合
            # -------------------------
            v_l = np.maximum(var_local, 1e-8)
            v_c = np.maximum(var_cloud, 1e-8)
            prec_l = 1.0 / v_l
            prec_c = 1.0 / v_c
            w_l = prec_l / (prec_l + prec_c)

            y_fused = w_l * y_local + (1.0 - w_l) * y_cloud

        y_local_hist.append(y_local.copy())
        y_cloud_hist.append(y_cloud.copy())
        y_fused_hist.append(y_fused.copy())
        var_local_hist.append(var_local.copy())
        var_cloud_hist.append(var_cloud.copy())

    y_local_hist = np.asarray(y_local_hist)
    y_cloud_hist = np.asarray(y_cloud_hist)
    y_fused_hist = np.asarray(y_fused_hist)
    var_local_hist = np.asarray(var_local_hist)
    var_cloud_hist = np.asarray(var_cloud_hist)

    # ============================================================
    # Save output
    # ============================================================
    out_df = df.copy()

    for j in range(7):
        out_df[f"offline_y_hat_local_{j+1}"] = y_local_hist[:, j]
        out_df[f"offline_y_hat_cloud_{j+1}"] = y_cloud_hist[:, j]
        out_df[f"offline_y_hat_fused_{j+1}"] = y_fused_hist[:, j]
        out_df[f"offline_var_local_{j+1}"] = var_local_hist[:, j]
        out_df[f"offline_var_cloud_{j+1}"] = var_cloud_hist[:, j]

    out_df.to_csv(args.out_csv, index=False)
    print(f"[INFO] saved replay results to {args.out_csv}")

    # ============================================================
    # Metrics
    # ============================================================
    metrics_local = compute_metrics(y_local_hist, tau_res_all, prefix="local_")
    metrics_cloud = compute_metrics(y_cloud_hist, tau_res_all, prefix="cloud_")
    metrics_fused = compute_metrics(y_fused_hist, tau_res_all, prefix="fused_")

    print("\n===== Offline Replay Metrics =====")
    print(f"local mae_mean : {metrics_local['local_mae_mean']:.6f}")
    print(f"local rmse_mean: {metrics_local['local_rmse_mean']:.6f}")
    print(f"cloud mae_mean : {metrics_cloud['cloud_mae_mean']:.6f}")
    print(f"cloud rmse_mean: {metrics_cloud['cloud_rmse_mean']:.6f}")
    print(f"fused mae_mean : {metrics_fused['fused_mae_mean']:.6f}")
    print(f"fused rmse_mean: {metrics_fused['fused_rmse_mean']:.6f}")

    plot_joint_comparison_grid(
        times=times,
        y_true=tau_res_all,
        y_local=y_local_hist,
        y_cloud=y_cloud_hist,
        y_fused=y_fused_hist,
        save_path="offline_gp_plots/all_joints_grid.png",
        num_joints=6
    )


if __name__ == "__main__":
    main()
    