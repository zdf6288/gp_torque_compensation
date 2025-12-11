#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os


# =============================
#   自动找列名工具
# =============================
def cols_1to7(df, prefix):
    return [f"{prefix}{i}" for i in range(1, 8) if f"{prefix}{i}" in df.columns]


# =============================
#   按圆周分割轨迹
# =============================
def split_by_round(df, frequency):
    """
    根据 Time(s) 和 frequency 将数据切成一圈一圈。
    返回 [(round_idx, df_round), ...]
    """
    if "Time(s)" not in df.columns:
        raise ValueError("CSV 中缺少 Time(s) 列")

    T = 1.0 / frequency
    time = df["Time(s)"].values

    rounds = []
    max_round = int(time[-1] / T) + 1

    for r in range(max_round):
        mask = (time >= r*T) & (time < (r+1)*T)
        df_r = df[mask].copy()

        if len(df_r) > 20:   # 必须有足够数据才算有效
            df_r["Time_round"] = df_r["Time(s)"] - r*T
            rounds.append((r, df_r))

    return rounds


# =============================
#   单圈绘图：Task-space Tracking
# =============================
def plot_round_tracking(df_r, round_idx, output_prefix):
    """
    绘制一个 round 的 task-space tracking 结果：
    x,y,z 的 actual vs desired
    """

    t = df_r["Time(s)"].to_numpy()

    cols_act = ["x_actual", "y_actual", "z_actual"]
    cols_des = ["x_desired", "y_desired", "z_desired"]
    axis_names = ["X", "Y", "Z"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f"Round {round_idx}: Task-Space Tracking", fontsize=14)

    for i in range(3):
        ax = axes[i]

        y_act = df_r[cols_act[i]].to_numpy()
        y_des = df_r[cols_des[i]].to_numpy()

        ax.plot(t, y_act, "b-", label=f"{axis_names[i]} actual")
        ax.plot(t, y_des, "r--", label=f"{axis_names[i]} desired")

        ax.set_ylabel(f"{axis_names[i]} (m)")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time (s)")

    out_name = f"{output_prefix}_round{round_idx}_tracking.png"
    fig.savefig(out_name, dpi=300, bbox_inches="tight")
    print(f"Saved {out_name}")

    plt.close(fig)

# =============================
#   单圈绘图：Torque
# =============================
def plot_round_tau(df_r, round_idx, output_prefix):
    t = df_r["Time_round"].values

    tau_cols = cols_1to7(df_r, "tau_")
    tau_meas_cols = cols_1to7(df_r, "tau_measured_")
    grav_cols = cols_1to7(df_r, "gravity_")

    if not (len(tau_cols)==7 and len(tau_meas_cols)==7 and len(grav_cols)==7):
        print(f"⚠ Round {round_idx}: torque columns missing, skip tau plot.")
        return

    tau = df_r[tau_cols].values
    tau_meas = df_r[tau_meas_cols].values
    grav = df_r[grav_cols].values

    tau_err = tau - (tau_meas - grav)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # ---- tau ----
    ax = axes[0]
    for j in range(7):
        ax.plot(t, tau[:, j], label=f"tau{j+1}")
    ax.set_title(f"Round {round_idx}: τ Command")
    ax.grid(True)
    ax.legend()

    # ---- measured ----
    ax = axes[1]
    for j in range(7):
        ax.plot(t, tau_meas[:, j], label=f"τ_meas{j+1}")
    ax.set_title(f"Round {round_idx}: τ measured")
    ax.grid(True)
    ax.legend()

    # ---- error ----
    ax = axes[2]
    for j in range(7):
        ax.plot(t, tau_err[:, j], label=f"err{j+1}")
    ax.set_title(f"Round {round_idx}: τ_error = τ_cmd - (τ_meas - gravity)")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    outfile = f"{output_prefix}_round{round_idx}_tau.png"
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"Saved {outfile}")


# =============================
#   单圈绘图：Local / Cloud / Combined vs tau_residual
# =============================
def plot_round_gp(df_r, round_idx, output_prefix):
    t = df_r["Time_round"].values

    res_cols = cols_1to7(df_r, "tau_residual_")
    yhat_cols = cols_1to7(df_r, "y_hat_")
    local_cols = cols_1to7(df_r, "y_hat_local_")
    cloud_cols = cols_1to7(df_r, "y_hat_cloud_")

    if len(res_cols) != 7:
        print(f"⚠ Round {round_idx}: tau_residual missing, skip GP plot.")
        return

    R = df_r[res_cols].values
    Yc = df_r[yhat_cols].values if len(yhat_cols)==7 else None
    Yl = df_r[local_cols].values if len(local_cols)==7 else None
    Yf = df_r[cloud_cols].values if len(cloud_cols)==7 else None

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axlist = [ax for row in axes for ax in row]

    for j in range(7):
        ax = axlist[j]
        ax.plot(t, R[:, j], label="tau_residual", linewidth=1.5)

        if Yc is not None:
            ax.plot(t, Yc[:, j], '--', label="combined")
        if Yl is not None:
            ax.plot(t, Yl[:, j], ':', label="local")
        if Yf is not None:
            ax.plot(t, Yf[:, j], '-.', label="cloud")

        ax.set_title(f"Joint {j+1}")
        ax.grid(True)
        ax.legend()

    # remove empty subplots
    axlist[7].axis("off")
    axlist[8].axis("off")

    fig.tight_layout()
    outfile = f"{output_prefix}_round{round_idx}_gp.png"
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"Saved {outfile}")

def plot_all_rounds_one_figure(round_dfs, output_prefix):
    """
    将所有 round 的 X/Y/Z tracking 放入一张大图中对比。
    tracking error 改为 mm。
    """

    colors = plt.cm.tab10(np.linspace(0, 1, len(round_dfs)))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    axes_names = ["X", "Y", "Z"]
    cols_act = ["x_actual", "y_actual", "z_actual"]
    cols_des = ["x_desired", "y_desired", "z_desired"]

    fig.suptitle("All Rounds Tracking Comparison (error in mm)", fontsize=18)

    for idx, df_r in enumerate(round_dfs):
        t = df_r["Time(s)"].to_numpy()
        color = colors[idx]

        for i in range(3):
            ax = axes[i]

            y_act = df_r[cols_act[i]].to_numpy()
            y_des = df_r[cols_des[i]].to_numpy()

            # --- error 改为 mm ---
            y_err_mm = (y_act - y_des) * 1000.0

            # Actual
            ax.plot(
                t, y_act * 1000.0,
                color=color,
                linewidth=1.0,
                alpha=0.7,
                label=f"Round {idx} actual" if i == 0 else None
            )

            # Desired（画一次）
            if idx == 0:
                ax.plot(
                    t, y_des * 1000.0,
                    "k--",
                    linewidth=2.0,
                    label="desired"
                )

            # Error
            ax.plot(
                t, y_err_mm,
                color=color,
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
                label=f"Round {idx} error (mm)" if i == 0 else None
            )

            ax.set_ylabel(f"{axes_names[i]} (mm)")
            ax.grid(True)

    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right", fontsize=10)

    out_name = f"{output_prefix}_ALL_rounds_tracking_mm.png"
    fig.savefig(out_name, dpi=300, bbox_inches="tight")
    print(f"Saved {out_name}")

    plt.close(fig)

def plot_round_mse_summary(round_dfs, output_prefix):
    """
    计算每个 round 的 tracking MSE（mm^2），并画成柱状图。
    """

    mse_x = []
    mse_y = []
    mse_z = []

    for df_r in round_dfs:
        x_err = (df_r["x_actual"] - df_r["x_desired"]).to_numpy() * 1000.0
        y_err = (df_r["y_actual"] - df_r["y_desired"]).to_numpy() * 1000.0
        z_err = (df_r["z_actual"] - df_r["z_desired"]).to_numpy() * 1000.0
        
        mse_x.append(np.mean(x_err**2))
        mse_y.append(np.mean(y_err**2))
        mse_z.append(np.mean(z_err**2))

    rounds = np.arange(len(round_dfs))

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.25

    ax.bar(rounds - width, mse_x, width, label="X MSE (mm²)")
    ax.bar(rounds,         mse_y, width, label="Y MSE (mm²)")
    ax.bar(rounds + width, mse_z, width, label="Z MSE (mm²)")

    ax.set_xlabel("Round Index")
    ax.set_ylabel("MSE (mm²)")
    ax.set_title("Tracking MSE per Round")
    ax.grid(True)
    ax.legend()

    out_name = f"{output_prefix}_MSE_per_round.png"
    fig.savefig(out_name, dpi=300, bbox_inches="tight")
    print(f"Saved {out_name}")

    plt.close(fig)


# =============================
#           MAIN
# =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="CSV file recorded by controller")
    parser.add_argument("--freq", type=float, default=0.1,
                        help="Circular motion frequency (Hz), default=0.1")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"CSV {args.csv_file} not found.")
        return

    df = pd.read_csv(args.csv_file)
    output_prefix = args.csv_file.replace(".csv", "")

    # ---- step 1: 自动分圈 ----
    rounds = split_by_round(df, args.freq)
    print(f"Detected {len(rounds)} rounds")

    if len(rounds) == 0:
        print("❌ No valid rounds detected!")
        return

    # ---- step 2: 不再逐圈绘图，只做一张所有圈的总对比图 ----
    round_dfs_only = [df_r for _, df_r in rounds]

    print("\n====== Plotting ALL rounds together in ONE figure ======\n")
    plot_all_rounds_one_figure(round_dfs_only, output_prefix)

    print("\n====== Plotting MSE summary ======\n")
    plot_round_mse_summary(round_dfs_only, output_prefix)


if __name__ == "__main__":
    main()
