#!/usr/bin/env python3
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt


def load_gp_log(csv_path):
    """
    读取 gp_debug_log.csv
    列结构（按你现在的代码）：
        time,
        y_slow_1..7,
        y_fast_1..7,
        y_hat_1..7,
        tau_residual_1..7
    """
    times = []
    y_slow = []
    y_fast = []
    y_hat = []
    tau_res = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # 丢掉表头

        for row in reader:
            if not row:
                continue
            # time 是第 0 列
            t = float(row[0])
            times.append(t)

            # 每组 7 维
            y_slow_i = list(map(float, row[1:1+7]))
            y_fast_i = list(map(float, row[1+7:1+7+7]))
            y_hat_i = list(map(float, row[1+7+7:1+7+7+7]))
            tau_i = list(map(float, row[1+7+7+7:1+7+7+7+7]))

            y_slow.append(y_slow_i)
            y_fast.append(y_fast_i)
            y_hat.append(y_hat_i)
            tau_res.append(tau_i)

    times = np.array(times)
    y_slow = np.array(y_slow)        # shape: (N, 7)
    y_fast = np.array(y_fast)
    y_hat = np.array(y_hat)
    tau_res = np.array(tau_res)

    return times, y_slow, y_fast, y_hat, tau_res


def plot_joint(times, y_slow, y_fast, y_hat, tau_res, joint_idx):
    """
    joint_idx: 0..6（关节1..7）
    """
    j = joint_idx
    plt.figure(figsize=(10, 5))
    plt.plot(times, tau_res[:, j], label="tau_residual", linewidth=1.5)
    plt.plot(times, y_slow[:, j], label="y_slow (delayed GP)", linewidth=1.0)
    plt.plot(times, y_fast[:, j], label="y_fast (fast GP)", linewidth=1.0)
    plt.plot(times, y_hat[:, j], label="y_hat (fused)", linewidth=1.5)

    plt.xlabel("time [s]")
    plt.ylabel(f"torque (joint {j+1}) [Nm]")
    plt.title(f"GP compensation vs residual (joint {j+1})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_all_joints(times, y_slow, y_fast, y_hat, tau_res):
    """
    一次性画 7 个关节，每个关节一个 subplot（共享 x 轴）。
    """
    fig, axes = plt.subplots(7, 1, figsize=(10, 14), sharex=True)

    for j in range(7):
        ax = axes[j]
        ax.plot(times, tau_res[:, j], label="tau_residual", linewidth=1.5)
        ax.plot(times, y_slow[:, j], label="y_slow", linewidth=1.0)
        ax.plot(times, y_fast[:, j], label="y_fast", linewidth=1.0)
        ax.plot(times, y_hat[:, j], label="y_hat", linewidth=1.0)

        ax.set_ylabel(f"J{j+1}")
        ax.grid(True)
        if j == 0:
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("GP slow / fast / fused vs tau_residual", y=0.99)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="path to gp_debug_log.csv")
    parser.add_argument(
        "--joint",
        type=int,
        default=None,
        help="joint index (1-7); if not set, plot all joints"
    )
    args = parser.parse_args()

    times, y_slow, y_fast, y_hat, tau_res = load_gp_log(args.csv_path)

    if args.joint is None:
        plot_all_joints(times, y_slow, y_fast, y_hat, tau_res)
    else:
        j = args.joint - 1
        if j < 0 or j > 6:
            raise ValueError("joint must be in [1, 7]")
        plot_joint(times, y_slow, y_fast, y_hat, tau_res, j)


if __name__ == "__main__":
    main()
