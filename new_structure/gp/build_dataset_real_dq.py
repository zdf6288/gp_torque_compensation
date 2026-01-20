#!/usr/bin/env python3
import numpy as np, pandas as pd, glob, argparse
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import os

def load_csvs(pattern="*.csv"):
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No CSV matched pattern: {pattern}")
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna().reset_index(drop=True)
    return df

def apply_decimate_and_smooth(df, decimate=5, smooth=10):
    """先下采样再滚动均值平滑；只处理数值列。"""
    # 下采样
    if decimate and decimate > 1:
        df = df.iloc[::decimate, :].reset_index(drop=True)

    # 平滑（滚动均值）
    if smooth and smooth > 1:
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].rolling(window=smooth, center=True).mean()
        df = df.dropna().reset_index(drop=True)

    return df


def build_xy_no_filter(
    df,
    dt=0.001,
    use_vel=False,           # False: [q, ddq_des]; True: [q, dq_des, ddq_des]
):
    """
    不对 y 进行任何滤波或中值去尖。
    X: [q, dq_des_joint, ddq_des_joint] 或 [q, ddq_des_joint]
    Y: tau_measured - gravity - tau_cmd (残差力矩)
    """
    X_list, Y_list = [], []
    C_list, M_list, G_list = [], [], []

    required = [
        "joint_pos_", "tau_", "tau_measured_", "gravity_",
        "dq_des_joint_", "ddq_des_joint_"
    ]
    for pref in required:
        for j in range(1, 8):
            col = f"{pref}{j}"
            if col not in df.columns:
                raise KeyError(f"缺少列: {col}")

    for j in range(1, 8):
        q        = df[f"joint_pos_{j}"].values.astype(float)
        dq_des   = df[f"dq_des_joint_{j}"].values.astype(float)
        ddq_des  = df[f"ddq_des_joint_{j}"].values.astype(float)
        tau_cmd  = df[f"tau_{j}"].values.astype(float)
        tau_meas = df[f"tau_measured_{j}"].values.astype(float)
        g        = df[f"gravity_{j}"].values.astype(float)

        # 残差力矩 (未滤波)
        y = tau_meas - g - tau_cmd

        # 特征
        X = np.stack([q, dq_des, ddq_des], axis=1) if use_vel else np.stack([q, ddq_des], axis=1)

        X_list.append(X.astype(np.float32))
        Y_list.append(y.astype(np.float32)[:, None])
        C_list.append(tau_cmd.astype(np.float32))
        M_list.append(tau_meas.astype(np.float32))
        G_list.append(g.astype(np.float32))

    return X_list, Y_list, C_list, M_list, G_list

def build_xy_full_input(
    df,
    dt=0.001,
    use_ddq=True,   # True: 使用 21 维输入；False: 只使用 q + dq → 14 维
):
    """
    每个关节都使用相同的高维输入 x_full:
        use_ddq=False: X = [q1..q7, dq1..dq7]              → 14 维
        use_ddq=True:  X = [q1..q7, dq1..dq7, ddq1..ddq7]  → 21 维

    注意：这里 dq 使用真实 joint_vel，而不是 dq_des_joint
    """

    X_list = [[] for _ in range(7)]
    Y_list = [[] for _ in range(7)]

    # -----------------------
    # 读取全关节的数据
    # -----------------------
    q_mat      = np.stack([df[f"joint_pos_{j}"].values       for j in range(1,8)], axis=1)
    dq_mat     = np.stack([df[f"joint_vel_{j}"].values       for j in range(1,8)], axis=1)  # ← 改为真实 dq
    ddq_mat    = np.stack([df[f"ddq_des_joint_{j}"].values   for j in range(1,8)], axis=1)
    tau_cmd    = np.stack([df[f"tau_{j}"].values             for j in range(1,8)], axis=1)
    tau_meas   = np.stack([df[f"tau_measured_{j}"].values    for j in range(1,8)], axis=1)
    g_mat      = np.stack([df[f"gravity_{j}"].values         for j in range(1,8)], axis=1)
    tau_residual = np.stack([df[f"tau_residual_{j}"].values         for j in range(1,8)], axis=1)

    # -----------------------
    # 构造 X_full
    # -----------------------
    if use_ddq:
        X_full = np.concatenate([q_mat, dq_mat, ddq_mat], axis=1)   # (N, 21)
    else:
        X_full = np.concatenate([q_mat, dq_mat], axis=1)            # (N, 14)

    X_full = X_full.astype(np.float32)

    # -----------------------
    # 输出 y
    # -----------------------
    Y_full = (tau_meas - g_mat - tau_cmd).astype(np.float32)
    # Y_full = (tau_residual).astype(np.float32)

    # -----------------------
    # 按关节构造 X_j, Y_j
    # -----------------------
    for j in range(7):
        X_list[j] = X_full
        Y_list[j] = Y_full[:, j][:, None]

    return X_list, Y_list



def save_per_joint_plots(X_list, Y_list, out_npz_path, use_vel=False):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    out_dir = os.path.dirname(out_npz_path) or "."
    prefix = os.path.splitext(os.path.basename(out_npz_path))[0]

    for j in range(7):
        X = X_list[j]
        Y = Y_list[j][:, 0]

        if X.shape[0] < 5:
            print(f"[warn] joint {j+1}: not enough samples ({X.shape[0]}) -> skip plot")
            continue

        if use_vel:
            q, dq_des, ddq_des = X[:, 0], X[:, 1], X[:, 2]
            ncols = 3
        else:
            q, ddq_des = X[:, 0], X[:, 1]
            ncols = 2

        fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 4))
        axes = np.atleast_1d(axes)

        def _scatter_fit(ax, x, y, xlab, title_prefix):
            ax.scatter(x, y, s=8, alpha=0.5)
            try:
                A = np.vstack([x, np.ones_like(x)]).T
                coef = np.linalg.lstsq(A, y, rcond=None)[0]
                a, b = float(coef[0]), float(coef[1])
                xfit = np.linspace(np.min(x), np.max(x), 200)
                ax.plot(xfit, a*xfit + b, linewidth=2, label=f'y={a:.3f}x+{b:.3f}')
            except Exception:
                pass
            ax.set_title(f'Joint {j+1}: {title_prefix}')
            ax.set_xlabel(xlab)
            ax.set_ylabel('Residual torque y [Nm]')
            ax.grid(True)
            ax.legend()

        _scatter_fit(axes[0], q, Y, 'q [rad]', 'q vs y')
        _scatter_fit(axes[1], ddq_des, Y, 'ddq_des [rad/s²]', 'ddq_des vs y')
        if use_vel and ncols == 3:
            _scatter_fit(axes[2], dq_des, Y, 'dq_des [rad/s]', 'dq_des vs y')

        plt.tight_layout()
        out_png = os.path.join(out_dir, f"{prefix}_joint{j+1}.png")
        fig.savefig(out_png, dpi=220, bbox_inches='tight')
        plt.close(fig)
        print(f"🖼 saved {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Build per-joint GP dataset without filtering")
    ap.add_argument("--pattern", default="cartesian_impedance_controller_data*.csv",
                    help="glob pattern of CSV files")
    ap.add_argument("--dt", type=float, default=0.001, help="original sample period (s)")
    ap.add_argument("--decimate", type=int, default=5, help="decimation factor (>=1)")
    ap.add_argument("--smooth", type=int, default=10, help="moving average window (>=1)")
    ap.add_argument("--use_vel", action="store_true", help="use [q, dq, ddq] as inputs")
    ap.add_argument("--out", default="gp_train_data_per_joint_no_filter.npz", help="output npz path")
    ap.add_argument("--plots-per-joint", action="store_true",
                    help="save 7 per-joint scatter plots")

    args = ap.parse_args()

    # 1) 读取并预处理
    df = load_csvs(args.pattern)
    df = apply_decimate_and_smooth(df, decimate=args.decimate, smooth=args.smooth)

    eff_dt = args.dt * (args.decimate if args.decimate and args.decimate > 1 else 1)

    # # 2) 构建未滤波数据
    # X_list, Y_list, C_list, M_list, G_list = build_xy_no_filter(
    #     df, dt=eff_dt, use_vel=args.use_vel
    # )

    X_list, Y_list = build_xy_full_input(
        df,
        dt=eff_dt,
        use_ddq=False  # 想14维就改成 False
    )


    # np.savez(
    #     args.out,
    #     **{f"X{j}": X_list[j-1] for j in range(1, 8)},
    #     **{f"Y{j}": Y_list[j-1] for j in range(1, 8)},
    #     **{f"C{j}": C_list[j-1] for j in range(1, 8)},
    #     **{f"M{j}": M_list[j-1] for j in range(1, 8)},
    #     **{f"G{j}": G_list[j-1] for j in range(1, 8)},
    #     meta=np.array({
    #         "decimate": args.decimate,
    #         "smooth": args.smooth,
    #         "eff_dt": eff_dt,
    #         "use_vel": args.use_vel
    #     }, dtype=object)
    # )

    np.savez(
    args.out,
    **{f"X{j}": X_list[j-1] for j in range(1, 8)},
    **{f"Y{j}": Y_list[j-1] for j in range(1, 8)},
    meta=np.array({
        "decimate": args.decimate,
        "smooth": args.smooth,
        "eff_dt": eff_dt,
        "input_dim": X_list[0].shape[1],
    }, dtype=object)
)

    print(f"✅ Saved {args.out} (no filtering applied).")

    # if args.plots_per_joint:
    #     save_per_joint_plots(X_list, Y_list, args.out, use_vel=args.use_vel)


if __name__ == "__main__":
    main()
