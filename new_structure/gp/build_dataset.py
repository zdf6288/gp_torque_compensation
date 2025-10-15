#!/usr/bin/env python3
import numpy as np, pandas as pd, glob, argparse
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import os


from scipy.signal import butter, filtfilt, savgol_filter, medfilt
import numpy as np
import pandas as pd

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

def make_ddq_from_dq(dq_series, dt):
    v = dq_series.values.astype(float)
    ddq = np.zeros_like(v)
    ddq[1:] = (v[1:] - v[:-1]) / dt
    ddq[0] = ddq[1] if len(v) > 1 else 0.0
    # 中值滤波去尖
    k = 5 if len(ddq) >= 5 else (len(ddq) // 2 * 2 + 1)  # 奇数核
    return medfilt(ddq, kernel_size=max(3, k))

def build_xy(df, dt=0.001, use_vel=False, fs=None, lp_tau=6.0, median_k=5):
    X_list, Y_list = [], []
    for j in range(1, 8):
        q  = df[f"joint_pos_{j}"].values
        dq = df.get(f"joint_vel_{j}", pd.Series(np.zeros_like(q))).values
        ddq = make_ddq_from_dq(pd.Series(dq), dt)

        tau_cmd = df[f"tau_{j}"].values
        tau_meas = df[f"tau_measured_{j}"].values
        g = df[f"gravity_{j}"].values

        # 残差力矩
        y = tau_meas - g - tau_cmd

        # 对 y 做去尖 + 低通
        y = median_despike(y, k=median_k)
        if lp_tau and lp_tau > 0 and fs and fs > 0:
            y = butter_lowpass_filtfilt(y, fs=fs, fc=lp_tau, order=4)

        x = np.stack([q, ddq] if not use_vel else [q, dq, ddq], axis=1)

        y_med, y_std = np.median(y), np.std(y) if np.std(y) > 0 else 1.0
        m = np.abs(y - y_med) < 5 * y_std

        X_list.append(x[m].astype(np.float32))
        Y_list.append(y[m].astype(np.float32)[:, None])
    return X_list, Y_list

def butter_lowpass_filtfilt(x, fs, fc, order=4):
    """
    零相位 Butterworth 低通。x: 1D ndarray，fs: 采样频率(Hz)，fc: 截止频率(Hz)
    """
    if fc is None or fc <= 0 or fs <= 0:
        return x
    wn = fc / (0.5 * fs)  # 归一化截止频率
    wn = min(max(wn, 1e-6), 0.999999)
    b, a = butter(order, wn, btype='low', analog=False)
    # filtfilt 需要长度> 3*(max(len(a),len(b))-1)
    if x.size < 3 * (max(len(a), len(b)) - 1) + 1:
        return x
    return filtfilt(b, a, x, method="pad")

def savgol_smooth(x, window=9, poly=3):
    """
    Savitzky–Golay 平滑。window 必须奇数且>= poly+2
    """
    w = int(window)
    if w % 2 == 0:
        w += 1
    w = max(w, poly + 3 if (poly + 3) % 2 == 1 else poly + 4)
    if x.size < w:
        return x
    return savgol_filter(x, window_length=w, polyorder=poly, mode='interp')

def median_despike(x, k=5):
    """中值滤波去尖点（奇数核）"""
    k = int(k)
    if k % 2 == 0:
        k += 1
    if x.size < k:
        return x
    return medfilt(x, kernel_size=k)


def save_per_joint_plots(X_list, Y_list, out_npz_path, use_vel=False):
    """
    为每个关节保存一张图：
      - 左：q vs y（残差力矩），带线性拟合
      - 右：ddq vs y（若 use_vel=True 也额外画 dq vs y）
    """
    out_dir = os.path.dirname(out_npz_path) or "."
    prefix = os.path.splitext(os.path.basename(out_npz_path))[0]

    for j in range(7):
        X = X_list[j]    # shape [N, 2] or [N, 3]
        Y = Y_list[j][:, 0]  # shape [N]

        if X.shape[0] < 5:
            print(f"[warn] joint {j+1}: not enough samples ({X.shape[0]}) -> skip plot")
            continue

        # 拆输入
        if use_vel:
            q, dq, ddq = X[:, 0], X[:, 1], X[:, 2]
        else:
            q, ddq = X[:, 0], X[:, 1]

        fig, axes = plt.subplots(1, 2 + (1 if use_vel else 0), figsize=(12, 4))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        # 1) q vs y
        ax = axes[0]
        ax.scatter(q, Y, s=8, alpha=0.5)
        # 线性拟合
        A = np.vstack([q, np.ones_like(q)]).T
        a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
        xfit = np.linspace(q.min(), q.max(), 200)
        yfit = a * xfit + b
        ax.plot(xfit, yfit, linewidth=2, label=f'fit: y={a:.3f}x+{b:.3f}')
        corr = np.corrcoef(q, Y)[0, 1]
        ax.set_title(f'Joint {j+1}: q vs y (corr={corr:.3f})')
        ax.set_xlabel('q [rad]')
        ax.set_ylabel('Residual torque y [Nm]')
        ax.grid(True)
        ax.legend(loc='best', fontsize=9)

        # 2) ddq vs y
        ax = axes[1]
        ax.scatter(ddq, Y, s=8, alpha=0.5)
        A = np.vstack([ddq, np.ones_like(ddq)]).T
        a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
        xfit = np.linspace(ddq.min(), ddq.max(), 200)
        yfit = a * xfit + b
        ax.plot(xfit, yfit, linewidth=2, label=f'fit: y={a:.3f}x+{b:.3f}')
        corr = np.corrcoef(ddq, Y)[0, 1]
        ax.set_title(f'Joint {j+1}: ddq vs y (corr={corr:.3f})')
        ax.set_xlabel('ddq [rad/s^2]')
        ax.set_ylabel('Residual torque y [Nm]')
        ax.grid(True)
        ax.legend(loc='best', fontsize=9)

        # 3) 可选：dq vs y
        if use_vel:
            ax = axes[2]
            ax.scatter(dq, Y, s=8, alpha=0.5)
            A = np.vstack([dq, np.ones_like(dq)]).T
            a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
            xfit = np.linspace(dq.min(), dq.max(), 200)
            yfit = a * xfit + b
            ax.plot(xfit, yfit, linewidth=2, label=f'fit: y={a:.3f}x+{b:.3f}')
            corr = np.corrcoef(dq, Y)[0, 1]
            ax.set_title(f'Joint {j+1}: dq vs y (corr={corr:.3f})')
            ax.set_xlabel('dq [rad/s]')
            ax.set_ylabel('Residual torque y [Nm]')
            ax.grid(True)
            ax.legend(loc='best', fontsize=9)

        plt.tight_layout()
        out_png = os.path.join(out_dir, f"{prefix}_joint{j+1}.png")
        fig.savefig(out_png, dpi=220, bbox_inches='tight')
        plt.close(fig)
        print(f"🖼 saved {out_png}")
        
def main():
    ap = argparse.ArgumentParser(description="Build per-joint GP dataset with decimation + smoothing")
    ap.add_argument("--pattern", default="cartesian_impedance_controller_data*.csv",
                    help="glob pattern of CSV files")
    ap.add_argument("--dt", type=float, default=0.001, help="original sample period (s), e.g., 0.001 for 1 kHz")
    ap.add_argument("--decimate", type=int, default=5, help="decimation factor (>=1); 5 -> 1/5 samples")
    ap.add_argument("--smooth", type=int, default=10, help="moving average window (>=1)")
    ap.add_argument("--use_vel", action="store_true", help="use [q, dq, ddq] as inputs instead of [q, ddq]")
    ap.add_argument("--out", default="gp_train_data_per_joint.npz", help="output npz path")
    ap.add_argument("--plots-per-joint", action="store_true",
                    help="save 7 per-joint figures after building dataset")
    ap.add_argument("--no-show", action="store_true",
                    help="(reserved) do not show figures (we always save)")
    ap.add_argument("--lp-dq", type=float, default=10.0, help="dq 低通截止频率(Hz), 0=不滤")
    ap.add_argument("--lp-tau", type=float, default=6.0,  help="tau残差低通截止频率(Hz), 0=不滤")
    ap.add_argument("--sg-window", type=int, default=9,   help="Savitzky-Golay 窗口(奇数)")
    ap.add_argument("--sg-poly", type=int, default=3,     help="Savitzky-Golay 多项式阶数")
    ap.add_argument("--median-k", type=int, default=5,    help="中值滤波核(奇数)")

    args = ap.parse_args()

    # 1) 读取并预处理（下采样 + 平滑）
    df = load_csvs(args.pattern)
    df = apply_decimate_and_smooth(df, decimate=args.decimate, smooth=args.smooth)

    # 2) 计算用于 ddq 的 dt（注意：下采样会放大采样周期）
    eff_dt = args.dt * (args.decimate if args.decimate and args.decimate > 1 else 1)

    # 3) 构建 X/Y
    fs = 1.0 / eff_dt
    X_list, Y_list = build_xy(df, dt=eff_dt, use_vel=args.use_vel, fs=fs,
                            lp_tau=args.lp_tau, median_k=args.median_k)

    # 4) 保存
    np.savez(
        args.out,
        **{f"X{j}": X_list[j - 1] for j in range(1, 8)},
        **{f"Y{j}": Y_list[j - 1] for j in range(1, 8)},
        meta=np.array({
            "decimate": args.decimate,
            "smooth": args.smooth,
            "eff_dt": eff_dt,
            "use_vel": args.use_vel
        }, dtype=object)
    )
    print(f"✅ Saved {args.out} | decimate={args.decimate}, smooth={args.smooth}, eff_dt={eff_dt:.6f}s, use_vel={args.use_vel}")
    
    # ... 生成 X_list, Y_list 并保存 npz 之后：
    if args.plots_per_joint:
        save_per_joint_plots(X_list, Y_list, args.out, use_vel=args.use_vel)


if __name__ == "__main__":
    main()
