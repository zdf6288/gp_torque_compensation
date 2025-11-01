#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import os
import sys
from typing import Tuple, Dict

def load_and_preprocess(csv_path: str, decimate: int, smooth_win: int, duration: float
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    读取CSV -> 子采样 -> 平滑 -> 截取前 duration 秒 -> 计算误差
    返回:
      t: 归一化时间(从0开始), shape [N]
      e_xyz: 三轴误差 [ex,ey,ez], shape [N,3]
      e_norm: 误差模长, shape [N]
      stats: 统计指标字典
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 基础列检查
    needed = ['Time(s)', 'x_actual', 'y_actual', 'z_actual',
              'x_desired', 'y_desired', 'z_desired']
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"{csv_path} 缺少列: {c}")

    # 子采样
    if decimate > 1:
        df = df.iloc[::decimate, :].reset_index(drop=True)

    # 平滑（滚动平均）
    if smooth_win > 1:
        df = df.rolling(window=smooth_win, center=True).mean().dropna().reset_index(drop=True)

    # 时间归一化并截取前 duration 秒
    t = df['Time(s)'].to_numpy(dtype=float)
    t0 = np.nanmin(t)
    t = t - t0
    mask = (t <= duration)
    if mask.sum() < 2:
        # 若采样太稀，则不截取，直接用全部
        mask = slice(None, None, None)

    t = t[mask]
    xa = df['x_actual'].to_numpy(dtype=float)[mask]
    ya = df['y_actual'].to_numpy(dtype=float)[mask]
    za = df['z_actual'].to_numpy(dtype=float)[mask]
    xd = df['x_desired'].to_numpy(dtype=float)[mask]
    yd = df['y_desired'].to_numpy(dtype=float)[mask]
    zd = df['z_desired'].to_numpy(dtype=float)[mask]

    ex = xa - xd
    ey = ya - yd
    ez = za - zd
    e_xyz = np.stack([ex, ey, ez], axis=1)
    e_norm = np.linalg.norm(e_xyz, axis=1)

    # 统计
    def nanstats(arr):
        return dict(
            RMS=float(np.sqrt(np.nanmean(arr**2))),
            Mean=float(np.nanmean(arr)),
            Max =float(np.nanmax(np.abs(arr))),
            P95 =float(np.nanpercentile(np.abs(arr), 95))
        )

    stats = {
        'norm': nanstats(e_norm),
        'ex': nanstats(ex),
        'ey': nanstats(ey),
        'ez': nanstats(ez),
        'N': len(t)
    }
    return t, e_xyz, e_norm, stats


def pretty_stats(name: str, stats: Dict[str, Dict[str, float]]):
    print(f"\n=== {name} ===")
    print(f"Samples: {stats['N']}")
    def line(tag, s):
        print(f"{tag:>6} | RMS={s['RMS']:.6f}  Mean={s['Mean']:.6f}  Max={s['Max']:.6f}  P95={s['P95']:.6f}")
    line('‖e‖', stats['norm'])
    line('ex ', stats['ex'])
    line('ey ', stats['ey'])
    line('ez ', stats['ez'])


def main():
    ap = argparse.ArgumentParser(description="Compare tracking error (with GP vs no GP) over first T seconds.")
    ap.add_argument('csv_gp',   help='CSV with GP compensation')
    ap.add_argument('csv_nogp', help='CSV without GP compensation')
    ap.add_argument('--duration', type=float, default=15.0, help='seconds to compare (default: 15)')
    ap.add_argument('--decimate', type=int, default=5, help='decimation factor (default: 5)')
    ap.add_argument('--smooth',   type=int, default=10, help='moving average window (default: 10)')
    ap.add_argument('--title',    type=str, default='Tracking Error Comparison (first {:.0f}s)',
                    help='figure title (supports one {:.0f} for duration)')
    args = ap.parse_args()

    try:
        t_gp,  e_xyz_gp,  e_norm_gp,  stats_gp  = load_and_preprocess(args.csv_gp,   args.decimate, args.smooth, args.duration)
        t_ngp, e_xyz_ngp, e_norm_ngp, stats_ngp = load_and_preprocess(args.csv_nogp, args.decimate, args.smooth, args.duration)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 打印统计
    pretty_stats("WITH GP", stats_gp)
    pretty_stats("NO GP ", stats_ngp)

    # 误差模长对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(args.title.format(args.duration), fontsize=16)

    ax0 = axes[0,0]
    ax0.plot(t_ngp, e_norm_ngp, label='No GP', linewidth=1.8)
    ax0.plot(t_gp,  e_norm_gp,  label='With GP', linestyle='--', linewidth=1.8)
    ax0.set_title('‖Position Error‖ (Euclidean)')
    ax0.set_xlabel('Time (s)'); ax0.set_ylabel('Meters')
    ax0.grid(True); ax0.legend()

    # XYZ 逐轴对比（X）
    ax1 = axes[0,1]
    ax1.plot(t_ngp, e_xyz_ngp[:,0], label='No GP', linewidth=1.5)
    ax1.plot(t_gp,  e_xyz_gp[:,0],  label='With GP', linestyle='--', linewidth=1.5)
    ax1.set_title('X-axis Error'); ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Meters')
    ax1.grid(True); ax1.legend()

    # Y
    ax2 = axes[1,0]
    ax2.plot(t_ngp, e_xyz_ngp[:,1], label='No GP', linewidth=1.5)
    ax2.plot(t_gp,  e_xyz_gp[:,1],  label='With GP', linestyle='--', linewidth=1.5)
    ax2.set_title('Y-axis Error'); ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Meters')
    ax2.grid(True); ax2.legend()

    # Z
    ax3 = axes[1,1]
    ax3.plot(t_ngp, e_xyz_ngp[:,2], label='No GP', linewidth=1.5)
    ax3.plot(t_gp,  e_xyz_gp[:,2],  label='With GP', linestyle='--', linewidth=1.5)
    ax3.set_title('Z-axis Error'); ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Meters')
    ax3.grid(True); ax3.legend()

    for ax in [ax0, ax1, ax2, ax3]:
        ax.relim(); ax.autoscale_view()

    plt.tight_layout()
    out_png = 'gp_vs_nogp_tracking_compare.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved as {out_png}")
    plt.show()


if __name__ == '__main__':
    main()
