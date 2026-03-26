#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import argparse
import sys
import os
import matplotlib.pyplot as plt
import scienceplots

def cols_1to7(df, prefix):
    # prefix 末尾自己带下划线，如 'tau_', 'tau_measured_', 'gravity_', 'y_hat_', 'tau_residual_'
    return [f'{prefix}{i}' for i in range(1, 8) if f'{prefix}{i}' in df.columns]

def plot_data_from_csv(csv_filename):
    """plot data from CSV file"""
    if not os.path.exists(csv_filename):
        print(f'Error: CSV file {csv_filename} not found')
        return
        
    try:
        df = pd.read_csv(csv_filename)

        # -------------------------------
        # 子采样 + 平滑组合处理
        # -------------------------------

        # 子采样率（越大 → 点越少）
        DECIMATE = 5     # 每5个取一个点  (1000Hz → 200Hz)
        # 平滑窗口大小
        SMOOTH_WINDOW = 1  # 相当于 ~50ms 平滑

        # Step 1: 子采样
        df = df.iloc[::DECIMATE, :].reset_index(drop=True)

        # Step 2: 平滑 (滚动均值)
        df_smooth = df.rolling(window=SMOOTH_WINDOW, center=True).mean()

        # 用平滑后的数据替换原 df，避免后面改很多
        df = df_smooth.dropna().reset_index(drop=True)

        out_prefix = csv_filename.replace('.csv', '')
        # out_prefix = csv_filename.replace('.csv', '')
        # plot_abs_error_pdf(df, out_prefix, use_kde=True, bins=80)

        print(f"✅ Applied decimation (/{DECIMATE}) and smoothing (window={SMOOTH_WINDOW})")
        print(f"Resulting data points: {len(df)}")
        
        time_history = df['Time(s)'].values
        
        
        x_history = df[['x_actual', 'y_actual', 'z_actual']].values
        x_des_history = df[['x_desired', 'y_desired', 'z_desired']].values
        
        dx_history = df[['dx_actual', 'dy_actual', 'dz_actual']].values
        dx_des_history = df[['dx_desired', 'dy_desired', 'dz_desired']].values

        # 严格取 1..7，且顺序正确
        tau_cols = cols_1to7(df, 'tau_')
        meas_cols = cols_1to7(df, 'tau_measured_')
        grav_cols = cols_1to7(df, 'gravity_')

        if len(tau_cols) != 7 or len(meas_cols) != 7 or len(grav_cols) != 7:
            print('Error: expected 7 columns for tau_/tau_measured_/gravity_.')
            return

        tau_history_array          = df[tau_cols].values
        tau_measured_history_array = df[meas_cols].values
        gravity_history_array      = df[grav_cols].values
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        fig.suptitle('Cartesian Impedance Controller Data', fontsize=14)
        
        # plot tau for 7 joints
        for i in range(tau_history_array.shape[1]):
            axes[0, 0].plot(time_history, tau_history_array[:, i], label=f'Joint {i+1}')
        axes[0, 0].set_title('Joint Torques (tau)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Torque (Nm)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # plot desired and actual velocity
        axes[0, 1].plot(time_history, dx_history[:, 0], 'b-', label='Actual dx', linewidth=2)
        axes[0, 1].plot(time_history, dx_des_history[:, 0], 'r--', label='Desired dx', linewidth=2)
        axes[0, 1].plot(time_history, dx_history[:, 1], 'g-', label='Actual dy', linewidth=2)
        axes[0, 1].plot(time_history, dx_des_history[:, 1], 'm--', label='Desired dy', linewidth=2)
        axes[0, 1].plot(time_history, dx_history[:, 2], 'c-', label='Actual dz', linewidth=2)
        axes[0, 1].plot(time_history, dx_des_history[:, 2], 'y--', label='Desired dz', linewidth=2)
        
        axes[0, 1].set_title('Desired vs Actual Velocity')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # plot position error on x, y, z axes separately
        x_errors = []
        y_errors = []
        z_errors = []
        for i in range(len(x_history)):
            actual_pos = x_history[i][:3]
            desired_pos = x_des_history[i][:3]
            x_errors.append(actual_pos[0] - desired_pos[0])
            y_errors.append(actual_pos[1] - desired_pos[1])
            z_errors.append(actual_pos[2] - desired_pos[2])
        
        axes[0, 2].plot(time_history, x_errors, 'r-', label='X Error', linewidth=2)
        axes[0, 2].plot(time_history, y_errors, 'g-', label='Y Error', linewidth=2)
        axes[0, 2].plot(time_history, z_errors, 'b-', label='Z Error', linewidth=2)
        axes[0, 2].set_title('Position Error on X, Y, Z Axes')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Error (m)')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # plot position trajectory on x axis in task space
        axes[1, 0].plot(time_history, x_history[:, 0], 'b-', label='Actual X', linewidth=2)
        axes[1, 0].plot(time_history, x_des_history[:, 0], 'r--', label='Desired X', linewidth=2)
        axes[1, 0].set_title('X Position Trajectory')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Position (m)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].yaxis.set_major_locator(ticker.MultipleLocator(0.02))
        
        # plot position trajectory on y axis in task space
        axes[1, 1].plot(time_history, x_history[:, 1], 'b-', label='Actual Y', linewidth=2)
        axes[1, 1].plot(time_history, x_des_history[:, 1], 'r--', label='Desired Y', linewidth=2)
        axes[1, 1].set_title('Y Position Trajectory')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Position (m)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].yaxis.set_major_locator(ticker.MultipleLocator(0.02))
        
        # plot position trajectory on z axis in task space
        axes[1, 2].plot(time_history, x_history[:, 2], 'b-', label='Actual Z', linewidth=2)
        axes[1, 2].plot(time_history, x_des_history[:, 2], 'r--', label='Desired Z', linewidth=2)
        axes[1, 2].set_title('Z Position Trajectory')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Position (m)')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        axes[1, 2].yaxis.set_major_locator(ticker.MultipleLocator(0.02))
        
        # adjust Z plot y-axis range to match the scale of X and Y plots
        z_data = np.concatenate([x_history[:, 2], x_des_history[:, 2]])
        z_range = np.max(z_data) - np.min(z_data)
        z_center = (np.max(z_data) + np.min(z_data)) / 2
        target_range = 0.1  # similar to X and Y plots range (0.1m)
        z_min = z_center - target_range / 2
        z_max = z_center + target_range / 2
        axes[1, 2].set_ylim(z_min, z_max)
        
        # plot measured joint torques (tau_measured)
        if tau_measured_history_array.size > 0:
            for i in range(tau_measured_history_array.shape[1]):
                axes[2, 0].plot(time_history, tau_measured_history_array[:, i], label=f'Joint {i+1}', linewidth=2)
            
            axes[2, 0].set_title('Measured Joint Torques (tau_measured)')
            axes[2, 0].set_xlabel('Time (s)')
            axes[2, 0].set_ylabel('Torque (Nm)')
            axes[2, 0].legend()
            axes[2, 0].grid(True)
        else:
            axes[2, 0].text(0.5, 0.5, 'No measured torque data available', 
                            ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 0].set_title('Measured Joint Torques (tau_measured)')
        
        # plot gravity compensation
        if gravity_history_array.size > 0:
            for i in range(gravity_history_array.shape[1]):
                axes[2, 1].plot(time_history, gravity_history_array[:, i], label=f'Joint {i+1}', linewidth=2)
            
            axes[2, 1].set_title('Gravity Compensation')
            axes[2, 1].set_xlabel('Time (s)')
            axes[2, 1].set_ylabel('Torque (Nm)')
            axes[2, 1].legend()
            axes[2, 1].grid(True)
        else:
            axes[2, 1].text(0.5, 0.5, 'No gravity data available', 
                            ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('Gravity Compensation')
        
        # plot error between controller output and measured torque minus gravity (all 7 joints)
        if (tau_history_array.size > 0 and tau_measured_history_array.size > 0 and gravity_history_array.size > 0):
            # error: (computed tau - (measured tau - gravity))
            tau_measured_minus_gravity = tau_measured_history_array - gravity_history_array
            error_array = tau_history_array - tau_measured_minus_gravity

            for i in range(error_array.shape[1]):
                axes[2, 2].plot(time_history, error_array[:, i], 
                                label=f'Joint {i+1}', linewidth=2)
            
            axes[2, 2].set_title('Error: Computed tau - (Measured tau - Gravity)')
            axes[2, 2].set_xlabel('Time (s)')
            axes[2, 2].set_ylabel('Torque Error (Nm)')
            axes[2, 2].legend()
            axes[2, 2].grid(True)
            
            mean_errors = np.mean(np.abs(error_array), axis=0)
            max_errors = np.max(np.abs(error_array), axis=0)
            
            print(f'\nTorque Error Statistics (Mean, Max):')
            for i in range(len(mean_errors)):
                print(f'Joint {i+1}: Mean={mean_errors[i]:.4f} Nm, Max={max_errors[i]:.4f} Nm')
        else:
            axes[2, 2].text(0.5, 0.5, 'Insufficient data for error analysis', 
                            ha='center', va='center', transform=axes[2, 2].transAxes)
            axes[2, 2].set_title('Error: Computed tau - (Measured tau - Gravity)')
        
        # auto-scale all axes
        for ax in axes.flat:
            ax.autoscale_view()
            ax.relim()
        
        plt.tight_layout()
        
        output_filename = csv_filename.replace('.csv', '_plot.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f'Figure saved as {output_filename}')
        
        # plt.show()
        
    except Exception as e:
        print(f'Error when plotting data: {str(e)}')

    # ===== 新增：读取关节位置 =====
    joint_pos_cols = [c for c in df.columns if c.startswith('joint_pos_')]
    if len(joint_pos_cols) == 7 and \
    (tau_measured_history_array.size > 0 and gravity_history_array.size > 0 and tau_history_array.size > 0):

        # 力矩误差：tau_cmd - (tau_meas - gravity)
        tau_err = tau_history_array - (tau_measured_history_array - gravity_history_array)  # shape: [N,7]
        q_all   = df[joint_pos_cols].values  # shape: [N,7]

        # 画 7 个关节的 位置-误差 散点 + 线性拟合
        fig2, axes2 = plt.subplots(3, 3, figsize=(18, 14))
        fig2.suptitle('Joint Position vs Torque Error', fontsize=14)

        # 只用到前 7 个子图
        import itertools
        grid_axes = list(itertools.chain.from_iterable(axes2))
        for j in range(6):
            ax = grid_axes[j]
            qj = q_all[:, j]
            ej = tau_err[:, j]

            vel_col = f'joint_vel_{j+1}'
            if vel_col in df.columns:
                vj = df[vel_col].values
                pos_mask = vj >= 0
                ax.scatter(qj[pos_mask], ej[pos_mask], s=8, c='red', alpha=0.5, label='v > 0')
                ax.scatter(qj[~pos_mask], ej[~pos_mask], s=8, c='blue', alpha=0.5, label='v < 0')
            else:
                ax.scatter(qj, ej, s=6, alpha=0.5, label=f'Joint {j+1}')

            # 线性拟合 y = a x + b（最小二乘）
            if len(qj) >= 2:
                A = np.vstack([qj, np.ones_like(qj)]).T
                a, b = np.linalg.lstsq(A, ej, rcond=None)[0]
                xfit = np.linspace(qj.min(), qj.max(), 100)
                yfit = a * xfit + b
                ax.plot(xfit, yfit, linewidth=2, label=f'fit: y={a:.3f}x+{b:.3f}')

                # 皮尔逊相关系数
                corr = np.corrcoef(qj, ej)[0, 1]
                ax.set_title(f'Joint {j+1}  (corr={corr:.3f})')
            else:
                ax.set_title(f'Joint {j+1}')

            ax.set_xlabel('Joint Position [rad]')
            ax.set_ylabel('Torque Error [Nm]')
            ax.grid(True)
            ax.legend()

        # 多出的第 9 个子图清空
        grid_axes[8].axis('off')

        plt.tight_layout()
        out2 = csv_filename.replace('.csv', '_pos_vs_tauerr.png')
        fig2.savefig(out2, dpi=300, bbox_inches='tight')
        print(f'Figure saved as {out2}')
    else:
        print('Skip joint position vs torque error plot: missing joint_pos_* columns or torque data.')

        # ===== 逐关节：local / cloud / combined y_hat vs tau_residual =====
    # 允许下面三种命名：
    #   y_hat_local_1..7, y_hat_cloud_1..7, y_hat_1..7 (combined)
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color='k', linestyle='-',  linewidth=2.0,
            label=r'$\tau_{\mathrm{res}}$'),
        Line2D([0], [0], color='tab:blue', linestyle='--', linewidth=1.6,
            label=r'$\hat{\tau}_{\mathrm{local}}$'),
        Line2D([0], [0], color='tab:orange', linestyle='-.', linewidth=1.6,
            label=r'$\hat{\tau}_{\mathrm{cloud}}$'),
        Line2D([0], [0], color='tab:red', linestyle='-', linewidth=1.2,
            label=r'$\Delta |e|$')
    ]

    yhat_comb_cols  = cols_1to7(df, 'y_hat_')
    yhat_local_cols = cols_1to7(df, 'y_hat_local_')
    yhat_cloud_cols = cols_1to7(df, 'y_hat_cloud_')
    yhat_mem_cols = cols_1to7(df,'y_hat_mem_')
    res_cols        = cols_1to7(df, 'tau_residual_')    
    if len(res_cols) == 7 and (len(yhat_comb_cols) == 7 or len(yhat_local_cols) == 7 or len(yhat_cloud_cols) == 7 or len(yhat_mem_cols) == 7):

        # ================== 数据读取 ==================
        TR = df[res_cols].values              # tau_residual, shape [N, 7]
        YH_local = df[yhat_local_cols].values if len(yhat_local_cols) == 7 else None
        YH_cloud = df[yhat_cloud_cols].values if len(yhat_cloud_cols) == 7 else None
        YH_comb  = df[yhat_comb_cols].values  if len(yhat_comb_cols)  == 7 else None
        YH_mem = df[yhat_mem_cols].values  if len(yhat_mem_cols)  == 7 else None
        # ================== Figure ==================
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        ax_list = [ax for row in axes for ax in row]

        # ---- 主标题 ----
        # fig.suptitle(
        #     r'Per-Joint GP-Based Residual Torque Prediction',
        #     fontsize=15,
        #     y=0.96
        # )

        # ================== 子图绘制 ==================
        for j in range(6):
            ax = ax_list[j]

            # ---- Ground truth residual ----
            tau_res = TR[:, j]
            ax.plot(
                time_history, tau_res,
                color='k', linestyle='-',
                linewidth=2.0
            )

            # ---- Local GP ----
            if YH_local is not None:
                tau_hat_local = YH_local[:, j]
                ax.plot(
                    time_history, tau_hat_local,
                    color='tab:blue', linestyle='--',
                    linewidth=1.6
                )

            # ---- Cloud GP ----
            if YH_cloud is not None:
                tau_hat_cloud = YH_cloud[:, j]
                ax.plot(
                    time_history, tau_hat_cloud,
                    color='tab:orange', linestyle='-.',
                    linewidth=1.6
                )
            
            if YH_mem is not None:
                tau_hat_mem = YH_mem[:, j]
                ax.plot(
                    time_history, tau_hat_mem,
                    color='tab:green', linestyle='-.',
                    linewidth=1.6
                )

            # ---- Error improvement (fused vs local) ----
            if (YH_local is not None) and (YH_comb is not None):
                e_local = tau_res - YH_local[:, j]
                e_fused = tau_res - YH_comb[:, j]

                delta_abs_err = np.abs(e_fused) - np.abs(e_local)

                ax.plot(
                    time_history, delta_abs_err,
                    color='tab:red', linestyle='-',
                    linewidth=1.2,
                    alpha=0.85
                )

                win_rate = np.mean(np.abs(e_fused) < np.abs(e_local)) * 100.0
                ax.set_title(
                    rf'Joint {j+1}  (Improvement: {win_rate:.1f}\%)',
                    fontsize=11
                )
            else:
                ax.set_title(rf'Joint {j+1}', fontsize=11)

            # ---- 轴 & 网格 ----
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Torque (Nm)')
            ax.grid(True, which='both', linestyle=':', linewidth=0.6)

        # ================== Layout & Save ==================
        plt.tight_layout(rect=[0, 0, 1, 0.88])

        out_fig = csv_filename.replace(
            '.csv',
            '_gp_residual_prediction_per_joint.png'
        )
        fig.legend(
            handles=legend_handles,
            loc='upper center',
            ncol=4,
            frameon=True,
            fontsize=11,
            bbox_to_anchor=(0.5, 0.915)
        )
        
        fig.savefig(out_fig, dpi=300, bbox_inches='tight')
        print(f'Figure saved as {out_fig}')
    else:
        print('Skip y_hat(local/cloud) vs tau_residual plot: missing columns.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_kde_from_csv(
    csv_file,
    error_type='abs',   # 'abs' or 'mse'
    use_log_x=False,
    save_path=None
):
    """
    Pure KDE plot of prediction error distributions from CSV.

    Parameters
    ----------
    csv_file : str
        Path to CSV file.
    error_type : str
        'abs' -> |error|
        'mse' -> error^2
    use_log_x : bool
        Whether to use log-scale on x-axis.
    save_path : str or None
        If provided, save figure to this path.
    """

    df = pd.read_csv(csv_file)

    # ---------- column detection ----------
    def cols(prefix):
        return [f'{prefix}{i}' for i in range(1, 8) if f'{prefix}{i}' in df.columns]

    res_cols   = cols('tau_residual_')
    local_cols = cols('y_hat_local_')
    cloud_cols = cols('y_hat_cloud_')
    comb_cols  = cols('y_hat_')

    if len(res_cols) != 7 or len(local_cols) != 7:
        raise RuntimeError('CSV must contain tau_residual_1..7 and y_hat_local_1..7')

    TR = df[res_cols].values
    YL = df[local_cols].values
    YC = df[cloud_cols].values if len(cloud_cols) == 7 else None
    YF = df[comb_cols].values  if len(comb_cols)  == 7 else None

    # ---------- error aggregation ----------
    err_local = []
    err_cloud = []
    err_fused = []

    for j in range(6):
        e_l = TR[:, j] - YL[:, j]
        err_local.append(e_l)

        if YC is not None:
            err_cloud.append(TR[:, j] - YC[:, j])

        if YF is not None:
            err_fused.append(TR[:, j] - YF[:, j])

    err_local = np.concatenate(err_local)
    err_cloud = np.concatenate(err_cloud) if err_cloud else None
    err_fused = np.concatenate(err_fused) if err_fused else None

    # ---------- choose metric ----------
    def transform(e):
        e = e[np.isfinite(e)]
        if error_type == 'abs':
            return np.abs(e)
        elif error_type == 'mse':
            return e ** 2
        else:
            raise ValueError("error_type must be 'abs' or 'mse'")

    err_local = transform(err_local)
    err_cloud = transform(err_cloud) if err_cloud is not None else None
    err_fused = transform(err_fused) if err_fused is not None else None

    # ---------- KDE plot ----------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    def plot_kde(data, label, color, linestyle, fill=False):
        kde = gaussian_kde(data, bw_method='scott')
        xmin, xmax = np.percentile(data, [1, 99])
        x = np.linspace(xmin, xmax, 400)
        y = kde(x)
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=2.0, label=label)
        if fill:
            ax.fill_between(x, y, color=color, alpha=0.25)

    plot_kde(err_local, 'Local GP', 'tab:blue', '--')

    if err_cloud is not None:
        plot_kde(err_cloud, 'Cloud GP', 'tab:orange', '-.', fill=True)

    if err_fused is not None:
        plot_kde(err_fused, 'Fused (SkyGP)', 'k', '-')

    # ---------- axis styling ----------
    xlabel = r'$|e|$ (Nm)' if error_type == 'abs' else r'$e^2$ (Nm$^2$)'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability Density')
    ax.grid(True, linestyle=':', linewidth=0.6)
    ax.legend(frameon=True)

    if use_log_x:
        ax.set_xscale('log')
        ax.set_xlabel(xlabel + ' [log scale]')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'KDE figure saved as {save_path}')

    return fig, ax

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_kde_per_joint_from_csv(
    csv_file,
    error_type='abs',   # 'abs' or 'mse'
    num_joints=6,
    use_log_x=True,
    save_path=None
):
    """
    Plot per-joint KDE (PDF) of prediction errors from CSV.

    Parameters
    ----------
    csv_file : str
        Path to CSV file.
    error_type : str
        'abs' -> |error|,  'mse' -> error^2
    num_joints : int
        Number of joints to plot (default: 6).
    use_log_x : bool
        Use log scale on x-axis (recommended for MSE).
    save_path : str or None
        Save figure if provided.
    """

    df = pd.read_csv(csv_file)

    # ---------- column detection ----------
    def cols(prefix):
        return [f'{prefix}{i}' for i in range(1, 8) if f'{prefix}{i}' in df.columns]

    res_cols   = cols('tau_residual_')
    local_cols = cols('y_hat_local_')
    cloud_cols = cols('y_hat_cloud_')
    comb_cols  = cols('y_hat_')

    if len(res_cols) != 7 or len(local_cols) != 7:
        raise RuntimeError('CSV must contain tau_residual_1..7 and y_hat_local_1..7')

    TR = df[res_cols].values
    YL = df[local_cols].values
    YC = df[cloud_cols].values if len(cloud_cols) == 7 else None
    YF = df[comb_cols].values  if len(comb_cols)  == 7 else None

    # ---------- error transform ----------
    def transform(e):
        e = e[np.isfinite(e)]
        if error_type == 'abs':
            return np.abs(e)
        elif error_type == 'mse':
            return e ** 2
        else:
            raise ValueError("error_type must be 'abs' or 'mse'")

    # ---------- figure ----------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax_list = [ax for row in axes for ax in row]

    # fig.suptitle(
    #     r'Per-Joint Prediction Error Density (KDE)',
    #     fontsize=15,
    #     y=0.96
    # )

    def plot_kde(ax, data, label, color, linestyle, fill=False):
        kde = gaussian_kde(data, bw_method='scott')
        xmin, xmax = np.percentile(data, [1, 99])
        x = np.linspace(xmin, xmax, 400)
        y = kde(x)
        ax.plot(x, y, color=color, linestyle=linestyle,
                linewidth=2.0, label=label)
        if fill:
            ax.fill_between(x, y, color=color, alpha=0.25)

    # ---------- per joint ----------
    for j in range(num_joints):
        ax = ax_list[j]

        # Local
        err_local = transform(TR[:, j] - YL[:, j])
        plot_kde(ax, err_local, 'Local GP', 'tab:blue', '--')

        # Cloud
        if YC is not None:
            err_cloud = transform(TR[:, j] - YC[:, j])
            plot_kde(ax, err_cloud, 'Cloud GP', 'tab:orange', '-.', fill=True)

        # Fused
        if YF is not None:
            err_fused = transform(TR[:, j] - YF[:, j])
            plot_kde(ax, err_fused, 'Fused (SkyGP)', 'k', '-')

        ax.set_title(f'Joint {j+1}', fontsize=11)
        ax.grid(True, linestyle=':', linewidth=0.6)

        if use_log_x:
            ax.set_xscale('log')

    # ---------- shared labels ----------
    xlabel = r'$|e|$ (Nm)' if error_type == 'abs' else r'$e^2$ (Nm$^2$)'
    fig.text(0.5, 0.04, xlabel, ha='center', fontsize=12)
    fig.text(0.03, 0.5, 'Probability Density', va='center',
             rotation='vertical', fontsize=12)

    # ---------- legend (only once) ----------
    handles, labels = ax_list[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=3,
        frameon=True,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.94)
    )

    plt.tight_layout(rect=[0.05, 0.06, 0.95, 0.90])

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Per-joint KDE figure saved as {save_path}')

    return fig, axes



def main():
    parser = argparse.ArgumentParser(description='Plot data from Cartesian Impedance Controller CSV file')
    parser.add_argument('csv_file', nargs='?', default='cartesian_impedance_controller_data.csv',
                       help='CSV file to plot (default: cartesian_impedance_controller_data.csv)')
    
    args = parser.parse_args()
    
    if not args.csv_file.endswith('.csv'):
        print('Error: Please provide a CSV file')
        sys.exit(1)
    
    plot_data_from_csv(args.csv_file)
    plot_kde_per_joint_from_csv(
        csv_file=args.csv_file,
        error_type='abs',
        save_path='kde_abs_error_per_joint.png'
    )

if __name__ == '__main__':
    main() 