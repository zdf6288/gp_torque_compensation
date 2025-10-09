#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import argparse
import sys
import os

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
        SMOOTH_WINDOW = 10  # 相当于 ~50ms 平滑

        # Step 1: 子采样
        df = df.iloc[::DECIMATE, :].reset_index(drop=True)

        # Step 2: 平滑 (滚动均值)
        df_smooth = df.rolling(window=SMOOTH_WINDOW, center=True).mean()

        # 用平滑后的数据替换原 df，避免后面改很多
        df = df_smooth.dropna().reset_index(drop=True)

        print(f"✅ Applied decimation (/{DECIMATE}) and smoothing (window={SMOOTH_WINDOW})")
        print(f"Resulting data points: {len(df)}")
        
        time_history = df['Time(s)'].values
        
        tau_columns = [col for col in df.columns if col.startswith('tau_') and not col.startswith('tau_measured_')]
        tau_history_array = df[tau_columns].values
        
        x_history = df[['x_actual', 'y_actual', 'z_actual']].values
        x_des_history = df[['x_desired', 'y_desired', 'z_desired']].values
        
        dx_history = df[['dx_actual', 'dy_actual', 'dz_actual']].values
        dx_des_history = df[['dx_desired', 'dy_desired', 'dz_desired']].values

        tau_measured_columns = [col for col in df.columns if col.startswith('tau_measured_')]
        tau_measured_history_array = df[tau_measured_columns].values

        gravity_columns = [col for col in df.columns if col.startswith('gravity_')]
        gravity_history_array = df[gravity_columns].values
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
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
        
        plt.show()
        
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
        for j in range(7):
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



def main():
    parser = argparse.ArgumentParser(description='Plot data from Cartesian Impedance Controller CSV file')
    parser.add_argument('csv_file', nargs='?', default='cartesian_impedance_controller_data.csv',
                       help='CSV file to plot (default: cartesian_impedance_controller_data.csv)')
    
    args = parser.parse_args()
    
    if not args.csv_file.endswith('.csv'):
        print('Error: Please provide a CSV file')
        sys.exit(1)
    
    plot_data_from_csv(args.csv_file)

if __name__ == '__main__':
    main() 