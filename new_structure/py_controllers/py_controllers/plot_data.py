#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
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
        
        # plot position error
        position_errors = []
        for i in range(len(x_history)):
            actual_pos = x_history[i][:3]
            desired_pos = x_des_history[i][:3]
            error = np.linalg.norm(actual_pos - desired_pos)
            position_errors.append(error)
        
        axes[0, 2].plot(time_history, position_errors, 'r-', linewidth=2)
        axes[0, 2].set_title('Position Error (Euclidean Distance)')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Error (m)')
        axes[0, 2].grid(True)
        
        # plot position trajectory on x axis in task space
        axes[1, 0].plot(time_history, x_history[:, 0], 'b-', label='Actual X', linewidth=2)
        axes[1, 0].plot(time_history, x_des_history[:, 0], 'r--', label='Desired X', linewidth=2)
        axes[1, 0].set_title('X Position Trajectory')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Position (m)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # plot position trajectory on y axis in task space
        axes[1, 1].plot(time_history, x_history[:, 1], 'b-', label='Actual Y', linewidth=2)
        axes[1, 1].plot(time_history, x_des_history[:, 1], 'r--', label='Desired Y', linewidth=2)
        axes[1, 1].set_title('Y Position Trajectory')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Position (m)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # plot position trajectory on z axis in task space
        axes[1, 2].plot(time_history, x_history[:, 2], 'b-', label='Actual Z', linewidth=2)
        axes[1, 2].plot(time_history, x_des_history[:, 2], 'r--', label='Desired Z', linewidth=2)
        axes[1, 2].set_title('Z Position Trajectory')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Position (m)')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
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