#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys
import os

def plot_xy_trajectory(csv_filename):
    """Plot desired vs actual position trajectory on x-y plane"""
    if not os.path.exists(csv_filename):
        print(f'Error: CSV file {csv_filename} not found')
        return
        
    try:
        df = pd.read_csv(csv_filename)
        
        x_actual = df['x_actual'].values
        y_actual = df['y_actual'].values
        x_desired = df['x_desired'].values
        y_desired = df['y_desired'].values
        
        plt.figure(figsize=(12, 10))

        plt.plot(x_desired, y_desired, 'r--', label='Desired Trajectory', linewidth=2, alpha=0.8)
        plt.plot(x_actual, y_actual, 'b-', label='Actual Trajectory', linewidth=2, alpha=0.8)
        
        plt.plot(x_desired[0], y_desired[0], 'go', markersize=10, label='Start Point (Desired)', alpha=0.7)
        plt.plot(x_actual[0], y_actual[0], 'bo', markersize=8, label='Start Point (Actual)', alpha=0.7)
        plt.plot(x_desired[-1], y_desired[-1], 'rs', markersize=10, label='End Point (Desired)', alpha=0.7)
        plt.plot(x_actual[-1], y_actual[-1], 'bs', markersize=8, label='End Point (Actual)', alpha=0.7)
        
        # add arrows to show trajectory direction
        # add arrows at regular intervals to show direction
        step = max(1, len(x_desired) // 10)  # show arrows every 10% of the trajectory
        for i in range(0, len(x_desired), step):
            if i < len(x_desired) - 1:
                # desired trajectory arrow
                dx_des = x_desired[i+1] - x_desired[i]
                dy_des = y_desired[i+1] - y_desired[i]
                plt.arrow(x_desired[i], y_desired[i], dx_des, dy_des, 
                         head_width=0.005, head_length=0.005, fc='red', ec='red', alpha=0.6)
                
                # actual trajectory arrow
                dx_act = x_actual[i+1] - x_actual[i]
                dy_act = y_actual[i+1] - y_actual[i]
                plt.arrow(x_actual[i], y_actual[i], dx_act, dy_act, 
                         head_width=0.005, head_length=0.005, fc='blue', ec='blue', alpha=0.6)
        
        # calculate error statistics
        x_error = x_actual - x_desired
        y_error = y_actual - y_desired
        position_error = np.sqrt(x_error**2 + y_error**2)
        
        mean_error = np.mean(position_error)
        max_error = np.max(position_error)
        rms_error = np.sqrt(np.mean(position_error**2))

        plt.text(0.02, 0.98, f'Mean Error: {mean_error:.4f} m\nMax Error: {max_error:.4f} m\nRMS Error: {rms_error:.4f} m', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.xlabel('X Position (m)', fontsize=12)
        plt.ylabel('Y Position (m)', fontsize=12)
        plt.title('Desired vs Actual Position Trajectory on X-Y Plane', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.minorticks_on()
        plt.grid(True, which='minor', alpha=0.2)
        plt.tight_layout()
        
        output_filename = csv_filename.replace('.csv', '_xy_trajectory.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f'X-Y trajectory plot saved as {output_filename}')
        
        print(f'\nTrajectory Error Statistics:')
        print(f'Mean Position Error: {mean_error:.4f} m')
        print(f'Max Position Error: {max_error:.4f} m')
        print(f'RMS Position Error: {rms_error:.4f} m')
        
        plt.show()
        
    except Exception as e:
        print(f'Error when plotting data: {str(e)}')

def main():
    parser = argparse.ArgumentParser(description='Plot desired vs actual position trajectory on x-y plane from CSV file')
    parser.add_argument('csv_file', nargs='?', default='cartesian_impedance_controller_data.csv',
                       help='CSV file to plot (default: cartesian_impedance_controller_data.csv)')
    
    args = parser.parse_args()
    
    if not args.csv_file.endswith('.csv'):
        print('Error: Please provide a CSV file')
        sys.exit(1)
    
    plot_xy_trajectory(args.csv_file)

if __name__ == '__main__':
    main() 