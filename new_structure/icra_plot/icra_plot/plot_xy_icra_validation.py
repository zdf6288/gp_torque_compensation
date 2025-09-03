#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys
import os

def plot_dual_xy_trajectory(before_csv, after_csv):
    """Plot desired vs actual position trajectory on x-y plane for both before and after GP data"""
    
    if not os.path.exists(before_csv):
        print(f'Error: CSV file {before_csv} not found')
        return
    if not os.path.exists(after_csv):
        print(f'Error: CSV file {after_csv} not found')
        return
        
    try:
        df_before = pd.read_csv(before_csv)
        df_after = pd.read_csv(after_csv)
        
        if df_before.empty:
            print(f'Error: {before_csv} contains no data rows')
            return
        if df_after.empty:
            print(f'Warning: {after_csv} contains no data rows. Plotting only before GP data.')
            # Plot only before GP data
            plot_single_xy_trajectory(df_before, "Before GP")
            return
        
        x_actual_before = df_before['x_actual'].values
        y_actual_before = df_before['y_actual'].values
        x_desired_before = df_before['x_desired'].values
        y_desired_before = df_before['y_desired'].values
        
        x_actual_after = df_after['x_actual'].values
        y_actual_after = df_after['y_actual'].values
        x_desired_after = df_after['x_desired'].values
        y_desired_after = df_after['y_desired'].values
        
        plt.figure(figsize=(14, 10))

        # 1. Desired trajectory before GP - blue dashed line
        plt.plot(x_desired_before, y_desired_before, 'b--', label='Desired Trajectory (Before GP)', linewidth=2, alpha=0.8)
        # 2. Actual trajectory before GP - blue continuous line
        plt.plot(x_actual_before, y_actual_before, 'b-', label='Actual Trajectory (Before GP)', linewidth=2, alpha=0.8)
        # 3. Desired trajectory after GP - red dashed line
        plt.plot(x_desired_after, y_desired_after, 'r--', label='Desired Trajectory (After GP)', linewidth=2, alpha=0.8)
        # 4. Actual trajectory after GP - red continuous line
        plt.plot(x_actual_after, y_actual_after, 'r-', label='Actual Trajectory (After GP)', linewidth=2, alpha=0.8)
        
        # plot start and end points
        plt.plot(x_desired_before[0], y_desired_before[0], 'bo', markersize=10, label='Start Point (Before GP)', alpha=0.7)
        plt.plot(x_actual_before[0], y_actual_before[0], 'bo', markersize=8, alpha=0.7)
        plt.plot(x_desired_after[0], y_desired_after[0], 'ro', markersize=10, label='Start Point (After GP)', alpha=0.7)
        plt.plot(x_actual_after[0], y_actual_after[0], 'ro', markersize=8, alpha=0.7)
        
        plt.plot(x_desired_before[-1], y_desired_before[-1], 'bs', markersize=10, label='End Point (Before GP)', alpha=0.7)
        plt.plot(x_actual_before[-1], y_actual_before[-1], 'bs', markersize=8, alpha=0.7)
        plt.plot(x_desired_after[-1], y_desired_after[-1], 'rs', markersize=10, label='End Point (After GP)', alpha=0.7)
        plt.plot(x_actual_after[-1], y_actual_after[-1], 'rs', markersize=8, alpha=0.7)
        
        # calculate error statistics for both datasets
        x_error_before = x_actual_before - x_desired_before
        y_error_before = y_actual_before - y_desired_before
        position_error_before = np.sqrt(x_error_before**2 + y_error_before**2)
        
        x_error_after = x_actual_after - x_desired_after
        y_error_after = y_actual_after - y_desired_after
        position_error_after = np.sqrt(x_error_after**2 + y_error_after**2)
        
        mean_error_before = np.mean(position_error_before)
        max_error_before = np.max(position_error_before)
        rms_error_before = np.sqrt(np.mean(position_error_before**2))
        
        mean_error_after = np.mean(position_error_after)
        max_error_after = np.max(position_error_after)
        rms_error_after = np.sqrt(np.mean(position_error_after**2))

        # display error statistics
        error_text = f'Before GP:\nMean: {mean_error_before:.4f} m\nMax: {max_error_before:.4f} m\nRMS: {rms_error_before:.4f} m\n\n'
        error_text += f'After GP:\nMean: {mean_error_after:.4f} m\nMax: {max_error_after:.4f} m\nRMS: {rms_error_after:.4f} m'
        
        plt.text(0.02, 0.98, error_text, 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        
        plt.xlabel('X Position (m)', fontsize=12)
        plt.ylabel('Y Position (m)', fontsize=12)
        plt.title('Trajectory Comparison: Before vs After GP', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.minorticks_on()
        plt.grid(True, which='minor', alpha=0.2)
        plt.tight_layout()
        
        output_filename = 'validation_comparison_xy_trajectory.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f'Comparison plot saved as {output_filename}')
        
        print(f'\nTrajectory Error Statistics:')
        print(f'Before GP:')
        print(f'  Mean Position Error: {mean_error_before:.4f} m')
        print(f'  Max Position Error: {max_error_before:.4f} m')
        print(f'  RMS Position Error: {rms_error_before:.4f} m')
        print(f'After GP:')
        print(f'  Mean Position Error: {mean_error_after:.4f} m')
        print(f'  Max Position Error: {max_error_after:.4f} m')
        print(f'  RMS Position Error: {rms_error_after:.4f} m')
        
        plt.show()
        
    except Exception as e:
        print(f'Error when plotting data: {str(e)}')

def plot_single_xy_trajectory(df, label):
    """Plot single trajectory when one of the files is empty"""
    x_actual = df['x_actual'].values
    y_actual = df['y_actual'].values
    x_desired = df['x_desired'].values
    y_desired = df['y_desired'].values
    
    plt.figure(figsize=(12, 10))
    
    color = 'blue' if 'Before' in label else 'red'
    
    plt.plot(x_desired, y_desired, f'{color[0]}--', label=f'Desired Trajectory ({label})', linewidth=2, alpha=0.8)
    plt.plot(x_actual, y_actual, f'{color[0]}-', label=f'Actual Trajectory ({label})', linewidth=2, alpha=0.8)
    
    plt.plot(x_desired[0], y_desired[0], f'{color[0]}o', markersize=10, label=f'Start Point ({label})', alpha=0.7)
    plt.plot(x_actual[0], y_actual[0], f'{color[0]}o', markersize=8, alpha=0.7)
    plt.plot(x_desired[-1], y_desired[-1], f'{color[0]}s', markersize=10, label=f'End Point ({label})', alpha=0.7)
    plt.plot(x_actual[-1], y_actual[-1], f'{color[0]}s', markersize=8, alpha=0.7)
    
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
    plt.title(f'Position Trajectory on X-Y Plane ({label})', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.minorticks_on()
    plt.grid(True, which='minor', alpha=0.2)
    plt.tight_layout()
    
    output_filename = f'{label.lower().replace(" ", "_")}_xy_trajectory.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f'X-Y trajectory plot saved as {output_filename}')
    
    print(f'\nTrajectory Error Statistics ({label}):')
    print(f'Mean Position Error: {mean_error:.4f} m')
    print(f'Max Position Error: {max_error:.4f} m')
    print(f'RMS Position Error: {rms_error:.4f} m')
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot comparison of desired vs actual position trajectory before and after GP')
    parser.add_argument('--before', default='validation_data_before_gp.csv',
                       help='CSV file for before GP data (default: validation_data_before_gp.csv)')
    parser.add_argument('--after', default='validation_data_after_gp.csv',
                       help='CSV file for after GP data (default: validation_data_after_gp.csv)')
    
    args = parser.parse_args()
    
    if not args.before.endswith('.csv'):
        print('Error: Before GP file must be a CSV file')
        sys.exit(1)
    if not args.after.endswith('.csv'):
        print('Error: After GP file must be a CSV file')
        sys.exit(1)
    
    plot_dual_xy_trajectory(args.before, args.after)

if __name__ == '__main__':
    main()