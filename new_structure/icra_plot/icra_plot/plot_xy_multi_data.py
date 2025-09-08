#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys
import os

def plot_xy_trajectory(csv_filename):
    """plot actual position trajectory on x-y plane"""
    if not os.path.exists(csv_filename):
        print(f'Error: CSV file {csv_filename} not found')
        return
        
    try:
        df = pd.read_csv(csv_filename)
        
        time_history = df['Time(s)'].values
        x_actual = df['x_actual'].values
        y_actual = df['y_actual'].values
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        ax.plot(x_actual, y_actual, 'b-', linewidth=2, label='Actual Trajectory')
        
        ax.plot(x_actual[0], y_actual[0], 'go', markersize=8, label='Start Point', markeredgecolor='darkgreen')
        ax.plot(x_actual[-1], y_actual[-1], 'ro', markersize=8, label='End Point', markeredgecolor='darkred')
        
        ax.set_title('Actual Position Trajectory on X-Y Plane', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        ax.autoscale_view()
        
        plt.tight_layout()
        
        output_filename = csv_filename.replace('.csv', '_xy_trajectory.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f'Image saved as: {output_filename}')
        
        plt.show()
        
    except Exception as e:
        print(f'Error when plotting data: {str(e)}')

def main():
    parser = argparse.ArgumentParser(description='Plot actual position trajectory on x-y plane from CSV file')
    parser.add_argument('csv_file', nargs='?', default='training_multi_data.csv',
                       help='CSV file to plot (default: training_multi_data.csv)')
    
    args = parser.parse_args()
    
    if not args.csv_file.endswith('.csv'):
        print('Error: Please provide a CSV file')
        sys.exit(1)
    
    plot_xy_trajectory(args.csv_file)

if __name__ == '__main__':
    main() 