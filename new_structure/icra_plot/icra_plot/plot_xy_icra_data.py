#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import rclpy
from rclpy.node import Node


class PlotXYICRADataNode(Node):
    def __init__(self):
        super().__init__('plot_xy_icra_data')
        
        self.declare_parameter('data_start_index', 0)
        self.data_start_index = self.get_parameter('data_start_index').get_parameter_value().integer_value
        self.get_logger().info(f'Using data_start_index: {self.data_start_index}')

    def plot_xy_trajectory(self, csv_filename):
        """Plot desired vs actual position trajectory on x-y plane"""
        
        if not os.path.exists(csv_filename):
            self.get_logger().error(f'CSV file {csv_filename} not found')
            return
            
        try:
            df = pd.read_csv(csv_filename)
            
            x_actual = df['x_actual'].values
            y_actual = df['y_actual'].values
            x_desired = df['x_desired'].values
            y_desired = df['y_desired'].values
            
            x_actual = x_actual[self.data_start_index:]
            y_actual = y_actual[self.data_start_index:]
            x_desired = x_desired[self.data_start_index:]
            y_desired = y_desired[self.data_start_index:]
            
            plt.figure(figsize=(12, 10))

            plt.plot(x_desired, y_desired, 'r--', label='Desired Trajectory', linewidth=2, alpha=0.8)
            plt.plot(x_actual, y_actual, 'b-', label='Actual Trajectory', linewidth=2, alpha=0.8)
            
            plt.plot(x_desired[0], y_desired[0], 'go', markersize=10, label='Start Point (Desired)', alpha=0.7)
            plt.plot(x_actual[0], y_actual[0], 'bo', markersize=8, label='Start Point (Actual)', alpha=0.7)
            plt.plot(x_desired[-1], y_desired[-1], 'rs', markersize=10, label='End Point (Desired)', alpha=0.7)
            plt.plot(x_actual[-1], y_actual[-1], 'bs', markersize=8, label='End Point (Actual)', alpha=0.7)
            
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
            self.get_logger().info(f'X-Y trajectory plot saved as {output_filename}')
            
            self.get_logger().info(f'Trajectory Error Statistics:')
            self.get_logger().info(f'Mean Position Error: {mean_error:.4f} m')
            self.get_logger().info(f'Max Position Error: {max_error:.4f} m')
            self.get_logger().info(f'RMS Position Error: {rms_error:.4f} m')
            
            plt.show()
            
        except Exception as e:
            self.get_logger().error(f'Error when plotting data: {str(e)}')

def main():
    rclpy.init()
    node = PlotXYICRADataNode()
    
    node.declare_parameter('csv_file', 'training_data.csv')
    csv_file = node.get_parameter('csv_file').get_parameter_value().string_value
    
    if not csv_file.endswith('.csv'):
        node.get_logger().error('Please provide a CSV file')
        rclpy.shutdown()
        sys.exit(1)
    
    node.plot_xy_trajectory(csv_file)
    rclpy.shutdown()

if __name__ == '__main__':
    main() 