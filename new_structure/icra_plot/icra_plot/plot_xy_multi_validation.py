#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import rclpy
from rclpy.node import Node


class PlotXYMultiValidationNode(Node):
    def __init__(self):
        super().__init__('plot_xy_multi_validation')
        
        self.declare_parameter('data_start_index', 0)
        self.data_start_index = self.get_parameter('data_start_index').get_parameter_value().integer_value
        self.get_logger().info(f'Using data_start_index: {self.data_start_index}')

    def plot_xy_trajectory(self, csv_filename):
        """plot actual position trajectory on x-y plane"""
        
        if not os.path.exists(csv_filename):
            self.get_logger().error(f'CSV file {csv_filename} not found')
            return
            
        try:
            df = pd.read_csv(csv_filename)
            
            time_history = df['Time(s)'].values
            x_actual = df['x_actual'].values
            y_actual = df['y_actual'].values
            
            time_history = time_history[self.data_start_index:]
            x_actual = x_actual[self.data_start_index:]
            y_actual = y_actual[self.data_start_index:]
            
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
            self.get_logger().info(f'Image saved as: {output_filename}')
            
            plt.show()
            
        except Exception as e:
            self.get_logger().error(f'Error when plotting data: {str(e)}')

def main():
    rclpy.init()
    node = PlotXYMultiValidationNode()
    
    node.declare_parameter('csv_file', 'validation_multi_data_before_gp.csv')
    csv_file = node.get_parameter('csv_file').get_parameter_value().string_value
    
    if not csv_file.endswith('.csv'):
        node.get_logger().error('Please provide a CSV file')
        rclpy.shutdown()
        sys.exit(1)
    
    node.plot_xy_trajectory(csv_file)
    rclpy.shutdown()

if __name__ == '__main__':
    main() 