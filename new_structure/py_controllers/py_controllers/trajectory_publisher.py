#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_msgs.msg import TaskSpaceCommand
from std_msgs.msg import Header
import numpy as np
import time


class TrajectoryPublisher(Node):
    
    def __init__(self):
        super().__init__('trajectory_publisher')
        
        # publish on /task_space_command
        self.trajectory_publisher = self.create_publisher(
            TaskSpaceCommand, '/task_space_command', 10)
        
        # publish frequency: 1000 Hz
        self.timer = self.create_timer(0.001, self.timer_callback)  # 0.001 second = 1000 Hz
        
        # circle trajectory parameters
        self.declare_parameter('circle_radius', 0.05)    # circle radius (meter)
        self.declare_parameter('circle_frequency', 0.1) # circle motion frequency (Hz)
        self.declare_parameter('circle_center_x', 0.6)  # circle center x coordinate
        self.declare_parameter('circle_center_y', 0.0)  # circle center y coordinate
        self.declare_parameter('circle_center_z', 0.45)  # circle center z coordinate
        
        self.radius = self.get_parameter('circle_radius').value
        self.frequency = self.get_parameter('circle_frequency').value
        self.center_x = self.get_parameter('circle_center_x').value
        self.center_y = self.get_parameter('circle_center_y').value
        self.center_z = self.get_parameter('circle_center_z').value
        
        self.start_time = self.get_clock().now()
        
        self.get_logger().info('Trajectory publisher node started')
        self.get_logger().info(f'Publishing circular trajectory at 1000 Hz')
        self.get_logger().info(f'Circle radius: {self.radius} m, frequency: {self.frequency} Hz')
        self.get_logger().info(f'Circle center: ({self.center_x}, {self.center_y}, {self.center_z})')
    
    def timer_callback(self):
        """timer callback function, period: 1ms"""
        try:
            # calculate elapsed time
            current_time = self.get_clock().now()
            elapsed_time = (current_time - self.start_time).nanoseconds / 1e9
            
            # calculate circle trajectory
            omega = 2.0 * np.pi * self.frequency  # angular velocity
            
            # position: x_des[:3] corresponds to (x, y, z)
            x = self.center_x + self.radius * np.cos(omega * elapsed_time)
            y = self.center_y + self.radius * np.sin(omega * elapsed_time)
            z = self.center_z
            
            # velocity: dx_des[:3] corresponds to (dx, dy, dz)
            dx = -self.radius * omega * np.sin(omega * elapsed_time)
            dy = self.radius * omega * np.cos(omega * elapsed_time)
            dz = 0.0
            
            # acceleration: ddx_des[:3] corresponds to (ddx, ddy, ddz)
            ddx = -self.radius * omega**2 * np.cos(omega * elapsed_time)
            ddy = -self.radius * omega**2 * np.sin(omega * elapsed_time)
            ddz = 0.0
            
            # publish message
            trajectory_msg = TaskSpaceCommand()
            trajectory_msg.header = Header()
            trajectory_msg.header.stamp = current_time.to_msg()
            trajectory_msg.header.frame_id = "base_link"
            trajectory_msg.x_des = [x, y, z, 0.0, 0.0, 0.0]         # position (x, y, z, roll, pitch, yaw)
            trajectory_msg.dx_des = [dx, dy, dz, 0.0, 0.0, 0.0]     # velocity
            trajectory_msg.ddx_des = [ddx, ddy, ddz, 0.0, 0.0, 0.0] # acceleration
            
            self.trajectory_publisher.publish(trajectory_msg)
            
            if int(elapsed_time * 1000) % 1000 == 0:
                self.get_logger().debug(f'Published trajectory at t={elapsed_time:.3f}s: pos=({x:.3f}, {y:.3f}, {z:.3f})')
                
        except Exception as e:
            self.get_logger().error(f'Error in trajectory publisher: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    trajectory_publisher_node = TrajectoryPublisher()
    
    try:
        rclpy.spin(trajectory_publisher_node)
    except KeyboardInterrupt:
        pass
    finally:
        trajectory_publisher_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 