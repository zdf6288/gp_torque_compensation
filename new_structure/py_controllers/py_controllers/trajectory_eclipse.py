#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_msgs.msg import TaskSpaceCommand
from std_msgs.msg import Header
import numpy as np
import time


class TrajectoryEclipsePublisher(Node):
    
    def __init__(self):
        super().__init__('trajectory_eclipse_publisher')
        
        # publish on /task_space_command
        self.trajectory_publisher = self.create_publisher(
            TaskSpaceCommand, '/task_space_command', 10)
        
        # publish frequency: 1000 Hz
        self.timer = self.create_timer(0.001, self.timer_callback)  # 0.001 second = 1000 Hz
        
        # eclipse trajectory parameters
        self.declare_parameter('eclipse_radius_x', 0.1)     # eclipse radius in x direction (meter)
        self.declare_parameter('eclipse_radius_y', 0.08)    # eclipse radius in y direction (meter)
        self.declare_parameter('eclipse_frequency', 0.1)    # eclipse motion frequency (Hz)
        self.declare_parameter('eclipse_center_x', 0.6)     # eclipse center x coordinate
        self.declare_parameter('eclipse_center_y', 0.0)     # eclipse center y coordinate
        self.declare_parameter('eclipse_center_z', 0.3)     # eclipse center z coordinate
        self.declare_parameter('z_amplitude', 0.05)         # z-axis sine motion amplitude (meter)
        self.declare_parameter('z_frequency', 0.2)          # z-axis sine motion frequency (Hz)
        self.declare_parameter('bank_angle', 5.0)          # bank angle in degrees (tilt of the eclipse plane)
        
        self.radius_x = self.get_parameter('eclipse_radius_x').value
        self.radius_y = self.get_parameter('eclipse_radius_y').value
        self.frequency = self.get_parameter('eclipse_frequency').value
        self.center_x = self.get_parameter('eclipse_center_x').value
        self.center_y = self.get_parameter('eclipse_center_y').value
        self.center_z = self.get_parameter('eclipse_center_z').value
        self.z_amplitude = self.get_parameter('z_amplitude').value
        self.z_frequency = self.get_parameter('z_frequency').value
        self.bank_angle = np.radians(self.get_parameter('bank_angle').value)  # convert to radians
        
        self.start_time = self.get_clock().now()
        
        self.get_logger().info('Trajectory Eclipse Publisher node started')
        self.get_logger().info(f'Publishing eclipse trajectory at 1000 Hz')
        self.get_logger().info(f'Eclipse radii: ({self.radius_x}, {self.radius_y}) m, frequency: {self.frequency} Hz')
        self.get_logger().info(f'Eclipse center: ({self.center_x}, {self.center_y}, {self.center_z})')
        self.get_logger().info(f'Z amplitude: {self.z_amplitude} m, Z frequency: {self.z_frequency} Hz')
        self.get_logger().info(f'Bank angle: {np.degrees(self.bank_angle):.1f} degrees')
    
    def timer_callback(self):
        """timer callback function, period: 1ms"""
        try:
            # calculate elapsed time
            current_time = self.get_clock().now()
            elapsed_time = (current_time - self.start_time).nanoseconds / 1e9
            
            # calculate eclipse trajectory parameters
            omega = 2.0 * np.pi * self.frequency  # angular velocity for eclipse
            omega_z = 2.0 * np.pi * self.z_frequency  # angular velocity for z motion
            
            # calculate base eclipse position (without bank angle)
            x_base = self.center_x + self.radius_x * np.cos(omega * elapsed_time)
            y_base = self.center_y + self.radius_y * np.sin(omega * elapsed_time)
            
            # apply bank angle rotation around x-axis
            # This creates a tilted ellipse where the z-coordinate varies with the bank angle
            cos_bank = np.cos(self.bank_angle)
            sin_bank = np.sin(self.bank_angle)
            
            # Apply rotation matrix around x-axis
            x = x_base
            y = y_base * cos_bank
            z = self.center_z + y_base * sin_bank + self.z_amplitude * np.sin(omega_z * elapsed_time)
            
            # calculate velocities
            # Base velocities (without bank angle)
            dx_base = -self.radius_x * omega * np.sin(omega * elapsed_time)
            dy_base = self.radius_y * omega * np.cos(omega * elapsed_time)
            
            # Apply bank angle to velocities
            dx = dx_base
            dy = dy_base * cos_bank
            dz = dy_base * sin_bank + self.z_amplitude * omega_z * np.cos(omega_z * elapsed_time)
            
            # calculate accelerations
            # Base accelerations (without bank angle)
            ddx_base = -self.radius_x * omega**2 * np.cos(omega * elapsed_time)
            ddy_base = -self.radius_y * omega**2 * np.sin(omega * elapsed_time)
            
            # Apply bank angle to accelerations
            ddx = ddx_base
            ddy = ddy_base * cos_bank
            ddz = ddy_base * sin_bank - self.z_amplitude * omega_z**2 * np.sin(omega_z * elapsed_time)
            
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
                self.get_logger().debug(f'Published eclipse trajectory at t={elapsed_time:.3f}s: pos=({x:.3f}, {y:.3f}, {z:.3f})')
                
        except Exception as e:
            self.get_logger().error(f'Error in trajectory eclipse publisher: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    trajectory_eclipse_node = TrajectoryEclipsePublisher()
    
    try:
        rclpy.spin(trajectory_eclipse_node)
    except KeyboardInterrupt:
        trajectory_eclipse_node.get_logger().info('收到键盘中断信号，正在停止轨迹发布...')
    except Exception as e:
        trajectory_eclipse_node.get_logger().error(f'程序运行时发生错误: {str(e)}')
    finally:
        trajectory_eclipse_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 