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
        
        # 发布到 /task_space_command 话题
        self.trajectory_publisher = self.create_publisher(
            TaskSpaceCommand, '/task_space_command', 10)
        
        # 设置发布频率为1000 Hz
        self.timer = self.create_timer(0.001, self.timer_callback)  # 0.001秒 = 1000 Hz
        
        # 圆形轨迹参数
        self.declare_parameter('circle_radius', 0.05)  # 圆形半径 (米)
        self.declare_parameter('circle_frequency', 0.1)  # 圆形运动频率 (Hz)
        self.declare_parameter('circle_center_x', 0.5)  # 圆心x坐标
        self.declare_parameter('circle_center_y', 0.0)  # 圆心y坐标
        self.declare_parameter('circle_center_z', 0.3)  # 圆心z坐标
        
        self.radius = self.get_parameter('circle_radius').value
        self.frequency = self.get_parameter('circle_frequency').value
        self.center_x = self.get_parameter('circle_center_x').value
        self.center_y = self.get_parameter('circle_center_y').value
        self.center_z = self.get_parameter('circle_center_z').value
        
        # 初始化时间
        self.start_time = self.get_clock().now()
        
        self.get_logger().info('Trajectory publisher node started')
        self.get_logger().info(f'Publishing circular trajectory at 1000 Hz')
        self.get_logger().info(f'Circle radius: {self.radius} m, frequency: {self.frequency} Hz')
        self.get_logger().info(f'Circle center: ({self.center_x}, {self.center_y}, {self.center_z})')
    
    def timer_callback(self):
        """定时器回调函数，每1ms执行一次"""
        try:
            # 计算经过的时间
            current_time = self.get_clock().now()
            elapsed_time = (current_time - self.start_time).nanoseconds / 1e9
            
            # 计算圆形轨迹
            omega = 2.0 * np.pi * self.frequency  # 角速度
            
            # 位置：x_des[:3] 对应 (x, y, z)
            x = self.center_x + self.radius * np.cos(omega * elapsed_time)
            y = self.center_y + self.radius * np.sin(omega * elapsed_time)
            z = self.center_z
            
            # 速度：dx_des[:3] 对应 (dx, dy, dz)
            dx = -self.radius * omega * np.sin(omega * elapsed_time)
            dy = self.radius * omega * np.cos(omega * elapsed_time)
            dz = 0.0
            
            # 加速度：ddx_des[:3] 对应 (ddx, ddy, ddz)
            ddx = -self.radius * omega**2 * np.cos(omega * elapsed_time)
            ddy = -self.radius * omega**2 * np.sin(omega * elapsed_time)
            ddz = 0.0
            
            # 创建消息
            trajectory_msg = TaskSpaceCommand()
            trajectory_msg.header = Header()
            trajectory_msg.header.stamp = current_time.to_msg()
            trajectory_msg.header.frame_id = "base_link"
            
            # 设置位置、速度和加速度
            trajectory_msg.x_des = [x, y, z, 0.0, 0.0, 0.0]  # 位置 (x, y, z, roll, pitch, yaw)
            trajectory_msg.dx_des = [dx, dy, dz, 0.0, 0.0, 0.0]  # 速度
            trajectory_msg.ddx_des = [ddx, ddy, ddz, 0.0, 0.0, 0.0]  # 加速度
            
            # 发布消息
            self.trajectory_publisher.publish(trajectory_msg)
            
            # 每1000次发布一次日志（即每秒一次）
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