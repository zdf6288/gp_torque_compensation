#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_msgs.msg import StateParameter, EffortCommand
import numpy as np


class EffortPDController(Node):
    
    def __init__(self):
        super().__init__('effort_pd')
        
        # Subscribe to /state_parameter
        self.param_subscription = self.create_subscription(
            StateParameter, '/state_parameter', self.stateParameterCallback, 10)
        
        # Publish on /effort_command
        self.effort_publisher = self.create_publisher(
            EffortCommand, '/effort_command', 10)
        
        # p_gain and d_gain, read from config, TO BE DONE
        self.declare_parameter('k_gains', np.array([24.0, 24.0, 24.0, 24.0, 10.0, 6.0, 2.0]))
        self.declare_parameter('d_gains', np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5]))
        
        # 目标位置（可以根据需要修改）
        # self.target_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        self.effort_msg = EffortCommand()
        self.get_logger().info('Effort PD controller node started')
    
    def stateParameterCallback(self, msg):
        """callback function for /state_parameter subscriber"""
        try:
            q = np.array(msg.position)
            dq = np.array(msg.velocity)
            
            # # 计算位置误差
            # position_error = self.target_position - current_position
            
            # # PD控制律
            # effort = self.kp * position_error - self.kd * current_velocity
            
            # 限制力矩范围（可选）
            effort = np.clip(effort, -100.0, 100.0)
            
            # Publish on topic /effort_command
            self.effort_msg.efforts = effort.tolist()
            self.effort_publisher.publish(self.effort_msg)
            self.get_logger().debug(f'published on topic /effort_command: {effort}')
            
        except Exception as e:
            self.get_logger().error(f'Parameter error: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    effort_pd_node = EffortPDController()
    try:
        rclpy.spin(effort_pd_node)
    except KeyboardInterrupt:
        pass
    finally:
        effort_pd_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 