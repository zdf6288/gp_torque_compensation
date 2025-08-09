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
        
        self.declare_parameter('k_gains', [24.0, 24.0, 24.0, 24.0, 10.0, 6.0, 2.0])  # k_gains in PD control
        self.declare_parameter('d_gains', [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5])       # d_gains in PD control
        self.k_gains = np.array(self.get_parameter('k_gains').value, dtype=float)
        self.d_gains = np.array(self.get_parameter('d_gains').value, dtype=float)
        
        self.q_initial = None  # initial joint position q0
        self.t_initial = None  # initial time
        
        self.effort_msg = EffortCommand()
        self.get_logger().info('Effort PD controller node started')
    
    def stateParameterCallback(self, msg):
        """callback function for /state_parameter subscriber"""
        try:
            # initialize t_initial and obtain t_elapsed
            t_now = self.get_clock().now()
            if self.t_initial is None:
                self.t_initial = t_now  
                t_elapsed = 0.0
            else:
                t_elapsed = (t_now - self.t_initial).nanoseconds / 1e9
            self.get_logger().debug(f"t_elapsed: {t_elapsed:.6f}s")

            # initialize q_initial and obtain q and dq
            q = np.array(msg.position)
            dq = np.array(msg.velocity)
            if self.q_initial is None:
                self.q_initial = q.copy()
                self.get_logger().info(f'Captured initial position q0: {self.q_initial.tolist()}')      
            
            # test trajectory, joint 4 and 5 move periodically
            q_delta = np.pi / 8.0 * (1 - np.cos(np.pi / 2.5 * t_elapsed))
            q_des = self.q_initial.copy()
            q_des[3] += q_delta
            q_des[4] += q_delta
            dq_delta = np.pi / 2.5 * np.sin(np.pi / 2.5 * t_elapsed)
            dq_des = np.zeros(7)
            dq_des[3] = dq_delta
            dq_des[4] = dq_delta
            
            # PD control 
            tau = self.k_gains * (q_des - q) + self.d_gains * (dq_des - dq)
            tau = np.clip(tau, -100.0, 100.0)
            
            # Publish on topic /effort_command
            self.effort_msg.efforts = tau.tolist()
            print(f'published on topic /effort_command: {tau}')
            print(f'self.effort_msg: {self.effort_msg}')
            self.effort_publisher.publish(self.effort_msg)
            self.get_logger().debug(f'published on topic /effort_command: {tau}')
            
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