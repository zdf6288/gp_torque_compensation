#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_msgs.msg import StateParameter, EffortCommand, TaskSpaceCommand
import numpy as np
from scipy.spatial.transform import Rotation

class CartesianImpedanceController(Node):
    
    def __init__(self):
        super().__init__('cartesian_impedance')
        
        # Subscribe to /state_parameter
        self.param_subscription = self.create_subscription(
            StateParameter, '/state_parameter', self.stateParameterCallback, 10)
        
        # Subscribe to /task_space_command
        self.task_command_subscription = self.create_subscription(
            TaskSpaceCommand, '/task_space_command', self.taskCommandCallback, 10)
        
        # Publish on /effort_command
        self.effort_publisher = self.create_publisher(
            EffortCommand, '/effort_command', 10)
        
        # self.declare_parameter('k_q', [24.0, 24.0, 24.0, 24.0, 10.0, 6.0, 2.0])  # k_gains in joint space
        # self.k_q = np.array(self.get_parameter('k_q').value, dtype=float)
        # self.K_q = np.diag(self.k_q)
        self.declare_parameter('k_gains', [20, 20, 20, 10, 10, 10])
        self.k_gains = np.array(self.get_parameter('k_gains').value, dtype=float)
        self.K_gains = np.diag(self.k_gains)
        self.eta = 0.707
        
        self.q_initial = None               # initial joint position q0
        self.t_initial = None               # initial time
        self.t_last = None                  # last time
        self.dq_buffer = None               # buffer for joint velocity dq
        self.body_jacobian_buffer = None    # buffer for body jacobian matrix in flange frame

        self.task_command_received = False
        self.x_des = None
        self.dx_des = None
        self.ddx_des = None

        self.effort_msg = EffortCommand()
        self.get_logger().info('Cartesian Impedance controller node started')
    
    def taskCommandCallback(self, msg):
        """callback function for /task_space_command subscriber"""
        self.task_command_received = True
        self.x_des = np.array(msg.x_des)
        self.dx_des = np.array(msg.dx_des)
        self.ddx_des = np.array(msg.ddx_des)
        self.get_logger().debug('Received task space command, enabling control execution')
        
    def stateParameterCallback(self, msg):
        """callback function for /state_parameter subscriber"""
        if not self.task_command_received:
            return
        
        try:
            # initialize t_initial, get t_elapsed, t_last and dt
            t_now = self.get_clock().now()
            if self.t_initial is None:
                self.t_initial = t_now
                self.t_last = t_now
                t_elapsed = 0.0
                dt = 1e-3
            else:
                t_elapsed = (t_now - self.t_initial).nanoseconds / 1e9
                dt = (t_now - self.t_last).nanoseconds / 1e9
                self.t_last = t_now
            self.get_logger().debug(f"t_elapsed: {t_elapsed:.6f}s")

            # initialize q_initial, get q, dq and ddq
            q = np.array(msg.position)
            dq = np.array(msg.velocity)
            if self.dq_buffer is None:
                ddq = np.zeros_like(dq)
                self.dq_buffer = dq.copy()
            else:                
                ddq = (dq - self.dq_buffer) / dt
                self.dq_buffer = dq.copy()

            # get O_T_F, mass, coriolis, flange-framed body jacobian matrix J(q) and dJ(q)
            o_t_f_array = np.array(msg.o_t_f)                           # vectorized 4x4 pose matrix in flange frame, column-major
            mass_matrix_array = np.array(msg.mass)                      # vectorized 7x7 mass matrix, column-major
            coriolis_matrix_array = np.array(msg.coriolis)              # vectorized diagonal elements of 7x7 coriolis matrix
            body_jacobian_array = np.array(msg.body_jacobian_flange)    # vectorized 6x7 body jacobian matrix in flange frame, column-major

            o_t_f = o_t_f_array.reshape(4, 4, order='F')                    # 4x4 pose matrix in flange frame, column-major
            mass = mass_matrix_array.reshape(7, 7, order='F')               # 7x7
            coriolis = np.diag(coriolis_matrix_array)                       # 7x7
            body_jacobian = body_jacobian_array.reshape(6, 7, order='F')    # 6x7
            body_jacobian_transpose = body_jacobian.T                       # 7x6
            body_jacobian_pseudoinverse = np.linalg.pinv(body_jacobian)     # 7x6, pseudoinverse obtained by SVD
            if self.body_jacobian_buffer is None:  
                dbody_jacobian = np.zeros_like(body_jacobian)
            else:
                dbody_jacobian = (body_jacobian - self.body_jacobian_buffer) / dt
                self.body_jacobian_buffer = body_jacobian.copy()

            # get x and dx
            position = o_t_f[:3, 3]
            rotation_matrix = o_t_f[:3, :3]
            r = Rotation.from_matrix(rotation_matrix).as_euler('xyz')
            x = np.concatenate((position, r))
            dx = body_jacobian @ dq
            # ddx = body_jacobian @ ddq + dbody_jacobian @ dq
            self.get_logger().info(f"x: {x.tolist()}, dx: {dx.tolist()}")

            # get K_gains and D_gains
            lambda_matrix = body_jacobian_pseudoinverse.T @ mass @ body_jacobian_pseudoinverse
            eigvals, _ = np.linalg.eig(lambda_matrix)
            D_gains = 2 * self.eta * np.sqrt(eigvals @ self.K_gains)
            
            # calculate tau
            tau = mass @ body_jacobian_pseudoinverse @ self.ddx_des \
                + (coriolis - mass @ body_jacobian_pseudoinverse @ dbody_jacobian) @ body_jacobian_pseudoinverse @ dx \
                - body_jacobian_transpose @ (self.K_gains @ (x - self.x_des) + D_gains @ (dx - self.dx_des))

            tau = np.clip(tau, -50.0, 50.0)
            
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
    cartesian_impedance_node = CartesianImpedanceController()
    try:
        rclpy.spin(cartesian_impedance_node)
    except KeyboardInterrupt:
        pass
    finally:
        cartesian_impedance_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 