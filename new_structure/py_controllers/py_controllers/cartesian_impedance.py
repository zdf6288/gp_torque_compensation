#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_msgs.msg import StateParameter, EffortCommand, TaskSpaceCommand
import numpy as np
import signal
import csv
import traceback
import sys

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
        
        self.declare_parameter('k_gains', [2000, 500, 2000, 200, 200, 200])
        self.k_gains = np.array(self.get_parameter('k_gains').value, dtype=float)
        self.K_gains = np.diag(self.k_gains)
        self.eta = 1.0
        
        self.q_initial = None               # initial joint position q0
        self.t_initial = None               # initial time
        self.t_last = None                  # last time
        self.dq_buffer = None               # buffer for joint velocity dq
        self.zero_jacobian_buffer = None    # buffer for zero jacobian matrix in flange frame

        self.task_command_received = False  # flag for task space command received
        self.x_des = None                   # desired position from task space command
        self.dx_des = None                  # desired velocity from task space command
        self.ddx_des = None                 # desired acceleration from task space command

        self.effort_msg = EffortCommand()
        self.get_logger().info('Cartesian Impedance controller node started')
    
        # list for data recording
        self.tau_history = []
        self.time_history = []
        self.x_history = []
        self.x_des_history = []
        self.dx_history = []           
        self.dx_des_history = []  
        self.tau_measured_history = []
        self.gravity_history = []

        # set signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        self._signal_handled = False                        # flag to avoid repeated data saving

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

            # get O_T_F, mass, coriolis, flange-framed zero jacobian matrix J(q) and dJ(q)
            o_t_f_array = np.array(msg.o_t_f)                           # vectorized 4x4 pose matrix in flange frame, column-major
            mass_matrix_array = np.array(msg.mass)                      # vectorized 7x7 mass matrix, column-major
            coriolis_matrix_array = np.array(msg.coriolis)              # vectorized diagonal elements of 7x7 coriolis matrix
            zero_jacobian_array = np.array(msg.zero_jacobian_flange)    # vectorized 6x7 zero jacobian matrix in flange frame, column-major

            o_t_f = o_t_f_array.reshape(4, 4, order='F')                    # 4x4 pose matrix in flange frame, column-major
            mass_matrix = mass_matrix_array.reshape(7, 7, order='F')        # 7x7
            coriolis_matrix = np.diag(coriolis_matrix_array)                # 7x7
            zero_jacobian = zero_jacobian_array.reshape(6, 7, order='F')    # 6x7
            zero_jacobian_t = zero_jacobian.T                               # 7x6, transpose of zero_jacobian
            zero_jacobian_pinv = np.linalg.pinv(zero_jacobian)              # 7x6, pseudoinverse obtained by SVD
            if self.zero_jacobian_buffer is None:  
                dzero_jacobian = np.zeros_like(zero_jacobian)
            else:
                dzero_jacobian = (zero_jacobian - self.zero_jacobian_buffer) / dt
                self.zero_jacobian_buffer = zero_jacobian.copy()

            # get x and dx
            x = o_t_f[:3, 3]            # 3x1 position, only x-y-z
            dx = zero_jacobian @ dq     # 6x1 velocity
            # ddx = zero_jacobian @ ddq + dzero_jacobian @ dq

            # get K_gains and D_gains
            lambda_matrix = np.linalg.inv(zero_jacobian @ np.linalg.inv(mass_matrix) @ zero_jacobian.T)
            eigvals, _ = np.linalg.eig(lambda_matrix)
            d_gains = 2 * self.eta * np.sqrt(eigvals @ self.K_gains)
            D_gains = np.diag(d_gains)
            
            # calculate tau
            tau = (
                mass_matrix @ zero_jacobian_pinv[:, :3] @ self.ddx_des[:3]
                + (coriolis_matrix - mass_matrix @ zero_jacobian_pinv[:, :3] @ dzero_jacobian[:3, :])
                    @ zero_jacobian_pinv[:, :3] @ dx[:3]
                - zero_jacobian_t[:, :3]
                    @ (self.K_gains[:3, :3] @ (x - self.x_des[:3])
                    + D_gains[:3, :3] @ (dx[:3] - self.dx_des[:3]))
            )
            tau = np.clip(tau, -50.0, 50.0)
            
            # Publish on topic /effort_command
            self.effort_msg.efforts = tau.tolist()
            self.effort_publisher.publish(self.effort_msg)
            
            # Record Data
            self.tau_history.append(tau.tolist())
            self.time_history.append(t_elapsed)
            self.x_history.append(x.tolist())
            self.x_des_history.append(self.x_des.tolist())
            self.dx_history.append(dx[:3].tolist())      
            self.dx_des_history.append(self.dx_des[:3].tolist()) 
            self.tau_measured_history.append(np.array(msg.effort_measured).tolist())
            self.gravity_history.append(np.array(msg.gravity).tolist())

        except Exception as e:
            self.get_logger().error(f'Parameter error: {str(e)}')

    def signal_handler(self, signum, frame):
        """signal handler, call save data function when program is interrupted"""
        try:
            if self._signal_handled:
                return
            self._signal_handled = True
            self.get_logger().info(f'Received signal {signum}, saving data...')
            self.save_data_to_file()
            self.get_logger().info(f'Signal handler completed successfully')
            sys.exit(0)
        except Exception as e:
            self.get_logger().error(f'Error in signal handler: {str(e)}')
            self._signal_handled = False
    
    def save_data_to_file(self):
        """save data to CSV file"""
        if not self.tau_history:
            self.get_logger().warning('No data to save - tau_history is empty')
            return
            
        try:
            filename = 'cartesian_impedance_controller_data.csv'
            
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                header = ['Time(s)']
                header.extend([f'tau_{i+1}' for i in range(len(self.tau_history[0]))])
                header.extend(['x_actual', 'y_actual', 'z_actual'])
                header.extend(['x_desired', 'y_desired', 'z_desired'])
                header.extend(['dx_actual', 'dy_actual', 'dz_actual'])
                header.extend(['dx_desired', 'dy_desired', 'dz_desired'])
                header.extend([f'tau_measured_{i+1}' for i in range(len(self.tau_history[0]))])
                header.extend([f'gravity_{i+1}' for i in range(len(self.tau_history[0]))])
                writer.writerow(header)
                
                for i, t in enumerate(self.time_history):
                    row = [t]
                    row.extend(self.tau_history[i])
                    row.extend(self.x_history[i][:3])
                    row.extend(self.x_des_history[i][:3])
                    row.extend(self.dx_history[i])
                    row.extend(self.dx_des_history[i])
                    row.extend(self.tau_measured_history[i])
                    row.extend(self.gravity_history[i])
                    writer.writerow(row)
                    
            self.get_logger().info(f'Successfully saved {len(self.tau_history)} data points to {filename}')
            
        except Exception as e:
            self.get_logger().error(f'Error when saving data: {str(e)}')
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')

def main(args=None):
    rclpy.init(args=args)
    cartesian_impedance_node = CartesianImpedanceController()
    
    try:
        rclpy.spin(cartesian_impedance_node)
    except KeyboardInterrupt:
        cartesian_impedance_node.get_logger().info('Received keyboard interrupt, saving data...')
    except Exception as e:
        cartesian_impedance_node.get_logger().error(f'Error when running program: {str(e)}')
    finally:
        try:
            # save data to file only if signal handler has not been executed
            if not cartesian_impedance_node._signal_handled:
                cartesian_impedance_node.get_logger().info('Signal handler not executed, saving data to file...')
                cartesian_impedance_node.save_data_to_file()
            else:
                cartesian_impedance_node.get_logger().info('Signal handler executed, data already saved, skipping...')
                
        except Exception as e:
            cartesian_impedance_node.get_logger().error(f'Error when saving data: {str(e)}')
        
        cartesian_impedance_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 