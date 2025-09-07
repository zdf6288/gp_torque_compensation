#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_msgs.msg import StateParameter, EffortCommand, TaskSpaceCommand, LambdaCommand, DataForGP
from custom_msgs.srv import JointPositionAdjust, GPPredict
from std_msgs.msg import Bool
import numpy as np
from scipy.spatial.transform import Rotation
import signal
import csv
import traceback
import sys

def vee(mat):
    return np.array([mat[2, 1], mat[0, 2], mat[1, 0]])

class CartesianImpedanceICRAValidation(Node):
    
    def __init__(self):
        super().__init__('cartesian_impedance_icra_validation')
        
        # subscribe to /state_parameter
        self.param_subscription = self.create_subscription(
            StateParameter, '/state_parameter', self.stateParameterCallback, 10)
        
        # subscribe to /task_space_command
        self.task_command_subscription = self.create_subscription(
            TaskSpaceCommand, '/task_space_command', self.taskCommandCallback, 10)
        
        # subscribe to /data_recording_enabled to know when to start recording data
        self.data_recording_subscription = self.create_subscription(
            Bool, '/data_recording_enabled', self.dataRecordingCallback, 10)
        
        # subscribe to /lambda_command to know when lambda is stopped
        self.lambda_command_subscription = self.create_subscription(
            LambdaCommand, '/TwistLeft', self.lambdaCommandCallback, 10)
        
        # publish on /effort_command
        self.effort_publisher = self.create_publisher(
            EffortCommand, '/effort_command', 10)

        # publish on /data_for_gp
        self.data_for_gp_publisher = self.create_publisher(
            DataForGP, '/data_for_gp', 10)
        
        # create service client for joint position adjustment
        self.joint_position_client = self.create_client(
            JointPositionAdjust, '/joint_position_adjust')

        # create service client for GP prediction
        self.gp_predict_client = self.create_client(
            GPPredict, '/gp_predict')
        
        self.declare_parameter('k_pd', [24.0, 24.0, 24.0, 24.0, 10.0, 6.0, 2.0])    # k_gains in PD control (joint space)
        self.declare_parameter('d_pd', [16.0, 16.0, 16.0, 16.0, 10.0, 6.0, 2.0])    # d_gains in PD control (joint space)
        self.declare_parameter('i_pid', [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])        # i_gains supplement to PD control (joint space)
        self.k_pd = np.array(self.get_parameter('k_pd').value, dtype=float)
        self.d_pd = np.array(self.get_parameter('d_pd').value, dtype=float)
        self.i_pid = np.array(self.get_parameter('i_pid').value, dtype=float)
        self.i_error = np.zeros(7)

        self.declare_parameter('k_gains', [750.0, 750.0, 750.0, 75.0, 75.0, 0.0])   # k_gains in impedance control (task space)
        self.k_gains = np.array(self.get_parameter('k_gains').value, dtype=float)
        self.K_gains = np.diag(self.k_gains)
        self.eta = 1.0                                                              # for calculating d_gains

        self.declare_parameter('kpn_gains', [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])    # kpn_gains for nullspace 
        self.kpn_gains = np.array(self.get_parameter('kpn_gains').value, dtype=float)
        self.dpn_gains = 2 * np.sqrt(np.array(self.kpn_gains))                      # dpn_gains for nullspace
        
        # Joint position control parameters
        self.declare_parameter('q_des', [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.0])     # desired joint positions
        self.declare_parameter('dq_des', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])               # desired joint velocities
        self.declare_parameter('joint_position_threshold', 0.2)                             # threshold for joint position convergence
        self.q_des = np.array(self.get_parameter('q_des').value, dtype=float)
        self.dq_des = np.array(self.get_parameter('dq_des').value, dtype=float)
        self.joint_position_threshold = self.get_parameter('joint_position_threshold').value
        
        self.q_initial = None               # initial joint position q0
        self.t_initial = None               # initial time
        self.t_last = None                  # last time
        self.dq_buffer = None               # buffer for joint velocity dq
        self.zero_jacobian_buffer = None    # buffer for zero jacobian matrix in flange frame
        self.jacobian_buffer = None         # buffer for jacobian matrix in flange frame

        self.task_command_received = False  # flag for task space command received
        self.x_des = None                   # desired position from task space command
        self.dx_des = None                  # desired velocity from task space command
        self.ddx_des = None                 # desired acceleration from task space command
        self.rotation_matrix_des = np.array(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)   # desired rotation matrix, z axis perpendicular to ground
        # joint position control state
        self.joint_position_control_active = True   # start with joint position control
        self.joint_position_adjusted = False        # flag for joint position adjustment
        self.trajectory_started = False             # flag for trajectory start, indicating the start of trajectory publishment

        # data recording control
        self.data_recording_enabled = False         # flag indicating whether to record data (controlled by trajectory_publisher)

        self.effort_msg = EffortCommand()
        self.get_logger().info('Cartesian Impedance controller node started')
        self.get_logger().info(f'Desired joint positions: {self.q_des}')
        self.get_logger().info(f'Joint position threshold: {self.joint_position_threshold}')

        # filter parameters
        self.filter_freq = 20.0                                      # filter frequency for tau
        self.filter_beta = 2 * np.pi * self.filter_freq / 1000.0
        self.tau_buffer = np.zeros_like(self.effort_msg.efforts)     # buffer for tau
    
        self.lambda_stopped = False                 # flag indicating lambda is stopped
        self.data_for_gp_msg = DataForGP()
        self.gp_predict_finished = False            # flag indicating the end of GP prediction
        self.gp_service_called = False              # flag indicating GP service has been called
        self.gp_service_in_progress = False         # flag indicating GP service is currently being processed

        # list for data recording
        self.tau_history = []
        self.time_history = []
        self.x_history = []
        self.x_des_history = []
        self.dx_history = []           
        self.dx_des_history = []
        self.tau_measured_history = []
        self.gravity_history = []
        # lists for recording data when lambda is not working
        self.tau_history_new = []
        self.time_history_new = []
        self.x_history_new = []
        self.x_des_history_new = []
        self.dx_history_new = []
        self.dx_des_history_new = []
        self.tau_measured_history_new = []
        self.gravity_history_new = []

        # set signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        self._signal_handled = False                # flag to avoid repeated data saving

    def taskCommandCallback(self, msg):
        """callback function for /task_space_command subscriber"""
        self.task_command_received = True
        self.x_des = np.array(msg.x_des)
        self.dx_des = np.array(msg.dx_des)
        self.ddx_des = np.array(msg.ddx_des)
        self.get_logger().debug('Received task space command, enabling control execution')
        
    def dataRecordingCallback(self, msg):
        """callback function for /data_recording_enabled subscriber"""
        self.data_recording_enabled = msg.data

    def lambdaCommandCallback(self, msg):
        """callback function for /lambda_command subscriber"""
        self.lambda_stopped = msg.lambda_stopped
        if self.lambda_stopped:
            if not self.tau_history:
                pass     # not in process of validation, lambda at initial state
            elif not self.gp_predict_finished \
                and not self.gp_service_called and not self.gp_service_in_progress:
                self.task_command_received = False
                self.call_gp_service()
                pass
            else:
                pass

    def stateParameterCallback(self, msg):
        """callback function for /state_parameter subscriber"""
        try:
            # initialize t_initial, get t_elapsed, t_last and dt
            # initialize q_initial, get q, dq and ddq
            t_now = self.get_clock().now()
            q = np.array(msg.position)
            dq = np.array(msg.velocity)
            if self.t_initial is None:
                self.t_initial = t_now
                self.t_last = t_now
                t_elapsed = 0.0
                dt = 1e-3
                ddq = np.zeros_like(dq)
                self.dq_buffer = dq.copy()
            else:
                t_elapsed = (t_now - self.t_initial).nanoseconds / 1e9
                dt = (t_now - self.t_last).nanoseconds / 1e9
                self.t_last = t_now
                ddq = (dq - self.dq_buffer) / dt
                self.dq_buffer = dq.copy()                         

            # joint position control (for joint position adjustment before trajectory_publisher starts to work)
            if self.joint_position_control_active and not self.joint_position_adjusted:
                # check if joint positions are close enough to desired positions
                joint_error = np.linalg.norm(q - self.q_des)
                if joint_error < self.joint_position_threshold:
                    self.joint_position_adjusted = True
                    self.get_logger().info(f'Joint positions adjusted! Error: {joint_error:.6f}')
                    
                    # start trajectory by calling service
                    if not self.trajectory_started:
                        self.start_trajectory()
                else:
                    # PD control for joint positions
                    self.i_error = self.i_error + (self.q_des - q) * dt
                    tau = self.k_pd * (self.q_des - q) + self.d_pd * (self.dq_des - dq) + self.i_pid * self.i_error
                    tau = np.clip(tau, -50.0, 50.0)
                    
                    # publish effort command
                    self.effort_msg.efforts = tau.tolist()
                    self.effort_publisher.publish(self.effort_msg)

                    return
            
            # cartesian impedance control (after joint position adjustment)
            if not self.task_command_received:
                return
            
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

            # to control the z axis perpendicular to ground, use 4*7 jacobian matrix
            jacobian = zero_jacobian[:5, :]                                 # 5x7
            jacobian_t = jacobian.T                                         # 7x5
            jacobian_pinv = np.linalg.pinv(jacobian)                        # 5x7, pseudoinverse obtained by SVD
            if self.jacobian_buffer is None:
                djacobian = np.zeros_like(jacobian)
            else:
                djacobian = (jacobian - self.jacobian_buffer) / dt
            self.jacobian_buffer = jacobian.copy()

  
            # get x and dx
            x = o_t_f[:3, 3]            # 3x1 position, only x-y-z
            dx = zero_jacobian @ dq     # 6x1 velocity
            # ddx = zero_jacobian @ ddq + dzero_jacobian @ dq

            rotation_matrix = o_t_f[:3, :3]     # 3x3 rotation matrix
            r_error = - 0.5 * (np.cross(rotation_matrix[:, 2], self.rotation_matrix_des[:, 2])
                + np.cross(rotation_matrix[:, 1], self.rotation_matrix_des[:, 1])
                + np.cross(rotation_matrix[:, 0], self.rotation_matrix_des[:, 0]))

            x_error = np.concatenate([x[:3] - self.x_des[:3], r_error])
            dx_error = np.concatenate([dx[:5] - self.dx_des[:5], [0.0]])

            # get K_gains and D_gains
            lambda_matrix = np.linalg.inv(zero_jacobian @ np.linalg.inv(mass_matrix) @ zero_jacobian.T)
            eigvals, _ = np.linalg.eig(lambda_matrix)
            d_gains = 2 * self.eta * np.sqrt(eigvals @ self.K_gains)
            D_gains = np.diag(d_gains)
            
            pd_term = self.K_gains @ x_error + D_gains @ dx_error
            tau = (
                mass_matrix @ jacobian_pinv @ self.ddx_des[:5]
                + (coriolis_matrix - mass_matrix @ jacobian_pinv@ djacobian)
                    @ jacobian_pinv @ dx[:5]
                - jacobian_t @ pd_term[:5]
            )

            tau_nullspace = ((np.eye(7) - zero_jacobian_pinv @ zero_jacobian) 
                @ (self.kpn_gains * (self.q_des - q) + self.dpn_gains * (self.dq_des - dq)))
            tau = tau + tau_nullspace

            tau = self.filter_beta * tau + (1 - self.filter_beta) * self.tau_buffer
            self.tau_buffer = tau.copy()
            tau = np.clip(tau, -50.0, 50.0)
            
            # publish on topic /effort_command
            self.effort_msg.efforts = tau.tolist()
            self.effort_publisher.publish(self.effort_msg)
            
            # record data only when data recording is enabled
            if self.data_recording_enabled and not self.lambda_stopped:
                self.tau_history.append(tau.tolist())
                self.time_history.append(t_elapsed)
                self.x_history.append(x.tolist())
                self.x_des_history.append(self.x_des.tolist())
                self.dx_history.append(dx[:3].tolist())      
                self.dx_des_history.append(self.dx_des[:3].tolist()) 
                self.tau_measured_history.append(np.array(msg.effort_measured).tolist())
                self.gravity_history.append(np.array(msg.gravity).tolist())

                # publish on topic /data_for_gp
                self.data_for_gp_msg.x_real = x[:3].tolist()
                self.data_for_gp_publisher.publish(self.data_for_gp_msg)
                
            elif self.data_recording_enabled and self.lambda_stopped:   
                self.tau_history_new.append(tau.tolist())
                self.time_history_new.append(t_elapsed)
                self.x_history_new.append(x.tolist())
                self.x_des_history_new.append(self.x_des.tolist())
                self.dx_history_new.append(dx[:3].tolist())      
                self.dx_des_history_new.append(self.dx_des[:3].tolist()) 
                self.tau_measured_history_new.append(np.array(msg.effort_measured).tolist())
                self.gravity_history_new.append(np.array(msg.gravity).tolist())

        except Exception as e:
            self.get_logger().error(f'Parameter error: {str(e)}')

    def start_trajectory(self):
        """start trajectory by calling the joint position adjust service"""
        try:
            if not self.joint_position_client.service_is_ready():
                self.get_logger().warn('Joint position adjust service not ready, retrying...')
                return
            
            request = JointPositionAdjust.Request()
            request.q_des = self.q_des.tolist()
            request.dq_des = self.dq_des.tolist()
            
            future = self.joint_position_client.call_async(request)
            future.add_done_callback(self.trajectory_start_callback)
            self.trajectory_started = True
            self.get_logger().info('Requested trajectory start via service call')
            
        except Exception as e:
            self.get_logger().error(f'Error calling joint position adjust service: {str(e)}')

    def trajectory_start_callback(self, future):
        """callback for trajectory start service call"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Trajectory started successfully: {response.message}')
                self.joint_position_control_active = False  # joint position adjestment completed
                # switch to cartesian impedance control, receiving task space command from trajectory_publisher
                # to move the robot to the start point of trajectory, then follow the trajectory
            else:
                self.get_logger().warn(f'Trajectory start failed: {response.message}')
                self.trajectory_started = False         # reset flag to retry
        except Exception as e:
            self.get_logger().error(f'Error in trajectory start callback: {str(e)}')
            self.trajectory_started = False             # reset flag to retry

    def call_gp_service(self):
        try:
            if not self.gp_predict_client.service_is_ready():
                self.get_logger().warn('GP prediction service not ready, retrying...')
                return

            # set service call state flags
            self.gp_service_called = True
            self.gp_service_in_progress = True
            
            future = self.gp_predict_client.call_async(GPPredict.Request())
            future.add_done_callback(self.gp_predict_callback)
            self.gp_predict_finished = False
            self.get_logger().info('Requested GP prediction via service call')
        except Exception as e:
            self.get_logger().error(f'Error calling GP prediction service: {str(e)}')
            self.gp_service_called = False
            self.gp_service_in_progress = False
    
    def gp_predict_callback(self, future):
        """Callback for GP prediction service call"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'GP prediction finished successfully: {response.message}')
                self.gp_predict_finished = True
            else:
                self.get_logger().warn(f'GP prediction failed: {response.message}')
                self.gp_predict_finished = False
        except Exception as e:
            self.get_logger().error(f'Error in GP prediction callback: {str(e)}')
            self.gp_predict_finished = False
        finally:
            self.gp_service_in_progress = False

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
            filename = 'validation_data_before_gp.csv'
            
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
            
            filename = 'validation_data_after_gp.csv'
            
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                header = ['Time(s)']
                header.extend([f'tau_{i+1}' for i in range(len(self.tau_history_new[0]))])
                header.extend(['x_actual', 'y_actual', 'z_actual'])
                header.extend(['x_desired', 'y_desired', 'z_desired'])
                header.extend(['dx_actual', 'dy_actual', 'dz_actual'])
                header.extend(['dx_desired', 'dy_desired', 'dz_desired'])
                header.extend([f'tau_measured_{i+1}' for i in range(len(self.tau_history_new[0]))])
                header.extend([f'gravity_{i+1}' for i in range(len(self.tau_history_new[0]))])
                writer.writerow(header)
                
                for i, t in enumerate(self.time_history_new):
                    row = [t]
                    row.extend(self.tau_history_new[i])
                    row.extend(self.x_history_new[i][:3])
                    row.extend(self.x_des_history_new[i][:3])
                    row.extend(self.dx_history_new[i])
                    row.extend(self.dx_des_history_new[i])
                    row.extend(self.tau_measured_history_new[i])
                    row.extend(self.gravity_history_new[i])
                    writer.writerow(row)
                    
            self.get_logger().info(f'Successfully saved {len(self.tau_history_new)} data points to {filename}')

        except Exception as e:
            self.get_logger().error(f'Error when saving data: {str(e)}')
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')

def main(args=None):
    rclpy.init(args=args)
    cartesian_impedance_icra_validation_node = CartesianImpedanceICRAValidation()
    
    try:
        rclpy.spin(cartesian_impedance_icra_validation_node)
    except KeyboardInterrupt:
        cartesian_impedance_icra_validation_node.get_logger().info('Received keyboard interrupt, saving data...')
    except Exception as e:
        cartesian_impedance_icra_validation_node.get_logger().error(f'Error when running program: {str(e)}')
    finally:
        try:
            # save data to file only if signal handler has not been executed
            if not cartesian_impedance_icra_validation_node._signal_handled:
                cartesian_impedance_icra_validation_node.get_logger().info('Signal handler not executed, saving data to file...')
                cartesian_impedance_icra_validation_node.save_data_to_file()
            else:
                cartesian_impedance_icra_validation_node.get_logger().info('Signal handler executed, data already saved, skipping...')
                
        except Exception as e:
            cartesian_impedance_icra_validation_node.get_logger().error(f'Error when saving data: {str(e)}')
        
        cartesian_impedance_icra_validation_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 