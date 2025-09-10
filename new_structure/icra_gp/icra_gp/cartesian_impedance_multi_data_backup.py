#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_msgs.msg import StateParameter, EffortCommand, TaskSpaceCommand
from custom_msgs.srv import JointPositionAdjust
from std_msgs.msg import Bool
import numpy as np
from scipy.spatial.transform import Rotation
import signal
import csv
import traceback
import sys


class CartesianImpedanceMultiData(Node):
    
    def __init__(self):
        super().__init__('cartesian_impedance')
        
        # subscribe to /state_parameter
        self.param_subscription = self.create_subscription(
            StateParameter, '/state_parameter', self.stateParameterCallback, 10)
        
        # publish on /effort_command
        self.effort_publisher = self.create_publisher(
            EffortCommand, '/effort_command', 10)
        
        # create service client for joint position adjustment
        self.joint_position_client = self.create_client(
            JointPositionAdjust, '/joint_position_adjust')
        
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
        
        self.t_initial = None               # initial time
        self.t_last = None                  # last time

        self.task_command_received = False  # flag for task space command received
        self.x_des = None                   # desired position from task space command
        self.dx_des = None                  # desired velocity from task space command
        self.ddx_des = None                 # desired acceleration from task space command
        self.rotation_matrix_des = np.array(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)   # desired rotation matrix, z axis perpendicular to ground
        # joint position control state
        self.joint_position_service_called = False  # flag for joint position service call
        self.joint_position_control_active = True   # start with joint position control
        self.joint_position_adjusted = False        # flag for joint position adjustment
        self.trajectory_started = False             # flag for trajectory start, indicating the start of trajectory publishment

        # data recording control
        self.data_recording_enabled = False         # flag indicating whether to record data (controlled by trajectory_publisher)

        self.effort_msg = EffortCommand()
        self.get_logger().info('Cartesian Impedance controller node started')
        self.get_logger().info(f'Desired joint positions: {self.q_des}')
        self.get_logger().info(f'Joint position threshold: {self.joint_position_threshold}')
    
        # list for data recording
        self.time_history = []
        self.x_history = []

        # set signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        self._signal_handled = False                # flag to avoid repeated data saving

        # filename for data recording
        self.declare_parameter('filename', 'training_multi_data.csv')
        self.filename = self.get_parameter('filename').get_parameter_value().string_value
        self.get_logger().info(f'Data will be recorded in goal filepath: {self.filename}')
        
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
            else:
                t_elapsed = (t_now - self.t_initial).nanoseconds / 1e9
                dt = (t_now - self.t_last).nanoseconds / 1e9
                self.t_last = t_now                       

            # joint position control (for joint position adjustment before trajectory_publisher starts to work)
            if self.joint_position_control_active and not self.joint_position_adjusted:
                # check if joint positions are close enough to desired positions
                joint_error = np.linalg.norm(q - self.q_des)
                if joint_error < self.joint_position_threshold:
                    self.joint_position_adjusted = True
                    self.get_logger().info(f'Joint positions adjusted! Error: {joint_error:.6f}')
                    
                    # start trajectory by calling service
                    if not self.trajectory_started and not self.joint_position_service_called:
                        self.k_gains = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        # start transition, clear ros2_control interface buffer
                        tau = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        self.effort_msg.efforts = tau.tolist()
                        self.effort_publisher.publish(self.effort_msg)
                        self.start_trajectory()
                        return
                else:
                    # PD control for joint positions
                    self.i_error = self.i_error + (self.q_des - q) * dt
                    tau = self.k_pd * (self.q_des - q) + self.d_pd * (self.dq_des - dq) + self.i_pid * self.i_error
                    tau = np.clip(tau, -50.0, 50.0)
                    
                    # publish effort command
                    self.effort_msg.efforts = tau.tolist()
                    self.effort_publisher.publish(self.effort_msg)

                    return
            
            # draw a trajectory by hand
            if not self.trajectory_started:
                return
            
            # get O_T_F matrix and position
            o_t_f_array = np.array(msg.o_t_f)               # vectorized 4x4 pose matrix in flange frame, column-major
            o_t_f = o_t_f_array.reshape(4, 4, order='F')    # 4x4 pose matrix in flange frame, column-major
            x = o_t_f[:3, 3]                                # 3x1 position, only x-y-z

            # record position data
            self.time_history.append(t_elapsed)
            self.x_history.append(x.tolist())

        except Exception as e:
            self.get_logger().error(f'Parameter error: {str(e)}')

    def start_trajectory(self):
        """start trajectory by calling the joint position adjust service"""
        try:
            if not self.joint_position_client.service_is_ready():
                self.get_logger().warn('Joint position adjust service not ready, retrying...')
                return
            
            self.joint_position_service_called = True
            request = JointPositionAdjust.Request()
            request.q_des = self.q_des.tolist()
            request.dq_des = self.dq_des.tolist()
            
            future = self.joint_position_client.call_async(request)
            future.add_done_callback(self.trajectory_start_callback)
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
                self.trajectory_started = True
                # switch to drawing trajectory by hand
            else:
                self.trajectory_started = False         # reset flag to retry
        except Exception as e:
            self.get_logger().error(f'Error in trajectory start callback: {str(e)}')
            self.trajectory_started = False             # reset flag to retry

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
        if not self.x_history:
            self.get_logger().warning('No data to save - x_history is empty')
            return
            
        try:
            filename = self.filename
            
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                header = ['Time(s)']
                header.extend(['x_actual', 'y_actual', 'z_actual'])
                writer.writerow(header)
                
                for i, t in enumerate(self.time_history):
                    row = [t]
                    row.extend(self.x_history[i][:3])
                    writer.writerow(row)                    
            self.get_logger().info(f'Successfully saved {len(self.time_history)} data points to {filename}')
            
        except Exception as e:
            self.get_logger().error(f'Error when saving data: {str(e)}')
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')

def main(args=None):
    rclpy.init(args=args)
    cartesian_impedance_multi_data_node = CartesianImpedanceMultiData()

    try:
        rclpy.spin(cartesian_impedance_multi_data_node)
    except KeyboardInterrupt:
        cartesian_impedance_multi_data_node.get_logger().info('Received keyboard interrupt, saving data...')
    except Exception as e:
        cartesian_impedance_multi_data_node.get_logger().error(f'Error when running program: {str(e)}')
    finally:
        try:
            # save data to file only if signal handler has not been executed
            if not cartesian_impedance_multi_data_node._signal_handled:
                cartesian_impedance_multi_data_node.get_logger().info('Signal handler not executed, saving data to file...')
                cartesian_impedance_multi_data_node.save_data_to_file()
            else:
                cartesian_impedance_multi_data_node.get_logger().info('Signal handler executed, data already saved, skipping...')
                
        except Exception as e:
            cartesian_impedance_multi_data_node.get_logger().error(f'Error when saving data: {str(e)}')
        
        cartesian_impedance_multi_data_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 