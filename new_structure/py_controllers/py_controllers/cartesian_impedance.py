#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_msgs.msg import StateParameter, EffortCommand, TaskSpaceCommand
import numpy as np
import matplotlib.pyplot as plt
import signal
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
        
        self.declare_parameter('k_gains', [1200, 500, 1000, 200, 200, 200])
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
    
        # 数据记录列表
        self.tau_history = []
        self.F_history = []
        self.time_history = []
        self.x_history = []
        self.x_des_history = []
        self.dx_history = []           
        self.dx_des_history = []  

        # 设置信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

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
            self.get_logger().info(f"x: {x.tolist()}, dx: {dx.tolist()}")

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
            print(f'published on topic /effort_command: {tau}')
            print(f'self.effort_msg: {self.effort_msg}')
            self.effort_publisher.publish(self.effort_msg)
            self.get_logger().debug(f'published on topic /effort_command: {tau}')
            
            # Record Data
            self.tau_history.append(tau.tolist())
            # self.F_history.append(F.tolist())
            self.time_history.append(t_elapsed)
            self.x_history.append(x.tolist())
            self.x_des_history.append(self.x_des.tolist())
            self.dx_history.append(dx[:3].tolist())      
            self.dx_des_history.append(self.dx_des[:3].tolist()) 

        except Exception as e:
            self.get_logger().error(f'Parameter error: {str(e)}')

    def signal_handler(self, signum, frame):
        """信号处理器，在程序被中断时调用绘图函数"""
        self.get_logger().info(f'收到信号 {signum}，正在绘制数据...')
        self.plot_data()
        sys.exit(0)
    
    def plot_data(self):
        """绘制记录的6个子图数据"""
        if not self.tau_history:
            self.get_logger().info('没有数据可绘制')
            return
            
        try:
            # 创建2行3列的子图
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Cartesian Impedance Controller Data', fontsize=16)
            
            # 第一行：关节力矩、速度对比、位置误差
            # 1. 关节力矩图
            tau_history_array = np.array(self.tau_history)
            for i in range(tau_history_array.shape[1]):
                axes[0, 0].plot(self.time_history, tau_history_array[:, i], label=f'Joint {i+1}')
            axes[0, 0].set_title('Joint Torques (tau)')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Torque (Nm)')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 2. 期望与实际速度对比图
            dx_history_array = np.array(self.dx_history)
            dx_des_history_array = np.array(self.dx_des_history)
            
            # 绘制x方向速度
            axes[0, 1].plot(self.time_history, dx_history_array[:, 0], 'b-', label='Actual dx', linewidth=2)
            axes[0, 1].plot(self.time_history, dx_des_history_array[:, 0], 'r--', label='Desired dx', linewidth=2)
            # 绘制y方向速度
            axes[0, 1].plot(self.time_history, dx_history_array[:, 1], 'g-', label='Actual dy', linewidth=2)
            axes[0, 1].plot(self.time_history, dx_des_history_array[:, 1], 'm--', label='Desired dy', linewidth=2)
            # 绘制z方向速度
            axes[0, 1].plot(self.time_history, dx_history_array[:, 2], 'c-', label='Actual dz', linewidth=2)
            axes[0, 1].plot(self.time_history, dx_des_history_array[:, 2], 'y--', label='Desired dz', linewidth=2)
            
            axes[0, 1].set_title('Desired vs Actual Velocity')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Velocity (m/s)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 3. 位置误差的欧几里得距离
            x_history_array = np.array(self.x_history)
            x_des_history_array = np.array(self.x_des_history)
            
            # 计算欧几里得距离
            position_errors = []
            for i in range(len(self.x_history)):
                actual_pos = np.array(self.x_history[i][:3])  # 只取x, y, z
                desired_pos = np.array(self.x_des_history[i][:3])
                error = np.linalg.norm(actual_pos - desired_pos)
                position_errors.append(error)
            
            axes[0, 2].plot(self.time_history, position_errors, 'r-', linewidth=2)
            axes[0, 2].set_title('Position Error (Euclidean Distance)')
            axes[0, 2].set_xlabel('Time (s)')
            axes[0, 2].set_ylabel('Error (m)')
            axes[0, 2].grid(True)
            
            # 第二行：x, y, z位置轨迹
            # 4. X位置轨迹
            axes[1, 0].plot(self.time_history, x_history_array[:, 0], 'b-', label='Actual X', linewidth=2)
            axes[1, 0].plot(self.time_history, x_des_history_array[:, 0], 'r--', label='Desired X', linewidth=2)
            axes[1, 0].set_title('X Position Trajectory')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Position (m)')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 5. Y位置轨迹
            axes[1, 1].plot(self.time_history, x_history_array[:, 1], 'b-', label='Actual Y', linewidth=2)
            axes[1, 1].plot(self.time_history, x_des_history_array[:, 1], 'r--', label='Desired Y', linewidth=2)
            axes[1, 1].set_title('Y Position Trajectory')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Position (m)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            # 6. Z位置轨迹
            axes[1, 2].plot(self.time_history, x_history_array[:, 2], 'b-', label='Actual Z', linewidth=2)
            axes[1, 2].plot(self.time_history, x_des_history_array[:, 2], 'r--', label='Desired Z', linewidth=2)
            axes[1, 2].set_title('Z Position Trajectory')
            axes[1, 2].set_xlabel('Time (s)')
            axes[1, 2].set_ylabel('Position (m)')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
            
            plt.tight_layout()
            
            # 保存图片
            plt.savefig('cartesian_impedance_controller_data.png', dpi=300, bbox_inches='tight')
            self.get_logger().info('数据图已保存为 cartesian_impedance_controller_data.png')
            
            # 显示图片
            plt.show()
            
        except Exception as e:
            self.get_logger().error(f'绘图时发生错误: {str(e)}')
    
    def save_data_to_file(self):
        """将数据保存到CSV文件"""
        if not self.tau_history:
            return
            
        try:
            import csv
            filename = 'cartesian_impedance_controller_data.csv'
            
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # 写入表头
                header = ['Time(s)']
                header.extend([f'tau_{i+1}' for i in range(len(self.tau_history[0]))])
                header.extend(['x_actual', 'y_actual', 'z_actual'])
                header.extend(['x_desired', 'y_desired', 'z_desired'])
                header.extend(['dx_actual', 'dy_actual', 'dz_actual'])
                header.extend(['dx_desired', 'dy_desired', 'dz_desired'])
                writer.writerow(header)
                
                # 写入数据
                for i, t in enumerate(self.time_history):
                    row = [t]
                    row.extend(self.tau_history[i])
                    row.extend(self.x_history[i][:3])
                    row.extend(self.x_des_history[i][:3])
                    row.extend(self.dx_history[i])
                    row.extend(self.dx_des_history[i])
                    writer.writerow(row)
                    
            self.get_logger().info(f'数据已保存到 {filename}')
            
        except Exception as e:
            self.get_logger().error(f'保存数据时发生错误: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    cartesian_impedance_node = CartesianImpedanceController()
    try:
        rclpy.spin(cartesian_impedance_node)
    except KeyboardInterrupt:
        cartesian_impedance_node.get_logger().info('收到键盘中断信号，正在保存数据...')
    except Exception as e:
        cartesian_impedance_node.get_logger().error(f'程序运行时发生错误: {str(e)}')
    finally:
        try:
            # 保存数据到文件
            cartesian_impedance_node.save_data_to_file()
            # 绘制数据
            cartesian_impedance_node.plot_data()
        except Exception as e:
            cartesian_impedance_node.get_logger().error(f'保存数据或绘图时发生错误: {str(e)}')
        
        cartesian_impedance_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 