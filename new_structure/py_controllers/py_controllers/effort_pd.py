#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_msgs.msg import StateParameter, EffortCommand
import numpy as np
import matplotlib.pyplot as plt
import time


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
        
        # 添加历史记录变量
        self.tau_history = []  # 记录所有tau命令
        self.time_history = []  # 记录对应的时间戳
        
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
            
            # 记录历史数据
            self.tau_history.append(tau.copy())
            self.time_history.append(t_elapsed)
            
            # Publish on topic /effort_command
            self.effort_msg.efforts = tau.tolist()
            self.effort_publisher.publish(self.effort_msg)
            
        except Exception as e:
            self.get_logger().error(f'Parameter error: {str(e)}')
    
    def plot_tau_history(self):
        """绘制tau命令的历史图像"""
        if not self.tau_history:
            self.get_logger().info('没有历史数据可绘制')
            return
            
        # 转换为numpy数组
        tau_array = np.array(self.tau_history)
        time_array = np.array(self.time_history)
        
        # 创建子图
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        fig.suptitle('Effort PD Controller - Tau Commands History', fontsize=16)
        
        # 绘制每个关节的tau命令
        joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7']
        for i in range(7):
            row = i // 2
            col = i % 2
            axes[row, col].plot(time_array, tau_array[:, i], 'b-', linewidth=2)
            axes[row, col].set_title(f'{joint_names[i]}')
            axes[row, col].set_xlabel('Time (s)')
            axes[row, col].set_ylabel('Tau (Nm)')
            axes[row, col].grid(True)
            axes[row, col].legend([f'Joint {i+1}'])
        
        # 隐藏最后一个空的子图
        axes[3, 1].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # 保存图像
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"tau_history_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.get_logger().info(f'图像已保存为: {filename}')
        
        # 打印统计信息
        self.get_logger().info(f'总共记录了 {len(self.tau_history)} 个数据点')
        self.get_logger().info(f'时间范围: {time_array[0]:.3f}s - {time_array[-1]:.3f}s')
        for i in range(7):
            tau_joint = tau_array[:, i]
            self.get_logger().info(f'Joint {i+1}: 最大值={np.max(tau_joint):.3f}Nm, 最小值={np.min(tau_joint):.3f}Nm, 平均值={np.mean(tau_joint):.3f}Nm')


def main(args=None):
    rclpy.init(args=args)
    effort_pd_node = EffortPDController()
    try:
        rclpy.spin(effort_pd_node)
    except KeyboardInterrupt:
        effort_pd_node.get_logger().info('收到Ctrl+C信号，正在绘制tau命令历史图像...')
        effort_pd_node.plot_tau_history()
    finally:
        effort_pd_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 