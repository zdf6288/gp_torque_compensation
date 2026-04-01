#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_msgs.msg import StateParameter, EffortCommand, TaskSpaceCommand
from custom_msgs.srv import JointPositionAdjust
from custom_msgs.srv import AsyncGPpredict
from custom_msgs.srv import GetFutureTrajectory

from std_msgs.msg import Bool
import numpy as np
from scipy.spatial.transform import Rotation
import signal
import csv
import traceback
import sys
import os, pickle
import importlib.util
import threading, time
from std_msgs.msg import String
from collections import deque

class CartesianImpedanceController(Node):
    
    def __init__(self):
        super().__init__('cartesian_impedance')
        
        # subscribe to /state_parameter
        self.param_subscription = self.create_subscription(
            StateParameter, '/state_parameter', self.stateParameterCallback, 10)
        
        # subscribe to /task_space_command
        self.task_command_subscription = self.create_subscription(
            TaskSpaceCommand, '/task_space_command', self.taskCommandCallback, 10)
        
        # subscribe to /data_recording_enabled to know when to start recording data
        self.data_recording_subscription = self.create_subscription(
            Bool, '/data_recording_enabled', self.dataRecordingCallback, 10)
        
        # publish on /effort_command
        self.effort_publisher = self.create_publisher(
            EffortCommand, '/effort_command', 10)
        
        # create service client for joint position adjustment
        self.joint_position_client = self.create_client(
            JointPositionAdjust, '/joint_position_adjust')
        
        # Ablation parameter
        self.gp_mode_sub = self.create_subscription(
            String, "/gp_mode", self.gp_mode_callback, 10
        )   
        self.shutdown_sub = self.create_subscription(
            Bool, "/shutdown_control", self.shutdown_callback, 10
        )
        
        self.declare_parameter('k_pd', [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 2.0])    # k_gains in PD control (joint space)
        self.declare_parameter('d_pd', [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])    # d_gains in PD control (joint space)
        self.declare_parameter('i_pid', [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0])        # i_gains supplement to PD control (joint space)
        self.k_pd = np.array(self.get_parameter('k_pd').value, dtype=float)
        self.d_pd = np.array(self.get_parameter('d_pd').value, dtype=float)
        self.i_pid = np.array(self.get_parameter('i_pid').value, dtype=float)
        self.i_error = np.zeros(7)

        self.declare_parameter('k_gains', [1500.0, 1250.0, 1500.0, 25.0, 25.0, 0.0])   # k_gains in impedance control (task space)
        self.k_gains = np.array(self.get_parameter('k_gains').value, dtype=float)
        self.K_gains = np.diag(self.k_gains)

        self.declare_parameter('D_gains', [150.0, 125.0, 150.0, 1.0, 1.0, 0.0])   # d_gains in impedance control (task space)
        self.d_gains = np.array(self.get_parameter('D_gains').value, dtype=float)
        self.d_gains = np.diag(self.d_gains)
        self.eta = 1.00                                                              # for calculating d_gains

        self.declare_parameter('kpn_gains', [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])    # kpn_gains for nullspace 
        self.kpn_gains = np.array(self.get_parameter('kpn_gains').value, dtype=float)
        self.dpn_gains = 1 * np.sqrt(np.array(self.kpn_gains))                        # dpn_gains for nullspace

        self.x_i_error = np.zeros(6, dtype=float)
        self.i_gains = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.prev_x_error = np.zeros(6, dtype=float)
        self.prev_dx_error = np.zeros(6, dtype=float)

        #friction compensation
        self.Fc = np.array([0.15, 0.95, 0.9, 0.15, 0.1, 0.1, 0.0], dtype=float)
        self.Bv = np.array([0, 0, 0, 0, 0, 0, 0.0], dtype=float)
        self.v_eps = 0.01
        
        # --- task-space integral ---
        self.declare_parameter('ki_task', [200.0, 0.0, 2000.0, 0.0, 0.0, 0.0])  # 先只给 z 积分
        self.ki_task = np.array(self.get_parameter('ki_task').value, dtype=float)
        self.Ki_task = np.diag(self.ki_task)

        self.e_int = np.zeros(6)          # 对应 x_error/dx_error 的6维
        self.e_int_limit = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # 防止windup，可调

        # Joint position control parameters
        self.declare_parameter('q_des', [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.0])     # desired joint positions
        self.declare_parameter('dq_des', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])               # desired joint velocities
        self.declare_parameter('joint_position_threshold', 0.4)                             # threshold for joint position convergence
        self.q_des = np.array(self.get_parameter('q_des').value, dtype=float)
        self.dq_des = np.array(self.get_parameter('dq_des').value, dtype=float)
        self.joint_position_threshold = self.get_parameter('joint_position_threshold').value
        
        self.q_initial = None               # initial joint position q0
        self.t_initial = None               # initial time
        self.t_last = None                  # last time
        self.dq_buffer = None               # buffer for joint velocity dq
        self.dq = None

        self.dq_raw = np.zeros(7)
        self.dq_filt = np.zeros(7)
        self.dq_filt_initialized = False

        # dq 一阶低通截止频率，先保守一点
        self.declare_parameter("dq_lpf_hz", 30.0)
        self.dq_lpf_hz = float(self.get_parameter("dq_lpf_hz").value)

        self.zero_jacobian_buffer = None    # buffer for zero jacobian matrix in flange frame
        self.jacobian_buffer = None         # buffer for jacobian matrix in flange frame
        self.declare_parameter("dls_lambda", 0.1)     # 阻尼系数 λ，0.01~0.2 常用
        self.declare_parameter("dls_lambda_ns", 0.1)  # nullspace 用的 λ（可与上面一样）
        self.dls_lambda = float(self.get_parameter("dls_lambda").value)
        self.dls_lambda_ns = float(self.get_parameter("dls_lambda_ns").value)

        self.djacobian = None

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
        self.tau_buffer = np.zeros_like(self.effort_msg.efforts)    # buffer for tau
    
        # list for data recording
        self.tau_history = []
        self.time_history = []
        self.x_history = []
        self.x_des_history = []
        self.dx_history = []           
        self.dx_des_history = []
        self.tau_measured_history = []
        self.gravity_history = []
        self.q_history = []
        self.dq_history = []
        self.dq_des_joint_history = []   # desired joint velocity
        self.ddq_des_joint_history = []  # desired joint acceleration
        # set signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        self._signal_handled = False                # flag to avoid repeated data saving


        # === GP ===
        self.q = np.zeros(7)
        self.dq_des_joint = np.zeros(7)
        self.ddq_des_joint = np.zeros(7)
        self.tau_residual = np.zeros(7)
        self.tau_memory = np.zeros(7)

        self.gp_stride = 1      # 每 10 个 state callback 做一次 GP（你可以调）
        self.gp_counter = 0

        self.y_hat_filtered = np.zeros(7)
        self.y_hat_alpha = 0.05   # 越小越平滑，0.01~0.1 合理      
        self.y_hat_combined = np.zeros(7)
        self.tau_filtered = np.zeros(7) 

        # 预测节流（你已有）from
        self.gp_update_y_clip = 10.0         # 训练用残差的幅值上限（Nm）

        self.y_hat_history = []          # 记录每次控制回路使用的 y_hat (7,)
        self.tau_residual_filtered = np.zeros(7)  # 滤波后的 tau_residual (7,)
        self.tau_residual_history = []   # 记录 tau_residual (7,)
        self.tau_residual_raw_history = []   # 滤波前残差

        # === local GP ===
        self.gp_models_small = {}
        self.gp_models_big = {}
        self.gp_ready = False    # 标记本地 GP 是否加载成功
        self.y_hat_local = np.zeros(7)
        self.y_hat_local_history = []
        self.offline_limit = 0

        # ===== local GP history memory =====
        self.printer = 0
        self.gp_hist_len = 250              # 固定长度，可调
        self.gp_hist_topk = 2                # 最近邻个数
        self.gp_hist_alpha_max = 0.35        # 历史项最多占比
        self.gp_hist_min_points = 20         # 少于这个数量不启用历史项

        # 每个元素：
        # {
        #   "x": np.ndarray(shape=(d,)),
        #   "y": np.ndarray(shape=(7,)),
        #   "var": np.ndarray(shape=(7,))
        # }
        self.local_gp_history = deque(maxlen=self.gp_hist_len)


        # 本地加载离线训练模型
        self._load_gp_models("./new_structure/gp/gp_models")

        if self.gp_ready:
            self.get_logger().info(f"[Controller] Local GP models loaded, will run local GP in control loop")
        else:
            self.get_logger().warn(f"[Controller] Local GP models NOT loaded, only using cloud GP")

        # GP service client
        self.gp_client = self.create_client(AsyncGPpredict, '/gp_predict')
        self.use_gp = True

        self.gp_sample_n = 10   # 比如每次随机采样 10 个点
        self.gp_sample_sigma = 0.02  # 采样扰动尺度
        
        self._gp_lock = threading.Lock()
        self._latest_y_hat = np.zeros(7, dtype=float)
        self._gp_warned = False  # 避免一直刷 warn
        self.y_hat_cloud = np.zeros(7)
        self.y_hat_cloud_history = []

        self.future_n_samples = 0         # 未来采样点个数
        self.future_ddq_noise_std = 0.1    # rad/s^2，加速度噪声强度    

        self.gp_send_seq = 0        # 发送给 cloud 的请求编号
        self.gp_recv_seq = -1      # 最近一次收到的请求编号

        self._cloud_lock = threading.Lock()
        self.cloud_state_valid = False

        self.q_cloud_used  = np.zeros(7)
        self.dq_cloud_used = np.zeros(7)
        self.ddq_cloud_used = np.zeros(7)
        self.y_hat_cloud_latest = np.zeros(7)
        self.cloud_seq_latest = -1

        # 记录误差（可选）
        self.q_err_history = []
        self.dq_err_history = []
        self.y_err_history = []


        # ===== Alpha-Beta filter for tau =====
        self.tau_ab      = np.zeros(7)   # τ̂
        self.dtau_ab     = np.zeros(7)   # τ̇̂

        self.tau_alpha = 0.1
        self.tau_beta  = 0.005

        #simulated dalay
        self.future_delay = self.declare_parameter(
            'future_delay', 0.015 # 默认 60 ms
        ).value
        self.delay_steps = 0
        self.state_delay_steps = 0   # 你想模拟的通信延迟：20个周期
        self.state_buffer = deque(maxlen=1000)  # 存2秒(1kHz)都够
        self.cloud_delay_steps = 100
        self.y_hat_cloud_buffer = deque(maxlen=self.cloud_delay_steps)
        
        # cloud queue
        self.cloud_queue = deque(maxlen=500)        # 存 cloud rollout 点 (7,)
        self.cloud_var_queue = deque(maxlen=500)    # 存对应方差 (7,)

        self.y_hat_cloud_hold = np.zeros(7)
        self.var_cloud_hold = np.ones(7) * 1e6

        # variance
        self.var_local = np.ones(7) * 1e6
        self.var_cloud = np.ones(7) * 1e6

        #cloud time pid
        self.q_corr_to_cloud  = np.zeros(7)
        self.dq_corr_to_cloud = np.zeros(7)
        self.corr_alpha = 0.1  # 低通系数
        # Prediction history memory
        self.y_hat_mem = np.zeros(7)
        self.y_hat_mem_history = []

        ## Ablation parameters
        self.declare_parameter("gp_mode", "fusion")  
        self.gp_mode = self.get_parameter("gp_mode").value
        self.get_logger().info(f"[GP] Running mode = {self.gp_mode}")


        # 只在轨迹真正开始之后再用 GP（由 /data_recording_enabled 控制）
        self.gp_active = False
        # 不要 while 死等，只尝试一次
        if not self.gp_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('/gp_predict service not available at start, GP disabled')
            self.use_gp = False
        else:
            self.get_logger().info('/gp_predict service is ready')

        # 未来轨迹 service client
        self.future_traj_client = self.create_client(
            GetFutureTrajectory,
            '/future_task_space'
        )

        # 存最新一次未来轨迹
        self._latest_future_traj = None   # dict: {"x_des": np.array(6,), "dx_des": ..., "ddx_des": ...}
        self._future_traj_counter = 0
        self._future_traj_warned = False

    def dls_dyn_pinv(self, J, M, lam):
        """
        Dynamically consistent DLS pseudoinverse:
        J# = M^{-1} J^T (J M^{-1} J^T + lam^2 I)^{-1}
        J: m×n, M: n×n
        """
        m = J.shape[0]
        Minv = np.linalg.inv(M)

        A = J @ Minv @ J.T + (lam**2) * np.eye(m)   # m×m
        return Minv @ J.T @ np.linalg.solve(A, np.eye(m))  # n×m

    def taskCommandCallback(self, msg):
        """callback function for /task_space_command subscriber"""
        self.task_command_received = True
        self.x_des = np.array(msg.x_des)
        self.dx_des = np.array(msg.dx_des)
        self.ddx_des = np.array(msg.ddx_des)
        
    def dataRecordingCallback(self, msg):
        """callback function for /data_recording_enabled subscriber"""
        self.data_recording_enabled = msg.data

        # 当 TrajectoryPublisher 认为“transition 完成”时，会发 True
        if msg.data and not self.gp_active:
            self.gp_active = True
            self.get_logger().info("[Controller] Data recording enabled -> GP compensation ACTIVATED")
        elif not msg.data and self.gp_active:
            # 如果你希望停轨迹时也关掉 GP，可以顺便关掉
            self.gp_active = False
            self.get_logger().info("[Controller] Data recording disabled -> GP compensation DEACTIVATED")

    def _future_traj_response_callback(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"[Controller] /future_task_space call failed: {e}")
            return

        x_f  = np.array(res.x_des, dtype=float)
        dx_f = np.array(res.dx_des, dtype=float)
        ddx_f = np.array(res.ddx_des, dtype=float)

        self._latest_future_traj = {
            "x_des": x_f,
            "dx_des": dx_f,
            "ddx_des": ddx_f,
        }
        # 调试时可以看看
        # self.get_logger().debug(f"Got future traj: x={x_f[:3]}")
    
    def gp_mode_callback(self, msg):
        self.gp_mode = msg.data
        self.get_logger().info(f"[Controller] GP mode switched to: {self.gp_mode}")

    def shutdown_callback(self, msg):
        if msg.data:
            self.get_logger().info("[Controller] Received shutdown signal — stopping robot, saving data & exiting.")

            # ------------------------------------------
            # 1) 立即停止力矩输出（关键！！！）
            # ------------------------------------------
            try:
                zero_tau = EffortCommand()
                zero_tau.efforts = [0.0] * 7
                self.effort_publisher.publish(zero_tau)
                self.get_logger().info("[Controller] Published zero torque to stop robot.")
            except Exception as e:
                self.get_logger().error(f"Error publishing zero torque: {e}")

            # ------------------------------------------
            # 2) 停止后再保存数据
            # ------------------------------------------
            try:
                self.save_data_to_file()
            except Exception as e:
                self.get_logger().error(f"Error saving data: {e}")

            # ------------------------------------------
            # 3) 自动画图
            # ------------------------------------------
            try:
                os.system("python3 ablation.py cartesian_impedance_controller_data.csv")
                self.get_logger().info("[Controller] Plotting completed.")
            except Exception as e:
                self.get_logger().error(f"Plotting error: {e}")

            # ------------------------------------------
            # 4) 安全退出
            # ------------------------------------------
            rclpy.shutdown()
            os._exit(0)
    

    def stateParameterCallback(self, msg):
        """callback function for /state_parameter subscriber"""
        try:
            # initialize t_initial, get t_elapsed, t_last and dt
            # initialize q_initial, get q, dq and ddq
            t_now = self.get_clock().now()
            q = np.array(msg.position, dtype=float)
            self.q = q

            dq_raw = np.array(msg.velocity, dtype=float)
            self.dq_raw = dq_raw
            if self.t_initial is None:
                self.t_initial = t_now
                self.t_last = t_now
                t_elapsed = 0.0
                dt = 1e-3   # 比原来的 1e-1 更合理，避免初值太怪

                self.dq_filt = dq_raw.copy()
                self.dq_filt_initialized = True

                # dq = self.dq_filt.copy()
                # self.dq = dq
                dq = dq_raw
                self.dq = dq

                ddq = np.zeros_like(dq)
                self.dq_buffer = dq.copy()
            else:
                t_elapsed = (t_now - self.t_initial).nanoseconds / 1e9
                dt = (t_now - self.t_last).nanoseconds / 1e9
                print(dt)
                self.t_last = t_now

                # 防止极小 dt
                dt = max(dt, 1e-6)

                if not self.dq_filt_initialized:
                    self.dq_filt = dq_raw.copy()
                    self.dq_filt_initialized = True
                else:
                    self.dq_filt = self._lowpass_vector(
                        dq_raw, self.dq_filt, dt, self.dq_lpf_hz
                    )

                dq = self.dq_filt.copy()
                self.dq = dq

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
                        # start transition, clear ros2_control interface buffer
                        tau = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        self.effort_msg.efforts = tau.tolist()
                        self.effort_publisher.publish(self.effort_msg)
                        self.start_trajectory()
                else:
                    # PD control for joint positions
                    e_q  = (self.q_des - q)
                    e_dq = (self.dq_des - dq)

                    tau_unsat = self.k_pd * e_q + self.d_pd * e_dq + self.i_pid * self.i_error
                    tau_sat   = np.clip(tau_unsat, -50.0, 50.0)

                    # anti-windup gain（每关节都可以不同）
                    k_aw = 15.0  # 典型 1~20 之间调；越大回退越快

                    # back-calculation: 积分更新（注意是用“力矩差”回退）
                    self.i_error = self.i_error + (e_q + k_aw * (tau_sat - tau_unsat)) * dt

                    # final command
                    tau = self.k_pd * e_q + self.d_pd * e_dq + self.i_pid * self.i_error
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
            gravity_measured = np.array(msg.gravity)
            tau_measured = np.array(msg.effort_measured)

            o_t_f = o_t_f_array.reshape(4, 4, order='F')                    # 4x4 pose matrix in flange frame, column-major
            mass_matrix = mass_matrix_array.reshape(7, 7, order='F')        # 7x7
            coriolis_matrix = np.diag(coriolis_matrix_array)                # 7x7
            zero_jacobian = zero_jacobian_array.reshape(6, 7, order='F')    # 6x7
            zero_jacobian_t = zero_jacobian.T                               # 7x6, transpose of zero_jacobian
            # zero_jacobian_pinv = np.linalg.pinv(zero_jacobian)              # 7x6, pseudoinverse obtained by SVD
            lam = self.dls_lambda
            lam_ns = self.dls_lambda_ns
            # 6×7
            zero_jacobian_pinv = self.dls_dyn_pinv(zero_jacobian, mass_matrix, lam_ns)   # 7×6       

            # to control the z axis perpendicular to ground, use 4*7 jacobian matrix
            # to control the z axis perpendicular to ground, use 5x7 Jacobian (3 pos + 2 rot constraints)
            jacobian = zero_jacobian[:5, :]        # 5x7
            jacobian_t = jacobian.T                # 7x5

            # DLS pseudoinverse: 7x5
            jacobian_pinv = self.dls_dyn_pinv(jacobian, mass_matrix, lam)   # 7x5
            if self.jacobian_buffer is None:
                djacobian = np.zeros_like(jacobian)
            else:
                djacobian = (jacobian - self.jacobian_buffer) / dt
                
            self.jacobian_buffer = jacobian.copy()
            # get x and dx
            x = o_t_f[:3, 3]            # 3x1 position, only x-y-z
            dx = zero_jacobian @ dq     # 6x1 velocity

            # === 映射任务空间期望到关节空间的期望速度/加速度 ===
            # 任务空间只用前5维（你的控制是5 DoF：3平移+2姿态约束）
            dx_des_5  = self.dx_des[:5]
            ddx_des_5 = self.ddx_des[:5]

            dq_des_joint = jacobian_pinv @ dx_des_5
            self.dq_des_joint = dq_des_joint
            ddq_des_joint = jacobian_pinv @ (ddx_des_5 - djacobian @ dq)
            self.ddq_des_joint = ddq_des_joint

            # 记录（仅当开启录数时）
            if self.data_recording_enabled:
                self.dq_des_joint_history.append(dq_des_joint.tolist())
                self.ddq_des_joint_history.append(ddq_des_joint.tolist())

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
            # 单独设置 Z 轴 damping（第 3 个分量，索引 2）
        
            # d_gains[2] = 0.9*d_gains[2]

            D_gains = np.diag(d_gains)
            
            # 只对 Z 轴积分
            # 分别对 x / y / z 三个方向积分

            self.x_i_error[0] += x_error[0] * dt
            self.x_i_error[0] = np.clip(self.x_i_error[0], -0.02, 0.02)

            self.x_i_error[1] += x_error[1] * dt
            self.x_i_error[1] = np.clip(self.x_i_error[1], -0.15, 0.15)

            self.x_i_error[2] += x_error[2] * dt
            self.x_i_error[2] = np.clip(self.x_i_error[2], -0.15, 0.15)

            # 三个方向各自的积分项
            i_term = np.zeros(6)
            i_term[0] = self.i_gains[0] * self.x_i_error[0]   # X
            i_term[1] = self.i_gains[1] * self.x_i_error[1]   # Y
            i_term[2] = self.i_gains[2] * self.x_i_error[2]   # Z

            pd_term = self.K_gains @ x_error + D_gains @ dx_error + i_term

            tau = (
                mass_matrix @ jacobian_pinv @ self.ddx_des[:5]
                + (coriolis_matrix - mass_matrix @ jacobian_pinv@ djacobian)
                    @ jacobian_pinv @ dx[:5]
                - jacobian_t @ pd_term[:5]
            )

            # tau_nullspace = ((np.eye(7) - zero_jacobian_pinv @ zero_jacobian) 
            #     @ (self.dpn_gains * (self.dq_des - dq)))
            N = np.eye(7) - zero_jacobian_pinv @ zero_jacobian   # or using your 5DoF jacobian
            tau_nullspace = N.T @ (- self.dpn_gains * dq)        # dq_des = 0 时就是减振
            tau = tau + tau_nullspace
            tau = tau + self.friction_compensation(dq)

            # === 计算残差 ===
            tau_residual = tau_measured - tau - gravity_measured
            if self.data_recording_enabled:
                self.tau_residual_raw_history.append(tau_residual.tolist())
            self.tau_residual_filtered = (
                0.02 * tau_residual + 0.98 * self.tau_residual_filtered
            )

            # tau = tau

            # === 异步请求未来轨迹（给 GP 用），比如 100Hz 请求一次 ===
            if self.future_traj_client.service_is_ready():
                self._future_traj_counter += 1
                if self._future_traj_counter % 1 == 0:
                    req_ft = GetFutureTrajectory.Request()
                    req_ft.t_delay = float(self.future_delay)
                    future_ft = self.future_traj_client.call_async(req_ft)
                    future_ft.add_done_callback(self._future_traj_response_callback)
            else:
                if not self._future_traj_warned:
                    self.get_logger().warn(
                        "[Controller] /future_task_space not ready; GP 没有未来轨迹信息"
                    )
                    self._future_traj_warned = True

            # === 控制循环的最后：按节拍触发一次“GP 更新”（本地 + 云端） ===
            if self.gp_active and self.use_gp:
                self.gp_counter += 1
                tick = (self.gp_counter % self.gp_stride == 0)
                # tick = True

                # ---------------------------------------------------------
                # A) 每帧：先消费 cloud 队列（填补空隙）
                # ---------------------------------------------------------
                if len(self.cloud_queue) > 0:
                    self.y_hat_cloud_hold = self.cloud_queue.popleft()
                    self.var_cloud_hold   = self.cloud_var_queue.popleft()

                self.y_hat_cloud = self.y_hat_cloud_hold
                self.var_cloud   = self.var_cloud_hold
                # ---------------------------------------------------------
                # B) local：只有 tick 更新，否则 hold
                # ---------------------------------------------------------
                if tick:
                    self.tau_memory = self.tau_residual_filtered

                    x_query = self._build_gp_feature(self.q, dq, self.ddq_des_joint)

                    # 1) 当前 local GP
                    y_hat_local_now, var_local_now = self._gp_predict_and_update(
                        self.q, dq, self.ddq_des_joint,
                        self.tau_residual_filtered,
                        self.gp_models_small,
                        update=True
                    )

                    # 2) 历史最近邻记忆项
                    y_hat_hist, var_hist, alpha_hist = self._query_local_gp_history(x_query)
                    self.y_hat_mem = y_hat_hist

                    # 3) 融合
                    # self.y_hat_local = (1.0 - alpha_hist) * y_hat_local_now + alpha_hist * y_hat_hist
                    self.y_hat_local = y_hat_local_now
                    self.var_local   = (1.0 - alpha_hist) * var_local_now   + alpha_hist * var_hist

                    # 4) 当前结果写入历史池
                    self._append_local_gp_history(x_query, y_hat_local_now, var_local_now)

                    # 5) rollout 保持你原来的逻辑
                    M = self.gp_stride
                    dt_step = 0.02
                    q_i  = self.q.copy()
                    dq_i = dq.copy()
                    ddq_i = self.ddq_des_joint.copy()
                    tau_recursive = self.tau_residual_filtered

                    y_i, v_i = self._gp_predict_and_update(
                        q_i, dq_i, ddq_i,
                        tau_recursive,
                        self.gp_models_big,
                        update=True
                    )
                    # self.y_hat_cloud = y_i
                    self._append_local_gp_history(x_query, y_i, v_i)
                    for i in range(M - 1):
                        q_i  = q_i  + dq_i * dt_step + 0.5 * ddq_i * (dt_step**2)
                        dq_i = dq_i + ddq_i * dt_step
                        y_i, v_i = self._gp_predict_and_update(
                            q_i, dq_i, ddq_i,
                            tau_recursive,
                            self.gp_models_big,
                            update=True
                        )
                        
                        tau_recursive = 1.0 * y_i + 0.0 * tau_recursive

                    self.cloud_queue.append(y_i.copy())
                    self.cloud_var_queue.append(v_i.copy())
                else:
                    self.var_local = np.minimum(self.var_local * 1.02, 1e6)
                
                # ---------------------------------------------------------
                # C) 每帧融合（不要只在 else 融合）
                # ---------------------------------------------------------
                eps = 1e-8
                v_l = np.maximum(self.var_local, eps)
                v_c = np.maximum(self.var_cloud, eps)

                prec_l = 1.0 / v_l
                prec_c = 1.0 / v_c
                w_l = prec_l / (prec_l + prec_c)

                self.y_hat_combined = w_l * self.y_hat_local + (1.0 - w_l) * self.y_hat_cloud

            # tau = tau - self.y_hat_local
            tau = tau

            # publish on topic /effort_command
            self.effort_msg.efforts = tau.tolist()
            self.effort_publisher.publish(self.effort_msg)

            # record data only when data recording is enabled
            if self.data_recording_enabled:
                self.tau_history.append(tau.tolist())
                self.time_history.append(t_elapsed)
                self.x_history.append(x.tolist())
                self.x_des_history.append(self.x_des.tolist())
                self.dx_history.append(dx[:3].tolist())
                self.dx_des_history.append(self.dx_des[:3].tolist())
                self.tau_measured_history.append(np.array(msg.effort_measured).tolist())
                self.gravity_history.append(np.array(msg.gravity).tolist())
                self.q_history.append(q.tolist())
                self.dq_history.append(dq.tolist())

                self.y_hat_history.append(self.y_hat_combined.tolist())      # combined
                self.y_hat_local_history.append(self.y_hat_local.tolist())    # 上一帧或刚更新的 local
                self.y_hat_cloud_history.append(self.y_hat_cloud.tolist())    # 上一帧或刚更新的 cloud
                self.y_hat_mem_history.append(self.y_hat_mem.tolist())
                self.tau_residual_history.append(self.tau_residual_filtered.tolist())

        except Exception as e:
            self.get_logger().error(f'Parameter error: {str(e)}')

    def predict_future_joint_state(q, dq, ddq_des, delay):
        dq_future = dq + ddq_des * delay
        q_future  = q + dq * delay + 0.5 * ddq_des * delay**2
        return q_future, dq_future

    def _sample_future_task_space(self, dx_f, ddx_f, n_samples=10, sigma=0.02):
        samples = []
        for _ in range(n_samples):
            dx_f_i  = dx_f.copy()
            ddx_f_i = ddx_f.copy()

            # --- 只扰动任务空间前5维（你的控制任务是5维）
            dx_f_i[:5]  += np.random.normal(0, sigma, size=5)
            ddx_f_i[:5] += np.random.normal(0, sigma, size=5)

            samples.append((dx_f_i, ddx_f_i))
        return samples

    def _build_gp_feature(self, q, dq_des_joint, ddq_des_joint=None):
        """
        和 local GP 使用同一套输入特征。
        当前版本使用 14 维: [q, dq_des_joint]
        如果你以后训练改成 21 维，再切到 [q, dq_des_joint, ddq_des_joint]
        """
        x_full = np.concatenate([q, dq_des_joint]).astype(np.float32)
        return x_full
    
    def _query_local_gp_history(self, x_query):
        """
        在固定长度历史池中找最近邻，用历史记录的预测值做加权平均。
        返回:
            y_hist: (7,)
            var_hist: (7,)
            alpha_hist: float  历史项建议融合权重
        """
        if len(self.local_gp_history) < self.gp_hist_min_points:
            return np.zeros(7, dtype=float), np.ones(7, dtype=float) * 1e6, 0.0

        # 取一个统一的标准化尺度，这里用 joint1 的 stats
        ref_pack = self.gp_models_small.get(1, None)
        if ref_pack is None:
            return np.zeros(7, dtype=float), np.ones(7, dtype=float) * 1e6, 0.0

        Xm, Xs, Ym, Ys = ref_pack["stats"]
        x_dim = ref_pack["x_dim"]

        Xm = np.asarray(Xm[:x_dim], dtype=np.float32)
        Xs = np.asarray(Xs[:x_dim], dtype=np.float32)
        Xs = np.where(np.abs(Xs) < 1e-8, 1.0, Xs)

        xq = ((x_query[:x_dim] - Xm) / Xs).astype(np.float32)

        dists = []
        ys = []
        vars_ = []

        for item in self.local_gp_history:
            xh = item["x"][:x_dim]
            yh = item["y"]
            vh = item["var"]

            dist = np.linalg.norm(xq - xh)
            dists.append(dist)
            ys.append(yh)
            vars_.append(vh)

        dists = np.asarray(dists, dtype=float)
        ys = np.asarray(ys, dtype=float)         # (N, 7)
        vars_ = np.asarray(vars_, dtype=float)   # (N, 7)

        k = min(self.gp_hist_topk, len(dists))
        idx = np.argpartition(dists, k - 1)[:k]

        d_k = dists[idx]
        y_k = ys[idx]
        v_k = vars_[idx]

        # ===== 用方差 PoE / precision weighting 替代距离加权 =====
        eps = 1e-8

        # precision: (k, 7)
        prec_k = 1.0 / np.maximum(v_k, eps)

        # 归一化后的每关节权重: (k, 7)
        w = prec_k / (np.sum(prec_k, axis=0, keepdims=True) + eps)

        # 历史融合预测：每个关节单独按 precision 加权
        y_hist = np.sum(y_k * w, axis=0)   # (7,)

        # PoE 合成后的历史方差：1 / sum(precision)
        var_hist = 1.0 / np.maximum(np.sum(prec_k, axis=0), eps)   # (7,)

        # 用历史融合后的整体置信度决定 alpha
        # 方差越小，alpha 越大
        conf_hist = 1.0 / (np.mean(var_hist) + eps)
        alpha_hist = self.gp_hist_alpha_max * (conf_hist / (conf_hist + 1.0))
        alpha_hist = float(np.clip(alpha_hist, 0.0, self.gp_hist_alpha_max))

        return y_hist, var_hist, alpha_hist
    
    def _append_local_gp_history(self, x_raw, y_hat, y_var):
        """
        把当前 local GP 查询点和预测结果塞进固定长度历史池
        """
        ref_pack = self.gp_models_small.get(1, None)
        if ref_pack is None:
            return

        Xm, Xs, Ym, Ys = ref_pack["stats"]
        x_dim = ref_pack["x_dim"]

        Xm = np.asarray(Xm[:x_dim], dtype=np.float32)
        Xs = np.asarray(Xs[:x_dim], dtype=np.float32)
        Xs = np.where(np.abs(Xs) < 1e-8, 1.0, Xs)

        x_std = ((x_raw[:x_dim] - Xm) / Xs).astype(np.float32)

        self.local_gp_history.append({
            "x": x_std.copy(),
            "y": np.asarray(y_hat, dtype=float).copy(),
            "var": np.asarray(y_var, dtype=float).copy(),
        })

    #friction compensation
    def friction_compensation(self, dq):
        dq = np.asarray(dq, dtype=float)
        return self.Fc * np.tanh(dq / self.v_eps) + self.Bv * dq

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

    def signal_handler(self, signum, frame):
        if self._signal_handled:
            return
        self._signal_handled = True

        self.get_logger().info(f"Received signal {signum}, saving data...")

        # 2. 保存数据
        try:
            self.save_data_to_file()
        except:
            pass

        # 3. 停止节点
        rclpy.shutdown()

        # 4. 直接退出程序
        os._exit(0)

    def _load_gp_models(self, dir_path="./new_structure/gp/gp_models"):
        """加载离线训练好的每关节GP，支持高维输入（14或21）""" 

        if not self._ensure_skygp_import():
            self.get_logger().error("[GP] skygp import failed; pickle loading will likely fail.")

        cwd = os.getcwd()
        abs_dir = os.path.abspath(dir_path)
        self.get_logger().info(f"[GP] 当前工作目录: {cwd}")
        self.get_logger().info(f"[GP] 模型目录绝对路径: {abs_dir}")
        
        # ===== 按关节定制 GP 参数（你可以自己改这些值） =====
        # key = 关节号（1..6），"default" 为所有关节的默认配置
        per_joint_cfg = {
            "default": dict(
                max_data_per_expert=50,
                nearest_k=1,
                max_experts=1,
                timescale=0.03,
            ),
            # 举例：如果你想让 6 号关节忘得快一点、专家少一点，可以单独改：
            6: dict(
                max_data_per_expert=50,
                nearest_k=1,
                max_experts=1,
                timescale=0.05,
            ),
        }

        self.gp_models_small = {}
        loaded = 0

        # ==== 必须改成加载 1..7 ====
        for j in range(1, 8):
            p = os.path.join(dir_path, f"joint{j}_local.pkl")
            abs_p = os.path.abspath(p)
            self.get_logger().info(f"[GP] 尝试加载模型: {abs_p}")

            if not os.path.isfile(p):
                self.get_logger().warn(f"[GP] model file not found: {abs_p}")
                continue

            try:
                with open(p, "rb") as f:
                    pack = pickle.load(f)

                model = pack["model"]
                stats = pack["stats"]   # (Xm, Xs, Ym, Ys)
                Xm, Xs, Ym, Ys = stats
                x_dim = int(len(Xm))    # 自动推断 14 或 21 维

                # ===== 在这里覆盖 SkyGP_rBCM 的参数 =====
                cfg = per_joint_cfg.get(j, per_joint_cfg["default"])
                try:
                    # 只有在模型里确实有这些属性时才改，避免旧版本崩溃
                    if hasattr(model, "max_data_per_expert"):
                        model.max_data_per_expert = int(cfg["max_data_per_expert"])
                    if hasattr(model, "nearest_k"):
                        model.nearest_k = int(cfg["nearest_k"])
                    if hasattr(model, "max_experts"):
                        model.max_experts = int(cfg["max_experts"])
                    if hasattr(model, "timescale"):
                        model.timescale = float(cfg["timescale"])

                    self.get_logger().info(
                        f"[GP] joint{j} loaded: x_dim={x_dim}, "
                        f"max_data_per_expert={getattr(model, 'max_data_per_expert', 'NA')}, "
                        f"nearest_k={getattr(model, 'nearest_k', 'NA')}, "
                        f"max_experts={getattr(model, 'max_experts', 'NA')}, "
                        f"timescale={getattr(model, 'timescale', 'NA')}"
                    )
                except Exception as e:
                    self.get_logger().warn(
                        f"[GP] joint{j}: override model params failed: {e}"
                    )

                self.gp_models_small[j] = {
                    "model": model,
                    "stats": stats,
                    "x_dim": x_dim
                }

                loaded += 1
                self.get_logger().info(f"[GP] 成功加载关节{j}模型, x_dim={x_dim}")

            except Exception as e:
                self.get_logger().error(f"[GP] fail loading {abs_p}: {e}")
        
        per_joint_cfg = {
            "default": dict(
                max_data_per_expert=25,
                nearest_k=4,
                max_experts=25,
                timescale=0.03,
            ),
            # 举例：如果你想让 6 号关节忘得快一点、专家少一点，可以单独改：
            6: dict(
                max_data_per_expert=25,
                nearest_k=4,
                max_experts=25,
                timescale=0.05,
            ),
        }

        self.gp_models_big = {}
        loaded = 0

        # ==== 必须改成加载 1..7 ====
        for j in range(1, 8):
            p = os.path.join(dir_path, f"joint{j}_local.pkl")
            abs_p = os.path.abspath(p)
            self.get_logger().info(f"[GP] 尝试加载模型: {abs_p}")

            if not os.path.isfile(p):
                self.get_logger().warn(f"[GP] model file not found: {abs_p}")
                continue

            try:
                with open(p, "rb") as f:
                    pack = pickle.load(f)

                model = pack["model"]
                stats = pack["stats"]   # (Xm, Xs, Ym, Ys)
                Xm, Xs, Ym, Ys = stats
                x_dim = int(len(Xm))    # 自动推断 14 或 21 维

                # ===== 在这里覆盖 SkyGP_rBCM 的参数 =====
                cfg = per_joint_cfg.get(j, per_joint_cfg["default"])
                try:
                    # 只有在模型里确实有这些属性时才改，避免旧版本崩溃
                    if hasattr(model, "max_data_per_expert"):
                        model.max_data_per_expert = int(cfg["max_data_per_expert"])
                    if hasattr(model, "nearest_k"):
                        model.nearest_k = int(cfg["nearest_k"])
                    if hasattr(model, "max_experts"):
                        model.max_experts = int(cfg["max_experts"])
                    if hasattr(model, "timescale"):
                        model.timescale = float(cfg["timescale"])

                    self.get_logger().info(
                        f"[GP] joint{j} loaded: x_dim={x_dim}, "
                        f"max_data_per_expert={getattr(model, 'max_data_per_expert', 'NA')}, "
                        f"nearest_k={getattr(model, 'nearest_k', 'NA')}, "
                        f"max_experts={getattr(model, 'max_experts', 'NA')}, "
                        f"timescale={getattr(model, 'timescale', 'NA')}"
                    )
                except Exception as e:
                    self.get_logger().warn(
                        f"[GP] joint{j}: override model params failed: {e}"
                    )

                self.gp_models_big[j] = {
                    "model": model,
                    "stats": stats,
                    "x_dim": x_dim
                }

                loaded += 1
                self.get_logger().info(f"[GP] 成功加载关节{j}模型, x_dim={x_dim}")

            except Exception as e:
                self.get_logger().error(f"[GP] fail loading {abs_p}: {e}")

        self.gp_ready = (loaded > 0)
        self.get_logger().info(f"[GP] 共加载 {loaded} 个模型，ready={self.gp_ready}")

    def _lowpass_vector(self, x_raw, x_prev, dt, cutoff_hz):
        """
        一阶低通:
            y_k = alpha * x_k + (1-alpha) * y_{k-1}
        alpha = dt / (tau + dt), tau = 1 / (2*pi*f_c)
        """
        if cutoff_hz <= 0.0:
            return x_raw.copy()

        tau = 1.0 / (2.0 * np.pi * cutoff_hz)
        alpha = dt / (tau + dt)
        alpha = np.clip(alpha, 0.0, 1.0)

        return alpha * x_raw + (1.0 - alpha) * x_prev

    def _gp_predict_and_update(self, q, dq_des_joint, ddq_des_joint, tau_residual, models, update = True):
        """
        本地 GP：高维输入版本（14维 or 21维）
        每个关节都使用相同的 x_full = concat([q, dq, ddq])
        """

        if not self.gp_ready or not self.use_gp:
            return np.zeros(7, dtype=float)

        y_hat = np.zeros(7, dtype=float)
        y_var = np.ones(7, dtype=float) * 1e6  # 默认给大方差，表示“不可信”
        # ==================================================
        # 1) 构造统一的高维输入 x_full
        # ==================================================
        # 如果你的训练是 q + dq + ddq → 21 维
        # x_full = np.concatenate([q, dq_des_joint, ddq_des_joint]).astype(np.float32)

        # 如果你训练使用的是 q + dq → 14 维，请改成：
        x_full = np.concatenate([q, dq_des_joint]).astype(np.float32)

        # ==================================================
        # 2) 每个关节都用同一 x_full 预测
        # ==================================================
        for j in range(1, 7):

            pack = models.get(j)
            if pack is None:
                continue

            model = pack["model"]
            Xm, Xs, Ym, Ys = pack["stats"]
            x_dim = pack["x_dim"]  # 训练时的真实维度
            Xm = np.asarray(Xm, dtype=np.float32)
            Xs = np.asarray(Xs, dtype=np.float32)
            Ym = float(Ym[0])
            Ys = float(Ys[0]) if float(Ys[0]) != 0.0 else 1.0

            # -------- 标准化整个向量 --------
            x_std = (x_full[:x_dim] - Xm[:x_dim]) / Xs[:x_dim]

            # -------- GP 预测 --------
            mu_std, var_std = model.predict(x_std.astype(np.float32))
            mu_std  = float(mu_std[0])
            var_std = float(var_std[0])

            y_pred = mu_std * Ys + Ym
            y_hat[j-1] = y_pred

            # 反标准化方差
            y_var[j-1] = max(var_std * (Ys**2), 1e-8)  # 防止为 0

            # -------- 在线更新 --------
            y_real = float(tau_residual[j - 1])
            y_std = (y_real - Ym) / Ys

            if np.isfinite(y_std):
                try:
                    if update:
                        model.add_point(
                            x_std.astype(np.float32),
                            np.array([y_std], dtype=np.float32)
                        )
                        self.offline_limit=self.offline_limit+1
                except Exception as e:
                    self.get_logger().error(f"[GP] joint{j} add_point failed: {e}")

        return y_hat, y_var


    def _ensure_skygp_import(self):
        """
        确保在当前进程中有名为 'skygp' 的模块，
        路径指向 repo 里的 /new_structure/gp/skygp.py
        """
        # 以当前脚本为基准，找到 gp/skygp.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        skygp_path = os.path.abspath(os.path.join(
            script_dir, "..","..", "..","..","..", "..", "new_structure","gp", "skygp.py"
        ))

        if not os.path.isfile(skygp_path):
            self.get_logger().error(f"[GP] skygp.py not found at: {skygp_path}")
            return False

        # 如果已加载则跳过
        if "skygp" in sys.modules:
            return True

        try:
            spec = importlib.util.spec_from_file_location("skygp", skygp_path)
            skygp_mod = importlib.util.module_from_spec(spec)
            sys.modules["skygp"] = skygp_mod   # 关键：模块名必须叫 'skygp'
            spec.loader.exec_module(skygp_mod)
            self.get_logger().info(f"[GP] skygp loaded from: {skygp_path}")
            return True
        except Exception as e:
            self.get_logger().error(f"[GP] failed to import skygp from {skygp_path}: {e}")
            return False
    
    def save_data_to_file(self):
        """save data to CSV file"""
        if not self.tau_history:
            self.get_logger().warning('No data to save - tau_history is empty')
            return

        try:
            filename = 'cartesian_impedance_controller_data.csv'

            # 计算可用的最小长度，避免某些列表短导致越界
            series_list = [
                self.time_history,
                self.tau_history,
                self.x_history,
                self.x_des_history,
                self.dx_history,
                self.dx_des_history,
                self.tau_measured_history,
                self.gravity_history,
                self.q_history,
                self.dq_history,
                self.dq_des_joint_history,
                self.ddq_des_joint_history,
                self.y_hat_history,            # <--- 新增
                self.tau_residual_history,     # <--- 新增
                self.tau_residual_raw_history,
            ]

            min_len = min(len(s) for s in series_list)

            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                header = ['Time(s)']
                n_j = len(self.tau_history[0])  # 通常是7
                header.extend([f'tau_{i+1}' for i in range(n_j)])
                header.extend(['x_actual', 'y_actual', 'z_actual'])
                header.extend(['x_desired', 'y_desired', 'z_desired'])
                header.extend(['dx_actual', 'dy_actual', 'dz_actual'])
                header.extend(['dx_desired', 'dy_desired', 'dz_desired'])
                header.extend([f'tau_measured_{i+1}' for i in range(n_j)])
                header.extend([f'gravity_{i+1}' for i in range(n_j)])
                header.extend([f'joint_pos_{i+1}' for i in range(7)])
                header.extend([f'joint_vel_{i+1}' for i in range(7)])
                header.extend([f'dq_des_joint_{i+1}' for i in range(7)])
                header.extend([f'ddq_des_joint_{i+1}' for i in range(7)])
                header.extend([f'y_hat_{i+1}' for i in range(7)])          # combined
                header.extend([f'y_hat_local_{i+1}' for i in range(7)])    # local
                header.extend([f'y_hat_cloud_{i+1}' for i in range(7)])    # cloud
                header.extend([f'y_hat_mem_{i+1}' for i in range(7)])    # cloud
                header.extend([f'tau_residual_{i+1}' for i in range(7)])
                header.extend([f'tau_residual_raw_{i+1}' for i in range(7)])
                writer.writerow(header)

                for i in range(min_len):
                    row = [self.time_history[i]]
                    row.extend(self.tau_history[i])
                    row.extend(self.x_history[i][:3])
                    row.extend(self.x_des_history[i][:3])
                    row.extend(self.dx_history[i])           # 已是3维
                    row.extend(self.dx_des_history[i])       # 已是3维
                    row.extend(self.tau_measured_history[i])
                    row.extend(self.gravity_history[i])
                    row.extend(self.q_history[i])
                    row.extend(self.dq_history[i])

                    # 防御性处理：如果某次未记录到 dq_des/ddq_des，填零
                    if i < len(self.dq_des_joint_history):
                        row.extend(self.dq_des_joint_history[i])
                    else:
                        row.extend([0.0]*7)

                    if i < len(self.ddq_des_joint_history):
                        row.extend(self.ddq_des_joint_history[i])
                    else:
                        row.extend([0.0]*7)
                    
                    # y_hat & tau_residual 新增（也做缺省保护）
                    if i < len(self.y_hat_history):
                        row.extend(self.y_hat_history[i])
                    else:
                        row.extend([0.0]*7)

                    if i < len(self.y_hat_local_history):
                        row.extend(self.y_hat_local_history[i])
                    else:
                        row.extend([0.0]*7)

                    if i < len(self.y_hat_cloud_history):
                        row.extend(self.y_hat_cloud_history[i])
                    else:
                        row.extend([0.0]*7)

                    if i < len(self.y_hat_mem_history):
                        row.extend(self.y_hat_mem_history[i])
                    else:
                        row.extend([0.0]*7)
                    
                    if i < len(self.tau_residual_history):
                        row.extend(self.tau_residual_history[i])
                    else:
                        row.extend([0.0]*7)
                    
                    if i < len(self.tau_residual_raw_history):
                        row.extend(self.tau_residual_raw_history[i])
                    else:
                        row.extend([0.0]*7)
                    writer.writerow(row)

            self.get_logger().info(f'Successfully saved {min_len} data points to {filename}')

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