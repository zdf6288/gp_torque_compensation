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

        # === local GP ===
        self.gp_models = {}
        self.gp_ready = False    # 标记本地 GP 是否加载成功
        self.y_hat_local = np.zeros(7)
        self.y_hat_local_history = []
        self.offline_limit = 0

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
        self.future_delay = self.declare_parameter(
            'future_delay', 0 # 默认 60 ms
        ).value

        # 存最新一次未来轨迹
        self._latest_future_traj = None   # dict: {"x_des": np.array(6,), "dx_des": ..., "ddx_des": ...}
        self._future_traj_counter = 0
        self._future_traj_warned = False

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
            q = np.array(msg.position)
            self.q = q
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
                        # start transition, clear ros2_control interface buffer
                        tau = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        self.effort_msg.efforts = tau.tolist()
                        self.effort_publisher.publish(self.effort_msg)
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
            gravity_measured = np.array(msg.gravity)
            tau_measured = np.array(msg.effort_measured)

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

            # === 映射任务空间期望到关节空间的期望速度/加速度 ===
            # 任务空间只用前5维（你的控制是5 DoF：3平移+2姿态约束）
            dx_des_5  = self.dx_des[:5]
            ddx_des_5 = self.ddx_des[:5]

            # dq_des (joint) = J^+ * dx_des
            dq_des_joint = jacobian_pinv @ dx_des_5
            self.dq_des_joint = dq_des_joint
            # ddq_des (joint) = J^+ * (ddx_des - dJ * dq)
            # 数值防护：dt很小时 dJ 可能抖；已按你代码用差分得到 djacobian
            ddq_des_joint = jacobian_pinv @ (ddx_des_5 - djacobian @ dq)
            self.ddq_des_joint = ddq_des_joint

            # 记录（仅当开启录数时）
            if self.data_recording_enabled:
                self.dq_des_joint_history.append(dq_des_joint.tolist())
                self.ddq_des_joint_history.append(ddq_des_joint.tolist())
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

            # === 计算残差 ===
            tau_residual = tau_measured - tau - gravity_measured
            self.tau_residual_filtered = (
                0.05 * tau_residual + 0.95 * self.tau_residual_filtered
            )
            # print("tau_residual:", tau_residual)

            # === 控制：先用“上一帧”融合好的 y_hat_combined 做补偿 ===
            if self.gp_active:
                tau = tau - self.y_hat_combined

            # --- 记录（含最终补偿用的 y_hat_combined）---
            if self.data_recording_enabled:
                self.y_hat_history.append(self.y_hat_combined.tolist())      # combined
                self.y_hat_local_history.append(self.y_hat_local.tolist())    # 上一帧或刚更新的 local
                self.y_hat_cloud_history.append(self.y_hat_cloud.tolist())    # 上一帧或刚更新的 cloud
                self.y_hat_mem_history.append(self.y_hat_mem.tolist())
                self.tau_residual_history.append(self.tau_residual_filtered.tolist())


            # === 异步请求未来轨迹（给 GP 用），比如 100Hz 请求一次 ===
            if self.future_traj_client.service_is_ready():
                self._future_traj_counter += 1
                if self._future_traj_counter % 10 == 0:
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
            # print("active:",self.gp_active)
            # print("use:",self.use_gp)
            if self.gp_active and self.use_gp:
                self.gp_counter += 1
                print("gp counter:",self.gp_counter)
                if self.gp_counter % self.gp_stride == 0:
                    # 1) 本地 GP 立刻算一次（同步）
                    self.y_hat_local = self._gp_predict_and_update(
                        self.q,
                        dq,
                        self.ddq_des_joint,
                        self.tau_residual_filtered,
                    )   
                    # print(self.gp_counter)
                # 2) 云端 GP：发异步请求（同一时刻的 q/dq/ddq/残差）
                # === CLOUD GP 调用 ===
                if self.gp_client.service_is_ready():
                    self.gp_counter = 0
                    req = AsyncGPpredict.Request()

                    # ---- 当前输入 ----
                    req.q = self.q.astype(np.float32).tolist()
                    req.dq_des_joint = dq.astype(np.float32).tolist()
                    req.ddq_des_joint = self.ddq_des_joint.astype(np.float32).tolist()
                    req.tau_residual = self.tau_residual_filtered.astype(np.float32).tolist()

                    # ============================================================
                    #   未来输入（不依赖 future_traj，不依赖任务空间）
                    #   永远可计算 —— 保证 cloud GP 每次都能收到预测输入
                    # ============================================================

                    # delay = float(self.future_delay)

                    # q_now  = self.q
                    # dq_now = dq
                    # ddq_now = self.ddq_des_joint

                    # # ---- 常加速度未来预测 ----
                    # q_future  = q_now + dq_now * delay + 0.5 * ddq_now * delay**2
                    # dq_future = dq_now + ddq_now * delay
                    # ============================================================
                    #   CLOUD GP：使用异步请求得到的未来任务空间轨迹
                    # ============================================================

                    if self._latest_future_traj is not None:

                        # 1) 取未来任务空间期望
                        dx_f  = self._latest_future_traj["dx_des"]
                        ddx_f = self._latest_future_traj["ddx_des"]

                        # 只用前 5 维（你的控制任务）
                        dx_f_5  = dx_f[:5]
                        ddx_f_5 = ddx_f[:5]

                        # 2) 用当前 Jacobian 映射到 joint space
                        dq_future  = jacobian_pinv @ dx_f_5
                        ddq_future = jacobian_pinv @ (ddx_f_5 - djacobian @ dq)

                    else:
                        # ❗ fallback：没有 future traj 时，直接置零 or 用当前 dq
                        dq_future  = dq.copy()
                        ddq_future = np.zeros_like(dq)


                    # req = AsyncGPpredict.Request()

                    # ---- 当前状态 ----
                    # req.q = self.q.astype(np.float32).tolist()
                    # req.dq_des_joint = dq.astype(np.float32).tolist()
                    # req.ddq_des_joint = self.ddq_des_joint.astype(np.float32).tolist()
                    # req.tau_residual = self.tau_residual_filtered.astype(np.float32).tolist()

                    # ---- 未来状态（来自轨迹）----
                    req.dq_des_joint_future  = dq_future.astype(np.float32).tolist()
                    req.ddq_des_joint_future = ddq_future.astype(np.float32).tolist()

                    # ---- 不使用采样 ----
                    req.n_future_samples = 0
                    req.dq_future_samples_flat = []
                    req.ddq_future_samples_flat = []

                    future = self.gp_client.call_async(req)
                    future.add_done_callback(self._gp_response_callback)


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

    def _gp_response_callback(self, future):
        try:
            resp = future.result()
            if resp is None:
                return

            y_cloud = np.array(resp.y_cloud, dtype=float)
            y_mem = np.array(resp.y_mem,dtype=float)

            with self._gp_lock:
                self.y_hat_cloud = y_cloud
                self.y_hat_mem = y_mem

            # ---------- 融合模式选择 ----------
            mode = self.gp_mode
            mode = "local"

            if mode == "local":
                # local-only
                self.y_hat_combined = self.y_hat_local

            elif mode == "cloud":
                # cloud-only
                self.y_hat_combined = self.y_hat_cloud

            elif mode == "fusion":
                # 0.5/0.5 融合
                self.y_hat_combined = 0 * self.y_hat_local + 1 * self.y_hat_cloud

            elif mode == "history_fusion":
                # 需要你先添加一个历史平滑：下面我会给你
                y_hist = self._push_and_smooth_history(self.y_hat_local)
                self.y_hat_combined = 0.5 * y_hist + 0.5 * self.y_hat_cloud

            elif mode == "none":
                # 不用 GP
                self.y_hat_combined = np.zeros(7)

            else:
                self.get_logger().warn(f"[GP] Unknown mode {mode}, fallback to fusion")
                self.y_hat_combined = 0.5 * self.y_hat_local + 0.5 * self.y_hat_cloud

        except Exception as e:
            self.get_logger().warn(f"[Controller] GP response error: {e}")


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

        self.gp_models = {}
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

                self.gp_models[j] = {
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

    def _gp_predict_and_update(self, q, dq_des_joint, ddq_des_joint, tau_residual):
        """
        本地 GP：高维输入版本（14维 or 21维）
        每个关节都使用相同的 x_full = concat([q, dq, ddq])
        """

        if not self.gp_ready or not self.use_gp:
            return np.zeros(7, dtype=float)

        y_hat = np.zeros(7, dtype=float)

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

            pack = self.gp_models.get(j)
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
            mu_std = float(mu_std[0])

            # -------- 反标准化 --------
            y_pred = mu_std * Ys + Ym

            y_hat[j - 1] = y_pred

            # -------- 在线更新 --------
            y_real = float(tau_residual[j - 1])
            y_std = (y_real - Ym) / Ys

            if np.isfinite(y_std):
                try:
                    model.add_point(
                        x_std.astype(np.float32),
                        np.array([y_std], dtype=np.float32)
                    )
                    self.offline_limit=self.offline_limit+1
                except Exception as e:
                    self.get_logger().error(f"[GP] joint{j} add_point failed: {e}")

        return y_hat


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