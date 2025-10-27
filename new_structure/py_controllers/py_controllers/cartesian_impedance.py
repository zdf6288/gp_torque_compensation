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
import os, pickle
import importlib.util
import threading, time

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

        # === Asychronous GP ===
        self._gp_lock = threading.Lock()
        self._gp_thread = None
        self._gp_stop = False
        self._gp_yhat = np.zeros(7, dtype=float)
        self._gp_thread_period = 0.002  # 后台线程循环周期 (s) → 2ms ≈ 500Hz

        self.gp_models = {}     # j: {'model':..., 'stats':(Xm,Xs,Ym,Ys), 'x_dim':D}
        self.gp_ready = False
        self.gp_use_features = ("q", "dq_des", "ddq_des")
        self._load_gp_models(dir_path="./new_structure/gp/gp_models")
        self._start_gp_thread()

        # 预测节流（你已有）
        self.use_gp = True
        self.gp_stride = 10
        self._step = 0
        self.y_hat_last = np.zeros(7, dtype=float)
        self.y_hat_max_age = 0.2
        self._y_hat_stamp = None

        # === 在线学习配置 ===
        self.gp_update_y_clip = 10.0         # 训练用残差的幅值上限（Nm）

        self.y_hat_history = []          # 记录每次控制回路使用的 y_hat (7,)
        self.tau_residual_history = []   # 记录 tau_residual (7,)



    def taskCommandCallback(self, msg):
        """callback function for /task_space_command subscriber"""
        self.task_command_received = True
        self.x_des = np.array(msg.x_des)
        self.dx_des = np.array(msg.dx_des)
        self.ddx_des = np.array(msg.ddx_des)
        
    def dataRecordingCallback(self, msg):
        """callback function for /data_recording_enabled subscriber"""
        self.data_recording_enabled = msg.data
        
    def stateParameterCallback(self, msg):
        """callback function for /state_parameter subscriber"""
        try:
            # initialize t_initial, get t_elapsed, t_last and dt
            # initialize q_initial, get q, dq and ddq
            t_now = self.get_clock().now()
            self._step += 1
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

            tau = self.filter_beta * tau + (1 - self.filter_beta) * self.tau_buffer
            self.tau_buffer = tau.copy()
            tau = np.clip(tau, -50.0, 50.0)

            tau_residual = tau_measured - tau - gravity_measured
            self.tau_residual = tau_residual
            print("tau_residual:",tau_residual)

            # === 主控制回路 ===
            with self._gp_lock:
                if not self._gp_stop:
                    y_hat = np.copy(self._gp_yhat)
                    print("yhat:",y_hat)
            tau = tau + y_hat

            # --- 记录 y_hat / tau_residual（与其他数据一起） ---
            if self.data_recording_enabled:
                self.y_hat_history.append(y_hat.tolist())
                self.tau_residual_history.append(tau_residual.tolist())

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
        """signal handler, call save data function when program is interrupted"""
        try:
            if self._signal_handled:
                return
            self._signal_handled = True
            self.get_logger().info(f'Received signal {signum}, saving data...')
            self.save_data_to_file()
            self.get_logger().info(f'Signal handler completed successfully')
        except Exception as e:
            self.get_logger().error(f'Error in signal handler: {str(e)}')
            self._signal_handled = False
    
    def _load_gp_models(self, dir_path="./new_structure/gp/gp_models"):
        """加载离线训练好的每关节GP：gp_models/joint{j}.pkl"""
        # 打印当前工作目录和 dir_path 的绝对路径，方便调试
        if not self._ensure_skygp_import():
            self.get_logger().error("[GP] skygp import failed; pickle loading will likely fail.")

        cwd = os.getcwd()
        abs_dir = os.path.abspath(dir_path)
        self.get_logger().info(f"[GP] 当前工作目录: {cwd}")
        self.get_logger().info(f"[GP] 模型目录绝对路径: {abs_dir}")

        loaded = 0
        for j in range(1, 7):
            p = os.path.join(dir_path, f"joint{j}.pkl")
            abs_p = os.path.abspath(p)
            self.get_logger().info(f"[GP] 尝试加载模型: {abs_p}")

            if not os.path.isfile(p):
                self.get_logger().warn(f"[GP] model file not found: {abs_p}")
                continue

            try:
                with open(p, "rb") as f:
                    pack = pickle.load(f)
                model = pack.get("model", None)
                stats = pack.get("stats", None)  # (Xm, Xs, Ym, Ys)
                if model is None or stats is None:
                    self.get_logger().warn(f"[GP] bad model pack: {abs_p}")
                    continue

                Xm, Xs, _, _ = stats
                x_dim = int(np.asarray(Xm).shape[0])
                self.gp_models[j] = {"model": model, "stats": stats, "x_dim": x_dim}
                loaded += 1
                self.get_logger().info(f"[GP] 成功加载关节{j}模型 ({x_dim}维输入) 来自: {abs_p}")
            except Exception as e:
                self.get_logger().error(f"[GP] fail loading {abs_p}: {e}")

        self.gp_ready = (loaded > 0)
        self.get_logger().info(f"[GP] 共加载 {loaded} 个模型. ready={self.gp_ready}")

    def _start_gp_thread(self):
        """启动 GP 异步预测+更新线程"""
        if self._gp_thread is not None:
            return  # 已经启动过
        self._gp_stop = False

        def gp_loop():
            self.get_logger().info("[GP] background thread started.")
            while not self._gp_stop and rclpy.ok():
                t0 = time.time()
                try:
                    with self._gp_lock:
                        # 复制当前状态（防止线程间冲突）
                        q = np.copy(self.q)
                        dq_des_joint = np.copy(self.dq_des_joint)
                        ddq_des_joint = np.copy(self.ddq_des_joint)
                        tau_residual = np.copy(self.tau_residual)
                    # === 执行预测 + 在线更新 ===
                    y_hat = self._gp_predict_and_update(q, dq_des_joint, ddq_des_joint, tau_residual)
                    # 存储预测结果
                    with self._gp_lock:
                        self._gp_yhat = y_hat
                except Exception as e:
                    self.get_logger().error(f"[GP] thread error: {e}")
                # 控制频率
                dt = time.time() - t0
                sleep_t = max(0.0, self._gp_thread_period - dt)
                time.sleep(sleep_t)
            self.get_logger().info("[GP] background thread stopped.")

        self._gp_thread = threading.Thread(target=gp_loop, daemon=True)
        self._gp_thread.start()
    
    def _stop_gp_thread(self):
        self._gp_stop = True
        if self._gp_thread is not None:
            self._gp_thread.join(timeout=1.0)
            self._gp_thread = None

    def _gp_predict_and_update(self, q, dq_des_joint, ddq_des_joint, tau_residual):
        """后台线程中调用：预测 + 在线更新（每次循环一次，带详细debug输出）"""
        if self._gp_stop:
            return
        if not self.gp_ready or not self.use_gp:
            return np.zeros(7, dtype=float)

        y_hat = np.zeros(7, dtype=float)

        for j in range(1, 7):
            pack = self.gp_models.get(j)
            if pack is None:
                self.get_logger().warn(f"[GP-debug] joint {j}: no model pack")
                continue

            model = pack["model"]
            Xm, Xs, Ym, Ys = pack["stats"]
            x_dim = pack["x_dim"]

            # === 1. 构造输入 ===
            if x_dim == 3:
                x = np.array([q[j-1], dq_des_joint[j-1], ddq_des_joint[j-1]], dtype=np.float32)
            elif x_dim == 2:
                x = np.array([q[j-1], ddq_des_joint[j-1]], dtype=np.float32)
            else:
                x = np.array([q[j-1]], dtype=np.float32)

            if not np.all(np.isfinite(x)):
                self.get_logger().warn(f"[GP-debug] joint {j}: invalid x = {x}")
                continue

            Xm = np.asarray(Xm, dtype=np.float32)
            Xs = np.asarray(Xs, dtype=np.float32)
            Ym = float(np.asarray(Ym)[0])
            Ys = float(np.asarray(Ys)[0]) if float(np.asarray(Ys)[0]) != 0.0 else 1.0
            # Xs[Xs < 1e-9] = 1.0

            # 标准化
            x_std = (x - Xm[:x_dim]) / Xs[:x_dim]
            # x_std = np.clip(x_std, -5.0, 5.0)

            # === 2. 预测 ===
            try:
                mu_std, _ = model.predict(x_std)
                mu_std = float(mu_std[0])
            except Exception as e:
                self.get_logger().error(f"[GP-debug] joint {j}: predict failed: {e}")
                continue

            if not np.isfinite(mu_std):
                self.get_logger().warn(f"[GP-debug] joint {j}: mu_std not finite ({mu_std}) → set 0")
                mu_std = 0.0

            y_pred = mu_std * Ys + Ym
            # y_pred_clipped = np.clip(y_pred, -5.0, 5.0)
            y_hat[j-1] = y_pred

            # # === 打印预测调试信息 ===
            # self.get_logger().info(
            #     f"[GP-debug] joint {j}: "
            #     f"x={x}, x_std={np.round(x_std,3)}, mu_std={mu_std:.4f}, "
            #     f"y_pred_raw={y_pred:.4f}, y_pred_clip={y_pred_clipped:.4f}, "
            #     f"Ys={Ys:.4f}, Ym={Ym:.4f}"
            # )

            # === 3. 在线更新 ===
            y_real = float(tau_residual[j-1])
            if not np.isfinite(y_real):
                self.get_logger().warn(f"[GP-debug] joint {j}: invalid y_real={y_real}")
                continue
            y_std = (y_real - Ym) / Ys

            if np.isfinite(y_std):
                try:
                    model.add_point(x_std, np.array([y_std], dtype=np.float32))
                    self.get_logger().info(
                        f"[GP-debug] joint {j}: update ok (y_real={y_real:.4f}, y_std={y_std:.4f})"
                    )
                except Exception as e:
                    self.get_logger().error(f"[GP-debug] joint {j}: add_point failed: {e}")
            else:
                self.get_logger().warn(f"[GP-debug] joint {j}: skip update (non-finite y_std={y_std})")

        # # 限幅
        # norm6 = np.linalg.norm(y_hat[:6])
        # if norm6 > 12.0:
        #     y_hat[:6] *= (12.0 / (norm6 + 1e-9))
        #     self.get_logger().warn(f"[GP-debug] y_hat norm={norm6:.3f} → clipped")

        # self.get_logger().info(f"[GP-debug] final y_hat={np.round(y_hat,4)}")

        return y_hat



    def _ensure_skygp_import(self):
        """
        确保在当前进程中有名为 'skygp' 的模块，
        路径指向 repo 里的 /new_structure/gp/skygp.py
        """
        # 以当前脚本为基准，找到 gp/skygp.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        skygp_path = os.path.abspath(os.path.join(
            script_dir, "..","..", "..", "new_structure","gp", "skygp.py"
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
                header.extend([f'y_hat_{i+1}' for i in range(7)])           # <--- 新增
                header.extend([f'tau_residual_{i+1}' for i in range(7)])    # <--- 新增
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
        cartesian_impedance_node._stop_gp_thread
        cartesian_impedance_node.get_logger().info('Received keyboard interrupt, saving data...')
    except Exception as e:
        cartesian_impedance_node.get_logger().error(f'Error when running program: {str(e)}')
    finally:
        try:
            # save data to file only if signal handler has not been executed
            if not cartesian_impedance_node._signal_handled:
                cartesian_impedance_node.get_logger().info('Signal handler not executed, saving data to file...')
                cartesian_impedance_node._stop_gp_thread
                cartesian_impedance_node.save_data_to_file()
            else:
                cartesian_impedance_node.get_logger().info('Signal handler executed, data already saved, skipping...')
                
        except Exception as e:
            cartesian_impedance_node.get_logger().error(f'Error when saving data: {str(e)}')
        cartesian_impedance_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 