#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_msgs.msg import StateParameter, EffortCommand, TaskSpaceCommand, JointSpaceCommand
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

        self.declare_parameter('reference_mode', 'cartesian')
        self.declare_parameter('joint_space_command_topic', '/joint_space_command')
        self.reference_mode = str(self.get_parameter('reference_mode').value).strip().lower()
        self.joint_space_command_topic = str(
            self.get_parameter('joint_space_command_topic').value
        ).strip()
        if self.reference_mode not in ('cartesian', 'joint'):
            self.get_logger().warn(
                f"Invalid reference_mode='{self.reference_mode}', falling back to 'cartesian'"
            )
            self.reference_mode = 'cartesian'
        if not self.joint_space_command_topic:
            self.get_logger().warn(
                "joint_space_command_topic is empty, falling back to /joint_space_command"
            )
            self.joint_space_command_topic = '/joint_space_command'
        
        # subscribe to /state_parameter
        self.param_subscription = self.create_subscription(
            StateParameter, '/state_parameter', self.stateParameterCallback, 10)
        
        # subscribe to /task_space_command
        self.task_command_subscription = self.create_subscription(
            TaskSpaceCommand, '/task_space_command', self.taskCommandCallback, 10)

        self.joint_space_command_subscription = None
        if self.reference_mode == 'joint':
            self.joint_space_command_subscription = self.create_subscription(
                JointSpaceCommand,
                self.joint_space_command_topic,
                self.jointSpaceCommandCallback,
                10
            )
        
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
        
        self.declare_parameter('k_pd', [20.0, 20.0, 20.0, 20.0, 5.0, 3.0, 2.0])    # k_gains in PD control (joint space)
        self.declare_parameter('d_pd', [16.0, 16.0, 16.0, 16.0, 5.0, 3.0, 2.0])    # d_gains in PD control (joint space)
        self.declare_parameter('i_pid', [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])        # i_gains supplement to PD control (joint space)
        self.k_pd = np.array(self.get_parameter('k_pd').value, dtype=float)
        self.d_pd = np.array(self.get_parameter('d_pd').value, dtype=float)
        self.i_pid = np.array(self.get_parameter('i_pid').value, dtype=float)
        self.i_error = np.zeros(7)

        # GOAL1 joint reference branch: explicit-only, low-gain, clipped, and rate-limited.
        self.declare_parameter('joint_reference_kp', [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        self.declare_parameter('joint_reference_kd', [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.declare_parameter('joint_reference_torque_clip_nm', 0.5)
        self.declare_parameter('joint_reference_torque_rate_limit_nm_per_s', 5.0)
        self.declare_parameter('joint_space_command_timeout_sec', 0.1)
        self.joint_reference_kp = self._get_7d_parameter(
            'joint_reference_kp', [2.0] * 7
        )
        self.joint_reference_kd = self._get_7d_parameter(
            'joint_reference_kd', [0.2] * 7
        )
        self.joint_reference_torque_clip_nm = self._get_positive_float_parameter(
            'joint_reference_torque_clip_nm', 0.5
        )
        self.joint_reference_torque_rate_limit_nm_per_s = self._get_positive_float_parameter(
            'joint_reference_torque_rate_limit_nm_per_s', 5.0
        )
        self.joint_space_command_timeout_sec = self._get_positive_float_parameter(
            'joint_space_command_timeout_sec', 0.1
        )
        self.joint_reference_last_tau = np.zeros(7, dtype=float)
        self.joint_reference_last_tau_time = None

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

        self.startup_interp_started = False
        self.startup_interp_start_time = None
        self.startup_interp_duration = 5.0   # 可调
        self.startup_x0 = None
        self.startup_rot0 = None

        self.declare_parameter('startup_linear_speed', 0.01)   # m/s
        self.startup_linear_speed = float(self.get_parameter('startup_linear_speed').value)


        self.declare_parameter('start_x', 0.35)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('start_z', 0.65)

        self.start_x = float(self.get_parameter('start_x').value)
        self.start_y = float(self.get_parameter('start_y').value)
        self.start_z = float(self.get_parameter('start_z').value)

        self.x_start_des = np.array([self.start_x, self.start_y, self.start_z], dtype=float)

        self.declare_parameter('startup_kp_task', [500.0, 500.0, 500.0, 10.0, 10.0, 1.0])
        self.declare_parameter('startup_kd_task', [50.0, 50.0, 50.0, 1.0, 1.0, 1.0])
        self.declare_parameter('startup_ki_task', [1.0, 1.0, 1.0, 1.0, 1.0, 0.0])

        self.startup_kp_task = np.array(self.get_parameter('startup_kp_task').value, dtype=float)
        self.startup_kd_task = np.array(self.get_parameter('startup_kd_task').value, dtype=float)
        self.startup_ki_task = np.array(self.get_parameter('startup_ki_task').value, dtype=float)

        self.startup_x_int_error = np.zeros(6, dtype=float)
        self.declare_parameter('startup_pos_threshold', 0.02)   # 1 cm
        self.startup_pos_threshold = float(self.get_parameter('startup_pos_threshold').value)
                
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

        self.dq_future_filt = np.zeros(7)
        self.dq_future_filt_initialized = False

        self.declare_parameter("dq_future_lpf_hz", 30.0)
        self.dq_future_lpf_hz = float(self.get_parameter("dq_future_lpf_hz").value)

        # ===== joint rollout for future prediction =====
        self.dq_prev = np.zeros(7, dtype=float)
        self.ddq_est = np.zeros(7, dtype=float)
        self.ddq_est_initialized = False

        self.declare_parameter("ddq_lpf_hz", 15.0)
        self.ddq_lpf_hz = float(self.get_parameter("ddq_lpf_hz").value)

        self.declare_parameter("delay_steps", 1)
        self.delay_steps = int(self.get_parameter("delay_steps").value)

        self.declare_parameter("cloud_rollout_n", 7)          # 采样点数
        self.declare_parameter("cloud_rollout_span", 0.001)    # 在 Td 附近 ±span/2 采样，单位秒

        self.cloud_rollout_n = int(self.get_parameter("cloud_rollout_n").value)
        self.cloud_rollout_span = float(self.get_parameter("cloud_rollout_span").value)

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
        self.joint_command_received = False
        self.joint_command_enabled = False
        self.joint_command_time = None
        self.q_des_joint = np.zeros(7, dtype=float)
        self._joint_reference_wait_logged = False
        self._joint_reference_stale_logged = False
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
        self.get_logger().info(
            f"Reference mode: {self.reference_mode}, "
            f"joint_space_command_topic='{self.joint_space_command_topic}'"
        )
        if self.reference_mode == 'joint':
            self.get_logger().warn(
                "Joint reference mode is explicit-only: waiting for enabled "
                "JointSpaceCommand before publishing joint-reference torque."
            )
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
        self.tau_nominal_history = []
        self.tau_final_history = []
        self.gp_source_code_history = []
        self.gp_selected_raw_history = []
        self.gp_scaled_history = []
        self.gp_applied_history = []
        self.gp_clip_active_history = []
        self.gp_shadow_historical_available_history = []
        self.gp_shadow_local_raw_history = []
        self.gp_shadow_cloud_raw_history = []
        self.gp_shadow_hist_raw_history = []
        self.gp_shadow_combined_paper_raw_history = []
        self.gp_shadow_var_local_history = []
        self.gp_shadow_var_cloud_history = []
        self.gp_shadow_var_hist_history = []
        self.gp_shadow_weight_local_history = []
        self.gp_shadow_weight_cloud_history = []
        self.gp_shadow_weight_hist_history = []
        self.gp_shadow_precision_local_history = []
        self.gp_shadow_precision_cloud_history = []
        self.gp_shadow_precision_hist_history = []
        self.gp_shadow_paper_scaled_history = []
        self.gp_shadow_paper_clip_proxy_applied_history = []
        self.gp_shadow_paper_clip_proxy_active_history = []
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
        self._tau_nominal = np.zeros(7, dtype=float)
        self._tau_final = np.zeros(7, dtype=float)
        self._gp_source_code = 0
        self._gp_selected_raw = np.zeros(7, dtype=float)
        self._gp_scaled = np.zeros(7, dtype=float)
        self._gp_applied = np.zeros(7, dtype=float)
        self._gp_clip_active = np.zeros(7, dtype=int)
        self.gp_shadow_paper_formula_available = True
        self._gp_shadow_variance_warned = False

        # Stage 1: frozen GP / compensation 实验开关。默认保持原 online update 和原 model 路径。
        # compensation 默认关闭，避免 GP prediction 在未显式开启时影响最终 tau。
        self.declare_parameter("gp_prediction_enabled", True)
        self.declare_parameter("gp_online_update_enabled", True)
        self.declare_parameter("gp_model_dir", "./new_structure/gp/gp_models")
        self.declare_parameter("gp_compensation_enabled", False)
        self.declare_parameter("gp_compensation_source", "local")
        self.declare_parameter("gp_compensation_scale", 0.1)
        self.declare_parameter("gp_compensation_clip_nm", 0.5)
        self.declare_parameter("gp_shadow_paper_fusion_logging_enabled", False)
        self.declare_parameter("gp_historical_shadow_enabled", False)
        self.declare_parameter("gp_historical_source_mode", "none")
        self.declare_parameter("gp_shadow_variance_eps", 1e-9)
        self.declare_parameter("gp_shadow_hist_fallback_variance", 1e6)

        self.gp_prediction_enabled = self._get_bool_parameter("gp_prediction_enabled")
        self.gp_online_update_enabled = self._get_bool_parameter("gp_online_update_enabled")
        self.gp_model_dir = str(self.get_parameter("gp_model_dir").value)
        self.gp_compensation_enabled = self._get_bool_parameter("gp_compensation_enabled")
        self.gp_compensation_source = str(self.get_parameter("gp_compensation_source").value).strip().lower()
        self.gp_compensation_scale = float(self.get_parameter("gp_compensation_scale").value)
        self.gp_compensation_clip_nm = float(self.get_parameter("gp_compensation_clip_nm").value)
        self.gp_shadow_paper_fusion_logging_enabled = self._get_bool_parameter(
            "gp_shadow_paper_fusion_logging_enabled"
        )
        self.gp_historical_shadow_enabled = self._get_bool_parameter("gp_historical_shadow_enabled")
        self.gp_historical_source_mode = str(
            self.get_parameter("gp_historical_source_mode").value
        ).strip().lower()
        self.gp_shadow_variance_eps = self._get_positive_float_parameter(
            "gp_shadow_variance_eps", 1e-9
        )
        self.gp_shadow_hist_fallback_variance = self._get_positive_float_parameter(
            "gp_shadow_hist_fallback_variance", 1e6
        )
        self._gp_compensation_logged = False

        # clip 只接受非负幅值；负数配置按绝对值处理，避免反向区间。
        if self.gp_compensation_clip_nm < 0.0:
            self.get_logger().warn(
                f"[GP] gp_compensation_clip_nm={self.gp_compensation_clip_nm} is negative; using abs value"
            )
            self.gp_compensation_clip_nm = abs(self.gp_compensation_clip_nm)

        # compensation source 只允许 local/cloud/combined，非法值保守 fallback 到 local。
        valid_gp_compensation_sources = ("local", "cloud", "combined")
        if self.gp_compensation_source not in valid_gp_compensation_sources:
            self.get_logger().warn(
                f"[GP] Invalid gp_compensation_source='{self.gp_compensation_source}', "
                "falling back to 'local'"
            )
            self.gp_compensation_source = "local"

        # Phase 1 shadow logging 暂不接入 historical retrieval；不能用 online_update 冒充 historical。
        valid_gp_historical_source_modes = ("none",)
        if self.gp_historical_source_mode not in valid_gp_historical_source_modes:
            self.get_logger().warn(
                f"[GP Shadow] Invalid gp_historical_source_mode='{self.gp_historical_source_mode}', "
                "falling back to 'none'"
            )
            self.gp_historical_source_mode = "none"
        self.gp_historical_source_mode_code = 0

        if not self.gp_prediction_enabled and self.gp_compensation_enabled:
            self.get_logger().warn(
                "[GP] gp_prediction_enabled=false forces GP compensation OFF "
                "even though gp_compensation_enabled=true was requested."
            )
            self.gp_compensation_enabled = False

        if not self.gp_prediction_enabled and self.gp_online_update_enabled:
            self.get_logger().warn(
                "[GP] gp_prediction_enabled=false skips GP prediction and online updates; "
                "gp_online_update_enabled has no effect in this run."
            )

        self.get_logger().info(
            "[GP] Experiment controls: "
            f"gp_prediction_enabled={self.gp_prediction_enabled}, "
            f"gp_online_update_enabled={self.gp_online_update_enabled}, "
            f"gp_model_dir='{self.gp_model_dir}', "
            f"gp_compensation_enabled={self.gp_compensation_enabled}, "
            f"gp_compensation_source='{self.gp_compensation_source}', "
            f"gp_compensation_scale={self.gp_compensation_scale}, "
            f"gp_compensation_clip_nm={self.gp_compensation_clip_nm}"
        )
        self.get_logger().info(
            "[GP Shadow] Paper fusion logging controls: "
            f"gp_shadow_paper_fusion_logging_enabled={self.gp_shadow_paper_fusion_logging_enabled}, "
            f"gp_historical_shadow_enabled={self.gp_historical_shadow_enabled}, "
            f"gp_historical_source_mode='{self.gp_historical_source_mode}', "
            f"gp_shadow_variance_eps={self.gp_shadow_variance_eps}, "
            f"gp_shadow_hist_fallback_variance={self.gp_shadow_hist_fallback_variance}"
        )

        self._reset_gp_shadow_state()

        self.gp_stride = 1      # 每 10 个 state callback 做一次 GP（你可以调）
        self.gp_counter = 0
        self.cloud_counter = 0
        self.last_q = np.zeros(7)
        self.last_dq = np.zeros(7)
        self.last_ddq = np.zeros(7)
        self.last_residual = np.zeros(7)

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
        self._load_gp_models(self.gp_model_dir)

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
            'future_delay', 0.00 # 默认 60 ms
        ).value
        self.delay_steps = 0
        self.state_delay_steps = 0   # 你想模拟的通信延迟：20个周期
        self.state_buffer = deque(maxlen=1000)  # 存2秒(1kHz)都够
        self.cloud_delay_steps = 100
        self.y_hat_cloud_buffer = deque(maxlen=self.cloud_delay_steps)
        
        #future prediction state comparison
        self.prev_q_pred = None
        self.prev_dq_pred = None
        self.prev_pred_time = None

        self.pred_time_history = []
        self.q_pred_history = []
        self.dq_pred_history = []
        self.q_future_actual_history = []
        self.dq_future_actual_history = []
        self.q_pred_err_history = []
        self.dq_pred_err_history = []

        
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

        # 未来轨迹 service client
        self.future_traj_client = self.create_client(
            GetFutureTrajectory,
            '/future_task_space'
        )

        # 存最新一次未来轨迹
        self._latest_future_traj = None   # dict: {"x_des": np.array(6,), "dx_des": ..., "ddx_des": ...}
        self._future_traj_counter = 0
        self._future_traj_warned = False

    def _get_bool_parameter(self, name):
        value = self.get_parameter(name).value
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("true", "1", "yes", "on"):
                return True
            if normalized in ("false", "0", "no", "off"):
                return False

        self.get_logger().warn(f"[GP] Parameter '{name}' is not bool-like ({value}); using bool(value)")
        return bool(value)

    def _get_7d_parameter(self, name, default_values):
        raw_value = self.get_parameter(name).value
        try:
            if isinstance(raw_value, str):
                values = [
                    float(item.strip())
                    for item in raw_value.split(',')
                    if item.strip()
                ]
            elif isinstance(raw_value, (list, tuple, np.ndarray)):
                values = [float(item) for item in raw_value]
            else:
                values = [float(raw_value)] * 7

            array = np.array(values, dtype=float)
            if array.shape != (7,) or not np.all(np.isfinite(array)):
                raise ValueError
            return array
        except (TypeError, ValueError):
            self.get_logger().warn(
                f"Parameter '{name}' must be a finite scalar or 7 values; "
                f"using default {default_values}"
            )
            return np.array(default_values, dtype=float)

    def _get_positive_float_parameter(self, name, default_value):
        try:
            value = float(self.get_parameter(name).value)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError
            return value
        except (TypeError, ValueError):
            self.get_logger().warn(
                f"Parameter '{name}' must be a finite value > 0.0; "
                f"using default {default_value}"
            )
            return float(default_value)

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

    def jointSpaceCommandCallback(self, msg):
        """callback function for /joint_space_command subscriber"""
        try:
            q_des = np.array(msg.q_des, dtype=float)
            dq_des = np.array(msg.dq_des, dtype=float)
            ddq_des = np.array(msg.ddq_des, dtype=float)
            if (
                q_des.shape != (7,)
                or dq_des.shape != (7,)
                or ddq_des.shape != (7,)
                or not np.all(np.isfinite(np.concatenate([q_des, dq_des, ddq_des])))
            ):
                raise ValueError("JointSpaceCommand q_des/dq_des/ddq_des must be finite length-7 arrays")

            self.q_des_joint = q_des
            self.dq_des_joint = dq_des
            self.ddq_des_joint = ddq_des
            self.joint_command_enabled = bool(msg.enable)
            self.joint_command_received = True
            self.joint_command_time = self.get_clock().now()
            self._joint_reference_stale_logged = False
            if self.joint_command_enabled:
                self._joint_reference_wait_logged = False
        except ValueError as e:
            self.joint_command_enabled = False
            self.get_logger().error(f"Invalid JointSpaceCommand ignored: {e}")
        
    def dataRecordingCallback(self, msg):
        """callback function for /data_recording_enabled subscriber"""
        self.data_recording_enabled = msg.data

        # 当 TrajectoryPublisher 认为“transition 完成”时，会发 True
        if msg.data and not self.gp_active:
            self.gp_active = True
            self.get_logger().info(
                "[Controller] Data recording enabled -> "
                f"gp_prediction_enabled={self.gp_prediction_enabled}, "
                f"gp_compensation_enabled={self.gp_compensation_enabled}"
            )
        elif not msg.data and self.gp_active:
            # 如果你希望停轨迹时也关掉 GP，可以顺便关掉
            self.gp_active = False
            self.get_logger().info("[Controller] Data recording disabled -> GP compensation DEACTIVATED")

    def _handle_joint_reference_control(
        self,
        t_now,
        t_elapsed,
        q,
        dq,
        dt,
        ddq_est,
        tau_measured,
        gravity_measured
    ):
        if not self.joint_command_received:
            if not self._joint_reference_wait_logged:
                self.get_logger().warn(
                    "reference_mode=joint but no JointSpaceCommand has been received; "
                    "no effort command is published."
                )
                self._joint_reference_wait_logged = True
            return

        if not self.joint_command_enabled:
            self.joint_reference_last_tau = np.zeros(7, dtype=float)
            self.joint_reference_last_tau_time = None
            return

        if self.joint_command_time is None:
            return

        command_age = (t_now - self.joint_command_time).nanoseconds / 1e9
        if command_age > self.joint_space_command_timeout_sec:
            self.joint_reference_last_tau = np.zeros(7, dtype=float)
            self.joint_reference_last_tau_time = None
            if not self._joint_reference_stale_logged:
                self.get_logger().warn(
                    "JointSpaceCommand is stale: "
                    f"age={command_age:.6f}s, "
                    f"timeout={self.joint_space_command_timeout_sec:.6f}s; "
                    "no effort command is published."
                )
                self._joint_reference_stale_logged = True
            return

        tau = (
            self.joint_reference_kp * (self.q_des_joint - q)
            + self.joint_reference_kd * (self.dq_des_joint - dq)
        )
        tau = np.clip(
            tau,
            -self.joint_reference_torque_clip_nm,
            self.joint_reference_torque_clip_nm
        )

        if self.joint_reference_last_tau_time is None:
            command_dt = max(dt, 1e-6)
        else:
            command_dt = (t_now - self.joint_reference_last_tau_time).nanoseconds / 1e9
            command_dt = max(command_dt, 1e-6)
        max_delta = self.joint_reference_torque_rate_limit_nm_per_s * command_dt
        tau = self.joint_reference_last_tau + np.clip(
            tau - self.joint_reference_last_tau,
            -max_delta,
            max_delta
        )
        self.joint_reference_last_tau = tau.copy()
        self.joint_reference_last_tau_time = t_now

        tau_residual = tau_measured - tau - gravity_measured
        if self.data_recording_enabled:
            self.dq_des_joint_history.append(self.dq_des_joint.tolist())
            self.ddq_des_joint_history.append(self.ddq_des_joint.tolist())
            self.tau_residual_raw_history.append(tau_residual.tolist())
        self.tau_residual_filtered = (
            0.02 * tau_residual + 0.98 * self.tau_residual_filtered
        )
        self.state_buffer.append({
            "t": t_elapsed,
            "q": q.copy(),
            "dq": self.dq_des_joint.copy(),
            "ddq_est": ddq_est.copy(),
            "tau_res": self.tau_residual_filtered.copy(),
        })

        # joint reference mode 使用 message 里的 dq_des 作为 GP feature 的第二段；
        # 这不改变现有 Cartesian 分支的 GP 调用语义。
        if self.use_gp and self.gp_prediction_enabled:
            y_hat_local, var_local = self._gp_predict_and_update(
                q,
                self.dq_des_joint,
                self.ddq_des_joint,
                self.tau_residual_filtered,
                self.gp_models_small,
                update=self.gp_online_update_enabled
            )
            self.y_hat_local = y_hat_local
            self.var_local = var_local

            y_hat_cloud, var_cloud = self._gp_predict_and_update(
                q,
                self.dq_des_joint,
                self.ddq_des_joint,
                self.tau_residual_filtered,
                self.gp_models_big,
                update=self.gp_online_update_enabled
            )
            self.y_hat_cloud = y_hat_cloud
            self.var_cloud = var_cloud

            eps = 1e-8
            v_l = np.maximum(self.var_local, eps)
            v_c = np.maximum(self.var_cloud, eps)
            prec_l = 1.0 / v_l
            prec_c = 1.0 / v_c
            w_l = prec_l / (prec_l + prec_c)
            self.y_hat_combined = w_l * self.y_hat_local + (1.0 - w_l) * self.y_hat_cloud

        self._update_gp_shadow_logging_state()
        self._tau_nominal = tau.copy()
        tau = self._apply_gp_compensation(tau)
        self._tau_final = tau.copy()
        self.effort_msg.efforts = tau.tolist()
        self.effort_publisher.publish(self.effort_msg)

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
        self.get_logger().debug(f"Got future traj: x={x_f[:3]}")
    
    def request_future_trajectory(self, t_delay):
        if not self.gp_prediction_enabled:
            return

        if not self.future_traj_client.service_is_ready():
            if not self._future_traj_warned:
                self.get_logger().warn("/future_task_space service not ready")
                self._future_traj_warned = True
            return

        req = GetFutureTrajectory.Request()
        req.t_delay = float(t_delay)

        future = self.future_traj_client.call_async(req)
        future.add_done_callback(self._future_traj_response_callback)

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
                dt = 1e-3

                self.dq_filt = dq_raw.copy()
                self.dq_filt_initialized = True

                dq = dq_raw
                self.dq = dq

                ddq = np.zeros_like(dq)
                self.dq_buffer = dq.copy()

                # ===== rollout init =====
                self.dq_prev = dq.copy()
                self.ddq_est = np.zeros_like(dq)
                self.ddq_est_initialized = True
                ddq_est = self.ddq_est.copy()
            else:
                t_elapsed = (t_now - self.t_initial).nanoseconds / 1e9
                dt = (t_now - self.t_last).nanoseconds / 1e9
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

                dq = dq_raw
                self.dq = dq

                # ===== estimate joint acceleration from measured dq =====
                ddq_raw = (dq - self.dq_prev) / dt
                self.dq_prev = dq.copy()

                if not self.ddq_est_initialized:
                    self.ddq_est = ddq_raw.copy()
                    self.ddq_est_initialized = True
                else:
                    self.ddq_est = self._lowpass_vector(
                        ddq_raw,
                        self.ddq_est,
                        dt,
                        self.ddq_lpf_hz
                    )

                ddq_est = self.ddq_est.copy()

                ddq = (dq - self.dq_buffer) / dt
                self.dq_buffer = dq.copy()                    
                
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
            coriolis_vec = np.array(msg.coriolis, dtype=float)
            zero_jacobian = zero_jacobian_array.reshape(6, 7, order='F')    # 6x7
            zero_jacobian_t = zero_jacobian.T                               # 7x6, transpose of zero_jacobian
            # zero_jacobian_pinv = np.linalg.pinv(zero_jacobian)              # 7x6, pseudoinverse obtained by SVD
            lam = self.dls_lambda
            lam_ns = self.dls_lambda_ns
            # 6×7
            zero_jacobian_pinv = self.dls_dyn_pinv(zero_jacobian, mass_matrix, lam_ns)   # 7×6 

            if self.reference_mode == 'joint':
                self._handle_joint_reference_control(
                    t_now,
                    t_elapsed,
                    q,
                    dq,
                    dt,
                    ddq_est,
                    tau_measured,
                    gravity_measured
                )
                return

            if self.joint_position_control_active and not self.joint_position_adjusted:
                tau, reached, pos_err_norm = self._startup_taskspace_control(
                    t_now, q, dq, dt, o_t_f, zero_jacobian, zero_jacobian_pinv
                )

                if reached:
                    self.joint_position_adjusted = True
                    self.get_logger().info(f"End-effector reached start point. Error={pos_err_norm:.6f}")

                    if not self.trajectory_started:
                        zero_tau = np.zeros(7)
                        self.effort_msg.efforts = zero_tau.tolist()
                        self.effort_publisher.publish(self.effort_msg)
                        self.start_trajectory()
                else:
                    self.effort_msg.efforts = tau.tolist()
                    self.effort_publisher.publish(self.effort_msg)

                return

            if not self.task_command_received:
                print("not received")
                return  

            # cartesian impedance control (after joint position adjustment)   

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

            if self.data_recording_enabled and self.prev_q_pred is not None and self.prev_dq_pred is not None:
                q_err = self.prev_q_pred - q
                dq_err = self.prev_dq_pred - dq

                self.pred_time_history.append(self.prev_pred_time)
                self.q_pred_history.append(self.prev_q_pred.tolist())
                self.dq_pred_history.append(self.prev_dq_pred.tolist())
                self.q_future_actual_history.append(q.tolist())
                self.dq_future_actual_history.append(dq_des_joint.tolist())
                self.q_pred_err_history.append(q_err.tolist())
                self.dq_pred_err_history.append(dq_err.tolist())

                self.prev_q_pred = None
                self.prev_dq_pred = None
                self.prev_pred_time = None
            

            # tau_nullspace = ((np.eye(7) - zero_jacobian_pinv @ zero_jacobian) 
            #     @ (self.dpn_gains * (self.dq_des - dq)))
            N = np.eye(7) - zero_jacobian_pinv @ zero_jacobian   # or using your 5DoF jacobian
            tau_nullspace = N.T @ (- self.dpn_gains * dq)        # dq_des = 0 时就是减振
            tau = tau + tau_nullspace
            tau = tau + self.friction_compensation(dq)

            # if self.data_recording_enabled:
            #     Td = dt   # 或 dt

            #     ddq_used = ddq_est.copy()
            #     dq_pred_nominal = dq.copy() + ddq_used * Td

            #     # ===== 当期望关节加速度接近 0 时，减少预测速度增量，防止过冲 =====
            #     ddq_des_abs = np.abs(self.ddq_des_joint)

            #     # 阈值：小于这个值就认为“接近 0”
            #     ddq_zero_th = 0.5   # 可调，单位 rad/s^2

            #     # 最小保留比例，避免压得太狠
            #     alpha_min = 0.05     # 可调，0~1

            #     # alpha in [alpha_min, 1]
            #     alpha = np.ones(7, dtype=float)
            #     mask = ddq_des_abs < ddq_zero_th
            #     alpha[mask] = alpha_min + (1.0 - alpha_min) * (ddq_des_abs[mask] / ddq_zero_th)

            #     # 只压缩“预测增量”
            #     dq_pred_next = dq.copy() + alpha * (dq_pred_nominal - dq.copy())

            #     q_pred_next = q.copy() + dq_pred_next * Td

            #     self.prev_q_pred = q_pred_next.copy()
            #     self.prev_dq_pred = dq_pred_next.copy()
            #     self.prev_pred_time = t_elapsed
            # if self.data_recording_enabled:
            #     Td = dt

            #     # 用测得/估计的当前关节加速度做常加速度外推
            #     ddq_used = ddq_est.copy()

            #     dq_pred_next = dq.copy() + ddq_used * Td
            #     q_pred_next = q.copy() + dq.copy() * Td + 0.5 * ddq_used * (Td ** 2)

            #     self.prev_q_pred = q_pred_next.copy()
            #     self.prev_dq_pred = dq_pred_next.copy()
            #     self.prev_pred_time = t_elapsed
            # else:
            #     self.prev_q_pred = None
            #     self.prev_dq_pred = None
            #     self.prev_pred_time = None
            # q_pred_next = q.copy()
            # dq_pred_next = dq.copy()

            dq_pred_next = dq_des_joint.copy()
            if self.data_recording_enabled and self.gp_prediction_enabled:
                Td = dt   # 或者固定 0.001
                self.request_future_trajectory(Td)

                if self._latest_future_traj is not None:
                    x_f = np.array(self._latest_future_traj["x_des"], dtype=float)
                    dx_f = np.array(self._latest_future_traj["dx_des"], dtype=float)
                    ddx_f = np.array(self._latest_future_traj["ddx_des"], dtype=float)
                    dq_future_ref = jacobian_pinv @ dx_f[0:5]
                    ddq_future_ref = jacobian_pinv @ (ddx_f[0:5] - djacobian @ dq)

                    dq_pred_next = dq_future_ref.copy()
                    q_pred_next = q.copy()
                    print(dq_pred_next * dt)
                    # dq_pred_next = dq_des_joint
                    # q_pred_next = q.copy()
                    self.prev_q_pred = q_pred_next.copy()
                    self.prev_dq_pred = dq_pred_next.copy()
                    self.prev_pred_time = t_elapsed
            
            # else:
                # self.prev_q_pred = None
                # self.prev_dq_pred = None
                # self.prev_pred_time = None

            # === 计算残差 ===
            tau_residual = tau_measured - tau - gravity_measured
            if self.data_recording_enabled:
                self.tau_residual_raw_history.append(tau_residual.tolist())
            self.tau_residual_filtered = (
                0.02 * tau_residual + 0.98 * self.tau_residual_filtered
            )
            self.state_buffer.append({
                "t": t_elapsed,
                "q": q.copy(),
                "dq": dq_des_joint.copy(),
                "ddq_est": ddq_est.copy(),
                "tau_res": self.tau_residual_filtered.copy(),
            })
            # tau = tau

            # === 控制循环的最后：按节拍触发一次“GP 更新”（本地 + 云端） ===
            if self.gp_active and self.use_gp and self.gp_prediction_enabled:
                self.gp_counter += 1
                tick = (self.gp_counter % self.gp_stride == 0)
                # # ---------------------------------------------------------
                y_hat_local, var_local = self._gp_predict_and_update(
                    self.q, dq, self.ddq_des_joint,
                    self.tau_residual_filtered,
                    self.gp_models_small,
                    # True 保持原 online update；False 用于 frozen GP evaluation，不允许 add_point。
                    update=self.gp_online_update_enabled
                )
                self.y_hat_local = y_hat_local
                self.var_local = var_local

                # Td = float(self.future_delay)
                # delay_steps = max(1, int(self.delay_steps))
                delay_steps = 2

                base_state = None
                if len(self.state_buffer) > delay_steps:
                    base_state = self.state_buffer[-(delay_steps + 1)]
                elif len(self.state_buffer) > 0:
                    base_state = self.state_buffer[0]

                if base_state is not None:
                    q_base = base_state["q"].copy()
                    dq_base = base_state["dq"].copy()
                    ddq_base = base_state["ddq_est"].copy()
                    tau_base = base_state["tau_res"].copy()
                else:
                    q_base = q.copy()
                    dq_base = dq.copy()
                    ddq_base = ddq_est.copy()
                    tau_base = self.tau_residual_filtered.copy()

                # # ===== big GP 用基准帧先更新 =====
                _, _ = self._gp_predict_and_update(
                    q_base, dq_base, ddq_des_joint,
                    tau_base,
                    self.gp_models_big,
                    # big GP 的主路径更新同样受 frozen GP 开关保护。
                    update=self.gp_online_update_enabled
                )

                # ===== 在 Td 附近均匀采样多个 rollout 点 =====
                self.cloud_rollout_n = 1
                self.cloud_rollout_span = 0.001
                Td_center = delay_steps * dt
                Td_samples = self._sample_rollout_times_uniform(
                    Td_center,
                    self.cloud_rollout_n,
                    self.cloud_rollout_span
                )

                y_list = []
                var_list = []
                Td_list = []

                # for Td_i in Td_samples:
                #     q_roll  = q_base + dq_base * Td_i + 0.5 * ddq_base * (Td_i ** 2)
                #     dq_roll = dq_base + ddq_base * Td_i
                #     ddq_roll = ddq_base.copy()

                #     # ===== 找历史中最近的点 =====
                #     nearest_state, nearest_dist = self._find_nearest_history_state(
                #         q_roll, dq_roll, ddq_roll, use_ddq=False
                #     )

                #     if nearest_state is not None:
                #         q_nn = nearest_state["q"].copy()
                #         dq_nn = nearest_state["dq"].copy()
                #         ddq_nn = nearest_state["ddq_est"].copy()
                #     else:
                #         q_nn = q_roll.copy()
                #         dq_nn = dq_roll.copy()
                #         ddq_nn = ddq_roll.copy()

                y_hat_i, var_i = self._gp_predict_and_update(
                    q, dq_pred_next, ddq,
                    tau_base,
                    self.gp_models_big,
                    update=False
                )
                self.y_hat_cloud = y_hat_i.copy()
                    # y_list.append(y_hat_i.copy())
                    # var_list.append(var_i.copy())
                    # Td_list.append(Td_i)

                # ===== variance-weighted fusion =====
                y_arr = np.asarray(y_list, dtype=float)      # (N, 7)
                var_arr = np.asarray(var_list, dtype=float)  # (N, 7)

                eps = 1e-8
                prec_arr = 1.0 / np.maximum(var_arr, eps)    # (N, 7)
                w_arr = prec_arr / np.sum(prec_arr, axis=0, keepdims=True)

                y_hat_cloud = np.sum(y_arr * w_arr, axis=0)
                var_cloud = 1.0 / np.maximum(np.sum(prec_arr, axis=0), eps)

                # self.y_hat_cloud = y_hat_cloud.copy()
                # self.var_cloud = var_cloud.copy()
                
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
            # 默认 compensation 关闭时返回原始 tau；开启后才按原注释方向补偿。
            self._update_gp_shadow_logging_state()
            self._tau_nominal = tau.copy()
            tau = self._apply_gp_compensation(tau)
            self._tau_final = tau.copy()
            # publish on topic /effort_command
            self.effort_msg.efforts = tau.tolist()
            self.effort_publisher.publish(self.effort_msg)

            # record data only when data recording is enabled
            if self.data_recording_enabled:
                self.tau_history.append(tau.tolist())
                self.tau_nominal_history.append(self._tau_nominal.tolist())
                self.tau_final_history.append(self._tau_final.tolist())
                self.gp_source_code_history.append(int(self._gp_source_code))
                self.gp_selected_raw_history.append(self._gp_selected_raw.tolist())
                self.gp_scaled_history.append(self._gp_scaled.tolist())
                self.gp_applied_history.append(self._gp_applied.tolist())
                self.gp_clip_active_history.append(self._gp_clip_active.tolist())
                self.gp_shadow_historical_available_history.append(
                    int(self.gp_shadow_historical_available)
                )
                self.gp_shadow_local_raw_history.append(self.gp_shadow_local_raw.tolist())
                self.gp_shadow_cloud_raw_history.append(self.gp_shadow_cloud_raw.tolist())
                self.gp_shadow_hist_raw_history.append(self.gp_shadow_hist_raw.tolist())
                self.gp_shadow_combined_paper_raw_history.append(
                    self.gp_shadow_combined_paper_raw.tolist()
                )
                self.gp_shadow_var_local_history.append(self.gp_shadow_var_local.tolist())
                self.gp_shadow_var_cloud_history.append(self.gp_shadow_var_cloud.tolist())
                self.gp_shadow_var_hist_history.append(self.gp_shadow_var_hist.tolist())
                self.gp_shadow_weight_local_history.append(self.gp_shadow_weight_local.tolist())
                self.gp_shadow_weight_cloud_history.append(self.gp_shadow_weight_cloud.tolist())
                self.gp_shadow_weight_hist_history.append(self.gp_shadow_weight_hist.tolist())
                self.gp_shadow_precision_local_history.append(self.gp_shadow_precision_local.tolist())
                self.gp_shadow_precision_cloud_history.append(self.gp_shadow_precision_cloud.tolist())
                self.gp_shadow_precision_hist_history.append(self.gp_shadow_precision_hist.tolist())
                self.gp_shadow_paper_scaled_history.append(self.gp_shadow_paper_scaled.tolist())
                self.gp_shadow_paper_clip_proxy_applied_history.append(
                    self.gp_shadow_paper_clip_proxy_applied.tolist()
                )
                self.gp_shadow_paper_clip_proxy_active_history.append(
                    self.gp_shadow_paper_clip_proxy_active.tolist()
                )
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

    def _rollout_with_frozen_tau(
        self, q0, dq0, Td, tau_cmd, mass_matrix, coriolis_vec, gravity_vec, tau_res_hat=None, n_steps=5
    ):
        q_pred = q0.copy()
        dq_pred = dq0.copy()
        ddq_pred = np.zeros(7, dtype=float)

        if tau_res_hat is None:
            tau_res_hat = np.zeros(7, dtype=float)

        dt_r = max(Td / max(1, n_steps), 1e-4)

        for _ in range(max(1, n_steps)):
            tau_fric = self.friction_compensation(dq_pred)
            rhs = tau_cmd - coriolis_vec - gravity_vec - tau_fric - tau_res_hat
            ddq_pred = np.linalg.solve(mass_matrix, rhs)
            dq_pred = dq_pred + ddq_pred * dt_r
            q_pred = q_pred + dq_pred * dt_r

        return q_pred, dq_pred, ddq_pred

    def _make_future_prediction(
        self,
        q,
        dq,
        tau_cmd,
        mass_matrix,
        coriolis_vec,
        gravity_vec,
        tau_res_hat,
        Td,
        n_steps=5,
    ):
        q_pred, dq_pred, ddq_pred = self._rollout_with_frozen_tau(
            q0=q,
            dq0=dq,
            Td=Td,
            tau_cmd=tau_cmd,
            mass_matrix=mass_matrix,
            coriolis_vec=coriolis_vec,
            gravity_vec=gravity_vec,
            tau_res_hat=tau_res_hat,
            n_steps=n_steps,
        )
        return q_pred, dq_pred, ddq_pred

    def _get_startup_task_reference(self, t_now, x_curr):
        """
        startup 阶段 task-space 插值参考
        总时间 T 按初始距离 / 参考速度 自适应确定
        """
        if not self.startup_interp_started:
            self.startup_interp_started = True
            self.startup_interp_start_time = t_now
            self.startup_x0 = x_curr.copy()

            dist = np.linalg.norm(self.x_start_des - self.startup_x0)
            self.startup_interp_total_dist = dist

            # 根据距离自适应设置总时间
            self.startup_interp_duration_adaptive = max(
                dist / max(self.startup_linear_speed, 1e-4),
                0.5   # 最短时间，防止距离太小导致过快
            )

        t_elapsed = (t_now - self.startup_interp_start_time).nanoseconds / 1e9
        T = max(self.startup_interp_duration_adaptive, 1e-6)

        if t_elapsed >= T:
            x_ref = self.x_start_des.copy()
            dx_ref = np.zeros(3, dtype=float)
            ddx_ref = np.zeros(3, dtype=float)
            finished = True
            return x_ref, dx_ref, ddx_ref, finished

        r = np.clip(t_elapsed / T, 0.0, 1.0)

        s = 10*r**3 - 15*r**4 + 6*r**5
        ds_dt = (30*r**2 - 60*r**3 + 30*r**4) / T
        d2s_dt2 = (60*r - 180*r**2 + 120*r**3) / (T**2)

        delta = self.x_start_des - self.startup_x0

        x_ref = self.startup_x0 + s * delta
        dx_ref = ds_dt * delta
        ddx_ref = d2s_dt2 * delta

        finished = False
        return x_ref, dx_ref, ddx_ref, finished
        
    def _startup_taskspace_control(
        self,
        t_now,
        q, dq, dt,
        o_t_f,
        zero_jacobian,
        zero_jacobian_pinv
    ):
        x = o_t_f[:3, 3]
        dx = zero_jacobian @ dq
        rotation_matrix = o_t_f[:3, :3]

        # 生成平滑插值参考
        x_ref, dx_ref_lin, ddx_ref_lin, finished = self._get_startup_task_reference(t_now, x)

        # 位置误差
        pos_error = x - x_ref

        # 姿态误差：仍然拉向固定目标姿态
        r_error = -0.5 * (
            np.cross(rotation_matrix[:, 2], self.rotation_matrix_des[:, 2]) +
            np.cross(rotation_matrix[:, 1], self.rotation_matrix_des[:, 1]) +
            np.cross(rotation_matrix[:, 0], self.rotation_matrix_des[:, 0])
        )

        x_error = np.concatenate([pos_error, r_error])

        dx_ref = np.concatenate([dx_ref_lin, np.zeros(3, dtype=float)])
        dx_error = dx - dx_ref

        self.startup_x_int_error += x_error * dt
        self.startup_x_int_error = np.clip(self.startup_x_int_error, -0.05, 0.05)

        F_task = (
            - self.startup_kp_task * x_error
            - self.startup_kd_task * dx_error
            - self.startup_ki_task * self.startup_x_int_error
        )

        tau = zero_jacobian.T @ F_task

        N = np.eye(7) - zero_jacobian_pinv @ zero_jacobian
        tau_nullspace = N.T @ (- self.dpn_gains * dq)
        tau = tau + tau_nullspace
        tau = tau + self.friction_compensation(dq)
        tau = np.clip(tau, -50.0, 50.0)

        reached = np.linalg.norm(pos_error) < self.startup_pos_threshold
        return tau, (finished and reached), np.linalg.norm(pos_error)

    def _sample_rollout_times_uniform(self, Td, n, span):
        """
        在 Td 附近按均匀分布采样 n 个时间点。
        例如 Td=0.02, span=0.01:
            采样区间 [0.015, 0.025]
        """
        if n <= 1:
            return np.array([Td], dtype=float)

        t_min = max(0.0, Td - 0.5 * span)
        t_max = max(t_min, Td + 0.5 * span)

        return np.linspace(t_min, t_max, n, dtype=float)

    def predict_future_joint_state(q, dq, ddq_des, delay):
        dq_future = dq + ddq_des * delay
        q_future  = q + dq * delay + 0.5 * ddq_des * delay**2
        return q_future, dq_future

    def _build_gp_feature(self, q, dq_des_joint, ddq_des_joint=None):
        """
        和 local GP 使用同一套输入特征。
        当前版本使用 14 维: [q, dq_des_joint]
        如果你以后训练改成 21 维，再切到 [q, dq_des_joint, ddq_des_joint]
        """
        x_full = np.concatenate([q, dq_des_joint]).astype(np.float32)
        return x_full
    
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
                max_data_per_expert=25,
                nearest_k=1,
                max_experts=1,
                timescale=0.03,
            ),
            # 举例：如果你想让 6 号关节忘得快一点、专家少一点，可以单独改：
            6: dict(
                max_data_per_expert=25,
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
                max_data_per_expert=50,
                nearest_k=2,
                max_experts=10,
                timescale=0.03,
            ),
            # 举例：如果你想让 6 号关节忘得快一点、专家少一点，可以单独改：
            6: dict(
                max_data_per_expert=50,
                nearest_k=2,
                max_experts=10,
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

    def _find_nearest_history_state(self, q_query, dq_query, ddq_query=None, use_ddq=False):
        """
        在 state_buffer 中找和查询状态最近的历史点。
        默认用 [q, dq] 做距离；
        如果 use_ddq=True，则用 [q, dq, ddq] 做距离。
        返回:
            nearest_state: dict
            nearest_dist: float
        """
        if len(self.state_buffer) == 0:
            return None, np.inf

        best_state = None
        best_dist = np.inf

        for item in self.state_buffer:
            q_h = item["q"]
            dq_h = item["dq"]

            if use_ddq:
                ddq_h = item["ddq_est"]
                d = np.linalg.norm(
                    np.concatenate([q_query - q_h, dq_query - dq_h, ddq_query - ddq_h])
                )
            else:
                d = np.linalg.norm(
                    np.concatenate([q_query - q_h, dq_query - dq_h])
                )

            if d < best_dist:
                best_dist = d
                best_state = item

        return best_state, best_dist

    def _as_finite_7d(self, value, fill_value=0.0):
        try:
            arr = np.asarray(value, dtype=float)
            if arr.shape != (7,):
                raise ValueError
            return np.where(np.isfinite(arr), arr, fill_value)
        except (TypeError, ValueError):
            return np.ones(7, dtype=float) * fill_value

    def _reset_gp_shadow_state(self):
        zero = np.zeros(7, dtype=float)
        zero_i = np.zeros(7, dtype=int)
        self.gp_shadow_historical_available = 0
        self.gp_shadow_local_raw = zero.copy()
        self.gp_shadow_cloud_raw = zero.copy()
        self.gp_shadow_hist_raw = zero.copy()
        self.gp_shadow_combined_paper_raw = zero.copy()
        self.gp_shadow_var_local = zero.copy()
        self.gp_shadow_var_cloud = zero.copy()
        self.gp_shadow_var_hist = zero.copy()
        self.gp_shadow_weight_local = zero.copy()
        self.gp_shadow_weight_cloud = zero.copy()
        self.gp_shadow_weight_hist = zero.copy()
        self.gp_shadow_precision_local = zero.copy()
        self.gp_shadow_precision_cloud = zero.copy()
        self.gp_shadow_precision_hist = zero.copy()
        self.gp_shadow_paper_scaled = zero.copy()
        self.gp_shadow_paper_clip_proxy_applied = zero.copy()
        self.gp_shadow_paper_clip_proxy_active = zero_i.copy()

    def _sanitize_shadow_variance(self, value, fallback_value):
        fallback_value = max(float(fallback_value), float(self.gp_shadow_variance_eps))
        try:
            arr = np.asarray(value, dtype=float)
            if arr.shape != (7,):
                raise ValueError
            invalid = (~np.isfinite(arr)) | (arr <= 0.0)
        except (TypeError, ValueError):
            arr = np.ones(7, dtype=float) * fallback_value
            invalid = np.ones(7, dtype=bool)

        if np.any(invalid):
            if not self._gp_shadow_variance_warned:
                self.get_logger().warn(
                    "[GP Shadow] Non-finite or non-positive variance detected; using fallback variance."
                )
                self._gp_shadow_variance_warned = True
            arr = arr.copy()
            arr[invalid] = fallback_value

        return np.maximum(arr, self.gp_shadow_variance_eps)

    def _get_historical_shadow_candidate(self):
        fallback_var = np.ones(7, dtype=float) * self.gp_shadow_hist_fallback_variance

        # Phase 1 没有可用的 past prediction pool；online_update 不能当 historical。
        if not self.gp_historical_shadow_enabled:
            return np.zeros(7, dtype=float), fallback_var, 0

        if self.gp_historical_source_mode == "none":
            return np.zeros(7, dtype=float), fallback_var, 0

        return np.zeros(7, dtype=float), fallback_var, 0

    def _compute_inverse_variance_weights(
        self,
        var_local,
        var_cloud,
        var_hist,
        historical_available
    ):
        v_l = self._sanitize_shadow_variance(
            var_local, self.gp_shadow_hist_fallback_variance
        )
        v_c = self._sanitize_shadow_variance(
            var_cloud, self.gp_shadow_hist_fallback_variance
        )
        v_h = self._sanitize_shadow_variance(
            var_hist, self.gp_shadow_hist_fallback_variance
        )

        eps = float(self.gp_shadow_variance_eps)
        prec_l = 1.0 / np.maximum(v_l, eps)
        prec_c = 1.0 / np.maximum(v_c, eps)
        if historical_available:
            prec_h = 1.0 / np.maximum(v_h, eps)
        else:
            prec_h = np.zeros(7, dtype=float)

        denom = prec_l + prec_c + prec_h
        valid = np.isfinite(denom) & (denom > 0.0)
        w_l = np.divide(prec_l, denom, out=np.ones(7, dtype=float) * 0.5, where=valid)
        w_c = np.divide(prec_c, denom, out=np.ones(7, dtype=float) * 0.5, where=valid)
        w_h = np.divide(prec_h, denom, out=np.zeros(7, dtype=float), where=valid)

        if np.any(~valid):
            if historical_available:
                w_l[~valid] = 1.0 / 3.0
                w_c[~valid] = 1.0 / 3.0
                w_h[~valid] = 1.0 / 3.0
            else:
                w_l[~valid] = 0.5
                w_c[~valid] = 0.5
                w_h[~valid] = 0.0

        return w_l, w_c, w_h, prec_l, prec_c, prec_h, v_l, v_c, v_h

    def _compute_paper_tri_temporal_shadow_fusion(
        self,
        y_hat_local,
        var_local,
        y_hat_cloud,
        var_cloud,
        y_hat_historical_shadow,
        var_historical_shadow,
        historical_available
    ):
        y_l = self._as_finite_7d(y_hat_local, 0.0)
        y_c = self._as_finite_7d(y_hat_cloud, 0.0)
        y_h = self._as_finite_7d(y_hat_historical_shadow, 0.0)

        (
            w_l,
            w_c,
            w_h,
            prec_l,
            prec_c,
            prec_h,
            v_l,
            v_c,
            v_h,
        ) = self._compute_inverse_variance_weights(
            var_local, var_cloud, var_historical_shadow, historical_available
        )

        y_fuse = w_l * y_l + w_c * y_c + w_h * y_h
        y_fuse = self._as_finite_7d(y_fuse, 0.0)

        return {
            "y_local": y_l,
            "y_cloud": y_c,
            "y_hist": y_h,
            "y_fuse": y_fuse,
            "var_local": v_l,
            "var_cloud": v_c,
            "var_hist": v_h,
            "weight_local": w_l,
            "weight_cloud": w_c,
            "weight_hist": w_h,
            "precision_local": prec_l,
            "precision_cloud": prec_c,
            "precision_hist": prec_h,
        }

    def _update_gp_shadow_logging_state(self):
        self._reset_gp_shadow_state()

        if (
            not self.gp_shadow_paper_fusion_logging_enabled
            or not self.gp_prediction_enabled
        ):
            return

        y_hist, var_hist, historical_available = self._get_historical_shadow_candidate()
        shadow = self._compute_paper_tri_temporal_shadow_fusion(
            self.y_hat_local,
            self.var_local,
            self.y_hat_cloud,
            self.var_cloud,
            y_hist,
            var_hist,
            historical_available,
        )

        self.gp_shadow_historical_available = int(historical_available)
        self.gp_shadow_local_raw = shadow["y_local"]
        self.gp_shadow_cloud_raw = shadow["y_cloud"]
        self.gp_shadow_hist_raw = shadow["y_hist"]
        self.gp_shadow_combined_paper_raw = shadow["y_fuse"]
        self.gp_shadow_var_local = shadow["var_local"]
        self.gp_shadow_var_cloud = shadow["var_cloud"]
        self.gp_shadow_var_hist = shadow["var_hist"]
        self.gp_shadow_weight_local = shadow["weight_local"]
        self.gp_shadow_weight_cloud = shadow["weight_cloud"]
        self.gp_shadow_weight_hist = shadow["weight_hist"]
        self.gp_shadow_precision_local = shadow["precision_local"]
        self.gp_shadow_precision_cloud = shadow["precision_cloud"]
        self.gp_shadow_precision_hist = shadow["precision_hist"]
        self.gp_shadow_paper_scaled = (
            self.gp_compensation_scale * self.gp_shadow_combined_paper_raw
        )
        self.gp_shadow_paper_clip_proxy_applied = np.clip(
            self.gp_shadow_paper_scaled,
            -self.gp_compensation_clip_nm,
            self.gp_compensation_clip_nm,
        )
        self.gp_shadow_paper_clip_proxy_active = (
            np.abs(
                self.gp_shadow_paper_scaled
                - self.gp_shadow_paper_clip_proxy_applied
            ) > 1e-12
        ).astype(int)

    def _apply_gp_compensation(self, tau):
        # 默认不改变最终 tau，只有显式开启 gp_compensation_enabled 才进入 torque command。
        if not self.gp_prediction_enabled or not self.gp_compensation_enabled:
            self._gp_source_code = 0
            self._gp_selected_raw = np.zeros(7, dtype=float)
            self._gp_scaled = np.zeros(7, dtype=float)
            self._gp_applied = np.zeros(7, dtype=float)
            self._gp_clip_active = np.zeros(7, dtype=int)
            return tau

        # combined 当前是 local/cloud variance fusion candidate；paper/historical 先走 shadow logging。
        if self.gp_compensation_source == "cloud":
            compensation = self.y_hat_cloud
            self._gp_source_code = 2
        elif self.gp_compensation_source == "combined":
            compensation = self.y_hat_combined
            self._gp_source_code = 3
        else:
            compensation = self.y_hat_local
            self._gp_source_code = 1

        # 先 scale 再 per-joint clip，避免 GP prediction 直接大幅影响 torque command。
        self._gp_selected_raw = np.asarray(compensation, dtype=float).copy()
        self._gp_scaled = self.gp_compensation_scale * self._gp_selected_raw
        self._gp_applied = np.clip(
            self._gp_scaled,
            -self.gp_compensation_clip_nm,
            self.gp_compensation_clip_nm
        )
        self._gp_clip_active = (
            np.abs(self._gp_scaled - self._gp_applied) > 1e-12
        ).astype(int)

        if not self._gp_compensation_logged:
            self.get_logger().warn(
                "[GP] Compensation ENABLED: "
                f"source='{self.gp_compensation_source}', "
                f"scale={self.gp_compensation_scale}, "
                f"clip_nm={self.gp_compensation_clip_nm}"
            )
            self._gp_compensation_logged = True

        # 符号方向沿用原注释：tau = tau - compensation。
        return tau - self._gp_applied

    def _gp_predict_and_update(self, q, dq_des_joint, ddq_des_joint, tau_residual, models, update = True):
        """
        本地 GP：高维输入版本（14维 or 21维）
        每个关节都使用相同的 x_full = concat([q, dq, ddq])
        """

        if not self.gp_prediction_enabled:
            return np.zeros(7, dtype=float), np.ones(7, dtype=float) * 1e6

        if not self.gp_ready or not self.use_gp:
            print("[GP] GP not ready or not enabled; skipping prediction")
            return np.zeros(7, dtype=float), np.ones(7, dtype=float) * 1e6

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
        for j in range(1, 8):

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
                    # update=False 时用于 frozen GP evaluation，内部也阻止 add_point。
                    if update and self.gp_online_update_enabled:
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
                self.y_hat_history,
                self.tau_residual_history,
                self.tau_residual_raw_history,
                self.tau_nominal_history,
                self.tau_final_history,
                self.gp_source_code_history,
                self.gp_selected_raw_history,
                self.gp_scaled_history,
                self.gp_applied_history,
                self.gp_clip_active_history,
                self.gp_shadow_historical_available_history,
                self.gp_shadow_local_raw_history,
                self.gp_shadow_cloud_raw_history,
                self.gp_shadow_hist_raw_history,
                self.gp_shadow_combined_paper_raw_history,
                self.gp_shadow_var_local_history,
                self.gp_shadow_var_cloud_history,
                self.gp_shadow_var_hist_history,
                self.gp_shadow_weight_local_history,
                self.gp_shadow_weight_cloud_history,
                self.gp_shadow_weight_hist_history,
                self.gp_shadow_precision_local_history,
                self.gp_shadow_precision_cloud_history,
                self.gp_shadow_precision_hist_history,
                self.gp_shadow_paper_scaled_history,
                self.gp_shadow_paper_clip_proxy_applied_history,
                self.gp_shadow_paper_clip_proxy_active_history,
                self.pred_time_history,
                self.q_pred_history,
                self.dq_pred_history,
                self.q_future_actual_history,
                self.dq_future_actual_history,
                self.q_pred_err_history,
                self.dq_pred_err_history,
            ]
            min_len = len(self.time_history)

            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                header = ['Time(s)', 'PredTime(s)']
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
                header.extend([f'q_pred_{i+1}' for i in range(7)])
                header.extend([f'dq_pred_{i+1}' for i in range(7)])
                header.extend([f'q_future_actual_{i+1}' for i in range(7)])
                header.extend([f'dq_future_actual_{i+1}' for i in range(7)])
                header.extend([f'q_pred_err_{i+1}' for i in range(7)])
                header.extend([f'dq_pred_err_{i+1}' for i in range(7)])
                header.extend([
                    'gp_prediction_enabled',
                    'gp_online_update_enabled',
                    'gp_compensation_enabled',
                    'gp_compensation_source_code',
                    'gp_compensation_scale',
                    'gp_compensation_clip_nm',
                ])
                header.extend([f'tau_nominal_{i+1}' for i in range(7)])
                header.extend([f'tau_final_{i+1}' for i in range(7)])
                header.extend([f'gp_selected_raw_{i+1}' for i in range(7)])
                header.extend([f'gp_scaled_{i+1}' for i in range(7)])
                header.extend([f'gp_applied_{i+1}' for i in range(7)])
                header.extend([f'gp_clip_active_{i+1}' for i in range(7)])
                header.extend([
                    'gp_shadow_paper_fusion_logging_enabled',
                    'gp_historical_shadow_enabled',
                    'gp_historical_source_mode_code',
                    'gp_shadow_paper_formula_available',
                    'gp_shadow_historical_available',
                    'gp_shadow_variance_eps',
                    'gp_shadow_hist_fallback_variance',
                ])
                header.extend([f'gp_shadow_local_raw_{i+1}' for i in range(7)])
                header.extend([f'gp_shadow_cloud_raw_{i+1}' for i in range(7)])
                header.extend([f'gp_shadow_hist_raw_{i+1}' for i in range(7)])
                header.extend([f'gp_shadow_combined_paper_raw_{i+1}' for i in range(7)])
                header.extend([f'gp_shadow_var_local_{i+1}' for i in range(7)])
                header.extend([f'gp_shadow_var_cloud_{i+1}' for i in range(7)])
                header.extend([f'gp_shadow_var_hist_{i+1}' for i in range(7)])
                header.extend([f'gp_shadow_weight_local_{i+1}' for i in range(7)])
                header.extend([f'gp_shadow_weight_cloud_{i+1}' for i in range(7)])
                header.extend([f'gp_shadow_weight_hist_{i+1}' for i in range(7)])
                header.extend([f'gp_shadow_precision_local_{i+1}' for i in range(7)])
                header.extend([f'gp_shadow_precision_cloud_{i+1}' for i in range(7)])
                header.extend([f'gp_shadow_precision_hist_{i+1}' for i in range(7)])
                header.extend([f'gp_shadow_paper_scaled_{i+1}' for i in range(7)])
                header.extend([f'gp_shadow_paper_clip_proxy_applied_{i+1}' for i in range(7)])
                header.extend([f'gp_shadow_paper_clip_proxy_active_{i+1}' for i in range(7)])
                writer.writerow(header)

                for i in range(min_len):
                    row = [self.time_history[i]]

                    if i < len(self.pred_time_history):
                        row.append(self.pred_time_history[i])
                    else:
                        row.append(0.0)

                    row.extend(self.tau_history[i])
                    row.extend(self.x_history[i][:3])
                    row.extend(self.x_des_history[i][:3])
                    row.extend(self.dx_history[i])
                    row.extend(self.dx_des_history[i])
                    row.extend(self.tau_measured_history[i])
                    row.extend(self.gravity_history[i])
                    row.extend(self.q_history[i])
                    row.extend(self.dq_history[i])

                    if i < len(self.dq_des_joint_history):
                        row.extend(self.dq_des_joint_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.ddq_des_joint_history):
                        row.extend(self.ddq_des_joint_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.y_hat_history):
                        row.extend(self.y_hat_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.y_hat_local_history):
                        row.extend(self.y_hat_local_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.y_hat_cloud_history):
                        row.extend(self.y_hat_cloud_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.y_hat_mem_history):
                        row.extend(self.y_hat_mem_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.tau_residual_history):
                        row.extend(self.tau_residual_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.tau_residual_raw_history):
                        row.extend(self.tau_residual_raw_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.q_pred_history):
                        row.extend(self.q_pred_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.dq_pred_history):
                        row.extend(self.dq_pred_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.q_future_actual_history):
                        row.extend(self.q_future_actual_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.dq_future_actual_history):
                        row.extend(self.dq_future_actual_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.q_pred_err_history):
                        row.extend(self.q_pred_err_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.dq_pred_err_history):
                        row.extend(self.dq_pred_err_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_source_code_history):
                        gp_source_code = int(self.gp_source_code_history[i])
                    else:
                        gp_source_code = int(self._gp_source_code)

                    row.extend([
                        int(bool(self.gp_prediction_enabled)),
                        int(bool(self.gp_online_update_enabled)),
                        int(bool(self.gp_compensation_enabled)),
                        gp_source_code,
                        self.gp_compensation_scale,
                        self.gp_compensation_clip_nm,
                    ])

                    if i < len(self.tau_nominal_history):
                        row.extend(self.tau_nominal_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.tau_final_history):
                        row.extend(self.tau_final_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_selected_raw_history):
                        row.extend(self.gp_selected_raw_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_scaled_history):
                        row.extend(self.gp_scaled_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_applied_history):
                        row.extend(self.gp_applied_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_clip_active_history):
                        row.extend([int(v) for v in self.gp_clip_active_history[i]])
                    else:
                        row.extend([0] * 7)

                    if i < len(self.gp_shadow_historical_available_history):
                        historical_available = int(
                            self.gp_shadow_historical_available_history[i]
                        )
                    else:
                        historical_available = int(self.gp_shadow_historical_available)

                    row.extend([
                        int(bool(self.gp_shadow_paper_fusion_logging_enabled)),
                        int(bool(self.gp_historical_shadow_enabled)),
                        int(self.gp_historical_source_mode_code),
                        int(bool(self.gp_shadow_paper_formula_available)),
                        historical_available,
                        self.gp_shadow_variance_eps,
                        self.gp_shadow_hist_fallback_variance,
                    ])

                    if i < len(self.gp_shadow_local_raw_history):
                        row.extend(self.gp_shadow_local_raw_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_shadow_cloud_raw_history):
                        row.extend(self.gp_shadow_cloud_raw_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_shadow_hist_raw_history):
                        row.extend(self.gp_shadow_hist_raw_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_shadow_combined_paper_raw_history):
                        row.extend(self.gp_shadow_combined_paper_raw_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_shadow_var_local_history):
                        row.extend(self.gp_shadow_var_local_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_shadow_var_cloud_history):
                        row.extend(self.gp_shadow_var_cloud_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_shadow_var_hist_history):
                        row.extend(self.gp_shadow_var_hist_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_shadow_weight_local_history):
                        row.extend(self.gp_shadow_weight_local_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_shadow_weight_cloud_history):
                        row.extend(self.gp_shadow_weight_cloud_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_shadow_weight_hist_history):
                        row.extend(self.gp_shadow_weight_hist_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_shadow_precision_local_history):
                        row.extend(self.gp_shadow_precision_local_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_shadow_precision_cloud_history):
                        row.extend(self.gp_shadow_precision_cloud_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_shadow_precision_hist_history):
                        row.extend(self.gp_shadow_precision_hist_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_shadow_paper_scaled_history):
                        row.extend(self.gp_shadow_paper_scaled_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_shadow_paper_clip_proxy_applied_history):
                        row.extend(self.gp_shadow_paper_clip_proxy_applied_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.gp_shadow_paper_clip_proxy_active_history):
                        row.extend([
                            int(v) for v in self.gp_shadow_paper_clip_proxy_active_history[i]
                        ])
                    else:
                        row.extend([0] * 7)

                    if len(row) != len(header):
                        self.get_logger().warning(
                            f"CSV row length mismatch at row {i}: "
                            f"header={len(header)}, row={len(row)}"
                        )

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
