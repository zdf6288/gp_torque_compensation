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
import json
import traceback
import sys
import os, pickle
import importlib.util
import threading, time
from pathlib import Path
from std_msgs.msg import String
from collections import deque

GOAL12_TIMING_FIELDS = [
    "event",
    "run_name",
    "control_frequency",
    "timing_output_dir",
    "data_output_dir",
    "callback_index",
    "ros_time_s",
    "callback_wall_ms",
    "callback_period_ms",
    "callback_deadline_ms",
    "callback_deadline_miss",
    "callback_deadline_ratio",
    "callback_wall_warn_sec",
    "callback_wall_over_warn_count",
    "callback_wall_over_20ms_count",
    "callback_wall_over_50ms_count",
    "callback_wall_over_100ms_count",
    "effort_published_this_tick",
    "effort_publish_skip_reason",
    "effort_publish_count",
    "effort_last_gap_ms",
    "effort_max_gap_ms",
    "effort_gap_warn_sec",
    "effort_gap_warn_count",
    "gp_total_ms",
    "gp_local_predict_ms",
    "gp_cloud_like_predict_ms",
    "gp_add_point_ms",
    "future_request_ms",
    "csv_append_ms",
    "csv_save_ms",
    "state_buffer_append_ms",
    "residual_update_ms",
    "data_recording_enabled",
    "gp_prediction_enabled",
    "gp_prediction_stride",
    "gp_prediction_updated_this_tick",
    "gp_prediction_age_sec",
    "gp_output_fresh",
    "gp_online_update_enabled",
    "gp_compensation_enabled",
    "gp_compensation_source",
    "gp_compensation_scale",
    "gp_compensation_clip_nm",
    "gp_compensation_disable_joint7",
    "delay_steps",
    "future_trajectory_request_stride",
    "future_trajectory_updated_this_tick",
    "local_gp_called",
    "cloud_like_gp_called",
    "add_point_count",
    "exception_flag",
]

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

        self.declare_parameter("delay_steps", 0)
        self.delay_steps = self._get_bounded_int_parameter("delay_steps", 0, 0, 100)
        self.get_logger().info(
            f"[GOAL12 Timing] Cloud-like delay_steps={self.delay_steps} callback(s); "
            "this delays only cloud-like prediction output, not local prediction"
        )

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
        self.gp_prediction_stride_history = []
        self.gp_prediction_updated_this_tick_history = []
        self.gp_prediction_age_sec_history = []
        self.gp_output_fresh_history = []
        self.future_trajectory_request_stride_history = []
        self.future_trajectory_updated_this_tick_history = []
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
        self.gp_shadow_hist_pool_size_history = []
        self.gp_shadow_hist_k_used_history = []
        self.gp_shadow_hist_nearest_distance_history = []
        self.gp_shadow_hist_mean_distance_topk_history = []
        self.hist_db_loaded_history = []
        self.hist_db_query_valid_history = []
        self.hist_db_available_history = []
        self.hist_db_online_disabled_history = []
        self.hist_db_distance_pass_history = []
        self.hist_db_k_used_history = []
        self.hist_db_nearest_distance_history = []
        self.hist_db_mean_topk_distance_history = []
        self.hist_db_gated_source_code_history = []
        self.hist_db_pred_history = []
        self.hist_db_gated_pred_history = []
        self.hist_db_query_stride_history = []
        self.hist_db_query_updated_this_tick_history = []
        self.hist_db_query_reused_history = []
        self.hist_db_query_counter_history = []
        self.hist_db_preflight_phase_history = []
        self.hist_db_preflight_pass_history = []
        self.hist_db_preflight_active_allowed_history = []
        self.hist_db_preflight_sample_count_history = []
        self.hist_db_preflight_pass_ratio_history = []
        self.hist_db_preflight_nearest_mean_history = []
        self.hist_db_preflight_nearest_p95_history = []
        self.hist_db_preflight_nearest_max_history = []
        self.hist_db_runtime_fallback_used_history = []
        self.hist_soft_valid_history = []
        self.hist_soft_nearest_distance_history = []
        self.hist_soft_raw_w_hist_history = []
        self.hist_soft_norm_w_local_history = []
        self.hist_soft_norm_w_cloud_history = []
        self.hist_soft_norm_w_hist_history = []
        self.hist_soft_pred_history = []
        self.hist_soft_delta_vs_local_cloud_history = []
        self.gp_triple_raw_history = []
        self.gp_triple_weight_local_history = []
        self.gp_triple_weight_cloud_history = []
        self.gp_triple_weight_hist_history = []
        self.gp_triple_available_history = []
        self.gp_triple_used_fallback_history = []
        self.gp_triple_fallback_source_code_history = []
        self.gp_triple_weight_mode_code_history = []
        self.gp_triple_hist_weight_cap_history = []
        self.gp_triple_rmse_local_history = []
        self.gp_triple_rmse_cloud_history = []
        self.gp_triple_rmse_hist_history = []
        self.gp_triple_dynamic_distance_ratio_history = []
        self.gp_triple_dynamic_hist_penalty_history = []
        self.gp_triple_dynamic_mode_code_history = []
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
        self.declare_parameter("gp_compensation_disable_joint7", False)
        self.declare_parameter("gp_shadow_paper_fusion_logging_enabled", False)
        self.declare_parameter("gp_historical_shadow_enabled", False)
        self.declare_parameter("gp_historical_source_mode", "none")
        self.declare_parameter("gp_shadow_variance_eps", 1e-9)
        self.declare_parameter("gp_shadow_hist_fallback_variance", 1e6)
        self.declare_parameter("gp_historical_shadow_max_points", 2000)
        self.declare_parameter("gp_historical_shadow_min_points", 10)
        self.declare_parameter("gp_historical_shadow_k", 5)
        self.declare_parameter("gp_historical_shadow_max_distance", 1e6)
        self.declare_parameter("gp_historical_shadow_variance_floor", 1e-8)
        self.declare_parameter("gp_historical_shadow_distance_eps", 1e-9)
        self.declare_parameter("gp_historical_db_enabled", False)
        self.declare_parameter("gp_historical_db_path", "")
        self.declare_parameter("gp_historical_db_k", 25)
        self.declare_parameter("gp_historical_db_q_scale", 0.1)
        self.declare_parameter("gp_historical_db_dq_scale", 0.1)
        self.declare_parameter("gp_historical_db_max_distance", 1.0)
        self.declare_parameter("gp_historical_db_query_stride", 1)
        self.declare_parameter("gp_historical_db_disable_when_online_update", True)
        self.declare_parameter("gp_historical_db_fallback_source", "cloud")
        self.declare_parameter("gp_historical_db_preflight_enabled", False)
        self.declare_parameter("gp_historical_db_preflight_required", False)
        self.declare_parameter("gp_disable_silent_hist_fallback", False)
        self.declare_parameter("gp_historical_db_preflight_mode", "segment")
        self.declare_parameter("gp_historical_db_preflight_duration_sec", 5.0)
        self.declare_parameter("gp_historical_db_preflight_min_samples", 50)
        self.declare_parameter("gp_historical_db_preflight_min_pass_ratio", 0.95)
        self.declare_parameter("gp_historical_db_preflight_p95_max_distance", 1.5)
        self.declare_parameter("gp_historical_db_preflight_max_distance", 2.0)
        self.declare_parameter("gp_historical_db_preflight_log_first_n", 5)
        self.declare_parameter("gp_triple_weight_mode", "inverse_rmse")
        self.declare_parameter("gp_triple_weight_local", 0.10)
        self.declare_parameter("gp_triple_weight_cloud", 0.20)
        self.declare_parameter("gp_triple_weight_hist", 0.70)
        self.declare_parameter("gp_triple_weight_normalize", True)
        self.declare_parameter("gp_triple_rmse_local", 0.330269)
        self.declare_parameter("gp_triple_rmse_cloud", 0.330278)
        self.declare_parameter("gp_triple_rmse_hist", 0.093071)
        self.declare_parameter("gp_triple_inverse_rmse_eps", 1e-9)
        self.declare_parameter("gp_triple_hist_distance_scale", 2.0)
        self.declare_parameter("gp_triple_hist_distance_power", 2.0)
        self.declare_parameter("gp_triple_hist_weight_cap", 0.70)
        self.declare_parameter("gp_triple_hist_min_weight", 0.0)
        self.declare_parameter("gp_triple_dynamic_eps", 1e-9)
        self.declare_parameter("gp_triple_min_weight_local", 0.05)
        self.declare_parameter("gp_triple_min_weight_cloud", 0.05)
        self.declare_parameter("gp_triple_require_hist_available", True)
        self.declare_parameter("gp_triple_fallback_source", "combined")
        self.declare_parameter("gp_triple_debug_safety_log_enabled", True)
        self.declare_parameter("gp_triple_debug_safety_log_first_n", 5)
        self.declare_parameter("gp_historical_soft_shadow_enabled", False)
        self.declare_parameter("gp_historical_soft_alpha", 1.0)
        self.declare_parameter("gp_historical_soft_distance_threshold", 0.2)
        self.declare_parameter("gp_historical_soft_online_scale", 0.02)
        self.declare_parameter("gp_historical_soft_non_online_scale", 1.0)
        self.declare_parameter("run_name", "")
        self.declare_parameter("data_output_dir", ".")
        self.declare_parameter("control_frequency", 50.0)
        self.declare_parameter("timing_logging_enabled", False)
        self.declare_parameter("timing_log_stride", 1)
        self.declare_parameter("timing_output_dir", "outputs/goal12_controller_timing")
        self.declare_parameter("deadline_ratio_warn_threshold", 0.8)
        self.declare_parameter("effort_gap_diagnostics_enabled", False)
        self.declare_parameter("effort_gap_log_stride", 100)
        self.declare_parameter("effort_gap_warn_sec", 0.2)
        self.declare_parameter("callback_wall_warn_sec", 0.02)
        self.declare_parameter("gp_prediction_stride", 5)
        self.declare_parameter("gp_output_timeout_sec", 0.5)
        self.declare_parameter("future_trajectory_request_stride", 5)

        self.gp_prediction_enabled = self._get_bool_parameter("gp_prediction_enabled")
        self.gp_prediction_stride = self._get_bounded_int_parameter(
            "gp_prediction_stride", 5, 1, 100
        )
        self.gp_output_timeout_sec = self._get_positive_float_parameter(
            "gp_output_timeout_sec", 0.5
        )
        self.future_trajectory_request_stride = self._get_bounded_int_parameter(
            "future_trajectory_request_stride", 5, 1, 100
        )
        self.gp_online_update_enabled = self._get_bool_parameter("gp_online_update_enabled")
        self.gp_model_dir = str(self.get_parameter("gp_model_dir").value)
        self.gp_compensation_enabled = self._get_bool_parameter("gp_compensation_enabled")
        self.gp_compensation_source = str(self.get_parameter("gp_compensation_source").value).strip().lower()
        self.gp_compensation_scale = float(self.get_parameter("gp_compensation_scale").value)
        self.gp_compensation_clip_nm = float(self.get_parameter("gp_compensation_clip_nm").value)
        self.gp_compensation_disable_joint7 = self._get_bool_parameter(
            "gp_compensation_disable_joint7"
        )
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
        self.gp_historical_shadow_max_points = self._get_bounded_int_parameter(
            "gp_historical_shadow_max_points", 2000, 1, 100000
        )
        self.gp_historical_shadow_min_points = self._get_bounded_int_parameter(
            "gp_historical_shadow_min_points", 10, 1, 100000
        )
        self.gp_historical_shadow_k = self._get_bounded_int_parameter(
            "gp_historical_shadow_k", 5, 1, 100000
        )
        self.gp_historical_shadow_max_distance = self._get_positive_float_parameter(
            "gp_historical_shadow_max_distance", 1e6
        )
        self.gp_historical_shadow_variance_floor = self._get_positive_float_parameter(
            "gp_historical_shadow_variance_floor", 1e-8
        )
        self.gp_historical_shadow_distance_eps = self._get_positive_float_parameter(
            "gp_historical_shadow_distance_eps", 1e-9
        )
        self.gp_historical_db_enabled = self._get_bool_parameter("gp_historical_db_enabled")
        self.gp_historical_db_path = str(
            self.get_parameter("gp_historical_db_path").value
        ).strip()
        self.gp_historical_db_k = self._get_bounded_int_parameter(
            "gp_historical_db_k", 25, 1, 1000000
        )
        self.gp_historical_db_q_scale = self._get_positive_float_parameter(
            "gp_historical_db_q_scale", 0.1
        )
        self.gp_historical_db_dq_scale = self._get_positive_float_parameter(
            "gp_historical_db_dq_scale", 0.1
        )
        self.gp_historical_db_max_distance = self._get_positive_float_parameter(
            "gp_historical_db_max_distance", 1.0
        )
        self.gp_historical_db_query_stride = self._get_bounded_int_parameter(
            "gp_historical_db_query_stride", 1, 1, 1000000
        )
        self.gp_historical_db_disable_when_online_update = self._get_bool_parameter(
            "gp_historical_db_disable_when_online_update"
        )
        self.gp_historical_db_fallback_source = str(
            self.get_parameter("gp_historical_db_fallback_source").value
        ).strip().lower()
        self.gp_historical_db_preflight_enabled = self._get_bool_parameter(
            "gp_historical_db_preflight_enabled"
        )
        self.gp_historical_db_preflight_required = self._get_bool_parameter(
            "gp_historical_db_preflight_required"
        )
        self.gp_disable_silent_hist_fallback = self._get_bool_parameter(
            "gp_disable_silent_hist_fallback"
        )
        self.gp_historical_db_preflight_mode = str(
            self.get_parameter("gp_historical_db_preflight_mode").value
        ).strip().lower()
        self.gp_historical_db_preflight_duration_sec = (
            self._get_nonnegative_float_parameter(
                "gp_historical_db_preflight_duration_sec", 5.0
            )
        )
        self.gp_historical_db_preflight_min_samples = self._get_bounded_int_parameter(
            "gp_historical_db_preflight_min_samples", 50, 1, 1000000
        )
        self.gp_historical_db_preflight_min_pass_ratio = (
            self._get_bounded_float_parameter(
                "gp_historical_db_preflight_min_pass_ratio", 0.95, 0.0, 1.0
            )
        )
        self.gp_historical_db_preflight_p95_max_distance = (
            self._get_positive_float_parameter(
                "gp_historical_db_preflight_p95_max_distance", 1.5
            )
        )
        self.gp_historical_db_preflight_max_distance = (
            self._get_positive_float_parameter(
                "gp_historical_db_preflight_max_distance", 2.0
            )
        )
        self.gp_historical_db_preflight_log_first_n = self._get_bounded_int_parameter(
            "gp_historical_db_preflight_log_first_n", 5, 0, 1000000
        )
        self.gp_triple_weight_mode = str(
            self.get_parameter("gp_triple_weight_mode").value
        ).strip().lower()
        self.gp_triple_weight_local_param = self._get_nonnegative_float_parameter(
            "gp_triple_weight_local", 0.10
        )
        self.gp_triple_weight_cloud_param = self._get_nonnegative_float_parameter(
            "gp_triple_weight_cloud", 0.20
        )
        self.gp_triple_weight_hist_param = self._get_nonnegative_float_parameter(
            "gp_triple_weight_hist", 0.70
        )
        self.gp_triple_weight_normalize = self._get_bool_parameter(
            "gp_triple_weight_normalize"
        )
        self.gp_triple_rmse_local = self._get_positive_float_parameter(
            "gp_triple_rmse_local", 0.330269
        )
        self.gp_triple_rmse_cloud = self._get_positive_float_parameter(
            "gp_triple_rmse_cloud", 0.330278
        )
        self.gp_triple_rmse_hist = self._get_positive_float_parameter(
            "gp_triple_rmse_hist", 0.093071
        )
        self.gp_triple_inverse_rmse_eps = self._get_positive_float_parameter(
            "gp_triple_inverse_rmse_eps", 1e-9
        )
        self.gp_triple_hist_distance_scale = self._get_positive_float_parameter(
            "gp_triple_hist_distance_scale", 2.0
        )
        self.gp_triple_hist_distance_power = self._get_positive_float_parameter(
            "gp_triple_hist_distance_power", 2.0
        )
        self.gp_triple_hist_weight_cap = self._get_nonnegative_float_parameter(
            "gp_triple_hist_weight_cap", 0.70
        )
        self.gp_triple_hist_min_weight = self._get_nonnegative_float_parameter(
            "gp_triple_hist_min_weight", 0.0
        )
        self.gp_triple_dynamic_eps = self._get_positive_float_parameter(
            "gp_triple_dynamic_eps", 1e-9
        )
        self.gp_triple_min_weight_local = self._get_nonnegative_float_parameter(
            "gp_triple_min_weight_local", 0.05
        )
        self.gp_triple_min_weight_cloud = self._get_nonnegative_float_parameter(
            "gp_triple_min_weight_cloud", 0.05
        )
        self.gp_triple_require_hist_available = self._get_bool_parameter(
            "gp_triple_require_hist_available"
        )
        self.gp_triple_fallback_source = str(
            self.get_parameter("gp_triple_fallback_source").value
        ).strip().lower()
        self.gp_triple_debug_safety_log_enabled = self._get_bool_parameter(
            "gp_triple_debug_safety_log_enabled"
        )
        self.gp_triple_debug_safety_log_first_n = self._get_bounded_int_parameter(
            "gp_triple_debug_safety_log_first_n", 5, 0, 1000000
        )
        self.gp_historical_soft_shadow_enabled = self._get_bool_parameter(
            "gp_historical_soft_shadow_enabled"
        )
        self.gp_historical_soft_alpha = self._get_nonnegative_float_parameter(
            "gp_historical_soft_alpha", 1.0
        )
        self.gp_historical_soft_distance_threshold = self._get_nonnegative_float_parameter(
            "gp_historical_soft_distance_threshold", 0.2
        )
        self.gp_historical_soft_online_scale = self._get_nonnegative_float_parameter(
            "gp_historical_soft_online_scale", 0.02
        )
        self.gp_historical_soft_non_online_scale = self._get_nonnegative_float_parameter(
            "gp_historical_soft_non_online_scale", 1.0
        )
        self.run_name = str(self.get_parameter("run_name").value).strip()
        self.data_output_dir = str(self.get_parameter("data_output_dir").value).strip() or "."
        self.control_frequency = self._get_positive_float_parameter(
            "control_frequency", 50.0
        )
        self.timing_logging_enabled = self._get_bool_parameter("timing_logging_enabled")
        self.timing_log_stride = self._get_bounded_int_parameter(
            "timing_log_stride", 1, 1, 1000000
        )
        self.timing_output_dir = str(self.get_parameter("timing_output_dir").value).strip()
        if not self.timing_output_dir:
            self.timing_output_dir = "outputs/goal12_controller_timing"
        self.deadline_ratio_warn_threshold = self._get_nonnegative_float_parameter(
            "deadline_ratio_warn_threshold", 0.8
        )
        # Diagnostics only: locate Python topic relay jitter; values never enter torque control.
        self.effort_gap_diagnostics_enabled = self._get_bool_parameter(
            "effort_gap_diagnostics_enabled"
        )
        self.effort_gap_log_stride = self._get_bounded_int_parameter(
            "effort_gap_log_stride", 100, 1, 1000000
        )
        self.effort_gap_warn_sec = self._get_nonnegative_float_parameter(
            "effort_gap_warn_sec", 0.2
        )
        self.callback_wall_warn_sec = self._get_nonnegative_float_parameter(
            "callback_wall_warn_sec", 0.02
        )
        self._gp_compensation_logged = False
        self._gp_triple_debug_safety_log_count = 0

        # clip 只接受非负幅值；负数配置按绝对值处理，避免反向区间。
        if self.gp_compensation_clip_nm < 0.0:
            self.get_logger().warn(
                f"[GP] gp_compensation_clip_nm={self.gp_compensation_clip_nm} is negative; using abs value"
            )
            self.gp_compensation_clip_nm = abs(self.gp_compensation_clip_nm)

        # compensation source 只允许显式列出的安全链路，非法值保守 fallback 到 local。
        valid_gp_compensation_sources = (
            "local",
            "cloud",
            "combined",
            "hist_db",
            "triple",
            "triple_dynamic",
        )
        if self.gp_compensation_source not in valid_gp_compensation_sources:
            self.get_logger().warn(
                f"[GP] Invalid gp_compensation_source='{self.gp_compensation_source}', "
                "falling back to 'local'"
            )
            self.gp_compensation_source = "local"

        # historical 只允许 shadow source；不能用 online_update 冒充 historical。
        valid_gp_historical_source_modes = ("none", "local_prediction_pool")
        if self.gp_historical_source_mode not in valid_gp_historical_source_modes:
            self.get_logger().warn(
                f"[GP Shadow] Invalid gp_historical_source_mode='{self.gp_historical_source_mode}', "
                "falling back to 'none'"
            )
            self.gp_historical_source_mode = "none"
        self.gp_historical_source_mode_code = (
            1 if self.gp_historical_source_mode == "local_prediction_pool" else 0
        )

        valid_gp_historical_db_fallback_sources = ("none", "local", "cloud", "combined")
        if self.gp_historical_db_fallback_source not in valid_gp_historical_db_fallback_sources:
            self.get_logger().warn(
                "[GP Hist DB] Invalid gp_historical_db_fallback_source="
                f"'{self.gp_historical_db_fallback_source}', falling back to 'cloud'"
            )
            self.gp_historical_db_fallback_source = "cloud"
        self.gp_historical_db_fallback_source_code = {
            "none": 0,
            "local": 1,
            "cloud": 2,
            "combined": 3,
        }[self.gp_historical_db_fallback_source]

        valid_preflight_modes = ("single", "segment", "single_and_segment")
        if self.gp_historical_db_preflight_mode not in valid_preflight_modes:
            self.get_logger().warn(
                "[GP Hist DB] Invalid gp_historical_db_preflight_mode="
                f"'{self.gp_historical_db_preflight_mode}', falling back to 'segment'"
            )
            self.gp_historical_db_preflight_mode = "segment"

        valid_gp_triple_weight_modes = ("fixed", "inverse_rmse")
        if self.gp_triple_weight_mode not in valid_gp_triple_weight_modes:
            self.get_logger().warn(
                f"[GP Triple] Invalid gp_triple_weight_mode='{self.gp_triple_weight_mode}', "
                "falling back to 'inverse_rmse'"
            )
            self.gp_triple_weight_mode = "inverse_rmse"
        self.gp_triple_weight_mode_code = {
            "fixed": 1,
            "inverse_rmse": 2,
        }[self.gp_triple_weight_mode]

        valid_gp_triple_fallback_sources = ("none", "local", "cloud", "combined", "hist_db")
        if self.gp_triple_fallback_source not in valid_gp_triple_fallback_sources:
            self.get_logger().warn(
                f"[GP Triple] Invalid gp_triple_fallback_source='{self.gp_triple_fallback_source}', "
                "falling back to 'combined'"
            )
            self.gp_triple_fallback_source = "combined"
        self.gp_triple_fallback_source_code = {
            "none": 0,
            "local": 1,
            "cloud": 2,
            "combined": 3,
            "hist_db": 4,
        }[self.gp_triple_fallback_source]
        self.gp_triple_weights = self._compute_gp_triple_weights()

        if self.gp_historical_shadow_min_points > self.gp_historical_shadow_max_points:
            self.get_logger().warn(
                "[GP Shadow] gp_historical_shadow_min_points exceeds max_points; "
                "using max_points"
            )
            self.gp_historical_shadow_min_points = self.gp_historical_shadow_max_points
        if self.gp_historical_shadow_k > self.gp_historical_shadow_max_points:
            self.get_logger().warn(
                "[GP Shadow] gp_historical_shadow_k exceeds max_points; using max_points"
            )
            self.gp_historical_shadow_k = self.gp_historical_shadow_max_points

        # Runtime historical shadow pool: 保存 past local GP prediction，不保存 residual。
        self.gp_hist_x_shadow = deque(maxlen=self.gp_historical_shadow_max_points)
        self.gp_hist_mu_shadow = deque(maxlen=self.gp_historical_shadow_max_points)
        self.gp_hist_var_shadow = deque(maxlen=self.gp_historical_shadow_max_points)
        self.gp_hist_t_shadow = deque(maxlen=self.gp_historical_shadow_max_points)
        self._gp_local_feature_shadow = np.zeros(14, dtype=np.float32)
        self._gp_local_prediction_sequence_shadow = 0
        self._gp_hist_last_appended_sequence_shadow = 0
        self._reset_gp_triple_state()

        # Persistent residual DB is separate from runtime prediction-pool paper fusion.
        # It enters active torque only through explicit hist_db/triple/triple_dynamic source selection.
        self.gp_historical_db_loaded = False
        self.gp_historical_db_row_count = 0
        self.gp_historical_db_x_scaled = None
        self.gp_historical_db_y_residual = None
        # hist DB 查询节流状态；默认 stride=1 时每个 callback 查询，保持旧行为。
        self._hist_db_query_counter = 0
        self._hist_db_last_query_result = None
        self.hist_db_query_reused = 0
        self.hist_db_query_counter = 0
        self.gp_historical_db_feature_scale = np.array(
            [self.gp_historical_db_q_scale] * 7
            + [self.gp_historical_db_dq_scale] * 7,
            dtype=float,
        )
        self._load_historical_residual_db()

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
            f"gp_compensation_clip_nm={self.gp_compensation_clip_nm}, "
            f"gp_compensation_disable_joint7={self.gp_compensation_disable_joint7}, "
            f"delay_steps={self.delay_steps}, "
            f"gp_prediction_stride={self.gp_prediction_stride}, "
            f"gp_output_timeout_sec={self.gp_output_timeout_sec}, "
            f"future_trajectory_request_stride={self.future_trajectory_request_stride}, "
            f"control_frequency={self.control_frequency}, "
            f"run_name='{self.run_name}', "
            f"data_output_dir='{self.data_output_dir}'"
        )
        self.get_logger().info(
            "[GP Shadow] Paper fusion logging controls: "
            f"gp_shadow_paper_fusion_logging_enabled={self.gp_shadow_paper_fusion_logging_enabled}, "
            f"gp_historical_shadow_enabled={self.gp_historical_shadow_enabled}, "
            f"gp_historical_source_mode='{self.gp_historical_source_mode}', "
            f"gp_shadow_variance_eps={self.gp_shadow_variance_eps}, "
            f"gp_shadow_hist_fallback_variance={self.gp_shadow_hist_fallback_variance}"
        )
        self.get_logger().info(
            "[GP Shadow] Historical source controls: "
            f"enabled={self.gp_historical_shadow_enabled}, "
            f"mode='{self.gp_historical_source_mode}', "
            f"max_points={self.gp_historical_shadow_max_points}, "
            f"min_points={self.gp_historical_shadow_min_points}, "
            f"k={self.gp_historical_shadow_k}, "
            f"max_distance={self.gp_historical_shadow_max_distance}, "
            f"variance_floor={self.gp_historical_shadow_variance_floor}, "
            f"distance_eps={self.gp_historical_shadow_distance_eps}; "
            "shadow-only and does not enter tau_final"
        )
        self.get_logger().info(
            "[GP Hist DB] Persistent residual DB controls: "
            f"enabled={self.gp_historical_db_enabled}, "
            f"path='{self.gp_historical_db_path}', "
            f"loaded={self.gp_historical_db_loaded}, "
            f"rows={self.gp_historical_db_row_count}, "
            f"k={self.gp_historical_db_k}, "
            f"q_scale={self.gp_historical_db_q_scale}, "
            f"dq_scale={self.gp_historical_db_dq_scale}, "
            f"max_distance={self.gp_historical_db_max_distance}, "
            "disable_when_online_update="
            f"{self.gp_historical_db_disable_when_online_update}, "
            f"fallback_source='{self.gp_historical_db_fallback_source}'; "
            f"preflight_enabled={self.gp_historical_db_preflight_enabled}, "
            f"preflight_required={self.gp_historical_db_preflight_required}, "
            f"preflight_mode='{self.gp_historical_db_preflight_mode}', "
            f"disable_silent_fallback={self.gp_disable_silent_hist_fallback}; "
            "active only with explicit hist_db/triple/triple_dynamic source"
        )
        self.get_logger().info(
            "[GP Triple] Fusion controls: "
            f"mode='{self.gp_triple_weight_mode}', "
            f"weights=({self.gp_triple_weights[0]:.6f}, "
            f"{self.gp_triple_weights[1]:.6f}, "
            f"{self.gp_triple_weights[2]:.6f}), "
            f"fixed_weights=({self.gp_triple_weight_local_param}, "
            f"{self.gp_triple_weight_cloud_param}, "
            f"{self.gp_triple_weight_hist_param}), "
            f"rmse=({self.gp_triple_rmse_local}, "
            f"{self.gp_triple_rmse_cloud}, "
            f"{self.gp_triple_rmse_hist}), "
            f"dynamic_distance_scale={self.gp_triple_hist_distance_scale}, "
            f"dynamic_distance_power={self.gp_triple_hist_distance_power}, "
            f"dynamic_eps={self.gp_triple_dynamic_eps}, "
            f"hist_weight_cap={self.gp_triple_hist_weight_cap}, "
            f"hist_min_weight={self.gp_triple_hist_min_weight}, "
            f"min_weight_local={self.gp_triple_min_weight_local}, "
            f"min_weight_cloud={self.gp_triple_min_weight_cloud}, "
            f"require_hist_available={self.gp_triple_require_hist_available}, "
            f"fallback_source='{self.gp_triple_fallback_source}', "
            f"debug_safety_log_enabled={self.gp_triple_debug_safety_log_enabled}, "
            f"debug_safety_log_first_n={self.gp_triple_debug_safety_log_first_n}; "
            "active only with gp_compensation_source='triple' or 'triple_dynamic' "
            "and compensation enabled"
        )
        self.get_logger().info(
            "[GP Hist Soft] Soft-weight shadow logging controls: "
            f"enabled={self.gp_historical_soft_shadow_enabled}, "
            f"alpha={self.gp_historical_soft_alpha}, "
            f"distance_threshold={self.gp_historical_soft_distance_threshold}, "
            f"online_scale={self.gp_historical_soft_online_scale}, "
            f"non_online_scale={self.gp_historical_soft_non_online_scale}; "
            "shadow-only and does not enter tau_final"
        )
        if (
            self.gp_historical_db_enabled
            and self.gp_historical_db_disable_when_online_update
            and self.gp_online_update_enabled
        ):
            self.get_logger().warn(
                "[GP Hist DB] Online GP update is enabled; persistent DB queries "
                "will be logged but the historical availability gate remains closed."
            )

        self.callback_deadline_ms = 1000.0 / self.control_frequency
        self.timing_history = []
        self.timing_callback_index = 0
        self._last_callback_perf = None
        self._timing_disabled_after_error = False
        self._last_csv_save_ms = None
        if self.timing_logging_enabled:
            self.get_logger().info(
                "[GOAL12 Timing] Timing logging enabled: "
                f"stride={self.timing_log_stride}, "
                f"output_dir='{self.timing_output_dir}', "
                f"control_frequency={self.control_frequency}, "
                f"deadline_ms={self.callback_deadline_ms:.3f}"
            )
        else:
            self.get_logger().info("[GOAL12 Timing] Timing logging disabled by default")
        if self.effort_gap_diagnostics_enabled:
            self.get_logger().info(
                "[EffortGapDiag] Diagnostics enabled: "
                f"log_stride={self.effort_gap_log_stride}, "
                f"effort_gap_warn_sec={self.effort_gap_warn_sec:.3f}, "
                f"callback_wall_warn_sec={self.callback_wall_warn_sec:.3f}"
            )

        self._reset_gp_shadow_state()
        self._reset_historical_db_preflight_state()
        self._reset_historical_residual_db_shadow_state()

        self.gp_stride = self.gp_prediction_stride
        self.gp_counter = 0
        self.future_trajectory_request_counter = 0
        self._gp_prediction_updated_this_tick = 0
        self._last_gp_prediction_time = None
        self._last_valid_gp_prediction = False
        self.gp_prediction_age_sec = 0.0
        self.gp_output_fresh = 0
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
        self._reset_gp_model_diagnostics()

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
        self.y_hat_cloud_current = np.zeros(7)
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

        # Watchdog variables for effort command publishing continuity
        self.last_publish_time = None
        self.publish_count = 0
        self.last_watchdog_warning_time = 0.0
        self._last_publish_perf = None
        self.last_effort_publish_gap_sec = 0.0
        self.max_effort_publish_gap_sec = 0.0
        self.effort_publish_gap_warn_count = 0
        self.effort_publish_gap_window = deque(maxlen=10000)
        self._effort_published_this_tick = 0
        self._effort_publish_skip_reason = ""
        self.callback_wall_over_warn_count = 0
        self.callback_wall_over_20ms_count = 0
        self.callback_wall_over_50ms_count = 0
        self.callback_wall_over_100ms_count = 0
        self._effort_gap_diag_callback_count = 0

        # simulated future trajectory request delay
        self.future_delay = self.declare_parameter(
            'future_delay', 0.00 # 默认 60 ms
        ).value
        self.state_buffer = deque(maxlen=1000)  # 存2秒(1kHz)都够
        self.y_hat_cloud_buffer = deque(maxlen=101)
        
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
        self._last_future_trajectory_time = None
        self._future_request_pending = False
        self._future_trajectory_updated_this_tick = 0
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

    def _disable_timing_logging(self, error):
        if not self._timing_disabled_after_error:
            self.get_logger().error(
                "[GOAL12 Timing] Timing instrumentation failed; "
                f"disabling timing logging: {error}"
            )
        self._timing_disabled_after_error = True
        self.timing_logging_enabled = False

    def _start_callback_timing(self, callback_start_perf):
        if not self.timing_logging_enabled:
            return None

        try:
            self.timing_callback_index += 1
            callback_index = self.timing_callback_index
            if self._last_callback_perf is None:
                callback_period_ms = ""
            else:
                callback_period_ms = (
                    callback_start_perf - self._last_callback_perf
                ) * 1000.0
            self._last_callback_perf = callback_start_perf

            if callback_index % self.timing_log_stride != 0:
                return None

            timing_row = {field: "" for field in GOAL12_TIMING_FIELDS}
            timing_row.update({
                "event": "callback",
                "run_name": self.run_name,
                "control_frequency": self.control_frequency,
                "timing_output_dir": self.timing_output_dir,
                "data_output_dir": self.data_output_dir,
                "callback_index": callback_index,
                "callback_period_ms": callback_period_ms,
                "callback_deadline_ms": self.callback_deadline_ms,
                "callback_wall_warn_sec": self.callback_wall_warn_sec,
                "gp_total_ms": 0.0,
                "gp_local_predict_ms": 0.0,
                "gp_cloud_like_predict_ms": 0.0,
                "gp_add_point_ms": 0.0,
                "future_request_ms": 0.0,
                "csv_append_ms": 0.0,
                "state_buffer_append_ms": 0.0,
                "residual_update_ms": 0.0,
                "gp_prediction_stride": self.gp_prediction_stride,
                "gp_prediction_updated_this_tick": 0,
                "gp_prediction_age_sec": self.gp_prediction_age_sec,
                "gp_output_fresh": int(self.gp_output_fresh),
                "future_trajectory_request_stride": self.future_trajectory_request_stride,
                "future_trajectory_updated_this_tick": 0,
                "local_gp_called": 0,
                "cloud_like_gp_called": 0,
                "add_point_count": 0,
                "exception_flag": 0,
                "effort_gap_warn_sec": self.effort_gap_warn_sec,
            })
            return timing_row
        except Exception as e:
            self._disable_timing_logging(e)
            return None

    def _finish_callback_timing(self, timing_row, callback_start_perf):
        if callback_start_perf is None:
            return

        try:
            callback_wall_ms = (time.perf_counter() - callback_start_perf) * 1000.0
            self._update_effort_gap_diagnostics(callback_wall_ms)
            if timing_row is None:
                return
            deadline_ms = self.callback_deadline_ms
            deadline_ratio = callback_wall_ms / deadline_ms if deadline_ms > 0.0 else 0.0
            timing_row.update({
                "callback_wall_ms": callback_wall_ms,
                "callback_deadline_ratio": deadline_ratio,
                "callback_deadline_miss": int(callback_wall_ms > deadline_ms),
                "callback_wall_warn_sec": self.callback_wall_warn_sec,
                "callback_wall_over_warn_count": int(self.callback_wall_over_warn_count),
                "callback_wall_over_20ms_count": int(self.callback_wall_over_20ms_count),
                "callback_wall_over_50ms_count": int(self.callback_wall_over_50ms_count),
                "callback_wall_over_100ms_count": int(self.callback_wall_over_100ms_count),
                "effort_published_this_tick": int(self._effort_published_this_tick),
                "effort_publish_skip_reason": self._effort_publish_skip_reason,
                "effort_publish_count": int(self.publish_count),
                "effort_last_gap_ms": self.last_effort_publish_gap_sec * 1000.0,
                "effort_max_gap_ms": self.max_effort_publish_gap_sec * 1000.0,
                "effort_gap_warn_sec": self.effort_gap_warn_sec,
                "effort_gap_warn_count": int(self.effort_publish_gap_warn_count),
                "data_recording_enabled": int(bool(self.data_recording_enabled)),
                "gp_prediction_enabled": int(bool(self.gp_prediction_enabled)),
                "gp_prediction_stride": int(self.gp_prediction_stride),
                "gp_prediction_updated_this_tick": int(self._gp_prediction_updated_this_tick),
                "gp_prediction_age_sec": float(self.gp_prediction_age_sec),
                "gp_output_fresh": int(self.gp_output_fresh),
                "gp_online_update_enabled": int(bool(self.gp_online_update_enabled)),
                "gp_compensation_enabled": int(bool(self.gp_compensation_enabled)),
                "gp_compensation_source": self.gp_compensation_source,
                "gp_compensation_scale": self.gp_compensation_scale,
                "gp_compensation_clip_nm": self.gp_compensation_clip_nm,
                "gp_compensation_disable_joint7": int(bool(self.gp_compensation_disable_joint7)),
                "delay_steps": self.delay_steps,
                "future_trajectory_request_stride": int(self.future_trajectory_request_stride),
                "future_trajectory_updated_this_tick": int(self._future_trajectory_updated_this_tick),
            })
            self.timing_history.append(timing_row)
        except Exception as e:
            self._disable_timing_logging(e)

    def _timing_add_ms(self, timing_row, field, start_perf):
        if timing_row is None:
            return

        try:
            duration_ms = (time.perf_counter() - start_perf) * 1000.0
            current_value = timing_row.get(field, 0.0)
            if current_value == "":
                current_value = 0.0
            timing_row[field] = float(current_value) + duration_ms
        except Exception as e:
            self._disable_timing_logging(e)

    def _mark_effort_publish_skipped(self, reason):
        if not self._effort_published_this_tick and not self._effort_publish_skip_reason:
            self._effort_publish_skip_reason = reason

    @staticmethod
    def _percentile(sorted_values, fraction):
        if not sorted_values:
            return 0.0
        idx = int(round((len(sorted_values) - 1) * fraction))
        idx = min(max(idx, 0), len(sorted_values) - 1)
        return float(sorted_values[idx])

    def _update_effort_gap_diagnostics(self, callback_wall_ms):
        callback_wall_sec = callback_wall_ms / 1000.0
        if callback_wall_sec > self.callback_wall_warn_sec:
            self.callback_wall_over_warn_count += 1
        if callback_wall_sec > 0.020:
            self.callback_wall_over_20ms_count += 1
        if callback_wall_sec > 0.050:
            self.callback_wall_over_50ms_count += 1
        if callback_wall_sec > 0.100:
            self.callback_wall_over_100ms_count += 1

        if not self.effort_gap_diagnostics_enabled:
            return

        self._effort_gap_diag_callback_count += 1
        if self._effort_gap_diag_callback_count % self.effort_gap_log_stride != 0:
            return

        gap_values = sorted(self.effort_publish_gap_window)
        p95_gap = self._percentile(gap_values, 0.95)
        p99_gap = self._percentile(gap_values, 0.99)
        log_msg = (
            "[EffortGapDiag] "
            f"callbacks={self._effort_gap_diag_callback_count}, "
            f"publish_count={self.publish_count}, "
            f"last_gap={self.last_effort_publish_gap_sec:.6f}s, "
            f"max_gap={self.max_effort_publish_gap_sec:.6f}s, "
            f"p95_gap={p95_gap:.6f}s, "
            f"p99_gap={p99_gap:.6f}s, "
            f"gap_warn_count={self.effort_publish_gap_warn_count}, "
            f"callback_wall_ms={callback_wall_ms:.3f}, "
            f"over_20ms={self.callback_wall_over_20ms_count}, "
            f"over_50ms={self.callback_wall_over_50ms_count}, "
            f"over_100ms={self.callback_wall_over_100ms_count}, "
            f"published_this_tick={self._effort_published_this_tick}, "
            f"skip_reason='{self._effort_publish_skip_reason}'"
        )
        if (
            self.last_effort_publish_gap_sec > self.effort_gap_warn_sec
            or callback_wall_sec > self.callback_wall_warn_sec
            or not self._effort_published_this_tick
        ):
            self.get_logger().warn(log_msg)
        else:
            self.get_logger().info(log_msg)

    def _append_csv_save_timing(self, csv_save_ms):
        if not self.timing_logging_enabled:
            return

        try:
            timing_row = {field: "" for field in GOAL12_TIMING_FIELDS}
            timing_row.update({
                "event": "csv_save",
                "run_name": self.run_name,
                "control_frequency": self.control_frequency,
                "timing_output_dir": self.timing_output_dir,
                "data_output_dir": self.data_output_dir,
                "callback_deadline_ms": self.callback_deadline_ms,
                "csv_save_ms": csv_save_ms,
                "data_recording_enabled": int(bool(self.data_recording_enabled)),
                "gp_prediction_enabled": int(bool(self.gp_prediction_enabled)),
                "gp_prediction_stride": int(self.gp_prediction_stride),
                "gp_prediction_updated_this_tick": int(self._gp_prediction_updated_this_tick),
                "gp_prediction_age_sec": float(self.gp_prediction_age_sec),
                "gp_output_fresh": int(self.gp_output_fresh),
                "gp_online_update_enabled": int(bool(self.gp_online_update_enabled)),
                "gp_compensation_enabled": int(bool(self.gp_compensation_enabled)),
                "gp_compensation_source": self.gp_compensation_source,
                "gp_compensation_scale": self.gp_compensation_scale,
                "gp_compensation_clip_nm": self.gp_compensation_clip_nm,
                "gp_compensation_disable_joint7": int(bool(self.gp_compensation_disable_joint7)),
                "delay_steps": self.delay_steps,
                "future_trajectory_request_stride": int(self.future_trajectory_request_stride),
                "future_trajectory_updated_this_tick": int(self._future_trajectory_updated_this_tick),
            })
            self.timing_history.append(timing_row)
        except Exception as e:
            self._disable_timing_logging(e)

    def _finish_csv_save_timing(self, csv_save_start):
        if csv_save_start is None:
            return

        try:
            csv_save_ms = (time.perf_counter() - csv_save_start) * 1000.0
            self._last_csv_save_ms = csv_save_ms
            self._append_csv_save_timing(csv_save_ms)
            self.save_timing_to_file()
        except Exception as e:
            self._disable_timing_logging(e)

    def save_timing_to_file(self):
        if not self.timing_logging_enabled or not self.timing_history:
            return

        try:
            output_dir = Path(self.timing_output_dir).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            run_name_stem = Path(self.run_name).name if self.run_name else ""
            filename_stem = (
                f"{run_name_stem}_goal12_controller_timing.csv"
                if run_name_stem
                else "goal12_controller_timing.csv"
            )
            filename = output_dir / filename_stem

            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=GOAL12_TIMING_FIELDS)
                writer.writeheader()
                for row in self.timing_history:
                    writer.writerow({
                        field: row.get(field, "")
                        for field in GOAL12_TIMING_FIELDS
                    })

            callback_rows = [
                row for row in self.timing_history
                if row.get("event") == "callback" and row.get("callback_wall_ms") != ""
            ]
            if callback_rows:
                miss_count = sum(
                    int(row.get("callback_deadline_miss", 0))
                    for row in callback_rows
                )
                max_ratio = max(
                    float(row.get("callback_deadline_ratio", 0.0))
                    for row in callback_rows
                )
                effort_gap_ms = sorted(
                    float(row.get("effort_last_gap_ms", 0.0))
                    for row in callback_rows
                    if row.get("effort_last_gap_ms") not in ("", None)
                )
                max_effort_gap_ms = max(effort_gap_ms) if effort_gap_ms else 0.0
                p95_effort_gap_ms = self._percentile(effort_gap_ms, 0.95)
                p99_effort_gap_ms = self._percentile(effort_gap_ms, 0.99)
                skip_count = sum(
                    1
                    for row in callback_rows
                    if str(row.get("effort_publish_skip_reason", "")).strip()
                )
                msg = (
                    f"[GOAL12 Timing] Saved {len(callback_rows)} callback timing rows "
                    f"to {filename}; deadline_miss_count={miss_count}, "
                    f"max_deadline_ratio={max_ratio:.3f}, "
                    f"effort_gap_ms(max/p95/p99)="
                    f"{max_effort_gap_ms:.3f}/{p95_effort_gap_ms:.3f}/{p99_effort_gap_ms:.3f}, "
                    f"effort_skip_count={skip_count}"
                )
                if max_ratio >= self.deadline_ratio_warn_threshold:
                    self.get_logger().warn(msg)
                else:
                    self.get_logger().info(msg)
            else:
                self.get_logger().info(
                    f"[GOAL12 Timing] Saved timing metadata to {filename}"
                )
        except Exception as e:
            self._disable_timing_logging(e)

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

    def _get_nonnegative_float_parameter(self, name, default_value):
        try:
            value = float(self.get_parameter(name).value)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError
            return value
        except (TypeError, ValueError):
            self.get_logger().warn(
                f"Parameter '{name}' must be a finite value >= 0.0; "
                f"using default {default_value}"
            )
            return float(default_value)

    def _get_bounded_float_parameter(self, name, default_value, min_value, max_value):
        try:
            raw_value = self.get_parameter(name).value
            if isinstance(raw_value, bool):
                raise ValueError
            value = float(raw_value)
            if not np.isfinite(value) or value < min_value or value > max_value:
                raise ValueError
            return value
        except (TypeError, ValueError, OverflowError):
            self.get_logger().warn(
                f"Parameter '{name}' must be a finite value in [{min_value}, {max_value}]; "
                f"using default {default_value}"
            )
            return float(default_value)

    def _get_bounded_int_parameter(self, name, default_value, min_value, max_value):
        try:
            raw_value = self.get_parameter(name).value
            if isinstance(raw_value, bool):
                raise ValueError
            numeric_value = float(raw_value)
            value = int(numeric_value)
            if (
                not np.isfinite(numeric_value)
                or numeric_value != value
                or value < min_value
                or value > max_value
            ):
                raise ValueError
            return value
        except (TypeError, ValueError, OverflowError):
            self.get_logger().warn(
                f"Parameter '{name}' must be an integer in [{min_value}, {max_value}]; "
                f"using default {default_value}"
            )
            return int(default_value)

    def _begin_control_tick(self):
        self._gp_prediction_updated_this_tick = 0
        self.hist_db_query_updated_this_tick = 0
        self.hist_db_runtime_fallback_used = 0
        self._future_trajectory_updated_this_tick = 0
        self._effort_published_this_tick = 0
        self._effort_publish_skip_reason = ""

    def _should_run_gp_prediction_this_tick(self):
        self.gp_counter += 1
        should_update = self.gp_counter % self.gp_prediction_stride == 0
        self._gp_prediction_updated_this_tick = int(should_update)
        return should_update

    def _gp_outputs_are_valid(self):
        vector_values = (
            self.y_hat_local,
            self.y_hat_cloud,
            self.y_hat_combined,
        )
        variance_values = (
            self.var_local,
            self.var_cloud,
        )

        try:
            for value in vector_values:
                arr = np.asarray(value, dtype=float)
                if arr.shape != (7,) or not np.all(np.isfinite(arr)):
                    return False
            for value in variance_values:
                arr = np.asarray(value, dtype=float)
                if (
                    arr.shape != (7,)
                    or not np.all(np.isfinite(arr))
                    or np.any(arr <= 0.0)
                ):
                    return False
        except (TypeError, ValueError):
            return False

        return True

    def _mark_gp_prediction_result(self, t_now):
        if self._gp_outputs_are_valid():
            self._last_valid_gp_prediction = True
            self._last_gp_prediction_time = t_now
        else:
            self._last_valid_gp_prediction = False
            self._last_gp_prediction_time = None
            self.y_hat_local = np.zeros(7, dtype=float)
            self.y_hat_cloud = np.zeros(7, dtype=float)
            self.y_hat_combined = np.zeros(7, dtype=float)
            self.var_local = np.ones(7, dtype=float) * 1e6
            self.var_cloud = np.ones(7, dtype=float) * 1e6

        self._update_gp_output_freshness(t_now)

    def _update_gp_output_freshness(self, t_now):
        self.gp_prediction_age_sec = 0.0
        self.gp_output_fresh = 0
        if not self._last_valid_gp_prediction or self._last_gp_prediction_time is None:
            return

        try:
            age_sec = (t_now - self._last_gp_prediction_time).nanoseconds / 1e9
        except Exception:
            age_sec = self.gp_output_timeout_sec + 1.0

        if not np.isfinite(age_sec) or age_sec < 0.0:
            age_sec = self.gp_output_timeout_sec + 1.0

        self.gp_prediction_age_sec = float(age_sec)
        self.gp_output_fresh = int(age_sec <= self.gp_output_timeout_sec)

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
        gravity_measured,
        timing_row=None
    ):
        if not self.joint_command_received:
            if not self._joint_reference_wait_logged:
                self.get_logger().warn(
                    "reference_mode=joint but no JointSpaceCommand has been received; "
                    "no effort command is published."
                )
                self._joint_reference_wait_logged = True
            self._mark_effort_publish_skipped("joint_command_missing")
            return

        if not self.joint_command_enabled:
            self.joint_reference_last_tau = np.zeros(7, dtype=float)
            self.joint_reference_last_tau_time = None
            self._mark_effort_publish_skipped("joint_command_disabled")
            return

        if self.joint_command_time is None:
            self._mark_effort_publish_skipped("joint_command_time_missing")
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
            self._mark_effort_publish_skipped("joint_command_stale")
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

        self._reset_historical_residual_db_shadow_state()

        # joint reference mode 使用 message 里的 dq_des 作为 GP feature 的第二段；
        # 这不改变现有 Cartesian 分支的 GP 调用语义。
        if self.use_gp and self.gp_prediction_enabled:
            gp_total_start = time.perf_counter() if timing_row is not None else None
            if self._should_run_gp_prediction_this_tick():
                y_hat_local, var_local = self._gp_predict_and_update(
                    q,
                    self.dq_des_joint,
                    self.ddq_des_joint,
                    self.tau_residual_filtered,
                    self.gp_models_small,
                    update=self.gp_online_update_enabled,
                    timing_label="local",
                    timing_row=timing_row
                )
                self.y_hat_local = y_hat_local
                self.var_local = var_local
                if (
                    self.gp_shadow_paper_fusion_logging_enabled
                    and self.gp_historical_shadow_enabled
                    and self.gp_historical_source_mode == "local_prediction_pool"
                ):
                    self._gp_local_feature_shadow = self._build_gp_shadow_feature(
                        q, self.dq_des_joint
                    )
                    self._gp_local_prediction_sequence_shadow += 1

                y_hat_cloud_current, var_cloud_current = self._gp_predict_and_update(
                    q,
                    self.dq_des_joint,
                    self.ddq_des_joint,
                    self.tau_residual_filtered,
                    self.gp_models_big,
                    update=self.gp_online_update_enabled,
                    timing_label="cloud_like",
                    timing_row=timing_row
                )
                self.y_hat_cloud_current = y_hat_cloud_current.copy()
                self.y_hat_cloud, self.var_cloud = self._delay_cloud_like_output(
                    y_hat_cloud_current,
                    var_cloud_current
                )

                eps = 1e-8
                v_l = np.maximum(self.var_local, eps)
                v_c = np.maximum(self.var_cloud, eps)
                prec_l = 1.0 / v_l
                prec_c = 1.0 / v_c
                w_l = prec_l / (prec_l + prec_c)
                self.y_hat_combined = w_l * self.y_hat_local + (1.0 - w_l) * self.y_hat_cloud
                self._mark_gp_prediction_result(t_now)
                if gp_total_start is not None:
                    self._timing_add_ms(timing_row, "gp_total_ms", gp_total_start)
            else:
                self._update_gp_output_freshness(t_now)

            self._update_historical_residual_db_shadow_state(q, dq, t_now)

        self._update_gp_shadow_logging_state(q, self.dq_des_joint)
        self._tau_nominal = tau.copy()
        tau = self._apply_gp_compensation(tau)
        self._tau_final = tau.copy()
        self.effort_msg.efforts = tau.tolist()
        self._publish_effort(self.effort_msg)

    def _future_traj_response_callback(self, future):
        try:
            res = future.result()
        except Exception as e:
            self._future_request_pending = False
            self.get_logger().error(f"[Controller] /future_task_space call failed: {e}")
            return

        x_f  = np.array(res.x_des, dtype=float)
        dx_f = np.array(res.dx_des, dtype=float)
        ddx_f = np.array(res.ddx_des, dtype=float)

        if (
            x_f.ndim != 1
            or dx_f.ndim != 1
            or ddx_f.ndim != 1
            or x_f.shape[0] < 3
            or dx_f.shape[0] < 5
            or ddx_f.shape[0] < 5
            or not np.all(np.isfinite(x_f))
            or not np.all(np.isfinite(dx_f))
            or not np.all(np.isfinite(ddx_f))
        ):
            self._future_request_pending = False
            return

        self._latest_future_traj = {
            "x_des": x_f,
            "dx_des": dx_f,
            "ddx_des": ddx_f,
        }
        self._last_future_trajectory_time = self.get_clock().now()
        self._future_request_pending = False
        # 调试时可以看看
        self.get_logger().debug(f"Got future traj: x={x_f[:3]}")
    
    def request_future_trajectory(self, t_delay):
        if not self.gp_prediction_enabled:
            return False

        if self._future_request_pending:
            return False

        if not self.future_traj_client.service_is_ready():
            if not self._future_traj_warned:
                self.get_logger().warn("/future_task_space service not ready")
                self._future_traj_warned = True
            return False

        req = GetFutureTrajectory.Request()
        req.t_delay = float(t_delay)

        try:
            future = self.future_traj_client.call_async(req)
            future.add_done_callback(self._future_traj_response_callback)
        except Exception as e:
            self._future_request_pending = False
            self.get_logger().error(f"[Controller] /future_task_space request failed: {e}")
            return False

        self._future_request_pending = True
        self._future_trajectory_updated_this_tick = 1
        return True

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
                self._publish_effort(zero_tau)
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
                # 使用 subprocess.Popen 后台非阻塞执行画图，并使用 nice -n 19 降低优先级
                # 避免同步执行阻塞 Python Executor 导致控制器停摆以及与实时控制抢占 CPU
                import subprocess
                subprocess.Popen(
                    ["nice", "-n", "19", "python3", "ablation.py", "cartesian_impedance_controller_data.csv"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.get_logger().info("[Controller] Background plotting process spawned successfully (nice -n 19).")
            except Exception as e:
                self.get_logger().error(f"Plotting error: {e}")

            # ------------------------------------------
            # 4) 安全退出
            # ------------------------------------------
            rclpy.shutdown()
            os._exit(0)
    

    def stateParameterCallback(self, msg):
        """callback function for /state_parameter subscriber"""
        timing_row = None
        callback_start_perf = None
        if self.timing_logging_enabled or self.effort_gap_diagnostics_enabled:
            callback_start_perf = time.perf_counter()
        if self.timing_logging_enabled:
            timing_row = self._start_callback_timing(callback_start_perf)

        try:
            # initialize t_initial, get t_elapsed, t_last and dt
            # initialize q_initial, get q, dq and ddq
            t_now = self.get_clock().now()
            self._begin_control_tick()
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

            if timing_row is not None:
                timing_row["ros_time_s"] = t_elapsed

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
                    gravity_measured,
                    timing_row=timing_row
                )
                return

            if self.joint_position_control_active and not self.joint_position_adjusted:
                tau, reached, pos_err_norm = self._startup_taskspace_control(
                    t_now, q, dq, dt, o_t_f, zero_jacobian, zero_jacobian_pinv
                )

                if reached:
                    if not self.trajectory_started:
                        self.start_trajectory()
                        self.get_logger().info(f"End-effector reached start point. Error={pos_err_norm:.6f}. Requesting trajectory start...")

                    # 只有真正收到了 task command 后才关闭启动期控制，完成无缝切换
                    # 防止由于轨迹发布器启动延迟导致 /effort_command 发送出现空闲周期 (Transition Gap)
                    if self.task_command_received:
                        self.joint_position_adjusted = True
                        self.get_logger().info(f"First task command received. Switching to active trajectory control.")

                # 无论是否 reached，均持续发布正常的阻抗保持力矩，不发零力矩，不断流
                self.effort_msg.efforts = tau.tolist()
                self._publish_effort(self.effort_msg)
                return

            if not self.task_command_received:
                # 真机实时控制中避免高频终端输出；这里保持静默，防止增加通信/调度负载。
                # 当定位完成后 (self.joint_position_adjusted == True) 若仍无轨迹数据导致不发布，则触发看门狗警告
                if self.joint_position_adjusted:
                    self._log_watchdog_warning("task_command_received is False (Transition Gap / Drop)", t_elapsed)
                self._mark_effort_publish_skipped("task_command_missing")
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
                self.future_trajectory_request_counter += 1
                should_request_future = (
                    self.future_trajectory_request_counter % self.future_trajectory_request_stride == 0
                )
                if should_request_future:
                    future_request_start = time.perf_counter() if timing_row is not None else None
                    self.request_future_trajectory(Td)
                    if future_request_start is not None:
                        self._timing_add_ms(
                            timing_row,
                            "future_request_ms",
                            future_request_start
                        )

                if self._latest_future_traj is not None:
                    x_f = np.array(self._latest_future_traj["x_des"], dtype=float)
                    dx_f = np.array(self._latest_future_traj["dx_des"], dtype=float)
                    ddx_f = np.array(self._latest_future_traj["ddx_des"], dtype=float)
                    dq_future_ref = jacobian_pinv @ dx_f[0:5]
                    ddq_future_ref = jacobian_pinv @ (ddx_f[0:5] - djacobian @ dq)

                    dq_pred_next = dq_future_ref.copy()
                    q_pred_next = q.copy()
                    # 真机实时控制中禁止每周期打印预测步长，避免 stdout I/O 造成负载抖动。
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
            residual_update_start = time.perf_counter() if timing_row is not None else None
            tau_residual = tau_measured - tau - gravity_measured
            if self.data_recording_enabled:
                self.tau_residual_raw_history.append(tau_residual.tolist())
            self.tau_residual_filtered = (
                0.02 * tau_residual + 0.98 * self.tau_residual_filtered
            )
            if residual_update_start is not None:
                self._timing_add_ms(
                    timing_row,
                    "residual_update_ms",
                    residual_update_start
                )
            state_buffer_append_start = time.perf_counter() if timing_row is not None else None
            self.state_buffer.append({
                "t": t_elapsed,
                "q": q.copy(),
                "dq": dq_des_joint.copy(),
                "ddq_est": ddq_est.copy(),
                "tau_res": self.tau_residual_filtered.copy(),
            })
            if state_buffer_append_start is not None:
                self._timing_add_ms(
                    timing_row,
                    "state_buffer_append_ms",
                    state_buffer_append_start
                )
            # tau = tau

            # === 控制循环的最后：按节拍触发一次“GP 更新”（本地 + 云端） ===
            if self.gp_active and self.use_gp and self.gp_prediction_enabled:
                gp_total_start = time.perf_counter() if timing_row is not None else None
                gp_tick = self._should_run_gp_prediction_this_tick()
                if gp_tick:
                    self._reset_historical_residual_db_shadow_state()
                    # # ---------------------------------------------------------
                    y_hat_local, var_local = self._gp_predict_and_update(
                        self.q, dq, self.ddq_des_joint,
                        self.tau_residual_filtered,
                        self.gp_models_small,
                        # True 保持原 online update；False 用于 frozen GP evaluation，不允许 add_point。
                        update=self.gp_online_update_enabled,
                        timing_label="local",
                        timing_row=timing_row
                    )
                    self.y_hat_local = y_hat_local
                    self.var_local = var_local
                    if (
                        self.gp_shadow_paper_fusion_logging_enabled
                        and self.gp_historical_shadow_enabled
                        and self.gp_historical_source_mode == "local_prediction_pool"
                    ):
                        self._gp_local_feature_shadow = self._build_gp_shadow_feature(
                            self.q, dq
                        )
                        self._gp_local_prediction_sequence_shadow += 1

                    y_hat_cloud_current, var_cloud_current = self._gp_predict_and_update(
                        q, dq_pred_next, ddq,
                        self.tau_residual_filtered,
                        self.gp_models_big,
                        update=self.gp_online_update_enabled,
                        timing_label="cloud_like",
                        timing_row=timing_row
                    )
                    self.y_hat_cloud_current = y_hat_cloud_current.copy()
                    self.y_hat_cloud, self.var_cloud = self._delay_cloud_like_output(
                        y_hat_cloud_current,
                        var_cloud_current
                    )
                    
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
                    self._mark_gp_prediction_result(t_now)
                    self._update_historical_residual_db_shadow_state(self.q, dq, t_now)
                    if gp_total_start is not None:
                        self._timing_add_ms(timing_row, "gp_total_ms", gp_total_start)
                else:
                    self._update_gp_output_freshness(t_now)

            # tau = tau - self.y_hat_local
            # 默认 compensation 关闭时返回原始 tau；开启后才按原注释方向补偿。
            self._update_gp_shadow_logging_state(self.q, dq)
            self._tau_nominal = tau.copy()
            tau = self._apply_gp_compensation(tau)
            self._tau_final = tau.copy()
            # publish on topic /effort_command
            self.effort_msg.efforts = tau.tolist()
            self._publish_effort(self.effort_msg)

            # record data only when data recording is enabled
            if self.data_recording_enabled:
                csv_append_start = time.perf_counter() if timing_row is not None else None
                self.tau_history.append(tau.tolist())
                self.tau_nominal_history.append(self._tau_nominal.tolist())
                self.tau_final_history.append(self._tau_final.tolist())
                self.gp_source_code_history.append(int(self._gp_source_code))
                self.gp_selected_raw_history.append(self._gp_selected_raw.tolist())
                self.gp_scaled_history.append(self._gp_scaled.tolist())
                self.gp_applied_history.append(self._gp_applied.tolist())
                self.gp_clip_active_history.append(self._gp_clip_active.tolist())
                self.gp_prediction_stride_history.append(int(self.gp_prediction_stride))
                self.gp_prediction_updated_this_tick_history.append(
                    int(self._gp_prediction_updated_this_tick)
                )
                self.gp_prediction_age_sec_history.append(
                    float(self.gp_prediction_age_sec)
                )
                self.gp_output_fresh_history.append(int(self.gp_output_fresh))
                self.future_trajectory_request_stride_history.append(
                    int(self.future_trajectory_request_stride)
                )
                self.future_trajectory_updated_this_tick_history.append(
                    int(self._future_trajectory_updated_this_tick)
                )
                # triple diagnostics 只在 recording block 中写入，避免 trajectory 启动前增加额外实时负担。
                self.gp_triple_raw_history.append(self.gp_triple_raw.tolist())
                self.gp_triple_weight_local_history.append(float(self.gp_triple_weight_local))
                self.gp_triple_weight_cloud_history.append(float(self.gp_triple_weight_cloud))
                self.gp_triple_weight_hist_history.append(float(self.gp_triple_weight_hist))
                self.gp_triple_available_history.append(int(self.gp_triple_available))
                self.gp_triple_used_fallback_history.append(int(self.gp_triple_used_fallback))
                self.gp_triple_fallback_source_code_history.append(
                    int(self.gp_triple_active_fallback_source_code)
                )
                self.gp_triple_weight_mode_code_history.append(
                    int(self.gp_triple_weight_mode_code)
                )
                self.gp_triple_hist_weight_cap_history.append(
                    float(self.gp_triple_hist_weight_cap)
                )
                self.gp_triple_rmse_local_history.append(float(self.gp_triple_rmse_local))
                self.gp_triple_rmse_cloud_history.append(float(self.gp_triple_rmse_cloud))
                self.gp_triple_rmse_hist_history.append(float(self.gp_triple_rmse_hist))
                self.gp_triple_dynamic_distance_ratio_history.append(
                    float(self.gp_triple_dynamic_distance_ratio)
                )
                self.gp_triple_dynamic_hist_penalty_history.append(
                    float(self.gp_triple_dynamic_hist_penalty)
                )
                self.gp_triple_dynamic_mode_code_history.append(
                    int(self.gp_triple_dynamic_mode_code)
                )
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
                self.gp_shadow_hist_pool_size_history.append(
                    int(self.gp_shadow_hist_pool_size)
                )
                self.gp_shadow_hist_k_used_history.append(
                    int(self.gp_shadow_hist_k_used)
                )
                self.gp_shadow_hist_nearest_distance_history.append(
                    float(self.gp_shadow_hist_nearest_distance)
                )
                self.gp_shadow_hist_mean_distance_topk_history.append(
                    float(self.gp_shadow_hist_mean_distance_topk)
                )
                self.hist_db_loaded_history.append(int(self.hist_db_loaded))
                self.hist_db_query_valid_history.append(int(self.hist_db_query_valid))
                self.hist_db_available_history.append(int(self.hist_db_available))
                self.hist_db_online_disabled_history.append(
                    int(self.hist_db_online_disabled)
                )
                self.hist_db_distance_pass_history.append(int(self.hist_db_distance_pass))
                self.hist_db_k_used_history.append(int(self.hist_db_k_used))
                self.hist_db_nearest_distance_history.append(
                    float(self.hist_db_nearest_distance)
                )
                self.hist_db_mean_topk_distance_history.append(
                    float(self.hist_db_mean_topk_distance)
                )
                self.hist_db_gated_source_code_history.append(
                    int(self.hist_db_gated_source_code)
                )
                self.hist_db_pred_history.append(self.hist_db_pred.tolist())
                self.hist_db_gated_pred_history.append(self.hist_db_gated_pred.tolist())
                self.hist_db_query_stride_history.append(
                    int(getattr(self, "gp_historical_db_query_stride", 1))
                )
                self.hist_db_query_updated_this_tick_history.append(
                    int(getattr(self, "hist_db_query_updated_this_tick", 0))
                )
                self.hist_db_query_reused_history.append(
                    int(getattr(self, "hist_db_query_reused", 0))
                )
                self.hist_db_query_counter_history.append(
                    int(getattr(self, "hist_db_query_counter", 0))
                )
                self.hist_db_preflight_phase_history.append(
                    str(getattr(self, "hist_db_preflight_phase", "disabled"))
                )
                self.hist_db_preflight_pass_history.append(
                    int(getattr(self, "hist_db_preflight_pass", 0))
                )
                self.hist_db_preflight_active_allowed_history.append(
                    int(getattr(self, "hist_db_preflight_active_allowed", 0))
                )
                self.hist_db_preflight_sample_count_history.append(
                    int(getattr(self, "hist_db_preflight_sample_count", 0))
                )
                self.hist_db_preflight_pass_ratio_history.append(
                    float(getattr(self, "hist_db_preflight_pass_ratio", 0.0))
                )
                self.hist_db_preflight_nearest_mean_history.append(
                    float(getattr(self, "hist_db_preflight_nearest_mean", 0.0))
                )
                self.hist_db_preflight_nearest_p95_history.append(
                    float(getattr(self, "hist_db_preflight_nearest_p95", 0.0))
                )
                self.hist_db_preflight_nearest_max_history.append(
                    float(getattr(self, "hist_db_preflight_nearest_max", 0.0))
                )
                self.hist_db_runtime_fallback_used_history.append(
                    int(getattr(self, "hist_db_runtime_fallback_used", 0))
                )
                self.hist_soft_valid_history.append(int(self.hist_soft_valid))
                self.hist_soft_nearest_distance_history.append(
                    float(self.hist_soft_nearest_distance)
                )
                self.hist_soft_raw_w_hist_history.append(
                    float(self.hist_soft_raw_w_hist)
                )
                self.hist_soft_norm_w_local_history.append(
                    float(self.hist_soft_norm_w_local)
                )
                self.hist_soft_norm_w_cloud_history.append(
                    float(self.hist_soft_norm_w_cloud)
                )
                self.hist_soft_norm_w_hist_history.append(
                    float(self.hist_soft_norm_w_hist)
                )
                self.hist_soft_pred_history.append(self.hist_soft_pred.tolist())
                self.hist_soft_delta_vs_local_cloud_history.append(
                    self.hist_soft_delta_vs_local_cloud.tolist()
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
                if csv_append_start is not None:
                    self._timing_add_ms(timing_row, "csv_append_ms", csv_append_start)

        except Exception as e:
            if timing_row is not None:
                timing_row["exception_flag"] = 1
            self._mark_effort_publish_skipped("exception")
            self.get_logger().error(f'Parameter error: {str(e)}')
        finally:
            self._finish_callback_timing(timing_row, callback_start_perf)

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

    def _publish_effort(self, msg):
        self.effort_publisher.publish(msg)
        publish_perf = time.perf_counter()
        if self._last_publish_perf is not None:
            self.last_effort_publish_gap_sec = publish_perf - self._last_publish_perf
            if self.last_effort_publish_gap_sec > self.max_effort_publish_gap_sec:
                self.max_effort_publish_gap_sec = self.last_effort_publish_gap_sec
            self.effort_publish_gap_window.append(self.last_effort_publish_gap_sec)
            if self.last_effort_publish_gap_sec > self.effort_gap_warn_sec:
                self.effort_publish_gap_warn_count += 1
        self._last_publish_perf = publish_perf
        self.publish_count += 1
        self.last_publish_time = self.get_clock().now()
        self._effort_published_this_tick = 1
        self._effort_publish_skip_reason = ""

    def _log_watchdog_warning(self, reason, elapsed_time):
        t_now_sec = self.get_clock().now().nanoseconds / 1e9
        if (t_now_sec - self.last_watchdog_warning_time) >= 1.0:
            self.get_logger().warn(
                f"[Watchdog] Controller active but callback skipped publishing effort! "
                f"Reason: {reason}, Elapsed: {elapsed_time:.3f}s, Publish Count: {self.publish_count}"
            )
            self.last_watchdog_warning_time = t_now_sec

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

    def _npz_scalar_to_python(self, value):
        arr = np.asarray(value)
        if arr.shape == ():
            item = arr.item()
            if isinstance(item, np.generic):
                return item.item()
            return item
        if arr.size == 1:
            item = arr.reshape(-1)[0]
            if isinstance(item, np.generic):
                return item.item()
            return item
        return arr.tolist()

    def _read_historical_db_metadata(self, db):
        metadata = {}
        metadata_keys = (
            "metadata_json",
            "source_files",
            "load_group",
            "load_gripper",
            "ee_load_model",
            "q_scale_recommended",
            "dq_scale_recommended",
        )

        for key in metadata_keys:
            if key not in db.files:
                continue
            try:
                value = self._npz_scalar_to_python(db[key])
            except Exception:
                continue
            metadata[key] = value
            if key == "source_files":
                try:
                    metadata["source_files_count"] = int(np.asarray(db[key]).size)
                except Exception:
                    pass

        raw_json = metadata.get("metadata_json")
        if raw_json:
            try:
                parsed = json.loads(str(raw_json))
                if isinstance(parsed, dict):
                    for key, value in parsed.items():
                        metadata.setdefault(key, value)
                    if "n_source_files" in parsed:
                        metadata.setdefault("source_files_count", parsed["n_source_files"])
            except json.JSONDecodeError as e:
                self.get_logger().warn(
                    "[GP Hist DB] metadata_json exists but cannot be parsed: "
                    f"{e}"
                )

        return metadata

    def _log_historical_db_metadata_summary(self, db_path):
        metadata = self.gp_historical_db_metadata
        if not metadata:
            return

        source_files_count = metadata.get("source_files_count")
        if source_files_count is None and "source_files" in metadata:
            try:
                source_files_count = len(metadata["source_files"])
            except TypeError:
                source_files_count = 1

        self.get_logger().info(
            "[GP Hist DB] Metadata summary: "
            f"path='{db_path}', "
            f"target_key='{self.gp_historical_db_target_key}', "
            f"source_files={source_files_count if source_files_count is not None else 'unknown'}, "
            f"load_group='{metadata.get('load_group', '')}', "
            f"load_gripper={metadata.get('load_gripper', '')}, "
            f"ee_load_model='{metadata.get('ee_load_model', '')}', "
            f"q_scale_recommended={metadata.get('q_scale_recommended', '')}, "
            f"dq_scale_recommended={metadata.get('dq_scale_recommended', '')}"
        )

    def _load_historical_residual_db(self):
        """Load a persistent residual DB once for shadow-only KNN queries."""
        self.gp_historical_db_loaded = False
        self.gp_historical_db_row_count = 0
        self.gp_historical_db_x_scaled = None
        self.gp_historical_db_y_residual = None
        self.gp_historical_db_metadata = {}
        self.gp_historical_db_target_key = ""

        if not self.gp_historical_db_enabled:
            self.get_logger().info("[GP Hist DB] Disabled; persistent DB will not be loaded.")
            return

        if not self.gp_historical_db_path:
            self.get_logger().warn(
                "[GP Hist DB] Enabled but gp_historical_db_path is empty; "
                "persistent DB remains unavailable."
            )
            return

        db_path = os.path.abspath(os.path.expanduser(self.gp_historical_db_path))
        try:
            if not os.path.isfile(db_path):
                raise FileNotFoundError(db_path)

            with np.load(db_path, allow_pickle=False) as db:
                target_key = "Y_residual" if "Y_residual" in db.files else "Y"
                missing = [name for name in ("X", target_key) if name not in db.files]
                if missing:
                    raise ValueError(f"missing arrays: {missing}")
                x = np.asarray(db["X"], dtype=float)
                y_residual = np.asarray(db[target_key], dtype=float)
                metadata = self._read_historical_db_metadata(db)

            if x.ndim != 2 or x.shape[1] != 14:
                raise ValueError(f"X must have shape (N, 14), got {x.shape}")
            if y_residual.ndim != 2 or y_residual.shape[1] != 7:
                raise ValueError(f"{target_key} must have shape (N, 7), got {y_residual.shape}")
            if x.shape[0] != y_residual.shape[0]:
                raise ValueError(
                    f"X and {target_key} row counts differ: "
                    f"{x.shape[0]} != {y_residual.shape[0]}"
                )
            if x.shape[0] <= 0:
                raise ValueError(f"X and {target_key} must contain at least one row")
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y_residual)):
                raise ValueError(f"X and {target_key} must contain only finite values")

            x_scaled = np.ascontiguousarray(
                x / self.gp_historical_db_feature_scale.reshape(1, -1),
                dtype=float,
            )
            if not np.all(np.isfinite(x_scaled)):
                raise ValueError("scaled X contains non-finite values")

            self.gp_historical_db_x_scaled = x_scaled
            self.gp_historical_db_y_residual = np.ascontiguousarray(
                y_residual,
                dtype=float,
            )
            self.gp_historical_db_row_count = int(x.shape[0])
            self.gp_historical_db_loaded = True
            self.gp_historical_db_target_key = target_key
            self.gp_historical_db_metadata = metadata
            self.get_logger().info(
                "[GP Hist DB] Loaded persistent residual DB: "
                f"path='{db_path}', rows={self.gp_historical_db_row_count}, "
                f"target_key='{self.gp_historical_db_target_key}'"
            )
            self._log_historical_db_metadata_summary(db_path)
        except Exception as e:
            self.gp_historical_db_loaded = False
            self.gp_historical_db_row_count = 0
            self.gp_historical_db_x_scaled = None
            self.gp_historical_db_y_residual = None
            self.gp_historical_db_metadata = {}
            self.gp_historical_db_target_key = ""
            self.get_logger().error(
                "[GP Hist DB] Failed to load persistent residual DB; "
                f"continuing with DB unavailable: {e}"
            )

    def _reset_gp_model_diagnostics(self):
        self.gp_model_local_loaded_count = 0
        self.gp_model_cloud_loaded_count = 0
        self.gp_model_cloud_fallback_count = 0
        self.gp_model_empty_or_prior_count = 0
        self.gp_model_cloud_uses_cloud_pkl = 0
        self.gp_model_cloud_uses_local_fallback = 0
        self.gp_model_local_files_by_joint = {}
        self.gp_model_cloud_files_by_joint = {}

    def _count_numeric_samples(self, value):
        if value is None:
            return None
        try:
            arr = np.asarray(value)
            if arr.size == 0:
                return 0
            if arr.dtype.kind in ("b", "i", "u", "f"):
                return int(np.sum(arr))
        except Exception:
            pass
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _model_sample_summary(self, model):
        summary = {}
        for attr in ("X_list", "Y_list", "y_list", "experts", "local_experts"):
            value = getattr(model, attr, None)
            if value is not None:
                try:
                    summary[f"{attr}_len"] = len(value)
                except TypeError:
                    summary[f"{attr}_len"] = "NA"

        sample_count = None
        for attr in ("localCount", "num_points", "N"):
            count = self._count_numeric_samples(getattr(model, attr, None))
            if count is not None:
                summary[attr] = count
                sample_count = count
                break

        if sample_count is None:
            for attr in ("experts", "local_experts"):
                experts = getattr(model, attr, None)
                if experts is None:
                    continue
                total = 0
                detected = False
                try:
                    for expert in experts:
                        for count_attr in ("localCount", "num_points", "N"):
                            count = self._count_numeric_samples(
                                getattr(expert, count_attr, None)
                            )
                            if count is not None:
                                total += count
                                detected = True
                                break
                except TypeError:
                    detected = False
                if detected:
                    summary[f"{attr}_sample_total"] = total
                    sample_count = total
                    break

        x_list_len = summary.get("X_list_len")
        empty_or_prior = (
            x_list_len == 0
            or sample_count is None
            or sample_count <= 0
        )
        return sample_count, summary, empty_or_prior

    def _format_gp_model_summary(self, summary):
        if not summary:
            return "no sample attributes detected"
        return ", ".join(f"{key}={value}" for key, value in sorted(summary.items()))

    def _apply_gp_model_runtime_cfg(self, model, joint, cfg):
        try:
            # 只有在模型里确实有这些属性时才改，避免旧版本崩溃。
            if hasattr(model, "max_data_per_expert"):
                model.max_data_per_expert = int(cfg["max_data_per_expert"])
            if hasattr(model, "nearest_k"):
                model.nearest_k = int(cfg["nearest_k"])
            if hasattr(model, "max_experts"):
                model.max_experts = int(cfg["max_experts"])
            if hasattr(model, "timescale"):
                model.timescale = float(cfg["timescale"])
        except Exception as e:
            self.get_logger().warn(
                f"[GP] joint{joint}: override model params failed: {e}"
            )

    def _load_gp_model_pack(self, path, joint, role, cfg):
        abs_path = os.path.abspath(path)
        self.get_logger().debug(f"[GP] 尝试加载 {role} 模型: {abs_path}")

        if not os.path.isfile(path):
            self.get_logger().warn(f"[GP] {role} model file not found: {abs_path}")
            return None

        try:
            with open(path, "rb") as f:
                pack = pickle.load(f)

            model = pack["model"]
            stats = pack["stats"]   # (Xm, Xs, Ym, Ys)
            Xm, Xs, Ym, Ys = stats
            x_dim = int(len(Xm))    # 自动推断 14 或 21 维

            self._apply_gp_model_runtime_cfg(model, joint, cfg)
            sample_count, sample_summary, empty_or_prior = self._model_sample_summary(model)
            if empty_or_prior:
                self.gp_model_empty_or_prior_count += 1
                self.get_logger().warn(
                    "[GP Sanity] "
                    f"joint{joint} {role} frozen model appears empty/prior-only; "
                    "non-online prediction may be constant; "
                    f"{self._format_gp_model_summary(sample_summary)}"
                )

            self.get_logger().debug(
                f"[GP] joint{joint} {role} loaded: x_dim={x_dim}, "
                f"file='{os.path.basename(path)}', "
                f"sample_count={sample_count if sample_count is not None else 'unknown'}, "
                f"max_data_per_expert={getattr(model, 'max_data_per_expert', 'NA')}, "
                f"nearest_k={getattr(model, 'nearest_k', 'NA')}, "
                f"max_experts={getattr(model, 'max_experts', 'NA')}, "
                f"timescale={getattr(model, 'timescale', 'NA')}"
            )

            return {
                "model": model,
                "stats": stats,
                "x_dim": x_dim,
                "file_basename": os.path.basename(path),
                "sample_count": sample_count,
                "sample_summary": sample_summary,
                "empty_or_prior": empty_or_prior,
            }

        except Exception as e:
            self.get_logger().error(f"[GP] fail loading {abs_path}: {e}")
            return None

    def _format_gp_model_files_by_joint(self, files_by_joint):
        parts = []
        for joint in range(1, 8):
            parts.append(f"joint{joint}:{files_by_joint.get(joint, 'missing')}")
        return ", ".join(parts)

    def _summarize_loaded_gp_models(self, dir_path):
        self.get_logger().info(
            "[GP] Model loading summary: "
            f"gp_model_dir='{dir_path}', "
            f"local_model_loaded_count={self.gp_model_local_loaded_count}, "
            f"cloud_model_loaded_count={self.gp_model_cloud_loaded_count}, "
            f"cloud_local_fallback_count={self.gp_model_cloud_fallback_count}, "
            f"empty_or_prior_model_count={self.gp_model_empty_or_prior_count}, "
            f"gp_online_update_enabled={self.gp_online_update_enabled}, "
            f"gp_compensation_source='{self.gp_compensation_source}', "
            f"delay_steps={self.delay_steps}"
        )
        self.get_logger().info(
            "[GP] Local model file basenames per joint: "
            f"{self._format_gp_model_files_by_joint(self.gp_model_local_files_by_joint)}"
        )
        self.get_logger().info(
            "[GP] Cloud-like delayed GP model file basenames per joint: "
            f"{self._format_gp_model_files_by_joint(self.gp_model_cloud_files_by_joint)}"
        )

    def _load_gp_models(self, dir_path="./new_structure/gp/gp_models"):
        """加载离线训练好的每关节GP，支持高维输入（14或21）"""

        if not self._ensure_skygp_import():
            self.get_logger().error("[GP] skygp import failed; pickle loading will likely fail.")

        cwd = os.getcwd()
        abs_dir = os.path.abspath(dir_path)
        self.get_logger().info(f"[GP] 当前工作目录: {cwd}")
        self.get_logger().info(f"[GP] 模型目录绝对路径: {abs_dir}")

        self._reset_gp_model_diagnostics()

        local_joint_cfg = {
            "default": dict(
                max_data_per_expert=25,
                nearest_k=1,
                max_experts=1,
                timescale=0.03,
            ),
            6: dict(
                max_data_per_expert=25,
                nearest_k=1,
                max_experts=1,
                timescale=0.05,
            ),
        }
        cloud_joint_cfg = {
            "default": dict(
                max_data_per_expert=50,
                nearest_k=2,
                max_experts=10,
                timescale=0.03,
            ),
            6: dict(
                max_data_per_expert=50,
                nearest_k=2,
                max_experts=10,
                timescale=0.05,
            ),
        }

        self.gp_models_small = {}
        self.gp_models_big = {}

        for j in range(1, 8):
            local_path = os.path.join(dir_path, f"joint{j}_local.pkl")
            pack = self._load_gp_model_pack(
                local_path,
                j,
                "local",
                local_joint_cfg.get(j, local_joint_cfg["default"])
            )
            if pack is None:
                continue
            self.gp_models_small[j] = pack
            self.gp_model_local_loaded_count += 1
            self.gp_model_local_files_by_joint[j] = pack["file_basename"]

        for j in range(1, 8):
            cloud_path = os.path.join(dir_path, f"joint{j}_cloud.pkl")
            local_fallback_path = os.path.join(dir_path, f"joint{j}_local.pkl")
            model_path = cloud_path

            if not os.path.isfile(cloud_path):
                if not os.path.isfile(local_fallback_path):
                    self.get_logger().warn(
                        "[GP] cloud-like model file not found and no local fallback "
                        f"exists for joint{j}: {os.path.abspath(cloud_path)}"
                    )
                    continue
                model_path = local_fallback_path
                self.gp_model_cloud_fallback_count += 1
                self.gp_model_cloud_uses_local_fallback = 1
                self.get_logger().warn(
                    "[GP] "
                    f"joint{j}_cloud.pkl not found; using joint{j}_local.pkl as "
                    "cloud-like local fallback. This is not an actual remote cloud "
                    "model and must not be treated as a silent cloud model load."
                )
            else:
                self.gp_model_cloud_uses_cloud_pkl = 1

            pack = self._load_gp_model_pack(
                model_path,
                j,
                "cloud-like",
                cloud_joint_cfg.get(j, cloud_joint_cfg["default"])
            )
            if pack is None:
                continue
            self.gp_models_big[j] = pack
            self.gp_model_cloud_loaded_count += 1
            self.gp_model_cloud_files_by_joint[j] = pack["file_basename"]

        self.gp_ready = (
            self.gp_model_local_loaded_count > 0
            or self.gp_model_cloud_loaded_count > 0
        )
        self._summarize_loaded_gp_models(dir_path)
        self.get_logger().info(
            f"[GP] 共加载 local={self.gp_model_local_loaded_count}, "
            f"cloud-like={self.gp_model_cloud_loaded_count} 个模型，ready={self.gp_ready}"
        )

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

    def _build_gp_shadow_feature(self, q, dq):
        """Build the same 14D [q, dq] feature used by the active GP predictor."""
        return np.concatenate([q, dq]).astype(np.float32)

    def _historical_db_source_requested(self):
        return (
            bool(getattr(self, "gp_compensation_enabled", False))
            and bool(getattr(self, "gp_prediction_enabled", False))
            and getattr(self, "gp_compensation_source", "") == "hist_db"
        )

    def _reset_historical_db_preflight_state(self):
        self.hist_db_preflight_enabled = int(
            bool(self.gp_historical_db_preflight_enabled)
        )
        self.hist_db_preflight_required = int(
            bool(self.gp_historical_db_preflight_required)
        )
        self.hist_db_preflight_mode = self.gp_historical_db_preflight_mode
        self.hist_db_preflight_phase = (
            "idle" if self.gp_historical_db_preflight_enabled else "disabled"
        )
        self.hist_db_preflight_pass = int(
            not self.gp_historical_db_preflight_enabled
        )
        self.hist_db_preflight_active_allowed = int(
            not self.gp_historical_db_preflight_enabled
            or not self.gp_historical_db_preflight_required
        )
        self.hist_db_preflight_sample_count = 0
        self.hist_db_preflight_pass_ratio = 0.0
        self.hist_db_preflight_nearest_mean = 0.0
        self.hist_db_preflight_nearest_p95 = 0.0
        self.hist_db_preflight_nearest_max = 0.0
        self.hist_db_runtime_fallback_used = 0
        self._hist_db_preflight_start_time = None
        self._hist_db_preflight_distances = []
        self._hist_db_preflight_pass_count = 0
        self._hist_db_preflight_distance_sum = 0.0
        self._hist_db_preflight_probe_logged = False
        self._hist_db_preflight_final_logged = False
        self._hist_db_runtime_diag_count = 0

    def _historical_db_result_passes_preflight_distance(self, result):
        try:
            nearest_distance = float(result["nearest_distance"])
            return bool(
                result["loaded"]
                and result["query_valid"]
                and not result.get("online_disabled", 0)
                and result["k_used"] > 0
                and np.isfinite(nearest_distance)
                and nearest_distance <= self.gp_historical_db_preflight_max_distance
            )
        except (KeyError, TypeError, ValueError):
            return False

    def _historical_db_result_has_finite_distance(self, result):
        try:
            nearest_distance = float(result["nearest_distance"])
            return bool(
                result["loaded"]
                and result["query_valid"]
                and not result.get("online_disabled", 0)
                and result["k_used"] > 0
                and np.isfinite(nearest_distance)
            )
        except (KeyError, TypeError, ValueError):
            return False

    def _historical_db_preflight_fail_reason(self, result):
        if not self.gp_historical_db_loaded:
            return "db_not_loaded"
        if self.gp_historical_db_row_count <= 0:
            return "db_empty"
        if not result.get("query_valid", 0):
            return "query_not_valid"
        if result.get("online_disabled", 0):
            return "hist_db_disabled_while_online_update_enabled"
        if result.get("k_used", 0) <= 0:
            return "knn_unavailable"
        try:
            nearest_distance = float(result["nearest_distance"])
        except (KeyError, TypeError, ValueError):
            return "nearest_distance_invalid"
        if not np.isfinite(nearest_distance):
            return "nearest_distance_invalid"
        if nearest_distance > self.gp_historical_db_preflight_max_distance:
            return (
                "nearest_distance_exceeds_preflight_max: "
                f"{nearest_distance:.6f} > "
                f"{self.gp_historical_db_preflight_max_distance:.6f}"
            )
        return "preflight_failed"

    def _log_historical_db_preflight_probe_once(self, result):
        if self._hist_db_preflight_probe_logged:
            return
        self._hist_db_preflight_probe_logged = True
        distance_pass = self._historical_db_result_passes_preflight_distance(result)
        message = (
            "[GP Hist DB] Single-state preflight probe: "
            f"source='{self.gp_compensation_source}', "
            f"loaded={int(result.get('loaded', 0))}, "
            f"rows={self.gp_historical_db_row_count}, "
            f"k={result.get('k_used', 0)}, "
            f"nearest_distance={float(result.get('nearest_distance', 0.0)):.6f}, "
            f"max_distance={self.gp_historical_db_preflight_max_distance}, "
            f"distance_pass={int(distance_pass)}, "
            f"required={int(self.gp_historical_db_preflight_required)}, "
            f"fallback_source='{self.gp_historical_db_fallback_source}'"
        )
        if distance_pass:
            self.get_logger().info(message)
        elif (
            self.gp_historical_db_preflight_required
            or self.gp_disable_silent_hist_fallback
        ):
            self.get_logger().error(message)
        else:
            self.get_logger().warn(message)

    def _log_historical_db_preflight_final(self, passed, reason):
        if self._hist_db_preflight_final_logged:
            return
        self._hist_db_preflight_final_logged = True
        self._refresh_historical_db_preflight_full_stats()
        message = (
            "[GP Hist DB] Preflight "
            f"{'PASS' if passed else 'FAIL'}: "
            f"source='{self.gp_compensation_source}', "
            f"phase='{self.hist_db_preflight_phase}', "
            f"samples={self.hist_db_preflight_sample_count}, "
            f"pass_ratio={self.hist_db_preflight_pass_ratio:.6f}, "
            f"nearest_mean={self.hist_db_preflight_nearest_mean:.6f}, "
            f"nearest_p95={self.hist_db_preflight_nearest_p95:.6f}, "
            f"nearest_max={self.hist_db_preflight_nearest_max:.6f}, "
            f"active_allowed={self.hist_db_preflight_active_allowed}, "
            f"required={int(self.gp_historical_db_preflight_required)}, "
            f"disable_silent_fallback={int(self.gp_disable_silent_hist_fallback)}, "
            f"reason='{reason}'"
        )
        if passed:
            self.get_logger().info(message)
        elif (
            self.gp_historical_db_preflight_required
            or self.gp_disable_silent_hist_fallback
        ):
            self.get_logger().error(message)
        else:
            self.get_logger().warn(message)

    def _set_historical_db_preflight_failed(self, reason):
        self.hist_db_preflight_phase = "failed"
        self.hist_db_preflight_pass = 0
        self.hist_db_preflight_active_allowed = int(
            not self.gp_historical_db_preflight_required
        )
        self._log_historical_db_preflight_final(False, reason)

    def _set_historical_db_preflight_passed(self, reason):
        self.hist_db_preflight_phase = "passed"
        self.hist_db_preflight_pass = 1
        self.hist_db_preflight_active_allowed = 1
        self._log_historical_db_preflight_final(True, reason)

    def _update_historical_db_preflight_stats(self, nearest_distance):
        nearest_distance = float(nearest_distance)
        self._hist_db_preflight_distances.append(nearest_distance)
        self._hist_db_preflight_distance_sum += nearest_distance
        if nearest_distance <= self.gp_historical_db_preflight_max_distance:
            self._hist_db_preflight_pass_count += 1

        self.hist_db_preflight_sample_count = len(self._hist_db_preflight_distances)
        if self.hist_db_preflight_sample_count <= 0:
            self.hist_db_preflight_pass_ratio = 0.0
            self.hist_db_preflight_nearest_mean = 0.0
            self.hist_db_preflight_nearest_p95 = 0.0
            self.hist_db_preflight_nearest_max = 0.0
            return

        self.hist_db_preflight_pass_ratio = (
            float(self._hist_db_preflight_pass_count)
            / float(self.hist_db_preflight_sample_count)
        )
        self.hist_db_preflight_nearest_mean = (
            self._hist_db_preflight_distance_sum
            / float(self.hist_db_preflight_sample_count)
        )
        self.hist_db_preflight_nearest_max = max(
            self.hist_db_preflight_nearest_max,
            nearest_distance,
        )

    def _refresh_historical_db_preflight_full_stats(self):
        distances = np.asarray(self._hist_db_preflight_distances, dtype=float)
        self.hist_db_preflight_sample_count = int(distances.size)
        if distances.size <= 0:
            self.hist_db_preflight_pass_ratio = 0.0
            self.hist_db_preflight_nearest_mean = 0.0
            self.hist_db_preflight_nearest_p95 = 0.0
            self.hist_db_preflight_nearest_max = 0.0
            return

        self.hist_db_preflight_pass_ratio = (
            float(self._hist_db_preflight_pass_count) / float(distances.size)
        )
        self.hist_db_preflight_nearest_mean = float(
            self._hist_db_preflight_distance_sum / float(distances.size)
        )
        self.hist_db_preflight_nearest_p95 = float(np.percentile(distances, 95.0))
        self.hist_db_preflight_nearest_max = float(np.max(distances))

    def _historical_db_preflight_elapsed_sec(self, t_now):
        if self._hist_db_preflight_start_time is None:
            if t_now is None:
                return (
                    self.hist_db_preflight_sample_count
                    / max(float(self.control_frequency), 1e-6)
                )
            self._hist_db_preflight_start_time = t_now
            return 0.0
        if t_now is None:
            return (
                self.hist_db_preflight_sample_count
                / max(float(self.control_frequency), 1e-6)
            )
        try:
            elapsed = (
                t_now - self._hist_db_preflight_start_time
            ).nanoseconds / 1e9
        except Exception:
            elapsed = (
                self.hist_db_preflight_sample_count
                / max(float(self.control_frequency), 1e-6)
            )
        if not np.isfinite(elapsed) or elapsed < 0.0:
            return 0.0
        return float(elapsed)

    def _update_historical_db_preflight_state(self, t_now, result, fresh_query):
        if not self._historical_db_source_requested():
            self.hist_db_preflight_phase = "disabled"
            self.hist_db_preflight_active_allowed = 0
            return

        if not self.gp_historical_db_preflight_enabled:
            self.hist_db_preflight_phase = "disabled"
            self.hist_db_preflight_pass = 1
            self.hist_db_preflight_active_allowed = 1
            return

        if self.hist_db_preflight_phase in ("passed", "failed"):
            return

        self._log_historical_db_preflight_probe_once(result)
        single_pass = self._historical_db_result_passes_preflight_distance(result)

        if self.gp_historical_db_preflight_mode == "single":
            if single_pass:
                self._set_historical_db_preflight_passed("single_state_pass")
            else:
                self._set_historical_db_preflight_failed(
                    self._historical_db_preflight_fail_reason(result)
                )
            return

        if (
            self.gp_historical_db_preflight_mode == "single_and_segment"
            and not single_pass
        ):
            self._set_historical_db_preflight_failed(
                self._historical_db_preflight_fail_reason(result)
            )
            return

        self.hist_db_preflight_phase = "segment_collecting"
        self.hist_db_preflight_active_allowed = int(
            not self.gp_historical_db_preflight_required
        )
        elapsed_sec = self._historical_db_preflight_elapsed_sec(t_now)

        if fresh_query and self._historical_db_result_has_finite_distance(result):
            self._update_historical_db_preflight_stats(
                float(result["nearest_distance"])
            )

        if elapsed_sec < self.gp_historical_db_preflight_duration_sec:
            return
        if (
            self.hist_db_preflight_sample_count
            < self.gp_historical_db_preflight_min_samples
        ):
            self._refresh_historical_db_preflight_full_stats()
            self._set_historical_db_preflight_failed("insufficient_preflight_samples")
            return

        self._refresh_historical_db_preflight_full_stats()
        pass_ratio_ok = (
            self.hist_db_preflight_pass_ratio
            >= self.gp_historical_db_preflight_min_pass_ratio
        )
        p95_ok = (
            self.hist_db_preflight_nearest_p95
            <= self.gp_historical_db_preflight_p95_max_distance
        )
        max_ok = (
            self.hist_db_preflight_nearest_max
            <= self.gp_historical_db_preflight_max_distance
        )
        if pass_ratio_ok and p95_ok and max_ok:
            self._set_historical_db_preflight_passed("segment_pass")
        else:
            reason_parts = []
            if not pass_ratio_ok:
                reason_parts.append("pass_ratio_below_threshold")
            if not p95_ok:
                reason_parts.append("p95_distance_above_threshold")
            if not max_ok:
                reason_parts.append("max_distance_above_threshold")
            self._set_historical_db_preflight_failed(";".join(reason_parts))

    def _historical_db_active_allowed(self):
        if not self._historical_db_source_requested():
            return False
        if not self.gp_historical_db_preflight_required:
            return True
        if not self.gp_historical_db_preflight_enabled:
            return True
        return bool(self.hist_db_preflight_active_allowed)

    def _maybe_log_hist_db_runtime_diag(self, selected_source):
        if not self._historical_db_source_requested():
            return
        if self._hist_db_runtime_diag_count >= self.gp_historical_db_preflight_log_first_n:
            return
        if self.hist_db_preflight_phase in ("idle", "segment_collecting"):
            return
        if (
            not self.hist_db_preflight_active_allowed
            and not self.hist_db_runtime_fallback_used
        ):
            return

        self._hist_db_runtime_diag_count += 1
        message = (
            "[GP Hist DB] Runtime active diagnostic: "
            f"source='{self.gp_compensation_source}', "
            f"available={int(self.hist_db_available)}, "
            f"distance={float(self.hist_db_nearest_distance):.6f}, "
            f"distance_pass={int(self.hist_db_distance_pass)}, "
            f"preflight_pass={int(self.hist_db_preflight_pass)}, "
            f"active_allowed={int(self.hist_db_preflight_active_allowed)}, "
            f"fallback_used={int(self.hist_db_runtime_fallback_used)}, "
            f"selected_source='{selected_source}'"
        )
        if self.hist_db_runtime_fallback_used:
            if (
                self.gp_historical_db_preflight_required
                or self.gp_disable_silent_hist_fallback
            ):
                self.get_logger().error(message)
            else:
                self.get_logger().warn(message)
        else:
            self.get_logger().info(message)

    def _get_historical_residual_db_fallback_candidate(self):
        zero = np.zeros(7, dtype=float)
        source = self.gp_historical_db_fallback_source
        if source == "none":
            return zero, 0

        candidate = {
            "local": self.y_hat_local,
            "cloud": self.y_hat_cloud,
            "combined": self.y_hat_combined,
        }.get(source)
        try:
            candidate_arr = np.asarray(candidate, dtype=float)
        except (TypeError, ValueError):
            return zero, 0

        if candidate_arr.shape != (7,) or not np.all(np.isfinite(candidate_arr)):
            return zero, 0
        return candidate_arr.copy(), self.gp_historical_db_fallback_source_code

    def _new_historical_residual_db_shadow_result(self):
        fallback_prediction, fallback_source_code = (
            self._get_historical_residual_db_fallback_candidate()
        )
        return {
            "loaded": int(bool(self.gp_historical_db_loaded)),
            "query_valid": 0,
            "available": 0,
            "online_disabled": int(
                bool(
                    self.gp_historical_db_disable_when_online_update
                    and self.gp_online_update_enabled
                )
            ),
            "distance_pass": 0,
            "k_used": 0,
            "nearest_distance": 0.0,
            "mean_topk_distance": 0.0,
            "prediction": np.zeros(7, dtype=float),
            "gated_prediction": fallback_prediction,
            "gated_source_code": int(fallback_source_code),
        }

    def _query_historical_residual_db_shadow(self, q, dq):
        """Query the persistent residual DB without affecting active torque."""
        result = self._new_historical_residual_db_shadow_result()

        try:
            q_arr = np.asarray(q, dtype=float)
            dq_arr = np.asarray(dq, dtype=float)
        except (TypeError, ValueError):
            return result
        if (
            q_arr.shape != (7,)
            or dq_arr.shape != (7,)
            or not np.all(np.isfinite(q_arr))
            or not np.all(np.isfinite(dq_arr))
        ):
            return result

        x_query = np.concatenate([q_arr, dq_arr])
        x_query_scaled = np.ascontiguousarray(
            x_query / self.gp_historical_db_feature_scale,
            dtype=float,
        )
        if not np.all(np.isfinite(x_query_scaled)):
            return result
        result["query_valid"] = 1

        if not self.gp_historical_db_enabled or not self.gp_historical_db_loaded:
            return result
        if (
            self.gp_historical_db_x_scaled is None
            or self.gp_historical_db_y_residual is None
            or self.gp_historical_db_row_count <= 0
        ):
            return result

        try:
            with np.errstate(over="ignore", invalid="ignore"):
                delta = self.gp_historical_db_x_scaled - x_query_scaled.reshape(1, -1)
                distance_sq = np.einsum("ij,ij->i", delta, delta)
            if (
                distance_sq.shape != (self.gp_historical_db_row_count,)
                or not np.all(np.isfinite(distance_sq))
                or np.any(distance_sq < 0.0)
            ):
                return result

            k_used = min(self.gp_historical_db_k, self.gp_historical_db_row_count)
            nearest_indices = np.argpartition(distance_sq, kth=k_used - 1)[:k_used]
            nearest_indices = nearest_indices[np.argsort(distance_sq[nearest_indices])]
            nearest_distances = np.sqrt(distance_sq[nearest_indices])
            prediction = np.mean(
                self.gp_historical_db_y_residual[nearest_indices],
                axis=0,
            )
        except (TypeError, ValueError, IndexError, FloatingPointError):
            return result

        result["k_used"] = int(k_used)
        result["nearest_distance"] = float(nearest_distances[0])
        result["mean_topk_distance"] = float(np.mean(nearest_distances))
        result["distance_pass"] = int(
            result["nearest_distance"] <= self.gp_historical_db_max_distance
        )
        if prediction.shape != (7,) or not np.all(np.isfinite(prediction)):
            return result

        result["prediction"] = prediction.copy()
        result["available"] = int(
            bool(
                result["loaded"]
                and result["query_valid"]
                and result["distance_pass"]
                and not result["online_disabled"]
            )
        )
        if result["available"]:
            result["gated_prediction"] = prediction.copy()
            result["gated_source_code"] = 4
        return result

    def _reset_historical_residual_db_shadow_state(self):
        zero = np.zeros(7, dtype=float)
        self.hist_db_loaded = int(bool(self.gp_historical_db_loaded))
        self.hist_db_query_valid = 0
        self.hist_db_available = 0
        self.hist_db_online_disabled = int(
            bool(
                self.gp_historical_db_disable_when_online_update
                and self.gp_online_update_enabled
            )
        )
        self.hist_db_distance_pass = 0
        self.hist_db_k_used = 0
        self.hist_db_nearest_distance = 0.0
        self.hist_db_mean_topk_distance = 0.0
        self.hist_db_pred = zero.copy()
        self.hist_db_gated_pred = zero.copy()
        self.hist_db_gated_source_code = 0
        self.hist_db_query_updated_this_tick = 0
        self._reset_historical_soft_shadow_state()

    def _update_historical_residual_db_shadow_state(self, q, dq, t_now=None):
        # 为降低真机 callback 负载，可按 stride 降频查询 hist DB。
        # 默认 gp_historical_db_query_stride=1 时，每周期查询，保持原始行为。
        # stride>1 时，中间 callback 复用上一帧 gated prediction 和 gate 诊断。
        try:
            q_arr = np.asarray(q, dtype=float)
            dq_arr = np.asarray(dq, dtype=float)
            input_valid = (
                q_arr.shape == (7,)
                and dq_arr.shape == (7,)
                and np.all(np.isfinite(q_arr))
                and np.all(np.isfinite(dq_arr))
            )
        except Exception:
            input_valid = False

        query_stride = max(1, int(getattr(self, "gp_historical_db_query_stride", 1)))
        should_query = (
            query_stride <= 1
            or self._hist_db_last_query_result is None
            or not input_valid
            or (self._hist_db_query_counter % query_stride == 0)
        )

        self.hist_db_query_reused = int(not should_query)
        self.hist_db_query_updated_this_tick = int(should_query)

        if should_query:
            result = self._query_historical_residual_db_shadow(q, dq)
            self._hist_db_last_query_result = dict(result)
        else:
            result = dict(self._hist_db_last_query_result)

        self._hist_db_query_counter += 1
        self.hist_db_query_counter = int(self._hist_db_query_counter)
        self.hist_db_loaded = int(result["loaded"])
        self.hist_db_query_valid = int(result["query_valid"])
        self.hist_db_available = int(result["available"])
        self.hist_db_online_disabled = int(result["online_disabled"])
        self.hist_db_distance_pass = int(result["distance_pass"])
        self.hist_db_k_used = int(result["k_used"])
        self.hist_db_nearest_distance = float(result["nearest_distance"])
        self.hist_db_mean_topk_distance = float(result["mean_topk_distance"])
        self.hist_db_pred = np.asarray(result["prediction"], dtype=float).copy()
        self.hist_db_gated_pred = np.asarray(
            result["gated_prediction"],
            dtype=float,
        ).copy()
        self.hist_db_gated_source_code = int(result["gated_source_code"])
        self._update_historical_db_preflight_state(t_now, result, bool(should_query))
        self._update_historical_soft_shadow_state()

    def _new_historical_soft_shadow_result(self):
        zero = np.zeros(7, dtype=float)
        return {
            "valid": 0,
            "nearest_distance": 0.0,
            "raw_w_hist": 0.0,
            "norm_w_local": 0.0,
            "norm_w_cloud": 0.0,
            "norm_w_hist": 0.0,
            "prediction": zero.copy(),
            "delta_vs_local_cloud": zero.copy(),
        }

    def _compute_historical_soft_shadow(self):
        """Compute persistent historical soft fusion for shadow logging only."""
        result = self._new_historical_soft_shadow_result()
        if not self.gp_historical_soft_shadow_enabled:
            return result

        # 这是 shadow-only evaluator，不写入 active torque，也不改变 tau_final。
        # DB hard gate fail 时必须 fail closed，避免 fallback prediction 被误记为
        # historical soft prediction。
        if (
            not self.hist_db_loaded
            or not self.hist_db_query_valid
            or not self.hist_db_available
            or not self.hist_db_distance_pass
            or self.hist_db_k_used <= 0
            or self.hist_db_gated_source_code != 4
        ):
            return result

        try:
            nearest_distance = float(self.hist_db_nearest_distance)
            hist_db_pred = np.asarray(self.hist_db_pred, dtype=float)
            hist_db_gated_pred = np.asarray(self.hist_db_gated_pred, dtype=float)
            y_hat_local = np.asarray(self.y_hat_local, dtype=float)
            y_hat_cloud = np.asarray(self.y_hat_cloud, dtype=float)
        except (TypeError, ValueError):
            return result

        inputs = (hist_db_pred, hist_db_gated_pred, y_hat_local, y_hat_cloud)
        if (
            not np.isfinite(nearest_distance)
            or nearest_distance < 0.0
            or any(value.shape != (7,) for value in inputs)
            or any(not np.all(np.isfinite(value)) for value in inputs)
        ):
            # finite check 失败时 fail closed，避免无效 shadow 数据被误判为可用。
            return result

        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            raw_w_hist = float(np.exp(-self.gp_historical_soft_alpha * nearest_distance))
        if nearest_distance > self.gp_historical_soft_distance_threshold:
            raw_w_hist = 0.0

        # online update 场景下 historical 与当前模型分布可能不一致，因此强降权。
        if self.gp_online_update_enabled:
            raw_w_hist *= self.gp_historical_soft_online_scale
        else:
            raw_w_hist *= self.gp_historical_soft_non_online_scale

        w_local_base = 0.5
        w_cloud_base = 0.5
        sum_w = w_local_base + w_cloud_base + raw_w_hist
        if not np.isfinite(raw_w_hist) or not np.isfinite(sum_w) or sum_w <= 0.0:
            return result

        norm_w_local = w_local_base / sum_w
        norm_w_cloud = w_cloud_base / sum_w
        norm_w_hist = raw_w_hist / sum_w
        local_cloud_shadow = 0.5 * y_hat_local + 0.5 * y_hat_cloud
        prediction = (
            norm_w_local * y_hat_local
            + norm_w_cloud * y_hat_cloud
            + norm_w_hist * hist_db_gated_pred
        )
        delta_vs_local_cloud = prediction - local_cloud_shadow

        scalar_outputs = (
            norm_w_local,
            norm_w_cloud,
            norm_w_hist,
        )
        vector_outputs = (local_cloud_shadow, prediction, delta_vs_local_cloud)
        if (
            any(not np.isfinite(value) for value in scalar_outputs)
            or any(not np.all(np.isfinite(value)) for value in vector_outputs)
        ):
            # 输出 finite check 同样 fail closed；这些值永远不进入 active torque。
            return result

        result.update({
            "valid": 1,
            "nearest_distance": nearest_distance,
            "raw_w_hist": raw_w_hist,
            "norm_w_local": norm_w_local,
            "norm_w_cloud": norm_w_cloud,
            "norm_w_hist": norm_w_hist,
            "prediction": prediction.copy(),
            "delta_vs_local_cloud": delta_vs_local_cloud.copy(),
        })
        return result

    def _reset_historical_soft_shadow_state(self):
        result = self._new_historical_soft_shadow_result()
        self.hist_soft_valid = int(result["valid"])
        self.hist_soft_nearest_distance = float(result["nearest_distance"])
        self.hist_soft_raw_w_hist = float(result["raw_w_hist"])
        self.hist_soft_norm_w_local = float(result["norm_w_local"])
        self.hist_soft_norm_w_cloud = float(result["norm_w_cloud"])
        self.hist_soft_norm_w_hist = float(result["norm_w_hist"])
        self.hist_soft_pred = np.asarray(result["prediction"], dtype=float).copy()
        self.hist_soft_delta_vs_local_cloud = np.asarray(
            result["delta_vs_local_cloud"],
            dtype=float,
        ).copy()

    def _update_historical_soft_shadow_state(self):
        result = self._compute_historical_soft_shadow()
        self.hist_soft_valid = int(result["valid"])
        self.hist_soft_nearest_distance = float(result["nearest_distance"])
        self.hist_soft_raw_w_hist = float(result["raw_w_hist"])
        self.hist_soft_norm_w_local = float(result["norm_w_local"])
        self.hist_soft_norm_w_cloud = float(result["norm_w_cloud"])
        self.hist_soft_norm_w_hist = float(result["norm_w_hist"])
        self.hist_soft_pred = np.asarray(result["prediction"], dtype=float).copy()
        self.hist_soft_delta_vs_local_cloud = np.asarray(
            result["delta_vs_local_cloud"],
            dtype=float,
        ).copy()

    def _gp_triple_default_weights(self):
        return np.array([0.10, 0.20, 0.70], dtype=float)

    def _normalize_gp_triple_weights(self, weights):
        default_weights = self._gp_triple_default_weights()
        try:
            weights_arr = np.asarray(weights, dtype=float)
            if (
                weights_arr.shape != (3,)
                or not np.all(np.isfinite(weights_arr))
                or np.any(weights_arr < 0.0)
            ):
                raise ValueError
            total = float(np.sum(weights_arr))
            if not np.isfinite(total) or total <= 0.0:
                raise ValueError
        except (TypeError, ValueError):
            weights_arr = default_weights
            total = float(np.sum(weights_arr))

        return weights_arr / total

    def _cap_gp_triple_hist_weight(self, weights):
        weights_arr = self._normalize_gp_triple_weights(weights)
        cap = float(self.gp_triple_hist_weight_cap)
        if not np.isfinite(cap) or cap < 0.0:
            cap = 0.70

        if cap < 1.0 and weights_arr[2] > cap:
            remaining = max(0.0, 1.0 - cap)
            local_cloud = weights_arr[:2].copy()
            local_cloud_total = float(np.sum(local_cloud))
            if not np.isfinite(local_cloud_total) or local_cloud_total <= 0.0:
                local_cloud = np.array([
                    self.gp_triple_min_weight_local,
                    self.gp_triple_min_weight_cloud,
                ], dtype=float)
                local_cloud_total = float(np.sum(local_cloud))
            if not np.isfinite(local_cloud_total) or local_cloud_total <= 0.0:
                local_cloud = np.array([0.5, 0.5], dtype=float)
                local_cloud_total = 1.0

            # hist cap 后的剩余权重按 cap 前 local/cloud 相对比例分回去。
            weights_arr[0] = remaining * local_cloud[0] / local_cloud_total
            weights_arr[1] = remaining * local_cloud[1] / local_cloud_total
            weights_arr[2] = cap

        return self._normalize_gp_triple_weights(weights_arr)

    def _apply_gp_triple_min_weights(self, weights):
        weights_arr = self._normalize_gp_triple_weights(weights)
        min_local = float(self.gp_triple_min_weight_local)
        min_cloud = float(self.gp_triple_min_weight_cloud)
        if not np.isfinite(min_local) or min_local < 0.0:
            min_local = 0.0
        if not np.isfinite(min_cloud) or min_cloud < 0.0:
            min_cloud = 0.0

        min_sum = min_local + min_cloud
        if min_sum <= 0.0:
            return weights_arr
        if min_sum >= 1.0:
            return np.array([min_local / min_sum, min_cloud / min_sum, 0.0], dtype=float)
        if weights_arr[0] >= min_local and weights_arr[1] >= min_cloud:
            return weights_arr

        floors = np.array([min_local, min_cloud, 0.0], dtype=float)
        excess = np.maximum(weights_arr - floors, 0.0)
        excess_total = float(np.sum(excess))
        remaining = 1.0 - min_sum
        if not np.isfinite(excess_total) or excess_total <= 0.0:
            weights_arr = np.array([min_local, min_cloud, remaining], dtype=float)
        else:
            weights_arr = floors + remaining * excess / excess_total

        return self._normalize_gp_triple_weights(weights_arr)

    def _apply_gp_triple_weight_safety(self, weights):
        # cap/min 都是比例约束；最终进入 active source 前统一归一化，避免权重和异常。
        weights_arr = self._normalize_gp_triple_weights(weights)
        weights_arr = self._cap_gp_triple_hist_weight(weights_arr)
        weights_arr = self._apply_gp_triple_min_weights(weights_arr)
        weights_arr = self._cap_gp_triple_hist_weight(weights_arr)
        return self._normalize_gp_triple_weights(weights_arr)

    def _apply_gp_triple_dynamic_hist_min_weight(self, weights):
        weights_arr = self._normalize_gp_triple_weights(weights)
        hist_min = float(self.gp_triple_hist_min_weight)
        if not np.isfinite(hist_min) or hist_min <= 0.0:
            return weights_arr

        hist_cap = float(self.gp_triple_hist_weight_cap)
        if not np.isfinite(hist_cap) or hist_cap < 0.0:
            hist_cap = 0.70
        if hist_cap < 1.0:
            hist_min = min(hist_min, hist_cap)
        hist_min = min(hist_min, 1.0)

        if weights_arr[2] >= hist_min:
            return weights_arr

        local_cloud = weights_arr[:2].copy()
        local_cloud_total = float(np.sum(local_cloud))
        if not np.isfinite(local_cloud_total) or local_cloud_total <= 0.0:
            local_cloud = np.array([
                self.gp_triple_min_weight_local,
                self.gp_triple_min_weight_cloud,
            ], dtype=float)
            local_cloud_total = float(np.sum(local_cloud))
        if not np.isfinite(local_cloud_total) or local_cloud_total <= 0.0:
            local_cloud = np.array([0.5, 0.5], dtype=float)
            local_cloud_total = 1.0

        remaining = max(0.0, 1.0 - hist_min)
        weights_arr[0] = remaining * local_cloud[0] / local_cloud_total
        weights_arr[1] = remaining * local_cloud[1] / local_cloud_total
        weights_arr[2] = hist_min
        return self._normalize_gp_triple_weights(weights_arr)

    def _apply_gp_triple_dynamic_weight_safety(self, weights):
        weights_arr = self._apply_gp_triple_weight_safety(weights)
        weights_arr = self._apply_gp_triple_dynamic_hist_min_weight(weights_arr)
        weights_arr = self._cap_gp_triple_hist_weight(weights_arr)
        return self._normalize_gp_triple_weights(weights_arr)

    def _compute_gp_triple_fixed_weights(self):
        weights = np.array([
            self.gp_triple_weight_local_param,
            self.gp_triple_weight_cloud_param,
            self.gp_triple_weight_hist_param,
        ], dtype=float)
        if (
            weights.shape != (3,)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or float(np.sum(weights)) <= 0.0
        ):
            weights = self._gp_triple_default_weights()
        if self.gp_triple_weight_normalize:
            weights = self._normalize_gp_triple_weights(weights)
        return self._apply_gp_triple_weight_safety(weights)

    def _compute_gp_triple_weights(self):
        if self.gp_triple_weight_mode == "fixed":
            return self._compute_gp_triple_fixed_weights()

        rmse = np.array([
            self.gp_triple_rmse_local,
            self.gp_triple_rmse_cloud,
            self.gp_triple_rmse_hist,
        ], dtype=float)
        eps = float(self.gp_triple_inverse_rmse_eps)
        if (
            rmse.shape != (3,)
            or not np.all(np.isfinite(rmse))
            or np.any(rmse <= 0.0)
            or not np.isfinite(eps)
            or eps <= 0.0
        ):
            return self._compute_gp_triple_fixed_weights()

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            weights = 1.0 / (rmse * rmse + eps)
        if (
            weights.shape != (3,)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or float(np.sum(weights)) <= 0.0
        ):
            return self._compute_gp_triple_fixed_weights()

        return self._apply_gp_triple_weight_safety(weights)

    def _compute_gp_triple_dynamic_weights(self, nearest_distance):
        ratio = 0.0
        penalty = 1.0
        fallback_weights = self._normalize_gp_triple_weights([
            self.gp_triple_weights[0],
            self.gp_triple_weights[1],
            0.0,
        ])

        try:
            distance = float(nearest_distance)
            distance_scale = float(self.gp_triple_hist_distance_scale)
            distance_power = float(self.gp_triple_hist_distance_power)
            eps = float(self.gp_triple_dynamic_eps)
            rmse_local = float(self.gp_triple_rmse_local)
            rmse_cloud = float(self.gp_triple_rmse_cloud)
            rmse_hist = float(self.gp_triple_rmse_hist)
        except (TypeError, ValueError):
            return fallback_weights, ratio, penalty, False

        if (
            not np.isfinite(distance)
            or distance < 0.0
            or not np.isfinite(distance_scale)
            or distance_scale <= 0.0
            or not np.isfinite(distance_power)
            or distance_power <= 0.0
            or not np.isfinite(eps)
            or eps <= 0.0
            or not np.isfinite(rmse_local)
            or rmse_local <= 0.0
            or not np.isfinite(rmse_cloud)
            or rmse_cloud <= 0.0
            or not np.isfinite(rmse_hist)
            or rmse_hist <= 0.0
        ):
            return fallback_weights, ratio, penalty, False

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            ratio = distance / distance_scale
            penalty = 1.0 + np.power(ratio, distance_power)
            precision_local = 1.0 / (rmse_local * rmse_local + eps)
            precision_cloud = 1.0 / (rmse_cloud * rmse_cloud + eps)
            precision_hist = 1.0 / (rmse_hist * rmse_hist * penalty + eps)
            weights = np.array([
                precision_local,
                precision_cloud,
                precision_hist,
            ], dtype=float)

        if (
            not np.isfinite(ratio)
            or not np.isfinite(penalty)
            or penalty < 1.0
            or weights.shape != (3,)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or float(np.sum(weights)) <= 0.0
        ):
            return fallback_weights, 0.0, 1.0, False

        weights = self._apply_gp_triple_dynamic_weight_safety(weights)
        return weights, float(ratio), float(penalty), True

    def _reset_gp_triple_state(self):
        self.gp_triple_raw = np.zeros(7, dtype=float)
        self.gp_triple_weight_local = 0.0
        self.gp_triple_weight_cloud = 0.0
        self.gp_triple_weight_hist = 0.0
        self.gp_triple_available = 0
        self.gp_triple_used_fallback = 0
        self.gp_triple_active_fallback_source_code = 0
        self.gp_triple_dynamic_distance_ratio = 0.0
        self.gp_triple_dynamic_hist_penalty = 1.0
        self.gp_triple_dynamic_mode_code = 0

    def _set_gp_triple_state(self, result):
        weights = np.asarray(result.get("weights", np.zeros(3)), dtype=float)
        if weights.shape != (3,) or not np.all(np.isfinite(weights)):
            weights = np.zeros(3, dtype=float)

        self.gp_triple_raw = self._as_finite_7d(result.get("raw"), 0.0).copy()
        self.gp_triple_weight_local = float(weights[0])
        self.gp_triple_weight_cloud = float(weights[1])
        self.gp_triple_weight_hist = float(weights[2])
        self.gp_triple_available = int(result.get("available", 0))
        self.gp_triple_used_fallback = int(result.get("used_fallback", 0))
        self.gp_triple_active_fallback_source_code = int(
            result.get("fallback_source_code", 0)
        )
        self.gp_triple_dynamic_distance_ratio = float(
            result.get("dynamic_distance_ratio", 0.0)
        )
        self.gp_triple_dynamic_hist_penalty = float(
            result.get("dynamic_hist_penalty", 1.0)
        )
        self.gp_triple_dynamic_mode_code = int(
            result.get("dynamic_mode_code", 0)
        )

    def _get_gp_triple_hist_candidate(self):
        zero = np.zeros(7, dtype=float)
        try:
            hist_db_gated_pred = np.asarray(self.hist_db_gated_pred, dtype=float)
        except (TypeError, ValueError):
            return zero, False

        hist_available = (
            bool(self.gp_historical_db_loaded)
            and int(self.hist_db_loaded) == 1
            and int(self.hist_db_query_valid) == 1
            and int(self.hist_db_available) == 1
            and int(self.hist_db_distance_pass) == 1
            and int(self.hist_db_gated_source_code) == 4
            and hist_db_gated_pred.shape == (7,)
            and np.all(np.isfinite(hist_db_gated_pred))
        )
        if not hist_available:
            return zero, False
        return hist_db_gated_pred.copy(), True

    def _get_gp_triple_fallback_candidate(self):
        zero = np.zeros(7, dtype=float)
        source = self.gp_triple_fallback_source
        if source == "none":
            return zero, 0
        if source == "hist_db":
            hist_candidate, hist_available = self._get_gp_triple_hist_candidate()
            if hist_available:
                return hist_candidate, self.gp_triple_fallback_source_code
            return zero, self.gp_triple_fallback_source_code

        candidate = {
            "local": self.y_hat_local,
            "cloud": self.y_hat_cloud,
            "combined": self.y_hat_combined,
        }.get(source)
        try:
            candidate_arr = np.asarray(candidate, dtype=float)
        except (TypeError, ValueError):
            return zero, 0

        if candidate_arr.shape != (7,) or not np.all(np.isfinite(candidate_arr)):
            return zero, 0
        return candidate_arr.copy(), self.gp_triple_fallback_source_code

    def _get_gp_triple_non_hist_fallback_candidate(self):
        zero = np.zeros(7, dtype=float)
        source = self.gp_triple_fallback_source
        if source not in ("local", "cloud", "combined"):
            source = "combined"

        source_code = {
            "local": 1,
            "cloud": 2,
            "combined": 3,
        }[source]
        candidate = {
            "local": self.y_hat_local,
            "cloud": self.y_hat_cloud,
            "combined": self.y_hat_combined,
        }.get(source)
        try:
            candidate_arr = np.asarray(candidate, dtype=float)
        except (TypeError, ValueError):
            return zero, 0

        if candidate_arr.shape != (7,) or not np.all(np.isfinite(candidate_arr)):
            return zero, 0
        return candidate_arr.copy(), source_code

    def _compute_gp_triple_prediction(self):
        zero = np.zeros(7, dtype=float)
        result = {
            "raw": zero.copy(),
            "weights": np.zeros(3, dtype=float),
            "available": 0,
            "used_fallback": 0,
            "fallback_source_code": 0,
        }

        hist_candidate, hist_available = self._get_gp_triple_hist_candidate()
        if not hist_available and self.gp_triple_require_hist_available:
            fallback, fallback_source_code = self._get_gp_triple_fallback_candidate()
            result.update({
                "raw": fallback,
                "used_fallback": 1,
                "fallback_source_code": fallback_source_code,
            })
            return result

        # triple 只复用本 callback 已经计算好的 local/cloud/hist_db_gated 结果，避免在实时回调里新增 GP 或 DB 查询负担。
        local_raw = self._as_finite_7d(self.y_hat_local, 0.0)
        cloud_raw = self._as_finite_7d(self.y_hat_cloud, 0.0)
        hist_raw = hist_candidate if hist_available else zero.copy()
        weights = np.asarray(self.gp_triple_weights, dtype=float).copy()

        if not hist_available:
            local_cloud_total = float(weights[0] + weights[1])
            if not np.isfinite(local_cloud_total) or local_cloud_total <= 0.0:
                fallback, fallback_source_code = self._get_gp_triple_fallback_candidate()
                result.update({
                    "raw": fallback,
                    "used_fallback": 1,
                    "fallback_source_code": fallback_source_code,
                })
                return result
            weights[0] = weights[0] / local_cloud_total
            weights[1] = weights[1] / local_cloud_total
            weights[2] = 0.0

        # 这里只选择 raw source；scale / clip / joint7-disable 仍由外层统一处理。
        triple_raw = (
            weights[0] * local_raw
            + weights[1] * cloud_raw
            + weights[2] * hist_raw
        )
        if triple_raw.shape != (7,) or not np.all(np.isfinite(triple_raw)):
            fallback, fallback_source_code = self._get_gp_triple_fallback_candidate()
            result.update({
                "raw": fallback,
                "used_fallback": 1,
                "fallback_source_code": fallback_source_code,
            })
            return result

        result.update({
            "raw": triple_raw.copy(),
            "weights": weights.copy(),
            "available": int(hist_available),
        })
        return result

    def _compute_gp_triple_dynamic_prediction(self):
        zero = np.zeros(7, dtype=float)
        result = {
            "raw": zero.copy(),
            "weights": np.zeros(3, dtype=float),
            "available": 0,
            "used_fallback": 0,
            "fallback_source_code": 0,
            "dynamic_distance_ratio": 0.0,
            "dynamic_hist_penalty": 1.0,
            "dynamic_mode_code": 0,
        }

        hist_candidate, hist_available = self._get_gp_triple_hist_candidate()
        try:
            nearest_distance = float(self.hist_db_nearest_distance)
        except (TypeError, ValueError):
            nearest_distance = float("inf")

        distance_gate_passed = (
            hist_available
            and np.isfinite(nearest_distance)
            and nearest_distance <= float(self.gp_historical_db_max_distance)
        )
        if not distance_gate_passed:
            fallback, fallback_source_code = self._get_gp_triple_non_hist_fallback_candidate()
            result.update({
                "raw": fallback,
                "used_fallback": 1,
                "fallback_source_code": fallback_source_code,
                "dynamic_mode_code": 1,
            })
            return result

        weights, distance_ratio, hist_penalty, weights_valid = (
            self._compute_gp_triple_dynamic_weights(nearest_distance)
        )
        result["dynamic_distance_ratio"] = distance_ratio
        result["dynamic_hist_penalty"] = hist_penalty
        if not weights_valid:
            fallback, fallback_source_code = self._get_gp_triple_non_hist_fallback_candidate()
            result.update({
                "raw": fallback,
                "used_fallback": 1,
                "fallback_source_code": fallback_source_code,
                "dynamic_mode_code": 1,
            })
            return result

        # dynamic triple 只复用本 callback 已经计算好的 local/cloud/hist_db_gated 结果。
        local_raw = self._as_finite_7d(self.y_hat_local, 0.0)
        cloud_raw = self._as_finite_7d(self.y_hat_cloud, 0.0)
        hist_raw = hist_candidate.copy()
        triple_raw = (
            weights[0] * local_raw
            + weights[1] * cloud_raw
            + weights[2] * hist_raw
        )
        if triple_raw.shape != (7,) or not np.all(np.isfinite(triple_raw)):
            fallback, fallback_source_code = self._get_gp_triple_non_hist_fallback_candidate()
            result.update({
                "raw": fallback,
                "used_fallback": 1,
                "fallback_source_code": fallback_source_code,
                "dynamic_mode_code": 1,
            })
            return result

        result.update({
            "raw": triple_raw.copy(),
            "weights": weights.copy(),
            "available": 1,
            "dynamic_mode_code": 2,
        })
        return result

    def _as_finite_7d(self, value, fill_value=0.0):
        try:
            arr = np.asarray(value, dtype=float)
            if arr.shape != (7,):
                raise ValueError
            return np.where(np.isfinite(arr), arr, fill_value)
        except (TypeError, ValueError):
            return np.ones(7, dtype=float) * fill_value

    def _log_gp_triple_debug_safety(self):
        if (
            not self.gp_triple_debug_safety_log_enabled
            or self.gp_compensation_source not in ("triple", "triple_dynamic")
            or not self.gp_compensation_enabled
            or (
                self._gp_triple_debug_safety_log_count
                >= self.gp_triple_debug_safety_log_first_n
            )
        ):
            return

        self._gp_triple_debug_safety_log_count += 1
        log_index = self._gp_triple_debug_safety_log_count
        max_abs_selected_raw = float(np.max(np.abs(self._gp_selected_raw)))
        max_abs_scaled = float(np.max(np.abs(self._gp_scaled)))
        max_abs_applied = float(np.max(np.abs(self._gp_applied)))
        fallback_source = (
            self.gp_triple_fallback_source
            if int(self.gp_triple_used_fallback) == 1
            else "none"
        )

        self.get_logger().warn(
            "[GP Triple Safety] "
            f"index={log_index}/{self.gp_triple_debug_safety_log_first_n}, "
            f"source='{self.gp_compensation_source}', "
            f"gp_compensation_scale={self.gp_compensation_scale}, "
            f"max_abs_selected_raw={max_abs_selected_raw:.9g}, "
            f"max_abs_scaled={max_abs_scaled:.9g}, "
            f"max_abs_applied={max_abs_applied:.9g}, "
            f"weight_local={self.gp_triple_weight_local:.9g}, "
            f"weight_cloud={self.gp_triple_weight_cloud:.9g}, "
            f"weight_hist={self.gp_triple_weight_hist:.9g}, "
            f"triple_available={int(self.gp_triple_available)}, "
            f"triple_used_fallback={int(self.gp_triple_used_fallback)}, "
            f"dynamic_distance_ratio={self.gp_triple_dynamic_distance_ratio:.9g}, "
            f"dynamic_hist_penalty={self.gp_triple_dynamic_hist_penalty:.9g}, "
            f"dynamic_mode_code={int(self.gp_triple_dynamic_mode_code)}, "
            f"fallback_source='{fallback_source}'"
        )

        if (
            float(self.gp_compensation_scale) == 0.0
            and (max_abs_scaled > 0.0 or max_abs_applied > 0.0)
        ):
            self.get_logger().error(
                "[GP Triple Safety] scale is exactly 0.0 but scaled/applied "
                f"is nonzero: max_abs_scaled={max_abs_scaled:.9g}, "
                f"max_abs_applied={max_abs_applied:.9g}"
            )

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
        self.gp_shadow_hist_pool_size = 0
        self.gp_shadow_hist_k_used = 0
        self.gp_shadow_hist_nearest_distance = 0.0
        self.gp_shadow_hist_mean_distance_topk = 0.0

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

    def _new_historical_shadow_diagnostics(self):
        return {
            "pool_size": len(self.gp_hist_x_shadow),
            "k_used": 0,
            "nearest_distance": 0.0,
            "mean_distance_topk": 0.0,
        }

    def _append_historical_shadow_candidate(self, x, y_hat, var, t):
        if (
            not self.gp_shadow_paper_fusion_logging_enabled
            or not self.gp_historical_shadow_enabled
            or self.gp_historical_source_mode != "local_prediction_pool"
            or not self.gp_prediction_enabled
            or not self.gp_ready
            or not self.use_gp
        ):
            return

        try:
            sequence = int(t)
            x_arr = np.asarray(x, dtype=np.float32)
            mu_arr = np.asarray(y_hat, dtype=float)
            var_arr = np.asarray(var, dtype=float)
        except (TypeError, ValueError, OverflowError):
            return

        if sequence <= self._gp_hist_last_appended_sequence_shadow:
            return
        if x_arr.shape != (14,) or mu_arr.shape != (7,) or var_arr.shape != (7,):
            return
        if (
            not np.all(np.isfinite(x_arr))
            or not np.all(np.isfinite(mu_arr))
            or not np.all(np.isfinite(var_arr))
            or np.any(var_arr <= 0.0)
        ):
            return

        self.gp_hist_x_shadow.append(x_arr.copy())
        self.gp_hist_mu_shadow.append(mu_arr.copy())
        self.gp_hist_var_shadow.append(var_arr.copy())
        self.gp_hist_t_shadow.append(sequence)
        self._gp_hist_last_appended_sequence_shadow = sequence

    def _query_historical_shadow_pool(self, x_query):
        fallback_var = np.ones(7, dtype=float) * self.gp_shadow_hist_fallback_variance
        diagnostics = self._new_historical_shadow_diagnostics()
        pool_size = diagnostics["pool_size"]

        try:
            x_arr = np.asarray(x_query, dtype=np.float32)
        except (TypeError, ValueError):
            return np.zeros(7, dtype=float), fallback_var, 0, diagnostics

        if x_arr.shape != (14,) or not np.all(np.isfinite(x_arr)) or pool_size == 0:
            return np.zeros(7, dtype=float), fallback_var, 0, diagnostics
        if (
            len(self.gp_hist_mu_shadow) != pool_size
            or len(self.gp_hist_var_shadow) != pool_size
            or len(self.gp_hist_t_shadow) != pool_size
        ):
            return np.zeros(7, dtype=float), fallback_var, 0, diagnostics

        try:
            hist_x = np.stack(self.gp_hist_x_shadow, axis=0).astype(np.float32)
            hist_mu = np.stack(self.gp_hist_mu_shadow, axis=0).astype(float)
            hist_var = np.stack(self.gp_hist_var_shadow, axis=0).astype(float)
        except (TypeError, ValueError):
            return np.zeros(7, dtype=float), fallback_var, 0, diagnostics

        if (
            hist_x.shape != (pool_size, 14)
            or hist_mu.shape != (pool_size, 7)
            or hist_var.shape != (pool_size, 7)
            or not np.all(np.isfinite(hist_x))
            or not np.all(np.isfinite(hist_mu))
            or not np.all(np.isfinite(hist_var))
        ):
            return np.zeros(7, dtype=float), fallback_var, 0, diagnostics

        distances = np.linalg.norm(hist_x - x_arr.reshape(1, -1), axis=1)
        if not np.all(np.isfinite(distances)):
            return np.zeros(7, dtype=float), fallback_var, 0, diagnostics

        k_candidate = min(self.gp_historical_shadow_k, pool_size)
        nearest_indices = np.argsort(distances)[:k_candidate]
        nearest_distances = distances[nearest_indices]
        diagnostics["nearest_distance"] = float(nearest_distances[0])
        diagnostics["mean_distance_topk"] = float(np.mean(nearest_distances))

        if (
            pool_size < self.gp_historical_shadow_min_points
            or diagnostics["nearest_distance"] > self.gp_historical_shadow_max_distance
        ):
            return np.zeros(7, dtype=float), fallback_var, 0, diagnostics

        selected_mu = hist_mu[nearest_indices]
        selected_var = np.maximum(
            hist_var[nearest_indices],
            self.gp_historical_shadow_variance_floor,
        )
        distance_weight = 1.0 / np.maximum(
            nearest_distances,
            self.gp_historical_shadow_distance_eps,
        )
        precision_distance_weight = distance_weight[:, None] / selected_var
        weight_sum = np.sum(precision_distance_weight, axis=0)
        if not np.all(np.isfinite(weight_sum)) or np.any(weight_sum <= 0.0):
            return np.zeros(7, dtype=float), fallback_var, 0, diagnostics

        normalized_weight = precision_distance_weight / weight_sum
        y_hat_hist = np.sum(normalized_weight * selected_mu, axis=0)
        var_hist = np.sum(normalized_weight * selected_var, axis=0)
        if (
            y_hat_hist.shape != (7,)
            or var_hist.shape != (7,)
            or not np.all(np.isfinite(y_hat_hist))
            or not np.all(np.isfinite(var_hist))
        ):
            return np.zeros(7, dtype=float), fallback_var, 0, diagnostics

        diagnostics["k_used"] = k_candidate
        return y_hat_hist, np.maximum(
            var_hist, self.gp_historical_shadow_variance_floor
        ), 1, diagnostics

    def _get_historical_shadow_candidate(self, x_query):
        fallback_var = np.ones(7, dtype=float) * self.gp_shadow_hist_fallback_variance
        diagnostics = self._new_historical_shadow_diagnostics()

        if not self.gp_historical_shadow_enabled:
            return np.zeros(7, dtype=float), fallback_var, 0, diagnostics

        if self.gp_historical_source_mode == "none":
            return np.zeros(7, dtype=float), fallback_var, 0, diagnostics

        if self.gp_historical_source_mode == "local_prediction_pool":
            return self._query_historical_shadow_pool(x_query)

        return np.zeros(7, dtype=float), fallback_var, 0, diagnostics

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

    def _update_gp_shadow_logging_state(self, q, dq):
        if (
            not self.gp_shadow_paper_fusion_logging_enabled
            or not self.gp_prediction_enabled
            or not self.gp_active
            or not self.use_gp
        ):
            self._reset_gp_shadow_state()
            return

        gp_tick = (
            self.gp_counter > 0
            and self.gp_counter % self.gp_prediction_stride == 0
        )
        if not gp_tick:
            return

        self._reset_gp_shadow_state()

        x_query = None
        if (
            self.gp_historical_shadow_enabled
            and self.gp_historical_source_mode == "local_prediction_pool"
        ):
            x_query = self._build_gp_shadow_feature(q, dq)
        (
            y_hist,
            var_hist,
            historical_available,
            diagnostics,
        ) = self._get_historical_shadow_candidate(x_query)
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
        self.gp_shadow_hist_pool_size = int(diagnostics["pool_size"])
        self.gp_shadow_hist_k_used = int(diagnostics["k_used"])
        self.gp_shadow_hist_nearest_distance = float(
            diagnostics["nearest_distance"]
        )
        self.gp_shadow_hist_mean_distance_topk = float(
            diagnostics["mean_distance_topk"]
        )

        # 必须先 query 旧池，再 append 当前 local prediction，避免检索到当前样本自身。
        self._append_historical_shadow_candidate(
            self._gp_local_feature_shadow,
            self.y_hat_local,
            self.var_local,
            self._gp_local_prediction_sequence_shadow,
        )

    def _delay_cloud_like_output(self, y_hat_current, var_current):
        """Return cloud-like prediction delayed by delay_steps controller ticks."""
        y_hat_current = self._as_finite_7d(y_hat_current, 0.0)
        try:
            var_current = np.asarray(var_current, dtype=float)
            if var_current.shape != (7,):
                raise ValueError
            var_current = np.where(
                np.isfinite(var_current) & (var_current > 0.0),
                var_current,
                1e6
            )
        except (TypeError, ValueError):
            var_current = np.ones(7, dtype=float) * 1e6

        self.y_hat_cloud_buffer.append((y_hat_current.copy(), var_current.copy()))
        if self.delay_steps <= 0:
            return y_hat_current.copy(), var_current.copy()

        if len(self.y_hat_cloud_buffer) <= self.delay_steps:
            return np.zeros(7, dtype=float), np.ones(7, dtype=float) * 1e6

        y_hat_delayed, var_delayed = self.y_hat_cloud_buffer[-(self.delay_steps + 1)]
        return y_hat_delayed.copy(), var_delayed.copy()

    def _apply_gp_compensation(self, tau):
        self._reset_gp_triple_state()
        # 真机安全 gate：smooth transition / 起步阶段不允许 GP compensation 进入 torque。
        # 只有 trajectory_publisher 发布 /data_recording_enabled=True 后，才开始比较各 GP source 的补偿效果。
        if (
            not self.data_recording_enabled
            or not self.gp_prediction_enabled
            or not self.gp_compensation_enabled
        ):
            self._gp_source_code = 0
            self._gp_selected_raw = np.zeros(7, dtype=float)
            self._gp_scaled = np.zeros(7, dtype=float)
            self._gp_applied = np.zeros(7, dtype=float)
            self._gp_clip_active = np.zeros(7, dtype=int)
            return tau

        # combined 当前是 local/cloud variance fusion candidate；hist_db 必须显式 opt-in。
        hist_compensation_ready = False
        if self.gp_compensation_source == "cloud":
            compensation = self.y_hat_cloud
            self._gp_source_code = 2
        elif self.gp_compensation_source == "combined":
            compensation = self.y_hat_combined
            self._gp_source_code = 3
        elif self.gp_compensation_source == "hist_db":
            compensation = np.zeros(7, dtype=float)
            self._gp_source_code = 4
            if (
                self.gp_historical_db_preflight_required
                and not self._historical_db_active_allowed()
            ):
                self.hist_db_runtime_fallback_used = 1
                self._gp_source_code = 0
                self._gp_selected_raw = np.zeros(7, dtype=float)
                self._gp_scaled = np.zeros(7, dtype=float)
                self._gp_applied = np.zeros(7, dtype=float)
                self._gp_clip_active = np.zeros(7, dtype=int)
                self._maybe_log_hist_db_runtime_diag("nominal")
                return tau
            try:
                hist_db_gated_pred = np.asarray(self.hist_db_gated_pred, dtype=float)
            except (TypeError, ValueError):
                hist_db_gated_pred = None

            if (
                bool(self.gp_historical_db_loaded)
                and int(self.hist_db_loaded) == 1
                and int(self.hist_db_query_valid) == 1
                and int(self.hist_db_available) == 1
                and int(self.hist_db_distance_pass) == 1
                and int(self.hist_db_gated_source_code) == 4
                and hist_db_gated_pred is not None
                and hist_db_gated_pred.shape == (7,)
                and np.all(np.isfinite(hist_db_gated_pred))
            ):
                compensation = hist_db_gated_pred.copy()
                hist_compensation_ready = True
            else:
                self.hist_db_runtime_fallback_used = 1
                if self.gp_disable_silent_hist_fallback:
                    self._gp_source_code = 0
                    self._gp_selected_raw = np.zeros(7, dtype=float)
                    self._gp_scaled = np.zeros(7, dtype=float)
                    self._gp_applied = np.zeros(7, dtype=float)
                    self._gp_clip_active = np.zeros(7, dtype=int)
                    self._maybe_log_hist_db_runtime_diag("nominal")
                    return tau
                if self.gp_historical_db_preflight_enabled:
                    self._maybe_log_hist_db_runtime_diag("zero_fallback")
        elif self.gp_compensation_source == "triple":
            triple_result = self._compute_gp_triple_prediction()
            self._set_gp_triple_state(triple_result)
            compensation = self.gp_triple_raw
            self._gp_source_code = 5
        elif self.gp_compensation_source == "triple_dynamic":
            triple_result = self._compute_gp_triple_dynamic_prediction()
            self._set_gp_triple_state(triple_result)
            compensation = self.gp_triple_raw
            self._gp_source_code = 6
        else:
            compensation = self.y_hat_local
            self._gp_source_code = 1

        if self.gp_compensation_source in ("local", "cloud", "combined") and not self.gp_output_fresh:
            self._gp_source_code = 0
            self._gp_selected_raw = np.zeros(7, dtype=float)
            self._gp_scaled = np.zeros(7, dtype=float)
            self._gp_applied = np.zeros(7, dtype=float)
            self._gp_clip_active = np.zeros(7, dtype=int)
            return tau

        # 先清洗成 finite 7D，再 scale / clip，避免 non-finite 进入 torque command。
        self._gp_selected_raw = self._as_finite_7d(compensation, 0.0).copy()
        self._gp_scaled = self.gp_compensation_scale * self._gp_selected_raw
        self._gp_applied = np.clip(
            self._gp_scaled,
            -self.gp_compensation_clip_nm,
            self.gp_compensation_clip_nm
        )
        self._gp_clip_active = (
            np.abs(self._gp_scaled - self._gp_applied) > 1e-12
        ).astype(int)
        if self.gp_compensation_disable_joint7:
            self._gp_applied[6] = 0.0

        if self.gp_compensation_source in ("triple", "triple_dynamic"):
            self._log_gp_triple_debug_safety()
        if self.gp_compensation_source == "hist_db" and hist_compensation_ready:
            self._maybe_log_hist_db_runtime_diag("hist_db")

        if not self._gp_compensation_logged:
            self.get_logger().warn(
                "[GP] Compensation ENABLED: "
                f"source='{self.gp_compensation_source}', "
                f"scale={self.gp_compensation_scale}, "
                f"clip_nm={self.gp_compensation_clip_nm}, "
                f"disable_joint7={self.gp_compensation_disable_joint7}"
            )
            self._gp_compensation_logged = True

        # 符号方向沿用原注释：tau = tau - compensation。
        return tau - self._gp_applied

    def _gp_predict_and_update(
        self,
        q,
        dq_des_joint,
        ddq_des_joint,
        tau_residual,
        models,
        update=True,
        timing_label=None,
        timing_row=None,
    ):
        """
        本地 GP：高维输入版本（14维 or 21维）
        每个关节都使用相同的 x_full = concat([q, dq, ddq])
        """

        predict_timing_field = None
        if timing_row is not None:
            if timing_label == "local":
                timing_row["local_gp_called"] = 1
                predict_timing_field = "gp_local_predict_ms"
            elif timing_label == "cloud_like":
                timing_row["cloud_like_gp_called"] = 1
                predict_timing_field = "gp_cloud_like_predict_ms"

        if not self.gp_prediction_enabled:
            return np.zeros(7, dtype=float), np.ones(7, dtype=float) * 1e6

        if not self.gp_ready or not self.use_gp:
            # GP 未 ready/未启用时可能在 callback 内高频发生；默认静默，避免实时路径打印。
            return np.zeros(7, dtype=float), np.ones(7, dtype=float) * 1e6

        y_hat = np.zeros(7, dtype=float)
        y_var = np.ones(7, dtype=float) * 1e6  # 默认给大方差，表示“不可信”
        # ==================================================
        # 1) 构造统一的高维输入 x_full
        # ==================================================
        # 如果你的训练是 q + dq + ddq → 21 维
        # x_full = np.concatenate([q, dq_des_joint, ddq_des_joint]).astype(np.float32)

        # 如果你训练使用的是 q + dq → 14 维，请改成：
        x_full = self._build_gp_shadow_feature(q, dq_des_joint)

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
            predict_start = (
                time.perf_counter()
                if timing_row is not None and predict_timing_field is not None
                else None
            )
            try:
                mu_std, var_std = model.predict(x_std.astype(np.float32))
            finally:
                if predict_start is not None:
                    self._timing_add_ms(
                        timing_row,
                        predict_timing_field,
                        predict_start
                    )
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
                        add_point_start = time.perf_counter() if timing_row is not None else None
                        try:
                            model.add_point(
                                x_std.astype(np.float32),
                                np.array([y_std], dtype=np.float32)
                            )
                        finally:
                            if add_point_start is not None:
                                self._timing_add_ms(
                                    timing_row,
                                    "gp_add_point_ms",
                                    add_point_start
                                )
                        if timing_row is not None:
                            timing_row["add_point_count"] = (
                                int(timing_row["add_point_count"]) + 1
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
        csv_save_start = time.perf_counter() if self.timing_logging_enabled else None
        if not self.tau_history:
            self.get_logger().warning('No data to save - tau_history is empty')
            self._finish_csv_save_timing(csv_save_start)
            return

        try:
            output_dir = Path(self.data_output_dir).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            run_name_stem = Path(self.run_name).name if self.run_name else ""
            filename_stem = (
                f"{run_name_stem}_cartesian_impedance_controller_data.csv"
                if run_name_stem
                else "cartesian_impedance_controller_data.csv"
            )
            filename = output_dir / filename_stem

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
                self.gp_prediction_stride_history,
                self.gp_prediction_updated_this_tick_history,
                self.gp_prediction_age_sec_history,
                self.gp_output_fresh_history,
                self.future_trajectory_request_stride_history,
                self.future_trajectory_updated_this_tick_history,
                self.gp_triple_raw_history,
                self.gp_triple_weight_local_history,
                self.gp_triple_weight_cloud_history,
                self.gp_triple_weight_hist_history,
                self.gp_triple_available_history,
                self.gp_triple_used_fallback_history,
                self.gp_triple_fallback_source_code_history,
                self.gp_triple_weight_mode_code_history,
                self.gp_triple_hist_weight_cap_history,
                self.gp_triple_rmse_local_history,
                self.gp_triple_rmse_cloud_history,
                self.gp_triple_rmse_hist_history,
                self.gp_triple_dynamic_distance_ratio_history,
                self.gp_triple_dynamic_hist_penalty_history,
                self.gp_triple_dynamic_mode_code_history,
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
                self.gp_shadow_hist_pool_size_history,
                self.gp_shadow_hist_k_used_history,
                self.gp_shadow_hist_nearest_distance_history,
                self.gp_shadow_hist_mean_distance_topk_history,
                self.hist_db_loaded_history,
                self.hist_db_query_valid_history,
                self.hist_db_available_history,
                self.hist_db_online_disabled_history,
                self.hist_db_distance_pass_history,
                self.hist_db_k_used_history,
                self.hist_db_nearest_distance_history,
                self.hist_db_mean_topk_distance_history,
                self.hist_db_gated_source_code_history,
                self.hist_db_pred_history,
                self.hist_db_gated_pred_history,
                self.hist_db_query_stride_history,
                self.hist_db_query_updated_this_tick_history,
                self.hist_db_query_reused_history,
                self.hist_db_query_counter_history,
                self.hist_db_preflight_phase_history,
                self.hist_db_preflight_pass_history,
                self.hist_db_preflight_active_allowed_history,
                self.hist_db_preflight_sample_count_history,
                self.hist_db_preflight_pass_ratio_history,
                self.hist_db_preflight_nearest_mean_history,
                self.hist_db_preflight_nearest_p95_history,
                self.hist_db_preflight_nearest_max_history,
                self.hist_db_runtime_fallback_used_history,
                self.hist_soft_valid_history,
                self.hist_soft_nearest_distance_history,
                self.hist_soft_raw_w_hist_history,
                self.hist_soft_norm_w_local_history,
                self.hist_soft_norm_w_cloud_history,
                self.hist_soft_norm_w_hist_history,
                self.hist_soft_pred_history,
                self.hist_soft_delta_vs_local_cloud_history,
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
                    'gp_model_local_loaded_count',
                    'gp_model_cloud_loaded_count',
                    'gp_model_cloud_fallback_count',
                    'gp_model_empty_or_prior_count',
                    'gp_model_cloud_uses_cloud_pkl',
                    'gp_model_cloud_uses_local_fallback',
                    'gp_prediction_stride',
                    'gp_prediction_updated_this_tick',
                    'gp_prediction_age_sec',
                    'gp_output_fresh',
                    'future_trajectory_request_stride',
                    'future_trajectory_updated_this_tick',
                ])
                header.extend([f'tau_nominal_{i+1}' for i in range(7)])
                header.extend([f'tau_final_{i+1}' for i in range(7)])
                header.extend([f'gp_selected_raw_{i+1}' for i in range(7)])
                header.extend([f'gp_scaled_{i+1}' for i in range(7)])
                header.extend([f'gp_applied_{i+1}' for i in range(7)])
                header.extend([f'gp_clip_active_{i+1}' for i in range(7)])
                header.extend([f'gp_triple_raw_{i+1}' for i in range(7)])
                header.extend([
                    'gp_triple_weight_local',
                    'gp_triple_weight_cloud',
                    'gp_triple_weight_hist',
                    'gp_triple_available',
                    'gp_triple_used_fallback',
                    'gp_triple_fallback_source_code',
                    'gp_triple_weight_mode_code',
                    'gp_triple_hist_weight_cap',
                    'gp_triple_rmse_local',
                    'gp_triple_rmse_cloud',
                    'gp_triple_rmse_hist',
                    'gp_triple_dynamic_distance_ratio',
                    'gp_triple_dynamic_hist_penalty',
                    'gp_triple_dynamic_mode_code',
                ])
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
                header.extend([
                    'gp_shadow_hist_pool_size',
                    'gp_shadow_hist_k_used',
                    'gp_shadow_hist_nearest_distance',
                    'gp_shadow_hist_mean_distance_topk',
                ])
                header.extend([
                    'hist_db_loaded',
                    'hist_db_query_valid',
                    'hist_db_available',
                    'hist_db_online_disabled',
                    'hist_db_distance_pass',
                    'hist_db_k_used',
                    'hist_db_nearest_distance',
                    'hist_db_mean_topk_distance',
                    'hist_db_q_scale',
                    'hist_db_dq_scale',
                    'hist_db_max_distance',
                    'hist_db_fallback_source_code',
                    'hist_db_gated_source_code',
                ])
                header.extend([f'hist_db_pred_{i+1}' for i in range(7)])
                header.extend([f'hist_db_gated_pred_{i+1}' for i in range(7)])
                header.extend([
                    'hist_db_query_stride',
                    'hist_db_query_updated_this_tick',
                    'hist_db_query_reused',
                    'hist_db_query_counter',
                    'hist_db_preflight_enabled',
                    'hist_db_preflight_required',
                    'hist_db_preflight_mode',
                    'hist_db_preflight_phase',
                    'hist_db_preflight_pass',
                    'hist_db_preflight_active_allowed',
                    'hist_db_preflight_sample_count',
                    'hist_db_preflight_pass_ratio',
                    'hist_db_preflight_nearest_mean',
                    'hist_db_preflight_nearest_p95',
                    'hist_db_preflight_nearest_max',
                    'hist_db_runtime_fallback_used',
                ])
                header.extend([
                    'hist_soft_enabled',
                    'hist_soft_valid',
                    'hist_soft_online_mode',
                    'hist_soft_alpha',
                    'hist_soft_distance_threshold',
                    'hist_soft_online_scale',
                    'hist_soft_non_online_scale',
                    'hist_soft_nearest_distance',
                    'hist_soft_raw_w_hist',
                    'hist_soft_norm_w_local',
                    'hist_soft_norm_w_cloud',
                    'hist_soft_norm_w_hist',
                ])
                header.extend([f'hist_soft_pred_{i+1}' for i in range(7)])
                header.extend([
                    f'hist_soft_delta_vs_local_cloud_{i+1}' for i in range(7)
                ])
                header.extend([
                    'run_name',
                    'control_frequency',
                    'delay_steps',
                    'data_output_dir',
                ])
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
                        int(self.gp_model_local_loaded_count),
                        int(self.gp_model_cloud_loaded_count),
                        int(self.gp_model_cloud_fallback_count),
                        int(self.gp_model_empty_or_prior_count),
                        int(self.gp_model_cloud_uses_cloud_pkl),
                        int(self.gp_model_cloud_uses_local_fallback),
                        self.gp_prediction_stride_history[i]
                        if i < len(self.gp_prediction_stride_history)
                        else int(self.gp_prediction_stride),
                        self.gp_prediction_updated_this_tick_history[i]
                        if i < len(self.gp_prediction_updated_this_tick_history)
                        else 0,
                        self.gp_prediction_age_sec_history[i]
                        if i < len(self.gp_prediction_age_sec_history)
                        else float(self.gp_prediction_age_sec),
                        self.gp_output_fresh_history[i]
                        if i < len(self.gp_output_fresh_history)
                        else int(self.gp_output_fresh),
                        self.future_trajectory_request_stride_history[i]
                        if i < len(self.future_trajectory_request_stride_history)
                        else int(self.future_trajectory_request_stride),
                        self.future_trajectory_updated_this_tick_history[i]
                        if i < len(self.future_trajectory_updated_this_tick_history)
                        else 0,
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

                    if i < len(self.gp_triple_raw_history):
                        row.extend(self.gp_triple_raw_history[i])
                    else:
                        row.extend([0.0] * 7)

                    gp_triple_weight_local = (
                        float(self.gp_triple_weight_local_history[i])
                        if i < len(self.gp_triple_weight_local_history)
                        else float(self.gp_triple_weight_local)
                    )
                    gp_triple_weight_cloud = (
                        float(self.gp_triple_weight_cloud_history[i])
                        if i < len(self.gp_triple_weight_cloud_history)
                        else float(self.gp_triple_weight_cloud)
                    )
                    gp_triple_weight_hist = (
                        float(self.gp_triple_weight_hist_history[i])
                        if i < len(self.gp_triple_weight_hist_history)
                        else float(self.gp_triple_weight_hist)
                    )
                    gp_triple_available = (
                        int(self.gp_triple_available_history[i])
                        if i < len(self.gp_triple_available_history)
                        else int(self.gp_triple_available)
                    )
                    gp_triple_used_fallback = (
                        int(self.gp_triple_used_fallback_history[i])
                        if i < len(self.gp_triple_used_fallback_history)
                        else int(self.gp_triple_used_fallback)
                    )
                    gp_triple_fallback_source_code = (
                        int(self.gp_triple_fallback_source_code_history[i])
                        if i < len(self.gp_triple_fallback_source_code_history)
                        else int(self.gp_triple_active_fallback_source_code)
                    )
                    gp_triple_weight_mode_code = (
                        int(self.gp_triple_weight_mode_code_history[i])
                        if i < len(self.gp_triple_weight_mode_code_history)
                        else int(self.gp_triple_weight_mode_code)
                    )
                    gp_triple_hist_weight_cap = (
                        float(self.gp_triple_hist_weight_cap_history[i])
                        if i < len(self.gp_triple_hist_weight_cap_history)
                        else float(self.gp_triple_hist_weight_cap)
                    )
                    gp_triple_rmse_local = (
                        float(self.gp_triple_rmse_local_history[i])
                        if i < len(self.gp_triple_rmse_local_history)
                        else float(self.gp_triple_rmse_local)
                    )
                    gp_triple_rmse_cloud = (
                        float(self.gp_triple_rmse_cloud_history[i])
                        if i < len(self.gp_triple_rmse_cloud_history)
                        else float(self.gp_triple_rmse_cloud)
                    )
                    gp_triple_rmse_hist = (
                        float(self.gp_triple_rmse_hist_history[i])
                        if i < len(self.gp_triple_rmse_hist_history)
                        else float(self.gp_triple_rmse_hist)
                    )
                    gp_triple_dynamic_distance_ratio = (
                        float(self.gp_triple_dynamic_distance_ratio_history[i])
                        if i < len(self.gp_triple_dynamic_distance_ratio_history)
                        else float(self.gp_triple_dynamic_distance_ratio)
                    )
                    gp_triple_dynamic_hist_penalty = (
                        float(self.gp_triple_dynamic_hist_penalty_history[i])
                        if i < len(self.gp_triple_dynamic_hist_penalty_history)
                        else float(self.gp_triple_dynamic_hist_penalty)
                    )
                    gp_triple_dynamic_mode_code = (
                        int(self.gp_triple_dynamic_mode_code_history[i])
                        if i < len(self.gp_triple_dynamic_mode_code_history)
                        else int(self.gp_triple_dynamic_mode_code)
                    )
                    row.extend([
                        gp_triple_weight_local,
                        gp_triple_weight_cloud,
                        gp_triple_weight_hist,
                        gp_triple_available,
                        gp_triple_used_fallback,
                        gp_triple_fallback_source_code,
                        gp_triple_weight_mode_code,
                        gp_triple_hist_weight_cap,
                        gp_triple_rmse_local,
                        gp_triple_rmse_cloud,
                        gp_triple_rmse_hist,
                        gp_triple_dynamic_distance_ratio,
                        gp_triple_dynamic_hist_penalty,
                        gp_triple_dynamic_mode_code,
                    ])

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

                    if i < len(self.gp_shadow_hist_pool_size_history):
                        hist_pool_size = int(self.gp_shadow_hist_pool_size_history[i])
                    else:
                        hist_pool_size = int(self.gp_shadow_hist_pool_size)

                    if i < len(self.gp_shadow_hist_k_used_history):
                        hist_k_used = int(self.gp_shadow_hist_k_used_history[i])
                    else:
                        hist_k_used = int(self.gp_shadow_hist_k_used)

                    if i < len(self.gp_shadow_hist_nearest_distance_history):
                        hist_nearest_distance = float(
                            self.gp_shadow_hist_nearest_distance_history[i]
                        )
                    else:
                        hist_nearest_distance = float(
                            self.gp_shadow_hist_nearest_distance
                        )

                    if i < len(self.gp_shadow_hist_mean_distance_topk_history):
                        hist_mean_distance_topk = float(
                            self.gp_shadow_hist_mean_distance_topk_history[i]
                        )
                    else:
                        hist_mean_distance_topk = float(
                            self.gp_shadow_hist_mean_distance_topk
                        )

                    row.extend([
                        hist_pool_size,
                        hist_k_used,
                        hist_nearest_distance,
                        hist_mean_distance_topk,
                    ])

                    hist_db_loaded = (
                        int(self.hist_db_loaded_history[i])
                        if i < len(self.hist_db_loaded_history)
                        else int(self.hist_db_loaded)
                    )
                    hist_db_query_valid = (
                        int(self.hist_db_query_valid_history[i])
                        if i < len(self.hist_db_query_valid_history)
                        else int(self.hist_db_query_valid)
                    )
                    hist_db_available = (
                        int(self.hist_db_available_history[i])
                        if i < len(self.hist_db_available_history)
                        else int(self.hist_db_available)
                    )
                    hist_db_online_disabled = (
                        int(self.hist_db_online_disabled_history[i])
                        if i < len(self.hist_db_online_disabled_history)
                        else int(self.hist_db_online_disabled)
                    )
                    hist_db_distance_pass = (
                        int(self.hist_db_distance_pass_history[i])
                        if i < len(self.hist_db_distance_pass_history)
                        else int(self.hist_db_distance_pass)
                    )
                    hist_db_k_used = (
                        int(self.hist_db_k_used_history[i])
                        if i < len(self.hist_db_k_used_history)
                        else int(self.hist_db_k_used)
                    )
                    hist_db_nearest_distance = (
                        float(self.hist_db_nearest_distance_history[i])
                        if i < len(self.hist_db_nearest_distance_history)
                        else float(self.hist_db_nearest_distance)
                    )
                    hist_db_mean_topk_distance = (
                        float(self.hist_db_mean_topk_distance_history[i])
                        if i < len(self.hist_db_mean_topk_distance_history)
                        else float(self.hist_db_mean_topk_distance)
                    )
                    hist_db_gated_source_code = (
                        int(self.hist_db_gated_source_code_history[i])
                        if i < len(self.hist_db_gated_source_code_history)
                        else int(self.hist_db_gated_source_code)
                    )
                    row.extend([
                        hist_db_loaded,
                        hist_db_query_valid,
                        hist_db_available,
                        hist_db_online_disabled,
                        hist_db_distance_pass,
                        hist_db_k_used,
                        hist_db_nearest_distance,
                        hist_db_mean_topk_distance,
                        self.gp_historical_db_q_scale,
                        self.gp_historical_db_dq_scale,
                        self.gp_historical_db_max_distance,
                        self.gp_historical_db_fallback_source_code,
                        hist_db_gated_source_code,
                    ])

                    if i < len(self.hist_db_pred_history):
                        row.extend(self.hist_db_pred_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.hist_db_gated_pred_history):
                        row.extend(self.hist_db_gated_pred_history[i])
                    else:
                        row.extend([0.0] * 7)

                    row.extend([
                        self.hist_db_query_stride_history[i]
                        if i < len(self.hist_db_query_stride_history)
                        else int(getattr(self, "gp_historical_db_query_stride", 1)),
                        self.hist_db_query_updated_this_tick_history[i]
                        if i < len(self.hist_db_query_updated_this_tick_history)
                        else int(getattr(self, "hist_db_query_updated_this_tick", 0)),
                        self.hist_db_query_reused_history[i]
                        if i < len(self.hist_db_query_reused_history)
                        else 0,
                        self.hist_db_query_counter_history[i]
                        if i < len(self.hist_db_query_counter_history)
                        else 0,
                        int(bool(self.gp_historical_db_preflight_enabled)),
                        int(bool(self.gp_historical_db_preflight_required)),
                        self.gp_historical_db_preflight_mode,
                        self.hist_db_preflight_phase_history[i]
                        if i < len(self.hist_db_preflight_phase_history)
                        else str(getattr(self, "hist_db_preflight_phase", "disabled")),
                        self.hist_db_preflight_pass_history[i]
                        if i < len(self.hist_db_preflight_pass_history)
                        else int(getattr(self, "hist_db_preflight_pass", 0)),
                        self.hist_db_preflight_active_allowed_history[i]
                        if i < len(self.hist_db_preflight_active_allowed_history)
                        else int(getattr(self, "hist_db_preflight_active_allowed", 0)),
                        self.hist_db_preflight_sample_count_history[i]
                        if i < len(self.hist_db_preflight_sample_count_history)
                        else int(getattr(self, "hist_db_preflight_sample_count", 0)),
                        self.hist_db_preflight_pass_ratio_history[i]
                        if i < len(self.hist_db_preflight_pass_ratio_history)
                        else float(getattr(self, "hist_db_preflight_pass_ratio", 0.0)),
                        self.hist_db_preflight_nearest_mean_history[i]
                        if i < len(self.hist_db_preflight_nearest_mean_history)
                        else float(getattr(self, "hist_db_preflight_nearest_mean", 0.0)),
                        self.hist_db_preflight_nearest_p95_history[i]
                        if i < len(self.hist_db_preflight_nearest_p95_history)
                        else float(getattr(self, "hist_db_preflight_nearest_p95", 0.0)),
                        self.hist_db_preflight_nearest_max_history[i]
                        if i < len(self.hist_db_preflight_nearest_max_history)
                        else float(getattr(self, "hist_db_preflight_nearest_max", 0.0)),
                        self.hist_db_runtime_fallback_used_history[i]
                        if i < len(self.hist_db_runtime_fallback_used_history)
                        else int(getattr(self, "hist_db_runtime_fallback_used", 0)),
                    ])

                    hist_soft_valid = (
                        int(self.hist_soft_valid_history[i])
                        if i < len(self.hist_soft_valid_history)
                        else int(self.hist_soft_valid)
                    )
                    hist_soft_nearest_distance = (
                        float(self.hist_soft_nearest_distance_history[i])
                        if i < len(self.hist_soft_nearest_distance_history)
                        else float(self.hist_soft_nearest_distance)
                    )
                    hist_soft_raw_w_hist = (
                        float(self.hist_soft_raw_w_hist_history[i])
                        if i < len(self.hist_soft_raw_w_hist_history)
                        else float(self.hist_soft_raw_w_hist)
                    )
                    hist_soft_norm_w_local = (
                        float(self.hist_soft_norm_w_local_history[i])
                        if i < len(self.hist_soft_norm_w_local_history)
                        else float(self.hist_soft_norm_w_local)
                    )
                    hist_soft_norm_w_cloud = (
                        float(self.hist_soft_norm_w_cloud_history[i])
                        if i < len(self.hist_soft_norm_w_cloud_history)
                        else float(self.hist_soft_norm_w_cloud)
                    )
                    hist_soft_norm_w_hist = (
                        float(self.hist_soft_norm_w_hist_history[i])
                        if i < len(self.hist_soft_norm_w_hist_history)
                        else float(self.hist_soft_norm_w_hist)
                    )
                    row.extend([
                        int(bool(self.gp_historical_soft_shadow_enabled)),
                        hist_soft_valid,
                        int(bool(self.gp_online_update_enabled)),
                        self.gp_historical_soft_alpha,
                        self.gp_historical_soft_distance_threshold,
                        self.gp_historical_soft_online_scale,
                        self.gp_historical_soft_non_online_scale,
                        hist_soft_nearest_distance,
                        hist_soft_raw_w_hist,
                        hist_soft_norm_w_local,
                        hist_soft_norm_w_cloud,
                        hist_soft_norm_w_hist,
                    ])

                    if i < len(self.hist_soft_pred_history):
                        row.extend(self.hist_soft_pred_history[i])
                    else:
                        row.extend([0.0] * 7)

                    if i < len(self.hist_soft_delta_vs_local_cloud_history):
                        row.extend(self.hist_soft_delta_vs_local_cloud_history[i])
                    else:
                        row.extend([0.0] * 7)

                    row.extend([
                        self.run_name,
                        self.control_frequency,
                        self.delay_steps,
                        self.data_output_dir,
                    ])

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

        self._finish_csv_save_timing(csv_save_start)


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
