#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from custom_msgs.msg import TaskSpaceCommand, StateParameter
from custom_msgs.srv import JointPositionAdjust, GetFutureTrajectory
from std_msgs.msg import Header, Bool
import json
import numpy as np
import time
from pathlib import Path
from rclpy.duration import Duration
from std_msgs.msg import String
from py_controllers.session_anchor_utils import (
    parse_vec3_parameter,
    read_anchor_vec3,
    compute_anchor_internal_residuals,
)
from py_controllers.session_relative_config import (
    DEFAULT_SESSION_RELATIVE_ANCHOR_DELTA_LIMIT_MODE,
    normalize_anchor_delta_limit_mode,
)

# session_relative 模式下，平移后的轨迹起点必须与 JSON 里的
# session_trajectory_start_xyz 一致；超过该容差说明控制器写 anchor 时用的
# nominal 轨迹起点和本节点的实际几何不一致（例如 multisine 参数被改过），
# 此时拒绝启用轨迹，避免两个节点用不同的起点。
SESSION_ANCHOR_START_CONSISTENCY_TOLERANCE_M = 0.010
SESSION_ANCHOR_NOMINAL_GEOMETRY_TOLERANCE_M = 0.005


class TrajectoryPublisher(Node):
    
    def __init__(self):
        super().__init__('trajectory_publisher')
        
        # publish on /task_space_command
        self.trajectory_publisher = self.create_publisher(
            TaskSpaceCommand, '/task_space_command', 10)
        
        # publish on /data_recording_enabled to inform other nodes when to start recording
        self.data_recording_publisher = self.create_publisher(
            Bool, '/data_recording_enabled', 10)

        self.declare_parameter('control_frequency', 100.0)
        self.legacy_control_frequency = float(self.get_parameter('control_frequency').value)
        if not np.isfinite(self.legacy_control_frequency) or self.legacy_control_frequency <= 0.0:
            self.get_logger().warning(
                f'Invalid control_frequency={self.legacy_control_frequency}; falling back to 100.0 Hz.'
            )
            self.legacy_control_frequency = 100.0

        self.declare_parameter('trajectory_publish_rate', self.legacy_control_frequency)
        self.trajectory_publish_rate = float(
            self.get_parameter('trajectory_publish_rate').value
        )
        if not np.isfinite(self.trajectory_publish_rate) or self.trajectory_publish_rate <= 0.0:
            self.get_logger().warning(
                f'Invalid trajectory_publish_rate={self.trajectory_publish_rate}; '
                f'falling back to control_frequency={self.legacy_control_frequency} Hz.'
            )
            self.trajectory_publish_rate = self.legacy_control_frequency

        self.control_frequency = self.trajectory_publish_rate
        self.timer = self.create_timer(1.0 / self.trajectory_publish_rate, self.timer_callback)

        # subscribe to /state_parameter to get robot current state
        self.state_subscription = self.create_subscription(
            StateParameter, '/state_parameter', self.stateCallback, 10)
        
        # service server for joint position adjustment
        self.joint_position_service = self.create_service(
            JointPositionAdjust, '/joint_position_adjust', self.joint_position_callback)
        
        self.future_traj_srv = self.create_service(
            GetFutureTrajectory,
            '/future_task_space',
            self.future_traj_callback
        )


        # circle trajectory parameters
        self.declare_parameter('circle_radius', 0.05)   # circle radius (meter)
        self.declare_parameter('circle_frequency', 0.1) # circle motion frequency (Hz)
        self.declare_parameter('circle_center_x', 0.3)  # circle center x coordinate
        self.declare_parameter('circle_center_y', 0.0)  # circle center y coordinate
        self.declare_parameter('circle_center_z', 0.65) # circle center z coordinate
        self.declare_parameter('anchor_trajectory_start_to_current_pose', False)
        # session_relative：整条轨迹（center + start）按 session anchor JSON 里
        # 的 anchor_delta 整体平移；形状/半径/频率/相位不变。默认 fixed_absolute
        # 保持旧的固定绝对几何，向后兼容。
        self.declare_parameter('trajectory_reference_mode', 'fixed_absolute')
        self.declare_parameter('session_anchor_path', '')
        self.declare_parameter('session_relative_apply_to_trajectory_center', True)
        self.declare_parameter('session_relative_max_anchor_delta_m', 0.250)
        # anchor_delta norm 超限策略（refuse|warn|off），默认 refuse 保持旧
        # hard-refuse 行为；与 cartesian 节点共用 session_relative_config 的
        # 默认值与归一化，保证两个节点语义一致。
        self.declare_parameter(
            'session_relative_anchor_delta_limit_mode',
            DEFAULT_SESSION_RELATIVE_ANCHOR_DELTA_LIMIT_MODE,
        )
        # launch/runner 以 STRING 传入这两个 vec3（ParameterValue value_type=str），
        # 但默认值是 list[float] 会让 rclpy 推断为 DOUBLE_ARRAY，导致 STRING 覆盖时
        # 抛 InvalidParameterTypeException。用 dynamic_typing 同时接受 STRING 与
        # DOUBLE_ARRAY；实际解析统一走 _parse_vec3_parameter（支持 JSON 字符串与
        # list[float]，并校验恰好 3 个有限数）。
        self.declare_parameter(
            'session_relative_nominal_trajectory_start_xyz',
            [0.3077306122468523, 0.043799833015107294, 0.6648721535244662],
            ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter(
            'session_relative_nominal_circle_center_xyz',
            [0.3, 0.0, 0.65],
            ParameterDescriptor(dynamic_typing=True),
        )
        # Stage 3A default-off：planar_circle 保持 Stage 1 / Stage 2A 的平面圆轨迹行为。
        # z_modulated_circle 只在显式设置 trajectory_mode 时启用。
        self.declare_parameter('trajectory_mode', 'planar_circle')
        # 默认 0.0，避免未显式选择 Stage 3A 时改变已有平面轨迹。
        self.declare_parameter('z_amplitude', 0.0)
        self.declare_parameter('z_frequency_multiplier', 0.5)

        # GOAL2-B spatial multisine trajectory. It only changes /task_space_command.
        self.declare_parameter('goal1_multisine_x_primary_amplitude', 0.040)
        self.declare_parameter('goal1_multisine_x_secondary_amplitude', 0.012)
        self.declare_parameter('goal1_multisine_y_primary_amplitude', 0.035)
        self.declare_parameter('goal1_multisine_y_secondary_amplitude', 0.012)
        self.declare_parameter('goal1_multisine_z_primary_amplitude', 0.030)
        self.declare_parameter('goal1_multisine_z_secondary_amplitude', 0.010)
        self.declare_parameter('goal1_multisine_x_primary_frequency_multiplier', 1.0)
        self.declare_parameter('goal1_multisine_x_secondary_frequency_multiplier', 2.1)
        self.declare_parameter('goal1_multisine_y_primary_frequency_multiplier', 1.3)
        self.declare_parameter('goal1_multisine_y_secondary_frequency_multiplier', 2.4)
        self.declare_parameter('goal1_multisine_z_primary_frequency_multiplier', 0.7)
        self.declare_parameter('goal1_multisine_z_secondary_frequency_multiplier', 1.9)
        self.declare_parameter('goal1_multisine_phi_x2', 0.7)
        self.declare_parameter('goal1_multisine_phi_y1', 0.4)
        self.declare_parameter('goal1_multisine_phi_y2', 1.3)
        self.declare_parameter('goal1_multisine_phi_z1', 0.2)
        self.declare_parameter('goal1_multisine_phi_z2', 1.1)

        self.radius = self.get_parameter('circle_radius').value
        self.frequency = self.get_parameter('circle_frequency').value
        self.center_x = self.get_parameter('circle_center_x').value
        self.center_y = self.get_parameter('circle_center_y').value
        self.center_z = self.get_parameter('circle_center_z').value
        self.anchor_trajectory_start_to_current_pose = bool(
            self.get_parameter('anchor_trajectory_start_to_current_pose').value
        )
        self.trajectory_reference_mode = str(
            self.get_parameter('trajectory_reference_mode').value
        ).strip().lower()
        self.session_anchor_path = str(
            self.get_parameter('session_anchor_path').value
        ).strip()
        self.session_relative_apply_to_trajectory_center = bool(
            self.get_parameter('session_relative_apply_to_trajectory_center').value
        )
        self.session_relative_max_anchor_delta_m = float(
            self.get_parameter('session_relative_max_anchor_delta_m').value
        )
        # 非法值在这里 fail fast（构造期抛 ValueError → 节点拒绝启动）。
        self.session_relative_anchor_delta_limit_mode = (
            normalize_anchor_delta_limit_mode(
                self.get_parameter(
                    'session_relative_anchor_delta_limit_mode'
                ).value
            )
        )
        self.session_relative_nominal_trajectory_start = (
            self._parse_vec3_parameter(
                self.get_parameter(
                    'session_relative_nominal_trajectory_start_xyz'
                ).value,
                'session_relative_nominal_trajectory_start_xyz',
            )
        )
        self.session_relative_nominal_circle_center = (
            self._parse_vec3_parameter(
                self.get_parameter(
                    'session_relative_nominal_circle_center_xyz'
                ).value,
                'session_relative_nominal_circle_center_xyz',
            )
        )
        if self.trajectory_reference_mode not in ('fixed_absolute', 'session_relative'):
            self.get_logger().error(
                f"Unsupported trajectory_reference_mode "
                f"'{self.trajectory_reference_mode}'. "
                "Supported: fixed_absolute, session_relative."
            )
            raise ValueError(
                f"Unsupported trajectory_reference_mode: "
                f"{self.trajectory_reference_mode}"
            )
        if self.trajectory_reference_mode == 'session_relative':
            if not self.session_anchor_path:
                raise ValueError(
                    "trajectory_reference_mode=session_relative requires a "
                    "non-empty session_anchor_path; refusing to start."
                )
            if self.anchor_trajectory_start_to_current_pose:
                # 两种平移机制叠加会让每次运行的轨迹漂移，破坏可比性。
                raise ValueError(
                    "anchor_trajectory_start_to_current_pose=true conflicts "
                    "with trajectory_reference_mode=session_relative; the "
                    "session anchor JSON is the only allowed shift source."
                )
            if not self.session_relative_apply_to_trajectory_center:
                raise ValueError(
                    "trajectory_reference_mode=session_relative requires "
                    "session_relative_apply_to_trajectory_center=true so the "
                    "whole trajectory center/start are shifted by the same "
                    "anchor_delta; refusing to start."
                )
            if (
                not np.isfinite(self.session_relative_max_anchor_delta_m)
                or self.session_relative_max_anchor_delta_m <= 0.0
            ):
                raise ValueError(
                    "session_relative_max_anchor_delta_m must be a positive "
                    f"finite value, got {self.session_relative_max_anchor_delta_m}."
                )
        self.session_anchor_applied = False
        self._session_anchor_error_logged = False
        self.trajectory_mode = self.get_parameter('trajectory_mode').value
        self.z_amplitude = self.get_parameter('z_amplitude').value
        self.z_frequency_multiplier = self.get_parameter('z_frequency_multiplier').value
        self.goal1_multisine_x_primary_amplitude = self.get_parameter('goal1_multisine_x_primary_amplitude').value
        self.goal1_multisine_x_secondary_amplitude = self.get_parameter('goal1_multisine_x_secondary_amplitude').value
        self.goal1_multisine_y_primary_amplitude = self.get_parameter('goal1_multisine_y_primary_amplitude').value
        self.goal1_multisine_y_secondary_amplitude = self.get_parameter('goal1_multisine_y_secondary_amplitude').value
        self.goal1_multisine_z_primary_amplitude = self.get_parameter('goal1_multisine_z_primary_amplitude').value
        self.goal1_multisine_z_secondary_amplitude = self.get_parameter('goal1_multisine_z_secondary_amplitude').value
        self.goal1_multisine_x_primary_frequency_multiplier = self.get_parameter('goal1_multisine_x_primary_frequency_multiplier').value
        self.goal1_multisine_x_secondary_frequency_multiplier = self.get_parameter('goal1_multisine_x_secondary_frequency_multiplier').value
        self.goal1_multisine_y_primary_frequency_multiplier = self.get_parameter('goal1_multisine_y_primary_frequency_multiplier').value
        self.goal1_multisine_y_secondary_frequency_multiplier = self.get_parameter('goal1_multisine_y_secondary_frequency_multiplier').value
        self.goal1_multisine_z_primary_frequency_multiplier = self.get_parameter('goal1_multisine_z_primary_frequency_multiplier').value
        self.goal1_multisine_z_secondary_frequency_multiplier = self.get_parameter('goal1_multisine_z_secondary_frequency_multiplier').value
        self.goal1_multisine_phi_x2 = self.get_parameter('goal1_multisine_phi_x2').value
        self.goal1_multisine_phi_y1 = self.get_parameter('goal1_multisine_phi_y1').value
        self.goal1_multisine_phi_y2 = self.get_parameter('goal1_multisine_phi_y2').value
        self.goal1_multisine_phi_z1 = self.get_parameter('goal1_multisine_phi_z1').value
        self.goal1_multisine_phi_z2 = self.get_parameter('goal1_multisine_phi_z2').value
        self.supported_trajectory_modes = (
            'planar_circle',
            'z_modulated_circle',
            'goal1_spatial_multisine',
        )

        if self.trajectory_mode not in self.supported_trajectory_modes:
            # 真实机器人上不静默 fallback，避免参数拼写错误导致运行了非预期轨迹。
            self.get_logger().error(
                f"Unsupported trajectory_mode '{self.trajectory_mode}'. "
                f"Supported modes: {self.supported_trajectory_modes}"
            )
            raise ValueError(f"Unsupported trajectory_mode: {self.trajectory_mode}")

        # transition parameters to reach the start point of trajectory smoothly
        # 'initial' means after the robot joint position is adjusted
        self.robot_initial_x = None
        self.robot_initial_y = None
        self.robot_initial_z = None
        self.robot_initial_received = False
        self.declare_parameter('transition_duration', 3.0)  # time to reach start point (s)
        self.declare_parameter('use_transition', True)      
        self.declare_parameter('trajectory_start_distance_warn_m', 0.03)
        self.declare_parameter('trajectory_start_distance_refuse_m', 0.12)
        self.declare_parameter('trajectory_start_distance_guard_enabled', True)
        self.declare_parameter('trajectory_max_cartesian_step_m', 0.0)
        self.transition_duration = self.get_parameter('transition_duration').value
        self.use_transition = self.get_parameter('use_transition').value
        self.trajectory_start_distance_warn_m = float(
            self.get_parameter('trajectory_start_distance_warn_m').value
        )
        self.trajectory_start_distance_refuse_m = float(
            self.get_parameter('trajectory_start_distance_refuse_m').value
        )
        self.trajectory_start_distance_guard_enabled = bool(
            self.get_parameter('trajectory_start_distance_guard_enabled').value
        )
        self.trajectory_max_cartesian_step_m = float(
            self.get_parameter('trajectory_max_cartesian_step_m').value
        )
        
        self.trajectory_enabled = False         # flag controlled by service
        
        self.start_time = self.get_clock().now()
        self.transition_start_time = None
        self.transition_start_position = None
        self.transition_target_position = None
        self.last_transition_command_position = None
        self.last_transition_command_time = None
        self.transition_step_clamp_logged = False
        self.transition_complete = False        # flag indicating the completion of moving to the start point of trajectory
        self.trajectory_start_anchored = False   # true after center/start point is anchored to measured EE pose
        
        # get start point of trajectory
        trajectory_start, _, _ = self._compute_task_space_trajectory(0.0)
        self.trajectory_start_x = trajectory_start[0]
        self.trajectory_start_y = trajectory_start[1]
        self.trajectory_start_z = trajectory_start[2]
        # session_relative 模式下平移前的名义几何，记录下来用于日志和一致性检查。
        self.nominal_center_xyz = np.array(
            [self.center_x, self.center_y, self.center_z], dtype=float
        )
        self.nominal_trajectory_start_xyz = np.array(
            [self.trajectory_start_x, self.trajectory_start_y, self.trajectory_start_z],
            dtype=float,
        )

        if self.trajectory_reference_mode == 'session_relative':
            nominal_start_mismatch = float(np.linalg.norm(
                self.nominal_trajectory_start_xyz
                - self.session_relative_nominal_trajectory_start
            ))
            nominal_center_mismatch = float(np.linalg.norm(
                self.nominal_center_xyz
                - self.session_relative_nominal_circle_center
            ))
            if (
                nominal_start_mismatch
                > SESSION_ANCHOR_NOMINAL_GEOMETRY_TOLERANCE_M
                or nominal_center_mismatch
                > SESSION_ANCHOR_NOMINAL_GEOMETRY_TOLERANCE_M
            ):
                raise ValueError(
                    "session_relative nominal geometry parameters do not "
                    "match trajectory_publisher's computed nominal geometry "
                    f"(start mismatch {nominal_start_mismatch:.4f} m, "
                    f"center mismatch {nominal_center_mismatch:.4f} m > "
                    f"{SESSION_ANCHOR_NOMINAL_GEOMETRY_TOLERANCE_M:.3f} m). "
                    "Refusing to start."
                )
            anchor_file = Path(self.session_anchor_path).expanduser()
            if anchor_file.is_file():
                # load 复用：anchor JSON 已存在，启动时立即校验并平移几何。
                # 任何校验失败直接抛异常 fail closed，不发布任何轨迹指令。
                self._load_and_apply_session_anchor()
            else:
                self.get_logger().warn(
                    '[TrajectoryPublisher] trajectory_reference_mode='
                    'session_relative: anchor JSON '
                    f"'{anchor_file}' does not exist yet (capture_first). "
                    'Trajectory stays disabled until the controller captures '
                    'and saves the session anchor; it will be loaded when the '
                    'joint position adjust service is called.'
                )

        # Ablation parameters
        self.gp_mode_pub = self.create_publisher(String, "/gp_mode", 10)
        # self.modes = ["none", "local", "cloud", "fusion", "history_fusion"]
        self.modes = ["local"]
        self.current_mode_index = 0
        self.period = 1.0 / self.frequency
        self.last_round = -1

        self.declare_parameter("rounds_per_mode", 6)
        self.rounds_per_mode = self.get_parameter("rounds_per_mode").value

        self.declare_parameter("max_rounds", 6)
        self.max_rounds = self.get_parameter("max_rounds").value

        self.shutdown_pub = self.create_publisher(Bool, "/shutdown_control", 10)

        # Post-run return coordination: when enabled, keep this node alive after
        # the final round so the launch-level on_exit=Shutdown does not kill the
        # controller while it slowly returns to the session home.
        self.declare_parameter('post_run_return_wait_enabled', False)
        self.declare_parameter('post_run_return_wait_timeout_sec', 90.0)
        self.post_run_return_wait_enabled = bool(
            self.get_parameter('post_run_return_wait_enabled').value
        )
        self.post_run_return_wait_timeout_sec = float(
            self.get_parameter('post_run_return_wait_timeout_sec').value
        )
        if (
            not np.isfinite(self.post_run_return_wait_timeout_sec)
            or self.post_run_return_wait_timeout_sec <= 0.0
        ):
            self.get_logger().warning(
                'Invalid post_run_return_wait_timeout_sec='
                f'{self.post_run_return_wait_timeout_sec}; falling back to 90.0 s.'
            )
            self.post_run_return_wait_timeout_sec = 90.0
        self.run_finished = False
        self.run_finished_time = None
        self.post_run_return_complete_received = False
        self.post_run_return_complete_sub = self.create_subscription(
            Bool,
            '/post_run_return_complete',
            self.post_run_return_complete_callback,
            10,
        )


        self.get_logger().info('Trajectory publisher node started')
        self.get_logger().info(
            f'Publishing trajectory at {self.trajectory_publish_rate:.1f} Hz '
            f'(legacy control_frequency={self.legacy_control_frequency:.1f} Hz)'
        )
        self.get_logger().info(f'Trajectory mode: {self.trajectory_mode}')
        self.get_logger().info(f'Circle radius: {self.radius} m, frequency: {self.frequency} Hz')
        self.get_logger().info(f'Circle center: ({self.center_x}, {self.center_y}, {self.center_z})')
        self.get_logger().info(
            f'Z modulation amplitude: {self.z_amplitude} m, '
            f'frequency multiplier: {self.z_frequency_multiplier}'
        )
        if self.trajectory_mode == 'goal1_spatial_multisine':
            self.get_logger().info(
                'GOAL2-B spatial multisine trajectory active; '
                'this changes task-space position commands only.'
            )
        self.get_logger().info(f'Trajectory start point: ({self.trajectory_start_x:.3f}, {self.trajectory_start_y:.3f}, {self.trajectory_start_z:.3f})')
        self.get_logger().info(
            f'Anchor trajectory start to current pose: {self.anchor_trajectory_start_to_current_pose}'
        )
        self.get_logger().info(
            f'trajectory_reference_mode: {self.trajectory_reference_mode}, '
            f"session_anchor_path: '{self.session_anchor_path}', "
            f'session_anchor_applied: {self.session_anchor_applied}'
        )
        if self.use_transition:
            self.get_logger().info(f'Transition duration: {self.transition_duration} s')
        self.get_logger().info(
            f'Trajectory start distance guard: enabled={self.trajectory_start_distance_guard_enabled}, '
            f'warn={self.trajectory_start_distance_warn_m:.3f} m, '
            f'refuse={self.trajectory_start_distance_refuse_m:.3f} m'
        )
        if self.trajectory_max_cartesian_step_m > 0.0:
            self.get_logger().info(
                f'Transition Cartesian step clamp enabled: '
                f'{self.trajectory_max_cartesian_step_m:.4f} m per publish'
            )
        self.get_logger().info('Waiting for joint position adjustment service call to enable trajectory...')
    
    def joint_position_callback(self, request, response):
        """Service callback for joint position adjustment"""
        try:
            self.get_logger().info(f'Received joint position adjustment request')
            self.get_logger().info(f'q_des: {request.q_des}')
            self.get_logger().info(f'dq_des: {request.dq_des}')

            if (
                self.trajectory_reference_mode == 'session_relative'
                and not self.session_anchor_applied
            ):
                # capture_first：控制器在发布任何 startup 力矩之前就已写好
                # anchor JSON，所以此时文件应当存在。加载/校验失败一律拒绝
                # 启用轨迹（trajectory_enabled 保持 False，不发布任何指令）。
                try:
                    self._load_and_apply_session_anchor()
                except Exception as e:
                    if not self._session_anchor_error_logged:
                        self._session_anchor_error_logged = True
                        self.get_logger().error(
                            '[TrajectoryPublisher] Refusing to enable '
                            f'trajectory: session anchor load failed: {e}'
                        )
                    response.success = False
                    response.message = (
                        f'session anchor load failed: {e}'
                    )
                    return response

            self.trajectory_enabled = True
            
            # reset timing for trajectory
            self.start_time = self.get_clock().now()
            self.transition_start_time = None
            self.transition_start_position = None
            self.transition_target_position = None
            self.last_transition_command_position = None
            self.last_transition_command_time = None
            self.transition_step_clamp_logged = False
            self.transition_complete = False
            self.robot_initial_received = False
            self.trajectory_start_anchored = False
            
            response.success = True
            response.message = "Trajectory enabled successfully"
            self.get_logger().info('Trajectory enabled via service call')
            
        except Exception as e:
            self.get_logger().error(f'Error in joint position callback: {str(e)}')
            response.success = False
            response.message = f"Error: {str(e)}"
            
        return response

    @staticmethod
    def _parse_vec3_parameter(value, name):
        # 纯解析逻辑抽到 session_anchor_utils.parse_vec3_parameter；解析/校验
        # 语义（STRING JSON 与 list 兼容、恰好 3 个有限数）保持不变。
        return parse_vec3_parameter(value, name)

    @staticmethod
    def _session_anchor_vec3(payload, key):
        """Read one finite 3-vector field from the anchor JSON; raise on failure."""
        # 校验逻辑抽到 session_anchor_utils.read_anchor_vec3；保留原有的
        # "session anchor " 错误前缀，错误文案逐字节不变。
        return read_anchor_vec3(payload, key, "session anchor ")

    def _load_and_apply_session_anchor(self):
        """Load the session anchor JSON and shift the whole trajectory by anchor_delta.

        只平移（center + start 同步移动 anchor_delta），不改变形状/半径/频率/
        相位。任何字段缺失、模式不匹配、delta 超限或起点一致性失败都抛
        ValueError，由调用方拒绝启用轨迹。
        """
        anchor_file = Path(self.session_anchor_path).expanduser()
        if not anchor_file.is_file():
            raise ValueError(
                f"session anchor JSON not found: '{anchor_file}' "
                "(controller capture may not have completed)."
            )
        try:
            payload = json.loads(anchor_file.read_text())
        except Exception as e:
            raise ValueError(
                f"failed to parse session anchor JSON '{anchor_file}': {e}"
            )
        if not isinstance(payload, dict):
            raise ValueError(
                f"session anchor JSON '{anchor_file}' is not an object."
            )

        file_mode = str(
            payload.get('trajectory_reference_mode', '')
        ).strip().lower()
        if file_mode != 'session_relative':
            raise ValueError(
                "session anchor JSON trajectory_reference_mode="
                f"'{file_mode}' does not match node mode 'session_relative'."
            )
        for key in ('version', 'created_at', 'source', 'notes'):
            if key not in payload:
                raise ValueError(
                    f"session anchor JSON '{anchor_file}' is missing "
                    f"required field '{key}'."
                )
        try:
            version = int(payload.get('version'))
        except (TypeError, ValueError):
            raise ValueError(
                f"session anchor JSON '{anchor_file}' field 'version' must "
                "be an integer."
            )
        if version < 2:
            raise ValueError(
                f"session anchor JSON '{anchor_file}' version={version} is "
                "too old for session_relative anchors; re-capture the anchor."
            )

        session_start = self._session_anchor_vec3(
            payload, 'session_trajectory_start_xyz'
        )
        ee_pose = self._session_anchor_vec3(payload, 'ee_pose_xyz')
        anchor_delta = self._session_anchor_vec3(payload, 'anchor_delta_xyz')
        shifted_center = self._session_anchor_vec3(
            payload, 'shifted_circle_center_xyz'
        )
        nominal_start_json = self._session_anchor_vec3(
            payload, 'nominal_trajectory_start_xyz'
        )
        nominal_center_json = self._session_anchor_vec3(
            payload, 'nominal_circle_center_xyz'
        )
        self._session_anchor_vec3(payload, 'nominal_fixed_start_xyz')

        # 内部自洽残差计算抽到 session_anchor_utils.compute_anchor_internal_residuals
        # （与 cartesian 共用同一份设计 B 不变量数学）；阈值判断与错误文案保持本节点
        # 原样不变。
        internal_tol = 1e-6
        ee_pose_residual, start_residual, center_residual = (
            compute_anchor_internal_residuals(
                ee_pose, session_start, nominal_start_json,
                nominal_center_json, shifted_center, anchor_delta,
            )
        )
        if (
            ee_pose_residual > internal_tol
            or start_residual > internal_tol
            or center_residual > internal_tol
        ):
            raise ValueError(
                "session anchor JSON is internally inconsistent "
                f"(ee_pose_residual={ee_pose_residual:.2e} m, "
                f"start_residual={start_residual:.2e} m, "
                f"center_residual={center_residual:.2e} m)."
            )

        anchor_delta_norm = float(np.linalg.norm(anchor_delta))
        # anchor_delta norm 策略与 cartesian 一致：refuse 保持旧 hard-refuse；
        # warn 只告警并继续（floating anchor）；off 跳过该 norm 检查。上方的
        # 内部自洽残差与下方的 nominal geometry / 起点一致性检查不受影响。
        if (
            self.session_relative_anchor_delta_limit_mode != 'off'
            and anchor_delta_norm > self.session_relative_max_anchor_delta_m
        ):
            if self.session_relative_anchor_delta_limit_mode == 'refuse':
                raise ValueError(
                    f"anchor_delta norm {anchor_delta_norm:.4f} m exceeds "
                    "session_relative_max_anchor_delta_m="
                    f"{self.session_relative_max_anchor_delta_m:.4f} m "
                    "(session_relative_anchor_delta_limit_mode=refuse)."
                )
            self.get_logger().warn(
                '[TrajectoryPublisher] anchor_delta norm '
                f"{anchor_delta_norm:.4f} m exceeds "
                "session_relative_max_anchor_delta_m="
                f"{self.session_relative_max_anchor_delta_m:.4f} m but "
                "session_relative_anchor_delta_limit_mode=warn; continuing "
                "with floating session anchor."
            )

        configured_start_mismatch = float(np.linalg.norm(
            nominal_start_json - self.session_relative_nominal_trajectory_start
        ))
        configured_center_mismatch = float(np.linalg.norm(
            nominal_center_json - self.session_relative_nominal_circle_center
        ))
        actual_start_mismatch = float(np.linalg.norm(
            nominal_start_json - self.nominal_trajectory_start_xyz
        ))
        actual_center_mismatch = float(np.linalg.norm(
            nominal_center_json - self.nominal_center_xyz
        ))
        geometry_tol = SESSION_ANCHOR_NOMINAL_GEOMETRY_TOLERANCE_M
        if (
            configured_start_mismatch > geometry_tol
            or configured_center_mismatch > geometry_tol
            or actual_start_mismatch > geometry_tol
            or actual_center_mismatch > geometry_tol
        ):
            raise ValueError(
                "session anchor nominal geometry does not match this run "
                f"(configured_start_mismatch={configured_start_mismatch:.4f} m, "
                "configured_center_mismatch="
                f"{configured_center_mismatch:.4f} m, "
                f"actual_start_mismatch={actual_start_mismatch:.4f} m, "
                f"actual_center_mismatch={actual_center_mismatch:.4f} m > "
                f"{geometry_tol:.3f} m). Re-capture the anchor for the "
                "current trajectory geometry."
            )

        old_center = self.nominal_center_xyz.copy()
        self.center_x = float(shifted_center[0])
        self.center_y = float(shifted_center[1])
        self.center_z = float(shifted_center[2])
        new_start, _, _ = self._compute_task_space_trajectory(0.0)
        new_start = np.asarray(new_start[:3], dtype=float)
        start_consistency_m = float(np.linalg.norm(new_start - session_start))
        if start_consistency_m > SESSION_ANCHOR_START_CONSISTENCY_TOLERANCE_M:
            # 回滚 center，保持未平移状态，拒绝启用轨迹。
            self.center_x = float(old_center[0])
            self.center_y = float(old_center[1])
            self.center_z = float(old_center[2])
            raise ValueError(
                "shifted trajectory start "
                f"{new_start.tolist()} deviates "
                f"{start_consistency_m:.4f} m from session_trajectory_start "
                f"{session_start.tolist()} (tolerance "
                f"{SESSION_ANCHOR_START_CONSISTENCY_TOLERANCE_M:.3f} m); the "
                "anchor JSON nominal geometry does not match this node's "
                "trajectory parameters."
            )

        self.trajectory_start_x = float(new_start[0])
        self.trajectory_start_y = float(new_start[1])
        self.trajectory_start_z = float(new_start[2])
        self.session_anchor_applied = True
        self.get_logger().warn(
            '[TrajectoryPublisher] Session-relative anchor applied: '
            f"anchor='{anchor_file}', "
            f'nominal_trajectory_start={self.nominal_trajectory_start_xyz.tolist()}, '
            f'nominal_trajectory_start_json={nominal_start_json.tolist()}, '
            f'nominal_circle_center_json={nominal_center_json.tolist()}, '
            f'session_trajectory_start={session_start.tolist()}, '
            f'anchor_delta={anchor_delta.tolist()} '
            f'(norm={anchor_delta_norm:.4f} m), '
            f'nominal_center={old_center.tolist()}, '
            f'shifted_circle_center={shifted_center.tolist()}, '
            f'shifted_trajectory_start={new_start.tolist()}, '
            f'start_consistency={start_consistency_m:.6f} m. '
            'Trajectory shape/radius/frequency/phase unchanged; global '
            'translation only. The start-distance guard now checks the '
            'shifted start.'
        )

    def future_traj_callback(self, request, response):
        t_delay = float(request.t_delay)
        future = self.get_future_task_space(t_delay)  # 你刚才写好的函数

        if future is None:
            # trajectory not ready
            # future trajectory 未 ready 可能高频发生；默认静默，避免真机运行时 stdout I/O 负载。
            response.x_des = [0.0]*6
            response.dx_des = [0.0]*6
            response.ddx_des = [0.0]*6
            return response

        x_des, dx_des, ddx_des = future
        response.x_des = x_des
        response.dx_des = dx_des
        response.ddx_des = ddx_des
        return response
    
    def stateCallback(self, msg):
        """callback function of /state_parameter subscriber"""
        if not self.trajectory_enabled:
            return
            
        if not self.robot_initial_received:
            try:
                # get initial position of robot arm (x, y, z) before transition
                o_t_f_array = np.array(msg.o_t_f, dtype=float)
                o_t_f = o_t_f_array.reshape(4, 4, order='F')       
                self.robot_initial_x = o_t_f[0, 3]
                self.robot_initial_y = o_t_f[1, 3]
                self.robot_initial_z = o_t_f[2, 3]
                current_position = np.array(
                    [self.robot_initial_x, self.robot_initial_y, self.robot_initial_z],
                    dtype=float
                )
                target_position = np.array(
                    [self.trajectory_start_x, self.trajectory_start_y, self.trajectory_start_z],
                    dtype=float
                )

                if (
                    self.anchor_trajectory_start_to_current_pose
                    and self.trajectory_mode == 'goal1_spatial_multisine'
                    and not self.trajectory_start_anchored
                ):
                    # 真机实验 anchor：保持 multisine 轨迹形状不变，只整体平移 center，
                    # 使 t=0 的 trajectory start point 精确对齐当前测得 EE pose。
                    # 这样每次 clear fault / unlock 后的毫米级初始位姿漂移不会造成 transition jump。
                    offset = target_position - np.array(
                        [self.center_x, self.center_y, self.center_z],
                        dtype=float
                    )
                    new_center = current_position - offset
                    self.center_x = float(new_center[0])
                    self.center_y = float(new_center[1])
                    self.center_z = float(new_center[2])
                    self.trajectory_start_x = float(current_position[0])
                    self.trajectory_start_y = float(current_position[1])
                    self.trajectory_start_z = float(current_position[2])
                    target_position = current_position.copy()
                    self.trajectory_start_anchored = True
                    self.get_logger().warn(
                        '[TrajectoryPublisher] Anchored trajectory start to current pose: '
                        f'new_center=({self.center_x:.4f}, {self.center_y:.4f}, {self.center_z:.4f}), '
                        f'new_start=({self.trajectory_start_x:.4f}, {self.trajectory_start_y:.4f}, {self.trajectory_start_z:.4f})'
                    )

                distance_to_start = float(np.linalg.norm(target_position - current_position))
                if self.transition_duration > 0.0:
                    average_speed = distance_to_start / float(self.transition_duration)
                else:
                    average_speed = float('inf')
                
                self.robot_initial_received = True
                self.get_logger().info(f'Robot initial position recorded: ({self.robot_initial_x:.3f}, {self.robot_initial_y:.3f}, {self.robot_initial_z:.3f})')
                if self.trajectory_reference_mode == 'session_relative':
                    self.get_logger().info(
                        '[TrajectoryPublisher] session_relative: '
                        f'distance_to_shifted_start={float(np.linalg.norm(target_position - current_position)):.4f} m '
                        f'(shifted start={target_position.tolist()}); the '
                        'start-distance guard below uses this shifted start, '
                        'not the old nominal trajectory start.'
                    )
                self.get_logger().info(
                    f'[TrajectoryPublisher] Smooth transition start: '
                    f'current=({current_position[0]:.4f}, {current_position[1]:.4f}, {current_position[2]:.4f}), '
                    f'target=({target_position[0]:.4f}, {target_position[1]:.4f}, {target_position[2]:.4f}), '
                    f'distance={distance_to_start:.4f} m, '
                    f'transition_duration={float(self.transition_duration):.3f} s, '
                    f'avg_speed={average_speed:.4f} m/s'
                )

                if self.trajectory_start_distance_guard_enabled:
                    if distance_to_start > self.trajectory_start_distance_refuse_m:
                        self.get_logger().error(
                            f'[TrajectoryPublisher] Refusing trajectory start: '
                            f'distance_to_start={distance_to_start:.4f} m exceeds '
                            f'trajectory_start_distance_refuse_m='
                            f'{self.trajectory_start_distance_refuse_m:.4f} m. '
                            f'Trajectory motion remains disabled.'
                        )
                        self.trajectory_enabled = False
                        self.robot_initial_received = False
                        self.transition_start_time = None
                        self.transition_start_position = None
                        self.transition_target_position = None
                        self.last_transition_command_position = None
                        self.last_transition_command_time = None
                        return

                    if distance_to_start > self.trajectory_start_distance_warn_m:
                        self.get_logger().warning(
                            f'[TrajectoryPublisher] Large trajectory start distance: '
                            f'distance_to_start={distance_to_start:.4f} m exceeds '
                            f'trajectory_start_distance_warn_m='
                            f'{self.trajectory_start_distance_warn_m:.4f} m.'
                        )
                
                # start moving to the start point of trajectory after receiving initial position
                if self.use_transition:
                    # transition 必须从当前 measured EE pose 开始，避免第一拍 desired command 跳到轨迹起点导致速度约束触发。
                    transition_start_now = self.get_clock().now()
                    self.transition_start_position = current_position
                    self.transition_target_position = target_position
                    self.last_transition_command_position = current_position.copy()
                    self.last_transition_command_time = transition_start_now
                    self.transition_start_time = transition_start_now
                    self.get_logger().info('Starting transition to trajectory start point')
                
            except Exception as e:
                self.get_logger().error(f'Error extracting robot initial position: {str(e)}')

    def _compute_transition_command(self, transition_elapsed):
        """Return position, velocity, and acceleration for the smooth start transition."""
        if self.transition_start_position is None or self.transition_target_position is None:
            return None

        duration = float(self.transition_duration)
        if duration <= 0.0:
            s = 1.0
            ds_dt = 0.0
            d2s_dt2 = 0.0
        else:
            t = transition_elapsed / duration
            s = min(max(t, 0.0), 1.0)
            ds_dt = (6.0 * s * (1.0 - s)) / duration
            d2s_dt2 = (6.0 - 12.0 * s) / (duration**2)

        alpha = s * s * (3.0 - 2.0 * s)
        delta = self.transition_target_position - self.transition_start_position
        position = self.transition_start_position + alpha * delta
        velocity = ds_dt * delta
        acceleration = d2s_dt2 * delta
        return position, velocity, acceleration

    def _apply_transition_step_clamp(self, position, velocity, acceleration, current_time):
        """Clamp only the per-cycle Cartesian position step during transition when enabled."""
        max_step = float(self.trajectory_max_cartesian_step_m)
        if max_step <= 0.0 or self.last_transition_command_position is None:
            return position, velocity, acceleration

        step = position - self.last_transition_command_position
        step_norm = float(np.linalg.norm(step))
        if step_norm <= max_step or step_norm <= 0.0:
            return position, velocity, acceleration

        clamped_step = step * (max_step / step_norm)
        clamped_position = self.last_transition_command_position + clamped_step

        if self.last_transition_command_time is not None:
            dt = (current_time - self.last_transition_command_time).nanoseconds / 1e9
        else:
            dt = 0.0

        if dt > 0.0:
            velocity = clamped_step / dt
        else:
            velocity = np.zeros(3)
        acceleration = np.zeros(3)

        if not self.transition_step_clamp_logged:
            self.get_logger().warning(
                f'[TrajectoryPublisher] Transition Cartesian step clamp active: '
                f'requested_step={step_norm:.5f} m, '
                f'clamped_to={max_step:.5f} m.'
            )
            self.transition_step_clamp_logged = True

        return clamped_position, velocity, acceleration

    def _compute_task_space_trajectory(self, t):
        """计算 post-transition 轨迹；live 发布和 /future_task_space 共用，避免预测不一致。"""
        omega = 2.0 * np.pi * self.frequency

        if self.trajectory_mode == 'goal1_spatial_multisine':
            ax1 = float(self.goal1_multisine_x_primary_amplitude)
            ax2 = float(self.goal1_multisine_x_secondary_amplitude)
            ay1 = float(self.goal1_multisine_y_primary_amplitude)
            ay2 = float(self.goal1_multisine_y_secondary_amplitude)
            az1 = float(self.goal1_multisine_z_primary_amplitude)
            az2 = float(self.goal1_multisine_z_secondary_amplitude)

            wx1 = float(self.goal1_multisine_x_primary_frequency_multiplier) * omega
            wx2 = float(self.goal1_multisine_x_secondary_frequency_multiplier) * omega
            wy1 = float(self.goal1_multisine_y_primary_frequency_multiplier) * omega
            wy2 = float(self.goal1_multisine_y_secondary_frequency_multiplier) * omega
            wz1 = float(self.goal1_multisine_z_primary_frequency_multiplier) * omega
            wz2 = float(self.goal1_multisine_z_secondary_frequency_multiplier) * omega

            phi_x2 = float(self.goal1_multisine_phi_x2)
            phi_y1 = float(self.goal1_multisine_phi_y1)
            phi_y2 = float(self.goal1_multisine_phi_y2)
            phi_z1 = float(self.goal1_multisine_phi_z1)
            phi_z2 = float(self.goal1_multisine_phi_z2)

            x = self.center_x + ax1 * np.sin(wx1 * t) + ax2 * np.sin(wx2 * t + phi_x2)
            y = self.center_y + ay1 * np.cos(wy1 * t + phi_y1) + ay2 * np.sin(wy2 * t + phi_y2)
            z = self.center_z + az1 * np.sin(wz1 * t + phi_z1) + az2 * np.sin(wz2 * t + phi_z2)

            dx = ax1 * wx1 * np.cos(wx1 * t) + ax2 * wx2 * np.cos(wx2 * t + phi_x2)
            dy = -ay1 * wy1 * np.sin(wy1 * t + phi_y1) + ay2 * wy2 * np.cos(wy2 * t + phi_y2)
            dz = az1 * wz1 * np.cos(wz1 * t + phi_z1) + az2 * wz2 * np.cos(wz2 * t + phi_z2)

            ddx = -ax1 * wx1**2 * np.sin(wx1 * t) - ax2 * wx2**2 * np.sin(wx2 * t + phi_x2)
            ddy = -ay1 * wy1**2 * np.cos(wy1 * t + phi_y1) - ay2 * wy2**2 * np.sin(wy2 * t + phi_y2)
            ddz = -az1 * wz1**2 * np.sin(wz1 * t + phi_z1) - az2 * wz2**2 * np.sin(wz2 * t + phi_z2)
        else:
            x = self.center_x + self.radius * np.cos(omega * t)
            y = self.center_y + self.radius * np.sin(omega * t)

            dx = -self.radius * omega * np.sin(omega * t)
            dy = self.radius * omega * np.cos(omega * t)

            ddx = -self.radius * omega**2 * np.cos(omega * t)
            ddy = -self.radius * omega**2 * np.sin(omega * t)

            if self.trajectory_mode == 'z_modulated_circle':
                z_omega = self.z_frequency_multiplier * omega
                z = self.center_z + self.z_amplitude * np.sin(z_omega * t)
                dz = self.z_amplitude * z_omega * np.cos(z_omega * t)
                ddz = -self.z_amplitude * z_omega**2 * np.sin(z_omega * t)
            else:
                z = self.center_z
                dz = 0.0
                ddz = 0.0

        x_des = [x, y, z, 0.0, 0.0, 0.0]
        dx_des = [dx, dy, dz, 0.0, 0.0, 0.0]
        ddx_des = [ddx, ddy, ddz, 0.0, 0.0, 0.0]

        return x_des, dx_des, ddx_des
    
    def post_run_return_complete_callback(self, msg):
        if msg.data and not self.post_run_return_complete_received:
            self.post_run_return_complete_received = True
            self.get_logger().info(
                '[TrajectoryPublisher] Controller reported post-run return '
                'complete.'
            )

    def _handle_post_run_return_wait(self):
        """After the final round, wait for controller return before exiting."""
        # 持续广播 recording=False，防止晚加入的订阅者误以为还在录数。
        stop_msg = Bool()
        stop_msg.data = False
        self.data_recording_publisher.publish(stop_msg)

        wait_elapsed = (
            self.get_clock().now() - self.run_finished_time
        ).nanoseconds / 1e9
        if self.post_run_return_complete_received:
            self.get_logger().info(
                '[TrajectoryPublisher] Post-run return complete; shutting down.'
            )
            if rclpy.ok():
                rclpy.shutdown()
            return
        if wait_elapsed > self.post_run_return_wait_timeout_sec:
            self.get_logger().warning(
                '[TrajectoryPublisher] Post-run return wait timed out after '
                f'{self.post_run_return_wait_timeout_sec:.1f} s; shutting down.'
            )
            if rclpy.ok():
                rclpy.shutdown()

    def timer_callback(self):
        """timer callback function at the configured control frequency."""
        if self.run_finished:
            self._handle_post_run_return_wait()
            return

        try:
            # check if joint position adjustment is completed
            if not self.trajectory_enabled:
                return
                
            # wait for initialization of robot position
            if not self.robot_initial_received:
                return
            
            # get time, initialize varaibles
            current_time = self.get_clock().now()
            elapsed_time = (current_time - self.start_time).nanoseconds / 1e9
            x, y, z = 0.0, 0.0, 0.0
            dx, dy, dz = 0.0, 0.0, 0.0
            ddx, ddy, ddz = 0.0, 0.0, 0.0
            
            if self.use_transition and not self.transition_complete:
                # transition: from adjusted robot position to trajectory start point
                transition_elapsed = (current_time - self.transition_start_time).nanoseconds / 1e9
                
                if transition_elapsed >= self.transition_duration:
                    # transition complete, start selected trajectory
                    self.transition_complete = True
                    self.get_logger().info(
                        f'Transition complete, starting trajectory mode: {self.trajectory_mode}'
                    )
                    # reset start time for selected trajectory
                    self.start_time = current_time
                    elapsed_time = 0.0
                else:
                    transition_command = self._compute_transition_command(transition_elapsed)
                    if transition_command is None:
                        return
                    position, velocity, acceleration = transition_command
                    position, velocity, acceleration = self._apply_transition_step_clamp(
                        position, velocity, acceleration, current_time
                    )

                    x, y, z = position[:3]
                    dx, dy, dz = velocity[:3]
                    ddx, ddy, ddz = acceleration[:3]
            
            # selected trajectory after smooth transition
            if self.transition_complete or not self.use_transition:
                x_des, dx_des, ddx_des = self._compute_task_space_trajectory(elapsed_time)
                x, y, z = x_des[:3]
                dx, dy, dz = dx_des[:3]
                ddx, ddy, ddz = ddx_des[:3]
            
            # publish on /task_space_command
            trajectory_msg = TaskSpaceCommand()
            trajectory_msg.header = Header()
            trajectory_msg.header.stamp = current_time.to_msg()
            trajectory_msg.header.frame_id = "base_link"
            trajectory_msg.x_des = [x, y, z, 0.0, 0.0, 0.0]         # position (x, y, z, roll, pitch, yaw)
            trajectory_msg.dx_des = [dx, dy, dz, 0.0, 0.0, 0.0]     # velocity
            trajectory_msg.ddx_des = [ddx, ddy, ddz, 0.0, 0.0, 0.0] # acceleration
            
            self.trajectory_publisher.publish(trajectory_msg)
            if self.use_transition and not self.transition_complete:
                self.last_transition_command_position = np.array([x, y, z], dtype=float)
                self.last_transition_command_time = current_time
            else:
                self.last_transition_command_position = None
                self.last_transition_command_time = None
            
            # publish data recording status
            data_recording_msg = Bool()
            data_recording_msg.data = self.transition_complete or not self.use_transition
            self.data_recording_publisher.publish(data_recording_msg)
            
            if int(elapsed_time * 1000) % 1000 == 0:
                if self.use_transition and not self.transition_complete:
                    transition_elapsed = (current_time - self.transition_start_time).nanoseconds / 1e9
                    self.get_logger().debug(f'Transition phase: t={transition_elapsed:.3f}s, pos=({x:.3f}, {y:.3f}, {z:.3f})')
                else:
                    self.get_logger().debug(
                        f'Trajectory mode {self.trajectory_mode}: '
                        f't={elapsed_time:.3f}s, pos=({x:.3f}, {y:.3f}, {z:.3f})'
                    )
                
        except Exception as e:
            self.get_logger().error(f'Error in trajectory publisher: {str(e)}')
            self.get_logger().error(f'Current state: transition_complete={self.transition_complete}, elapsed_time={elapsed_time}')
        
        # ---------------------------
        #   Round Detection + GP Mode Switch
        # ---------------------------
        if self.transition_complete:   # 只有真正跑圆轨迹才切换模式
            current_round = int(elapsed_time / self.period)

            if current_round != self.last_round:
                self.last_round = current_round

                # === 是否该切换 mode ===
                if current_round > 0 and (current_round % self.rounds_per_mode) == 0:
                    self.current_mode_index = (self.current_mode_index + 1) % len(self.modes)

                    mode_msg = String()
                    mode_msg.data = self.modes[self.current_mode_index]
                    self.gp_mode_pub.publish(mode_msg)

                    self.get_logger().info(
                        f"[TrajectoryPublisher] Switching GP Mode → {mode_msg.data}"
                    )


            # ========== Auto stop here ==========
            total_rounds = self.rounds_per_mode * len(self.modes)

            if current_round >= total_rounds:
                self.get_logger().info(
                    f"Reached total {total_rounds} rounds, stopping trajectory publisher..."
                )

                # Stop recording
                stop_msg = Bool()
                stop_msg.data = False
                self.data_recording_publisher.publish(stop_msg)

                # Notify controller
                shutdown_msg = Bool()
                shutdown_msg.data = True
                self.shutdown_pub.publish(shutdown_msg)

                if self.post_run_return_wait_enabled:
                    # 控制器将执行 post-run return；本节点保持存活，
                    # 避免 launch on_exit=Shutdown 在归位途中杀掉控制器。
                    self.run_finished = True
                    self.run_finished_time = self.get_clock().now()
                    self.trajectory_enabled = False
                    self.get_logger().info(
                        '[TrajectoryPublisher] Waiting up to '
                        f'{self.post_run_return_wait_timeout_sec:.1f} s for '
                        'controller post-run return to session home...'
                    )
                    return

                if rclpy.ok():
                    rclpy.shutdown()
                return


    def get_future_task_space(self, t_delay):
        """
        计算 t_delay 秒之后的期望 (x, dx, ddx)，不改变当前节点内部状态。

        返回:
            (x_des, dx_des, ddx_des)
            其中每个都是长度 6 的 list:
            [x, y, z, roll, pitch, yaw] / 对应的一阶 / 二阶
        """
        # 还没启用轨迹或者还没收到初始位姿，就没法算未来轨迹
        if not self.trajectory_enabled or not self.robot_initial_received:
            return None

        # 当前 ROS 时间
        now = self.get_clock().now()
        # 未来的 ROS 时间
        future_time = now + Duration(seconds=float(t_delay))

        x = y = z = 0.0
        dx = dy = dz = 0.0
        ddx = ddy = ddz = 0.0

        # 有/没有平滑过渡两种情况分开
        if self.use_transition:
            # 还没开始 transition（一般是还没在 stateCallback 里记录初始位姿）
            if self.transition_start_time is None:
                return None

            # 相对于 transition 开始的时间
            transition_elapsed = (future_time - self.transition_start_time).nanoseconds / 1e9

            if 0.0 <= transition_elapsed < self.transition_duration:
                # —— 还在 smoothstep 过渡阶段 —— #
                transition_command = self._compute_transition_command(transition_elapsed)
                if transition_command is None:
                    return None
                position, velocity, acceleration = transition_command
                x, y, z = position[:3]
                dx, dy, dz = velocity[:3]
                ddx, ddy, ddz = acceleration[:3]

            else:
                # —— 已经过渡完，进入圆轨迹 —— #
                # 圆轨迹的时间从 transition 结束时刻算起：
                t_circle = transition_elapsed - self.transition_duration
                if t_circle < 0.0:
                    t_circle = 0.0

                return self._compute_task_space_trajectory(t_circle)

        else:
            # —— 没有过渡，直接圆轨迹，从 start_time 开始 —— #
            elapsed_time = (future_time - self.start_time).nanoseconds / 1e9
            if elapsed_time < 0.0:
                elapsed_time = 0.0

            return self._compute_task_space_trajectory(elapsed_time)

        x_des  = [x,  y,  z,  0.0, 0.0, 0.0]
        dx_des = [dx, dy, dz, 0.0, 0.0, 0.0]
        ddx_des = [ddx, ddy, ddz, 0.0, 0.0, 0.0]

        return x_des, dx_des, ddx_des

def main(args=None):
    rclpy.init(args=args)
    trajectory_publisher_node = TrajectoryPublisher()
    
    try:
        rclpy.spin(trajectory_publisher_node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            trajectory_publisher_node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
