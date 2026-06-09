#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_msgs.msg import TaskSpaceCommand, StateParameter
from custom_msgs.srv import JointPositionAdjust, GetFutureTrajectory
from std_msgs.msg import Header, Bool
import numpy as np
import time
from rclpy.duration import Duration
from std_msgs.msg import String


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
        self.control_frequency = float(self.get_parameter('control_frequency').value)
        if self.control_frequency <= 0.0:
            self.get_logger().warning(
                f'Invalid control_frequency={self.control_frequency}; falling back to 100.0 Hz.'
            )
            self.control_frequency = 100.0

        self.timer = self.create_timer(1.0 / self.control_frequency, self.timer_callback)

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
        
        # get start point of trajectory
        trajectory_start, _, _ = self._compute_task_space_trajectory(0.0)
        self.trajectory_start_x = trajectory_start[0]
        self.trajectory_start_y = trajectory_start[1]
        self.trajectory_start_z = trajectory_start[2]

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


        self.get_logger().info('Trajectory publisher node started')
        self.get_logger().info(f'Publishing trajectory at {self.control_frequency:.1f} Hz')
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
            
            response.success = True
            response.message = "Trajectory enabled successfully"
            self.get_logger().info('Trajectory enabled via service call')
            
        except Exception as e:
            self.get_logger().error(f'Error in joint position callback: {str(e)}')
            response.success = False
            response.message = f"Error: {str(e)}"
            
        return response
    
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
                distance_to_start = float(np.linalg.norm(target_position - current_position))
                if self.transition_duration > 0.0:
                    average_speed = distance_to_start / float(self.transition_duration)
                else:
                    average_speed = float('inf')
                
                self.robot_initial_received = True
                self.get_logger().info(f'Robot initial position recorded: ({self.robot_initial_x:.3f}, {self.robot_initial_y:.3f}, {self.robot_initial_z:.3f})')
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
    
    def timer_callback(self):
        """timer callback function at the configured control frequency."""
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
        trajectory_publisher_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
