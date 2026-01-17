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
        
        self.timer = self.create_timer(0.001, self.timer_callback)  # publish at 1000 Hz

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
        self.radius = self.get_parameter('circle_radius').value
        self.frequency = self.get_parameter('circle_frequency').value
        self.center_x = self.get_parameter('circle_center_x').value
        self.center_y = self.get_parameter('circle_center_y').value
        self.center_z = self.get_parameter('circle_center_z').value

        # transition parameters to reach the start point of trajectory smoothly
        # 'initial' means after the robot joint position is adjusted
        self.robot_initial_x = None
        self.robot_initial_y = None
        self.robot_initial_z = None
        self.robot_initial_received = False
        self.declare_parameter('transition_duration', 3.0)  # time to reach start point (s)
        self.declare_parameter('use_transition', True)      
        self.transition_duration = self.get_parameter('transition_duration').value
        self.use_transition = self.get_parameter('use_transition').value
        
        self.trajectory_enabled = False         # flag controlled by service
        
        self.start_time = self.get_clock().now()
        self.transition_start_time = None
        self.transition_complete = False        # flag indicating the completion of moving to the start point of trajectory
        
        # get start point of trajectory
        self.trajectory_start_x = self.center_x + self.radius
        self.trajectory_start_y = self.center_y
        self.trajectory_start_z = self.center_z

        # Ablation parameters
        self.gp_mode_pub = self.create_publisher(String, "/gp_mode", 10)
        # self.modes = ["none", "local", "cloud", "fusion", "history_fusion"]
        self.modes = ["local"]
        self.current_mode_index = 0
        self.period = 1.0 / self.frequency
        self.last_round = -1

        self.declare_parameter("rounds_per_mode", 5)
        self.rounds_per_mode = self.get_parameter("rounds_per_mode").value

        self.declare_parameter("max_rounds", 4)
        self.max_rounds = self.get_parameter("max_rounds").value

        self.shutdown_pub = self.create_publisher(Bool, "/shutdown_control", 10)


        self.get_logger().info('Trajectory publisher node started')
        self.get_logger().info(f'Publishing circular trajectory at 1000 Hz')
        self.get_logger().info(f'Circle radius: {self.radius} m, frequency: {self.frequency} Hz')
        self.get_logger().info(f'Circle center: ({self.center_x}, {self.center_y}, {self.center_z})')
        self.get_logger().info(f'Trajectory start point: ({self.trajectory_start_x:.3f}, {self.trajectory_start_y:.3f}, {self.trajectory_start_z:.3f})')
        if self.use_transition:
            self.get_logger().info(f'Transition duration: {self.transition_duration} s')
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
                
                self.robot_initial_received = True
                self.get_logger().info(f'Robot initial position recorded: ({self.robot_initial_x:.3f}, {self.robot_initial_y:.3f}, {self.robot_initial_z:.3f})')
                
                # start moving to the start point of trajectory after receiving initial position
                if self.use_transition:
                    self.transition_start_time = self.get_clock().now()
                    self.get_logger().info('Starting transition to trajectory start point')
                
            except Exception as e:
                self.get_logger().error(f'Error extracting robot initial position: {str(e)}')
    
    def timer_callback(self):
        """timer callback function, period: 1ms"""
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
                    # transition complete, start circular trajectory
                    self.transition_complete = True
                    self.get_logger().info('Transition complete, starting circular trajectory')
                    # reset start time for circular trajectory
                    self.start_time = current_time
                    elapsed_time = 0.0
                    
                    # set initial position to trajectory start point
                    x = self.trajectory_start_x
                    y = self.trajectory_start_y
                    z = self.trajectory_start_z
                else:
                    # generate smooth transition trajectory from adjusted robot position to start point
                    # use 5th order polynomial for interpolation
                    t = transition_elapsed / self.transition_duration
                    s = 10*t**3 - 15*t**4 + 6*t**5
                    
                    # interpolation
                    x = self.robot_initial_x + s * (self.trajectory_start_x - self.robot_initial_x)
                    y = self.robot_initial_y + s * (self.trajectory_start_y - self.robot_initial_y)
                    z = self.robot_initial_z + s * (self.trajectory_start_z - self.robot_initial_z)

                    ds_dt = (30*t**2 - 60*t**3 + 30*t**4) / self.transition_duration
                    d2s_dt2 = (60*t - 180*t**2 + 120*t**3) / (self.transition_duration**2)
                    
                    dx = ds_dt * (self.trajectory_start_x - self.robot_initial_x)
                    dy = ds_dt * (self.trajectory_start_y - self.robot_initial_y)
                    dz = ds_dt * (self.trajectory_start_z - self.robot_initial_z)
                    
                    ddx = d2s_dt2 * (self.trajectory_start_x - self.robot_initial_x)
                    ddy = d2s_dt2 * (self.trajectory_start_y - self.robot_initial_y)
                    ddz = d2s_dt2 * (self.trajectory_start_z - self.robot_initial_z)
            
            # trajectory for uniform circular trajectory
            if self.transition_complete or not self.use_transition:
                if elapsed_time > 0.0:
                    omega = 2.0 * np.pi * self.frequency  # angular velocity
                    
                    # position: (x, y, z) for x_des[:3]
                    x = self.center_x + self.radius * np.cos(omega * elapsed_time)
                    y = self.center_y + self.radius * np.sin(omega * elapsed_time)
                    z = self.center_z
                    
                    # velocity: (dx, dy, dz) for dx_des[:3]
                    dx = -self.radius * omega * np.sin(omega * elapsed_time)
                    dy = self.radius * omega * np.cos(omega * elapsed_time)
                    dz = 0.0
                    
                    # acceleration: (ddx, ddy, ddz) for ddx_des[:3]
                    ddx = -self.radius * omega**2 * np.cos(omega * elapsed_time)
                    ddy = -self.radius * omega**2 * np.sin(omega * elapsed_time)
                    ddz = 0.0
            
            # publish on /task_space_command
            trajectory_msg = TaskSpaceCommand()
            trajectory_msg.header = Header()
            trajectory_msg.header.stamp = current_time.to_msg()
            trajectory_msg.header.frame_id = "base_link"
            trajectory_msg.x_des = [x, y, z, 0.0, 0.0, 0.0]         # position (x, y, z, roll, pitch, yaw)
            trajectory_msg.dx_des = [dx, dy, dz, 0.0, 0.0, 0.0]     # velocity
            trajectory_msg.ddx_des = [ddx, ddy, ddz, 0.0, 0.0, 0.0] # acceleration
            
            self.trajectory_publisher.publish(trajectory_msg)
            
            # publish data recording status
            data_recording_msg = Bool()
            data_recording_msg.data = self.transition_complete or not self.use_transition
            self.data_recording_publisher.publish(data_recording_msg)
            
            if int(elapsed_time * 1000) % 1000 == 0:
                if self.use_transition and not self.transition_complete:
                    transition_elapsed = (current_time - self.transition_start_time).nanoseconds / 1e9
                    self.get_logger().debug(f'Transition phase: t={transition_elapsed:.3f}s, pos=({x:.3f}, {y:.3f}, {z:.3f})')
                else:
                    self.get_logger().debug(f'Circular trajectory: t={elapsed_time:.3f}s, pos=({x:.3f}, {y:.3f}, {z:.3f})')
                
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
                # —— 还在 5 次多项式过渡阶段 —— #
                t = transition_elapsed / self.transition_duration
                s = 10*t**3 - 15*t**4 + 6*t**5

                x = self.robot_initial_x + s * (self.trajectory_start_x - self.robot_initial_x)
                y = self.robot_initial_y + s * (self.trajectory_start_y - self.robot_initial_y)
                z = self.robot_initial_z + s * (self.trajectory_start_z - self.robot_initial_z)

                ds_dt = (30*t**2 - 60*t**3 + 30*t**4) / self.transition_duration
                d2s_dt2 = (60*t - 180*t**2 + 120*t**3) / (self.transition_duration**2)

                dx = ds_dt * (self.trajectory_start_x - self.robot_initial_x)
                dy = ds_dt * (self.trajectory_start_y - self.robot_initial_y)
                dz = ds_dt * (self.trajectory_start_z - self.robot_initial_z)

                ddx = d2s_dt2 * (self.trajectory_start_x - self.robot_initial_x)
                ddy = d2s_dt2 * (self.trajectory_start_y - self.robot_initial_y)
                ddz = d2s_dt2 * (self.trajectory_start_z - self.robot_initial_z)

            else:
                # —— 已经过渡完，进入圆轨迹 —— #
                # 圆轨迹的时间从 transition 结束时刻算起：
                t_circle = transition_elapsed - self.transition_duration
                if t_circle < 0.0:
                    t_circle = 0.0

                omega = 2.0 * np.pi * self.frequency

                x = self.center_x + self.radius * np.cos(omega * t_circle)
                y = self.center_y + self.radius * np.sin(omega * t_circle)
                z = self.center_z

                dx = -self.radius * omega * np.sin(omega * t_circle)
                dy =  self.radius * omega * np.cos(omega * t_circle)
                dz = 0.0

                ddx = -self.radius * omega**2 * np.cos(omega * t_circle)
                ddy = -self.radius * omega**2 * np.sin(omega * t_circle)
                ddz = 0.0

        else:
            # —— 没有过渡，直接圆轨迹，从 start_time 开始 —— #
            elapsed_time = (future_time - self.start_time).nanoseconds / 1e9
            if elapsed_time < 0.0:
                elapsed_time = 0.0

            omega = 2.0 * np.pi * self.frequency

            x = self.center_x + self.radius * np.cos(omega * elapsed_time)
            y = self.center_y + self.radius * np.sin(omega * elapsed_time)
            z = self.center_z

            dx = -self.radius * omega * np.sin(omega * elapsed_time)
            dy =  self.radius * omega * np.cos(omega * elapsed_time)
            dz = 0.0

            ddx = -self.radius * omega**2 * np.cos(omega * elapsed_time)
            ddy = -self.radius * omega**2 * np.sin(omega * elapsed_time)
            ddz = 0.0

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