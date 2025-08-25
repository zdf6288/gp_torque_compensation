#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_msgs.msg import TaskSpaceCommand, StateParameter, LambdaCommand
from custom_msgs.srv import JointPositionAdjust
from std_msgs.msg import Header
import numpy as np
import time


class TrajectoryPublisherLambda(Node):
    
    def __init__(self):
        super().__init__('trajectory_publisher_lambda')
        
        # publish on /task_space_command
        self.trajectory_publisher = self.create_publisher(
            TaskSpaceCommand, '/task_space_command', 10)
        self.timer = self.create_timer(0.001, self.timer_callback)  # publish at 1000 Hz

        # subscribe to /state_parameter to get robot current state
        self.state_subscription = self.create_subscription(
            StateParameter, '/state_parameter', self.stateCallback, 10)
        
        # subscribe to /TwistLeft or /TwistRight according to arm parameter
        self.declare_parameter('arm', 'left')
        self.arm = self.get_parameter('arm').value
        if self.arm == "left":
            self.twist_subscription = self.create_subscription(
                LambdaCommand, '/TwistLeft', self.lambdaCallback, 10)
        elif self.arm == "right":
            self.twist_subscription = self.create_subscription(
                LambdaCommand, '/TwistRight', self.lambdaCallback, 10)
        else:
            self.get_logger().error(f'Invalid arm parameter: {self.arm}. Must be "left" or "right"')
            self.twist_subscription = None
        
        # Create service server for joint position adjustment
        self.joint_position_service = self.create_service(
            JointPositionAdjust, '/joint_position_adjust', self.joint_position_callback)

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
        
        # start point of trajectory
        self.trajectory_start_x = 0.5
        self.trajectory_start_y = 0.0
        self.trajectory_start_z = 0.5

        # for convertion from lambda command to trajectory
        self.t_buffer = None
        self.x_buffer = self.trajectory_start_x
        self.y_buffer = self.trajectory_start_y
        self.z_buffer = self.trajectory_start_z
        self.lambda_linear_x_buffer = 0.0
        self.lambda_linear_y_buffer = 0.0
        self.lambda_linear_z_buffer = 0.0
        
        self.get_logger().info('Trajectory publisher node started')
        self.get_logger().info(f'Publishing trajectory at 1000 Hz')
        self.get_logger().info(f'Arm parameter: {self.arm}')
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
    
    def lambdaCallback(self, msg):
        """callback function of /TwistLeft or /TwistRight subscriber"""
        try:
            self.lambda_linear_x = msg.linear.x
            self.lambda_linear_y = msg.linear.y
            self.lambda_linear_z = msg.linear.z
        except Exception as e:
            self.get_logger().error(f'Error in lambda callback: {str(e)}')
    
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
            
            # trajectory determined by lambda command
            if self.transition_complete or not self.use_transition:
                if elapsed_time > 0.0:
                    if self.t_buffer is None:
                        dt = 1e-3
                    else:
                        dt = elapsed_time - self.t_buffer
                    self.t_buffer = elapsed_time

                    # position: (x, y, z) for x_des[:3]
                    x = self.x_buffer + self.lambda_linear_x * dt
                    y = self.y_buffer + self.lambda_linear_y * dt
                    z = self.z_buffer + self.lambda_linear_z * dt
                    
                    # velocity: (dx, dy, dz) for dx_des[:3]
                    dx = self.lambda_linear_x
                    dy = self.lambda_linear_y
                    dz = self.lambda_linear_z

                    # acceleration: (ddx, ddy, ddz) for ddx_des[:3]
                    ddx = 0
                    ddy = 0
                    ddz = 0
            
            # publish on /task_space_command
            trajectory_msg = TaskSpaceCommand()
            trajectory_msg.header = Header()
            trajectory_msg.header.stamp = current_time.to_msg()
            trajectory_msg.header.frame_id = "base_link"
            trajectory_msg.x_des = [x, y, z, 0.0, 0.0, 0.0]         # position (x, y, z, roll, pitch, yaw)
            trajectory_msg.dx_des = [dx, dy, dz, 0.0, 0.0, 0.0]     # velocity
            trajectory_msg.ddx_des = [ddx, ddy, ddz, 0.0, 0.0, 0.0] # acceleration
            
            self.trajectory_publisher.publish(trajectory_msg)
            
            if int(elapsed_time * 1000) % 1000 == 0:
                if self.use_transition and not self.transition_complete:
                    transition_elapsed = (current_time - self.transition_start_time).nanoseconds / 1e9
                    self.get_logger().debug(f'Transition phase: t={transition_elapsed:.3f}s, pos=({x:.3f}, {y:.3f}, {z:.3f})')
                else:
                    self.get_logger().debug(f'Circular trajectory: t={elapsed_time:.3f}s, pos=({x:.3f}, {y:.3f}, {z:.3f})')
                
        except Exception as e:
            self.get_logger().error(f'Error in trajectory publisher: {str(e)}')
            self.get_logger().error(f'Current state: transition_complete={self.transition_complete}, elapsed_time={elapsed_time}')


def main(args=None):
    rclpy.init(args=args)
    trajectory_publisher_lambda_node = TrajectoryPublisherLambda()
    
    try:
        rclpy.spin(trajectory_publisher_lambda_node)
        pass
    finally:
        trajectory_publisher_lambda_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()