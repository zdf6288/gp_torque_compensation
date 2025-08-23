#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_msgs.msg import TaskSpaceCommand, StateParameter
from std_msgs.msg import Header
import numpy as np
import time


class TrajectoryEclipsePublisher(Node):
    
    def __init__(self):
        super().__init__('trajectory_eclipse_publisher')
        
        # publish on /task_space_command
        self.trajectory_publisher = self.create_publisher(
            TaskSpaceCommand, '/task_space_command', 10)
        
        # subscribe to /state_parameter to get robot current state
        self.state_subscription = self.create_subscription(
            StateParameter, '/state_parameter', self.stateCallback, 10)
        
        # publish frequency: 1000 Hz
        self.timer = self.create_timer(0.001, self.timer_callback)  # 0.001 second = 1000 Hz
        
        # circular trajectory parameters
        self.declare_parameter('circle_radius', 0.1)        # circle radius (meter)
        self.declare_parameter('circle_frequency', 0.1)     # circle motion frequency (Hz)
        self.declare_parameter('circle_center_x', 0.6)      # circle center x coordinate
        self.declare_parameter('circle_center_y', 0.0)      # circle center y coordinate
        self.declare_parameter('circle_center_z', 0.45)     # circle center z coordinate
        self.declare_parameter('bank_angle', 10.0)          # bank angle in degrees (tilt of the circle plane relative to xOy)
        
        self.radius = self.get_parameter('circle_radius').value
        self.frequency = self.get_parameter('circle_frequency').value
        self.center_x = self.get_parameter('circle_center_x').value
        self.center_y = self.get_parameter('circle_center_y').value
        self.center_z = self.get_parameter('circle_center_z').value
        self.bank_angle = np.radians(self.get_parameter('bank_angle').value)  # convert to radians
        
        # transition parameters to reach the start point of trajectory smoothly
        self.robot_initial_x = None
        self.robot_initial_y = None
        self.robot_initial_z = None
        self.robot_initial_received = False
        self.declare_parameter('transition_duration', 3.0)  # time to reach start point (s)
        self.declare_parameter('use_transition', True)      
        self.transition_duration = self.get_parameter('transition_duration').value
        self.use_transition = self.get_parameter('use_transition').value
        
        self.start_time = self.get_clock().now()
        self.transition_start_time = None
        self.transition_complete = False
        
        # get start point of trajectory (on the tilted circle)
        # The circle is tilted around x-axis, so the start point will be at (center_x + radius, center_y, center_z + radius*sin(bank_angle))
        self.trajectory_start_x = self.center_x + self.radius
        self.trajectory_start_y = self.center_y
        self.trajectory_start_z = self.center_z 
        
        self.get_logger().info('Trajectory Circle Publisher node started')
        self.get_logger().info(f'Publishing circular trajectory at 1000 Hz')
        self.get_logger().info(f'Circle radius: {self.radius} m, frequency: {self.frequency} Hz')
        self.get_logger().info(f'Circle center: ({self.center_x}, {self.center_y}, {self.center_z})')
        self.get_logger().info(f'Bank angle: {np.degrees(self.bank_angle):.1f} degrees')
        self.get_logger().info(f'Trajectory start point: ({self.trajectory_start_x:.3f}, {self.trajectory_start_y:.3f}, {self.trajectory_start_z:.3f})')
        if self.use_transition:
            self.get_logger().info(f'Transition duration: {self.transition_duration} s')
    
    def stateCallback(self, msg):
        """callback function of /state_parameter subscriber"""
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
                
                # start transition after receiving initial position
                if self.use_transition:
                    self.transition_start_time = self.get_clock().now()
                    self.get_logger().info('Starting transition to trajectory start point')
                
            except Exception as e:
                self.get_logger().error(f'Error extracting robot initial position: {str(e)}')
    
    def timer_callback(self):
        """timer callback function, period: 1ms"""
        try:
            # wait for initialization of robot position
            if not self.robot_initial_received:
                return
            
            # get time, initialize variables
            current_time = self.get_clock().now()
            elapsed_time = (current_time - self.start_time).nanoseconds / 1e9
            x, y, z = 0.0, 0.0, 0.0
            dx, dy, dz = 0.0, 0.0, 0.0
            ddx, ddy, ddz = 0.0, 0.0, 0.0
            
            if self.use_transition and not self.transition_complete:
                # transition: move from initial robot position to trajectory start point
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
                    # generate smooth transition trajectory from robot initial position to start point
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
            
            # circular trajectory on tilted plane
            if self.transition_complete or not self.use_transition:
                if elapsed_time > 0.0:
                    # calculate circular trajectory parameters
                    omega = 2.0 * np.pi * self.frequency  # angular velocity
                    
                    # Generate a perfect circle in the tilted plane
                    # The circle is tilted around x-axis by bank_angle
                    # In the tilted plane, we use standard circle equations
                    angle = omega * elapsed_time
                    
                    # Base circle coordinates (in the tilted plane)
                    x_base = self.center_x + self.radius * np.cos(angle)
                    y_base = self.center_y + self.radius * np.sin(angle)
                    
                    # Apply rotation matrix around x-axis to tilt the circle
                    # This creates a tilted circle where the z-coordinate varies with the bank angle
                    cos_bank = np.cos(self.bank_angle)
                    sin_bank = np.sin(self.bank_angle)
                    
                    # Apply rotation matrix around x-axis
                    x = x_base
                    y = y_base * cos_bank
                    z = self.center_z + y_base * sin_bank
                    
                    # Calculate velocities
                    # Base velocities (in the tilted plane)
                    dx_base = -self.radius * omega * np.sin(angle)
                    dy_base = self.radius * omega * np.cos(angle)
                    
                    # Apply bank angle to velocities
                    dx = dx_base
                    dy = dy_base * cos_bank
                    dz = dy_base * sin_bank
                    
                    # Calculate accelerations
                    # Base accelerations (in the tilted plane)
                    ddx_base = -self.radius * omega**2 * np.cos(angle)
                    ddy_base = -self.radius * omega**2 * np.sin(angle)
                    
                    # Apply bank angle to accelerations
                    ddx = ddx_base
                    ddy = ddy_base * cos_bank
                    ddz = ddy_base * sin_bank
            
            # publish message
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
            self.get_logger().error(f'Error in trajectory circle publisher: {str(e)}')
            self.get_logger().error(f'Current state: transition_complete={self.transition_complete}, elapsed_time={elapsed_time}')


def main(args=None):
    rclpy.init(args=args)
    trajectory_eclipse_node = TrajectoryEclipsePublisher()
    
    try:
        rclpy.spin(trajectory_eclipse_node)
    except KeyboardInterrupt:
        trajectory_eclipse_node.get_logger().info('Received keyboard interrupt, stopping trajectory publishing...')
    except Exception as e:
        trajectory_eclipse_node.get_logger().error(f'Error when running program: {str(e)}')
    finally:
        trajectory_eclipse_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 