#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_msgs.msg import StateParameter
from custom_msgs.srv import JointPositionAdjust
from std_msgs.msg import Header, Bool
import numpy as np
import time
from pynput import keyboard


class TrajectoryPublisherMultiValidation(Node):
    
    def __init__(self):
        super().__init__('trajectory_publisher_multi_validation')

        # subscribe to /state_parameter to get robot current state
        self.state_subscription = self.create_subscription(
            StateParameter, '/state_parameter', self.stateCallback, 10)

        # subscribe to /data_recording_enabled to know when to start recording data
        self.data_recording_subscription = self.create_subscription(
            Bool, '/data_recording_enabled', self.dataRecordingCallback, 10)
        
        # Create service server for joint position adjustment
        self.joint_position_service = self.create_service(
            JointPositionAdjust, '/joint_position_adjust', self.joint_position_callback)

        # transition parameters to reach the start point of trajectory smoothly
        # 'initial' means after the robot joint position is adjusted
        self.robot_initial_x = None
        self.robot_initial_y = None
        self.robot_initial_z = None
        self.robot_initial_received = False
        self.declare_parameter('use_transition', True)      # in multi-class test, transition is a manual process
        self.use_transition = self.get_parameter('use_transition').value
        
        self.trajectory_enabled = False                     # flag controlled by service

        # keyboard listening variables
        self.key_pressed_d = False              # flag indicating the 'd' key on keyboard is pressed
        self.key_pressed_v = False              # flag indicating the 'v' key on keyboard is pressed
        self.keyboard_listener = None           # keyboard listener object
        self.keyboard_thread = None             # thread for keyboard listening
        
        self.start_time = self.get_clock().now()
        self.transition_complete = False        # flag indicating the completion of moving to the start point of trajectory
        self.transition_start_time = None

        self.get_logger().info('Trajectory publisher node started')
        self.get_logger().info(f'Publishing trajectory at 1000 Hz')
        self.get_logger().info('Waiting for joint position adjustment service call to enable trajectory...')
    
    # keyboard listening functions
    def on_key_press(self, key):
        """callback function of keyboard listener"""
        try:
            if key.char == 'd':
                self.key_pressed_d = True
                self.get_logger().info('Key "d" pressed! Continuing trajectory execution...')
            if key.char == 'v':
                self.key_pressed_v = True
                self.get_logger().info('Key "v" pressed! Stop recording data probe, start GP prediction...')
        except AttributeError:
            pass
    
    def start_keyboard_listener(self):
        """start keyboard listener"""
        self.key_pressed_d = False
        self.key_pressed_v = False
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()
        self.get_logger().info('Keyboard listener started. Press "d" to continue...')
    
    def stop_keyboard_listener(self):
        """stop keyboard listener"""
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
    
    def joint_position_callback(self, request, response):
        """Service callback for joint position adjustment"""
        try:
            self.get_logger().info(f'Received joint position adjustment request')
            self.get_logger().info(f'q_des: {request.q_des}')
            self.get_logger().info(f'dq_des: {request.dq_des}')

            self.trajectory_enabled = True
            
            self.start_keyboard_listener()
            while not self.key_pressed_d:
                pass
            self.key_pressed_d = False
            self.transition_complete = True
            
            # reset timing for trajectory
            self.start_time = self.get_clock().now()
            self.transition_start_time = None
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
    
    def timer_callback(self):
        """timer callback function, period: 1ms"""
        if not self.transition_complete:
            return
        
        if not self.key_pressed_v:
            # publish data recording status
            data_recording_msg = Bool()
            data_recording_msg.data = True
            self.data_recording_publisher.publish(data_recording_msg)
        else:
            self.stop_keyboard_listener()

            data_recording_msg = Bool()
            data_recording_msg.data = False
            self.data_recording_publisher.publish(data_recording_msg)


        return

def main(args=None):
    rclpy.init(args=args)
    trajectory_publisher_multi_validation_node = TrajectoryPublisherMultiValidation()
    
    try:
        rclpy.spin(trajectory_publisher_multi_validation_node)
        pass
    finally:
        trajectory_publisher_multi_validation_node.stop_keyboard_listener()
        trajectory_publisher_multi_validation_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()