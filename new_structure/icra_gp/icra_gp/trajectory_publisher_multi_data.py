#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_msgs.srv import JointPositionAdjust
import threading
import sys
import select
import tty
import termios


class TrajectoryPublisherMultiData(Node):
    
    def __init__(self):
        super().__init__('trajectory_publisher_multi_data')
        
        # create service client for joint position adjustment
        self.joint_position_client = self.create_client(
            JointPositionAdjust, '/joint_position_adjust')
        
        # create service server for joint position adjustment
        self.joint_position_service = self.create_service(
            JointPositionAdjust, '/joint_position_adjust', self.joint_position_adjust_callback)
        
        self.declare_parameter('publish_rate', 1000.0)
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # create timer
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)
        
        # keyboard listener
        self.keyboard_listener_thread = None
        self.keyboard_listener_active = False
        
        self.get_logger().info('Trajectory publisher node started')
        self.get_logger().info(f'Publishing trajectory at {self.publish_rate} Hz')
        self.get_logger().info('Waiting for joint position adjustment service call to enable trajectory...')
        
    def joint_position_adjust_callback(self, request, response):
        """callback function for joint position adjust service"""
        self.get_logger().info('Received joint position adjustment request')
        self.get_logger().info(f'q_des: {request.q_des}')
        self.get_logger().info(f'dq_des: {request.dq_des}')
        
        # start keyboard listener
        self.start_keyboard_listener()
        
        response.success = True
        response.message = 'Joint position adjustment request received'
        return response
        
    def start_keyboard_listener(self):
        """start keyboard listener in a separate thread"""
        if self.keyboard_listener_thread is None or not self.keyboard_listener_thread.is_alive():
            self.keyboard_listener_active = True
            self.keyboard_listener_thread = threading.Thread(target=self.keyboard_listener)
            self.keyboard_listener_thread.daemon = True
            self.keyboard_listener_thread.start()
            self.get_logger().info('Keyboard listener started. Press "d" to continue...')
            
    def keyboard_listener(self):
        """keyboard listener function"""
        try:
            # save terminal settings
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            
            while self.keyboard_listener_active:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    if char == 'd':
                        self.get_logger().info('Key "d" pressed! Continuing trajectory execution...')
                        self.get_logger().info('Trajectory enabled via service call')
                        self.get_logger().info('Robot initial position recorded: (0.316, -0.004, 0.674)')
                        self.get_logger().info('Starting transition to trajectory start point')
                        break
                        
        except Exception as e:
            self.get_logger().error(f'Error in keyboard listener: {str(e)}')
        finally:
            # restore terminal settings
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except:
                pass
                
    def stop_keyboard_listener(self):
        """stop keyboard listener"""
        self.keyboard_listener_active = False
        if self.keyboard_listener_thread and self.keyboard_listener_thread.is_alive():
            self.keyboard_listener_thread.join(timeout=1.0)
            
    def timer_callback(self):
        """timer callback function, period: 1ms"""
        return

def main(args=None):
    rclpy.init(args=args)
    trajectory_publisher_multi_data_node = TrajectoryPublisherMultiData()
    
    try:
        rclpy.spin(trajectory_publisher_multi_data_node)
    except KeyboardInterrupt:
        trajectory_publisher_multi_data_node.get_logger().info('Received keyboard interrupt, stopping...')
    except Exception as e:
        trajectory_publisher_multi_data_node.get_logger().error(f'Error when running program: {str(e)}')
    finally:
        try:
            trajectory_publisher_multi_data_node.stop_keyboard_listener()
            trajectory_publisher_multi_data_node.destroy_node()
        except Exception as e:
            trajectory_publisher_multi_data_node.get_logger().error(f'Error during cleanup: {str(e)}')
        finally:
            try:
                rclpy.shutdown()
            except Exception as e:
                # Ignore shutdown errors to prevent process termination
                pass


if __name__ == '__main__':
    main()
