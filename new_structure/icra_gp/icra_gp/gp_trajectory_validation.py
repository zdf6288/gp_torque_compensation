#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_msgs.msg import DataForGP, TaskSpaceCommand
from custom_msgs.srv import GPPredict
from std_msgs.msg import Header
import numpy as np
import pandas as pd
import time


class GPTrajectoryValidation(Node):
    
    def __init__(self):
        super().__init__('gp_trajectory_validation')
        
        # subscribe to /data_for_gp topic
        self.data_subscription = self.create_subscription(
            DataForGP, '/data_for_gp', self.data_callback, 10)

        # create a server for GP prediction
        self.gp_predict_server = self.create_service(
            GPPredict, '/gp_predict', self.gp_predict_callback)
        
        # publish on /task_space_command topic
        self.task_space_command_publisher = self.create_publisher(
            TaskSpaceCommand, '/task_space_command', 10)

        # timer for publishing task space command
        self.timer = self.create_timer(0.001, self.timer_callback)  # publish at 1000 Hz

        # list for storing trajectory data
        self.x_real = []        # store position data [x, y, z]
        self.time_stamp = []    # store time stamp
        self.x_pred = []        # store predicted position data [x, y, z]
        self.z_des = None       # desired z to keep the pen contact with the paper
        
        # trajectory publishing control
        self.predicted_trajectory_index = 0             # current index for publishing predicted trajectory
        self.predicted_trajectory_finished = False      # flag indicating publishment ofpredicted trajectory is finished
        self.point_repeat_count = 0                     # counter for repeating each point 10 times
        self.points_per_repeat = 20                     # number of times to publish each point
        
        self.declare_parameter('filename', 'validation_data.csv')
        self.filename = self.get_parameter('filename').get_parameter_value().string_value
        self.get_logger().info(f'Data will be recorded in goal filepath: {self.filename}')

        self.get_logger().info('GP trajectory data collection node started')

        self.gp_finished = False  # flag indicating the end of GP prediction
    
    def data_callback(self, msg):
        """callback function for /data_for_gp subscriber"""
        try:         
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            position = list(msg.x_real)  # [x, y, z]  
            self.x_real.append(position)
            self.time_stamp.append(timestamp)
            self.z_des = position[2] - 0.15
            # z_desired is slightly lower than the last captured z-position
            # to keep the pen contact with the paper
                
        except Exception as e:
            self.get_logger().error(f'Error when processing data message: {str(e)}')

    def gp_predict_callback(self, request, response):
        """callback function for GP prediction service"""
        try:
            self.get_logger().info('Received GP prediction request')

            success = self.gp_predict(self.x_real)
            
            if success:
                response.success = True
                response.message = "GP prediction completed successfully"
                response.gp_finished = self.gp_finished
                self.get_logger().info('GP prediction completed successfully')
            else:
                response.success = False
                response.message = "GP prediction failed"
                response.gp_finished = False
                self.get_logger().error('GP prediction failed')
                
        except Exception as e:
            self.get_logger().error(f'Error in GP prediction callback: {str(e)}')
            response.success = False
            response.message = f"Error in GP prediction: {str(e)}"
            response.gp_finished = False
            
        return response

    def gp_predict(self, x):

        try:
            self.get_logger().info('Starting GP prediction...')
            
            # Convert the input data to numpy array for processing
            # x_real_array = np.array(x).reshape(-1, 3)  # reshape to [N, 3] for [x, y, z]
            
            with open(self.filename, "rb") as f:
                df = pd.read_csv(f)
                x_temp = df['x_actual'].values - df['x_actual'].values[0]
                y_temp = df['y_actual'].values - df['y_actual'].values[0]
                x_pred = x_temp + self.x_real[-1][0]
                y_pred = y_temp + self.x_real[-1][1]
                
                self.x_pred = np.column_stack((x_pred, y_pred, self.z_des))   

            self.gp_finished = True

            # predicted trajectory publishing control variables
            self.predicted_trajectory_index = 0
            self.predicted_trajectory_finished = False
            self.point_repeat_count = 0      # reset repeat counter for new prediction

            self.get_logger().info(f'GP prediction completed. Predicted {len(self.x_pred)} points.')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error in gp_predict function: {str(e)}')
            return False

    def timer_callback(self):
        """timer callback function"""
        if not self.gp_finished:
            pass
        else:
            if self.predicted_trajectory_finished:
                return          # all predicted trajectory published
            
            task_cmd = TaskSpaceCommand()            
            if self.predicted_trajectory_index < len(self.x_pred):
                # Send next [x, y, z] from x_pred
                xyz = self.x_pred[self.predicted_trajectory_index]
                task_cmd.x_des = [xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0]
                task_cmd.dx_des = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                task_cmd.ddx_des = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

                # publish task space command
                self.task_space_command_publisher.publish(task_cmd)
                
                # publish every point 10 times
                self.point_repeat_count += 1
                if self.point_repeat_count >= self.points_per_repeat:
                    self.predicted_trajectory_index += 1
                    self.point_repeat_count = 0  # reset counter for next point

            else:
                # all predicted trajectory published
                self.predicted_trajectory_finished = True
                self.get_logger().info('All trajectory points have been sent, trajectory publishing finished')
                return

def main(args=None):
    rclpy.init(args=args)
    gp_trajectory_validation_node = GPTrajectoryValidation()
    
    try:
        rclpy.spin(gp_trajectory_validation_node)
        pass
    finally:
        gp_trajectory_validation_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()