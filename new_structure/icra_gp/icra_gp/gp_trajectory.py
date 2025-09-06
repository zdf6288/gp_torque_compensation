#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_msgs.msg import DataForGP, TaskSpaceCommand
from custom_msgs.srv import GPPredict
from std_msgs.msg import Header
import numpy as np
import time
from icra_gp.gp_predictor import GP_predictor
import pickle


class GPTrajectory(Node):
    
    def __init__(self):
        super().__init__('gp_trajectory')
        
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
        
        # trajectory publishing control
        self.predicted_trajectory_index = 0             # current index for publishing predicted trajectory
        self.predicted_trajectory_finished = False      # flag indicating publishment ofpredicted trajectory is finished
        
        self.get_logger().info('GP trajectory data collection node started')

        self.gp_finished = False  # flag indicating the end of GP prediction
    
    def data_callback(self, msg):
        """callback function for /data_for_gp subscriber"""
        try:         
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            position = list(msg.x_real)  # [x, y, z]  
            self.x_real.append(position)
            self.time_stamp.append(timestamp)
                
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
            
            
            x_array = np.array(x)
            probe2d = x_array[:, :2]
            probe = probe2d[::10]
            # print(f"ref.shape: {ref.shape}")
            # print(f"x_array.shape: {x_array.shape}")
            # print(f"probe.shape: {probe.shape}")
            # print(probe)
            with open("gp_model.pkl", "rb") as f:
                gp_predictor = pickle.load(f)
            predicted = gp_predictor.predict_from_probe(probe)
            self.x_pred = []
            for point in predicted:
                self.x_pred.append([point[0], point[1], 0.65])     

            self.gp_finished = True

            # predicted trajectory publishing control variables
            self.predicted_trajectory_index = 0
            self.predicted_trajectory_finished = False

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
                self.predicted_trajectory_index += 1

                # publish task space command
                self.task_space_command_publisher.publish(task_cmd)

            else:
                # all predicted trajectory published
                self.predicted_trajectory_finished = True
                self.get_logger().info('All trajectory points have been sent, trajectory publishing finished')
                return     

def main(args=None):
    rclpy.init(args=args)
    gp_trajectory_node = GPTrajectory()
    
    try:
        rclpy.spin(gp_trajectory_node)
        pass
    finally:
        gp_trajectory_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()