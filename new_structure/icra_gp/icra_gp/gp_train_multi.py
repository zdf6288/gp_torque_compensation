#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from icra_gp.gp_predictor import GP_predictor
import pickle
import rclpy
from rclpy.node import Node


class GPTrainMultiNode(Node):
    def __init__(self):
        super().__init__('gp_train_multi')
        
        self.declare_parameter('data_start_index', 1000)
        self.data_start_index = self.get_parameter('data_start_index').get_parameter_value().integer_value
        self.get_logger().info(f'Using data_start_index: {self.data_start_index}')

        self.declare_parameter('filename', 'training_multi_data.csv')
        self.filename = self.get_parameter('filename').get_parameter_value().string_value
        self.get_logger().info(f'Training Multi Data from File: {self.filename}')


    def train(self):
        try:
            df = pd.read_csv(self.filename)
            x_real = df['x_actual'].values
            y_real = df['y_actual'].values
            x_real = x_real[self.data_start_index:]
            y_real = y_real[self.data_start_index:]
            ref = list(zip(x_real.tolist(), y_real.tolist()))
            self.get_logger().info("Start Training")
        except Exception as e:
            self.get_logger().error(f"Error in training: {str(e)}")
            return
        
        if os.path.exists("gp_multi_model.pkl") and os.path.getsize("gp_multi_model.pkl") > 0:
            try:
                with open("gp_multi_model.pkl", "rb") as f:
                    gp_predictor = pickle.load(f)
                self.get_logger().info("Loaded existing GP model from gp_multi_model.pkl")
            except (pickle.PickleError, EOFError) as e:
                self.get_logger().warn(f"Failed to load existing model: {e}. Creating new model.")
                gp_predictor = GP_predictor()
        else:
            self.get_logger().info("No existing model found. Creating new GP model.")
            gp_predictor = GP_predictor()
        
        gp_predictor.train_gp(ref)
        
        with open("gp_multi_model.pkl", "wb") as f:
            pickle.dump(gp_predictor, f)

        self.get_logger().info("GP model saved to gp_multi_model.pkl")

def main():
    rclpy.init()
    node = GPTrainMultiNode()
    node.train()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 