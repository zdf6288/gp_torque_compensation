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


class GPTrainNode(Node):
    def __init__(self):
        super().__init__('gp_train')
        
        self.declare_parameter('data_start_index', 1000)
        self.data_start_index = self.get_parameter('data_start_index').get_parameter_value().integer_value
        self.get_logger().info(f'Using data_start_index: {self.data_start_index}')


    def train(self):
        df = pd.read_csv('training_data.csv')
        x_real = df['x_actual'].values
        y_real = df['y_actual'].values
        x_real = x_real[self.data_start_index:]
        y_real = y_real[self.data_start_index:]
        ref = list(zip(x_real.tolist(), y_real.tolist()))
        self.get_logger().info("Start Training")
        gp_predictor = GP_predictor()
        gp_predictor.train_gp(ref)
        
        with open("gp_model.pkl", "wb") as f:
            pickle.dump(gp_predictor, f)

        self.get_logger().info("GP model saved to gp_model.pkl")

def main():
    rclpy.init()
    node = GPTrainNode()
    node.train()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 