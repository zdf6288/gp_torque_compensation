#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from icra_gp.gp_predictor import GP_predictor
import pickle
import rclpy


def train():
    df = pd.read_csv('training_data.csv')
    x_real = df['x_actual'].values
    y_real = df['y_actual'].values
    # x_real = x_real[18000:]
    # y_real = y_real[10000:]
    # x_real = x_real[::2]
    # y_real = y_real[::2]
    ref = list(zip(x_real.tolist(), y_real.tolist()))
    print("Start Training")
    gp_predictor = GP_predictor()
    gp_predictor.train_gp(ref)
    
    with open("gp_model.pkl", "wb") as f:
        pickle.dump(gp_predictor, f)

    print("GP model saved to gp_model.pkl")

def main():
    rclpy.init()
    train()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 