#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from icra_gp.gp_predictor import train_reference_from_array

def train():
    df = pd.read_csv('training_data.csv')
    x_real = df['x_actual'].values
    y_real = df['y_actual'].values
    x_real = x_real[::10]
    y_real = y_real[::10]
    ref = list(zip(x_real.tolist(), y_real.tolist()))
    model_bundle = train_reference_from_array(ref)
    
    # 将tensor转换为numpy数组并保存为CSV
    sampled_data = model_bundle['sampled'].detach().cpu().numpy()
    
    # 创建DataFrame并保存
    df_sampled = pd.DataFrame(sampled_data, columns=['x', 'y'])
    df_sampled.to_csv('sampled_trajectory.csv', index=False)
    
    print(f"已保存 {len(sampled_data)} 个点到 sampled_trajectory.csv")
    print("前5个点:")
    print(df_sampled.head())

def main():
    train()

if __name__ == '__main__':
    main() 