import numpy as np
from gp_predictor import train_reference_from_array, predict_trajectory_from_probe

# 生成 sin 曲线作为参考轨迹
x_ref = np.linspace(0, 4*np.pi, 50)
y_ref = np.sin(x_ref)
ref = list(zip(x_ref.tolist(), y_ref.tolist()))

# 生成平移后的 sin 曲线作为 probe
x_probe = np.linspace(0, 4*np.pi, 50)
y_probe = np.sin(x_probe) + 5.0  # Y轴平移
probe = list(zip(x_probe.tolist(), y_probe.tolist()))

# 训练
model_bundle = train_reference_from_array(ref)

# 预测
predicted = predict_trajectory_from_probe(model_bundle, probe)

# 打印预测结果
print("✅ 预测轨迹：")
for pt in predicted:
    print(f"{pt[0]:.3f}, {pt[1]:.3f}")