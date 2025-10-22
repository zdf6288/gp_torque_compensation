# train_gp.py —— 在线预测 + 增量学习 + SMSE（全局方差作分母，无滑动窗口）
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from skygp import SkyGP_rBCM
from hyperparam_training import fit_hparams_gpytorch


# ---------------- Utils ----------------
def standardize(X, Y):
    Xm, Xs = X.mean(0), X.std(0); Xs[Xs < 1e-9] = 1.0
    Ym, Ys = Y.mean(0), Y.std(0); Ys[Ys < 1e-9] = 1.0
    Xn = (X - Xm) / Xs
    Yn = (Y - Ym) / Ys
    return Xn.astype(np.float32), Yn.astype(np.float32), (Xm, Xs, Ym, Ys)

def destandardize(y_std_vec, stats):
    """y 标准化值 -> 原单位"""
    _, _, Ym, Ys = stats
    return y_std_vec * Ys[0] + Ym[0]

def plot_curve(curve, out_png, title="SMSE vs #Samples"):
    plt.figure(figsize=(7,4.2))
    x = np.arange(1, len(curve)+1)
    plt.plot(x, curve, linewidth=2)
    plt.xlabel("Number of samples (prefix)")
    plt.ylabel("SMSE")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()

def smse_curve_from_preds(y_true, y_pred, warmup=0):
    """
    计算逐前缀 SMSE，分母使用全局方差（稳定）。
    前 warmup 个点只沿用上一值（首点为 0），不计误差。
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    N = len(y_true)
    curve = np.zeros(N, dtype=np.float64)

    var_global = float(np.var(y_true, ddof=1))
    if var_global < 1e-12:
        var_global = 1e-12

    sum_sq = 0.0
    for i in range(N):
        if i < warmup:
            curve[i] = 0.0 if i == 0 else curve[i-1]
            continue
        err = float(y_pred[i] - y_true[i])
        sum_sq += err * err
        t_eff = i - warmup + 1
        mse_t = sum_sq / t_eff
        curve[i] = mse_t / var_global
    return curve


# --------------- Hparam fit ---------------
def fit_global_hparams(Xn, Yn, max_pts_hparam=2000, iters=600, lr=0.05):
    """
    在标准化后的数据上拟合超参（子采样加速）。
    返回 (sigma_f, sigma_n, lengthscale)
    """
    N = len(Xn)
    if N > max_pts_hparam:
        idx = np.random.choice(N, max_pts_hparam, replace=False)
        Xh, Yh = Xn[idx], Yn[idx, 0]
    else:
        Xh, Yh = Xn, Yn[:, 0]
    outputscale, noise, lengthscale = fit_hparams_gpytorch(
        Xh.astype(np.float32), Yh.astype(np.float32),
        max_points=min(max_pts_hparam, len(Xh)),
        iters=iters, lr=lr, use_cuda_if_available=True, print_every=100
    )
    return outputscale, noise, lengthscale


# --------------- Online train & eval ---------------
def online_predict_update_smse(
    Xn, Yn, hps, x_dim,
    nearest_k=4, max_experts=80, max_data_per_expert=50, timescale=0.0,
    warmup=200,
    print_each=False,
    stats=None,              # 新增：用于反标准化
):
    """
    逐样本在线预测+更新，打印标准化和物理单位下的预测/真值。
    返回：model, y_pred_std(N,1), smse_std_curve(N,), final_smse_std
    """
    model = SkyGP_rBCM(
        x_dim=x_dim, y_dim=1,
        max_data_per_expert=max_data_per_expert,
        nearest_k=nearest_k, max_experts=max_experts,
        replacement=False,
        pretrained_params=hps,
        timescale=timescale,
    )

    N = len(Xn)
    y_pred_std = np.zeros((N, 1), dtype=np.float32)

    var_global_std = float(np.var(Yn[:, 0], ddof=1))
    if var_global_std < 1e-12:
        var_global_std = 1e-12

    smse_std_curve = np.zeros(N, dtype=np.float64)
    sum_sq_err_std = 0.0

    # 若有 stats，则预取反标准化参数
    if stats is not None:
        _, _, Ym, Ys = stats
        Ym, Ys = float(Ym[0]), float(Ys[0])

    for i in tqdm(range(N), desc="Online predict+update"):
        x = Xn[i].reshape(-1)
        y = Yn[i].reshape(-1)

        if i < warmup:
            model.add_point(x, y)
            smse_std_curve[i] = 0.0 if i == 0 else smse_std_curve[i - 1]
            if print_each:
                if stats is not None:
                    y_real = y[0] * Ys + Ym
                    tqdm.write(f"[warmup {i:5d}] y_true = {y_real:10.6f} Nm (std={y[0]:.6f})")
                else:
                    tqdm.write(f"[warmup {i:5d}] y_true_std = {y[0]:.6f}")
            continue

        # --- 预测 ---
        mu, _ = model.predict(x)
        y_pred_std[i, 0] = mu[0]

        # --- 打印 ---
        if print_each:
            err_std = mu[0] - y[0]
            if stats is not None:
                y_pred_real = mu[0] * Ys + Ym
                y_true_real = y[0] * Ys + Ym
                err_real = y_pred_real - y_true_real
                tqdm.write(
                    f"[step {i:5d}] y_pred = {y_pred_real:10.6f} Nm | y_true = {y_true_real:10.6f} Nm | err = {err_real:9.6f} Nm"
                )
            else:
                tqdm.write(
                    f"[step {i:5d}] y_pred_std = {mu[0]:.6f} | y_true_std = {y[0]:.6f} | err_std = {err_std:.6f}"
                )

        # --- 更新 ---
        model.add_point(x, y)

        # --- SMSE ---
        err_i = float(y_pred_std[i, 0] - Yn[i, 0])
        sum_sq_err_std += err_i * err_i
        t_eff = (i - warmup + 1)
        mse_t = sum_sq_err_std / t_eff
        smse_std_curve[i] = mse_t  # 若要真SMSE, 可除以 var_global_std

    final_smse_std = float(smse_std_curve[-1])
    return model, y_pred_std, smse_std_curve, final_smse_std




# ---------------- Main ----------------
if __name__ == "__main__":
    data = np.load("gp_train_data_per_joint.npz", allow_pickle=True)
    os.makedirs("gp_models", exist_ok=True)

    # 选择关节（示例：只跑第5个；可改为 range(1,8)）
    for j in range(1,7):
        X = data[f"X{j}"]        # (N, D)
        Y = data[f"Y{j}"]        # (N, 1)

        # 1) 标准化
        Xn, Yn, stats = standardize(X, Y)

        # 2) 拟合全局超参（标准化空间）
        hps = fit_global_hparams(
            Xn, Yn, max_pts_hparam=2000, iters=200, lr=0.02
        )

        # 3) 在线预测 + 增量更新 + 标准化 SMSE（分母=全局方差）
        warmup = 50
        model, y_pred_std, smse_std_curve, final_smse_std = online_predict_update_smse(
            Xn, Yn, hps=hps, x_dim=Xn.shape[1],
            nearest_k=2, max_experts=80, max_data_per_expert=50, timescale=0.05,
            warmup=warmup,print_each=True
        )

        # 4) 原单位下 SMSE（同样用全局方差；与标准化逻辑一致）
        y_pred_real = destandardize(y_pred_std[:, 0], stats)
        y_true_real = Y[:, 0].astype(np.float64)
        smse_real_curve = smse_curve_from_preds(y_true_real, y_pred_real, warmup=warmup)
        final_smse_real = float(smse_real_curve[-1])

        print(f"[joint {j}] SMSE (std space, prefix)  = {final_smse_std:.6f}")
        print(f"[joint {j}] SMSE (real unit, prefix) = {final_smse_real:.6f}")

        # 5) 画图并保存
        out_png_std  = f"gp_models/joint{j}_smse_std.png"
        out_png_real = f"gp_models/joint{j}_smse_real.png"
        plot_curve(smse_std_curve,  out_png_std,  title=f"Joint {j} - SMSE (standardized, global var)")
        plot_curve(smse_real_curve, out_png_real, title=f"Joint {j} - SMSE (real unit, global var)")
        print(f"🖼 saved {out_png_std} & {out_png_real}")

        # 6) 存档
        with open(f"gp_models/joint{j}.pkl", "wb") as f:
            pickle.dump({
                "model": model,
                "stats": stats,                # (Xm, Xs, Ym, Ys)
                "hps_std": hps,                # (sf, sn, ls) —— 标准化空间
                "smse_std_curve": smse_std_curve,
                "smse_real_curve": smse_real_curve,
                "final_smse_std": final_smse_std,
                "final_smse_real": final_smse_real,
                "y_pred_std": y_pred_std,      # 逐样本预测（标准化）
                "warmup": warmup,
            }, f)
        print(f"✔ saved gp_models/joint{j}.pkl ({X.shape[0]} samples)")

        # 读取额外列
        C = data[f"C{j}"].astype(np.float64)  # tau_cmd
        M = data[f"M{j}"].astype(np.float64)  # tau_measured
        G = data[f"G{j}"].astype(np.float64)  # gravity

        # 反标准化预测残差（Nm）
        y_pred_real = destandardize(y_pred_std[:, 0], stats)
        y_true_real = Y[:, 0].astype(np.float64)

        # 计算预测的关节力矩
        tau_pred_with_g = C + G + y_pred_real      # 和 tau_measured 对比
        tau_pred_no_g   = C + y_pred_real          # 和 (tau_measured - gravity) 对比（可选）

        # 画图（含重力版本，通常更直观）
        # 画图（含重力，对比 tau_measured / 预测力矩 / 指令力矩）
        plt.figure(figsize=(10,4))
        plt.plot(M,                label="tau_measured",                    linewidth=1.5)
        plt.plot(tau_pred_with_g,  label="tau_pred = tau_cmd + g + y_pred", linewidth=1.5)
        plt.plot(C + G,            label="tau_cmd + g",                      linewidth=1.0, linestyle="--")  # ← 指令（含重力）
        plt.plot(C,                label="tau_cmd (no g)",                   linewidth=1.0, linestyle=":")   # ← 你要的 tau_cmd
        plt.xlabel("Sample index")
        plt.ylabel("Torque [Nm]")
        plt.title(f"Joint {j} — Measured vs Predicted Torque")
        plt.grid(True); plt.legend()
        out_png_torque = f"gp_models/joint{j}_tau_pred_vs_measured.png"
        plt.tight_layout(); plt.savefig(out_png_torque, dpi=200, bbox_inches='tight'); plt.close()
        print(f"🖼 saved {out_png_torque}")


        # 去重力对比（sanity check）：(measured - g) / (command + y_pred) / command
        plt.figure(figsize=(10,4))
        plt.plot(M - G,           label="tau_measured - gravity", linewidth=1.5)
        plt.plot(tau_pred_no_g,   label="tau_cmd + y_pred",        linewidth=1.5)
        plt.plot(C,               label="tau_cmd",                 linewidth=1.0, linestyle="--")  # ← 你要的 tau_cmd
        plt.xlabel("Sample index")
        plt.ylabel("Torque [Nm]")
        plt.title(f"Joint {j} — (Measured - g) vs (Command + y_pred)")
        plt.grid(True); plt.legend()
        out_png_torque2 = f"gp_models/joint{j}_tau_pred_vs_measured_minus_g.png"
        plt.tight_layout(); plt.savefig(out_png_torque2, dpi=200, bbox_inches='tight'); plt.close()
        print(f"🖼 saved {out_png_torque2}")

