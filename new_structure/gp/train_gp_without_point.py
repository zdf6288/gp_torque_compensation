# train_gp.py —— 只训练超参+离线建模并保存（无预测/加点/画图）
import os
import pickle
import numpy as np

from skygp import SkyGP_rBCM
from hyperparam_training import fit_hparams_gpytorch


# ---------- Utils ----------
def standardize(X, Y):
    Xm, Xs = X.mean(0), X.std(0); Xs[Xs < 1e-9] = 1.0
    Ym, Ys = Y.mean(0), Y.std(0); Ys[Ys < 1e-9] = 1.0
    Xn = (X - Xm) / Xs
    Yn = (Y - Ym) / Ys
    return Xn.astype(np.float32), Yn.astype(np.float32), (Xm, Xs, Ym, Ys)


def fit_global_hparams(Xn, Yn, max_pts_hparam=2000, iters=600, lr=0.03, print_every=100):
    """
    在标准化后的数据上拟合超参（子采样加速）。
    返回 (outputscale, noise, lengthscale)
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
        iters=iters, lr=lr, use_cuda_if_available=True, print_every=print_every
    )
    return outputscale, noise, lengthscale


def build_rBCM(x_dim, hps,
               max_data_per_expert=64, nearest_k=3, max_experts=64,
               replacement=False, timescale=0.0):
    """
    根据已拟合超参构建 rBCM 模型（不做预测，不在线学习）
    """
    model = SkyGP_rBCM(
        x_dim=x_dim, y_dim=1,
        max_data_per_expert=max_data_per_expert,
        nearest_k=nearest_k, max_experts=max_experts,
        replacement=replacement,
        pretrained_params=hps,   # (sf, sn, ls) 在标准化空间
        timescale=timescale,
    )
    return model


# ---------- Main ----------
if __name__ == "__main__":
    np.random.seed(0)

    data_path = "gp_train_data_per_joint.npz"
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = np.load(data_path, allow_pickle=True)
    os.makedirs("gp_models", exist_ok=True)

    # 训练 1..6 关节（按需修改）
    for j in range(1, 7):
        X = data[f"X{j}"]        # (N, D)  训练用输入，例如 [q] 或 [q, dq_des, ddq_des]
        Y = data[f"Y{j}"]        # (N, 1)  训练用输出：残差力矩 y

        print(f"\n==== Joint {j} ====")
        print(f"Samples: {X.shape[0]} | x_dim: {X.shape[1]}")

        # 1) 标准化
        Xn, Yn, stats = standardize(X, Y)

        # 2) 拟合全局超参（标准化空间）
        hps = fit_global_hparams(
            Xn, Yn,
            max_pts_hparam=2000,   # 子采样加速上限
            iters=300,             # 迭代步数（按需调）
            lr=0.03,               # 学习率（按需调）
            print_every=100
        )
        sf, sn, ls = hps
        ls_arr = np.asarray(ls).ravel()
        ls_str = np.array2string(ls_arr, precision=4, separator=',', suppress_small=True)
        print(f"[joint {j}] fitted hparams: sf^2={float(sf):.4g}, sn^2={float(sn):.4g}, ls={ls_str}")



        # 3) 构建 rBCM，并用全量数据离线灌入（不优化超参）
        model = build_rBCM(
            x_dim=Xn.shape[1], hps=hps,
            max_data_per_expert=64, nearest_k=3, max_experts=64,
            replacement=False, timescale=0.03
        )

        # 离线预填充专家
        # optimize_hparams=False -> 固定用上面的全局超参
        # print(f"[joint {j}] offline_pretrain ...")
        # model.offline_pretrain(Xn, Yn, optimize_hparams=False, show_progress=True)
        # print(f"[joint {j}] offline_pretrain done.")

        # 4) 保存
        out_pkl = f"gp_models/joint{j}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump({
                "model": model,
                "stats": stats,     # (Xm, Xs, Ym, Ys)
                "hps_std": hps,     # (sf, sn, ls) —— 标准化空间
            }, f)
        print(f"✔ saved {out_pkl} ({X.shape[0]} samples)")
