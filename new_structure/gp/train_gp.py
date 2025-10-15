import numpy as np, pickle, os
from skygp import SkyGP_rBCM
from hyperparam_training import fit_hparams_gpytorch

def standardize(X, Y):
    Xm, Xs = X.mean(0), X.std(0); Xs[Xs<1e-9]=1.0
    Ym, Ys = Y.mean(0), Y.std(0); Ys[Ys<1e-9]=1.0
    Xn = (X - Xm)/Xs; Yn = (Y - Ym)/Ys
    return Xn.astype(np.float32), Yn.astype(np.float32), (Xm, Xs, Ym, Ys)

def train_one_joint(X, Y, x_dim, max_pts_hparam=1000):
    # 1) 用 GPyTorch 在子集上学全局超参（和你的脚本一致）
    Xh = X if len(X)<=max_pts_hparam else X[np.random.choice(len(X), max_pts_hparam, replace=False)]
    Yh = Y if len(Y)<=max_pts_hparam else Y[np.random.choice(len(Y), max_pts_hparam, replace=False)]
    outputscale, noise, lengthscale = fit_hparams_gpytorch(
        Xh.astype(np.float32), Yh[:,0].astype(np.float32),
        max_points=min(max_pts_hparam, len(Xh)),
        iters=300, lr=0.1, use_cuda_if_available=True, print_every=50
    )
    # 2) 构建并离线填充专家
    model = SkyGP_rBCM(
        x_dim=x_dim, y_dim=1,
        max_data_per_expert=64, nearest_k=3, max_experts=16,
        replacement=False,
        pretrained_params=(outputscale, noise, lengthscale),
        timescale=0.05,
    )
    model.offline_pretrain(X, Y, optimize_hparams=False, show_progress=True)
    return model, (outputscale, noise, lengthscale)

if __name__ == "__main__":
    data = np.load("gp_train_data_per_joint.npz", allow_pickle=True)
    os.makedirs("gp_models", exist_ok=True)
    for j in range(1,8):
        X = data[f"X{j}"]; Y = data[f"Y{j}"]
        Xn, Yn, stats = standardize(X, Y)  # 重要：标准化并保存
        model, hps = train_one_joint(Xn, Yn, x_dim=Xn.shape[1])
        with open(f"gp_models/joint{j}.pkl", "wb") as f:
            pickle.dump({"model":model, "stats":stats, "hps":hps}, f)
        print(f"✔ saved gp_models/joint{j}.joblib ({X.shape[0]} samples)")
