#!/usr/bin/env python3
import os
import pickle
import numpy as np
import argparse

from skygp import SkyGP_rBCM
from hyperparam_training import fit_hparams_gpytorch


# ---------- Utils ----------
def standardize(X, Y):
    """
    标准化到 0 均值、1 方差，并返回 (Xm, Xs, Ym, Ys)
    """
    Xm = X.mean(0)
    Xs = X.std(0)
    Xs[Xs < 1e-9] = 1.0

    Ym = Y.mean(0)
    Ys = Y.std(0)
    Ys[Ys < 1e-9] = 1.0

    Xn = (X - Xm) / Xs
    Yn = (Y - Ym) / Ys

    return Xn.astype(np.float32), Yn.astype(np.float32), (Xm, Xs, Ym, Ys)


def fit_global_hparams(
    Xn, Yn,
    max_pts_hparam=3000,
    iters=600,
    lr=0.03,
    print_every=100
):
    """
    使用 gpytorch 拟合 rBF 核的全局超参。
    """
    N = len(Xn)
    if N > max_pts_hparam:
        idx = np.random.choice(N, max_pts_hparam, replace=False)
        Xh = Xn[idx]
        Yh = Yn[idx, 0]
    else:
        Xh = Xn
        Yh = Yn[:, 0]

    outputscale, noise, lengthscale = fit_hparams_gpytorch(
        Xh.astype(np.float32),
        Yh.astype(np.float32),
        max_points=min(max_pts_hparam, len(Xh)),
        iters=iters,
        lr=lr,
        use_cuda_if_available=True,
        print_every=print_every
    )
    return outputscale, noise, lengthscale


def build_rBCM(x_dim, hps,
               max_data_per_expert=64,
               nearest_k=4,
               max_experts=64,
               replacement=False,
               timescale=0.03):
    """
    根据拟合好的超参构建 rBCM GP（不进行预测/不在线学习）
    """
    model = SkyGP_rBCM(
        x_dim=x_dim,
        y_dim=1,
        max_data_per_expert=max_data_per_expert,
        nearest_k=nearest_k,
        max_experts=max_experts,
        replacement=replacement,
        pretrained_params=hps,
        timescale=timescale,
    )
    return model


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Train per-joint GP models with full control.")

    parser.add_argument("--data", default="gp_train_data_per_joint_no_filter.npz",
                        help="输入的训练数据 npz 文件")
    parser.add_argument("--joint", default="all",
                        help="'all' 或 指定关节编号 (1..7)")
    parser.add_argument("--iters", type=int, default=800,
                        help="超参优化迭代次数")
    parser.add_argument("--lr", type=float, default=0.04,
                        help="超参学习率")
    parser.add_argument("--max-hp", type=int, default=3000,
                        help="用于超参拟合的最大数据点数量")
    parser.add_argument("--max-exp", type=int, default=64,
                        help="最大 experts 数量")
    parser.add_argument("--k", type=int, default=4,
                        help="nearest_k")
    parser.add_argument("--mde", type=int, default=64,
                        help="max_data_per_expert")
    parser.add_argument("--timescale", type=float, default=0.03,
                        help="在线学习遗忘因子（训练模型时也存进去）")
    parser.add_argument("--out-dir", default="gp_models",
                        help="输出模型路径")

    args = parser.parse_args()

    # -------------------------
    # 载入数据
    # -------------------------
    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")

    data = np.load(args.data, allow_pickle=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # 解析关节列表
    if args.joint == "all":
        joints = list(range(1, 7))
    else:
        j = int(args.joint)
        if not (1 <= j <= 7):
            raise ValueError("Joint must be 1..7 or 'all'")
        joints = [j]

    # -------------------------
    # 逐关节训练
    # -------------------------
    for j in joints:
        print(f"\n========== Training joint {j} ==========")

        X = data[f"X{j}"]
        Y = data[f"Y{j}"]

        print(f"Samples: {X.shape[0]} | x_dim={X.shape[1]}")

        # 1) 标准化
        Xn, Yn, stats = standardize(X, Y)

        # 2) 拟合超参
        hps = fit_global_hparams(
            Xn, Yn,
            max_pts_hparam=args.max_hp,
            iters=args.iters,
            lr=args.lr,
            print_every=100
        )
        sf, sn, ls = hps
        print(f"[joint {j}] hparams: sf^2={float(sf):.4f}, sn^2={float(sn):.4f}, ls={np.array(ls)}")

        # 3) 构建模型（不在线学习）
        model = build_rBCM(
            x_dim=Xn.shape[1],
            hps=hps,
            max_data_per_expert=args.mde,
            nearest_k=args.k,
            max_experts=args.max_exp,
            timescale=args.timescale
        )

        # 4) 保存
        out_pkl = os.path.join(args.out_dir, f"joint{j}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump({
                "model": model,
                "stats": stats,         # (Xm, Xs, Ym, Ys)
                "hps_std": hps,         # (sf, sn, ls)
            }, f)

        print(f"✔ Saved GP model for joint {j}: {out_pkl}")


# -------------------------
if __name__ == "__main__":
    main()
