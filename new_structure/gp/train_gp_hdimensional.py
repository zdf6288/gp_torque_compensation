#!/usr/bin/env python3
import os
import pickle
import numpy as np
import argparse

from skygp import SkyGP_rBCM
try:
    from hyperparam_training import fit_hparams_gpytorch
except ModuleNotFoundError as exc:
    fit_hparams_gpytorch = None
    GPYTORCH_IMPORT_ERROR = exc
else:
    GPYTORCH_IMPORT_ERROR = None


# ---------- Utils ----------
def standardize(X, Y):
    """
    标准化到 0 均值 / 1 方差
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
    使用 GPyTorch 拟合 RBF kernel 的全局超参
    """
    N = len(Xn)
    if N > max_pts_hparam:
        idx = np.random.choice(N, max_pts_hparam, replace=False)
        Xh = Xn[idx]
        Yh = Yn[idx, 0]
    else:
        Xh = Xn
        Yh = Yn[:, 0]

    if fit_hparams_gpytorch is not None:
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

    print(
        "[WARN] GPyTorch hyperparameter fitting unavailable "
        f"({GPYTORCH_IMPORT_ERROR}); using standardized default GP hyperparameters."
    )
    y_std = float(np.std(Yh))
    outputscale = np.array([max(y_std, 1e-6)], dtype=float)
    noise = np.array([max(1e-4, 0.05 * y_std)], dtype=float)
    lengthscale = np.ones(Xn.shape[1], dtype=float)
    return outputscale, noise, lengthscale


def reset_prediction_state(model):
    model.last_sorted_experts = None
    model.last_prediction_cache = {}
    model.last_x = None
    model.last_expert_idx = None


def validate_frozen_local_model(model, joint):
    required_attrs = [
        "X_list",
        "Y_list",
        "localCount",
        "expert_centers",
        "model_params",
        "L_all",
        "alpha_all",
    ]
    missing_or_empty = [
        name for name in required_attrs
        if len(getattr(model, name, [])) == 0
    ]
    if missing_or_empty:
        raise RuntimeError(
            f"joint {joint}: frozen local model has empty state: {missing_or_empty}"
        )

    if not any(int(count) > 0 for count in model.localCount):
        raise RuntimeError(f"joint {joint}: frozen local model has no stored samples")


def build_rBCM(x_dim, hps,
               max_data_per_expert=100,
               nearest_k=4,
               max_experts=64,
               replacement=False,
               timescale=0.03):
    """
    构建 rBCM GP（离线版，用于 ROS 推理时在线更新）
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
    parser = argparse.ArgumentParser(description="Train per-joint GP models (full input)")

    parser.add_argument("--data", default="gp_train_data_per_joint_no_filter.npz",
                        help="高维训练数据 npz （包含 X1..X7, Y1..Y7）")
    parser.add_argument("--joint", default="all",
                        help="'all' 或 1..7")
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--max-hp", type=int, default=3000)
    parser.add_argument("--max-exp", type=int, default=64)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--mde", type=int, default=64)
    parser.add_argument("--timescale", type=float, default=0.03)
    parser.add_argument("--out-dir", default="gp_models",
                        help="保存模型的目录")

    args = parser.parse_args()

    # -------------------------
    # 载入训练数据
    # -------------------------
    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")

    data = np.load(args.data, allow_pickle=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # 关节列表
    if args.joint == "all":
        joints = list(range(1, 8))   # <-- 这里改成 1..7 全部关节
    else:
        j = int(args.joint)
        if not (1 <= j <= 7):
            raise ValueError("Joint must be 1..7 or 'all'")
        joints = [j]

    # -------------------------
    # 逐关节训练
    # -------------------------
        # -------------------------
    # 逐关节训练
    # -------------------------
    for j in joints:
        print(f"\n========== Training joint {j} ==========")

        X = data[f"X{j}"]   # (N, 14) or (N, 21)
        Y = data[f"Y{j}"]   # (N, 1)

        print(f"Samples: {X.shape[0]} | x_dim={X.shape[1]}")

        # --- 标准化 ---
        Xn, Yn, stats = standardize(X, Y)

        # --- 拟合超参（GPyTorch）---
        hps = fit_global_hparams(
            Xn, Yn,
            max_pts_hparam=args.max_hp,
            iters=args.iters,
            lr=args.lr,
            print_every=100
        )

        sf, sn, ls = hps
        print(f"[joint {j}] hparams: sf²={float(sf):.6f}, sn²={float(sn):.6f}")
        print(f"lengthscale = {np.array(ls)}")

        # ---------------------------------------------------
        # ① Local GP：带所有训练点（用于本地在线更新）
        # ---------------------------------------------------
        # 构建 local 模型
        frozen_mde = max(args.mde, Xn.shape[0])
        local_model = build_rBCM(
            x_dim=X.shape[1],
            hps=hps,
            max_data_per_expert=frozen_mde,
            nearest_k=1,
            max_experts=1,
            timescale=args.timescale
        )

        local_model.offline_pretrain(
            Xn,
            Yn,
            show_progress=True,
            optimize_hparams=False,
        )
        validate_frozen_local_model(local_model, j)
        reset_prediction_state(local_model)
        mu_check, var_check = local_model.predict(Xn[0])
        if not (np.all(np.isfinite(mu_check)) and np.all(np.isfinite(var_check))):
            raise RuntimeError(f"joint {j}: frozen local model predict sanity check failed")
        reset_prediction_state(local_model)
        print(
            f"[joint {j}] frozen local state: experts={len(local_model.X_list)}, "
            f"samples={int(np.sum(local_model.localCount))}"
        )

        # ---------------------------------------------------
        # ② Cloud GP：无训练数据（只共享超参）
        # ---------------------------------------------------
        cloud_model = build_rBCM(
            x_dim=X.shape[1],
            hps=hps,
            max_data_per_expert=args.mde,
            nearest_k=args.k,
            max_experts=args.max_exp,
            timescale=args.timescale
        )

        # cloud_model 不加入任何点

        # ---------------------------------------------------
        # 保存 local & cloud 两种模型
        # ---------------------------------------------------
        out_local = os.path.join(args.out_dir, f"joint{j}_local.pkl")
        out_cloud = os.path.join(args.out_dir, f"joint{j}_cloud.pkl")

        with open(out_local, "wb") as f:
            pickle.dump({
                "model": local_model,
                "stats": stats,
                "hps_std": hps,
                "type": "local"
            }, f)

        with open(out_cloud, "wb") as f:
            pickle.dump({
                "model": cloud_model,
                "stats": stats,
                "hps_std": hps,
                "type": "cloud"
            }, f)

        print(f"✔ Saved LOCAL GP model to: {out_local}")
        print(f"✔ Saved CLOUD GP model to: {out_cloud}")


# -------------------------
if __name__ == "__main__":
    main()
