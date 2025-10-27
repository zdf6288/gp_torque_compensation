# gpytorch_hparam_fit.py
import numpy as np
import torch
import gpytorch

# ---------------------------
# Exact GP 模型（常数均值 + ARD RBF）
# ---------------------------
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ard_num_dims):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)

        # 长度尺度先验（标准化后 ls~1 左右较合理）
        base_kernel.register_prior(
            "lengthscale_prior",
            gpytorch.priors.LogNormalPrior(loc=0.0, scale=0.5),  # 对数空间 ~ N(0, 0.5^2)
            "lengthscale",
        )

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        # outputscale 先验（方差），防止塌到 0
        self.covar_module.register_prior(
            "outputscale_prior",
            gpytorch.priors.LogNormalPrior(loc=0.0, scale=0.8),
            "outputscale",
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


@torch.no_grad()
def _to_numpy_hparams(model, likelihood, x_std=None, y_std=None):
    """
    导出 (sigma_f, sigma_n, lengthscale) 为 numpy。
    x_std/y_std 可选，仅保留兼容；若传入也不做缩放（参数已在标准化空间训练）。
    """
    sf2 = float(model.covar_module.outputscale.detach().cpu().item())
    sf  = np.sqrt(sf2) if sf2 > 0 else 1.0

    sn2 = float(likelihood.noise.detach().cpu().item())
    sn  = np.sqrt(sn2) if sn2 > 0 else 1e-3

    ls  = model.covar_module.base_kernel.lengthscale.detach().cpu().view(-1).numpy()
    ls  = np.maximum(ls.astype(float), 1e-9)  # 数值下限，防止0

    return np.array([sf], dtype=float), np.array([sn], dtype=float), ls


def fit_hparams_gpytorch(X, y, max_points=1000, iters=300, lr=0.1,
                         use_cuda_if_available=True, print_every=50):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).reshape(-1).astype(np.float32)

    # 采样
    N = X.shape[0]
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        Xs, ys = X[idx], y[idx]
    else:
        Xs, ys = X, y

    # 过滤非有限
    finite_mask = np.isfinite(Xs).all(axis=1) & np.isfinite(ys)
    Xs, ys = Xs[finite_mask], ys[finite_mask]
    if Xs.shape[0] < 5:
        raise ValueError("Too few finite samples for hparam fitting.")

    device = "cuda" if (use_cuda_if_available and torch.cuda.is_available()) else "cpu"
    tx = torch.from_numpy(Xs).to(device)
    ty = torch.from_numpy(ys).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    # model = ExactGPModel(tx, ty, likelihood, ard_num_dims=tx.shape(-1)).to(device)
    model = ExactGPModel(tx, ty, likelihood, ard_num_dims=tx.size(-1)).to(device)

    # 约束
    from gpytorch.constraints import GreaterThan, Interval
    likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-6))
    model.covar_module.base_kernel.register_constraint(
        "raw_lengthscale",
        Interval(lower_bound=torch.tensor(1e-3, device=tx.device),
                 upper_bound=torch.tensor(1e3,  device=tx.device))
    )
    model.covar_module.register_constraint("raw_outputscale", GreaterThan(1e-6))

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    jitter_ctx = gpytorch.settings.cholesky_jitter(1e-3)

    # 强制用 Cholesky（推荐）
    chol_ctx = gpytorch.settings.max_cholesky_size(float('inf'))

    # Adam 预热
    model.train(); likelihood.train()
    opt_adam = torch.optim.Adam(model.parameters(), lr=lr)
    adam_iters = max(100, iters // 2)
    with chol_ctx:
        for i in range(1, adam_iters + 1):
            opt_adam.zero_grad()
            with jitter_ctx:
                out = model(tx)
                loss = -mll(out, ty)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt_adam.step()
            if print_every and (i % print_every == 0 or i == 1 or i == adam_iters):
                sf2 = model.covar_module.outputscale.detach().item()
                sn2 = likelihood.noise.detach().item()
                ls0 = model.covar_module.base_kernel.lengthscale.detach().view(-1)[0].item()
                print(f"[GPyTorch opt P1] iter {i:4d} | nll {loss.item():.4f} | sf^2 {sf2:.4g} | sn^2 {sn2:.4g} | ls0 {ls0:.4g}")


    # 导出参数
    model.eval(); likelihood.eval()
    return _to_numpy_hparams(model, likelihood)
