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
def _to_numpy_hparams(model, likelihood, x_std, y_std):
    """
    把标准化空间学习到的参数还原到原始单位：
      - lengthscale: 乘以对应列的 x_std
      - sigma_f, sigma_n（标准差，不是方差）: 乘以 y_std
    """
    sf2 = float(model.covar_module.outputscale.detach().cpu().item())
    sn2 = float(likelihood.noise.detach().cpu().item())
    ls_scaled = (
        model.covar_module.base_kernel.lengthscale.detach()
        .cpu()
        .view(-1)
        .numpy()
        .astype(np.float64)
    )

    x_std = np.asarray(x_std, dtype=np.float64).reshape(-1)
    y_std = float(y_std)

    ls = ls_scaled * (x_std + 1e-12)
    sf = np.sqrt(max(sf2, 1e-12)) * y_std
    sn = np.sqrt(max(sn2, 1e-12)) * y_std

    return np.array([sf], dtype=float), np.array([sn], dtype=float), ls.astype(float)


def fit_hparams_gpytorch(
    X,
    y,
    max_points=2000,
    iters=500,
    lr=0.05,
    use_cuda_if_available=True,
    print_every=50,
    freeze_noise_steps=50,   # 先冻结噪声训练核的步数；0 关闭
    use_lbfgs_refine=True,   # 是否用 LBFGS 小步精修
):
    """
    拟合 ARD RBF 超参（lengthscale, outputscale, noise），返回:
      outputscale: (1,)  -> sigma_f（标准差）
      noise:       (1,)  -> sigma_n（标准差）
      lengthscale: (D,)
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    N, D = X.shape

    # 子采样，保证可训练
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        Xs, ys = X[idx], y[idx]
    else:
        Xs, ys = X, y

    # ------- 标准化（很关键） -------
    x_mean = Xs.mean(axis=0, keepdims=True)
    x_std = Xs.std(axis=0, keepdims=True) + 1e-12
    Xn = (Xs - x_mean) / x_std

    y_mean = ys.mean()
    y_std = ys.std() + 1e-12
    yn = (ys - y_mean) / y_std

    device = "cuda" if (use_cuda_if_available and torch.cuda.is_available()) else "cpu"
    tx = torch.from_numpy(Xn).to(device)
    ty = torch.from_numpy(yn).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    # 把噪声限制在合理范围（标准化空间）
    likelihood.noise_covar.register_constraint(
        "raw_noise", gpytorch.constraints.Interval(1e-6, 0.3)  # sn^2 ∈ [1e-6, 0.3] -> sn ∈ [1e-3, ~0.55]
    )

    model = ExactGPModel(tx, ty, likelihood, ard_num_dims=D).to(device)

    # ------- 初始化 -------
    with torch.no_grad():
        model.mean_module.initialize(constant=0.0)
        model.covar_module.base_kernel.initialize(lengthscale=torch.ones(D, device=device))
        model.covar_module.initialize(outputscale=1.0)  # sf^2 ~ 1
        likelihood.initialize(noise=0.05)               # sn^2 ~ 0.05

    model.train()
    likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # ------- 优化器：Adam 预训练 -------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Phase 1：可选，先固定噪声只学核（防止“全靠增大噪声来解释”）
    if freeze_noise_steps and freeze_noise_steps > 0:
        for p in likelihood.parameters():
            p.requires_grad_(False)
        for i in range(1, freeze_noise_steps + 1):
            optimizer.zero_grad()
            loss = -mll(model(tx), ty)
            loss.backward()
            optimizer.step()
            if print_every and (i % print_every == 0 or i == 1 or i == freeze_noise_steps):
                sf2 = model.covar_module.outputscale.detach().item()
                sn2 = likelihood.noise.detach().item()
                print(f"[GPyTorch opt P1] iter {i:4d} | nll {loss.item():.4f} | sf^2 {sf2:.4g} | sn^2 {sn2:.4g}")

        for p in likelihood.parameters():
            p.requires_grad_(True)

    # Phase 2：放开噪声一起学
    total_iters = iters
    for i in range(1, total_iters + 1):
        optimizer.zero_grad()
        loss = -mll(model(tx), ty)
        loss.backward()
        optimizer.step()
        if print_every and (i % print_every == 0 or i == 1 or i == total_iters):
            sf2 = model.covar_module.outputscale.detach().item()
            sn2 = likelihood.noise.detach().item()
            ls = model.covar_module.base_kernel.lengthscale.detach().view(-1)
            ls_show = [float(v) for v in ls[: min(3, D)]]
            print(f"[GPyTorch opt P2] iter {i:4d} | nll {loss.item():.4f} "
                  f"| sf^2 {sf2:.4g} | sn^2 {sn2:.4g} | ls[:3] {ls_show}")

    # ------- 可选：LBFGS 小步精修（常能再降一点 NLL） -------
    if use_lbfgs_refine:
        model.train(); likelihood.train()

        def closure():
            optimizer_lbfgs.zero_grad()
            out = model(tx)
            loss = -mll(out, ty)
            loss.backward()
            return loss

        # 只做少量步
        optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=50, line_search_fn="strong_wolfe")
        loss = optimizer_lbfgs.step(closure)
        if print_every:
            sf2 = model.covar_module.outputscale.detach().item()
            sn2 = likelihood.noise.detach().item()
            print(f"[LBFGS refine] nll {float(loss):.4f} | sf^2 {sf2:.4g} | sn^2 {sn2:.4g}")

    # ------- 导出参数（反标准化） -------
    model.eval()
    likelihood.eval()
    return _to_numpy_hparams(model, likelihood, x_std=x_std.reshape(-1), y_std=y_std)
