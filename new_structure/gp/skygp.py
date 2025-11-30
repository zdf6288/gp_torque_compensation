import time
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from tqdm import tqdm


class SkyGP_rBCM:
    """
    两阶段流程（含可选超参数训练）：
      - offline_pretrain(X_train, Y_train, optimize_hparams=True, ...):
          可选 L-BFGS-B 优化 RBF 超参（全局），再构建专家并分解
      - online_step(x_new, y_new):
          对新点先预测再增量更新
      - predict(x): 仅预测

    超参训练说明：
      - 优化目标：标准 GP 边缘负对数似然（全局单 GP 近似）
      - 变量：log_lengthscale (D), log_outputscale, log_noise
      - 先用子集（可设上限）优化以控内存/时间；结果用于初始化所有专家
    """

    def __init__(
        self,
        x_dim,
        y_dim=1,
        max_data_per_expert=50,
        nearest_k=2,
        max_experts=8,
        replacement=False,
        pretrained_params=None,  # (outputscale, noise, lengthscale)
        timescale=0.0,
    ):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.max_data = max_data_per_expert
        self.max_experts = max_experts
        self.nearest_k = nearest_k
        self.pretrained_params = pretrained_params

        self.X_list = []
        self.Y_list = []
        self.localCount = []
        self.expert_centers = []
        self.drop_centers = []
        self.drop_counts = []
        self.model_params = {}
        self.L_all = []
        self.alpha_all = []

        self.last_sorted_experts = None
        self.last_prediction_cache = {}
        self.replacement = replacement
        self.expert_creation_order = []
        self.expert_weights = []
        self.last_x = None
        self.last_expert_idx = None
        self.expert_dict = {}
        self.expert_id_counter = 0
        self.timescale = timescale

    # ---------- 核函数与参数 ----------

    def kernel_np(self, X1, X2, lengthscale, sigma_f):
        # 统一形状
        X1 = np.asarray(X1).reshape(self.x_dim, -1)   # (D, N1)
        X2 = np.asarray(X2).reshape(self.x_dim, -1)   # (D, N2)
        ls = np.asarray(lengthscale, dtype=float).reshape(-1)
        if ls.size == 1 and self.x_dim > 1:           # 标量 -> 广播到每一维
            ls = np.repeat(ls, self.x_dim)
        elif ls.size != self.x_dim:                   # 容错：尺寸不对也广播
            ls = np.broadcast_to(ls, (self.x_dim,))
        X1_scaled = X1 / ls[:, None]
        X2_scaled = X2 / ls[:, None]
        dists = np.sum((X1_scaled[:, :, None] - X2_scaled[:, None, :]) ** 2, axis=0)
        return float(sigma_f) ** 2 * np.exp(-0.5 * dists)


    def _set_or_guess_hparams(self, X, Y):
        """若没给超参，就用启发式估计：lengthscale=各维中位间距；outputscale=std(Y)；noise=0.05*std(Y)+eps"""
        if self.pretrained_params is not None:
            return
        X = np.asarray(X)
        Y = np.asarray(Y).reshape(-1, 1)
        N = min(2000, X.shape[0])
        idx = np.random.choice(X.shape[0], N, replace=False) if X.shape[0] > N else np.arange(X.shape[0])
        Xs = X[idx]
        # 估计 lengthscale
        if Xs.shape[0] > 1:
            diffs = []
            for d in range(self.x_dim):
                xd = np.sort(Xs[:, d])
                diff_d = np.median(np.diff(xd)) if xd.size > 1 else 1.0
                diffs.append(max(diff_d, 1e-3))
            ls = np.array(diffs, dtype=float)
        else:
            ls = np.ones(self.x_dim)
        out_std = float(np.std(Y)) if np.std(Y) > 0 else 1.0
        outputscale = np.array([max(out_std, 1e-6)], dtype=float)
        noise = np.array([max(1e-4, 0.05 * out_std)], dtype=float)
        self.pretrained_params = (outputscale, noise, ls)

    def init_model_params(self, model_id, pretrained_params=None):
        if pretrained_params:
            self.pretrained_params = pretrained_params

        if self.pretrained_params:
            outputscale, noise, lengthscale = self.pretrained_params
            log_sigma_f = np.log(np.atleast_1d(outputscale).flatten())
            log_sigma_n = np.log(np.atleast_1d(noise).flatten())

            ls = np.asarray(lengthscale, dtype=float).reshape(-1)
            if ls.size == 1 and self.x_dim > 1:
                ls = np.repeat(ls, self.x_dim)
            # y_dim==1 时用 (D,)；多输出时你可以扩成 (D, y_dim)
            log_lengthscale = np.log(ls)
        else:
            log_sigma_f = np.log(np.ones(self.y_dim))
            log_sigma_n = np.log(np.ones(self.y_dim) * 0.01)
            log_lengthscale = np.log(np.ones((self.x_dim,)))

        self.model_params[model_id] = {
            "log_sigma_f": log_sigma_f,
            "log_sigma_n": log_sigma_n,
            "log_lengthscale": log_lengthscale,
        }


    # ---------- 专家管理 ----------

    def _create_new_expert(self, model_id=None):
        if model_id is None:
            model_id = self.expert_id_counter
            self.expert_id_counter += 1

        self.X_list.append(np.zeros((self.x_dim, self.max_data)))
        self.Y_list.append(np.zeros((self.y_dim, self.max_data)))
        self.localCount.append(0)
        self.expert_centers.append(np.zeros(self.x_dim))
        self.drop_centers.append(np.zeros(self.x_dim))
        self.drop_counts.append(0)
        self.L_all.append(np.zeros((self.max_data, self.max_data)))
        self.alpha_all.append(np.zeros((self.max_data, self.y_dim)))
        self.expert_creation_order.append(model_id)
        self.expert_weights.append(1.0)
        self.init_model_params(model_id)
        self.expert_dict[model_id] = {"center": self.expert_centers[-1], "usage": 0, "created": True}
        return len(self.X_list) - 1

    def _insert_new_expert_near(self, near_idx):
        if self.last_x is None or len(self.expert_centers) <= 1:
            return self._create_new_expert()

        left_idx = max(near_idx - 1, 0)
        right_idx = min(near_idx + 1, len(self.expert_centers) - 1)

        outputscale, noise, lengthscale = self.pretrained_params
        sigma_f = np.atleast_1d(outputscale)[0]
        lengthscale = lengthscale if np.ndim(lengthscale) == 1 else lengthscale[:, 0]

        def k_to(idx):
            return self.kernel_np(self.last_x[:, None], self.expert_centers[idx][:, None], lengthscale, sigma_f)[0, 0]

        dist_left = k_to(left_idx)
        dist_right = k_to(right_idx)

        insert_after = near_idx if dist_right < dist_left else near_idx - 1
        insert_pos = min(max(insert_after + 1, 0), len(self.expert_centers))

        new_id = max(self.expert_dict.keys(), default=0) + 1

        self.X_list.insert(insert_pos, np.zeros((self.x_dim, self.max_data)))
        self.Y_list.insert(insert_pos, np.zeros((self.y_dim, self.max_data)))
        self.localCount.insert(insert_pos, 0)
        self.expert_centers.insert(insert_pos, np.zeros(self.x_dim))
        self.drop_centers.insert(insert_pos, np.zeros(self.x_dim))
        self.drop_counts.insert(insert_pos, 0)
        self.L_all.insert(insert_pos, np.zeros((self.max_data, self.max_data)))
        self.alpha_all.insert(insert_pos, np.zeros((self.max_data, self.y_dim)))
        self.expert_creation_order.insert(insert_pos, new_id)
        self.expert_weights.insert(insert_pos, 1.0)

        self.init_model_params(new_id)
        self.expert_dict[new_id] = {"center": self.expert_centers[insert_pos], "usage": 0}
        return insert_pos

    # ---------- 参数更新（批量/增量） ----------

    def update_param(self, model):
        p = 0
        idx = self.localCount[model]
        params = self.model_params[self.expert_creation_order[model]]
        sigma_f = np.exp(params["log_sigma_f"][p])
        sigma_n = np.exp(params["log_sigma_n"][p])
        lengthscale = np.exp(params["log_lengthscale"]) if self.y_dim == 1 else np.exp(params["log_lengthscale"][:, p])

        X_subset = self.X_list[model][:, :idx]
        Y_subset = self.Y_list[model][p, :idx]
        K = self.kernel_np(X_subset, X_subset, lengthscale, sigma_f)
        K[np.diag_indices_from(K)] += sigma_n**2
        try:
            L = np.linalg.cholesky(K + 1e-6 * np.eye(idx))
        except np.linalg.LinAlgError:
            print(f"⚠️ Cholesky failed for model {model}, using identity fallback")
            L = np.eye(idx)
        self.L_all[model][:idx, :idx] = L
        aux_alpha = solve_triangular(L, Y_subset, lower=True)
        self.alpha_all[model][:idx, p] = solve_triangular(L.T, aux_alpha, lower=False)

    def update_param_incremental(self, x, y, model):
        p = 0
        idx = self.localCount[model]
        if idx == 0:
            return

        params = self.model_params[self.expert_creation_order[model]]
        sigma_f = np.exp(params["log_sigma_f"][p])
        sigma_n = np.exp(params["log_sigma_n"][p])
        lengthscale = np.exp(params["log_lengthscale"]) if self.y_dim == 1 else np.exp(params["log_lengthscale"][:, p])

        if idx == 1:
            kxx = self.kernel_np(x[:, None], x[:, None], lengthscale, sigma_f)[0, 0] + sigma_n**2
            L = np.sqrt(kxx)
            self.L_all[model][0, 0] = L
            self.alpha_all[model][0, p] = y[p] / (L * L)
            return

        X_prev = self.X_list[model][:, : idx - 1]
        y_vals = self.Y_list[model][p, :idx]

        cached = self.last_prediction_cache.get(model, {}).get(p, None)
        if cached is not None and cached["k_star"].shape[0] == idx - 1:
            b = cached["k_star"]
            Lx = cached["v"]
        else:
            b = self.kernel_np(X_prev, x[:, None], lengthscale, sigma_f).flatten()
            L_prev = self.L_all[model][: idx - 1, : idx - 1]
            Lx = solve_triangular(L_prev, b, lower=True)

        c = self.kernel_np(x[:, None], x[:, None], lengthscale, sigma_f)[0, 0] + sigma_n**2
        Lii = np.sqrt(max(c - np.dot(Lx, Lx), 1e-9))
        self.L_all[model][: idx - 1, idx - 1] = 0.0
        self.L_all[model][idx - 1, : idx - 1] = Lx
        self.L_all[model][idx - 1, idx - 1] = Lii
        L_now = self.L_all[model][:idx, :idx]
        aux_alpha = solve_triangular(L_now, y_vals, lower=True)
        self.alpha_all[model][:idx, p] = solve_triangular(L_now.T, aux_alpha, lower=False)

    # ---------- 数据插入（内部） ----------

    def _insert_into_expert(self, model, x, y):
        idx = self.localCount[model]
        self.X_list[model][:, idx] = x
        self.Y_list[model][:, idx] = y
        self.localCount[model] += 1
        self.expert_centers[model] = x if idx == 0 else (self.expert_centers[model] * idx + x) / (idx + 1)
        expert_id = self.expert_creation_order[model]
        self.expert_dict[expert_id]["center"] = self.expert_centers[model]
        self.update_param_incremental(x, y, model)

    def add_point(self, x, y):
            x = np.asarray(x)
            y = np.asarray(y).reshape(-1)
            
            x_uncat = x.copy()
            y_uncat = y.copy()

            expert_order = self.last_sorted_experts if self.last_sorted_experts is not None else []
            # sorting experts by distance to x_uncat
            if self.last_sorted_experts is None:
                # if no last sorted experts, recalculate the distances
                # print("🔄 Recalculating expert order...")
                outputscale, noise, lengthscale = self.pretrained_params
                sigma_f = np.atleast_1d(outputscale)[0]
                lengthscale = (
                    lengthscale if lengthscale.ndim == 1
                    else lengthscale[:, 0]
                )
                dists = [(self.kernel_np(x_uncat[None, :], center[None, :], lengthscale, sigma_f)[0, 0], i) for i, center in enumerate(self.expert_centers)]
                if len(self.expert_centers) == 0:
                    expert_order = []
                else:
                    dists = [(self.kernel_np(x_uncat[None, :], center[None, :], lengthscale, sigma_f)[0, 0], i) for i, center in enumerate(self.expert_centers)]
                    dists.sort()
                    expert_order = [i for _, i in dists]
            
            for model in expert_order[:self.nearest_k]:
                self.expert_weights[model] = 1.0  # reset weight
                # if expert does not exist
                if model >= len(self.X_list):
                    self._create_new_expert(model)
                if self.localCount[model] < self.max_data:
                    idx = self.localCount[model]
                    self.X_list[model][:, idx] = x_uncat
                    self.Y_list[model][:, idx] = y_uncat
                    self.localCount[model] += 1
                    if idx == 0:
                        self.expert_centers[model] = x_uncat
                    else:
                        self.expert_centers[model] = (
                            self.expert_centers[model] * idx + x_uncat
                        ) / (idx + 1)
                    self.update_param_incremental(x_uncat, y_uncat, model)
                    expert_id = self.expert_creation_order[model]
                    self.expert_dict[expert_id]['center'] = self.expert_centers[model]
                    return
                else:
                    if not self.replacement:
                        # if no replacement logic, just skip
                        continue
                    elif self.drop_counts[model] == 0:
                        # initial drop logic
                        d_center = np.linalg.norm(self.expert_centers[model] - x_uncat)
                        stored = self.X_list[model][:, :self.max_data]
                        dists = np.linalg.norm(stored - self.expert_centers[model][:, None], axis=0)
                        max_idx = np.argmax(dists)
                        if d_center < dists[max_idx]:
                            # replace the farthest point
                            x_old = self.X_list[model][:, max_idx].copy()
                            y_old = self.Y_list[model][:, max_idx].copy()
                            self.X_list[model][:, max_idx] = x_uncat
                            self.Y_list[model][:, max_idx] = y_uncat
                            # update center
                            self.expert_centers[model] += (x_uncat - x_old) / self.max_data
                            expert_id = self.expert_creation_order[model]
                            self.expert_dict[expert_id]['center'] = self.expert_centers[model]
                            self.drop_centers[model] = x_old
                            self.drop_counts[model] += 1
                            x_uncat = x_old
                            y_uncat = y_old
                            self.update_param(model)
                            return
                    else:
                        stored = self.X_list[model][:, :self.max_data]
                        d_keep = np.linalg.norm(stored - self.expert_centers[model][:, None], axis=0)
                        d_drop = np.linalg.norm(stored - self.drop_centers[model][:, None], axis=0)
                        d_new_keep = np.linalg.norm(x_uncat - self.expert_centers[model])
                        d_new_drop = np.linalg.norm(x_uncat - self.drop_centers[model])
                        d_diff = np.concatenate([(d_keep - d_drop), [d_new_keep - d_new_drop]])
                        drop_idx = np.argmax(d_diff)
                        if drop_idx < self.max_data:
                            # replace the point in the expert
                            x_old = self.X_list[model][:, drop_idx].copy()
                            y_old = self.Y_list[model][:, drop_idx].copy()
                            self.X_list[model][:, drop_idx] = x_uncat
                            self.Y_list[model][:, drop_idx] = y_uncat
                            self.expert_centers[model] += (x_uncat - x_old) / self.max_data
                            self.drop_centers[model] = (
                                self.drop_centers[model] * self.drop_counts[model] + x_old
                            ) / (self.drop_counts[model] + 1)
                            self.drop_counts[model] += 1
                            x_uncat = x_old
                            y_uncat = y_old
                            self.update_param(model)
                            expert_id = self.expert_creation_order[model]
                            self.expert_dict[expert_id]['center'] = self.expert_centers[model]
                            return
            # if no expert can accommodate the point, create a new expert
            if self.last_expert_idx is not None:
                model = self._insert_new_expert_near(self.last_expert_idx)
            else:
                model = self._create_new_expert()
            self._insert_into_expert(model, x_uncat, y_uncat) 

    # ---------- 预测 ----------

    def predict(self, x_query):
        x_query = np.asarray(x_query).reshape(-1)
        self.last_prediction_cache.clear()

        if len(self.expert_centers) == 0:
            return np.zeros(self.y_dim), np.ones(self.y_dim) * 10.0

        raw_ls = self.model_params[self.expert_creation_order[0]]["log_lengthscale"]
        lengthscale_ref = np.exp(raw_ls[:, 0]) if np.ndim(raw_ls) == 2 else np.exp(raw_ls)

        if self.last_x is not None:
            norm_dist = np.linalg.norm((x_query - self.last_x) / lengthscale_ref)
        else:
            norm_dist = np.inf

        search_k = int(min(self.max_experts, max(1, np.exp(norm_dist / self.timescale))))
        # # print("search_k:",search_k)
        search_k = 4
        n_experts = len(self.expert_centers)

        if self.last_expert_idx is None:
            candidate_idxs = list(range(n_experts))
        else:
            half_k = search_k // 2
            start = max(0, self.last_expert_idx - half_k)
            end = min(n_experts, self.last_expert_idx + half_k + 1)
            candidate_idxs = list(range(start, end))

        min_weight_threshold = 1e-3
        valid_idxs = [idx for idx in candidate_idxs if self.expert_weights[idx] > min_weight_threshold]

        outputscale, _, lengthscale_all = self.pretrained_params
        sigma_f_ref = np.atleast_1d(outputscale)[0]
        lengthscale_all = lengthscale_all if np.ndim(lengthscale_all) == 1 else lengthscale_all[:, 0]

        dists = []
        for idx in valid_idxs:
            if self.localCount[idx] == 0:
                continue
            k_val = self.kernel_np(self.expert_centers[idx][:, None], x_query[:, None], lengthscale_all, sigma_f_ref)[0, 0]
            dists.append((k_val, idx))
        if not dists:
            return np.zeros(self.y_dim), np.ones(self.y_dim) * 10.0

        dists.sort(reverse=True)
        selected = [idx for _, idx in dists[: self.nearest_k]]
        self.last_sorted_experts = selected
        self.last_x = x_query
        self.last_expert_idx = selected[0]

        mus, vars_ = [], []
        for idx in selected:
            L = self.L_all[idx]
            alpha = self.alpha_all[idx]
            X_snapshot = self.X_list[idx][:, : self.localCount[idx]]
            n_valid = self.localCount[idx]

            mu = np.zeros(self.y_dim)
            var = np.zeros(self.y_dim)
            for p in range(self.y_dim):
                params = self.model_params[self.expert_creation_order[idx]]
                sigma_f = np.exp(params["log_sigma_f"][p])
                sigma_n = np.exp(params["log_sigma_n"][p])
                lengthscale = (
                    np.exp(params["log_lengthscale"][:, p]) if self.y_dim > 1 else np.exp(params["log_lengthscale"])
                )

                k_star = self.kernel_np(X_snapshot, x_query[:, None], lengthscale, sigma_f).flatten()
                k_xx = sigma_f**2

                mu[p] = np.dot(k_star, alpha[:n_valid, p])
                v = solve_triangular(L[:n_valid, :n_valid], k_star, lower=True)
                var[p] = max(k_xx - np.sum(v**2), 1e-12)

                if idx not in self.last_prediction_cache:
                    self.last_prediction_cache[idx] = {}
                self.last_prediction_cache[idx][p] = {"k_star": k_star.copy(), "v": v.copy(), "mu_part": mu[p]}

            mus.append(mu)
            vars_.append(var)

        mus = np.stack(mus)
        vars_ = np.stack(vars_)
        inv_vars = 1.0 / (vars_ + 1e-9)

        sigma0_sq = np.exp(self.model_params[self.expert_creation_order[selected[0]]]["log_sigma_f"][0]) ** 2
        mu_weighted = np.sum(inv_vars * mus, axis=0)
        denom = np.sum(inv_vars, axis=0)
        denom_corr = denom - (len(mus) - 1) / sigma0_sq
        mu_bcm = mu_weighted / (denom_corr + 1e-9)
        var_bcm = 1.0 / (denom_corr + 1e-9)
        return mu_bcm, var_bcm

    # ---------- 超参数优化（全局） ----------

    def _neg_log_marginal_likelihood(self, theta, X, y):
        """
        theta = [log_l1, ..., log_lD, log_sf, log_sn]
        单 GP 的 NLL，用于估计全局超参
        """
        D = self.x_dim
        log_l = theta[:D]
        log_sf = theta[D]
        log_sn = theta[D + 1]

        ls = np.exp(log_l)
        sf = np.exp(log_sf)
        sn = np.exp(log_sn)

        Xc = X.T  # (D, N)
        K = self.kernel_np(Xc, Xc, ls, sf)
        np.fill_diagonal(K, K.diagonal() + sn**2)

        try:
            L = np.linalg.cholesky(K + 1e-6 * np.eye(K.shape[0]))
        except np.linalg.LinAlgError:
            return 1e20  # 不可逆时返回大损失

        # NLL = 0.5*y^T K^-1 y + sum(log(diag(L))) + 0.5*N*log(2*pi)
        alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True), lower=False)
        nll = 0.5 * float(y.T @ alpha) + np.sum(np.log(np.diag(L))) + 0.5 * y.size * np.log(2 * np.pi)
        return nll

    def _fit_hparams_global(
        self,
        X,
        y,
        max_points=2000,
        n_restarts=2,
        lengthscale_bounds=(1e-3, 1e3),
        outputscale_bounds=(1e-6, 1e3),
        noise_bounds=(1e-6, 1.0),
        verbose=True,
    ):
        """用子集 + L-BFGS-B 优化全局超参（log 参数空间）。"""
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        N = X.shape[0]
        if N > max_points:
            idx = np.random.choice(N, max_points, replace=False)
            Xs, ys = X[idx], y[idx]
        else:
            Xs, ys = X, y

        # 启发式初始值
        self._set_or_guess_hparams(Xs, ys)
        outputscale0, noise0, ls0 = self.pretrained_params

        best_nll = np.inf
        best_theta = None

        def pack(ls, sf, sn):
            return np.concatenate([np.log(ls), [np.log(sf)], [np.log(sn)]])

        def bounds_for_theta():
            D = self.x_dim
            lb = [np.log(lengthscale_bounds[0])] * D + [np.log(outputscale_bounds[0])] + [np.log(noise_bounds[0])]
            ub = [np.log(lengthscale_bounds[1])] * D + [np.log(outputscale_bounds[1])] + [np.log(noise_bounds[1])]
            return list(zip(lb, ub))

        bounds = bounds_for_theta()

        initials = [pack(ls0, outputscale0[0], noise0[0])]
        # 简单随机扰动的多重启动
        for _ in range(n_restarts):
            jitter = np.random.uniform(-0.5, 0.5, size=self.x_dim + 2)
            initials.append(initials[0] + jitter)

        for init in initials:
            res = minimize(
                fun=self._neg_log_marginal_likelihood,
                x0=init,
                args=(Xs, ys),
                method="L-BFGS-B",
                bounds=bounds,
            )
            if verbose:
                print("Hparam opt try -> nll: {:.4f}".format(res.fun))
            if res.success and res.fun < best_nll:
                best_nll = res.fun
                best_theta = res.x

        if best_theta is None:
            if verbose:
                print("超参优化失败，保留启发式超参。")
            return  # 保持已有 self.pretrained_params

        # 解包并存回
        D = self.x_dim
        log_l = best_theta[:D]
        log_sf = best_theta[D]
        log_sn = best_theta[D + 1]
        ls = np.exp(log_l)
        sf = np.exp(log_sf)
        sn = np.exp(log_sn)
        self.pretrained_params = (np.array([sf], dtype=float), np.array([sn], dtype=float), ls)
        if verbose:
            print("优化完成 -> sf: {:.4g}, sn: {:.4g}, ls[:3]: {}".format(sf, sn, np.round(ls[:min(3, D)], 4)))

    def _reinit_all_expert_params(self):
        """在超参更新后，重置所有已存在专家的模型参数。"""
        for mid in self.expert_creation_order:
            self.init_model_params(mid, pretrained_params=self.pretrained_params)

    # ---------- 阶段一：离线预训练（带可选超参训练） ----------

    def offline_pretrain(
        self,
        X_train,
        Y_train,
        max_samples=None,
        show_progress=True,
        optimize_hparams=True,
        hparam_max_points=2000,
        hparam_restarts=2,
        verbose_hparam=True,
    ):
        """
        用离线数据集把模型（专家、分解等）建好。
        - 若 optimize_hparams=True，则先用子集最大化边缘似然训练超参
        - 然后按当前策略路由并填充专家，做增量分解

        返回：None
        """
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(-1, 1)
        elif Y_train.shape[1] > 1:
            Y_train = Y_train[:, :1]

        # 1) 先估计或训练超参
        if optimize_hparams:
            self._fit_hparams_global(
                X_train,
                Y_train,
                max_points=hparam_max_points,
                n_restarts=hparam_restarts,
                verbose=verbose_hparam,
            )
        else:
            self._set_or_guess_hparams(X_train, Y_train)

        # 如果已存在专家，用新的超参重置其参数表
        if len(self.expert_creation_order) == 0:
            self._create_new_expert()
        self._reinit_all_expert_params()

        # 2) 构建专家与分解（流式插入）
        N = X_train.shape[0]
        if max_samples is None:
            max_samples = N
        steps = min(max_samples, N)

        iterator = range(steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Offline pretraining (build experts)")

        for i in iterator:
            x = X_train[i].reshape(-1)
            y = Y_train[i].reshape(-1)
            # 先预测，得到候选专家顺序与缓存
            _ , _ = self.predict(x)
            # 不走 add_point 的“重排序”分支，直接用缓存路由并增量更新
            self.add_point(x, y)

    # ---------- 阶段二：在线单点预测 + 增量学习 ----------

    def online_step(self, x_new, y_new):
        """
        对单个新点执行：
          1) 先预测（不包含该点）
          2) 再把该点加入模型并做增量更新

        返回：
          y_pred, var  —— 预测均值、方差（在加入前的预测）
        """
        x_new = np.asarray(x_new).reshape(-1)
        y_new = np.asarray(y_new).reshape(-1)

        if len(self.expert_creation_order) == 0:
            # 未预训练时做最小初始化
            self._set_or_guess_hparams(x_new[None, :], y_new[None, None])
            self._create_new_expert()
            self._reinit_all_expert_params()

        y_pred, var = self.predict(x_new)
        self.add_point(x_new, y_new)
        return y_pred, var
