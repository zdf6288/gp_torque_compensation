# GOAL1 Historical Paper Fusion Shadow Design

本文档记录 Phase 1 的 shadow/logging-only 实现边界。该 patch 只用于离线分析 CSV，不改变真实 torque command。

## Formula

paper tri-temporal fusion 在本实验中映射为：

`f_hat_fuse = w_local * f_hat_local + w_cloud * f_hat_cloud + w_hist * f_hat_hist`

权重按 inverse variance 归一化：

`w_source = sigma_source^-2 / (sigma_local^-2 + sigma_cloud^-2 + sigma_hist^-2)`

代码中使用 predictive variance：

- `prec_local = 1.0 / max(var_local, gp_shadow_variance_eps)`
- `prec_cloud = 1.0 / max(var_cloud, gp_shadow_variance_eps)`
- `prec_hist = 1.0 / max(var_hist, gp_shadow_variance_eps)`

## Source Mapping

- `local`: current local GP prediction `self.y_hat_local`
- `cloud`: current cloud-like GP prediction `self.y_hat_cloud`
- `historical`: stored past inference / past prediction retrieval

`online_update` 不是 historical source，不能用当前 residual 或在线更新过程替代 historical prediction。

## Runtime Historical Shadow Source

historical source 现在支持两个 mode：

- `"none"`：默认值，保持 historical unavailable。
- `"local_prediction_pool"`：维护 bounded runtime pool，保存 past local GP prediction。

pool 保存：

- 14D feature `x=[q, dq]`，与 `_gp_predict_and_update()` 当前启用的 GP input helper 一致。
- past local prediction mean `self.y_hat_local`。
- past local prediction variance `self.var_local`。
- local prediction sequence，用于避免同一 prediction 重复 append。

pool 不保存 current residual，不调用 `model.add_point`，也不改变 GP model。`online_update` 仍然只是 runtime adaptation，不是 historical source。

每次 shadow update 按以下顺序执行：

1. 用当前 14D feature query 已存在的 past prediction pool。
2. 使用 nearest/top-k candidate，以 inverse distance 和 per-joint variance precision 组合权重。
3. 计算 shadow-only historical mean / variance 和 paper fusion。
4. 最后 append 当前 local prediction，避免当前样本检索到自身。

在 pool size 小于 `gp_historical_shadow_min_points`、nearest distance 超过
`gp_historical_shadow_max_distance`，或输入/输出无效时，historical 保持 unavailable：

- `gp_shadow_historical_available = 0`
- `gp_shadow_hist_raw_* = 0`
- `gp_shadow_var_hist_* = gp_shadow_hist_fallback_variance`
- `gp_shadow_weight_hist_* = 0`

因此 historical unavailable 时，paper shadow candidate 仍退化为 local/cloud inverse-variance fusion。

## Historical Parameters

默认参数保持旧行为：

- `gp_historical_shadow_enabled = false`
- `gp_historical_source_mode = "none"`
- `gp_historical_shadow_max_points = 2000`
- `gp_historical_shadow_min_points = 10`
- `gp_historical_shadow_k = 5`
- `gp_historical_shadow_max_distance = 1e6`
- `gp_historical_shadow_variance_floor = 1e-8`
- `gp_historical_shadow_distance_eps = 1e-9`

只有显式启用 `gp_shadow_paper_fusion_logging_enabled=true`、
`gp_historical_shadow_enabled=true` 且选择
`gp_historical_source_mode="local_prediction_pool"` 时，runtime pool 才会收集 past local prediction。

新增 optional CSV diagnostics：

- `gp_shadow_hist_pool_size`
- `gp_shadow_hist_k_used`
- `gp_shadow_hist_nearest_distance`
- `gp_shadow_hist_mean_distance_topk`

## Clip Proxy

CSV 中的 `gp_shadow_paper_scaled_*`、`gp_shadow_paper_clip_proxy_applied_*` 和 `gp_shadow_paper_clip_proxy_active_*` 只是 hypothetical proxy：

- 使用当前 `gp_compensation_scale`
- 使用当前 `gp_compensation_clip_nm`
- 不写入 `gp_applied_*`
- 不进入 `tau_final_*`
- 不发布到 `/effort_command`

## Safety Boundary

默认参数保持关闭：

- `gp_shadow_paper_fusion_logging_enabled = false`
- `gp_historical_shadow_enabled = false`
- `gp_historical_source_mode = "none"`

`gp_compensation_source:=combined` 仍保持 Phase 0 的 local/cloud variance fusion torque behavior。paper/historical fusion 只记录 shadow columns，不进入真实 torque path。

在 offline/fake/sanity review 之前，不应将 paper/historical fusion 接入真实机器人 torque。
当前 runtime historical shadow pool 尚未通过新生成 CSV 验证 historical availability，也不代表 real-robot tracking 会改善。

## Explicit Safety Wording

Current implementation is shadow/logging-only. `historical` and `paper_tri_temporal` fusion must not enter `tau_final`, and `gp_shadow_paper_clip_proxy_applied_*` is only a hypothetical clip proxy.

For intended experiments, the safety clip bound remains `gp_compensation_clip_nm=0.5`. No no-clip mode should be introduced, and no clip increase should be used for historical/paper fusion validation.

No real-robot run should be performed before offline/fake/sanity validation. `online_update` is not treated as historical.

## Fake Validation Status

The no-robot fake launch now exposes the GP prediction and historical shadow
parameters with disabled defaults. A runtime CSV validation was not run on
2026-06-04 because the current workspace cannot satisfy the offline prerequisites:

- `ros2 pkg prefix py_controllers` reports `Package not found` after sourcing the
  current workspace install.
- `sklearn` is missing in the clean-shell Python environment.
- `new_structure/gp/gp_models` is not present as an extracted GOAL1 model directory.

No CSV has therefore confirmed `gp_shadow_historical_available=1` or nonzero
historical weights yet. A model directory from another experiment was not used.
