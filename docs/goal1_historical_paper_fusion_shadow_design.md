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

## Current Historical Fallback

Phase 1 尚未接入可用的 past prediction pool。当前 `gp_historical_source_mode` 只接受 `"none"`：

- `gp_shadow_historical_available = 0`
- `gp_shadow_hist_raw_* = 0`
- `gp_shadow_var_hist_* = gp_shadow_hist_fallback_variance`
- `gp_shadow_weight_hist_* = 0`

因此在 historical unavailable 时，paper shadow candidate 退化为 local/cloud inverse-variance fusion。

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
