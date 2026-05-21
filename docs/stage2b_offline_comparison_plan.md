# Stage 2B Offline Comparison Plan

Stage 2B 的目标是在 WSL / PC 上对 Stage 2A 已保存 CSV 做离线横向比较，回答一个务实问题：在已有成功数据采集基础上，`pure_no_gp`、`compute_only`、`gp_on_conservative` 等模式的 tracking、residual、GP prediction、torque 分布是否出现可解释差异。

这一步不修改任何 real-robot controller、launch file、controller config 或 torque computation logic。它只读取 `data/stage2a/csv/` 下的 CSV，并把结果写到 `outputs/stage2b_comparison/`。

## Compared CSVs

默认命令会加载 `data/stage2a/csv/*.csv`，因此当前会比较：

- `pure_no_gp_20260520_212521.csv`
- `stage1_baseline_20260520.csv`
- `stage2a_gpon_conservative_20260520_211748.csv`

如果之后加入 compute-only CSV，脚本会自动包含它；也可以用 `--include` / `--exclude` 按文件名筛选。

## Generated Metrics

脚本 `scripts/compare_stage2a_modes.py` 会生成：

- `stage2b_timing_summary.csv`: CSV health、row/column count、time column、duration、dt、estimated Hz、NaN / inf count。
- `stage2b_cartesian_tracking_summary.csv`: Cartesian actual vs desired tracking error，包括 `rmse_x`、`rmse_y`、`rmse_z`、`rmse_norm`、`p95_norm`。
- `stage2b_tau_residual_metrics.csv`: per-joint `tau_residual` statistics，包括 mean、std、rms、abs_mean、p95_abs、min、max。
- `stage2b_tau_residual_comparison.csv`: 如果同时检测到 `pure_no_gp` 和 `gp_on_conservative`，计算 `rms_change_percent`。
- `stage2b_gp_prediction_stats.csv`: per-joint `y_hat` / GP prediction statistics，包括 mean、std、range、rms、p95_abs、variation_ratio。
- `stage2b_clip_proxy.csv`: 使用 `scaled_y_hat = gp_scale * y_hat` 和 `gp_clip_nm` 估算 conservative clip saturation proxy。
- `stage2b_tau_metrics.csv`: per-joint commanded torque-like statistics。

PNG plots 会保存到：

- `outputs/stage2b_comparison/plots/`

## How To Run

从 repo root 运行：

- `python3 scripts/compare_stage2a_modes.py --input-dir data/stage2a/csv --output-dir outputs/stage2b_comparison`

可选参数：

- `--include PATTERN`
- `--exclude PATTERN`
- `--gp-scale 0.1`
- `--gp-clip-nm 0.5`

## How To Interpret

先看 `stage2b_timing_summary.csv`，确认每个 CSV 的 rows、dt、estimated Hz、NaN / inf 是否健康。然后看 `stage2b_cartesian_tracking_summary.csv` 对比 tracking error，再看 `stage2b_tau_residual_metrics.csv` 和 `stage2b_tau_residual_comparison.csv` 判断 GP-on conservative 是否让 residual RMS 在某些 joints 上下降或上升。

`stage2b_gp_prediction_stats.csv` 和 `stage2b_clip_proxy.csv` 主要用于判断 `y_hat` 是否有明显变化、是否接近 conservative clip。clip proxy 只是离线近似，不等同于 controller 内部最终 torque application。

## Limitations

- Stage 2B 是 engineering progression 的离线比较，不是最终 robust statistical proof。
- 每个 mode 有一次成功完成 target trajectory 并保存有效 CSV，就足够支持下一步工程判断。
- `communication_constraints_violation`、non-clean shutdown、CPU / realtime 限制是已知 limitation。
- 论文或报告中可以讨论 successful runs 和 method effects，但不要声称 `fully stable`、`robust repeated validation` 或类似过强结论。
