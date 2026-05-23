# Stage 4 Formal Result Summary

## Scope

本文档总结 Stage 4 formal frozen GP comparison 的离线结果，用于项目文档、Obsidian 笔记和后续论文草稿整理。这里记录的是一次 formal fullrun 结果集，不是 repeated validation，也不作为最终论文结论。

## Experimental Design

本次 formal comparison 比较三种模式：

- `strict_no_gp`：strict no-GP baseline。
- `gp_planar_scale03`：frozen GP trained on planar trajectory。
- `gp_spatial_scale03`：frozen GP trained on spatial / tilted trajectory。

三组数据使用相同 formal test trajectory：

| parameter | value |
| --- | --- |
| `trajectory_mode` | `z_modulated_circle` |
| `z_amplitude` | `0.03` |
| `z_frequency_multiplier` | `0.5` |

离线指标使用各 CSV 的完整长度计算；timeseries overlay plot 裁剪到最短 run 长度。

## Input Files

| mode | CSV | rows | nan | inf |
| --- | --- | ---: | ---: | ---: |
| `strict_no_gp` | `data/stage4/test/strict_no_gp/usable_runs/strict_no_gp_zmod_3000pts_20260523_154902.csv` | 3000 | 0 | 0 |
| `gp_planar_scale03` | `data/stage4/test/gp_planar_scale03/usable_runs/gp_planar_scale03_zmod_2999pts_20260523_161222.csv` | 2999 | 0 | 0 |
| `gp_spatial_scale03` | `data/stage4/test/gp_spatial_scale03/usable_runs/gp_spatial_scale03_zmod_3000pts_20260523_163907.csv` | 3000 | 0 | 0 |

## GP Configuration

`gp_planar_scale03` 和 `gp_spatial_scale03` 使用 frozen local GP compensation：

| ROS2 parameter | GP modes value |
| --- | --- |
| `gp_prediction_enabled` | `true` |
| `gp_online_update_enabled` | `false` |
| `gp_compensation_enabled` | `true` |
| `gp_compensation_source` | `local` |
| `gp_compensation_scale` | `0.3` |
| `gp_compensation_clip_nm` | `0.5` |

`strict_no_gp` 使用 GP 完全关闭设置：

| ROS2 parameter | strict baseline value |
| --- | --- |
| `gp_prediction_enabled` | `false` |
| `gp_online_update_enabled` | `false` |
| `gp_compensation_enabled` | `false` |

## Key Results

核心结果来自 `outputs/stage4_formal_analysis/stage4_formal_summary.csv` 和 `outputs/stage4_formal_analysis/stage4_formal_summary.md`。

| mode | rows | 3D RMS tracking error mm | improvement vs strict no-GP % | tau residual all RMS | `y_hat_local` all RMS | compensation proxy all RMS | clip ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `strict_no_gp` | 3000 | 4.092 | 0.00 | 0.380749 | 0.000000 | 0.000000 | 0.000000 |
| `gp_planar_scale03` | 2999 | 4.011 | 1.99 | 0.376481 | 0.046064 | 0.013819 | 0.000000 |
| `gp_spatial_scale03` | 3000 | 4.147 | -1.33 | 0.386266 | 0.042182 | 0.012655 | 0.000000 |

相关 plots：

- `outputs/stage4_formal_analysis/tracking_3d_rms_bar.png`
- `outputs/stage4_formal_analysis/tracking_axis_error_rms_bar.png`
- `outputs/stage4_formal_analysis/tracking_3d_error_timeseries.png`
- `outputs/stage4_formal_analysis/tau_residual_rms_per_joint.png`
- `outputs/stage4_formal_analysis/y_hat_local_rms_per_joint.png`
- `outputs/stage4_formal_analysis/compensation_proxy_rms_per_joint.png`
- `outputs/stage4_formal_analysis/compensation_clip_ratio_per_joint.png`

## Interpretation

在这一次 Stage 4 formal run 中，`gp_planar_scale03` 的 3D RMS tracking error 从 `strict_no_gp` 的 `4.092 mm` 降到 `4.011 mm`，约 `1.99%` improvement。

`gp_spatial_scale03` 的 3D RMS tracking error 为 `4.147 mm`，相对 `strict_no_gp` 的 improvement 为 `-1.33%`，也就是本次 run 中 tracking 略差于 strict no-GP baseline。

相对 `gp_planar_scale03`，`gp_spatial_scale03` 的 3D RMS tracking error 约高 `3.39%`。因此，本次结果不支持“spatial-trained GP 在当前设置下优于 planar-trained GP”的结论。

不过，这不等价于证明 spatial training 无效。当前结果只来自一组 fullrun，并且使用 fallback standardized hyperparameters、single `gp_compensation_scale=0.3`、single `gp_compensation_clip_nm=0.5` 设置。spatial-trained GP 的表现仍可能受 model quality、training data distribution、feature/target definition、trajectory mismatch 等因素影响。

## Data Quality

三组 CSV 均满足基本离线分析质量要求：`nan=0`，`inf=0`。

| mode | rows | columns | nan | inf | time span s | estimated Hz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `strict_no_gp` | 3000 | 147 | 0 | 0 | 59.980 | 50.00 |
| `gp_planar_scale03` | 2999 | 147 | 0 | 0 | 59.960 | 50.00 |
| `gp_spatial_scale03` | 3000 | 147 | 0 | 0 | 59.980 | 50.01 |

Row count 不完全一致：`strict_no_gp=3000`，`gp_planar_scale03=2999`，`gp_spatial_scale03=3000`。分析脚本对数值指标使用各自完整长度；timeseries overlay 裁剪到最短长度。

三组 compensation clip ratio 均为 `0.000000`。这说明在当前 `gp_compensation_scale=0.3` 和 `gp_compensation_clip_nm=0.5` 下，离线 compensation proxy 没有 hit clip。

## Caveats

- 这是一次 Stage 4 formal run，不是 robust repeated validation。
- 本结果不应表述为系统已经 `fully stable`。
- 真机日志中存在 post-save `communication_constraints_violation` / shutdown caveat；但数据保存和 plotting 已完成，不阻止本次离线分析。
- 当前 frozen GP models 使用 fallback standardized hyperparameters；如果后续具备 `torch/gpytorch` 环境，可以重新训练 hparams 后复测。
- Spatial-trained GP 的表现可能受 model quality、training data distribution、single scale、clip、feature/target definition、trajectory mismatch 影响。
- Offline compensation proxy 根据 `y_hat_local_*` 和 scale/clip 重建，不替代 controller 内部完整安全判断。

## Suggested Next Steps

- 不建议继续无计划地反复跑真机。
- 先做离线诊断：per-joint residual、`y_hat_local` magnitude、compensation proxy、failure/partial runs 对比。
- 可考虑后续单独设计 `GP_spatial scale01` 或重新训练 hparams 的 diagnostic。
- 若要论文级结论，需要 repeated runs 或更系统的 ablation。

## Related Files

- `scripts/analyze_stage4_formal_results.py`
- `outputs/stage4_formal_analysis/stage4_formal_summary.md`
- `outputs/stage4_formal_analysis/stage4_tracking_metrics.csv`
- `outputs/stage4_formal_analysis/stage4_formal_summary.csv`
- `docs/stage4_roadmap.md`
- `docs/stage4_collection_plan.md`
- `docs/stage4_dataset_preparation.md`
