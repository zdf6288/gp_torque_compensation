# GOAL1 Spatial Multisine Complete Matrix Offline Analysis

本文档记录 `goal1_spatial_multisine_full_frozen_and_online_matrix_20260603.tar.gz` 的离线分析入口、输出文件和解释边界。实际数值结论以脚本生成的 CSV 和 `analysis_summary.md` 为准。

## Scope

- 只做离线 CSV 分析、summary table、plot 和 Markdown summary。
- 不运行 ROS2 launch。
- 不连接机器人。
- 不修改 controller、launch、torque command、franka_hardware、controllers.yaml 或 robot bringup 文件。
- 不删除、不修改 raw data 或 archive。

## Script

脚本路径：

- `scripts/analyze_goal1_spatial_multisine_complete_matrix.py`

推荐输入 archive：

- `/mnt/c/Users/dummd/Downloads/goal1_spatial_multisine_full_frozen_and_online_matrix_20260603.tar.gz`

默认输出目录：

- `outputs/goal1_spatial_multisine_complete_matrix_20260603/`

示例命令：

- `python3 scripts/analyze_goal1_spatial_multisine_complete_matrix.py --archive /mnt/c/Users/dummd/Downloads/goal1_spatial_multisine_full_frozen_and_online_matrix_20260603.tar.gz`

## Generated Evidence Tables

脚本至少生成：

- `run_manifest.csv`
- `sanity_summary.csv`
- `tracking_summary.csv`
- `gp_compensation_summary.csv`
- `clip_summary.csv`
- `analysis_summary.md`

如果 `matplotlib` 可用且没有传入 `--no-plots`，还会生成：

- `frozen_tracking_rmse_comparison.png`
- `online_tracking_rmse_comparison.png`
- `max_abs_gp_applied_by_run.png`
- `clip_active_count_by_run.png`
- `clip_active_count_by_joint.png`
- `nogp_begin_vs_end_drift.png`

## Latest Generated Snapshot

基于 `/mnt/c/Users/dummd/Downloads/goal1_spatial_multisine_full_frozen_and_online_matrix_20260603.tar.gz` 的一次离线分析结果：

- 16 个 expected runs 全部读取成功。
- `run_manifest.csv`、`sanity_summary.csv`、`tracking_summary.csv`、`gp_compensation_summary.csv`、`clip_summary.csv` 中所有 run 的 `status=ok`。
- Cartesian tracking columns 自动识别为 `x_desired/y_desired/z_desired` 和 `x_actual/y_actual/z_actual`。
- no-GP begin 的 `rmse_3d_mm=4.12932038771`，no-GP repeat end 的 `rmse_3d_mm=4.16348870163`。
- Frozen scale 0.1 main runs 的 `rmse_3d_mm`：local `4.08269989993`，cloud `4.10723504825`，combined `4.11299333511`。
- Frozen scale 1.0 main runs 的 `rmse_3d_mm`：local `3.88166745441`，cloud `3.86317647475`，combined `3.88730898431`。
- Online diagnostic scale 1.0 runs 的 `rmse_3d_mm`：local `3.6459047089`，cloud `3.66419946816`，combined `3.76082489141`。
- Frozen scale 1.0 runs 的 `total_clip_active_count=0`。
- Online local scale 1.0 的 `max_abs_gp_applied=0.5`，`total_clip_active_count=3`，clip active 在 j3。
- Online cloud scale 1.0 的 `max_abs_gp_applied=0.481494182318`，`total_clip_active_count=0`。
- Online combined scale 1.0 的 `max_abs_gp_applied=0.5`，`total_clip_active_count=4`，clip active 在 j3 和 j4。

## Interpretation Boundary

Main frozen GP matrix 是主要 controlled comparison：

- `gp_online_update_enabled=false`
- GP model 在单次 run 内固定
- 用同一条 `goal1_spatial_multisine` 轨迹比较 no-GP、local、cloud、combined、scale 0.1 和 scale 1.0

Legacy online-update diagnostic matrix 是补充诊断：

- `gp_online_update_enabled=true`
- 用来解释 legacy behavior
- 因为模型状态会在 run 中变化，不应替代 frozen GP 作为主结论

## Clip Interpretation

- `clip=0.5` 是安全边界。
- 如果 frozen scale 1.0 runs 没有 clip active，说明本矩阵没有必要把 clip 提到更高。
- 如果 online scale 1.0 runs 出现 clip active，说明 online update 可能把 GP output 推近安全边界，支持继续保留 `clip=0.5`。

## Caveats

- 部分真机 run 可能保存了有效 CSV/plots，但结束时出现 `User Stop`、`communication_constraints_violation` 或 `rclpy.shutdown()` traceback。
- 这些应作为 engineering caveats，而不是自动判定数据无效。
- 推荐表述：usable real-robot data、complete evidence matrix for offline comparison、post-run shutdown caveats。
- 避免表述：fully stable、robust repeated validation、clean shutdown、communication constraints are solved。

## Conclusion Style

可以使用：

- The real-robot dataset provides a complete evidence matrix for offline comparison.
- The frozen GP matrix is the primary controlled comparison.
- The online-update matrix is treated as a legacy diagnostic.
- The clip summary confirms whether the safety bound was active and where.
- Tracking improvement should be concluded only from the computed tracking summary, not assumed from GP-on configuration.
