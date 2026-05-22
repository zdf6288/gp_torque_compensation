# Stage 3A 结果总结：Frozen GP Spatial Trajectory 离线分析

## 1. 总体结论

Stage 3A 已经完成 `gp_torque_compensation` 项目中 frozen GP spatial trajectory 的第一次可用真机验证。

本阶段最重要的正式对比是：

- 基线模式：`comp_off fullrun`
- Frozen GP 补偿模式：`gp_on_scale03 fullrun`

主要结论是：

> 相比 `comp_off fullrun`，`gp_on_scale03 fullrun` 在 7 个关节上的 tau residual RMS 全部下降，同时 3D Cartesian tracking error 保持在相近水平，没有明显破坏轨迹跟踪质量。

因此，Stage 3A 可以视为一次成功的初步真机验证，结果足以支持进入 Stage 4 formal spatial / 7DoF experiment 的准备阶段。

需要注意的是，本阶段结论不应表述为 fully stable 或 robust repeated validation。`communication_constraints_violation`、User Stop、`rclpy` shutdown context errors 等问题应作为工程 caveat 记录，但不阻止当前离线分析和下一阶段推进。

---

## 2. 当前仓库与数据状态

当前分支：

- `frozen_gp_spatial_trajectory`

当前 PC WSL 路径：

- `~/projects/gp_torque_compensation`

当前最新相关 commit：

- `4affe5c Add Stage 3A offline analysis workflow`

Stage 3A 数据目录：

- `data/stage3a/csv/comp_off/`
- `data/stage3a/csv/gp_on_scale03/`
- `data/stage3a/csv/gp_on_scale05/`
- `data/stage3a/csv/gp_on_scale07/`
- `data/stage3a/plots/`
- `data/stage3a/logs/launch.log`

新增并已 push 的离线分析脚本和文档：

- `scripts/analyze_stage3a_csv.py`
- `scripts/compare_stage3a_modes.py`
- `docs/stage3a_offline_analysis.md`

---

## 3. Stage 3A 实验数据概览

当前已有 CSV 数据如下：

| Mode | Run 类型 | 数据点数 | 解释 |
|---|---:|---:|---|
| `comp_off` | fullrun | 3749 | 主基线数据 |
| `gp_on_scale03` | fullrun | 3749 | 主 frozen GP 对比数据 |
| `gp_on_scale05` | partial | 2024 | 趋势参考，不作为完整验证 |
| `gp_on_scale07` | nearfull | 3103 | 趋势参考，不作为完整验证 |

当前 Stage 3A 的正式结论只基于：

- `comp_off fullrun`
- `gp_on_scale03 fullrun`

`gp_on_scale05` 和 `gp_on_scale07` 仍然有参考价值，尤其适合观察 GP scale 增大后 `scaled_y_hat_abs` 的变化趋势和 clip proxy 情况，但它们不应被描述为完整 fullrun validation。

---

## 4. CSV 完整性检查结果

Stage 3A CSV 基础检查结果如下：

- columns 数量：`147`
- NaN 数量：`0`
- inf 数量：`0`
- estimated Hz：约 `50 Hz`
- `columns_match=True`
- `rows_match=False`

其中，`rows_match=False` 是预期结果，因为：

- `gp_on_scale05` 是 partial run
- `gp_on_scale07` 是 nearfull run

这不是 CSV schema 错误，也不是数据列不一致问题。

`columns_match=True` 表明不同 mode 的 CSV 列结构一致，可以用于后续统一分析和横向比较。

---

## 5. 主对比：`gp_on_scale03 fullrun` vs `comp_off fullrun`

### 5.1 Tau Residual RMS 结果

相比 `comp_off fullrun`，`gp_on_scale03 fullrun` 在 7 个关节上的 tau residual RMS 全部下降：

| Joint | Tau Residual RMS 变化 |
|---:|---:|
| J1 | `-26.89%` |
| J2 | `-17.17%` |
| J3 | `-13.64%` |
| J4 | `-17.58%` |
| J5 | `-19.34%` |
| J6 | `-17.35%` |
| J7 | `-74.86%` |

解释：

- Frozen GP compensation 在 Stage 3A 的 spatial trajectory 设置下产生了明确的 residual torque 降低效果。
- 所有关节均下降，说明这个结果不是单一关节上的偶然变化。
- J7 的下降幅度最大，但在写论文或报告时不建议只强调 J7，而应强调“7 个关节全部下降”这一整体结果。
- 当前结果可以支持“GP compensation 有效降低 tau residual”的初步结论。
- 当前结果不能支持“系统已经完成鲁棒重复验证”的结论。

推荐表述：

> 在 Stage 3A 中，`gp_on_scale03` 相比 `comp_off` 在所有 7 个关节上降低了 tau residual RMS，说明 frozen GP compensation 在该 spatial trajectory 条件下产生了有效补偿作用。

不建议表述为：

> GP compensation 已经被完全验证。

> 系统已经稳定可靠。

> GP compensation 在所有情况下都能降低 residual torque。

---

## 6. Cartesian Tracking 结果

3D Cartesian error norm RMSE 结果如下：

| Mode | 3D Error Norm RMSE |
|---|---:|
| `comp_off fullrun` | 约 `0.003956 m` |
| `gp_on_scale03 fullrun` | 约 `0.003716 m` |

解释：

- `gp_on_scale03` 没有明显破坏 Cartesian tracking。
- `gp_on_scale03` 的 RMSE 略低于 `comp_off`，但差距较小。
- 因此建议将该结果描述为 tracking comparable，而不是强行声称 tracking 明显改善。
- 当前最稳妥的结论是：`gp_on_scale03` 在降低 tau residual RMS 的同时保持了相近的 tracking 质量。

推荐表述：

> `gp_on_scale03` 在降低 tau residual RMS 的同时保持了与 `comp_off` 相近的 3D Cartesian tracking error，未观察到明显 tracking degradation。

不建议表述为：

> GP compensation 显著提升了 tracking accuracy。

> GP compensation 一定改善了末端轨迹误差。

---

## 7. Clip Proxy 与 GP 输出幅值

Clip proxy 检查结果：

- 所有 mode 的 `clip_proxy_ratio = 0`
- 当前 clip threshold 为 `1.0 Nm`

不同 GP scale 下的最大 `scaled_y_hat_abs` 为：

| Mode | Max `scaled_y_hat_abs` |
|---|---:|
| `gp_on_scale03` | 约 `0.251674 Nm` |
| `gp_on_scale05` | 约 `0.410120 Nm` |
| `gp_on_scale07` | 约 `0.556491 Nm` |

解释：

- 当前所有测试 mode 都没有触及 `1.0 Nm` clip threshold。
- 随着 GP scale 从 `0.3` 增加到 `0.5` 和 `0.7`，最大 GP 输出幅值按预期增大。
- 当前结果说明测试过的 scale 范围内没有发生 clip proxy activation。
- 这不意味着更大 GP scale 一定安全。
- 这也不意味着可以移除 clip 或运行 unlimited GP-on。

安全边界：

- 不建议做 no-clip GP-on。
- 不建议做 unlimited GP compensation。
- 不应把 `clip_proxy_ratio = 0` 解读为可以无限增大 GP scale。
- Stage 4 应继续保留 clip 或等效 torque safety bound。

---

## 8. 工程 Caveats

当前 Stage 3A 仍需记录以下工程 caveats：

- `communication_constraints_violation`
- User Stop
- `rclpy` shutdown context errors
- 部分运行不是 clean shutdown
- `gp_on_scale05` 和 `gp_on_scale07` 不是完整 fullrun

这些问题应被记录，但不应被过度放大为“实验失败”。

本项目当前采用的工程标准是：

> 成功一次跑出可用数据即可推进；不要求每次都 clean shutdown。

因此，Stage 3A 的判断应区分：

| 判断对象 | 当前状态 |
|---|---|
| 可用数据采集 | 已完成 |
| `comp_off` fullrun | 已完成 |
| `gp_on_scale03` fullrun | 已完成 |
| 完全稳定 repeated validation | 不声称 |
| 每次 clean shutdown | 不要求 |
| Stage 4 准备条件 | 已具备 |

---

## 9. Stage 3A 结论

Stage 3A 成功获得了用于 offline analysis 的真机 spatial trajectory 数据。

本阶段最重要的结论是：

> `gp_on_scale03 fullrun` 相比 `comp_off fullrun` 在 7 个关节上的 tau residual RMS 全部下降，同时 3D Cartesian tracking error 保持相近水平，且所有 mode 的 clip proxy ratio 均为 0。

这说明 frozen GP compensation 在当前 Stage 3A 设置下已经表现出有效补偿趋势，可以作为 Stage 4 formal spatial experiment 的基础。

但该结论应保持谨慎：

- 不声称 fully stable。
- 不声称 robust repeated validation。
- 不把 partial / nearfull 的 scale05 / scale07 当作正式完整验证。
- 不声称更高 GP scale 必然更好。
- 不建议取消 clip 或扩大到 unlimited GP-on。

---

## 10. Stage 4 准备方向

Stage 4 应聚焦于 formal spatial / 7DoF experiment。

建议 Stage 4 的核心目标是：

> 在最终 formal spatial / 7DoF trajectory 设置下，复现并确认 `comp_off` 与 `gp_on_scale03` 的正式 fullrun 对比。

Stage 4 不应一开始就追求更高 GP scale，而应先建立稳定、可解释、可复现的正式主对比。

推荐的 Stage 4 基础实验矩阵：

| 优先级 | Mode | 目的 | 是否必需 |
|---:|---|---|---|
| 1 | `comp_off` | 正式 baseline fullrun | 必需 |
| 2 | `gp_on_scale03` | 正式 frozen GP comparison | 必需 |
| 3 | `gp_on_scale05` | 更强补偿趋势参考 | 可选 |
| 4 | `gp_on_scale07` | 高 scale 谨慎趋势探索 | 可选，谨慎 |

推荐运行顺序：

1. 先运行 `comp_off` fullrun。
2. 再运行 `gp_on_scale03` fullrun。
3. 立刻做 CSV 完整性检查。
4. 对比 tracking、tau residual RMS 和 clip proxy。
5. 如果 `scale03` 数据可用且没有明显异常，再考虑 `scale05`。
6. 只有在前面结果安全且必要时，再考虑 `scale07`。
7. 不进行 no-clip 或 unlimited GP-on 实验。

---

## 11. Stage 4 中应保持不变的条件

为了让 Stage 4 的结论可解释，建议一次 formal comparison 中保持以下条件不变：

- 相同 trajectory geometry
- 相同 controller configuration
- 相同 logging columns
- 相同 CSV analysis pipeline 或兼容扩展
- 相同 GP clip threshold
- 相同 sampling-rate expectation，约 `50 Hz`
- 相同数据归档结构
- 相同成功标准和 caveat 记录方式

Stage 4 中不建议同时改变多个变量。

不建议同时做：

- 新 trajectory
- 新 controller 参数
- 新 GP scale
- 新 logging format
- 新 shutdown 逻辑
- 新 safety boundary

否则后续无法判断结果变化到底来自哪个因素。

---

## 12. Stage 4 最低成功标准

Stage 4 的最低成功标准建议为：

- 至少一次可用的 `comp_off` fullrun
- 至少一次可用的 `gp_on_scale03` fullrun
- CSV columns 一致
- NaN 数量为 `0`
- inf 数量为 `0`
- estimated Hz 接近预期值
- clip proxy 没有异常
- `gp_on_scale03` 不明显破坏 tracking
- tau residual RMS 在多个关节上下降，理想情况下多数或全部关节下降

更强但非必需的标准：

- `comp_off` 重复 fullrun
- `gp_on_scale03` 重复 fullrun
- `gp_on_scale05` 可选趋势运行
- `gp_on_scale07` 可选谨慎趋势运行
- shutdown 行为更干净
- 通信异常减少

但这些不是推进到离线分析的硬性前提。

当前工程标准仍然是：

> 单次成功跑出可用数据即可推进，但必须如实记录 caveats。

---

## 13. 推荐论文 / 报告表述

### 13.1 安全表述

可以使用以下表述：

> Stage 3A produced a usable real-robot spatial trajectory dataset for comparing compensation-off behavior with frozen GP compensation.

中文对应：

> Stage 3A 获得了一组可用于离线分析的真机 spatial trajectory 数据，可用于比较 compensation-off 与 frozen GP compensation 的效果。

可以使用以下表述：

> Compared with `comp_off`, `gp_on_scale03` reduced tau residual RMS across all seven joints while maintaining comparable Cartesian tracking performance.

中文对应：

> 相比 `comp_off`，`gp_on_scale03` 在 7 个关节上均降低了 tau residual RMS，同时保持了相近的 Cartesian tracking performance。

可以使用以下表述：

> No clip-proxy activation was observed under the tested GP scales.

中文对应：

> 在当前测试过的 GP scale 下，没有观察到 clip-proxy activation。

### 13.2 不建议使用的表述

不建议写：

> The system is fully stable.

不建议写：

> The controller is robustly validated.

不建议写：

> GP compensation always improves tracking.

不建议写：

> Higher GP scale is better.

不建议写：

> The clip is unnecessary.

不建议写：

> The real-robot system is safe under unlimited GP compensation.

---

## 14. 建议保存命令

在 PC WSL 中执行：

- `cd ~/projects/gp_torque_compensation`
- `mkdir -p docs`
- `nano docs/stage3a_result_summary.md`

将本文件内容粘贴进去后保存。

也可以使用 VS Code：

- `cd ~/projects/gp_torque_compensation`
- `code docs/stage3a_result_summary.md`

保存后检查 Git 状态：

- `git status --short`

建议 commit：

- `git add docs/stage3a_result_summary.md`
- `git commit -m "Add Stage 3A result summary"`
- `git push`

---

## 15. 离线复查命令

如需在 commit 前重新检查 Stage 3A 离线结果，可在 PC WSL 中执行：

- `cd ~/projects/gp_torque_compensation`
- `python3 scripts/analyze_stage3a_csv.py`
- `python3 scripts/compare_stage3a_modes.py`

如果使用虚拟环境，应先激活对应 `.venv`：

- `cd ~/projects/gp_torque_compensation`
- `source .venv/bin/activate`

然后再执行分析脚本。

---

## 16. 后续建议

完成 `docs/stage3a_result_summary.md` 后，下一步建议单独创建 Stage 4 planning 文档，例如：

- `docs/stage4_formal_experiment_plan.md`

Stage 4 planning 文档应覆盖：

- 实验目标
- 实验矩阵
- 真机运行顺序
- 每个 mode 的成功标准
- 数据目录结构
- CSV 命名规则
- offline analysis 流程
- caveat 记录方式
- 禁止项，例如 no-clip GP-on 和 unlimited GP compensation

建议 Stage 4 先只把 `comp_off` 与 `gp_on_scale03` 作为正式主线。`gp_on_scale05` 和 `gp_on_scale07` 可以保留为可选趋势实验，而不是 Stage 4 的硬性要求。

---

## 17. Stage 3A 最终状态记录

Stage 3A 当前状态：

| 项目 | 状态 |
|---|---|
| 真机可用数据 | 已获得 |
| `comp_off fullrun` | 已获得 |
| `gp_on_scale03 fullrun` | 已获得 |
| `gp_on_scale05 partial` | 已获得，趋势参考 |
| `gp_on_scale07 nearfull` | 已获得，趋势参考 |
| CSV NaN / inf 检查 | 通过 |
| CSV columns 一致性 | 通过 |
| estimated Hz | 约 `50 Hz` |
| tau residual RMS | `scale03` 相比 `comp_off` 全部关节下降 |
| Cartesian tracking | `scale03` 与 `comp_off` 相近 |
| clip proxy | 所有 mode 为 `0` |
| fully stable repeated validation | 不声称 |
| Stage 4 准备 | 可以开始 |

最终一句话总结：

> Stage 3A 已经完成一次可用的 frozen GP spatial trajectory 真机验证；`gp_on_scale03` 在不明显破坏 tracking 的前提下降低了所有关节的 tau residual RMS，因此可以进入 Stage 4 formal spatial experiment 的准备阶段，同时继续记录 communication constraints、User Stop 和非 clean shutdown 等工程 caveats。