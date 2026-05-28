# Stage 5A Meeting Summary

## 1. One-Sentence Summary

Stage 4 已经完成 frozen GP offline residual evaluation，但 real-robot GP-on 暂缓；Stage 5A 进一步发现问题不是单纯 q7 mismatch，而是完整 `joint_pos_1..7 + joint_vel_1..7` 的 14D joint-space support mismatch，所以下一步应做 no-GP live q7 / 14D logging，而不是直接 GP-on。

## 2. What Has Been Completed

### Stage 4

- frozen GP cross-trajectory residual evaluator。
- `GP_A_planar_train` vs `GP_B_zmod_train` offline comparison。
- held-out C offline residual prediction。
- support validator and conservative GP-on gate framework。
- Stage 4 closure / result docs。

### Stage 5A Offline Tooling

- `scripts/validate_stage5_q7_support.py`
- `scripts/run_stage5_support_matrix.py`
- `docs/stage5_support_aware_trajectory_plan.md`
- `docs/stage5a_no_gp_live_q7_logging_runbook.md`
- `docs/stage5a_offline_support_diagnostic_note.md`
- `outputs/stage5_support_matrix/stage5_support_matrix.md`
- `outputs/stage5_support_matrix/stage5_support_matrix.csv`

## 3. Main Offline Result

Stage 4 的核心 offline finding 是：`GP_B_zmod_train` 在 held-out C 上的 offline residual prediction 好于 `GP_A_planar_train`。

- `RMSE_tau_A_to_C = 0.539815817`
- `RMSE_tau_B_to_C = 0.426237082`
- `delta_rmse_A_minus_B = 0.113578736`
- better model on held-out C: `GP_B_zmod_train`
- `GP_A_planar_train`: 7/7 joints constant prediction
- `GP_B_zmod_train`: 7/7 joints input-dependent prediction
- C held-out 数据没有参与训练。
- training 使用 `train-max-samples=2000`。

`GP_B_zmod_train` model artifact 已确认存在于：

- `data/stage4/cross_traj/models/GP_B_zmod_train`

该目录包含 `joint1_local.pkl` 到 `joint7_local.pkl`、`joint1_cloud.pkl` 到 `joint7_cloud.pkl`、`metadata.json` 和 `README.md`。

重要 caveat：

- 这是 offline residual prediction result。
- 这不是 real-robot GP-on tracking improvement proof。
- 这不说明可以直接打开 GP compensation。
- 这不授权跳过 support gate。

## 4. Why Stage 4 GP-on Was Not Forced

Stage 4 没有强行继续 GP-on，是因为 conservative support gate 暴露了 q7 / support mismatch。

已知 q7 distribution：

- B1 q7 mean 约 `0.1238`
- B2 q7 mean 约 `-0.1296`
- C1 q7 mean 约 `-0.1139`
- C2 q7 mean 约 `-0.8341`
- live q7 observed before potential GP-on 约 `-0.349`

当时 `stage4_cross_traj_GP_B_conservative_C2` 的 validator failed：

- status: `fail_formal_out_of_support`
- blocking reason: `formal_out_of_support`
- worst dimension: `q7`

虽然 `stage4_cross_traj_GP_B_conservative_B2_to_C1` 曾经通过 conservative gate：

- `preflight_gate_pass = true`
- `overall_status = pass_ready_for_conservative_robot_validation`
- `gp_online_update_enabled=false`
- `gp_compensation_scale=0.1`
- `gp_compensation_clip_nm=0.5`

但 live q7 约 `-0.349` 不在当时可用 support gate 内，所以不应该继续 GP-on。

原因是当前 `z_modulated_circle` 是 Cartesian 3D trajectory，不是 full 7DoF joint-space trajectory。Franka 是 7DoF redundant robot；类似的 Cartesian path 可能对应不同 q7 / nullspace posture family。GP input 又使用 joint positions + joint velocities，所以 q7 / posture family mismatch 会直接影响 frozen GP support。

因此 Stage 4 的收束是合理的，不是失败。它把结果停在 offline residual diagnostic 和 support-aware gate，而不是把不在 support 内的 real-robot GP-on 强行跑下去。

## 5. New Stage 5A Finding: q7 Alone Is Not Enough

Stage 5A batch support matrix 的新发现是：q7 很重要，但 q7 alone is insufficient。

Matrix result：

- `pairs_total: 12`
- `pairs_evaluated: 12`
- `pairs_skipped: 0`
- full 14D support pass: `0`

q7 pass 但 14D fail 的 cases：

- `GP_B_zmod_train_vs_C1`
- `B1_B2_reference_vs_C1`

这两个 pair 的共同结果：

- `q7_support_pass=true`
- `joint_space_14d_pass=false`
- `overall_status=fail_14d_out_of_support`
- `worst_dimension=joint_pos_4`

其他关键结果：

- `GP_B_zmod_train_vs_C2`: q7 fail，`worst_dimension=joint_pos_7`
- `GP_A_planar_train` 对 C1 / C2 都 q7 fail

Worst dimension distribution：

- `joint_pos_1`: 5 次
- `joint_pos_7`: 5 次
- `joint_pos_4`: 2 次

解释：

- q7 是早期暴露出来的明显 blocker。
- 但完整 frozen GP support gate 必须看 `joint_pos_1..7 + joint_vel_1..7`。
- 即使 q7 pass，也可能因为 `joint_pos_4` 或其他维度 out of support 而 fail。
- 所以 Stage 5A 不能只修 q7，应升级为 14D joint-space support-aware preflight。

## 6. Updated Interpretation

旧理解：

- q7 / nullspace support mismatch 是 GP-on blocker。

更新后理解：

- q7 是 blocker 之一，但不是唯一 blocker。
- Stage 5A 应该做 `q7 / 14D support-aware posture consistency`。
- 目标是让 train / test / live posture family 在 14D joint-space feature support 上更一致。
- 当前不应直接 full 7DoF joint-space excitation。
- 当前也不应直接 GP-on。

更准确的 Stage 5A framing：

- `q7 / 14D support-aware posture consistency`
- `14D joint-space support-aware trajectory/preflight design`
- no-GP live support logging before any compensation discussion

## 7. Next Real-Robot Step

下一次有真机时，只做 no-GP live q7 / 14D logging。

建议流程：

- 使用已 review 的 `docs/stage5a_no_gp_live_q7_logging_runbook.md`。
- 保持 no-GP：不做 compensation experiment。
- 记录 `trajectory_mode`、`z_amplitude`、`z_frequency_multiplier`、`circle_frequency`、`transition_duration`。
- 保存 CSV 和 terminal log。
- 确认 CSV 有 `joint_pos_1..7` 和 `joint_vel_1..7`。
- 用 `scripts/validate_stage5_q7_support.py` 检查 live CSV。
- 必要时用 `scripts/run_stage5_support_matrix.py` 把 live data 加入 matrix。

Validator exit code：

- `0 = preflight pass`
- `1 = invalid input / usage error`
- `2 = valid input but support/preflight fail`

解释规则：

- no-GP live logging 不是 compensation experiment。
- exit code `0` 也不直接授权 GP-on。
- exit code `2` 或任何 q7 / 14D support fail 都应停止，不进入 GP-on。
- validator pass 只代表可以进入下一轮 review，不代表可以直接打开 compensation。

## 8. What We Should Not Do Yet

现在不应该做：

- 不做 GP-on。
- 不做 no-clip。
- 不做 high scale。
- 不做 scale sweep。
- 不做 online update。
- 不改 controller / torque path。
- 不改 GP compensation path。
- 不改 launch defaults。
- 不直接上 full 7DoF joint-space excitation。
- 不把 offline residual prediction 说成 real-robot tracking improvement。
- 不把 offline support matrix 说成 GP-on validation。
- 不把 validator pass 说成 GP-on approval。

## 9. How to Explain This in a Meeting

### Short version

Stage 4 已经收束了，不是失败。我们完成了 frozen GP 的 cross-trajectory offline residual evaluation，结果显示 `GP_B_zmod_train` 在 held-out C 上比 planar GP 的 residual prediction 更好，RMSE 从约 `0.5398` 降到 `0.4262`。但这个结果只是 offline residual prediction，不是 GP-on tracking improvement proof。后来 support gate 显示 q7 / nullspace posture 有 mismatch，所以没有强行上 GP-on。Stage 5A 进一步发现 q7 不是唯一问题，因为有些 case q7 pass 但完整 14D joint-space support fail，worst dimension 是 `joint_pos_4`。所以下一步不是继续 GP-on，而是先做 no-GP live q7 / 14D logging，把真实机器人 live posture family 查清楚。

### Slightly More Technical Version

Stage 4 的主要价值是把 frozen GP comparison 和 support-aware gate 搭起来了。`GP_B_zmod_train` 在 held-out C 上的 offline residual prediction 明显好于 `GP_A_planar_train`，而且 `GP_A` 出现 7/7 joints constant prediction，`GP_B` 则是 7/7 joints input-dependent prediction。不过这只能说明 residual model diagnostic 更好，不能直接说明 real-robot GP-on tracking 会变好。

真正阻止继续 GP-on 的是 support mismatch。早期最明显的是 q7：Franka 是 7DoF redundant robot，当前 `z_modulated_circle` 只定义 Cartesian path，不定义完整 joint posture family，所以 live q7 可能落在 frozen GP training support 外。Stage 5A matrix 又进一步说明，q7 pass 也不够；`GP_B_zmod_train_vs_C1` 和 `B1_B2_reference_vs_C1` 都是 q7 pass 但 14D fail，worst dimension 是 `joint_pos_4`。所以现在的科学问题已经从“能不能继续 GP-on”变成“train / test / live 的 14D joint-space support 是否一致”。下一步应先 no-GP 采 live q7 / 14D 数据，再离线检查 support，GP-on 和 full 7DoF excitation 都继续暂缓。

## 10. Current Status and Decision

- Stage 4: closed cleanly。
- Stage 5A offline docs/tools: ready。
- 当前主要结论：q7 alone is insufficient，14D support-aware gate is needed。
- real-robot next step: no-GP live q7 / 14D logging。
- GP-on: deferred。
- full 7DoF joint-space excitation: deferred。
- immediate offline task after this doc: data inventory / lab quick checklist only if needed。

## 11. Safety Notes

本 summary 的安全边界：

- offline result only。
- no GP-on proof。
- no real-robot launch in this task。
- no controller change。
- no torque path change。
- no launch/config change。
- no GP compensation path change。
- no online update。
- no no-clip。
- no high scale。
- no scale sweep。
- no GP-on claim。
- no tracking improvement claim。
- validator pass is not GP-on approval。

Meeting 里应避免的说法：

- 不说 “GP_B 已经证明真机 tracking 更好”。
- 不说 “matrix pass 就可以 GP-on”。
- 不说 “q7 调好就一定够了”。
- 不说 “可以靠更大 scale 试出来”。

更稳妥的说法：

- `GP_B_zmod_train` 是 better offline residual predictor。
- Stage 5A 的重点是 support-aware live logging。
- GP-on re-entry 需要单独 review、单独 safety gate、单独 decision。
