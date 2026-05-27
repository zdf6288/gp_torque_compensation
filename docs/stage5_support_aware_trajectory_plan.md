# Stage 5 Support-Aware / 7DoF-Aware Trajectory Plan

## 1. Purpose

Stage 5 的目标是从 Stage 4 的 frozen GP 实验结果中退一步，先解决 support mismatch 和 7DoF redundancy 问题，而不是继续强行推进 GP-on real-robot validation。

Stage 5A 定义为 support-aware `z_modulated_circle` / q7 posture consistency 阶段：

- 不再强行推进 Stage 4 GP-on。
- 先解决 frozen GP support mismatch。
- 重点关注 q7 / nullspace posture family。
- 保持第一版工作 offline/preflight-first。
- 把 full 7DoF joint-space excitation 放到 Stage 5B 或更后。

Stage 5A 的成功标准不是证明 GP compensation 改善 tracking，而是确认 candidate/live trajectory 的 joint-space support，尤其是 q7 posture family，是否与 frozen GP training support 一致到足够保守。

## 2. Stage 4 Closure Summary

Stage 4 achieved:

- frozen GP cross-trajectory residual evaluator。
- support validator。
- conservative GP-on gate framework。
- `GP_B_zmod_train` 在 held-out C offline residual prediction 上优于 `GP_A_planar_train`。
- `GP_A_planar_train` 在 held-out C 上出现 7/7 joints constant prediction。
- `GP_B_zmod_train` 在 held-out C 上出现 7/7 joints input-dependent prediction。

Stage 4 did not prove:

- real-robot GP-on tracking improvement。
- robust repeated GP-on validation。
- stable compensation behavior on real robot。

因此，Stage 4 的 offline residual prediction 结果只能作为 residual-model diagnostic。它不能被写成 GP-on tracking improvement proof，也不能跳过 support gate 直接作为继续真机 GP-on 的理由。

## 3. q7 / Nullspace Support Problem

Franka 是 7DoF redundant robot。当前 `z_modulated_circle` 是 Cartesian trajectory，不是 full joint-space trajectory；相同或相似的 Cartesian path 可能落到不同 q7 / nullspace posture family。

当前 GP input 使用 joint positions + joint velocities。也就是说，q7 不是旁枝信息，而是 frozen GP feature vector 的一部分。即使 end-effector Cartesian trajectory 看起来相同，如果 q7 posture family 不同，frozen GP support gate 仍然可能 fail。

Stage 4 support checks 已经把 q7 / nullspace posture mismatch 暴露为主要 blocker。live q7 observed before potential GP-on 约为 `-0.349`，这不应被视为可以靠 GP compensation 强行试出来的问题。正确路径是先把 q7 support mismatch 变成 explicit gate，再决定是否有资格进入 conservative GP-on re-entry。

## 4. Current Architecture Constraint

基于 Stage 5A read-only analysis，当前架构约束如下：

- `trajectory_publisher.py` 当前是 Cartesian-only trajectory publisher。
- `trajectory_publisher.py` 发布 `/task_space_command`，内容是 Cartesian `x_des` / `dx_des` / `ddx_des`。
- 没有发现 q7 target。
- 没有发现 elbow target。
- 没有发现 full 7DoF joint-space target。
- 没有发现 active nullspace posture target。
- `cartesian_impedance.py` 有 nullspace velocity damping，但不是 active posture target control。
- 当前 q7 / nullspace posture family 主要由真实 robot 当前姿态、startup 到 Cartesian start point 的过程、controller redundancy resolution 和 nullspace damping 间接决定。

因此，Stage 5A 第一版不应该改 controller、controller callback、`_apply_gp_compensation()` 或 torque path。第一版应先固定 offline/preflight workflow 和 q7 logging checklist。

## 5. Stage 5A Plan: Support-Aware zmod / q7 Posture Consistency

Stage 5A 推荐先保持 Cartesian `z_modulated_circle` 主体不变，同时把 q7 / posture support family 变成可检查、可记录、可 fail 的前置条件。

推荐路线：

- 先保持 Cartesian `z_modulated_circle` 主体不变。
- 先分析和约束 q7 / posture support family。
- 先做 offline/preflight validator。
- 先做 no-GP live q7 data collection。
- 只有 support gate pass 后，才考虑 conservative GP-on re-entry。
- 不做 scale sweep。
- 不做 no-clip compensation。
- 不开启 online update。

Stage 5A 子任务：

1. `5A.1` write roadmap and checklist。
2. `5A.2` q7-focused support report。
3. `5A.3` no-GP live q7 logging。
4. `5A.4` compare train/test/live q7 support。
5. `5A.5` decide whether support-aware zmod is enough。
6. `5A.6` only then consider conservative GP-on gate。

`5A.6` 不是自动进入 GP-on。它只是判断是否具备提交新的 conservative GP-on review 的条件。

## 6. Stage 5B Plan: Conservative 7DoF Joint-Space Excitation

Stage 5B 暂缓。full 7DoF joint-space excitation 不是下一步。

如果未来进入 Stage 5B，目标应先是 data collection / support coverage，而不是直接 GP-on compensation。full 7DoF trajectory 至少需要以下前置条件：

- read-only design doc。
- joint limit checker。
- velocity limit checker。
- acceleration limit checker。
- jerk / smoothness checker。
- workspace and self-collision sanity check。
- no-GP dry-run protocol。
- data quality and support coverage report。
- separate safety review。

Stage 5B 不应直接通过 controller 大改来实现复杂 nullspace/posture control。任何 controller/nullspace/posture modification 都应有单独设计、单独 review、单独验证，不应和 GP-on re-entry 混在同一个 patch 或同一次真机实验里。

## 7. No-GP Live q7 Logging Checklist

这个 checklist 用于之后真机 Linux 采集 no-GP q7 data 前确认。它不是 GP-on checklist。

### Pre-run checks

- [ ] Confirm branch is `frozen_gp_spatial_trajectory`。
- [ ] Confirm `git status --short` is clean or all changes are documented and reviewed。
- [ ] Confirm robot IP is correct and intentionally selected。
- [ ] Confirm `gp_prediction_enabled=false` or `gp_compensation_enabled=false` for this run。
- [ ] Confirm `gp_online_update_enabled=false`。
- [ ] Confirm `gp_compensation_enabled=false`。
- [ ] Confirm there are no unreviewed controller changes。
- [ ] Confirm there are no unreviewed launch/config changes。
- [ ] Record `trajectory_mode`。
- [ ] Record `z_amplitude`。
- [ ] Record `z_frequency_multiplier`。
- [ ] Record `circle_frequency`。
- [ ] Record `transition_duration`。
- [ ] Prepare and record output directory。
- [ ] Confirm the run purpose is no-GP q7 logging only。

### Run constraints

- [ ] Do not run GP-on。
- [ ] Do not enable online update。
- [ ] Do not enable compensation。
- [ ] Collect joint positions and joint velocities。
- [ ] Confirm CSV includes `joint_pos_1..7` and `joint_vel_1..7` or equivalent columns。
- [ ] Focus first review on q7 range / mean / std。
- [ ] Stop immediately if robot motion becomes abnormal。
- [ ] Stop immediately if vibration, abnormal sound, reflex stop, or unsafe communication behavior appears。

### Post-run checks

- [ ] Verify CSV exists。
- [ ] Verify CSV columns include joint positions / velocities。
- [ ] Extract q7 min / max / mean / std。
- [ ] Compare q7 against Stage 4 train/test support。
- [ ] Compare 14D support against model stats and training CSV support。
- [ ] Record whether q7 is inside or outside support。
- [ ] Record worst support dimension。
- [ ] Do not proceed to GP-on unless validator passes。

## 8. Validator / Preflight Requirements

The next script-level implementation may either:

- extend `scripts/validate_frozen_gp_support.py`; or
- add a new q7-focused support preflight script.

The first implementation should remain offline-first. It should consume existing CSV/model artifacts and write reports only. It should not modify controller, launch/config, GP model files, or torque command behavior.

Required outputs:

- q7 min / max / mean / std。
- q7 support pass/fail。
- worst support dimension。
- 14D support pass/fail。
- comparison between train CSV, candidate/live CSV, and model stats。
- formal gate recommendation。
- explicit blocking reason when q7 is out of support。

Recommended inputs:

- `--model-dir`
- `--train-csv`
- `--candidate-csv` or `--live-csv`
- `--feature-source`
- `--mode-name`
- optional conservative safety parameters for report metadata, such as `gp_online_update_enabled`, `gp_compensation_enabled`, `gp_compensation_scale`, and `gp_compensation_clip_nm`

The validator should make q7 support failure obvious. A q7 support fail should be a blocking reason for GP-on re-entry, not a warning that can be ignored during real-robot launch.

## 9. Safety Boundaries

Stage 5A must preserve these boundaries:

- Do not touch `tau`。
- Do not touch `_apply_gp_compensation()`。
- Do not modify controller callback。
- Do not enable online update。
- Do not remove clip。
- Do not use high scale。
- Do not run no-clip GP compensation。
- Do not treat offline residual prediction as real-robot tracking proof。
- Do not launch GP-on when support gate fails。
- Do not directly implement full 7DoF trajectory without separate design and safety review。

Additional real-robot caveats:

- Stage 5A no-GP q7 logging is still a real-robot activity and should use the normal Franka safety workflow.
- Support-aware validation must happen before any GP-on re-entry.
- If q7 support mismatch remains unresolved, the correct result is `blocked`, not "try a larger compensation scale"。

## 10. Next Recommended Implementation

Preferred next implementation after this doc:

1. Add a q7-focused support report / preflight script; or
2. extend `scripts/validate_frozen_gp_support.py` with q7-focused report fields and blocking reasons.

The next implementation should remain offline first:

- no controller modification。
- no trajectory publisher modification。
- no launch/config modification。
- no real-robot launch。
- no torque path modification。
- no GP compensation behavior modification。

After the offline/preflight report exists, the next reviewed step can be a no-GP live q7 logging run following the checklist above. Conservative GP-on re-entry should only be considered after support gate pass and separate safety review.
