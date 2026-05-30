# GOAL1 Status Before Lab Validation

## 当前阶段一句话总结

GOAL1 现在处于 lab-side readiness validation 前的 offline 准备阶段；导师已经认可当前轨迹复杂性，因此当前重点不再是继续增加 trajectory complexity，而是准备实验室 Linux 上的 state-only validation 和 safety review。

## 当前已完成内容

- WSL offline trajectory 已完成。
- FR3 MuJoCo replay 已完成。
- 60s / 50Hz polished video 已完成。
- default-disabled joint-space replay skeleton 已完成。
- 导师已认可当前轨迹复杂性。

## 当前尚未完成内容

- 在 lab Linux 上 pull latest branch。
- build `custom_msgs` / `py_controllers`。
- pure dry-run CLI。
- node-only launch dry-run。
- 使用 `state_only=true publish_effort=false` 检查 `/state_parameter`。
- 确认 `/effort_command` 不被发布。
- 单独处理 `cpp_relayer` safety blocker。

## 明确禁止事项

- 不运行 `publish_effort=true`。
- 不做 GP-on。
- 不做 full 60s trajectory。
- 不让机器人运动。
- 不发布 `/effort_command`。
- 不把 WSL fake/offline 成功等同于 lab Linux real-robot readiness。

## 下一步推荐顺序

1. 完成本 GOAL1 当前状态文档。
2. 准备 GOAL1 K lab checklist。
3. read-only review 默认安全边界。
4. 单独做 `cpp_relayer` safety blocker diagnosis。
5. 上实验室 Linux 做 GOAL1 K state-only validation。

## Engineering standard

本项目采用“成功一次跑出可用数据即可推进”的工程标准：只要一次实验能够产生可信、可解释、可复现到足够程度的数据，就可以作为下一阶段推进依据。

但当前 GOAL1 K 不是正式数据采集，而是 lab readiness gate。它的目标是确认实验室 Linux、ROS2 package、launch 参数、安全默认值、state-only 数据路径和 no-effort 边界是否可用，而不是评估 GP compensation 或正式轨迹表现。

`communication_constraints_violation`、User Stop、`rclpy` shutdown errors 可以作为 caveat 记录；它们不等于完全阻塞离线分析。只要已获得的数据仍然可解释，并且问题不影响结论边界，就可以继续整理 offline 结果。

但是，在任何 `publish_effort=true` 前，`cpp_relayer` stale torque / hold-last-command 风险必须单独处理。需要独立 review 或 patch timeout、zero fallback、stale-command refusal，或者形成明确 operator abort plan。

## Notes for future real-robot work

- WSL 用于代码、Git、Python、离线分析和文档。
- Windows 用于看图、视频、截图、临时备份。
- 实验室 Linux 才用于真实 Franka / ROS2 / hardware validation。
- WSL 成功不代表真机 Linux 一定可运行。
