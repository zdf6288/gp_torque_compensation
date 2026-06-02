# GOAL1 Real Robot Route Decision - 2026-06-02

## Scope

本文档记录 2026-06-02 GOAL1 lab findings 和当前 real-robot route decision。

这是 documentation-only note。

本次结论不修改 controller / launch / config，不要求运行 `ros2 launch`、`ros2 run`、`ros2 control`，也不要求 controller activation / deactivation、`/effort_command` publish、real robot command、commit 或 push。

## Summary

GOAL1 direct real-robot final-effort replay 目前停止。

今天的结果说明：直接把 `goal1_joint_space_replay` 作为 final effort publisher，并通过 active `cpp_relayer` 写入 `panda_joint*/effort`，在当前架构和启动顺序下不再适合作为下一步 real-robot motion route。

本次没有产生 usable real motion tracking data，但产生了有用的 safety evidence 和 architecture evidence。

## Successful Gates

今天确认成功的 gate：

- lab pull/build 到 `01143bb Allow GOAL1 final effort replay from joint states`。
- current-q anchored CSV generation 成功。
- state-only preflight pass。
- no-motion `/effort_command` publish-path test 成功，且测试时 `cpp_relayer inactive`。
- patch `ecb8ad7 Require fresh command before cpp_relayer activation` 在 lab build success。
- `require_fresh_command_on_activate=true` runtime parameter visible。

这些 gate 说明 offline / no-motion / state-only 路径有实际价值，也说明 fresh-command activation gate 已经进入 lab runtime 可见状态。

## Blockers Found

今天发现的 blocker：

- direct `cpp_relayer active` 导致 `communication_constraints_violation`。
- fresh-command activation gate 拒绝了 activation，但后续仍出现 claimed `panda_joint*/effort` 和 Franka reflex。
- `cartesian_impedance_launch.py` 当前 launch order 与 fresh-command gate 不兼容，因为它先 spawns `cpp_relayer`，再启动 `cartesian_impedance`。
- `cartesian_impedance.py` 创建 `/effort_command` publisher，但在 `/state_parameter` 到来前不会自然 publish fresh command。
- `/state_parameter` 依赖 `cpp_relayer`。

因此，当前问题不是简单的参数重试问题，而是启动依赖和安全 gate 之间存在架构性冲突。

## Architecture Conclusion

当前相关依赖链可以概括为：

`cpp_relayer active -> /state_parameter -> cartesian_impedance computes tau -> /effort_command -> cpp_relayer activation gate`

这个链条形成 dependency loop：

- `cpp_relayer` active 后才提供 `/state_parameter`。
- `cartesian_impedance.py` 需要 `/state_parameter` 才能计算 `tau`。
- `cartesian_impedance.py` 计算 `tau` 后才会 publish `/effort_command`。
- fresh-command activation gate 又要求 activation 前已经有 fresh `/effort_command`。

因此，在不重新设计 startup strategy / state source / hold path 的情况下，现有 Cartesian launch 不能直接在 fresh-command gate 下用于 real robot。

## Route Decision

当前 route decision：

- 不再 retry real robot 上的 `cpp_relayer` activation。
- 不再 retry real robot 上的 `goal1_joint_space_replay` direct effort。
- 不在当前 launch order 下直接运行 real robot `cartesian_impedance_launch.py`。
- direct replay 只保留给 offline / fake / no-motion 使用。
- 保留 `ecb8ad7` fresh-command gate，除非未来经过单独 review 的 route 明确需要不同 startup strategy。

这个决定的重点是避免把已知的 startup dependency loop 当作可通过重复尝试解决的问题。

## Next Possible Directions

后续可能方向：

1. 设计 dedicated safe-start / hold controller。
2. 设计 prearm/hold mode，并使用经过 review 的 state source。
3. 在 fake hardware 上验证 staged launch。
4. 使用 GOAL1 no-motion/offline evidence，并把 thesis focus 转向 GOAL2 / offline residual prediction / timing。
5. 只有在 separate review 之后，通过 lab-proven pipeline 再重新考虑 real motion。

## Caveats

项目标准 caveat：

- `communication_constraints_violation` 和 non-clean shutdown 是 caveats。
- usable data success 与 robust repeated stability 是不同标准。
- 今天的 GOAL1 没有产生 usable real motion tracking data。
- 今天的 GOAL1 产生了有用的 safety evidence 和 architecture evidence。

因此，今天的结果不应被写成 real-robot tracking success；更准确的表述是：GOAL1 direct real-robot route 被 safety / architecture evidence 停止，offline / no-motion 证据仍可用于后续论文路线判断。
