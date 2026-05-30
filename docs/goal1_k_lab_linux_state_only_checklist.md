# GOAL1 K Lab Linux State-only Validation Checklist

## Scope

这是 GOAL1 K 的 lab-side readiness validation checklist，用于在实验室 Linux 上确认当前 branch、build、dry-run、node-only launch 和 state-only topic 行为是否满足进入后续安全审查的基础条件。

当前不是正式 GP-on 实验，不让机器人运动，也不验证 torque control。当前只检查：

- build 是否能通过。
- dry-run / entry point 是否可被安全确认。
- node-only launch 参数和启动边界是否清楚。
- `state_only=true publish_effort=false` 时 `/state_parameter` 是否正常。
- `/effort_command` 是否不被发布，或至少没有可疑 publisher。

## Global safety rules

- 禁止 `publish_effort:=true`。
- 禁止 GP-on。
- 禁止 full 60s trajectory。
- 禁止发布 `/effort_command`。
- 禁止任何让机器人运动的操作。
- 即使 `cpp_relayer` safety guard 已添加，本 checklist 仍不进入 torque publishing path。

## Gate 0: Machine / path / branch / working tree check

目标：确认当前在正确机器、正确 workspace、正确 branch，并且没有会干扰 pull/build 的本地修改。

命令：

```bash
hostname
whoami
pwd
cd ~/projects/gp_torque_compensation 2>/dev/null || cd ~/dongfa/tt_dgp 2>/dev/null || pwd
pwd
git branch --show-current
git status --short
git remote -v
git log --oneline -5
```

Expected result：

- `pwd` 指向实验室 Linux 上的目标 workspace。
- branch 是 `frozen_gp_spatial_trajectory`。
- `git status --short` 为空，或只有已确认无关且不会被覆盖的本地文件。
- remote 指向预期的 `gp_torque_compensation` repo。
- 最近 commit 能对应当前准备验证的代码状态。

Stop condition：

- 当前目录不是目标 workspace。
- branch 不是 `frozen_gp_spatial_trajectory`。
- `git status --short` 有未确认 local modifications。
- remote 明显不是目标 repo。

如果触发 stop condition，把完整输出贴给 ChatGPT 判断，不要继续。

## Gate 1: Pull latest safely

目标：只在工作树干净且 branch 正确时，安全更新到最新代码。

命令：

```bash
git status --short
git branch --show-current
git fetch --all --prune
git status -sb
```

如果 `git status -sb` 显示当前 branch behind remote，且工作树干净，再运行：

```bash
git pull --ff-only
git status -sb
git log --oneline -5
```

Expected result：

- pull 前 `git status --short` 为空。
- branch 是 `frozen_gp_spatial_trajectory`。
- `git pull --ff-only` 成功，或确认本来就是 up to date。

Stop condition：

- 有 local modifications：停止，不要 pull。
- branch 不对：停止，不要切 branch，先确认。
- `git pull --ff-only` 失败：停止，保存完整输出。

## Gate 2: Build only required packages

目标 package：

- `custom_msgs`
- `py_controllers`

以下命令只在实验室 Linux 运行：

```bash
source /opt/ros/humble/setup.bash
# 根据 workspace 实际情况 source existing install if needed
colcon build --packages-select custom_msgs py_controllers --symlink-install
```

Expected result：

- `custom_msgs` build 成功。
- `py_controllers` build 成功。
- 没有缺失 message、Python package、ROS dependency 或 setup/install 错误。

Stop condition：

- build fail：停止，保存完整 error，不要继续 launch。
- 出现 dependency 或 overlay 相关异常：停止，记录当前 sourced environment。
- 需要安装依赖时：停止，先确认，不要现场随意改系统环境。

## Gate 3: Pure dry-run CLI

目标：只检查 Python entry point / CLI 是否能被 import 或显示 help，不接 robot，不 publish effort。

由于具体 entry point 需要从 package 文件确认，先做 read-only 检查，不盲跑任何 node：

```bash
grep -R "entry_points" -n new_structure/py_controllers setup.py setup.cfg pyproject.toml 2>/dev/null || true
grep -R "console_scripts" -n new_structure/py_controllers setup.py setup.cfg pyproject.toml 2>/dev/null || true
```

Expected result：

- 能看清 `py_controllers` 的 console scripts / entry points。
- 能判断哪些命令只是 help/import，哪些命令可能启动 controller、publisher 或 trajectory。

Stop condition：

- entry point 指向 controller、trajectory、replay、relayer 或任何可能发布 `/effort_command` 的路径。
- 无法确认参数是否安全。
- 需要运行 `ros2 run` 才能判断行为。

只有确认安全参数后，才考虑 dry-run。禁止任何会发布 `/effort_command` 的命令。

## Gate 4: Node-only launch dry-run

目标：只做 launch 参数和 node startup 层面的 dry-run，不进入 robot motion 或 effort publishing。

launch 前必须先 inspect launch file 参数默认值：

```bash
grep -R "state_only" -n new_structure/py_controllers new_structure/new_bringup 2>/dev/null || true
grep -R "publish_effort" -n new_structure/py_controllers new_structure/new_bringup 2>/dev/null || true
grep -R "gp_.*enabled\|gp_prediction\|compensation" -n new_structure/py_controllers new_structure/new_bringup 2>/dev/null || true
```

Safe parameters 必须明确包含：

- `state_only:=true`
- `publish_effort:=false`
- GP disabled if parameter exists

Expected result：

- launch / node 参数存在明确 safe gating。
- 默认值不会进入 torque publishing path。
- 能明确指定 `state_only:=true publish_effort:=false`。

Stop condition：

- launch file 没有明确 safe gating。
- `publish_effort` 默认值不清楚。
- GP compensation / GP prediction 是否启用不清楚。
- launch 可能启动 trajectory、relayer、controller torque path 或 robot motion。

如果触发 stop condition，停止并贴出 grep 结果，不要 launch。

## Gate 5: State-only topic inspection

目标：

- `/state_parameter` 正常。
- `/effort_command` 不应出现，或不应有 publisher。

topic 检查命令：

```bash
ros2 topic list
ros2 topic info /state_parameter
ros2 topic echo /state_parameter --once
ros2 topic info /effort_command
```

Expected result：

- `/state_parameter` 存在，并能 echo 到一次合理 state-only message。
- `/effort_command` 不存在，或 `ros2 topic info /effort_command` 显示没有 publisher。
- 没有 trajectory command、effort command 或 robot motion 相关 topic 被主动发布。

Stop condition：

- `/state_parameter` 不存在或 echo 失败。
- `/effort_command` 有可疑 publisher。
- 发现任何可能驱动机器人运动的 publisher。
- node 输出显示进入 torque publishing path。

如果 `/effort_command` 有可疑 publisher，立即停止并记录 node/topic 输出。

## Gate 6: Shutdown and post-check

目标：确认 dry-run / state-only 检查结束后没有残留 node 或相关进程。

操作：

- 用 Ctrl+C 停止 node / launch。
- 等待终端完全返回 shell。
- 检查没有残留 ROS2 node 或相关 process。

命令：

```bash
ros2 node list
ps aux | grep -E "cartesian|trajectory|replay|gp|relayer" | grep -v grep || true
```

Expected result：

- `ros2 node list` 中没有本次测试残留 node。
- `ps aux` 中没有 `cartesian`、`trajectory`、`replay`、`gp`、`relayer` 相关残留进程。

Stop condition：

- 有残留相关 node/process。
- Ctrl+C 后 node 未正常退出。
- 需要手动 kill 时，先记录完整输出并确认，不要继续下一步。

## GO / NO-GO decision table

| Gate | Pass 条件 | Stop 条件 |
| --- | --- | --- |
| Gate 0 | 机器、路径、branch、remote 正确，working tree 干净 | 路径错误、branch 错误、remote 错误、有未确认 local modifications |
| Gate 1 | 已安全 fetch/pull，branch up to date 或确认无需更新 | local modifications、branch 不对、`git pull --ff-only` 失败 |
| Gate 2 | `custom_msgs` 和 `py_controllers` build 成功 | build fail、依赖缺失、overlay/environment 异常 |
| Gate 3 | entry points 已 read-only 确认，未运行危险命令 | entry point 行为不清楚，可能发布 `/effort_command` |
| Gate 4 | launch 参数明确 safe，包含 `state_only:=true publish_effort:=false` | safe gating 不明确，可能进入 torque publishing path |
| Gate 5 | `/state_parameter` 正常，`/effort_command` 无 publisher | `/state_parameter` 异常，`/effort_command` 有可疑 publisher |
| Gate 6 | shutdown 后无残留 node/process | 有残留 node/process 或 shutdown 异常 |

## What this checklist does NOT validate

- 不验证真实 torque control。
- 不验证 GP-on。
- 不验证 60s full trajectory。
- 不验证 `cpp_relayer` safety。
- 不证明机器人可以安全运动。
- 只证明 lab Linux 上 state-only readiness 有一定基础。

## cpp_relayer safety guard status

`cpp_relayer` 代码现在包含 runtime safety guard：

- configurable `command_timeout_sec`
- no-command / stale-command zero fallback
- stale-command refusal
- invalid `EffortCommand` refusal for wrong length or non-finite values
- deactivate-time command interface zeroing

这只解决 stale torque / hold-last-command 的代码层 blocker，不代表 GOAL1 K 可以直接进入 torque publishing path。

本 checklist 仍然只允许 `state_only=true publish_effort=false` validation。任何 `publish_effort=true` 仍然需要单独 lab-side review、operator abort plan、短时 guarded test，并在现场确认 robot / controller / topic 状态后才能考虑。
