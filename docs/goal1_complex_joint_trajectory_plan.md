# GOAL1 Complex Joint Trajectory Plan

## Background

GOAL1 的目标是设计一个所有 `q1..q7` 都运动的复杂 joint-space trajectory，让末端轨迹在后续仿真和安全评审阶段也更复杂。本阶段是 GOAL1 B，只做离线 trajectory generator 和 preliminary safety check。

本阶段不启动 ROS2，不运行 `ros2 launch`，不连接 Franka，不 replay 到 controller，不启用 GP-on，不做 online update，也不修改 controller / launch / config / torque command logic。

## GOAL1 A Key Findings

GOAL1 A read-only feasibility review 的关键结论：

- 当前 repo 主路径是 `Cartesian-only trajectory interface + Cartesian impedance torque controller`。
- `TaskSpaceCommand` 只有 `x_des / dx_des / ddx_des`。
- `trajectory_publisher.py` 发布 `/task_space_command`。
- `cartesian_impedance.py` 订阅 `/task_space_command`。
- controller 内部会把 task-space `dx_des / ddx_des` 通过 Jacobian pseudo-inverse 映射成 `dq_des_joint / ddq_des_joint`，但这些是派生量，不是外部 streaming joint desired trajectory。
- 当前没有完整 `trajectory_msgs/JointTrajectory` all-q replay pipeline。
- 当前 `cartesian_impedance.py` 不能直接接收 joint-space trajectory。
- `effort_pd.py` 更接近 joint-space 雏形，但会直接发布 `/effort_command`，不适合直接作为 all-q CSV replay 入口。
- repo 内没有发现 MuJoCo / Isaac Lab / Gazebo 实质仿真入口。
- ROS2 fake hardware 有代码级支持，但当前 `controllers.yaml` 没有 `joint_trajectory_controller`。

因此 GOAL1 B 不尝试接 ROS controller，只生成 offline all-q CSV、plots 和 summary。

## GOAL1 B Scope

GOAL1 B 新增：

- `scripts/generate_goal1_joint_trajectory.py`
- offline conservative all-q multi-sine trajectory
- CSV output
- q / dq / ddq plots
- optional jerk output and jerk plot
- JSON summary
- Markdown summary
- preliminary safety checker

GOAL1 B 不做：

- ROS2 replay
- real robot validation
- FK
- controller integration
- GP-on
- online update
- torque command logic changes

## Script

脚本位置：

- `scripts/generate_goal1_joint_trajectory.py`

默认运行：

- `python3 scripts/generate_goal1_joint_trajectory.py`

包含 jerk 的 30s / 100Hz 示例：

- `python3 scripts/generate_goal1_joint_trajectory.py --duration 30 --sample-rate 100 --include-jerk`

常用参数：

- `--duration`
- `--sample-rate`
- `--output-dir`
- `--prefix`
- `--include-jerk`
- `--no-plots`
- `--fail-on-unsafe`
- `--profile`

第一版只支持 `conservative` profile，但脚本结构保留了未来扩展 profile 的位置。

## Outputs

默认输出目录：

- `outputs/goal1_joint_trajectory/`

默认输出文件：

- `goal1_allq_conservative.csv`
- `goal1_allq_conservative_summary.json`
- `goal1_allq_conservative_summary.md`
- `goal1_allq_conservative_q.png`
- `goal1_allq_conservative_dq.png`
- `goal1_allq_conservative_ddq.png`
- `goal1_allq_conservative_jerk.png`, only when `--include-jerk` is used

CSV columns 至少包括：

- `time`
- `joint_pos_1..7`
- `joint_vel_1..7`
- `joint_acc_1..7`
- `joint_jerk_1..7`, only when `--include-jerk` is used

## Safety Checker Meaning

脚本内显式定义了 conservative position / velocity / acceleration / jerk limits。这些 limits 只用于 offline preliminary screening。

`overall_safety_status=safe` 只表示生成的 CSV 在脚本设定阈值下没有越界。它不表示：

- 可以直接真机执行；
- 可以直接 ROS replay；
- controller 一定能稳定 tracking；
- Franka 硬件一定安全；
- GP compensation 可以打开；
- no-clip / high-scale experiment 可以进行。

正式仿真或真机前必须重新确认项目配置、Franka 实际 limits、controller tracking 行为、collision / workspace / singularity 风险，以及独立 safety review。

## Why No FK Here

本脚本不强行实现 FK。当前 repo 没有一个简单、明确、已接入的 standalone FK library 入口。为了避免在 GOAL1 B 阶段引入错误的机器人模型或隐藏假设，end-effector FK 明确留到后续 MuJoCo / Isaac Lab 阶段处理。

## Explicit Boundaries

本阶段不能直接真机。

本阶段不能直接 ROS replay。

本阶段不启用 GP-on。

本阶段不修改 controller。

本阶段不修改 launch / config。

本阶段不修改任何 torque command logic。

## Later Route

建议后续路线：

- GOAL1 C-M: MuJoCo standalone simulation
- GOAL1 C-I: Isaac Lab standalone simulation
- GOAL1 C-R: optional ROS2 fake hardware / replay
- GOAL1 D: future no-GP real robot data collection, only after separate safety review

GOAL1 D 之前仍应保持 GP-off / no real robot replay 的边界，直到仿真、轨迹 envelope、controller 接口和独立安全评审全部完成。
