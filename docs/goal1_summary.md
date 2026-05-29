# GOAL1 Summary: All-q Joint Trajectory and FR3 MuJoCo Offline Replay

## Executive Summary

GOAL1 当前已经完成一条从 all-q joint trajectory generation 到 FR3 MuJoCo kinematic replay 和 video export 的 offline pipeline。

这条 pipeline 的核心意义是：先用离线方式证明 `q1..q7` 都运动的 conservative joint-space trajectory 可以被生成、检查、映射到 MuJoCo model，并可视化为 end-effector path 和 video。

当前结果只证明 offline kinematic feasibility / visualization。它不代表 ROS2 controller tracking，不代表 real robot safety，不涉及 GP-on，也不说明 torque compensation 已可用于这条 trajectory。

## Scope and Safety Boundary

本阶段边界非常明确：

- no real robot run
- no ROS2 controller integration
- no torque control
- no actuator control
- no `mj_step` dynamic simulation
- no GP-on
- no online update experiment
- no `/effort_command`
- no claim of controller tracking
- no claim of hardware safety

所有 MuJoCo replay 都是 standalone kinematic replay：读取 offline CSV，设置 MuJoCo `qpos`，调用 `mj_forward`，记录或渲染 robot pose。Video 只是 visualization，不是 tracking validation。

## Completed Milestones

### GOAL1 B - Offline all-q trajectory generator

- script: `scripts/generate_goal1_joint_trajectory.py`
- doc: `docs/goal1_complex_joint_trajectory_plan.md`
- commit: `7d12e51 Add GOAL1 B offline all-q trajectory generator`
- main output: `outputs/goal1_joint_trajectory/goal1_allq_conservative.csv`
- additional outputs: q / dq / ddq / jerk plots, summary JSON/MD

Core result:

- `q1..q7` all move.
- `q7` uses small amplitude motion.
- Conservative offline safety summary status is `safe`.
- The generated CSV includes position, velocity, acceleration, and optional jerk columns.

Caveat:

- This is an offline CSV generator and preliminary checker only.
- It is not replay-ready for the real robot.
- It does not authorize ROS2 replay, controller tracking, GP-on, or Franka execution.

### GOAL1 C-M - Panda MuJoCo replay prototype

- script: `scripts/replay_goal1_trajectory_mujoco.py`
- doc: `docs/goal1_mujoco_replay_plan.md`
- commit: `76eed6e Add GOAL1 C-M MuJoCo kinematic replay`
- model: `/home/dummd/mujoco_models/mujoco_menagerie/franka_emika_panda/panda.xml`
- body/EE candidates: `hand`, `link7`

Core result:

- The CSV replay pipeline works in standalone MuJoCo.
- The script maps `joint_pos_1..7` into MuJoCo `qpos`.
- The script calls `mj_forward` and generates EE xyz path and 3D path plots.

Caveat:

- Panda replay is a useful pipeline prototype, not an FR3-specific result.
- It should not be used as the model-specific claim for the real FR3 / Research 3 setup.

### GOAL1 C-M-FR3 - FR3 MuJoCo replay

- doc: `docs/goal1_fr3_mujoco_replay_summary.md`
- commit: `a69f701 Add GOAL1 C-M FR3 MuJoCo replay summary`
- primary model: `/home/dummd/mujoco_models/mujoco_menagerie/franka_fr3/fr3.xml`
- primary joints: `fr3_joint1..fr3_joint7`
- primary EE: `site:attachment_site`
- optional model: `/home/dummd/mujoco_models/mujoco_menagerie/franka_fr3_v2/fr3v2.xml`
- optional joints: `fr3v2_joint1..fr3v2_joint7`
- optional EE: `body:fr3v2_link8`

Core result:

- FR3-specific kinematic replay works.
- The replay uses FR3 joint names and an FR3-specific EE target.
- Outputs were generated in ignored `outputs/goal1_mujoco_replay/`.
- The FR3 primary replay produced a smooth 3D EE path from the all-q trajectory.

Caveat:

- This is still kinematic replay only.
- It does not prove ROS2 controller tracking, torque safety, or real robot readiness.

### GOAL1 C-M-V - FR3 MuJoCo video export

- script: `scripts/render_goal1_mujoco_video.py`
- doc: `docs/goal1_mujoco_video_plan.md`
- commit: `8cc5bd6 Add GOAL1 C-M-V FR3 MuJoCo video export`
- video output: `outputs/goal1_mujoco_video/goal1_allq_fr3_mujoco_video_640.mp4`

Important technical note:

- Initial 1280x720 rendering failed because the MuJoCo offscreen framebuffer default width was `640`.
- 640x480 rendering succeeded.
- Writer backend: `imageio-mp4`.

Core result:

- A complete FR3 kinematic replay video can be exported.
- The video provides a quick visual check of the all-q trajectory on the FR3 MuJoCo model.

Caveat:

- This video is visualization only.
- It is not dynamic simulation, not torque control, not actuator control, and not controller tracking validation.

## Key Technical Decisions

GOAL1 uses a joint-space all-q trajectory because the goal is to make all `q1..q7` move and make the end-effector path more complex. A Cartesian-only command path does not directly expose a streaming `q_des/dq_des/ddq_des` interface for this purpose.

The current repo controller path is not directly used because the main path is based on `TaskSpaceCommand` and Cartesian impedance control. `trajectory_publisher.py` publishes Cartesian desired state, and `cartesian_impedance.py` derives joint-space quantities internally through the Jacobian. That is different from replaying an external all-q joint trajectory.

Standalone MuJoCo was chosen before ROS2 fake hardware because it gives a simpler offline verification path. It does not touch the controller, launch files, hardware interface, GP compensation path, or torque command logic. It also gives EE path plots and video quickly.

The Panda result was kept because it validated the first replay pipeline with an available MuJoCo Menagerie model. It remains useful as a prototype, but it is not FR3-specific.

FR3 replay was needed because the intended real robot is FR3 / Research 3. A Panda-only replay is not enough for a model-specific FR3 claim, so the later replay uses FR3 MJCF files, FR3 joint names, and an FR3 EE target.

## Outputs to Inspect Manually

Important output locations:

- `outputs/goal1_joint_trajectory/`
- `outputs/goal1_mujoco_replay/`
- `outputs/goal1_mujoco_video/goal1_allq_fr3_mujoco_video_640.mp4`

These are ignored outputs and should not be committed. Video, plots, external MJCF models, and `.venv` contents should stay outside git history unless a separate review explicitly decides otherwise.

## How to Reproduce Key Offline Outputs

Use the existing `.venv` and run these only in a separately approved reproduction task:

Generate the all-q trajectory:

    .venv/bin/python scripts/generate_goal1_joint_trajectory.py --duration 20 --sample-rate 100 --include-jerk

Replay on the primary FR3 MuJoCo model:

    .venv/bin/python scripts/replay_goal1_trajectory_mujoco.py --model /home/dummd/mujoco_models/mujoco_menagerie/franka_fr3/fr3.xml --joint-names fr3_joint1,fr3_joint2,fr3_joint3,fr3_joint4,fr3_joint5,fr3_joint6,fr3_joint7 --ee-site attachment_site --prefix goal1_allq_mujoco_replay_fr3_attachment_site

Render the 640x480 FR3 video:

    .venv/bin/python scripts/render_goal1_mujoco_video.py --width 640 --height 480 --prefix goal1_allq_fr3_mujoco_video_640

Known Python dependencies:

- `mujoco`
- `imageio`
- `imageio-ffmpeg`
- `numpy`
- `matplotlib`

This note only documents the requirements. It does not suggest installing packages during this docs-only task.

## What Can Be Said in a Meeting

我已经完成了从 all-q trajectory generation 到 FR3 MuJoCo kinematic replay 和 video export 的 offline pipeline。这条 trajectory 让 `q1..q7` 都运动，其中 `q7` 是小幅运动。

在 FR3 MuJoCo model 上 replay 后，末端有平滑的 3D 空间运动，并且已经能导出 video，方便直观看 trajectory。

但目前这只是 `qpos + mj_forward` 的 kinematic replay，不是 ROS2 controller tracking，不代表真机安全，也没有 GP-on。下一步可以选择继续 polish visualization，或者评估 ROS2 fake hardware / replay，但那需要单独的 controller / replay / safety review。

## Next Steps

Recommended immediate next step:

- Inspect the FR3 video manually.
- Optionally create a short presentation/demo note.

Possible later steps:

- Add video timestamp or camera customization.
- Add EE trace overlay.
- Compare Panda vs FR3 qualitatively.
- Only later consider ROS2 fake hardware / replay.
- Do not run the real robot until a separate controller/replay/safety review is completed.

## Caveats / Non-Claims

Do not claim:

- real robot validated
- controller can track this trajectory
- torque safety
- actuator-control safety
- GP compensation works for this trajectory
- repeated robust validation
- ROS2 replay readiness
- direct Franka execution readiness

