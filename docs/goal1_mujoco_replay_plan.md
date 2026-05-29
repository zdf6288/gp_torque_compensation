# GOAL1 C-M MuJoCo Kinematic Replay Plan

## Goal

GOAL1 C-M adds a standalone MuJoCo kinematic replay path for the GOAL1 B all-q joint-space trajectory. The first version replays `joint_pos_1..7` from the offline CSV, updates the MuJoCo arm joint `qpos`, calls `mj_forward`, and records the selected end-effector body or site xyz path.

This is an offline inspection tool only. It does not connect to ROS2, does not run a controller, does not command Franka, and does not enable GP.

## Model Source

The first implementation uses an external MuJoCo Menagerie Panda model:

- `/home/dummd/mujoco_models/mujoco_menagerie/franka_emika_panda/panda.xml`

The model stays outside this repo because it is an external asset with its own source and license. The replay script accepts the model path through `--model`, so the repo does not need to vendor or commit MJCF files.

Current confirmed model properties:

- `nq: 9`
- `nv: 9`
- `nu: 8`
- `nbody: 12`
- `nsite: 0`
- arm joints: `joint1,joint2,joint3,joint4,joint5,joint6,joint7`
- body candidates: `hand`, `link7`

Because this model has `nsite=0`, the first version should use `--ee-body hand` by default. `--ee-body link7` is useful for comparing flange-side and hand-side paths. `--ee-site` is retained for future models that provide a TCP site.

## Why Kinematic Replay First

The first version intentionally uses kinematic replay instead of dynamic torque simulation:

- GOAL1 C-M needs to inspect whether the all-q CSV produces a complex end-effector path.
- The current repo control path is Cartesian command based, not streaming all-q trajectory replay.
- MuJoCo actuator mapping, torque gains, contact behavior, and controller tracking are separate problems.
- Calling `mj_forward` after setting `qpos` is sufficient for forward-kinematic path inspection.

The script must not call `mj_step`, set `ctrl`, simulate torque control, or claim tracking or hardware safety.

## CLI Usage

Default hand replay:

```bash
.venv/bin/python scripts/replay_goal1_trajectory_mujoco.py --ee-body hand
```

Compare `link7`:

```bash
.venv/bin/python scripts/replay_goal1_trajectory_mujoco.py --ee-body link7 --prefix goal1_allq_mujoco_replay_link7
```

List model names without replay:

```bash
.venv/bin/python scripts/replay_goal1_trajectory_mujoco.py --list-model-names
```

Use an explicit model path:

```bash
.venv/bin/python scripts/replay_goal1_trajectory_mujoco.py \
  --csv outputs/goal1_joint_trajectory/goal1_allq_conservative.csv \
  --model /home/dummd/mujoco_models/mujoco_menagerie/franka_emika_panda/panda.xml \
  --ee-body hand
```

## Outputs

Default output directory:

- `outputs/goal1_mujoco_replay/`

Default files:

- `goal1_allq_mujoco_replay_ee_path.csv`
- `goal1_allq_mujoco_replay_summary.json`
- `goal1_allq_mujoco_replay_summary.md`
- `goal1_allq_mujoco_replay_ee_xyz.png`
- `goal1_allq_mujoco_replay_ee_path_3d.png`

The EE path CSV contains:

- `time`
- `ee_x`
- `ee_y`
- `ee_z`
- `joint_pos_1..7`

The summary records the CSV path, model path, model sizes, selected joints, selected EE body or site, sample count, time range, median dt, EE xyz min/max/range, output paths, and caveats.

## Caveats

- MuJoCo standalone kinematic replay only.
- No torque control.
- No actuator control.
- No ROS2 integration.
- No real robot validation.
- No GP-on.
- No guarantee of controller tracking.
- No guarantee of hardware safety.

## Future Extensions

- Compare `hand` vs `link7` as a standard GOAL1 C-M check.
- Use `--ee-site` when a model provides a TCP site.
- Add an optional viewer only after offline path generation is stable.
- Consider ROS2 fake hardware or controller replay only in a later, separately reviewed task.
