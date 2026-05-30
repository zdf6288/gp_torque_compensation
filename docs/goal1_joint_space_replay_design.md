# GOAL1 H Joint-Space Torque Replay Skeleton

## Purpose

GOAL1 H adds a default-disabled Python skeleton for reviewing a possible joint-space torque replay route:

- `new_structure/py_controllers/py_controllers/goal1_joint_space_replay.py`
- `new_structure/py_controllers/launch/goal1_joint_space_replay_launch.py`

The first version is for guarded CSV validation and a heavily guarded optional `/effort_command` skeleton only. It is not a real-robot safety proof.

## Why This Does Not Modify `cartesian_impedance.py`

The current real-robot main path is Cartesian impedance:

`trajectory_publisher.py -> /task_space_command -> cartesian_impedance.py -> /effort_command -> cpp_relayer -> franka_hardware effort interface`

The GOAL1 D CSV is a joint-space trajectory. Feeding it into `cartesian_impedance.py` would mix a joint-space replay experiment into the existing Cartesian impedance and GP compensation path. GOAL1 H therefore keeps the path independent and default-disabled.

## Default-Disabled Behavior

The defaults are intentionally no-motion and no-effort:

- `dry_run=true`
- `start_replay=false`
- `publish_effort=false`
- `max_duration=3.0`

With `dry_run=true` or `publish_effort=false`, the script validates the CSV and exits without publishing torque. It prints the CSV path, selected point count, selected source time range, first/last `q`, max absolute `dq`, `ddq`, optional jerk, and the guard states.

## CSV Validation

The dry-run path checks:

- required columns: `time`, `joint_pos_1..7`, `joint_vel_1..7`, `joint_acc_1..7`
- optional complete columns: `joint_jerk_1..7`
- finite numeric values
- strictly increasing time
- selected segment from `start_time` to `start_time + max_duration`

The pure CLI dry-run path does not need to initialize a ROS graph.

## Optional Publish Skeleton

Publishing can only be attempted when all three guards are explicitly changed:

- `dry_run=false`
- `start_replay=true`
- `publish_effort=true`

Even then, the first version only uses joint PD:

`tau = kp * (q_des - q) + kd * (dq_des - dq)`

It does not use GP, GP compensation, `ddq_des` feedforward, inverse dynamics, model terms, or a full 60s run by default.

## Refusal Conditions

The publish skeleton refuses before sending any torque if:

- no fresh `/state_parameter` message arrives within `state_timeout_sec`
- current `q` differs from the first CSV `q` by more than `start_position_tolerance_rad`
- state values are malformed or non-finite

During replay, stale state or command timer overrun stops further publishing.

## Torque Limits

The skeleton applies:

- per-joint `torque_clip_nm`
- per-joint `torque_rate_limit_nm_per_s`

The defaults are small and conservative. They are not a safety certification.

## Shutdown Caveat

This skeleton does not implement a complex shutdown torque command. On selected-segment completion or refusal, it stops publishing. The downstream `cpp_relayer` last-command behavior must be reviewed on lab Linux before any real-robot attempt.

## Launch Scope

`goal1_joint_space_replay_launch.py` is node-only. It does not include `new_bringup/launch/franka.launch.py`, does not start `cartesian_impedance.py`, does not start `trajectory_publisher.py`, does not start `gp_server.py`, and does not spawn `cpp_relayer`.

This avoids accidentally starting the existing real-robot path from a review launch. The exposed `robot_ip` argument is kept only for interface review symmetry and is not used to bring up hardware.

## Lab Safety Requirement

WSL implementation and static validation do not prove real robot safety. Lab Linux validation, a separate safety review, and explicit operator approval are required before any hardware-side test.

Do not run `publish_effort=true` without separate lab-side safety review.
