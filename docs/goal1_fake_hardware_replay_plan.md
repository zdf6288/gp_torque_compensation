# GOAL1 E1 Fake-only ROS2 Joint Trajectory Replay Path

## Goal

GOAL1 E1 adds a local WSL fake-only ROS2 path for replaying the GOAL1 D CSV through a standard `joint_trajectory_controller` and visualizing the result in RViz.

This is a wiring and replay-path check before any real robot discussion. It does not validate the lab real-robot Linux environment, real Franka hardware behavior, torque safety, or GP compensation.

## Scope and Caveats

This path is fake-only:

- no real robot
- no GP
- no torque command logic
- no `/effort_command`
- no `cartesian_impedance.py`
- no `trajectory_publisher.py`
- no main `controllers.yaml`
- no main `franka.launch.py`

The fake launch uses the repository's current Panda-based `franka_description` through `panda_arm.urdf.xacro` and `panda_joint1..panda_joint7`. This is for ROS2 wiring, fake hardware, and RViz replay-path validation only. It is not an FR3-specific geometry validation. FR3-specific geometry visualization was already checked through the MuJoCo GOAL1 pipeline.

The current GOAL1 CSV is joint-space data (`time`, `joint_pos_1..7`). The main project controller path is Cartesian-only: `trajectory_publisher.py` publishes Cartesian desired state and `cartesian_impedance.py` handles the torque controller path. That path cannot directly consume the GOAL1 all-q CSV, so GOAL1 E1 stays isolated.

## Files

- `new_structure/new_bringup/config/goal1_fake_joint_trajectory_controllers.yaml`
- `new_structure/new_bringup/launch/goal1_fake_joint_trajectory_replay.launch.py`
- `new_structure/py_controllers/py_controllers/goal1_csv_joint_trajectory_replay.py`

The fake controller config defines `joint_state_broadcaster` and `goal1_joint_trajectory_controller`. The trajectory controller uses the `position` command interface and `position` / `velocity` state interfaces for `panda_joint1..panda_joint7`.

The fake launch starts `robot_state_publisher`, `ros2_control_node`, `joint_state_broadcaster`, and `goal1_joint_trajectory_controller`. It does not add `joint_state_publisher` by default because `joint_state_broadcaster` should provide `/joint_states` from ros2_control fake hardware.

## Replay Node

The replay node reads:

- default CSV: `outputs/goal1_joint_trajectory/goal1_allq_spatial_rich_60s_50hz.csv`
- columns: `time`, `joint_pos_1`, ..., `joint_pos_7`

Important parameters:

- `csv_path`
- `joint_names`
- `controller_name`
- `topic_or_action`
- `max_duration`
- `start_time`
- `time_scale`
- `dry_run`
- `hold_final`
- `prepend_current_state`
- `ramp_duration`

Defaults are intentionally conservative:

- `dry_run=true`
- `max_duration=5.0`
- `controller_name=goal1_joint_trajectory_controller`
- `joint_names=panda_joint1,...,panda_joint7`
- `topic_or_action=topic`

Dry-run mode only reads and checks the CSV, then prints columns, first/last q, selected duration, point count, and joint names. It does not publish a trajectory and does not send a goal.

## Start Pose Caveat

Fake hardware replay can tolerate an initial mismatch for wiring visualization, but this does not make the trajectory safe for a real robot. Before any real robot launch, a separate safety review must implement and validate current-state matching, ramp-in behavior, limits, collision/workspace checks, stop procedure, and controller tracking behavior.

In this v1, `prepend_current_state` and `ramp_duration` are placeholders/caveats for the fake-only path. They must not be treated as real robot safety features.

## Example Commands

Codex did not run `ros2 launch` or `ros2 run` during this implementation. The following commands are for later manual review only.

Pure dry-run without ROS graph:

    python3 new_structure/py_controllers/py_controllers/goal1_csv_joint_trajectory_replay.py --dry-run --csv-path outputs/goal1_joint_trajectory/goal1_allq_spatial_rich_60s_50hz.csv --max-duration 5.0

Fake hardware launch for manual local WSL review only:

    ros2 launch new_bringup goal1_fake_joint_trajectory_replay.launch.py

Start replay node in launch, still dry-run by default:

    ros2 launch new_bringup goal1_fake_joint_trajectory_replay.launch.py start_replay:=true dry_run:=true max_duration:=5.0

Publish a short fake-only trajectory segment only after manual review:

    ros2 launch new_bringup goal1_fake_joint_trajectory_replay.launch.py start_replay:=true dry_run:=false max_duration:=5.0

These commands are not real robot commands. Do not use this path as evidence of lab Linux validation, FR3 geometry validation, torque safety, GP readiness, or hardware safety.
