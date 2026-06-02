# GOAL1 O Fake/No-Motion Staged Launch Plan

## Purpose

GOAL1 O exists because the previous real-robot direct effort route is stopped. The known dependency loop is:

`cpp_relayer active -> /state_parameter -> cartesian_impedance computes tau -> /effort_command -> cpp_relayer activation gate`

The fake/no-motion prototype validates whether the Cartesian launch ordering can be represented cleanly without connecting to Franka, without activating `cpp_relayer`, and without treating direct effort replay as a real-robot route.

## What This Validates

- A staged launch shape can be expressed without including `franka.launch.py`.
- A mock `/state_parameter` publisher can provide the fields consumed by `cartesian_impedance.py`.
- `trajectory_publisher` can expose `/joint_position_adjust`, `/task_space_command`, `/data_recording_enabled`, and `/future_task_space` in the fake graph.
- `cartesian_impedance` can be started with GP prediction, online update, and compensation all disabled for offline topic-graph inspection.
- Topic endpoints can be inspected in an offline WSL shell before any lab-side discussion.

## What This Does Not Validate

- Real Franka readiness.
- Real `franka_hardware` timing.
- Real `cpp_relayer` activation behavior.
- Real effort interface claiming.
- Torque safety under motion.
- Tracking quality, vibration, reflex stop risk, or `communication_constraints_violation` absence.
- Any GP-on behavior.

Passing this fake/no-motion prototype does not authorize real robot motion.

## Forbidden Real-Robot Actions

Do not use GOAL1 O to justify:

- running real `franka.launch.py robot_ip:=...`
- activating or deactivating real `ros2 control` controllers
- spawning `cpp_relayer` against real hardware
- publishing real `/effort_command` in the lab environment
- running `goal1_joint_space_replay.py` final effort replay on the real robot
- enabling GP compensation on the real robot
- changing `cartesian_impedance.py`, `cpp_relayer.cpp`, `franka_hardware`, or real controller safety defaults

## Implemented Fake/No-Motion Flow

File:

- `new_structure/py_controllers/launch/goal1_fake_cartesian_staged_launch.py`

Staged order:

1. Start `goal1_mock_state_parameter_publisher`.
2. After `trajectory_start_delay_sec`, start `trajectory_publisher`.
3. After `cartesian_start_delay_sec`, start `cartesian_impedance`.

The launch file intentionally does not:

- declare `robot_ip`
- include `new_bringup/launch/franka.launch.py`
- start `gp_server`
- start `cpp_relayer`
- start `controller_manager`
- activate controllers
- provide any hardware connection path

`cartesian_impedance` is launched with:

- `gp_prediction_enabled=false`
- `gp_online_update_enabled=false`
- `gp_compensation_enabled=false`
- `gp_compensation_source=local`
- `gp_compensation_scale=0.1`
- `gp_compensation_clip_nm=0.5`

`trajectory_publisher` is launched with:

- `trajectory_mode=planar_circle` by default
- `z_amplitude=0.0`

## Mock State Publisher

File:

- `new_structure/py_controllers/py_controllers/goal1_mock_state_parameter_publisher.py`

It publishes `custom_msgs/msg/StateParameter` to `/state_parameter` with finite mock values:

- `position`: default 7-joint startup pose
- `velocity`: zeros
- `effort_measured`: zeros
- `gravity`: zeros
- `o_t_f`: identity pose with translation `(0.35, 0.0, 0.65)`
- `mass`: 7x7 identity matrix
- `coriolis`: zeros
- `zero_jacobian_flange`: simple finite 6x7 mock Jacobian

It does not publish `/effort_command`, does not start controllers, and does not connect to any hardware interface.

## Topic Boundary

When `cartesian_impedance` is started, it creates a publisher on `/effort_command` because that is part of the existing controller node. In GOAL1 O this is acceptable only because the fake launch has no `cpp_relayer`, no real controller, and no real hardware consumer.

This means GOAL1 O can inspect topic endpoints, but it must not be copied into a lab command sequence with `cpp_relayer` or `franka.launch.py` added around it.

## Suggested Offline Runbook

Use a clean WSL shell, source only the required ROS2/workspace setup, then inspect the fake graph. Example commands are documentation only; do not run them on the real robot machine:

1. `source /opt/ros/humble/setup.bash`
2. `cd /home/feizao/projects/gp_torque_compensation`
3. `colcon build --packages-select custom_msgs py_controllers`
4. `source install/setup.bash`
5. `ros2 launch py_controllers goal1_fake_cartesian_staged_launch.py`
6. In a second WSL shell, inspect:
   - `ros2 topic list`
   - `ros2 topic info /state_parameter`
   - `ros2 topic info /task_space_command`
   - `ros2 topic info /data_recording_enabled`
   - `ros2 topic info /effort_command`

Expected offline endpoint shape:

- `/state_parameter`: publisher is `goal1_mock_state_parameter_publisher`; subscribers include `trajectory_publisher` and `cartesian_impedance`.
- `/task_space_command`: publisher is `trajectory_publisher`; subscriber is `cartesian_impedance`.
- `/effort_command`: publisher may be `cartesian_impedance`; there should be no `cpp_relayer` subscriber in this prototype.

## GenericSystem Fake Hardware Limitation

GOAL1 O does not rely on ROS2 `GenericSystem` fake hardware. That is intentional: fake hardware can help topic/controller integration, but it does not reproduce Franka FCI timing, effort-interface claiming, reflex behavior, or `communication_constraints_violation` conditions. A future fake-hardware test must still be treated as offline-only unless separately reviewed.

## Relation To The cpp_relayer Blocker

GOAL1 O avoids the blocker instead of solving it. The prototype checks whether mock `/state_parameter` can break the Cartesian dependency loop for offline sequencing. It does not prove that `cpp_relayer` can be activated safely, and it does not validate the fresh-command gate on hardware.

## Future Real Lab Gates

Before any future lab attempt, require a separate review that confirms:

- no direct final effort replay route is being used
- `cpp_relayer` activation risk is resolved or explicitly bypassed
- no-GP route is validated before GP-on
- GP compensation remains disabled by default
- any compensation has explicit scale and clip
- operator stop plan is clear
- Franka web interface unlock / FCI activation / shutdown workflow is followed

Passing fake/no-motion does not authorize real robot motion.
