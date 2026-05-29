# GOAL1 C-M-FR3 MuJoCo Kinematic Replay Summary

## Purpose

GOAL1 C-M-FR3 checks the GOAL1 B all-q joint-space trajectory with an FR3-specific MuJoCo MJCF model. The previous Panda replay remains useful as a kinematic replay prototype, but it is not FR3-specific.

This task only generates standalone MuJoCo kinematic replay outputs. It does not connect to ROS2, does not run a controller, does not command a real Franka robot, and does not enable GP.

## Primary FR3 Replay

- csv_path: `outputs/goal1_joint_trajectory/goal1_allq_conservative.csv`
- model_path: `/home/dummd/mujoco_models/mujoco_menagerie/franka_fr3/fr3.xml`
- selected_joints: `fr3_joint1..fr3_joint7`
- selected_EE: `site:attachment_site`
- model sizes: `nq=7`, `nv=7`, `nu=7`, `nbody=10`, `nsite=1`
- samples: `2001`
- time range: `0.0` to `20.0` s
- median dt: `0.009999999999999787` s

EE xyz summary:

- x_min: `0.2676955906699702` m
- x_max: `0.3272372210038226` m
- x_range: `0.0595416303338524` m
- y_min: `-0.11104437390094282` m
- y_max: `0.12236157230763306` m
- y_range: `0.23340594620857588` m
- z_min: `0.5345673815384899` m
- z_max: `0.6313430136418708` m
- z_range: `0.09677563210338092` m

Generated outputs:

- `outputs/goal1_mujoco_replay/goal1_allq_mujoco_replay_fr3_attachment_site_ee_path.csv`
- `outputs/goal1_mujoco_replay/goal1_allq_mujoco_replay_fr3_attachment_site_summary.json`
- `outputs/goal1_mujoco_replay/goal1_allq_mujoco_replay_fr3_attachment_site_summary.md`
- `outputs/goal1_mujoco_replay/goal1_allq_mujoco_replay_fr3_attachment_site_ee_xyz.png`
- `outputs/goal1_mujoco_replay/goal1_allq_mujoco_replay_fr3_attachment_site_ee_path_3d.png`

## Optional FR3 v2 Replay

- model_path: `/home/dummd/mujoco_models/mujoco_menagerie/franka_fr3_v2/fr3v2.xml`
- selected_joints: `fr3v2_joint1..fr3v2_joint7`
- selected_EE: `body:fr3v2_link8`
- model sizes: `nq=7`, `nv=7`, `nu=7`, `nbody=11`, `nsite=0`
- result: replay completed

The FR3 v2 replay produced the same EE xyz range for this selected body and trajectory:

- x_range: `0.0595416303338524` m
- y_range: `0.23340594620857588` m
- z_range: `0.09677563210338092` m

Generated outputs:

- `outputs/goal1_mujoco_replay/goal1_allq_mujoco_replay_fr3v2_link8_ee_path.csv`
- `outputs/goal1_mujoco_replay/goal1_allq_mujoco_replay_fr3v2_link8_summary.json`
- `outputs/goal1_mujoco_replay/goal1_allq_mujoco_replay_fr3v2_link8_summary.md`
- `outputs/goal1_mujoco_replay/goal1_allq_mujoco_replay_fr3v2_link8_ee_xyz.png`
- `outputs/goal1_mujoco_replay/goal1_allq_mujoco_replay_fr3v2_link8_ee_path_3d.png`

## Replay Method

The replay script loads the offline CSV, maps `joint_pos_1..7` into the selected MuJoCo joints, writes `data.qpos`, and calls `mj_forward` for each sample. It records the selected EE body or site xyz position.

Important boundaries:

- kinematic replay only
- uses `mj_forward`
- no `mj_step`
- no `ctrl`
- no torque control
- no actuator control
- no ROS2 integration
- no real robot validation
- no GP-on
- no controller tracking claim
- no hardware safety claim

## Panda / FR3 Wording Boundary

- Panda result: `Panda model kinematic replay prototype, not FR3-specific`.
- FR3 result: FR3-specific kinematic replay using `/home/dummd/mujoco_models/mujoco_menagerie/franka_fr3/fr3.xml`, `fr3_joint1..fr3_joint7`, and `attachment_site`.
- FR3 v2 result: optional FR3 v2 kinematic replay using `/home/dummd/mujoco_models/mujoco_menagerie/franka_fr3_v2/fr3v2.xml`, `fr3v2_joint1..fr3v2_joint7`, and `fr3v2_link8`.

## Suggested Next Steps

- Optionally add FR3 video export from the same kinematic replay outputs.
- Compare Panda vs FR3 qualitative path differences.
- Keep ROS2 fake hardware, controller replay, and real robot validation as later separately reviewed tasks.
