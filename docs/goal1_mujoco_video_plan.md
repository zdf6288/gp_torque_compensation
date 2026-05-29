# GOAL1 C-M-V MuJoCo FR3 Video Export Plan

## Goal

GOAL1 C-M-V adds a standalone MuJoCo video export for the GOAL1 B all-q joint-space trajectory. The script replays `joint_pos_1..7` from the offline CSV in an FR3 MuJoCo model, renders frames with the MuJoCo Python renderer, and writes a video plus summaries.

This is only a kinematic visualization step. It does not connect to ROS2, does not run a controller, does not command a real Franka robot, does not enable GP, and does not perform torque control.

## Why FR3 Instead Of Panda

The previous GOAL1 C-M replay used the external MuJoCo Menagerie Panda model as a prototype. That was useful for checking the replay pipeline, but it was not FR3-specific.

GOAL1 C-M-V defaults to the local MuJoCo Menagerie FR3 model so the rendered robot geometry and joint names match the intended FR3 replay target:

- `/home/dummd/mujoco_models/mujoco_menagerie/franka_fr3/fr3.xml`

The external MJCF assets remain outside this repository and are not committed.

## Inputs

Trajectory CSV:

- `outputs/goal1_joint_trajectory/goal1_allq_conservative.csv`

Default FR3 model:

- `/home/dummd/mujoco_models/mujoco_menagerie/franka_fr3/fr3.xml`

Default joints:

- `fr3_joint1,fr3_joint2,fr3_joint3,fr3_joint4,fr3_joint5,fr3_joint6,fr3_joint7`

## Replay Method

For each selected video frame, the script:

1. reads `time` and `joint_pos_1..7` from the CSV;
2. maps the selected joint names to MuJoCo `qpos` addresses;
3. writes `joint_pos_1..7` into `data.qpos`;
4. calls `mj_forward`;
5. renders the scene with `mujoco.Renderer`;
6. appends the RGB frame to the selected video writer.

Important boundaries:

- set `qpos`
- use `mj_forward`
- no `mj_step`
- no `ctrl`
- no actuator control
- no torque control
- no contact-control simulation

## CLI Usage

Dry-run validation:

    .venv/bin/python scripts/render_goal1_mujoco_video.py --dry-run --max-frames 120

List model names:

    .venv/bin/python scripts/render_goal1_mujoco_video.py --list-model-names

Short test render:

    .venv/bin/python scripts/render_goal1_mujoco_video.py --max-frames 120 --prefix goal1_allq_fr3_mujoco_video_test

Full render:

    .venv/bin/python scripts/render_goal1_mujoco_video.py --prefix goal1_allq_fr3_mujoco_video

Explicit FR3 inputs:

    .venv/bin/python scripts/render_goal1_mujoco_video.py \
      --csv outputs/goal1_joint_trajectory/goal1_allq_conservative.csv \
      --model /home/dummd/mujoco_models/mujoco_menagerie/franka_fr3/fr3.xml \
      --joint-names fr3_joint1,fr3_joint2,fr3_joint3,fr3_joint4,fr3_joint5,fr3_joint6,fr3_joint7

Optional controls:

- `--width 1280`
- `--height 720`
- `--fps 30`
- `--playback-speed 1.0`
- `--max-frames 120`
- `--frame-stride 2`
- `--camera CAMERA_NAME`
- `--output-format mp4`

## Outputs

Default output directory:

- `outputs/goal1_mujoco_video/`

Default full-render files:

- `goal1_allq_fr3_mujoco_video.mp4`
- `goal1_allq_fr3_mujoco_video_summary.json`
- `goal1_allq_fr3_mujoco_video_summary.md`

Short test-render files use the selected prefix, for example:

- `goal1_allq_fr3_mujoco_video_test.mp4`
- `goal1_allq_fr3_mujoco_video_test_summary.json`
- `goal1_allq_fr3_mujoco_video_test_summary.md`

The summary records input paths, selected joints, MuJoCo model sizes, video settings, frame counts, writer backend, writer package availability, rendering success, and caveats.

## Rendering Caveats

Headless MuJoCo rendering can fail if the OpenGL backend is not available. If rendering fails with an OpenGL or display-related error, run from an environment with a working display or explicitly test one of:

- `MUJOCO_GL=egl`
- `MUJOCO_GL=osmesa`

Do not modify shell profile or system OpenGL configuration without a separate review.

Video writing requires an available writer package. The script checks:

- `imageio`
- `imageio_ffmpeg`
- `cv2`

For MP4, install either `imageio` plus `imageio-ffmpeg`, or `opencv-python`, only after explicit approval:

- `.venv/bin/python -m pip install imageio imageio-ffmpeg`
- `.venv/bin/python -m pip install opencv-python`

## Explicit Non-Goals

- no ROS2 integration
- no real robot run
- no GP-on
- no torque control
- no actuator control
- no controller tracking validation
- no hardware safety validation
- no Panda default
- no xacro to MJCF conversion

## Future Extensions

- add camera customization presets;
- add EE trace overlay, if needed;
- add Panda vs FR3 side-by-side video only after the current FR3 video export is stable.
