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

Polished short test render with the default stable 640x480 size, `iso` camera preset, and timestamp overlay:

    .venv/bin/python scripts/render_goal1_mujoco_video.py \
      --width 640 \
      --height 480 \
      --max-frames 120 \
      --camera-preset iso \
      --show-timestamp \
      --prefix goal1_allq_fr3_mujoco_video_test_polished

High-resolution polished short test render in a headless shell:

    MUJOCO_GL=egl .venv/bin/python scripts/render_goal1_mujoco_video.py \
      --width 1280 \
      --height 720 \
      --auto-offscreen-xml \
      --offscreen-width 1280 \
      --offscreen-height 720 \
      --max-frames 120 \
      --camera-preset close_iso \
      --show-timestamp \
      --start-time 8.0 \
      --prefix goal1_allq_fr3_mujoco_video_test_polished_720p

High-resolution polished short test render with EE trace overlay plus trace CSV/PNG outputs:

    MUJOCO_GL=egl .venv/bin/python scripts/render_goal1_mujoco_video.py \
      --width 1280 \
      --height 720 \
      --auto-offscreen-xml \
      --offscreen-width 1280 \
      --offscreen-height 720 \
      --max-frames 120 \
      --camera-preset close_iso \
      --show-timestamp \
      --show-ee-trace \
      --start-time 8.0 \
      --prefix goal1_allq_fr3_mujoco_video_test_trace_720p

Full render:

    .venv/bin/python scripts/render_goal1_mujoco_video.py --prefix goal1_allq_fr3_mujoco_video

Explicit FR3 inputs:

    .venv/bin/python scripts/render_goal1_mujoco_video.py \
      --csv outputs/goal1_joint_trajectory/goal1_allq_conservative.csv \
      --model /home/dummd/mujoco_models/mujoco_menagerie/franka_fr3/fr3.xml \
      --joint-names fr3_joint1,fr3_joint2,fr3_joint3,fr3_joint4,fr3_joint5,fr3_joint6,fr3_joint7

Optional controls:

- `--width 640`
- `--height 480`
- `--fps 30`
- `--playback-speed 1.0`
- `--start-time 0.0`
- `--max-frames 120`
- `--frame-stride 2`
- `--auto-offscreen-xml`
- `--offscreen-width 1280`
- `--offscreen-height 720`
- `--temp-model-dir outputs/goal1_mujoco_video/temp_models`
- `--camera CAMERA_NAME`
- `--camera-preset default|front|side|top|iso|close_iso`
- `--show-timestamp`
- `--timestamp-format 't={time:.2f}s'`
- `--ee-site attachment_site`
- `--show-ee-trace`
- `--trace-length 0`
- `--trace-radius 0.02`
- `--trace-color 0.0,1.0,0.1,1.0`
- `--output-format mp4`

## GOAL1 C-M-V2 Video Polish

GOAL1 C-M-V2 keeps the same offline FR3 kinematic replay path and only polishes the exported video for human review.

Camera controls:

- `--camera-preset iso` is the default preset.
- Other presets are `default`, `front`, `side`, `top`, and `close_iso`.
- `close_iso` is a closer free-camera preset intended for human inspection videos where the FR3 should occupy more of the frame.
- `--camera CAMERA_NAME` still selects a model-defined MuJoCo camera and overrides `--camera-preset`.
- Presets use MuJoCo free camera configuration through `Renderer.update_scene`.

Timestamp overlay:

- `--show-timestamp` draws a small overlay in the upper-left corner of the rendered RGB frame.
- The overlay includes the shorter title `GOAL1 FR3 replay` and the formatted replay time.
- `--timestamp-format` defaults to `t={time:.2f}s`.
- The overlay uses a positive margin and bounded background box so the title/timestamp should not start outside the image.
- If the text overlay dependency or format fails, video rendering continues and the summary records the fallback.

EE trace overlay:

- `--show-ee-trace` records the history of the selected `--ee-site`, defaulting to `attachment_site`.
- The script attempts to draw the trace in the MuJoCo scene using bright marker geoms.
- The script also draws a visible lower-right image overlay panel from the EE x/y history, so the trace-enabled video should be visibly different from the no-trace video.
- `--trace-length 0` shows all rendered history; a positive value shows only the latest N rendered frames.
- `--trace-radius` controls marker size in meters.
- `--trace-color` accepts comma-separated RGB or RGBA values in `[0.0, 1.0]`.
- When EE trace is enabled, the script also writes `<prefix>_ee_trace.csv` and `<prefix>_ee_trace_3d.png` so the trace remains inspectable even if video overlay support is limited in a specific renderer environment.

Start time:

- `--start-time` selects the CSV source time where rendering starts.
- This is useful for short review videos, for example `--start-time 8.0 --max-frames 120`.

High-resolution offscreen mode:

- `--auto-offscreen-xml` writes an ignored temporary patched MJCF into `outputs/goal1_mujoco_video/temp_models/`.
- The original Menagerie XML is not modified.
- The patched XML adds or updates `<visual><global offwidth="..." offheight="..."/></visual>`.
- Relative asset paths are preserved by converting the temporary XML `compiler meshdir` to the original model directory or original `meshdir` as an absolute path.
- The summary records the original model path, effective model path, offscreen size, and whether a patched XML was created.
- Use this mode for 1280x720 rendering instead of editing `/home/dummd/mujoco_models/mujoco_menagerie/franka_fr3/fr3.xml`.

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

For GOAL1 C-M-V2, the summary also records camera preset, timestamp overlay state, EE trace overlay state, EE trace CSV/PNG fallback outputs, width, height, offscreen width/height, start time, effective model path, FPS, rendered frame count, and writer backend.

## Rendering Caveats

Headless MuJoCo rendering can fail if the OpenGL backend is not available. If rendering fails with an OpenGL or display-related error, run from an environment with a working display or explicitly test one of:

- `MUJOCO_GL=egl`
- `MUJOCO_GL=osmesa`

Do not modify shell profile or system OpenGL configuration without a separate review.

The original Menagerie FR3 XML may only provide a 640-wide default offscreen framebuffer. 1280x720 can fail with an error like `Image width 1280 > framebuffer width 640`. Use `--auto-offscreen-xml --offscreen-width 1280 --offscreen-height 720` to generate an ignored temporary MJCF for high-resolution rendering. Do not edit the external Menagerie XML for this GOAL1 C-M-V2 polish step.

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
