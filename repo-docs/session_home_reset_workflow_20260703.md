# Session Home and Session-Relative Trajectory Workflow (2026-07-03)

## Why home-only reset was not enough

The first `session_home` patch solved only the reset target: startup and
post-run return could use the captured safe EE pose, but the formal trajectory
still started from the old absolute world point. That is home-only reset:

- startup target = current captured `session_home`;
- post-run return target = the same `session_home`;
- trajectory start and `circle_center` = old fixed absolute geometry.

That mode can still fail a real matrix run because the robot returns to the
captured home, then `trajectory_publisher` asks it to transition toward the old
fixed trajectory start. If that old start is too far away, the start-distance
guard still refuses or the transition becomes unnecessarily large.

## `fixed_absolute`

`trajectory_reference_mode=fixed_absolute` is the backward-compatible default.

- nominal fixed start remains an absolute world coordinate;
- nominal trajectory start remains an absolute world coordinate;
- nominal `circle_center` remains an absolute world coordinate;
- existing manual commands continue to work when no new session-relative
  arguments are set.

`session_home_mode=capture_first/load` can still be used in this mode, but it is
only a startup/return reset target. It does not move the formal trajectory.

## `session_relative`

`trajectory_reference_mode=session_relative` is the recommended mode for
continuous real-robot compensation matrices.

The first run uses `session_home_mode=capture_first`:

- capture the current stable, safe EE pose;
- store it as `session_trajectory_start`;
- treat nominal trajectory geometry only as a shape template;
- compute `anchor_delta = session_trajectory_start - nominal_trajectory_start`;
- compute `shifted_circle_center = nominal_circle_center + anchor_delta`;
- save one shared JSON anchor.

Later runs use `session_home_mode=load`:

- controller loads the same JSON and uses `session_trajectory_start` for
  startup and post-run return;
- `trajectory_publisher` loads the same JSON and uses the shifted start/center;
- local, cloud, combined, and triple runs can compare on the same shifted
  trajectory instead of recapturing a new one.

Do not capture again for every source/scale group. Re-capturing per group would
change the global trajectory translation and make tracking-error rows no longer
directly comparable. Capture once for the session, then load the same anchor.

## Why not `anchor_trajectory_start_to_current_pose`

`anchor_trajectory_start_to_current_pose=true` shifts the trajectory based on the
measured pose at enable time. That is useful for a one-off drift workaround, but
it is the wrong mechanism for a formal matrix because every run can get a
slightly different trajectory. In `session_relative`, the only allowed shift
source is the saved anchor JSON, and the publisher refuses to start if both
mechanisms are enabled.

## Anchor JSON

The anchor JSON contains at least:

- `version`, `created_at`, `source`, and `notes`;
- `trajectory_reference_mode=session_relative`;
- `ee_pose_xyz` and `session_trajectory_start_xyz`;
- `nominal_trajectory_start_xyz`;
- `nominal_circle_center_xyz`;
- `shifted_circle_center_xyz`;
- `anchor_delta_xyz`;
- `nominal_fixed_start_xyz`;
- `q_at_capture` when available.

Load paths fail closed if the JSON is missing, unparsable, not
`session_relative`, has non-finite 3-vectors, has inconsistent
`session_start = nominal_start + anchor_delta`, has inconsistent
`shifted_center = nominal_center + anchor_delta`, exceeds the anchor-delta
limit, or does not match the current launch nominal geometry.

## First capture run

Terminal C only, after Terminal A/B are already in the required robot state:

```
scripts/run_goal12_python_only_f50_scale1_manual_matrix_20260703.sh \
  --source local \
  --scale 0.25 \
  --strict-start \
  --session-relative \
  --session-home-mode capture_first
```

`--session-relative` sets the recommended repeated-matrix safety defaults:
`trajectory_reference_mode=session_relative`,
`post_run_return_to_session_home_enabled=true`,
`normal_run_start_gate_enabled=true`,
`normal_run_start_refuse_m=0.150`, and
`emergency_return_start_refuse_m=0.300`.

## Later load runs

Reuse the JSON produced by the first run:

```
scripts/run_goal12_python_only_f50_scale1_manual_matrix_20260703.sh \
  --source cloud \
  --scale 0.25 \
  --strict-start \
  --session-relative \
  --session-home-mode load \
  --session-home-path outputs/manual_compensation/<stamp>/session_home.json
```

Run `--source combined` or `--source triple` the same way to keep the identical
shifted trajectory.

Within one runner invocation, `capture_first` applies to the first case only;
later cases automatically switch to `load` and reuse the same JSON path.

## Data labeling

Do not mix old `fixed_absolute` CSVs with `session_relative` CSVs without
marking the mode. `session_relative` deliberately changes the global trajectory
translation while preserving the shape, radius, frequency, and phase. Tracking
metrics are comparable within the same anchor session, but not silently
interchangeable with older fixed-absolute runs.

No GP math, fusion source selection, `cpp_relayer`, bringup, controller manager,
or Franka hardware interface is part of this workflow change.
