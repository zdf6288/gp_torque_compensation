# Session-home reset and post-run cleanup for GP split-run experiments (2026-07-03)

## Why session_home was added

Repeated combined/scale1 matrix runs do not naturally end at the same pose. The
next run then starts too far from the hardcoded `fixed_start=[0.35, 0.0, 0.65]`
or from an unsafe posture. A real failure case started at
`distance_to_fixed_start≈0.2169 m`: above `startup_distance_warn_m=0.100` but
below `startup_distance_refuse_m=0.300`, so the run continued and later hit a
`joint_velocity_violation` reflex. Manual re-dragging before every run does not
scale for matrix runs.

`session_home` is a per-session, validated safe end-effector pose that:

- replaces the hardcoded fixed start **only** as the startup interpolation
  target and post-run return target (trajectory geometry is unchanged);
- is captured once under validation gates (position stability over N samples,
  z-range, max distance from the nominal fixed start) or loaded from JSON;
- enables a stricter three-tier run-start gate and an automatic slow no-GP
  return to home after every run.

## Runtime pieces

- `cartesian_impedance` parameters: `session_home_mode`
  (`fixed`/`capture_first`/`load`), `session_home_path`,
  `session_home_capture_*` validation gates, `normal_run_start_gate_enabled`
  + `normal_run_start_warn_m=0.100` / `normal_run_start_refuse_m=0.150` /
  `emergency_return_start_refuse_m=0.300` / `return_only_if_too_far_enabled`,
  and `post_run_return_*` cleanup parameters.
- Three-tier run-start gate (only when `normal_run_start_gate_enabled=true`):
  - `d <= 0.150`: normal startup allowed (warn above 0.100);
  - `0.150 < d <= 0.300`: official GP recording refused; optional no-GP
    return-only cleanup if `return_only_if_too_far_enabled=true`;
  - `d > 0.300`: all automatic motion refused; operator recovery required.
- Post-run return: after the final round, the controller receives
  `/shutdown_control`, disables data recording and GP compensation/online
  update, then reuses the existing conservative startup interpolation (default
  0.005 m/s, startup torque clip and slew limits still active) to return to
  `session_home`, holds `post_run_return_hold_sec`, publishes zero torque,
  saves the CSV, publishes `/post_run_return_complete`, and exits.
  `trajectory_publisher` (`post_run_return_wait_enabled`) stays alive until
  that completion message or a timeout so the launch-level `on_exit=Shutdown`
  cannot kill the controller mid-return.
- CSV purity: controller history buffers append only while
  `data_recording_enabled=true`; the return phase runs strictly after the
  publisher sets it to `false`, so official RMSE rows are never contaminated
  (no phase column was added — that would have touched dozens of columns).
- `session_home.json` records version, created_at, source, `ee_pose_xyz`,
  `nominal_fixed_start_xyz`, `q_at_capture` (diagnostic), and notes. The
  runner manifest records `session_home_mode`, `session_home_path`, and
  `post_run_return` per case.

## How to run

First run of a session (capture):

```
scripts/run_goal12_python_only_f50_scale1_manual_matrix_20260703.sh \
  --source local --scale 0.25 --strict-start \
  --session-home-mode capture_first
```

Later runs of the same session (reuse):

```
scripts/run_goal12_python_only_f50_scale1_manual_matrix_20260703.sh \
  --source local --scale 0.5 --strict-start \
  --session-home-mode load \
  --session-home-path outputs/manual_compensation/<stamp>/session_home.json
```

Within one runner invocation, `capture_first` automatically degrades to `load`
from the second case on, so a `--source all` matrix shares one home. `--plan`
prints all session/return settings and creates no directories or JSON files.

## Design notes

- `anchor_trajectory_start_to_current_pose` stays `false` by default: anchoring
  would silently shift the multisine center per run and break the formal
  fixed-geometry comparison across runs. `session_home` fixes the *reset* pose
  instead of moving the *trajectory*.
- `normal_run_start_refuse_m≈0.150` for repeated real-robot matrix runs: the
  observed reflex started from 0.217 m, which the old single 0.300 m refuse
  threshold allowed. 0.150 m keeps official runs close to the validated home
  while the 0.300 m emergency threshold still permits supervised return-only
  cleanup in between.
- Defaults are backward-compatible: `session_home_mode=fixed`, gate off,
  post-run return off. Nothing changes unless the new args are set.
- No changes to GP math, fusion source selection, cpp_relayer, bringup, or
  controller_manager. The repo has no authoritative Franka joint-limit table in
  this controller, so no joint-margin thresholds were invented; captured `q`
  is stored in the JSON for offline diagnostics only.
