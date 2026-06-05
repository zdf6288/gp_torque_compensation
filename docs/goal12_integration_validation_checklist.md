# GOAL12 Integration Validation Checklist

## Scope

Branch:

- `goal12_integrated_histdb_delay`

Integrated scope:

- GOAL1 historical DB / soft shadow logging.
- q7 GP compensation disable switch.
- GOAL2 delay / frequency / timing logging.

This checklist is for integration validation. It does not claim robust robot validation or GP tracking improvement.

## Home WSL Static Validation

Run from `/home/dummd/projects/gp_torque_compensation`:

- `pwd`
- `git branch --show-current`
- `git status --short`
- `python3 -m py_compile scripts/goal2_timing_csv_sanity_check.py`
- `python3 -m py_compile scripts/plot_goal2_timing_summary.py`
- `git diff --check`

Home WSL does not need ROS launch or `colcon build` if the Franka / ROS environment is incompatible.

## Lab Linux Build Validation

Run only on the intended lab Linux workspace after confirming branch, path, and clean status:

- `colcon build --packages-select py_controllers new_bringup --symlink-install`
- `source install/setup.bash`

Do not patch `franka_hardware`, `libfranka`, controller files, launch files, or config files as part of this checklist.

## Fake / No-Robot Validation

Use fake hardware only, with conservative flags:

- `gp_compensation_enabled:=false`
- `gp_online_update_enabled:=false`
- `timing_logging_enabled:=true`
- expected 50 Hz first-pass `control_frequency`

Optional trajectory mode:

- Pass `trajectory_mode:=goal1_spatial_multisine` explicitly to use the integration branch GOAL2-style multisine trajectory for fake / no-robot checks or later real shadow-only validation.

Check generated logs / CSVs for:

- `hist_db_*` columns or messages when historical DB shadow logging is enabled.
- `hist_soft_*` columns or messages when soft shadow logging is enabled.
- `delay_steps`.
- `control_frequency`.
- timing CSV presence and non-empty rows.
- `gp_applied=0` when compensation is disabled.

Stop if the launch uses the wrong workspace prefix or if compensation is unexpectedly active.

## Real Shadow-Only Validation

Use only after fake / no-robot validation is acceptable:

- `gp_compensation_enabled:=false`
- historical DB / soft shadow may be enabled only as shadow logging.
- timing logging is optional but recommended.
- keep 50 Hz as the first real validation frequency.

Do not treat shadow-only logging as proof that active compensation is safe.

## Active GP Validation Later

Active GP compensation belongs to a later, separately reviewed step:

- keep `gp_compensation_clip_nm` enabled.
- keep conservative `gp_compensation_scale`.
- optionally use `gp_compensation_disable_joint7:=true`.
- do not run no-clip or unlimited GP-on tests.

## Caveats

These are caveats, not automatic blockers, if a usable CSV exists and safety columns are valid:

- `communication_constraints_violation`.
- `User Stop`.
- `rclpy` shutdown errors after operator stop.

Document the caveat with the run notes and sanity checker output.

## Stop Conditions

Stop immediately if any item occurs:

- abnormal motion, vibration, abnormal sound, reflex stop, or operator concern.
- unexpected nonzero GP applied when `gp_compensation_enabled=false`.
- missing or invalid CSV safety columns needed for the run decision.
- launch uses the wrong workspace prefix.
- branch, path, or `git status --short` does not match the intended lab state.
