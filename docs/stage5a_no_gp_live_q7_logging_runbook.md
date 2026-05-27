# Stage 5A No-GP Live q7 Logging Runbook

## 1. Purpose

This runbook is for Stage 5A no-GP live q7 / 14D joint-space data collection.

The purpose is to collect live joint posture/support data and check whether the real robot posture family is close enough to the frozen GP training support before any later conservative GP-on discussion.

This runbook explicitly does not:

- run a GP-on experiment.
- run a compensation experiment.
- prove tracking improvement.
- run full 7DoF joint-space excitation.
- modify controller, launch, config, GP compensation, or torque command logic.

The expected data are live joint positions and joint velocities, especially `joint_pos_7` / q7 and the full 14D vector `joint_pos_1..7` + `joint_vel_1..7`.

After collection, use `scripts/validate_stage5_q7_support.py` to compare the live CSV against the reference support. A validator pass does not directly authorize GP-on. A support fail means stop and do not proceed to GP-on.

## 2. When to Use This Runbook

Use this runbook for Stage 5A support-aware `z_modulated_circle` / q7 posture consistency checks:

- before any conservative GP-on re-entry.
- on the lab real Linux machine for actual no-GP data collection.
- after confirming the lab checkout contains the Stage 5 q7 validator and this runbook.
- when the goal is live posture/support data only.

WSL is for code, docs, and offline analysis. The lab real Linux machine is for actual Franka data collection.

## 3. Hard Safety Boundary

This run is no-GP support data collection only.

Required boundaries:

- no GP-on.
- no online update.
- no compensation.
- no no-clip run.
- no high compensation scale.
- no scale sweep.
- no controller modification.
- no torque path modification.
- no launch/config hot patch on the lab machine.
- stop on abnormal motion, vibration, abnormal sound, reflex stop, or unexpected posture behavior.
- stop on unsafe communication behavior.
- support gate fail means do not proceed to GP-on.

Do not treat a clean terminal exit as the only definition of usable data. If CSV data were saved and are complete, communication caveats can be recorded separately. Robot safety always takes priority over collecting a clean shutdown.

## 4. Pre-Run Checklist on Lab Linux

Run these checks on the lab real Linux machine before any real robot launch. These are templates; do not treat them as already executed.

- Confirm the lab path:
  - `cd ~/dongfa/tt_dgp`
  - or replace with the actual reviewed lab checkout path.
- Confirm branch:
  - `git branch --show-current`
  - expected: `frozen_gp_spatial_trajectory`
- Confirm local status:
  - `git status --short`
  - expected: no unreviewed local diff.
- Confirm recent commits:
  - `git log --oneline -5`
  - verify the current commit includes `scripts/validate_stage5_q7_support.py` and this docs update.
- Confirm robot IP:
  - expected robot IP: `172.16.0.4`
- Confirm normal Franka workflow:
  - Franka web interface reachable.
  - FCI activation follows the lab standard workflow.
  - joints are unlocked using the normal Franka web interface process.
  - no active robot faults before starting.
- Confirm no unreviewed local modifications to controller, launch, config, trajectory, scripts, or GP path.
- Prepare output directory for CSV and terminal log.
- Record operator, date, commit hash, robot IP, and planned trajectory settings.

Do not proceed if the checkout has unexplained local diffs or the branch is not the reviewed Stage 5A branch.

## 5. Required Runtime Settings

Before running, verify exact parameter names from `new_structure/py_controllers/launch/cartesian_impedance_launch.py`.

Required no-GP semantics:

- `gp_prediction_enabled:=false`, or otherwise ensure the GP prediction path is not used.
- `gp_compensation_enabled:=false`.
- `gp_online_update_enabled:=false`.
- `gp_compensation_scale` must not participate in this no-GP run.
- `gp_compensation_clip_nm` must not be used as a reason to enable compensation.
- record `trajectory_mode`.
- record `z_amplitude`.
- record `z_frequency_multiplier`.
- record `circle_frequency`.
- record `transition_duration`.

If any parameter name is different on the lab branch, stop and verify against the launch file before running. Do not invent new launch arguments during the lab run.

## 6. Suggested No-GP Logging Command Template

This is a command template only. Before running, verify exact launch arguments in `new_structure/py_controllers/launch/cartesian_impedance_launch.py`.

Do not add GP-on flags. Do not enable compensation. Do not enable online update.

Suggested shell sequence:

    source /opt/ros/humble/setup.bash
    cd ~/dongfa/tt_dgp
    source install/setup.bash
    mkdir -p outputs/stage5a_live_q7_logs
    ros2 launch py_controllers cartesian_impedance_launch.py robot_ip:=172.16.0.4 gp_prediction_enabled:=false gp_compensation_enabled:=false gp_online_update_enabled:=false trajectory_mode:=z_modulated_circle z_amplitude:=TODO_VERIFY z_frequency_multiplier:=TODO_VERIFY circle_frequency:=TODO_VERIFY transition_duration:=TODO_VERIFY 2>&1 | tee outputs/stage5a_live_q7_logs/live_no_gp_zmod_DATE_OR_LABEL_terminal.log

If `gp_prediction_enabled` does not exist, use the reviewed no-GP launch semantics that exist in the launch file. If the trajectory settings use different argument names, replace the `TODO_VERIFY` arguments only after read-only confirmation from the launch file or reviewed lab notes.

This command is not to be run from WSL and is not executed as part of this docs task.

## 7. During-Run Monitoring

Monitor the run directly:

- robot motion smoothness.
- q7 / joint posture if visible in logs.
- communication warnings.
- controller errors.
- CSV save message.
- abnormal vibration.
- abnormal sound.
- reflex stop.
- unexpected trajectory behavior.

Stop immediately with the normal safe stop procedure or `Ctrl+C` if motion, sound, communication, posture, or controller behavior looks unsafe.

A single successful saved no-GP live data file is enough to move to offline analysis. Repeated live runs are not required before the first validator check.

Known caveat handling:

- `communication_constraints_violation` should be recorded as a caveat.
- User Stop should be recorded as a caveat.
- `rclpy` shutdown errors should be recorded as a caveat.
- If the CSV is complete and readable, these caveats do not automatically make the data unusable.
- If robot motion was abnormal, do not use the data to justify GP-on.

## 8. Post-Run Data Check

After the no-GP run, locate the new CSV and terminal log.

Suggested read-only checks:

- Find likely CSV files:
  - `find outputs -type f -name "*.csv" -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" | sort`
- Check file size:
  - `ls -lh PATH_TO_LIVE_CSV`
- Inspect header:
  - `python3 -c 'import csv,sys; print(next(csv.reader(open(sys.argv[1]))))' PATH_TO_LIVE_CSV`
- Confirm required columns:
  - `joint_pos_1`
  - `joint_pos_2`
  - `joint_pos_3`
  - `joint_pos_4`
  - `joint_pos_5`
  - `joint_pos_6`
  - `joint_pos_7`
  - `joint_vel_1`
  - `joint_vel_2`
  - `joint_vel_3`
  - `joint_vel_4`
  - `joint_vel_5`
  - `joint_vel_6`
  - `joint_vel_7`

If the column names differ, record the exact names and pass explicit validator column options if needed. Do not rename or edit raw CSV data in place.

Optional q7 quick check:

    python3 - PATH_TO_LIVE_CSV <<'PY'
    import csv, sys, statistics
    path = sys.argv[1]
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    values = [float(r["joint_pos_7"]) for r in rows if r.get("joint_pos_7")]
    print("count", len(values))
    print("min", min(values))
    print("max", max(values))
    print("mean", statistics.fmean(values))
    print("std", statistics.pstdev(values))
    PY

## 9. Offline Validation with Stage 5 q7 Validator

Run the offline validator after the live CSV exists and its columns are understood.

Command template:

    python3 scripts/validate_stage5_q7_support.py --model-dir data/stage4/cross_traj/models/GP_B_zmod_train --candidate-csv PATH_TO_LIVE_CSV --output-dir outputs/stage5_q7_support_preflight/live_DATE_OR_LABEL --label-reference GP_B_zmod_train --label-candidate live_no_gp_zmod --overwrite

Expected exit codes:

- `0 = preflight pass`
- `1 = invalid input / usage error`
- `2 = valid input but support/preflight fail`

Interpretation:

- exit code `0` means the preflight support check passed. It does not approve GP-on.
- exit code `1` means fix input path, usage, or columns first.
- exit code `2` means do not proceed to GP-on.

The validator should generate JSON and Markdown summaries. Review both q7 support and complete 14D support. The recent offline test showed `q7_support_pass=true` while `joint_space_14d_pass=false` with `worst_dimension=joint_pos_4`, so q7 pass alone is not enough.

## 10. Decision Rules

Proceed only to offline analysis if:

- CSV exists.
- terminal log exists or run caveats were recorded.
- columns are valid or column mapping is documented.
- q7 stats are readable.
- validator summary was generated.

Do not proceed to GP-on if:

- q7 support fails.
- 14D support fails.
- worst dimension indicates large support mismatch.
- CSV is missing required joint position or velocity columns.
- robot motion was abnormal.
- local unreviewed code diff exists.

Even if the validator passes:

- discuss possible conservative GP-on re-entry only in a separate reviewed step.
- do not do direct same-day GP-on without read-only review.
- do not run no-clip.
- do not use high scale.
- do not enable online update.

Validator pass is a support preflight result, not a GP-on approval.

## 11. Expected Outputs to Archive

Archive these outputs together:

- live CSV.
- terminal log.
- `stage5_q7_support_summary.json`.
- `stage5_q7_support_summary.md`.
- short note with run parameters:
  - commit hash.
  - branch.
  - robot IP.
  - trajectory settings.
  - no-GP settings.
  - operator and date.
- caveats:
  - communication warnings.
  - User Stop.
  - `rclpy` shutdown messages.
  - any abnormal motion or safety intervention.

If any abnormal motion occurred, mark the run as blocked for GP-on regardless of validator output.

## 12. Next Step After This Runbook

Recommended next steps:

1. Commit this docs-only runbook after review.
2. Use the runbook on lab Linux for no-GP live q7 / 14D logging.
3. Run `scripts/validate_stage5_q7_support.py` offline against `GP_B_zmod_train`.
4. Bring the CSV, terminal log, JSON summary, Markdown summary, and run notes back to WSL for analysis.
5. Only after support pass and read-only review, consider a separate conservative GP-on re-entry prompt.

Full 7DoF joint-space excitation remains deferred. It needs a separate design and safety review before any implementation or live robot attempt.
