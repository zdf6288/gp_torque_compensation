# GOAL2 Lab Validation Checklist

## Purpose

This checklist prepares the first GOAL2 lab validation pass for controller timing data.

The immediate goal is to validate conservative timing data collection, not to prove GP compensation improves tracking. The first lab pass should focus on:

- prediction-on / compensation-off timing
- `delay_steps` effects on timing and error signals
- `control frequency` / `update_rate` effects on deadline ratio, after the 50 Hz gates pass

This document is an execution checklist, not a real robot safety proof. Do not describe fake/sim timing as real robot validation.

## Environment Boundary: WSL vs Lab Linux

| Environment | Franka connected | Modify files | Launch allowed | Real robot allowed | Continue condition |
|---|---:|---:|---:|---:|---|
| Home WSL / local PC | No | No for lab validation | No | No | Only read docs/code and prepare analysis |
| Lab Linux, before fake/sim gate | No | No | Fake/sim only after checks | No | Workspace, branch, HEAD, and install state are confirmed |
| Lab Linux, fake/sim smoke | No | No | Fake/sim only | No | Timing CSV is generated and sanity check is acceptable |
| Lab Linux, first real robot run | Yes | No | Approved conservative command placeholder only | 50 Hz only | Fake/sim gate passed and operator confirms Franka readiness |

Stop immediately if the workspace path, branch, or HEAD is not the intended lab state.

## Required Commits / Branch Check

Before any launch, confirm the lab machine has the expected GOAL2 work:

```bash
pwd
git branch --show-current
git status --short
git log --oneline -8
```

Expected:

- Path is the intended lab checkout for `gp_torque_goal2_delay`.
- Branch is `stage6_goal2_delay`.
- `git status --short` is clean.
- Recent commits include:
  - `224d934 Parameterize GOAL2 D2 cloud delay steps`
  - `5e34556 Add GOAL2 D1 controller timing logging`
  - `437df59 Add GOAL2 C offline mock GP timing benchmark`

Continue only if all checks match. If any check fails, stop and record the actual path, branch, status, and HEAD.

## Preflight Commands

These checks do not connect to Franka, do not modify files, and do not launch ROS nodes:

```bash
test -f install/setup.bash
echo $ROS_DISTRO
which ros2
which colcon
git status --short
```

If `install/setup.bash` exists:

- Do not rebuild just because it exists.
- Continue to fake/sim preflight after sourcing the intended environment in the lab shell.

If `install/setup.bash` does not exist:

- Stop before building.
- Confirm lab `libfranka` / `franka_ros2` compatibility first.
- Do not patch `franka_hardware`.
- Do not patch `libfranka`.

## Fake/sim Preflight Gate

Purpose: confirm the lab workspace can start a conservative fake/sim setup before any real robot action.

| Check | Franka connected | Modify files | Launch allowed | Real robot allowed | Continue condition |
|---|---:|---:|---:|---:|---|
| Workspace / branch / HEAD confirmed | No | No | No | No | All expected |
| `git status --short` clean | No | No | No | No | Empty output |
| `install/setup.bash` confirmed | No | No | No | No | Exists, or compatibility is confirmed before build |
| Fake/sim launch parameters reviewed | No | No | Fake/sim only | No | Conservative flags are explicit |

Required fake/sim parameter intent:

- `use_fake_hardware:=true`
- `fake_sensor_commands:=true`
- `load_gripper:=false`
- no real Franka IP is used
- `timing_logging_enabled:=true`
- `gp_prediction_enabled:=false` for the first preflight
- `gp_compensation_enabled:=false`
- `gp_online_update_enabled:=false`
- `update_rate` remains 50 Hz

Do not continue if any flag is unclear.

## Fake/sim Timing CSV Smoke Gate

Purpose: prove that GOAL2 D timing CSV generation works in fake/sim before conservative real robot timing.

Allowed:

- Franka connected: no
- Modify files: no
- Launch: fake/sim only
- Real robot: no

Smoke intent:

- Run one short fake/sim timing session.
- Confirm `goal2d_controller_timing.csv` is generated.
- Run the offline sanity checker on the generated CSV.
- Accept caveats such as `User Stop`, `communication_constraints_violation`, or `rclpy` shutdown noise only if a valid timing CSV exists and the sanity checker does not show severe timing or exception issues.

Continue only if:

- CSV rows are sufficient for the configured smoke duration.
- `callback_deadline_ratio` is below the warning threshold.
- `callback_deadline_miss` is zero or explicitly understood.
- `exception_flag` is zero.
- `gp_compensation_enabled` remains off.
- `gp_online_update_enabled` remains off.

Stop if the CSV is missing, empty, has repeated exceptions, or shows clear deadline misses.

## Conservative First Real-robot Timing Matrix

Do not enter this matrix until fake/sim preflight and fake/sim timing CSV smoke have passed.

Real-robot commands are intentionally omitted here. Use only a separately reviewed lab command placeholder after the operator confirms Franka readiness, web interface state, FCI activation, and stop plan.

| Run | Frequency | `delay_steps` | Prediction | Compensation | Online update | Franka connected | Modify files | Launch allowed | Continue condition |
|---|---:|---:|---|---|---|---:|---:|---:|---|
| R1 | 50 Hz | N/A | off | off | off | Yes | No | Conservative real robot only | Baseline timing CSV is acceptable |
| R2 | 50 Hz | 0 | on | off | off | Yes | No | Conservative real robot only | R1 acceptable, no abnormal behavior |
| R3 | 50 Hz | 1 | on | off | off | Yes | No | Conservative real robot only | R2 acceptable, no abnormal behavior |
| R4 | 50 Hz | 2 | on | off | off | Yes | No | Conservative real robot only | R3 acceptable, no abnormal behavior |
| R5 | 50 Hz | 5 | on | off | off | Yes | No | Conservative real robot only | R4 acceptable, no abnormal behavior |
| R6 | 50 Hz | 2 | on | off | off | Yes | No | Conservative real robot only | Repeatability check after R5 |

Forbidden in the first pass:

- direct 75 Hz real robot run
- direct 100 Hz real robot run
- GP compensation on
- online update on
- no-clip or high-scale compensation
- hardware interface patching
- treating fake/sim timing as real robot safety proof
- describing `cloud_like` as real network cloud timing

## Stop Conditions

Stop immediately and do not continue the matrix if any item occurs:

- Wrong workspace path, wrong branch, or unexpected HEAD.
- `git status --short` is not clean before lab execution.
- `install/setup.bash` is missing and compatibility has not been confirmed.
- Any temptation to patch `franka_hardware` or `libfranka` for this validation.
- Fake/sim cannot generate timing CSV.
- Sanity checker reports severe missing key columns, deadline misses, or nonzero `exception_flag`.
- Franka vibration, abnormal sound, reflex stop, trajectory abnormality, unexpected torque behavior, or operator concern.
- Any run accidentally enables `gp_compensation_enabled`.
- Any run accidentally enables `gp_online_update_enabled`.
- A 75 Hz or 100 Hz real robot run is proposed before the 50 Hz matrix has been reviewed.

## Data Files to Collect

Collect these files for each accepted fake/sim or real robot run:

- timing CSV, expected name: `goal2d_controller_timing.csv`
- sanity checker stdout log
- optional sanity checker JSON summary
- controller console log
- launch command text or lab run note, with sensitive robot IP omitted if needed
- `git log --oneline -8`
- `git status --short`
- run manifest copied from the template below

Do not commit generated `outputs/`, timing CSVs, or raw lab logs unless a separate data-management decision says to do so.

## Run Manifest Template

```text
GOAL2 Lab Validation Run Manifest

run_id:
date_time_local:
operator:
machine:
workspace_path:
branch:
head_commit:
git_status_short:

environment:
  ROS_DISTRO:
  install_setup_bash_exists:
  Franka_connected:
  fake_hardware:
  real_robot:

run_config:
  frequency_hz:
  update_rate_source:
  delay_steps:
  gp_prediction_enabled:
  gp_compensation_enabled:
  gp_online_update_enabled:
  timing_logging_enabled:
  timing_output_dir:

data_files:
  timing_csv:
  sanity_stdout:
  sanity_json:
  console_log:

sanity_summary:
  status:
  rows:
  callback_deadline_ratio_p95:
  callback_deadline_ratio_max:
  callback_deadline_miss_count:
  exception_flag_count:

caveats:
operator_decision:
```

## Caveats for Paper Writing

- GOAL2 C offline/mock timing is model-level timing only.
- GOAL2 D fake/sim timing is not a real robot safety proof.
- `cloud_like` means local cloud-like GP branch or local delay simulation, not real network cloud communication.
- A single successful run is enough to proceed to the next conservative gate, but it is not evidence of fully stable or robust repeated validation.
- `communication_constraints_violation`, `User Stop`, and `rclpy` shutdown errors may be documented as caveats when useful data was produced.
- The first lab pass should report timing, deadline ratio, prediction/tracking error context, and caveats without claiming GP compensation benefit.
