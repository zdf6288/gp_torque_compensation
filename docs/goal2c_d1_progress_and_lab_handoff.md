# GOAL2 C/D1 - Progress Summary and Lab Handoff

## Scope

This note summarizes the current GOAL2 C and GOAL2 D1 status for the Goal 2 workspace.

It covers:

- GOAL2 C offline/mock GP model-level timing benchmark.
- GOAL2 D1 Python controller timing logging implementation.
- Current local PC / WSL build limitation.
- Lab handoff checklist for later fake/sim validation.

It does not cover GOAL2 D2, GOAL2 D3, or GOAL2 D4. It is not a fake/sim runtime result, not a real robot experiment record, and not a real robot safety proof. No fake hardware validation or real robot validation is claimed here.

## GOAL2 C Summary

GOAL2 C added an offline/mock GP timing benchmark. Its purpose was to measure model-level timing without running ROS, without running the controller, without connecting Franka, and without starting fake hardware.

Added files:

- `scripts/goal2c_offline_mock_timing.py`
- `docs/goal2c_offline_mock_timing_summary.md`

Benchmark model source:

- Model zip: `new_structure/gp/gp_models.zip`
- Temp extracted model dir: `outputs/goal2c_tmp_gp_models_from_zip/gp_models`
- Benchmark output dir: `outputs/goal2c_offline_mock_timing_smoke_real_model`

Result status:

- Benchmark status: success
- Records count: `1109`
- Skipped count: `0`
- Failed count: `0`
- Synthetic input: yes, timing only
- Fallback: none
- Mock cloud: JSON mock only
- `add_point()`: measured on copied models only

Timing highlights from the smoke benchmark:

| Operation | Model kind | Count | p50_ms | p95_ms | p99_ms | max_ms | Caveat |
|---|---|---:|---:|---:|---:|---:|---|
| predict per-joint | local | 350 | 0.006273 | 0.023739 | 0.035299 | 0.071110 | synthetic input |
| predict 7-joint total | local | 50 | 0.190777 | 0.352677 | 0.440985 | 0.450665 | synthetic input |
| predict per-joint | cloud_like | 350 | 0.006154 | 0.009940 | 0.025836 | 0.039098 | local cloud-like pickle, not network |
| predict 7-joint total | cloud_like | 50 | 0.185976 | 0.283656 | 0.310416 | 0.318660 | not real cloud delay |
| predict local + cloud total | combined | 50 | 0.074819 | 0.107127 | 0.127663 | 0.139744 | combined local process timing |
| add_point total | local | 5 | 0.822784 | 1.601021 | 1.740074 | 1.774837 | copied model only |
| add_point total | cloud_like | 5 | 0.576435 | 0.748816 | 0.779057 | 0.786617 | copied model only |
| mock roundtrip | mock_cloud | 50 | 0.009282 | 0.015336 | 0.045764 | 0.047500 | JSON mock only |

GOAL2 C caveats:

- Offline/mock timing is not fake hardware timing.
- Offline/mock timing is not real robot safety proof.
- Synthetic input is for timing only and should not be used for accuracy conclusions.
- `cloud_like` is a local pickle model path, not real cloud communication.
- Mock cloud timing is JSON serialization / parse only, not ROS service timing.
- There is no controller callback wall-duration measurement in GOAL2 C.
- There is no ROS executor, pub-sub, Franka communication, or real robot timing in GOAL2 C.

## GOAL2 D1 Summary

GOAL2 D1 added Python controller timing logging instrumentation. The purpose was to prepare callback and GP timing visibility while preserving existing controller behavior by default.

Commit:

- `5e34556 Add GOAL2 D1 controller timing logging`

Modified files:

- `new_structure/py_controllers/py_controllers/cartesian_impedance.py`
- `new_structure/py_controllers/launch/cartesian_impedance_launch.py`

New timing parameters:

- `timing_logging_enabled`
- `timing_log_stride`
- `timing_output_dir`
- `deadline_ratio_warn_threshold`
- `controller_update_rate_label`

Expected timing CSV fields include:

- `callback_wall_ms`
- `callback_period_ms`
- `callback_deadline_ms`
- `callback_deadline_ratio`
- `callback_deadline_miss`
- `gp_total_ms`
- `gp_local_predict_ms`
- `gp_cloud_like_predict_ms`
- `gp_add_point_ms`
- `future_request_ms`
- `csv_append_ms`
- `csv_save_ms`
- flags such as `gp_compensation_enabled`, `gp_online_update_enabled`, and `exception_flag`

Default behavior and safety boundary:

- `timing_logging_enabled=false` by default.
- No torque semantic change is intended.
- No command publication change is intended.
- No GP compensation behavior change is intended.
- No YAML `update_rate` change was made.
- No trajectory publisher change was made.
- No C++ relayer change was made.
- No Franka hardware interface change was made.
- No model files or raw data were changed.
- GP compensation was not enabled by default.
- Online update behavior was not changed by this GOAL2 D1 timing work.

Validation status:

- Static review completed.
- AST / syntax-oriented review completed.
- Self-review completed.
- Commit and push completed.
- No ROS launch was run.
- No ROS node was started.
- No fake hardware was started.
- No real robot run was performed.

Remaining validation gap:

- Runtime timing CSV generation has not yet been proven.
- Fake/sim callback timing has not yet been measured.
- Real robot timing and safety have not been proven.

## Local PC Build Limitation

The current home PC / WSL environment is not confirmed to match the lab Franka environment. During local fake/sim preparation, the build was blocked by a Franka dependency compatibility issue.

Observed sequence:

- `install/setup.bash` was initially missing.
- Initial build attempts could not find `FrankaConfig.cmake`.
- `FrankaConfig.cmake` was found at `/home/dummd/impl_course/libfranka/build/FrankaConfig.cmake`.
- With `Franka_DIR=/home/dummd/impl_course/libfranka/build`, CMake could find Franka.
- `franka_hardware` then failed to build.

Current local dependency finding:

- `/home/dummd/impl_course/libfranka` is `libfranka 0.21.2`.
- `Robot::startTorqueControl()` returns `std::unique_ptr<franka::ActiveControlBase>`.
- The repo's `franka_ros2/franka_hardware` code still expects `std::unique_ptr<franka::ActiveControl>`.
- Typical errors are in:
  - `franka_ros2/franka_hardware/src/robot.cpp`
  - `franka_ros2/franka_hardware/include/franka_hardware/robot.hpp`

Interpretation:

- This is not a GOAL2 D1 timing code issue.
- This is a local PC / WSL `libfranka` and `franka_ros2` API compatibility mismatch.
- It is not recommended to patch `franka_hardware` inside GOAL2 D1 just to force fake/sim locally.
- Fake/sim runtime validation should wait for the lab environment or an existing compatible build workspace.

## What Is Proven / Not Proven

| Claim | Status | Evidence | Caveat |
|---|---|---|---|
| GOAL2 C model-level timing benchmark works | Proven for offline/mock scope | `scripts/goal2c_offline_mock_timing.py`, `docs/goal2c_offline_mock_timing_summary.md`, success result with `1109` records | Offline/mock only; not ROS or robot timing |
| GOAL2 C local and cloud-like model timing was measured | Proven for smoke benchmark | Output dir `outputs/goal2c_offline_mock_timing_smoke_real_model` | Synthetic input; `cloud_like` is local pickle, not network |
| GOAL2 D1 static implementation was reviewed | Proven for static review scope | Commit `5e34556 Add GOAL2 D1 controller timing logging` | Static review is not runtime proof |
| GOAL2 D1 changes preserve default runtime behavior | Intended and statically reviewed | `timing_logging_enabled=false`, no planned torque or publication semantic change | Needs runtime smoke validation later |
| Timing CSV runtime generation works | Not yet proven | No fake/sim or controller runtime was executed | Requires later D1 fake/sim smoke |
| Fake/sim timing behavior is acceptable | Not yet proven | Local build blocked by Franka dependency mismatch | Validate in lab-compatible workspace |
| Real robot safety is proven | Not proven | No real robot run was performed | Requires separate approved real robot plan |
| 75 Hz / 100 Hz behavior is proven | Not proven | No 75 Hz or 100 Hz runtime test was performed | Do not enter frequency escalation here |
| GP-on compensation timing and behavior are proven | Not proven | GP compensation was not enabled | Keep `gp_compensation_enabled=false` for D1 smoke |
| Real cloud timing is proven | Not proven | Mock cloud is JSON-only | No real network/cloud/ROS service timing measured |

## Lab Handoff Checklist

Before any lab fake/sim validation:

1. Confirm the lab workspace path and current branch.
2. Confirm `git log --oneline -5` includes `5e34556 Add GOAL2 D1 controller timing logging`.
3. Confirm whether `install/setup.bash` already exists in the lab workspace.
4. If `install/setup.bash` does not exist, confirm lab `libfranka` / `franka_ros2` version compatibility before building.
5. Do not patch the hardware interface unless separately approved.
6. Before fake/sim run, confirm:
   - `use_fake_hardware:=true`
   - `fake_sensor_commands:=true`
   - `load_gripper:=false`
   - no real Franka IP is used
   - `gp_compensation_enabled:=false`
   - `gp_online_update_enabled:=false`
   - `timing_logging_enabled:=true`
   - `update_rate` remains 50 Hz
7. Run only a GOAL2 D1 fake/sim smoke validation.
8. Check the generated timing CSV.
9. Do not enter real robot testing.
10. Do not enter 75 Hz or 100 Hz testing.
11. Do not enter GP-on testing.

## Recommended Next Steps

- On the current local PC, stop chasing the `franka_hardware` build issue for GOAL2 D1.
- Keep this progress and handoff note as the current documentation output.
- Back in the lab, first do a fake/sim preflight in the compatible workspace.
- If the lab build is already valid or builds cleanly, do an explicit GOAL2 D1 fake/sim smoke run.
- After fake/sim success, write a separate GOAL2 D1 runtime result summary.
- Do not enter GOAL2 D2, GOAL2 D3, or GOAL2 D4 from this task.
- Continue to forbid real robot experiments unless there is a separate approved GOAL2 E plan.

## Commit Recommendation

This documentation can be committed separately.

Suggested commit file:

- `docs/goal2c_d1_progress_and_lab_handoff.md`

Do not commit:

- `build/`
- `install/`
- `log/`
- `outputs/`
- model files
- raw data
- any unrelated local artifacts

Suggested commit message:

`Add GOAL2 C D1 progress handoff note`
