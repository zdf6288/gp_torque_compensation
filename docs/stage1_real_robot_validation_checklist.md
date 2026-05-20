# Stage 1 Real-Robot Validation Checklist

## 1. Current Branch and Commits

- branch: `frozen_gp_spatial_trajectory`
- Stage 1 commit: `b378acb Add frozen GP experiment controls`
- AGENTS commit: `01cb132 Add project agent instructions`
- repo path on WSL dev machine: `/home/dummd/projects/gp_torque_compensation`
- lab machine should `git pull` this branch before testing

## 2. Stage 1 Code Changes Summary

Stage 1 added these controller / launch parameters:

- `gp_online_update_enabled`
- `gp_model_dir`
- `gp_compensation_enabled`
- `gp_compensation_source`
- `gp_compensation_scale`
- `gp_compensation_clip_nm`

Modified files:

- `new_structure/py_controllers/py_controllers/cartesian_impedance.py`
- `new_structure/py_controllers/launch/cartesian_impedance_launch.py`

## 3. Default Safety Behavior

Default values:

- `gp_online_update_enabled=True`
- `gp_model_dir="./new_structure/gp/gp_models"`
- `gp_compensation_enabled=False`
- `gp_compensation_source="local"`
- `gp_compensation_scale=0.1`
- `gp_compensation_clip_nm=0.5`

Safety behavior:

- default does not change final `tau`
- default does not enable GP compensation
- default keeps the original online update behavior
- compensation enters torque command only when explicitly enabled

## 4. Known Environment Notes

Current WSL check results:

- clean shell `AMENT_PREFIX_PATH=/opt/ros/humble`
- clean shell `COLCON_PREFIX_PATH=`
- `numpy` OK
- `scipy` OK
- `sklearn` missing
- `torch` missing

Notes:

- missing `sklearn` / `torch` does not affect the completed static code modification
- missing `sklearn` / `torch` may affect GP training or runtime
- re-check dependencies on the lab real-robot machine
- avoid `.bashrc` automatically sourcing other workspaces and polluting ROS2 overlays

## 5. Lab Machine Pull Checklist

Suggested checks on the lab machine:

- `pwd`
- `git branch --show-current`
- `git status --short`
- `git pull`
- `git log --oneline -5`
- confirm `b378acb Add frozen GP experiment controls` is present

## 6. Clean Shell Checklist

Suggested clean shell checks:

- source only the required ROS2 / workspace setup files
- check `ROS_DISTRO`
- check `AMENT_PREFIX_PATH`
- check `COLCON_PREFIX_PATH`
- check `which ros2`
- check `which colcon`
- check `python3 --version`
- check `python3 -c "import numpy, scipy; print('numpy scipy ok')"`
- check `sklearn` / `torch` if needed for the planned test

## 7. Build Checklist

- decide whether to build from repo root or from `new_structure`
- do not directly delete `build/`, `install/`, or `log/`
- if a clean build is needed, confirm the current path first
- if build fails, record the full error and do not continue to real-robot launch

## 8. First Real-Robot Test: Baseline Only

The first real-robot run must keep:

- `gp_compensation_enabled:=false`
- do not change `gp_compensation_scale`
- do not change `gp_compensation_clip_nm`
- do not modify trajectory
- do not increase `update_rate`
- do not change the Franka hardware interface

Goals:

- validate default no-compensation behavior
- confirm the Stage 1 patch did not break original controller startup
- confirm logs show Stage 1 parameters
- observe whether there is vibration, abnormal sound, or reflex stop

## 8.1 Stage 1 Shutdown / Plotting Safety Update

Default shutdown behavior is intentionally lightweight for real-robot runs:

- controller shutdown saves `cartesian_impedance_controller_data.csv` by default
- controller shutdown does not run runtime plotting by default
- controller shutdown does not run `ablation.py` by default
- plotting should be run offline after the robot is stopped and the ROS2 launch has exited

New launch/controller parameters:

- `save_csv_on_shutdown` default: `true`
- `enable_runtime_plotting` default: `false`
- `run_ablation_on_shutdown` default: `false`

Recommended baseline values:

- `save_csv_on_shutdown:=true`
- `enable_runtime_plotting:=false`
- `run_ablation_on_shutdown:=false`

If `communication_constraints_violation` appears at shutdown, do not repeatedly rerun the robot. First check shutdown/plotting load, network/realtime stability, residual ROS2 processes, and Franka Desk state.

This shutdown fix does not mean GP compensation is ready to enable.

## 9. Second Low-Risk Test: Frozen GP Without Compensation

Only after the baseline is stable, test:

- `gp_online_update_enabled:=false`
- `gp_compensation_enabled:=false`

Goals:

- validate only frozen GP logging and stopped `add_point`
- do not validate compensation yet
- do not change final `tau`

## 10. Do Not Test First

Do not do these in the first real-robot test:

- do not set `gp_compensation_enabled:=true`
- do not increase `gp_compensation_scale`
- do not increase `gp_compensation_clip_nm`
- do not test spatial / 3D trajectory
- do not increase to 100 Hz
- do not modify torque command logic

## 11. Stop Conditions

Immediately press `Ctrl+C` if any of these occur:

- robot vibration
- abnormal sound
- reflex stop
- obvious trajectory abnormality
- abnormal log spam
- command / state clearly out of sync
- controller error

## 12. Run Log Template

- Date:
- Machine:
- Branch:
- Commit:
- Build command:
- Launch command:
- Parameters:
- `gp_online_update_enabled`:
- `gp_compensation_enabled`:
- `gp_model_dir`:
- Observed behavior:
- Logs:
- Errors:
- Reflex stop:
- Ctrl+C stopped cleanly:
- Next action:

## 13. Validation Status

- WSL static validation: done
- py_compile: done before `.pyc` restore
- self-review: done
- build: not done
- launch: not done
- real robot validation: not done
