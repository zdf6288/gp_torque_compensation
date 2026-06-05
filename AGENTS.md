# AGENTS.md

Repo-level instructions for Codex / AI coding assistants working on `gp_torque_compensation`.

This file is not a README. It defines project-specific behavior, safety constraints, environment notes, and Stage 1 rules for future assistant work in this repository.

## 1. Language and Naming

- Default explanations to the user should be in Chinese.
- Keep file names, paths, branch names, function names, class names, parameter names, ROS2 package names, and commands in English.
- Prefer concise, explicit engineering explanations over broad speculation.

## 2. Repository Identity and Required Path

Project:

- `gp_torque_compensation`

GitHub repo:

- `https://github.com/zdf6288/gp_torque_compensation.git`

Current base branch:

- `ggwp`

Default main implementation branch:

- `frozen_gp_spatial_trajectory`

Required main development path:

- `~/projects/gp_torque_compensation`
- Absolute path: `/home/dummd/projects/gp_torque_compensation`

Do not primarily modify the old Windows desktop repo:

- `C:/Users/dummd/Desktop/GP-TORQUE-COMPENSATION/...`
- `/mnt/c/Users/dummd/Desktop/GP-TORQUE-COMPENSATION/...`

If the current path is under `/mnt/c/...GP-TORQUE-COMPENSATION...`, stop and report the wrong development path. Do not continue modifying files.

Before code changes, check:

- `pwd`
- `git branch --show-current`
- `git status --short`
- `git rev-parse --show-toplevel`

Branch policy:

1. Codex must always print or report:
   - `pwd`
   - `git branch --show-current`
   - `git status --short`
2. Default main implementation branch remains `frozen_gp_spatial_trajectory`.
3. Task-specific branches are allowed when the user explicitly states the task belongs there:
   - `goal1_historical_shadow_source`
   - `stage6_goal2_delay`
   - any branch whose name clearly starts with or contains `goal1`, `goal2`, `historical`, `shadow`, `offline`, or `analysis`
4. If the branch is not `frozen_gp_spatial_trajectory`, Codex should not automatically stop when:
   - the user explicitly named the current branch or task,
   - the task is offline-only, docs-only, or analysis-only,
   - or the task clearly belongs to the current branch.
5. Codex must still stop and ask/report when:
   - the branch is unrelated to the task,
   - the user did not authorize the branch,
   - or the task touches controller, torque, launch, config, or real-robot safety and the branch is ambiguous.
6. Do not switch branches unless the user explicitly confirms.

Dirty workspace policy:

1. If `git status --short` is dirty, Codex must list dirty files before editing.
2. Codex may continue only when:
   - the requested task explicitly allows modifying those dirty files,
   - or the task is limited to a new file and Codex can avoid touching existing dirty files,
   - or the task explicitly limits changes to specific files and Codex can avoid all unrelated dirty files.
3. Codex must not overwrite, restore, stage, commit, or push unrelated dirty files.
4. For offline-only scripts, docs, or reports, Codex may create or edit only the requested offline files even if unrelated dirty files exist, but must explicitly state:
   - which dirty files were pre-existing,
   - which files it actually modified,
   - that it did not touch pre-existing dirty files.

## 3. Development Environment Model

Recommended workflow:

- Windows: VS Code UI, ChatGPT, Obsidian, GitHub web.
- WSL Ubuntu 22.04: repo, VS Code Remote WSL, Codex plugin, git, colcon, ROS2 checks.
- Lab Linux real robot machine: `git pull`, `colcon build`, `ros2 launch`, Franka real robot run.

Main development should happen in:

- `~/projects/gp_torque_compensation`

Do not edit the same branch independently in both the Windows desktop repo and the WSL repo.

## 4. Current Environment Notes

Latest environment check summary:

- current path: `/home/dummd/projects/gp_torque_compensation`
- git top-level: `/home/dummd/projects/gp_torque_compensation`
- branch: `frozen_gp_spatial_trajectory`
- upstream: `origin/frozen_gp_spatial_trajectory`
- remote: `https://github.com/zdf6288/gp_torque_compensation.git`
- working tree status: clean
- OS / kernel: `Linux feizaodemon 6.6.87.2-microsoft-standard-WSL2 ... x86_64 GNU/Linux`
- user: `dummd`
- shell: `/bin/bash`
- `ROS_DISTRO=humble`
- ros2 path: `/opt/ros/humble/bin/ros2`
- colcon path: `/usr/bin/colcon`
- python version: `Python 3.10.12`
- python path: `/usr/bin/python3`
- no visible repo-local `build/`, `install/`, or `log/` artifacts were present at the time of the check.

Known environment risks:

- Python dependency check found missing modules:
  - `sklearn`
  - `torch`
- `.bashrc` automatically sources other ROS2 workspaces:
  - `~/ws_moveit/install/setup.bash`
  - `~/impl_course/ros2_ws/install/setup.bash`

These additional overlays can pollute ROS2 package resolution. Before build, test, or dependency-related work, remind the user to prefer a clean shell and source only the required environment.

Recommended clean shell check:

- `source /opt/ros/humble/setup.bash`
- `cd ~/projects/gp_torque_compensation`
- `echo $ROS_DISTRO`
- `python3 -c "import sklearn, torch; print('deps ok')"`

If dependencies are missing, report them. Do not install dependencies unless the user explicitly asks.

Do not modify `.bashrc` unless the user explicitly asks.

## 5. Workspace and Build Scope

Current workspace structure:

- repo root contains multiple ROS2 packages and directories, including `franka_ros2/`, `custom_msgs/`, and `new_structure/`.
- `new_structure` itself contains multiple packages, including `py_controllers`, `new_bringup`, and `cpp_relayer`.

Build scope guidance:

- If only developing current Stage 1, building from `new_structure` may be sufficient.
- If `franka_ros2` and `custom_msgs` are needed, building from repo root may be required.
- Do not assume a permanent build location until the user confirms the intended scope.
- Do not automatically delete `build/`, `install/`, or `log/`.
- Do not run `rm -rf build install log` unless the user explicitly asks and the current path has been re-confirmed.

## 6. Current Project Understanding

Friend note:

> 现在不需要 server 了，全写在 impedance controller 里了。

Current interpretation:

- Main GP prediction / update logic is in `cartesian_impedance.py`.
- `gp_server.py` may still be launched, but its content may be mostly commented or no longer the main control path.
- Do not assume a real cloud GP server is participating in the main control loop.
- `small GP` and `big/cloud-like GP` both run locally inside the impedance controller.
- The current `cloud` branch is more like a locally simulated cloud-like prediction branch than a remote server.
- Do not treat `gp_server.py` as the primary control logic entry point.
- Do not design Stage 1 main changes around `gp_server.py` unless the user explicitly requests it.

## 7. Key Files

Important files:

- `new_structure/py_controllers/py_controllers/cartesian_impedance.py`
- `new_structure/py_controllers/py_controllers/gp_server.py`
- `new_structure/py_controllers/py_controllers/trajectory_publisher.py`
- `new_structure/py_controllers/py_controllers/trajectory_eclipse_publisher.py`
- `new_structure/gp/build_dataset_no_filter.py`
- `new_structure/gp/train_gp_hdimensional.py`
- `new_structure/py_controllers/launch/cartesian_impedance_launch.py`
- `new_structure/new_bringup/config/controllers.yaml`

Stage 1 should primarily focus on:

- `new_structure/py_controllers/py_controllers/cartesian_impedance.py`
- `new_structure/py_controllers/launch/cartesian_impedance_launch.py`, only if needed.

Stage 1 should not modify:

- `trajectory_publisher.py`
- `trajectory_eclipse_publisher.py`
- `cpp_relayer.cpp`
- Franka hardware interface
- large parts of `controllers.yaml`
- large parts of GP training scripts
- GP model files
- dataset files

## 8. Current GP Mechanism

`cartesian_impedance.py` currently contains local GP model containers:

- `self.gp_models_small = {}`
- `self.gp_models_big = {}`

Both model sets run locally in the controller.

Current GP model loading may still use a hardcoded path:

- `_load_gp_models("./new_structure/gp/gp_models")`

Current GP input is 14 dimensional:

- `x_full = [q1..q7, dq_des_joint1..dq_des_joint7]`
- equivalently: `q + dq_des_joint`

There may be an older 21-dimensional comment version:

- `q + dq_des_joint + ddq_des_joint`

Current enabled path is understood to be 14-dimensional.

Current GP target / residual torque:

- `tau_residual = tau_measured - tau - gravity_measured`

Offline dataset builder equivalent:

- `Y = tau_measured - gravity - tau_cmd`

## 9. Current Key Limitations

Do not assume the project is ready for a complete frozen GP compensation tracking-error comparison yet.

Current limitations:

1. GP compensation may not actually enter final `tau`.
2. GP may still online update during testing.
3. GP model path may not be parameterized.
4. Frozen GP evaluation launch/controller switches are still missing.

Latest code-state understanding:

- hardcoded `gp_model_dir`
- online update likely enabled
- compensation not safely parameterized
- Stage 1 controls not yet implemented

## 10. Experiment Goal

Overall experiment goal:

- Compare how different training trajectories affect frozen GP torque compensation.

Core workflow:

1. Collect GP training data using different training trajectories.
2. Train separate GP models.
3. Freeze GP during test; do not online update.
4. Use the exact same test trajectory to compare tracking error.

Example experiment groups:

- `E0`: no GP baseline
- `E1`: GP trained on planar circle
- `E2`: GP trained on spatial / tilted / 3D trajectory

Final comparison metrics:

- RMS tracking error
- max tracking error
- z-axis RMSE
- tau smoothness
- y_hat magnitude
- control dt mean/std
- whether vibration occurs
- whether Franka reflex stop is triggered

## 11. Recommended Stage Plan

Recommended order:

1. `Stage 1`: Add Minimal Frozen GP Experiment Controls
2. `Stage 2`: Validate no-GP / frozen GP / compensation switch using original planar circle
3. `Stage 3`: Use existing tilted trajectory or add spatial trajectory for richer training
4. `Stage 4`: Run formal 7DoF spatial / 3D circle experiments

Current intended work:

- `Stage 1: Add Minimal Frozen GP Experiment Controls`

Do not simultaneously make large GP changes and large trajectory changes at the beginning.

## 12. Stage 1 Requirements

Stage 1 goal:

- Add minimal experiment control parameters in `cartesian_impedance.py` and, only if necessary, the launch file.

Required parameters:

- `gp_online_update_enabled`
- `gp_model_dir`
- `gp_compensation_enabled`
- `gp_compensation_source`
- `gp_compensation_scale`
- `gp_compensation_clip_nm`

Recommended defaults:

- `gp_online_update_enabled=True`
- `gp_model_dir="./new_structure/gp/gp_models"`
- `gp_compensation_enabled=False`
- `gp_compensation_source="local"`
- `gp_compensation_scale=0.1`
- `gp_compensation_clip_nm=0.5`

Stage 1 must satisfy:

1. Default behavior remains the same as the current code.
2. GP compensation is not enabled by default.
3. When `gp_online_update_enabled=False`, main-path `_gp_predict_and_update()` calls must not `add_point`.
4. `y_hat` may enter `tau` only when `gp_compensation_enabled=True`.
5. Compensation must have `scale` and `clip`.
6. `gp_model_dir` must allow selecting different GP model directories.
7. `gp_compensation_source` must support `local`, `cloud`, and `combined`.
8. Do not modify `trajectory_publisher.py`.
9. Do not modify `cpp_relayer.cpp`.
10. Do not modify the Franka hardware interface.
11. Add enough logging to confirm whether a run is frozen or online, and whether compensation is ON or OFF.
12. Do not enable any safety-relevant feature by default.

## 13. Real Robot Safety Rules

Always follow these rules:

1. Do not suggest direct large changes to torque command.
2. Do not enable GP compensation by default.
3. GP compensation must have a clip.
4. GP compensation must start with a very small scale.
5. If vibration, abnormal sound, reflex stop, or obvious trajectory abnormality occurs, stop immediately with `Ctrl+C`.
6. 100 Hz previously showed obvious vibration; do not treat 100 Hz as the current main experiment direction.
7. Prefer 50 Hz validation first, then consider 75 Hz.
8. Do not skip Franka web interface joint unlock / FCI activation / shutdown workflow.
9. Do not run real robot launch commands unless the user explicitly requests it.
10. Do not automatically send `/effort_command`.
11. Do not modify the Franka hardware interface.
12. Do not delete safety clamp, clip, rate limit, or abnormality-checking logic.
13. Do not enable any safety-relevant behavior by default.

High-risk changes require extra caution and a read-only self-review recommendation after implementation when they touch:

- `new_structure/py_controllers/py_controllers/cartesian_impedance.py`
- `_apply_gp_compensation`
- `tau_final`
- controller callback logic
- `new_structure/new_bringup/config/controllers.yaml`
- `franka.launch.py`
- `new_structure/py_controllers/launch/cartesian_impedance_launch.py`
- `trajectory_publisher.py`
- `cpp_relayer`
- real robot launch/config paths

Codex must never:

- enable active historical compensation by default
- add `gp_compensation_source:=historical` as an active path without an explicit separate task
- remove clip, finite checks, or safety gates
- increase `gp_compensation_clip_nm` unless explicitly requested as a separate high-risk task
- make historical or soft-fusion compensation enter `tau_final` unless explicitly requested in a separate high-risk task

Offline-only scripts may read CSV, NPZ, logs, build DBs, evaluate support coverage, and generate reports. They must not import ROS, run real robot commands, send `/effort_command`, or launch robot/controller processes. Offline-only tasks must not modify controller, launch, or config files unless the user explicitly scopes those files in the task.

## 14. Commands That Must Not Be Run Automatically

Do not run these unless the user explicitly asks and the path/context is verified:

- `rm -rf`
- `git reset --hard`
- `git clean -fd`
- real robot `ros2 launch`
- any command that directly drives a Franka robot
- any command that sends `/effort_command`

Do not automatically `commit` or `push`.

## 15. Codex Behavior Rules

Codex must:

- Explain in Chinese by default.
- Keep commands, paths, file names, function names, class names, and parameter names in English.
- State a brief plan before modifying code.
- List changed files after modifying code.
- Explain whether build / test is needed after modifying code.
- Avoid automatic `push`.
- Avoid automatic `commit` unless the user explicitly asks.
- Ask for confirmation before switching branches.
- Follow the branch policy in Section 2 instead of assuming only `frozen_gp_spatial_trajectory` is valid.
- Stop and report if the current path is under `/mnt/c/...GP-TORQUE-COMPENSATION...`.
- Report a dirty working tree before editing.
- Prefer minimal patches.
- Avoid large refactors.
- Avoid changing unrelated modules.
- Avoid enabling experiment features by default.
- Avoid installing dependencies unless explicitly requested.
- Avoid modifying `.bashrc` unless explicitly requested.
- Avoid assuming the current shell environment is clean.

For every task, Codex should report:

1. `pwd`
2. branch
3. dirty status before editing
4. whether files were modified
5. modified file list
6. validation commands run
7. whether unrelated dirty files were left untouched
8. whether `commit` or `push` was performed; default is no unless the user explicitly asks

## 16. Suggested Checks Before Each Task

Reference commands:

- `pwd`
- `git branch --show-current`
- `git status --short`
- `git remote -v`
- `git rev-parse --show-toplevel`
- `echo $ROS_DISTRO`
- `which ros2`
- `which colcon`
- `grep -n "update_rate" new_structure/new_bringup/config/controllers.yaml`
- `grep -n "_gp_predict_and_update" new_structure/py_controllers/py_controllers/cartesian_impedance.py`
- `grep -n "_load_gp_models" new_structure/py_controllers/py_controllers/cartesian_impedance.py`
- `grep -n "add_point" new_structure/py_controllers/py_controllers/cartesian_impedance.py`
- `grep -n "tau = tau" new_structure/py_controllers/py_controllers/cartesian_impedance.py`

## 17. Prompt Output Preference

When the user asks to generate a Codex prompt, Claude Code prompt, GitHub Copilot prompt, or another AI coding assistant task prompt:

- Default output should be one complete copyable Markdown-style prompt block.
- The format should generally start with `# Codex Prompt: ...`.
- Do not insert explanation text in the middle of the prompt.
- Do not split the prompt into multiple small code blocks.
- Unless the user explicitly asks for explanation, prioritize one complete prompt that can be pasted into Codex at once.
- To avoid copy splitting, avoid using triple-backtick code blocks inside the prompt content.
- If commands or code need to be listed, use normal lists, indented text, or inline code.
