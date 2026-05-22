# Stage 4 Roadmap: GP Training Trajectory Comparison

## 1. Purpose

This document defines the Stage 4 roadmap for `gp_torque_compensation`.

Stage 4 is not just a simple comparison between:

- `planar circle, GP OFF`
- `planar circle, GP ON`
- `3D spatial circle, GP OFF`
- `3D spatial circle, GP ON`

Those runs are useful only as early feasibility checks or sanity checks.

The formal Stage 4 goal is to compare how different GP training trajectories affect frozen GP torque compensation on the same formal test trajectory.

Stage 4's formal comparison is not a GP ON/OFF comparison across different test trajectories. It first collects the same number of GP training points, or a similar and explicitly recorded number of GP training points, from different training trajectories. It then trains separate frozen GP models and tests those frozen models on the exact same formal test trajectory. This is required so the observed difference can be attributed to training trajectory richness rather than test trajectory or training data size differences.

The intended final comparison is:

- `strict_no_gp_test`
- `frozen_gp_planar_train_scale03_test`
- `frozen_gp_spatial_train_scale03_test`

The central question is whether richer `spatial / tilted` training data improves GP prediction and torque compensation for a 7DoF / spatial test trajectory compared with a GP trained only on a `planar circle`.

The `GP_planar` and `GP_spatial` training trajectories may be different, but the selected GP training point counts should be equal or explicitly matched as closely as possible. If raw runs have different lengths, the training datasets should be cropped to the same number of points, or the point count difference must be recorded and treated as a caveat.

## 2. Current State

The project is currently on the `frozen_gp_spatial_trajectory` branch.

Stage 3A has completed initial real-robot feasibility validation. A `gp_prediction_enabled` patch may be under review or close to commit. That patch is part of Stage 4A preparation and should remain narrowly scoped.

Current understanding:

- GP prediction / update logic is mainly inside `new_structure/py_controllers/py_controllers/cartesian_impedance.py`.
- The `small GP` and `big/cloud-like GP` paths are local controller-side paths.
- `gp_server.py` should not be treated as the main controller path unless explicitly reintroduced.
- Stage 4 should avoid large controller, trajectory, GP model selection, and nullspace changes in the same patch.

Completed or basically verified work:

- Stage 0: current DoF / logging / GP ON-OFF structure understanding.
- Stage 1: low-risk spatial trajectory implementation.
- Stage 2: logging and offline metrics pipeline.
- Stage 3A: spatial trajectory + `scale03` initial real-robot validation.

## 3. What Stage 3A Proved and Did Not Prove

Stage 3A proved that a low-risk spatial / z-modulated trajectory can run on the real robot and produce usable data.

Stage 3A run summary:

- `comp_off fullrun`: 3749 pts.
- `gp_on_scale03 fullrun`: 3749 pts.
- `gp_on_scale05 partial`: 2024 pts.
- `gp_on_scale07 nearfull`: 3103 pts.
- `gp_on_scale03` reduced tau residual RMS on all 7 joints compared with `comp_off`.
- Tracking was comparable between `comp_off` and `gp_on_scale03`.
- Clip proxy ratio was 0.

Stage 3A also verified that the current task-space logging and offline analysis are usable:

- Task-space logging includes `x_actual/y_actual/z_actual` and `x_desired/y_desired/z_desired`.
- CSV logs include `tau`, `q`, `dq`, `dq_des_joint`, `ddq_des_joint`, and `y_hat`.
- Offline analysis can compute 3D tracking error, tau residual RMS, and clip proxy metrics.
- Existing checks cover NaN / inf, required columns, column matching, and estimated Hz.

Stage 3A did not prove final Stage 4 conclusions.

It should be treated as feasibility validation, not robust repeated validation. It should not be used to claim that spatial-trained GP is better than planar-trained GP, because the formal training-trajectory comparison has not been run yet.

Partial or near-full runs such as `gp_on_scale05 partial` and `gp_on_scale07 nearfull` should be recorded as useful observations, but they should not support final formal conclusions.

## 4. Final Stage 4 Experimental Question

The final Stage 4 experimental question is:

Can a frozen GP model trained on richer `spatial / tilted` trajectory data produce better residual-related prediction and torque compensation on the same 7DoF / spatial formal test trajectory than a frozen GP model trained only on `planar circle` data?

The required formal structure is:

1. Collect GP training data with a `planar circle` trajectory.
2. Record raw and selected / cropped planar training point counts.
3. Collect GP training data with a `spatial / tilted` trajectory.
4. Record raw and selected / cropped spatial training point counts.
5. Match the selected training point count for `GP_planar` and `GP_spatial` where possible.
6. Train `GP_planar` and `GP_spatial` with the same preprocessing pipeline, sampling-rate expectation, feature selection, target definition, and GP hyperparameters.
7. Freeze both GP models for testing.
8. Run the exact same formal test trajectory under:
   - strict no-GP baseline;
   - frozen GP trained on planar trajectory;
   - frozen GP trained on spatial / tilted trajectory.
9. Compare tracking error and residual-related metrics.
10. Conclude only from the formal comparison, not from early feasibility runs.

If `GP_spatial` clearly outperforms `GP_planar` on the same formal test trajectory, the result supports the claim that richer spatial training improves 7DoF generalization. If it does not clearly outperform `GP_planar`, the result should be reported honestly.

Do not directly compare `planar GP ON` and `spatial GP ON` tracking error on different test trajectories to make the final conclusion. That comparison can be a feasibility / sanity check only. The formal conclusion must come from frozen models tested on the same formal test trajectory.

Do not write the conclusion before the data supports it.

## 5. Stage 4 Roadmap

### Stage 4A: `gp_prediction_enabled` / strict no-GP capability preparation

Goal:

- Prepare a strict no-GP baseline mode.
- Support disabling GP prediction, online update, and compensation at the same time.

Strict no-GP parameter semantics:

- `gp_prediction_enabled=false`
- `gp_online_update_enabled=false`
- `gp_compensation_enabled=false`

This stage is not intended to improve controller performance. It exists to make the final baseline definition rigorous.

This patch should only modify:

- `new_structure/py_controllers/py_controllers/cartesian_impedance.py`
- `new_structure/py_controllers/launch/cartesian_impedance_launch.py`

It should not modify:

- `trajectory_publisher.py`
- `trajectory_eclipse_publisher.py`
- `gp_server.py`
- `cpp_relayer.cpp`
- `controllers.yaml`
- `franka.launch.py`
- hardware interface
- GP model files
- data / outputs

### Stage 4B: collect planar circle GP training data

Purpose:

- Use a `planar circle` trajectory to collect GP training data.
- Train `GP_planar` from this dataset.

Suggested run semantics:

- `trajectory_type=planar_circle`
- `gp_prediction_enabled=true`
- `gp_online_update_enabled=true`
- `gp_compensation_enabled=false`

Principles:

- Training data collection may use online update / `add_point`.
- GP compensation should not enter the final `tau`.
- Training data should come from nominal / no-compensation behavior as much as possible.
- Record the raw planar training point count.
- Record the selected / cropped planar training point count.
- Before training, compare this selected count with the spatial training dataset.
- Prefer using the same number of selected GP training points as `GP_spatial`.

Suggested directories:

- `data/stage4/train/planar_circle/`
- `models/stage4/gp_planar_circle/`

### Stage 4C: collect spatial / tilted trajectory GP training data

Purpose:

- Use a `spatial / tilted` trajectory to collect GP training data.
- Train `GP_spatial` from this dataset.

Suggested run semantics:

- `trajectory_type=spatial_or_tilted`
- `gp_prediction_enabled=true`
- `gp_online_update_enabled=true`
- `gp_compensation_enabled=false`

Principles:

- Training data collection should not enable final compensation.
- The goal is to cover richer 3D / 7DoF excitation.
- Record the raw spatial / tilted training point count.
- Record the selected / cropped spatial / tilted training point count.
- Before training, compare this selected count with the planar training dataset.
- Prefer using the same number of selected GP training points as `GP_planar`.

Suggested directories:

- `data/stage4/train/spatial_tilted/`
- `models/stage4/gp_spatial_tilted/`

### Stage 4D: offline train two frozen GP models

Goals:

- Train `GP_planar` from planar training data.
- Train `GP_spatial` from spatial / tilted training data.
- Freeze both models for testing.

Test-stage requirement:

- `gp_online_update_enabled=false`

Before implementation, confirm:

- Whether the current project supports selecting different training datasets.
- Whether the current project supports saving / loading different GP models.
- Whether launch / runtime can specify different GP model paths.
- Whether both datasets use the same number of selected training points, or whether the point count difference is explicitly recorded.
- Whether both datasets use the same preprocessing pipeline.
- Whether both datasets use the same sampling-rate expectation, feature selection, and target definition.
- Whether both models use the same GP hyperparameters.
- Whether both models follow the same model save / load convention.

If any of these are unsupported, do a separate read-only review first, then make a minimal patch. Do not mix model selection changes with controller behavior, trajectory changes, or nullspace changes.

### Stage 4E: run strict no-GP baseline on the formal test trajectory

Goal:

- Collect the formal no-GP baseline on a fixed formal test trajectory.

Parameter semantics:

- `gp_prediction_enabled=false`
- `gp_online_update_enabled=false`
- `gp_compensation_enabled=false`

Principles:

- No GP prediction.
- No online update.
- No compensation in final `tau`.
- This run is the final comparison baseline.

### Stage 4F: run planar-trained frozen GP on the formal test trajectory

Goal:

- Load `GP_planar`.
- Test it on the exact same formal test trajectory.

Parameter semantics:

- `gp_prediction_enabled=true`
- `gp_online_update_enabled=false`
- `gp_compensation_enabled=true`
- `gp_compensation_scale=0.3`
- clip remains enabled.

Purpose:

- Evaluate how well a GP trained only on `planar circle` data generalizes to the 7DoF / spatial formal test trajectory.

The test trajectory must match Stage 4G in geometry, duration, sampling-rate expectation, controller configuration, GP compensation scale, and clip threshold.

### Stage 4G: run spatial-trained frozen GP on the formal test trajectory

Goal:

- Load `GP_spatial`.
- Test it on the exact same formal test trajectory.

Parameter semantics:

- `gp_prediction_enabled=true`
- `gp_online_update_enabled=false`
- `gp_compensation_enabled=true`
- `gp_compensation_scale=0.3`
- clip remains enabled.

Purpose:

- Evaluate whether `spatial / tilted` training data improves residual-related prediction and compensation on the 7DoF / spatial formal test trajectory.

The test trajectory must match Stage 4F in geometry, duration, sampling-rate expectation, controller configuration, GP compensation scale, and clip threshold.

### Stage 4H: unified offline comparison analysis

Final main comparison:

- `strict_no_gp_test`
- `frozen_gp_planar_train_scale03_test`
- `frozen_gp_spatial_train_scale03_test`

Main metrics:

- 3D tracking RMSE.
- Max tracking error.
- Per-axis tracking error.
- Tau residual RMS.
- Per-joint tau residual RMS change.
- GP output magnitude.
- Clip proxy ratio.
- Estimated Hz.
- NaN / inf / `columns_match`.

Interpretation rule:

- If spatial-trained GP outperforms planar-trained GP, the result supports the claim that richer spatial training improves 7DoF generalization.
- If spatial-trained GP does not clearly outperform planar-trained GP, record that directly.
- Do not force the conclusion.

### Stage 4I: paper figures and result summary

Suggested outputs:

- `docs/stage4_result_summary.md`
- `docs/stage4_experiment_log.md`
- `docs/stage4_data_inventory.md`
- `outputs/stage4_analysis/`
- `outputs/stage4_analysis/plots/`

Suggested figures and tables:

- Tracking error norm comparison.
- Tau residual RMS per joint.
- GP output magnitude.
- Clip proxy summary.
- Planar-trained vs spatial-trained comparison table.

## 6. Run Type Definitions

### 1. Training data collection

Use:

- Collect GP training points.

Typical parameters:

- `gp_prediction_enabled=true`
- `gp_online_update_enabled=true`
- `gp_compensation_enabled=false`

Meaning:

- Online update / `add_point` is allowed for collecting training data.
- Final torque should not be affected by GP compensation.
- The dataset should represent nominal / no-compensation behavior where possible.
- Raw and selected / cropped training point counts should be recorded.
- The selected training point count should be matched across `GP_planar` and `GP_spatial` where possible.

### 2. Frozen GP test

Use:

- Test a GP model that has already been trained.

Typical parameters:

- `gp_prediction_enabled=true`
- `gp_online_update_enabled=false`
- `gp_compensation_enabled=true`
- `gp_compensation_scale=0.3`
- clip enabled.

Meaning:

- The test stage no longer performs online update.
- The GP model is frozen.
- This run type allows comparison of different training trajectories through their frozen GP models.
- `GP_planar` and `GP_spatial` must be tested on the same formal test trajectory.
- The formal test trajectory should keep the same geometry, duration, sampling-rate expectation, controller configuration, GP compensation scale, and clip threshold.

### 3. Strict no-GP baseline

Use:

- Provide a formal baseline with no GP involvement.

Typical parameters:

- `gp_prediction_enabled=false`
- `gp_online_update_enabled=false`
- `gp_compensation_enabled=false`

Meaning:

- No GP prediction request.
- No online update.
- No compensation contribution to final `tau`.

## 7. Recommended Data and Model Layout

Recommended training data layout:

- `data/stage4/train/planar_circle/`
- `data/stage4/train/spatial_tilted/`

Recommended model layout:

- `models/stage4/gp_planar_circle/`
- `models/stage4/gp_spatial_tilted/`

Recommended formal test data layout:

- `data/stage4/test/strict_no_gp_test/`
- `data/stage4/test/frozen_gp_planar_train_scale03_test/`
- `data/stage4/test/frozen_gp_spatial_train_scale03_test/`

Recommended analysis output layout:

- `outputs/stage4_analysis/`
- `outputs/stage4_analysis/plots/`

Recommended documentation outputs:

- `docs/stage4_experiment_log.md`
- `docs/stage4_data_inventory.md`
- `docs/stage4_result_summary.md`

The data inventory should record raw point counts, selected / cropped point counts, dataset source paths, preprocessing choices, model output paths, and the loaded `gp_model_dir` used for each formal test run.

## 8. Metrics

Required metrics:

- Raw and selected / cropped training point counts for each GP model.
- 3D tracking RMSE.
- Max tracking error.
- Per-axis tracking RMSE.
- Tau residual RMS.
- Per-joint tau residual RMS.
- Per-joint tau residual RMS change relative to strict no-GP baseline.
- GP output magnitude.
- Clip proxy ratio.
- Estimated Hz.
- NaN / inf status.
- Required columns status.
- `columns_match` status across compared runs.

The final analysis should separate:

- Dataset size and preprocessing caveats.
- Tracking performance.
- Residual-related torque metrics.
- GP output magnitude and clipping behavior.
- Runtime / data quality caveats.

## 9. Safety and Scope Boundaries

Do not do the following in the current Stage 4 preparation:

- Do not directly jump to 100 Hz.
- Do not run no-clip GP-on tests.
- Do not run unlimited GP compensation.
- Do not use large trajectory amplitudes yet.
- Do not introduce a helix trajectory yet.
- Do not modify nullspace posture optimization yet.
- Do not add joint limit avoidance yet.
- Do not modify the Franka hardware interface.
- Do not modify the controller torque core without a separate focused reason.
- Do not mix trajectory, controller, GP model selection, and nullspace changes in one patch.
- Do not compare GP ON runs from different test trajectories as final evidence.
- Do not use partial / near-full runs to support final formal conclusions.
- Do not describe Stage 3A as robust repeated validation.

Real-robot safety reminders:

- GP compensation must stay disabled by default.
- Compensation must keep a clip.
- Compensation should start with a small scale such as `gp_compensation_scale=0.3`.
- If vibration, abnormal sound, reflex stop, or obvious trajectory abnormality occurs, stop immediately with `Ctrl+C`.
- Do not run real-robot launch commands unless explicitly requested.
- Do not send `/effort_command` automatically.

## 10. Engineering Caveats

Current project standard:

- One successful run that produces usable data can be enough to proceed to analysis.
- Clean shutdown is desirable but not required for every real-robot run.
- `communication_constraints_violation`, User Stop, and `rclpy` shutdown errors can be recorded as caveats.
- These caveats should not automatically invalidate a usable dataset.
- They also should not be used to claim that the system is fully stable or robustly validated.

Stage 3A should be described as initial feasibility validation.

Future formal Stage 4 results should clearly state:

- Which runs were full, partial, or near-full.
- Whether shutdown was clean.
- Whether any reflex stop, User Stop, or communication constraint issue occurred.
- Whether clipping occurred.
- Whether estimated Hz was acceptable.
- Whether required columns matched across compared logs.

## 11. Recommended Execution Order

Recommended order:

1. Finish Stage 4A `gp_prediction_enabled` / strict no-GP preparation.
2. Review whether dataset selection and GP model save / load paths are sufficient for Stage 4D.
3. If needed, make a minimal model-path / dataset-selection patch separately.
4. Collect planar circle GP training data for Stage 4B.
5. Collect spatial / tilted GP training data for Stage 4C.
6. Record raw point counts and choose equal or explicitly matched selected point counts.
7. Train `GP_planar` and `GP_spatial` offline for Stage 4D using the same preprocessing and GP hyperparameters.
8. Run strict no-GP baseline on the formal test trajectory for Stage 4E.
9. Run frozen `GP_planar` on the same formal test trajectory for Stage 4F.
10. Run frozen `GP_spatial` on the same formal test trajectory for Stage 4G.
11. Run unified offline comparison analysis for Stage 4H.
12. Write Stage 4 result summary, experiment log, data inventory, and paper figures for Stage 4I.

Codex should handle:

- Narrow code review.
- Minimal patches.
- Documentation.
- Offline analysis scripts and summaries.

Real robot sessions should handle:

- Dataset collection.
- Formal baseline and frozen GP tests.
- Safety observation and run notes.

Offline analysis should handle:

- Data validation.
- Metrics.
- Plots.
- Final comparison tables.

## 12. Open Checks Before Real-Robot Stage 4

Before real-robot Stage 4, check:

- Whether `gp_prediction_enabled` is merged and works as intended.
- Whether strict no-GP mode truly avoids GP prediction and online update.
- Whether `gp_online_update_enabled=false` prevents main-path `add_point`.
- Whether `gp_compensation_enabled=false` keeps `y_hat` out of final `tau`.
- Whether clip and scale are still active when GP compensation is enabled.
- Whether launch / runtime can select `gp_model_dir`.
- Whether the current project supports saving two different GP models.
- Whether separate `GP_planar` and `GP_spatial` model directories can be saved and loaded.
- Whether launch / runtime can specify the GP model path for each test run.
- Whether the controller logs or startup output record the loaded model name / model path.
- Whether the project can export or confirm training point count.
- Whether training scripts can select the intended dataset.
- Whether planar and spatial training datasets can be cropped or selected to the same number of GP training points.
- Whether preprocessing, feature selection, target definition, and GP hyperparameters can be kept identical across both models.
- Whether the formal test trajectory is fixed and identical across test runs.
- Whether the shell environment is clean enough for ROS2 package resolution.
- Whether `sklearn` and `torch` are available if needed for training or analysis.

If model selection is unsupported, do a separate read-only review and then a minimal patch. Do not mix model selection changes with controller behavior, trajectory changes, nullspace changes, data changes, or outputs.

Recommended clean shell reminder:

- `source /opt/ros/humble/setup.bash`
- `cd ~/projects/gp_torque_compensation`
- `echo $ROS_DISTRO`
- `python3 -c "import sklearn, torch; print('deps ok')"`

Do not install dependencies unless explicitly requested.

## 13. Summary

Stage 4 should answer a training-trajectory question, not just a GP OFF / GP ON question.

The final comparison must use one fixed formal test trajectory and compare:

- strict no-GP baseline;
- frozen GP trained on planar data;
- frozen GP trained on spatial / tilted data.

The two frozen GP models should be trained from equal or explicitly matched GP training point counts wherever possible. The preprocessing pipeline, sampling-rate expectation, feature selection, target definition, and GP hyperparameters should remain the same across both models.

Training data collection may use online update, but frozen GP tests must disable online update. Strict no-GP baseline must disable prediction, online update, and compensation.

The project should proceed with small, verifiable steps: finish strict no-GP support, confirm model selection, collect separate training datasets, train separate frozen models, run the same formal test trajectory, then analyze with consistent metrics.

Stage 3A is a useful feasibility result. It is not the final formal Stage 4 conclusion.
