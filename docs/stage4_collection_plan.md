# Stage 4 Collection Plan: Pre-Real-Machine Execution Checklist

## 1. Purpose

This document is a pre-real-machine execution checklist for Stage 4.

It is not a paper conclusion document. Its goal is to make the real-machine Linux session boring and controlled: no last-minute code edits, no accidental mode mixing, no wrong branch, and no confusion between training data collection and frozen GP test runs.

Stage 4 should be executed as:

- different training trajectories
- matched training datasets
- frozen GP models
- same formal test trajectory
- tracking comparison under controlled conditions

## 2. Current Repository State Before Real-Machine Work

Before going to the real machine, the repository should already include these key preparations:

- `gp_prediction_enabled` strict no-GP gating has been added.
- The GP prediction / update loop now covers joint 1..7.
- The Stage 4 GP training trajectory roadmap has been completed.
- The matched dataset preparation script has been completed.
- `docs/stage4_dataset_preparation.md` exists.
- WSL should be git clean before transferring or pulling on the real-machine Linux PC.
- The real-machine Linux PC should sync to the same branch and commit as WSL.

Expected branch:

- `frozen_gp_spatial_trajectory`

Important current commits include:

- `3f84973 Add Stage 4 matched GP dataset preparation script`
- `7c276cf Fix GP prediction loop to include joint 7`
- `9bdd02a Add Stage 4 GP training trajectory roadmap`
- `0edacf5 Add gp_prediction_enabled strict no-GP gating`
- `d29bac1 Add Stage 3A result summary`
- `4affe5c Add Stage 3A offline analysis workflow`
- `328ebcc Add Stage 3A z-modulated spatial trajectory`

## 3. Stage 4 Experimental Logic

Stage 4 is not a simple comparison of GP ON / GP OFF on different test trajectories.

The formal logic is:

- Training trajectory can be different.
- Training point count should be as equal / matched as possible.
- Each matched dataset trains one frozen GP model.
- The formal test trajectory must be the same for all comparison groups.
- During the formal frozen test, GP online update must be disabled.
- No `add_point` should happen during frozen GP test runs.
- Tracking error from different test trajectories should not be directly used as the final conclusion.

The intended comparison is:

- `planar circle` training data -> matched dataset -> `GP_planar`
- `spatial / tilted trajectory` training data -> matched dataset -> `GP_spatial`
- same formal test trajectory for:
  - `strict_no_gp_test`
  - `frozen_gp_planar_train_scale03_test`
  - `frozen_gp_spatial_train_scale03_test`

## 4. Run Type Definitions

Training data collection:

- `gp_prediction_enabled=true`
- `gp_online_update_enabled=true`
- `gp_compensation_enabled=false`

Purpose:

- Use online GP prediction / update path to collect training points.
- Do not let GP compensation affect final torque.

Strict no-GP baseline:

- `gp_prediction_enabled=false`
- `gp_online_update_enabled=false`
- `gp_compensation_enabled=false`

Purpose:

- Disable GP prediction, online update, and compensation.
- Produce the strict baseline for the same formal test trajectory.

Frozen GP test:

- `gp_prediction_enabled=true`
- `gp_online_update_enabled=false`
- `gp_compensation_enabled=true`
- `gp_compensation_scale=0.3`
- clip enabled

Purpose:

- Load a pre-trained frozen GP model.
- Apply small, clipped GP compensation.
- Do not update the model online.

## 5. Pre-Real-Machine WSL Checklist

Run from WSL:

- `cd ~/projects/gp_torque_compensation`
- `git status --short`
- `git log --oneline -8`
- `python3 scripts/prepare_stage4_gp_dataset.py --help`

Confirm:

- The latest expected commits are present.
- The branch is `frozen_gp_spatial_trajectory`.
- `git status --short` is clean or only contains intentional documentation changes.
- No controller, launch, trajectory, config, data, output, or model edits are needed on the real-machine Linux PC.
- The planned data directories are known before going to the robot.
- The run names are written down before launching anything.
- The intended `gp_model_dir` paths for later frozen tests are known.

## 6. Real-Machine Linux Preparation Checklist

On the real-machine Linux PC:

- Pull the latest `frozen_gp_spatial_trajectory` branch.
- Confirm the branch is exactly `frozen_gp_spatial_trajectory`.
- Confirm the commit matches WSL.
- Build the workspace if needed.
- Source the intended ROS2 / workspace environment.
- Confirm there is no unexpected overlay pollution if package resolution looks strange.
- Confirm Franka web interface readiness.
- Confirm FCI activation.
- Confirm brakes / joint unlock state.
- Confirm emergency stop and User Stop process.
- Confirm launch logs and CSV output paths before running.

Safety reminders:

- Do not run GP compensation without clip.
- Do not run unlimited GP-on compensation.
- If vibration, abnormal sound, reflex stop, communication errors, or obvious trajectory abnormality occurs, stop the launch.
- If a run fails, do not expand the experiment scope on site to "try more things".
- Do not make temporary controller edits on the robot PC unless a separate decision is made.

## 7. Stage 4B: Planar Training Data Collection

Goal:

- Collect `planar_circle` training data.
- Use it later to train `GP_planar`.

Suggested mode name:

- `train_planar_circle`

Suggested directory:

- `data/stage4/train/planar_circle/`

Parameter semantics:

- `trajectory_type=planar_circle`
- `z_amplitude=0.0`
- `gp_prediction_enabled=true`
- `gp_online_update_enabled=true`
- `gp_compensation_enabled=false`

Execution notes:

- Online update / `add_point` is used to collect training points.
- Final torque should not be affected by GP compensation.
- Save launch logs and CSV files.
- If the data is usable, the run can be accepted even if shutdown is not perfectly clean.
- Do not mix this run with frozen GP test settings.

## 8. Stage 4C: Spatial / Tilted Training Data Collection

Goal:

- Collect spatial / tilted training data.
- Use it later to train `GP_spatial`.

Suggested mode name:

- `train_spatial_tilted`

Suggested directory:

- `data/stage4/train/spatial_tilted/`

Parameter semantics:

- `trajectory_type=spatial` or the current project spatial / z-modulated trajectory type
- use conservative small amplitude
- `gp_prediction_enabled=true`
- `gp_online_update_enabled=true`
- `gp_compensation_enabled=false`

Execution notes:

- Do not enable final GP compensation.
- The goal is to collect richer 3D / 7DoF excitation training data.
- Do not directly jump to large-amplitude trajectory, `helix`, or `100 Hz`.
- Save launch logs and CSV files.
- Do not mix this run with frozen GP test settings.

## 9. Post-Collection Data Transfer to WSL

Suggested WSL / PC directories:

- `data/stage4/train/planar_circle/`
- `data/stage4/train/spatial_tilted/`
- `data/stage4/logs/`

Transfer rules:

- Keep original CSV files.
- Keep launch logs.
- Do not overwrite Stage 3A data.
- Do not rename files in a way that loses run order or run type.
- Do not commit generated data unless the repo policy explicitly allows it.
- After transfer, run quick CSV checks before preparing matched datasets.

## 10. Dataset Matching Workflow

Use the existing script:

- `scripts/prepare_stage4_gp_dataset.py`

Dry-run command:

```bash
python3 scripts/prepare_stage4_gp_dataset.py --planar-pattern "data/stage4/train/planar_circle/*.csv" --spatial-pattern "data/stage4/train/spatial_tilted/*.csv" --out-dir data/stage4/datasets --dry-run
```

Real run command:

```bash
python3 scripts/prepare_stage4_gp_dataset.py --planar-pattern "data/stage4/train/planar_circle/*.csv" --spatial-pattern "data/stage4/train/spatial_tilted/*.csv" --out-dir data/stage4/datasets
```

Expected outputs:

- `GP_planar_matched.npz`
- `GP_spatial_matched.npz`
- `stage4_dataset_manifest.json`

Checklist:

- Confirm both datasets use the same feature definition.
- Confirm both datasets use the same target definition.
- Confirm point count is equal / matched.
- Confirm CSV NaN / inf / columns / estimated Hz checks pass.

## 11. Offline GP Training Workflow

Training commands:

```bash
python3 new_structure/gp/train_gp_hdimensional.py --data data/stage4/datasets/GP_planar_matched.npz --out-dir data/stage4/models/GP_planar --joint all
```

```bash
python3 new_structure/gp/train_gp_hdimensional.py --data data/stage4/datasets/GP_spatial_matched.npz --out-dir data/stage4/models/GP_spatial --joint all
```

Training requirements:

- `GP_planar` and `GP_spatial` should use the same feature definition.
- `GP_planar` and `GP_spatial` should use the same target definition.
- Use the same hyperparameters unless there is a documented reason not to.
- Record model path and training manifest.
- During test, switch models through `gp_model_dir`.
- Do not change controller logic for model switching.

## 12. Formal Test Runs After Model Training

Final formal test groups:

- `strict_no_gp_test`
- `frozen_gp_planar_train_scale03_test`
- `frozen_gp_spatial_train_scale03_test`

All three must use:

- same formal test trajectory
- same controller config
- same duration
- same `gp_compensation_scale=0.3` for frozen GP tests
- same clip threshold for frozen GP tests
- `gp_online_update_enabled=false`

Suggested semantics:

- `strict_no_gp_test`
  - `gp_prediction_enabled=false`
  - `gp_online_update_enabled=false`
  - `gp_compensation_enabled=false`
- `frozen_gp_planar_train_scale03_test`
  - `gp_prediction_enabled=true`
  - `gp_online_update_enabled=false`
  - `gp_compensation_enabled=true`
  - `gp_model_dir=data/stage4/models/GP_planar`
  - `gp_compensation_scale=0.3`
  - clip enabled
- `frozen_gp_spatial_train_scale03_test`
  - `gp_prediction_enabled=true`
  - `gp_online_update_enabled=false`
  - `gp_compensation_enabled=true`
  - `gp_model_dir=data/stage4/models/GP_spatial`
  - `gp_compensation_scale=0.3`
  - clip enabled

Do not compare final tracking conclusions across different formal test trajectories.

## 13. Data and Naming Conventions

Recommended directory structure:

- `data/stage4/train/planar_circle/`
- `data/stage4/train/spatial_tilted/`
- `data/stage4/datasets/`
- `data/stage4/models/GP_planar/`
- `data/stage4/models/GP_spatial/`
- `data/stage4/test/strict_no_gp/`
- `data/stage4/test/gp_planar_scale03/`
- `data/stage4/test/gp_spatial_scale03/`
- `outputs/stage4_analysis/`

Recommended run naming:

- `train_planar_circle`
- `train_spatial_tilted`
- `strict_no_gp_test`
- `frozen_gp_planar_train_scale03_test`
- `frozen_gp_spatial_train_scale03_test`

Recommended metadata to preserve:

- branch
- commit hash
- run type
- trajectory type
- `gp_prediction_enabled`
- `gp_online_update_enabled`
- `gp_compensation_enabled`
- `gp_compensation_source`
- `gp_compensation_scale`
- `gp_compensation_clip_nm`
- `gp_model_dir`
- launch log path
- CSV path

## 14. Safety Boundaries

Do not do:

- no-clip GP-on
- unlimited GP compensation
- controller torque core changes
- nullspace changes
- hardware interface changes
- temporary `100 Hz` experiments
- temporary trajectory amplitude expansion
- training and frozen test mixed in one run
- direct large changes to torque command
- real-machine launch without reviewing run parameters

If anything abnormal happens:

- stop the launch
- preserve logs
- copy CSV if available
- do offline checks before the next risky run

## 15. Success Criteria

Minimum success standard:

- Planar training data has at least one usable run.
- Spatial training data has at least one usable run.
- Dataset matching dry-run passes.
- Matched `.npz` files can be generated.
- `GP_planar` can be trained.
- `GP_spatial` can be trained.
- Three formal test groups can be collected:
  - `strict_no_gp_test`
  - `frozen_gp_planar_train_scale03_test`
  - `frozen_gp_spatial_train_scale03_test`
- CSV NaN / inf / columns / estimated Hz checks pass.
- Clip proxy is not abnormal.
- Logs clearly show whether the run was online training, strict no-GP, or frozen GP test.

## 16. Caveats and Non-Goals

Engineering caveats:

- `communication_constraints_violation` can happen.
- User Stop can happen.
- `rclpy` shutdown errors can happen.
- A usable data run may still be accepted even if shutdown is noisy.

Non-goals for this stage:

- Do not claim fully stable / robust repeated validation.
- Do not make `scale05` or `scale07` the main line.
- Do not add nullspace redesign.
- Do not add joint limit avoidance redesign.
- Do not add `helix` as the main experiment.
- Do not move to `100 Hz` as the main experiment.
- Do not modify the controller torque core.
- Do not modify the Franka hardware interface.

## 17. Final Go / No-Go Checklist

- [ ] WSL repo is clean.
- [ ] True latest commit is confirmed.
- [ ] Real-machine branch is synced to `frozen_gp_spatial_trajectory`.
- [ ] Real-machine commit matches WSL.
- [ ] Franka web interface is ready.
- [ ] FCI / brakes / joint unlock process is confirmed.
- [ ] Emergency stop / User Stop process is known.
- [ ] Run parameters are reviewed before each launch.
- [ ] Logging paths are prepared.
- [ ] CSV output paths are prepared.
- [ ] No unbounded GP compensation is possible in the planned runs.
- [ ] Training data collection settings are separate from frozen GP test settings.
- [ ] After each run, data and logs are copied.
- [ ] Offline checks are done before the next risky run.
- [ ] No source code, launch, trajectory, config, data, output, or model files need to be edited on the robot PC.
