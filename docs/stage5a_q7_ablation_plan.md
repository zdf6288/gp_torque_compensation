# Stage 5A q7 / joint7 GP Offline Ablation Plan

## Scope

This is an offline-only ablation plan for q7 / joint7 GP decisions. It uses existing CSV, Stage 4 data, and existing GP tooling. It does not connect to Franka, does not run ROS2 launch, and does not modify controller, launch, config, GP compensation, or torque command logic.

The questions are:

1. Whether `joint_pos_7` / `joint_vel_7` help residual GP prediction.
2. Whether `tau_residual_*_7` should remain a GP output target.
3. How 14D support gate decisions differ from a 12D gate that excludes q7/dq7.
4. Whether planar and zmod data show different q7/dq7 behavior.
5. Whether there is offline evidence worth considering before any future real-time `tau7` compensation work.

This document deliberately does not authorize GP-on. A support pass or offline prediction improvement is not real-robot validation.

## Available Data

Current Stage 4 cross-trajectory data is available under `data/stage4/cross_traj/`:

- `A_no_gp_planar/usable_runs/*.csv`
- `B_no_gp_zmod/usable_runs/*.csv`
- `C_no_gp_zmod_heldout/usable_runs/*.csv`
- `datasets/GP_A_planar_train.npz`
- `datasets/GP_B_zmod_train.npz`
- `datasets/GP_C_zmod_heldout_eval.npz`
- `models/GP_A_planar_train/metadata.json`
- `models/GP_B_zmod_train/metadata.json`

The checked CSV headers include:

- `joint_pos_1..7`
- `joint_vel_1..7`
- `dq_des_joint_1..7`
- `ddq_des_joint_1..7`
- `tau_1..7`
- `tau_measured_1..7`
- `gravity_1..7`
- `tau_residual_1..7`
- `tau_residual_raw_1..7`
- `y_hat_1..7`, `y_hat_local_1..7`, `y_hat_cloud_1..7`

The checked `GP_A_planar_train` and `GP_B_zmod_train` metadata both report:

- `X_shape: [2000, 14]`
- `Y_shape: [2000, 7]`
- `feature_names: joint_pos_1..7 + joint_vel_1..7`
- `target_names: tau_residual_raw_1..7`
- per-joint `joint1..joint7` local/cloud model files exist

This means the current offline model artifacts can evaluate the existing 14D design, but they cannot answer a true 12D-input GP prediction question without retraining reduced-input models.

## Available Scripts

Relevant existing scripts:

- `scripts/validate_stage5_q7_support.py`
  - Offline q7 and 14D support validator.
  - Reads reference CSV or `metadata.json` `source_csv`.
  - Computes per-dimension min/max support checks for `joint_pos_1..7 + joint_vel_1..7`.

- `scripts/run_stage5_support_matrix.py`
  - Batches support validation over Stage 4 model and CSV pairs.
  - Already reports q7 pass, 14D pass, `worst_dimension`, and `blocking_reason`.

- `scripts/evaluate_stage4_cross_traj_residuals.py`
  - Builds combined Stage 4 datasets.
  - Trains/evaluates existing 14D frozen GP models.
  - Uses `X = joint_pos_1..7 + joint_vel_1..7`.
  - Uses `Y = tau_residual_raw_1..7`.
  - Reports per-joint RMSE, prediction span, target span, correlation, and constant prediction flags.

- `scripts/train_stage4_matched_frozen_gp.py`
  - Trains controller/validator-compatible per-joint GP pickles from `.npz`.
  - Currently validates `X.shape[1] == 14` and `Y.shape[1] == 7`.

- `new_structure/gp/train_gp_hdimensional.py`
  - Underlying GP training utility supports `x_dim=X.shape[1]`, but the Stage 4 wrapper currently enforces 14D.

## Current GP Input / Output Structure

The current Stage 4 cross-trajectory design is:

- Input: `joint_pos_1..7 + joint_vel_1..7`, 14D.
- Target: `tau_residual_raw_1..7`, 7 outputs trained as per-joint GP pickles.

The original high-dimensional design therefore includes q7/dq7 in the input and includes joint7 residual as an output target. However, this offline fact is separate from runtime torque compensation. It does not imply that `tau[6]` should be compensated in real time.

## Ablations That Can Be Done Immediately

### S1: 14D support gate vs 12D support gate

This can be done immediately because existing CSVs contain the full 14D joint-space columns.

Compare:

- `14d`: `joint_pos_1..7 + joint_vel_1..7`
- `12d_without_q7_dq7`: same columns excluding `joint_pos_7` and `joint_vel_7`

Outputs:

- Which pairs fail in 14D but pass in 12D.
- Which pairs fail in 14D and still fail in 12D.
- Whether q7/dq7 is the only blocking support dimension for any pair.
- Whether other dimensions, such as `joint_pos_4`, remain blockers after q7/dq7 removal.

Interpretation boundary:

- 12D pass does not authorize GP-on.
- Removing q7/dq7 from a support gate is an ablation, not a safety decision.
- If the model input is 14D, real GP support should still respect 14D unless a model is retrained without q7/dq7.

### S2: Existing 14D model inventory and residual target inspection

This can be done from existing metadata/CSV:

- Confirm whether `joint7_local.pkl` and `joint7_cloud.pkl` exist.
- Confirm whether `tau_residual_raw_7` has nonzero variance in train and held-out CSVs.
- Confirm whether existing Stage 4 evaluation flags joint7 predictions as constant or unstable.

This is inspection only. It does not isolate the causal value of q7/dq7 input because existing models were trained with 14D inputs.

## Ablations That Need Retraining

### P1: 14D-input vs 12D-input residual prediction

Train:

- `GP_A_planar_14D`
- `GP_A_planar_12D_no_q7dq7`
- `GP_B_zmod_14D`
- `GP_B_zmod_12D_no_q7dq7`

Evaluate on:

- held-out C zmod data
- planar held-out data if available later
- zmod intra-domain split if a clean split is defined

Metrics:

- overall RMSE over `tau_residual_raw_1..6`
- overall RMSE over `tau_residual_raw_1..7`
- per-joint RMSE
- prediction variance / span
- constant prediction flag
- correlation with target
- support gate status for the model's actual input dimensions

Required implementation change for this offline path:

- Build `.npz` datasets with selectable feature columns.
- Allow `scripts/train_stage4_matched_frozen_gp.py` or a new offline trainer to accept `X.shape[1] == 12`.
- Write model metadata that records the exact feature columns.
- Ensure any 12D model is clearly marked as offline-ablation-only unless the runtime controller is explicitly updated in a later, separate task.

Do not compare a 12D support gate against a 14D-trained GP as if it proves the 14D model is safe.

### P2: tau1..6 vs tau1..7 output target

Train/evaluate:

- outputs `tau_residual_raw_1..6`
- outputs `tau_residual_raw_1..7`

Report:

- joint7 residual variance and RMS
- joint7 prediction RMSE
- joint7 prediction span and constant prediction flag
- whether adding joint7 target changes joints 1..6 in any shared training/evaluation pipeline

Because the existing pipeline trains per-joint pickles, including joint7 should not mathematically change joint1..6 models unless shared preprocessing or sampling changes. That assumption should be verified in code before making a conclusion.

### P3: planar vs zmod q7 contribution

Compare the q7/dq7 drop effect separately for:

- planar train -> zmod held-out
- zmod train -> zmod held-out
- planar train -> planar held-out if available
- zmod train -> zmod held-out or intra-domain split

Expected interpretation:

- If 12D and 14D RMSE are similar on planar but different on zmod, q7/dq7 may matter more for zmod-induced joint-space motion.
- If both planar and zmod 12D models perform similarly to 14D, q7/dq7 may be weakly useful for residual prediction, but this still does not decide runtime support safety.
- If joint7 residual prediction is weak, unstable, or constant-like, future `tau7` compensation should remain deferred.

## Future Work Only

The following should remain future work until the offline ablation is complete:

- real-time `tau7` compensation
- changing `cartesian_impedance.py`
- changing GP compensation scale/clip/defaults
- changing launch/config defaults
- full 7DoF joint-space trajectory design
- any Franka-connected validation

## First Implemented Offline Step

The first implementation target is `scripts/run_stage5_q7_ablation.py` in `support-gate` mode:

- discover existing Stage 4 reference/candidate pairs with `--auto-stage4`
- optionally read an existing support matrix CSV with `--matrix-csv`
- generate `stage5_q7_ablation_support_gate.csv`
- generate `stage5_q7_ablation_support_gate.md`
- report each pair twice: `14d` and `12d_without_q7_dq7`

This first step intentionally avoids GP retraining.

## Safety Boundary

- No controller modification.
- No trajectory publisher modification.
- No `gp_server.py` runtime modification.
- No launch/config modification.
- No GP compensation path modification.
- No torque command logic modification.
- No `ros2 launch`.
- No `ros2 run`.
- No Franka connection.
- No claim that offline support or prediction ablation proves GP-on tracking improvement.
- No claim that a 12D support pass permits ignoring q7/dq7 for a 14D-trained model.
