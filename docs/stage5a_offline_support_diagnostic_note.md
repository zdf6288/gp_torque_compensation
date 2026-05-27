# Stage 5A Offline Support Diagnostic Note

## 1. Purpose

This note records the Stage 5A offline support matrix result generated from `scripts/run_stage5_support_matrix.py`.

The purpose is to:

- check whether q7 alone is sufficient as a support gate.
- document the observed q7 / 14D joint-space support mismatch.
- clarify that this is an offline support diagnostic, not GP-on tracking proof.
- provide motivation for the next no-GP live q7 / 14D joint-space data collection.

The main conclusion is that Stage 5A should no longer be framed as only q7-focused. A more accurate framing is:

- `q7 / 14D support-aware posture consistency`
- `14D joint-space support-aware trajectory/preflight design`

## 2. Inputs

Tools and outputs used:

- `scripts/validate_stage5_q7_support.py`
- `scripts/run_stage5_support_matrix.py`
- `outputs/stage5_support_matrix/stage5_support_matrix.csv`
- `outputs/stage5_support_matrix/stage5_support_matrix.md`

References and candidates recorded in the matrix report:

- `GP_A_planar_train`
- `GP_B_zmod_train`
- `GP_matched_strict_no_gp_zmod`
- `A1+A2_planar_reference`
- `B1+B2_zmod_reference`
- `B1_zmod_reference`
- `B2_zmod_reference`
- `C1_heldout_zmod`
- `C2_heldout_zmod`
- `B2_zmod_candidate`

Important source paths from the matrix:

- `data/stage4/cross_traj/models/GP_A_planar_train`
- `data/stage4/cross_traj/models/GP_B_zmod_train`
- `data/stage4/models/GP_matched_strict_no_gp_zmod`
- `data/stage4/cross_traj/C_no_gp_zmod_heldout/usable_runs/C_no_gp_zmod_heldout_3000pts_20260526_211759.csv`
- `data/stage4/cross_traj/C_no_gp_zmod_heldout/usable_runs/C_no_gp_zmod_heldout_3000pts_20260526_212556.csv`

## 3. Main Matrix Results

The offline support matrix reported:

- 12 pairs evaluated.
- 0 pairs skipped.
- 0 pairs passed full 14D support.
- q7 passed but 14D support failed in 2 pairs:
  - `GP_B_zmod_train_vs_C1`
  - `B1_B2_reference_vs_C1`

Important pair-level results:

- `GP_B_zmod_train_vs_C1`
  - `q7_support_pass=true`
  - `joint_space_14d_pass=false`
  - `overall_status=fail_14d_out_of_support`
  - `worst_dimension=joint_pos_4`
- `GP_B_zmod_train_vs_C2`
  - q7 failed
  - `overall_status=fail_q7_out_of_support`
  - `worst_dimension=joint_pos_7`
- `GP_A_planar_train` failed q7 support for both C1 and C2.

Worst-dimension distribution:

- `joint_pos_1`: 5
- `joint_pos_7`: 5
- `joint_pos_4`: 2

## 4. Interpretation

q7 support mismatch remains important, but q7 alone is not sufficient as a support gate.

The matrix shows that a candidate can pass q7 support and still fail full 14D joint-space support. The clearest case is `GP_B_zmod_train_vs_C1`: q7 passed, but the full support check failed with `worst_dimension=joint_pos_4`.

Therefore, Stage 5A should check the full 14D feature support:

- `joint_pos_1..7`
- `joint_vel_1..7`

This result also suggests that `joint_pos_4` may be another relevant mismatch dimension for C1-like data. `GP_B_zmod_train` improved offline residual prediction in Stage 4 diagnostics, but that does not mean it automatically covers all C held-out joint-space support.

This strengthens the need for no-GP live data collection before any conservative GP-on discussion.

## 5. Updated Stage 5A Framing

Old framing:

- q7-focused support preflight

Updated framing:

- q7 / 14D support-aware posture consistency
- full joint-space support preflight before GP-on re-entry
- no-GP live 14D logging before any compensation experiment

Stage 5A can still start from the `z_modulated_circle` Cartesian trajectory, but the gate should evaluate 14D joint-space support, not only q7.

Stage 5B full 7DoF joint-space excitation remains deferred. Stage 5A should first answer whether the current support-aware zmod workflow can produce a candidate/live trajectory inside frozen GP support.

## 6. Implications for Next Real-Robot Step

The next real-robot step is not GP-on.

The next real-robot step should be:

- no-GP live q7 / 14D joint-space logging.
- same or known zmod parameters.
- offline validator / support matrix after data collection.
- compare live CSV against `GP_B_zmod_train` and relevant references.
- do not proceed if q7 or 14D support fails.

Decision rules:

- Validator pass does not authorize GP-on.
- Support fail blocks GP-on.
- Real-robot GP-on re-entry requires a separate reviewed prompt.
- A q7 pass is not enough if 14D support fails.

## 7. Safety Notes

This note records offline support analysis only.

Safety boundaries:

- no real-robot launch in this step.
- no controller change.
- no trajectory publisher change.
- no launch/config change.
- no torque path change.
- no online update.
- no no-clip run.
- no high scale.
- no scale sweep.
- no GP-on claim.
- no tracking improvement claim.

The offline support matrix is a diagnostic tool. It does not prove real-robot tracking improvement and does not authorize GP-on.

## 8. Recommended Next Steps

Recommended next steps:

1. Confirm `scripts/run_stage5_support_matrix.py` is committed in the reviewed branch.
2. Commit this diagnostic note.
3. Use `docs/stage5a_no_gp_live_q7_logging_runbook.md` for no-GP live q7 / 14D data collection when the robot is available.
4. Run `scripts/validate_stage5_q7_support.py` and/or `scripts/run_stage5_support_matrix.py` on the live CSV.
5. Only after support pass plus read-only review, discuss conservative GP-on re-entry in a separate prompt.
6. Keep full 7DoF joint-space excitation deferred.
