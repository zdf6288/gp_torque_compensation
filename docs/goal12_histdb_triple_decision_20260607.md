# GOAL12 Historical DB / Triple Fusion Decision Summary

## Context

This report summarizes the home-PC offline analysis after the GOAL12 real lab archive:

- Archive: `goal12_v3_shadow_active_histdb_real_20260607_193111.tar.gz`
- Branch context: `goal12_integrated_histdb_delay`
- Key code state: high-frequency runtime prints removed in commit `6300d3a`
- Main real runs:
  - Shadow: `goal12_real_v3_goal2clean_histdb_shadow_holdout_delay2_slowtransition_stride10_20260607_190542`
  - Active histdb: `goal12_real_v3_active_histdb_scale002_clip02_delay2_lowload_20260607_192757`

## Run Summary

From `goal12_run_summary.md`:

### Shadow run

- rows: `5234`
- gp_compensation_enabled: `0`
- gp_applied max_abs: `0.0`
- hist_db_available ratio: `0.99866`
- hist_db_distance_pass ratio: `0.99866`
- deadline_miss_count: `0`
- callback_deadline_ratio_max: `0.327`

### Active histdb run

- rows: `5678`
- gp_compensation_enabled: `1`
- gp_compensation_source_code: `4`
- gp_applied max_abs: `0.0162 Nm`
- hist_db_available ratio: `0.9512`
- deadline_miss_count: `0`
- callback_deadline_ratio_max: `0.173`

## Residual Prediction RMSE

From `goal12_residual_prediction_rmse_comparison.md`:

| predictor | error_rmse | interpretation |
|---|---:|---|
| zero baseline | 0.9939 | no residual prediction |
| local | 1.0047 | worse than zero in this run |
| cloud | 1.0047 | worse than zero in this run |
| combined_mean | 1.0047 | worse than zero in this run |
| hist_db | 0.3939 | clearly strongest on this run |
| plain_w01 | 0.9247 | better than combined but much weaker than hist |
| sign_w01 | 0.9732 | sign gate did not improve over plain_w01 |
| sign_w02 | 0.9441 | still far weaker than hist |

## q7-excluded RMSE

From `goal12_q7_excluded_rmse.txt`:

| predictor | q1-q6 RMSE |
|---|---:|
| zero baseline | 0.9919 |
| local | 1.0023 |
| cloud | 1.0023 |
| combined | 1.0023 |
| hist_db | 0.3938 |

Conclusion:

- Hist DB remains clearly best even when q7 is excluded.
- q7 compensation should remain disabled in the next real-robot matrix.
- This is consistent with the earlier decision that q7 is not a reliable GP compensation target.

## Triple Fusion Feasibility

From `goal12_shadow_triple_offline_feasibility.md`:

- local/cloud combined norm RMS: `0.1735`
- hist norm RMS: `0.8624`
- hist / combined magnitude ratio mean: `4.86`
- hist / combined magnitude ratio p95: `6.50`
- hist vs combined sign agreement mean: `0.439`
- joint7 sign agreement: approximately `0.0013`

From `goal12_sign_gated_triple_feasibility.md`:

- plain_w01_over_combined_mean: `1.01`
- sign_w01_over_combined_mean: `1.17`
- sign_w02_over_combined_mean: `1.38`
- sign gate active joint ratio mean: `0.437`
- joint7 active ratio: `0.0` by design

Interpretation:

- Equal-weight triple fusion is not justified.
- Local/cloud predictions were weak on this specific slow real-shadow run.
- Historical DB is the strongest current predictor.
- Sign-gated triple is safe-ish but does not clearly outperform plain low-weight fusion.
- A future triple source should be treated as an ablation, not the mainline compensation strategy.

## Current Mainline Decision

Mainline next real-robot validation should be:

- `gp_compensation_source:=hist_db`
- `gp_compensation_disable_joint7:=true`
- `gp_compensation_scale:=0.02`
- `gp_compensation_clip_nm:=0.2`
- `control_frequency:=50`
- `circle_frequency`: fixed across all comparison runs
- `transition_duration:=10.0`
- `timing_log_stride:=20` or `50`
- `gp_online_update_enabled:=false`

## Future Triple Candidate

If implemented later, use a new explicit source:

- `gp_compensation_source:=triple`

Do not modify existing `combined` semantics.

Recommended first triple candidate:

    base = 0.5 * (y_hat_local + y_hat_cloud)

    if hist_db_available:
        w_hist = min(exp(-alpha * hist_db_nearest_distance), w_hist_max)
    else:
        w_hist = 0

    triple = (1 - w_hist) * base + w_hist * hist_db_gated_pred

Current data suggests hist-dominant or hist-only compensation is more justified than equal fusion.

## Future Prediction-Memory DB

Do not mix GP-applied data into current `Y_residual` DB.

Keep DB types separate:

| DB type | X | Y | role |
|---|---|---|---|
| hist_residual_db | `[q, dq]` | `tau_residual` | current mainline |
| prediction_memory_db | `[q, dq]` | `y_hat_*` or `gp_applied` | future paper-style memory ablation |
| reconstructed_residual_db | `[q, dq]` | `tau_residual + gp_applied` | experimental; requires sign audit |

## Load-Reduction Direction

Already done:

- removed high-frequency runtime prints: `6300d3a`

Next low-risk engineering patches to consider:

1. `controller_log_stride`
2. `hist_db_query_stride`
3. keep `timing_log_stride:=20` or `50`
4. do not increase to `100 Hz` until `50 Hz` is stable

## Final Decision

For the next lab session:

1. Use identical trajectory parameters across all runs.
2. Disable q7 compensation.
3. First compare no-GP vs hist_db active.
4. Add local/cloud/combined/triple only as controlled ablations.
5. Do not claim global local/cloud failure; only claim they were weak on this analyzed run.
