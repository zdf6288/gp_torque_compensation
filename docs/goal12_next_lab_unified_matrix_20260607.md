# GOAL12 Next Lab Unified Matrix Plan

## Goal

Run all compensation modes under identical trajectory settings to make results comparable.

## Fixed Conditions

Use the same values for every run:

    trajectory_mode:=goal1_spatial_multisine
    control_frequency:=50
    circle_frequency:=0.05
    transition_duration:=10.0
    delay_steps:=2
    gp_online_update_enabled:=false
    gp_historical_db_enabled:=true
    gp_compensation_disable_joint7:=true
    gp_compensation_clip_nm:=0.2
    timing_logging_enabled:=true
    timing_log_stride:=20 or 50
    load_gripper:=false
    spawn_gp_server:=false
    spawn_fake_state_parameter_publisher:=false

## Minimum Matrix

| ID | compensation | source | scale | purpose |
|---|---:|---|---:|---|
| A | false | local | 0.0 | no-GP baseline |
| B | true | hist_db | 0.02 | main historical active validation |
| C | true | hist_db | 0.05 | historical scale ablation |
| D | true | combined | 0.02 | local/cloud current predictor baseline |
| E | true | triple | 0.02 | future triple ablation, only after implementation/review |

## Expanded Matrix

| ID | compensation | source | scale | purpose |
|---|---:|---|---:|---|
| A | false | local | 0.0 | no-GP baseline |
| B | true | hist_db | 0.02 | hist active low-scale |
| C | true | hist_db | 0.05 | hist active medium-scale |
| D | true | local | 0.02 | local-only ablation |
| E | true | cloud | 0.02 | cloud-only ablation |
| F | true | combined | 0.02 | local/cloud combined ablation |
| G | true | triple | 0.02 | memory-augmented ablation |

## Success Criteria

A run is usable if:

    controller rows >= 3000
    CSV exists and contains expected safety columns
    no abnormal motion
    gp_applied <= clip
    deadline_miss_count small or zero
    communication violation after CSV save is caveat, not automatic failure

## Stop Conditions

Stop immediately on:

    abnormal motion
    vibration / noise
    Franka reflex
    unexpected nonzero gp_applied when compensation disabled
    frequent clipping
    NaN / Inf
    wrong workspace prefix
    missing safety columns

## Current Recommendation

Do not implement active triple before:

1. `controller_log_stride` / `hist_db_query_stride` are considered
2. triple source is read-only reviewed
3. fake/no-robot validation passes
4. no-GP and hist_db active are validated under identical trajectory
