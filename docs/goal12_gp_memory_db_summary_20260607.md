# GOAL12 GP Memory DB Summary

## Purpose

This document records the offline construction and initial evaluation of GP-related memory DBs from the GOAL12 real archive.

These DBs are not replacements for the current residual historical DB. They are separate memory targets for future ablation and paper-style previous prediction memory analysis.

## Source archive

`goal12_v3_shadow_active_histdb_real_20260607_193111.tar.gz`

## Generated DBs

| DB | Rows | Target | Intended role |
|---|---:|---|---|
| `goal12_prediction_memory_all_sources_20260607.npz` | 10912 | multiple: local/cloud/hist/selected_raw/applied | full prediction memory inspection |
| `goal12_gp_selected_raw_memory_20260607.npz` | 5401 | `gp_selected_raw` | preferred previous prediction memory candidate |
| `goal12_gp_applied_memory_20260607.npz` | 5401 | `gp_applied` | actual active torque compensation memory |

## Scale / clip consistency check

The applied memory was verified against the selected raw memory:

| item | value |
|---|---:|
| scale | 0.02 |
| clip | 0.2 |
| max_abs_raw | 0.8104 |
| max_abs_applied | 0.0162 |
| max_abs_error between `gp_applied` and `scale * gp_selected_raw` | 0.0 |
| clip_active_ratio | 0.0 |

Conclusion:

- `gp_selected_raw_memory` stores the raw selected compensation prediction.
- `gp_applied_memory` stores the actual scaled compensation entering the torque path.
- For previous-prediction-memory analysis, `gp_selected_raw_memory` is more informative than `gp_applied_memory`.

## Coverage check

Coverage was checked using the same scaled `[q, dq]` distance convention as the residual hist DB:

| target run | coverage <= 1.0 | nearest mean | nearest p95 | nearest max |
|---|---:|---:|---:|---:|
| active histdb run | 0.9998 | 0.0144 | 0.0 | 1.1603 |
| shadow holdout run | 0.0038 | 1.2478 | 1.4036 | 1.8021 |

Conclusion:

- The GP selected raw memory DB covers the active run itself.
- It does not cover the shadow holdout run.
- Therefore, this memory DB is valid as a record of the active run but should not be used as a general compensation DB.

## Active vs shadow q/dq distribution

| run | rows | note |
|---|---:|---|
| active | 5678 | larger q/dq ranges in several joints |
| shadow | 5234 | different trajectory distribution |
| scaled center distance | 1.3015 | exceeds the current `max_distance=1.0` gate |

This explains why selected raw memory coverage is poor on the shadow run.

## Current interpretation

The GP memory DB construction is successful, but the resulting DB is distribution-specific.

It should be kept as a future ablation:

- `prediction_memory_db`
- not `hist_residual_db`
- not current mainline active compensation DB

## Mainline decision

The current mainline remains:

- residual historical DB
- `gp_compensation_source:=hist_db`
- `gp_compensation_disable_joint7:=true`
- unified trajectory matrix in the next lab session

## Future work

A meaningful previous-prediction-memory DB should be built from a unified trajectory matrix where all runs share:

- same trajectory mode
- same `circle_frequency`
- same `transition_duration`
- same `control_frequency`
- same q7-disabled compensation convention
