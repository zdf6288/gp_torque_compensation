# Stage 2A Feasibility Log

## Project Standard

For this project, a real-robot mode or stage is considered usable when at least one successful run:

- completes the target trajectory
- saves valid CSV/log data
- has traceable branch, commit, command, and parameters

Perfect repeatability and clean shutdown are not required for engineering progress. Known issues such as `communication_constraints_violation`, non-clean shutdown, CPU/realtime limitations, and intermittent original-branch instability should be recorded as engineering notes, but they do not block progress if usable data was collected.

For thesis/paper writing, focus on successful runs and method effect. Do not overstate results with claims such as fully stable or robust repeated validation.

## Current Modes

| Mode | Intended purpose | Current status |
| --- | --- | --- |
| Pure no-GP | Baseline with `gp_prediction_enabled:=false`, `gp_online_update_enabled:=false`, `gp_compensation_enabled:=false` | Usable run available |
| Compute-only / no-apply | GP prediction/update checks without applying compensation | Useful intermediate validation mode; document exact run when used |
| GP-on conservative | Frozen local GP compensation with `gp_compensation_scale:=0.1` and `gp_compensation_clip_nm:=0.5` | Usable run available |
| Stage 1 baseline archive | Earlier baseline/reference archive | Usable archived data available |

## Current Result Summary

- Pure no-GP: usable run available; real-robot logs indicate the trajectory completed 6 rounds and saved 3000 points at least once.
- GP-on conservative: usable run available; real-robot logs indicate the trajectory completed 6 rounds and saved 3000 points at least once.
- Stage 1 baseline archive: usable archived data and plots are available for reference.

## Known Limitations

- Intermittent `communication_constraints_violation` remains present.
- The original `ggwp` branch also showed this issue, so it should not be attributed only to the Stage 2A frozen-GP changes without more evidence.
- A shutdown-time abort does not invalidate CSV data that was already saved after a completed trajectory.
- Current notes support feasibility and comparison preparation, not a claim of fully stable repeated validation.

## Next Stage

1. Run minimal offline CSV analysis with `scripts/analyze_stage2a_csv.py`.
2. Inspect `outputs/stage2a_analysis/stage2a_summary.csv` and quick-look plots.
3. Use the successful Stage 2A runs as the basis for Stage 3 / spatial trajectory preparation.
4. Keep real-robot claims conservative: successful usable datasets exist, while repeated stability remains a separate validation topic.
