# Stage 3A Offline Analysis

This note documents the offline comparison workflow for Stage 3A `z_modulated_circle` CSV runs.

## Data Layout

Stage 3A data is stored under:

- `data/stage3a/csv/comp_off/`
- `data/stage3a/csv/gp_on_scale03/`
- `data/stage3a/csv/gp_on_scale05/`
- `data/stage3a/csv/gp_on_scale07/`
- `data/stage3a/plots/`
- `data/stage3a/logs/launch.log`

Main CSV files:

- `data/stage3a/csv/comp_off/stage3a_comp_off_zmod_fullrun_20260522_3749pts.csv`
- `data/stage3a/csv/gp_on_scale03/stage3a_gp_on_zmod_conservative_fullrun_20260522_3749pts.csv`
- `data/stage3a/csv/gp_on_scale05/stage3a_gp_on_zmod_scale05_clip1_partial_20260522_2024pts.csv`
- `data/stage3a/csv/gp_on_scale07/stage3a_gp_on_zmod_scale07_clip1_nearfull_20260522_3103pts.csv`

## Run Status Meaning

- `fullrun`: complete intended validation run, expected to be close to 6 rounds / `3749` samples.
- `partial`: stopped before the complete validation length.
- `nearfull`: close to full length, but still not a complete 6-round validation.
- `short`: short attempt, useful for debugging context only.
- `attempt`: incomplete attempt, useful for log traceability only.

For Stage 3A conclusions, the main comparison is `comp_off` fullrun versus `gp_on_scale03` fullrun. `gp_on_scale05` partial and `gp_on_scale07` nearfull are trend references, not complete stable validation evidence.

## Scripts

Inventory and quick-look analysis:

- `python3 scripts/analyze_stage3a_csv.py`
- `python3 scripts/analyze_stage3a_csv.py data/stage3a/csv/comp_off/stage3a_comp_off_zmod_fullrun_20260522_3749pts.csv`

Comparison summaries and plots:

- `python3 scripts/compare_stage3a_modes.py`
- `python3 scripts/compare_stage3a_modes.py --include '*fullrun*'`

Default paths:

- input: `data/stage3a/csv`
- analysis output: `outputs/stage3a_analysis`
- comparison output: `outputs/stage3a_comparison`

## Caveats

`communication_constraints_violation` indicates the robot/control stack reported a communication timing constraint issue. Treat the related run as operationally interrupted, not as robust repeated validation.

`User Stop` means the run was manually stopped. The CSV may still be useful if it contains enough samples, but its `run_status` and `sample_completion_ratio` must be considered.

`rclpy shutdown context errors` can occur during ROS2 shutdown after interruption. They should be recorded as run caveats, but they do not automatically invalidate already written CSV samples.

Current project standard: one successful usable run is enough to move the offline comparison forward. Do not claim Stage 3A is fully stable or robustly repeated. In particular, `gp_on_scale05` and `gp_on_scale07` are trend references only, not full stable 6-round validations.
