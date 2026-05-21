# Stage 2A Data Inventory

This document records the local Stage 2A real-robot CSV/archive files used for offline analysis. Raw data and generated analysis outputs should remain local and should not be committed to git.

## Local Data Files

| File | Size | Intended mode | Commit to git? |
| --- | ---: | --- | --- |
| `data/stage2a/csv/pure_no_gp_20260520_212521.csv` | 5.2 MB | Pure no-GP baseline: `gp_prediction_enabled:=false`, `gp_online_update_enabled:=false`, `gp_compensation_enabled:=false` | No |
| `data/stage2a/csv/stage1_baseline_20260520.csv` | 8.0 MB | Stage 1 archived baseline / reference run | No |
| `data/stage2a/csv/stage2a_gpon_conservative_20260520_211748.csv` | 8.1 MB | GP-on conservative: `gp_prediction_enabled:=true`, `gp_online_update_enabled:=false`, `gp_compensation_enabled:=true`, `gp_compensation_source:=local`, `gp_compensation_scale:=0.1`, `gp_compensation_clip_nm:=0.5` | No |
| `data/stage2a/raw_archives/stage1_baseline_20260520.tar.gz` | local archive | Stage 1 baseline archive with existing plots and `repo_state.txt` | No |

## Analysis Outputs

Offline analysis outputs are generated under:

- `outputs/stage2a_analysis/stage2a_summary.csv`
- `outputs/stage2a_analysis/plots/<csv_stem>/`

The repository `.gitignore` already ignores `data/` and `outputs/`, so raw CSV/archive files and generated plots should remain local.

## Notes

- The CSV files listed above are real-robot experimental data and should be treated as local research artifacts.
- Do not use `git add data/`, `git add outputs/`, or stage generated `.csv`, `.tar.gz`, `.png`, or ROS log files.
- The offline analysis script is `scripts/analyze_stage2a_csv.py` and does not require ROS2.
