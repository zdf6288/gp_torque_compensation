# Stage 4 Dataset Preparation

This note documents the offline dataset matching step before training `GP_planar`
and `GP_spatial`.

## Purpose

Stage 4 compares frozen GP models trained from different trajectory families.
The training sample count should therefore be matched before model training, so
the comparison is not biased by one model seeing more training points.

The matching script is:

- `scripts/prepare_stage4_gp_dataset.py`

It reads planar and spatial / tilted training CSV files, builds compatible
`X1..X7` and `Y1..Y7` arrays, crops both datasets to the same deterministic
sample count, and writes:

- `GP_planar_matched.npz`
- `GP_spatial_matched.npz`
- `stage4_dataset_manifest.json`

## Feature And Target Choice

Default builder mode:

- `--builder-mode runtime-real-dq`

Default feature:

- `X = [joint_pos_1..7, joint_vel_1..7]`

Default target:

- `Y_j = tau_residual_j`

Reason:

- `build_dataset_no_filter.py` currently uses the 14D `q + dq_des_joint`
  convention in its enabled main path.
- `build_dataset_real_dq.py` uses the 14D `q + joint_vel` convention.
- The current controller GP runtime main call passes measured `dq` into
  `_gp_predict_and_update()`.

For Stage 4, the default script mode therefore uses `q + joint_vel` so the
offline training dataset is closer to the runtime input. This should be kept
consistent across both `GP_planar` and `GP_spatial`.

Caveat:

- If the project later decides to strictly follow the older
  `build_dataset_no_filter.py` feature convention, regenerate both matched
  datasets with `--builder-mode no-filter-dq-des`.
- Do not train one model with `q + joint_vel` and the other model with
  `q + dq_des_joint`.
- If controller runtime feature construction changes, regenerate the Stage 4
  datasets with the same feature definition.

## Commands

Dry-run:

- `python3 scripts/prepare_stage4_gp_dataset.py --planar-pattern "data/stage4/train/planar_circle/*.csv" --spatial-pattern "data/stage4/train/spatial_tilted/*.csv" --out-dir data/stage4/datasets --dry-run`

Real run:

- `python3 scripts/prepare_stage4_gp_dataset.py --planar-pattern "data/stage4/train/planar_circle/*.csv" --spatial-pattern "data/stage4/train/spatial_tilted/*.csv" --out-dir data/stage4/datasets`

Training:

- `python3 new_structure/gp/train_gp_hdimensional.py --data data/stage4/datasets/GP_planar_matched.npz --out-dir data/stage4/models/GP_planar --joint all`
- `python3 new_structure/gp/train_gp_hdimensional.py --data data/stage4/datasets/GP_spatial_matched.npz --out-dir data/stage4/models/GP_spatial --joint all`

## Manifest

`stage4_dataset_manifest.json` records:

- source CSV files and raw row counts
- rows after `dropna`
- raw dataset samples after `decimate` and `smooth`
- selected / cropped samples
- `builder_mode`
- feature definition
- target definition
- `dt`, `decimate`, `smooth`, and effective `eff_dt`
- `crop_mode`
- output `.npz` paths
- timestamp
- suggested training commands

## Safety Boundary

This preparation step is offline-only. It does not modify controller code,
launch files, trajectory code, config files, model files, raw data files, or
Franka hardware interface code. It does not run any real-robot launch command.

## Git Policy for Generated Artifacts

Generated matched datasets, manifests, trained GP model directories, and analysis outputs under `data/stage4/` or `outputs/` should not be committed unless the repository policy explicitly allows it.
