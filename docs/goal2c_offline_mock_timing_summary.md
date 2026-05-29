# GOAL2 C - Offline / Mock GP Timing Benchmark Summary

## Purpose

GOAL2 C adds and validates an offline-only timing benchmark for GP model-level timing. It measures:

- GP model load timing.
- Local GP `predict()` timing.
- Cloud-like GP `predict()` timing.
- Combined local + cloud-like prediction timing.
- Copied-model `add_point()` timing.
- Mock cloud JSON request / response roundtrip timing.

The purpose is to provide a conservative reference for later GOAL2 D timing instrumentation design. These results are not fake hardware timing and are not real robot safety proof.

## Environment

- Workspace path: `/home/dummd/projects/gp_torque_goal2_delay`
- Branch: `stage6_goal2_delay`
- Benchmark script: `scripts/goal2c_offline_mock_timing.py`
- Model zip: `new_structure/gp/gp_models.zip`
- Temp extracted model dir: `outputs/goal2c_tmp_gp_models_from_zip/gp_models`
- Benchmark output dir: `outputs/goal2c_offline_mock_timing_smoke_real_model`
- ROS / launch / robot command: not run
- Outputs: ignored and not intended for commit
- Python CSV output: `pandas` was unavailable, so the script used its stdlib CSV fallback

## Model Source

The default model directory `new_structure/gp/gp_models` was not present in this workspace. The complete model set was found in `new_structure/gp/gp_models.zip`.

The zip contains an internal `gp_models/` directory with:

- `joint1_local.pkl` through `joint7_local.pkl`
- `joint1_cloud.pkl` through `joint7_cloud.pkl`

For the smoke benchmark, the zip was extracted only to the ignored temp directory `outputs/goal2c_tmp_gp_models_from_zip`. The original zip was not modified, and no source model directory was created or changed. The benchmark used:

`outputs/goal2c_tmp_gp_models_from_zip/gp_models`

No fallback from cloud-like to local models was needed because all 7 cloud-like pickle files were present.

## Commands

Syntax check:

```bash
python3 -m py_compile scripts/goal2c_offline_mock_timing.py
```

Offline smoke benchmark:

```bash
python3 scripts/goal2c_offline_mock_timing.py --model-dir outputs/goal2c_tmp_gp_models_from_zip/gp_models --num-samples 50 --warmup 5 --add-point-samples 5 --include-add-point --mock-cloud --output-dir outputs/goal2c_offline_mock_timing_smoke_real_model
```

Representative output checks:

```bash
ls -lh outputs/goal2c_offline_mock_timing_smoke_real_model
head -5 outputs/goal2c_offline_mock_timing_smoke_real_model/goal2c_timing_records.csv
head -5 outputs/goal2c_offline_mock_timing_smoke_real_model/goal2c_timing_summary.csv
cat outputs/goal2c_offline_mock_timing_smoke_real_model/goal2c_timing_summary.md
```

## Result Status

- Benchmark status: success
- Records count: 1109
- Skipped count: 0
- Failed count: 0
- Fallback: none
- Synthetic input: yes, for timing only
- `add_point()`: measured only on copied models
- Mock cloud: JSON serialization / parse mock only
- Real cloud communication: none
- ROS service timing: none
- Fake hardware timing: none
- Real robot timing: none

## Timing Summary

| Operation | Model kind | Count | p50_ms | p95_ms | p99_ms | max_ms | Caveat |
|---|---|---:|---:|---:|---:|---:|---|
| predict per-joint | local | 350 | 0.006273 | 0.023739 | 0.035299 | 0.071110 | synthetic input |
| predict 7-joint total | local | 50 | 0.190777 | 0.352677 | 0.440985 | 0.450665 | synthetic input |
| predict per-joint | cloud_like | 350 | 0.006154 | 0.009940 | 0.025836 | 0.039098 | local cloud-like pickle, not network |
| predict 7-joint total | cloud_like | 50 | 0.185976 | 0.283656 | 0.310416 | 0.318660 | not real cloud delay |
| predict local + cloud total | combined | 50 | 0.074819 | 0.107127 | 0.127663 | 0.139744 | combined local process timing |
| add_point per-joint | local | 35 | 0.054379 | 0.128513 | 0.709123 | 1.000331 | copied model only |
| add_point total | local | 5 | 0.822784 | 1.601021 | 1.740074 | 1.774837 | copied model only |
| add_point per-joint | cloud_like | 35 | 0.051106 | 0.084668 | 0.135015 | 0.158143 | copied model only |
| add_point total | cloud_like | 5 | 0.576435 | 0.748816 | 0.779057 | 0.786617 | copied model only |
| mock roundtrip | mock_cloud | 50 | 0.009282 | 0.015336 | 0.045764 | 0.047500 | JSON mock only |

## Interpretation

The local 7-joint prediction p95 was about 0.35 ms. The cloud-like 7-joint prediction p95 was about 0.28 ms. Copied-model `add_point()` total p95 was about 1.6 ms for local models and about 0.75 ms for cloud-like models. Mock JSON roundtrip p95 was about 0.015 ms.

These values suggest the model-level compute path is small in this offline benchmark. They do not directly imply ROS callback duration, controller loop duration, deadline miss behavior, fake hardware timing, or real robot timing.

The combined timing is lower than the local and cloud-like separate 7-joint totals. This should not be over-interpreted. It may be affected by benchmark function structure, cache state, measurement granularity, synthetic input, and the exact operation definition used by the script.

The `cloud_like` model is a local pickle model loaded from disk. It is not a real network/cloud delay measurement.

## Caveats

- Offline/mock timing is not fake hardware timing.
- Offline/mock timing is not real robot safety proof.
- Synthetic input is for timing only and is not an accuracy result.
- `cloud_like` is not real cloud communication.
- Mock cloud timing is not ROS service timing.
- `add_point()` timing is measured only on copied models and does not mutate saved pickle files.
- Output files under `outputs/` should not be committed.
- `pandas` fallback does not block basic CSV output, but the missing dependency should be recorded.
- There is no controller callback wall-duration measurement here.
- There is no deadline miss count or deadline ratio here.
- There is no ROS executor, pub-sub, or service timing here.
- There is no Franka communication timing here.

## Commit Recommendation

The benchmark script and this docs summary can be prepared for commit after user confirmation. Generated outputs should not be committed, and `.gitignore` does not need to be changed because `outputs/` is already ignored.

Suggested commit message:

`Add GOAL2 C offline mock GP timing benchmark`

## Next Steps

1. Commit `scripts/goal2c_offline_mock_timing.py` and `docs/goal2c_offline_mock_timing_summary.md` after user confirmation.
2. Keep GOAL2 C results separate from GOAL2 D.
3. Start GOAL2 D in a separate thread / prompt.
4. If GOAL2 D adds controller timing instrumentation, design it first in Plan mode.
5. Continue to forbid real robot experiments at this stage.
6. Do not describe GOAL2 C timing as real robot timing.
