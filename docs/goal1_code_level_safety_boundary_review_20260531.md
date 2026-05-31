# GOAL1 Code-Level Safety Boundary Review — 2026-05-31

## Scope

This note summarizes a read-only code-level safety boundary review for GOAL1 state-only validation.

This is not a lab real-robot go/no-go decision.

This review only covers readiness for:

- `state_only=true`
- `publish_effort=false`
- no real motion
- no GP-on compensation

This review does not authorize:

- `publish_effort=true`
- real robot motion
- GP-on
- controller activation / deactivation
- manual `/effort_command` publishing

## Environment

Reviewed workspace:

- path: `/home/dummd/projects/gp_torque_compensation`
- branch: `frozen_gp_spatial_trajectory`
- latest commit includes: `4a0cc40 Add GOAL1 JointState state-only preflight`

Current untracked note:

- `docs/goal1_n_lab_safety_review_20260531.md`

This existing untracked document was not modified during the review and is not considered a blocker.

## Review method

Only read-only commands were used, such as:

- `pwd`
- `git rev-parse --show-toplevel`
- `git branch --show-current`
- `git log --oneline -5`
- `git status --short`
- `git diff --stat`
- `git diff --name-only`
- `rg --files`
- `rg -n ...`
- `sed -n ...`

Not run:

- `colcon build`
- `ros2 launch`
- `ros2 run`
- controller activation / deactivation
- manual `/effort_command` publishing
- real robot commands

## Files inspected

- `new_structure/py_controllers/py_controllers/goal1_joint_space_replay.py`
- `new_structure/py_controllers/launch/goal1_joint_space_replay_launch.py`
- `new_structure/py_controllers/py_controllers/cartesian_impedance.py`
- `new_structure/py_controllers/launch/cartesian_impedance_launch.py`
- `new_structure/new_bringup/config/controllers.yaml`
- `new_structure/new_bringup/launch/franka.launch.py`
- `new_structure/cpp_relayer/src/cpp_relayer.cpp`
- `docs/goal1_status_before_lab_validation.md`
- `docs/goal1_joint_space_replay_design.md`

## Default GOAL1 replay safety boundary

Observed defaults:

- `DEFAULT_DRY_RUN=True`
- `DEFAULT_START_REPLAY=False`
- `DEFAULT_PUBLISH_EFFORT=False`
- `DEFAULT_STATE_ONLY=False`

Launch defaults:

- `dry_run=true`
- `start_replay=false`
- `publish_effort=false`
- `state_only=false`

Safety interpretation:

- default launch does not enter the effort publish path
- default launch is not state-only unless `state_only=true` is explicitly passed
- lab state-only validation must explicitly set `state_only=true`

## State-only path

`state_only=true` has priority over the publish path.

In the state-only path:

- the node subscribes to robot state
- it validates state reception/freshness
- it does not create an `/effort_command` publisher
- it does not publish torque

This supports lab state-only validation under:

- `state_only=true`
- `publish_effort=false`

## Publish path

The effort publish path is only reachable when explicitly configured.

The dangerous combination is:

- `dry_run=false`
- `publish_effort=true`
- `start_replay=true`
- suitable motion state source

This path is outside the scope of this state-only review.

If entered, the GOAL1 replay path publishes final joint PD effort to `/effort_command`, not a trajectory/reference.

## Cartesian impedance / GP compensation boundary

`cartesian_impedance.py` is the original Cartesian torque path.

Observed GP defaults:

- `gp_compensation_enabled=False`
- `gp_compensation_scale=0.1`
- `gp_compensation_clip_nm=0.5`

The GP compensation logic only affects final `tau` when both are true:

- `gp_prediction_enabled`
- `gp_compensation_enabled`

Compensation is scaled and then clipped per joint before being applied.

Safety interpretation:

- GP compensation is not enabled by default
- default configuration does not inject `y_hat` into final `tau`
- explicit GP-on remains outside this review scope

## cpp_relayer boundary

`cpp_relayer`:

- subscribes to `/effort_command`
- writes `panda_joint*/effort` command interfaces
- publishes `/state_parameter`
- contains `command_timeout_sec=0.2`
- contains stale-command zero fallback
- refuses invalid commands

Caveat:

Even with stale-command zero fallback in source code, `cpp_relayer active` is not harmless.

Do not activate `cpp_relayer` just to obtain state.

Before any `publish_effort=true` motion attempt, lab-side behavior must be separately verified with controller state, topic inventory, timeout behavior, and operator abort plan.

## Topic boundary

`/state_parameter`:

- published by `cpp_relayer`
- consumed by `cartesian_impedance.py`, `trajectory_publisher.py`, and GOAL1 replay/state-only paths

`/effort_command`:

- published by `cartesian_impedance.py` or explicit GOAL1 publish path
- consumed by `cpp_relayer`
- leads to joint effort command interfaces

For state-only validation:

- GOAL1 should not create `/effort_command`
- no torque should be published

## Readiness decision

Code-level result:

GOAL1 supports lab Linux state-only validation under the following boundary:

- use `state_only=true`
- keep `publish_effort=false`
- do not activate motion path
- do not enable GP compensation
- confirm no `/effort_command` publisher

Blocking issue for `state_only=true / publish_effort=false`:

- none found at code level

This does not authorize:

- real motion
- `publish_effort=true`
- GP-on compensation
- controller activation / deactivation
- manual `/effort_command` publishing

## Caveats

- Default `state_only=false`, so state-only validation must explicitly set `state_only=true`.
- If using `state_source=state_parameter`, `/state_parameter` depends on `cpp_relayer`; do not treat `cpp_relayer active` as harmless.
- Prefer `state_source=joint_states` for no-motion preflight when available.
- Any future `publish_effort=true` attempt requires separate lab-side read-only controller/topic inventory and runbook gate.
