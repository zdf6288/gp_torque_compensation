# GOAL1 Replay vs GP Compensation Architecture Review — 2026-05-31

## Scope

This note summarizes a read-only code-level architecture review.

This is not a lab real-robot go/no-go decision.

No files were modified during the Codex review. No `ros2 launch`, `ros2 run`, controller activation, motion command, or `/effort_command` publish command was run.

## Main question

Can the current `goal1_joint_space_replay.py` naturally support the formal comparison:

- without GP compensation
- with GP compensation

under the existing `cartesian_impedance.py` GP compensation interface?

## Current GOAL1 replay dataflow

Current GOAL1 replay dataflow:

`goal1_joint_space_replay.py -> /effort_command -> cpp_relayer -> panda_joint*/effort`

The replay node:

- reads a joint-space CSV
- expects `time`, `joint_pos_1..7`, `joint_vel_1..7`, `joint_acc_1..7`
- computes guarded low-gain joint PD effort:
  `tau = kp * (q_des - q) + kd * (dq_des - dq)`
- clips / rate-limits the effort
- publishes `custom_msgs/msg/EffortCommand` to `/effort_command`

Conclusion:

- it publishes final effort / torque command
- it does not publish a desired trajectory/reference
- it bypasses `cartesian_impedance.py`
- it bypasses `_apply_gp_compensation()`
- it bypasses the `gp_compensation_enabled` gate

## Existing GP compensation interface

The existing GP compensation path lives in `cartesian_impedance.py`.

Current controller dataflow:

`/task_space_command -> cartesian_impedance.py -> _apply_gp_compensation(tau) -> /effort_command -> cpp_relayer`

`cartesian_impedance.py` subscribes to:

- `/state_parameter`
- `/task_space_command`
- `/data_recording_enabled`
- `/gp_mode`
- `/shutdown_control`

It publishes:

- `/effort_command`

The reference interface is Cartesian:

- `custom_msgs/msg/TaskSpaceCommand`
- `x_des`
- `dx_des`
- `ddx_des`

The controller computes `tau` inside `stateParameterCallback()`, then applies:

`tau = self._apply_gp_compensation(tau)`

GP compensation only affects final `tau` when both are true:

- `gp_prediction_enabled`
- `gp_compensation_enabled`

When enabled, it applies source selection, scale, and per-joint clip before returning:

`tau - clipped_compensation`

## cpp_relayer role

`cpp_relayer`:

- subscribes to `/effort_command`
- consumes `custom_msgs/msg/EffortCommand`
- writes to `panda_joint1/effort` through `panda_joint7/effort`
- publishes `/state_parameter`
- has stale-command zero fallback via `command_timeout_sec=0.2`
- rejects invalid / non-finite effort arrays

If `goal1_joint_space_replay.py` publishes `/effort_command`, `cpp_relayer` consumes it directly and writes to the joint effort command interfaces.

This path bypasses:

- `cartesian_impedance.py` tau calculation
- GP prediction/update path inside `cartesian_impedance.py`
- `_apply_gp_compensation()`
- `gp_compensation_enabled`

## Architecture options

### Option A: keep current GOAL1 replay as final effort publisher

Dataflow:

`goal1_joint_space_replay.py -> /effort_command -> cpp_relayer -> effort interface`

Use case:

- suitable for GOAL1 N first no-GP low-torque validation
- useful as a guarded short replay skeleton

Limitations:

- bypasses `cartesian_impedance.py`
- bypasses existing GP compensation
- not suitable as the final architecture for formal no-GP vs GP-on tracking comparison
- direct `/effort_command` publisher path requires strict lab-side safety gates

### Option B: add reference-publisher mode

Dataflow:

`GOAL1 reference publisher -> controller reference input -> cartesian_impedance.py computes tau -> _apply_gp_compensation() -> /effort_command -> cpp_relayer`

Use case:

- better architecture for formal without-GP vs with-GP tracking comparison
- allows one controller torque pipeline
- allows no-GP / GP-on switching through existing parameters
- avoids duplicating GP compensation logic in the replay node

Challenges:

- `cartesian_impedance.py` currently uses Cartesian `TaskSpaceCommand`
- current GOAL1 CSV is joint-space all-q trajectory
- must decide whether to add joint-space reference support or convert GOAL1 trajectory to Cartesian reference
- any controller-side change requires separate safety review

## Recommendation

Recommended architecture direction:

Keep the current `/effort_command` final effort publisher only for GOAL1 N first motion / no-GP low-torque validation.

For formal no-GP vs GP-on comparison, add a reference-publisher mode so the GOAL1 trajectory becomes a controller reference and the existing controller computes `tau` and applies GP compensation through the existing gates.

## Next task

Do a separate read-only feasibility review before implementation.

The next review should answer:

- whether to add a new `JointSpaceCommand` message
- whether `cartesian_impedance.py` should add a joint-space reference branch
- how `dq_des_joint` and `ddq_des_joint` should be defined
- how GP feature `q + dq_des_joint` behaves under joint-space reference
- whether no-GP / GP-on can be switched only by launch parameters
- how to keep defaults safe:
  - no motion by default
  - no GP compensation by default
  - clip / scale preserved
  - no accidental `/effort_command` publish
