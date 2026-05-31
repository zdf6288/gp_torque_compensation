# GOAL1 N Lab Safety Review Summary — 2026-05-31

## Scope

This note summarizes the lab-side read-only safety review for GOAL1 N.

No real motion was executed during this review.

No `/effort_command` was published.

`cpp_relayer` was not activated.

## GOAL1 N target

GOAL1 N is the first low-risk real replay attempt:

- 3s
- 50Hz
- current-q anchored CSV
- low-torque
- without GP compensation
- hard safety gates
- one short attempt only

This is not the formal GP comparison experiment.

The first motion attempt should be explicitly treated as:

`without GP compensation`

with:

`gp_compensation_enabled=False`

## Confirmed today

### GP compensation gate

Stage 1 GP control logic was checked.

Confirmed:

- `gp_compensation_enabled=False` is the safe default.
- Default configuration does not inject `y_hat` into final `tau`.
- `gp_online_update_enabled=True` only gates `model.add_point()`.
- `gp_prediction_enabled` remains an additional kill switch.
- `gp_compensation_clip_nm` is applied before compensation enters `tau`.
- No active path was found that bypasses `gp_compensation_enabled` and directly injects GP prediction into final torque.

Conclusion:

GOAL1 N first motion should be run as `without GP compensation`.

### Lab workspace

Confirmed lab workspace state:

- path: `/home/mirmi_ros2_2/dongfa/tt_dgp`
- branch: `frozen_gp_spatial_trajectory`
- recent commit: `4a0cc40 Add GOAL1 JointState state-only preflight`
- working tree: clean

### Workspace overlay

After `source install/setup.bash`, package prefix aligned with repo-local install:

- `cpp_relayer`
- `py_controllers`
- `custom_msgs`

### Runtime/process inventory before bringup

No residual process was found for:

- `ros2_control_node`
- `controller_manager`
- `cpp_relayer`
- `cartesian_impedance`
- `goal1_joint_space_replay`
- `trajectory_publisher`
- `ros2 launch`
- `ros2 run`

Current `/effort_command` topic was absent before bringup.

### cpp_relayer artifact

`cpp_relayer` source/build/install alignment looked acceptable:

- source contains stale-command zero guard
- build artifact timestamp is later than source
- install library symlink points to build artifact

Caveat:

`install/new_bringup/share/new_bringup/config/controllers.yaml` may miss explicit `command_timeout_sec: 0.2`, although the code default is still `0.2`.

Next lab run should either confirm the runtime parameter after bringup or rebuild/reinstall `new_bringup`.

## Remaining blockers

GOAL1 N cannot directly enter motion yet.

Remaining gates:

1. Robot bringup must be done in a clean session.
2. After bringup, controller_manager must be queried read-only.
3. Confirm `joint_state_broadcaster` active.
4. Confirm `cpp_relayer` is not active during state-only preflight.
5. Confirm no unknown `/effort_command` publisher.
6. Confirm `/state_parameter` and `/franka/joint_states` topic state.
7. Confirm `cpp_relayer.command_timeout_sec=0.2` or rebuild/reinstall `new_bringup`.
8. Run no-motion state-only preflight before any motion.
9. Confirm first CSV q vs current q mismatch is within tolerance.
10. Only after all gates pass should a separate first-motion command be prepared.

## cpp_relayer risk

The current GOAL1 N real motion path likely needs active `cpp_relayer` to consume `/effort_command` and write joint effort command interfaces.

Known caveat:

Standalone active `cpp_relayer` previously caused:

- visible robot motion
- yellow state
- `communication_constraints_violation`

Therefore:

- Do not activate `cpp_relayer` for state-only checks.
- Do not treat `cpp_relayer active` as harmless.
- Do not mix GOAL1 replay with existing Cartesian launch.
- Do not allow unknown `/effort_command` publishers.

## Current conclusion

GOAL1 N status at the end of 2026-05-31:

- GP gate: pass
- first motion mode: without GP compensation
- lab workspace / branch / commit: verified
- working tree: clean
- repo-local package overlay: verified
- `cpp_relayer` artifact alignment: likely OK
- `/effort_command`: absent before bringup
- residual GOAL1 / Cartesian / `cpp_relayer` processes: none
- controller_manager: unavailable before robot bringup
- conclusion: can enter runbook-only planning next time, but cannot directly motion

## Next lab step

Next lab session should start from:

GOAL1 N runbook-only planning before first motion.

Do not directly run motion.

Recommended phase order:

1. Shell / repo / overlay check
2. Franka Web Interface and physical safety check
3. Bringup read-only controller/topic inventory
4. No-motion state-only preflight
5. Pre-motion go/no-go gate
6. Separate first-motion command only if all gates pass
7. Post-attempt inventory and caveat logging
