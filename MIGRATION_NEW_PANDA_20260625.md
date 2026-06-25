# MIGRATION_NEW_PANDA_20260625

- New PC for gp_torque_compensation on another Panda robot.
- Purpose: TRO follow-up experiment support, not graduation thesis.
- Repo: ~/dongfa/tt_dgp
- Branch: goal12_triple_combined_base_shadow_20260613
- Commit: 91f1350
- New PC: Ubuntu 22.04.5 + ROS2 Humble + ROS_DOMAIN_ID=75 + CycloneDDS.
- Installed libfranka 0.12.1 from source into /usr/local.
- Build-only patch: added controller_interface dependency to franka_semantic_components.
- Build passed for franka_hardware, franka_semantic_components, franka_robot_state_broadcaster, franka_bringup, new_bringup, cpp_relayer, py_controllers.
- No controller, torque, GP compensation, trajectory, launch, or runtime logic changed.
- Not yet validated: Panda System Image, Desk, FCI, robot IP, state-only bringup, no-GP baseline, GP prediction-only, GP compensation-on.
- Safety: do not directly run GP-on or full trajectory experiments after build.
- Next order: Desk/FCI/IP -> state-only -> controller manager -> no-GP -> prediction-only -> local GP small scale/clip.

## gp_torque bridge build against IMPL Franka stack PASS

- Build underlay:
  - `/opt/ros/humble`
  - `/home/impl-user/impl-groups/group3/ros2_ws/install`
- gp_torque overlay:
  - `install_impl_bridge`
- Built packages:
  - `custom_msgs`
  - `new_bringup`
  - `cpp_relayer`
  - `py_controllers`
- Packages intentionally not built from gp_torque:
  - `franka_hardware`
  - `franka_bringup`
  - `franka_semantic_components`
  - `franka_robot_state_broadcaster`
- These Franka packages are provided by IMPL workspace instead.
- `cpp_relayer` required a small compatibility patch for the IMPL old `FrankaRobotModel` API:
  - `getPoseMatrix(...)` -> `getPose(...)`
  - `getMassMatrix()` -> `getMass()`
  - `getCoriolisForceVector()` -> `getCoriolis()`
  - `getGravityForceVector()` -> `getGravity()`
- `cpp_relayer.xml` still exports:
  - `cpp_relayer/CPPRelayer`
  - `cpp_relayer/UpdateRateDiagnosticController`
- No GP logic, torque computation, trajectory generation, or Python controller behavior was changed.
- Next step:
  - create a temporary IMPL-compatible controller YAML that adds `cpp_relayer`;
  - validate loading/configuring carefully before running any trajectory or GP.

## gp_torque bridge build against IMPL Franka stack PASS

- Build underlay:
  - `/opt/ros/humble`
  - `/home/impl-user/impl-groups/group3/ros2_ws/install`
- gp_torque overlay:
  - `install_impl_bridge`
- Built packages:
  - `custom_msgs`
  - `new_bringup`
  - `cpp_relayer`
  - `py_controllers`
- Packages intentionally not built from gp_torque:
  - `franka_hardware`
  - `franka_bringup`
  - `franka_semantic_components`
  - `franka_robot_state_broadcaster`
- These Franka packages are provided by IMPL workspace instead.
- `cpp_relayer` required a small compatibility patch for the IMPL old `FrankaRobotModel` API:
  - `getPoseMatrix(...)` -> `getPose(...)`
  - `getMassMatrix()` -> `getMass()`
  - `getCoriolisForceVector()` -> `getCoriolis()`
  - `getGravityForceVector()` -> `getGravity()`
- `cpp_relayer.xml` still exports:
  - `cpp_relayer/CPPRelayer`
  - `cpp_relayer/UpdateRateDiagnosticController`
- No GP logic, torque computation, trajectory generation, or Python controller behavior was changed.
- Next step:
  - create a temporary IMPL-compatible controller YAML that adds `cpp_relayer`;
  - validate loading/configuring carefully before running any trajectory or GP.

## cpp_relayer active zero-fallback validation PASS

- Underlay:
  - `/opt/ros/humble`
  - `/home/impl-user/impl-groups/group3/ros2_ws/install`
- Overlay:
  - `~/dongfa/tt_dgp/install_impl_bridge`
- Temporary controller YAML:
  - `/tmp/impl_gp_torque_cpp_relayer_controllers.yaml`
- Runtime result:
  - `cpp_relayer` loaded successfully.
  - `cpp_relayer` configured successfully.
  - `cpp_relayer` activated successfully for a short zero-fallback validation.
  - While active, `panda_joint1/effort` to `panda_joint7/effort` were `[claimed]`.
  - `/state_parameter` published successfully.
  - Sample contained:
    - `position`
    - `velocity`
    - `effort_measured`
    - `gravity`
    - `o_t_f`
    - `mass`
  - `cpp_relayer` deactivated successfully.
  - After deactivation, `panda_joint1/effort` to `panda_joint7/effort` returned to `[unclaimed]`.
- No Python trajectory, no GP prediction, and no GP compensation were run in this validation.
- Interpretation:
  - The IMPL Franka stack can host the gp_torque `cpp_relayer`.
  - The old-API model patch is sufficient for build and runtime state-parameter publication.
  - Next step should be Python-controller compatibility inspection before any trajectory or GP run.

## IMPL underlay + gp_torque overlay checkpoint

- The IMPL underlay + gp_torque overlay build passes.
- `cpp_relayer` uses the IMPL old `FrankaRobotModel` API names:
  - `getPose(franka::Frame::kFlange)`
  - `getMass()`
  - `getCoriolis()`
  - `getGravity()`
- Active zero-fallback validation was manually completed before this checkpoint task.
- `/state_parameter` published model/state fields including `position`, `velocity`, `effort_measured`, `gravity`, `o_t_f`, and `mass`.
- Do not run a Python trajectory, GP prediction, or GP compensation yet.
- Next step: Python controller safety review or an isolated no-trajectory/no-GP launch design.
