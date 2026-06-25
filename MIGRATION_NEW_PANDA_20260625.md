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
