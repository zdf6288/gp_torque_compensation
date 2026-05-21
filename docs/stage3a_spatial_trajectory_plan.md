# Stage 3A Spatial Trajectory Plan

## Purpose

Stage 3A adds a conservative, default-off spatial trajectory option for real-robot feasibility validation after Stage 2A/2B. The goal is to make the test trajectory slightly richer than a planar circle while leaving the torque controller and GP compensation core unchanged.

## Trajectory Choice

The new `z_modulated_circle` mode keeps the existing x-y circle and adds a small smooth z modulation:

- `x = center_x + radius * cos(omega * t)`
- `y = center_y + radius * sin(omega * t)`
- `z = center_z + z_amplitude * sin(z_frequency_multiplier * omega * t)`

This is chosen because it is continuous, easy to inspect, starts at the same point as the planar circle, and can be made very small for first validation.

## Default Behavior

The default remains `trajectory_mode=planar_circle` with `z_amplitude=0.0`. Existing planar-circle parameters are unchanged by default.

## Recommended First Validation Parameters

For the first explicit Stage 3A compensation-off baseline run:

- `robot_ip:=<ROBOT_IP>`
- `trajectory_mode:=z_modulated_circle`
- `z_amplitude:=0.005`
- `z_frequency_multiplier:=0.5`
- `circle_frequency:=0.08`
- `transition_duration:=5.0`
- `gp_online_update_enabled:=false`
- `gp_compensation_enabled:=false`

The current launch file does not expose `gp_prediction_enabled`, so the first Stage 3A run should be described as compensation-off rather than strict prediction-off no-GP.

## Validation Order

1. Compensation-off baseline first: compensation off, no online GP update.
2. Compute-only / prediction-logging check second, only if supported by current code or added as a separate implementation task.
3. GP-on conservative last: small compensation scale and clip, after the first two runs look stable.

## Safety Caveats

- Do not claim GP improvement from Stage 3A alone.
- Keep the first validation around 50 Hz.
- Do not modify the torque core for this stage.
- Keep conservative GP compensation settings.
- Stop immediately if vibration, abnormal sound, reflex stop, or obvious trajectory abnormality occurs.
