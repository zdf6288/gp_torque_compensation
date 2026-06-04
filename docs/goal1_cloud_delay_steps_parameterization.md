# GOAL1 Cloud Delay Steps Parameterization

`delay_steps` is a cloud-like control-step delay used by the GOAL1 big GP path.
It controls which buffered state and residual are used for big-GP input timing
and state selection. It is not real network latency and is not a GP model
hyperparameter.

Changing `delay_steps` does not change the GP kernel, GP capacity, expert size,
training points, or model files.

- Default: `2`, preserving the previous hardcoded GOAL1 behavior.
- Allowed range: `[0, 100]`; invalid values fall back to `2`.
- `delay_steps=0` uses the latest/current buffered state.
- Changing the value affects cloud-like timing and input selection only.

Keep validation offline or on fake hardware until the change has been reviewed.
This parameterization does not add a no-clip mode and does not increase any
torque safety clip.
