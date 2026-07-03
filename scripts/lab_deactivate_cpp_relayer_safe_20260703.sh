#!/usr/bin/env bash
set -euo pipefail

# 只用于 lab cleanup：尝试把已加载的 cpp_relayer 置为 inactive。
# 所有 ROS2 命令都带 timeout，避免网络/daemon/controller_manager 异常时无限等待。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=75

mkdir -p outputs/runtime_logs

log() {
  printf '[cpp_relayer-cleanup] %s\n' "$*"
}

source_environment() {
  set +u
  # shellcheck source=/opt/ros/humble/setup.bash
  source /opt/ros/humble/setup.bash
  # shellcheck source=/dev/null
  source /home/impl-user/impl-groups/group3/ros2_ws/install/setup.bash
  # shellcheck source=/dev/null
  source /home/impl-user/dongfa/tt_dgp/install_impl_bridge/setup.bash
  set -u
}

source_environment

log "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
log "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
log "Checking /controller_manager reachability with timeout."

if ! controllers_output="$(
  timeout 5 ros2 control list_controllers --controller-manager /controller_manager 2>&1
)"; then
  log "/controller_manager is not reachable or list_controllers timed out."
  log "No cleanup action was taken; exiting successfully because this helper is best-effort."
  exit 0
fi

printf '%s\n' "${controllers_output}"

cpp_line="$(printf '%s\n' "${controllers_output}" | awk '$1 == "cpp_relayer" {print; exit}')"
if [[ -z "${cpp_line}" ]]; then
  log "cpp_relayer is not loaded; no cleanup action is needed."
  exit 0
fi

cpp_state="$(printf '%s\n' "${cpp_line}" | awk '{print $3}')"
log "cpp_relayer state appears to be '${cpp_state}'."

if [[ "${cpp_state}" != "active" ]]; then
  log "cpp_relayer is not active; no deactivate command is needed."
  exit 0
fi

log "Deactivating cpp_relayer with timeout. This should zero command interfaces in cpp_relayer::on_deactivate()."
if timeout 10 ros2 control set_controller_state cpp_relayer inactive --controller-manager /controller_manager; then
  log "cpp_relayer deactivate command completed."
else
  log "ERROR: cpp_relayer deactivate command failed or timed out."
  log "Check Terminal A/B state manually before any further robot-facing run."
  exit 1
fi

