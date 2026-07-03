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

strip_ansi_and_cr() {
  # ros2controlcli 输出可能带 ANSI 颜色码（如 ^[[92mactive^[[0m）和 \r，
  # 不剥离会导致字段匹配失败，把 active 的 cpp_relayer 误判为未加载/未激活，
  # 从而静默跳过 deactivate 清理。
  awk '{ gsub(/\033\[[0-9;]*m/, ""); sub(/\r$/, "") } 1'
}

cpp_relayer_is_active() {
  # 先剥离颜色码/\r 再匹配：首字段必须是 cpp_relayer，末字段（state）必须是
  # active；中间的 controller type 列（cpp_relayer/CPPRelayer）不参与匹配。
  awk '
    {
      gsub(/\033\[[0-9;]*m/, "")
      sub(/\r$/, "")
    }
    $1 == "cpp_relayer" && $NF == "active" { found = 1 }
    END { exit found ? 0 : 1 }
  '
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

controllers_log="outputs/runtime_logs/cpp_relayer_cleanup_list_controllers_$(date +%Y%m%d_%H%M%S).txt"
if ! controllers_output="$(
  timeout 5 ros2 control list_controllers --controller-manager /controller_manager 2>&1
)"; then
  printf '%s\n' "${controllers_output}" > "${controllers_log}"
  log "/controller_manager is not reachable or list_controllers timed out."
  log "Controller-list diagnostic saved to ${controllers_log}"
  log "No cleanup action was taken; exiting successfully because this helper is best-effort."
  exit 0
fi

printf '%s\n' "${controllers_output}" > "${controllers_log}"
log "Controller-list diagnostic saved to ${controllers_log}"
printf '%s\n' "${controllers_output}"

cpp_line="$(printf '%s\n' "${controllers_output}" | strip_ansi_and_cr | awk '$1 == "cpp_relayer" {print; exit}')"
if [[ -z "${cpp_line}" ]]; then
  log "cpp_relayer is not loaded; no cleanup action is needed."
  exit 0
fi

cpp_state="$(printf '%s\n' "${cpp_line}" | awk '{print $NF}')"
log "cpp_relayer state appears to be '${cpp_state}'."

if ! printf '%s\n' "${controllers_output}" | cpp_relayer_is_active; then
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
