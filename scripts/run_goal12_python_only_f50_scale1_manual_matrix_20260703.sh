#!/usr/bin/env bash
set -euo pipefail

# Split architecture:
# Terminal A owns Franka bringup, hardware, and the 1000 Hz controller bridge.
# Terminal B owns loading/activating cpp_relayer.
# This script is Terminal C only: it starts the Python controller and trajectory publisher.
# It deliberately does not clean up cpp_relayer between successful runs, because reactivation
# belongs to Terminal B and must stay visible to the operator.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PLAN_ONLY=0
LAUNCH_REPEAT="1"
ROUNDS_PER_LAUNCH="6"
SOURCE_FILTER="all"
GP_SCALE="1.0"
GP_SCALE_EXPLICIT=0
GP_CLIP="0.5"
# true GP-off：--gp-off 时把 gp_prediction/online_update/compensation 全部关掉
# 并强制 scale=0.0；与 --scale 0.0（GP 仍在预测、只是零幅值补偿）语义不同。
GP_OFF=0
# 频率参数化：三个 rate 独立可调，--frequency 一次性设置三者（未被显式
# per-rate 覆盖时）。默认 50/50/50 与旧的硬编码 F50 行为一致。
CONTROL_FREQUENCY="50"
TRAJECTORY_PUBLISH_RATE="50"
STATE_PARAMETER_PUBLISH_RATE="50"
CONTROL_FREQUENCY_EXPLICIT=0
TRAJECTORY_PUBLISH_RATE_EXPLICIT=0
STATE_PARAMETER_PUBLISH_RATE_EXPLICIT=0
FREQUENCY_ALL=""
MODEL_DIR="${REPO_ROOT}/outputs/gp_models_extracted_20260625_164901/gp_models"
OUTPUT_ROOT="outputs/manual_compensation"
HIST_DB_PATH=""
NO_CLEANUP=1
TRANSITION_DURATION="60.0"
TORQUE_RATE_LIMIT_NM_PER_S="20.0"
STARTUP_LINEAR_SPEED="0.005"
STARTUP_DISTANCE_WARN_M="0.100"
STARTUP_DISTANCE_REFUSE_M="0.300"
STARTUP_DISTANCE_REFUSE_ENABLED="true"
TRAJECTORY_REFERENCE_MODE="fixed_absolute"
TRAJECTORY_REFERENCE_MODE_EXPLICIT=0
SESSION_RELATIVE_SHORTCUT=0
SESSION_HOME_MODE="fixed"
SESSION_HOME_MODE_EXPLICIT=0
SESSION_HOME_PATH=""
SESSION_RELATIVE_MAX_ANCHOR_DELTA_M="0.250"
SESSION_RELATIVE_WARN_ANCHOR_DELTA_M="0.150"
# raw 默认 refuse 保持旧 hard-refuse 行为；--session-relative shortcut 在用户
# 未显式覆盖时改为 warn（floating anchor：从当前摆好的 pose 整体平移轨迹）。
SESSION_RELATIVE_ANCHOR_DELTA_LIMIT_MODE="refuse"
SESSION_RELATIVE_ANCHOR_DELTA_LIMIT_MODE_EXPLICIT=0
SESSION_RELATIVE_MIN_Z="0.45"
SESSION_RELATIVE_MAX_Z="0.85"
SESSION_RELATIVE_REQUIRES_STABLE_STATE="true"
SESSION_RELATIVE_STABILITY_SAMPLES="10"
SESSION_RELATIVE_STABILITY_POSITION_STD_M="0.003"
SESSION_RELATIVE_APPLY_TO_STARTUP_AND_RETURN="true"
SESSION_RELATIVE_APPLY_TO_TRAJECTORY_CENTER="true"
SESSION_RELATIVE_NOMINAL_TRAJECTORY_START_XYZ="[0.3077306122468523, 0.043799833015107294, 0.6648721535244662]"
SESSION_RELATIVE_NOMINAL_CIRCLE_CENTER_XYZ="[0.3, 0.0, 0.65]"
NORMAL_RUN_START_GATE_ENABLED="false"
NORMAL_RUN_START_WARN_M="0.100"
NORMAL_RUN_START_REFUSE_M="0.150"
EMERGENCY_RETURN_START_REFUSE_M="0.300"
RETURN_ONLY_IF_TOO_FAR_ENABLED="false"
POST_RUN_RETURN="false"
POST_RUN_RETURN_LINEAR_SPEED="0.005"
POST_RUN_RETURN_TIMEOUT_SEC="60.0"
POST_RUN_RETURN_HOLD_SEC="2.0"
SESSION_CASE_INDEX=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_goal12_python_only_f50_scale1_manual_matrix_20260703.sh [options]

Options:
  --plan                 Print the planned Terminal-C-only runs and exit.
  --launch-repeat N      Restart the whole launch N independent times per
                         selected source. Default: 1.
  --repeat N             Alias for --launch-repeat N, kept for compatibility.
                         It does NOT mean internal trajectory rounds.
  --rounds-per-launch N  Run N full trajectory rounds inside each independent
                         launch (wired to trajectory_publisher rounds_per_mode).
                         Default: 6. Each launch should produce one controller
                         CSV containing all internal rounds.
  --source VALUE         local, cloud, combined, triple, triple_dynamic,
                         triple_dynamic_gated, or all.
                         Default: all. all preserves the older
                         local/cloud/combined matrix; use --source triple
                         explicitly for triple fusion on the same anchor.
  --scale VALUE          GP compensation scale. Default: 1.0. Note scale 0.0
                         still runs GP prediction/online update with zero
                         compensation amplitude; use --gp-off for true GP-off.
  --gp-off               True GP-off for clean historical-DB raw runs. Sets
                         gp_prediction_enabled=false,
                         gp_online_update_enabled=false,
                         gp_compensation_enabled=false, and forces
                         gp_compensation_scale=0.0. Conflicts with an explicit
                         non-zero --scale. Default: off (GP enabled as before).
  --frequency VALUE      Convenience switch: set control frequency, trajectory
                         publish rate, and state parameter publish rate to the
                         same value (e.g. 50, 100, 200, 500) unless an explicit
                         per-rate option overrides one of them. Default: 50.
                         F500 is only a parameterized runner target; do not run
                         it on the real robot unless ros2_control update rate,
                         cpp_relayer, controller config, and timing have been
                         explicitly validated.
  --control-frequency VALUE
                         Python controller frequency in Hz. Default: 50.
  --trajectory-publish-rate VALUE
                         Trajectory publisher rate in Hz. Default: 50.
  --state-parameter-publish-rate VALUE
                         State parameter publish rate in Hz. Default: 50.
  --clip VALUE           GP compensation clip in Nm. Default: 0.5.
  --model-dir PATH       GP model directory.
  --output-root DIR      Output root. Default: outputs/manual_compensation.
  --hist-db-path PATH    Historical residual DB .npz path. Required only for
                         --source triple_dynamic_gated in this runner.
  --transition-duration VALUE
                         Fixed-start to trajectory-start transition duration
                         in seconds. Default: 60.0.
  --torque-rate-limit-nm-per-s VALUE
                         Per-joint torque slew-rate limit in Nm/s. Default: 20.0.
  --startup-linear-speed VALUE
                         Startup move-to-fixed-start linear speed in m/s.
                         Default: 0.005.
  --startup-distance-warn-m VALUE
                         Warn threshold for distance to fixed start in m.
                         Default: 0.100.
  --startup-distance-refuse-m VALUE
                         Hard-refuse threshold for distance to fixed start in m.
                         Default: 0.300.
  --startup-distance-refuse-enabled true|false
                         Refuse startup when far from the fixed start.
                         Default: true.
  --trajectory-reference-mode fixed_absolute|session_relative
                         fixed_absolute keeps the old absolute trajectory;
                         session_relative shifts start/center from the shared
                         anchor JSON. Default: fixed_absolute.
  --session-relative     Convenience mode for repeated matrix runs. Sets
                         trajectory_reference_mode=session_relative,
                         post-run return enabled, strict run-start gate enabled,
                         normal_run_start_refuse_m=0.150, and
                         emergency_return_start_refuse_m=0.300. If
                         --session-home-mode was not explicit, the first case
                         uses capture_first and later cases load the same JSON.
                         Unless --session-relative-anchor-delta-limit-mode is
                         explicit, this shortcut also sets it to warn so the
                         current manually placed pose can serve as a floating
                         session home even when it is far from the nominal
                         trajectory start.
  --session-relative-anchor-delta-limit-mode refuse|warn|off
                         Policy when anchor_delta norm exceeds
                         session_relative_max_anchor_delta_m. refuse keeps the
                         legacy hard refuse (default in raw mode without the
                         --session-relative shortcut); warn only warns and
                         continues (default with --session-relative; suits
                         starting from the current pose as a floating session
                         home); off disables this norm check. Has no effect
                         when trajectory_reference_mode=fixed_absolute.
  --session-home-mode fixed|capture_first|load
                         Session home source. capture_first captures the
                         current safe EE pose on the first case and later
                         cases in this invocation automatically use load.
                         Default: fixed.
  --session-home-path PATH
                         Session home JSON path. Default when mode is not
                         fixed: <output-root>/<stamp>/session_home.json.
  --normal-run-start-refuse-m VALUE
                         Refuse official GP run above this distance to the
                         session home (needs --strict-start to enable the
                         gate). Default: 0.150.
  --emergency-return-start-refuse-m VALUE
                         Refuse all automatic motion above this distance to
                         the session home. Default: 0.300.
  --post-run-return true|false
                         Slowly return to session home after each run before
                         exit. Default: false.
  --post-run-return-linear-speed VALUE
                         Post-run return linear speed in m/s. Default: 0.005.
  --post-run-return-timeout-sec VALUE
                         Post-run return timeout in seconds. Default: 60.0.
  --post-run-return-hold-sec VALUE
                         Hold time at session home in seconds. Default: 2.0.
  --strict-start         Enable the run-start gate with
                         normal_run_start_refuse_m=0.150,
                         emergency_return_start_refuse_m=0.300, and
                         post-run return enabled. Later explicit options can
                         still override individual values.
  --no-cleanup           Keep cpp_relayer active between runs. This is the default.
  -h, --help             Show this help.

Before running:
  Terminal A must already own external IMPL bringup/controller_manager.
  Terminal B must already have cpp_relayer active.

Safety:
  scale1 is NOT a post-reflex first rerun. After a Franka reflex, rerun the
  chain in order: predict-only/no-GP -> scale0.25 -> scale0.5 -> scale1.
  Far-start protection is enabled by default (startup_distance_refuse_enabled=true).
USAGE
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

confirm_start_or_die() {
  # 真机启动确认必须来自操作员终端。历史 bug：主循环曾用
  # `done < <(selected_sources)` 重定向整个循环体的 stdin，导致这里的 read
  # 读到 EOF 后被 set -e 静默退出。现在显式从 /dev/tty 读取；拿不到终端或
  # 读取失败一律大声失败，绝不静默通过，也绝不自动继续。
  local answer=""
  echo "Type START to launch this real-robot Python-only case, or Ctrl+C to stop."
  if [[ ! -e /dev/tty ]]; then
    die "no controlling terminal (/dev/tty unavailable); refusing to launch without an explicit interactive START"
  fi
  if ! read -r -p "Type START to launch this real-robot Python-only case: " answer < /dev/tty; then
    die "could not read START confirmation from /dev/tty (no interactive terminal?); refusing to launch"
  fi
  if [[ "${answer}" != "START" ]]; then
    die "operator did not type START"
  fi
}

while (($# > 0)); do
  case "$1" in
    --plan)
      PLAN_ONLY=1
      ;;
    --repeat)
      [[ $# -ge 2 ]] || die "--repeat requires a value"
      LAUNCH_REPEAT="$2"
      shift
      ;;
    --launch-repeat)
      [[ $# -ge 2 ]] || die "--launch-repeat requires a value"
      LAUNCH_REPEAT="$2"
      shift
      ;;
    --rounds-per-launch)
      [[ $# -ge 2 ]] || die "--rounds-per-launch requires a value"
      ROUNDS_PER_LAUNCH="$2"
      shift
      ;;
    --source)
      [[ $# -ge 2 ]] || die "--source requires a value"
      SOURCE_FILTER="$2"
      shift
      ;;
    --scale)
      [[ $# -ge 2 ]] || die "--scale requires a value"
      GP_SCALE="$2"
      GP_SCALE_EXPLICIT=1
      shift
      ;;
    --gp-off)
      GP_OFF=1
      ;;
    --frequency)
      [[ $# -ge 2 ]] || die "--frequency requires a value"
      FREQUENCY_ALL="$2"
      shift
      ;;
    --control-frequency)
      [[ $# -ge 2 ]] || die "--control-frequency requires a value"
      CONTROL_FREQUENCY="$2"
      CONTROL_FREQUENCY_EXPLICIT=1
      shift
      ;;
    --trajectory-publish-rate)
      [[ $# -ge 2 ]] || die "--trajectory-publish-rate requires a value"
      TRAJECTORY_PUBLISH_RATE="$2"
      TRAJECTORY_PUBLISH_RATE_EXPLICIT=1
      shift
      ;;
    --state-parameter-publish-rate)
      [[ $# -ge 2 ]] || die "--state-parameter-publish-rate requires a value"
      STATE_PARAMETER_PUBLISH_RATE="$2"
      STATE_PARAMETER_PUBLISH_RATE_EXPLICIT=1
      shift
      ;;
    --clip)
      [[ $# -ge 2 ]] || die "--clip requires a value"
      GP_CLIP="$2"
      shift
      ;;
    --model-dir)
      [[ $# -ge 2 ]] || die "--model-dir requires a value"
      MODEL_DIR="$2"
      shift
      ;;
    --output-root)
      [[ $# -ge 2 ]] || die "--output-root requires a value"
      OUTPUT_ROOT="$2"
      shift
      ;;
    --hist-db-path)
      [[ $# -ge 2 ]] || die "--hist-db-path requires a value"
      HIST_DB_PATH="$2"
      shift
      ;;
    --transition-duration)
      [[ $# -ge 2 ]] || die "--transition-duration requires a value"
      TRANSITION_DURATION="$2"
      shift
      ;;
    --torque-rate-limit-nm-per-s)
      [[ $# -ge 2 ]] || die "--torque-rate-limit-nm-per-s requires a value"
      TORQUE_RATE_LIMIT_NM_PER_S="$2"
      shift
      ;;
    --startup-linear-speed)
      [[ $# -ge 2 ]] || die "--startup-linear-speed requires a value"
      STARTUP_LINEAR_SPEED="$2"
      shift
      ;;
    --startup-distance-warn-m)
      [[ $# -ge 2 ]] || die "--startup-distance-warn-m requires a value"
      STARTUP_DISTANCE_WARN_M="$2"
      shift
      ;;
    --startup-distance-refuse-m)
      [[ $# -ge 2 ]] || die "--startup-distance-refuse-m requires a value"
      STARTUP_DISTANCE_REFUSE_M="$2"
      shift
      ;;
    --startup-distance-refuse-enabled)
      [[ $# -ge 2 ]] || die "--startup-distance-refuse-enabled requires true or false"
      STARTUP_DISTANCE_REFUSE_ENABLED="$2"
      shift
      ;;
    --trajectory-reference-mode)
      [[ $# -ge 2 ]] || die "--trajectory-reference-mode requires fixed_absolute or session_relative"
      TRAJECTORY_REFERENCE_MODE="$2"
      TRAJECTORY_REFERENCE_MODE_EXPLICIT=1
      shift
      ;;
    --session-relative)
      SESSION_RELATIVE_SHORTCUT=1
      TRAJECTORY_REFERENCE_MODE="session_relative"
      NORMAL_RUN_START_GATE_ENABLED="true"
      NORMAL_RUN_START_REFUSE_M="0.150"
      EMERGENCY_RETURN_START_REFUSE_M="0.300"
      POST_RUN_RETURN="true"
      ;;
    --session-relative-anchor-delta-limit-mode)
      [[ $# -ge 2 ]] || die "--session-relative-anchor-delta-limit-mode requires refuse, warn, or off"
      SESSION_RELATIVE_ANCHOR_DELTA_LIMIT_MODE="$2"
      SESSION_RELATIVE_ANCHOR_DELTA_LIMIT_MODE_EXPLICIT=1
      shift
      ;;
    --session-home-mode)
      [[ $# -ge 2 ]] || die "--session-home-mode requires fixed, capture_first, or load"
      SESSION_HOME_MODE="$2"
      SESSION_HOME_MODE_EXPLICIT=1
      shift
      ;;
    --session-home-path)
      [[ $# -ge 2 ]] || die "--session-home-path requires a value"
      SESSION_HOME_PATH="$2"
      shift
      ;;
    --normal-run-start-refuse-m)
      [[ $# -ge 2 ]] || die "--normal-run-start-refuse-m requires a value"
      NORMAL_RUN_START_REFUSE_M="$2"
      shift
      ;;
    --emergency-return-start-refuse-m)
      [[ $# -ge 2 ]] || die "--emergency-return-start-refuse-m requires a value"
      EMERGENCY_RETURN_START_REFUSE_M="$2"
      shift
      ;;
    --post-run-return)
      [[ $# -ge 2 ]] || die "--post-run-return requires true or false"
      POST_RUN_RETURN="$2"
      shift
      ;;
    --post-run-return-linear-speed)
      [[ $# -ge 2 ]] || die "--post-run-return-linear-speed requires a value"
      POST_RUN_RETURN_LINEAR_SPEED="$2"
      shift
      ;;
    --post-run-return-timeout-sec)
      [[ $# -ge 2 ]] || die "--post-run-return-timeout-sec requires a value"
      POST_RUN_RETURN_TIMEOUT_SEC="$2"
      shift
      ;;
    --post-run-return-hold-sec)
      [[ $# -ge 2 ]] || die "--post-run-return-hold-sec requires a value"
      POST_RUN_RETURN_HOLD_SEC="$2"
      shift
      ;;
    --strict-start)
      NORMAL_RUN_START_GATE_ENABLED="true"
      NORMAL_RUN_START_REFUSE_M="0.150"
      EMERGENCY_RETURN_START_REFUSE_M="0.300"
      POST_RUN_RETURN="true"
      ;;
    --no-cleanup)
      NO_CLEANUP=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
  shift
done

[[ "${LAUNCH_REPEAT}" =~ ^[1-9][0-9]*$ ]] || die "--launch-repeat/--repeat must be a positive integer"
[[ "${ROUNDS_PER_LAUNCH}" =~ ^[1-9][0-9]*$ ]] || die "--rounds-per-launch must be a positive integer"

is_positive_number() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]] || return 1
  awk -v v="$1" 'BEGIN { exit (v > 0) ? 0 : 1 }'
}

# --frequency 一次性设置三个 rate；显式 per-rate 选项优先，与出现顺序无关。
if [[ -n "${FREQUENCY_ALL}" ]]; then
  is_positive_number "${FREQUENCY_ALL}" || die "--frequency must be a positive number, got: ${FREQUENCY_ALL}"
  (( ! CONTROL_FREQUENCY_EXPLICIT )) && CONTROL_FREQUENCY="${FREQUENCY_ALL}"
  (( ! TRAJECTORY_PUBLISH_RATE_EXPLICIT )) && TRAJECTORY_PUBLISH_RATE="${FREQUENCY_ALL}"
  (( ! STATE_PARAMETER_PUBLISH_RATE_EXPLICIT )) && STATE_PARAMETER_PUBLISH_RATE="${FREQUENCY_ALL}"
fi
is_positive_number "${CONTROL_FREQUENCY}" || die "--control-frequency must be a positive number, got: ${CONTROL_FREQUENCY}"
is_positive_number "${TRAJECTORY_PUBLISH_RATE}" || die "--trajectory-publish-rate must be a positive number, got: ${TRAJECTORY_PUBLISH_RATE}"
is_positive_number "${STATE_PARAMETER_PUBLISH_RATE}" || die "--state-parameter-publish-rate must be a positive number, got: ${STATE_PARAMETER_PUBLISH_RATE}"

# true GP-off 与显式非零 --scale 冲突：宁可 fail fast 也不静默覆盖操作员意图。
if ((GP_OFF)) && ((GP_SCALE_EXPLICIT)) && [[ ! "${GP_SCALE}" =~ ^0+([.]0+)?$ ]]; then
  die "--gp-off forces gp_compensation_scale=0.0; drop --scale or pass --scale 0.0 explicitly"
fi

if ((GP_OFF)); then
  GP_OFF_BOOL="true"
  EFFECTIVE_GP_PREDICTION_ENABLED="false"
  EFFECTIVE_GP_ONLINE_UPDATE_ENABLED="false"
  EFFECTIVE_GP_COMPENSATION_ENABLED="false"
  EFFECTIVE_GP_SCALE="0.0"
else
  GP_OFF_BOOL="false"
  EFFECTIVE_GP_PREDICTION_ENABLED="true"
  EFFECTIVE_GP_ONLINE_UPDATE_ENABLED="true"
  EFFECTIVE_GP_COMPENSATION_ENABLED="true"
  EFFECTIVE_GP_SCALE="${GP_SCALE}"
fi

rate_exceeds_200() {
  awk -v v="$1" 'BEGIN { exit (v > 200) ? 0 : 1 }'
}

print_high_frequency_warning_if_needed() {
  if rate_exceeds_200 "${CONTROL_FREQUENCY}" \
    || rate_exceeds_200 "${TRAJECTORY_PUBLISH_RATE}" \
    || rate_exceeds_200 "${STATE_PARAMETER_PUBLISH_RATE}"; then
    echo "########################################################################"
    echo "# HIGH-FREQUENCY WARNING (>200 Hz requested)"
    echo "# F500 is only a parameterized runner target; do not run on real robot"
    echo "# unless ros2_control update rate, cpp_relayer, controller config, and"
    echo "# timing have been explicitly validated."
    echo "########################################################################"
  fi
}

case "${SOURCE_FILTER}" in
  local|cloud|combined|triple|triple_dynamic|triple_dynamic_gated|all)
    ;;
  *)
    die "--source must be local, cloud, combined, triple, triple_dynamic, triple_dynamic_gated, or all"
    ;;
esac

if [[ "${SOURCE_FILTER}" == "triple_dynamic_gated" ]]; then
  [[ -n "${HIST_DB_PATH}" ]] || die "--source triple_dynamic_gated requires --hist-db-path"
  [[ -f "${HIST_DB_PATH}" ]] || die "--hist-db-path does not exist or is not a regular file: ${HIST_DB_PATH}"
fi

case "${TRAJECTORY_REFERENCE_MODE}" in
  fixed_absolute|session_relative)
    ;;
  *)
    die "--trajectory-reference-mode must be fixed_absolute or session_relative"
    ;;
esac

if ((SESSION_RELATIVE_SHORTCUT)) && [[ "${TRAJECTORY_REFERENCE_MODE}" != "session_relative" ]]; then
  die "--session-relative conflicts with --trajectory-reference-mode fixed_absolute"
fi

# --session-relative shortcut 默认 floating anchor（warn）；显式
# --session-relative-anchor-delta-limit-mode 始终优先，与选项顺序无关。
# raw 模式（含显式 --trajectory-reference-mode session_relative）保持 refuse。
if ((SESSION_RELATIVE_SHORTCUT)) && (( ! SESSION_RELATIVE_ANCHOR_DELTA_LIMIT_MODE_EXPLICIT )); then
  SESSION_RELATIVE_ANCHOR_DELTA_LIMIT_MODE="warn"
fi

case "${SESSION_RELATIVE_ANCHOR_DELTA_LIMIT_MODE}" in
  refuse|warn|off)
    ;;
  *)
    die "--session-relative-anchor-delta-limit-mode must be refuse, warn, or off"
    ;;
esac

case "${STARTUP_DISTANCE_REFUSE_ENABLED}" in
  true|false)
    ;;
  *)
    die "--startup-distance-refuse-enabled must be true or false"
    ;;
esac

if [[ "${TRAJECTORY_REFERENCE_MODE}" == "session_relative" ]]; then
  if (( ! SESSION_HOME_MODE_EXPLICIT )); then
    SESSION_HOME_MODE="capture_first"
  elif [[ "${SESSION_HOME_MODE}" == "fixed" ]]; then
    die "trajectory_reference_mode=session_relative requires --session-home-mode capture_first or load"
  fi
fi

case "${SESSION_HOME_MODE}" in
  fixed|capture_first|load)
    ;;
  *)
    die "--session-home-mode must be fixed, capture_first, or load"
    ;;
esac

case "${POST_RUN_RETURN}" in
  true|false)
    ;;
  *)
    die "--post-run-return must be true or false"
    ;;
esac

cd "${REPO_ROOT}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT%/}/${RUN_STAMP}"
MANIFEST_PATH="${OUTPUT_ROOT}/manifest.csv"

if [[ "${SESSION_HOME_MODE}" != "fixed" && -z "${SESSION_HOME_PATH}" ]]; then
  SESSION_HOME_PATH="${OUTPUT_ROOT}/session_home.json"
fi

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=75
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}/new_structure/gp:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

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

scale_tag() {
  case "$1" in
    0|0.0|0.00) printf '0' ;;
    1|1.0|1.00) printf '1' ;;
    0.5|.5|0.50|.50) printf '05' ;;
    0.25|.25|0.250|.250) printf '025' ;;
    *) printf '%s' "$1" | tr -cd '[:alnum:]' ;;
  esac
}

freq_tag() {
  printf '%s' "$1" | tr -cd '[:alnum:]'
}

source_upper() {
  printf '%s' "$1" | tr '[:lower:]' '[:upper:]'
}

selected_sources() {
  if [[ "${SOURCE_FILTER}" == "all" ]]; then
    printf '%s\n' local cloud combined
  else
    printf '%s\n' "${SOURCE_FILTER}"
  fi
}

print_repo_status() {
  echo "== Repository =="
  echo "pwd: $(pwd)"
  echo "branch: $(git branch --show-current)"
  echo "HEAD: $(git rev-parse --short HEAD)"
  echo "git status --short:"
  git status --short
  echo
}

init_manifest() {
  mkdir -p "${OUTPUT_ROOT}"
  printf 'case_name,run_name,data_output_dir,source,scale,clip,gp_off,j7_disabled,control_frequency,trajectory_publish_rate,state_parameter_publish_rate,launch_repeat,rounds_per_launch,trajectory_reference_mode,session_home_mode,session_home_path,post_run_return,status\n' > "${MANIFEST_PATH}"
}

append_manifest_row() {
  local case_name="$1"
  local run_name="$2"
  local data_output_dir="$3"
  local source="$4"
  local effective_session_home_mode="$5"
  local status="$6"
  printf '%s,%s,%s,%s,%s,%s,%s,true,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${case_name}" "${run_name}" "${data_output_dir}" "${source}" \
    "${EFFECTIVE_GP_SCALE}" "${GP_CLIP}" "${GP_OFF_BOOL}" \
    "${CONTROL_FREQUENCY}" "${TRAJECTORY_PUBLISH_RATE}" \
    "${STATE_PARAMETER_PUBLISH_RATE}" "${LAUNCH_REPEAT}" "${ROUNDS_PER_LAUNCH}" \
    "${TRAJECTORY_REFERENCE_MODE}" \
    "${effective_session_home_mode}" "${SESSION_HOME_PATH}" \
    "${POST_RUN_RETURN}" "${status}" >> "${MANIFEST_PATH}"
}

cpp_relayer_list_shows_active() {
  # ros2controlcli 输出可能带 ANSI 颜色码（例如 ^[[92mcpp_relayer...^[[0m 和
  # ^[[92mactive^[[0m），直接按字段比较会误报 inactive。先剥离颜色码和 \r，
  # 再要求首字段是 cpp_relayer 且末字段（state）是 active；中间的
  # controller type 列（cpp_relayer/CPPRelayer）不参与匹配。
  awk '
    {
      gsub(/\033\[[0-9;]*m/, "")
      sub(/\r$/, "")
    }
    $1 == "cpp_relayer" && $NF == "active" { found = 1 }
    END { exit found ? 0 : 1 }
  ' "$@"
}

check_cpp_relayer_active() {
  local output
  local controllers_log
  controllers_log="outputs/runtime_logs/grouped_runner_list_controllers_$(date +%Y%m%d_%H%M%S).txt"
  mkdir -p outputs/runtime_logs
  if ! output="$(timeout 5 ros2 control list_controllers --controller-manager /controller_manager 2>&1)"; then
    printf '%s\n' "${output}" > "${controllers_log}"
    printf '%s\n' "${output}"
    echo "Controller-list diagnostic saved to ${controllers_log}"
    die "Cannot reach external controller manager. Start/check Terminal A before Terminal C."
  fi
  printf '%s\n' "${output}" > "${controllers_log}"
  printf '%s\n' "${output}"
  echo "Controller-list diagnostic saved to ${controllers_log}"
  if ! cpp_relayer_list_shows_active "${controllers_log}"; then
    die "cpp_relayer is not active. Run Terminal B load/activate sequence before continuing."
  fi
}

run_one_case() {
  local source="$1"
  local repeat_index="$2"
  local source_tag
  local gpoff_name_tag
  local case_name
  local run_name
  local data_output_dir
  local effective_session_home_mode
  local effective_capture_enabled
  local effective_session_relative_capture_enabled

  # capture_first 只对本次 session 的第一个 case 生效；
  # 之后的 case 自动改用 load 复用同一个 session_home.json。
  effective_session_home_mode="${SESSION_HOME_MODE}"
  if [[ "${SESSION_HOME_MODE}" == "capture_first" && ${SESSION_CASE_INDEX} -gt 0 ]]; then
    effective_session_home_mode="load"
  fi
  if [[ "${effective_session_home_mode}" == "capture_first" ]]; then
    effective_capture_enabled="true"
  else
    effective_capture_enabled="false"
  fi
  if [[ "${TRAJECTORY_REFERENCE_MODE}" == "session_relative" && "${effective_session_home_mode}" == "capture_first" ]]; then
    effective_session_relative_capture_enabled="true"
  else
    effective_session_relative_capture_enabled="false"
  fi
  SESSION_CASE_INDEX=$((SESSION_CASE_INDEX + 1))

  source_tag="$(source_upper "${source}")"
  gpoff_name_tag=""
  ((GP_OFF)) && gpoff_name_tag="_GPOFF"
  case_name="COMP_${source_tag}_F$(freq_tag "${CONTROL_FREQUENCY}")${gpoff_name_tag}_SCALE$(scale_tag "${EFFECTIVE_GP_SCALE}")_CLIP$(scale_tag "${GP_CLIP}")_J7OFF_1000HZ_BRIDGE_ROUND${ROUNDS_PER_LAUNCH}_R${repeat_index}"
  run_name="${case_name}_${RUN_STAMP}"
  data_output_dir="${OUTPUT_ROOT}/${case_name}"

  local launch_args=(
    "run_name:=${run_name}"
    "data_output_dir:=${data_output_dir}"
    "csv_output_profile:=full"
    "trajectory_mode:=goal1_spatial_multisine"
    "circle_frequency:=0.05"
    "control_frequency:=${CONTROL_FREQUENCY}"
    "trajectory_publish_rate:=${TRAJECTORY_PUBLISH_RATE}"
    "state_parameter_publish_rate:=${STATE_PARAMETER_PUBLISH_RATE}"
    "rounds_per_mode:=${ROUNDS_PER_LAUNCH}"
    "transition_duration:=${TRANSITION_DURATION}"
    "torque_rate_limit_nm_per_s:=${TORQUE_RATE_LIMIT_NM_PER_S}"
    "startup_linear_speed:=${STARTUP_LINEAR_SPEED}"
    "startup_distance_warn_m:=${STARTUP_DISTANCE_WARN_M}"
    "startup_distance_refuse_m:=${STARTUP_DISTANCE_REFUSE_M}"
    "startup_distance_refuse_enabled:=${STARTUP_DISTANCE_REFUSE_ENABLED}"
    "trajectory_reference_mode:=${TRAJECTORY_REFERENCE_MODE}"
    "session_home_mode:=${effective_session_home_mode}"
    "session_home_path:=${SESSION_HOME_PATH}"
    "session_home_capture_enabled:=${effective_capture_enabled}"
    "session_relative_capture_enabled:=${effective_session_relative_capture_enabled}"
    "session_relative_max_anchor_delta_m:=${SESSION_RELATIVE_MAX_ANCHOR_DELTA_M}"
    "session_relative_warn_anchor_delta_m:=${SESSION_RELATIVE_WARN_ANCHOR_DELTA_M}"
    "session_relative_anchor_delta_limit_mode:=${SESSION_RELATIVE_ANCHOR_DELTA_LIMIT_MODE}"
    "session_relative_min_z:=${SESSION_RELATIVE_MIN_Z}"
    "session_relative_max_z:=${SESSION_RELATIVE_MAX_Z}"
    "session_relative_requires_stable_state:=${SESSION_RELATIVE_REQUIRES_STABLE_STATE}"
    "session_relative_stability_samples:=${SESSION_RELATIVE_STABILITY_SAMPLES}"
    "session_relative_stability_position_std_m:=${SESSION_RELATIVE_STABILITY_POSITION_STD_M}"
    "session_relative_apply_to_startup_and_return:=${SESSION_RELATIVE_APPLY_TO_STARTUP_AND_RETURN}"
    "session_relative_apply_to_trajectory_center:=${SESSION_RELATIVE_APPLY_TO_TRAJECTORY_CENTER}"
    "session_relative_nominal_trajectory_start_xyz:=${SESSION_RELATIVE_NOMINAL_TRAJECTORY_START_XYZ}"
    "session_relative_nominal_circle_center_xyz:=${SESSION_RELATIVE_NOMINAL_CIRCLE_CENTER_XYZ}"
    "normal_run_start_gate_enabled:=${NORMAL_RUN_START_GATE_ENABLED}"
    "normal_run_start_warn_m:=${NORMAL_RUN_START_WARN_M}"
    "normal_run_start_refuse_m:=${NORMAL_RUN_START_REFUSE_M}"
    "emergency_return_start_refuse_m:=${EMERGENCY_RETURN_START_REFUSE_M}"
    "return_only_if_too_far_enabled:=${RETURN_ONLY_IF_TOO_FAR_ENABLED}"
    "post_run_return_to_session_home_enabled:=${POST_RUN_RETURN}"
    "post_run_return_linear_speed:=${POST_RUN_RETURN_LINEAR_SPEED}"
    "post_run_return_timeout_sec:=${POST_RUN_RETURN_TIMEOUT_SEC}"
    "post_run_return_hold_sec:=${POST_RUN_RETURN_HOLD_SEC}"
    "gp_model_dir:=${MODEL_DIR}"
    "gp_prediction_enabled:=${EFFECTIVE_GP_PREDICTION_ENABLED}"
    "gp_online_update_enabled:=${EFFECTIVE_GP_ONLINE_UPDATE_ENABLED}"
    "gp_compensation_enabled:=${EFFECTIVE_GP_COMPENSATION_ENABLED}"
    "gp_compensation_source:=${source}"
    "gp_compensation_scale:=${EFFECTIVE_GP_SCALE}"
    "gp_compensation_clip_nm:=${GP_CLIP}"
    "gp_compensation_disable_joint7:=true"
    "delay_steps:=0"
    "gp_prediction_stride:=5"
    "future_trajectory_request_stride:=5"
    "gp_output_timeout_sec:=0.5"
  )

  if [[ "${source}" == "triple_dynamic_gated" ]]; then
    launch_args+=(
      "gp_historical_db_enabled:=true"
      "gp_historical_db_path:=${HIST_DB_PATH}"
      "gp_historical_db_preflight_enabled:=true"
      "gp_historical_db_preflight_required:=true"
      "gp_historical_db_disable_when_online_update:=false"
      "gp_disable_silent_hist_fallback:=true"
    )
  fi

  echo "========================================================================"
  echo "Terminal-C-only case: ${case_name}"
  echo "run_name: ${run_name}"
  echo "data_output_dir: ${data_output_dir}"
  echo "source: ${source}, scale: ${EFFECTIVE_GP_SCALE}, clip: ${GP_CLIP}, J7OFF: true"
  echo "gp_off=${GP_OFF_BOOL}"
  echo "gp_prediction_enabled=${EFFECTIVE_GP_PREDICTION_ENABLED}"
  echo "gp_online_update_enabled=${EFFECTIVE_GP_ONLINE_UPDATE_ENABLED}"
  echo "gp_compensation_enabled=${EFFECTIVE_GP_COMPENSATION_ENABLED}"
  echo "gp_compensation_scale=${EFFECTIVE_GP_SCALE}"
  if [[ "${source}" == "triple_dynamic_gated" ]]; then
    echo "hist_db_path=${HIST_DB_PATH}"
    echo "hist_db_preflight_required=true"
    echo "gp_historical_db_disable_when_online_update=false"
    echo "gp_disable_silent_hist_fallback=true"
  fi
  echo "control_frequency=${CONTROL_FREQUENCY}"
  echo "trajectory_publish_rate=${TRAJECTORY_PUBLISH_RATE}"
  echo "state_parameter_publish_rate=${STATE_PARAMETER_PUBLISH_RATE}"
  echo "launch_repeat_index=${repeat_index}/${LAUNCH_REPEAT}"
  echo "rounds_per_launch=${ROUNDS_PER_LAUNCH} (one launch, one controller CSV containing all rounds)"
  echo "transition_duration=${TRANSITION_DURATION}"
  echo "torque_rate_limit_nm_per_s=${TORQUE_RATE_LIMIT_NM_PER_S}"
  echo "startup_linear_speed=${STARTUP_LINEAR_SPEED}"
  echo "startup_distance_warn_m=${STARTUP_DISTANCE_WARN_M}"
  echo "startup_distance_refuse_m=${STARTUP_DISTANCE_REFUSE_M}"
  echo "startup_distance_refuse_enabled=${STARTUP_DISTANCE_REFUSE_ENABLED}"
  echo "trajectory_reference_mode=${TRAJECTORY_REFERENCE_MODE}"
  echo "session_home_mode=${effective_session_home_mode}"
  echo "session_home/session_anchor path=${SESSION_HOME_PATH}"
  echo "session_relative_capture_enabled=${effective_session_relative_capture_enabled}"
  echo "session_relative_max_anchor_delta_m=${SESSION_RELATIVE_MAX_ANCHOR_DELTA_M}"
  echo "session_relative_warn_anchor_delta_m=${SESSION_RELATIVE_WARN_ANCHOR_DELTA_M}"
  echo "session_relative_anchor_delta_limit_mode=${SESSION_RELATIVE_ANCHOR_DELTA_LIMIT_MODE}"
  echo "session_relative_min_z=${SESSION_RELATIVE_MIN_Z}"
  echo "session_relative_max_z=${SESSION_RELATIVE_MAX_Z}"
  echo "session_relative_requires_stable_state=${SESSION_RELATIVE_REQUIRES_STABLE_STATE}"
  echo "session_relative_stability_samples=${SESSION_RELATIVE_STABILITY_SAMPLES}"
  echo "session_relative_stability_position_std_m=${SESSION_RELATIVE_STABILITY_POSITION_STD_M}"
  echo "session_relative_apply_to_startup_and_return=${SESSION_RELATIVE_APPLY_TO_STARTUP_AND_RETURN}"
  echo "session_relative_apply_to_trajectory_center=${SESSION_RELATIVE_APPLY_TO_TRAJECTORY_CENTER}"
  echo "session_relative_nominal_trajectory_start_xyz=${SESSION_RELATIVE_NOMINAL_TRAJECTORY_START_XYZ}"
  echo "session_relative_nominal_circle_center_xyz=${SESSION_RELATIVE_NOMINAL_CIRCLE_CENTER_XYZ}"
  echo "normal_run_start_gate_enabled=${NORMAL_RUN_START_GATE_ENABLED}"
  echo "normal_run_start_warn_m=${NORMAL_RUN_START_WARN_M}"
  echo "normal_run_start_refuse_m=${NORMAL_RUN_START_REFUSE_M}"
  echo "emergency_return_start_refuse_m=${EMERGENCY_RETURN_START_REFUSE_M}"
  echo "post_run_return_to_session_home_enabled=${POST_RUN_RETURN}"
  echo "post_run_return_linear_speed=${POST_RUN_RETURN_LINEAR_SPEED}"
  echo "post_run_return_timeout_sec=${POST_RUN_RETURN_TIMEOUT_SEC}"
  echo "post_run_return_hold_sec=${POST_RUN_RETURN_HOLD_SEC}"
  echo "This will start only the Python-only compensation trajectory launch."
  echo "========================================================================"

  if ((PLAN_ONLY)); then
    printf 'ros2 launch py_controllers cartesian_impedance_python_only_compensation_trajectory_launch.py'
    printf ' %q' "${launch_args[@]}"
    printf '\n'
    return 0
  fi

  check_cpp_relayer_active
  echo "cpp_relayer active check passed; proceeding to START confirmation."
  confirm_start_or_die

  mkdir -p "${data_output_dir}"
  append_manifest_row "${case_name}" "${run_name}" "${data_output_dir}" "${source}" "${effective_session_home_mode}" "starting"
  if ros2 launch py_controllers cartesian_impedance_python_only_compensation_trajectory_launch.py "${launch_args[@]}"; then
    append_manifest_row "${case_name}" "${run_name}" "${data_output_dir}" "${source}" "${effective_session_home_mode}" "launch_exited_zero"
  else
    append_manifest_row "${case_name}" "${run_name}" "${data_output_dir}" "${source}" "${effective_session_home_mode}" "launch_failed"
    echo "Terminal-C launch failed or was interrupted."
    echo "Default behavior is no cleanup. Check robot state and Terminal B manually."
    if ((NO_CLEANUP)); then
      echo "If cleanup is needed, run scripts/lab_deactivate_cpp_relayer_safe_20260703.sh manually, then reactivate cpp_relayer in Terminal B."
    fi
    exit 1
  fi

  echo "Case exited. Inspect robot state, Terminal logs, and CSV before continuing."
  echo "cpp_relayer is intentionally left active; next iteration will verify it before launch."
}

print_scale1_safety_warning() {
  echo "########################################################################"
  echo "# SAFETY WARNING: scale1 runner"
  echo "# scale1 is NOT a post-reflex first rerun. After any Franka reflex,"
  echo "# rerun the chain in order:"
  echo "#   predict-only/no-GP -> scale0.25 -> scale0.5 -> scale1"
  echo "# Far-start protection is enabled by default:"
  echo "#   startup_distance_refuse_enabled=${STARTUP_DISTANCE_REFUSE_ENABLED}"
  echo "#   (refuse threshold startup_distance_refuse_m=${STARTUP_DISTANCE_REFUSE_M} m)"
  echo "########################################################################"
  echo
}

main() {
  print_repo_status
  print_scale1_safety_warning
  if ((PLAN_ONLY)); then
    echo "Plan mode: no output directories or manifest files will be created."
  else
    init_manifest
  fi
  echo "Output root: ${OUTPUT_ROOT}"
  echo "Manifest: ${MANIFEST_PATH}"
  echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
  echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
  echo "Model dir: ${MODEL_DIR}"
  if [[ -n "${HIST_DB_PATH}" ]]; then
    echo "Hist DB path: ${HIST_DB_PATH}"
  fi
  echo "Cleanup between successful runs: disabled"
  echo "== GP compensation / frequency / repeat =="
  echo "gp_off=${GP_OFF_BOOL}"
  echo "gp_prediction_enabled=${EFFECTIVE_GP_PREDICTION_ENABLED}"
  echo "gp_online_update_enabled=${EFFECTIVE_GP_ONLINE_UPDATE_ENABLED}"
  echo "gp_compensation_enabled=${EFFECTIVE_GP_COMPENSATION_ENABLED}"
  echo "gp_compensation_scale=${EFFECTIVE_GP_SCALE}"
  echo "control_frequency=${CONTROL_FREQUENCY}"
  echo "trajectory_publish_rate=${TRAJECTORY_PUBLISH_RATE}"
  echo "state_parameter_publish_rate=${STATE_PARAMETER_PUBLISH_RATE}"
  echo "launch_repeat=${LAUNCH_REPEAT}"
  echo "rounds_per_launch=${ROUNDS_PER_LAUNCH}"
  echo "NOTE: --repeat is an alias for --launch-repeat. Internal trajectory"
  echo "rounds are controlled only by --rounds-per-launch."
  print_high_frequency_warning_if_needed
  echo "== Trajectory reference / session anchor =="
  echo "trajectory_reference_mode=${TRAJECTORY_REFERENCE_MODE}"
  echo "session_home/session_anchor path=${SESSION_HOME_PATH}"
  echo "session_relative_max_anchor_delta_m=${SESSION_RELATIVE_MAX_ANCHOR_DELTA_M}"
  echo "session_relative_warn_anchor_delta_m=${SESSION_RELATIVE_WARN_ANCHOR_DELTA_M}"
  echo "session_relative_anchor_delta_limit_mode=${SESSION_RELATIVE_ANCHOR_DELTA_LIMIT_MODE}"
  echo "session_relative_min_z=${SESSION_RELATIVE_MIN_Z}"
  echo "session_relative_max_z=${SESSION_RELATIVE_MAX_Z}"
  echo "session_relative_requires_stable_state=${SESSION_RELATIVE_REQUIRES_STABLE_STATE}"
  echo "session_relative_stability_samples=${SESSION_RELATIVE_STABILITY_SAMPLES}"
  echo "session_relative_stability_position_std_m=${SESSION_RELATIVE_STABILITY_POSITION_STD_M}"
  echo "session_relative_apply_to_startup_and_return=${SESSION_RELATIVE_APPLY_TO_STARTUP_AND_RETURN}"
  echo "session_relative_apply_to_trajectory_center=${SESSION_RELATIVE_APPLY_TO_TRAJECTORY_CENTER}"
  echo "session_relative_nominal_trajectory_start_xyz=${SESSION_RELATIVE_NOMINAL_TRAJECTORY_START_XYZ}"
  echo "session_relative_nominal_circle_center_xyz=${SESSION_RELATIVE_NOMINAL_CIRCLE_CENTER_XYZ}"
  echo "== Session home / return cleanup =="
  echo "session_home_mode=${SESSION_HOME_MODE}"
  echo "session_home_path=${SESSION_HOME_PATH}"
  echo "normal_run_start_gate_enabled=${NORMAL_RUN_START_GATE_ENABLED}"
  echo "normal_run_start_warn_m=${NORMAL_RUN_START_WARN_M}"
  echo "normal_run_start_refuse_m=${NORMAL_RUN_START_REFUSE_M}"
  echo "emergency_return_start_refuse_m=${EMERGENCY_RETURN_START_REFUSE_M}"
  echo "return_only_if_too_far_enabled=${RETURN_ONLY_IF_TOO_FAR_ENABLED}"
  echo "post_run_return_to_session_home_enabled=${POST_RUN_RETURN}"
  echo "post_run_return_linear_speed=${POST_RUN_RETURN_LINEAR_SPEED}"
  echo "post_run_return_timeout_sec=${POST_RUN_RETURN_TIMEOUT_SEC}"
  echo "post_run_return_hold_sec=${POST_RUN_RETURN_HOLD_SEC}"
  if [[ "${SESSION_HOME_MODE}" == "capture_first" ]]; then
    echo "NOTE: capture_first applies to the first case only; later cases in"
    echo "this invocation automatically load ${SESSION_HOME_PATH}."
  fi
  echo

  # 用 mapfile 代替 `done < <(selected_sources)`：后者会把整个循环体
  # （包括 run_one_case 里的 START 确认 read）的 stdin 重定向到
  # process substitution，导致确认读到 EOF 后 set -e 静默退出。
  local sources=()
  mapfile -t sources < <(selected_sources)
  local source
  local repeat_index
  for source in "${sources[@]}"; do
    for ((repeat_index = 1; repeat_index <= LAUNCH_REPEAT; repeat_index++)); do
      run_one_case "${source}" "${repeat_index}"
    done
  done
}

main "$@"
