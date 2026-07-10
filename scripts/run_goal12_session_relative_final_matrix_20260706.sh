#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="core"
FREQUENCIES="50,100,200"
FREQUENCY_OVERRIDE=0
ANCHOR="outputs/session_relative_floating_anchor_scale0_smoke_20260706_144203/session_home.json"
HIST_DB_PATH=""
HOME_PREFLIGHT_CURRENT_Q=""
HOME_PREFLIGHT_CURRENT_DQ=""
OUTPUT_ROOT=""
PLAN_ONLY=0
NO_PROMPT=0
STOP_ON_FAILURE=1
ALLOW_HIGH_FREQUENCY_REAL_RUN=0
INCLUDE_F500_LOCAL=0
LAUNCH_REPEAT="1"
START_INDEX="1"
END_INDEX=""
ROUNDS_PER_LAUNCH="6"
GP_MODEL_DIR="/home/impl-user/dongfa/tt_dgp/outputs/gp_models_extracted_20260625_164901/gp_models"
GP_SCALE="1.0"
GP_CLIP="0.5"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_PATH=""

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_goal12_session_relative_final_matrix_20260706.sh [options]

Options:
  --mode core|expanded|stress
  --frequency VALUE[,VALUE...]   Frequencies. Defaults: core/expanded=50,100,200;
                                 stress=500 unless overridden.
  --anchor PATH                  Existing session_home.json anchor to load.
  --hist-db-path PATH            Required for real hist_db/triple_dynamic cases.
  --home-preflight-current-q CSV Current q1..q7 for read-only pre-START gate.
  --home-preflight-current-dq CSV
                                 Current dq1..dq7 for read-only pre-START gate.
  --output-root DIR              Output root for logs, CSVs, and manifest.
  --launch-repeat N              Independent launches per official case. Default: 1.
  --start-index N                First global matrix case number to run. Default: 1.
  --end-index N                  Last global matrix case number to run. Default:
                                 total matrix cases after launch-repeat.
  --rounds-per-launch N          Internal trajectory rounds per launch. Default: 6.
  --model-dir PATH               GP model directory. Default:
                                 /home/impl-user/dongfa/tt_dgp/outputs/gp_models_extracted_20260625_164901/gp_models.
                                 GP-on cases use online_update=true by default.
  --plan                         Print matrix/commands only; do not source ROS or launch.
  --no-prompt                    Do not require manual START confirmation.
  --stop-on-failure              Stop at first failed launch/check. Default: enabled.
  --allow-high-frequency-real-run
                                 Allow real runs above 200 Hz. Only stress mode
                                 may use this; core/expanded F500 remains refused.
  --include-f500-local           In stress mode, add optional F500 local D0 smoke.
  -h, --help                     Show this help.

Core matrix per F50/F100/F200:
  no-GP D0; local D0; cloud D0/D2/D5; combined D0/D2/D5;
  hist_db D2; triple_dynamic D0/D2/D5. Total: 12 per frequency, 36 cases.

Expanded matrix per F50/F100/F200:
  no-GP D0; local D0; cloud D0/D2/D5/D10; combined D0/D2/D5/D10;
  hist_db D0/D2; triple_dynamic D0/D2/D5/D10. Total: 16 per frequency, 48 cases.

Stress mode:
  F500 plan/stress only. Real F500 requires --allow-high-frequency-real-run and
  never runs the full matrix; default stress real set is F500 no-GP D0 smoke only.
USAGE
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while (($# > 0)); do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || die "--mode requires core, expanded, or stress"
      MODE="$2"
      shift
      ;;
    --frequency|--frequencies)
      [[ $# -ge 2 ]] || die "$1 requires a frequency list"
      FREQUENCIES="$2"
      FREQUENCY_OVERRIDE=1
      shift
      ;;
    --anchor)
      [[ $# -ge 2 ]] || die "--anchor requires a path"
      ANCHOR="$2"
      shift
      ;;
    --hist-db-path)
      [[ $# -ge 2 ]] || die "--hist-db-path requires a path"
      HIST_DB_PATH="$2"
      shift
      ;;
    --home-preflight-current-q)
      [[ $# -ge 2 ]] || die "--home-preflight-current-q requires 7 CSV values"
      HOME_PREFLIGHT_CURRENT_Q="$2"
      shift
      ;;
    --home-preflight-current-dq)
      [[ $# -ge 2 ]] || die "--home-preflight-current-dq requires 7 CSV values"
      HOME_PREFLIGHT_CURRENT_DQ="$2"
      shift
      ;;
    --output-root)
      [[ $# -ge 2 ]] || die "--output-root requires a directory"
      OUTPUT_ROOT="$2"
      shift
      ;;
    --launch-repeat)
      [[ $# -ge 2 ]] || die "--launch-repeat requires a positive integer"
      LAUNCH_REPEAT="$2"
      shift
      ;;
    --start-index)
      [[ $# -ge 2 ]] || die "--start-index requires a positive integer"
      START_INDEX="$2"
      shift
      ;;
    --end-index)
      [[ $# -ge 2 ]] || die "--end-index requires a positive integer"
      END_INDEX="$2"
      shift
      ;;
    --rounds-per-launch)
      [[ $# -ge 2 ]] || die "--rounds-per-launch requires a positive integer"
      ROUNDS_PER_LAUNCH="$2"
      shift
      ;;
    --model-dir)
      [[ $# -ge 2 ]] || die "--model-dir requires a path"
      GP_MODEL_DIR="$2"
      shift
      ;;
    --plan)
      PLAN_ONLY=1
      ;;
    --no-prompt)
      NO_PROMPT=1
      ;;
    --stop-on-failure)
      STOP_ON_FAILURE=1
      ;;
    --allow-high-frequency-real-run)
      ALLOW_HIGH_FREQUENCY_REAL_RUN=1
      ;;
    --include-f500-local)
      INCLUDE_F500_LOCAL=1
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

case "${MODE}" in
  core|expanded|stress)
    ;;
  *)
    die "--mode must be core, expanded, or stress"
    ;;
esac

if [[ "${MODE}" == "stress" && "${FREQUENCY_OVERRIDE}" == "0" ]]; then
  FREQUENCIES="500"
fi

[[ "${LAUNCH_REPEAT}" =~ ^[1-9][0-9]*$ ]] || die "--launch-repeat must be a positive integer"
[[ "${START_INDEX}" =~ ^[1-9][0-9]*$ ]] || die "--start-index must be a positive integer"
if [[ -n "${END_INDEX}" ]]; then
  [[ "${END_INDEX}" =~ ^[1-9][0-9]*$ ]] || die "--end-index must be a positive integer"
fi
[[ "${ROUNDS_PER_LAUNCH}" =~ ^[1-9][0-9]*$ ]] || die "--rounds-per-launch must be a positive integer"

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="outputs/session_relative_goal12_${MODE}_matrix_20260706"
fi
OUTPUT_ROOT="${OUTPUT_ROOT%/}"
MANIFEST_PATH="${OUTPUT_ROOT}/manifest.csv"

cd "${REPO_ROOT}"

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

to_upper_tag() {
  printf '%s' "$1" | tr '[:lower:]' '[:upper:]' | tr -c '[:alnum:]' '_'
}

freq_tag() {
  printf '%s' "$1" | tr -cd '[:alnum:]'
}

split_frequencies() {
  local raw="${FREQUENCIES//,/ }"
  local f
  for f in ${raw}; do
    [[ "${f}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid frequency: ${f}"
    printf '%s\n' "${f}"
  done
}

rate_gt_200() {
  awk -v v="$1" 'BEGIN { exit (v > 200) ? 0 : 1 }'
}

case_specs_for_mode() {
  case "${MODE}" in
    core)
      cat <<'SPECS'
no_gp:0
local:0
cloud:0
cloud:2
cloud:5
combined:0
combined:2
combined:5
hist_db:2
triple_dynamic:0
triple_dynamic:2
triple_dynamic:5
SPECS
      ;;
    expanded)
      cat <<'SPECS'
no_gp:0
local:0
cloud:0
cloud:2
cloud:5
cloud:10
combined:0
combined:2
combined:5
combined:10
hist_db:0
hist_db:2
triple_dynamic:0
triple_dynamic:2
triple_dynamic:5
triple_dynamic:10
SPECS
      ;;
    stress)
      echo "no_gp:0"
      if ((INCLUDE_F500_LOCAL)); then
        echo "local:0"
      fi
      ;;
  esac
}

source_needs_hist_db() {
  case "$1" in
    hist_db|triple_dynamic|triple_dynamic_gated)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

source_needs_gp_models() {
  [[ "$1" != "no_gp" ]]
}

source_uses_delay() {
  case "$1" in
    cloud|combined|triple_dynamic)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

source_display_tag() {
  case "$1" in
    no_gp) echo "NOGP" ;;
    hist_db) echo "HISTDB" ;;
    triple_dynamic) echo "TRIPLE_DYNAMIC" ;;
    *) to_upper_tag "$1" ;;
  esac
}

case_name_for() {
  local frequency="$1"
  local source="$2"
  local delay="$3"
  local launch_index="$4"
  printf 'GOAL12_%s_F%s_%s_D%s_L%s' \
    "$(to_upper_tag "${MODE}")" "$(freq_tag "${frequency}")" \
    "$(source_display_tag "${source}")" "${delay}" "${launch_index}"
}

gp_flags_for_source() {
  local source="$1"
  if [[ "${source}" == "no_gp" ]]; then
    GP_PREDICTION_ENABLED="false"
    GP_ONLINE_UPDATE_ENABLED="false"
    GP_COMPENSATION_ENABLED="false"
    EFFECTIVE_GP_SCALE="0.0"
    EFFECTIVE_SOURCE="local"
    GP_OFF="true"
  else
    GP_PREDICTION_ENABLED="true"
    GP_ONLINE_UPDATE_ENABLED="true"
    GP_COMPENSATION_ENABLED="true"
    EFFECTIVE_GP_SCALE="${GP_SCALE}"
    EFFECTIVE_SOURCE="${source}"
    GP_OFF="false"
  fi
}

build_launch_args() {
  local case_name="$1"
  local data_output_dir="$2"
  local frequency="$3"
  local source="$4"
  local delay="$5"
  local run_name="${case_name}_${RUN_STAMP}"

  gp_flags_for_source "${source}"

  LAUNCH_ARGS=(
    "run_name:=${run_name}"
    "data_output_dir:=${data_output_dir}"
    "trajectory_reference_mode:=session_relative"
    "session_home_mode:=load"
    "session_home_path:=${ANCHOR}"
    "session_home_capture_enabled:=false"
    "session_relative_capture_enabled:=false"
    "session_relative_anchor_delta_limit_mode:=warn"
    "transition_duration:=10"
    "rounds_per_mode:=${ROUNDS_PER_LAUNCH}"
    "control_frequency:=${frequency}"
    "trajectory_publish_rate:=${frequency}"
    "state_parameter_publish_rate:=${frequency}"
    "post_run_return_to_session_home_enabled:=true"
    "post_run_return_linear_speed:=0.005"
    "post_run_return_timeout_sec:=60.0"
    "post_run_return_hold_sec:=2.0"
    "normal_run_start_gate_enabled:=true"
    "normal_run_start_warn_m:=0.100"
    "normal_run_start_refuse_m:=0.150"
    "emergency_return_start_refuse_m:=0.300"
    "return_only_if_too_far_enabled:=false"
    "csv_output_profile:=full"
    "trajectory_mode:=goal1_spatial_multisine"
    "circle_frequency:=0.05"
    "torque_rate_limit_nm_per_s:=20.0"
    "startup_linear_speed:=0.005"
    "startup_distance_warn_m:=0.100"
    "startup_distance_refuse_m:=0.300"
    "startup_distance_refuse_enabled:=true"
    "gp_model_dir:=${GP_MODEL_DIR}"
    "gp_prediction_enabled:=${GP_PREDICTION_ENABLED}"
    "gp_online_update_enabled:=${GP_ONLINE_UPDATE_ENABLED}"
    "gp_compensation_enabled:=${GP_COMPENSATION_ENABLED}"
    "gp_compensation_source:=${EFFECTIVE_SOURCE}"
    "gp_compensation_scale:=${EFFECTIVE_GP_SCALE}"
    "gp_compensation_clip_nm:=${GP_CLIP}"
    "gp_compensation_disable_joint7:=true"
    "delay_steps:=${delay}"
    "gp_prediction_stride:=5"
    "future_trajectory_request_stride:=5"
    "gp_output_timeout_sec:=0.5"
  )

  if source_needs_hist_db "${source}"; then
    LAUNCH_ARGS+=(
      "gp_historical_db_enabled:=true"
      "gp_historical_db_path:=${HIST_DB_PATH}"
      "gp_historical_db_preflight_enabled:=true"
      "gp_historical_db_preflight_required:=true"
      "gp_disable_silent_hist_fallback:=true"
      "gp_historical_db_disable_when_online_update:=false"
      "gp_historical_db_k:=25"
      "gp_historical_db_q_scale:=0.1"
      "gp_historical_db_dq_scale:=0.1"
      "gp_historical_db_max_distance:=2.0"
      "gp_historical_db_require_distance_pass_for_active:=true"
      "gp_historical_db_distance_contribution_logging:=true"
      "gp_historical_db_metadata_enforcement_enabled:=true"
      "session_home_joint_check_enabled:=true"
      "session_home_joint_check_required_for_hist:=true"
      "gp_historical_db_preflight_mode:=single"
      "gp_historical_db_preflight_max_distance:=2.0"
      "gp_historical_db_preflight_p95_max_distance:=2.0"
    )
  else
    LAUNCH_ARGS+=(
      "gp_historical_db_enabled:=false"
      "gp_historical_db_preflight_enabled:=false"
      "gp_historical_db_preflight_required:=false"
      "gp_disable_silent_hist_fallback:=false"
    )
  fi
}

print_ros_command() {
  printf 'ros2 launch py_controllers cartesian_impedance_python_only_compensation_trajectory_launch.py'
  printf ' %q' "${LAUNCH_ARGS[@]}"
  printf '\n'
}

count_official_cases() {
  local freq_count spec_count
  freq_count="$(split_frequencies | wc -l | tr -d '[:space:]')"
  spec_count="$(case_specs_for_mode | wc -l | tr -d '[:space:]')"
  echo $((freq_count * spec_count))
}

resolve_index_window() {
  TOTAL_MATRIX_CASES=$(( $(count_official_cases) * LAUNCH_REPEAT ))
  if [[ -z "${END_INDEX}" ]]; then
    FINAL_INDEX="${TOTAL_MATRIX_CASES}"
  else
    FINAL_INDEX="${END_INDEX}"
  fi
  ((START_INDEX <= FINAL_INDEX)) || die "--start-index must be <= --end-index/final index"
  ((FINAL_INDEX <= TOTAL_MATRIX_CASES)) || die "--end-index must be <= total matrix cases (${TOTAL_MATRIX_CASES})"
  PLANNED_CASE_COUNT=$((FINAL_INDEX - START_INDEX + 1))
}

init_manifest() {
  mkdir -p "${OUTPUT_ROOT}"
  printf '%s\n' \
    'mode,global_case_index,frequency,source,delay_steps,launch_repeat_total,launch_repeat_index,rounds_per_launch,expected_csv_count,gp_off,gp_prediction_enabled,gp_online_update_enabled,gp_compensation_enabled,gp_compensation_source,gp_compensation_scale,gp_compensation_clip_nm,hist_db_required,hist_db_path,session_home_path,data_output_dir,launch_log,status' \
    > "${MANIFEST_PATH}"
}

append_manifest_row() {
  local global_case_index="$1"
  local frequency="$2"
  local source="$3"
  local delay="$4"
  local launch_index="$5"
  local data_output_dir="$6"
  local launch_log="$7"
  local status="$8"
  local hist_required="false"
  source_needs_hist_db "${source}" && hist_required="true"
  gp_flags_for_source "${source}"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${MODE}" "${global_case_index}" "${frequency}" "${source}" "${delay}" \
    "${LAUNCH_REPEAT}" "${launch_index}" "${ROUNDS_PER_LAUNCH}" \
    "${TOTAL_MATRIX_CASES}" "${GP_OFF}" \
    "${GP_PREDICTION_ENABLED}" "${GP_ONLINE_UPDATE_ENABLED}" \
    "${GP_COMPENSATION_ENABLED}" "${EFFECTIVE_SOURCE}" \
    "${EFFECTIVE_GP_SCALE}" "${GP_CLIP}" "${hist_required}" \
    "${HIST_DB_PATH}" "${ANCHOR}" "${data_output_dir}" "${launch_log}" \
    "${status}" >> "${MANIFEST_PATH}"
}

bad_log_patterns() {
  cat <<'PATTERNS'
joint_velocity_violation
motion aborted by reflex
communication violation
communication_constraints_violation
safe_abort
launch_failed
InvalidParameterTypeException
ModuleNotFoundError
Refusing to start
PATTERNS
}

check_log_absent_bad_signs() {
  local log_path="$1"
  local pattern
  while IFS= read -r pattern; do
    [[ -n "${pattern}" ]] || continue
    if grep -Fqi "${pattern}" "${log_path}"; then
      echo "Bad log sign found: ${pattern}"
      return 1
    fi
  done < <(bad_log_patterns)
}

require_log_text() {
  local log_path="$1"
  local text="$2"
  if ! grep -Fq "${text}" "${log_path}"; then
    echo "Required log text missing: ${text}"
    return 1
  fi
}

check_csv_count() {
  local data_output_dir="$1"
  local csvs=()
  mapfile -t csvs < <(
    find "${data_output_dir}" -maxdepth 1 -type f \
      \( -name '*_cartesian_impedance_controller_data.csv' \
      -o -name 'cartesian_impedance_controller_data.csv' \) \
      | sort
  )
  if [[ "${#csvs[@]}" -ne 1 ]]; then
    echo "Expected exactly one controller CSV in ${data_output_dir}, found ${#csvs[@]}"
    echo "Discovered controller CSV candidates:"
    printf '%s\n' "${csvs[@]}"
    return 1
  fi
}

check_hist_db_active_evidence() {
  local log_path="$1"
  require_log_text "${log_path}" "HIST_DB_PREFLIGHT_PASS"
  require_log_text "${log_path}" "HIST_DB_ACTIVE_OK"
  if grep -Eq "source='hist_db'.*fallback_used=1|source='hist_db'.*selected_source='nominal'" "${log_path}"; then
    echo "Hist DB case used fallback/nominal path; invalid matrix data"
    return 1
  fi
}

check_triple_dynamic_active_evidence() {
  local log_path="$1"
  require_log_text "${log_path}" "HIST_DB_PREFLIGHT_PASS"
  require_log_text "${log_path}" "TRIPLE_DYNAMIC_ACTIVE_OK"
  if grep -Eq "source='triple_dynamic'.*triple_used_fallback=1|source='triple_dynamic'.*triple_available=0|triple_dynamic fallback rejected" "${log_path}"; then
    echo "triple_dynamic case used fallback or was unavailable; invalid matrix data"
    return 1
  fi
}

validate_launch_output() {
  local log_path="$1"
  local data_output_dir="$2"
  local source="$3"
  check_log_absent_bad_signs "${log_path}"
  require_log_text "${log_path}" "Trajectory round 1/${ROUNDS_PER_LAUNCH} started"
  require_log_text "${log_path}" "Trajectory round ${ROUNDS_PER_LAUNCH}/${ROUNDS_PER_LAUNCH} started"
  require_log_text "${log_path}" "Return cleanup reached session home"
  require_log_text "${log_path}" "Published /post_run_return_complete=True"
  require_log_text "${log_path}" "process has finished cleanly"
  case "${source}" in
    hist_db)
      check_hist_db_active_evidence "${log_path}"
      ;;
    triple_dynamic)
      check_triple_dynamic_active_evidence "${log_path}"
      ;;
  esac
  check_csv_count "${data_output_dir}"
}

print_matrix_definition() {
  case "${MODE}" in
    core)
      echo "Core matrix: per frequency 12 cases"
      echo "  no-GP: D0 true GP-off"
      echo "  local: D0"
      echo "  cloud: D0 D2 D5"
      echo "  combined: D0 D2 D5"
      echo "  hist_db: D2"
      echo "  triple_dynamic: D0 D2 D5"
      ;;
    expanded)
      echo "Expanded matrix: per frequency 16 cases"
      echo "  no-GP: D0 true GP-off"
      echo "  local: D0"
      echo "  cloud: D0 D2 D5 D10"
      echo "  combined: D0 D2 D5 D10"
      echo "  hist_db: D0 D2"
      echo "  triple_dynamic: D0 D2 D5 D10"
      ;;
    stress)
      echo "Stress matrix: F500 plan/stress smoke only, no full matrix"
      echo "  no-GP: D0 true GP-off"
      if ((INCLUDE_F500_LOCAL)); then
        echo "  local: D0 optional smoke"
      fi
      ;;
  esac
}

print_delay_support() {
  echo "Delay support inspection:"
  echo "  cartesian_impedance.py logs: Cloud-like delay_steps delays only cloud-like prediction output, not local prediction."
  echo "  Affected sources: cloud, combined, triple_dynamic."
  echo "  Not delay-driven: local and pure hist_db; hist_db D cases are included only where the official matrix explicitly asks for them."
}

print_plan() {
  resolve_index_window

  echo "Goal12 session-relative final matrix plan"
  echo "mode=${MODE}"
  echo "frequencies=${FREQUENCIES}"
  echo "total_planned_cases=${TOTAL_MATRIX_CASES}"
  echo "start_index=${START_INDEX}"
  echo "final_index=${FINAL_INDEX}"
  echo "planned_case_count=${PLANNED_CASE_COUNT}"
  echo "launch_repeat=${LAUNCH_REPEAT}"
  echo "rounds_per_launch=${ROUNDS_PER_LAUNCH}"
  echo "expected_csv_count=${TOTAL_MATRIX_CASES}"
  echo "anchor=${ANCHOR}"
  echo "hist_db_path=${HIST_DB_PATH:-MISSING}"
  echo "hist_db_path_required_for=hist_db,triple_dynamic"
  echo "model_dir=${GP_MODEL_DIR}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "gp_on_flags=gp_prediction_enabled=true,gp_online_update_enabled=true,gp_compensation_enabled=true,gp_compensation_scale=${GP_SCALE},gp_compensation_clip_nm=${GP_CLIP},gp_compensation_disable_joint7=true"
  echo "true_gp_off_flags=gp_prediction_enabled=false,gp_online_update_enabled=false,gp_compensation_enabled=false,gp_compensation_scale=0.0"
  echo "session_relative=true"
  echo "session_home_mode=load"
  echo "hist_home_support_contract=joint_check+distance_pass+metadata_binding"
  echo "session_relative_anchor_delta_limit_mode=warn"
  echo "post_run_return_to_session_home_enabled=true"
  echo "stop_on_failure=${STOP_ON_FAILURE}"
  echo "no_prompt=${NO_PROMPT}"
  echo
  print_matrix_definition
  echo
  print_delay_support
  echo

  if [[ "${MODE}" == "stress" ]]; then
    echo "HIGH-FREQUENCY WARNING: stress mode is plan/stress smoke only; F500 real run is refused unless --allow-high-frequency-real-run is passed."
    echo
  fi

  local frequency spec source delay launch_index case_name data_output_dir
  local global_case_index=0
  while IFS= read -r frequency; do
    if rate_gt_200 "${frequency}"; then
      echo "Frequency F$(freq_tag "${frequency}") warning: >200 Hz; plan-only safe, real run requires stress mode and explicit allow flag."
    fi
    while IFS= read -r spec; do
      source="${spec%%:*}"
      delay="${spec#*:}"
      if source_needs_hist_db "${source}" && [[ -z "${HIST_DB_PATH}" ]]; then
        echo "Requirement: ${source} D${delay} needs --hist-db-path for real run."
      fi
      if ! source_uses_delay "${source}" && [[ "${delay}" != "0" ]]; then
        echo "Note: ${source} D${delay} is official matrix coverage, but delay_steps does not affect this source directly."
      fi
      for ((launch_index = 1; launch_index <= LAUNCH_REPEAT; launch_index++)); do
        global_case_index=$((global_case_index + 1))
        if ((global_case_index < START_INDEX || global_case_index > FINAL_INDEX)); then
          continue
        fi
        case_name="$(case_name_for "${frequency}" "${source}" "${delay}" "${launch_index}")"
        data_output_dir="${OUTPUT_ROOT}/${case_name}"
        build_launch_args "${case_name}" "${data_output_dir}" "${frequency}" "${source}" "${delay}"
        echo "Case ${global_case_index}/${TOTAL_MATRIX_CASES}: ${case_name}"
        echo "frequency=${frequency} source=${source} delay_steps=${delay} data_output_dir=${data_output_dir}"
        print_ros_command
        echo
      done
    done < <(case_specs_for_mode)
  done < <(split_frequencies)
}

confirm_start_or_die() {
  if ((NO_PROMPT)); then
    return 0
  fi
  local answer=""
  echo "Type START to run the first real matrix launch, or Ctrl+C to stop."
  if [[ ! -e /dev/tty ]]; then
    die "no controlling terminal; pass --no-prompt only when unattended real run is intended"
  fi
  if ! read -r -p "Type START to run the matrix: " answer < /dev/tty; then
    die "could not read START confirmation"
  fi
  [[ "${answer}" == "START" ]] || die "operator did not type START"
  NO_PROMPT=1
}

run_one_launch() {
  local frequency="$1"
  local source="$2"
  local delay="$3"
  local launch_index="$4"
  local global_case_index="$5"
  local case_name data_output_dir launch_log rc
  case_name="$(case_name_for "${frequency}" "${source}" "${delay}" "${launch_index}")"
  data_output_dir="${OUTPUT_ROOT}/${case_name}"
  launch_log="${data_output_dir}/launch.log"
  build_launch_args "${case_name}" "${data_output_dir}" "${frequency}" "${source}" "${delay}"

  echo "========================================================================"
  echo "Running ${case_name} (${global_case_index}/${TOTAL_MATRIX_CASES})"
  echo "frequency=${frequency} source=${source} delay_steps=${delay}"
  echo "data_output_dir=${data_output_dir}"
  echo "launch_log=${launch_log}"
  print_ros_command
  echo "========================================================================"

  mkdir -p "${data_output_dir}"
  append_manifest_row "${global_case_index}" "${frequency}" "${source}" "${delay}" "${launch_index}" "${data_output_dir}" "${launch_log}" "starting"

  set +e
  ros2 launch py_controllers cartesian_impedance_python_only_compensation_trajectory_launch.py "${LAUNCH_ARGS[@]}" 2>&1 | tee "${launch_log}"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ "${rc}" != "0" ]]; then
    append_manifest_row "${global_case_index}" "${frequency}" "${source}" "${delay}" "${launch_index}" "${data_output_dir}" "${launch_log}" "failed_exit_${rc}"
    echo "Launch exit code failed: ${rc}"
    return 1
  fi

  if ! validate_launch_output "${launch_log}" "${data_output_dir}" "${source}"; then
    append_manifest_row "${global_case_index}" "${frequency}" "${source}" "${delay}" "${launch_index}" "${data_output_dir}" "${launch_log}" "failed_postcheck"
    return 1
  fi

  append_manifest_row "${global_case_index}" "${frequency}" "${source}" "${delay}" "${launch_index}" "${data_output_dir}" "${launch_log}" "passed"
}

preflight_real_run_or_die() {
  [[ -f "${ANCHOR}" ]] || die "anchor does not exist: ${ANCHOR}"

  local frequency spec source needs_gp_models=0 needs_hist_db=0
  while IFS= read -r frequency; do
    if rate_gt_200 "${frequency}"; then
      if [[ "${MODE}" != "stress" ]]; then
        die "real frequency >200 Hz is refused outside --mode stress"
      fi
      ((ALLOW_HIGH_FREQUENCY_REAL_RUN)) || die "real F${frequency} requires --allow-high-frequency-real-run"
    fi
  done < <(split_frequencies)

  while IFS= read -r spec; do
    source="${spec%%:*}"
    if source_needs_gp_models "${source}"; then
      needs_gp_models=1
    fi
    if source_needs_hist_db "${source}"; then
      needs_hist_db=1
      [[ -n "${HIST_DB_PATH}" ]] || die "${source} real run requires --hist-db-path"
      [[ -f "${HIST_DB_PATH}" ]] || die "hist DB path does not exist: ${HIST_DB_PATH}"
      [[ -f "${HIST_DB_PATH%.npz}_metadata.json" ]] \
        || die "hist DB metadata sidecar does not exist: ${HIST_DB_PATH%.npz}_metadata.json"
    fi
  done < <(case_specs_for_mode)

  if ((needs_hist_db)); then
    [[ -n "${HOME_PREFLIGHT_CURRENT_Q}" ]] \
      || die "active hist matrix requires --home-preflight-current-q"
    [[ -n "${HOME_PREFLIGHT_CURRENT_DQ}" ]] \
      || die "active hist matrix requires --home-preflight-current-dq"
    echo "== Read-only session-home joint preflight before START =="
    python3 scripts/check_canonical_home_feasibility.py \
      --session-home "${ANCHOR}" \
      --current-q "${HOME_PREFLIGHT_CURRENT_Q}" \
      --current-dq "${HOME_PREFLIGHT_CURRENT_DQ}"
  fi

  if ((needs_gp_models)); then
    [[ -d "${GP_MODEL_DIR}" ]] || die "GP model dir does not exist: ${GP_MODEL_DIR}"
    [[ -f "${GP_MODEL_DIR}/joint1_local.pkl" ]] || die "missing GP local model: ${GP_MODEL_DIR}/joint1_local.pkl"
    [[ -f "${GP_MODEL_DIR}/joint1_cloud.pkl" ]] || die "missing GP cloud model: ${GP_MODEL_DIR}/joint1_cloud.pkl"
  fi
}

main() {
  echo "== Repository =="
  echo "pwd: $(pwd)"
  echo "branch: $(git branch --show-current)"
  echo "git status --short:"
  git status --short
  echo

  if ((PLAN_ONLY)); then
    print_plan
    return 0
  fi

  resolve_index_window
  preflight_real_run_or_die
  init_manifest
  confirm_start_or_die

  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export ROS_DOMAIN_ID=75
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH="${REPO_ROOT}/new_structure/gp:${PYTHONPATH:-}"
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export NUMEXPR_MAX_THREADS=1
  source_environment

  local frequency spec source delay launch_index global_case_index=0
  while IFS= read -r frequency; do
    while IFS= read -r spec; do
      source="${spec%%:*}"
      delay="${spec#*:}"
      for ((launch_index = 1; launch_index <= LAUNCH_REPEAT; launch_index++)); do
        global_case_index=$((global_case_index + 1))
        if ((global_case_index < START_INDEX || global_case_index > FINAL_INDEX)); then
          continue
        fi
        if ! run_one_launch "${frequency}" "${source}" "${delay}" "${launch_index}" "${global_case_index}"; then
          echo "Stopping after failed case ${global_case_index}: frequency=${frequency} source=${source} delay_steps=${delay} launch_index=${launch_index}"
          if ((STOP_ON_FAILURE)); then
            exit 1
          fi
        fi
      done
    done < <(case_specs_for_mode)
  done < <(split_frequencies)
}

main "$@"
