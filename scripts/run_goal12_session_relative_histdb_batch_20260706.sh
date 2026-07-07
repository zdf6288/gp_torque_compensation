#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="db5"
LAUNCH_REPEAT=""
START_INDEX="1"
END_INDEX=""
ROUNDS_PER_LAUNCH="6"
FREQUENCY="50"
ANCHOR="outputs/session_relative_floating_anchor_scale0_smoke_20260706_144203/session_home.json"
OUTPUT_ROOT=""
PLAN_ONLY=0
NO_PROMPT=1
STOP_ON_FAILURE=1

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_PATH=""

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_goal12_session_relative_histdb_batch_20260706.sh [options]

Options:
  --mode db5|db10           Historical DB batch size. Default: db5.
  --launch-repeat N         Independent ros2 launch count. Defaults: db5=5, db10=10.
  --start-index N           First independent launch index to run. Default: 1.
                            Example: --mode db5 --start-index 2 runs R2-R5.
  --end-index N             Last independent launch index to run. Default:
                            launch-repeat total for the selected mode.
  --rounds-per-launch N     Internal trajectory rounds per launch. Default: 6.
  --frequency VALUE         control_frequency, trajectory_publish_rate, and
                            state_parameter_publish_rate. Default: 50.
  --anchor PATH             Existing session_home.json anchor to load.
  --output-root DIR         Output root. Defaults:
                            db5  -> outputs/session_relative_histdb5_true_gpoff_raw_20260706
                            db10 -> outputs/session_relative_histdb10_true_gpoff_raw_20260706
  --plan                    Print commands only; do not source ROS or launch.
  --no-prompt               Do not require manual confirmation. Default: enabled.
  --stop-on-failure         Stop at first failed launch/check. Default: enabled.
  -h, --help                Show this help.

DB semantics:
  DB5  = 5 independent launches x 6 rounds per launch = 5 CSVs = 30 rounds.
  DB10 = 10 independent launches x 6 rounds per launch = 10 CSVs = 60 rounds.
  This script uses true GP-off, not GP-enabled scale=0.0.
USAGE
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while (($# > 0)); do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || die "--mode requires db5 or db10"
      MODE="$2"
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
    --frequency)
      [[ $# -ge 2 ]] || die "--frequency requires a positive number"
      FREQUENCY="$2"
      shift
      ;;
    --anchor)
      [[ $# -ge 2 ]] || die "--anchor requires a path"
      ANCHOR="$2"
      shift
      ;;
    --output-root)
      [[ $# -ge 2 ]] || die "--output-root requires a directory"
      OUTPUT_ROOT="$2"
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
  db5)
    DEFAULT_LAUNCH_REPEAT=5
    DEFAULT_OUTPUT_ROOT="outputs/session_relative_histdb5_true_gpoff_raw_20260706"
    MODE_TAG="DB5"
    ;;
  db10)
    DEFAULT_LAUNCH_REPEAT=10
    DEFAULT_OUTPUT_ROOT="outputs/session_relative_histdb10_true_gpoff_raw_20260706"
    MODE_TAG="DB10"
    ;;
  *)
    die "--mode must be db5 or db10"
    ;;
esac

[[ "${ROUNDS_PER_LAUNCH}" =~ ^[1-9][0-9]*$ ]] || die "--rounds-per-launch must be a positive integer"
[[ "${FREQUENCY}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--frequency must be a positive number"
[[ "${START_INDEX}" =~ ^[1-9][0-9]*$ ]] || die "--start-index must be a positive integer"

if [[ -z "${LAUNCH_REPEAT}" ]]; then
  LAUNCH_REPEAT="${DEFAULT_LAUNCH_REPEAT}"
fi
[[ "${LAUNCH_REPEAT}" =~ ^[1-9][0-9]*$ ]] || die "--launch-repeat must be a positive integer"
if [[ -z "${END_INDEX}" ]]; then
  END_INDEX="${LAUNCH_REPEAT}"
fi
[[ "${END_INDEX}" =~ ^[1-9][0-9]*$ ]] || die "--end-index must be a positive integer"
((START_INDEX <= END_INDEX)) || die "--start-index must be <= --end-index"
((END_INDEX <= LAUNCH_REPEAT)) || die "--end-index must be <= launch-repeat total (${LAUNCH_REPEAT})"
PLANNED_LAUNCH_COUNT=$((END_INDEX - START_INDEX + 1))

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}"
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

mode_lower() {
  printf '%s' "${MODE_TAG}" | tr '[:upper:]' '[:lower:]'
}

freq_tag() {
  printf '%s' "$1" | tr -cd '[:alnum:]'
}

case_name_for_index() {
  local index="$1"
  printf 'HISTDB_%s_F%s_GPOFF_ROUND%s_R%s' \
    "${MODE_TAG}" "$(freq_tag "${FREQUENCY}")" "${ROUNDS_PER_LAUNCH}" "${index}"
}

build_launch_args() {
  local case_name="$1"
  local data_output_dir="$2"
  local run_name="${case_name}_${RUN_STAMP}"
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
    "control_frequency:=${FREQUENCY}"
    "trajectory_publish_rate:=${FREQUENCY}"
    "state_parameter_publish_rate:=${FREQUENCY}"
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
    "gp_prediction_enabled:=false"
    "gp_online_update_enabled:=false"
    "gp_compensation_enabled:=false"
    "gp_compensation_source:=local"
    "gp_compensation_scale:=0.0"
    "gp_compensation_clip_nm:=0.5"
    "gp_compensation_disable_joint7:=true"
    "delay_steps:=0"
    "gp_prediction_stride:=5"
    "future_trajectory_request_stride:=5"
    "gp_output_timeout_sec:=0.5"
  )
}

print_ros_command() {
  printf 'ros2 launch py_controllers cartesian_impedance_python_only_compensation_trajectory_launch.py'
  printf ' %q' "${LAUNCH_ARGS[@]}"
  printf '\n'
}

init_manifest() {
  mkdir -p "${OUTPUT_ROOT}"
  printf '%s\n' \
    'mode,launch_repeat_total,launch_repeat_index,rounds_per_launch,expected_csv_count,frequency,gp_off,gp_prediction_enabled,gp_online_update_enabled,gp_compensation_enabled,gp_compensation_scale,session_home_path,data_output_dir,launch_log,status' \
    > "${MANIFEST_PATH}"
}

append_manifest_row() {
  local index="$1"
  local data_output_dir="$2"
  local launch_log="$3"
  local status="$4"
  printf '%s,%s,%s,%s,%s,%s,true,false,false,false,0.0,%s,%s,%s,%s\n' \
    "${MODE}" "${LAUNCH_REPEAT}" "${index}" "${ROUNDS_PER_LAUNCH}" \
    "${LAUNCH_REPEAT}" "${FREQUENCY}" "${ANCHOR}" "${data_output_dir}" \
    "${launch_log}" "${status}" >> "${MANIFEST_PATH}"
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

validate_launch_output() {
  local log_path="$1"
  local data_output_dir="$2"

  check_log_absent_bad_signs "${log_path}"
  require_log_text "${log_path}" "Data recording enabled -> gp_prediction_enabled=False, gp_compensation_enabled=False"
  require_log_text "${log_path}" "Trajectory round 1/${ROUNDS_PER_LAUNCH} started"
  require_log_text "${log_path}" "Trajectory round ${ROUNDS_PER_LAUNCH}/${ROUNDS_PER_LAUNCH} started"
  require_log_text "${log_path}" "Return cleanup reached session home"
  require_log_text "${log_path}" "Published /post_run_return_complete=True"
  require_log_text "${log_path}" "process has finished cleanly"
  check_csv_count "${data_output_dir}"
}

print_plan() {
  echo "Goal12 session-relative hist DB batch plan"
  echo "mode=${MODE}"
  echo "launch_repeat=${LAUNCH_REPEAT}"
  echo "start_index=${START_INDEX}"
  echo "final_index=${END_INDEX}"
  echo "planned_launch_count=${PLANNED_LAUNCH_COUNT}"
  echo "rounds_per_launch=${ROUNDS_PER_LAUNCH}"
  echo "expected_csv_count=${LAUNCH_REPEAT}"
  echo "planned_csv_count=${PLANNED_LAUNCH_COUNT}"
  echo "total_expected_rounds=$((LAUNCH_REPEAT * ROUNDS_PER_LAUNCH))"
  echo "planned_rounds=$((PLANNED_LAUNCH_COUNT * ROUNDS_PER_LAUNCH))"
  echo "frequency=${FREQUENCY}"
  echo "anchor=${ANCHOR}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "true_gp_off=true"
  echo "gp_prediction_enabled=false"
  echo "gp_online_update_enabled=false"
  echo "gp_compensation_enabled=false"
  echo "gp_compensation_scale=0.0"
  echo "no_manual_pause_between_clean_launches=${NO_PROMPT}"
  echo "stop_on_failure=${STOP_ON_FAILURE}"
  echo

  local index case_name data_output_dir
  for ((index = START_INDEX; index <= END_INDEX; index++)); do
    case_name="$(case_name_for_index "${index}")"
    data_output_dir="${OUTPUT_ROOT}/${case_name}"
    build_launch_args "${case_name}" "${data_output_dir}"
    echo "Case ${index}/${LAUNCH_REPEAT}: ${case_name}"
    echo "data_output_dir=${data_output_dir}"
    print_ros_command
    echo
  done
}

run_one_launch() {
  local index="$1"
  local case_name data_output_dir launch_log rc
  case_name="$(case_name_for_index "${index}")"
  data_output_dir="${OUTPUT_ROOT}/${case_name}"
  launch_log="${data_output_dir}/launch.log"
  build_launch_args "${case_name}" "${data_output_dir}"

  echo "========================================================================"
  echo "Running ${case_name} (${index}/${LAUNCH_REPEAT})"
  echo "data_output_dir=${data_output_dir}"
  echo "launch_log=${launch_log}"
  print_ros_command
  echo "========================================================================"

  mkdir -p "${data_output_dir}"
  append_manifest_row "${index}" "${data_output_dir}" "${launch_log}" "starting"

  set +e
  ros2 launch py_controllers cartesian_impedance_python_only_compensation_trajectory_launch.py "${LAUNCH_ARGS[@]}" 2>&1 | tee "${launch_log}"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ "${rc}" != "0" ]]; then
    append_manifest_row "${index}" "${data_output_dir}" "${launch_log}" "failed_exit_${rc}"
    echo "Launch exit code failed: ${rc}"
    return 1
  fi

  if ! validate_launch_output "${launch_log}" "${data_output_dir}"; then
    append_manifest_row "${index}" "${data_output_dir}" "${launch_log}" "failed_postcheck"
    return 1
  fi

  append_manifest_row "${index}" "${data_output_dir}" "${launch_log}" "passed"
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

  [[ -f "${ANCHOR}" ]] || die "anchor does not exist: ${ANCHOR}"
  init_manifest
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

  local index
  for ((index = START_INDEX; index <= END_INDEX; index++)); do
    if ! run_one_launch "${index}"; then
      echo "Stopping after failed launch index ${index}."
      if ((STOP_ON_FAILURE)); then
        exit 1
      fi
    fi
  done
}

main "$@"
