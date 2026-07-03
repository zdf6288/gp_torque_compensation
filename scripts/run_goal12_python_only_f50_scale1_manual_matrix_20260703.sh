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
REPEAT_COUNT=1
SOURCE_FILTER="all"
GP_SCALE="1.0"
GP_CLIP="0.5"
MODEL_DIR="${REPO_ROOT}/outputs/gp_models_extracted_20260625_164901/gp_models"
OUTPUT_ROOT="outputs/manual_compensation"
NO_CLEANUP=1

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_goal12_python_only_f50_scale1_manual_matrix_20260703.sh [options]

Options:
  --plan                 Print the planned Terminal-C-only runs and exit.
  --repeat N             Repeat each selected source N times. Default: 1.
  --source VALUE         local, cloud, combined, or all. Default: all.
  --scale VALUE          GP compensation scale. Default: 1.0.
  --clip VALUE           GP compensation clip in Nm. Default: 0.5.
  --model-dir PATH       GP model directory.
  --output-root DIR      Output root. Default: outputs/manual_compensation.
  --no-cleanup           Keep cpp_relayer active between runs. This is the default.
  -h, --help             Show this help.

Before running:
  Terminal A must already own external IMPL bringup/controller_manager.
  Terminal B must already have cpp_relayer active.
USAGE
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while (($# > 0)); do
  case "$1" in
    --plan)
      PLAN_ONLY=1
      ;;
    --repeat)
      [[ $# -ge 2 ]] || die "--repeat requires a value"
      REPEAT_COUNT="$2"
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

[[ "${REPEAT_COUNT}" =~ ^[1-9][0-9]*$ ]] || die "--repeat must be a positive integer"

case "${SOURCE_FILTER}" in
  local|cloud|combined|all)
    ;;
  *)
    die "--source must be local, cloud, combined, or all"
    ;;
esac

cd "${REPO_ROOT}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT%/}/${RUN_STAMP}"
MANIFEST_PATH="${OUTPUT_ROOT}/manifest.csv"

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
    1|1.0|1.00) printf '1' ;;
    0.5|.5|0.50|.50) printf '05' ;;
    0.25|.25|0.250|.250) printf '025' ;;
    *) printf '%s' "$1" | tr -cd '[:alnum:]' ;;
  esac
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
  printf 'case_name,run_name,data_output_dir,source,scale,clip,j7_disabled,control_frequency,trajectory_publish_rate,state_parameter_publish_rate,status\n' > "${MANIFEST_PATH}"
}

append_manifest_row() {
  local case_name="$1"
  local run_name="$2"
  local data_output_dir="$3"
  local source="$4"
  local status="$5"
  printf '%s,%s,%s,%s,%s,%s,true,50,50,50,%s\n' \
    "${case_name}" "${run_name}" "${data_output_dir}" "${source}" \
    "${GP_SCALE}" "${GP_CLIP}" "${status}" >> "${MANIFEST_PATH}"
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
  if ! printf '%s\n' "${output}" | awk '$1 == "cpp_relayer" && $NF == "active" {found=1} END {exit found ? 0 : 1}'; then
    die "cpp_relayer is not active. Run Terminal B load/activate sequence before continuing."
  fi
}

run_one_case() {
  local source="$1"
  local repeat_index="$2"
  local source_tag
  local case_name
  local run_name
  local data_output_dir
  source_tag="$(source_upper "${source}")"
  case_name="COMP_${source_tag}_F50_SCALE$(scale_tag "${GP_SCALE}")_CLIP$(scale_tag "${GP_CLIP}")_J7OFF_1000HZ_BRIDGE_R${repeat_index}"
  run_name="${case_name}_${RUN_STAMP}"
  data_output_dir="${OUTPUT_ROOT}/${case_name}"

  local launch_args=(
    "run_name:=${run_name}"
    "data_output_dir:=${data_output_dir}"
    "csv_output_profile:=full"
    "trajectory_mode:=goal1_spatial_multisine"
    "circle_frequency:=0.05"
    "control_frequency:=50"
    "trajectory_publish_rate:=50"
    "state_parameter_publish_rate:=50"
    "transition_duration:=10.0"
    "torque_rate_limit_nm_per_s:=20.0"
    "gp_model_dir:=${MODEL_DIR}"
    "gp_compensation_source:=${source}"
    "gp_compensation_scale:=${GP_SCALE}"
    "gp_compensation_clip_nm:=${GP_CLIP}"
    "gp_compensation_disable_joint7:=true"
    "delay_steps:=0"
    "gp_prediction_stride:=5"
    "future_trajectory_request_stride:=5"
    "gp_output_timeout_sec:=0.5"
  )

  echo "========================================================================"
  echo "Terminal-C-only case: ${case_name}"
  echo "run_name: ${run_name}"
  echo "data_output_dir: ${data_output_dir}"
  echo "source: ${source}, scale: ${GP_SCALE}, clip: ${GP_CLIP}, J7OFF: true"
  echo "This will start only the Python-only compensation trajectory launch."
  echo "========================================================================"

  if ((PLAN_ONLY)); then
    printf 'ros2 launch py_controllers cartesian_impedance_python_only_compensation_trajectory_launch.py'
    printf ' %q' "${launch_args[@]}"
    printf '\n'
    return 0
  fi

  check_cpp_relayer_active
  read -r -p "Type START to run this Terminal-C case, or Ctrl+C to stop: " answer
  if [[ "${answer}" != "START" ]]; then
    die "operator did not type START"
  fi

  mkdir -p "${data_output_dir}"
  append_manifest_row "${case_name}" "${run_name}" "${data_output_dir}" "${source}" "starting"
  if ros2 launch py_controllers cartesian_impedance_python_only_compensation_trajectory_launch.py "${launch_args[@]}"; then
    append_manifest_row "${case_name}" "${run_name}" "${data_output_dir}" "${source}" "launch_exited_zero"
  else
    append_manifest_row "${case_name}" "${run_name}" "${data_output_dir}" "${source}" "launch_failed"
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

main() {
  print_repo_status
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
  echo "Cleanup between successful runs: disabled"
  echo

  local source
  local repeat_index
  while IFS= read -r source; do
    for ((repeat_index = 1; repeat_index <= REPEAT_COUNT; repeat_index++)); do
      run_one_case "${source}" "${repeat_index}"
    done
  done < <(selected_sources)
}

main "$@"
