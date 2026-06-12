#!/usr/bin/env bash
set -euo pipefail

INCLUDE_CLIP05_SHORT=0
INCLUDE_F100_SMOKE=0
INCLUDE_F100_CLOUD_COMBINED=0

usage() {
  cat <<'USAGE'
Usage:
  ./run_goal12_final_matrix_50hz_100hz_after_commfix_20260612.sh [options]

Options:
  --include-clip05-short        Add G12_F50_LOCAL_ONLINE_S1_C05_SHORT.
  --include-f100-smoke          Add F100 smoke cases.
  --include-f100-cloud-combined Add extra F100 cloud/combined smoke cases.
  -h, --help                    Show this help.

Default runs only the F50 main matrix. Every case waits for manual confirmation.
USAGE
}

while (($# > 0)); do
  case "$1" in
    --include-clip05-short)
      INCLUDE_CLIP05_SHORT=1
      ;;
    --include-f100-smoke)
      INCLUDE_F100_SMOKE=1
      ;;
    --include-f100-cloud-combined)
      INCLUDE_F100_CLOUD_COMBINED=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

require_repo_root() {
  local missing=0
  for path in \
    "new_structure/py_controllers/launch/cartesian_impedance_launch.py" \
    "new_structure/py_controllers/py_controllers/cartesian_impedance.py"
  do
    if [[ ! -f "$path" ]]; then
      echo "ERROR: missing required repo file: $path" >&2
      missing=1
    fi
  done

  if ((missing)); then
    echo "ERROR: please cd to the gp_torque_compensation repo root before running this script." >&2
    exit 1
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

confirm_if_dirty() {
  local status
  status="$(git status --short)"
  if [[ -n "$status" ]]; then
    echo "WARNING: working tree is dirty. This script will not clean or restore files."
    echo "$status"
    read -r -p "Continue with dirty working tree? Type YES to continue: " answer
    if [[ "$answer" != "YES" ]]; then
      echo "Aborted by user."
      exit 1
    fi
  fi
}

validate_options() {
  if ((INCLUDE_F100_CLOUD_COMBINED && ! INCLUDE_F100_SMOKE)); then
    echo "ERROR: --include-f100-cloud-combined requires --include-f100-smoke." >&2
    exit 2
  fi
}

source_environment() {
  echo "== Environment =="
  # shellcheck source=/opt/ros/humble/setup.bash
  source /opt/ros/humble/setup.bash
  # shellcheck source=/dev/null
  source install/setup.bash

  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH="/home/mirmi_ros2_2/dongfa/tt_dgp/new_structure/gp:${PYTHONPATH:-}"
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export NUMEXPR_MAX_THREADS=1

  echo "ROS_DISTRO=${ROS_DISTRO:-}"
  echo "PYTHONPATH=$PYTHONPATH"
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
  echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
  echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
  echo "NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
  echo "NUMEXPR_MAX_THREADS=$NUMEXPR_MAX_THREADS"
  echo
}

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="outputs/goal12_final_matrix_after_commfix_20260612/${RUN_STAMP}"

COMMON_ARGS=(
  "robot_ip:=172.16.0.4"
  "load_gripper:=false"
  "use_fake_hardware:=false"
  "trajectory_mode:=goal1_spatial_multisine"
  "circle_frequency:=0.05"
  "transition_duration:=10.0"
  "anchor_trajectory_start_to_current_pose:=true"
  "trajectory_start_distance_warn_m:=0.003"
  "trajectory_start_distance_refuse_m:=0.02"
  "gp_model_dir:=/home/mirmi_ros2_2/dongfa/tt_dgp/new_structure/gp/gp_models"
  "csv_output_profile:=final"
  "gp_compensation_disable_joint7:=true"
  "gp_output_timeout_sec:=0.5"
  "timing_logging_enabled:=false"
  "torque_rate_limit_enabled:=true"
  "torque_rate_limit_nm_per_s:=40.0"
  "torque_rate_limit_log_first_n:=5"
  "allow_high_ros2_control_rate:=true"
  "ros2_control_update_rate:=1000"
  "spawn_cpp_relayer:=true"
  "spawn_update_rate_diagnostic:=false"
)

F50_ARGS=(
  "control_frequency:=50"
  "trajectory_publish_rate:=50"
  "state_parameter_publish_rate:=50"
  "gp_prediction_stride:=5"
  "future_trajectory_request_stride:=5"
)

F100_ARGS=(
  "control_frequency:=100"
  "trajectory_publish_rate:=100"
  "state_parameter_publish_rate:=100"
  "gp_prediction_stride:=10"
  "future_trajectory_request_stride:=10"
)

print_args() {
  local -n args_ref=$1
  for arg in "${args_ref[@]}"; do
    printf '  %s\n' "$arg"
  done
}

run_case() {
  local case_name="$1"
  local case_note="$2"
  local rate_tuple_name="$3"
  shift 3
  local case_args=("$@")
  local run_name="${case_name}_${RUN_STAMP}"
  local data_output_dir="${OUTPUT_ROOT}/${case_name}"
  local rate_args=()

  case "$rate_tuple_name" in
    F50)
      rate_args=("${F50_ARGS[@]}")
      ;;
    F100)
      rate_args=("${F100_ARGS[@]}")
      ;;
    *)
      echo "ERROR: unknown rate tuple: $rate_tuple_name" >&2
      exit 2
      ;;
  esac

  local launch_args=(
    "${COMMON_ARGS[@]}"
    "${rate_args[@]}"
    "run_name:=${run_name}"
    "data_output_dir:=${data_output_dir}"
    "${case_args[@]}"
  )

  echo "========================================================================"
  echo "Case: $case_name"
  echo "Note: $case_note"
  echo "run_name: $run_name"
  echo "data_output_dir: $data_output_dir"
  echo "Launch args:"
  print_args launch_args
  echo "========================================================================"
  echo "Before starting, confirm Franka web interface, FCI activation, joint unlock,"
  echo "robot area, and emergency-stop readiness are all checked."
  read -r -p "Press Enter to START this case, or Ctrl+C to stop."

  mkdir -p "$data_output_dir"
  ros2 launch py_controllers cartesian_impedance_launch.py "${launch_args[@]}"

  echo
  echo "Case exited: $case_name"
  echo "Please inspect robot state, terminal logs, CSV/data directory, and any"
  echo "communication_constraints_violation / controller_torque_discontinuity /"
  echo "joint_velocity_violation messages before continuing."
  read -r -p "Press Enter to continue to the next case, or Ctrl+C to stop."
}

add_f50_main_cases() {
  run_case \
    "G12_F50_NOGP_BASE_A" \
    "F50 main table: no-GP baseline at the beginning." \
    "F50" \
    "gp_prediction_enabled:=false" \
    "gp_online_update_enabled:=false" \
    "gp_compensation_enabled:=false" \
    "delay_steps:=0"

  run_case \
    "G12_F50_PRED_FROZEN_LOCALCLOUD" \
    "F50 diagnostic only: frozen prediction path; not a main compensation result." \
    "F50" \
    "gp_prediction_enabled:=true" \
    "gp_online_update_enabled:=false" \
    "gp_compensation_enabled:=false" \
    "delay_steps:=0"

  run_case \
    "G12_F50_PRED_ONLINE_ONLY" \
    "F50 diagnostic: online update enabled, prediction only, compensation off." \
    "F50" \
    "gp_prediction_enabled:=true" \
    "gp_online_update_enabled:=true" \
    "gp_compensation_enabled:=false" \
    "delay_steps:=0"

  run_case \
    "G12_F50_LOCAL_ONLINE_S1_C02" \
    "F50 main table: local online GP compensation, scale 1.0, clip 0.2." \
    "F50" \
    "gp_prediction_enabled:=true" \
    "gp_online_update_enabled:=true" \
    "gp_compensation_enabled:=true" \
    "gp_compensation_source:=local" \
    "gp_compensation_scale:=1.0" \
    "gp_compensation_clip_nm:=0.2" \
    "delay_steps:=0"

  run_case \
    "G12_F50_CLOUD_ONLINE_S1_C02_D2" \
    "F50 main table: cloud-like online GP compensation; D2 is about 40 ms at 50 Hz." \
    "F50" \
    "gp_prediction_enabled:=true" \
    "gp_online_update_enabled:=true" \
    "gp_compensation_enabled:=true" \
    "gp_compensation_source:=cloud" \
    "gp_compensation_scale:=1.0" \
    "gp_compensation_clip_nm:=0.2" \
    "delay_steps:=2"

  run_case \
    "G12_F50_COMBINED_ONLINE_S1_C02_D2" \
    "F50 main table: combined online GP compensation; D2 is about 40 ms at 50 Hz." \
    "F50" \
    "gp_prediction_enabled:=true" \
    "gp_online_update_enabled:=true" \
    "gp_compensation_enabled:=true" \
    "gp_compensation_source:=combined" \
    "gp_compensation_scale:=1.0" \
    "gp_compensation_clip_nm:=0.2" \
    "delay_steps:=2"

  run_case \
    "G12_F50_NOGP_BASE_B" \
    "F50 main table: no-GP baseline at the end for drift check." \
    "F50" \
    "gp_prediction_enabled:=false" \
    "gp_online_update_enabled:=false" \
    "gp_compensation_enabled:=false" \
    "delay_steps:=0"
}

add_clip05_short_case() {
  run_case \
    "G12_F50_LOCAL_ONLINE_S1_C05_SHORT" \
    "OPTIONAL short sanity only, not main table: local online GP compensation, clip 0.5." \
    "F50" \
    "gp_prediction_enabled:=true" \
    "gp_online_update_enabled:=true" \
    "gp_compensation_enabled:=true" \
    "gp_compensation_source:=local" \
    "gp_compensation_scale:=1.0" \
    "gp_compensation_clip_nm:=0.5" \
    "delay_steps:=0"
}

add_f100_smoke_cases() {
  run_case \
    "G12_F100_NOGP_SMOKE" \
    "OPTIONAL F100 smoke only: true F100 tuple, no GP compensation." \
    "F100" \
    "gp_prediction_enabled:=false" \
    "gp_online_update_enabled:=false" \
    "gp_compensation_enabled:=false" \
    "delay_steps:=0"

  run_case \
    "G12_F100_PRED_ONLINE_ONLY_SMOKE" \
    "OPTIONAL F100 smoke only: true F100 tuple, online prediction only." \
    "F100" \
    "gp_prediction_enabled:=true" \
    "gp_online_update_enabled:=true" \
    "gp_compensation_enabled:=false" \
    "delay_steps:=0"

  run_case \
    "G12_F100_LOCAL_ONLINE_S1_C02_SHORT" \
    "OPTIONAL F100 short smoke only: local online GP compensation, clip 0.2." \
    "F100" \
    "gp_prediction_enabled:=true" \
    "gp_online_update_enabled:=true" \
    "gp_compensation_enabled:=true" \
    "gp_compensation_source:=local" \
    "gp_compensation_scale:=1.0" \
    "gp_compensation_clip_nm:=0.2" \
    "delay_steps:=0"
}

add_f100_cloud_combined_cases() {
  run_case \
    "G12_F100_CLOUD_ONLINE_S1_C02_D4_SHORT" \
    "EXTRA F100 smoke only: cloud online GP compensation; D4 is about 40 ms at 100 Hz." \
    "F100" \
    "gp_prediction_enabled:=true" \
    "gp_online_update_enabled:=true" \
    "gp_compensation_enabled:=true" \
    "gp_compensation_source:=cloud" \
    "gp_compensation_scale:=1.0" \
    "gp_compensation_clip_nm:=0.2" \
    "delay_steps:=4"

  run_case \
    "G12_F100_COMBINED_ONLINE_S1_C02_D4_SHORT" \
    "EXTRA F100 smoke only: combined online GP compensation; D4 is about 40 ms at 100 Hz." \
    "F100" \
    "gp_prediction_enabled:=true" \
    "gp_online_update_enabled:=true" \
    "gp_compensation_enabled:=true" \
    "gp_compensation_source:=combined" \
    "gp_compensation_scale:=1.0" \
    "gp_compensation_clip_nm:=0.2" \
    "delay_steps:=4"
}

main() {
  require_repo_root
  validate_options
  print_repo_status
  confirm_if_dirty
  source_environment

  echo "Output root: $OUTPUT_ROOT"
  echo "Default matrix: F50 main table only."
  echo "include_clip05_short=$INCLUDE_CLIP05_SHORT"
  echo "include_f100_smoke=$INCLUDE_F100_SMOKE"
  echo "include_f100_cloud_combined=$INCLUDE_F100_CLOUD_COMBINED"
  echo

  add_f50_main_cases

  if ((INCLUDE_CLIP05_SHORT)); then
    add_clip05_short_case
  fi

  if ((INCLUDE_F100_SMOKE)); then
    add_f100_smoke_cases
  fi

  if ((INCLUDE_F100_CLOUD_COMBINED)); then
    add_f100_cloud_combined_cases
  fi

  echo "All selected GOAL12 final matrix cases have exited."
}

main "$@"
