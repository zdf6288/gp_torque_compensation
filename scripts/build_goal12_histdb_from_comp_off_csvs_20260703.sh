#!/usr/bin/env bash
set -euo pipefail

# Offline helper: build a persistent historical residual DB from compensation-off CSVs.
# 默认不使用 compensation-on 数据，避免把 active GP feedback 混进 Y_residual。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PLAN_ONLY=0
OUTPUT_DIR="outputs/goal12_histdb_20260703"
USER_INPUTS=()

usage() {
  cat <<'USAGE'
Usage:
  scripts/build_goal12_histdb_from_comp_off_csvs_20260703.sh [options]

Options:
  --plan              Print selected inputs and commands without building.
  --output-dir DIR    Output directory for goal1_historical_residual_db.npz.
  --input tag=path    Explicit compensation-off CSV input. Can be repeated.
  -h, --help          Show this help.

Default input discovery searches compensation-off / predict-only / no-GP outputs:
  outputs/manual_predict_only
  outputs/manual_nogp_smoke
  outputs paths containing predict or nogp

Default discovery excludes:
  outputs/manual_compensation
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
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a value"
      OUTPUT_DIR="$2"
      shift
      ;;
    --input)
      [[ $# -ge 2 ]] || die "--input requires tag=path"
      USER_INPUTS+=("$2")
      shift
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

cd "${REPO_ROOT}"

sanitize_tag() {
  printf '%s' "$1" | tr -cs '[:alnum:]_' '_' | sed 's/^_//; s/_$//'
}

validate_input_item() {
  local item="$1"
  [[ "${item}" == *=* ]] || die "--input must be tag=path, got: ${item}"
  local tag="${item%%=*}"
  local path="${item#*=}"
  [[ -n "${tag}" && -n "${path}" ]] || die "invalid --input: ${item}"
  [[ -f "${path}" ]] || die "input CSV does not exist: ${path}"
  case "${path}" in
    *outputs/manual_compensation*)
      die "refusing compensation output as hist DB input: ${path}"
      ;;
  esac
}

discover_inputs() {
  local found=()
  local path

  for root in outputs/manual_predict_only outputs/manual_nogp_smoke; do
    if [[ -d "${root}" ]]; then
      while IFS= read -r path; do
        found+=("${path}")
      done < <(find "${root}" -type f -name 'cartesian_impedance_controller_data.csv' | sort)
    fi
  done

  if [[ -d outputs ]]; then
    while IFS= read -r path; do
      case "${path}" in
        *outputs/manual_compensation*)
          continue
          ;;
        *predict*|*PREDICT*|*nogp*|*NOGP*|*no_gp*|*NO_GP*)
          found+=("${path}")
          ;;
      esac
    done < <(find outputs -type f -name 'cartesian_impedance_controller_data.csv' | sort)
  fi

  printf '%s\n' "${found[@]}" | awk 'NF && !seen[$0]++'
}

INPUT_ITEMS=()
if ((${#USER_INPUTS[@]} > 0)); then
  for item in "${USER_INPUTS[@]}"; do
    validate_input_item "${item}"
    INPUT_ITEMS+=("${item}")
  done
else
  while IFS= read -r path; do
    [[ -n "${path}" ]] || continue
    parent_tag="$(sanitize_tag "$(basename "$(dirname "${path}")")")"
    [[ -n "${parent_tag}" ]] || parent_tag="run"
    INPUT_ITEMS+=("${parent_tag}=${path}")
  done < <(discover_inputs)
fi

if ((${#INPUT_ITEMS[@]} == 0)); then
  echo "No compensation-off / predict-only / no-GP CSV inputs were found."
  echo "Pass explicit inputs with --input tag=/path/to/cartesian_impedance_controller_data.csv."
  if ((PLAN_ONLY)); then
    exit 0
  fi
  exit 1
fi

echo "Selected hist DB inputs:"
for item in "${INPUT_ITEMS[@]}"; do
  echo "  ${item}"
done
echo

BUILD_CMD=(python3 scripts/build_goal1_historical_residual_db.py --output-dir "${OUTPUT_DIR}")
for item in "${INPUT_ITEMS[@]}"; do
  BUILD_CMD+=(--input "${item}")
done

DB_PATH="${OUTPUT_DIR%/}/goal1_historical_residual_db.npz"
CHECK_CMD=(python3 scripts/check_goal1_historical_residual_db.py --db "${DB_PATH}")

printf 'Build command:'
printf ' %q' "${BUILD_CMD[@]}"
printf '\n'
printf 'Check command:'
printf ' %q' "${CHECK_CMD[@]}"
printf '\n'

if ((PLAN_ONLY)); then
  echo "Plan only; DB was not built."
  echo "Expected DB path: ${DB_PATH}"
  exit 0
fi

"${BUILD_CMD[@]}"
"${CHECK_CMD[@]}"

echo "Historical residual DB ready: ${DB_PATH}"
