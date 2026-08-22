#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/zhao/miniconda3/envs/myenv/bin/python}"
GPU_CSV="${1:-0,1,2,3,4,5,6,7}"
LOG_DIR="${LOG_DIR:-logs}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"
NUM_RUNS="${NUM_RUNS:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
VALID_SIZE="${VALID_SIZE:-}"
TEST_SIZE="${TEST_SIZE:-}"
PREPROCESS_NUM_PROCESSES="${PREPROCESS_NUM_PROCESSES:-8}"
RESULTS_OUT="${RESULTS_OUT:-results_summary.csv}"
IFS=',' read -r -a GPUS <<< "${GPU_CSV}"

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPU ids supplied." >&2
  exit 2
fi
if ! [[ "${SLOTS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SLOTS_PER_GPU must be a positive integer." >&2
    exit 2
fi
if [[ -n "${NUM_RUNS}" ]] && ! [[ "${NUM_RUNS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_RUNS must be a positive integer." >&2
    exit 2
fi
if [[ -n "${NUM_WORKERS}" ]] && ! [[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]]; then
    echo "NUM_WORKERS must be a non-negative integer." >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
mapfile -t CONFIGS < <(printf '%s\n' configs/*_balanced_random.yaml configs/*_balanced_scaffold.yaml | sort)
mapfile -t CACHE_CONFIGS < <(printf '%s\n' configs/*_balanced_random.yaml | sort)

echo "Prebuilding four dataset caches..."
"${PYTHON_BIN}" scripts/prebuild_dataset_cache.py "${CACHE_CONFIGS[@]}" \
  --preprocess-num-processes "${PREPROCESS_NUM_PROCESSES}" \
  --num-workers "${NUM_WORKERS:-8}"

declare -a SLOT_PIDS=()
declare -a SLOT_GPUS=()

for ((replica = 0; replica < SLOTS_PER_GPU; replica++)); do
    for gpu in "${GPUS[@]}"; do
        SLOT_GPUS+=("${gpu}")
    done
done

for index in "${!CONFIGS[@]}"; do
  slot=$((index % ${#SLOT_GPUS[@]}))
  if [[ -n "${SLOT_PIDS[$slot]:-}" ]]; then
    wait "${SLOT_PIDS[$slot]}"
  fi

  config="${CONFIGS[$index]}"
  gpu="${SLOT_GPUS[$slot]}"
  name="$(basename "${config}" .yaml)"
  log_path="${LOG_DIR}/${name}.log"
  main_args=(main.py --config "${config}")
  if [[ -n "${NUM_RUNS}" ]]; then
    main_args+=(--num-runs "${NUM_RUNS}")
  fi
  if [[ -n "${NUM_WORKERS}" ]]; then
    main_args+=(--num-workers "${NUM_WORKERS}")
  fi
  if [[ -n "${VALID_SIZE}" ]]; then
    main_args+=(--valid-size "${VALID_SIZE}")
  fi
  if [[ -n "${TEST_SIZE}" ]]; then
    main_args+=(--test-size "${TEST_SIZE}")
  fi
  echo "Starting ${name} on physical GPU ${gpu}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${main_args[@]}" >"${log_path}" 2>&1 &
  pid=$!
  SLOT_PIDS[$slot]="${pid}"
done

for pid in "${SLOT_PIDS[@]}"; do
  wait "${pid}"
done

"${PYTHON_BIN}" scripts/collect_latest_results.py \
  --root experiment_records \
  --out "${RESULTS_OUT}"
echo "All runs completed."
echo "Run-level summary: ${PROJECT_ROOT}/${RESULTS_OUT}"
echo "Grouped summary: ${PROJECT_ROOT}/${RESULTS_OUT%.*}_grouped.${RESULTS_OUT##*.}"
