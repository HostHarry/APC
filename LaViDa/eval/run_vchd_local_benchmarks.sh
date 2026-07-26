#!/usr/bin/env bash
# Run local M3CoT and VLind-Bench adapters with the three decoding modes.
set -euo pipefail

CKPT=${1:-/autodl-fs/data/lavida-ckpts/lavida-llada-reason}
BENCHMARKS=${BENCHMARKS:-m3cot,vlind}
MODES=${MODES:-original,vchd,vchd_ccaw}
NUM_PROCESSES=${NUM_PROCESSES:-2}
RUN_ID=${RUN_ID:-lavida_local_benchmarks_$(date +%Y%m%d_%H%M%S)}

export PATH=/root/miniconda3/envs/CrossMatch/bin:${PATH}
export PYTHONPATH="/root/autodl-tmp/LaViDa:/root/autodl-tmp/LaViDa/eval:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/autodl-fs/data/hf_home}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/autodl-fs/data/hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/autodl-fs/data/hf_home}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export LLADA_VISION_ENCODER=${LLADA_VISION_ENCODER:-/autodl-fs/data/lavida-ckpts/siglip-so400m-patch14-384}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export DEBUG_PRINT_IMAGE_RES=${DEBUG_PRINT_IMAGE_RES:-0}
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_ROOT="${ROOT}/eval/logs/${RUN_ID}"
mkdir -p "${LOG_ROOT}"
echo "${RUN_ID}" > "${ROOT}/eval/logs/LATEST_LAVIDA_LOCAL_RUN"

status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_ROOT}/status.log"
}

# Prints: task max_tokens total_steps block_length
# Defaults aligned to MMaDA schedules: M3CoT 512/256/64, VLind 128/128/64
benchmark_config() {
  case "$1" in
    m3cot)
      printf '%s %s %s %s\n' m3cot_full 512 256 64
      ;;
    vlind)
      printf '%s %s %s %s\n' vlind_bench_full 128 128 64
      ;;
    *)
      echo "Unknown benchmark: $1" >&2
      return 2
      ;;
  esac
}

port=25621
IFS=',' read -r -a benchmark_array <<<"${BENCHMARKS}"
IFS=',' read -r -a mode_array <<<"${MODES}"

status "START run_id=${RUN_ID} ckpt=${CKPT} benchmarks=${BENCHMARKS} modes=${MODES}"
for benchmark in "${benchmark_array[@]}"; do
  benchmark="$(echo "${benchmark}" | xargs)"
  read -r task max_tokens total_steps block_length <<<"$(benchmark_config "${benchmark}")"
  num_blocks=$((max_tokens / block_length))
  step_per_block=$((total_steps / num_blocks))

  for mode in "${mode_array[@]}"; do
    mode="$(echo "${mode}" | xargs)"
    out_dir="${LOG_ROOT}/${benchmark}/${mode}"
    log_file="${LOG_ROOT}/${benchmark}__${mode}.log"
    mkdir -p "$(dirname "${out_dir}")"
    status "START benchmark=${benchmark} task=${task} mode=${mode} L/T/B=${max_tokens}/${total_steps}/${block_length} step_per_block=${step_per_block}"

    set +e
    MODE="${mode}" \
    TASKS="${task}" \
    LIMIT=0 \
    NUM_PROCESSES="${NUM_PROCESSES}" \
    MAX_NEW_TOKENS="${max_tokens}" \
    BLOCK_LENGTH="${block_length}" \
    STEP_PER_BLOCK="${step_per_block}" \
    CCAW_MAX_MASK_CAPACITY="${block_length}" \
    MAIN_PROCESS_PORT="${port}" \
    OUTPUT_PATH="${out_dir}" \
      bash "${ROOT}/eval/run_vchd_llada.sh" "${CKPT}" \
      >>"${log_file}" 2>&1
    rc=$?
    set -e

    if [[ ${rc} -eq 0 ]] && rg -q \
      "Error during evaluation:|Traceback \\(most recent call last\\)|FAILED mode" \
      "${log_file}"; then
      rc=1
    fi
    if [[ ${rc} -eq 0 ]] && ! compgen -G "${out_dir}/*/*results*.json" >/dev/null; then
      rc=1
    fi

    status "DONE benchmark=${benchmark} mode=${mode} rc=${rc}"
    if [[ ${rc} -ne 0 ]]; then
      status "FAILED benchmark=${benchmark} mode=${mode}; see ${log_file}"
      exit "${rc}"
    fi
    port=$((port + 1))
  done
done

status "ALL COMPLETE run_id=${RUN_ID}"
