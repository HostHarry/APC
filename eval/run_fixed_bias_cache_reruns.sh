#!/usr/bin/env bash

set -euo pipefail

CKPT=${1:?checkpoint path required}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/LaViDa/eval/logs/lavida_llada_fixed_bias_cache_$(date +%Y%m%d_%H%M%S)"}

export PATH="/root/miniconda3/envs/CrossMatch/bin:${PATH}"
export PYTHONPATH="/root/autodl-tmp/LaViDa:/root/autodl-tmp/LaViDa/eval"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HOME="${HF_HOME:-/autodl-fs/data/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}}"
export LLADA_VISION_ENCODER="${LLADA_VISION_ENCODER:-/autodl-fs/data/lavida-ckpts/siglip-so400m-patch14-384}"
export DEBUG_PRINT_IMAGE_RES="${DEBUG_PRINT_IMAGE_RES:-0}"
# Keep model weights offline; datasets for LLaVABench use a local path in the task yaml.
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# The MMBench / LLaVABench GPT judges and HF dataset fetches go through the
# reverse SSH tunnel from bridge_terminal_proxy.sh.
if [[ -f "/root/autodl-tmp/LaViDa/eval/bridge_terminal_proxy.sh" ]]; then
  # shellcheck disable=SC1091
  source /root/autodl-tmp/LaViDa/eval/bridge_terminal_proxy.sh
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-REPLACE_WITH_YOUR_OPENAI_API_KEY}"
export OPENAI_API_URL="${OPENAI_API_URL:-https://api.openai.com/v1/chat/completions}"
export GPT_EVAL_MODEL_NAME="${GPT_EVAL_MODEL_NAME:-gpt-4o}"

mkdir -p "${RUN_ROOT}"
printf 'run_root=%s\nstarted_at=%s\n' "${RUN_ROOT}" "$(date --iso-8601=seconds)" | tee "${RUN_ROOT}/status.log"

run_job() {
  local label=$1
  local mode=$2
  local tasks=$3
  local max_new_tokens=$4
  local block_length=$5
  local step_per_block=$6
  local ccaw_capacity=$7
  local port=$8

  printf 'START %s %s\n' "${label}" "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
  MODE="${mode}" \
  TASKS="${tasks}" \
  LIMIT=0 \
  NUM_PROCESSES=2 \
  MAX_NEW_TOKENS="${max_new_tokens}" \
  BLOCK_LENGTH="${block_length}" \
  STEP_PER_BLOCK="${step_per_block}" \
  CCAW_MAX_MASK_CAPACITY="${ccaw_capacity}" \
  MAIN_PROCESS_PORT="${port}" \
  OUTPUT_PATH="${RUN_ROOT}/${label}/${mode}" \
    bash eval/run_vchd_llada.sh "${CKPT}" 2>&1 | tee "${RUN_ROOT}/${label}__${mode}.log"
  printf 'DONE %s %s\n' "${label}" "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
}

# 0) LaViDa LLaVABench first (open-ended; 60 samples; α=0.25 via run_vchd_llada.sh)
run_job llavabench vchd_prefix_cache llava_in_the_wild 512 64 64 64 26010
run_job llavabench vchd_ccaw_prefix_cache llava_in_the_wild 512 64 64 64 26011

run_job mmmu vchd_prefix_cache mmmu_dev_val_full 128 64 64 64 26012
run_job mmbench_4329 vchd_prefix_cache mmbench_en_dev 128 64 64 64 26013
run_job mmmu vchd_ccaw_prefix_cache mmmu_dev_val_full 128 64 64 64 26014
run_job mmbench_4329 vchd_ccaw_prefix_cache mmbench_en_dev 128 64 64 64 26015
run_job m3cot vchd_prefix_cache m3cot_full 512 64 64 64 26016
run_job m3cot vchd_ccaw_prefix_cache m3cot_full 512 64 64 64 26017
run_job vlind vchd_prefix_cache vlind_bench_full 16 16 16 64 26018
run_job vlind vchd_ccaw_prefix_cache vlind_bench_full 16 16 16 64 26019

printf 'ALL_DONE %s\n' "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
