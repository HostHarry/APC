#!/usr/bin/env bash
# Multiple-choice regression check under the aligned protocol (L/T/B = 128/128/64):
#   MMMU dev_val_full : original + vchd_ccaw_prefix_cache
#                       (vchd_prefix_cache reuses logs/lavida_llada_bidir_fix_20260727_060142)
#   MMBench en_dev    : original + vchd_prefix_cache + vchd_ccaw_prefix_cache
set -euo pipefail

CKPT=${1:-/autodl-fs/data/lavida-ckpts/lavida-llada-reason}
RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/LaViDa/eval/logs/mc_aligned_ours_20260727}

export PATH="/root/miniconda3/envs/CrossMatch/bin:${PATH}"
export PYTHONPATH="/root/autodl-tmp/LaViDa:/root/autodl-tmp/LaViDa/eval"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HOME="${HF_HOME:-/autodl-fs/data/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}}"
export LLADA_VISION_ENCODER="${LLADA_VISION_ENCODER:-/autodl-fs/data/lavida-ckpts/siglip-so400m-patch14-384}"
export DEBUG_PRINT_IMAGE_RES=0
export PYTHONUNBUFFERED=1

# Proxy + judge key: MMBench scoring uses GPT-based choice extraction.
if [[ -f /root/autodl-tmp/LaViDa/eval/bridge_terminal_proxy.sh ]]; then
  source /root/autodl-tmp/LaViDa/eval/bridge_terminal_proxy.sh
fi
eval "$(rg '^export (OPENAI_API_KEY|OPENAI_API_URL|GPT_EVAL_MODEL_NAME)=' /root/autodl-tmp/LaViDa/eval/run_fixed_bias_cache_reruns.sh)"

mkdir -p "${RUN_ROOT}"

run_one() {
  local task=$1
  local mode=$2
  local port=$3
  printf 'START %s %s %s\n' "${task}" "${mode}" "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
  MODE="${mode}" \
  TASKS="${task}" \
  LIMIT=0 \
  NUM_PROCESSES=2 \
  MAX_NEW_TOKENS=128 \
  BLOCK_LENGTH=64 \
  STEP_PER_BLOCK=64 \
  CCAW_MAX_MASK_CAPACITY=64 \
  MAIN_PROCESS_PORT="${port}" \
  OUTPUT_PATH="${RUN_ROOT}/${task}/${mode}" \
    bash /root/autodl-tmp/LaViDa/eval/run_vchd_llada.sh "${CKPT}" 2>&1 \
    | tee "${RUN_ROOT}/${task}__${mode}.log"
  printf 'DONE %s %s %s\n' "${task}" "${mode}" "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
}

run_one mmmu_dev_val_full original 26130
run_one mmmu_dev_val_full vchd_ccaw_prefix_cache 26131
run_one mmbench_en_dev original 26132
run_one mmbench_en_dev vchd_prefix_cache 26133
run_one mmbench_en_dev vchd_ccaw_prefix_cache 26134

printf 'ALL_DONE_MC %s\n' "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
