#!/usr/bin/env bash
# 3-way LaViDa comparison on MMMU dev (150 samples, standard split).
# Original vs VCHD prefix_cache vs VCHD_CCAW prefix_cache.
# Expected ~5 min per config × 3 = ~15 min on DDP-2.

set -euo pipefail

CKPT=${1:-/autodl-fs/data/lavida-ckpts/lavida-llada-reason}
RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/LaViDa/eval/logs/lavida_llada_bidir_fix_alpha025_20260727_084747}

export PATH="/root/miniconda3/envs/CrossMatch/bin:${PATH}"
export PYTHONPATH="/root/autodl-tmp/LaViDa:/root/autodl-tmp/LaViDa/eval"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HOME="${HF_HOME:-/autodl-fs/data/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}}"
export LLADA_VISION_ENCODER="${LLADA_VISION_ENCODER:-/autodl-fs/data/lavida-ckpts/siglip-so400m-patch14-384}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DEBUG_PRINT_IMAGE_RES="${DEBUG_PRINT_IMAGE_RES:-0}"

if [[ -f /root/autodl-tmp/LaViDa/eval/bridge_terminal_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /root/autodl-tmp/LaViDa/eval/bridge_terminal_proxy.sh
fi

# MMMU is self-scored (multiple-choice / exact match), no GPT judge needed,
# but we still export the key so the launcher can be reused later.
eval "$(rg '^export (OPENAI_API_KEY|OPENAI_API_URL|GPT_EVAL_MODEL_NAME)=' /root/autodl-tmp/LaViDa/eval/run_fixed_bias_cache_reruns.sh)"

# One config: mode, label, port.
run_one() {
  local mode=$1
  local port=$2
  local label="${mode}"

  printf 'START mmmu_dev %s %s\n' "${label}" "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"

  MODE="${mode}" \
  TASKS=mmmu_dev \
  LIMIT=0 \
  NUM_PROCESSES=2 \
  MAX_NEW_TOKENS=128 \
  BLOCK_LENGTH=64 \
  STEP_PER_BLOCK=64 \
  CCAW_MAX_MASK_CAPACITY=64 \
  MAIN_PROCESS_PORT="${port}" \
  OUTPUT_PATH="${RUN_ROOT}/mmmu_dev/${label}" \
    bash /root/autodl-tmp/LaViDa/eval/run_vchd_llada.sh "${CKPT}" 2>&1 \
    | tee "${RUN_ROOT}/mmmu_dev__${label}.log"

  printf 'DONE mmmu_dev %s %s\n' "${label}" "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
}

run_one original 26040
run_one vchd_prefix_cache 26041
run_one vchd_ccaw_prefix_cache 26042

printf 'ALL_DONE_MMMU_DEV %s\n' "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
