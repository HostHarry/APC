#!/usr/bin/env bash
# Wait for the main MC queue, then rerun vchd_prefix_cache MMMU under the
# current checkout so every leaderboard number shares one code state.
set -euo pipefail
RUN_ROOT=/root/autodl-tmp/LaViDa/eval/logs/mc_aligned_ours_20260727
while ! rg -q 'ALL_DONE_MC' "${RUN_ROOT}/status.log" 2>/dev/null; do sleep 120; done

CKPT=/autodl-fs/data/lavida-ckpts/lavida-llada-reason
export PATH="/root/miniconda3/envs/CrossMatch/bin:${PATH}"
export PYTHONPATH="/root/autodl-tmp/LaViDa:/root/autodl-tmp/LaViDa/eval"
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/autodl-fs/data/hf_home
export HUGGINGFACE_HUB_CACHE=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}
export LLADA_VISION_ENCODER=/autodl-fs/data/lavida-ckpts/siglip-so400m-patch14-384
export DEBUG_PRINT_IMAGE_RES=0
export PYTHONUNBUFFERED=1
source /root/autodl-tmp/LaViDa/eval/bridge_terminal_proxy.sh || true

printf 'START mmmu_dev_val_full vchd_prefix_cache(rerun) %s\n' "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
MODE=vchd_prefix_cache TASKS=mmmu_dev_val_full LIMIT=0 NUM_PROCESSES=2 \
MAX_NEW_TOKENS=128 BLOCK_LENGTH=64 STEP_PER_BLOCK=64 CCAW_MAX_MASK_CAPACITY=64 \
MAIN_PROCESS_PORT=26135 \
OUTPUT_PATH="${RUN_ROOT}/mmmu_dev_val_full/vchd_prefix_cache" \
  bash /root/autodl-tmp/LaViDa/eval/run_vchd_llada.sh "${CKPT}" 2>&1 \
  | tee "${RUN_ROOT}/mmmu_dev_val_full__vchd_prefix_cache.log"
printf 'ALL_DONE_MC_EXTRA %s\n' "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
