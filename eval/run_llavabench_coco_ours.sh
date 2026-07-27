#!/usr/bin/env bash
# Our-tree LLaVA-Bench COCO runs under the exact repro protocol:
#   llava_bench_coco, 90 samples, L/T/B = 256/128/256, judge deferred
# Modes: original (cross-tree sanity) + vchd_prefix_cache + vchd_ccaw_prefix_cache.
set -euo pipefail

CKPT=${1:-/autodl-fs/data/lavida-ckpts/lavida-llada-reason}
RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/LaViDa/eval/logs/llavabench_coco_ours_20260727}

export PATH="/root/miniconda3/envs/CrossMatch/bin:${PATH}"
export PYTHONPATH="/root/autodl-tmp/LaViDa:/root/autodl-tmp/LaViDa/eval"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HOME="${HF_HOME:-/autodl-fs/data/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}}"
export LLADA_VISION_ENCODER="${LLADA_VISION_ENCODER:-/autodl-fs/data/lavida-ckpts/siglip-so400m-patch14-384}"
export DEBUG_PRINT_IMAGE_RES=0
export LLAVA_JUDGE_SKIP=1
export PYTHONUNBUFFERED=1

if [[ -f /root/autodl-tmp/LaViDa/eval/bridge_terminal_proxy.sh ]]; then
  source /root/autodl-tmp/LaViDa/eval/bridge_terminal_proxy.sh
fi

mkdir -p "${RUN_ROOT}"

run_one() {
  local mode=$1
  local port=$2
  printf 'START llava_bench_coco %s %s\n' "${mode}" "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
  MODE="${mode}" \
  TASKS=llava_bench_coco \
  LIMIT=0 \
  NUM_PROCESSES=2 \
  MAX_NEW_TOKENS=256 \
  BLOCK_LENGTH=256 \
  STEP_PER_BLOCK=128 \
  CCAW_MAX_MASK_CAPACITY=64 \
  MAIN_PROCESS_PORT="${port}" \
  OUTPUT_PATH="${RUN_ROOT}/${mode}" \
    bash /root/autodl-tmp/LaViDa/eval/run_vchd_llada.sh "${CKPT}" 2>&1 \
    | tee "${RUN_ROOT}/llava_bench_coco__${mode}.log"
  printf 'DONE llava_bench_coco %s %s\n' "${mode}" "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
}

run_one original 26120
run_one vchd_prefix_cache 26121
run_one vchd_ccaw_prefix_cache 26122

printf 'ALL_DONE_OURS %s\n' "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
