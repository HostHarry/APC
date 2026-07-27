#!/usr/bin/env bash
# Parallel launcher: one VCHD mode per GPU concurrently, four waves total.
# Each process pins itself to a single physical device via CUDA_VISIBLE_DEVICES
# and uses NUM_PROCESSES=1 (no DDP). Total wall time ~= max(GPU0, GPU1).
#
# Layout (chosen to balance the two GPU queues around the M3CoT ccaw
# bottleneck):
#
#   GPU 0                                    GPU 1
#   ---------------------------------------  ---------------------------------
#   M3CoT vchd_ccaw_prefix_cache   (~9h39)   M3CoT vchd_prefix_cache   (~6h26)
#   MMBench vchd_prefix_cache      (~2h24)   MMBench vchd_ccaw_prefix_cache (~3h36)
#   MMMU  vchd_prefix_cache        (~0h35)   MMMU  vchd_ccaw_prefix_cache   (~0h52)
#   VLind vchd_prefix_cache        (~0h26)   VLind vchd_ccaw_prefix_cache   (~0h40)
#                                = ~13h04                                    = ~11h34
#
# Longest queue drives the wall time.

set -euo pipefail

CKPT=${1:?checkpoint path required}
RUN_ROOT=${RUN_ROOT:-"/root/autodl-tmp/LaViDa/eval/logs/lavida_llada_bidir_fix_par_$(date +%Y%m%d_%H%M%S)"}

export PATH="/root/miniconda3/envs/CrossMatch/bin:${PATH}"
export PYTHONPATH="/root/autodl-tmp/LaViDa:/root/autodl-tmp/LaViDa/eval"
export HF_HOME="${HF_HOME:-/autodl-fs/data/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}}"
export LLADA_VISION_ENCODER="${LLADA_VISION_ENCODER:-/autodl-fs/data/lavida-ckpts/siglip-so400m-patch14-384}"
export DEBUG_PRINT_IMAGE_RES="${DEBUG_PRINT_IMAGE_RES:-0}"

if [[ -f "/root/autodl-tmp/LaViDa/eval/bridge_terminal_proxy.sh" ]]; then
  # shellcheck disable=SC1091
  source /root/autodl-tmp/LaViDa/eval/bridge_terminal_proxy.sh
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-REPLACE_WITH_YOUR_OPENAI_API_KEY}"
export OPENAI_API_URL="${OPENAI_API_URL:-https://api.openai.com/v1/chat/completions}"
export GPT_EVAL_MODEL_NAME="${GPT_EVAL_MODEL_NAME:-gpt-4o}"

mkdir -p "${RUN_ROOT}"
printf 'run_root=%s\nstarted_at=%s\nlayout=parallel-1-per-gpu\n' \
  "${RUN_ROOT}" "$(date --iso-8601=seconds)" | tee "${RUN_ROOT}/status.log"

# Run a single job pinned to one physical GPU.
run_job_on_gpu() {
  local gpu=$1
  local label=$2
  local mode=$3
  local tasks=$4
  local max_new_tokens=$5
  local block_length=$6
  local step_per_block=$7
  local ccaw_capacity=$8
  local port=$9

  printf '[gpu%s] START %s %s\n' "${gpu}" "${label}__${mode}" "$(date --iso-8601=seconds)" \
    | tee -a "${RUN_ROOT}/status.log"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  MODE="${mode}" \
  TASKS="${tasks}" \
  LIMIT=0 \
  NUM_PROCESSES=1 \
  MAX_NEW_TOKENS="${max_new_tokens}" \
  BLOCK_LENGTH="${block_length}" \
  STEP_PER_BLOCK="${step_per_block}" \
  CCAW_MAX_MASK_CAPACITY="${ccaw_capacity}" \
  MAIN_PROCESS_PORT="${port}" \
  OUTPUT_PATH="${RUN_ROOT}/${label}/${mode}" \
    bash eval/run_vchd_llada.sh "${CKPT}" 2>&1 \
    | tee "${RUN_ROOT}/${label}__${mode}.log"

  printf '[gpu%s] DONE  %s %s\n' "${gpu}" "${label}__${mode}" "$(date --iso-8601=seconds)" \
    | tee -a "${RUN_ROOT}/status.log"
}

# Serial queue on a single GPU, runs jobs one-by-one in the order given.
gpu_queue() {
  local gpu=$1; shift
  local queue_name=$1; shift
  printf '[gpu%s] QUEUE %s START %s\n' "${gpu}" "${queue_name}" "$(date --iso-8601=seconds)" \
    | tee -a "${RUN_ROOT}/status.log"
  while (( "$#" )); do
    # Each argument is a space-separated tuple, e.g. "mmmu vchd_prefix_cache mmmu_dev_val_full 128 64 64 64 27011"
    local args=($1); shift
    run_job_on_gpu "${gpu}" "${args[@]}"
  done
  printf '[gpu%s] QUEUE %s DONE  %s\n' "${gpu}" "${queue_name}" "$(date --iso-8601=seconds)" \
    | tee -a "${RUN_ROOT}/status.log"
}

# GPU 0 queue (heaviest job first for early progress signal).
gpu_queue 0 gpu0_queue \
  "m3cot        vchd_ccaw_prefix_cache m3cot_full         512 64 64 64 27101" \
  "mmbench_4329 vchd_prefix_cache      mmbench_en_dev     128 64 64 64 27102" \
  "mmmu         vchd_prefix_cache      mmmu_dev_val_full  128 64 64 64 27103" \
  "vlind        vchd_prefix_cache      vlind_bench_full    16 16 16 64 27104" \
  &
QUEUE0_PID=$!

# GPU 1 queue.
gpu_queue 1 gpu1_queue \
  "m3cot        vchd_prefix_cache      m3cot_full         512 64 64 64 27201" \
  "mmbench_4329 vchd_ccaw_prefix_cache mmbench_en_dev     128 64 64 64 27202" \
  "mmmu         vchd_ccaw_prefix_cache mmmu_dev_val_full  128 64 64 64 27203" \
  "vlind        vchd_ccaw_prefix_cache vlind_bench_full    16 16 16 64 27204" \
  &
QUEUE1_PID=$!

printf 'gpu0_queue_pid=%s gpu1_queue_pid=%s\n' "${QUEUE0_PID}" "${QUEUE1_PID}" \
  | tee -a "${RUN_ROOT}/status.log"

wait "${QUEUE0_PID}"
wait "${QUEUE1_PID}"

printf 'ALL_DONE %s\n' "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
