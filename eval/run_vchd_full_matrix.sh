#!/usr/bin/env bash
# Full LaViDa-LLaDA matrix: MMMU+MMBench / M3CoT / VLind × original|vchd|vchd_ccaw
# with MMaDA-aligned schedules:
#   MMMU/MMBench  128/128/64
#   M3CoT         512/256/64
#   VLind         128/128/64
set -euo pipefail

CKPT=${1:-/autodl-fs/data/lavida-ckpts/lavida-llada-reason}
MODES=${MODES:-original,vchd,vchd_ccaw}
# NOTE: do not name this GROUPS — bash reserves GROUPS as a readonly array.
BENCH_GROUPS=${BENCH_GROUPS:-mmmu_mmbench,m3cot,vlind}
NUM_PROCESSES=${NUM_PROCESSES:-2}
RUN_ID=${RUN_ID:-lavida_llada_aligned_$(date +%Y%m%d_%H%M%S)}

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
echo "${RUN_ID}" > "${ROOT}/eval/logs/LATEST_LAVIDA_FULL_RUN"
echo "${RUN_ID}" > "${ROOT}/eval/logs/LATEST_LAVIDA_LOCAL_RUN"

status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_ROOT}/status.log"
}

# group -> task max_tokens steps block  (steps = total; step_per_block = steps / (tokens/block))
group_config() {
  case "$1" in
    mmmu_mmbench)
      printf '%s %s %s %s\n' 'mmmu_dev_val_full,mmbench_en_full' 128 128 64
      ;;
    m3cot)
      printf '%s %s %s %s\n' m3cot_full 512 256 64
      ;;
    vlind)
      printf '%s %s %s %s\n' vlind_bench_full 128 128 64
      ;;
    *)
      echo "Unknown group: $1" >&2
      return 2
      ;;
  esac
}

port=25721
IFS=',' read -r -a group_array <<<"${BENCH_GROUPS}"
IFS=',' read -r -a mode_array <<<"${MODES}"

status "START run_id=${RUN_ID} ckpt=${CKPT} groups=${BENCH_GROUPS} modes=${MODES}"
status "schedules: mmmu_mmbench=128/128/64 m3cot=512/256/64 vlind=128/128/64"

for group in "${group_array[@]}"; do
  group="$(echo "${group}" | xargs)"
  read -r task max_tokens total_steps block_length <<<"$(group_config "${group}")"
  num_blocks=$((max_tokens / block_length))
  step_per_block=$((total_steps / num_blocks))
  if (( step_per_block * num_blocks != total_steps )); then
    status "ERROR non-divisible schedule group=${group} L=${max_tokens} T=${total_steps} B=${block_length}"
    exit 3
  fi

  for mode in "${mode_array[@]}"; do
    mode="$(echo "${mode}" | xargs)"
    out_dir="${LOG_ROOT}/${group}/${mode}"
    log_file="${LOG_ROOT}/${group}__${mode}.log"
    mkdir -p "${out_dir}"
    status "START group=${group} task=${task} mode=${mode} L/T/B=${max_tokens}/${total_steps}/${block_length} step_per_block=${step_per_block}"

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

    status "DONE group=${group} mode=${mode} rc=${rc}"
    if [[ ${rc} -ne 0 ]]; then
      status "FAILED group=${group} mode=${mode}; see ${log_file}"
      exit "${rc}"
    fi
    port=$((port + 1))
  done
done

status "ALL COMPLETE run_id=${RUN_ID}"
