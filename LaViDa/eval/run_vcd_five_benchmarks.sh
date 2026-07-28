#!/usr/bin/env bash
# LaViDa pure VCD (Leng et al.) on five aligned benchmarks.
# Modes: vcd_prefix_cache (paper α=1.0, β=0.1, noise_step=500)
set -euo pipefail

CKPT=${1:-/root/autodl-tmp/LaViDa/lavida-ckpts/lavida-llada-hd-reason}
MODES=${MODES:-vcd_prefix_cache}
BENCHMARKS=${BENCHMARKS:-llava_bench,mmmu,mmbench,vlind,m3cot}
NUM_PROCESSES=${NUM_PROCESSES:-2}
RUN_ID=${RUN_ID:-lavida_vcd_five_$(date +%Y%m%d_%H%M%S)}

export PATH=/root/miniconda3/bin:${PATH}
export PYTHONPATH="/root/autodl-tmp/LaViDa:/root/autodl-tmp/LaViDa/eval:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/autodl-fs/data/hf_home}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export LLADA_VISION_ENCODER=${LLADA_VISION_ENCODER:-/autodl-fs/data/lavida-ckpts/siglip-so400m-patch14-384}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export DEBUG_PRINT_IMAGE_RES=0
export PYTHONUNBUFFERED=1
export http_proxy=${http_proxy:-http://127.0.0.1:7897}
export https_proxy=${https_proxy:-http://127.0.0.1:7897}
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE HF_ENDPOINT || true

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_ROOT="${ROOT}/eval/logs/${RUN_ID}"
mkdir -p "${LOG_ROOT}"
echo "${RUN_ID}" > "${ROOT}/eval/logs/LATEST_LAVIDA_VCD_RUN"

status() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_ROOT}/status.log"; }

bench_config() {
  case "$1" in
    llava_bench) printf '%s %s %s %s\n' llava_bench_coco 256 128 256 ;;
    mmmu)        printf '%s %s %s %s\n' mmmu_dev_val_full 128 128 64 ;;
    mmbench)     printf '%s %s %s %s\n' mmbench_en_dev 128 128 64 ;;
    vlind)       printf '%s %s %s %s\n' vlind_bench_full 128 128 64 ;;
    m3cot)       printf '%s %s %s %s\n' m3cot_full 512 256 64 ;;
    *) echo "Unknown benchmark: $1" >&2; return 2 ;;
  esac
}

port=26421
status "START run_id=${RUN_ID} ckpt=${CKPT} benches=${BENCHMARKS} modes=${MODES}"
IFS=',' read -r -a benches <<<"${BENCHMARKS}"
IFS=',' read -r -a modes <<<"${MODES}"

for bench in "${benches[@]}"; do
  bench="$(echo "${bench}" | xargs)"
  read -r task max_tokens total_steps block_length <<<"$(bench_config "${bench}")"
  num_blocks=$((max_tokens / block_length))
  step_per_block=$((total_steps / num_blocks))
  for mode in "${modes[@]}"; do
    mode="$(echo "${mode}" | xargs)"
    out_dir="${LOG_ROOT}/${bench}/${mode}"
    log_file="${LOG_ROOT}/${bench}__${mode}.log"
    mkdir -p "${out_dir}"
    status "START bench=${bench} task=${task} mode=${mode} L/T/B=${max_tokens}/${total_steps}/${block_length}"
    set +e
    MODE="${mode}" TASKS="${task}" LIMIT=0 \
    NUM_PROCESSES="${NUM_PROCESSES}" \
    MAX_NEW_TOKENS="${max_tokens}" BLOCK_LENGTH="${block_length}" \
    STEP_PER_BLOCK="${step_per_block}" \
    MAIN_PROCESS_PORT="${port}" OUTPUT_PATH="${out_dir}" \
      bash "${ROOT}/eval/run_vchd_llada.sh" "${CKPT}" >>"${log_file}" 2>&1
    rc=$?
    set -e
    if [[ ${rc} -eq 0 ]] && rg -q "Error during evaluation:|Traceback \\(most recent call last\\)" "${log_file}"; then
      rc=1
    fi
    if [[ ${rc} -eq 0 ]] && ! compgen -G "${out_dir}/*/*results*.json" >/dev/null; then
      rc=1
    fi
    status "DONE bench=${bench} mode=${mode} rc=${rc}"
    if [[ ${rc} -ne 0 ]]; then
      status "FAILED bench=${bench} mode=${mode}; see ${log_file}"
      exit "${rc}"
    fi
    port=$((port + 1))
  done
done
status "ALL COMPLETE run_id=${RUN_ID}"
