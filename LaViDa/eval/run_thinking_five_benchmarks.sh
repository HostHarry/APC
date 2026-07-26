#!/usr/bin/env bash
# LaViDa-LLaDA: five aligned benchmarks × swd|psp|psp_vrg by default.
# Aligned schedules are written as max_new_tokens / total_steps / block_length:
#   MMBench, MMMU, VLind  128 / 128 / 64
#   M3CoT                 512 / 256 / 64
#   LLaVA-Bench           256 / 128 / 256
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT=${1:-"${ROOT}/lavida-ckpts/lavida-llada-hd-reason"}
MODES=${MODES:-swd,psp,psp_vrg}
BENCHMARKS=${BENCHMARKS:-vlind,m3cot,mmmu,mmbench,llava_bench}
NUM_PROCESSES=${NUM_PROCESSES:-2}
LIMIT=${LIMIT:-0}
RUN_ID=${RUN_ID:-lavida_thinking_5bench_$(date +%Y%m%d_%H%M%S)}
START_PORT=${START_PORT:-25831}
JUDGE_ENV_FILE=${JUDGE_ENV_FILE:-/root/autodl-tmp/lladav_vchd_server_bundle/VLMEvalKit/.env}

export PYTHONPATH="${ROOT}:${ROOT}/eval:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/autodl-fs/data/hf_home}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export LLADA_VISION_ENCODER=${LLADA_VISION_ENCODER:-/autodl-fs/data/lavida-ckpts/siglip-so400m-patch14-384}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export DEBUG_PRINT_IMAGE_RES=${DEBUG_PRINT_IMAGE_RES:-0}
export PYTHONUNBUFFERED=1
export LLAVA_JUDGE_MODEL=${LLAVA_JUDGE_MODEL:-gpt-4o}
export LLAVA_JUDGE_SKIP=${LLAVA_JUDGE_SKIP:-1}

if [[ -f "${JUDGE_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${JUDGE_ENV_FILE}"
  set +a
fi

# Prefer local SSH/Clash bridge when present (AutoDL often has no direct egress).
if [[ -z "${http_proxy:-}${HTTP_PROXY:-}" ]] && (echo >/dev/tcp/127.0.0.1/7897) >/dev/null 2>&1; then
  export http_proxy=http://127.0.0.1:7897
  export https_proxy=http://127.0.0.1:7897
  export HTTP_PROXY=http://127.0.0.1:7897
  export HTTPS_PROXY=http://127.0.0.1:7897
  export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
  export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"
fi

LOG_ROOT="${ROOT}/eval/logs/${RUN_ID}"
mkdir -p "${LOG_ROOT}"
printf '%s\n' "${RUN_ID}" >"${ROOT}/eval/logs/LATEST_LAVIDA_THINKING_RUN"

status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_ROOT}/status.log"
}

# Prints: task max_new_tokens total_steps block_length
benchmark_config() {
  case "$1" in
    mmbench)
      printf '%s %s %s %s\n' mmbench_en_dev 128 128 64
      ;;
    mmmu)
      printf '%s %s %s %s\n' mmmu_dev_val_full 128 128 64
      ;;
    m3cot)
      printf '%s %s %s %s\n' m3cot_full 512 256 64
      ;;
    vlind)
      printf '%s %s %s %s\n' vlind_bench_full 128 128 64
      ;;
    llava_bench)
      printf '%s %s %s %s\n' llava_bench_coco 256 128 256
      ;;
    *)
      echo "Unknown benchmark: $1" >&2
      return 2
      ;;
  esac
}

port=${START_PORT}
IFS=',' read -r -a benchmark_array <<<"${BENCHMARKS}"
IFS=',' read -r -a mode_array <<<"${MODES}"

status "START run_id=${RUN_ID} ckpt=${CKPT} benchmarks=${BENCHMARKS} modes=${MODES} limit=${LIMIT}"
status "schedules: mmbench=128/128/64 mmmu=128/128/64 m3cot=512/256/64 vlind=128/128/64 llava_bench=256/128/256"
status "judge: skip=${LLAVA_JUDGE_SKIP} model=${LLAVA_JUDGE_MODEL} key=$([[ -n "${OPENAI_API_KEY:-}" ]] && echo set || echo unset)"

for benchmark in "${benchmark_array[@]}"; do
  benchmark="$(echo "${benchmark}" | xargs)"
  read -r task max_tokens total_steps block_length <<<"$(benchmark_config "${benchmark}")"
  num_blocks=$((max_tokens / block_length))
  step_per_block=$((total_steps / num_blocks))
  if ((step_per_block * num_blocks != total_steps)); then
    status "ERROR non-divisible schedule benchmark=${benchmark} L/T/B=${max_tokens}/${total_steps}/${block_length}"
    exit 3
  fi

  for mode in "${mode_array[@]}"; do
    mode="$(echo "${mode}" | xargs)"
    out_dir="${LOG_ROOT}/${benchmark}/${mode}"
    log_file="${LOG_ROOT}/${benchmark}__${mode}.log"
    mkdir -p "${out_dir}"
    status "START benchmark=${benchmark} task=${task} mode=${mode} L/T/B=${max_tokens}/${total_steps}/${block_length} step_per_block=${step_per_block}"

    set +e
    MODE="${mode}" \
    TASKS="${task}" \
    LIMIT="${LIMIT}" \
    NUM_PROCESSES="${NUM_PROCESSES}" \
    MAX_NEW_TOKENS="${max_tokens}" \
    BLOCK_LENGTH="${block_length}" \
    STEP_PER_BLOCK="${step_per_block}" \
    MAIN_PROCESS_PORT="${port}" \
    OUTPUT_PATH="${out_dir}" \
      bash "${ROOT}/eval/run_thinking_llada.sh" "${CKPT}" \
      >>"${log_file}" 2>&1
    rc=$?
    set -e

    if [[ ${rc} -eq 0 ]] && rg -q \
      "Error during evaluation:|Traceback \\(most recent call last\\)|FAILED mode" \
      "${log_file}"; then
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
