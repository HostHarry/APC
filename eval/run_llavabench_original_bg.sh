#!/usr/bin/env bash
# Wrapper: LaViDa `original` LLaVABench 60-sample DDP baseline, mirroring the
# full-matrix env (proxy + GPT judge key) so postprocessing can auto-score.
# Only new logic vs run_vchd_llada.sh is env plumbing.

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

# Reuse the API key already embedded in the reruns launcher without duplicating
# the secret literal here. `eval` extracts only the three OPENAI/GPT export
# lines from that script.
eval "$(rg '^export (OPENAI_API_KEY|OPENAI_API_URL|GPT_EVAL_MODEL_NAME)=' /root/autodl-tmp/LaViDa/eval/run_fixed_bias_cache_reruns.sh)"

printf '[env] OPENAI_API_KEY set: %s\n' "$([[ -n "${OPENAI_API_KEY:-}" && "$OPENAI_API_KEY" != "YOUR_API_KEY" ]] && echo yes || echo no)"

mkdir -p "${RUN_ROOT}/llavabench/original"

printf 'START llavabench original (baseline 60) %s\n' "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"

cd /root/autodl-tmp/LaViDa
MODE=original \
TASKS=llava_in_the_wild \
LIMIT=0 \
NUM_PROCESSES=2 \
MAX_NEW_TOKENS=512 \
BLOCK_LENGTH=64 \
STEP_PER_BLOCK=64 \
MAIN_PROCESS_PORT=26020 \
OUTPUT_PATH="${RUN_ROOT}/llavabench/original" \
  bash eval/run_vchd_llada.sh "${CKPT}" 2>&1 | tee "${RUN_ROOT}/llavabench__original.log"

printf 'DONE llavabench original %s\n' "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
