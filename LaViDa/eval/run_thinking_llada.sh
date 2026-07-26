#!/usr/bin/env bash
# LaViDa-LLaDA original / SWD / PSP / VRG evaluation runner.
# Usage:
#   MODE=swd bash eval/run_thinking_llada.sh lavida-ckpts/lavida-llada-hd-reason
#   MODE=psp_vrg TASKS=m3cot_full LIMIT=0 bash eval/run_thinking_llada.sh CKPT
set -euo pipefail

CKPT=${1:-lavida-ckpts/lavida-llada-hd-reason}
MODE=${MODE:-swd}
TASKS=${TASKS:-mmmu_val}
LIMIT=${LIMIT:-1}
NUM_PROCESSES=${NUM_PROCESSES:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
BLOCK_LENGTH=${BLOCK_LENGTH:-${MAX_NEW_TOKENS}}
STEP_PER_BLOCK=${STEP_PER_BLOCK:-${BLOCK_LENGTH}}
SWD_LAMBDA=${SWD_LAMBDA:-5.0}
PSP_GAMMA=${PSP_GAMMA:-0.5}
VRG_SCALE=${VRG_SCALE:-0.5}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-25531}
OUTPUT_PATH=${OUTPUT_PATH:-"./logs/thinking_${MODE}"}

export LLADA_VISION_ENCODER=${LLADA_VISION_ENCODER:-google/siglip-so400m-patch14-384}
export DEBUG_PRINT_IMAGE_RES=${DEBUG_PRINT_IMAGE_RES:-0}

COMMON="block_length=${BLOCK_LENGTH},step_per_block=${STEP_PER_BLOCK},max_new_tokens=${MAX_NEW_TOKENS}"
case "${MODE}" in
  original)
    GEN_KWARGS="prefix_lm=True,${COMMON}"
    ;;
  swd)
    GEN_KWARGS="decode_strategy=swd,prefix_lm=True,${COMMON},thinking__swd_lambda=${SWD_LAMBDA}"
    ;;
  psp)
    GEN_KWARGS="decode_strategy=psp,prefix_lm=True,${COMMON},thinking__psp_gamma=${PSP_GAMMA}"
    ;;
  vrg)
    GEN_KWARGS="decode_strategy=vrg,prefix_lm=True,${COMMON},thinking__vrg_scale=${VRG_SCALE}"
    ;;
  psp_vrg)
    GEN_KWARGS="decode_strategy=psp_vrg,prefix_lm=True,${COMMON},thinking__psp_gamma=${PSP_GAMMA},thinking__vrg_scale=${VRG_SCALE}"
    ;;
  *)
    echo "Unknown MODE=${MODE} (expected original|swd|psp|vrg|psp_vrg)" >&2
    exit 2
    ;;
esac

EXTRA=()
if [[ -n "${LIMIT}" && "${LIMIT}" != "0" ]]; then
  EXTRA+=(--limit "${LIMIT}")
fi

cd "$(dirname "$0")"
echo "[run_thinking_llada] mode=${MODE} tasks=${TASKS} L=${MAX_NEW_TOKENS} block=${BLOCK_LENGTH} step_per_block=${STEP_PER_BLOCK}"
echo "[run_thinking_llada] gen_kwargs=${GEN_KWARGS}"
accelerate launch \
  --num_processes="${NUM_PROCESSES}" \
  --main_process_port="${MAIN_PROCESS_PORT}" \
  -m lmms_eval \
  --model llava_llada \
  --model_args "pretrained=${CKPT},conv_template=llada,model_name=llava_llada" \
  --tasks "${TASKS}" \
  --batch_size 1 \
  --gen_kwargs "${GEN_KWARGS}" \
  --log_samples \
  --log_samples_suffix "llada_${MODE}" \
  --output_path "${OUTPUT_PATH}" \
  "${EXTRA[@]}" \
  "${@:2}"
