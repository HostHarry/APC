#!/usr/bin/env bash
# Smoke / ablation runner for LaViDa-LLaDA + VCHD/CCAW.
# Usage:
#   bash eval/run_vchd_llada.sh /path/to/lavida-llada-hd
#   MODE=vchd_ccaw_prefix_cache TASKS=mmmu_val LIMIT=1 bash eval/run_vchd_llada.sh CKPT
#
# Schedule knobs (MMaDA-style L/T/block):
#   MAX_NEW_TOKENS / BLOCK_LENGTH / STEP_PER_BLOCK
#   total_steps ≈ (MAX_NEW_TOKENS / BLOCK_LENGTH) * STEP_PER_BLOCK
#   e.g. 128/128/64 → MAX_NEW_TOKENS=128 BLOCK_LENGTH=64 STEP_PER_BLOCK=64

set -euo pipefail

CKPT=${1:?checkpoint path required}
MODE=${MODE:-vchd}
TASKS=${TASKS:-mmmu_val}
LIMIT=${LIMIT:-1}
NUM_PROCESSES=${NUM_PROCESSES:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
BLOCK_LENGTH=${BLOCK_LENGTH:-${MAX_NEW_TOKENS}}
# Default: full steps within each block (step_per_block == block_length).
STEP_PER_BLOCK=${STEP_PER_BLOCK:-${BLOCK_LENGTH}}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-25521}
OUTPUT_PATH=${OUTPUT_PATH:-"./logs/vchd_${MODE}"}
export LLADA_VISION_ENCODER=${LLADA_VISION_ENCODER:-google/siglip-so400m-patch14-384}
export DEBUG_PRINT_IMAGE_RES=${DEBUG_PRINT_IMAGE_RES:-1}

# CCAW upper bound: allow expanding at least to block length.
CCAW_MAX_MASK=${CCAW_MAX_MASK_CAPACITY:-${BLOCK_LENGTH}}

case "$MODE" in
  original)
    GEN_KWARGS="prefix_lm=True,block_length=${BLOCK_LENGTH},step_per_block=${STEP_PER_BLOCK},max_new_tokens=${MAX_NEW_TOKENS}"
    ;;
  vchd)
    # VCHD pops block/step schedule; length is controlled by max_new_tokens.
    # alpha/beta: CD-APC contrast coeffs (paper default α=0.25, β=0.1).
    GEN_KWARGS="decode_strategy=vchd,prefix_lm=False,max_new_tokens=${MAX_NEW_TOKENS},vchd__ccaw_enabled=false,vchd__enable_g_gate=false,vchd__mask_capacity=16,vchd__alpha=0.25,vchd__beta=0.1,vchd__tau_base=0.1,vchd__tau_contrast=0.9"
    ;;
  vchd_ccaw)
    GEN_KWARGS="decode_strategy=vchd,prefix_lm=False,max_new_tokens=${MAX_NEW_TOKENS},vchd__ccaw_enabled=true,vchd__ccaw_mode=inverse_window,vchd__enable_g_gate=false,vchd__mask_capacity=16,vchd__ccaw_max_mask_capacity=${CCAW_MAX_MASK},vchd__alpha=0.25,vchd__beta=0.1,vchd__tau_base=0.1,vchd__tau_contrast=0.9"
    ;;
  vchd_prefix)
    GEN_KWARGS="decode_strategy=vchd,prefix_lm=True,max_new_tokens=${MAX_NEW_TOKENS},vchd__prefix_prompt_cache=false,vchd__ccaw_enabled=false,vchd__enable_g_gate=false,vchd__mask_capacity=16,vchd__alpha=0.25,vchd__beta=0.1,vchd__tau_base=0.1,vchd__tau_contrast=0.9"
    ;;
  vchd_prefix_cache)
    GEN_KWARGS="decode_strategy=vchd,prefix_lm=True,max_new_tokens=${MAX_NEW_TOKENS},vchd__prefix_prompt_cache=true,vchd__ccaw_enabled=false,vchd__enable_g_gate=false,vchd__mask_capacity=16,vchd__alpha=0.25,vchd__beta=0.1,vchd__tau_base=0.1,vchd__tau_contrast=0.9"
    ;;
  vchd_ccaw_prefix)
    GEN_KWARGS="decode_strategy=vchd,prefix_lm=True,max_new_tokens=${MAX_NEW_TOKENS},vchd__prefix_prompt_cache=false,vchd__ccaw_enabled=true,vchd__ccaw_mode=inverse_window,vchd__enable_g_gate=false,vchd__mask_capacity=16,vchd__ccaw_max_mask_capacity=${CCAW_MAX_MASK},vchd__alpha=0.25,vchd__beta=0.1,vchd__tau_base=0.1,vchd__tau_contrast=0.9"
    ;;
  vchd_ccaw_prefix_cache)
    GEN_KWARGS="decode_strategy=vchd,prefix_lm=True,max_new_tokens=${MAX_NEW_TOKENS},vchd__prefix_prompt_cache=true,vchd__ccaw_enabled=true,vchd__ccaw_mode=inverse_window,vchd__enable_g_gate=false,vchd__mask_capacity=16,vchd__ccaw_max_mask_capacity=${CCAW_MAX_MASK},vchd__alpha=0.25,vchd__beta=0.1,vchd__tau_base=0.1,vchd__tau_contrast=0.9"
    ;;
  vcd)
    # Pure VCD (Leng et al. CVPR 2024): noised-image negative branch + CD-APC.
    # Paper defaults α=1.0, β=0.1, noise_step=500. Uses original L/T/B schedule.
    GEN_KWARGS="decode_strategy=vcd,prefix_lm=False,max_new_tokens=${MAX_NEW_TOKENS},block_length=${BLOCK_LENGTH},step_per_block=${STEP_PER_BLOCK},vcd__noise_step=500,vcd__alpha=1.0,vcd__beta=0.1"
    ;;
  vcd_prefix)
    GEN_KWARGS="decode_strategy=vcd,prefix_lm=True,max_new_tokens=${MAX_NEW_TOKENS},block_length=${BLOCK_LENGTH},step_per_block=${STEP_PER_BLOCK},vcd__prefix_prompt_cache=false,vcd__noise_step=500,vcd__alpha=1.0,vcd__beta=0.1"
    ;;
  vcd_prefix_cache)
    GEN_KWARGS="decode_strategy=vcd,prefix_lm=True,max_new_tokens=${MAX_NEW_TOKENS},block_length=${BLOCK_LENGTH},step_per_block=${STEP_PER_BLOCK},vcd__prefix_prompt_cache=true,vcd__noise_step=500,vcd__alpha=1.0,vcd__beta=0.1"
    ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    exit 1
    ;;
esac

EXTRA=()
if [[ -n "${LIMIT}" && "${LIMIT}" != "0" ]]; then
  EXTRA+=(--limit "${LIMIT}")
fi

cd "$(dirname "$0")"
echo "[run_vchd_llada] MODE=${MODE} TASKS=${TASKS} L/T/block~ ${MAX_NEW_TOKENS}/${STEP_PER_BLOCK}x$((MAX_NEW_TOKENS/BLOCK_LENGTH))/${BLOCK_LENGTH} GEN_KWARGS=${GEN_KWARGS}"
accelerate launch --num_processes="${NUM_PROCESSES}" --main_process_port="${MAIN_PROCESS_PORT}" \
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
