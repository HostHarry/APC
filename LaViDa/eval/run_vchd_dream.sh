#!/usr/bin/env bash
# Smoke / ablation runner for LaViDa-Dream + VCHD/CCAW.
# Usage:
#   bash eval/run_vchd_dream.sh /path/to/lavida-dream-hd
#   MODE=original|vchd|vchd_ccaw TASKS=mmmu_val LIMIT=1 bash eval/run_vchd_dream.sh CKPT

set -euo pipefail

CKPT=${1:?checkpoint path required}
MODE=${MODE:-vchd}
TASKS=${TASKS:-mmmu_val}
LIMIT=${LIMIT:-1}
NUM_PROCESSES=${NUM_PROCESSES:-1}
export LLADA_VISION_ENCODER=${LLADA_VISION_ENCODER:-google/siglip-so400m-patch14-384}
export DEBUG_PRINT_IMAGE_RES=${DEBUG_PRINT_IMAGE_RES:-1}

case "$MODE" in
  original)
    GEN_KWARGS="alg=topk_margin,prefix_lm=True,max_new_tokens=64,steps=64"
    ;;
  vchd)
    GEN_KWARGS="decode_strategy=vchd,prefix_lm=False,max_new_tokens=64,vchd__ccaw_enabled=false,vchd__enable_g_gate=false,vchd__mask_capacity=16,vchd__tau_base=0.1,vchd__tau_contrast=0.9"
    ;;
  vchd_ccaw)
    GEN_KWARGS="decode_strategy=vchd,prefix_lm=False,max_new_tokens=64,vchd__ccaw_enabled=true,vchd__ccaw_mode=inverse_window,vchd__enable_g_gate=false,vchd__mask_capacity=16,vchd__ccaw_max_mask_capacity=64,vchd__tau_base=0.1,vchd__tau_contrast=0.9"
    ;;
  *)
    echo "Unknown MODE=$MODE (expected original|vchd|vchd_ccaw)" >&2
    exit 1
    ;;
esac

EXTRA=()
if [[ -n "${LIMIT}" && "${LIMIT}" != "0" ]]; then
  EXTRA+=(--limit "${LIMIT}")
fi

cd "$(dirname "$0")"
accelerate launch --num_processes="${NUM_PROCESSES}" --main_process_port=25522 \
  -m lmms_eval \
  --model llava_dream \
  --model_args "pretrained=${CKPT},conv_template=dream,model_name=llava_dream" \
  --tasks "${TASKS}" \
  --batch_size 1 \
  --gen_kwargs "${GEN_KWARGS}" \
  --log_samples \
  --log_samples_suffix "dream_${MODE}" \
  --output_path "./logs/vchd_dream_${MODE}" \
  "${EXTRA[@]}" \
  "${@:2}"
