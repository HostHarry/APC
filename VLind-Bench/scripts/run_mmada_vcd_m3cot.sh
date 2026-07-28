#!/usr/bin/env bash
# MMaDA VCD on M3CoT via LightChen233 official scorer.
set -euo pipefail
ROOT=/root/autodl-tmp
OUT_DIR=${OUT_DIR:-${ROOT}/VLind-Bench/outputs/mmada_vcd_m3cot_$(date +%Y%m%d_%H%M%S)}
mkdir -p "${OUT_DIR}"
export PATH=/root/miniconda3/bin:${PATH}
export HF_HOME=${HF_HOME:-/autodl-fs/data/hf_home}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export http_proxy=${http_proxy:-http://127.0.0.1:7897}
export https_proxy=${https_proxy:-http://127.0.0.1:7897}
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true

VQ=${VQ:-/root/autodl-tmp/MMaDA-8B-MixCoT}
# Prefer dedicated magvit checkpoint when present
if [[ -d /root/autodl-tmp/magvitv2 ]]; then VQ=/root/autodl-tmp/magvitv2; fi
if [[ -d /autodl-fs/data/magvitv2 ]]; then VQ=/autodl-fs/data/magvitv2; fi

python /root/autodl-tmp/VLind-Bench/eval/mmada_m3cot_eval.py \
  --strategy "${STRATEGY:-vcd}" \
  --model-path /root/autodl-tmp/MMaDA-8B-MixCoT \
  --vq-model-path "${VQ}" \
  --max-new-tokens 512 \
  --steps 256 \
  --block-length 64 \
  --limit "${LIMIT:-0}" \
  --output "${OUT_DIR}/summary.json" \
  2>&1 | tee "${OUT_DIR}/run.log"

echo "OUT_DIR=${OUT_DIR}"
