#!/bin/bash
# Defer-only Phase D FOCUSED sweep: 60 samples x 2 configs on LLaVABench.
#
# This is a targeted verification, not exploration:
#   - v4 smoke identified S3 (hard, text_only, tau=-3.0) as the best defer_only
#     variant (baseline-preserving, minimal NFE overhead).
#   - Sample-mean scores at 5 samples fell in GPT-4 noise range (3.80 vs 4.00).
#   - Question: on 60 samples, does noise average out to reveal a
#     baseline-beating trend, or does defer_only stay at ~baseline?
#
# Configs:
#   D_focal  hard text_only tau=-3.0 beta=1 lambda=0.5 gain_type=logit
#   D_sanity same params but lambda=0 (MUST match baseline DCD exactly)
#
# Success criteria (relative to baseline mean 3.82 over 60 samples):
#   1. D_sanity mean == baseline mean (validates code path)
#   2. D_focal mean > D_sanity mean + 0.15 (real improvement above noise)
#   3. D_focal preserves samples where baseline scored >= 6 (no destruction)
#   4. D_focal avg #steps <= 1.5x baseline (compute cost acceptable)
#
# Expected runtime: ~2h on a single H100 (60 samples * 2 configs * ~60s/sample).
#
# Usage:
#   nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
#   bash scripts/run_defer_only_phase_d_focused.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VLMEVAL_ROOT}"

PYTHON="${PYTHON:-/home/user/anaconda3/envs/mmada/bin/python}"
BASE_RUN_ID="${MMADA_RUN_ID:-defer_only_phase_d_focused_$(date +%Y%m%d_%H%M%S)}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU status before Phase D focused ==="
  nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader || true
  BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "${BUSY}" != "0" ]; then
    echo "WARNING: ${BUSY} process(es) using GPU. If <25 GiB free, may OOM."
  fi
  echo ""
fi

# 60 samples (all of LLaVABench). Do NOT set MMADA_INDICES.
unset MMADA_INDICES || true

export MMADA_DECODE_STRATEGY=cv_dcd
export MMADA_CV_MODE=defer_only
export MMADA_CV_ALPHA="${MMADA_CV_ALPHA:-0.1}"
export MMADA_CV_STRIDE="${MMADA_CV_STRIDE:-1}"
export MMADA_CV_RETURN_DEBUG="${MMADA_CV_RETURN_DEBUG:-1}"

DEBUG_ROOT="${VLMEVAL_ROOT}/attention_analysis/cv_debug_v4/${BASE_RUN_ID}"
mkdir -p "./outputs/cvdcd_sweep/${BASE_RUN_ID}"
mkdir -p "${DEBUG_ROOT}"

# tag                 veto  tau   beta  lambda  drop        gain_type
CONFIGS=(
  "D_focal_hard_txt_t-3.0   hard  -3.0  1.0   0.5   text_only   logit"
  "D_sanity_lambda0         hard  -3.0  1.0   0.0   text_only   logit"
)

for cfg in "${CONFIGS[@]}"; do
  read -r tag veto tau beta lam drop gain_type <<< "${cfg}"
  OUT_DIR="./outputs/cvdcd_sweep/${BASE_RUN_ID}/${tag}"
  export MMADA_DEFER_VETO="${veto}"
  export MMADA_DEFER_TAU="${tau}"
  export MMADA_DEFER_BETA="${beta}"
  export MMADA_DEFER_GAIN_TYPE="${gain_type}"
  export MMADA_CV_LAMBDA="${lam}"
  export MMADA_CV_DROP="${drop}"
  export MMADA_CV_DEBUG_DIR="${DEBUG_ROOT}/${tag}"
  mkdir -p "${MMADA_CV_DEBUG_DIR}"
  echo "========================================"
  echo "[$(date +%H:%M:%S)] Phase D focused tag=${tag}"
  echo "  veto=${veto} tau=${tau} beta=${beta} lambda=${lam}"
  echo "  drop=${drop} gain_type=${gain_type}"
  echo "  60 samples (all LLaVABench)"
  echo "  out=${OUT_DIR}"
  echo "========================================"
  "${PYTHON}" run.py \
    --data LLaVABench \
    --model MMaDA-MixCoT-CV-DCD \
    --work-dir "${OUT_DIR}" \
    "$@" || {
      rc=$?
      echo "WARNING: run.py exited with code ${rc} for tag=${tag}"
    }
  sleep 5
done

echo ""
echo "[$(date +%H:%M:%S)] Phase D focused complete."
echo "  base dir:  ./outputs/cvdcd_sweep/${BASE_RUN_ID}"
echo "  debug dir: ${DEBUG_ROOT}"
echo ""
echo "Next: analyze with"
echo "  ${PYTHON} - <<PY"
echo "    # (analysis script printed after run)"
echo "PY"
