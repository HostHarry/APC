#!/bin/bash
# Defer-only CV smoke test v4: 5 samples x 4 configs on LLaVABench.
#
# v4 rationale (from v3 analysis):
#   - v3 S4 (hard txt tau=-1.5, defer 7.8%) reached mean 3.80 (baseline 4.00),
#     preserving #54=10 and losing only 1pt on #7 (7 -> 6).
#   - Question: can EVEN MORE NEGATIVE tau eliminate the #7 loss (by sparser
#     intervention) while still catching a chance on #12?
#   - Target: bring #steps close to baseline (S7 ~50) while keeping quality.
#
# Config table (all hard veto, text_only drop, gain_type=logit):
#   S1  hard txt tau=-2.0   (expected defer ~5%, #steps ~65)
#   S2  hard txt tau=-2.5   (expected defer ~4%, #steps ~60)
#   S3  hard txt tau=-3.0   (expected defer ~2-3%, #steps ~55)
#   S4  lambda=0 sanity     (must match baseline exactly)
#
# Success metrics (NOT debug-side defer_active, but real decoding cost):
#   1. avg #steps close to S4 sanity (~50)
#   2. #7 score >= 7 (baseline; v3 S4 got 6)
#   3. #54 score = 10 (baseline; v3 shuf variants failed here)
#   4. #12 score > 1 (bonus rescue; v2 S1 got 8, v3 all got 1)
#
# Usage:
#   nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
#   bash scripts/run_defer_only_smoke_v4.sh
#
# Expected runtime: ~30 min on a single H100 (5 samples * 4 configs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VLMEVAL_ROOT}"

PYTHON="${PYTHON:-/home/user/anaconda3/envs/mmada/bin/python}"
BASE_RUN_ID="${MMADA_RUN_ID:-defer_only_smoke_v4_$(date +%Y%m%d_%H%M%S)}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU status before smoke ==="
  nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader || true
  BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "${BUSY}" != "0" ]; then
    echo "WARNING: ${BUSY} process(es) using GPU. If <25 GiB free, may OOM."
  fi
  echo ""
fi

export MMADA_INDICES="${MMADA_INDICES:-3,7,12,45,54}"
export MMADA_DECODE_STRATEGY=cv_dcd
export MMADA_CV_MODE=defer_only
export MMADA_CV_ALPHA="${MMADA_CV_ALPHA:-0.1}"
export MMADA_CV_STRIDE="${MMADA_CV_STRIDE:-1}"
export MMADA_CV_RETURN_DEBUG="${MMADA_CV_RETURN_DEBUG:-1}"

DEBUG_ROOT="${VLMEVAL_ROOT}/attention_analysis/cv_debug_v4/${BASE_RUN_ID}"
mkdir -p "./outputs/cvdcd_sweep/${BASE_RUN_ID}"
mkdir -p "${DEBUG_ROOT}"

# tag                     veto  tau   beta  lambda  drop        gain_type
CONFIGS=(
  "S1_hard_txt_t-2.0      hard  -2.0  1.0   0.5     text_only   logit"
  "S2_hard_txt_t-2.5      hard  -2.5  1.0   0.5     text_only   logit"
  "S3_hard_txt_t-3.0      hard  -3.0  1.0   0.5     text_only   logit"
  "S4_lambda0_sanity      hard   0.0  1.0   0.0     text_only   logit"
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
  echo "[$(date +%H:%M:%S)] SMOKE v4 tag=${tag}"
  echo "  veto=${veto} tau=${tau} beta=${beta} lambda=${lam}"
  echo "  drop=${drop} gain_type=${gain_type}"
  echo "========================================"
  "${PYTHON}" run.py \
    --data LLaVABench \
    --model MMaDA-MixCoT-CV-DCD \
    --work-dir "${OUT_DIR}" \
    "$@" || {
      rc=$?
      echo "WARNING: run.py exited with code ${rc} for tag=${tag}"
    }
  sleep 3
done

echo ""
echo "[$(date +%H:%M:%S)] Defer-only smoke v4 complete."
echo "  base dir:  ./outputs/cvdcd_sweep/${BASE_RUN_ID}"
echo "  debug dir: ${DEBUG_ROOT}"
