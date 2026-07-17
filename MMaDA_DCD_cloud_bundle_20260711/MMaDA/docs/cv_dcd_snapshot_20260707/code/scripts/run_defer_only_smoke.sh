#!/bin/bash
# Defer-only CV smoke test: 5 samples x 4 configs on LLaVABench.
#
# S1 mult tau=0    beta=1  lambda=0.5  shuffle  (primary candidate)
# S2 hard tau=0    -       lambda=0.5  shuffle  (hardest veto)
# S3 min  -        beta=1  lambda=0.5  shuffle  (two-sided consensus)
# S4 mult tau=0    beta=1  lambda=0.0  shuffle  (sanity: MUST match baseline DCD)
#
# Usage:
#   bash scripts/run_defer_only_smoke.sh
#
# Expected runtime: ~30 min on a single H100 (5 samples * 4 configs * ~90s).
# Expected VLM: mostly near baseline (36.7). S4 should equal baseline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VLMEVAL_ROOT}"

PYTHON="${PYTHON:-/home/user/anaconda3/envs/mmada/bin/python}"
BASE_RUN_ID="${MMADA_RUN_ID:-defer_only_smoke_$(date +%Y%m%d_%H%M%S)}"

# Only run these 5 indices for smoke.
export MMADA_INDICES="${MMADA_INDICES:-3,7,12,45,54}"

# Shared: defer-only mode with drop_forward enabled.
export MMADA_DECODE_STRATEGY=cv_dcd
export MMADA_CV_MODE=defer_only
export MMADA_CV_ALPHA="${MMADA_CV_ALPHA:-0.1}"       # unused by defer_only but harmless
export MMADA_CV_DROP="${MMADA_CV_DROP:-shuffle}"
export MMADA_CV_STRIDE="${MMADA_CV_STRIDE:-1}"
export MMADA_CV_RETURN_DEBUG="${MMADA_CV_RETURN_DEBUG:-1}"

DEBUG_ROOT="${VLMEVAL_ROOT}/attention_analysis/cv_debug_v4/${BASE_RUN_ID}"
mkdir -p "./outputs/cvdcd_sweep/${BASE_RUN_ID}"
mkdir -p "${DEBUG_ROOT}"

# tag              veto  tau   beta  lambda
CONFIGS=(
  "S1_mult_t0_b1   mult  0.0   1.0   0.5"
  "S2_hard_t0      hard  0.0   1.0   0.5"
  "S3_min_b1       min   0.0   1.0   0.5"
  "S4_lambda0      mult  0.0   1.0   0.0"
)

for cfg in "${CONFIGS[@]}"; do
  read -r tag veto tau beta lam <<< "${cfg}"
  OUT_DIR="./outputs/cvdcd_sweep/${BASE_RUN_ID}/${tag}"
  export MMADA_DEFER_VETO="${veto}"
  export MMADA_DEFER_TAU="${tau}"
  export MMADA_DEFER_BETA="${beta}"
  export MMADA_CV_LAMBDA="${lam}"
  export MMADA_CV_DEBUG_DIR="${DEBUG_ROOT}/${tag}"
  mkdir -p "${MMADA_CV_DEBUG_DIR}"
  echo "========================================"
  echo "[$(date +%H:%M:%S)] SMOKE tag=${tag}"
  echo "  veto=${veto} tau=${tau} beta=${beta} lambda=${lam} drop=${MMADA_CV_DROP}"
  echo "  indices=${MMADA_INDICES}"
  echo "  out=${OUT_DIR}"
  echo "========================================"
  "${PYTHON}" run.py \
    --data LLaVABench \
    --model MMaDA-MixCoT-CV-DCD \
    --work-dir "${OUT_DIR}" \
    "$@"
done

echo ""
echo "[$(date +%H:%M:%S)] Defer-only smoke complete."
echo "  base dir:  ./outputs/cvdcd_sweep/${BASE_RUN_ID}"
echo "  debug dir: ${DEBUG_ROOT}"
echo ""
echo "Sanity check: S4 (lambda=0) should equal baseline DCD on the same 5 indices."
echo "  diff \\"
echo "    outputs/cvdcd_sweep/${BASE_RUN_ID}/S4_lambda0/MMaDA-MixCoT-CV-DCD/T*/MMaDA-MixCoT-CV-DCD_LLaVABench_openai_result.xlsx \\"
echo "    outputs/MMaDA-MixCoT-DCD-DualCache/T20260613_G/MMaDA-MixCoT-DCD-DualCache_LLaVABench_openai_result.xlsx"
