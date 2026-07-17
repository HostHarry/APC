#!/bin/bash
# Defer-only v5 matrix: (lambda, tau) 2D grid on FULL 60 LLaVABench samples.
#
# v5 semantic change: causal_lambda is now INTERVENTION STRENGTH in [0, 1],
# blending base_conf and veto_conf linearly:
#     eff_conf = (1 - lambda) * base_conf + lambda * veto_conf(base_conf, gain, tau)
#
#   - lambda = 0    : no intervention (== plain DCD ; sanity)
#   - lambda = 1    : full veto        (== v1-v4 defer_only)
#   - lambda in (0,1): partial veto    (soft interpolation)
#
# tau controls WHICH positions get intervention (frequency).
# lambda controls HOW MUCH intervention is applied (strength).
#
# Matrix (7 configs total = ~1h GPU on H100):
#   E0 sanity            lambda=0    tau=0
#   E1 weak veto  wide   lambda=0.25 tau=-2.0
#   E2 weak veto  narrow lambda=0.25 tau=-3.0
#   E3 mid  veto  wide   lambda=0.50 tau=-2.0
#   E4 mid  veto  narrow lambda=0.50 tau=-3.0
#   E5 full veto  wide   lambda=1.00 tau=-2.0
#   E6 full veto  narrow lambda=1.00 tau=-3.0   (== Phase D focused, reference)
#
# Common config: hard veto, text_only drop, logit gain (v4's best combo).
#
# Success criteria:
#   1. E0 (sanity) ≈ prior sanity mean 2.58 within +/-0.15 (validates env stability)
#   2. Any (lambda, tau) point mean > E0 by >= 0.20 (larger than typical GPT noise)
#   3. That point's #steps <= 1.5 * E0 #steps (compute-cost acceptable)
#   4. That point preserves baseline text_match >= 25/60
#
# Expected runtime: ~1h on a single H100 (60 samples * 7 configs * ~6-8 min inference + scoring).
#
# Usage:
#   nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
#   bash scripts/run_defer_only_matrix_v5.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VLMEVAL_ROOT}"

PYTHON="${PYTHON:-/home/user/anaconda3/envs/mmada/bin/python}"
BASE_RUN_ID="${MMADA_RUN_ID:-defer_only_matrix_v5_$(date +%Y%m%d_%H%M%S)}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU status before matrix v5 ==="
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
export MMADA_CV_ALPHA="${MMADA_CV_ALPHA:-0.1}"          # unused in defer_only, kept for schema
export MMADA_CV_STRIDE="${MMADA_CV_STRIDE:-1}"
export MMADA_CV_RETURN_DEBUG="${MMADA_CV_RETURN_DEBUG:-1}"
export MMADA_CV_DROP="text_only"                        # v4's best drop strategy
export MMADA_DEFER_VETO="hard"                          # v4's best veto (soft explored elsewhere)
export MMADA_DEFER_BETA="1.0"
export MMADA_DEFER_GAIN_TYPE="logit"

DEBUG_ROOT="${VLMEVAL_ROOT}/attention_analysis/cv_debug_v4/${BASE_RUN_ID}"
mkdir -p "./outputs/cvdcd_sweep/${BASE_RUN_ID}"
mkdir -p "${DEBUG_ROOT}"

# tag                       lambda  tau
CONFIGS=(
  "E0_sanity_l0             0.0     0.0"
  "E1_l0.25_t-2.0           0.25    -2.0"
  "E2_l0.25_t-3.0           0.25    -3.0"
  "E3_l0.50_t-2.0           0.50    -2.0"
  "E4_l0.50_t-3.0           0.50    -3.0"
  "E5_l1.00_t-2.0           1.00    -2.0"
  "E6_l1.00_t-3.0           1.00    -3.0"
)

echo "=== Matrix v5: 7 configs x 60 samples ==="
echo ""
printf "  %-25s %s %s\n" "config" "lambda" "tau"
for cfg in "${CONFIGS[@]}"; do
  read -r tag lam tau <<< "${cfg}"
  printf "  %-25s %s     %s\n" "${tag}" "${lam}" "${tau}"
done
echo ""

for cfg in "${CONFIGS[@]}"; do
  read -r tag lam tau <<< "${cfg}"
  OUT_DIR="./outputs/cvdcd_sweep/${BASE_RUN_ID}/${tag}"
  export MMADA_DEFER_TAU="${tau}"
  export MMADA_CV_LAMBDA="${lam}"
  export MMADA_CV_DEBUG_DIR="${DEBUG_ROOT}/${tag}"
  mkdir -p "${MMADA_CV_DEBUG_DIR}"
  echo "========================================"
  echo "[$(date +%H:%M:%S)] Matrix v5 tag=${tag}"
  echo "  lambda=${lam}  tau=${tau}  (veto=hard drop=text_only gain=logit)"
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
echo "[$(date +%H:%M:%S)] Matrix v5 complete."
echo "  base dir:  ./outputs/cvdcd_sweep/${BASE_RUN_ID}"
echo "  debug dir: ${DEBUG_ROOT}"
