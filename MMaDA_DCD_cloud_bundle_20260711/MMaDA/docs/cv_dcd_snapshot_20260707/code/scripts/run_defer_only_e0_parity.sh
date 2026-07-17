#!/bin/bash
# E0-parity head-to-head: verify that after the dispatcher fix,
#   defer_only + causal_lambda=0 (E0)   ==   plain DCD (dcd_decode_text_dual_cache)
# byte-identically, and re-verify E4/E6 with the corrected baseline.
#
# Background
# ----------
# The Matrix v5 sweep's E0 (defer_only, lambda=0) was intended as a
# byte-identical sanity for plain DCD, but ``pick_transfer_defer_only``
# computed base_conf via a memory-efficient ``logsumexp(logits.bfloat16)``
# path that diverged from plain DCD's ``F.softmax(logits.to(float64))`` at
# the ~2^-7 precision level. Near threshold=0.9 this could flip commit
# order and produce different text/scores (see D_sanity 2.58 vs baseline
# 3.82 -- most of that gap is GPT scorer version + temperature sampling,
# but the numerical drift contributes non-trivially).
#
# After the fix in ``models/defer_only/dispatcher.py`` (softmax(fp64) path),
# this script proves E0 == plain DCD byte-identically on 60 LLaVABench
# samples, and re-runs E4/E6 to give a same-run relative delta.
#
# Configs (4 total = ~35 min GPU each on H100, ~2.5h total):
#   B0_plain_dcd            decode_strategy=dcd, cv_mode=off, cache_type=dual
#   E0_defer_l0             cv_dcd, defer_only, lambda=0.0
#   E4_defer_l0.5_t-3.0     cv_dcd, defer_only, lambda=0.5, tau=-3.0
#   E6_defer_l1.0_t-3.0     cv_dcd, defer_only, lambda=1.0, tau=-3.0
#
# Success criteria:
#   1. B0 and E0 produce byte-identical LLaVABench predictions (60/60).
#      Verify via ``diff <(jq -r .prediction B0/....xlsx.json) <(jq -r .prediction E0/....json)``.
#   2. E4 and E6 GPT-4 mean score >= B0 mean score - 0.1 (fair baseline).
#   3. E6 preserves "detail" category (was v5's uniqueness).
#
# Usage:
#   nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
#   bash scripts/run_defer_only_e0_parity.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VLMEVAL_ROOT}"

PYTHON="${PYTHON:-/home/user/anaconda3/envs/mmada/bin/python}"
BASE_RUN_ID="${MMADA_RUN_ID:-defer_only_e0_parity_$(date +%Y%m%d_%H%M%S)}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU status before e0-parity run ==="
  nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader || true
  BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "${BUSY}" != "0" ]; then
    echo "WARNING: ${BUSY} process(es) using GPU. If <25 GiB free, may OOM."
  fi
  echo ""
fi

# 60 samples (all of LLaVABench). Do NOT set MMADA_INDICES.
unset MMADA_INDICES || true

# Common defer_only knobs (only consumed when cv_mode==defer_only).
export MMADA_CV_ALPHA="${MMADA_CV_ALPHA:-0.1}"
export MMADA_CV_STRIDE="${MMADA_CV_STRIDE:-1}"
export MMADA_CV_RETURN_DEBUG="${MMADA_CV_RETURN_DEBUG:-1}"
export MMADA_CV_DROP="text_only"
export MMADA_DEFER_VETO="hard"
export MMADA_DEFER_BETA="1.0"
export MMADA_DEFER_GAIN_TYPE="logit"

DEBUG_ROOT="${VLMEVAL_ROOT}/attention_analysis/cv_debug_v4/${BASE_RUN_ID}"
mkdir -p "./outputs/cvdcd_sweep/${BASE_RUN_ID}"
mkdir -p "${DEBUG_ROOT}"

# tag                       strategy   cv_mode      lambda  tau
# NB: for the plain-DCD baseline (B0) we override MMADA_DECODE_STRATEGY=dcd,
# which routes through dcd_decode_text_dual_cache (bypassing _pick_transfer_cv).
CONFIGS=(
  "B0_plain_dcd             dcd        off          0.0     0.0"
  "E0_defer_l0              cv_dcd     defer_only   0.0     0.0"
  "E4_defer_l0.5_t-3.0      cv_dcd     defer_only   0.5     -3.0"
  "E6_defer_l1.0_t-3.0      cv_dcd     defer_only   1.0     -3.0"
)

echo "=== E0 parity head-to-head: 4 configs x 60 samples ==="
echo ""
printf "  %-25s %-8s %-12s %s %s\n" "config" "strategy" "cv_mode" "lambda" "tau"
for cfg in "${CONFIGS[@]}"; do
  read -r tag strategy cv_mode lam tau <<< "${cfg}"
  printf "  %-25s %-8s %-12s %s     %s\n" "${tag}" "${strategy}" "${cv_mode}" "${lam}" "${tau}"
done
echo ""

for cfg in "${CONFIGS[@]}"; do
  read -r tag strategy cv_mode lam tau <<< "${cfg}"
  OUT_DIR="./outputs/cvdcd_sweep/${BASE_RUN_ID}/${tag}"

  export MMADA_DECODE_STRATEGY="${strategy}"
  export MMADA_CV_MODE="${cv_mode}"
  export MMADA_DEFER_TAU="${tau}"
  export MMADA_CV_LAMBDA="${lam}"
  export MMADA_CV_DEBUG_DIR="${DEBUG_ROOT}/${tag}"
  mkdir -p "${MMADA_CV_DEBUG_DIR}"

  echo "========================================"
  echo "[$(date +%H:%M:%S)] E0-parity tag=${tag}"
  echo "  strategy=${strategy}  cv_mode=${cv_mode}"
  echo "  lambda=${lam}  tau=${tau}"
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
echo "[$(date +%H:%M:%S)] E0-parity complete."
echo "  base dir:  ./outputs/cvdcd_sweep/${BASE_RUN_ID}"
echo "  debug dir: ${DEBUG_ROOT}"
echo ""
echo "Next steps (analysis):"
echo "  1. Byte-identity check between B0 and E0:"
echo "     ${PYTHON} scripts/compare_e0_parity.py \\"
echo "         --base ./outputs/cvdcd_sweep/${BASE_RUN_ID}/B0_plain_dcd \\"
echo "         --alt  ./outputs/cvdcd_sweep/${BASE_RUN_ID}/E0_defer_l0"
echo "  2. Score deltas vs corrected baseline B0:"
echo "     ${PYTHON} scripts/summarize_defer_only.py \\"
echo "         --root ./outputs/cvdcd_sweep/${BASE_RUN_ID}"
