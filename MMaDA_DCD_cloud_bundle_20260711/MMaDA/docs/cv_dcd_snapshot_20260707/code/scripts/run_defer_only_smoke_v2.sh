#!/bin/bash
# Defer-only CV smoke test v2: 5 samples x 6 configs on LLaVABench.
#
# Fixes over v1:
#   - Adds `soft` veto (asymmetric exp penalty, no zero-point pathology).
#   - Adds `gain_type=logit` (v3.2-scale gain magnitudes, ~+/-1.5 for shuffle).
#   - Adds `image_drop=text_only` (stronger gain signal than shuffle).
#
# S1  soft   shuffle   gain=logit    tau=0 beta=1  lambda=0.5   (revised mult, primary)
# S2  soft   text_only gain=logit    tau=0 beta=1  lambda=0.5   (soft + strong drop)
# S3  hard   shuffle   gain=logit    tau=0         lambda=0.5   (baseline vs v1 S2)
# S4  hard   text_only gain=logit    tau=0         lambda=0.5   (hard + strong drop)
# S5  soft   shuffle   gain=logprob  tau=0 beta=10 lambda=0.5   (CD-paper gain, large beta)
# S6  soft   shuffle   gain=logit    tau=0 beta=1  lambda=0.0   (SANITY: must equal baseline)
#
# Usage:
#   bash scripts/run_defer_only_smoke_v2.sh
#
# Expected runtime: ~45 min on a single H100 (5 samples * 6 configs * ~90s).
# Expected VLM: near baseline (36.7). S6 MUST equal baseline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VLMEVAL_ROOT}"

PYTHON="${PYTHON:-/home/user/anaconda3/envs/mmada/bin/python}"
BASE_RUN_ID="${MMADA_RUN_ID:-defer_only_smoke_v2_$(date +%Y%m%d_%H%M%S)}"

# Reduce CUDA fragmentation for the paired base+drop forward passes.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Pre-flight GPU check: warn (do NOT abort) if the GPU is already busy.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU status before smoke ==="
  nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader || true
  BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "${BUSY}" != "0" ]; then
    echo "WARNING: ${BUSY} process(es) currently using the GPU. Peak memory in the"
    echo "         paired forward is ~22 GiB; if free memory < 25 GiB this may OOM."
    echo "         To clean up: nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9"
  fi
  echo ""
fi

# Same 5 indices as v1 for direct comparison.
export MMADA_INDICES="${MMADA_INDICES:-3,7,12,45,54}"

# Shared: defer-only mode with drop_forward enabled.
export MMADA_DECODE_STRATEGY=cv_dcd
export MMADA_CV_MODE=defer_only
export MMADA_CV_ALPHA="${MMADA_CV_ALPHA:-0.1}"       # unused by defer_only
export MMADA_CV_STRIDE="${MMADA_CV_STRIDE:-1}"
export MMADA_CV_RETURN_DEBUG="${MMADA_CV_RETURN_DEBUG:-1}"

DEBUG_ROOT="${VLMEVAL_ROOT}/attention_analysis/cv_debug_v4/${BASE_RUN_ID}"
mkdir -p "./outputs/cvdcd_sweep/${BASE_RUN_ID}"
mkdir -p "${DEBUG_ROOT}"

# tag                      veto  tau   beta   lambda  drop        gain_type
CONFIGS=(
  "S1_soft_shuf_logit      soft  0.0   1.0    0.5     shuffle     logit"
  "S2_soft_txt_logit       soft  0.0   1.0    0.5     text_only   logit"
  "S3_hard_shuf_logit      hard  0.0   1.0    0.5     shuffle     logit"
  "S4_hard_txt_logit       hard  0.0   1.0    0.5     text_only   logit"
  "S5_soft_shuf_logprob    soft  0.0   10.0   0.5     shuffle     logprob"
  "S6_lambda0_sanity       soft  0.0   1.0    0.0     shuffle     logit"
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
  echo "[$(date +%H:%M:%S)] SMOKE tag=${tag}"
  echo "  veto=${veto} tau=${tau} beta=${beta} lambda=${lam}"
  echo "  drop=${drop} gain_type=${gain_type}"
  echo "  indices=${MMADA_INDICES}"
  echo "  out=${OUT_DIR}"
  echo "========================================"
  "${PYTHON}" run.py \
    --data LLaVABench \
    --model MMaDA-MixCoT-CV-DCD \
    --work-dir "${OUT_DIR}" \
    "$@" || {
      rc=$?
      echo "WARNING: run.py exited with code ${rc} for tag=${tag}"
      echo "         (continuing to next config; check ${OUT_DIR} for logs)"
    }
  # Force a brief settle so PyTorch's caching allocator releases between configs.
  # Prevents a stuck allocator state from carrying over on shared GPUs.
  sleep 3
done

echo ""
echo "[$(date +%H:%M:%S)] Defer-only smoke v2 complete."
echo "  base dir:  ./outputs/cvdcd_sweep/${BASE_RUN_ID}"
echo "  debug dir: ${DEBUG_ROOT}"
echo ""
echo "Post-run checks:"
echo "  1. S6 (lambda=0) output text MUST equal baseline DCD (sanity)."
echo "  2. Gain distribution:"
echo "     - S1/S3 (shuffle, logit): median should NOT be ~0 (unlike v1)"
echo "     - S2/S4 (text_only, logit): stronger signal expected"
echo "     - S5 (shuffle, logprob): should match v1's near-zero distribution"
