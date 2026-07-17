#!/bin/bash
# Defer-only CV smoke test v3: 5 samples x 7 configs on LLaVABench.
#
# v3 rationale (from v2 analysis):
#   - Math fixes (gain_type=logit, apply_veto_soft) confirmed working.
#   - v2 defer rate 20-30% at tau=0 destroyed DCD parallelism
#     (commits/step 5.08 -> 1.5-2.0).
#   - Beta softening does NOT reduce intervention rate: DCD threshold=0.9
#     vs base_conf~0.95 leaves only 5% margin, so any penalty triggers
#     fallback. Tau shift is the correct knob.
#   - Target defer rate: 5-15% (comparable to v1's 7.66% but with real
#     logit-scale signal instead of near-zero logprob signal).
#
# Config table:
#   Shuffle/logit:
#     S1  hard tau=-1.0            (target ~10% defer)
#     S2  soft tau=-1.0 beta=1     (soft variant of S1)
#     S3  soft tau=-1.5 beta=1     (target ~5% defer)
#   Text_only/logit:
#     S4  hard tau=-1.5            (target ~12% defer)
#     S5  soft tau=-1.5 beta=1     (soft variant of S4)
#     S6  soft tau=-2.0 beta=1     (target ~7% defer)
#   Sanity:
#     S7  lambda=0                 (MUST match baseline DCD text-for-text)
#
# Success criteria per config (relative to v2's failing 20-30% variants):
#   1. defer_rate in [3%, 15%]
#   2. commits/step >= 3.5 (baseline was 5.08; v2 was ~1.5-2.0)
#   3. at least 3/5 samples identical to baseline
#   4. mean_score not worse than baseline (4.0) minus 0.5 = 3.5
#
# Usage:
#   nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9   # clean GPU
#   bash scripts/run_defer_only_smoke_v3.sh
#
# Expected runtime: ~50 min on a single H100 (5 samples * 7 configs * ~90s).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VLMEVAL_ROOT}"

PYTHON="${PYTHON:-/home/user/anaconda3/envs/mmada/bin/python}"
BASE_RUN_ID="${MMADA_RUN_ID:-defer_only_smoke_v3_$(date +%Y%m%d_%H%M%S)}"

# Reduce fragmentation for the paired base+drop forward passes.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Pre-flight GPU check.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU status before smoke ==="
  nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader || true
  BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "${BUSY}" != "0" ]; then
    echo "WARNING: ${BUSY} process(es) using GPU. If <25 GiB free, may OOM."
    echo "  Cleanup: nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9"
  fi
  echo ""
fi

# Same 5 indices as v1/v2 for direct comparison.
export MMADA_INDICES="${MMADA_INDICES:-3,7,12,45,54}"

# Shared: defer-only with drop_forward enabled.
export MMADA_DECODE_STRATEGY=cv_dcd
export MMADA_CV_MODE=defer_only
export MMADA_CV_ALPHA="${MMADA_CV_ALPHA:-0.1}"       # unused by defer_only
export MMADA_CV_STRIDE="${MMADA_CV_STRIDE:-1}"
export MMADA_CV_RETURN_DEBUG="${MMADA_CV_RETURN_DEBUG:-1}"

DEBUG_ROOT="${VLMEVAL_ROOT}/attention_analysis/cv_debug_v4/${BASE_RUN_ID}"
mkdir -p "./outputs/cvdcd_sweep/${BASE_RUN_ID}"
mkdir -p "${DEBUG_ROOT}"

# tag                     veto  tau   beta  lambda  drop        gain_type
CONFIGS=(
  "S1_hard_shuf_t-1.0     hard  -1.0  1.0   0.5     shuffle     logit"
  "S2_soft_shuf_t-1.0     soft  -1.0  1.0   0.5     shuffle     logit"
  "S3_soft_shuf_t-1.5     soft  -1.5  1.0   0.5     shuffle     logit"
  "S4_hard_txt_t-1.5      hard  -1.5  1.0   0.5     text_only   logit"
  "S5_soft_txt_t-1.5      soft  -1.5  1.0   0.5     text_only   logit"
  "S6_soft_txt_t-2.0      soft  -2.0  1.0   0.5     text_only   logit"
  "S7_lambda0_sanity      soft   0.0  1.0   0.0     shuffle     logit"
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
  echo "[$(date +%H:%M:%S)] SMOKE v3 tag=${tag}"
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
      echo "         (continuing to next config)"
    }
  # Brief settle so PyTorch's caching allocator releases between configs.
  sleep 3
done

echo ""
echo "[$(date +%H:%M:%S)] Defer-only smoke v3 complete."
echo "  base dir:  ./outputs/cvdcd_sweep/${BASE_RUN_ID}"
echo "  debug dir: ${DEBUG_ROOT}"
echo ""
echo "Priority sanity checks (see docs/cv_dcd_v4_design.md):"
echo "  1. S7 (lambda=0) MUST match baseline DCD text-for-text on all 5 samples."
echo "  2. Was #7 (baseline=7) preserved? v2 shuf variants dropped it to 1."
echo "  3. Was #54 (baseline=10) preserved? v2 shuf variants dropped it to 1."
echo "  4. commits/step should be >= 3.5 (baseline 5.08, v2 was 1.5-2.0)."
