#!/bin/bash
# Defer-only CV Phase D sweep: 60 samples x 6 configs on LLaVABench.
#
# 6 configs isolate the fixed defer-only design:
#   - soft veto: asymmetric exp penalty, no sigmoid zero-point pathology
#   - gain_type: logit default plus logprob ablation
#   - drop: shuffle vs text_only
#
#   D0 soft tau=0 beta=1  lambda=0.0  shuffle   logit    Sanity: MUST equal DCD baseline.
#   D1 soft tau=0 beta=1  lambda=0.5  shuffle   logit    Primary smooth candidate.
#   D2 soft tau=0 beta=1  lambda=0.5  text_only logit    Stronger uncond signal.
#   D3 hard tau=0         lambda=0.5  shuffle   logit    Compare with previous hard sanity.
#   D4 hard tau=0         lambda=0.5  text_only logit    Hard + stronger uncond signal.
#   D5 soft tau=0 beta=10 lambda=0.5  shuffle   logprob  CD-paper gain ablation.
#
# Reference points:
#   v3.2 B3 min_base_blended  VLM 28.3   (previous CV-DCD SOTA)
#   DCD baseline              VLM 36.7   (target)
#
# Expected runtime: ~4h on a single H100.
# Success criteria:
#   D0 VLM ~ 36.7 (+/- 0.5)                       [internal consistency]
#   at least one of D1..D5 VLM > 28.3             [beat previous CV-DCD SOTA]
#   ideally at least one D1..D5 VLM > 34          [approach baseline]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VLMEVAL_ROOT}"

PYTHON="${PYTHON:-/home/user/anaconda3/envs/mmada/bin/python}"
BASE_RUN_ID="${MMADA_RUN_ID:-defer_only_phase_d_$(date +%Y%m%d_%H%M%S)}"

# Shared config.
export MMADA_DECODE_STRATEGY=cv_dcd
export MMADA_CV_MODE=defer_only
export MMADA_CV_ALPHA="${MMADA_CV_ALPHA:-0.1}"       # unused but kept for schema
export MMADA_CV_STRIDE="${MMADA_CV_STRIDE:-1}"
export MMADA_CV_RETURN_DEBUG="${MMADA_CV_RETURN_DEBUG:-1}"

DEBUG_ROOT="${VLMEVAL_ROOT}/attention_analysis/cv_debug_v4/${BASE_RUN_ID}"
mkdir -p "./outputs/cvdcd_sweep/${BASE_RUN_ID}"
mkdir -p "${DEBUG_ROOT}"

# tag                    veto  tau   beta  lambda  drop       gain_type
CONFIGS=(
  "D0_sanity_l0          soft  0.0   1.0    0.0     shuffle    logit"
  "D1_soft_shuf_logit    soft  0.0   1.0    0.5     shuffle    logit"
  "D2_soft_txt_logit     soft  0.0   1.0    0.5     text_only  logit"
  "D3_hard_shuf_logit    hard  0.0   1.0    0.5     shuffle    logit"
  "D4_hard_txt_logit     hard  0.0   1.0    0.5     text_only  logit"
  "D5_soft_shuf_logprob  soft  0.0   10.0   0.5     shuffle    logprob"
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
  echo "[$(date +%H:%M:%S)] Phase D tag=${tag}"
  echo "  veto=${veto} tau=${tau} beta=${beta} lambda=${lam} drop=${drop} gain_type=${gain_type}"
  echo "  out=${OUT_DIR}"
  echo "  debug=${MMADA_CV_DEBUG_DIR}"
  if [[ -n "${MMADA_INDICES:-}" ]]; then
    echo "  indices=${MMADA_INDICES}"
  fi
  echo "========================================"
  "${PYTHON}" run.py \
    --data LLaVABench \
    --model MMaDA-MixCoT-CV-DCD \
    --work-dir "${OUT_DIR}" \
    "$@"
done

echo ""
echo "[$(date +%H:%M:%S)] Phase D complete."
echo "  base dir:  ./outputs/cvdcd_sweep/${BASE_RUN_ID}"
echo "  debug dir: ${DEBUG_ROOT}"
echo ""
echo "Analysis suggested:"
echo "  ${PYTHON} attention_analysis/summarize_cv_dcd_phase1.py \\"
echo "    --sweep-dir ./outputs/cvdcd_sweep/${BASE_RUN_ID} \\"
echo "    --baseline outputs/MMaDA-MixCoT-DCD-DualCache/T20260613_G"
