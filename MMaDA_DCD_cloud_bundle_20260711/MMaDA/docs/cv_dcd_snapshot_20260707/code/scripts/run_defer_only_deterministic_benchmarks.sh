#!/bin/bash
# Deterministic-scoring benchmarks: bypass LLaVABench GPT noise
#
# Motivation
# ----------
# The E0-parity run (2026-07-07) proved that LLaVABench GPT-4 scoring has
# ~0.9 point of noise on 60 samples (E0 == B0 byte-identical predictions
# but scored 31.8 vs 30.9). This noise dominates the ~0.3-0.6 point
# intervention effect of defer_only (E4, E6).
#
# To determine whether defer_only has ANY real effect, we run on
# deterministically-scored benchmarks:
#   - ScienceQA_VAL: pure rule-based (exact_matching), zero GPT noise
#   - MathVision_MINI: GPT only used for answer extraction, is_equal for
#     final scoring — much lower noise than LLaVABench
#
# We skip E0 because head-to-head proved E0 == B0 byte-identical (60/60).
#
# Configs (3 total):
#   B0_plain_dcd            decode_strategy=dcd (true plain DCD baseline)
#   E4_defer_l0.5_t-3.0     cv_dcd, defer_only, lambda=0.5, tau=-3.0
#   E6_defer_l1.0_t-3.0     cv_dcd, defer_only, lambda=1.0, tau=-3.0
#
# Datasets (sequential, chosen from LMUData/):
#   MathVision_MINI: 1844 samples total; use MMADA_INDICES to subsample
#                    to ~200 for tractable runtime (~1.5h per config)
#   ScienceQA_VAL:   11567 samples total; subsample to ~200 (~1h per config)
#
# Total expected runtime: 3 configs * 2 datasets * ~1.5h = ~9h on H100.
#
# Usage:
#   # Small sanity first (60 samples each):
#   MMADA_SAMPLE_N=60 bash scripts/run_defer_only_deterministic_benchmarks.sh
#
#   # Full 200 samples each (~9h):
#   bash scripts/run_defer_only_deterministic_benchmarks.sh
#
# Success criteria:
#   1. On ScienceQA (zero noise), if E4/E6 accuracy > B0 by any amount,
#      defer_only has a real effect. If equal, defer_only is null.
#   2. On MathVision, similar interpretation but allow ~0.5pp noise budget.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VLMEVAL_ROOT}"

PYTHON="${PYTHON:-/home/user/anaconda3/envs/mmada/bin/python}"
SAMPLE_N="${MMADA_SAMPLE_N:-200}"       # samples per dataset
BASE_RUN_ID="${MMADA_RUN_ID:-defer_only_det_bench_$(date +%Y%m%d_%H%M%S)}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU status ==="
  nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader 2>/dev/null || true
  BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "${BUSY}" != "0" ]; then
    echo "WARNING: ${BUSY} process(es) using GPU."
  fi
  echo ""
fi

# Sub-sample first SAMPLE_N indices for a tractable run.
INDICES=$(seq 0 $((SAMPLE_N - 1)) | tr '\n' ',' | sed 's/,$//')
export MMADA_INDICES="${INDICES}"
echo "Using MMADA_INDICES: first ${SAMPLE_N} samples"

# Common defer_only knobs (only consumed when cv_mode==defer_only).
export MMADA_CV_ALPHA="${MMADA_CV_ALPHA:-0.1}"
export MMADA_CV_STRIDE="${MMADA_CV_STRIDE:-1}"
export MMADA_CV_RETURN_DEBUG="${MMADA_CV_RETURN_DEBUG:-0}"    # off to save time
export MMADA_CV_DROP="text_only"
export MMADA_DEFER_VETO="hard"
export MMADA_DEFER_BETA="1.0"
export MMADA_DEFER_GAIN_TYPE="logit"

mkdir -p "./outputs/cvdcd_sweep/${BASE_RUN_ID}"

# tag                       strategy   cv_mode      lambda  tau
CONFIGS=(
  "B0_plain_dcd             dcd        off          0.0     0.0"
  "E4_defer_l0.5_t-3.0      cv_dcd     defer_only   0.5     -3.0"
  "E6_defer_l1.0_t-3.0      cv_dcd     defer_only   1.0     -3.0"
)

DATASETS=(
  "MathVision_MINI"
  "ScienceQA_VAL"
)

echo "=== Deterministic-benchmark sweep ==="
echo "  configs : 3 (B0, E4, E6)"
echo "  datasets: ${DATASETS[*]}"
echo "  samples : ${SAMPLE_N} per dataset"
echo ""

for dataset in "${DATASETS[@]}"; do
  echo "########################################"
  echo "#### Dataset: ${dataset}"
  echo "########################################"
  for cfg in "${CONFIGS[@]}"; do
    read -r tag strategy cv_mode lam tau <<< "${cfg}"
    OUT_DIR="./outputs/cvdcd_sweep/${BASE_RUN_ID}/${dataset}/${tag}"

    export MMADA_DECODE_STRATEGY="${strategy}"
    export MMADA_CV_MODE="${cv_mode}"
    export MMADA_DEFER_TAU="${tau}"
    export MMADA_CV_LAMBDA="${lam}"

    echo "----------------------------------------"
    echo "[$(date +%H:%M:%S)] ${dataset} / ${tag}"
    echo "  strategy=${strategy}  cv_mode=${cv_mode}  lambda=${lam}  tau=${tau}"
    echo "  out=${OUT_DIR}"
    echo "----------------------------------------"
    "${PYTHON}" run.py \
      --data "${dataset}" \
      --model MMaDA-MixCoT-CV-DCD \
      --work-dir "${OUT_DIR}" \
      "$@" || {
        rc=$?
        echo "WARNING: run.py exited with ${rc} for ${dataset}/${tag}"
      }
    sleep 5
  done
done

echo ""
echo "[$(date +%H:%M:%S)] Deterministic-benchmark sweep complete."
echo "  base dir: ./outputs/cvdcd_sweep/${BASE_RUN_ID}"
echo ""
echo "Aggregate scores:"
echo "  find ./outputs/cvdcd_sweep/${BASE_RUN_ID} -name '*_score.csv' -o -name '*_acc.csv' | sort | xargs -I{} sh -c 'echo === {} ===; cat {}'"
