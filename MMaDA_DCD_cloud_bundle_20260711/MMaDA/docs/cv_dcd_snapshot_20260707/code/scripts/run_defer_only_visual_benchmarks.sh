#!/bin/bash
# Visual-focused benchmarks (POPE, MME) for defer_only.
#
# Motivation
# ----------
# Previous deterministic runs on ScienceQA / MathVision showed defer_only
# is either neutral (ScienceQA -0.5pp, image-independent data) or hurts
# (MathVision -2.3pp, math+vision). The remaining hypothesis is that
# defer_only might help on *hallucination-focused* Y/N benchmarks where
# the language prior can hallucinate objects that aren't in the image.
# POPE and MME (perception subset) are the canonical tests for that.
#
# Data preparation
# ----------------
# Our sandbox cannot reach opencompass.openxlab.space (VLMEval's default TSV
# host), so we mirror lmms-lab HF parquets and re-encode via
# `scripts/hf_to_vlmeval_tsv.py --shuffle-seed 42`. All rows are globally
# shuffled with a fixed seed so `MMADA_INDICES=0..N-1` selects a
# category-stratified subset instead of a single-category block.
#
# Full coverage (all splits/categories):
#   POPE           : 9000 rows across random / popular / adversarial
#   MME            : 2374 rows across 14 categories (existence, count,
#                    OCR, code_reasoning, celebrity, ...)
#   MMBench_DEV_EN : 4377 rows across 20 fine-grained categories
#                    (6 l2-categories: coarse_perception,
#                    finegrained_perception, attribute_reasoning, ...)
#
# Configs (identical to run_defer_only_deterministic_benchmarks.sh):
#   B0_plain_dcd            decode_strategy=dcd (true plain DCD baseline)
#   E4_defer_l0.5_t-3.0     cv_dcd, defer_only, lambda=0.5, tau=-3.0
#   E6_defer_l1.0_t-3.0     cv_dcd, defer_only, lambda=1.0, tau=-3.0
#
# Usage
# -----
#   # Smoke first (20 samples each):
#   MMADA_SAMPLE_N=20 bash scripts/run_defer_only_visual_benchmarks.sh
#
#   # Full 200-sample run per dataset (~1h/config on H100):
#   bash scripts/run_defer_only_visual_benchmarks.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VLMEVAL_ROOT}"

# --- LMUData: workspace-local so we can write predictions/*.xlsx --------------
export LMUData="${LMUData:-${VLMEVAL_ROOT}/LMUData}"
for tsv in POPE.tsv MME.tsv MMBench_DEV_EN.tsv; do
  if [ ! -f "${LMUData}/${tsv}" ]; then
    echo "ERROR: LMUData at ${LMUData} is missing ${tsv}."
    echo "Run scripts/hf_to_vlmeval_tsv.py first to prepare the TSVs."
    exit 1
  fi
done

# Skip VLMEval's LOCALIZE step (needs mp.Pool which is blocked in sandbox);
# with TSVs >1 GiB we would otherwise fail with a semaphore PermissionError.
export MMADA_SKIP_LOCALIZE=1

PYTHON="${PYTHON:-/home/user/anaconda3/envs/mmada/bin/python}"
SAMPLE_N="${MMADA_SAMPLE_N:-200}"       # samples per dataset
BASE_RUN_ID="${MMADA_RUN_ID:-defer_only_visual_bench_$(date +%Y%m%d_%H%M%S)}"

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

# Subsample first SAMPLE_N indices.
INDICES=$(seq 0 $((SAMPLE_N - 1)) | tr '\n' ',' | sed 's/,$//')
export MMADA_INDICES="${INDICES}"
echo "Using MMADA_INDICES: first ${SAMPLE_N} samples"

# Common defer_only knobs (only consumed when cv_mode==defer_only).
export MMADA_CV_ALPHA="${MMADA_CV_ALPHA:-0.1}"
export MMADA_CV_STRIDE="${MMADA_CV_STRIDE:-1}"
export MMADA_CV_RETURN_DEBUG="${MMADA_CV_RETURN_DEBUG:-0}"
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
  "POPE"
  "MME"
  "MMBench_DEV_EN"
)

echo "=== Visual-benchmark sweep ==="
echo "  configs : 3 (B0, E4, E6)"
echo "  datasets: ${DATASETS[*]}"
echo "  samples : ${SAMPLE_N} per dataset"
echo "  out dir : ./outputs/cvdcd_sweep/${BASE_RUN_ID}"
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
echo "[$(date +%H:%M:%S)] Visual-benchmark sweep complete."
echo "  base dir: ./outputs/cvdcd_sweep/${BASE_RUN_ID}"
echo ""
echo "Aggregate scores:"
echo "  find ./outputs/cvdcd_sweep/${BASE_RUN_ID} -type f \\( -name '*_score.*' -o -name '*_acc.*' -o -name '*_rating.*' \\) | sort | xargs -I{} sh -c 'echo === {} ===; cat {}; echo'"
