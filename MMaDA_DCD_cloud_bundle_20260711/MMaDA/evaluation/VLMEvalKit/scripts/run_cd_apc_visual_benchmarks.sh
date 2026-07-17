#!/bin/bash
# Phase 3: Deterministic visual benchmarks for the refactored cd_apc.
#
# 200 samples × 3 datasets × 5 configs on POPE / MME / MMBench_DEV_EN.
# All three benchmarks use rule-based scoring (Y/N or MCQ), so noise is minimal.
#
# Prerequisite: TSVs prepared in LMUData (see run_defer_only_visual_benchmarks.sh
# header for details on hf_to_vlmeval_tsv.py).
#
# Usage:
#   bash scripts/run_cd_apc_visual_benchmarks.sh              # 200/dataset
#   MMADA_SAMPLE_N=20 bash scripts/run_cd_apc_visual_benchmarks.sh   # smoke
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VLMEVAL_ROOT}"

# --- LMUData: workspace-local -----------------------------------------------
export LMUData="${LMUData:-${VLMEVAL_ROOT}/LMUData}"
for tsv in POPE.tsv MME.tsv MMBench_DEV_EN.tsv; do
  if [ ! -f "${LMUData}/${tsv}" ]; then
    echo "ERROR: LMUData at ${LMUData} is missing ${tsv}."
    echo "Run scripts/hf_to_vlmeval_tsv.py first to prepare the TSVs."
    exit 1
  fi
done

export MMADA_SKIP_LOCALIZE=1
PYTHON="${PYTHON:-/home/user/anaconda3/envs/mmada/bin/python}"
SAMPLE_N="${MMADA_SAMPLE_N:-200}"
BASE_RUN_ID="${MMADA_RUN_ID:-cd_apc_visual_bench_$(date +%Y%m%d_%H%M%S)}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU status ==="
  nvidia-smi --query-gpu=name,memory.free,memory.used --format=csv,noheader
  echo ""
fi

INDICES=$(seq 0 $((SAMPLE_N - 1)) | tr '\n' ',' | sed 's/,$//')
export MMADA_INDICES="${INDICES}"
echo "Using MMADA_INDICES: first ${SAMPLE_N} samples"

# Common CD knobs.
export MMADA_CV_ALPHA="${MMADA_CV_ALPHA:-0.1}"
export MMADA_CV_STRIDE="${MMADA_CV_STRIDE:-1}"
export MMADA_CV_DROP="${MMADA_CV_DROP:-shuffle}"
export MMADA_CV_CONF_SOURCE="${MMADA_CV_CONF_SOURCE:-min_base_blended}"

mkdir -p "./outputs/cvdcd_sweep/${BASE_RUN_ID}"

# tag                       strategy   cv_mode   lambda   gate_tau
CONFIGS=(
  "B0_plain_dcd             dcd        off       0.0      0.0"
  "E0_cd_apc_l0             cv_dcd     cd_apc    0.0      0.0"
  "E1_cd_apc_default        cv_dcd     cd_apc    0.5      0.0"
  "E2_cd_apc_gate090        cv_dcd     cd_apc    0.5      0.9"
  "E3_cd_apc_gate095        cv_dcd     cd_apc    0.5      0.95"
)

DATASETS=(
  "POPE"
  "MME"
  "MMBench_DEV_EN"
)

echo "=== Phase 3: cd_apc visual-benchmark sweep ==="
echo "  configs : 5 (B0, E0, E1, E2, E3)"
echo "  datasets: ${DATASETS[*]}"
echo "  samples : ${SAMPLE_N} per dataset"
echo "  out dir : ./outputs/cvdcd_sweep/${BASE_RUN_ID}"
echo ""

for dataset in "${DATASETS[@]}"; do
  echo "########################################"
  echo "#### Dataset: ${dataset}"
  echo "########################################"
  for cfg in "${CONFIGS[@]}"; do
    read -r tag strategy cv_mode lam gate_tau <<< "${cfg}"
    OUT_DIR="./outputs/cvdcd_sweep/${BASE_RUN_ID}/${dataset}/${tag}"

    export MMADA_DECODE_STRATEGY="${strategy}"
    export MMADA_CV_MODE="${cv_mode}"
    export MMADA_CV_LAMBDA="${lam}"
    export MMADA_CV_GATE_TAU="${gate_tau}"

    echo "----------------------------------------"
    echo "[$(date +%H:%M:%S)] ${dataset} / ${tag}"
    echo "  strategy=${strategy}  cv_mode=${cv_mode}  lambda=${lam}  gate_tau=${gate_tau}"
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
echo "[$(date +%H:%M:%S)] Phase 3 complete."
echo ""
echo "Aggregate scores:"
echo "  find ./outputs/cvdcd_sweep/${BASE_RUN_ID} -type f \\( -name '*_score.*' -o -name '*_acc.*' -o -name '*_rating.*' \\) | sort | xargs -I{} sh -c 'echo === {} ===; cat {}; echo'"
