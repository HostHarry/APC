#!/bin/bash
# Phase 4: CHAIR long-form caption hallucination for the refactored cd_apc.
#
# 200 MSCOCO val2017 images × 3 configs (B0, best cd_apc from Phase 3, and
# ungated cd_apc default). Paired forward makes this ~5-6h.
#
# Prerequisite: CHAIR.tsv built via scripts/build_chair_tsv.py.
#
# Usage:
#   bash scripts/run_cd_apc_chair.sh               # 200 samples
#   MMADA_SAMPLE_N=20 bash scripts/run_cd_apc_chair.sh    # smoke
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VLMEVAL_ROOT}"

# --- CHAIR data / NLTK -------------------------------------------------------
export LMUData="${LMUData:-${VLMEVAL_ROOT}/LMUData}"
if [ ! -f "${LMUData}/CHAIR.tsv" ]; then
  echo "ERROR: LMUData at ${LMUData} is missing CHAIR.tsv."
  echo "Run scripts/build_chair_tsv.py first."
  exit 1
fi
export CHAIR_COCO_ANN="${CHAIR_COCO_ANN:-/home/user/大模型/LLava/data/coco/annotations/instances_val2017.json}"
export CHAIR_COCO_CAPS="${CHAIR_COCO_CAPS:-/home/user/大模型/LLava/data/coco/annotations/captions_val2017.json}"
export NLTK_DATA="${NLTK_DATA:-${VLMEVAL_ROOT}/nltk_data}"
export MMADA_SKIP_LOCALIZE=1

for req in "${CHAIR_COCO_ANN}" "${CHAIR_COCO_CAPS}"; do
  if [ ! -f "${req}" ]; then
    echo "ERROR: missing ${req}"
    exit 1
  fi
done

PYTHON="${PYTHON:-/home/user/anaconda3/envs/mmada/bin/python}"
SAMPLE_N="${MMADA_SAMPLE_N:-200}"
BASE_RUN_ID="${MMADA_RUN_ID:-cd_apc_chair_$(date +%Y%m%d_%H%M%S)}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU status ==="
  nvidia-smi --query-gpu=name,memory.free,memory.used --format=csv,noheader || true
  echo ""
fi

INDICES=$(seq 0 $((SAMPLE_N - 1)) | tr '\n' ',' | sed 's/,$//')
export MMADA_INDICES="${INDICES}"

# Common CD knobs.
export MMADA_CV_ALPHA="${MMADA_CV_ALPHA:-0.1}"
export MMADA_CV_STRIDE="${MMADA_CV_STRIDE:-1}"
export MMADA_CV_DROP="${MMADA_CV_DROP:-shuffle}"
export MMADA_CV_CONF_SOURCE="${MMADA_CV_CONF_SOURCE:-min_base_blended}"

mkdir -p "./outputs/cvdcd_sweep/${BASE_RUN_ID}"

# tag                       strategy   cv_mode   lambda   gate_tau
# NOTE: dropped E2 (gate=0.9) - Phase 2 showed the gate essentially never
# triggers on typical model confidences, so it produces predictions identical
# to E1 while doubling GPU time. Re-add if you want to verify long-form
# generation, where per-token confidences might be lower.
CONFIGS=(
  "B0_plain_dcd             dcd        off       0.0      0.0"
  "E1_cd_apc_default        cv_dcd     cd_apc    0.5      0.0"
)

DATASET="CHAIR"
echo "=== Phase 4: cd_apc CHAIR sweep ==="
echo "  configs : 3 (B0, E1, E2)"
echo "  samples : ${SAMPLE_N}"
echo "  out dir : ./outputs/cvdcd_sweep/${BASE_RUN_ID}"
echo ""

for cfg in "${CONFIGS[@]}"; do
  read -r tag strategy cv_mode lam gate_tau <<< "${cfg}"
  OUT_DIR="./outputs/cvdcd_sweep/${BASE_RUN_ID}/${DATASET}/${tag}"

  # Idempotent skip: if a completed xlsx already exists, don't rerun.
  DONE_XLSX="${OUT_DIR}/MMaDA-MixCoT-CV-DCD/MMaDA-MixCoT-CV-DCD_${DATASET}.xlsx"
  if [ -f "${DONE_XLSX}" ]; then
    echo "[$(date +%H:%M:%S)] SKIP ${DATASET}/${tag} (xlsx already present)"
    continue
  fi

  export MMADA_DECODE_STRATEGY="${strategy}"
  export MMADA_CV_MODE="${cv_mode}"
  export MMADA_CV_LAMBDA="${lam}"
  export MMADA_CV_GATE_TAU="${gate_tau}"

  echo "----------------------------------------"
  echo "[$(date +%H:%M:%S)] ${DATASET} / ${tag}"
  echo "  strategy=${strategy}  cv_mode=${cv_mode}  lambda=${lam}  gate_tau=${gate_tau}"
  echo "  out=${OUT_DIR}"
  echo "----------------------------------------"
  "${PYTHON}" run.py \
    --data "${DATASET}" \
    --model MMaDA-MixCoT-CV-DCD \
    --work-dir "${OUT_DIR}" \
    "$@" || {
      rc=$?
      echo "WARNING: run.py exited with ${rc} for ${DATASET}/${tag}"
    }
  sleep 5
done

echo ""
echo "[$(date +%H:%M:%S)] Phase 4 complete."
echo ""
echo "Aggregate CHAIR scores:"
echo "  find ./outputs/cvdcd_sweep/${BASE_RUN_ID} -name '*_chair_score.csv' | sort"
