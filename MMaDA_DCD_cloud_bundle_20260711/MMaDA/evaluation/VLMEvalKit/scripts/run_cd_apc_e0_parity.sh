#!/bin/bash
# Phase 1: E0-parity smoke for the refactored cd_apc.
#
# Verifies that `cv_mode=cd_apc, causal_lambda=0.0` produces byte-identical
# predictions to plain DCD on real GPU (LLaVABench, 8 samples, ~5-8 min).
#
# Usage:
#   MMADA_SAMPLE_N=8 bash scripts/run_cd_apc_e0_parity.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VLMEVAL_ROOT}"

PYTHON="${PYTHON:-/home/user/anaconda3/envs/mmada/bin/python}"
SAMPLE_N="${MMADA_SAMPLE_N:-8}"
BASE_RUN_ID="${MMADA_RUN_ID:-cd_apc_e0_parity_$(date +%Y%m%d_%H%M%S)}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU status ==="
  nvidia-smi --query-gpu=name,memory.free,memory.used --format=csv,noheader
  echo ""
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
INDICES=$(seq 0 $((SAMPLE_N - 1)) | tr '\n' ',' | sed 's/,$//')
export MMADA_INDICES="${INDICES}"

# Fixed for both configs so decoding is deterministic w/ temp=0.
export MMADA_CV_ALPHA="${MMADA_CV_ALPHA:-0.1}"
export MMADA_CV_STRIDE="${MMADA_CV_STRIDE:-1}"
export MMADA_CV_DROP="${MMADA_CV_DROP:-shuffle}"   # irrelevant when λ=0
export MMADA_CV_CONF_SOURCE="${MMADA_CV_CONF_SOURCE:-min_base_blended}"
export MMADA_CV_GATE_TAU="${MMADA_CV_GATE_TAU:-0.0}"

mkdir -p "./outputs/cvdcd_sweep/${BASE_RUN_ID}"

# tag                    strategy   cv_mode      lambda
CONFIGS=(
  "B0_plain_dcd          dcd        off          0.0"
  "E0_cd_apc_l0          cv_dcd     cd_apc       0.0"
)

DATASET="LLaVABench"
echo "=== Phase 1: E0-parity smoke ==="
echo "  dataset : ${DATASET}"
echo "  samples : ${SAMPLE_N}"
echo "  out dir : ./outputs/cvdcd_sweep/${BASE_RUN_ID}"
echo ""

for cfg in "${CONFIGS[@]}"; do
  read -r tag strategy cv_mode lam <<< "${cfg}"
  OUT_DIR="./outputs/cvdcd_sweep/${BASE_RUN_ID}/${DATASET}/${tag}"

  export MMADA_DECODE_STRATEGY="${strategy}"
  export MMADA_CV_MODE="${cv_mode}"
  export MMADA_CV_LAMBDA="${lam}"

  echo "----------------------------------------"
  echo "[$(date +%H:%M:%S)] ${tag}"
  echo "  strategy=${strategy}  cv_mode=${cv_mode}  lambda=${lam}"
  echo "  out=${OUT_DIR}"
  echo "----------------------------------------"
  "${PYTHON}" run.py \
    --data "${DATASET}" \
    --model MMaDA-MixCoT-CV-DCD \
    --work-dir "${OUT_DIR}" \
    "$@" || {
      rc=$?
      echo "WARNING: run.py exited with ${rc} for ${tag}"
    }
  sleep 3
done

echo ""
echo "[$(date +%H:%M:%S)] Phase 1 complete. Compare predictions:"
echo ""

# Byte-diff between B0 and E0 predictions.
B0_XLSX=$(find ./outputs/cvdcd_sweep/${BASE_RUN_ID}/${DATASET}/B0_plain_dcd \
    -name '*_LLaVABench.xlsx' | head -1)
E0_XLSX=$(find ./outputs/cvdcd_sweep/${BASE_RUN_ID}/${DATASET}/E0_cd_apc_l0 \
    -name '*_LLaVABench.xlsx' | head -1)

if [ -z "${B0_XLSX}" ] || [ -z "${E0_XLSX}" ]; then
  echo "ERROR: could not find B0 or E0 prediction xlsx"
  exit 1
fi

"${PYTHON}" - <<PYEOF
import pandas as pd
b0 = pd.read_excel("${B0_XLSX}")
e0 = pd.read_excel("${E0_XLSX}")
n = len(b0)
if not (b0['prediction'].values == e0['prediction'].values).all():
    diffs = [(i, b0['prediction'][i], e0['prediction'][i])
             for i in range(n) if b0['prediction'][i] != e0['prediction'][i]]
    print(f"E0-PARITY FAIL: {len(diffs)}/{n} predictions differ")
    for idx, b, e in diffs[:3]:
        print(f"  [{idx}] B0={b!r}")
        print(f"  [{idx}] E0={e!r}")
    exit(1)
print(f"E0-PARITY PASS: {n}/{n} predictions byte-identical between B0 and E0 (cd_apc,λ=0)")
PYEOF
