#!/bin/bash
# Phase 2: LLaVABench smoke on cd_apc variants.
#
# 20 samples × 5 configs. Verifies:
#   1. E0 parity holds on real GPU (must match B0 byte-identically).
#   2. E1..E3 diverge from B0 (proves refactor wired end-to-end).
#   3. Early GPT-scored feel + NFE cost per sample.
#
# Usage:
#   MMADA_SAMPLE_N=20 bash scripts/run_cd_apc_llavabench_smoke.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VLMEVAL_ROOT}"

PYTHON="${PYTHON:-/home/user/anaconda3/envs/mmada/bin/python}"
SAMPLE_N="${MMADA_SAMPLE_N:-20}"
BASE_RUN_ID="${MMADA_RUN_ID:-cd_apc_smoke_$(date +%Y%m%d_%H%M%S)}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU status ==="
  nvidia-smi --query-gpu=name,memory.free,memory.used --format=csv,noheader
  echo ""
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
INDICES=$(seq 0 $((SAMPLE_N - 1)) | tr '\n' ',' | sed 's/,$//')
export MMADA_INDICES="${INDICES}"

# Common CD knobs.
export MMADA_CV_ALPHA="${MMADA_CV_ALPHA:-0.1}"
export MMADA_CV_STRIDE="${MMADA_CV_STRIDE:-1}"
export MMADA_CV_DROP="${MMADA_CV_DROP:-shuffle}"    # v3.2 winner
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

DATASET="LLaVABench"
echo "=== Phase 2: LLaVABench smoke on cd_apc ==="
echo "  configs : 5 (B0, E0, E1, E2, E3)"
echo "  samples : ${SAMPLE_N}"
echo "  out dir : ./outputs/cvdcd_sweep/${BASE_RUN_ID}"
echo ""

for cfg in "${CONFIGS[@]}"; do
  read -r tag strategy cv_mode lam gate_tau <<< "${cfg}"
  OUT_DIR="./outputs/cvdcd_sweep/${BASE_RUN_ID}/${DATASET}/${tag}"

  export MMADA_DECODE_STRATEGY="${strategy}"
  export MMADA_CV_MODE="${cv_mode}"
  export MMADA_CV_LAMBDA="${lam}"
  export MMADA_CV_GATE_TAU="${gate_tau}"

  echo "----------------------------------------"
  echo "[$(date +%H:%M:%S)] ${tag}"
  echo "  strategy=${strategy}  cv_mode=${cv_mode}  lambda=${lam}  gate_tau=${gate_tau}"
  echo "  cv_conf_source=${MMADA_CV_CONF_SOURCE}  cv_alpha=${MMADA_CV_ALPHA}  drop=${MMADA_CV_DROP}"
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
  sleep 5
done

echo ""
echo "[$(date +%H:%M:%S)] Phase 2 complete."
echo ""

# Cross-config byte-diff analysis
"${PYTHON}" - <<PYEOF
import glob, pandas as pd
base = f"./outputs/cvdcd_sweep/${BASE_RUN_ID}/${DATASET}"
tags = ["B0_plain_dcd", "E0_cd_apc_l0", "E1_cd_apc_default",
        "E2_cd_apc_gate090", "E3_cd_apc_gate095"]
preds = {}
for tag in tags:
    xlsx = glob.glob(f"{base}/{tag}/**/*_LLaVABench.xlsx", recursive=True)
    if xlsx:
        df = pd.read_excel(xlsx[0])
        preds[tag] = df['prediction'].tolist()
        print(f"  {tag}: {len(preds[tag])} predictions")
    else:
        print(f"  {tag}: MISSING")
print()
if "B0_plain_dcd" in preds:
    b0 = preds["B0_plain_dcd"]
    for tag in tags[1:]:
        if tag not in preds:
            continue
        n = min(len(b0), len(preds[tag]))
        diffs = sum(1 for i in range(n) if b0[i] != preds[tag][i])
        status = "PARITY-OK" if diffs == 0 else "DIVERGE"
        print(f"  {status}: {tag} has {diffs}/{n} predictions different from B0")
PYEOF
