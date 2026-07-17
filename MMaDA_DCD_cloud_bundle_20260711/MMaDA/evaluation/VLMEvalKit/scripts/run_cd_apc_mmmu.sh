#!/bin/bash
# Phase 5: cd_apc refactor on MMMU_DEV_VAL (multi-discipline MCQ).
#
# 200 samples × 2 configs (B0 plain DCD, E1 cd_apc λ=0.5).
# B0 ~40m + E1 ~1h15 ≈ 2h total.
#
# TSV built by scripts/hf_to_vlmeval_tsv.py from lmms-lab/MMMU parquet.
# 667 rows total after (single-image ∩ multi-choice ∩ 4-option) filter,
# shuffled with seed=42 so MMADA_INDICES=0..199 gives a diverse subject mix.
#
# Usage:
#   bash scripts/run_cd_apc_mmmu.sh                    # 200 samples
#   MMADA_SAMPLE_N=20 bash scripts/run_cd_apc_mmmu.sh  # smoke
#
# The script is idempotent: SKIPs (dataset, config) pairs whose xlsx already
# exists in the RUN_ID directory, so it can be re-launched safely after a
# crash / GPU driver recovery.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VLMEVAL_ROOT}"

export LMUData="${LMUData:-${VLMEVAL_ROOT}/LMUData}"
if [ ! -f "${LMUData}/MMMU_DEV_VAL.tsv" ]; then
  echo "ERROR: ${LMUData}/MMMU_DEV_VAL.tsv missing."
  echo "Build via: python scripts/hf_to_vlmeval_tsv.py --dataset MMMU_DEV_VAL \\"
  echo "             --parquet /tmp/hfd_mmmu/dev.parquet /tmp/hfd_mmmu/val.parquet \\"
  echo "             --shuffle-seed 42 --out-dir ${LMUData}"
  exit 1
fi
# Skip VLMEval's LOCALIZE (multiprocessing) - TSV is 60 MB, easily loads directly.
export MMADA_SKIP_LOCALIZE=1

PYTHON="${PYTHON:-/home/user/anaconda3/envs/mmada/bin/python}"
SAMPLE_N="${MMADA_SAMPLE_N:-200}"
BASE_RUN_ID="${MMADA_RUN_ID:-cd_apc_mmmu_$(date +%Y%m%d_%H%M%S)}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
INDICES=$(seq 0 $((SAMPLE_N - 1)) | tr '\n' ',' | sed 's/,$//')
export MMADA_INDICES="${INDICES}"

# Common CD knobs (align with Phase 3 defaults, so results directly compare).
export MMADA_CV_ALPHA="${MMADA_CV_ALPHA:-0.1}"
export MMADA_CV_STRIDE="${MMADA_CV_STRIDE:-1}"
export MMADA_CV_DROP="${MMADA_CV_DROP:-shuffle}"
export MMADA_CV_CONF_SOURCE="${MMADA_CV_CONF_SOURCE:-min_base_blended}"

CONFIGS=(
  "B0_plain_dcd             dcd        off       0.0      0.0"
  "E1_cd_apc_default        cv_dcd     cd_apc    0.5      0.0"
)
DATASET="MMMU_DEV_VAL"

echo "=== cd_apc / MMMU_DEV_VAL sweep ==="
echo "  run_id : ${BASE_RUN_ID}"
echo "  samples: ${SAMPLE_N}"
echo ""

RUN_ROOT="./outputs/cvdcd_sweep/${BASE_RUN_ID}"
mkdir -p "${RUN_ROOT}"

check_gpu_ready() {
  for attempt in 1 2 3; do
    if OUT=$(nvidia-smi --query-gpu=temperature.gpu,memory.free --format=csv,noheader 2>/dev/null); then
      TEMP=$(echo "$OUT" | cut -d, -f1 | xargs)
      MEM=$(echo "$OUT" | cut -d, -f2 | xargs)
      echo "  GPU ready: temp=${TEMP}°C  free=${MEM}"
      if [ "${TEMP:-0}" -gt 80 ] 2>/dev/null; then
        echo "  GPU hot (${TEMP}°C), sleeping 120s ..."
        sleep 120
      fi
      return 0
    fi
    echo "  attempt $attempt: nvidia-smi failed, sleeping 5s ..."
    sleep 5
  done
  echo "  WARNING: nvidia-smi unresponsive - assuming CUDA still works (NVML query lag)"
  return 0
}

for cfg in "${CONFIGS[@]}"; do
  read -r tag strategy cv_mode lam gate_tau <<< "${cfg}"
  OUT_DIR="${RUN_ROOT}/${DATASET}/${tag}"

  if find "${OUT_DIR}" -name "*_${DATASET}.xlsx" 2>/dev/null | grep -q .; then
    echo "----------------------------------------"
    echo "[$(date +%H:%M:%S)] SKIP ${DATASET}/${tag}   (xlsx already present)"
    continue
  fi

  echo "----------------------------------------"
  echo "[$(date +%H:%M:%S)] RUN  ${DATASET}/${tag}"
  echo "  strategy=${strategy}  cv_mode=${cv_mode}  lambda=${lam}  gate_tau=${gate_tau}"
  echo "  out=${OUT_DIR}"
  check_gpu_ready

  export MMADA_DECODE_STRATEGY="${strategy}"
  export MMADA_CV_MODE="${cv_mode}"
  export MMADA_CV_LAMBDA="${lam}"
  export MMADA_CV_GATE_TAU="${gate_tau}"

  "${PYTHON}" run.py \
    --data "${DATASET}" \
    --model MMaDA-MixCoT-CV-DCD \
    --work-dir "${OUT_DIR}" \
    "$@" || {
      rc=$?
      echo "WARNING: run.py exited with ${rc} for ${DATASET}/${tag}"
    }

  echo "[$(date +%H:%M:%S)] cool-down 30s ..."
  sleep 30
done

echo ""
echo "[$(date +%H:%M:%S)] MMMU sweep complete."
echo ""
echo "Aggregate:"
echo "  find ${RUN_ROOT} -name '*_MMMU_DEV_VAL.xlsx' | sort"
