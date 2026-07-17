#!/bin/bash
# Resume-mode Phase 3: only run (dataset, config) pairs whose xlsx is missing
# in the target RUN_ID directory. Reduced to B0 + E1 only (Phase 2 already
# showed E0 ≡ B0, and E2/E3 gates don't fire in this token distribution).
#
# Overheat protection: 30 s cooldown between configs.
#
# Usage:
#   MMADA_RUN_ID=cd_apc_visual_bench_20260710_012143 \
#     bash scripts/run_cd_apc_visual_resume.sh
#
# Set MMADA_SAMPLE_N=20 for smoke; default 200.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VLMEVAL_ROOT}"

export LMUData="${LMUData:-${VLMEVAL_ROOT}/LMUData}"
for tsv in POPE.tsv MME.tsv MMBench_DEV_EN.tsv; do
  if [ ! -f "${LMUData}/${tsv}" ]; then
    echo "ERROR: LMUData missing ${tsv}"; exit 1
  fi
done
export MMADA_SKIP_LOCALIZE=1

PYTHON="${PYTHON:-/home/user/anaconda3/envs/mmada/bin/python}"
SAMPLE_N="${MMADA_SAMPLE_N:-200}"
BASE_RUN_ID="${MMADA_RUN_ID:?please set MMADA_RUN_ID to an existing run dir}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
INDICES=$(seq 0 $((SAMPLE_N - 1)) | tr '\n' ',' | sed 's/,$//')
export MMADA_INDICES="${INDICES}"

# Common CD knobs (Phase 3 defaults).
export MMADA_CV_ALPHA="${MMADA_CV_ALPHA:-0.1}"
export MMADA_CV_STRIDE="${MMADA_CV_STRIDE:-1}"
export MMADA_CV_DROP="${MMADA_CV_DROP:-shuffle}"
export MMADA_CV_CONF_SOURCE="${MMADA_CV_CONF_SOURCE:-min_base_blended}"

# Reduced: B0 + E1 only
CONFIGS=(
  "B0_plain_dcd             dcd        off       0.0      0.0"
  "E1_cd_apc_default        cv_dcd     cd_apc    0.5      0.0"
)
DATASETS=( "POPE" "MME" "MMBench_DEV_EN" )

echo "=== Phase 3 RESUME · reduced B0+E1 ==="
echo "  run_id : ${BASE_RUN_ID}"
echo "  samples: ${SAMPLE_N}"
echo ""

RUN_ROOT="./outputs/cvdcd_sweep/${BASE_RUN_ID}"
[ -d "${RUN_ROOT}" ] || { echo "ERROR: ${RUN_ROOT} does not exist"; exit 1; }

check_gpu_ready() {
  # Try to recover the driver; wait until GPU responds; capture temperature.
  for attempt in 1 2 3; do
    if OUT=$(nvidia-smi --query-gpu=temperature.gpu,memory.free --format=csv,noheader 2>/dev/null); then
      TEMP=$(echo "$OUT" | cut -d, -f1 | xargs)
      MEM=$(echo "$OUT" | cut -d, -f2 | xargs)
      echo "  GPU ready: temp=${TEMP}°C  free=${MEM}"
      # Wait if hot (> 80 C)
      if [ "${TEMP:-0}" -gt 80 ] 2>/dev/null; then
        echo "  GPU hot (${TEMP}°C), sleeping 120s ..."
        sleep 120
      fi
      return 0
    fi
    echo "  attempt $attempt: nvidia-smi failed, sleeping 5s ..."
    sleep 5
  done
  echo "  ERROR: GPU driver not responding after 3 attempts"
  return 1
}

for dataset in "${DATASETS[@]}"; do
  echo ""
  echo "########################################"
  echo "#### Dataset: ${dataset}"
  echo "########################################"
  for cfg in "${CONFIGS[@]}"; do
    read -r tag strategy cv_mode lam gate_tau <<< "${cfg}"
    OUT_DIR="${RUN_ROOT}/${dataset}/${tag}"

    # Skip if any prediction xlsx already present in inner dir.
    if find "${OUT_DIR}" -name "*_${dataset}.xlsx" 2>/dev/null | grep -q .; then
      echo "----------------------------------------"
      echo "[$(date +%H:%M:%S)] SKIP ${dataset}/${tag}   (xlsx already present)"
      continue
    fi

    echo "----------------------------------------"
    echo "[$(date +%H:%M:%S)] RUN  ${dataset}/${tag}"
    echo "  strategy=${strategy}  cv_mode=${cv_mode}  lambda=${lam}  gate_tau=${gate_tau}"
    echo "  out=${OUT_DIR}"
    check_gpu_ready || { echo "GPU not ready, aborting"; exit 2; }

    export MMADA_DECODE_STRATEGY="${strategy}"
    export MMADA_CV_MODE="${cv_mode}"
    export MMADA_CV_LAMBDA="${lam}"
    export MMADA_CV_GATE_TAU="${gate_tau}"

    "${PYTHON}" run.py \
      --data "${dataset}" \
      --model MMaDA-MixCoT-CV-DCD \
      --work-dir "${OUT_DIR}" \
      "$@" || {
        rc=$?
        echo "WARNING: run.py exited with ${rc} for ${dataset}/${tag}"
      }

    echo "[$(date +%H:%M:%S)] cool-down 30s ..."
    sleep 30
  done
done

echo ""
echo "[$(date +%H:%M:%S)] Phase 3 RESUME complete."
