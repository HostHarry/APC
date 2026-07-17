#!/bin/bash
# CHAIR hallucination benchmark (MSCOCO val2017 -> object hallucination)
# ----------------------------------------------------------------------
#
# Rationale
# ---------
# POPE + MME + MMBench_DEV_EN all showed defer_only is neutral-to-harmful,
# but they answer with a single Yes/No/MCQ token, so intervention on
# such answers is inevitably lossy. CHAIR is the classical *long-form*
# hallucination probe: prompt the model to describe the image freely,
# then count how many mentioned objects are NOT in the ground-truth
# instance annotations.
#
# If defer_only really cures visual-hallucination text, we expect CHAIRi
# and CHAIRs to drop under E4/E6 relative to B0. If not, this is the last
# chance to see any benefit before we retire the direction.
#
# Data prep
# ---------
# We do not download anything. The TSV is built offline from MSCOCO val2017:
#     python scripts/build_chair_tsv.py \
#         --coco-root /home/user/大模型/LLava/data/coco \
#         --n 200 --shuffle-seed 42
# which writes $LMUData/CHAIR.tsv. The COCO instances_val2017.json is used
# by the CHAIR scorer at eval time via CHAIR_COCO_ANN.
#
# Configs (mirrors run_defer_only_visual_benchmarks.sh):
#   B0_plain_dcd           decode_strategy=dcd (true plain DCD baseline)
#   E4_defer_l0.5_t-3.0    cv_dcd, defer_only, lambda=0.5, tau=-3.0
#   E6_defer_l1.0_t-3.0    cv_dcd, defer_only, lambda=1.0, tau=-3.0
#
# Usage
# -----
#   # Smoke (8 samples, ~5min/config)
#   MMADA_SAMPLE_N=8 bash scripts/run_defer_only_chair.sh
#
#   # Full 200-sample sweep (3 configs, ~3-4h on H100 total)
#   bash scripts/run_defer_only_chair.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${VLMEVAL_ROOT}"

export LMUData="${LMUData:-${VLMEVAL_ROOT}/LMUData}"
if [ ! -f "${LMUData}/CHAIR.tsv" ]; then
  echo "ERROR: ${LMUData}/CHAIR.tsv is missing."
  echo "Run: python scripts/build_chair_tsv.py --coco-root <coco_root>"
  exit 1
fi

# CHAIR scorer needs the COCO instance annotations at eval time.
export CHAIR_COCO_ANN="${CHAIR_COCO_ANN:-/home/user/大模型/LLava/data/coco/annotations/instances_val2017.json}"
if [ ! -f "${CHAIR_COCO_ANN}" ]; then
  echo "ERROR: instances_val2017.json not found at ${CHAIR_COCO_ANN}"
  echo "Set CHAIR_COCO_ANN=/path/to/instances_val2017.json"
  exit 1
fi

# Optional: canonical CHAIR uses caption-derived GT as well. If we can find
# captions_val2017.json (or CHAIR_COCO_CAPS is set), we go into
# "segments+captions" GT mode (matches LisaAnne/Hallucination behaviour).
# Set CHAIR_STRICT_GT=1 to force strict (segments-only) scoring.
export CHAIR_COCO_CAPS="${CHAIR_COCO_CAPS:-/home/user/大模型/LLava/data/coco/annotations/captions_val2017.json}"

# NLTK data (workspace-local, installed by:
#   NLTK_DATA=./nltk_data python -m nltk.downloader -d ./nltk_data \
#     punkt punkt_tab wordnet omw-1.4
# )
export NLTK_DATA="${NLTK_DATA:-${VLMEVAL_ROOT}/nltk_data}"

export MMADA_SKIP_LOCALIZE=1

PYTHON="${PYTHON:-/home/user/anaconda3/envs/mmada/bin/python}"
SAMPLE_N="${MMADA_SAMPLE_N:-200}"
BASE_RUN_ID="${MMADA_RUN_ID:-defer_only_chair_$(date +%Y%m%d_%H%M%S)}"
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

INDICES=$(seq 0 $((SAMPLE_N - 1)) | tr '\n' ',' | sed 's/,$//')
export MMADA_INDICES="${INDICES}"
echo "Using MMADA_INDICES: first ${SAMPLE_N} samples of CHAIR.tsv"

# Common defer_only knobs (mirror the visual bench sweep exactly).
export MMADA_CV_ALPHA="${MMADA_CV_ALPHA:-0.1}"
export MMADA_CV_STRIDE="${MMADA_CV_STRIDE:-1}"
export MMADA_CV_RETURN_DEBUG="${MMADA_CV_RETURN_DEBUG:-0}"
export MMADA_CV_DROP="text_only"
export MMADA_DEFER_VETO="hard"
export MMADA_DEFER_BETA="1.0"
export MMADA_DEFER_GAIN_TYPE="logit"

mkdir -p "./outputs/cvdcd_sweep/${BASE_RUN_ID}"

CONFIGS=(
  "B0_plain_dcd             dcd        off          0.0     0.0"
  "E4_defer_l0.5_t-3.0      cv_dcd     defer_only   0.5     -3.0"
  "E6_defer_l1.0_t-3.0      cv_dcd     defer_only   1.0     -3.0"
)

echo "=== CHAIR sweep ==="
echo "  configs : ${#CONFIGS[@]}"
echo "  dataset : CHAIR"
echo "  samples : ${SAMPLE_N}"
echo "  out dir : ./outputs/cvdcd_sweep/${BASE_RUN_ID}"
echo ""

for cfg in "${CONFIGS[@]}"; do
  read -r tag strategy cv_mode lam tau <<< "${cfg}"
  OUT_DIR="./outputs/cvdcd_sweep/${BASE_RUN_ID}/CHAIR/${tag}"

  export MMADA_DECODE_STRATEGY="${strategy}"
  export MMADA_CV_MODE="${cv_mode}"
  export MMADA_DEFER_TAU="${tau}"
  export MMADA_CV_LAMBDA="${lam}"

  echo "----------------------------------------"
  echo "[$(date +%H:%M:%S)] CHAIR / ${tag}"
  echo "  strategy=${strategy}  cv_mode=${cv_mode}  lambda=${lam}  tau=${tau}"
  echo "  out=${OUT_DIR}"
  echo "----------------------------------------"
  "${PYTHON}" run.py \
    --data CHAIR \
    --model MMaDA-MixCoT-CV-DCD \
    --work-dir "${OUT_DIR}" \
    "$@" || {
      rc=$?
      echo "WARNING: run.py exited with ${rc} for CHAIR/${tag}"
    }
  sleep 5
done

echo ""
echo "[$(date +%H:%M:%S)] CHAIR sweep complete."
echo "  base dir: ./outputs/cvdcd_sweep/${BASE_RUN_ID}"
echo ""
echo "Aggregate scores:"
echo "  find ./outputs/cvdcd_sweep/${BASE_RUN_ID} -type f -name '*_chair_score.csv' -exec sh -c 'echo === {} ===; cat {}; echo' \\;"
