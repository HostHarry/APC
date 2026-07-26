#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MMADA_ROOT="$(cd "${VLMEVAL_ROOT}/../.." && pwd)"
cd "${VLMEVAL_ROOT}"

PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
SAMPLE_N="${MMADA_SAMPLE_N:-5000}"
CV_LAMBDA="${MMADA_CV_LAMBDA:-0.25}"
RUN_ID="${MMADA_RUN_ID:-chair_karpathy_5k_128_cd_apc_text_only_l${CV_LAMBDA}_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${VLMEVAL_ROOT}/outputs/${RUN_ID}"

export PYTHONPATH="${MMADA_ROOT}:${VLMEVAL_ROOT}"
export LMUData="${LMUData:-${VLMEVAL_ROOT}/LMUData}"
export MMADA_MODEL_PATH="${MMADA_MODEL_PATH:-/root/autodl-tmp/MMaDA-8B-MixCoT}"
export MMADA_TOKENIZER_PATH="${MMADA_TOKENIZER_PATH:-${MMADA_MODEL_PATH}}"
export MMADA_VQ_MODEL_PATH="${MMADA_VQ_MODEL_PATH:-/root/autodl-tmp/magvitv2}"
export CHAIR_COCO_ANN="${CHAIR_COCO_ANN:-/root/autodl-tmp/datasets/coco2014_karpathy/annotations/instances_val2014.json}"
export CHAIR_COCO_CAPS="${CHAIR_COCO_CAPS:-/root/autodl-tmp/datasets/coco2014_karpathy/annotations/captions_val2014.json}"
export CHAIR_STRICT_GT=0
export NLTK_DATA="${NLTK_DATA:-${VLMEVAL_ROOT}/nltk_data}"
export MMADA_SKIP_LOCALIZE=1
export MMADA_COLLECT_ATTENTION=0
export MMADA_CV_RETURN_DEBUG=0
export MMADA_DECODE_STRATEGY=cv_dcd
export MMADA_CV_MODE=cd_apc
export MMADA_CV_LAMBDA="${CV_LAMBDA}"
export MMADA_CV_ALPHA=0.1
export MMADA_CV_CLIP=4.0
export MMADA_CV_STRIDE=1
export MMADA_CV_DROP=text_only
export MMADA_CV_CONF_SOURCE=min_base_blended
export MMADA_CV_GATE_TAU=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for required in \
    "${LMUData}/CHAIR.tsv" \
    "${MMADA_MODEL_PATH}/config.json" \
    "${CHAIR_COCO_ANN}" \
    "${CHAIR_COCO_CAPS}"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: missing ${required}" >&2
        exit 1
    fi
done

CHAIR_ROWS="$("${PYTHON}" -c \
    "import pandas as pd; print(len(pd.read_csv('${LMUData}/CHAIR.tsv', sep='\\t', usecols=['index'])))")"
if (( SAMPLE_N < 1 || SAMPLE_N > CHAIR_ROWS )); then
    echo "ERROR: MMADA_SAMPLE_N=${SAMPLE_N}, but CHAIR.tsv has ${CHAIR_ROWS} rows" >&2
    exit 1
fi

unset MMADA_INDICES
if (( SAMPLE_N < CHAIR_ROWS )); then
    export MMADA_INDICES
    MMADA_INDICES="$(seq 0 $((SAMPLE_N - 1)) | tr '\n' ',' | sed 's/,$//')"
fi
mkdir -p "${RUN_ROOT}"

echo "CHAIR protocol: COCO2014 Karpathy Test"
echo "CHAIR samples: ${SAMPLE_N}/${CHAIR_ROWS}"
echo "CHAIR GT: instance masks + human captions"
echo "Generation: max_new_tokens=128, steps=128, block_length=64"
echo "CD-APC: drop=text_only, lambda=${CV_LAMBDA}, alpha=0.1, clip=4.0"
echo "Output: ${RUN_ROOT}"

"${PYTHON}" run.py \
    --data CHAIR \
    --model MMaDA-MixCoT-CV-DCD \
    --work-dir "${RUN_ROOT}"

echo "CHAIR CD-APC text-only completed: ${RUN_ROOT}"
