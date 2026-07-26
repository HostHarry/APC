#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMEVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MMADA_ROOT="$(cd "${VLMEVAL_ROOT}/../.." && pwd)"
cd "${VLMEVAL_ROOT}"

PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
DATASET="${M3COT_DATASET:-M3CoT_COT}"
MODEL_NAME="${MMADA_MODEL_NAME:-MMaDA-MixCoT}"
SAMPLE_N="${MMADA_SAMPLE_N:-0}"
RUN_ID="${MMADA_RUN_ID:-m3cot_parallel2_${DATASET}_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${VLMEVAL_ROOT}/outputs/${RUN_ID}"

case "${DATASET}" in
    M3CoT|M3CoT_COT|M3CoT_DIRECT) ;;
    *)
        echo "ERROR: unsupported M3COT_DATASET=${DATASET}" >&2
        exit 1
        ;;
esac

case "${MODEL_NAME}" in
    *CV-DCD*) DEFAULT_STRATEGY="cv_dcd" ;;
    *DCD*) DEFAULT_STRATEGY="dcd" ;;
    *) DEFAULT_STRATEGY="original" ;;
esac

export PYTHONPATH="${MMADA_ROOT}:${VLMEVAL_ROOT}"
export LMUData="${LMUData:-${VLMEVAL_ROOT}/LMUData}"
export MMADA_MODEL_PATH="${MMADA_MODEL_PATH:-/root/autodl-tmp/MMaDA-8B-MixCoT}"
export MMADA_TOKENIZER_PATH="${MMADA_TOKENIZER_PATH:-${MMADA_MODEL_PATH}}"
export MMADA_VQ_MODEL_PATH="${MMADA_VQ_MODEL_PATH:-/root/autodl-tmp/magvitv2}"
export MMADA_DECODE_STRATEGY="${MMADA_DECODE_STRATEGY:-${DEFAULT_STRATEGY}}"
export MMADA_SKIP_LOCALIZE=1
export MMADA_COLLECT_ATTENTION=0
export MMADA_CV_RETURN_DEBUG=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

for required in \
    "${LMUData}/M3CoT.tsv" \
    "${MMADA_MODEL_PATH}/config.json" \
    "${MMADA_VQ_MODEL_PATH}"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: missing ${required}" >&2
        exit 1
    fi
done

mkdir -p "${RUN_ROOT}/indices"

read -r TOTAL_ROWS EFFECTIVE_N < <(
    "${PYTHON}" - "${LMUData}/M3CoT.tsv" "${SAMPLE_N}" "${RUN_ROOT}/indices" <<'PY'
import os
import sys

import pandas as pd

tsv_path, sample_n, output_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
indices = pd.read_csv(tsv_path, sep='\t', usecols=['index'])['index'].tolist()
total = len(indices)
effective = total if sample_n == 0 else sample_n
if effective < 2 or effective > total:
    raise SystemExit(
        f'MMADA_SAMPLE_N must be 0 or in [2, {total}], got {sample_n}.')
indices = indices[:effective]
for worker in range(2):
    shard = indices[worker::2]
    pd.DataFrame({'index': shard}).to_csv(
        os.path.join(output_dir, f'worker{worker}.csv'), index=False)
print(total, effective)
PY
)

echo "M3CoT dataset: ${DATASET}"
echo "Model: ${MODEL_NAME} (${MMADA_DECODE_STRATEGY})"
echo "Samples: ${EFFECTIVE_N}/${TOTAL_ROWS}"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Launching two independent MMaDA workers"
echo "Output: ${RUN_ROOT}"

pids=()
for worker in 0 1; do
    worker_root="${RUN_ROOT}/worker${worker}"
    worker_log="${RUN_ROOT}/worker${worker}.log"
    index_file="${RUN_ROOT}/indices/worker${worker}.csv"
    mkdir -p "${worker_root}"
    (
        unset MMADA_INDICES
        export MMADA_INDEX_FILE="${index_file}"
        export MMADA_RUN_ID="${RUN_ID}_worker${worker}"
        exec "${PYTHON}" run.py \
            --data "${DATASET}" \
            --model "${MODEL_NAME}" \
            --work-dir "${worker_root}" \
            --mode infer
    ) >"${worker_log}" 2>&1 &
    pids+=("$!")
    echo "worker${worker}: pid=${pids[worker]}, log=${worker_log}"
done

terminate_workers() {
    for pid in "${pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap terminate_workers INT TERM

failed=0
for worker in 0 1; do
    if ! wait "${pids[worker]}"; then
        echo "ERROR: worker${worker} failed; inspect ${RUN_ROOT}/worker${worker}.log" >&2
        failed=1
    fi
done
trap - INT TERM

if (( failed )); then
    exit 1
fi

"${PYTHON}" "${SCRIPT_DIR}/merge_m3cot_parallel.py" \
    --run-root "${RUN_ROOT}" \
    --dataset "${DATASET}" \
    --model "${MODEL_NAME}" \
    --expected "${EFFECTIVE_N}"

echo "M3CoT parallel run completed: ${RUN_ROOT}"
