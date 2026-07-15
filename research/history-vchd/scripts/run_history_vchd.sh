#!/usr/bin/env bash
# History-VCHD (VCHD + Sparse History + CCAW) cloud sweep.
#
# What this runs:
#   * profile = history_ccaw
#     - decode_strategy=vchd
#     - MMADA_VCHD_HISTORY=1, MMADA_VCHD_CCAW=1
#     - dual-gate: tau_base=0.10, tau_contrast=0.90
#     - alpha=0.5 (CD-APC strength), beta=0.1 (APC keep threshold)
#     - mask_capacity=16, max_commit=16, ccaw_max_mask_capacity=64
#     - COLLECT_TRACE=1, RETURN_REPORT=1  (per-sample ledger JSON dumped)
#
# By default runs on both datasets sequentially. Override DATASETS to restrict.
#
# Usage:
#   bash scripts/run_history_vchd.sh
#   DATASETS="MMMU_DEV_VAL_FULL" bash scripts/run_history_vchd.sh
#   RUN_ID=my_run bash scripts/run_history_vchd.sh
#
# Env overrides:
#   PY / PYTHON     python interpreter
#   RUN_ID          output subdir tag (default: history_vchd_<timestamp>)
#   DATASETS        space-separated dataset list (default: both)
#   CUDA_VISIBLE_DEVICES  which GPU (default: 0)
#   MMADA_VCHD_MASK_CAPACITY / MAX_COMMIT / etc. — override VCHD knobs

set -uo pipefail
shopt -s globstar nullglob

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMADA_ROOT="${REPO_ROOT}/MMaDA"
VLMEVAL_ROOT="${MMADA_ROOT}/evaluation/VLMEvalKit"
cd "${VLMEVAL_ROOT}"

PYTHON="${PYTHON:-${PY:-python}}"
RUN_ID="${RUN_ID:-history_vchd_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${VLMEVAL_ROOT}/outputs/${RUN_ID}"
LOG_ROOT="${VLMEVAL_ROOT}/logs/${RUN_ID}"
STATUS_LOG="${LOG_ROOT}/status.log"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

export LMUData="${LMUData:-${VLMEVAL_ROOT}/LMUData}"
export MMADA_SKIP_LOCALIZE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${MMADA_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Prefer already-downloaded local weights on the current machine while
# retaining Hugging Face IDs as portable cloud fallbacks. All paths remain
# explicitly overridable by the caller.
LOCAL_MMADA_MODEL_PATH="${MMADA_LOCAL_MODEL_PATH:-/root/autodl-tmp/MMaDA-8B-MixCoT}"
LOCAL_MMADA_VQ_MODEL_PATH="${MMADA_LOCAL_VQ_MODEL_PATH:-/root/autodl-tmp/magvitv2}"
if [[ -z "${MMADA_MODEL_PATH:-}" && -d "${LOCAL_MMADA_MODEL_PATH}" ]]; then
  MMADA_MODEL_PATH="${LOCAL_MMADA_MODEL_PATH}"
fi
if [[ -z "${MMADA_VQ_MODEL_PATH:-}" && -d "${LOCAL_MMADA_VQ_MODEL_PATH}" ]]; then
  MMADA_VQ_MODEL_PATH="${LOCAL_MMADA_VQ_MODEL_PATH}"
fi
export MMADA_MODEL_PATH="${MMADA_MODEL_PATH:-Gen-Verse/MMaDA-8B-MixCoT}"
export MMADA_TOKENIZER_PATH="${MMADA_TOKENIZER_PATH:-${MMADA_MODEL_PATH}}"
export MMADA_VQ_MODEL_PATH="${MMADA_VQ_MODEL_PATH:-showlab/magvitv2}"

# --- History-VCHD profile ---
export MMADA_DECODE_STRATEGY=vchd
export MMADA_VCHD_TAU_BASE="${MMADA_VCHD_TAU_BASE:-0.10}"
export MMADA_VCHD_TAU_CONTRAST="${MMADA_VCHD_TAU_CONTRAST:-0.90}"
export MMADA_VCHD_MASK_CAPACITY="${MMADA_VCHD_MASK_CAPACITY:-16}"
export MMADA_VCHD_MAX_COMMIT="${MMADA_VCHD_MAX_COMMIT:-16}"
export MMADA_VCHD_FORCE_MATH_SDPA="${MMADA_VCHD_FORCE_MATH_SDPA:-1}"
export MMADA_VCHD_FALLBACK_TO_RAW="${MMADA_VCHD_FALLBACK_TO_RAW:-0}"
export MMADA_VCHD_HISTORY=1
export MMADA_VCHD_HISTORY_TOP_V="${MMADA_VCHD_HISTORY_TOP_V:-8}"
export MMADA_VCHD_HISTORY_EMA_DECAY="${MMADA_VCHD_HISTORY_EMA_DECAY:-0.7}"
export MMADA_VCHD_CCAW=1
export MMADA_VCHD_CCAW_MAX_CAPACITY="${MMADA_VCHD_CCAW_MAX_CAPACITY:-64}"
export MMADA_VCHD_CCAW_PRESSURE_DECAY="${MMADA_VCHD_CCAW_PRESSURE_DECAY:-0.8}"
export MMADA_VCHD_CCAW_EXPAND_STEP="${MMADA_VCHD_CCAW_EXPAND_STEP:-8}"
export MMADA_VCHD_CCAW_SHRINK_STEP="${MMADA_VCHD_CCAW_SHRINK_STEP:-4}"
export MMADA_VCHD_COLLECT_TRACE=1
export MMADA_VCHD_RETURN_REPORT=1

# Alpha/beta are consumed via CV_LAMBDA / CV_ALPHA env by mmada.py (see H7 in
# vchd_cross_check_20260714.md). Values below match v3 recommendations.
export MMADA_CV_LAMBDA="${MMADA_CV_LAMBDA:-0.5}"
export MMADA_CV_ALPHA="${MMADA_CV_ALPHA:-0.1}"

status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${STATUS_LOG}"
}

status "History-VCHD sweep starting"
status "  RUN_ID=${RUN_ID}"
status "  model_path=${MMADA_MODEL_PATH}"
status "  tokenizer_path=${MMADA_TOKENIZER_PATH}"
status "  vq_model_path=${MMADA_VQ_MODEL_PATH}"
status "  alpha=${MMADA_CV_LAMBDA} beta=${MMADA_CV_ALPHA}"
status "  tau_base=${MMADA_VCHD_TAU_BASE} tau_contrast=${MMADA_VCHD_TAU_CONTRAST}"
status "  window=${MMADA_VCHD_MASK_CAPACITY} max_commit=${MMADA_VCHD_MAX_COMMIT}"
status "  history=1 ccaw=1 ema_decay=${MMADA_VCHD_HISTORY_EMA_DECAY}"

DATASETS="${DATASETS:-MMMU_DEV_VAL_FULL MMBench_DEV_EN}"

declare -A SAMPLE_COUNT=(
  [MMMU_DEV_VAL_FULL]=1050
  [MMBench_DEV_EN]=4377
)

for dataset in ${DATASETS}; do
  n="${SAMPLE_COUNT[${dataset}]:-}"
  if [[ -z "${n}" ]]; then
    status "SKIP ${dataset}: unknown sample_count (add to SAMPLE_COUNT map)"
    continue
  fi

  if [[ ! -f "${LMUData}/${dataset}.tsv" ]]; then
    status "SKIP ${dataset}: ${LMUData}/${dataset}.tsv missing (run prepare_datasets.sh)"
    continue
  fi

  out_dir="${RUN_ROOT}/${dataset}"
  log_file="${LOG_ROOT}/${dataset}.log"
  report_dir="${LOG_ROOT}/${dataset}_reports"

  if compgen -G "${out_dir}"/**/*_"${dataset}".xlsx > /dev/null; then
    status "SKIP ${dataset}: result xlsx already present"
    continue
  fi

  mkdir -p "${out_dir}" "${report_dir}"
  export MMADA_VCHD_REPORT_DIR="${report_dir}"
  export MMADA_INDICES="$(seq -s, 0 "$((n - 1))")"

  status "START ${dataset} n=${n}"
  nvidia-smi --query-gpu=temperature.gpu,memory.free --format=csv,noheader \
    >> "${log_file}" 2>&1 || true

  "${PYTHON}" run.py \
    --data "${dataset}" \
    --model MMaDA-MixCoT-VCHD \
    --work-dir "${out_dir}" \
    >> "${log_file}" 2>&1
  rc=$?

  if compgen -G "${out_dir}"/**/*_"${dataset}".xlsx > /dev/null; then
    xlsx=$(ls "${out_dir}"/**/*_"${dataset}".xlsx 2>/dev/null | head -n 1)
    status "DONE  ${dataset} rc=${rc} result=${xlsx}"
  else
    status "FAILED ${dataset} rc=${rc}; see ${log_file}"
  fi

  sleep 30
done

status "History-VCHD sweep finished. Outputs under ${RUN_ROOT}"
