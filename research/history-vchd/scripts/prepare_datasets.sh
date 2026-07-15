#!/usr/bin/env bash
# Download MMBench_DEV_EN + MMMU_DEV_VAL_FULL parquets from HuggingFace and
# convert them to VLMEval-style TSVs at $LMUData/{name}.tsv.
#
# Usage:
#   bash scripts/prepare_datasets.sh
#
# Env overrides:
#   PY          python interpreter (default: python)
#   LMUData     TSV output dir (default: MMaDA/evaluation/VLMEvalKit/LMUData)
#   HF_HOME     HuggingFace cache root (default: ~/.cache/huggingface)
#
# Datasets:
#   * lmms-lab/MMBench_EN  (dev split, 4377 rows)
#   * lmms-lab/MMMU        (dev+validation splits, ~1050 rows post-filter)
#
# The resulting TSVs match the schemas that hf_to_vlmeval_tsv.py emits, so
# `run.py --data MMBench_DEV_EN` / `--data MMMU_DEV_VAL_FULL` will just work.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMADA_ROOT="${REPO_ROOT}/MMaDA"
VLMEVAL_ROOT="${MMADA_ROOT}/evaluation/VLMEvalKit"

PY="${PY:-python}"
LMUData="${LMUData:-${VLMEVAL_ROOT}/LMUData}"
WORK="${WORK:-${REPO_ROOT}/.dataset_cache}"

mkdir -p "${LMUData}" "${WORK}"
export LMUData

echo "[data] LMUData     : ${LMUData}"
echo "[data] work / cache: ${WORK}"

# -------------------------------------------------------------------
# 1. MMBench_DEV_EN
# -------------------------------------------------------------------
MMBENCH_TSV="${LMUData}/MMBench_DEV_EN.tsv"
if [[ -f "${MMBENCH_TSV}" ]]; then
  echo "[data] SKIP MMBench_DEV_EN.tsv (already present)"
else
  echo "[data] downloading lmms-lab/MMBench_EN (dev) ..."
  ${PY} - <<PY_DOWNLOAD
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="lmms-lab/MMBench_EN",
    repo_type="dataset",
    allow_patterns=["*dev*.parquet", "data/*dev*.parquet"],
    local_dir="${WORK}/mmbench_en",
    local_dir_use_symlinks=False,
)
print("[data] snapshot at", path)
PY_DOWNLOAD

  MMBENCH_PARQUET=$(find "${WORK}/mmbench_en" -type f -name '*dev*.parquet' | head -n 1)
  if [[ -z "${MMBENCH_PARQUET}" ]]; then
    echo "[data] ERROR: no MMBench dev parquet found in ${WORK}/mmbench_en" >&2
    exit 1
  fi
  ${PY} "${VLMEVAL_ROOT}/scripts/hf_to_vlmeval_tsv.py" \
    --dataset MMBench_DEV_EN \
    --parquet "${MMBENCH_PARQUET}"
fi

# -------------------------------------------------------------------
# 2. MMMU_DEV_VAL_FULL (dev 150 + val 900 -> 1050 after grid concat)
# -------------------------------------------------------------------
MMMU_TSV="${LMUData}/MMMU_DEV_VAL_FULL.tsv"
if [[ -f "${MMMU_TSV}" ]]; then
  echo "[data] SKIP MMMU_DEV_VAL_FULL.tsv (already present)"
else
  echo "[data] downloading lmms-lab/MMMU (dev+validation) ..."
  ${PY} - <<PY_DOWNLOAD
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="lmms-lab/MMMU",
    repo_type="dataset",
    allow_patterns=[
        "*dev*.parquet", "*validation*.parquet",
        "data/*dev*.parquet", "data/*validation*.parquet",
    ],
    local_dir="${WORK}/mmmu",
    local_dir_use_symlinks=False,
)
print("[data] snapshot at", path)
PY_DOWNLOAD

  MMMU_PARQUETS=$(find "${WORK}/mmmu" -type f -name '*.parquet' | sort | tr '\n' ' ')
  if [[ -z "${MMMU_PARQUETS}" ]]; then
    echo "[data] ERROR: no MMMU parquet files found in ${WORK}/mmmu" >&2
    exit 1
  fi
  ${PY} "${VLMEVAL_ROOT}/scripts/hf_to_vlmeval_tsv.py" \
    --dataset MMMU_DEV_VAL_FULL \
    --parquet ${MMMU_PARQUETS}
fi

echo ""
echo "[data] done."
ls -lh "${LMUData}" | tail -n +2
