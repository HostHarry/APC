#!/usr/bin/env bash
# MMBench two-cycle evaluation harness used by the thinking-method chain.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MMADA_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
EXTERNAL_ROOT="${MMADA_EXTERNAL_ROOT:-/root/autodl-tmp}"
VLMEVAL="$REPO_ROOT/MMaDA_DCD_cloud_bundle_20260711/MMaDA/evaluation/VLMEvalKit"
DATASET="${MMADA_MMBENCH_DATASET:-MMBench_DEV_EN_2C}"
RUN_ID="${MMADA_RUN_ID:-mmbench_4way_2cycle_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${MMADA_OUTPUT_ROOT:-$VLMEVAL/outputs/$RUN_ID}"
LOG_ROOT="${MMADA_LOG_ROOT:-$VLMEVAL/logs/$RUN_ID}"
STATUS_LOG="$LOG_ROOT/status.log"
PYTHON="${PYTHON:-python}"
GPU_IDS="${MMADA_GPU_IDS:-0,1}"
NPROC_PER_NODE="${MMADA_NPROC_PER_NODE:-2}"
LMU_DATA_ROOT="${LMUData:-$VLMEVAL/LMUData}"
TWO_CYCLE_SOURCE="${MMADA_MMBENCH_SOURCE_TSV:-$LMU_DATA_ROOT/MMBench_DEV_EN.tsv}"
TWO_CYCLE_TSV="$LMU_DATA_ROOT/MMBench_DEV_EN_2C.tsv"
TWO_CYCLE_GENERATOR="$SCRIPT_DIR/make_mmbench_two_cycle.py"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATUS_LOG"
}

prepare_two_cycle_dataset() {
  if [[ "$DATASET" != "MMBench_DEV_EN_2C" ]]; then
    return 0
  fi
  if [[ -s "$TWO_CYCLE_TSV" ]]; then
    status "using existing two-cycle TSV: $TWO_CYCLE_TSV"
    return 0
  fi
  if [[ ! -s "$TWO_CYCLE_SOURCE" ]]; then
    status "ERROR missing source TSV: $TWO_CYCLE_SOURCE"
    status "Prepare it, then run:"
    status "  $PYTHON $TWO_CYCLE_GENERATOR $TWO_CYCLE_SOURCE $TWO_CYCLE_TSV"
    return 2
  fi

  status "generating two-cycle TSV from $TWO_CYCLE_SOURCE"
  if ! "$PYTHON" "$TWO_CYCLE_GENERATOR" \
      "$TWO_CYCLE_SOURCE" "$TWO_CYCLE_TSV" | tee -a "$STATUS_LOG"; then
    status "ERROR two-cycle TSV generation failed"
    return 2
  fi
  if [[ ! -s "$TWO_CYCLE_TSV" ]]; then
    status "ERROR generator did not create $TWO_CYCLE_TSV"
    return 2
  fi
}

clear_method_env() {
  unset MMADA_DECODE_STRATEGY MMADA_CACHE_TYPE MMADA_CV_MODE
  unset MMADA_DCD_INITIAL_WINDOW_LENGTH MMADA_DCD_MAX_WINDOW_LENGTH
  unset MMADA_DCD_REFRESH_COUNT
  unset MMADA_VCHD_ALPHA MMADA_VCHD_BETA
  unset MMADA_VCHD_TAU_BASE MMADA_VCHD_TAU_CONTRAST
  unset MMADA_VCHD_MASK_CAPACITY MMADA_VCHD_MAX_COMMIT
  unset MMADA_VCHD_HISTORY MMADA_VCHD_CCAW
  unset MMADA_VCHD_CCAW_MODE MMADA_VCHD_CCAW_BLOCK_SIZE
  unset MMADA_VCHD_CCAW_MIN_COMMIT MMADA_VCHD_CCAW_QUALIFIED_BUDGET
  unset MMADA_VCHD_CCAW_MAX_CAPACITY MMADA_VCHD_CCAW_PRESSURE_DECAY
  unset MMADA_VCHD_CCAW_PRESSURE_SCALE MMADA_VCHD_CCAW_EXPAND_STEP
  unset MMADA_VCHD_CCAW_SHRINK_STEP MMADA_VCHD_CCAW_PRESSURE_FILTER
  unset MMADA_VCHD_CACHE_TYPE MMADA_VCHD_REPORT_DIR
  unset MMADA_THINKING_PSP MMADA_THINKING_PSP_GAMMA
  unset MMADA_THINKING_VRG MMADA_THINKING_VRG_SCALE
  unset MMADA_THINKING_SWD MMADA_THINKING_SWD_LAMBDA
}

common_env() {
  export LMUData="$LMU_DATA_ROOT"
  export MMADA_MODEL_PATH="${MMADA_MODEL_PATH:-$EXTERNAL_ROOT/MMaDA-8B-MixCoT}"
  export MMADA_TOKENIZER_PATH="${MMADA_TOKENIZER_PATH:-$EXTERNAL_ROOT/MMaDA-8B-MixCoT}"
  export MMADA_VQ_MODEL_PATH="${MMADA_VQ_MODEL_PATH:-$EXTERNAL_ROOT/magvitv2}"
  export MMADA_SKIP_LOCALIZE=1
  export MMADA_TEMPERATURE="${MMADA_TEMPERATURE:-0.0}"
  export PRINT_VANILLA=1
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  unset USE_COT
  if [[ -n "${MMBENCH_INDICES:-}" ]]; then
    export MMADA_INDICES="$MMBENCH_INDICES"
  else
    unset MMADA_INDICES MMADA_INDEX_FILE
  fi
}

configure_method() {
  local tag="$1"
  clear_method_env
  common_env
  case "$tag" in
    original)
      export MMADA_DECODE_STRATEGY=original
      export MMADA_CACHE_TYPE=none
      ;;
    official_dcd)
      export MMADA_DECODE_STRATEGY=official_dcd
      export MMADA_CACHE_TYPE=dual-delay2
      export MMADA_DCD_INITIAL_WINDOW_LENGTH=16
      export MMADA_DCD_MAX_WINDOW_LENGTH=128
      export MMADA_DCD_REFRESH_COUNT=32
      ;;
    vchd)
      export MMADA_DECODE_STRATEGY=vchd
      export MMADA_VCHD_ALPHA=0.5
      export MMADA_VCHD_BETA=0.1
      export MMADA_VCHD_TAU_BASE=0.1
      export MMADA_VCHD_TAU_CONTRAST=0.9
      export MMADA_VCHD_MASK_CAPACITY=16
      export MMADA_VCHD_MAX_COMMIT=16
      export MMADA_VCHD_HISTORY=0
      export MMADA_VCHD_CCAW=0
      export MMADA_VCHD_CACHE_TYPE=none
      ;;
    vchd_ccaw)
      export MMADA_DECODE_STRATEGY=vchd
      export MMADA_VCHD_ALPHA=0.5
      export MMADA_VCHD_BETA=0.1
      export MMADA_VCHD_TAU_BASE=0.1
      export MMADA_VCHD_TAU_CONTRAST=0.9
      export MMADA_VCHD_MASK_CAPACITY=16
      export MMADA_VCHD_MAX_COMMIT=16
      export MMADA_VCHD_HISTORY=0
      export MMADA_VCHD_CCAW=1
      export MMADA_VCHD_CCAW_MODE=inverse_window
      export MMADA_VCHD_CCAW_BLOCK_SIZE=32
      export MMADA_VCHD_CCAW_MIN_COMMIT=1
      export MMADA_VCHD_CCAW_QUALIFIED_BUDGET=1
      export MMADA_VCHD_CCAW_MAX_CAPACITY=64
      export MMADA_VCHD_CCAW_PRESSURE_DECAY=0.8
      export MMADA_VCHD_CCAW_PRESSURE_SCALE=3.0
      export MMADA_VCHD_CCAW_EXPAND_STEP=8
      export MMADA_VCHD_CCAW_SHRINK_STEP=4
      export MMADA_VCHD_CCAW_PRESSURE_FILTER=ema
      export MMADA_VCHD_CACHE_TYPE=none
      ;;
    swd)
      export MMADA_DECODE_STRATEGY=original
      export MMADA_CACHE_TYPE=none
      export MMADA_THINKING_SWD=1
      export MMADA_THINKING_SWD_LAMBDA=5.0
      export MMADA_THINKING_PSP=0
      export MMADA_THINKING_VRG=0
      ;;
    psp)
      export MMADA_DECODE_STRATEGY=original
      export MMADA_CACHE_TYPE=none
      export MMADA_THINKING_PSP=1
      export MMADA_THINKING_PSP_GAMMA=0.5
      export MMADA_THINKING_VRG=0
      export MMADA_THINKING_SWD=0
      ;;
    psp_vrg)
      export MMADA_DECODE_STRATEGY=original
      export MMADA_CACHE_TYPE=none
      export MMADA_THINKING_PSP=1
      export MMADA_THINKING_PSP_GAMMA=0.5
      export MMADA_THINKING_VRG=1
      export MMADA_THINKING_VRG_SCALE=0.5
      export MMADA_THINKING_SWD=0
      ;;
    *)
      status "Unknown method: $tag"
      return 2
      ;;
  esac
}

run_method() {
  local tag="$1"
  local out_dir="$RUN_ROOT/$tag"
  local log_file="$LOG_ROOT/$tag.log"
  mkdir -p "$out_dir"
  configure_method "$tag" || return $?
  status "START $tag gpus=$GPU_IDS dataset=$DATASET indices=${MMBENCH_INDICES:-all}"
  status "  decode=${MMADA_DECODE_STRATEGY:-} thinking PSP=${MMADA_THINKING_PSP:-0} VRG=${MMADA_THINKING_VRG:-0} SWD=${MMADA_THINKING_SWD:-0}"
  (
    cd "$VLMEVAL"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" torchrun \
      --standalone \
      --nproc-per-node="$NPROC_PER_NODE" \
      run.py \
      --data "$DATASET" \
      --model MMaDA-MixCoT \
      --mode all \
      --work-dir "$out_dir" \
      --reuse \
      --verbose
  ) >"$log_file" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    status "DONE $tag rc=0"
  else
    status "FAIL $tag rc=$rc log=$log_file"
  fi
  return "$rc"
}

summarize() {
  "$PYTHON" - "$RUN_ROOT" "$DATASET" <<'PY'
from pathlib import Path
import sys

import pandas as pd

root = Path(sys.argv[1])
dataset = sys.argv[2]
rows = []
preferred = [
    "original", "official_dcd", "vchd", "vchd_ccaw",
    "swd", "psp", "psp_vrg",
]
tags = [tag for tag in preferred if (root / tag).exists()]
for path in sorted(root.iterdir()):
    if path.is_dir() and path.name not in tags:
        tags.append(path.name)
for tag in tags:
    files = list((root / tag).rglob(f"*_{dataset}_acc_all.csv"))
    if not files:
        files = list((root / tag).rglob(f"*_{dataset}_acc.csv"))
    if not files:
        print(f"{tag}: score missing")
        continue
    score = pd.read_csv(files[-1])
    score.insert(0, "method", tag)
    rows.append(score)
    print(f"{tag}: {files[-1]}")
if rows:
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(root / "summary.csv", index=False)
    print(out.to_string(index=False))
    print("wrote", root / "summary.csv")
PY
}

main() {
  status "RUN START id=$RUN_ID"
  if ! prepare_two_cycle_dataset; then
    status "RUN ABORT id=$RUN_ID: two-cycle dataset unavailable"
    return 2
  fi
  local failed=0
  local methods="${MMADA_METHODS:-swd psp psp_vrg}"
  for tag in $methods; do
    if ! run_method "$tag"; then
      failed=1
    fi
  done

  summarize | tee -a "$STATUS_LOG"
  status "RUN END id=$RUN_ID"
  [[ $failed -eq 0 ]]
}

main "$@"
