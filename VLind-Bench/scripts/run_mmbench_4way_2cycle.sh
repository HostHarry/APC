#!/usr/bin/env bash
# Four-way MMBench two-cycle evaluation on two GPUs.
set -uo pipefail

ROOT=/root/autodl-tmp
VLMEVAL="$ROOT/MMaDA_DCD_cloud_bundle_20260711/MMaDA/evaluation/VLMEvalKit"
DATASET=MMBench_DEV_EN_2C
RUN_ID="${MMADA_RUN_ID:-mmbench_4way_2cycle_20260723}"
RUN_ROOT="$VLMEVAL/outputs/$RUN_ID"
LOG_ROOT="$VLMEVAL/logs/$RUN_ID"
STATUS_LOG="$LOG_ROOT/status.log"
PYTHON="${PYTHON:-python}"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATUS_LOG"
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
  export LMUData="$VLMEVAL/LMUData"
  export MMADA_MODEL_PATH="$ROOT/MMaDA-8B-MixCoT"
  export MMADA_TOKENIZER_PATH="$ROOT/MMaDA-8B-MixCoT"
  export MMADA_VQ_MODEL_PATH="$ROOT/magvitv2"
  export MMADA_SKIP_LOCALIZE=1
  export MMADA_TEMPERATURE=0.0
  export PRINT_VANILLA=1
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
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
      # Thinking SWD on original remasking (same hyperparams as VLind/LLaVABench)
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
  status "START $tag gpus=0,1 dataset=$DATASET indices=${MMBENCH_INDICES:-all}"
  status "  decode=${MMADA_DECODE_STRATEGY:-} thinking PSP=${MMADA_THINKING_PSP:-0} VRG=${MMADA_THINKING_VRG:-0} SWD=${MMADA_THINKING_SWD:-0}"
  (
    cd "$VLMEVAL"
    CUDA_VISIBLE_DEVICES=0,1 torchrun \
      --standalone \
      --nproc-per-node=2 \
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
  "$PYTHON" - "$RUN_ROOT" <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
rows = []
# Prefer methods that exist under RUN_ROOT; keep a stable preferred order.
preferred = [
    "original", "official_dcd", "vchd", "vchd_ccaw",
    "swd", "psp", "psp_vrg",
]
tags = []
for tag in preferred:
    if (root / tag).exists():
        tags.append(tag)
for p in sorted(root.iterdir()):
    if p.is_dir() and p.name not in tags:
        tags.append(p.name)
for tag in tags:
    files = list((root / tag).rglob("*_MMBench_DEV_EN_2C_acc_all.csv"))
    if not files:
        files = list((root / tag).rglob("*_MMBench_DEV_EN_2C_acc.csv"))
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
  local failed=0
  local methods="${MMADA_METHODS:-official_dcd original vchd vchd_ccaw}"
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
