#!/usr/bin/env bash
# LLaVABench inference for SWD / PSP / PSP+VRG on original remasking.
# Waits for plain_noccaw VLind jobs, then runs --mode infer (no GPT judge).
set -euo pipefail

ROOT=/root/autodl-tmp
VLMEVAL="$ROOT/MMaDA_DCD_cloud_bundle_20260711/MMaDA/evaluation/VLMEvalKit"
OUT_ROOT="$ROOT/VLind-Bench/outputs/llavabench_thinking"
LOG_ROOT="$OUT_ROOT/logs"
RUN_ID="${MMADA_RUN_ID:-llavabench_thinking_$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-python}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
STATUS_LOG="$LOG_ROOT/status.log"

status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATUS_LOG"
}

wait_plain_noccaw() {
  status "waiting for mmada_vchd_plain_noccaw_v302 to finish..."
  while pgrep -f -- '--model-identifier mmada_vchd_plain_noccaw_v302' >/dev/null 2>&1; do
    sleep 60
  done
  status "plain_noccaw finished (or not running)"
}

setup_lmudata() {
  local src="$ROOT/lladav_vchd_server_bundle/VLMEvalKit/LMUData"
  mkdir -p "$VLMEVAL/LMUData/images"
  ln -sfn "$src/LLaVABench.tsv" "$VLMEVAL/LMUData/LLaVABench.tsv"
  ln -sfn "$src/images/LLaVABench" "$VLMEVAL/LMUData/images/LLaVABench"
}

clear_thinking_env() {
  unset MMADA_THINKING_PSP MMADA_THINKING_PSP_GAMMA
  unset MMADA_THINKING_VRG MMADA_THINKING_VRG_SCALE
  unset MMADA_THINKING_SWD MMADA_THINKING_SWD_LAMBDA
  export MMADA_THINKING_PSP=0
  export MMADA_THINKING_VRG=0
  export MMADA_THINKING_SWD=0
}

configure_common() {
  export LMUData="$VLMEVAL/LMUData"
  export MMADA_MODEL_PATH="$ROOT/MMaDA-8B-MixCoT"
  export MMADA_TOKENIZER_PATH="$ROOT/MMaDA-8B-MixCoT"
  export MMADA_VQ_MODEL_PATH="$ROOT/magvitv2"
  export MMADA_DECODE_STRATEGY=original
  export MMADA_CACHE_TYPE=none
  export MMADA_CV_MODE=off
  export MMADA_SKIP_LOCALIZE=1
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  unset MMADA_INDICES MMADA_INDEX_FILE USE_COT
  unset MMADA_VCHD_REPORT_DIR MMADA_CV_DEBUG_DIR
}

run_one() {
  local tag="$1"
  local gpu="$2"
  shift 2
  # remaining args are env assignments applied after clear
  local out_dir="$OUT_ROOT/${RUN_ID}/${tag}"
  local log_file="$LOG_ROOT/${RUN_ID}_${tag}.log"
  local pred

  mkdir -p "$out_dir"
  clear_thinking_env
  # Apply method-specific exports passed as KEY=VAL pairs
  local kv
  for kv in "$@"; do
    export "$kv"
  done

  # Skip if prediction workbook already present
  shopt -s nullglob globstar
  pred=("$out_dir"/**/*_LLaVABench.xlsx)
  shopt -u nullglob globstar
  if ((${#pred[@]})); then
    status "SKIP ${tag}: prediction exists at ${pred[0]}"
    return 0
  fi

  status "START ${tag} gpu=${gpu} decode=original thinking env:"
  status "  PSP=${MMADA_THINKING_PSP} gamma=${MMADA_THINKING_PSP_GAMMA:-} VRG=${MMADA_THINKING_VRG} scale=${MMADA_THINKING_VRG_SCALE:-} SWD=${MMADA_THINKING_SWD} lambda=${MMADA_THINKING_SWD_LAMBDA:-}"

  (
    cd "$VLMEVAL"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u run.py \
      --data LLaVABench \
      --model MMaDA-MixCoT \
      --mode infer \
      --work-dir "$out_dir" \
      --verbose
  ) >"$log_file" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    status "FAIL ${tag} exit=${rc} log=${log_file}"
    return "$rc"
  fi
  status "DONE ${tag} log=${log_file}"
}

export_predictions_json() {
  "$PYTHON" - <<'PY'
import json
from pathlib import Path
import pandas as pd

out_root = Path("/root/autodl-tmp/VLind-Bench/outputs/llavabench_thinking")
run_dirs = sorted([p for p in out_root.iterdir() if p.is_dir() and p.name.startswith("llavabench_thinking_")], reverse=True)
if not run_dirs:
    print("no run dirs")
    raise SystemExit(0)
run_dir = run_dirs[0]
print("export from", run_dir)
merged = {}
for tag_dir in sorted(run_dir.iterdir()):
    if not tag_dir.is_dir():
        continue
    tag = tag_dir.name
    xlsx = list(tag_dir.rglob("*_LLaVABench.xlsx"))
    # prefer non-openai result
    xlsx = [x for x in xlsx if "_openai" not in x.name]
    if not xlsx:
        print(f"  {tag}: missing xlsx")
        continue
    df = pd.read_excel(xlsx[0])
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "index": int(r["index"]) if "index" in r and pd.notna(r["index"]) else None,
            "question": r.get("question", ""),
            "category": r.get("category", ""),
            "prediction": r.get("prediction", ""),
            "gpt4_ans": r.get("gpt4_ans", ""),
        })
    merged[tag] = rows
    out_json = out_root / f"preds_{tag}.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {tag}: {len(rows)} -> {out_json}")

(out_root / "preds_all_methods.json").write_text(
    json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("wrote", out_root / "preds_all_methods.json")
PY
}

main() {
  setup_lmudata
  configure_common
  wait_plain_noccaw
  status "GPU status before LLaVABench:"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "$STATUS_LOG" || true

  # Parallel wave 1: SWD + PSP
  run_one swd 0 \
    MMADA_THINKING_SWD=1 MMADA_THINKING_SWD_LAMBDA=5.0 &
  pid_swd=$!
  run_one psp 1 \
    MMADA_THINKING_PSP=1 MMADA_THINKING_PSP_GAMMA=0.5 &
  pid_psp=$!
  wait "$pid_swd"
  rc_swd=$?
  wait "$pid_psp"
  rc_psp=$?
  status "wave1 done swd=${rc_swd} psp=${rc_psp}"

  # Wave 2: PSP+VRG (heavier ~2x NFE)
  run_one psp_vrg 0 \
    MMADA_THINKING_PSP=1 MMADA_THINKING_PSP_GAMMA=0.5 \
    MMADA_THINKING_VRG=1 MMADA_THINKING_VRG_SCALE=0.5
  rc_vrg=$?
  status "wave2 done psp_vrg=${rc_vrg}"

  export_predictions_json || status "WARN export_predictions_json failed"
  status "ALL DONE run_id=${RUN_ID} out=${OUT_ROOT}/${RUN_ID}"
  [[ $rc_swd -eq 0 && $rc_psp -eq 0 && $rc_vrg -eq 0 ]]
}

main "$@"
