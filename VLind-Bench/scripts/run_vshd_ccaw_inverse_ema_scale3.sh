#!/usr/bin/env bash
# VSHD + CCAW recipe used for VLind-Bench v302:
#   mmada_vchd_ccaw_inverse_ema_scale3_v302
#
# Key knobs (verified from run reports):
#   strategy=vchd, profile=ccaw
#   ccaw_mode=inverse_window
#   ccaw_pressure_filter=ema
#   ccaw_pressure_scale=3.0
#   history / focus_longtail disabled
#   cache=none, 128/128/64, temperature=0.8
#
# Reported scores on the 302-context set:
#   a/b/c/d_M/c_raw/d_raw =
#   46.0 / 57.6 / 76.3 / 21.1 / 65.6 / 24.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL="$ROOT/VLind-Bench/eval/mmada_eval.py"
OUT="$ROOT/VLind-Bench/outputs"
TAG="${TAG:-mmada_vchd_ccaw_inverse_ema_scale3_v302}"
GLOBAL_IDS="${GLOBAL_IDS:-}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

mkdir -p "$OUT"

cmd=(
  python -u "$EVAL"
  --strategy vchd
  --vchd-profile ccaw
  --vchd-ccaw-mode inverse_window
  --vchd-ccaw-pressure-filter ema
  --vchd-ccaw-pressure-scale 3.0
  --vchd-ccaw-block-size 32
  --vchd-ccaw-min-commit 1
  --vchd-ccaw-qualified-budget 1
  --vchd-ccaw-max-capacity 64
  --vchd-ccaw-pressure-decay 0.8
  --vchd-ccaw-expand-step 8
  --vchd-ccaw-shrink-step 4
  --vchd-alpha 0.5
  --vchd-beta 0.1
  --vchd-tau-base 0.1
  --vchd-tau-contrast 0.9
  --vchd-mask-capacity 16
  --vchd-max-commit 16
  --vchd-cache-type none
  --model-identifier "$TAG"
  --output-path "$OUT/data_${TAG}.json"
  --vchd-report-dir "$OUT/${TAG}_reports"
  --max-new-tokens 128
  --steps 128
  --block-length 64
  --temperature 0.8
  --vchd-save-reports
  --vchd-collect-trace
  --resume
)

if [[ -n "$GLOBAL_IDS" ]]; then
  cmd+=(--global-ids "$GLOBAL_IDS")
fi

echo "[run] $(date) CUDA_VISIBLE_DEVICES=$CUDA_DEVICE TAG=$TAG"
echo "[run] ${cmd[*]}"
env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${cmd[@]}"

python "$ROOT/VLind-Bench/eval/score_pipeline.py" \
  --data_path "$OUT/data_${TAG}.json" \
  --model_identifier "$TAG" \
  | tee "$OUT/${TAG}_score.txt"
