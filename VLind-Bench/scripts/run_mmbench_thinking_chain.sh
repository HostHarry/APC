#!/usr/bin/env bash
# Wait for current MMBench 4-way run, then evaluate SWD / PSP / PSP+VRG
# on MMBench_DEV_EN_2C with two-cycle scoring (same harness as 4-way).
set -euo pipefail

ROOT=/root/autodl-tmp
VLMEVAL="$ROOT/MMaDA_DCD_cloud_bundle_20260711/MMaDA/evaluation/VLMEvalKit"
PREV_ID=mmbench_4way_2cycle_full_dist_20260723
PREV_STATUS="$VLMEVAL/logs/$PREV_ID/status.log"
WAIT_LOG="$VLMEVAL/logs/mmbench_thinking_wait.log"
RUN_ID="${MMADA_RUN_ID:-mmbench_thinking_swd_psp_psp_vrg_20260724}"
METHODS="${MMADA_METHODS:-swd psp psp_vrg}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"

mkdir -p "$(dirname "$WAIT_LOG")"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$WAIT_LOG"
}

wait_prev() {
  log "WAITING for $PREV_ID to finish before thinking methods: $METHODS"
  while true; do
    if [[ -f "$PREV_STATUS" ]] && grep -q "RUN END id=$PREV_ID" "$PREV_STATUS"; then
      log "detected RUN END in $PREV_STATUS"
      break
    fi
    if ! pgrep -f "$PREV_ID" >/dev/null 2>&1 \
      && ! pgrep -af 'run.py' | grep -q "$PREV_ID"; then
      log "no processes referencing $PREV_ID"
      break
    fi
    local prog=""
    local clog="$VLMEVAL/logs/$PREV_ID/vchd_ccaw.log"
    if [[ -f "$clog" ]]; then
      prog=$(tr '\r' '\n' <"$clog" | grep -oE 'Rank 0/2: +[0-9]+%\|[^ ]* +[0-9]+/4327' | tail -1 || true)
      if [[ -z "$prog" ]]; then
        prog=$(tr '\r' '\n' <"$clog" | grep -oE '[0-9]+/4327' | tail -1 || true)
      fi
    fi
    log "still waiting... ${prog:-no progress yet}"
    sleep 120
  done
  sleep 15
  log "GPU before start:"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "$WAIT_LOG" || true
}

main() {
  wait_prev
  export MMADA_RUN_ID="$RUN_ID"
  export MMADA_METHODS="$METHODS"
  export PYTHON
  log "START thinking MMBench RUN_ID=$RUN_ID METHODS=$METHODS"
  bash "$ROOT/VLind-Bench/scripts/run_mmbench_4way_2cycle.sh"
  local rc=$?
  log "END thinking MMBench rc=$rc"
  exit "$rc"
}

main "$@"
