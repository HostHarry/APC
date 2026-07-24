#!/usr/bin/env bash
# Wait for an optional preceding MMBench run, then evaluate SWD / PSP / PSP+VRG.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MMADA_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VLMEVAL="$REPO_ROOT/MMaDA_DCD_cloud_bundle_20260711/MMaDA/evaluation/VLMEvalKit"
PREV_ID="${MMADA_PREVIOUS_RUN_ID:-mmbench_4way_2cycle_full_dist_20260723}"
PREV_STATUS="${MMADA_PREVIOUS_STATUS:-$VLMEVAL/logs/$PREV_ID/status.log}"
WAIT_LOG="${MMADA_WAIT_LOG:-$VLMEVAL/logs/mmbench_thinking_wait.log}"
RUN_ID="${MMADA_RUN_ID:-mmbench_thinking_swd_psp_psp_vrg_$(date +%Y%m%d)}"
METHODS="${MMADA_METHODS:-swd psp psp_vrg}"
PYTHON="${PYTHON:-python}"

mkdir -p "$(dirname "$WAIT_LOG")"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$WAIT_LOG"
}

wait_prev() {
  if [[ "${MMADA_WAIT_FOR_PREVIOUS:-1}" != "1" ]]; then
    log "previous-run wait disabled"
    return
  fi

  log "WAITING for $PREV_ID to finish before thinking methods: $METHODS"
  while true; do
    if [[ -f "$PREV_STATUS" ]] && grep -q "RUN END id=$PREV_ID" "$PREV_STATUS"; then
      log "detected RUN END in $PREV_STATUS"
      break
    fi
    if ! pgrep -f "$PREV_ID" >/dev/null 2>&1; then
      log "no processes referencing $PREV_ID"
      break
    fi
    log "still waiting for $PREV_ID..."
    sleep 120
  done
  sleep "${MMADA_POST_WAIT_SECONDS:-15}"
  log "GPU before start:"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "$WAIT_LOG" || true
}

main() {
  wait_prev
  export MMADA_RUN_ID="$RUN_ID"
  export MMADA_METHODS="$METHODS"
  export PYTHON
  log "START thinking MMBench RUN_ID=$RUN_ID METHODS=$METHODS"
  bash "$SCRIPT_DIR/run_mmbench_4way_2cycle.sh"
  local rc=$?
  log "END thinking MMBench rc=$rc"
  exit "$rc"
}

main "$@"
