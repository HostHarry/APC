#!/usr/bin/env bash
# Wait for the currently-running PSP+VRG rerun (llava_bench, mmmu, mmbench)
# to finish, then launch M3CoT + VLind on SWD, PSP, and PSP+VRG.
#
# The current run is tracked via eval/logs/PSP_VRG_RERUN_LATEST which points at
# eval/logs/<run_id>/status.log. We poll for a terminal marker (ALL COMPLETE or
# FAILED) and then hand off to eval/run_thinking_five_benchmarks.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT=${CKPT:-"${ROOT}/lavida-ckpts/lavida-llada-hd-reason"}

# Which run to wait on. Defaults to the pointer written by the current run.
WAIT_RUN_ID=${WAIT_RUN_ID:-$(cat "${ROOT}/eval/logs/PSP_VRG_RERUN_LATEST" 2>/dev/null || true)}
if [[ -z "${WAIT_RUN_ID}" ]]; then
  echo "[queue] No prior run to wait on; starting M3CoT/VLind immediately." >&2
  WAIT_RUN_ID=""
else
  echo "[queue] Waiting for run '${WAIT_RUN_ID}' to reach a terminal state..."
fi

if [[ -n "${WAIT_RUN_ID}" ]]; then
  status_log="${ROOT}/eval/logs/${WAIT_RUN_ID}/status.log"
  # Poll every 60s. Exit polling loop when status.log contains ALL COMPLETE or FAILED.
  while true; do
    if [[ -f "${status_log}" ]] && rg -q '(^|\s)ALL COMPLETE run_id=|(^|\s)FAILED benchmark=' "${status_log}"; then
      break
    fi
    sleep 60
  done
  # Preserve the prior status.log terminal line for provenance.
  echo "[queue] Prior run terminated:"
  rg -N '(^|\s)ALL COMPLETE run_id=|(^|\s)FAILED benchmark=' "${status_log}" | tail -1
fi

# Launch the follow-up run. Fresh RUN_ID so logs are separated.
export RUN_ID=${RUN_ID:-lavida_m3cot_vlind_swd_psp_pspvrg_$(date +%Y%m%d_%H%M%S)}
export MODES=${MODES:-swd,psp,psp_vrg}
export BENCHMARKS=${BENCHMARKS:-m3cot,vlind}
export LIMIT=${LIMIT:-0}
export NUM_PROCESSES=${NUM_PROCESSES:-2}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
# Skip the GPT-4o judge for LLaVA-Bench (not queued here anyway); makes runner
# tolerant of missing OPENAI_API_KEY.
export LLAVA_JUDGE_SKIP=${LLAVA_JUDGE_SKIP:-1}
# Offline caches / mirrors are inherited via the runner script.

mkdir -p "${ROOT}/eval/logs/${RUN_ID}"
printf '%s\n' "${RUN_ID}" >"${ROOT}/eval/logs/LATEST_LAVIDA_THINKING_RUN"
printf '%s\n' "${RUN_ID}" >"${ROOT}/eval/logs/M3COT_VLIND_LATEST"

echo "[queue] Launching follow-up run id=${RUN_ID} modes=${MODES} benchmarks=${BENCHMARKS}"
bash "${ROOT}/eval/run_thinking_five_benchmarks.sh" "${CKPT}"
