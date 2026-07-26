#!/usr/bin/env bash
# Wait for lavida_attentionfix_3bench (mmmu/mmbench/llava) to finish successfully,
# then run M3CoT + VLind with the same Thinking modes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PHASE1_ID=${PHASE1_ID:-lavida_attentionfix_3bench_20260727}
PHASE1_STATUS="${ROOT}/eval/logs/${PHASE1_ID}/status.log"
PHASE2_ID=${PHASE2_ID:-lavida_attentionfix_m3cot_vlind_20260727}
PHASE2_LOG="${ROOT}/eval/logs/${PHASE2_ID}_screen.log"
POLL_SEC=${POLL_SEC:-60}

mkdir -p "${ROOT}/eval/logs"
exec > >(tee -a "${PHASE2_LOG}") 2>&1

stamp() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

stamp "queue waiter start; waiting for ${PHASE1_STATUS} ALL COMPLETE"

while true; do
  if [[ -f "${PHASE1_STATUS}" ]] && rg -q 'ALL COMPLETE' "${PHASE1_STATUS}"; then
    stamp "phase1 complete: ${PHASE1_ID}"
    break
  fi
  if [[ -f "${PHASE1_STATUS}" ]] && rg -q '^\[.*\] FAILED ' "${PHASE1_STATUS}"; then
    stamp "ERROR phase1 failed; see ${PHASE1_STATUS}"
    exit 1
  fi
  # also stop if the phase1 screen vanished without completion
  if ! screen -list 2>/dev/null | rg -q 'lavida_attentionfix_3bench'; then
    if [[ -f "${PHASE1_STATUS}" ]] && rg -q 'ALL COMPLETE' "${PHASE1_STATUS}"; then
      break
    fi
    stamp "ERROR phase1 screen gone without ALL COMPLETE"
    exit 2
  fi
  sleep "${POLL_SEC}"
done

# Preflight: offline M3CoT local snapshot + VLind assets
export HF_HOME=${HF_HOME:-/autodl-fs/data/hf_home}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export http_proxy=${http_proxy:-http://127.0.0.1:7897}
export https_proxy=${https_proxy:-http://127.0.0.1:7897}
export HTTP_PROXY=${HTTP_PROXY:-$http_proxy}
export HTTPS_PROXY=${HTTPS_PROXY:-$https_proxy}
export NO_PROXY=${NO_PROXY:-localhost,127.0.0.1,::1}
export no_proxy=${no_proxy:-localhost,127.0.0.1,::1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONUNBUFFERED=1
export DEBUG_PRINT_IMAGE_RES=0
export LLAVA_JUDGE_SKIP=1

stamp "preflight M3CoT/VLind"
python - <<'PY'
from pathlib import Path
from datasets import load_dataset
import pandas as pd

snap = Path(
    "/autodl-fs/data/hf_home/hub/datasets--LightChen2333--M3CoT/"
    "snapshots/48cf35001d595a6b0290c82c897a4b4563390821"
)
assert snap.is_dir(), snap
ds = load_dataset(str(snap), split="test")
assert len(ds) == 2318, len(ds)

tsv = Path(
    "/root/autodl-tmp/MMaDA_DCD_cloud_bundle_20260711/MMaDA/evaluation/"
    "VLMEvalKit/LMUData/VLind-Bench.tsv"
)
df = pd.read_csv(tsv, sep="\t")
assert len(df) == 6360, len(df)
img = Path(str(df.iloc[0]["image_path"]))
assert img.is_file(), img
print(f"preflight_ok m3cot={len(ds)} vlind={len(df)}")
PY

stamp "START phase2 RUN_ID=${PHASE2_ID} benchmarks=m3cot,vlind modes=swd,psp,psp_vrg"
cd "${ROOT}"
RUN_ID="${PHASE2_ID}" \
BENCHMARKS=m3cot,vlind \
MODES=swd,psp,psp_vrg \
LIMIT=0 \
NUM_PROCESSES=2 \
START_PORT=26901 \
LLAVA_JUDGE_SKIP=1 \
LLAVA_JUDGE_MODEL=gpt-4o \
  bash eval/run_thinking_five_benchmarks.sh \
    /root/autodl-tmp/LaViDa/lavida-ckpts/lavida-llada-hd-reason
rc=$?
stamp "phase2 rc=${rc}"
exit "${rc}"
