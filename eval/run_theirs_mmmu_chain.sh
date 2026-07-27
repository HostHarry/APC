#!/usr/bin/env bash
# After the ours-MC queue (incl. extra rerun) completes, run SWD / PSP /
# PSP_VRG on MMMU with the patched (bidirectional) APC tree and OUR
# checkpoint, giving a fully controlled same-machine comparison against the
# 43133-server numbers. MMMU scores locally; no GPT judge involved.
set -euo pipefail
MAIN_ROOT=/root/autodl-tmp/LaViDa/eval/logs/mc_aligned_ours_20260727
while ! rg -q 'ALL_DONE_MC_EXTRA' "${MAIN_ROOT}/status.log" 2>/dev/null; do sleep 120; done

ROOT=/root/autodl-tmp/APC_43133/LaViDa
RUN_ROOT=${ROOT}/eval/logs/theirs_mmmu_bidir_20260727
CKPT=/autodl-fs/data/lavida-ckpts/lavida-llada-reason
export PATH="/root/miniconda3/envs/CrossMatch/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}:${ROOT}/eval"
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/autodl-fs/data/hf_home
export HUGGINGFACE_HUB_CACHE=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}
export LLADA_VISION_ENCODER=/autodl-fs/data/lavida-ckpts/siglip-so400m-patch14-384
export DEBUG_PRINT_IMAGE_RES=0
export PYTHONUNBUFFERED=1
source /root/autodl-tmp/LaViDa/eval/bridge_terminal_proxy.sh || true
mkdir -p "${RUN_ROOT}"

cd "${ROOT}"
port=26140
for mode in swd psp psp_vrg; do
  printf 'START theirs_mmmu %s %s\n' "${mode}" "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
  MODE="${mode}" TASKS=mmmu_dev_val_full LIMIT=0 NUM_PROCESSES=2 \
  MAX_NEW_TOKENS=128 BLOCK_LENGTH=64 STEP_PER_BLOCK=64 \
  MAIN_PROCESS_PORT="${port}" \
  OUTPUT_PATH="${RUN_ROOT}/${mode}" \
    bash eval/run_thinking_llada.sh "${CKPT}" 2>&1 \
    | tee "${RUN_ROOT}/mmmu__${mode}.log"
  printf 'DONE theirs_mmmu %s %s\n' "${mode}" "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
  port=$((port + 1))
done
printf 'ALL_DONE_THEIRS_MMMU %s\n' "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
