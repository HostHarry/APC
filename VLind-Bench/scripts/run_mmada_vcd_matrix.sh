#!/usr/bin/env bash
# MMaDA pure VCD matrix:
#   - VLind: dual-shard mmada_eval.py (official VLind harness)
#   - MMBench/MMMU/LLaVABench: Gen-Verse VLMEvalKit (official MMaDA VLM harness)
#   - M3CoT: LightChen233/M3CoT evaluate.py (official M3CoT scorer; NOT in VLMEvalKit)
set -euo pipefail

ROOT=/root/autodl-tmp
MMADA_ROOT=${MMADA_ROOT:-${ROOT}/MMaDA_DCD_cloud_bundle_20260711/MMaDA}
MODEL=${MODEL:-${ROOT}/MMaDA-8B-MixCoT}
VQ=${VQ:-${ROOT}/magvitv2}
RUN_ID=${RUN_ID:-mmada_vcd_matrix_$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-${ROOT}/VLind-Bench/outputs/${RUN_ID}}
BENCHES=${BENCHES:-vlind,mmbench,mmmu,llavabench,m3cot}
EVAL=${ROOT}/VLind-Bench/eval/mmada_eval.py
TAG=${TAG:-mmada_vcd_v302}
SHARD_A_IDS=${SHARD_A_IDS:-${ROOT}/VLind-Bench/outputs/ids_thinking_shardA.txt}
SHARD_B_IDS=${SHARD_B_IDS:-${ROOT}/VLind-Bench/outputs/ids_thinking_shardB.txt}

mkdir -p "${OUT}"
echo "${RUN_ID}" > "${ROOT}/VLind-Bench/outputs/LATEST_MMADA_VCD_RUN"
export PATH=/root/miniconda3/bin:${PATH}
export HF_HOME=${HF_HOME:-/autodl-fs/data/hf_home}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export http_proxy=${http_proxy:-http://127.0.0.1:7897}
export https_proxy=${https_proxy:-http://127.0.0.1:7897}
export MMADA_MODEL_PATH=${MODEL}
export MMADA_TOKENIZER_PATH=${MODEL}
export MMADA_VQ_MODEL_PATH=${VQ}
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true

status() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${OUT}/status.log"; }

wait_tag() {
  local tag=$1
  while pgrep -f -- "python.*--model-identifier ${tag}" >/dev/null 2>&1; do
    sleep 30
  done
}

status "START run_id=${RUN_ID} benches=${BENCHES} tag=${TAG}"

IFS=',' read -r -a arr <<<"${BENCHES}"
for b in "${arr[@]}"; do
  b="$(echo "$b" | xargs)"
  case "$b" in
    vlind)
      status "START vlind dual-shard"
      nohup env CUDA_VISIBLE_DEVICES=0 python -u "${EVAL}" \
        --strategy vcd \
        --model-identifier "${TAG}" \
        --mmada-root "${MMADA_ROOT}" \
        --model-path "${MODEL}" \
        --tokenizer-path "${MODEL}" \
        --vq-model-path "${VQ}" \
        --output-path "${OUT}/data_${TAG}_shardA.json" \
        --global-ids "$(cat "${SHARD_A_IDS}")" \
        --max-new-tokens 128 --steps 128 --block-length 64 --temperature 0.0 \
        --resume \
        >"${OUT}/${TAG}_shardA.log" 2>&1 &
      echo "VLIND_A=$!"
      nohup env CUDA_VISIBLE_DEVICES=1 python -u "${EVAL}" \
        --strategy vcd \
        --model-identifier "${TAG}" \
        --mmada-root "${MMADA_ROOT}" \
        --model-path "${MODEL}" \
        --tokenizer-path "${MODEL}" \
        --vq-model-path "${VQ}" \
        --output-path "${OUT}/data_${TAG}_shardB.json" \
        --global-ids "$(cat "${SHARD_B_IDS}")" \
        --max-new-tokens 128 --steps 128 --block-length 64 --temperature 0.0 \
        --resume \
        >"${OUT}/${TAG}_shardB.log" 2>&1 &
      echo "VLIND_B=$!"
      wait_tag "${TAG}"
      python3 - <<PY
import json
from pathlib import Path
base=Path("${OUT}"); tag="${TAG}"
a=json.loads((base/f"data_{tag}_shardA.json").read_text())
b=json.loads((base/f"data_{tag}_shardB.json").read_text())
by={int(r["global_id"]):r for r in a+b}
merged=[by[i] for i in sorted(by)]
(base/f"data_{tag}.json").write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
print(f"merged={len(merged)}")
PY
      python "${ROOT}/VLind-Bench/eval/score_pipeline.py" \
        --data_path "${OUT}/data_${TAG}.json" \
        --model_identifier "${TAG}" \
        | tee "${OUT}/${TAG}_score.txt"
      status "DONE vlind"
      ;;
    mmbench|mmmu|llavabench)
      case "$b" in
        mmbench) DATA=MMBench_DEV_EN_2C ;;
        mmmu) DATA=MMMU_DEV_VAL ;;
        llavabench) DATA=LLaVABench ;;
      esac
      status "START ${DATA} (VLMEvalKit official)"
      cd "${MMADA_ROOT}/evaluation/VLMEvalKit"
      mkdir -p "${OUT}/vlmeval" LMUData/images
      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} \
      torchrun --nproc-per-node=2 --master-port=$((27100 + RANDOM % 200)) \
        run.py --data "${DATA}" --model MMaDA-MixCoT-VCD \
        --work-dir "${OUT}/vlmeval" \
        2>&1 | tee "${OUT}/${b}.log"
      status "DONE ${DATA}"
      ;;
    m3cot)
      status "START m3cot (LightChen233 official evaluate.py)"
      LIMIT=0 STRATEGY=vcd OUT_DIR="${OUT}/m3cot" \
        bash "${ROOT}/VLind-Bench/scripts/run_mmada_vcd_m3cot.sh"
      status "DONE m3cot"
      ;;
    *)
      status "Unknown bench ${b}"; exit 2 ;;
  esac
done

status "ALL COMPLETE run_id=${RUN_ID}"
