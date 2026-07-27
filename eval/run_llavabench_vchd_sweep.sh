#!/usr/bin/env bash
# LaViDa VCHD hyperparameter sweep on LLaVABench 60 samples.
# Each config runs a separate DDP-2 pass with custom (alpha, tau_contrast).

set -euo pipefail

CKPT=${1:-/autodl-fs/data/lavida-ckpts/lavida-llada-reason}
RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/LaViDa/eval/logs/lavida_llada_bidir_fix_alpha025_20260727_084747}

export PATH="/root/miniconda3/envs/CrossMatch/bin:${PATH}"
export PYTHONPATH="/root/autodl-tmp/LaViDa:/root/autodl-tmp/LaViDa/eval"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HOME="${HF_HOME:-/autodl-fs/data/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}}"
export LLADA_VISION_ENCODER="${LLADA_VISION_ENCODER:-/autodl-fs/data/lavida-ckpts/siglip-so400m-patch14-384}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DEBUG_PRINT_IMAGE_RES="${DEBUG_PRINT_IMAGE_RES:-0}"

if [[ -f /root/autodl-tmp/LaViDa/eval/bridge_terminal_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /root/autodl-tmp/LaViDa/eval/bridge_terminal_proxy.sh
fi

# Reuse the API key embedded in the reruns launcher.
eval "$(rg '^export (OPENAI_API_KEY|OPENAI_API_URL|GPT_EVAL_MODEL_NAME)=' /root/autodl-tmp/LaViDa/eval/run_fixed_bias_cache_reruns.sh)"

# One config: alpha, tau_contrast, label, port.
sweep_one() {
  local alpha=$1
  local tau=$2
  local label=$3
  local port=$4

  local outdir="${RUN_ROOT}/llavabench/${label}"
  mkdir -p "${outdir}"

  local gen_kwargs="decode_strategy=vchd,prefix_lm=True,max_new_tokens=512,vchd__prefix_prompt_cache=true,vchd__ccaw_enabled=false,vchd__enable_g_gate=false,vchd__mask_capacity=16,vchd__alpha=${alpha},vchd__beta=0.1,vchd__tau_base=0.1,vchd__tau_contrast=${tau}"

  printf 'START llavabench %s (alpha=%s tau=%s) %s\n' "${label}" "${alpha}" "${tau}" "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"

  cd /root/autodl-tmp/LaViDa
  accelerate launch --num_processes=2 --main_process_port="${port}" \
    -m lmms_eval \
    --model llava_llada \
    --model_args "pretrained=${CKPT},conv_template=llada,model_name=llava_llada" \
    --tasks llava_in_the_wild \
    --batch_size 1 \
    --gen_kwargs "${gen_kwargs}" \
    --log_samples \
    --log_samples_suffix "llada_${label}" \
    --output_path "${outdir}" 2>&1 | tee "${RUN_ROOT}/llavabench__${label}.log"

  printf 'DONE llavabench %s %s\n' "${label}" "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
}

# Sweep configs
sweep_one 0.5 0.5 vchd_prefix_cache_a05_t05 26030
sweep_one 1.0 0.1 vchd_prefix_cache_a10_t01 26031

printf 'ALL_DONE_SWEEP %s\n' "$(date --iso-8601=seconds)" | tee -a "${RUN_ROOT}/status.log"
