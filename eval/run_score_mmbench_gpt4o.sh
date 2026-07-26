#!/usr/bin/env bash
# Score LaViDa MMBench submissions with GPT-4o (local prefetch + API).
#
# Usage:
#   1) Put your key below (or export OPENAI_API_KEY before running)
#   2) bash eval/run_score_mmbench_gpt4o.sh
#   3) optional: MODE=vchd bash eval/run_score_mmbench_gpt4o.sh
#               INPUT=/path/to/mmbench_en_dev_results.xlsx bash eval/run_score_mmbench_gpt4o.sh
#
set -euo pipefail

########################################
# >>> put credentials here <<<
OPENAI_API_KEY="${OPENAI_API_KEY:-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}"   # e.g. sk-...
OPENAI_API_URL="${OPENAI_API_URL:-https://api.openai.com/v1/chat/completions}"
# If your provider uses a base URL without /chat/completions, set full chat URL above.
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"
# Bridge local terminal network into this AutoDL box via SSH -R, then point here.
# Existing reverse tunnel on this machine listens at 127.0.0.1:7897.
HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7897}"
HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7897}"
########################################

if [[ -z "${OPENAI_API_KEY}" || "${OPENAI_API_KEY}" == "sk-..." ]]; then
  echo "Please set OPENAI_API_KEY in this script or in the environment." >&2
  exit 2
fi

export OPENAI_API_KEY OPENAI_API_URL OPENAI_MODEL
export https_proxy="${HTTPS_PROXY}" http_proxy="${HTTP_PROXY}"
export HTTPS_PROXY HTTP_PROXY
echo "[proxy] https_proxy=${https_proxy}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_DIR="${ROOT}/eval"
PY="${PY:-/root/miniconda3/envs/CrossMatch/bin/python}"
export PATH="$(dirname "${PY}"):${PATH}"
export PYTHONPATH="${ROOT}/eval:${PYTHONPATH:-}"
export OPENAI_API_KEY OPENAI_API_URL OPENAI_MODEL

# Default: score the three completed full-matrix modes.
# Override with MODE=original|vchd|vchd_ccaw  or INPUT=/path/to.xlsx
RUN_DIR="${RUN_DIR:-${EVAL_DIR}/logs/lavida_llada_full_20260722_235615}"
MODE="${MODE:-}"
INPUT="${INPUT:-}"
GPT_WORKERS="${GPT_WORKERS:-16}"

score_one() {
  local mode="$1"
  local input="$2"
  local out_dir="$3"
  echo "[score] mode=${mode} input=${input}"
  "${PY}" "${EVAL_DIR}/score_mmbench_gpt4o.py" \
    "${input}" \
    --output-dir "${out_dir}" \
    --mode-name "${mode}" \
    --model "${OPENAI_MODEL}" \
    --api-url "${OPENAI_API_URL}" \
    --workers "${GPT_WORKERS}"
}

if [[ -n "${INPUT}" ]]; then
  mode_name="${MODE:-custom}"
  input_dir="$(dirname "${INPUT}")"
  if [[ "$(basename "${input_dir}")" == "submissions" ]]; then
    default_out="$(dirname "${input_dir}")/gpt4o_score"
  else
    default_out="${input_dir}/gpt4o_score"
  fi
  out_dir="${OUTPUT_DIR:-${default_out}}"
  score_one "${mode_name}" "${INPUT}" "${out_dir}"
  exit 0
fi

if [[ -n "${MODE}" ]]; then
  modes=("${MODE}")
else
  modes=(original vchd vchd_ccaw)
fi

for mode in "${modes[@]}"; do
  input="${RUN_DIR}/${mode}/submissions/mmbench_en_dev_results.xlsx"
  if [[ ! -f "${input}" ]]; then
    echo "Missing submission: ${input}" >&2
    exit 1
  fi
  score_one "${mode}" "${input}" "${RUN_DIR}/${mode}/gpt4o_score"
done

echo "All done. Summaries:"
for mode in "${modes[@]}"; do
  summary="${RUN_DIR}/${mode}/gpt4o_score/summary.json"
  echo "---- ${mode} ----"
  "${PY}" -c "
import json
s=json.load(open('${summary}'))
print(f\"circular={s['accuracy_percent']:.2f}%  flat={s['flat_accuracy_percent']:.2f}%  rows={s['rows']}\")
print('${summary}')
"
done
