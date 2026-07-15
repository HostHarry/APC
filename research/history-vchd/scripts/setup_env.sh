#!/usr/bin/env bash
# History-VCHD cloud environment bootstrap.
#
# Usage:
#   bash scripts/setup_env.sh                  # into current active env
#   PY=/opt/conda/bin/python bash scripts/setup_env.sh   # explicit python
#
# What this does:
#   1. Install PyTorch + Transformers stack (versions frozen by MMaDA).
#   2. Install MMaDA runtime deps (from MMaDA/requirements.txt).
#   3. Editable-install VLMEvalKit (needed for run.py + vlmeval package).
#   4. Do NOT download model weights (fetched lazily on first run).
#
# On success you can immediately run scripts/prepare_datasets.sh, then
# scripts/run_history_vchd.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMADA_ROOT="${REPO_ROOT}/MMaDA"
VLMEVAL_ROOT="${MMADA_ROOT}/evaluation/VLMEvalKit"

PY="${PY:-python}"
PIP="${PIP:-${PY} -m pip}"

echo "[setup] repo root       : ${REPO_ROOT}"
echo "[setup] MMaDA root      : ${MMADA_ROOT}"
echo "[setup] VLMEvalKit root : ${VLMEVAL_ROOT}"
echo "[setup] python          : $(${PY} -c 'import sys; print(sys.executable)')"

# 1. Torch stack. Pin cu121 wheels; users on other CUDA majors should adjust.
if ! ${PY} -c 'import torch' 2>/dev/null; then
  echo "[setup] installing PyTorch 2.4 (cu121) ..."
  ${PIP} install --extra-index-url https://download.pytorch.org/whl/cu121 \
    'torch==2.4.1' 'torchvision==0.19.1' 'torchaudio==2.4.1'
else
  echo "[setup] PyTorch already present: $(${PY} -c 'import torch; print(torch.__version__)')"
fi

# 2. MMaDA runtime deps.
echo "[setup] installing MMaDA requirements ..."
${PIP} install -r "${MMADA_ROOT}/requirements.txt"

# 3. VLMEvalKit (editable).
echo "[setup] installing VLMEvalKit (editable) ..."
${PIP} install -e "${VLMEVAL_ROOT}"

# 4. Extra tooling used by cloud scripts.
${PIP} install --upgrade 'huggingface_hub>=0.24' 'openpyxl>=3.1'

echo ""
echo "[setup] sanity check: importing decoding + models ..."
PYTHONPATH="${MMADA_ROOT}:${PYTHONPATH:-}" ${PY} - <<'PY_SANITY'
import importlib, sys
for mod in ('decoding', 'models', 'decoding.decoder', 'decoding.mmada_adapter',
           'decoding.history', 'decoding.window', 'decoding.selector'):
    importlib.import_module(mod)
    print(f"  ok  {mod}")
from decoding import VCHDDecodeConfig
cfg = VCHDDecodeConfig()
cfg.validate()
print(f"  ok  VCHDDecodeConfig defaults validate")
print("[setup] all imports OK.")
PY_SANITY

echo ""
echo "[setup] done. Next:"
echo "  bash scripts/prepare_datasets.sh    # ~10 GB parquet -> ~5 GB TSV"
echo "  bash scripts/run_history_vchd.sh    # runs the History-VCHD sweep"
