#!/usr/bin/env bash
# Helpers for bridging AutoDL -> your local terminal network.
#
# Prerequisite (run on YOUR LOCAL machine that can reach OpenAI):
#   1) Start a local HTTP/SOCKS proxy, e.g. Clash listening on 7890
#   2) Reverse-tunnel it into AutoDL (copy SSH host/port from AutoDL console):
#
#      ssh -CNg -R 7897:127.0.0.1:7890 -p <SSH_PORT> root@<SSH_HOST>
#
#   Keep that SSH session alive. Then on AutoDL:
#      source /root/autodl-tmp/LaViDa/eval/bridge_terminal_proxy.sh
#      curl -I https://api.openai.com
#
set -euo pipefail

export http_proxy="${http_proxy:-http://127.0.0.1:7897}"
export https_proxy="${https_proxy:-http://127.0.0.1:7897}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"

echo "[bridge] http_proxy=${http_proxy}"
echo "[bridge] https_proxy=${https_proxy}"

if [[ "${1:-}" == "test" ]]; then
  echo "[bridge] probing api.openai.com ..."
  curl -sS --connect-timeout 8 --max-time 20 \
    -o /tmp/openai_bridge_probe.json \
    -w 'http=%{http_code} time=%{time_total}s\n' \
    https://api.openai.com/v1/models || true
  head -c 200 /tmp/openai_bridge_probe.json; echo
  rm -f /tmp/openai_bridge_probe.json
fi
