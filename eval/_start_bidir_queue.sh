#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/LaViDa
RUN_ROOT=$(cat /tmp/lavida_bidir_new_root.txt)
RUN_ROOT="$RUN_ROOT" bash eval/run_fixed_bias_cache_reruns.sh /autodl-fs/data/lavida-ckpts/lavida-llada-reason 2>&1 | tee "${RUN_ROOT}.screen.log"
