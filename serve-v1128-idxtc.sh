#!/bin/bash
# A/B boot: same as baseline but DS4_INDEX_TC=1 (tensor-core int8 indexer scoring, exact).
cd /home/aisever/maya-ds4-main
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export DS4_INDEX_TC=1
export DS4_RECALL=0
exec /home/aisever/maya-ds4-main/.venv/bin/python serve/server.py \
  --engine strata --config maya-ds4.json --port 8080 \
  > /home/aisever/maya-ds4-main/ds4-baseline.log 2>&1
