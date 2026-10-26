#!/bin/bash
# Baseline boot: DS4 v1.1.24 engine, TP2, both 3090s, port 8080.
# Usage: bash serve-v1124-baseline.sh   (run via sudo systemd-run, see start-ds4.sh)
cd /home/aisever/maya-ds4-main
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
exec /home/aisever/maya-ds4-main/.venv/bin/python serve/server.py \
  --engine strata --config maya-ds4.json --port 8080 \
  > /home/aisever/maya-ds4-main/ds4-baseline.log 2>&1
