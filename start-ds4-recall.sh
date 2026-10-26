#!/bin/bash
# Recall boot: pr344 + Recall drafter + qsa-select identity skip, TP1, port 8080.
# The config's exe stays maya-ds4-pr344; this script points at the new binary via DS4_EXE override.
cd /home/aisever/maya-ds4-main
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export DS4_EXE=/home/aisever/maya-ds4-main/build-ds4/maya-ds4-recall
exec /home/aisever/maya-ds4-main/.venv/bin/python serve/server.py \
  --engine strata --config maya-ds4.json --port 8080 \
  > /home/aisever/maya-ds4-main/ds4-recall.log 2>&1
