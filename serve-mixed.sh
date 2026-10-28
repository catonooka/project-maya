#!/bin/bash
# Mixed-lane S5 boot (0xSero 2.7, waterline placement + swap cache), TP1 layer split, port 8080.
# Run via: sudo systemd-run --unit=maya-ds4 --working-directory=/home/aisever/maya-ds4-main bash serve-mixed.sh
cd /home/aisever/maya-ds4-main
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
# campaign base env (the 2.9 FINAL line)
export MAYA_DS4_DSPARK_CUM=0.5
export DS4_CACHE_THREADS=4
export MAYA_DS4_STREAM=0
export MAYA_DS4_PROMPT_SKIP=1
export MAYA_DS4_Q8_BATCH=1
export MAYA_DS4_SLOT_KEEP=1
# S5 mixed-lane line
export DS4_MIX_SWAP=1
export DS4_PROMPT_SWAPS=${DS4_PROMPT_SWAPS:-4}
export DS4_WARM_CAP_GB=${DS4_WARM_CAP_GB:-90}
export DS4_LOAD_THREADS=${DS4_LOAD_THREADS:-24}
# A/B passthroughs (set before calling to override)
exec /home/aisever/maya-ds4-main/.venv/bin/python serve/server.py \
  --engine strata --config maya-ds4.json --port 8080 \
  > /home/aisever/maya-ds4-main/ds4-mixed.log 2>&1
