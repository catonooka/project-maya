#!/bin/bash
# A/B boot: decode experiments. Usage: serve-ds4-ab.sh <TAG> [extra env as KEY=VAL...]
# TAG selects the env block; extra KEY=VAL pairs appended.
cd /home/aisever/maya-ds4-main
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
TAG="$1"; shift
case "$TAG" in
  cum04) export STRATA_DS4_DSPARK_CUM=0.4 ;;
  cum03) export STRATA_DS4_DSPARK_CUM=0.3 ;;
  cum06) export STRATA_DS4_DSPARK_CUM=0.6 ;;
  ahead2) export STRATA_DS4_DSPARK_CUM=0.5; export DS4_AHEAD_EXTRA=2 ;;
  recent64) export STRATA_DS4_DSPARK_CUM=0.5; export DS4_RECENT=64 ;;
  *) : ;;  # plain baseline
esac
for kv in "$@"; do export "$kv"; done
exec /home/aisever/maya-ds4-main/.venv/bin/python serve/server.py \
  --engine strata --config maya-ds4.json --port 8080 \
  > /home/aisever/maya-ds4-main/ds4-baseline.log 2>&1
