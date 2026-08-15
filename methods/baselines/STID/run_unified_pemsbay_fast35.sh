#!/usr/bin/env bash
# STID on the existing fixed STD-MAE PEMS-BAY 144->144 split.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${TRAFFICFM_PYTHON:-python}"
RUN_ROOT="/tmp/trafficfm_benchmark_runs/STID"
cd "$ROOT/methods/baselines/STID"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" "$PYTHON_BIN" run_unified_144.py \
  --dataset PEMS-BAY --epochs 35 --patience 8 --batch-size 16 --seed 99 --output-dir "$RUN_ROOT"
