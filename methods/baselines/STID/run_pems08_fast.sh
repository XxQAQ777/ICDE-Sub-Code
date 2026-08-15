#!/usr/bin/env bash
# Native PEMS08 protocol: existing fixed 12->12 index, fast budget.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${TRAFFICFM_PYTHON:-python}"
RUN_ROOT="/tmp/trafficfm_benchmark_runs/STID/PEMS08/seed_42"
"$PYTHON_BIN" "$ROOT/methods/baselines/STID/prepare_pems08.py" \
  --source "$ROOT/data/processed/HimNet/PEMS08" --output "$RUN_ROOT/data"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
  "$PYTHON_BIN" "$ROOT/methods/baselines/STID/train.py" \
  --data "$RUN_ROOT/data" --num_nodes 170 --input_len 12 --output_len 12 \
  --epochs 40 --patience 8 --batch_size 32 --test_batch_size 32 --seed 42 \
  --horizons 3,6,12 --save "$RUN_ROOT/checkpoints/best_model.pt"
