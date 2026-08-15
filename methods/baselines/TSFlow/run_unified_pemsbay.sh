#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${TSFLOW_PYTHON:-python}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
  "$PYTHON_BIN" "$REPO_ROOT/methods/baselines/TSFlow/run_unified_144.py" \
  --dataset PEMS-BAY --setting univariate --use-ema --epochs 100 --patience 20 \
  --batch-size 128 --train-batches-per-epoch 128 \
  --output-dir /tmp/trafficfm_benchmark_runs/TSFlow_univariate
