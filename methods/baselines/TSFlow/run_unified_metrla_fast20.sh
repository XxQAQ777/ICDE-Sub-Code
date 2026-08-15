#!/usr/bin/env bash
# Quick diagnostic only: fixed split, but abbreviated training/validation.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${TSFLOW_PYTHON:-python}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
  "$PYTHON_BIN" "$REPO_ROOT/methods/baselines/TSFlow/run_unified_144.py" \
  --dataset METR-LA --setting univariate --use-ema --epochs 20 --patience 6 \
  --batch-size 256 --train-batches-per-epoch 64 --max-validation-batches 128 \
  --sampling-steps 4 --test-point-samples 1 \
  --output-dir /tmp/trafficfm_benchmark_runs/TSFlow_univariate_fast20
