#!/usr/bin/env bash
# Evaluate the post-missing-mask METR-LA checkpoint with 20 true flow samples.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${TSFLOW_PYTHON:-python}"
# Existing checkpoints are multivariate (one joint model over all sensors).
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
  "$PYTHON_BIN" "$ROOT/methods/baselines/TSFlow/run_unified_144.py" \
  --dataset METR-LA --setting multivariate --use-ema --batch-size 64 \
  --sampling-steps 16 --test-probability-samples 20 --evaluate-only \
  --output-dir /tmp/trafficfm_benchmark_runs/TSFlow \
  --metrics-output-dir /tmp/trafficfm_benchmark_runs/TSFlow_probabilistic
