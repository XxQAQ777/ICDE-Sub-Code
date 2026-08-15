#!/usr/bin/env bash
# Full fixed test split, accelerated 1-sample / 4-step point forecast.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${TSFLOW_PYTHON:-python}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
  "$PYTHON_BIN" "$REPO_ROOT/methods/baselines/TSFlow/run_unified_144.py" \
  --dataset METR-LA --setting univariate --use-ema --batch-size 512 \
  --sampling-steps 4 --test-point-samples 1 --evaluate-only \
  --output-dir /tmp/trafficfm_benchmark_runs/TSFlow_univariate_fast20
