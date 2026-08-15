#!/usr/bin/env bash
# Diagnostic-only point forecast: not the formal 5-sample / 16-step report.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${TSFLOW_PYTHON:-python}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
  "$PYTHON_BIN" "$ROOT/methods/baselines/TSFlow/run_unified_144.py" \
  --dataset METR-LA --batch-size 4 --sampling-steps 4 --test-point-samples 1 --evaluate-only \
  --metrics-output-dir /tmp/trafficfm_benchmark_runs/TSFlow_diagnostic
