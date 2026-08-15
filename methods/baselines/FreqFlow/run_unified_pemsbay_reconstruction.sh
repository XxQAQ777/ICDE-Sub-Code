#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${TRAFFICFM_PYTHON:-python}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
  "$PYTHON_BIN" "$REPO_ROOT/methods/baselines/FreqFlow/run_unified_144_reconstruction.py" \
  --dataset PEMS-BAY --epochs 100 --patience 20 --batch-size 1
