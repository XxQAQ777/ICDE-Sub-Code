#!/usr/bin/env bash
# Fast screening run: same fixed 144->144 split, max 35 epochs, patience 8.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${TRAFFICFM_PYTHON:-python}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
  "$PYTHON_BIN" "$REPO_ROOT/methods/baselines/STUM/run_unified_144.py" \
  --dataset PEMS-BAY --epochs 35 --patience 8 --batch-size 2
