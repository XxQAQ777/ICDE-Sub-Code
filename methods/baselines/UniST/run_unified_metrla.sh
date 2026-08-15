#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${TRAFFICFM_PYTHON:-python}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
  "$PYTHON_BIN" "$REPO_ROOT/methods/baselines/UniST/run_unified_144.py" \
  --dataset METR-LA --epochs 50 --patience 10 --batch-size 1 --t-patch-size 12
