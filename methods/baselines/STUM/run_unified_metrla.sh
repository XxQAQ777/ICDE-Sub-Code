#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${TRAFFICFM_PYTHON:-python}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
  "$PYTHON_BIN" "$REPO_ROOT/methods/baselines/STUM/run_unified_144.py" \
  --dataset METR-LA --epochs 100 --patience 20 --batch-size 4
