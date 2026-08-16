#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${TRAFFICFM_PYTHON:-python}"
: "${TRAFFICFM_CHECKPOINT:?Set TRAFFICFM_CHECKPOINT to the Table-II METR-LA checkpoint}"
cd "$ROOT/methods/TrafficFM"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" "$PYTHON_BIN" table2_probabilistic_evaluator.py \
  --dataset METR-LA --dataset-dir "$ROOT/data/processed/STD-MAE/METR-LA" \
  --adjdata "$ROOT/data/processed/STD-MAE/METR-LA/adj_mx.pkl" \
  --checkpoint "$TRAFFICFM_CHECKPOINT" --output-dir /tmp/trafficfm_benchmark_runs/TrafficFM/TableII/METR_LA \
  --num-nodes 207 --num-samples 10 --sampling-steps 10 --batch-size 4
