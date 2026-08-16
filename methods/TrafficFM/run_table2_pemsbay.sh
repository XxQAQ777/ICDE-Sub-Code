#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${TRAFFICFM_PYTHON:-python}"
: "${TRAFFICFM_CHECKPOINT:?Set TRAFFICFM_CHECKPOINT to the Table-II PEMS-BAY checkpoint}"
cd "$ROOT/methods/TrafficFM"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" "$PYTHON_BIN" table2_probabilistic_evaluator.py \
  --dataset PEMS-BAY --dataset-dir "$ROOT/data/processed/STD-MAE/PEMS-BAY" \
  --adjdata "$ROOT/data/processed/STD-MAE/PEMS-BAY/adj_mx.pkl" \
  --checkpoint "$TRAFFICFM_CHECKPOINT" --output-dir /tmp/trafficfm_benchmark_runs/TrafficFM/TableII/PEMS_BAY \
  --num-nodes 325 --num-samples 10 --sampling-steps 10 --batch-size 4
