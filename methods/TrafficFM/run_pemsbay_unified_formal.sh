#!/usr/bin/env bash
# Paper-grade PEMS-BAY 144->144 training on the complete fixed STD-MAE split.
# This intentionally has no per-epoch batch cap.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="/home/guyuanhao/.conda/envs/trafficfm/bin/python"
RUN_ROOT="/tmp/trafficfm_benchmark_runs/TrafficFM/PEMS_BAY/formal_seed_99"
DATA_ROOT="$ROOT/data/processed/STD-MAE/PEMS-BAY"

cd "$ROOT/methods/TrafficFM"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" train_pems.py \
  --model default \
  --train_objective model \
  --devices 0 \
  --unified_data "$DATA_ROOT" \
  --skip_final_test \
  --adjdata "$DATA_ROOT/adj_mx.pkl" \
  --adjtype doubletransition \
  --gcn_bool \
  --addaptadj \
  --seq_length 144 \
  --num_nodes 325 \
  --in_dim 3 \
  --nhid 16 \
  --blocks 8 \
  --layers 3 \
  --batch_size 4 \
  --epochs 30 \
  --patience 8 \
  --max_train_batches 0 \
  --max_validation_batches 0 \
  --print_every 250 \
  --learning_rate 0.0001 \
  --dropout 0.5 \
  --weight_decay 0.0005 \
  --seed 99 \
  --save "$RUN_ROOT/checkpoints/trafficfm"
