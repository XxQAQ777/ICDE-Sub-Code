#!/usr/bin/env bash
# Backward-compatible name for the Table-II PEMS-BAY evaluator.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash "$ROOT/methods/TrafficFM/run_table2_pemsbay.sh"
