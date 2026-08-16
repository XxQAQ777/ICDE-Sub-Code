#!/usr/bin/env bash
# Table-II entry point: exactly ten trajectories and the documented W1 pool.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash "$ROOT/methods/TrafficFM/run_table2_pemsbay.sh"
