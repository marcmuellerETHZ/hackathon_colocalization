#!/usr/bin/env bash
# Family C: development data from the simulation agent's (uncommitted, M3.5-in-progress) train API.
# valid_dev: validation windows, seed 20261029, all levels incl. held-out L6, W/G down-sampled at generation.
# train_dev: train windows, seed 20261030, L6 held out, natural class mixture, NW workers (disjoint window steps).
# Usage: bash src/famC/run_gen_dev.sh <n_valid> <n_train_per_worker> <n_workers>
set -euo pipefail
P="${COLOC_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source $P/env/activate.sh
export PYTHONPATH=$P/src
cd $P
NV=${1:-15000}; NT=${2:-60000}; NW=${3:-3}
mkdir -p logs/famC
flock -s results/.machine.lock python -m famC.gen_shards --set valid_dev --split valid --n-loci $NV --lpw 128 \
  --accept '{"S":1,"D":1,"U":1,"G":0.5,"W":0.3}' > logs/famC/gen_valid_dev.log 2>&1 &
for w in $(seq 0 $((NW-1))); do
  flock -s results/.machine.lock python -m famC.gen_shards --set train_dev --split train --n-loci $NT --lpw 128 \
    --worker $w --n-workers $NW > logs/famC/gen_train_dev_w$w.log 2>&1 &
done
wait
echo "all generators finished"
