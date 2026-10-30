#!/usr/bin/env bash
# Build the frozen screening set screen_v1 (DESIGN §6.1), one process per mismatch level.
set -euo pipefail
cd /cluster/scratch/mumarc/hackathon_eqtl_gwas
source env/activate.sh
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LVL=$1
flock -s results/.machine.lock python -m sim.cells --set screen_v1 --split valid --seed 20261027 --per-bucket 500 \
  --K 200 --M 3 --cells "$LVL|P1|r_p=0.0" "$LVL|P3|rho=0.5" --stats-suffix ".$LVL"
