#!/usr/bin/env bash
set -euo pipefail
cd /cluster/scratch/mumarc/hackathon_eqtl_gwas; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source env/activate.sh
for P in FIN L6c ASW_ACB B; do
  PYTHONPATH=src python -m sim.validate_mosaic --pool $P --n-windows 40 --ne 250 500 1000 2000 3000 --out src/sim/validation/mosaic_validation_$P.json > logs/sim/validate_mosaic_$P.log 2>&1
done
