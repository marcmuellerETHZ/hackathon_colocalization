#!/usr/bin/env bash
# Spot check part 2 (CPU): fresh run_susie016_mm.R on one locus vs stored r016 JSON, then G-mm CPU torch fp64 (no Triton) vs fresh R.
# Usage: S=<scratch dir under /tmp/claude-*> bash spot_pip.sh ; run from the project root after `source env/activate.sh`.
set -euo pipefail
export PYTHONPATH=src R_PROFILE_USER=/dev/null OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
L='valid_grid_v1__L2_P1_r_p=0.0__G__0004402'
mkdir -p $S/work && ln -sfn $PWD/runs/famG_mm_identity_ref_v2/out/loci/$L $S/work/$L && echo $L > $S/work/ids.txt
R_LIBS=$PWD/env/rlib_susie016 Rscript src/famG/validation/run_susie016_mm.R $S/work $S/r $L
python - "$S/r/$L.json" "results/famG_mm/identity_v2/r016/$L.json" <<'PY'
import json, sys, numpy as np
a, b = (json.load(open(f)) for f in sys.argv[1:3])
for t in ("t1", "t2"): print(t, "fresh-vs-stored R max|dPIP|", np.abs(np.array(a[t]["pip"]) - np.array(b[t]["pip"])).max())
print("pairs PP equal:", a["pairs"]["PP"] == b["pairs"]["PP"])
PY
python -m famG.validation.compare_mm --work $S/work --rdir $S/r --out $S/cmp --device cpu --no-triton
