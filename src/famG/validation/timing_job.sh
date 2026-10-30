#!/usr/bin/env bash
# DESIGN §7.7 timing job for G-mm (one gated orch job in the exclusive slot; see reports/p9.famG_mm.md).
#  1. stream-count pick: 2-config bench (streams 1 vs 3) on 512 strided slice loci (diagnostic, recorded);
#  2. timing_mm run: page-cache prewarm, then G-mm x3, coloc_susie_mm x3, coloc-SuSiE 0.14.2 x1 (PI fallback set
#     first), optionally more 0.14.2 repeats via PLAN.
set -euo pipefail
export PYTHONPATH=$ORCH_CODE/src:${PYTHONPATH:-}
O=${OUT:-results/famG_mm/timing_v1}
PLAN=${PLAN:-gmm:3,mm:3,cs:1}
mkdir -p "$O"
python -m famG.validation.bench_mm --n 512 --batch 512 --mem-gb 9.0 --dtype float64 \
    --sweep 1024:8:1.5:1,1024:8:1.5:3 --out "$O/stream_pick.json" 2>&1 | grep '^{'
S=$(python -c "import json; r = json.load(open('$O/stream_pick.json')); print(max(r, key=lambda x: x['fit_loci_per_s'])['streams'])")
echo "picked streams=$S"
python -m famG.validation.timing_mm run --ids results/famG_mm/slice2000_v1/ids.txt --plan "$PLAN" --out "$O" \
    --overrides "{\"gmm\": {\"n_workers\": 1, \"batch_size\": 2000, \"kwargs\": {\"streams\": $S, \"mem_budget_gb\": 9.0}}}"
