#!/usr/bin/env bash
# Real-data arms (DESIGN §2.5): build check, candidate pairs (+R1), train/valid locus sets, verification.
# Prerequisites: scripts/real/run_downloads.sh finished; datasets/splits/blocks_v1.tsv + genes_v1.tsv.gz and
# data/processed/variants/chrN.parquet from the data agent.
# Usage: scripts/real/build_real_arms.sh [workers=32]
set -euo pipefail
P="${COLOC_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; cd "$P"; source env/activate.sh
export PYTHONPATH=src
W=${1:-32}
mkdir -p logs/real
grep -q "eqtlcat downloads complete" logs/real/download_eqtlcat.log
grep -q "finngen downloads complete" logs/real/download_finngen.log
python -m pytest -q tests/data_real -p no:cacheprovider 2>&1 | tee logs/real/tests.log
python -m data.real.qc_build 2>&1 | tee logs/real/qc_build.log
python -m data.real.pairs --workers 16 2>&1 | tee logs/real/pairs.log
python -m data.real.extract --workers "$W" 2>&1 | tee logs/real/extract.log
python -m data.real.rho_update --workers "$W" 2>&1 | tee logs/real/rho_update.log
python -m data.real.verify 2>&1 | tee logs/real/verify.log
