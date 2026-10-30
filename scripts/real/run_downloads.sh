#!/usr/bin/env bash
# Driver: run all real-data downloads in the background with logs under logs/real/.
# Usage: scripts/real/run_downloads.sh   (re-runnable; completed+verified files are skipped)
set -euo pipefail
P="${COLOC_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; cd "$P"; source env/activate.sh
mkdir -p logs/real
nohup bash -c 'set -euo pipefail; scripts/real/download_finngen.sh data/raw/finngen/r12 4' > logs/real/download_finngen.log 2>&1 &
echo "finngen pid $!"
nohup bash -c 'set -euo pipefail; scripts/real/download_eqtlcat.sh data/raw/eqtlcat 5' > logs/real/download_eqtlcat.log 2>&1 &
echo "eqtlcat pid $!"
