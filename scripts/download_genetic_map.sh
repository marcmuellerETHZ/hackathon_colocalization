#!/usr/bin/env bash
# Beagle GRCh38 genetic maps (plink format), derived from HapMap II; used for haplotype-mosaic simulation.
set -euo pipefail
OUT=${1:-data/raw/genmap}; mkdir -p "$OUT"
curl -sSfL -o "$OUT/plink.GRCh38.map.zip" https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh38.map.zip
(cd "$OUT" && unzip -oq plink.GRCh38.map.zip && sha256sum plink.GRCh38.map.zip > SHA256SUMS)
