#!/usr/bin/env bash
# MacDonald et al. 2022 native-GRCh38 LD blocks (jmacdon/LDblocks_GRCh38) and GENCODE v39 (matches eQTL Catalogue r6+).
set -euo pipefail
OUT=${1:-data/raw/annot}; mkdir -p "$OUT"
LB=https://raw.githubusercontent.com/jmacdon/LDblocks_GRCh38/master/data
for f in pyrho_EUR_LD_blocks.bed deCODE_EUR_LD_blocks.bed pyrho_AFR_LD_blocks.bed; do curl -sSfL -o "$OUT/$f" "$LB/$f"; done
curl -sSfL -o "$OUT/gencode.v39.basic.annotation.gtf.gz" https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_39/gencode.v39.basic.annotation.gtf.gz
(cd "$OUT" && sha256sum *.bed *.gtf.gz > SHA256SUMS)
