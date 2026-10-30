#!/usr/bin/env bash
# Positions carrying >1 SNV/indel record (split multiallelics, SNV+indel at one POS) in the RAW
# 1000G 30x VCF, counting only records polymorphic (AC >= 1) in our EUR+AFR sample subset.
# Done on the raw VCF because the pgen was already filtered at EUR MAF >= 0.005, which can hide
# the second record of a split multiallelic (e.g. an allele common in AFR but rare in EUR).
# SVs are excluded (-v snps,indels), matching scripts/prep_1kg_pgen.sh.
# Tools: env/tools/bin (bcftools 1.24), the PI-managed tools env.
# Usage: src/data/multirecord_positions.sh <chr-number>
# Output: data/processed/variants/multirec/chrN.pos (POS<TAB>n_records, only n_records > 1)
#         data/processed/variants/multirec/chrN.summary (chr, n polymorphic snv/indel records, n multi-record positions)
set -euo pipefail
c=$1
P="${COLOC_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RAW=$P/data/raw/1kg30x; OUT=$P/data/processed/variants/multirec; BIN=$P/env/tools/bin
mkdir -p "$OUT"
# unique per-run temp dir: concurrent runs on the same chromosome cannot clobber each other;
# the final mv is atomic, so the last finished run wins with a complete file.
TMPD=$(mktemp -d "$OUT/tmp_chr$c.XXXXXX"); TMP=$TMPD/chr$c
trap 'rm -rf "$TMPD"' EXIT
"$BIN/bcftools" view --threads 2 -S "$P/data/processed/1kg/samples_eur_afr.txt" --force-samples \
    -v snps,indels -c 1 -Ou "$RAW/chr$c.vcf.gz" 2> >(grep -v "bcf_hdr_check_sanity" >&2) \
  | "$BIN/bcftools" query -f '%POS\n' 2> >(grep -v "bcf_hdr_check_sanity" >&2) > "$TMP.allpos"
sort -n -S 1G -T "$TMPD" "$TMP.allpos" | uniq -c | awk '$1>1{print $2"\t"$1}' > "$TMP.pos"
n_all=$(wc -l < "$TMP.allpos")
n_multi=$(wc -l < "$TMP.pos")
echo -e "chr$c\t$n_all\t$n_multi" > "$TMP.summary"
mv "$TMP.pos" "$OUT/chr$c.pos"
mv "$TMP.summary" "$OUT/chr$c.summary"
cat "$OUT/chr$c.summary"
