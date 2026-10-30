#!/usr/bin/env bash
# Convert 1000G 30x GRCh38 phased VCFs to plink2 pgen for unrelated EUR (503) + AFR (661) samples.
# Keeps biallelic SNVs and indels (no SVs), variant ID = chr:pos:ref:alt, and EUR MAF >= 0.005.
# Usage: scripts/prep_1kg_pgen.sh [rawdir] [outdir]
set -euo pipefail
RAW=${1:-data/raw/1kg30x}; OUT=${2:-data/processed/1kg}
BIN=env/tools/bin; mkdir -p "$OUT"
awk 'NR>1 && ($3=="EUR"||$3=="AFR"){print $1}' "$RAW/phase3_2504.panel" > "$OUT/samples_eur_afr.txt"
awk 'NR>1 && $3=="EUR"{print $1}' "$RAW/phase3_2504.panel" > "$OUT/samples_eur.txt"
awk 'NR>1 && $3=="AFR"{print $1}' "$RAW/phase3_2504.panel" > "$OUT/samples_afr.txt"
awk 'NR>1{print $1"\t"$2"\t"$3"\t"$4}' "$RAW/phase3_2504.panel" > "$OUT/panel.tsv"
conv() {
  set -euo pipefail
  c=$1
  $BIN/bcftools view --threads 2 -S "$OUT/samples_eur_afr.txt" --force-samples -v snps,indels -m2 -M2 "$RAW/chr$c.vcf.gz" -Ou 2> >(grep -v "bcf_hdr_check_sanity" >&2) \
   | $BIN/bcftools annotate -x INFO,^FORMAT/GT --set-id '%CHROM:%POS:%REF:%ALT' -Ob -o "$OUT/tmp_chr$c.bcf" 2> >(grep -v "bcf_hdr_check_sanity" >&2)
  $BIN/plink2 --bcf "$OUT/tmp_chr$c.bcf" --make-pgen --out "$OUT/tmp_chr$c" --threads 2 --memory 3500 >/dev/null
  $BIN/plink2 --pfile "$OUT/tmp_chr$c" --keep "$OUT/samples_eur.txt" --freq --out "$OUT/tmp_chr$c.eur" --threads 2 --memory 3500 >/dev/null
  awk 'NR>1{m=$5; if(m>0.5)m=1-m; if(m>=0.005) print $2}' "$OUT/tmp_chr$c.eur.afreq" > "$OUT/tmp_chr$c.keep"
  $BIN/plink2 --pfile "$OUT/tmp_chr$c" --extract "$OUT/tmp_chr$c.keep" --make-pgen --output-chr chrM --out "$OUT/chr$c" --threads 2 --memory 3500 >/dev/null
  rm -f "$OUT"/tmp_chr$c.*
  n=$(grep -vc '^#' "$OUT/chr$c.pvar"); span=$(grep -v '^#' "$OUT/chr$c.pvar" | awk 'NR==1{a=$2} {b=$2} END{print int((b-a)/1e6)}')
  echo "chr$c $n variants over ${span} Mb ($((n/(span+1))) per Mb)"
  # sanity: 1000G EUR at MAF>=0.005 has ~4-6k variants/Mb; fail loudly if far below
  if (( n/(span+1) < 2000 )); then echo "ERROR chr$c variant density too low" >&2; exit 1; fi
}
export -f conv; export OUT RAW BIN
CHRS=${CHRS:-$(seq 1 22)}
scripts/verify_1kg_md5.sh "$RAW" > /dev/null
echo $CHRS | tr " " "\n" | xargs -P 22 -I{} bash -c 'conv {}'
