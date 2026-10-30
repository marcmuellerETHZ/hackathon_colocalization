#!/usr/bin/env bash
# FinnGen R12 (GRCh38; Kurki et al. 2023 Nature) summary statistics + SuSiE fine-mapping for the traits
# in scripts/real/finngen_traits.tsv. Access: public bucket after the FinnGen results access form; no
# redistribution of raw files; acknowledge FinnGen (finngen.fi/en/access_results).
# Usage: scripts/real/download_finngen.sh [outdir=data/raw/finngen/r12] [parallel=4]
# Verification: upstream GCS md5Hash for every object, size, full gzip/BGZF test, sha256 recorded.
# Also stores the R12 manifest, the Kanta lab documentation and the ld_matrix README + tabix indices
# (the LD matrix itself is queried remotely by region with tabix; 200 GB is not downloaded).
set -euo pipefail
P="${COLOC_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$P"
OUT=${1:-data/raw/finngen/r12}; NPAR=${2:-4}
BUCKET=finngen-public-data-r12; GS=https://storage.googleapis.com/$BUCKET
PY=$P/.venv/bin/python
source scripts/real/lib_fetch.sh
mkdir -p "$OUT/listing"
lst="$OUT/listing/objects.tsv"; : > "$lst.tmp"
for pre in summary_stats/finngen_R12_manifest.tsv lab_values/analysis_documentation/ ld_matrix/; do
  $PY -I scripts/real/gcs_list.py $BUCKET "$pre" >> "$lst.tmp"
done
tail -n +2 scripts/real/finngen_traits.tsv | while IFS=$'\t' read -r ph kind label ssp sup; do
  $PY -I scripts/real/gcs_list.py $BUCKET "$ssp/finngen_R12_$ph.gz" >> "$lst.tmp"
  $PY -I scripts/real/gcs_list.py $BUCKET "$sup/finngen_R12_$ph.SUSIE." >> "$lst.tmp"
done
sort -u "$lst.tmp" > "$lst"; rm "$lst.tmp"
# keep: sumstats .gz/.tbi, SUSIE cred/cred_99/snp(+tbi), manifest, docs, ld README + .tbi
awk -F'\t' '$1 ~ /(summary_stats\/release|lab_values\/summary_stats)\/finngen_R12_[^.]+\.gz(\.tbi)?$/ ||
            $1 ~ /SUSIE\.(cred|cred_99|snp)\.bgz(\.tbi)?$/ || $1 ~ /manifest\.tsv$/ ||
            $1 ~ /analysis_documentation\// || $1 ~ /ld_matrix\/(README|.*\.tbi)$/' "$lst" > "$OUT/listing/selected.tsv"
n=$(wc -l < "$OUT/listing/selected.tsv"); echo "selected $n objects, $(awk -F'\t' '{s+=$2} END{printf "%.2f", s/1e9}' "$OUT/listing/selected.tsv") GB"
nt=$(($(wc -l < scripts/real/finngen_traits.tsv)-1))
for ph in $(tail -n +2 scripts/real/finngen_traits.tsv | cut -f1); do
  for suf in ".gz" ".gz.tbi" ".SUSIE.cred.bgz" ".SUSIE.snp.bgz" ".SUSIE.snp.bgz.tbi"; do
    grep -q "finngen_R12_$ph$suf"$'\t' "$OUT/listing/selected.tsv" || { echo "ERROR missing $ph$suf in bucket listing" >&2; exit 1; }
  done
done
echo "all $nt traits have sumstats + SuSiE objects"
export -f fetch remote_size md5_to_hex record_sha256
awk -F'\t' -v gs="$GS" -v out="$OUT" '{n=$1; sub(/^.*\//,"",n); d=$1; sub(/\/[^\/]*$/,"",d); print gs"/"$1, out"/"d"/"n, $3}' "$OUT/listing/selected.tsv" \
  | xargs -P "$NPAR" -L 1 bash -c 'set -euo pipefail; fetch "$0" "$1" "${2:-}"'
echo "finngen downloads complete"; du -sh "$OUT"
