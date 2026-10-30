#!/usr/bin/env bash
# eQTL Catalogue r7 (GRCh38, CC BY 4.0; Kerimov et al. 2021 Nat Genet) — metadata + curated ge datasets.
# Usage: scripts/real/download_eqtlcat.sh [outdir=data/raw/eqtlcat] [parallel=4]
# Downloads, per dataset in scripts/real/eqtlcat_datasets.tsv: *.all.tsv.gz (+.tbi), *.permuted.tsv.gz,
# SuSiE *.credible_sets.tsv.gz and *.lbf_variable.txt.gz. No upstream checksums exist on the EBI FTP:
# verification = Content-Length + full gzip test + sha256 recorded (SHA256SUMS per directory).
# The metadata tables come from eQTL-Catalogue-resources at a pinned commit.
set -euo pipefail
P="${COLOC_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$P"
OUT=${1:-data/raw/eqtlcat}; NPAR=${2:-4}
COMMIT=c90087edbdde407ccbf88611deeff0a43e9a4d89   # eQTL-Catalogue-resources master on 2026-10-26
GH=https://raw.githubusercontent.com/eQTL-Catalogue/eQTL-Catalogue-resources/$COMMIT
FTP=https://ftp.ebi.ac.uk/pub/databases/spot/eQTL
source scripts/real/lib_fetch.sh
mkdir -p "$OUT/meta"
for f in tabix/tabix_ftp_paths.tsv data_tables/dataset_metadata_r7.tsv data_tables/population_assignments.tsv \
         data_tables/dataset_id_map.tsv data_tables/sceQTL_num_individuals_per_dataset.tsv; do
  fetch "$GH/$f" "$OUT/meta/$(basename "$f")"
done
# eQTL Catalogue sample metadata (Zenodo 4678936, CC BY 4.0; upstream md5) — GEUVADIS individuals for R4b.
fetch https://zenodo.org/api/records/4678936/files/GEUVADIS.tsv/content "$OUT/meta/zenodo4678936_GEUVADIS.tsv" 7514d69f70bb1bdd252941afb44c32ec
# GEUVADIS sample/data relationship file (ArrayExpress E-GEUV-1; 1000 Genomes sample IDs; no upstream checksum)
fetch https://ftp.ebi.ac.uk/biostudies/fire/E-GEUV-/001/E-GEUV-1/Files/E-GEUV-1.sdrf.txt "$OUT/meta/E-GEUV-1.sdrf.txt"

# consistency: every curated dataset must be a ge dataset in the r7 path index with the expected study id
tail -n +2 scripts/real/eqtlcat_datasets.tsv | while IFS=$'\t' read -r qtd qts lab grp rest; do
  awk -F'\t' -v q="$qtd" -v s="$qts" '$2==q && $1==s && $9=="ge"{f=1} END{exit !f}' "$OUT/meta/tabix_ftp_paths.tsv" \
    || { echo "ERROR $qtd/$qts not a ge dataset in r7 index" >&2; exit 1; }
done

jobs_file=$(mktemp /tmp/claude-eqtlcat.XXXXXX)
tail -n +2 scripts/real/eqtlcat_datasets.tsv | while IFS=$'\t' read -r qtd qts rest; do
  d="$OUT/r7/$qtd"
  for f in all.tsv.gz all.tsv.gz.tbi permuted.tsv.gz; do echo "$FTP/sumstats/$qts/$qtd/$qtd.$f $d/$qtd.$f"; done
  for f in credible_sets.tsv.gz lbf_variable.txt.gz; do echo "$FTP/susie/$qts/$qtd/$qtd.$f $d/$qtd.$f"; done
done > "$jobs_file"
export -f fetch remote_size md5_to_hex record_sha256
# largest files first are interleaved by xargs order; each job fails loudly, xargs exits non-zero on any failure
xargs -P "$NPAR" -L 1 bash -c 'set -euo pipefail; fetch "$0" "$1"' < "$jobs_file"
rm -f "$jobs_file"
echo "eqtlcat downloads complete"
du -sh "$OUT"
