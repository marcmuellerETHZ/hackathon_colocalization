#!/usr/bin/env bash
# Download 1000 Genomes 30x (NYGC) phased SNV/INDEL/SV panel, GRCh38, autosomes.
# Source: Byrska-Bishop et al. 2022 Cell; open access (IGSR data reuse policy, no restrictions).
# Usage: scripts/download_1kg30x.sh [outdir]
set -euo pipefail
OUT=${1:-data/raw/1kg30x}
B=http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV
mkdir -p "$OUT"
for f in 20220804_manifest.txt README_1kGP_phased_panel_110722.pdf change_log.txt; do curl -sSf -o "$OUT/$f" "$B/$f"; done
curl -sSf -o "$OUT/integrated_call_samples.ped" http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/20130606_g1k_3202_samples_ped_population.txt
# 2504 unrelated phase-3 samples (EUR 503, AFR 661)
curl -sSf -o "$OUT/phase3_2504.panel" http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel
seq 1 22 | xargs -P 11 -I{} sh -c "curl -sSf -C - -o $OUT/chr{}.vcf.gz $B/1kGP_high_coverage_Illumina.chr{}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz && curl -sSf -o $OUT/chr{}.vcf.gz.tbi $B/1kGP_high_coverage_Illumina.chr{}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz.tbi"
scripts/verify_1kg_md5.sh "$OUT"
(cd "$OUT" && sha256sum chr*.vcf.gz > SHA256SUMS)
echo done
