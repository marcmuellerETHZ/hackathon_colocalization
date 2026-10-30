#!/usr/bin/env bash
# Verify downloaded 1000G 30x VCFs against the upstream md5 manifest. Exit non-zero on any mismatch.
set -euo pipefail
RAW=${1:-data/raw/1kg30x}; bad=0
while read -r f md5; do
  [[ $f == *.vcf.gz ]] || continue
  c=$(echo "$f" | sed -E 's/.*\.(chr[0-9]+)\.filtered.*/\1/'); [[ $c == chrX* || $f == *chrX* ]] && continue
  got=$(md5sum "$RAW/$c.vcf.gz" | cut -d' ' -f1)
  if [[ $got != "$md5" ]]; then echo "MISMATCH $c"; bad=1; else echo "ok $c"; fi
done < "$RAW/20220804_manifest.txt"
exit $bad
