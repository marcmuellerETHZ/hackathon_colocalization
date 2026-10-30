# src/data — data engineering (owner: data agent)

Run everything with `source env/activate.sh` and `PYTHONPATH=src`.

| Script | Output | DESIGN |
|---|---|---|
| `python -m data.pools` | `data/processed/pools/pools_v1.tsv`, committed copy `datasets/splits/pools_v1.tsv` | §2.1 |
| `src/data/multirecord_positions.sh N` (22 in parallel, ~5 min) | `data/processed/variants/multirec/chrN.{pos,summary}` | §2.1 |
| `python -m data.variants [--chroms ..] [--jobs 11]` (~15 s) | `data/processed/variants/chrN.parquet`, `chrN.summary.json` | §2.1 |
| `python -m data.make_splits` (~45 s) | `datasets/splits/blocks_v1.tsv`, `datasets/splits/genes_v1.tsv.gz`, `datasets/splits/SHA256SUMS` | §3 |
| `data.genotypes.load_genotypes / load_haplotypes / variant_table` | loader API (pgenlib) | §2.1 |

Tests: `PYTHONPATH=src:tests/data python -m pytest -q tests/data` (temp files go to `/tmp/claude-<uid>/`).

**Coordinate convention (everywhere in this module):** blocks and windows are BED intervals
(0-based start, exclusive end). A 1-based position `pos` (pvar/VCF POS, GTF TSS) is inside iff
`start < pos <= end`. The genotype loader uses the same rule, so a gene window from `genes_v1.tsv.gz`
can be passed to it directly.

## Decisions / assumptions

### Pools (`pools.py`)
- Sample universe: samples in `data/processed/1kg/chr*.psam` (all 22 psam files are identical; 503 EUR + 661 AFR unrelated phase-3 samples), with population labels from the phase-3 panel.
- NFE = CEU (99) + GBR (91) + IBS (107) + TSI (107) = 404.
- Pool B (150) is drawn stratified by subpopulation: slots allocated by largest remainder of 150·n_pop/404 (ties broken alphabetically by population code), giving CEU 37, GBR 34, IBS 40, TSI 39. Within each subpopulation, sample IDs are sorted and permuted with `rng(20261026, "pools_v1", "poolB", pop)`; the first k go to B. Pool A = the other 254 NFE.
- **L6c**: DESIGN says "a mosaic of 99 random NFE individuals not in pool B". Since A ∪ B = all NFE, L6c is a 99-sample subset of pool A, drawn stratified by subpopulation in the same way (`rng(20261026, "pools_v1", "L6c", pop)`): CEU 24, GBR 22, IBS 26, TSI 27. Consequence: L6c's mosaic haplotypes are a subset of the primary mosaic's source haplotypes (pool A). That is intended: L6c differs from the main NFE world only by the smaller founder set (99, same as FIN), so L6 − L6c isolates the population difference from the founder-count effect.
- FIN = all 99 FIN; ASW_ACB = all 61 ASW + 96 ACB.
- File format: long TSV `sample_id, pop, superpop, pool`; a sample appears twice if it is in A and L6c. Pool names: `A, B, FIN, ASW_ACB, L6c`.

### Variant QC (`variants.py`, `multirecord_positions.sh`)
- Input: `data/processed/1kg/chrN.pgen` (biallelic SNV+indel, EUR MAF ≥ 0.005, 503 EUR + 661 AFR). Note the upstream EUR-MAF ≥ 0.005 filter means FIN/ASW_ACB-specific variants rarer than that in EUR are absent; any downstream MAF filter is therefore effectively "EUR MAF ≥ 0.005 AND pool MAF ≥ x".
- SNV = REF and ALT each one of A/C/G/T (MNPs and indels dropped).
- "Positions with > 1 record" is evaluated on the **raw** 1000G VCF (SNV/indel records with AC ≥ 1 among our 1164 samples; SVs not counted) **and** on the pgen. Reason: the pgen's EUR-MAF filter can drop one record of a split multiallelic, leaving a record that looks biallelic but whose REF genotype hides carriers of the other ALT. This removes 186,665 additional SNVs genome-wide on top of the 111,811 duplicated within the pgen. Spanning deletions (deletion starting upstream and covering the SNV) are not counted as multi-record.
- MHC: chr6:28,500,000–33,400,000, 1-based inclusive.
- No MAF filter in the stored table; AFs (ALT allele frequency) stored for pools A, B, FIN, ASW_ACB, L6c (added: needed for the L6c control's MAF filter) and NFE (= A ∪ B).
- `pgen_idx` = row in `chrN.pvar` for direct pgenlib access.
- The multi-record scan was cross-checked: an earlier plink2-based implementation (`--mac 1 --make-just-pvar`) gave identical multi-record position counts on chr13–chr22 to the committed bcftools version.

### Genotype loader (`genotypes.py`)
- Returns int8 ALT dosages `[n_samples, p]` and phased ALT haplotypes `[2·n_samples, p]` (rows 2i, 2i+1 = VCF GT order a|b of `samples[i]`), rows in the requested sample order.
- Phase: the pgen was made by `plink2 --make-pgen` from the phased VCF; `hardcall_phase_present()` is true and the loader raises if any het call lacks phase. Verified against the raw VCF with bcftools for 80 samples × all QC'd SNVs in chr22:30.0–30.1 Mb (`tests/data/test_genotypes.py`).
- Missing calls raise (none exist: 0 missing calls genome-wide among the QC'd SNVs).

### Blocks and splits (`make_splits.py`)
- Units: MacDonald 2022 pyrho EUR blocks (`data/raw/annot/pyrho_EUR_LD_blocks.bed`, 1,336 blocks; no overlaps, 17 inter-block gaps, all centromeric and all ≥ 1 Mb (4.1–28.9 Mb)). No gap is < 1 Mb, so the midpoint rule is implemented and tested on toy data but never triggers on real data.
- **Chromosome ends (not specified in DESIGN):** the stretch before the first / after the last block follows the same rule: ≥ 1 Mb becomes its own block (`gap_start`/`gap_end`), < 1 Mb is absorbed into the first/last block. Ends = 0 and the contig length from the raw 1000G VCF header. Result: 5 `gap_start` blocks (acrocentric p-arms of chr13/14/15/21/22) and 0 `gap_end` blocks. In total 1,358 tiled blocks (1,336 pyrho + 22 gap).
- Large gap blocks are kept whole as DESIGN says (e.g. chr1 1q12–q21 gap block of 28.9 Mb with 54 protein-coding genes). They contain 463 protein-coding genes in total; 464 protein-coding TSSs lie outside the raw pyrho blocks (DESIGN quotes 509; that figure came from a reviewer, with a different TSS/gene set).
- **MHC**: every tiled block overlapping chr6:28.5–33.4 Mb gets `split = excluded` (8 blocks, chr6:27,258,180–33,988,587). These blocks stay in `blocks_v1.tsv` so the tiling stays gap-free; genes with their TSS there are in `genes_v1.tsv.gz` with `split = excluded` and window −1/−1 (510 genes, 223 protein-coding). **Consumers must filter `split in {train, valid, test}`.** Variants in the 27.26–28.5 Mb and 33.4–33.99 Mb flanks pass variant QC but belong to an excluded block.
- Assignment: per chromosome, block counts per split are set by largest remainder of 60/20/20 (random tie-break, `rng(20261026, "splits_v1", "counts")`). Then 5,000 random permutations within chromosomes (`rng(20261026, "splits_v1", "candidates")`) are scored by Σ_split (pc genes − target)². The best one is kept (loss 65.8, i.e. within ±6 genes of target; median random loss 135,407).
- "Balanced on protein-coding gene counts" is read as the global protein-coding shares per split. It is not per chromosome, and gene types other than protein-coding play no part in balancing.
- Genes: GENCODE v39 basic `gene` records on chr1–22 (all gene types, 58,476 genes). TSS = gene start (+ strand) or gene end (− strand) of the gene record. This is the gene-level TSS used by eQTL Catalogue; it is not the TSS of the canonical transcript.
- Windows: BED `[tss−1−500,000, tss+500,000)` (i.e. 1-based tss±500 kb inclusive), clipped to the maximal run of consecutive same-split blocks containing the TSS block. Simulated loci/leads use `data.make_splits.clip_window(load_blocks(), chrom, pos)` with the same rule.
- File names follow the M3 task: `datasets/splits/blocks_v1.tsv` (DESIGN §3 calls it `split_v1.tsv`), and the test is `tests/data/test_splits.py` (DESIGN: `tests/test_splits.py`).
- `datasets/splits/SHA256SUMS` also lists `pools_v1.tsv`.
- `genes_v1.tsv` is stored gzipped (`genes_v1.tsv.gz`, gzip header mtime 0 and no file name, so the bytes are reproducible), because the plain TSV is 5.5 MB, over the 5 MB commit limit. `pd.read_csv` reads it directly.
