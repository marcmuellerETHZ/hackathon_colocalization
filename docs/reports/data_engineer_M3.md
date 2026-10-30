# Data engineer — M3 report (src/data/, tests/data/)

Date 2026-10-26. Commits:
- efd317c: pools
- f316ee6: variant QC
- 2239fe9: genotype loader
- b03edc7: blocks/splits

Every choice not fixed by DESIGN is listed in `src/data/README.md` under "Decisions / assumptions".

## 1. What was built

| Step | Code | Output |
|---|---|---|
| Pools | `src/data/pools.py` | `splits/pools_v1.tsv` (committed), `data/processed/pools/pools_v1.tsv` |
| Raw multi-record scan | `src/data/multirecord_positions.sh` | `data/processed/variants/multirec/chrN.pos` |
| Variant QC + AFs | `src/data/variants.py` | `data/processed/variants/chrN.parquet` and `chrN.summary.json` |
| Blocks / splits / genes | `src/data/make_splits.py` | `splits/blocks_v1.tsv`, `splits/genes_v1.tsv.gz`, `splits/SHA256SUMS` |
| Loader | `src/data/genotypes.py` | `load_genotypes`, `load_haplotypes`, `variant_table` |

**Coordinate convention, used throughout the module:** blocks and windows are BED intervals. A 1-based position is inside one iff `start < pos <= end`. The loader uses the same rule, so a gene window can be passed to it unchanged.

## 2. Counts

### Pools (seed 20261026, stratified by subpopulation)

| Pool | Size | By subpopulation |
|---|---|---|
| A | 254 | CEU 62, GBR 57, IBS 67, TSI 68 |
| B | 150 | CEU 37, GBR 34, IBS 40, TSI 39 |
| FIN | 99 | all FIN |
| ASW_ACB | 157 | ASW 61, ACB 96 |
| L6c | 99 | subset of A: CEU 24, GBR 22, IBS 26, TSI 27 |

- **L6c rule:** A ∪ B is all 404 NFE, so "NFE not in pool B" means pool A. L6c is therefore a stratified 99-sample subset of A. This matches FIN's founder count, so L6 − L6c isolates ancestry from founder number.

### Variant QC (chr1–22, chr7 rebuilt; 4,807 pvar records/Mb)

Starting from 13,055,668 pgen records:
- 10,290,529 SNVs;
- −111,811 duplicate positions within the pgen;
- −186,665 positions that are multi-record only in the raw VCF;
- −54,521 in the MHC.

**9,937,532 QC'd SNVs remain.** At MAF ≥ 0.01 that is 8,517,873 in pool A, 8,885,299 in pool B and 8,584,205 in FIN.

| chr | QC'd SNVs | chr | QC'd SNVs | chr | QC'd SNVs |
|---|---|---|---|---|---|
| 1 | 789,446 | 9 | 448,927 | 17 | 270,761 |
| 2 | 831,826 | 10 | 510,146 | 18 | 281,203 |
| 3 | 703,382 | 11 | 488,166 | 19 | 232,265 |
| 4 | 714,107 | 12 | 474,550 | 20 | 231,190 |
| 5 | 635,669 | 13 | 369,098 | 21 | 144,415 |
| 6 | 604,211 | 14 | 325,166 | 22 | 151,526 |
| 7 | 587,217 | 15 | 290,362 | | |
| 8 | 538,160 | 16 | 315,739 | | |

- No genotype calls are missing.
- Phase is retained. `hardcall_phase_present` is true, and the haplotypes match the raw VCF GTs exactly for 80 samples × all SNVs in chr22:30.0–30.1 Mb.
- Loader speed for pool A in a ±500 kb window (~4,200 SNVs): 4 ms (dosages) and 20 ms (haplotypes).

### Blocks and splits

- **Tiling:** the 1,336 pyrho blocks plus 22 gap blocks give 1,358 blocks. The gap blocks are 17 centromeric gaps (all ≥ 1 Mb) and 5 acrocentric p-arms.
- **Blocks per split:**

  | Split | Blocks | Same-split runs |
  |---|---|---|
  | train | 809 | 327 |
  | valid | 274 | 223 |
  | test | 267 | 217 |
  | excluded (MHC) | 8 | — |

  Per chromosome, each split's count is within 1 of 60/20/20.
- **Genes** (58,476 GENCODE v39 basic, all types):

  | Split | All genes | Protein-coding | Target (protein-coding) |
  |---|---|---|---|
  | train | 34,388 | 11,289 | 11,295.6 |
  | valid | 11,947 | 3,769 | 3,765.2 |
  | test | 11,631 | 3,768 | 3,765.2 |
  | excluded | 510 | 223 | — |

- **Clipped window size (kb):**

  | Split | 5% | 10% | 25% | Median | Mean | Min | Full width (1 Mb) | < 500 kb |
  |---|---|---|---|---|---|---|---|---|
  | train | 622 | 740 | 1000 | 1000 | 948 | 245 | 79.6% | 0.0% |
  | valid | 563 | 633 | 818 | 1000 | 901 | 423 | 60.0% | 0.2% |
  | test | 561 | 628 | 827 | 1000 | 903 | 287 | 62.3% | 0.4% |

## 3. Tests

Command:

```
source env/activate.sh && PYTHONPATH=src:tests/data python -m pytest -q tests/data
```

Result: `69 passed in 54.46s`.

- **`test_pools`:**
  - known-answer allocation;
  - toy pools;
  - real sizes, A∩B = ∅, A∪B = NFE, L6c ⊂ A;
  - the committed file equals a regeneration.
- **`test_variants`:**
  - toy VCF→pgen known-answer test: indel, MNP, duplicate position, raw-only multi-record, MHC boundaries 28,500,000/33,400,000 inclusive and 28,499,999/33,400,001 kept, exact AFs per pool;
  - a missing multirec file raises;
  - invariants on all 22 real files;
  - pool-B and ASW_ACB AFs equal an independent `plink2 --freq` to 1e-6.
- **`test_genotypes`:**
  - toy known-answer haplotypes in a permuted sample order;
  - dosage = sum of haplotypes;
  - BED boundaries;
  - MAF filter and error cases;
  - real haplotypes vs raw VCF via bcftools;
  - loader AF = stored AF.
- **`test_splits`:**
  - toy tiling (midpoint split, own-block gap, absorbed or own-block chromosome ends, overlap error), toy window clipping and seeded balance;
  - real: no gaps or overlaps, ends at 0 and the contig length, raw blocks preserved, only MHC-overlapping blocks excluded;
  - real: each gene's block is its TSS block, every window contains its TSS, lies in one split run and is maximal (exact formula);
  - real: every QC'd variant is in exactly one block, and every variant in a window has the gene's split (all 22 chromosomes);
  - real: split fractions within 0.02 overall and within 1 per chromosome, protein-coding shares within 0.005;
  - real: SHA256SUMS match, and a regeneration is byte-identical;
  - negative controls: a flipped block split, a widened window and a shifted window are each detected by `audit_windows`.

## 4. Deviations from DESIGN, and choices it does not fix

1. **File names follow the task, not DESIGN §3.**
   - `splits/blocks_v1.tsv` (DESIGN: `split_v1.tsv`) and `tests/data/test_splits.py` (DESIGN: `tests/test_splits.py`).
   - `genes_v1.tsv` is committed gzipped (`genes_v1.tsv.gz`, deterministic gzip) because the plain file is 5.5 MB, over the 5 MB rule.
2. **Multi-record positions are taken from the raw VCF** (AC ≥ 1 in our 1,164 samples), not only from the pgen.
   - Reason: the pgen's EUR-MAF filter hides second alleles of split multiallelics.
   - Effect: +186,665 SNVs removed.
   - Cross-check: a plink2 implementation and the bcftools implementation gave identical counts on chr13–22.
3. **The MHC is excluded at block level in the splits.**
   - Effect: 8 blocks, chr6:27.26–33.99 Mb.
   - These blocks are kept in `blocks_v1.tsv` as `split = excluded`, so the tiling has no gaps.
   - Consumers must filter `split ∈ {train, valid, test}`.
4. **Chromosome ends follow the gap rule** (DESIGN does not say).
5. **Gap blocks are kept whole, as DESIGN says**, even when huge.
   - Examples: 28.9 Mb at chr1q12–21 and 26 Mb at chr9.
   - DESIGN says the tiling puts 509 protein-coding genes back inside a block; we get 464.
6. **AFs are also stored for L6c.**
7. **TSS** is taken from the GENCODE gene record: start for + strand, end for − strand.

## 5. Incidents (for LESSONS)

- **bcftools was removed from `env/renv`.** A 14:18 micromamba R-package install uninstalled it. This silently broke the chr7 redo; the PI has since fixed it with `env/tools`.
- **A duplicate copy of this agent** (started by a PI message) edited `multirecord_positions.sh` while it was running and killed my scan. Bash reads scripts incrementally, which made this worse. I re-ran the whole scan from a frozen copy (identical sha256 to the committed script).
- `src/data/NOTE_FROM_PI.md` is still on disk (deleting it was denied by the permission classifier). It is not committed; please delete it.

## 6. For the PI to decide

1. Whether to accept the `blocks_v1.tsv` and `genes_v1.tsv.gz` names, or have DESIGN updated.
2. Whether huge gap blocks should be sub-tiled. Currently one 28.9 Mb block counts as a single bootstrap unit.
3. **R4b:** pool B contains GEUVADIS individuals (CEU/GBR/TSI). The R4b reference must exclude them, which needs the GEUVADIS sample list (not downloaded). Options:
   - a pool-B-minus-GEUVADIS reference;
   - an IBS-only reference.
4. Whether DATA.md should gain an entry for the derived variant table: script `src/data/variants.py`, deterministic from inputs.
