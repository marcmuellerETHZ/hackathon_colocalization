# Real-data engineer — M3 report (arms R1, R2, R3b, R4b)

Date 2026-10-26. Code: `src/data/real/` (README there lists every decision), `scripts/real/`, `tests/data_real/`.
Commits: 139d1f9, 1e3cc12 (+ final commit with this report's checksum table). Data (git-ignored): `data/raw/eqtlcat/`,
`data/raw/finngen/r12/`, `data/processed/real/`.

## 1. What was built

| Piece | File | Notes |
|---|---|---|
| Resumable verified fetch | `scripts/real/lib_fetch.sh` | `.part` + `curl -C -`; size == Content-Length; upstream md5 when one exists; full gzip test; sha256 appended to `SHA256SUMS` per directory; already-verified files skipped |
| eQTL Catalogue r7 | `scripts/real/download_eqtlcat.sh`, `eqtlcat_datasets.tsv` | metadata tables at pinned commit c90087e of eQTL-Catalogue-resources; 9 `ge` datasets × {all + tbi, permuted, credible_sets, lbf_variable}; GEUVADIS sample metadata (Zenodo 4678936, md5) and E-GEUV-1 SDRF |
| FinnGen R12 | `scripts/real/download_finngen.sh`, `finngen_traits.tsv`, `gcs_list.py` | 8 traits: sumstats + tbi, SuSiE cred/cred_99/snp + tbi; manifest; Kanta documentation; LD-matrix README + 23 `.tbi` (matrix queried remotely) |
| Driver | `scripts/real/run_downloads.sh`, `scripts/real/build_real_arms.sh` | background with logs in `logs/real/` |
| Readers | `src/data/real/sources.py` | |
| Harmonisation | `src/data/real/harmonise.py` | DESIGN §2.5 rules + GRCh38 build check |
| Splits/windows adapter | `src/data/real/regions.py` | uses data agent's `splits/blocks_v1.tsv`, `genes_v1.tsv.gz` (sha a86651bb…, a56c3cf1…, = commit b03edc7) |
| LD look-ups | `src/data/real/ld.py` | pool-B r² (data agent's loader); FinnGen R12 r² by remote tabix (`URL##idx##local.tbi`), cached |
| Pair builders + R1 | `src/data/real/pairs.py` | R2, R3b, R4b, R1 |
| GEUVADIS | `src/data/real/geuvadis.py` | 445-sample list, 6 genotype PCs |
| Extraction | `src/data/real/extract.py` | INTERFACES §2 HDF5 via `src/eval/io.py` (train/valid only), sidecar per-variant parquet, labels parquet, ρ̂ via `stats.rho.rho_hat_window` |
| Verification | `src/data/real/verify.py`, `src/data/real/qc_build.py` | end-to-end checks; build check |

## 2. Downloads (all verified)

- **eQTL Catalogue r7: 52 files, 45.1 GB.** No upstream checksums exist on the EBI FTP; verification = Content-Length + full gzip decompression + sha256. Per dataset (GB): GTEx blood QTD000356 4.33, GTEx liver QTD000266 3.44, Lepik_2017 blood QTD000373 4.41, TwinsUK blood QTD000549 4.00, BLUEPRINT monocyte QTD000021 5.69, Quach_2016 monocyte QTD000409 8.36, OneK1K CD14 QTD000609 1.56, GEUVADIS LCL QTD000110 7.43, TwinsUK LCL QTD000539 5.88.
- **FinnGen R12: 76 objects, 15.1 GB, every object md5-verified against the GCS `md5Hash`** (76/76 "[md5] ok", 76/76 sha256 recorded). Per trait (GB): LDL 1.29, bilirubin 0.81, urate 0.66, ferritin 0.64, platelets 5.22, T2D 2.83, K11_IBD_STRICT 1.35, E4_HYTHY_AI_STRICT 2.25.
- **Total 60.2 GB downloaded; derived locus sets 34 GB** (total ≈ 94 GB, inside the ~100 GB budget). Full list with sha256: `scripts/real/downloads_sha256.tsv` (committed, 128 files).
- Incident: I edited `download_eqtlcat.sh` (to add the SDRF fetch) while it was running. Bash reads scripts incrementally, so after `xargs` finished it executed a garbled line (`emote_size: command not found`) and exited. All files had already been fetched and verified; a full re-run skipped all 60 files as verified and printed "complete" (`logs/real/download_eqtlcat.log`). **Proposed LESSON:** never edit a shell script while it runs; copy it or wait.

**GRCh38 build check** (`python -m data.real.qc_build`, `logs/real/qc_build.log`): all 17 sources PASS.
- REF/ALT equal to the GRCh38 1000G record at shared positions in 5 train windows: eQTL Catalogue 1.0000 (same imputation panel); FinnGen 0.9967–0.9975, with REF = GRCh38 REF at 100% of shared positions (the rest are different ALTs at multi-allelic sites).
- rs429358, rs7412, rs12740374, rs4149056 are found at their GRCh38 positions in all 9 eQTL datasets and in the 3 FinnGen endpoints. The FinnGen lab files have no rsID column.

## 3. Candidate pairs (counts per arm per split)

| Arm | train | valid | test (ID list only) | unique genes (train+valid / test) | LD blocks (train+valid) |
|---|---|---|---|---|---|
| R2 (partial positives) | 8,518 | 2,807 | 2,674 | 5,151 / 1,180 | 855 |
| R3b (pseudo-negatives) | 3,948 | 1,203 | 1,237 | 2,027 / 441 | 314 |
| R4b (GEUVADIS exact in-sample LD) | 4,081 | 1,259 | 1,279 | 3,121 / 724 | 750 |

- **R2 by cohort pair (train/valid/test):**
  - GTEx blood × Lepik: 2031/690/644
  - GTEx blood × TwinsUK blood: 1247/398/382
  - Lepik × TwinsUK blood: 1437/487/462
  - BLUEPRINT × Quach: 1507/486/489
  - BLUEPRINT × OneK1K: 153/50/42
  - Quach × OneK1K: 118/36/31
  - GEUVADIS × TwinsUK LCL: 2025/660/624

  The top leads are identical in only 14% of R2 pairs. This is recorded but **not used for selection**.
- **R3b funnel (all splits):** 8,959 pairs pass the CS rules → 1,373 have no computable pool-B lead r² (no CS member SNV in the QC'd panel) → pool-B r² < 0.1 → FinnGen r² < 0.1 (status: 4,204 below 0.01, 2,405 found) → **6,388 kept** (3,948 / 1,203 / 1,237).
  - In train+valid there are only 419 distinct FinnGen CSs and 2,291 trait × gene combinations, so pairs are strongly clustered. Bootstrap by block/gene is required.
  - Only 0.1% of pairs have any cross-CS member pair with pool-B r² > 0.5 (`max_cs_r2_poolB`, label-noise guard).
  - Median lead |z|: 6.7 for FinnGen, 7.6 for eQTL.
- **R4b by partner (train+valid):** TwinsUK LCL 2,685; platelets 1,377; T2D 427; hypothyroidism 334; LDL 194; IBD 147; bilirubin 77; urate 59; ferritin 40.
- **Test pairs** are written as identifier lists only: `data/processed/real/pairs/*_test_ids.tsv`. Their sha256 values are R2 546b139b…, R3b 769060f6…, R4b 71ebdc48…. Candidates sha256: R2 d71d43e2…, R3b 5baabaa1…, R4b 7b6530fb….

## 4. Locus sets (train/valid; INTERFACES §2, written by `eval.io.write_locus_set`, gzip)

| File | loci | p (median) |
|---|---|---|
| `R2_train.h5` / `R2_valid.h5` | 8,513 / 2,806 (6 dropped: < 2 variants) | 2,371 / 2,284 |
| `R3b_train.h5` / `R3b_valid.h5` | 3,948 / 1,203 | 2,410 / 2,382 |
| `R4b_{train,valid}.nfe_minus_geuv.h5` (reference LD, n = 138) | 4,081 / 1,259 | 2,512 / 2,440 |
| `R4b_{train,valid}.geuv_insample_pcadj.h5` (in-sample, PC-residualised, n = 445) | same loci | same variants |
| `R4b_{train,valid}.geuv_insample_raw.h5` | same loci | same variants |

Every file has `<stem>.truth.parquet` (labels and selection columns), `<stem>.variants.parquet` (source beta/se/AF, flags) and `<stem>.extract_log.tsv` (harmonisation counts per locus). Sizes: R2 5.7 GB, R3b 2.7 GB, R4b 24.6 GB (pcadj alone is 17.3 GB, because residualised dosages barely compress).

## 5. Tests and verification (commands run, outputs)

- `PYTHONPATH=src python -m pytest -q tests/data_real -p no:cacheprovider` → **21 passed**. They cover:
  - harmonisation known answers: flip, palindromic AF-drop and swap-drop, AF flag kept, strand mismatch, multi-record, non-autosomal, identity negative control, flip round-trip, build detection with shifted positions, known rsIDs;
  - gene gate (lead outside window, p ≥ 1e-5, impure CS);
  - R2 gene-level selection and the same-donor exclusion;
  - every R3b rule, positive and negative (disjointness with allele-swapped IDs, second CS in window, low purity, weak p);
  - LD thresholds and NaN handling;
  - test-split refusal (extract and io guard);
  - FinnGen LD parsing;
  - correlation from standardisation, residual orthogonality;
  - real chr22 r² self-check, the LD-lead fallback, and GEUVADIS counts 445 = 358 EUR + 87 AFR.
- `python -m data.real.verify` (`logs/real/verify.log`, `locus_sets/verify.json`): 0 failures in all 10 files.
  - diag(X'X/n) − 1 ≤ 1.1e-7; min maf_ref = 0.01; z identical to source beta/se (max abs diff 0.0).
  - **R2 positive control: same sign of z at the lead in 99.8% of loci (6,809 train, 2,257 valid with both |z| > 5).** Misaligned alleles would give about 50%.
  - corr(study AF, pool-B AF): 0.977–0.988. AF-flag rate: 0.003–1.2% (highest in eQTL cohorts with AFR donors). No allele flips were needed; all sources use GRCh38 REF / 1000G ALT.

## 6. R1 realism (train split only; `data/processed/real/pairs/counts.json` → "R1")

| eQTL dataset | genes tested | gate pass | lead \|z\| of gate genes q05/q50/q95 | CS per gate gene 1/2/≥3 |
|---|---|---|---|---|
| GTEx blood | 9,511 | 0.27 | 5.4/8.0/19.4 | 2045/441/93 |
| GTEx liver | 11,734 | 0.10 | 5.6/7.6/15.6 | 989/126/11 |
| Lepik blood | 9,451 | 0.41 | 5.3/8.6/22.7 | 2773/844/282 |
| TwinsUK blood | 10,583 | 0.15 | 5.5/7.7/16.7 | 1424/174/22 |
| BLUEPRINT mono | 9,442 | 0.35 | 5.4/8.3/20.1 | 2517/659/162 |
| Quach mono | 8,961 | 0.28 | 5.8/8.3/17.3 | 1818/529/133 |
| OneK1K CD14 | 4,258 | 0.04 | 5.6/7.7/16.3 | 159/13/1 |
| GEUVADIS | 10,053 | 0.26 | 5.7/8.2/19.3 | 2056/469/94 |
| TwinsUK LCL | 11,587 | 0.29 | 5.3/8.0/19.7 | 2675/567/81 |

| FinnGen trait | pure CS | frac p < 5e-8 | lead \|z\| q05/q50/q95 | CS per region 1/2/≥3 |
|---|---|---|---|---|
| LDL | 107 | 0.89 | 4.9/7.2/26.1 | 28/16/12 |
| bilirubin | 37 | 0.97 | 5.7/8.7/70.2 | 20/3/2 |
| urate | 20 | 0.85 | 5.3/7.0/26.4 | 11/6/2 |
| ferritin | 18 | 0.83 | 3.5/7.1/19.1 | 8/2/3 |
| platelets | 710 | 0.88 | 4.4/7.3/17.6 | 91/83/139 |
| T2D | 171 | 0.92 | 5.2/6.8/12.3 | 69/43/23 |
| IBD | 42 | 0.93 | 5.4/6.9/11.7 | 33/12/1 |
| hypothyroidism | 130 | 0.89 | 4.8/6.7/12.5 | 52/27/18 |

Mismatch diagnostics (extracted train loci): see §5 (AF concordance), §7 (lead missingness), and the R3b LD comparison between pool-B and FinnGen.

## 7. Findings the PI should see

1. **The per-window ρ̂ is biased upward when the two traits share a signal.** `stats.rho.rho_hat_window` on real loci, whose cohorts are all disjoint so true ρ ≈ 0:
   - R2 median ρ̂ = **0.157** (SD 0.161, median SE 0.119);
   - R4b median 0.077;
   - R3b (distinct signals) median −0.002.

   Pruned |z| < 2 SNPs still carry LD-tagged shared signal. A method that trusts the window ρ̂ would partly explain true sharing as overlap, which costs S recall exactly in the R2 arm. This matters for C4 and for every ρ̂-using method on real data. Options:
   - estimate ρ̂ per cohort pair from all train windows pooled, excluding windows with any CS;
   - set ρ = 0 for documented disjoint cohorts;
   - mask CS members ± LD before the MLE.

   ρ̂ is written as computed; I have not changed the estimator (it belongs to the simulation agent).
2. **Lead-variant missingness is 20–29% per trait.** The trait's lead is absent from the harmonised SNV set in that share of loci (R2 train: trait 1 21%, trait 2 28%). About 65% of these leads are indels; the rest are SNVs absent from the QC'd panel (multi-record positions, MAF < 0.01 in the panel). This is logged per locus (`lead1_present`/`lead2_present` attributes). The PI may want results stratified by it.
3. OneK1K CD14_Mono in r7 is weak: only 4% of tested genes pass the gate. It contributes few R2 pairs (271 train).
4. R3b drops 15% of rule-passing pairs because no CS member SNV is in the pool-B panel (mostly Finnish-enriched or rare FinnGen leads).

## 8. Deviations from DESIGN / task

- Harmonisation lives in `src/data/real/harmonise.py` (task) rather than `src/data/harmonise.py` (DESIGN §2.5 text).
- FinnGen LD PSD repair (DESIGN §2.5) is not implemented: no FinnGen LD matrix is produced (R4 is deferred). R3b uses FinnGen LD only for lead r², by remote tabix, so **R3b uses both pool-B and FinnGen LD**.
- R3b "lead" for LD = the highest-PIP CS member that is an SNV present in pool B, because the top variant is often an indel.
- R4b reference = NFE (A ∪ B) minus GEUVADIS, n = 138 (IBS 107, TSI 16, CEU 10, GBR 5), instead of pool B. Pool B minus GEUVADIS would leave only 46.
- R4b also includes TwinsUK-LCL × GEUVADIS pairs. Only trait 2 has in-sample LD.
- The GEUVADIS donor list is inferred (E-GEUV-1 ∩ 1000G 2,504 = 445 = 358 EUR + 87 YRI, which exactly matches eQTL Catalogue). It is not published.
- **Test-lock exposure:** building the test ID lists used the same rules on test genes. That meant reading their CS/permuted metadata, pool-B lead r², and FinnGen LD lead r² by remote tabix. The build check read 4 single positions, 2 of which may lie in test blocks. No summary-statistic windows or LD matrices of test loci were extracted.

## 9. Decisions needed from the PI

1. How methods get ρ on real data (finding 1).
2. Which R4b in-sample version is primary: pcadj (my proposal) or raw. Whether to delete one (17.3 GB vs 4.5 GB).
3. Whether to freeze the test ID lists' sha256 (above) in `configs/` for the test lock.
4. Whether to keep OneK1K, or replace it with a stronger monocyte cohort.
5. R2/R3b clustering: bootstrap unit = LD block (as DESIGN), and possibly a cap on pairs per FinnGen CS.

## 10. Proposed DATA.md entries

| Dataset | Source | Version | Build | Checksum | Licence | Local path | Script |
|---|---|---|---|---|---|---|---|
| eQTL Catalogue r7: 9 ge datasets (QTD000356, 266, 373, 549, 021, 409, 609, 110, 539) — all + tbi, permuted, credible_sets, lbf_variable | https://ftp.ebi.ac.uk/pub/databases/spot/eQTL/{sumstats,susie}/ (Kerimov 2021) | r7 (June 2024) | GRCh38 | sha256 per file in `data/raw/eqtlcat/r7/*/SHA256SUMS`, mirrored in `scripts/real/downloads_sha256.tsv` (no upstream checksums) | CC BY 4.0 | `data/raw/eqtlcat/r7` | `scripts/real/download_eqtlcat.sh` |
| eQTL Catalogue metadata (tabix_ftp_paths, dataset_metadata_r7, population_assignments, dataset_id_map, sceQTL counts) | github eQTL-Catalogue-resources @ c90087e | c90087e | — | `data/raw/eqtlcat/meta/SHA256SUMS` | CC BY 4.0 / Apache 2.0 | `data/raw/eqtlcat/meta` | same |
| GEUVADIS sample metadata | Zenodo 4678936 `GEUVADIS.tsv` (md5 7514d69f…) ; ArrayExpress E-GEUV-1 SDRF | 2021 / current | — | sha256 84a1f323…, 7d3a28e0… | CC BY 4.0 / EBI terms | `data/raw/eqtlcat/meta` | same |
| FinnGen R12 summary stats + SuSiE (LDL 3001308, bilirubin 3006140, urate 3026493, ferritin 3001122, platelets 3007461, T2D, K11_IBD_STRICT, E4_HYTHY_AI_STRICT) + manifest + Kanta documentation + LD-matrix README/tbi | gs://finngen-public-data-r12 (Kurki 2023) | R12 (2024-11-04) | GRCh38 | upstream GCS md5 verified for all 76 objects; sha256 in `data/raw/finngen/r12/**/SHA256SUMS` and `scripts/real/downloads_sha256.tsv`; object list `data/raw/finngen/r12/listing/selected.tsv` | FinnGen terms: free after the access form; no redistribution; acknowledge FinnGen and cite Kurki 2023 | `data/raw/finngen/r12` | `scripts/real/download_finngen.sh` |
| FinnGen R12 LD matrix (remote, lead pairs only) | storage.googleapis.com/finngen-public-data-r12/ld_matrix | R12 | GRCh38 | queried by tabix with local verified .tbi; results cached in `data/processed/real/cache/finngen_ld_pairs.tsv` | as FinnGen | — | `src/data/real/ld.py` |
| Derived: real locus sets R2/R3b/R4b (train, valid) | this project | v1 | GRCh38 | regenerate (`scripts/real/build_real_arms.sh`); candidate/test-ID sha256 in this report | derived; FinnGen-derived content not redistributed | `data/processed/real/` | `scripts/real/build_real_arms.sh` |
