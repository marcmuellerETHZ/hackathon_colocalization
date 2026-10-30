# src/data/real — real-data arms R1, R2, R3b, R4b (owner: real-data engineer)

Run everything after `source env/activate.sh`, with `PYTHONPATH=src`, from the project root.
DESIGN v4 §2.5 is the specification; this README lists what the code does and every choice that DESIGN does
not fix.

## Pipeline

| Step | Command | Output (git-ignored, under `data/`) |
|---|---|---|
| 1. Downloads (background, resumable, verified) | `scripts/real/run_downloads.sh` (calls `download_eqtlcat.sh`, `download_finngen.sh`) | `data/raw/eqtlcat/`, `data/raw/finngen/r12/` (+ `SHA256SUMS` per directory); logs in `logs/real/` |
| 2. Build check (GRCh38) | `python -m data.real.qc_build` | `data/processed/real/qc/build_check.json` |
| 3. Candidate pairs + R1 summaries | `python -m data.real.pairs --workers 16` | `data/processed/real/pairs/{R2,R3b,R4b}_candidates.tsv` (train + valid), `*_test_ids.tsv` (test: identifiers only), `counts.json` (incl. R1) |
| 4. Locus sets (train, valid only) | `python -m data.real.extract --workers 32` | `data/processed/real/locus_sets/<arm>_<split>[.<ld mode>].h5` (INTERFACES §2 via `src/eval/io.py`), `*.truth.parquet` (labels), `*.variants.parquet` (per-variant source values), `*.extract_log.tsv`, `*.summary.json` |
| 4b. ρ̂ v2 (P3-1) | `python -m data.real.rho_update --workers 32` | attributes in place; `locus_sets/rho_v2.parquet`, `rho_v2_pairs.tsv`, `rho_v2_summary.json` |
| 5. Verification | `python -m data.real.verify` | `data/processed/real/locus_sets/verify.json` |
| 6. Test-list metadata (P3-14/P3-24; IDs + CS metadata only) | `python -m data.real.test_meta --workers 3` (`--check` to verify) | `data/processed/real_testmeta_v1/<arm>_test_meta_v1.tsv`, `test_meta_v1.{verify.json,sha256}` (artifact real_test_meta@1; committed copy `datasets/configs/real_test_meta_v1.sha256`) |
| 7. Test arms (inside `scripts/eval_test.py` only) | entry point `data.real.extract:build_test_arm` (= `build_test_arms(real_spec, out_dir, workers=)`) | `<out_dir>/<arm>_test[.<ld mode>].h5` + truth/variants/log/summary |
| 7v. Same builder on validation ID lists (check) | `python scripts/real/check_testarm_valid.py --workers 16` | `data/processed/real_testarm_valid_check_v1/` (`check.json`) |
| Tests | `python -m pytest tests/data_real` | |

`scripts/real/build_real_arms.sh` runs steps 2–5 with logs.

## Modules

- `config.py` — paths, dataset/trait tables (`scripts/real/eqtlcat_datasets.tsv`, `scripts/real/finngen_traits.tsv`), constants.
- `sources.py` — readers for eQTL Catalogue r7 (`permuted`, `credible_sets`, `all` by tabix) and FinnGen R12 (sumstats by tabix, SuSiE `cred` + `snp` CS membership, cached as parquet).
- `harmonise.py` — DESIGN §2.5 harmonisation, lead-variant missingness, GRCh38 build check.
- `regions.py` — adapter to the data agent's `datasets/splits/blocks_v1.tsv`, `datasets/splits/genes_v1.tsv.gz` (BED convention).
- `ld.py` — pool-B r² (1000G, data agent's genotype loader) and FinnGen R12 in-sample r² by remote tabix.
- `pairs.py` — gene gate, R2, R3b, R4b builders, R1 summaries.
- `geuvadis.py` — GEUVADIS sample list and genotype PCs for R4b.
- `extract.py` — locus extraction and HDF5 writing; `verify.py` — end-to-end checks; `qc_build.py` — build check.

## Decisions / assumptions

### Data selection
- **eQTL Catalogue release r7** (stable; TSV + tabix). The r8 beta (GTEx v10) is not used: it is a pre-release and may change.
- **Datasets** (all `ge`, i.e. gene expression with full nominal summary statistics):
  - whole blood: GTEx v8 blood QTD000356 (N 670), Lepik_2017 / Estonian Biobank QTD000373 (471), TwinsUK blood QTD000549 (195);
  - monocytes: BLUEPRINT QTD000021 (191), Quach_2016 naive QTD000409 (200), OneK1K CD14_Mono QTD000609 (pseudobulk; r7 metadata says 560, path index 959; weak in r7: 363 genes with p_beta < 1e-3);
  - LCL: GEUVADIS QTD000110 (445), TwinsUK LCL QTD000539 (418);
  - liver: GTEx v8 liver QTD000266 (208), used in R1 and R3b only.
- **R2 cohort pairs** = all pairs within a tissue/cell-type group, except pairs from the same donor pool. Same-donor pools never paired: GTEx blood/liver, TwinsUK blood/LCL. **Sample-overlap status** of the pairs used: GTEx (US post-mortem) × EstBB × TwinsUK: no shared donors (distinct recruitment; not documented as overlapping anywhere we found); BLUEPRINT (Cambridge BioResource, UK) × Quach (Belgium) × OneK1K (Tasmania): none; GEUVADIS (1000 Genomes LCLs) × TwinsUK LCL: none. "None" here means distinct cohorts with no documented overlap; individual-level checks are impossible (controlled access).
- **FinnGen R12 traits** (8): Kanta lab values LDL 3001308, total bilirubin 3006140, urate 3026493, ferritin 3001122, platelets 3007461; endpoints T2D, K11_IBD_STRICT, E4_HYTHY_AI_STRICT. Platelets added for blood-tissue relevance (eQTL blood/monocyte); I9_CHD not used (budget; CHD has few liver-eQTL-explained loci beyond LDL).
- FinnGen per-SNP N: labs N_total from the Kanta documentation; endpoints **N_eff = 4/(1/N_cases + 1/N_controls)** (constant per trait). eQTL per-SNP N = an/2.
- FinnGen in-sample LD (200 GB) is **not downloaded**; it is queried by remote tabix for R3b lead pairs only (local `.tbi` copies, `URL##idx##local.tbi`). So R3b uses **both** pool-B 1000G LD and FinnGen LD. The PSD repair rule of DESIGN §2.5 applies to FinnGen LD *matrices*, which only R4 (deferred) needs; not implemented here.

### Harmonisation (DESIGN §2.5)
- Reference records = data agent's QC'd 1000G SNV table (`data/processed/variants/chrN.parquet`: biallelic SNVs, single-record positions in the raw VCF, MHC excluded). Study records at positions with >1 record (any type) are dropped before the SNV filter.
- AF concordance always against **pool B** (`af_B`) — also for R4b, whose LD panels differ. FinnGen (Finnish) AFs differ genuinely from NFE; flagged non-palindromic SNPs are kept and counted (DESIGN), not dropped.
- Strand-complement matches of non-palindromic SNPs are treated as allele mismatches and dropped (all sources are forward-strand GRCh38; such records indicate errors).
- Variant set of a locus = SNVs present in both traits and in the reference with **MAF ≥ 0.01 in every LD panel of that arm** (for R4b: in the NFE-minus-GEUVADIS reference and in the GEUVADIS sample), inside the gene window.
- z = beta/se after alignment (source values; for FinnGen endpoints beta is log-OR). Standardized beta/se as DESIGN §2.2 (sdY = 1): se = 1/√(2·N·f·(1−f)) with f the study's own ALT frequency (FinnGen `af_alt`, eQTL `ac/an`); pool-B AF is used where the study AF is missing or 0/1 (count logged). Source beta/se are kept in the sidecar `*.variants.parquet`.
- **Build check**: (a) in 5 fixed train-split 1-Mb windows, REF/ALT equal (or swapped) to the GRCh38 1000G record at shared positions; (b) rs429358, rs7412 (APOE), rs12740374 (SORT1), rs4149056 (SLCO1B1) at their GRCh38 positions (sources with rsIDs).

### Windows, splits, gates
- Locus unit = gene window from `datasets/splits/genes_v1.tsv.gz` (TSS ± 500 kb clipped to the same-split block run; BED convention, a 1-based position x is inside iff start < x ≤ end). Every pair inherits its gene's split; MHC-excluded genes are dropped. Genes of any biotype that eQTL Catalogue tests are allowed (genes_v1 lists all GENCODE v39 basic genes).
- **Observed power gate** per trait and window: ≥ 1 credible set with purity ≥ 0.5 (eQTL Catalogue `cs_min_r2` ≥ 0.25; FinnGen `low_purity == FALSE`, which is the same 0.5 rule) whose lead (max PIP) lies in the window and has p < α (molecular 1e-5; GWAS-like 5e-8; DESIGN v4 thresholds by trait type, so both eQTL cohorts of R2 use 1e-5).
- **R2** selection is gene-level only; `same_top_lead` is recorded for information, never used.
- **R3b**: "exactly one CS" for the eQTL = the gene has one CS in the eQTL Catalogue SuSiE output (whole ±1 Mb cis region, any purity) and it passes the gate; for FinnGen = exactly one FinnGen 95% CS (any purity, any region file) with ≥ 1 member in the window, and it is pure, its lead is in the window and p < 5e-8. CSs must share no variant (allele-order-insensitive site match). **LD leads** = the highest-PIP SNV member of each CS that is present and polymorphic in the QC'd pool-B panel (the top variant is often an indel or absent from the panel); lead r² < 0.1 in pool B **and** in FinnGen R12 LD (absent pair in the FinnGen file while variant 1 is present ⇒ r² ≤ 0.01 by construction of the file). Pairs whose LD cannot be computed (no panel SNV in a CS, or variant 1 absent from FinnGen LD) are excluded. Recorded per pair for the label-noise guard of lit/gap_and_evaluation.md §7.5: `max_cs_r2_poolB` (max r² over all cross-CS member pairs; report results with and without pairs > 0.5), lead p-values and z for power matching.
- **R4b**: pairs = GEUVADIS genes passing the gate × (a) each FinnGen trait with a pure CS, lead p < 5e-8, lead in the window (one row per trait; the strongest such CS recorded) and (b) TwinsUK LCL passing the gate (an eQTL × eQTL pair with GEUVADIS as trait 2). Three locus sets with one shared variant set: reference LD = **NFE (pools A ∪ B) minus all GEUVADIS individuals** (n = 138: IBS 107, TSI 16, CEU 10, GBR 5; pool B minus GEUVADIS would leave only 46), in-sample LD = the 445 GEUVADIS individuals with genotypes residualised on 6 genotype PCs (`pcadj`, primary; eQTL Catalogue's qtlmap adjusts for 6 genotype PCs + 6 expression PCs; expression PCs are ignored), and raw in-sample LD (`raw`, sensitivity). Only trait 2 (GEUVADIS) has in-sample LD; trait 1 (FinnGen or TwinsUK) is always with reference LD.
- **GEUVADIS sample list**: eQTL Catalogue does not publish donor IDs. E-GEUV-1 individuals ∩ 1000G phase-3 unrelated (2,504) gives 445 = 358 EUR + 87 YRI, identical to eQTL Catalogue's counts (445; 358 EUR / 87 AFR) — identity inferred, not published. All 445 are in the project pgen (EUR+AFR unrelated).
- **R1** summaries use train-split genes (eQTL; tested genes from the `permuted` file, |z| of the gene lead from its nominal p) and train-split FinnGen CSs (split of the block containing the CS lead).

### Locus-set fields
- `ld_type = "factor"`, `X_ref` = standardized dosages (population SD), `n_ref` = number of panel individuals. Loci of one file with the same panel and identical variant list share one stored `X_ref` (`ld_key`, HDF5 hard link; transparent to readers).
- **ρ̂ (v2, DECISIONS P3-1; `rho_update.py`, real_locus_sets@2).** `rho_hat`, `rho_se` = **per-window** `stats.rho.rho_hat_window` (PRIMARY) with the LD stored in *that* file (the LD given to methods; each R4b LD mode has its own value), simulator settings of `sim.generator.rho_hat_for` (mask max|z| > 4 plus r² > 0.1; pruning r² < 0.1 in a random order seeded `derive(20261029, "real_rho_prune_order", locus_id)`; |z| < 2; unit-variance MLE; 10-block jackknife). `residualize_signal=False` for all real files: the simulator's rule (L0 in-sample for both traits, or n_ref ≥ 1,000) never holds (pool B n = 150, NFE-minus-GEUVADIS n = 138, GEUVADIS in-sample n = 445 for trait 2 only). Diagnostics `rho_n_used`, `rho_n_masked`, `rho_n_pruned`, `rho_method`. `rho_hat_pair`, `rho_se_pair` = **per-cohort-pair pooled** estimate (SENSITIVITY): `rho_hat_pooled` over all **train** windows of (arm, LD mode, dataset1, dataset2), computed from per-window sufficient statistics (identical point estimate, tested), jackknife deleting clusters of windows that share an LD block (overlapping gene windows share SNPs, and FinnGen z are identical across genes); the windows-as-blocks SE of `rho_hat_pooled` is `rho_se_pair_win`; `rho_pair_n_windows`, `rho_pair_n_clusters`. Valid loci carry their pair's train estimate. True ρ is expected ≈ 0 (no shared donors). The M3 values (v1: pruning only, no signal mask, first mode's LD) are kept in `locus_sets/rho_v2.parquet` (`rho_hat_v1`) and in the extract logs. Run `python -m data.real.rho_update` after `extract` (extract writes the per-window attributes itself; the pooled ones need all train windows).
- `ld_role`: R4b `geuv_insample_pcadj` = `primary_insample` (P3 decision: PC-adjusted in-sample LD is the primary in-sample version), `geuv_insample_raw` = `sensitivity_insample`, reference panels = `reference`. `boot_unit`: R3b = FinnGen credible-set id `fg_cs_id` = `<FG trait>|<fg_region>|cs<fg_cs>` (pairs are clustered on 419 CSs); R2/R4b = the candidate's LD block id (DESIGN default).
- `pairing_type` = `real_gwas_x_eqtl` or `real_eqtl_x_eqtl`; `mismatch_level` = `real_ref_poolB`, `real_ref_nfe_minus_geuvadis`, `real_insample_geuvadis_pcadj`, `real_insample_geuvadis_raw`; `cell_id` = arm. Extra attributes: arm, gene, datasets, trait types `trait1_type`/`trait2_type` (`gwas` / `mol`, the simulator's convention, `sim.params.ALPHA` keys; FinnGen = GWAS-like, eQTL = molecular; for the type-specific thresholds of DESIGN v4 §1), leads and whether they survived harmonisation, AF-flag counts, LD panel.
- `<stem>.truth.parquet` holds the arm label (`S_partial_positive` for R2, `D_pseudo_negative` for R3b, `unlabelled_agreement` for R4b) and the selection columns; methods must not open it.

### Test lock
- Test-split pairs are written as identifiers only (`*_test_ids.tsv`: locus_id, arm, datasets, gene, split, window). `extract.py` refuses test rows, and `src/eval/io.py` refuses test loci independently. Building the test ID lists necessarily read the test genes' credible-set metadata (same rule for every split), but no summary statistics or LD of test windows were extracted.
- **Frozen test ID lists (P3-14):** `python -m data.real.freeze_test_ids` wrote `pairs/<arm>_test_ids_sub_v1.tsv` (R2 800, R3b all 1,237, R4b 600 loci; `rng(20261028, "real_test_subsample_v1", arm).choice(n, k, replace=False)`, file order kept; identifier columns only) and `datasets/configs/real_test_ids_v1.sha256` (sha256 of the full and subsampled lists; refuses to overwrite). Verify with `python -m data.real.freeze_test_ids --check` or `sha256sum -c datasets/configs/real_test_ids_v1.sha256`.

### Test arms (p6.realdata_testarm; DECISIONS P3-14, P3-23, P3-24, P3-26)
- **Builder.** `extract.build_test_arm(arm, id_list_path, out_dir, workers=16, mode="test"|"valid")` builds one arm
  for an ID list in the frozen format with `extract_pair` (the train/valid code path: harmonisation, variant set,
  LD panels and modes, per-window P3-1 ρ̂ with each mode's own LD, trait types, `ld_role`, `boot_unit`/`fg_cs_id`).
  It then adds what `rho_update` writes to train/valid loci: `rho_hat_pair`, `rho_se_pair`, `rho_se_pair_win`,
  `rho_pair_n_windows`, `rho_pair_n_clusters` = the cohort pair's **train** pooled estimate, read from
  `locus_sets/rho_v2.parquet` (exact float64 values of real_locus_sets@2). Two P3-23(b) attributes are new:
  `donor_disjoint` (bool; `extract.donor_disjoint`: distinct datasets not in one `SAME_DONOR_POOLS` set; true for
  every R2/R3b/R4b pair by construction) and `rho_sens_zero` (0.0 for donor-disjoint pairs, else NaN).
- **Entry point for the test run.** `src/eval/testrun.py` calls `builder(real_spec, out_dir, workers=n)`.
  `build_test_arm` dispatches a mapping first argument to `build_test_arms`, which builds R2, R3b and R4b from the
  frozen sub lists `pairs/<arm>_test_ids_sub_v1.tsv` (checks `real_spec[arm]["n"]` against the list size) and
  returns `real_R2`, `real_R3b`, `real_R4b_insample` (= `geuv_insample_pcadj`), `real_R4b_reference`
  (= `nfe_minus_geuv`) and `real_R4b_insample_raw` (sensitivity; ignored by testrun), each `{"path", "ids"}`;
  `ids` = loci written (a locus with < 2 shared variants is dropped, as in train/valid, and listed in
  `failed_ids`). Methods file: `real_data_arms.builder: "data.real.extract:build_test_arm"`.
- **Test lock.** Mode "test" refuses unless `eval.io.has_test_access()` is true and `scripts/eval_test.py` is on
  the call stack (parent check, as `sim.testset`); every spawned worker re-checks `has_test_access()` before
  touching a test row (`_job`, `extract_pair(allow_test=True)`), and access errors are raised, never logged as a
  per-locus failure. The ID list's sha256 must equal the frozen `<arm>_test_ids_sub_v1.tsv` hash and the metadata
  file's sha256 must equal `datasets/configs/real_test_meta_v1.sha256`. Outputs never go inside `data/processed/real/` (the registered @2 tree). Workers use
  the `spawn` start method (the test run is multi-threaded).
- **Metadata of the frozen lists** (`test_meta.py`). The frozen lists hold identifiers only, but extraction also
  needs `gene_name`, `block_id` (boot_unit), `lead1`/`lead2` (lead-present flags) and, for R3b, `fg_region`/`fg_cs`
  (boot_unit = FinnGen CS id). M3 kept full columns only for train/valid (`<arm>_candidates.tsv` has no test
  rows), so they are re-derived by re-running the M3 selection code (`eqtl_gene_cs`, `build_r2`,
  `r3b_prefilter`, `build_r4b`; gene table + credible-set metadata only, as at M3; the R3b LD filter is not
  re-run). Verified: rebuilt train+valid rows byte-identical to `R2/R4b_candidates.tsv` and identical on every
  pre-LD column of `R3b_candidates.tsv`; rebuilt test ID columns hash to the frozen sha256 of all six frozen
  files (R3b: pre-LD rows restricted to the frozen ids). The metadata files contain no p-values, z-scores, CS
  members or LD.

## Known properties of the extracted data (M3; details in reports/real_data_engineer_M3.md)
- M3 (v1) per-window ρ̂ was biased upward when the two traits share a signal (true ρ ≈ 0 for these disjoint cohorts): median ρ̂ 0.157 in R2, 0.077 in R4b, −0.002 in R3b. v2 (P3-1 estimator with signal mask): see reports/p4.realdata_rho.md.
- 20–29 % of loci miss a trait's lead variant after SNV-only harmonisation (≈65 % of those leads are indels); flagged per locus (`lead1_present`, `lead2_present`).
- No allele flips were needed (all sources use the GRCh38 REF and the 1000G ALT); R2 lead-sign concordance 99.8 %.
