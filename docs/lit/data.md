# Data & benchmark resources review (GWAS × molecular-QTL colocalization)

Reviewer: data-resources sub-agent. Date of probing: **2026-10-26**.
Method: every URL marked **[probed]** was checked with `curl -sI` / `curl --range` / the bucket JSON or XML listing API on 2026-10-26, and the size given is the server's `Content-Length` or the listing size. Items marked **[unverified]** were not confirmed. Nothing larger than ~80 MB was downloaded; the 79 MB Open Targets colocalisation shard was read only to check its schema.
Abbreviations: CS = credible set, lbf = SuSiE per-variant log Bayes factor (one per single-effect), SS = summary statistics, N = sample size.

---

## 0. Key findings (read this first)

1. **eQTL Catalogue is the backbone QTL source.** It is GRCh38-native and CC BY 4.0, and gives full nominal SS (beta and SE for every variant–gene pair), SuSiE CS, and **per-variant SuSiE lbf for L=1..10** in one uniform format.
   - The stable FTP release is **r7** (June 2024; TSV plus tabix). It includes GTEx **v8** and OneK1K.
   - A **r8 beta** has been published since Sep 2026 (parquet). It includes **GTEx v10** (N up to +20–120% per tissue: blood 670→800, liver 208→261, LCL 147→326). The final r8 is announced for Dec 2026.
   - Full nominal SS (`all`) are released **only for `ge`, `microarray` and `aptamer`**. Splicing and transcript QTLs (`exon`, `tx`, `txrevise`, `leafcutter`) ship only `cc` files: full SS restricted to traits with FDR<1% and at least one CS.
2. **FinnGen** is the best hg38 GWAS source.
   - R12 (500k individuals, Nov 2024) and **R13 (June 2026)** are public with beta/SE, SuSiE (`snp` files carry `lbf_variable1..10`) and FINEMAP.
   - **R12 uniquely ships an in-sample LD matrix**: PLINK2 unphased r for all pairs within 3 Mb with r²>0.01, tabix-indexed, 200 GB in total. This gives an exact *in-sample vs reference LD* contrast on real data.
   - R12 also has **Kanta lab-value quantitative GWAS**: LDL N=369k, bilirubin 190k, urate 81k, ferritin 169k, Hb 447k, all with SuSiE.
3. **No open in-sample LD exists for any eQTL dataset.** GTEx and eQTL Catalogue individual genotypes are controlled-access. Real-data eQTL coloc therefore always runs with mismatched reference LD.
   - GTEx is ancestry-mixed: per the eQTL Catalogue table, 705 of 838 donors are EUR, 82 AFR, 10 EAS, 3 SAS and 38 unassigned.
   - So GTEx-vs-1000G-EUR LD is a *natural* mismatch case.
4. **Open individual-level genotypes are small.** 1000G 30x NYGC (GRCh38) has 3,202 samples, of which about 503 are unrelated EUR. Realistic GWAS-scale N and sample overlap must be simulated at the **summary-statistic level from real LD**: z ~ MVN, with cross-trait noise correlation ρ = N_s·r_p/√(N₁N₂). Individual-level simulation is used only to validate that generator at small N.
5. **Ready-made real-data truth and negative controls are available.**
   - *cis*-pQTL-anchored causal genes (Tambets et al. 2024, built from the INTERVAL aptamer pQTL inside eQTL Catalogue).
   - Open Targets L2G gold standards (26.09).
   - Multiple SuSiE CS per locus in both FinnGen and eQTL Catalogue, with FinnGen in-sample LD, which give "two distinct well-powered signals" negatives.
   - Open Targets 26.09 already has 1.4M+ precomputed coloc rows (COLOC H3/H4 plus eCAVIAR CLPP) for a baseline comparison.

---

## 1. Dataset table

### 1a. Molecular QTL summary statistics

| Name | Content | Build | Size (probed) | URL | Version | Licence | Open? | Notes |
|---|---|---|---|---|---|---|---|---|
| **eQTL Catalogue r7** (stable) | Full nominal cis SS (`*.all.tsv.gz`: molecular_trait_id, chr, pos, ref, alt, ma_samples, maf, pvalue, beta, se, ac, an, r2, gene_id, median_tpm, rsid) + `.tbi`; SuSiE CS (`*.credible_sets.tsv.gz`: pip, cs_id, cs_size, cs_min_r2, region); **SuSiE lbf** (`*.lbf_variable.txt.gz`: molecular_trait_id, region, variant, lbf_variable1..10); permuted p-values | GRCh38 (genotypes re-imputed to 1000G-30x GRCh38 since r5; GENCODE v39 / Ensembl 105 since r6) | Per GTEx tissue: `all` 2.8–4.0 GB, lbf 0.5–2.5 GB, CS 1–5 MB. e.g. liver QTD000266: 2.9 GB / 526 MB / 1.3 MB; blood QTD000356: 2.8 GB / 1.5 GB / 3.3 MB | https://ftp.ebi.ac.uk/pub/databases/spot/eQTL/sumstats/ and `/susie/` **[probed]**. Path index: https://raw.githubusercontent.com/eQTL-Catalogue/eQTL-Catalogue-resources/master/tabix/tabix_ftp_paths.tsv (758 datasets: 280 ge, 113 each of exon/tx/txrev/leafcutter, 25 microarray, 1 aptamer) **[probed]** | r7 (June 2024) | **CC BY 4.0** (data), Apache 2.0 (code) — "License" page **[probed]** | Yes | `all` files only for ge/microarray/aptamer; splicing has `cc` only. N = an/2. Remote tabix by region over HTTPS is the documented use case (tutorial in repo); not run here because tabix is not installed in env. Ancestry per study: `data_tables/population_assignments.tsv` **[probed]** |
| **eQTL Catalogue r8 beta** | Same content, **parquet** (`all`, `cc`, `permuted`, `credible_sets`, `lbf_variable`); adds **GTEx_v10** (49 tissues × 5 quant methods = 294 datasets), MAGE, IBDverse, INTERVAL_RNA, re-annotated OneK1K/Perez/Randolph/Cytoimmgen | GRCh38 | GTEx_v10 adipose QTD000116: all 4.8 GB, lbf 3.4 GB, CS 14 MB; liver QTD000266: 3.6 GB / 1.2 GB / 4.8 MB; blood QTD000356 all 3.6 GB | https://ftp.ebi.ac.uk/pub/databases/spot/eQTL/r8_beta/ **[probed]**; metadata `data_tables/dataset_metadata_r8_beta.tsv` (592 datasets) and `manuscript/r8_tables/GTEx_v8_vs_v10_sample_sizes.tsv` **[probed]** | r8 pre-release (dirs dated 2026-09-09/10); final r8 "Dec 2026" per release notes | CC BY 4.0 | Yes | Beta: files may change. Record sha256 at download, or pin r7 for the primary analysis |
| eQTL Catalogue chromatin QTLs (caQTL/histone) | Dataset ID map only (BLUEPRINT H3K27ac/H3K4me1, Kumasaka ATAC, AFGR, iPSCORE …; N 56–152) | — | — | `data_tables/chromatin_QTLs/chromatin_dataset_id_map.tsv` **[probed]**; FTP `sumstats/QTS1000xx` → **404** **[probed]** | upcoming | — | **Not yet released** | No open, uniformly processed caQTL full SS found. Other caQTL sources **[unverified]** |
| GTEx v8 (portal) | eGenes/signif pairs tar 1.56 GB; EUR-only tar; fine-mapping CAVIAR 0.90 GB, DAP-G 0.76 GB, CaVEMaN 17 MB; edQTL allpairs | GRCh38 | as listed | https://storage.googleapis.com/adult-gtex/bulk-qtl/v8/ **[probed]** | v8 | GTEx open-access tier (summary data public; no explicit CC licence found) **[licence text unverified]** | Partly | **All-pairs eQTL/sQTL SS are in a *requester-pays* GCP bucket** (`gtex-resources`; sQTL allpairs about 690 GB per portal API) → not free. Use the eQTL Catalogue recomputation instead (free, has SE and lbf) |
| GTEx v10 (portal) | eQTL tar 2.56 GB (eGenes + signif_pairs.parquet), sQTL 1.96 GB, apaQTL 0.42 GB; **SuSiE eQTL 174 MB** (`*.v10.eQTLs.SuSiE_summary.parquet`), SuSiE sQTL 169 MB, apaQTL 36 MB; expression matrices 1.5 GB; small-RNA eQTL | GRCh38 (variant IDs `_b38`, dbSNP155) | as listed | https://storage.googleapis.com/adult-gtex/bulk-qtl/v10/ **[probed]** | v10 | as above | Partly | Public `…/eQTL-Tissue-Specific-All-SNP-Gene-Associations/` folder is **empty** in the public listing **[probed]**. Full SS via eQTL Catalogue r8 beta |
| OneK1K | sc-eQTL pseudobulk, 17 (r7) to 24 cell types, N 534–981 (e.g. CD4_Naive 981, CD14_Mono 959) | GRCh38 (eQTL Catalogue reprocessing) | CD4_Naive QTD000612: 2.7 GB all, 1.2 GB lbf; CD14_Mono QTD000609: 1.4 GB / 123 MB | eQTL Catalogue QTS000038 **[probed]** | r7 / r8β (re-annotated) | CC BY 4.0 (via eQTL Catalogue) | Yes | Original OneK1K portal not probed |
| INTERVAL SomaScan pQTL (Sun 2018) | Plasma protein (aptamer) full nominal SS, CS, lbf; N=3,301 (Tambets 2024) | GRCh38 (eQTL Catalogue) | QTD000584: 732 MB all, 460 MB lbf, 1 MB CS | https://ftp.ebi.ac.uk/pub/databases/spot/eQTL/sumstats/QTS000035/QTD000584/ **[probed]** | r7 | CC BY 4.0 | Yes | **Open cis-pQTL source for "causal gene = coding gene" truth** (Tambets 2024). INTERVAL also contributes to blood-trait GWAS meta-analyses → a real **sample-overlap** case |
| UKB-PPP (Sun 2023) | Olink pQTL SS, ~2,923 proteins, ~54k UKB participants | per study: GRCh38 and GRCh37 **[unverified]** | not probed | Synapse syn51364943 (https://doi.org/10.7303/syn51364943) | 2023 | AWS Open-Data registry lists CC BY **[unverified]** | Needs free Synapse account + access form, no DAC | Fully overlaps UKB GWAS (Pan-UKBB/Neale), which makes it the ideal real-data *overlap* case. **Open Targets 26.09 already has UKB-PPP EUR SuSiE CS (`input/molqtl/study/ukb_ppp_eur/`, CC0)** **[probed]** |
| deCODE pQTL (Ferkingstad 2021; Eldjarn 2023) | SomaScan pQTL, ~35k Icelanders | **[unverified]** | not probed | decode.com/summarydata (download form) | — | deCODE terms **[unverified]** | Form-gated | Not needed for minimal plan |
| eQTLGen phase I | Blood cis-eQTL meta-analysis, 31,684 samples / 37 cohorts. Columns: Pvalue, SNP, SNPChr, SNPPos, AssessedAllele, OtherAllele, **Zscore**, Gene, NrCohorts, NrSamples, FDR — **no beta/SE** | **hg19** | full cis file 3.88 GB | https://molgenis26.gcc.rug.nl/downloads/eqtlgen/cis-eqtl/2019-12-11-cis-eQTLsFDR-ProbeLevel-CohortInfoRemoved-BonferroniAdded.txt.gz **[probed]** | 2019-12-11 | not stated on site **[unverified]** | Yes | Needs liftover. beta/SE derivable from z, allele freq (separate AF file) and N. Per-SNP N varies. **Overlaps many GWAS cohorts** (e.g. FHS, EstBB, LifeLines, Rotterdam) |
| eQTLGen phase II | Genome-wide blood eQTL meta-analysis | — | — | eqtlgen.org says "currently in its second phase" **[probed]** | — | — | **No public SS found** | |
| MetaBrain | Brain cis-eQTL meta-analysis, n≤2,759 | **[unverified]** | — | metabrain.nl **[unverified]** | 2023 | — | **[unverified]** | Not needed |
| sQTL (open, full SS) | eQTL Catalogue leafcutter/txrevise `cc` files; GTEx v10 sQTL signif pairs + SuSiE | GRCh38 | per dataset `cc` ≈ 0.7–2 GB (r8β adipose `cc` 1.9 GB) | as above | r7/r8β | CC BY 4.0 | Yes (cc only) | `cc` is sufficient for coloc of significant traits but is ascertained. State this when used |

### 1b. GWAS summary statistics

| Name | Content | Build | Size (probed) | URL | Version | Licence | Open? | Notes |
|---|---|---|---|---|---|---|---|---|
| **FinnGen R12** | ~2,469 endpoints (manifest). Columns: #chrom, pos (GRCh38), ref, alt (effect), rsids, nearest_genes, pval, mlogp, beta, sebeta, af_alt, af_alt_cases, af_alt_controls. **SuSiE** (`*.SUSIE.snp.bgz` with alpha1..10, mean/sd, **lbf_variable1..10**; `.cred`, `.cred_99`) + FINEMAP | **GRCh38** | per endpoint ≈ 0.82 GB (T2D 821 MB); summary_stats total 1.98 TB; finemap total 173 GB; SuSiE snp file per trait ~0.8–2 GB (tabix-indexed) | https://storage.googleapis.com/finngen-public-data-r12/summary_stats/finngen_R12_manifest.tsv **[probed]**; `…/summary_stats/release/finngen_R12_<PHENO>.gz` **[probed]**; `…/finemap/full/susie/` **[probed]** | R12 (public 2024-11-04) | No formal licence; free after online form; must acknowledge FinnGen and cite Kurki 2023 (finngen.fi/en/access_results) **[probed]** | Yes (form) | N≈500k. T2D 82,878 cases / 403,489 controls; I9_CHD 56,650 / 443,698; E4_HYTHY_AI_STRICT 54,752 / 362,639 |
| **FinnGen R12 in-sample LD** | All variant pairs within 3 Mb with r²>0.01; columns chr, pos, variant1, variant2, **r**, r2 (PLINK2 `--r-unphased`, 520,210 samples) | GRCh38 | 47 files, **199.8 GB** total; chr1 13.2 GB; tabix-indexed | https://storage.googleapis.com/finngen-public-data-r12/ld_matrix/ (README) **[probed]** | R12 | as FinnGen | Yes | **Region queries by tabix avoid full download.** Truncated at r²≤0.01 (set to 0) — note when used as "true" LD |
| **FinnGen R12 Kanta lab values** | Quantitative lab GWAS (IRN-transformed) + SuSiE/FINEMAP: LDL (OMOP 3001308, N=368,976), total bilirubin (3006140, 190,407), ferritin (3001122, 168,595), urate (3026493, 81,374), Hb (3000963, 447,271), platelets, creatinine/eGFR … | GRCh38 | LDL SS 490 MB, LDL SuSiE snp 795 MB | https://storage.googleapis.com/finngen-public-data-r12/lab_values/ (summary_stats/, finemap/, analysis_documentation/Kanta_labs_GWAS_results_v2_summary.txt) **[probed]** | R12 | as FinnGen | Yes | **Quantitative traits with known liver-eQTL biology (LDL–SORT1, bilirubin–UGT1A1).** Same individuals as R12 LD matrix |
| FinnGen R13 | 2,754 endpoints (manifest), same format; SuSiE + FINEMAP; annotated variants 29.8 GB | GRCh38 | summary_stats 2.17 TB total (~0.76–0.80 GB per endpoint); finemap 191 GB | https://storage.googleapis.com/finngen-public-data-r13/ **[probed]** | R13 (public 2026-06-02; N=500,186) | as FinnGen | Yes | **No LD matrix in R13 bucket** (no `ld_matrix/` prefix) → use R12 when in-sample LD is needed. R10/R11 buckets also exist **[probed]** |
| GWAS Catalog harmonised | `harmonised/*.h.tsv.gz` + `.tbi` + meta YAML (GWAS-SSF / pre-GWAS-SSF), GRCh38 coordinates. e.g. GLGC 2021 LDL EUR **GCST90239658**: N up to 1,320,016, per-variant `n`, N_studies, `hm_code` | **GRCh38** (harmonised) | GCST90239658.h.tsv.gz **1.3 GB** | https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90239001-GCST90240000/GCST90239658/harmonised/ **[probed]** | continuous | "majority CC0 or EMBL-EBI terms; check per-study *Usage License*" (ebi.ac.uk/gwas/downloads/summary-statistics) | Yes | Meta-analyses with variable per-SNP N and many cohorts → **sample overlap with QTL cohorts is likely and undocumented**. Good *stress test*, poor primary source |
| Pan-UKBB | Per-phenotype multi-ancestry SS (e.g. phenocode 30690 cholesterol: 2.29 GB) | **GRCh37** | manifest 1.4 MB; per-phenotype ~2 GB | https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_release/phenotype_manifest.tsv.bgz **[probed]** | 2020/2024 (Karczewski medRxiv 2024) | CC BY 4.0 **[unverified]** | Yes | Needs liftover. UKB overlaps UKB-PPP pQTL fully |
| UKB Neale lab round 2 | Per-phenotype additive SS (`*.varorder.tsv.bgz`, e.g. 30780_irnt both_sexes 464 MB) | **GRCh37** | ~0.46 GB per phenotype | https://broad-ukb-sumstats-us-east-1.s3.amazonaws.com/round2/additive-tsvs/ **[probed listing]** | round 2 (2018) | **[unverified]** | Yes | Variant manifest needed for alleles |
| Open Targets Platform **26.09** | `credible_set` (2.85 GB parquet; GWAS SuSiE / SuSiE-inf / PICS **and** QTL SuSiE CS incl. eQTL Catalogue, **GTEx_v10**, UKB-PPP EUR, FinnGen, GWAS Catalog SS fine-mapped). `colocalisation` (15.8 GB; columns leftStudyLocusId, rightStudyLocusId, rightStudyType, colocalisationMethod, numberColocalisingVariants, h3, h4, clpp, betaRatioSignAverage; method label `COLOC_PIP_ECAVIAR` in shard 0; rightStudyType eqtl/sqtl/tuqtl/pqtl/sceqtl/gwas). `l2g_prediction` 0.49 GB; `study` 93 MB; `input/l2g/gold_standard.json` | GRCh38 | as listed | https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.09/ (released 2026-09-24) **[probed]** | 26.09 | **CC0 1.0** (croissant.json) **[probed]**; underlying sources keep their own terms | Yes | Baseline comparator and source of real-data labels. **Caution:** OT uses its own CS and fine-mapping (circularity if used as truth for a coloc method) |
| BBJ | East-Asian biobank GWAS | mostly hg19 **[unverified]** | — | https://pheweb.jp/ (HTTP 200) **[probed]** | — | **[unverified]** | Yes | Out of scope for minimal plan (EUR LD) |

### 1c. Sample-overlap matrix (QTL × GWAS)

| QTL ↓ / GWAS → | FinnGen | UKB (Pan-UKBB, Neale) | GWAS Catalog meta-analyses (GLGC, DIAGRAM, blood-cell …) |
|---|---|---|---|
| GTEx v8/v10 (US post-mortem donors) | none | none | none expected |
| eQTL Catalogue BLUEPRINT, OneK1K (Australia), ROSMAP, CommonMind … | none | none | none / small |
| eQTL Catalogue FUSION (Finnish muscle/adipose) | **possible, undocumented** (Finnish cohorts) **[unverified]** | none | possible (FUSION in DIAGRAM) **[unverified]** |
| eQTL Catalogue Lepik_2017 (Estonian Biobank), TwinsUK | none | none (TwinsUK ≠ UKB) | **likely** (EstBB, TwinsUK in many consortia) |
| INTERVAL pQTL (Sun 2018, eQTL Catalogue QTS000035) | none | none | **yes** for blood-cell GWAS that include INTERVAL **[study IDs unverified]** |
| eQTLGen phase I | none known | none | **yes** (e.g. FHS, EstBB, Rotterdam, LifeLines) |
| UKB-PPP | none | **complete (pQTL sample ⊂ UKB)** | yes for UKB-containing meta-analyses (e.g. GLGC 2021) |

**Clean no-overlap pairs:** GTEx/BLUEPRINT/OneK1K × FinnGen. **Natural overlap pairs:** UKB-PPP × Pan-UKBB (needs Synapse plus liftover; OT has UKB-PPP CS under CC0), and INTERVAL pQTL × INTERVAL-containing blood-trait GWAS.

### 1d. Individual-level genotypes (open) and LD resources

| Name | Content | Build | Size (probed) | URL | Version | Licence | Open? | Notes |
|---|---|---|---|---|---|---|---|---|
| **1000 Genomes 30x (NYGC)** phased panel | 3,202 samples (2,504 unrelated + 698 related), SNV/INDEL/SV, statistically phased, filtered | **GRCh38** | chr1 2.2 GB … chr22 425 MB; chrX v2 2.5 GB; **autosomes ≈ 33 GB** | https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/1kGP_high_coverage_Illumina.chr{N}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz (+ .tbi) **[probed]**; populations: `…/1000G_2504_high_coverage/20130606_g1k_3202_samples_ped_population.txt` **[probed]** | 2022-04-22 panel (chrX 2022-08-04) | IGSR open data; data-reuse README asks for Coriell/NYGC acknowledgement and citation of Byrska-Bishop 2022 **[probed]** | Yes | EUR in 3,202 set: CEU 179, FIN 99, GBR 91, IBS 157, TSI 107 (633; 525 founders); 503 EUR among the unrelated 2,504 (standard). Same panel eQTL Catalogue used for imputation (r5+) |
| HGDP+1kGP (gnomAD v3.1.2 subset) | 4,151 genomes; joint VCF | GRCh38 | **chr22 59 GB, chr1 280 GB** (full VCF); phased BCFs `resources/hgdp_1kg/phased_haplotypes/…chr{N}.merged.bcf` 10–18 GB each | https://storage.googleapis.com/gcp-public-data--gnomad/release/3.1.2/vcf/genomes/gnomad.genomes.v3.1.2.hgdp_tgp.chr22.vcf.bgz **[probed]**; phased BCF listing **[probed]** | v3.1.2 | gnomAD: open (CC0 for gnomAD data **[unverified for HGDP subset]**) | Yes | Adds ~150 European HGDP samples **[count unverified]**. Not needed for minimal plan |
| HAPNEST synthetic | 1,008,000 synthetic individuals, 6.8M common variants, 9 traits | **[unverified]** | not probed | https://www.ebi.ac.uk/biostudies/studies/S-BSST936 | 2023 | **[unverified]** | Yes | Optional large-N individual-level option; build must be checked |
| **FinnGen R12 LD** | see 1b | GRCh38 | 200 GB (tabix) | see 1b | R12 | FinnGen | Yes | **Only open hg38 in-sample biobank LD found** |
| UKB LD (Weissbrod/PolyFun) | 337k British-ancestry UKB, 3 Mb windows (1 Mb step), `.npz` + variant list `.gz` | **GRCh37** (PolyFun docs; positions consistent with hg19 **[partly verified]**) | **2,763 windows, 3.06 TB**; ~0.9–1.0 GB per window | https://broad-alkesgroup-ukbb-ld.s3.amazonaws.com/UKBB_LD/chr1_1_3000001.npz **[probed]** | 2020 | **[unverified]** | Yes | Would require liftover of variant positions (matrix content build-independent) |
| Pan-UKBB LD | Hail BlockMatrix per ancestry (`UKBB.EUR.ldadj.bm`); variant index tables include **`ldadj.variant.b38.ht`** (GRCh38 coords) | GRCh37 (with b38 variant table) | **EUR BlockMatrix 15.5 TB (124,283 parts)** | https://pan-ukb-us-east-1.s3.amazonaws.com/ld_release/ **[probed]** | 2020 | CC BY 4.0 **[unverified]** | Yes | Needs Hail; block-wise random access possible |
| gnomAD v2.1.1 LD | Hail BlockMatrix per pop (afr, amr, asj, eas, est, fin, nfe, nwe, seu), common variants | **GRCh37** | not summed | gs://gcp-public-data--gnomad/release/2.1.1/ld/ (listing) **[probed]** | 2.1.1 | gnomAD open | Yes | Used by Open Targets for GWAS Catalog SuSiE-inf |

### 1e. LD regions, annotation, liftover, harmonisation

| Name | Content | Build | Size | URL | Licence | Notes |
|---|---|---|---|---|---|---|
| **MacDonald et al. 2022 GRCh38 LD blocks** | LDetect on 1000G GRCh38: `deCODE_EUR_LD_blocks.bed` (**1,361 blocks**, mean 2.06 Mb), `pyrho_{AFR,EAS,EUR,SAS}_LD_blocks.bed` (EUR **1,336 blocks**, mean 1.96 Mb) | **GRCh38** | KB | https://raw.githubusercontent.com/jmacdon/LDblocks_GRCh38/master/data/ **[probed]** | GitHub licence "NOASSERTION" → cite, download by script, do not vendor | Native hg38 (not lifted). Recommended split unit |
| Berisa & Pickrell 2016 | LDetect blocks (EUR 1,703) | hg19 | KB | bitbucket nygcresearch/ldetect-data **[unverified]** | — | Do not liftover blocks (fragmentation, per MacDonald) |
| **GENCODE v39** | Gene annotation used by eQTL Catalogue (r6+) | GRCh38 | `gencode.v39.annotation.gtf.gz` 47.6 MB; basic 28.4 MB | https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_39/ **[probed]** | Open (no restrictions) | Latest is release_50 **[probed]**. **Use v39** so gene IDs match eQTL Catalogue and GTEx v10 |
| UCSC chain hg19→hg38 | liftOver chain | — | 228 KB | https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz **[probed]** | UCSC, free download | Also hg38ToHg19 (1.2 MB) **[probed]** and Ensembl `GRCh37_to_GRCh38.chain.gz` (285 KB) **[probed]** |
| GRCh38 reference FASTA (1000G analysis set) | needed for allele/strand checks in liftover | GRCh38 | 3.26 GB | https://ftp.ncbi.nlm.nih.gov/1000genomes/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa **[probed]** | open | |
| Tools | **BCFtools/liftover** (Genovese 2024; handles indels and ref/alt swaps), **MungeSumstats** (Bioconductor; QC + liftover), **GWAS-SSF** format and harmonisation pipeline (GWAS Catalog) | — | — | — | open source | Only Pan-UKBB, Neale, eQTLGen and UKB-LD need liftover in the plan below — and the minimal plan needs none |

### 1f. Ground-truth / validation resources

| Resource | What it gives | Access | Notes |
|---|---|---|---|
| **Tambets et al. 2024 (HGG Adv)** benchmark | 793 proteins with cis-pQTL where the causal gene is assumed to be the protein-coding gene; benchmarked coloc.abf, coloc.susie, CLPP and 5 MR methods against 131 eQTL Catalogue datasets. coloc.susie recall 46.3%, precision 45.1%; nearest-gene to fine-mapped pQTL: 71.9% / 76.9% | Built entirely from open eQTL Catalogue data (INTERVAL aptamer QTD000584 + eQTL datasets) | **Best open, reproducible real-data gene-level benchmark.** Shows eQTL co-regulation of neighbours → precision is the hard part |
| Open Targets gold standards | Curated GWAS locus → causal gene (OTG 2019 set: `gwas_gold_standards.191108.tsv` 1.3 MB; current `26.09/input/l2g/gold_standard.json` ~5 parts of 3–4.5 MB + `predefined_test_for_split/`) | https://raw.githubusercontent.com/opentargets/genetics-gold-standards/master/gold_standards/processed/gwas_gold_standards.191108.tsv **[probed]**; https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.09/input/l2g/ **[probed]** | Gene-level, not variant-level. OT's own split must not leak into ours |
| Open Targets coloc 26.09 | h3/h4/CLPP for all CS pairs | see 1b | Comparator, not truth |
| MPRA | Abell et al. 2022 (Science; GTEx eQTL variants, allelic MPRA in LCL/HepG2…); Siraj et al. 2024 bioRxiv (MPRA of fine-mapped complex and molecular trait variants) | supplementary tables **[not probed]** | Variant-level positives for "shared causal variant" |
| CRISPR | Morris et al. 2023 Science (STING-seq: blood-trait GWAS variants → target genes); ENCODE-rE2G CRISPRi enhancer–gene benchmark (Gschwind et al. bioRxiv 2023) | supplementary / ENCODE **[not probed]** | Gene-level positives |
| Natural negatives | Loci with ≥2 SuSiE CS in the QTL **or** GWAS (eQTL Catalogue lbf L=1..10; FinnGen SuSiE L=1..10) where leads are in low LD (r²<0.1 in FinnGen in-sample LD) and both CS are well-powered (e.g. |z|>6) | Open (above) | Gives "two distinct, both detectable" pairs. **Labels need care:** a CS pair across studies is "distinct" only if it shows low LD in *both* panels |
| Replication positives | Same gene, same tissue, two independent cohorts (e.g. GTEx blood vs Lepik_2017 blood vs OneK1K pseudo-bulk; GTEx v10 liver vs another liver if available) | eQTL Catalogue | eQTL–eQTL "should-colocalise" pairs with real LD/ancestry mismatch (GTEx admixed vs Estonian) |
| Simulation frameworks in the literature | coloc (Giambartolomei 2014: simulated from haplotype panel); coloc-SuSiE (Wallace 2021; simGWAS); simGWAS (Fortune & Wallace 2019: fast case-control SS simulation from reference haplotypes); SharePro (Zhang 2024: UKB-genotype simulations); ColocBoost (Cao et al. medRxiv 2025: multi-trait simulations from ROSMAP genotypes **[details unverified]**); eCAVIAR (Hormozdiari 2016); Hukku 2021 (enloc/fastENLOC simulation) | papers | No public *benchmark dataset* for coloc was found. Benchmarks are simulation-based or gene-level gold standards |

---

## 2. Recommended data plan (minimal but sufficient)

All builds GRCh38 natively. **No liftover needed in the core plan.** Pin versions and record sha256 in DATA.md. Commit only derived results.

### (a) Simulation with real hg38 LD — "1000G-30x LD → summary-statistic generator"

**Download** (≈33 GB plus about 5 GB of derived PLINK2 files):
- 1000G 30x phased panel, chr1–22, from the URL in 1d.
- The population file `20130606_g1k_3202_samples_ped_population.txt`.
- Keep the 2,504 unrelated samples. Filter to biallelic SNVs (optionally indels) with MAF ≥ 0.01 in EUR (or ≥ 0.005).
- Convert to PLINK2 `pgen` per chromosome.

**Loci.** Sample loci from the MacDonald hg38 EUR blocks (pyrho_EUR, 1,336 blocks).
- Within each block, take a ~1 Mb window around a GENCODE v39 protein-coding TSS, to mimic a cis window of ±1 Mb as in eQTL Catalogue.
- Assign the **whole block** (and hence every gene whose TSS is in it) to one split: train / val / test.
- Genes whose ±1 Mb cis window spans two blocks go with the block that contains the TSS. Both blocks are then forced into the same split (union-find on blocks linked by cis windows).

**Generator** (truth known exactly):
- Let R_true = LD of the "study" panel and b₁, b₂ = sparse causal effects.
- Draw z_k = √N_k·R_true·b_k + e_k, with e ~ MVN(0, R_true) and Cov(e₁, e₂) = ρ·R_true.
- Set ρ = N_s·r_p/√(N₁N₂) (LDSC cross-trait result: Bulik-Sullivan et al. 2015, Nat Genet, doi:10.1038/ng.3406 — DOI not re-probed here).
- **This gives exact control of sample overlap** (N_s = 0 … min(N₁,N₂); r_p = phenotypic correlation, e.g. 0–0.5).
- Use N_QTL ∈ {100, 300, 700, 1,000}, matching GTEx/OneK1K, and N_GWAS ∈ {50k, 200k, 500k} (FinnGen), with heritability per causal variant tuned to target NCPs.

**Scenarios** must include the key comparison:
- H4: shared causal variant.
- H3-strong: two distinct causal variants at controlled LD r² ∈ {<0.1, 0.1–0.5, 0.5–0.8, >0.8}, **both above a power threshold**.
- H3-weak / "too weak": one or both effects below detection.
- Multiple causal variants (1–3 per trait).

**LD-reference mismatch.** Split EUR (503) into disjoint panels:
- **in-sample**: the R used to generate = the R given to the method.
- **out-of-sample, same population**: disjoint EUR subsets of size 100 / 200 / 400.
- **subpopulation mismatch**: generate with non-Finnish EUR (CEU+GBR+TSI+IBS, ~404 unrelated), analyse with FIN (99) or vice versa.
- **ancestry-mixed study**: generate with ~85% EUR + 10% AFR, to mimic GTEx 705 EUR / 82 AFR; analyse with EUR-only.
- **reference size**: shrinkage / regularisation of small panels.
- Also cover **allele-flip / strand errors** and **missing variants** (drop 5–20% of variants from the reference).

**Individual-level sanity check (small N).** Simulate expression and phenotype on actual 1000G genotypes, including shared individuals for overlap, run real regressions, and confirm that the SS generator reproduces method behaviour. Optional: HAPNEST (S-BSST936) for large-N individual-level checks, after confirming its build.

**Gap:** a true hg38 LD from ~400 haplotype samples is noisy at rare variants. Restrict causal variants to MAF ≥ 0.01, or use FinnGen R12 r as the "population" LD for a FIN-specific arm (truncated at r²>0.01).

### (b) Real-data evaluation — eQTL Catalogue (hg38) × FinnGen R12 (hg38)

**QTL (eQTL Catalogue r7, stable; ~35 GB).** `all` + `lbf` + `credible_sets` for:
- GTEx liver QTD000266, blood QTD000356, thyroid QTD000341, adipose subcutaneous QTD000116, colon transverse QTD000231, artery tibial QTD000141.
- OneK1K CD4_Naive QTD000612 and CD14_Mono QTD000609.
- BLUEPRINT monocyte QTD000021.
- INTERVAL aptamer pQTL QTD000584.
- Sensitivity arm: the same GTEx tissues from **r8 beta (GTEx v10)**, parquet, ~+30 GB. Larger N gives a power comparison; checksum at download.

**GWAS (FinnGen R12; ~15 GB).**
- Lab values: LDL 3001308, total bilirubin 3006140, urate 3026493, ferritin 3001122, Hb 3000963, platelets 3007461.
- Endpoints: T2D, E4_HYTHY_AI_STRICT, I9_CHD, K11_IBD_STRICT (10,960 cases / 489,388 controls), L12_PSORIASIS, M13_RHEUMA.
- Plus the matching `SUSIE.snp.bgz` (≈0.8–2 GB each; or tabix by region).

**In-sample GWAS LD.** Use FinnGen R12 `ld_matrix` **by tabix region only** for the loci used, a few GB in total instead of 200 GB.
- This enables: GWAS-side in-sample vs 1000G-EUR vs 1000G-FIN reference.
- QTL side is always reference LD (1000G EUR, or EUR+AFR mixture for GTEx).

**Overlap.** GTEx/OneK1K/BLUEPRINT × FinnGen has none. That is clean, so overlap robustness comes from simulation plus an optional real overlap arm:
- **INTERVAL pQTL × INTERVAL-containing blood-cell GWAS** (GWAS Catalog harmonised hg38; study ID to be confirmed), or
- **UKB-PPP (Synapse) × Pan-UKBB** (requires liftover; heavier).

**Second GWAS source for robustness.** GLGC 2021 LDL EUR GCST90239658 (GWAS Catalog harmonised hg38, 1.3 GB, N up to 1.32M, contains UKB; per-SNP N varies).

**Truth / labels.**
- (i) Tambets-style cis-pQTL gene truth: INTERVAL pQTL vs eQTL coloc → coding gene.
- (ii) OT 26.09 gold standards for FinnGen/GLGC loci (gene-level).
- (iii) MPRA/STING-seq positives where loci overlap.
- (iv) **Natural negatives**: multi-CS loci with well-powered distinct signals, defined using FinnGen in-sample LD.
- (v) eQTL–eQTL replication positives (GTEx blood vs OneK1K/BLUEPRINT for the same gene).
- (vi) **Comparator**: OT 26.09 coloc table (COLOC/eCAVIAR) restricted to the same CS pairs.

**Splits.** Same MacDonald pyrho_EUR hg38 blocks and GENCODE v39 gene→block assignment as in (a). The test-set blocks are fixed before any real-data look.

### (c) Tissue–trait pairs with known biology (positive anchors)

| GWAS (FinnGen R12) | QTL dataset | Expected locus / gene | Rationale |
|---|---|---|---|
| LDL (3001308) | GTEx liver | *SORT1* / *CELSR2–PSRC1* (1p13; rs12740374), *HMGCR*, *PCSK9* region | Classic liver-eQTL→LDL mechanism (Musunuru 2010) |
| Total bilirubin (3006140) | GTEx liver | *UGT1A1* | Very strong cis effect (Gilbert); positive control with huge NCP |
| Urate (3026493) | GTEx kidney cortex (N 73/103 — underpowered) or liver | *SLC2A9*, *ABCG2* | Good "too weak to call in QTL" example (tiny kidney N) |
| Ferritin (3001122) / Hb | GTEx liver / blood | *HFE*, *TMPRSS6* | Iron biology |
| Platelets / Hb | GTEx blood, OneK1K, BLUEPRINT | many blood-cell eQTLs | Multi-dataset replication |
| Autoimmune hypothyroidism | GTEx thyroid (N 574/681) | thyroid-expressed genes (e.g. *FOXE1* region **[expected, unverified]**) | Large-N tissue |
| IBD, psoriasis, RA | OneK1K CD4_Naive / CD14_Mono, BLUEPRINT | immune-cell eQTLs (e.g. *IL23R*/*IL18RAP* regions **[expected, unverified]**) | Cell-type specificity tests |
| T2D | GTEx adipose / muscle / pancreas | e.g. *TCF7L2* (**no eQTL expected** → canonical "GWAS hit without QTL") | Absence-of-coloc case |
| Plasma protein (INTERVAL) | eQTL Catalogue, all tissues | coding gene of the protein | Tambets-style gene truth |

---

## 3. Licences and what may be committed

| Source | Redistribution of raw files | Derived results in git |
|---|---|---|
| eQTL Catalogue | CC BY 4.0 → allowed with attribution, but **do not commit (size)**; script the download | Yes (cite Kerimov 2021) |
| Open Targets Platform | CC0 | Yes |
| GWAS Catalog | CC0 / EMBL-EBI terms (check per-study "Usage License") | Yes, with citation |
| FinnGen R10–R13 | No explicit licence; access via form; acknowledgement and citation required → **do not redistribute raw files** | Derived statistics, metrics and figures: yes (standard practice), with the acknowledgement sentence |
| GTEx (open tier) | Public summary data; no explicit CC licence found → do not redistribute | Yes, cite GTEx 2020 |
| 1000G 30x | Open; acknowledge Coriell/NYGC and cite Byrska-Bishop 2022 | Derived LD and simulated SS: yes, but large → store outside git and commit generator script + seeds + checksums |
| UKB-PPP, Pan-UKBB, Neale, UKB LD | Terms **[unverified]**; UKB-PPP gated by Synapse → do not redistribute | Derived only |
| MacDonald LD blocks | No licence asserted → download by script, cite; do not copy into repo | Split assignment table (block IDs → split): yes |
| GENCODE, UCSC chains | Open | Gene→block table: yes |
| **Never commit** | any individual-level genotype (even open 1000G: size), any controlled-access data (none planned) | — |

---

## 4. Verified references (DOIs resolved via Crossref/doi.org on 2026-10-26)

- Giambartolomei et al. 2014, PLoS Genet — coloc. https://doi.org/10.1371/journal.pgen.1004383
- Wallace 2021, PLoS Genet — coloc + SuSiE. https://doi.org/10.1371/journal.pgen.1009440
- Zou et al. 2022, PLoS Genet — SuSiE-RSS (summary-data fine-mapping, LD mismatch). https://doi.org/10.1371/journal.pgen.1010299
- Hormozdiari et al. 2016, AJHG — eCAVIAR. https://doi.org/10.1016/j.ajhg.2016.10.003
- Hukku et al. 2021, AJHG — probabilistic colocalisation, promise and limitations. https://doi.org/10.1016/j.ajhg.2020.11.012
- Zhang et al. 2024, Bioinformatics — SharePro coloc. https://doi.org/10.1093/bioinformatics/btae295
- Cao et al. 2025, medRxiv — ColocBoost (multi-omics xQTL coloc). https://doi.org/10.1101/2025.04.17.25326042
- Fortune & Wallace 2019, Bioinformatics — simGWAS. https://doi.org/10.1093/bioinformatics/bty898
- Chen et al. 2021, Nat Commun — DENTIST (SS/LD heterogeneity QC). https://doi.org/10.1038/s41467-021-27438-7
- Kanai et al. 2022, Cell Genomics — SLALOM / miscalibrated meta-analysis fine-mapping. https://doi.org/10.1016/j.xgen.2022.100210
- Tambets et al. 2024, HGG Adv — eQTL coloc benchmark with cis-pQTL truth. https://doi.org/10.1016/j.xhgg.2024.100348
- Mostafavi et al. 2023, Nat Genet — systematic differences between eQTL and GWAS discovery. https://doi.org/10.1038/s41588-023-01529-1
- Kerimov et al. 2021, Nat Genet — eQTL Catalogue. https://doi.org/10.1038/s41588-021-00924-w
- GTEx Consortium 2020, Science — GTEx v8. https://doi.org/10.1126/science.aaz1776
- Yazar et al. 2022, Science — OneK1K. https://doi.org/10.1126/science.abf3041
- Sun et al. 2018, Nature — INTERVAL plasma proteome. https://doi.org/10.1038/s41586-018-0175-2
- Sun et al. 2023, Nature — UKB-PPP. https://doi.org/10.1038/s41586-023-06592-6
- Võsa et al. 2021, Nat Genet — eQTLGen phase I. https://doi.org/10.1038/s41588-021-00913-z
- Kurki et al. 2023, Nature — FinnGen. https://doi.org/10.1038/s41586-022-05473-8
- Graham et al. 2021, Nature — GLGC lipids (GCST90239658). https://doi.org/10.1038/s41586-021-04064-3
- Karczewski et al. 2024, medRxiv — Pan-UKBB. https://doi.org/10.1101/2024.03.13.24303864
- Mountjoy et al. 2021, Nat Genet — Open Targets Genetics / L2G. https://doi.org/10.1038/s41588-021-00945-5
- Buniello et al. 2025 (NAR database issue; Crossref issued 2024) — Open Targets Platform. https://doi.org/10.1093/nar/gkae1128
- Byrska-Bishop et al. 2022, Cell — 1000G 30x. https://doi.org/10.1016/j.cell.2022.08.004
- Weissbrod et al. 2020, Nat Genet — PolyFun (UKB LD). https://doi.org/10.1038/s41588-020-00735-5
- Berisa & Pickrell 2016 (Crossref issued 2015), Bioinformatics — LD blocks hg19. https://doi.org/10.1093/bioinformatics/btv546
- MacDonald et al. 2022, bioRxiv — GRCh38 LD blocks. https://doi.org/10.1101/2022.03.04.483057
- Murphy et al. 2021, Bioinformatics — MungeSumstats. https://doi.org/10.1093/bioinformatics/btab665
- Genovese et al. 2024, Bioinformatics — BCFtools/liftover. https://doi.org/10.1093/bioinformatics/btae038
- Hayhurst et al. 2022, bioRxiv — GWAS-SSF standard. https://doi.org/10.1101/2022.07.15.500230
- Abell et al. 2022, Science — MPRA of eQTL variants. https://doi.org/10.1126/science.abj5117
- Siraj et al. 2024, bioRxiv — MPRA of fine-mapped variants. https://doi.org/10.1101/2024.05.05.592437
- Morris et al. 2023, Science — STING-seq. https://doi.org/10.1126/science.adh7699
- Gschwind et al. 2023, bioRxiv — ENCODE-rE2G / CRISPRi benchmark. https://doi.org/10.1101/2023.11.09.563812
- Wharrie et al. 2023, Bioinformatics — HAPNEST. https://doi.org/10.1093/bioinformatics/btad535
- Not re-verified here, cited from knowledge: Bulik-Sullivan et al. 2015 Nat Genet cross-trait LDSC (ρ = N_s·r_p/√(N₁N₂)), doi:10.1038/ng.3406; Musunuru et al. 2010 Nature (SORT1), doi:10.1038/nature09266.

## 5. Open items / unverified

- Licence text for GTEx, Pan-UKBB, Neale UKB, UKB-PPP and UKB LD not read directly.
- UKB-PPP build and current Synapse access flow not checked.
- deCODE, MetaBrain, BBJ, HAPNEST build not checked.
- INTERVAL-containing blood-cell GWAS study ID in the GWAS Catalog not identified.
- Remote tabix on eQTL Catalogue/FinnGen HTTPS was not executed (no htslib in env). Install htslib/bcftools in `env/` first.
- Whether r8 beta files change before the final r8 (Dec 2026): pin by checksum.
