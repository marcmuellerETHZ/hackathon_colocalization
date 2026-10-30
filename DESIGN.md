# DESIGN — fast, powerful colocalization robust to sample overlap and LD mismatch

Version: **v4 — FROZEN design** (after review rounds 1–3), 2026-10-26. Later changes require a decision note in RESEARCH_TREE.md and are listed in §15. Owner: PI agent. Sources: LITERATURE.md, lit/*.md, reviews/round1_*.md.
Changes vs v1 are logged in §15. Items marked (R1-x) come from round-1 reviews.

## 0. Goal, claims, minimum viable result

**Goal (fixed).**
- Benchmark existing GWAS–molecular-QTL colocalization methods.
- Build a method that is faster *and* more powerful.
- It must cope with (i) overlapping samples and (ii) mismatched LD references.
- It must (iii) distinguish *two distinct, well-powered causal variants* from *a signal too weak to call*.

**Pre-registered claims** (the evidence for each is defined in §8):

| Claim | Statement |
|---|---|
| C1 (power, S vs D) | On the held-out test split, under the realistic stress grid, the primary candidate's AUPRC for S vs D exceeds every baseline's. The paired-bootstrap 95% CI of the difference excludes 0. |
| C2 (speed) | On the defined genome-scale workloads (§7.7), whole-machine wall-clock throughput is ≥ 10× (stretch ≥ 100×) that of coloc-SuSiE on 64 cores, including any per-trait precompute. Accuracy must be non-inferior: realistic AUPRC not lower by more than 0.01. Our GPU coloc-SuSiE (G) and gpu-coloc are *reported* to separate engineering from modelling gains; they are not pass bars (R2-PI). |
| C3 (distinct vs weak) | **Thresholds.** For each method, on validation: t_S is the smallest threshold with confident-S rate on W ≤ 2.5%; then t_D is the smallest threshold with confident-D rate on W ≤ 2.5%. Calls give S precedence: S if P_S ≥ t_S; else D if P_D ≥ t_D; else insufficient. **Test.** At these thresholds, the candidate's test recall of D and of S is non-inferior to every eligible baseline (paired cluster bootstrap, one-sided, margin 0.01). It is also superior to the best eligible baseline on at least one of the two (one-sided paired bootstrap p < 0.05, Holm-adjusted over the two). **Eligible baselines:** all tier-1 methods with a defined P_D; the proportional test and the lead-SNP rule are excluded. AUROC of P_D on D ∪ W is a co-primary endpoint (§8.1). Baselines get ungated P_D (§4.3). |
| C4 (overlap) | Under estimated ρ̂, the slope of the candidate's false-S rate vs ρ has a 95% CI inside ±0.02 per 0.1ρ, on D and on W loci separately. **Tested only in pairing types where |ρ| ≥ 0.3 is reachable (P2, P3; §2.4).** For independent-likelihood baselines the slope is reported and expected to be positive. |
| C5 (benchmark contribution) | False-S/false-D rates of LD-using methods (coloc-SuSiE variants, SharePro, ColocBoost, mvSuSiE) as functions of LD-reference choice and of ρ. coloc.abf, which uses no LD, is a reference only. |

**Minimum viable result (MVR, R1-PI).** If time runs short, this alone is a complete deliverable:
1. The simulation benchmark on the test split, comparing the core baselines (§4, tier 1) with **one** primary candidate on the full realistic grid (§8.1). This includes the shifted arm built with the cheap pool-swap world (§8.5).
2. Runtimes for the same comparison.
3. Real-data arms R1, R2, R3b and R4b (R2-PI).

**Critical path:** M3 → M4 (tier-1 baselines) → M5 (kill-tests A, B, C) → M6 → M7 → M8.

**Deferred until the MVR exists:** eCAVIAR, HyPrColoc, L5 (per-SNP N), layer F, R4, R5a, R5b, R6, and the msprime LD world (the shifted arm uses the pool-swap world first).

**Out of scope:** target-gene prioritisation, more than 2 traits, trans-QTLs.

## 1. Estimand and classes

**Unit.** A *locus pair*: one window, trait 1 (GWAS-like) and trait 2 (QTL-like). Signal-pair analyses are secondary (§8.4).

**Power (oracle).**
- For causal variant j of trait t, the marginal non-centrality is μ_tj = E[z_tj | study-t genotypes, β_t]. This is the exact expected z given the realised study sample. For factor-form traits it equals (R_pop λ_t)_j; for individual-level traits it is computed from G_t. Labels therefore use the realised noncentrality, not the nominal λ (R2-stat).
- Own-effect λ_tj is the joint standardized effect, √N_t·β_std,j, with β_std,j² = q_j² (per-variant PVE in standardized-genotype units). This is the single definition used in §1 and §2.2 (R3-stat 5).
- Total PVE per trait (causal variants plus background, including LD between them) is capped at 0.5. Architecture draws that exceed it are redrawn, so residual variance stays positive.
- **Powered** means P(|z| > z_α) ≥ 0.8. This needs **both** |μ_tj| ≥ τ_t and own-effect |λ_tj| ≥ τ_t (R1-stat 19, which guards against cancellation by opposite-sign LD partners).
  - Thresholds depend on the **trait type**, not the trait index (R3-PI 1). GWAS-like traits: α = 5e-8, τ = 6.29. Molecular traits (eQTL, pQTL): α = 1e-5, τ = 5.26.
  - In P1 and P2, trait 1 is GWAS-like and trait 2 molecular. In P3, eQTL × eQTL has both molecular; lab × lab has both GWAS-like.
- **Weak** means power < 0.2: |μ| < 4.61 for GWAS-like traits and < 3.58 for molecular traits.
- **Grey** otherwise.
- These thresholds were checked by three reviewers and are frozen.

**Identifiability index (R1-stat 10).** For a D-type pair (c1, c2):

d = min(|λ1,c1|, |λ2,c2|) · √(1 − r²(c1, c2))

Here r² is computed in R_true (pooled population LD).

**Oracle classes: a precedence-ordered partition** implemented in one function `oracle_class(θ)`. A unit test asserts that every locus gets exactly one label.

| Order | Class | Rule |
|---|---|---|
| 1 | **S** | ≥ 1 causal variant is shared and powered in both traits. |
| 2 | **U** (unidentifiable) | Each trait has a powered variant, none shared and powered, and the class-determining pair has d < 2. |
| 3 | **D** | Each trait has ≥ 1 powered causal variant, and no shared causal variant has power ≥ 0.2 in both traits. The class-determining pair is the cross-trait pair of powered variants with maximal r²; D is stratified by r² bins and by d. |
| 4 | **W** | At least one trait has no causal variant with power ≥ 0.2. |
| 5 | **G** | Everything else: grey class-determining variants, and powered distinct variants plus a shared variant with power ≥ 0.2 in both traits that is not powered in both. A shared variant that is weak (< 0.2) in at least one trait does **not** block D (R2-PI). |

Grey status applies only to class-determining variants. Class shares of the generated mixture are reported in M3 before anything is frozen.

**Class-determining (μ1, μ2)**, used for power matching and stratification:
- for S: the shared powered variant with the largest min-power;
- for D and U: the class-determining pair;
- for W: the strongest variant per trait (0 if none).

**Observed power gate** (what a method may use on real data):
- each trait separately has min p < α;
- and ≥ 1 credible set (coverage 0.95, purity ≥ 0.5).

The confusion between oracle and observed labels is reported.

**Method outputs.** Each method outputs, through pre-registered adapters (§4.3):
- **s**: an *ungated* ranking score for S vs D (R1-stat 2);
- **P_D**: a score for D vs W;
- a three-way call. Gates apply to the call only.

The three-way call:
- S if P_S ≥ t_S;
- D if P_D ≥ t_D and the exclusion probabilities E₁, E₂ ≥ 0.95 (§5, layer D) where available;
- otherwise "insufficient".

Thresholds t_S and t_D are set on validation by the matched-W rule (§8.1). The fixed 0.8 thresholds are also reported.

## 2. Data

### 2.1 Variants, populations, LD worlds

**Variant set** (R1-data 9):
- biallelic SNVs only;
- positions with more than one record (split multiallelics) removed;
- MAF ≥ 0.01 in the target pool;
- MHC (chr6:28.5–33.4 Mb) excluded.

The pgen files from `scripts/prep_1kg_pgen.sh` keep indels and multiallelic splits. The filtering is done downstream in `src/data/variants.py` and verified by a test.

**Populations** (1000G 30x, phase-3 unrelated, GRCh38):

| Pool | Content | Role |
|---|---|---|
| NFE | CEU, GBR, TSI, IBS: 404 individuals, split by seed stratified by subpopulation | |
| Pool A | 254 NFE individuals | target population haplotypes |
| Pool B | 150 NFE individuals | external reference (n_eff measured, not assumed) |
| FIN | 99 individuals | cross-population GWAS world (L6) |
| ASW + ACB | 157 individuals | admixed component of a GTEx-like study sample (L3) |

**LD world 1 (primary): a Li–Stephens mosaic synthetic population.**
- Built from pool-A haplotypes with the Beagle GRCh38 genetic map; N_syn = 20,000.
- Parameters (switch rate via N_e, mutation/miscopy rate) are pinned in config and validated against pool A before use:
  - r² decay by distance bin (0–10, 10–50, 50–100, 100–500 kb) within 10% relative;
  - effective rank / eigen-spectrum of 1-Mb windows reported (R1-data 3).
- The mosaic's known bias (inherits pool-A sample LD; slight long-range LD loss) is documented. It is not corrected.

**LD world 2 (test-only shifted arm, §8.5): the pool-swap world.** The target population is a mosaic of pool-B haplotypes; references are pool-A individuals (n = 254) or an independent pool-B mosaic draw (n = 2,000). Used only in the shifted test arm, so that no method was developed on it. An msprime OutOfAfrica_3G09 world is optional and deferred.

### 2.2 Generator (R1-stat 7, R1-data 2, R1-alg 1)

Each study draws its own sample of rows from the synthetic population.

- **Small-N trait** (N_t ≤ 20,000, e.g. eQTL N 100–5,000): **individual-level**.
  - Take a row subsample G_t (N_t × p, standardized) and compute y = G_t β + ε.
  - z_t = G_tᵀ y / (√N_t · σ̂_y) (OLS z on standardized genotypes).
  - z_t therefore follows that study's own in-sample LD R_t, with its sampling noise.
- **Large-N trait** (GWAS, N ≥ 20k): **factor form**.

  ```
  z_t = R_pop λ_t + e_t,   e_t = G_popᵀ u / √N_syn,   u ~ N(0, I_{N_syn})
  ```

  This uses one scaling only (fixing E7). In-sample LD sampling noise at N ≥ 20k is negligible relative to reference noise and is ignored (documented).
- **Overlap.** The target is always the realized null correlation ρ = N_s·r_p/√(N1·N2).
  - Two factor-form traits: Cov(e1, e2) = ρR_pop exactly, via correlated u.
  - Small-N trait nested in a factor-form study: shared-row noise correlation chosen so the realized ρ equals the target. The per-row correlation is r_p·√(N_syn/N1) when N1 > N_syn (R2-alg, R2-data).
  - Two individual-level traits (same-donor eQTL × eQTL): shared rows with residual correlation r_p.
  - A verification test checks the realized ρ empirically for every mode.
  - The generator **asserts reachability**: the required |r_p| ≤ 0.9 (R3-stat 1).
  - Studies from different populations (L3 trait 2 vs trait 1; L6) cannot share donors. Those levels are only combined with ρ = 0 (R3-stat 2).
  - Note: for a small-N trait nested in a large study, |ρ| ≤ |r_p|·√(N2/N1). High ρ therefore needs pairing types P2/P3 (§2.4).
- **Sample sizes and effects** (R1-stat E8, R1-data 7). Per locus, by pairing type (§2.4):
  - N1 ∈ {20k, 100k, 400k};
  - N2 ∈ {100, 300, 1,000, 5,000} for P1, {20k, 50k} for P2, and N2 = N1 for P3;
  - per-variant PVE log-uniform: trait 1 in [1e-5, 2e-3] (main stratum) and [2e-3, 2e-2] in a 20% "biobank-strong" stratum, which gives λ up to about 150; trait 2 in [1e-3, 0.3];
  - λ = √(N·q²/(1−q²)); random sign.

  The R1 real-data realism check (§2.6) may adjust these ranges *before* any test-split use, with a decision note.
- **Inputs given to baselines:** beta, se, N, MAF and z, with se_j = 1/√(2N·f_j(1−f_j)) and sdY = 1 (R1-stat 7b).
  - f_j is the in-sample allele frequency.
  - The reference allele frequency comes with the reference LD.

### 2.3 Inference-time LD (mismatch levels; methods never receive R_true except at L0)

| Level | Study sample(s) | LD given to methods |
|---|---|---|
| L0 | mosaic-NFE | in-sample LD of trait 1's study, and of trait 2's study where relevant (methods taking one R get trait 1's) |
| L1 | mosaic-NFE | independent mosaic draw from pool A, n = 2,000 (finite-sample noise only) |
| L2 | mosaic-NFE | real pool-B individuals, n = 150 ("small external reference"; n_eff reported) |
| L3 | trait-2 study = 85% mosaic-NFE + 15% mosaic-ASW/ACB (GTEx-like admixed study, R1-data 1); trait 1 NFE | pool-B NFE reference |
| L4 | as L2 | L2 + 0.5% of variants with a flipped allele in one trait + 5% of variants missing from the reference (dropped) |
| L6 | trait 1 from mosaic-FIN (FinnGen-like), trait 2 from mosaic-NFE | pool-B NFE reference |
| L6c | control for L6: trait 1 from a mosaic of 99 random NFE individuals not in pool B | pool-B NFE reference |
| L5 (deferred) | per-SNP N heterogeneity via sample masks | L1 |

Further rules for these levels:
- **L6 effects** are drawn on the allelic scale and converted per population using that population's allele frequency. Variants monomorphic in FIN have no trait-1 effect (R2-data).
- **L3 admixed trait-2 study:** z is computed with the top 2 genotype PCs of the study sample as covariates, as eQTL pipelines do.
- **Reference LD** is computed from genotype dosages of the reference individuals. Methods receive n_ref = the number of reference individuals.
- **Assertions:**
  - a test asserts that no method input file at levels ≥ L1 contains R_true (R1-AFM 3);
  - in R4b, the reference excludes all GEUVADIS individuals.

### 2.4 Causal architecture and grid

**Locus mixture (the training prior).**

| Type | Share | Content |
|---|---|---|
| S | 30% | 1 shared variant, plus 1 extra trait-specific variant in 50% |
| D | 30% | 1 + 1 distinct variants, with partner chosen to fill r² bins roughly uniformly; extras in 30% |
| W | 25% | one trait with signal; the other with none (50%) or a sub-powered shared variant (50%) |
| M | 15% | 2–3 per trait, mixed |

- Causal MAF is uniform over MAF bins (0.01–0.05, 0.05–0.2, 0.2–0.5).
- An optional infinitesimal background (local h² equivalent to Σλ² = 2) is added in 20% of loci.
- The oracle class is computed after generation. Evaluation cells are filled by **rejection sampling on the oracle class** (R1-AFM 7), so each cell has its target count.

**Overlap grid by pairing type (R2 consensus).** Under the constraint |ρ| ≤ |r_p|·√(N2/N1), each pairing type gets its own reachable grid:

| Pairing type | Content | N | Overlap grid | Headline weight |
|---|---|---|---|---|
| P1: eQTL vs GWAS | small-N QTL; nested (N_s = N2) or disjoint | N2 ≤ 5k | r_p ∈ {0, 0.3, 0.6}; realized ρ ≤ 0.3, mostly ≈ 0.01–0.1 | 0.5 |
| P2: large-N molecular trait vs biobank GWAS (pQTL-in-UKB-like) | both factor form | **N1 = 100k, N2 = 50k fixed**, so ρ cells are comparable and |ρ| ≤ 0.707·r_p (R3) | ρ ∈ {0, 0.1, 0.2, 0.3} | 0.25 |
| P3: full overlap, same donors / same biobank (eQTL × eQTL across tissues, N1 = N2 ∈ {300, 1,000}, individual-level, both molecular-type; and lab trait × lab trait, N1 = N2 = 100k, factor form, both GWAS-type) | | | ρ = r_p ∈ {0, 0.3, 0.5, 0.8}. Training and validation use both signs; the test uses positive ρ | 0.25 |

C4 is evaluated within P2 ∪ P3.

**ρ̂ (R1 consensus, R2-data).** Methods never receive the true ρ in headline results.
- One estimator, `src/stats/rho.py`: a truncated-bivariate-normal MLE on LD-pruned SNPs with |z| < 2 in both traits. It returns ρ̂ and its SE.
- What it is computed on:
  - **P1, and P3 eQTL × eQTL:** the locus window's own SNPs. This mirrors cis-window-only QTL data and gives realistic noise (SE ≈ 0.06–0.1).
  - **P2 and P3 lab × lab:** a simulated genome-wide set of 20,000 independent SNPs (null plus 1% polygenic signal) from the same (N1, N2, N_s, r_p).
- Methods receive (ρ̂, SE) and may shrink ρ̂.
- A test checks unbiasedness (|bias| < 0.01 at ρ ∈ {0, 0.3, 0.5, 0.8} with genome-wide SNP counts).

**Power matching.** Within each r²/d bin, S and D are reweighted to equal joint distributions of (μ1, μ2) in 5×5 trait-specific quantile bins. Weighted S prevalence in S ∪ D is 0.5.

### 2.5 Real data (validation of realism and of conclusions; secondary to simulation)

All real data are natively hg38: eQTL Catalogue r7, FinnGen R12, Open Targets 26.09, 1000G 30x.

**Harmonisation** (R1-data 9) follows one script, `src/data/harmonise.py`:
- contig naming → `chrN`;
- SNV only;
- effect allele = alt;
- AF concordance check against pool B (flag |ΔAF| > 0.2, or > 0.1 at palindromic SNPs; drop flagged palindromes);
- lead-variant missingness logged per locus;
- FinnGen truncated LD repaired to PSD by one rule for all methods (eigenvalue floor + renormalise), with the count of repaired eigenvalues logged.

| Arm | Content |
|---|---|
| R1 | **Realism.** Distributions of lead \|z\|, number of credible sets, λ ranges and mismatch diagnostics, in simulation vs eQTL Catalogue and FinnGen. Run on train-split regions only. Adjustments go in a decision note before M5. |
| R2 | **Partial positives.** The same gene in the same tissue in two non-overlapping eQTL Catalogue cohorts, both with a significant cis signal (gene-level selection; *not* selected by SuSiE lead agreement, R1-PI 3). Used for relative recall (true sharing is high but below 1). |
| R3b | **Pseudo-negatives (locus-level D).** FinnGen trait × eQTL pairs at the same locus where each has exactly one credible set (catalogue / FinnGen in-sample SuSiE). Both are powered, the CSs are disjoint, and lead r² < 0.1 in both pool B and FinnGen LD. Guards are in lit/gap_and_evaluation.md §7.5. |
| R3 | **Signal-level negatives.** For methods taking per-signal Bayes factors only: catalogue genes with ≥ 2 CSs in cohort A, where cohort B replicates only one. |
| R4 | **Real LD mismatch.** FinnGen traits with FinnGen in-sample LD vs pool-B / 1000G EUR reference. Agreement of each method's calls under reference vs in-sample LD, with in-sample calls as pseudo-truth. |
| R4b | **Exact in-sample QTL LD.** GEUVADIS LCL eQTL (eQTL Catalogue) individuals are in 1000G, so in-sample QTL LD is exact (R1-data 10). Run in-sample vs out-of-sample. |
| R5b | **Real overlap.** FinnGen trait pairs (full overlap, ρ̂ by the same estimator). The metric is the confident-S rate on pairs where trait 2 has no signal (W-like), compared between overlapping and non-overlapping comparators (R1-data 11). |

**Pre-registered real-data success criterion** (R1-PI 3): at the per-method threshold giving a 5% confident-S rate on R3b, the candidate's S recall on R2 is ≥ that of the best tier-1 baseline. Under R4/R4b, its agreement with in-sample calls is ≥ the best baseline's. Both are reported with bootstrap CIs over genes. A failure is reported as a failure.

### 2.6 Realism check before freezing grids

M3 ends with a comparison of the simulated vs R1 real distributions. Grid changes are allowed only here, documented, and applied to all methods.

## 3. Splits (LD region; each gene in a single split) (R1 consensus)

- **Units:** MacDonald 2022 EUR pyrho LD blocks (hg38, 1,336), excluding MHC blocks. They are made to **tile** each chromosome (R2-data):
  - an inter-block gap ≥ 1 Mb becomes its own block;
  - a smaller gap is split at its midpoint between its neighbours.

  This puts 509 protein-coding genes back inside a block.
- **Assignment:** each block goes to train/validation/test (60/20/20), by seeded random assignment (seed 20261026), stratified by chromosome and balanced on protein-coding gene counts.
- **Genes:** assigned by the block containing their TSS. A test asserts that every window contains its own TSS or lead.
- **Windows** (simulated loci and real genes/loci): TSS or lead ± 500 kb, **clipped to the maximal run of consecutive same-split blocks containing the TSS or lead**. Every variant therefore belongs to exactly one split.
- **Script and files:** `src/data/make_splits.py` writes `splits/split_v1.tsv` (block → split) and `splits/genes_v1.tsv` (gene → block, split). Both are committed with sha256 in `splits/SHA256SUMS`.
- **Audit test** `tests/test_splits.py` checks:
  - no gene, block or variant appears in two splits;
  - every simulated window lies inside one split run;
  - the counts match the decision note.
- **Bootstrap unit:** the LD block, about 267 test units. CIs are paired cluster-bootstrap (BCa) over blocks; a sign test over blocks is also reported. Power-matching weights are recomputed inside each replicate.
- **Screening kill rules** use a paired locus-level bootstrap on the fixed validation screening set.

## 4. Baselines

### 4.1 Tier 1 (required for the MVR)

1. **coloc.abf** (coloc 5.2.3, CRAN): beta/varbeta/N/MAF input. Prior effect SD W tuned on validation over {0.1, 0.15, 0.2}; p12 = 5e-6 and 1e-5 reported. p12 does not change its AUPRC, so tuning targets W (R1-stat 6).
2. **coloc-SuSiE (standard):** susieR 0.14.2 `susie_rss(z, R, n, L = 10)`, without regularisation (the standard pipeline), then `coloc.bf_bf` (p12 = 5e-6). Falls back to coloc.abf when either trait has no CS.
3. **coloc-SuSiE (mismatch-aware):** susieR 0.16.x (pinned version/commit) `susie_rss(..., R_finite = n_ref, R_mismatch = "eb_mix")` with its SER fallback diagnostic, then coloc.bf_bf. If the interface differs from the docs, the wrapper documents what was used (R1-PI 4, R1-prior M3).
4. **SharePro** (official implementation, pinned commit), run in its documented LD-mismatch-robust mode at levels ≥ L1.
5. **ColocBoost** (R, pinned), pairwise mode, with its documented mismatch handling [preprint method].
6. **mvSuSiE-RSS** (mvsusieR, pinned) with the singleton + shared prior and residual correlation from ρ̂. This is the closest prior work to Family A (R1-prior M1).
7. **Proportional test:** colocPropTest v0.9.3 and/or prop.coloc v2.0.0. These are the only methods whose null is "distinct" (R1-prior M3).
8. **Lead-SNP rule:** a sanity floor.

### 4.2 Tier 2 (after the MVR)

gpu-coloc (speed reference for pairing), eCAVIAR/CLPP (capped subset), HyPrColoc (with trait correlation).

### 4.3 Adapters (pre-registered)

Baseline P_D scores are **ungated**, so D-vs-W AUROC is meaningful for every method (R2-PI). Gates apply only to three-way calls.

| Method | s (S vs D) | P_S | P_D (ungated) |
|---|---|---|---|
| coloc.abf | PP4/(PP3+PP4) | PP4 | PP3 |
| coloc-SuSiE (both) | max over CS pairs of PP4/(PP3+PP4); coloc.abf's s on fallback | max PP4 over pairs | (1 − max PP4) · max PP3 over pairs; fallback coloc.abf PP3 |
| SharePro / ColocBoost | their locus colocalization probability | the same | (1 − P_S) · P(trait-1 signal) · P(trait-2 signal), where P(trait-t signal) = the method's per-trait signal probability if reported, else 1 − ∏_j(1 − PIP_tj) over its per-trait PIPs |
| mvSuSiE-RSS | from CS-level sharing (same inclusion–exclusion formulas as Family A, §5, with lfsr < 0.05 defining "active") | the same | the same |
| Proportional test | **p_prop** (p-value of the "proportional = shared" null; a high p means consistent with sharing; R2-stat) | — | — (S vs D only) |
| Lead-SNP rule | binary | — | — |

- **Ties and undefined values:** s = 0.5. The share of ties is reported per method. AUPRC uses step-wise average precision (ties treated as one threshold).
- **Failures** (crash, timeout, non-convergence) are counted, reported per method and cell, and scored as the method's *least favourable* outcome (s = 0 for S, s = 1 for D), not 0.5. More than 1% failures in a cell blocks claims involving that method until resolved (R1-AFM 5).
- **Failure scoring (R2-AFM MC3):**
  - metrics are computed over the full expected locus list (outer join);
  - "no output" (crash, timeout, NaN) counts as a failure;
  - non-converged runs that return output are used and flagged;
  - claims must hold both with failures scored least-favourably and with failures excluded.
- **Timeouts and subsampling** are frozen per method from a pilot of 200 validation loci (R2-alg 5). Slow methods (mvSuSiE-RSS, ColocBoost) may run on a pre-registered random subsample of 400 loci per cell, with the subsample identical across methods for paired comparisons.
- **Install fallback:** if a tier-1 baseline cannot be installed or run after reasonable effort, it is reported as "not run (reason)". The comparison proceeds with the rest and no substitute is invented.
- **Wrapper validation:** each wrapper must reproduce its package's documented example output. Known-direction controls are run with every benchmark (§7).

## 5. Candidate model families (≥ 3 kept alive through screening, incl. a bold one)

**Shared notation:**
- z_t: z-scores of trait t;
- R: the inference LD;
- V = [[1, ρ̂], [ρ̂, 1]];
- stacking is trait-major, z = [z1; z2];
- noise covariance is V ⊗ R̃, and the mean map is I₂ ⊗ R̃ (R1 E1).

### Family A — Joint bivariate SuSiE-RSS with sharing states (JointSuSiE)

**Model.**
- L single effects. Each has a variant j_l and a state s_l ∈ {0 (null), 1-only, 2-only, both} (an explicit null state, R1-prior M2 / R1-stat 3).
- Effect b_l ~ N(0, U_{s_l}). U_both allows a correlated effect; the scales are estimated by empirical Bayes per locus, and the state prior π by EB, or tuned on train.
- Likelihood: z ~ N((I₂ ⊗ R̃) b, V ⊗ R̃).
- LD error enters through η estimated *jointly inside IBSS* on residual z (R1-stat 15), plus a reference-size-aware variance term (layer E) as an ablation.

**Locus outputs (R2-stat: inclusion–exclusion form).** Effects are treated as independent under the mean-field posterior. For effect l define per-effect event probabilities, computable from its state posterior and its Gaussian effect posterior:
- **a_l** = P(l is shared and powered in both traits): the S event;
- **w_l** = P(l is shared with power ≥ 0.2 in both traits): blocks D;
- **p_l^t** = P(l has a powered effect in trait t).

Then:
- P_S = 1 − Π_l (1 − a_l)
- with B = Π_l (1 − w_l) (no blocking shared effect):
  - P_D = B − Π_l (1 − w_l − p̃_l^1) − Π_l (1 − w_l − p̃_l^2) + Π_l (1 − w_l − p̃_l^{1∨2})
  - where p̃_l^t = P(l powered in trait t and not blocking), and p̃_l^{1∨2} = P(l powered in trait 1 or 2 and not blocking)
- P_W = 1 − P_S − P_D, which also absorbs G-type mass
- s = P_S/(P_S + P_D)

A unit test checks the formula: a single signal that could belong to either trait gives P_D = 0; two independent strong single-trait signals give P_D ≈ 1. Exclusion probabilities E_t are also reported (layer D). They use the 0.2-power thresholds.

**Contribution, honestly stated (R1-prior M1).** As a model, A is close to mvSuSiE-RSS with a singleton + shared prior, and to SharePro/CAFEH with a residual covariance. Its claimed contributions are:
- GPU batching (speed);
- the LD-error model inside the joint likelihood;
- coherent oracle-event posteriors {S, D, W};
- overlap via ρ̂.

**Required checks:**
- **Identity test:** with η = 0 and ρ = 0, A reproduces `mvsusie_rss` with the same prior (PIP/state agreement < 1e-2 on 50 loci).
- **Ablation ladder:** base → +ρ̂ → +η → +E-layer → +power-aware outputs. Each step is credited separately.

**Engineering (R1-alg 5, 6).**
- Group loci by p (no padding to p_max). References with n < p are represented as genotype factors (n_ref × p); larger ones as dense LD.
- η is computed from a thin SVD.
- fp32 compute and fp64 accumulation of log-BFs; no bf16/fp16 for LD; TF32 disabled and asserted.

**Kill-test A1:** on the validation screening set (L0, ρ = 0), AUPRC ≥ max(coloc-SuSiE, SharePro, mvSuSiE-RSS) − 1 paired SE, with runtime ≤ coloc-SuSiE. On failure: diagnose once; if it still fails, kill or demote A to a GPU re-implementation of the best baseline (speed-only claim).

### Family B — Exact Bayesian enumeration on GPU (ExactColoc)

**Screen (R1-ML 1).**
- Take the union of the top M/2 SNPs per trait, plus the top SNPs by each trait's z after removing its lead's LD-projected effect.
- Collapse near-duplicates into one representative only when they are unidentifiable at the observed signal strength, i.e. max|z|²·(1 − r²) < 1. Keep the multiplicity for the prior (R2-alg 2).
- R̃ = (1 − η)R + ηI throughout, with η ≥ 1e-3.
- Cap the LD-proxy expansion.
- **Pre-registered recall test:** ≥ 0.99 of causal variants (or their r² > 0.95 proxies) are retained, stratified by λ1/λ2.

**Enumeration.** Configurations with ≤ 2 causal variants per trait, M ≤ 100 (R1-alg 3). An extended ≤ 3 mode is allowed only for M ≤ 30 or with pruned search.

**Exact Bayes factor in support space** (R1-alg 2). For union set u, with mean map I₂ ⊗ R̃_uu and noise V ⊗ R̃_uu, the BF is evaluated in the ≤ 4-dimensional effect space:

BF = |I + DΦ|^{-1/2} · exp(½ · bᵀ (I + DΦ)^{-1} D b)

where:
- Φ = V^{-1} ⊗ R̃_uu;
- b = (V^{-1} ⊗ I) z_u;
- D is the prior covariance of the effects of the active (trait, variant) pairs. The rank-1 shared prior is allowed.

Implementation conditions (R2-stat):
- b and Φ are built over the union u of both traits' causal SNPs (both trait rows), and then restricted to the active (trait, variant) pairs. Using only the active z's is wrong (logBF 16.5 vs 32.9 in a check).
- Verified to 2e-13 against the full form for ρ ∈ {0, 0.4, −0.7}.
- The test reference for near-singular R̃ uses extended precision (mpmath), because the fp64 full form fails there.
- Configuration sums are done in fp64.
- **coloc mode** uses prior variance W²/SE_j² on the z scale.

**Exactness scope (R1-stat E2):**
- exact under the RSS model with the same R̃ in mean and covariance, within the screen and ≤ 2 per trait;
- for variance-inflated mismatch models, B evaluates the full-z_M likelihood (rank-|u| updates);
- B is misspecified on loci with 3 causal variants per trait, and this is reported.

**Priors:**
- *coloc mode:* single-SNP, ρ = 0, coloc's W/SE_j² priors and p1/p2/p12 at full p.
- *Gaussian mode:* EB.
- *Simulator-prior mode* (R1-stat 4, R2-stat 4, R2-alg 3). |λ| is drawn from an 8-component scale mixture of Gaussians, and that mixture is itself used as the generator's prior. The configuration prior is §2.4 restricted to ≤ 2 per trait. The mode outputs posteriors of the *oracle classes*, by Monte Carlo over each configuration's Gaussian effect posterior against the power thresholds.
  - It runs in a dedicated world **L0\***: short windows (p ≤ 30); the same LD for both traits; data generated *with R̃ itself*, the matrix used for inference, so B's model is exactly correct; no background and no M loci; both methods given the true ρ; class-prior correction for any rejection sampling. The Monte Carlo error of the class posteriors is reported (R3-stat 5).
  - In L0\* it gives the **Bayes-optimal ceiling** reported in M4/M5. It is not claimed to be a ceiling in other worlds.

**Kill-test B1:**
- *Correctness:* coloc mode with full-p single-SNP enumeration reproduces coloc.abf to < 1e-3 on 200 loci.
- *Usefulness:* on the D r² 0.3–0.8 and M strata (L0, L2), the false-S rate is below coloc-SuSiE's by more than 1 paired SE.
- Otherwise B is kept as gold standard and ceiling only.

### Family C (bold) — Amortised neural posterior estimation (NeuralColoc)

**Input.**
- The same per-trait-union screen as B, M = 128–256.
- Per-SNP features: z1, z2, MAF, per-SNP N, and log n_ref.
- Global features: ρ̂, N1, N2, the LD eigen-spectrum and the screen's top-z statistics.

**Architecture.** A transformer over SNPs with an LD-derived attention bias (learned function of r_ij and r_ij²), a sign-equivariant treatment of z (allele-flip augmentation), and a permutation-invariant pooled readout.

**Outputs:** 5-class probabilities {S, D, W, G, U} with hard labels (R1-ML 2), giving P_S, P_D, P_W and s = P_S/(P_S + P_D). There is also a per-SNP shared-PIP head.

**Training.**
- On-the-fly GPU simulation from **train-split windows only**, over all mismatch levels and ρ, with ρ̂ (not ρ) as input.
- The training prior is made explicit and separate from evaluation strata (R1-ML 7).
- One mismatch mechanism (L6) is held out of training, to test generalisation to an unseen mechanism (R1-ML 8).
- Outside its training input range (e.g. lead |z| beyond the max trained value), C falls back to A or B, flagged.

**Kill-tests, cheapest first (R1-ML 3):**
- **C0:** in coloc.abf's own world (single variant, its priors), C reproduces coloc.abf's PPs. Expected KL < 0.01 and 99th-percentile |ΔPP4| < 0.05 on validation loci. No B needed.
- **C1:** in world L0\* (B's simulator-prior world), C matches B's oracle-class posteriors (expected KL < 0.02). It is run as a learning curve over training-set size.
- **Floor:** C must beat a gradient-boosted classifier on hand-built features (lead z's, lead r², conditional z's, coloc.abf and SuSiE summaries) and a no-LD ablation of itself.
- **C2:** on the realistic validation grid, AUPRC ≥ best of A/B at equal or better *recalibrated* ECE. The train-region vs validation-region gap must be < 0.02 AUPRC.
- Failing C0 or C1 → kill. Failing C2 → demote C to a fast pre-filter.

**Learned components inside A/B (R1-ML):** an amortised LD-mismatch correction is the best candidate. Its cheapest kill-test is to plug the *true* mismatch into B: if oracle-corrected B is not better, a learned correction cannot help.

### Cross-cutting layers

- **D — power-aware decision (R1-stat 14).** Exclusion probability E₂ = P(|λ_{2,j}| < w₂ for all j ∈ CS₁ | z) under the fitted model, and symmetrically E₁. Here w_t is the weak (0.2-power) threshold of trait t's type: 4.61 for GWAS-like, 3.58 for molecular (R3-stat 4). D is called only if both observed gates pass and E₁, E₂ ≥ 0.95; otherwise "insufficient". Available for A and B. For baselines a plug-in version is applied as an optional post-hoc layer, reported separately.
- **E — LD-mismatch layer.**
  - Joint η.
  - An explicit reference-size variance, Var((R̂b)_k) ≈ Σ_j b_j²(1 − r_jk²)²/n_ref, plus cross terms (R1-stat 15).
  - A DENTIST-S outlier mask.
  - Screened as ablations.
- **F — per-SNP N:** deferred.

### Engineering baseline G — GPU coloc-SuSiE

A batched SuSiE-RSS on GPU plus bf_bf pairing. The identity test (|ΔPIP| < 1e-3 vs susieR 0.14.2 on 100 loci) is the *first* engineering task (R1-alg). It separates engineering speed from modelling gains in C2.

## 6. Screening protocol and decision rules

### 6.1 Screening set

- From the **validation** split only.
- 500 loci per cell: {S, D r² < 0.5, D r² 0.5–0.8, W} × {L0, L2, L4} × {P1 with r_p = 0, P3 with ρ = 0.5}.
- Generated once with seed 20261027, with sha256.
- **NPE tuning loci** are a separate validation draw (seed 20261029), disjoint from the screening set (R1-ML 11).

### 6.2 Kill rules

A family is killed after a good-faith implementation that passes its correctness tests, if any of:
- (a) AUPRC(S vs D) at L0, ρ = 0 is below the best tier-1 baseline by more than 1 paired SE;
- (b) confident calls on W exceed coloc.abf's (p12 = 5e-6) by > 2× **and** by > 2 percentage points, with one-sided bootstrap p < 0.05 (R1-stat 17);
- (c) false-S at L4 or ρ = 0.5 is more than 2× (and +2 pp) that of the best baseline;
- (d) runtime is more than 10× coloc-SuSiE without an accuracy gain.

Kill-test outcomes are recorded in the research tree, and never re-run with altered metrics. A, B and C are all screened before any is promoted.

### 6.3 Primary candidate

Before the test split is touched, the PI names exactly **one** primary candidate (plus at most two secondary candidates) in a decision note. All candidates and baselines are evaluated in one test run (R1-PI 5, R1-AFM 2).

## 7. Verification plan and process guards

### 7.1 Tests

Tests live in `tests/` and run via `scripts/run_tests.sh`. A result is used only if tests pass at its commit.

1. **Generator:**
   - mean/covariance vs individual-level;
   - empirical cross-trait null correlation equals ρ (both modes);
   - per-trait in-sample LD noise has the predicted magnitude;
   - oracle power equals the empirical detection rate (1,000 replicates per μ, ± 2 SE).
2. **oracle_class:** a partition (exactly one label per locus, 10⁵ random draws). Class shares are reported.
3. **ρ̂ estimator:** unbiased within 0.01 at ρ ∈ {0, 0.3, 0.5, 0.8}.
4. **B:**
   - support-space BF equals the full-form BF (< 1e-8 relative);
   - coloc.abf identity;
   - a hand-computed 2-SNP toy example;
   - screen recall ≥ 0.99.
5. **A:**
   - mvsusie_rss identity;
   - reduces to SuSiE-RSS when one trait is absent;
   - toy known answers: identical z → high P_S; two r² = 0 signals → high P_D; noise → P_W ≈ 1.
6. **G:** susieR identity.
7. **Splits:** the audit (§3).
8. **Truth isolation:** method inputs at ≥ L1 never contain R_true or the true ρ.
9. **Controls in every benchmark run:**
   - λ = 30 shared at r² = 1 must give s > 0.9 for every LD-using method, except the proportional test. Its p_prop is uniform under sharing by construction, so its control is: two strong distinct signals at r² < 0.1 give p_prop < 0.05 in ≥ 80% of replicates (R3-stat 3);
   - null loci must give high P_W;
   - shuffled-label AUPRC equals the weighted prevalence.
10. **Precision:** TF32 off is asserted; fp32 vs fp64 PIPs agree within 1e-4.

### 7.2 Run records (R1-AFM 4)

- Every run writes `results/<run_id>/manifest.json`: git hash, clean-tree flag, config, seeds, env lock hash, start/end time, counts completed/failed. It also writes per-locus predictions in one parquet file per run (no per-locus files: file-quota guard).
- An append-only `results/LEDGER.tsv` lists every run.
- `src/eval/metrics.py` is the **single** metrics implementation.
- `scripts/verify_run.py` recomputes the reported metrics from the predictions and the manifest.

### 7.3 Test lock (R1-AFM 2, R2-AFM MC4)

- The test split's simulated loci and real-data test genes are produced only by `scripts/eval_test.py`. That script refuses to run unless:
  - HEAD carries a PI-created git tag `frozen-*`;
  - the **committed** `results/test_access.log` has no previous completed entry.
- It appends to the log and the PI commits it. **One test run only.**
- The test set is fully defined *now* in `configs/testset_v1.yaml`: cells, counts per class, pooling weights and bootstrap unit (§8.6).
- Other loaders refuse test-split blocks (asserted in tests).

### 7.4 Hash-protected definitions

`configs/FROZEN_SHA256` lists the sha256 of `src/eval/metrics.py`, `src/sim/oracle.py`, the adapters, the split files, `configs/testset_v1.yaml` and this DESIGN.md (whole file, v4). Any change needs a decision note and a re-run of all affected methods.

### 7.5 Environment

- `env/activate.sh` keeps all caches and config in the project and sets `HOME` to `env/home` (R2-AFM MC6). Nothing is written to the user's home. Python entry points assert the environment is active.
- `env/requirements.lock` (uv pip freeze) and `env/renv.lock` (micromamba env export) are committed.
- Development agents own disjoint directories (`src/<family>/`, `tests/<family>/`). Shared modules (`src/sim`, `src/eval`) are changed only by their owner. Code is imported from the repo path, not via an editable install. Tests assert the import path and the R package versions (two susieR libraries coexist).
- **Machine lock:** heavy jobs take a shared lock (`flock -s results/.machine.lock`). Timing runs take it exclusively, record machine idle state, and repeat 3× (R2-AFM MC5).

### 7.6 Independent review

Every key result is re-checked by an agent that did not produce it. The reviewer reads the code, reruns a subset with `verify_run.py`, and logs a verdict in reviews/. The PI spot-checks reported numbers against result files with a script, never against agent prose.

### 7.7 Timing protocol (R1-alg 4)

- Timed runs hold an exclusive machine lock (`flock results/.timing.lock`) and run on an otherwise idle machine.
- CPU baselines: single-threaded BLAS per worker, worker count limited by memory (≤ 2 GB/core; susieR peaks reach about 3 GB at p ≈ 6k, so those use fewer workers). Rfast is installed for coloc-SuSiE.
- GPU timing uses `torch.cuda.synchronize`.
- Each locus loads its own LD from disk (LD I/O included).
- Reported:
  - per-locus median and p95 per mismatch level;
  - whole-machine throughput;
  - scaling in p ∈ {500, 2k, 5k, 10k}.
- **Genome-scale workload definition:**
  - (i) 1 GWAS × 20,000 genes × 1 tissue;
  - (ii) 1 GWAS × 20,000 genes × 50 tissues;
  - in both, windows of ±500 kb, with amortised per-trait fine-mapping allowed (precompute once, pair many times, as in gpu-coloc).

## 8. Metrics (frozen before test use)

### 8.1 Co-primary endpoints

1. **AUPRC for S vs D** (both powered, U and G excluded, power-matched, weighted prevalence 0.5), reported with AUROC.
2. **AUROC of P_D on D ∪ W** (distinct vs too weak).

Both endpoints are reported in two settings:
- **realistic:** pooled over L1–L6 and the pairing-type × overlap cells, with the weights in §8.6. This is the headline.
- **ideal:** L0, ρ = 0.

Macro-averaged per-cell values are also reported.

**Runtime** is always reported alongside (§7.7).

### 8.2 Secondary

- **Matched-W operating point (C3):** recall of S and of D at the C3 thresholds (confident-S and confident-D rates on W each ≤ 2.5%, set on validation; S precedence).
- Three-way confusion matrix.
- **Calibration:** ECE and Brier score on the *unweighted* simulation distribution, both raw and after a common one-parameter logit-intercept recalibration fitted on validation for every method (R1-stat 6, R1-ML 4); also per mismatch level × ρ.
- Realized FDR vs nominal.
- False-S rate by r² bin, by d, and vs ρ (slope with CI, for C4).
- ΔAUPRC by mismatch level.
- Shared-credible-set coverage and size.

### 8.3 Uncertainty

Paired cluster bootstrap over test LD blocks (BCa, 2,000 replicates), plus a sign test.

### 8.4 Signal-level

Signal-pair AUPRC on M loci for methods with signal output.

### 8.5 Shifted test arm (R1-PI 2, R2-PI 3)

A pre-registered, test-only arm. It differs from training in five ways:
- a class mix of S 15%, D 15%, W 50%, M 20%;
- effects up to the biobank-strong stratum in 50% of loci;
- up to 4 causal variants per trait;
- polygenic background in 50% of loci;
- a **pool-swap LD world**: the target population is a mosaic of pool-B haplotypes and the reference is pool-A individuals. msprime is added only if time permits.

C1 and C3 must hold *in direction* there. The magnitude is reported, not claimed.

### 8.6 Test-set definition (frozen in configs/testset_v1.yaml)

**Main grid:**

| Factor | Levels |
|---|---|
| Mismatch | {L0, L1, L2, L3, L4, L6} |
| Pairing type × overlap | P1: r_p ∈ {0, 0.6}, nested; P2: ρ ∈ {0, 0.3}; P3: ρ ∈ {0, 0.5, 0.8} |
| Cells | 6 × 7 = 42 |

**Per cell:**
- 400 S, 400 D (stratified uniformly over the d/r² bins), 400 W;
- G and U kept as they arise, up to 200;
- filled by rejection sampling with seed 20261028.

**Cell exclusions:** L3 and L6 are combined only with ρ = 0 cells (P1 r_p = 0, P2 ρ = 0, P3 ρ = 0), so the main grid has 4 × 7 + 2 × 3 = **34 cells**. P2 cells use N1 = 100k, N2 = 50k.

**Pooling for the headline:**
- equal weight over L1–L6 (L0 is "ideal" only);
- pairing types weighted P1 0.5, P2 0.25, P3 0.25, with equal weights within type, over the cells that exist.

**Power-matching bins (R3-PI 6):**
- S and D are matched on the class-determining (μ1, μ2) in 5 × 5 bins, using trait-specific quantiles computed on the pooled S ∪ D loci of the cell;
- r² bin edges are [0, 0.1, 0.3, 0.5, 0.8, 0.95);
- d bin edges are [2, 4, 8, 16, ∞);
- r²/d strata are used for stratified reporting, not for matching.

**Shifted arm (R3-PI 5):**
- 6,000 loci in the pool-swap world, with seed derive(20261028, "shifted");
- mismatch analogues: L1-swap (pool-B mosaic, n = 2,000) and L2-swap (pool-A individuals, n = 254), half each;
- pairing P1 with r_p ∈ {0, 0.6};
- class mix and architecture as in §8.5;
- power matching as in the main grid.

**Bootstrap unit:** the test LD block.

## 9. Reproducibility

- **SEEDS = [20261026, 20261027, 20261028, 20261029, 20261030]:**

  | Seed | Use |
  |---|---|
  | 20261026 | splits and pools |
  | 20261027 | screening set |
  | 20261028 | test set |
  | 20261029 | NPE tuning / validation draws |
  | 20261030 | replicate training run |

  On-the-fly NPE training draws use a seed derived as hash(20261030, step).
- **Configs:** YAML in `configs/`; results in `results/<run_id>/`; summary tables committed.
- **DATA.md** gives source, version, sha256 and licence for every file. Download scripts are in `scripts/`. Restricted or large data are never committed.

## 10. Compute plan (adapted from R1-alg)

| Milestone | CPU | GPU |
|---|---|---|
| M3 (mosaic for 1,336 blocks, LD, splits, tests) | ~4–6 h wall | ~1 h |
| M4 (tier-1 baselines on screening set ~12k loci plus validation grid ~50k loci) | coloc-SuSiE 1–5 s/locus (up to ~125 s under mismatch) with ~40 workers; mvSuSiE and ColocBoost on 400 loci per cell; budget 36–72 h wall (R2-alg 5: v2 budget was 3–6× too low); timeouts frozen from a 200-locus pilot | — |
| M5 (kill-tests) | — | A and B: minutes; C0/C1: ≤ 6 h |
| M6 (development) | — | NPE ≤ 24 h, 2 seeds |
| M7 (test run, one pass, all methods; resumable per method × cell, and a resumed run counts as the same single run) | 48–96 h (same scale as M4) | ~12 h |

## 11. Milestones and stop rules

| Milestone | Content | Gate |
|---|---|---|
| M3 | data, pools, mosaic, generator, oracle, ρ̂, splits, harmonisation, tests | all tests pass; independent review; R1 realism decision note |
| M4 | tier-1 baselines on the validation grid; Bayes-optimal ceiling (B, simulator-prior mode); C5 benchmark | |
| M5 | kill-tests A1, B1, C0/C1 (+ floor), layers D and E | |
| M6 | development of survivors on validation; primary candidate named | |
| M7 | tagged freeze, then one test run (simulation, plus the shifted arm and the real-data arms) | |
| M8 | REPORT.md | |

**Stop rule:** stop development when two consecutive iterations improve the validation realistic AUPRC by < 0.005, and D-vs-W AUROC by < 0.005, and runtime by < 20%. Also stop near the time limit.

## 12. Risks and guards (summary; details in §7)

| Risk | Guard |
|---|---|
| Metric or label drift | hash-protected definitions |
| Tuning on test | test lock |
| Optimistic simulation | headline uses mismatch levels only; per-trait in-sample LD; ρ̂ not ρ; a shifted test arm with a second LD world |
| Unfair baselines | mismatch-aware versions; tuned on validation; wrapper example checks; independent wrapper review; failure scoring |
| Bugs that make results look good | improvements > 0.05 AUPRC trigger an independent re-implementation check |
| Fabrication or overstatement | manifests, ledger, verify_run, PI spot-checks |
| Context or compute runaway | narrow agent scopes, ≤ 400-word summaries, result files |
| Writing outside the project | never write to $HOME; all caches go to env/cache |

## 13. Division of labour (plan)

| Role | Model | Task |
|---|---|---|
| Data/simulation engineer | Opus | M3 core: pools, mosaic, generator, oracle, ρ̂, splits, tests |
| Real-data engineer | Sonnet for downloads/harmonisation, Opus for the arm definitions | |
| Baseline engineer | Opus | wrappers, adapters, runtime harness |
| Family A developer | Opus | GPU JointSuSiE and G |
| Family B developer | Opus | |
| Family C developer | Opus | |
| Independent reviewers | Opus, fresh agents | each key result |

## 14. Open questions

None blocking. Baseline installability is being checked (env/BASELINES_INSTALL.md); see the §4.3 install-fallback rule.

## 15. Revision log

- **v1 (2026-10-26):** initial draft.
- **v2 (2026-10-26), after round 1** (7 reviewers; reviews/round1_*.md). Main changes:
  - partitioned oracle classes with U (identifiability index d) and narrowed grey;
  - ungated scores;
  - D-vs-W co-primary; C3 restated at a matched W rate; C2 against G and gpu-coloc on defined workloads; C4 with an equivalence margin;
  - block-level splits (about 267 test units) with window clipping; BCa paired bootstrap;
  - per-trait in-sample LD generator (individual-level for small N), realistic N grids, beta/SE/N/MAF inputs to baselines;
  - L3 direction fixed (admixed study, EUR reference); L6 cross-population added; SNV-only QC; MHC excluded;
  - ρ̂ via truncated MLE, used for all methods;
  - Family A: null state, coherent P_S/P_D, novelty restated, mvSuSiE identity test, ablation ladder;
  - Family B: Kronecker error fixed, support-space BF, exactness scope, ≤ 2 per trait, simulator-prior mode as Bayes ceiling;
  - Family C: per-trait screen, 5-class labels, C0/C1/floor/C2 kill ladder, ρ̂ input, held-out mechanism;
  - baselines: + mismatch-aware coloc-SuSiE (susieR 0.16.x), mvSuSiE-RSS, proportional test, SharePro/ColocBoost mismatch modes; failure scoring;
  - process: test lock via tag, manifests/ledger/verify_run, hash-protected definitions, timing lock, env locks, no writes to $HOME;
  - real data: R2 by gene identity, R3b locus-level pseudo-negatives, R4b GEUVADIS exact in-sample LD, R5b metric fixed, one pre-registered real-data criterion;
  - MVR, deferrals, shifted test arm.
  - LITERATURE.md corrections: "susieR 2.x" → susieR 0.16.x; the 0.26-FDR figure is from a preprint; JLIM's r² 0.8 is a definitional threshold, not an empirical breakdown point.
- **v3 (2026-10-26), after round 2** (5 reviewers; reviews/round2_*.md; all verdicts were "minor revision"). Changes:
  - overlap grid redefined by pairing type (P1/P2/P3), because |ρ| ≤ |r_p|·√(N2/N1) made the v2 grid unreachable; realized-ρ construction specified; C4 restricted to P2 ∪ P3;
  - ρ̂ computed per window for cis-only QTL pairs (realistic SE), returned with its SE;
  - C2 bar is now coloc-SuSiE on 64 cores, with a non-inferiority margin, precompute included, and G/gpu-coloc reported only;
  - C3 uses separate 2.5% W limits for S and D, non-inferiority plus superiority on one, and ungated baseline P_D;
  - G-row wording fixed; class-determining (μ1, μ2) defined; oracle μ = exact expected z given the realized sample;
  - Family A P_D by inclusion–exclusion (the v2 product formula was wrong);
  - proportional-test score is p_prop (the v2 score was inverted);
  - B: BF written as (I+DΦ)⁻¹D, built over the union of both traits' SNPs; mpmath reference test; W²/SE² in coloc mode; strength-aware duplicate collapse; R̃ defined;
  - simulator-prior mode and Bayes ceiling in a dedicated L0\* world (p ≤ 30, MC);
  - LD blocks tiled (gaps become blocks or are split), and each window must contain its TSS/lead;
  - L6 effects on the allelic scale per population; L6c control; L3 with PC covariates; reference from dosages, n_ref = individuals; R4b reference excludes GEUVADIS;
  - MVR includes the shifted arm via the pool-swap world, and real arms R1, R2, R3b, R4b; R4 and R5b deferred;
  - test set fully defined (§8.6); one test run enforced by a committed access log;
  - failure scoring by outer join over expected loci; timeouts from a pilot; install-fallback rule;
  - HOME set inside the project; machine locks; disjoint agent directories; import-path and R-version assertions;
  - compute budget for baselines raised;
  - chr7 VCF found corrupt by a reviewer, re-downloaded and verified against upstream md5 (scripts/verify_1kg_md5.sh).
- **v4 (2026-10-26) — FROZEN, after round 3** (2 reviewers; reviews/round3_*.md). Changes:
  - power thresholds by trait type (P3 eQTL×eQTL both molecular);
  - P2 fixed at N1 = 100k, N2 = 50k, with a reachability assertion (r_p ≤ 0.9);
  - L3 and L6 only with ρ = 0, giving a 34-cell main grid;
  - C3 thresholds, S precedence, superiority test and eligible baselines defined;
  - §8.2 changed from 5% to 2.5%; §2.1 shifted arm made consistent (pool-swap); test ρ positive only;
  - shifted arm fully specified; power-matching bins and r²/d edges defined;
  - layer D uses weak thresholds;
  - proportional-test control fixed;
  - L0\* fully specified;
  - λ defined once, with a total-PVE cap of 0.5;
  - M7 budget 48–96 h and a resumable single run;
  - the whole DESIGN.md is hash-protected.
