# GWAS–molecular-QTL colocalization: methods literature review

*Status: current to 2026-10-26. Compiled for the hackathon_eqtl_gwas project (see `PROMPT.md`).*

**How this was checked.**
- Every reference in §7 was checked against PubMed (NCBI E-utilities), the bioRxiv/medRxiv API, the publisher page or the software repository. I confirmed first author, year, venue and DOI for each.
- Quantitative claims marked **[FT]** come from the full text, read through Europe PMC / NCBI PMC XML.
- Claims marked **[Abs]** come from the abstract only.
- Claims marked **[Doc]** come from software documentation or source code.
- Claims marked **[mem]** are standard knowledge I could not re-check in the full text in this session. Treat them as needing a check before they go into a paper.
- I could not retrieve the full texts of CARMA, eCAVIAR, Hukku 2021 or the ColocBoost preprint. Statements about them rely on abstracts, docs or memory and are marked as such.

---

## 0. TL;DR (what matters for our design)

1. **No published colocalization method handles all three of our requirements together:** sample overlap, LD-reference mismatch, and an explicit "distinct-and-powered vs underpowered" call. Coverage of each requirement is patchy:
   - **Sample overlap is modelled in only a few methods.**
     - gwas-pw: z-correlation estimated from summary statistics.
     - HyPrColoc: an optional `trait.cor` / `sample.overlap` matrix, documented as giving "reasonable results" even when ignored.
     - SS2 / LocusFocus: explicit covariance n0·ρ/√(n1·n2).
     - mvSuSiE: a residual covariance V estimated from null z-scores. It is a fine-mapper, not a coloc test.
     - ColocBoost: overlap is included in its simulations, per the authors.
     - Neither coloc (ABF or SuSiE) nor SharePro, CAFEH, eCAVIAR, ENLOC or JLIM models it.
   - **LD mismatch is handled almost entirely in the fine-mapping layer**, not in the coloc test. The tools are SuSiE-RSS λ-regularisation plus kriging, DENTIST, SLALOM, CARMA, RSparsePro, and the new susieR 2.x `R_finite` / `R_mismatch="eb_mix"` with an SER fallback.
     - coloc-ABF needs no LD, which is why it is still used at scale. The price is the single-causal-variant assumption.
     - Zou et al. 2022 [FT]: "performance depended more on choice of LD matrix than on choice of method". With out-of-sample panels of 500–1,000 people, 95% credible sets no longer reached their nominal coverage.
   - **Separating H3 from low power.**
     - Only JLIM (permutation test with the distinct-variant null as an upper bound), the proportional tests (Wallace's colocPropTest; Patel & Burgess's prop-coloc, Hum Hered 2026) and SS2 (composite null that includes "distinct") treat "distinct causal variants" as a hypothesis to be tested.
     - coloc reports PP.H3 separately from PP.H0–H2. However, coloc-SuSiE simply does not test when SuSiE returns no credible set, and PP.H3 is prior- and region-size-dependent (prior odds scale as n(n−1)p1p2).
     - Rosen et al. (AJHG 2026) estimate that an eQTL study of N=500 detects <0.1–60% of eQTL depending on effect size, and that N=2,000 would detect 36.8% of all eQTL. "Absence of colocalization" is therefore mostly a power statement.
2. **Speed: state of the art.**
   - Precompute per-signal SuSiE log-Bayes factors once (eQTL Catalogue does this), then do pairwise coloc as dense tensor algebra.
   - gpu-coloc (Jesse et al., PLoS Genet 2026, Torch) is ~800× (CUDA) / ~190× (CPU) faster than R `coloc.bf_bf` for the coloc step. It ran 15.65M signal pairs in 1 h 30 min on a laptop. Results match R to within a maximum |ΔPP.H4| of 0.0087.
   - CLPP (Σ PIP1·PIP2) is another ~200× faster still. It recovers ~95% of high-confidence gpu-coloc calls.
   - The fine-mapping step (SuSiE-RSS: mean 1.3 s per locus without refinement, 5.5 s with refinement) is now the bottleneck. **I found no published GPU/batched SuSiE-RSS.** tensorQTL ships a Torch SuSiE, but only for individual-level data.
3. **Accuracy leaders in recent independent or semi-independent benchmarks.**
   - SharePro: 79.6% of MR-significant protein–trait pairs supported, versus 40–47% for coloc, coloc-SuSiE and PWCoCo (Zhang et al. GPB 2026). This benchmark comes from SharePro's own group.
   - ColocBoost reports greater sensitivity for weak, gene-distal colocalizations (preprint, v3 Sept 2026, not peer reviewed).
   - coloc-SuSiE had the highest recall (46.3%) but the lowest precision (45.1%) for target-gene prioritisation (Tambets 2024).
   - Positive-control benchmark (Hwang et al. 2025 preprint): the same protein measured on different platforms or in different populations is often *not* called colocalized. In the worst case, coloc, coloc-SuSiE, prop-coloc and colocPropTest all agreed for only 20% of proteins. Existing methods are therefore fragile exactly where LD or measurement differs between studies.
4. **New failure modes documented in 2025–26.**
   - "Variable information": per-SNP N or imputation-quality differences cause confident false H3 and false H4. Wallace, Robins & Johnson (bioRxiv 2025) report up to 100% false H3 under extreme missingness.
   - SuSiE merging of causal variants in small studies produces "colocalisation colliders": 2.5–7.3% of clusters (Jesse 2026).
   - Multi-gene colocalisation is common (~47% of OTG peaks; Reales et al. 2026) and largely reflects co-regulation.
5. **Gap we can own:** a single, GPU-batched, joint two-trait model that
   - fine-maps both traits with a shared/distinct effect structure (SharePro/CAFEH/ColocBoost-like);
   - uses a cross-trait error correlation ρ (overlap) in the likelihood;
   - has an LD-mismatch error model (finite-reference plus mismatch variance, with kriging/DENTIST-S masking);
   - outputs a three-way call per signal pair, {shared, distinct-and-both-powered, insufficient}, with calibrated probabilities.
   - Exact enumeration over ≤2–3 causal variants per trait on GPU is feasible and would serve both as a method and as a gold standard. Amortised neural posterior estimation is the bold option (§6).

---

## 1. Comparison table

Abbreviations: PP = posterior probability; ABF = Wakefield approximate Bayes factor; CS = credible set; p = number of SNPs in the region; L = number of single effects; K = number of components; T = number of traits.

| Method | Year | Model | Multi-causal | Sample-overlap handling | LD-mismatch handling | H3 vs low-power separation | Complexity / speed | Software | Key refs |
|---|---|---|---|---|---|---|---|---|---|
| **coloc (ABF)** | 2014 | Enumerates ≤1 causal per trait; ABF per SNP; PP H0–H4 with priors p1, p2, p12 | No (single causal) | Not modelled: the joint BF is the product of per-trait BFs, i.e. it assumes independent errors | Needs no LD, so immune to LD mismatch, but missed second signals give H3/H4 errors | Reports PP.H3 separately; H3 prior grows with p² | O(p) per pair; ~0.1 s per locus (SharePro paper) | R `coloc` (CRAN, v6.0.3) | Giambartolomei 2014; Wallace 2020 |
| **coloc priors / cond / mask** | 2020 | Prior elicitation (p12 = 5e-6 recommended); conditioning (COJO) or masking for multiple signals | Via conditioning (needs aligned LD) or masking (no LD) | No | Masking avoids LD; conditioning is sensitive to reference-panel LD | Masking can give false H3 when H4 is true and causal variants are in LD | Fast | `coloc.signals` (deprecated in v5) | Wallace 2020 |
| **coloc-SuSiE** | 2021 | SuSiE-RSS per trait → per-signal log-BFs → coloc on every pair of CS signals (L1×L2) | Yes (one test per CS pair) | No | Inherits SuSiE-RSS (λ, kriging); no coloc-level model; spurious mismatch signals become extra pairs | PP.H3 per pair; no test if no CS (falls back to coloc-ABF) | SuSiE ~1–5 s per locus + O(L1·L2·p); 14 s per locus in the SharePro benchmark | `coloc::coloc.susie` | Wallace 2021 |
| **Variant-specific priors** | 2025 | `prior_weights` (e.g., TSS-distance density) in coloc v6 | As coloc / coloc-SuSiE | No | No | Improved PP.H4 under H4 and lowered it under H3 in simulations; real gain ≤0.04 recall, 0.03 precision | Negligible overhead | coloc ≥6.0 | Pullin & Wallace 2025 |
| **Proportional test (colocPropTest)** | 2013 / 2024–25 | Frequentist test that β1 ∝ β2 across SNP pairs (Fieller, θ = atan η, profile likelihood), FDR across pairs | Implicitly (all shared vs not) | No (the supplement scales SEs for shared samples *within* a study only) | **Requires the same LD in both studies**, so vulnerable to mismatch | Rejection = "not all causal variants shared"; designed to catch false H3 from coloc; only valid when both traits are clearly associated | O(n²) pairs, restricted subset; runtime not stated | R `colocPropTest` v0.9.3 (2025) | Wallace 2013; Wallace, Robins & Johnson 2025 (bioRxiv) |
| **prop-coloc (cond / full)** | 2024 / 2026 | Frequentist proportionality test (null = colocalization); the -cond version conditions on selected lead variants to account for selection uncertainty | Implicit | No | Needs an LD matrix per trait | Non-rejection ≠ evidence for coloc; authors recommend use alongside coloc | Cheap | R `prop.coloc` v2.0.0 (MRC-IEU r-universe) | Patel & Burgess, Hum Hered 2026 (arXiv 2402.12171) |
| **HyPrColoc** | 2021 | Multi-trait ABF, ≤1 causal per trait; PP(full coloc) ≈ regional × alignment; branch-and-bound clustering | No; accuracy fell up to 38% with secondary signals | **Optional** `trait.cor` and `sample.overlap` matrices in a joint ABF; ignoring them "gives reasonable results" | No LD needed; assumes the same population | Separates "not all associated" (regional) from "misaligned" (alignment) | 100 traits × 1,000 SNPs <1 s; ~10⁵× faster than 45 eCAVIAR runs | `hyprcoloc` (GitHub) | Foley 2021 |
| **moloc** | 2018 | coloc extended to 3–4 traits (all sharing configurations) | No | No | No LD | Full set of configurations | Exponential in T; ~1 h for 5 traits (per HyPrColoc) | `moloc` | Giambartolomei 2018 |
| **gwas-pw** | 2016 | Four regional models (1 only, 2 only, shared, distinct) with genome-wide empirical-Bayes priors across ldetect blocks | No (single per trait per block) | **Yes:** expected z-correlation from phenotype correlation/overlap, estimated from summary statistics; conservative | Not needed (ABF-like) | Separates shared (M3) from distinct (M4), but not in strong LD | Fast, genome-wide | `gwas-pw` | Pickrell 2016 |
| **eCAVIAR (CLPP)** | 2016 | CAVIAR per trait (exhaustive ≤c causal); CLPP_j = PIP1_j·PIP2_j | Yes (small c) | No | Separate GWAS and eQTL LD files accepted (`-l` twice) [Doc]; no mismatch model | No explicit H3; CLPP is low both when signals are distinct and when underpowered | Exponential in c; 228 s per locus (SharePro benchmark) | `caviar` (eCAVIAR) | Hormozdiari 2016 |
| **CLPP from precomputed PIPs** | — | Σ_j PIP1_j·PIP2_j over SuSiE CSs | Yes | No | Inherits fine-mapping | No | ~200× faster than gpu-coloc | Open Targets gentropy, gpu-coloc | Jesse 2026 |
| **ENLOC / fastENLOC** | 2017 / 2021–22 | Hierarchical: molQTL PIPs (DAP-G) form an enrichment prior on GWAS causality; EM estimates α1; outputs SCP and RCP (regional colocalization probability) | Yes (DAP-G) | Assumes non-overlapping samples | DAP-G on summary stats with reference LD; no mismatch model | No H3; RCP is power-limited (power <40% at 20% FDR in Wen 2017) | Enrichment 4–5 min; analysis 10–20 min per dataset | `fastenloc` (GitHub xqwen) | Wen 2017; Hukku 2021, 2022 |
| **INTACT** | 2023 | TWAS BF × empirical-Bayes prior f(coloc probability) | Inherited | Two-sample design assumed | No direct LD | No | Cheap post-processing | `INTACT` (R) | Okamoto 2023 |
| **SMR / HEIDI** | 2016 | Two-step MR with the top cis-QTL as instrument; HEIDI tests heterogeneity of b_xy across SNPs in LD | HEIDI flags multiple causal variants (linkage) | Not modelled [mem] | Reference LD for HEIDI | HEIDI rejection = "distinct (linkage)"; cannot separate pleiotropy from causality | Very fast | `SMR` (C++) | Zhu 2016 |
| **JLIM** | 2017 | Joint-likelihood statistic Λ; permutation of secondary-trait phenotypes; tests the null of *distinct* causal variants (r² < θ = 0.8) | ≤1 per trait assumed; robust in simulations | Assumes non-overlapping | Reference LD for the primary trait; "mild population mismatch" tolerated | **Yes:** the distinct-variant null is explicit; estimates π0, π1, π2 | Needs individual-level data for trait 2 and 10³–10⁶ permutations | `jlim` (GitHub cotsapaslab) | Chun 2017 |
| **SS2 / LocusFocus** | 2022 | Two-stage frequentist: eQTL χ² test, then Simple Sum of weighted z² with Z ~ N(0, Σ); composite null H01–H04 | Robust to allelic heterogeneity | **Yes:** cov(Z, T) = n0·ρ/√(n_GWAS·n_eQTL), with decorrelation | Needs LD Σ (in-sample in the paper) | Composite null includes "distinct"; stage 1 handles "no eQTL" | Not stated (cheap) | LocusFocus (web), GitHub | Wang F 2022 |
| **PWCoCo** | 2020 / 2022 | Pairwise GCTA-COJO conditional analysis of each signal in each trait, then coloc on every pair | Yes (conditional) | No | COJO is sensitive to reference LD (DENTIST: false secondary signals fall from >28% to <2% after QC) | coloc PPs per pair | 38 s per locus (SharePro benchmark) | `pwcoco` (C++) | Zheng 2020; Robinson 2022 (bioRxiv) |
| **POEMColoc** | 2021 | Imputes the missing trait's z from the lead SNP: E[z_j] = z_top·r_j,top; then coloc | No (main failure mode: 9% false negatives with multiple signals) | No | Reference LD; ancestry mismatch has a modest effect | coloc PPs | Cheap | GitHub (AbbVie) | King 2021 |
| **CAFEH** | 2022 | Multi-trait SuSiE-like model: K components with a shared variant selector and per-trait activity; CAVI (CAFEH-S uses RSS + SVI) | Yes (K = 20) | No (assumes independent measurements) | One LD matrix for all traits; mismatch "could impact inference" | Per-component, per-trait activity probability (no H3) | 20 s for 2 traits, 37 s for 20 traits (5k SNPs, per mvSuSiE paper) | `cafeh` (GitHub) | Arvanitis 2022 |
| **SharePro** | 2024 | Shared sparse projection into K effect groups; trait-specific Bernoulli activity; VI on summary stats; P(both active) per group | Yes (K = 10) | No (simulations used non-overlapping samples) | Reference LD; "severe mismatch → convergence issues"; companion RSparsePro for mismatch | Group-level PP > 0.8 coloc / < 0.2 against; no power-aware H3 | 4.3 ± 1.1 s per 1-Mb locus | `SharePro_coloc` (Python) | Zhang W 2024 |
| **mvSuSiE** | 2026 | Multivariate SuSiE: Y ~ MN(XB, I, V), mixture-of-MVN effect prior, IBSS; summary-stat version mvSuSiE-RSS | Yes (L = 10) | **Residual covariance V estimated from null SNPs**, which absorbs overlap and correlated noise | Needs LD; in-sample used | Per-trait average lfsr per CS (no H3) | 41 s for 2 traits, 2 min for 20 traits (5k SNPs, n = 250k) | `mvsusieR` | Zou 2026 (Nat Genet) |
| **flashfm** | 2021 | Joint multi-trait fine-mapping (≤6 traits) with shared-variant prior; built on FINEMAP/JAM outputs | Yes | Uses trait covariance; handles missing measurements and related individuals | Reference LD | Shared vs trait-specific models | ~5 min (per mvSuSiE table) | `flashfm` (R) | Hernández 2021 |
| **ColocBoost** | 2025–26 (preprint) | Multi-task gradient boosting of coupled weak learners across traits with "LD-proximity smoothing" | Yes | Overlapping samples included in simulations (authors' claim; model details not verified) | Reported to run without LD data (authors) | Per-trait colocalization confidence sets (no explicit H3 vs power) | Scales to hundreds of traits | `colocboost` (CRAN v1.0.x) | Cao et al. medRxiv 10.1101/2025.04.17.25326042 (v3 2026-09) |
| **gpu-coloc** | 2026 | Torch re-implementation of `coloc.bf_bf` on precomputed SuSiE LBFs or ABFs; batched 100×100 | Via SuSiE signals | No | None at the coloc step | PP.H3 computed; adds a "collider" metric for threshold choice | ~795× (CUDA) / 190× (CPU) vs R; 15.65M pairs in 1 h 30 min (M2 Max) | `gpu-coloc` (GitHub mjesse-github) | Jesse 2026 |
| **Multi-ancestry coloc** | 2026 | SuSiEx or MsCAVIAR fine-mapping → coloc or eCAVIAR | Yes (SuSiEx) | No | Ancestry-specific LD; in-sample recommended | coloc PP.H3 (>0.94 for distinct) | SuSiEx 2–4 min vs MsCAVIAR hours per locus | MACA (GitHub) | Shen 2026 |
| **scJLIM** | 2025 (preprint) | JLIM extended to single-cell data: eQTL × continuous cell-state interaction, so colocalization is tested per cell without clustering | ≤1 per trait (JLIM) | Not stated | Reference LD | Distinct-variant null (JLIM) | Not stated | — | Mitchel et al. bioRxiv 10.1101/2025.10.10.681685 |
| **MystraColoc** | 2026 (preprint) | Bayesian multi-trait colocalisation over hundreds to thousands of GWAS; claims better clustering than alternatives | Not stated in the abstract | Not stated | Not stated | Clustering | Scales to 1000s of traits | Proprietary platform (Mystra) | Iotchkova & Weale bioRxiv 10.64898/2026.03.30.715409 |
| **HLA-coloc** | 2026 | SuSiE on HLA alleles → Bayesian regression on PIPs | Yes | No | Allele-level | — | — | `hlacoloc` | Butler-Laporte 2026 |
| **FUSION / TWAS** | 2016 | Predicted expression × GWAS z (weighted sum); conditional and permutation tests; coloc used as a follow-up | n/a (gene level) | Two-sample | Reference LD (1000G) | No (LD hitchhiking) | Very fast | FUSION | Gusev 2016 |
| **FOCUS** | 2019 | Fine-maps genes from TWAS z with an LD-induced correlation and a pleiotropy intercept; null model included | Multiple genes (≤5 in big regions) | No | Reference LD (1000G, n = 489) | Includes the null in credible sets | — | `focus` | Mancuso 2019 |
| **PTWAS / cTWAS / TGFM** | 2020 / 2024 / 2024 | Probabilistic TWAS; cTWAS adjusts for genetic confounders (variants and genes); TGFM fine-maps gene-tissue pairs | Yes | No | in-sample LD (TGFM) | — | — | — | Zhang Y 2020; Zhao 2024; Strober 2024 |
| **MRLocus** | 2021 | eCAVIAR-like per-cluster colocalization, then Bayesian slope over LD-independent clusters (Stan) | Yes (across clusters) | No | Reference LD | Slope interval overlapping 0 = coloc without mediation | ~75 s per locus in simulations; ~4 min per real locus | `mrlocus` | Zhu A 2021 |
| **LCV** | 2018 | Genome-wide latent causal variable model (genetic causality proportion) | n/a (genome-wide) | Robust to overlap via LDSC-like intercept [mem] | LD scores | n/a | Fast | `LCV` | O'Connor 2018 |

**Fine-mapping engines that the coloc methods sit on.** These set both the speed and the LD-mismatch behaviour.

| Engine | Model | Multi-causal | LD mismatch | Speed (reported) | Ref |
|---|---|---|---|---|---|
| SuSiE / SuSiE-RSS | Sum of L single-effect regressions, IBSS (variational) | L effects | R̂_λ = (1−λ)R̂ + λI, with λ estimated by ML under z = 0; kriging diagnostic t_j | Mean 1.33 s per locus (refined: 5.5 s) vs DAP-G 5.7 s, FINEMAP 16.1 s, CAVIAR 1,517 s | Wang 2020; Zou 2022 |
| susieR 2.x "SuSiE-rRSS" | Adds a finite-reference term (`R_finite = B`) and an EB mismatch term (`R_mismatch="eb"/"eb_mix"`); multi-panel mixtures; `SER_fallback` flag | L effects | Explicit (documentation only, no paper found) | ~5× faster for summary statistics (SuSiE 2.0 preprint) | McCreight 2025 (bioRxiv); susieR docs |
| SuSiE-ash | SuSiE plus an adaptive-shrinkage background | L + polygenic | — | FDR reduced 1.5–3× under polygenic backgrounds | McCreight 2025 |
| SuSiE-inf / FINEMAP-inf | Sparse + infinitesimal (τ²) via eigendecomposition | Yes | Needs in-sample LD | Replication failure rate 15% (SuSiE) / 12% (FINEMAP) vs ~2% expected; "substantially decreased" with -inf | Cui 2024 |
| FINEMAP-miss | FINEMAP with variant-specific missingness / sample size (meta-analysis) | Yes | Corrects *variable information*; well calibrated where standard fine-mapping fails | — | Kartau & Pirinen 2025 (Bioinformatics, btaf616) |
| FINEMAP | Shotgun stochastic search over configurations (≤K causal) | Yes | λ helpful; degrades with out-of-sample LD | Cost grows quickly with K | Benner 2016, 2017, 2025 |
| DAP-G | Deterministic approximation of posteriors | Yes | — | 5.7 s mean | Wen 2016 |
| CARMA | Flexible effect prior + annotations + Bayesian outlier detection for z/LD discrepancies | Yes | **Yes**, designed for meta-analysis mismatch [Abs] | Slower (stochastic search) [mem] | Yang 2023 |
| SparsePro / RSparsePro | Sparse projection, VI; RSparsePro is robust to LD mismatch | K ≤ 10 | RSparsePro: higher power/coverage than SuSiE under mismatch (qualitative) | "Marginal" increase with K | Zhang 2023; Zhang 2024 (bioRxiv) |
| BEATRICE | Binary-concrete posterior with a neural network as the inference machine (deep VI) | Yes | — | GPU-able | Ghosal 2024 |
| FiniMOM | Non-local prior | Yes | — | — | Karhunen 2023 |
| GWFM (SBayesRC) | Genome-wide Bayesian fine-mapping with annotations | Yes | Genome-wide LD eigen-blocks | Genome-wide | Wu Y 2026 (Nat Genet) |
| tensorQTL `susie.py` / `coloc.py` | Torch SuSiE and coloc (individual-level genotypes) | L effects | in-sample | GPU | Taylor-Weiner 2019; [Doc] |

QC layers for LD mismatch:
- **DENTIST** (Chen 2021): z_i is predicted from neighbouring SNPs, and the test is T_d = (z_i − z̃_i)² / (1 − R_it R_tt⁻¹ R_ti) ~ χ²₁. Recommended n_ref ≥ 5,000. After QC, COJO false secondary signals fell from >28% to <2%. FINEMAP's correct causal count for 2 causal variants went from 49.8% to 78.5% (in-sample: 87.2%).
- **SLALOM** (Kanai 2022): DENTIST-S against the lead variant, with P < 1e-4 and r² > 0.6. AUROC 0.74–0.80. 67% of GBMI meta-analysis loci are "suspicious". Meta-analysis FDR reached up to 37% in heterogeneous configurations.

---

## 2. Per-method notes

### 2.1 coloc family (Wallace group)
- **coloc ABF (Giambartolomei 2014).**
  - Model: Wakefield ABF per SNP, with prior SD W = 0.15 for quantitative traits (scaled by sdY) and 0.2 for case-control [Doc: coloc R source]. Priors p1 = p2 = 1e-4 and p12 = 1e-5 (default). Five hypotheses.
  - The 2014 paper already proposed conditioning on secondary signals (COJO) for multiple causal variants. It notes that the single-variant assumption makes the method consider only the strongest signal per trait [FT].
  - Strength: no LD needed, which makes it robust to LD mismatch and trivially fast.
  - Weaknesses:
    - Single causal variant per trait. When a trait has two signals, coloc "equivocates between H3 and H4" (Wallace 2021).
    - Strong prior dependence: Wallace 2020 shows that p12 = 1e-5 can be too liberal and recommends 5e-6.
    - Under H3 the prior scales as n(n−1)·p1·p2, so wider regions automatically favour H3.
    - Does not model sample overlap: the per-trait BFs are multiplied.
- **Wallace 2020 (priors, conditioning, masking).**
  - Of 25 applied 2018 papers, 22 used default priors and only 4 considered multiple causal variants.
  - Masking (setting BF = 1 for SNPs in LD with other lead SNPs) needs no LD alignment. It can give false H3 when causal variants are in LD.
  - The paper provides `coloc::sensitivity()` for prior sensitivity.
- **coloc-SuSiE (Wallace 2021).**
  - Run susie_rss on each trait, keep the per-CS log-BFs, and run coloc for every CS pair.
  - Recommendation: a hybrid of coloc-SuSiE when SuSiE finds CSs, else coloc-ABF.
  - Limitations from the paper [FT]:
    - SuSiE can merge causal variants in low-power settings.
    - No CS → no test (power floor at roughly p_min ≲ 1e-6).
    - Requires LD that matches each dataset.
- **coloc v6 / variant-specific priors (Pullin & Wallace 2025).**
  - Gains are modest: ≤0.04 recall and 0.03 precision. Priors changed significance in up to 14.1% of GWAS–eQTL colocs with some evidence.
  - PPs cluster near 0 and 1, which limits how much priors can move them.
- **Proportional colocalisation.**
  - Wallace 2013 introduced it. colocPropTest reimplements it, and a 2025 bioRxiv paper by Wallace, Robins & Johnson revives it.
  - "Variable information" across SNPs (per-SNP N, imputation r², missingness, rounding) distorts BF-based fine-mapping. At r² = 0.95 on one of two perfectly linked SNPs, that SNP's PIP has a median of 0 at N = 500k (expected 0.5).
  - Effect on coloc: up to 25% false H4 (when H3 is true) and up to 100% false H3 (when H4 is true) under heavy missingness. These are deliberately extreme simulations with 80% missingness at the top three SNPs.
  - The proportional test is recommended as a complement. It needs the same LD in both studies.
- **Positive-control benchmark (Hwang, Pullin, Wallace, Whittaker & Burgess, bioRxiv 2025-11).**
  - Compares coloc, coloc-SuSiE, prop-coloc and colocPropTest on pairs of datasets that measure the *same protein*, so colocalization is expected by construction.
  - Split-half of one dataset: all methods report colocalization for the majority of proteins.
  - Same protein measured on different platforms or in different populations: methods often call non-colocalization and often disagree. In the worst scenario, all four methods agreed on colocalization for only 20% of proteins.
  - **This is a ready-made template for our positive controls.** "Same protein, different population" tests LD mismatch. "Split-half" tests shared samples, if the halves are re-merged with overlap.
- **Large-scale evaluation (Reales et al. PLoS Genet 2026).**
  - Single-variant coloc only, with p12 = 5e-6.
  - 5% FDR thresholds of PP.H4 ≥ 0.86–0.88, assuming PP.H4 is calibrated (this was not validated).
  - Multi-gene colocalisation in ~47% (OTG) and ~37% (IMD) of peaks.
  - High-granularity, small-N single-cell eQTLs colocalise at a higher rate per eQTL.

### 2.2 Variant-level / enrichment approaches (Wen group)
- **ENLOC (Wen 2017) and fastENLOC.**
  - Uses molQTL fine-mapping (DAP-G PIPs) as a probabilistic annotation in a logistic prior for GWAS causal status. The enrichment is estimated by EM with multiple imputation.
  - Outputs SCP and RCP. Bayesian FDR = mean(1 − RCP).
  - The enrichment estimate is biased toward 0, which makes results conservative. Power was <40% at 20% FDR in simulations [FT].
- **Hukku 2021 (AJHG).**
  - Mis-specifying the prior enrichment "can lead to severe inflation of false-positive colocalization findings".
  - Recommends estimating priors from data and separating fine-mapping from colocalization.
  - Across 4,091 traits × GTEx v8, "only a small proportion [of colocalizations] can be confidently identified … due to a lack of power" [Abs].
- **Hukku 2022.** Reconciles TWAS and colocalization from an inferential-reproducibility viewpoint. It explains when the two disagree: TWAS is sensitive to LD hitchhiking, while coloc is power-limited.
- **INTACT (Okamoto 2023).** Uses the colocalization probability as a prior gate on TWAS evidence. Realised FDR 0.035 and power 0.41, versus TWAS-only FDR 0.18–0.39 [FT].

### 2.3 Joint multi-trait fine-mapping / coloc models
- **CAFEH (Arvanitis 2022).**
  - Components share a causal-variant selector, with per-tissue activity. Fit by CAVI.
  - Simulated from GTEx genotypes (N = 838).
  - Assigns causal variants with r² > 0.7 to one component.
  - Per the mvSuSiE paper, its PIPs were poorly calibrated under correlated traits (FDR 0.13 at PIP 0.95).
- **SharePro (Zhang W 2024).**
  - Effect groups with trait-specific activity. Fit by VI on summary statistics.
  - Simulations: K_C = 1, K_S = 1 → SharePro 100% detection, coloc-SuSiE 97.6%, coloc 44.4%. Runtime 4.3 s per locus.
  - Independent-ish benchmark (Zhang W, GPB 2026, same senior authors): 79.6% of 201 MR-significant protein–trait pairs supported, versus coloc 40.3%, coloc-SuSiE 46.8% and PWCoCo 45.8%. All methods had "well-controlled false discoveries".
  - Weaknesses: no overlap model; convergence issues under severe LD mismatch.
- **mvSuSiE (Zou 2026).**
  - A multivariate SuSiE with a data-driven mixture prior on cross-trait effects.
  - **Residual covariance V estimated from weak or "null" SNPs. This is the cleanest published way to absorb overlap-induced z-correlation in a summary-statistics fine-mapper.**
  - Not a coloc test, but per-trait lfsr per CS gives "which traits share this signal".
  - Forcing independence reduced performance and calibration.
- **flashfm (Hernández 2021).** Multi-trait fine-mapping for ≤6 traits that uses the trait covariance and handles missing measurements and related individuals. Reduces potential causal variants by 30% when traits share signals.
- **ColocBoost (Cao et al., medRxiv 2025, v3 2026-09).**
  - Multi-task boosting with an LD-proximity smoothing step.
  - Applied to 17 xQTL datasets in ROSMAP (average N = 595): 16,503 colocalization events and 10.7-fold heritability enrichment. Up to 2.5× more AD-colocalized loci.
  - The authors claim robustness to overlapping samples and missing LD. Not peer reviewed; an independent benchmark is needed.
- **HyPrColoc / moloc.**
  - Both are ABF single-causal multi-trait models. HyPrColoc is very fast.
  - HyPrColoc's `sample.overlap` default is a matrix of 1s (complete overlap) [Doc]. Users should be aware of this.
- **gwas-pw (Pickrell 2016).** Early, explicit four-model shared/distinct framework with an overlap correction estimated from summary statistics. Notes that the estimated correlation is non-zero even for known non-overlapping pairs, which makes it conservative.

### 2.4 Testing-based methods (explicit "distinct" null)
- **JLIM (Chun 2017).**
  - Asymmetric. Needs individual-level data for the QTL trait.
  - The permutation p-value is conservative for the distinct-variant null.
  - AUC 0.883 even at 0.7 < r² < 0.8 between causal variants. Shows coloc and gwas-pw have substantial false positives as LD between distinct causal variants rises.
  - Weak when eQTL p > 0.01.
- **SS2 / LocusFocus (Wang F 2022).**
  - Composite null (no eQTL, no GWAS, distinct). Explicit overlap covariance.
  - Type I error ≤0.045 across nulls, versus COLOC2 FPR 0.086 under H03.
  - Less powerful than coloc when eQTL evidence is weak. Gives a p-value, not a PP.
- **SMR / HEIDI (Zhu 2016).**
  - HEIDI tests heterogeneity of the SMR estimate across SNPs in LD (linkage vs pleiotropy) [mem].
  - Conservative: SS2 reports SMR and SMR-multi as very conservative.
  - Single-instrument and LD-reference dependent.

### 2.5 Conditional / imputation pipelines
- **PWCoCo.** COJO conditioning per signal plus coloc on all pairs. 38 s per locus. In SharePro's simulation, PP > 0.8 in 1.2% of K_S = 3 null cases. COJO with a mismatched reference creates false secondary signals (DENTIST).
- **POEMColoc.** Imputes the second trait from a lead SNP. R² = 0.81 with full coloc in real data. Fails with multiple signals: 9% false negatives, all called H3.

### 2.6 TWAS / MR-based
- **FUSION, FOCUS, PTWAS, cTWAS, TGFM, MRLocus, INTACT.** These are gene-level. TWAS z is inflated by LD hitchhiking and co-regulation.
  - FOCUS: 90% credible gene sets contain the causal gene 83% of the time (N_eQTL = 500).
  - TGFM: models uncertainty in cis-predicted expression, which gives correct calibration.
  - cTWAS: adjusts for genetic confounders.
  - MRLocus: coverage drops to 50–70% at low h²g or low h²med.
  - These methods rank genes. They do not separate variant-level shared from distinct signals. Useful mainly as orthogonal validation.
- **LCV.** Genome-wide, not locus-level. Not a competitor.

### 2.7 Scale / engineering
- **gpu-coloc (Jesse 2026).**
  - Groups signals by position into chunks of ≤1,000 and evaluates 100×100 batches.
  - Drops signals with max LBF < 5 (~¾ of signals).
  - Missing variants are set to LBF = −1e6.
  - Recommended settings: p12 = 1e-6 with PP.H4 ≥ 0.8, which corresponds to CLPP ≥ 0.04.
  - Collider rate 2.5% (gpu-coloc at p12 = 1e-6) versus 1.1% (CLPP).
  - Marginal-GWAS ABF recovers molQTL colocalisations for ~80% of loci that have fine-mapping.
- **tensorQTL.** Torch SuSiE and coloc on genotypes, GPU [Doc]. This is the only GPU SuSiE I found, and it is individual-level.
- **SuSiE 2.0 (McCreight 2025).** Modular rewrite; up to 5× faster for summary statistics; SuSiE-ash.

---

## 3. Cross-cutting issue 1: sample overlap

- **Theory.** With N_s shared samples and phenotypic correlation r_p, the sampling errors of z1 and z2 at the same SNP have correlation ρ ≈ N_s·r_p / √(N1·N2). This equals the LDSC cross-trait intercept (Bulik-Sullivan 2015 [mem]). Across SNPs, Cov(e1, e2) = ρR.
  - Correctly modelling ρ does two things: it removes spurious "shared noise" peaks under H1/H2/H3, and it **increases power**, because e2 | e1 has variance (1 − ρ²)R.
  - Ignoring ρ makes PPs over-confident in the direction of whatever the shared noise suggests.
  - It does **not** create a spurious true-effect association, because sampling error is not signal. The most dangerous case is therefore a weak signal in one trait plus large ρ, e.g. a pQTL and a GWAS both from UK Biobank.
- **Magnitude in practice.**
  - eQTL Catalogue / GTEx (N ≈ 100–1,000) vs biobank GWAS (N ≈ 400k): even with full containment, √(N_eQTL/N_GWAS) ≈ 0.02–0.05, so ρ is tiny.
  - UKB-PPP pQTL (~50k) inside UKB GWAS (~400k): √(50/400) ≈ 0.35 × r_p.
  - GWAS-vs-GWAS within a biobank: ρ can reach 0.3–0.6.
  - **Kill-test 0 (cheapest):** compute LDSC-intercept ρ̂ for our actual dataset pairs before investing in overlap modelling.
- **Who models it:**
  - gwas-pw: correlation from summary statistics.
  - HyPrColoc: optional matrices.
  - SS2: n0·ρ/√(n1·n2) with decorrelation.
  - mvSuSiE: V from null SNPs.
  - flashfm: trait covariance.
  - ColocBoost: claimed in simulations.
  - Classic meta-analysis precedent: Zhu X 2015 (CPASSOC) estimates the correlation matrix from null z.
- **Empirical calibration from meta-analysis.** Xu, Zhang & Elliott (bioRxiv 2026-10) compared five meta-analysis methods against a mega-analysis with known overlap. When overlap is small, its impact is "minimal, even at 5×10⁻⁸". This is consistent with the ρ-magnitude argument above.
- **Nobody** has combined overlap modelling with multi-causal coloc and LD-mismatch robustness.

## 4. Cross-cutting issue 2: LD-reference mismatch

- **Size of the effect.**
  - Zou 2022 [FT]: with 500–1,000-sample out-of-sample panels, SuSiE-RSS, FINEMAP and DAP-G all lost 95% CS coverage. The smaller panel was notably worse.
  - Benner 2017: reference panel size must scale with GWAS N [mem].
  - DENTIST recommends n_ref ≥ 5,000. Using HRS (n = 8.6k) instead of in-sample LD inflated UKB height COJO signals by 15.5%.
  - SLALOM: meta-analysis heterogeneity (arrays, imputation panels, per-SNP N) drives FDR up to 37%.
- **Remedies, ranked by maturity:**
  1. Use in-sample LD or LD sketches. This is often impossible for public GWAS; Ahmed et al. (2026 preprint) call for sharing in-sample LD.
  2. λ-regularisation estimated by ML under z = 0, plus kriging/DENTIST outlier removal (SuSiE-RSS).
  3. Explicit finite-reference and mismatch variance with multi-panel mixing and a single-effect (SER) fallback (susieR 2.x `R_finite`, `R_mismatch = "eb_mix"`). Documented only in package vignettes; I found no paper.
  4. Mismatch-robust variational models: RSparsePro (bioRxiv 2024), CARMA outlier model.
  5. LD-free coloc: ABF/single-variant, or CLPP on in-sample-fine-mapped QTLs (eQTL Catalogue uses individual-level SuSiE).
- **Consequence for coloc.** Mismatch creates spurious secondary SuSiE signals in the GWAS. These pair with QTL signals as H3 ("distinct") or, if they land near a QTL, as false H4. It also inflates collider rates. No coloc paper quantifies H3/H4 error as a function of mismatch. **This is a publishable benchmark gap.**
- **Variable information** (per-SNP N, imputation quality) is a closely related, under-modelled failure mode (Wallace 2025; SLALOM). The first fine-mapper that models it explicitly is FINEMAP-miss (Kartau & Pirinen 2025); I found no coloc method that does. A likelihood that uses per-SNP N_j can absorb part of it: z_j has a non-centrality proportional to √N_j, and the correlation of z_i and z_j scales with N_ij/√(N_i·N_j).

## 5. Cross-cutting issue 3: "distinct and powered" vs "too weak"

- coloc's PP.H3 is the posterior that both traits have a causal variant and that they differ. It is conditioned on the locus definition and the priors, and it is low whenever either trait is underpowered (mass moves to H0, H1 or H2). So coloc's 5-way output *can* separate the cases in principle.
  - In practice PP.H3 is miscalibrated when there are multiple signals (single-variant ABF), when signals are merged (SuSiE), and under variable information.
  - Many pipelines also only threshold PP.H4, which throws the distinction away.
- **Natural negative controls.** Conditionally independent signals for the *same* trait at one locus (e.g. two SuSiE CSs of one eQTL) are distinct by construction, and both are powered. The rate at which a method calls H4 between them, or between one of them and a GWAS signal that colocalises with the other, is a direct false-H4 metric. Jesse 2026's "collider" metric is the large-scale version of this idea.
- **Methods with an explicit distinct null:** JLIM / scJLIM, SS2, the proportional tests (colocPropTest, prop-coloc), gwas-pw (M4).
- **Ingredient to add:** a power statement. For trait B, compute P(detect | shared effect of the size implied by trait A), using N_B, MAF and the posterior effect from A. Then report "insufficient" when that predictive power is low, rather than H3 or H4. I found no published coloc method that outputs this explicitly.
  - Hukku 2021 and Rosen 2026 motivate it.
  - Rosen 2026: N = 500 eQTL studies detect <0.1–60% of eQTL depending on effect size.
  - Mostafavi 2023: GWAS and eQTL discovery is systematically biased toward different variant classes.

## 5b. Speed summary (per locus unless noted)

| Step | Fastest reported | Notes |
|---|---|---|
| ABF coloc (CPU, R) | ~0.1 s | O(p) |
| gpu-coloc pairwise (precomputed LBF) | ~2,900 pairs/s on a laptop; ~800× faster than R on an H200 | Batched; fine-mapping not included |
| CLPP | ~200× faster than gpu-coloc | Needs credible sets for both traits |
| SuSiE-RSS (p ≈ 1–5k) | 1.3 s mean (no refinement), 5.5 s with refinement | Per IBSS sweep O(L·p²) with dense R, or O(L·p·k) with a rank-k factor [mem]; reading LD from disk often dominates |
| SharePro | 4.3 s | VI |
| coloc-SuSiE end-to-end | 14 s | — |
| mvSuSiE | 41 s (2 traits) | 5k SNPs |
| eCAVIAR | 228 s | Exhaustive |
| MsCAVIAR | hours | — |
| HyPrColoc | <1 s for 100 traits | Single-causal |

The fine-mapping step, LD I/O and LD QC dominate. A GPU-batched SuSiE-RSS / joint model over thousands of loci with padded LD tensors is the obvious engineering win.
- An RTX 4090 has 24 GB. A 2,000×2,000 float32 LD matrix is 16 MB, so ~1,000 loci fit per batch.

---

## 6. Gaps & opportunities, with my ranking and the cheapest kill-tests

Notation: z1, z2 are the GWAS and QTL z-vectors; R is the LD; ρ is the cross-trait error correlation.

### Opportunity A (most promising; main family): joint bivariate SuSiE-RSS with a shared/distinct selector, overlap ρ, and an LD-error model, GPU-batched
- **Model.**
  - Stack z = (z1, z2). Each of L effects picks one variant (shared selector, as in CAFEH/SharePro) and a sharing state s_l ∈ {1-only, 2-only, both}, with effect sizes (b1_l, b2_l). The prior on s_l carries p1, p2 and p12.
  - Likelihood: z ~ N((√N1·R·b1, √N2·R·b2), [[R̃, ρR̃], [ρR̃, R̃]]), where R̃ = (1−λ)R_ref + λI plus a finite-reference / mismatch term.
  - Whitening across traits, z2' = (z2 − ρz1)/√(1−ρ²), turns it into two conditionally independent RSS likelihoods. The IBSS updates stay closed-form.
- **Output per signal pair:** P(shared), P(distinct), and P(either signal absent). Combined with the predictive-power check (§5), this gives the three-way call.
- **Why it is promising:**
  - It subsumes coloc-SuSiE (ρ = 0, independent selectors), SharePro and mvSuSiE-style overlap.
  - Fitting both traits jointly borrows strength, which is what SharePro and ColocBoost exploit to gain sensitivity for weak shared signals.
  - IBSS is matrix-vector heavy, so it batches well on GPU.
- **Cheapest kill-tests:**
  1. On ~200 simulated loci from real LD (1 or 2 causal variants per trait; shared / distinct / one-trait-only), compare its AUPRC for shared vs distinct against coloc-SuSiE and SharePro, using identical in-sample LD and ρ = 0. If it is not at least as good as SharePro, stop and reuse SharePro's model.
  2. Repeat with ρ ∈ {0.2, 0.5}. If naive coloc-SuSiE shows no H4/H3 miscalibration at realistic ρ (see kill-test 0 in §3), downgrade the overlap component to an option.

### Opportunity B (second family): exact Bayesian enumeration over ≤2–3 causal variants per trait on GPU
- **Model.**
  - Screen the top K variants (K ≈ 100–300) by marginal and conditional evidence.
  - Enumerate all configurations of ≤2 causal variants per trait: ~K²/2 ≈ 45k configurations per trait at K = 300.
  - Compute exact BFs under the RSS likelihood, with optional λ, ρ and N_j adjustments, as batched 2×2/4×4 solves.
  - Sum over joint configuration pairs with the overlap structure to get exact PPs for {H0, H1, H2, H3, H4, "H4+3" (one shared plus one distinct)}.
  - With ρ = 0 the double sum factorises into per-variant marginal sums plus inclusion–exclusion, so the cost is O(#configs), not O(#configs²).
- **Why it matters:**
  - Removes the variational pathologies: merged causal variants, missed weak second signals, local optima.
  - Gives a *gold standard* to calibrate A and C against.
  - Extends coloc's 5-hypothesis output naturally to "partially shared".
  - ~1e5 configurations × thousands of loci is trivial on a 4090.
- **Kill-test:** on loci with two causal variants per trait in moderate LD (r² 0.3–0.8), compare the false-H4 rate on natural negative controls (conditionally independent same-trait signals) and the H3/H4 calibration (ECE) against coloc-SuSiE. If there is no improvement, keep it only as the gold standard.

### Opportunity C (bold): amortised neural posterior estimation (NPE) for colocalization
- **Model.**
  - Train a network on millions of simulated loci. Use real LD blocks from the target ancestry, and draw the reference LD from a *different* subsample or population to teach mismatch. Vary N1, N2, ρ, per-SNP missingness, and causal configurations: shared, distinct, partially shared, one-only, none.
  - The network maps (z1, z2, R_ref, N_j) to class posteriors {shared, distinct-both-powered, insufficient} plus per-variant shared-PIPs.
  - Architecture: a set transformer over SNPs, with LD as an attention bias or a low-rank eigen-projection as input. Check calibration with simulation-based calibration and conformal decision thresholds.
- **Why it is bold:**
  - Millisecond inference.
  - It can learn robustness to mismatch, overlap and variable information that no closed-form model captures.
  - The "insufficient power" class is learned directly from the simulator's ground truth.
- **Risks:**
  - Distribution shift between simulator and reality.
  - O(p²) LD input.
  - Calibration outside the training domain.
- **Kill-tests:**
  1. Train on a single LD block to reproduce exact coloc-ABF PPs (B, ρ = 0). If it cannot reach |ΔPP| < 0.02, stop.
  2. Train on mismatched-LD simulations and test on held-out LD regions and chromosomes. It must beat A or B with λ-regularisation on AUPRC (shared vs distinct) at equal ECE.

### Cross-cutting add-ons (cheap, high expected value)
- **D. Power-aware three-way decision layer**, usable on top of any method (§5): predictive detection power for each trait under the "shared" hypothesis. Kill-test: on simulations, does it reduce "H3" calls in underpowered loci without losing true distinct calls?
- **E. LD-mismatch layer:**
  - Estimate λ, run a kriging / DENTIST-S outlier mask, use an R_finite-style inflation, and fall back to a single effect when the fit is unstable.
  - Benchmark H3/H4 error versus reference panel (in-sample, UKB-subsample n = 500 / 5,000, 1000G EUR n ≈ 500, mismatched ancestry). This benchmark is itself a novel contribution.
- **F. Variable-information correction:** use per-SNP N_j in the likelihood. Cheap; especially relevant to meta-analysed GWAS.
- **G. Batched GPU SuSiE-RSS / coloc as an engineering baseline:** identity to susieR (|ΔPIP| < 1e-3 on 100 loci) is the first test. Precompute LBFs per signal (eQTL Catalogue style), so pairwise coloc runs at gpu-coloc speed.
- **Lower priority:**
  - Variant-specific priors (≤0.04 recall gain).
  - Multi-trait (>2) extensions: HyPrColoc and ColocBoost already exist.
  - TWAS-style gene-level methods: they answer a different question.

### My ranking
1. **A + D + E**: the main bet, most likely to beat coloc-SuSiE and SharePro on both power and calibration while running 10–100× faster on GPU.
2. **B**: the gold standard and a possible winner for small loci. Cheap to build.
3. **C**: the bold family. Highest upside on speed and on mismatch robustness, highest risk.

Keep all three alive until their kill-tests have been run.

### Known failure modes to include in our benchmark (each needs its own simulation scenario)
1. Two causal variants per trait in moderate LD (coloc-ABF H3/H4 equivocation; SuSiE merging).
2. LD-reference mismatch: small panel, wrong ancestry, meta-analysis heterogeneity. Expect spurious secondary signals and false H3, and coverage loss.
3. Variable per-SNP N / imputation quality / missingness. Expect confident false H3/H4 (Wallace 2025; SLALOM).
4. Sample overlap with r_p ≠ 0. Expect miscalibration of independent-likelihood methods.
5. Prior sensitivity: p12 = 1e-5 vs 5e-6 vs 1e-6 (Wallace 2020; Jesse 2026).
6. Region-size dependence of H3 priors.
7. Low QTL power (N ≈ 100–500), which turns true shared signals into H1/H3 (Rosen 2026; Hukku 2021).
8. Co-regulation: multiple genes share one QTL variant, so H4 with several genes is correct but uninformative about the target gene (Tambets 2024; Reales 2026).
9. Allele flips and strand issues (kriging catches these; the proportional test is very sensitive to them).

---

## 7. Verified reference list

All entries were checked against PubMed (PMID shown), the bioRxiv/medRxiv API, or the repository. "Preprint" means not peer reviewed as of 2026-10-26.

**Colocalization methods**
- Giambartolomei C, et al. Bayesian test for colocalisation between pairs of genetic association studies using summary statistics. *PLoS Genet* 2014;10:e1004383. doi:10.1371/journal.pgen.1004383 (PMID 24830394)
- Wallace C. Statistical testing of shared genetic control for potentially related traits. *Genet Epidemiol* 2013. doi:10.1002/gepi.21765 (PMID 24227294)
- Wallace C. Eliciting priors and relaxing the single causal variant assumption in colocalisation analyses. *PLoS Genet* 2020;16:e1008720. doi:10.1371/journal.pgen.1008720 (PMID 32310995). Correction 2026: doi:10.1371/journal.pgen.1012155
- Wallace C. A more accurate method for colocalisation analysis allowing for multiple causal variants. *PLoS Genet* 2021;17:e1009440. doi:10.1371/journal.pgen.1009440 (PMID 34587156)
- Pullin JM, Wallace C. Variant-specific priors clarify colocalisation analysis. *PLoS Genet* 2025;21:e1011697. doi:10.1371/journal.pgen.1011697 (PMID 40424384)
- Wallace C, Robins C, Johnson T. Variable information across SNPs in GWAS data can cause false rejections of colocalisation which can be resolved by proportional colocalisation tests. *bioRxiv* 2025. doi:10.1101/2025.09.08.674910 (preprint). Software: CRAN `colocPropTest` v0.9.3.
- Reales G, Pullin JM, Manipur I, Vigorito E, Wallace C. Design and interpretation of eQTL-GWAS colocalisation studies: lessons from a large-scale evaluation. *PLoS Genet* 2026;22:e1012141. doi:10.1371/journal.pgen.1012141
- Giambartolomei C, et al. A Bayesian framework for multiple trait colocalization from summary association statistics (moloc). *Bioinformatics* 2018;34:2538. doi:10.1093/bioinformatics/bty147 (PMID 29579179)
- Foley CN, et al. A fast and efficient colocalization algorithm for identifying shared genetic risk factors across multiple traits (HyPrColoc). *Nat Commun* 2021;12:764. doi:10.1038/s41467-020-20885-8 (PMID 33536417)
- Pickrell JK, et al. Detection and interpretation of shared genetic influences on 42 human traits (gwas-pw). *Nat Genet* 2016;48:709. doi:10.1038/ng.3570 (PMID 27182965)
- Hormozdiari F, et al. Colocalization of GWAS and eQTL signals detects target genes (eCAVIAR). *Am J Hum Genet* 2016;99:1245. doi:10.1016/j.ajhg.2016.10.003 (PMID 27866706)
- Wen X, Pique-Regi R, Luca F. Integrating molecular QTL data into genome-wide genetic association analysis: probabilistic assessment of enrichment and colocalization (ENLOC). *PLoS Genet* 2017;13:e1006646. doi:10.1371/journal.pgen.1006646 (PMID 28278150)
- Hukku A, et al. Probabilistic colocalization of genetic variants from complex and molecular traits: promise and limitations. *Am J Hum Genet* 2021;108:25. doi:10.1016/j.ajhg.2020.11.012 (PMID 33308443)
- Hukku A, et al. Analyzing and reconciling colocalization and transcriptome-wide association studies from the perspective of inferential reproducibility. *Am J Hum Genet* 2022. doi:10.1016/j.ajhg.2022.04.005 (PMID 35523146)
- Okamoto J, et al. Probabilistic integration of transcriptome-wide association studies and colocalization analysis identifies key molecular pathways of complex traits (INTACT). *Am J Hum Genet* 2023. doi:10.1016/j.ajhg.2022.12.002 (PMID 36608684)
- Zhu Z, et al. Integration of summary data from GWAS and eQTL studies predicts complex trait gene targets (SMR/HEIDI). *Nat Genet* 2016;48:481. doi:10.1038/ng.3538 (PMID 27019110)
- Wu Y, et al. Integrative analysis of omics summary data reveals putative mechanisms underlying complex traits (SMR multi-omics). *Nat Commun* 2018. doi:10.1038/s41467-018-03371-0 (PMID 29500431)
- Chun S, et al. Limited statistical evidence for shared genetic effects of eQTLs and autoimmune-disease-associated loci in three major immune-cell types (JLIM). *Nat Genet* 2017;49:600. doi:10.1038/ng.3795 (PMID 28218759)
- Wang F, et al. A flexible summary statistics-based colocalization method with application to the mucin cystic fibrosis lung disease modifier locus (SS2 / LocusFocus). *Am J Hum Genet* 2022. doi:10.1016/j.ajhg.2021.12.012 (PMID 35065708)
- Zheng J, et al. Phenome-wide Mendelian randomization mapping the influence of the plasma proteome on complex diseases (introduces PWCoCo). *Nat Genet* 2020. doi:10.1038/s41588-020-0682-6 (PMID 32895551)
- Robinson JW, et al. An efficient and robust tool for colocalisation: Pair-wise Conditional and Colocalisation (PWCoCo). *bioRxiv* 2022. doi:10.1101/2022.08.08.503158 (preprint; cited from the GitHub README github.com/jwr-git/pwcoco; journal version not found)
- King EA, Dunbar F, Davis JW, Degner JF. Estimating colocalization probability from limited summary statistics (POEMColoc). *BMC Bioinformatics* 2021;22:254. doi:10.1186/s12859-021-04170-z (PMID 34000989)
- Arvanitis M, et al. Redefining tissue specificity of genetic regulation of gene expression in the presence of allelic heterogeneity (CAFEH). *Am J Hum Genet* 2022. doi:10.1016/j.ajhg.2022.01.002 (PMID 35085493)
- Zhang W, et al. SharePro: an accurate and efficient genetic colocalization method accounting for multiple causal signals. *Bioinformatics* 2024;40:btae295. doi:10.1093/bioinformatics/btae295 (PMID 38688586)
- Zhang W, et al. Accounting for genetic effect heterogeneity in fine-mapping and improving power to detect gene-environment interactions with SharePro. *Nat Commun* 2024. doi:10.1038/s41467-024-53818-w (PMID 39478020)
- Zhang W, Yoshiji S, Sladek R, Dupuis J, Lu T. Benchmarking Bayesian colocalization methods in validating Mendelian randomization-identified targets. *Genomics Proteomics Bioinformatics* 2026:qzag087. doi:10.1093/gpbjnl/qzag087 (PMID 42613742)
- Cao X, et al. (Wang G senior). Integrative multi-omics QTL colocalization maps regulatory architecture in aging human brain (ColocBoost). *medRxiv* 2025, v3 2026-09-04. doi:10.1101/2025.04.17.25326042 (preprint). Software: CRAN `colocboost`.
- Patel A, Burgess S. A frequentist test of proportional colocalization after selecting relevant genetic variants (prop-coloc). *Hum Hered* 2026. doi:10.1159/000553247 (PMID 42696482). arXiv 2402.12171. R `prop.coloc` v2.0.0.
- Hwang S, Pullin JM, Wallace C, Whittaker J, Burgess S. Systematic comparison of colocalization methods using protein quantitative trait loci. *bioRxiv* 2025. doi:10.1101/2025.11.07.686776 (preprint)
- Mitchel J, et al. (Chun S, Cotsapas C, Kharchenko P). A single-cell genetic colocalization test improves power and resolves disease-mediating cell types (scJLIM). *bioRxiv* 2025. doi:10.1101/2025.10.10.681685 (preprint)
- Iotchkova V, Weale ME. Multi-trait colocalisation using MystraColoc: improved performance, deeper insights. *bioRxiv* 2026. doi:10.64898/2026.03.30.715409 (preprint)
- Diana MA, et al. COLSTATS: an interactive web platform for systematic colocalization of GWAS and QTL summary statistics. *bioRxiv* 2026. doi:10.64898/2026.01.28.702279 (preprint). A resource of 1.65M harmonised hg38 summary-stat datasets running coloc.abf; potentially useful for data.
- Xu Z, Zhang L, Elliott LT. Sample overlap in genome-wide association meta-analysis: practical comparisons across significance thresholds. *bioRxiv* 2026. doi:10.64898/2026.10.14.759234 (preprint)
- Jesse M, Riet AE, Alasoo K. Ultra-fast genetic colocalisation across millions of association signals (gpu-coloc). *PLoS Genet* 2026;22:e1012209. doi:10.1371/journal.pgen.1012209 (PMID 42308237). Preprint 10.1101/2025.08.25.672103. Code: github.com/mjesse-github/gpu-coloc
- Shen C, Dupuis J, Zhang Q. Multi-ancestry colocalization approaches. *PLoS Genet* 2026;22:e1012221. doi:10.1371/journal.pgen.1012221 (PMID 42479736)
- Butler-Laporte G, et al. An accurate genetic colocalisation method for the HLA locus. *HLA* 2026;107:e70759. doi:10.1111/tan.70759 (PMID 42174329)
- Zou Y, et al. Fast and flexible joint fine-mapping of multiple traits via the Sum of Single Effects model (mvSuSiE). *Nat Genet* 2026. doi:10.1038/s41588-025-02486-7 (PMID 41634413)
- Hernández N, et al. The flashfm approach for fine-mapping multiple quantitative traits. *Nat Commun* 2021;12:6147. doi:10.1038/s41467-021-26364-y (PMID 34686674)

**TWAS / MR / gene-level**
- Gusev A, et al. Integrative approaches for large-scale transcriptome-wide association studies (FUSION). *Nat Genet* 2016;48:245. doi:10.1038/ng.3506 (PMID 26854917)
- Mancuso N, et al. Probabilistic fine-mapping of transcriptome-wide association studies (FOCUS). *Nat Genet* 2019;51:675. doi:10.1038/s41588-019-0367-1 (PMID 30926970)
- Zhang Y, et al. PTWAS: investigating tissue-relevant causal molecular mechanisms of complex traits using probabilistic TWAS analysis. *Genome Biol* 2020. doi:10.1186/s13059-020-02026-y (PMID 32912253)
- Zhao S, et al. Adjusting for genetic confounders in transcriptome-wide association studies improves discovery of risk genes of complex traits (cTWAS). *Nat Genet* 2024. doi:10.1038/s41588-023-01648-9 (PMID 38279041)
- Strober BJ, et al. Fine-mapping causal tissues and genes at disease-associated loci (TGFM). *Nat Genet* 2024. doi:10.1038/s41588-024-01994-2
- Zhu A, et al. MRLocus: identifying causal genes mediating a trait through Bayesian estimation of allelic heterogeneity. *PLoS Genet* 2021. doi:10.1371/journal.pgen.1009455 (PMID 33872308)
- Barfield R, et al. Transcriptome-wide association studies accounting for colocalization using Egger regression (LDA MR-Egger). *Genet Epidemiol* 2018. doi:10.1002/gepi.22131 (PMID 29808603)
- O'Connor LJ, Price AL. Distinguishing genetic correlation from causation across 52 diseases and complex traits (LCV). *Nat Genet* 2018;50:1728. doi:10.1038/s41588-018-0255-0 (PMID 30374074)

**Fine-mapping, LD mismatch and QC**
- Wang G, Sarkar A, Carbonetto P, Stephens M. A simple new approach to variable selection in regression, with application to genetic fine mapping (SuSiE). *JRSS-B* 2020;82:1273. doi:10.1111/rssb.12388 (PMID 37220626)
- Zou Y, Carbonetto P, Wang G, Stephens M. Fine-mapping from summary data with the "Sum of Single Effects" model (SuSiE-RSS). *PLoS Genet* 2022;18:e1010299. doi:10.1371/journal.pgen.1010299 (PMID 35853082)
- McCreight A, et al. SuSiE 2.0: improved methods and implementations for genetic fine-mapping and phenotype prediction. *bioRxiv* 2025. doi:10.1101/2025.11.25.690514 (preprint, v2 2025-12-25)
- susieR vignette "Modeling and accounting for LD reference mismatch" (`R_finite`, `R_mismatch`), https://stephenslab.r-universe.dev/susieR/doc/rss_mismatch.html [Doc; no paper found]
- Cui R, et al. Improving fine-mapping by modeling infinitesimal effects (SuSiE-inf / FINEMAP-inf). *Nat Genet* 2024;56:162. doi:10.1038/s41588-023-01597-3 (PMID 38036779)
- Benner C, et al. FINEMAP: efficient variable selection using summary data from genome-wide association studies. *Bioinformatics* 2016;32:1493. doi:10.1093/bioinformatics/btw018 (PMID 26773131)
- Benner C, et al. Prospects of fine-mapping trait-associated genomic regions by using summary statistics from genome-wide association studies. *Am J Hum Genet* 2017;101:539. doi:10.1016/j.ajhg.2017.08.012 (PMID 28942963)
- Benner C, Mahajan A, Pirinen M. Refining fine-mapping: effect sizes and regional heritability. *PLoS Genet* 2025;21:e1011480. doi:10.1371/journal.pgen.1011480 (PMID 39787248)
- Kartau J, Pirinen M. FINEMAP-miss: fine-mapping genome-wide association studies with missing genotype information. *Bioinformatics* 2025. doi:10.1093/bioinformatics/btaf616 (preprint medRxiv 10.1101/2025.09.30.25336878). R `finemapmiss`.
- Wen X, Lee Y, Luca F, Pique-Regi R. Efficient integrative multi-SNP association analysis via deterministic approximation of posteriors (DAP). *Am J Hum Genet* 2016. doi:10.1016/j.ajhg.2016.03.029 (PMID 27236919)
- Weissbrod O, et al. Functionally informed fine-mapping and polygenic localization of complex trait heritability (PolyFun). *Nat Genet* 2020;52:1355. doi:10.1038/s41588-020-00735-5 (PMID 33199916)
- Yang Z, et al. CARMA is a new Bayesian model for fine-mapping in genome-wide association meta-analyses. *Nat Genet* 2023;55:1057. doi:10.1038/s41588-023-01392-0 (PMID 37169873)
- Chen W, et al. Improved analyses of GWAS summary statistics by reducing data heterogeneity and errors (DENTIST). *Nat Commun* 2021;12:7117. doi:10.1038/s41467-021-27438-7 (PMID 34880243)
- Kanai M, et al. Meta-analysis fine-mapping is often miscalibrated at single-variant resolution (SLALOM). *Cell Genom* 2022;2:100210. doi:10.1016/j.xgen.2022.100210 (PMID 36643910)
- Zhang W, Najafabadi H, Li Y. SparsePro: an efficient fine-mapping method integrating summary statistics and functional annotations. *PLoS Genet* 2023. doi:10.1371/journal.pgen.1011104 (PMID 38153934)
- Zhang W, Lu T, Sladek R, Dupuis J, Lettre G. Robust fine-mapping in the presence of linkage disequilibrium mismatch (RSparsePro). *bioRxiv* 2024. doi:10.1101/2024.10.29.620968 (preprint)
- Zhu X, Stephens M. Bayesian large-scale multiple regression with summary statistics from genome-wide association studies (RSS). *Ann Appl Stat* 2017;11:1561. doi:10.1214/17-aoas1046 (PMID 29399241)
- Yang J, et al. Conditional and joint multiple-SNP analysis of GWAS summary statistics (GCTA-COJO). *Nat Genet* 2012;44:369. doi:10.1038/ng.2213 (PMID 22426310)
- Ghosal S, Schatz M, Venkataraman A. BEATRICE: Bayesian fine-mapping from summary data using deep variational inference. *Bioinformatics* 2024. doi:10.1093/bioinformatics/btae590 (PMID 39360993)
- Karhunen V, et al. Genetic fine-mapping from summary data using a nonlocal prior improves the detection of multiple causal variants (FiniMOM). *Bioinformatics* 2023. doi:10.1093/bioinformatics/btad396 (PMID 37348543)
- Wu Y, et al. Genome-wide fine-mapping improves identification of causal variants (GWFM). *Nat Genet* 2026. doi:10.1038/s41588-026-02549-3 (preprint medRxiv 10.1101/2024.07.18.24310667)
- Gao B, et al. MESuSiE enables scalable and powerful multi-ancestry fine-mapping. *Nat Genet* 2024. doi:10.1038/s41588-023-01604-7 (PMID 38168930)
- Yuan K, et al. Fine-mapping across diverse ancestries drives the discovery of putative causal variants (SuSiEx). *Nat Genet* 2024. doi:10.1038/s41588-024-01870-z (PMID 39187616)
- Rossen J, et al. MultiSuSiE improves multi-ancestry fine-mapping in All of Us whole-genome sequencing data. *Nat Genet* 2026. doi:10.1038/s41588-025-02450-5 (PMID 41491094)
- Lu Z, et al. Improved multiancestry fine-mapping identifies cis-regulatory variants underlying molecular traits and disease risk (SuShiE). *Nat Genet* 2025. doi:10.1038/s41588-025-02262-7 (PMID 40691406)
- Taylor-Weiner A, et al. Scaling computational genomics to millions of individuals with GPUs (tensorQTL; includes Torch `susie.py` and `coloc.py`). *Genome Biol* 2019;20:228. doi:10.1186/s13059-019-1836-7 (PMID 31675989)

**Overlap / statistical background**
- Bulik-Sullivan BK, et al. LD Score regression distinguishes confounding from polygenicity in genome-wide association studies. *Nat Genet* 2015;47:291. doi:10.1038/ng.3211 (PMID 25642630)
- Zhu X, et al. Meta-analysis of correlated traits via summary statistics from GWASs with an application in hypertension (CPASSOC; null-z correlation for overlap). *Am J Hum Genet* 2015;96:21. doi:10.1016/j.ajhg.2014.11.011 (PMID 25500260)
- Wakefield J. Bayes factors for genome-wide association studies: comparison with P-values. *Genet Epidemiol* 2009;33:79. doi:10.1002/gepi.20359 (PMID 18642345)

**Context: why coloc rates are low**
- Rosen JD, et al. Higher eQTL power reveals signals that boost GWAS colocalization. *Am J Hum Genet* 2026. doi:10.1016/j.ajhg.2026.02.009 (PMID 41785861)
- Mostafavi H, et al. Systematic differences in discovery of genetic effects on gene expression and complex traits. *Nat Genet* 2023;55:1866. doi:10.1038/s41588-023-01529-1 (PMID 37857933)
- Connally NJ, et al. The missing link between genetic association and regulatory function. *eLife* 2022;11:e74970. doi:10.7554/eLife.74970 (PMID 36515579)
- Tambets R, et al. Extensive co-regulation of neighboring genes complicates the use of eQTLs in target gene prioritization. *HGG Adv* 2024. doi:10.1016/j.xhgg.2024.100348 (PMID 39210598)
- Kerimov N, et al. A compendium of uniformly processed human gene expression and splicing quantitative trait loci (eQTL Catalogue). *Nat Genet* 2021;53:1290. doi:10.1038/s41588-021-00924-w (PMID 34493866)
- Ahmed OY, et al. Current challenges in GWAS integration and fine-mapping for variant interpretation. *bioRxiv* 2026. doi:10.64898/2026.07.04.736511 (preprint)

**Searched for but not found**
- **"COLOC-reporter":** no such tool found.
- **A deep-learning or amortised colocalization method:** none found. BEATRICE is the closest; it is neural VI fine-mapping, not amortised and not coloc.
- **A GPU summary-statistics SuSiE:** none found. tensorQTL's version is individual-level.
- **A coloc paper quantifying H3/H4 error versus LD-reference mismatch:** none found.
- **A peer-reviewed ColocBoost:** not yet published; v3 preprint only.
