# The colocalization gap and how to evaluate colocalization methods

Literature review for the project "colocalization: speed and power". Compiled 2026-10-26.

**How this was verified.** Every citation in the reference list (Section 8) was checked against Crossref and/or PubMed: title, first author, year, venue and DOI. Quantitative statements were taken from abstracts or PMC/Europe PMC full texts, quoted where possible.

**Tags used:**
- **[PREPRINT]**: not peer reviewed.
- **[UNVERIFIED]**: the number or detail was not seen in the primary source (for example, it sits in an inaccessible supplement). Treat it as a lead, not a fact.

**Scope:**
- Why GWAS signals and molecular QTLs rarely colocalize, with the coloc rates reported.
- Power and prior sensitivity of colocalization.
- Sample overlap.
- LD-reference mismatch.
- How colocalization methods have been evaluated.
- A proposed evaluation protocol for this project (Section 7).

---

## 0. Executive summary: the ten facts that should shape our design

1. **Locus-level coloc rates are moderate, not tiny, and depend strongly on denominator and threshold.**

   | Study | Comparison | Coloc rate |
   |---|---|---|
   | GTEx v8 (enloc) | 5,385 GWAS loci × 49 tissues | 43% with an eQTL, 23% with an sQTL |
   | Open Targets, Mountjoy 2021 | 70,364 loci × 92 QTL datasets, H4 > 0.8 | 50.6% with ≥1 gene |
   | Reales 2026 | >1.3M Open Targets tests | 50% of GWAS peaks |
   | Reales 2026 | Immune-mediated diseases (IMD) × 12 immune eQTL studies | 34% |
   | Mu 2021 | Immune traits × 18 immune cell types | 40.4% |
   | Chun 2017 (JLIM) | Autoimmune loci, 3 immune cell types | ~25% |

   Per trait the rate is lower: a median of 21% of hits per trait in GTEx (as cited by Mostafavi 2023), and about 20% for a typical trait in Barbeira 2021.

2. **Gene-level and heritability-level views are much bleaker.**
   - Only 11 ± 2% of heritability is mediated by assayed bulk expression (MESC; Yao 2020).
   - For 220 genes known through coding variants, coloc implicated 8%, TWAS 2% and annotation+distance 4% (Connally 2022).
   - The "colocalization gap" is still at least 40% of GWAS loci (Reales 2026).

3. **The main explanations are, in rough order of evidence: power, systematic differences, context, conservatism, and co-regulation.**
   - **Power of the QTL study:**
     - eQTL N = 500 detects <0.1–60% of eQTLs, and N = 2,000 about 37% (Rosen 2026).
     - Doubling eQTL N roughly doubled colocalizing GWAS signals (Rosen 2026).
     - fastENLOC finds only about 10.7% of the expected colocalizations (Hukku 2021).
   - **Systematic differences between GWAS and eQTL discovery** (Mostafavi 2023):
     - GWAS hits sit further from the TSS (median 36 kb vs 13 kb for eQTLs).
     - GWAS hits are near constrained genes.
     - Selection purges large-effect eQTLs at constrained genes.
   - **Context:**
     - 60% of coloc genes in CD4+ T cells were seen only upon activation (Soskic 2022).
     - 48% of colocalizing IMD peaks colocalized only with sorted or single-cell eQTLs (Reales 2026).
   - **Method conservatism:** height GWAS colocalized against itself (GIANT vs UKB) gave enloc colocalization probability < 0.01 for 66% of credible sets (Barbeira 2021).
   - **Co-regulation and multi-gene coloc:** 37–47% of colocalizing peaks colocalize with more than one gene (Reales 2026).

4. **"No colocalization" is mostly "no information".** Giambartolomei 2014 already said that a low PP4 with a low PP3 "may simply be the result of limited power". Only H3 (two distinct, well-powered signals) is a real negative. This is exactly the project's framing, and the literature supports it.

5. **Priors matter more than users think.**
   - 22 of 25 studies used the coloc default p12 = 1e-5 (Wallace 2020).
   - Wallace recommends 5e-6. The default favours H4 under true H3, especially at small N.
   - The prior on H3 scales with n² (n = number of SNPs in the region), so region size alone shifts H3 vs H4.
   - With default priors, coloc's realized FDR was 0.258 at a nominal 5% (Hukku 2021 preprint).

6. **Distinguishing "shared" from "distinct but in LD" breaks down above r² ≈ 0.8** between the two causal variants (JLIM simulations). Any benchmark must stratify the distinct class by r².

7. **Sample overlap.**
   - Theory: the null z-correlation is ρ_z = N_s·ρ_p / √(N1·N2), where N_s is the number of shared samples and ρ_p the phenotypic correlation. It is estimable from the cross-trait LDSC intercept or from the correlation of null-SNP z-scores (Bulik-Sullivan 2015; PLACO; OPERA).
   - No peer-reviewed paper quantifies how overlap changes coloc.abf / coloc-SuSiE posteriors in the main text. HyPrColoc (supplement), coloc GitHub replies and ColocBoost [PREPRINT] all report "limited practical impact" at modest N. That remains an open, testable question for us.

8. **LD-reference mismatch.**
   - It is a major source of false fine-mapped signals. Kanai 2022: FDR up to 37% vs ≤2% expected in meta-analysis fine-mapping; 28% of GWAS Catalog loci flagged as suspicious.
   - DENTIST: COJO false-positive rate (FPR) of 28% with ancestry-mismatched LD, cut to about 3% after QC. Allele flips create PIP = 1 false credible sets (SuSiE-RSS).
   - For colocalization specifically, the only quantification found is that fastENLOC type-I error stayed controlled while power halved under severe mismatch (Hukku 2021 preprint).
   - No peer-reviewed study quantifies coloc-SuSiE errors with reference vs in-sample LD. This is a gap we can fill.

9. **Benchmarks in the literature are mostly self-benchmarks by method authors, mostly in-sample LD, and almost never simulate sample overlap or split by LD region.**
   - No published coloc benchmark groups train/test by LD block or gene.
   - No paper uses conditionally independent signals as explicit negative controls for coloc. GTEx 2020 used "LD-matched eGenes with low colocalization probability" as controls, and PoPS used independent signals as positive labels.
   - Our planned design is therefore novel. It must be done carefully (see Section 7.5).

10. **Runtime is reported ad hoc.** SharePro reports per-locus seconds: coloc 0.1 s, SharePro 4.3 s, coloc-SuSiE 14 s, PWCoCo 38 s, eCAVIAR 228 s. HyPrColoc runs 100 traits in about 1 s. Nobody reports runtime on a Pareto front with accuracy under matched hardware.

---

## 1. The colocalization gap ("missing regulation")

### 1.1 Quantitative coloc rates reported

| Study | Data | Threshold / method | Reported rate |
|---|---|---|---|
| Chun 2017, Nat Genet | 272 autoimmune loci (ImmunoChip, 7 diseases) × LCL/CD4/CD14 eQTL (n≈250) | JLIM | "shared effects in only ∼25% of the loci examined" |
| Giambartolomei 2014, PLoS Genet | Liver eQTL (n=966) × lipids | coloc PP4>75% | 26/38 previously reported eQTL–lipid links supported; 14 new |
| GTEx 2020, Science | 5,385 GWAS loci (1,167 LD blocks), 87 traits × 49 tissues | enloc | "43% and 23% were colocalized with a cis-eQTL and cis-sQTL"; per trait median 21% (as cited in Mostafavi 2023) |
| GTEx 2020 / Kim-Hellmuth 2020, Science | Cell-type interaction eQTLs | coloc | 1,370 (10.3%) ieQTLs and 89 (3.7%) isQTLs coloc with ≥1 trait; for 34% of these only the ieQTL colocalizes; ~32% of neutrophil-ieQTL coloc loci not found with bulk eQTL |
| Barbeira 2021, Genome Biol | Same GTEx v8 / 87 traits | enloc rcp>0.5 | 3,477 genes (15%) colocalize; of 3,899 loci with a PrediXcan association, 55% had an eQTL coloc; per typical trait ~20% of loci have a colocalized, significant gene (11% for splicing) |
| Mountjoy 2021, Nat Genet (Open Targets) | 70,364 loci × 92 QTL datasets | coloc H4>0.8 | 49.4% no coloc gene, 25.5% exactly one, 25.2% >1 |
| Mu 2021, Genome Biol | 72 GWAS × 18 immune cell types e/sQTL | coloc | e/sQTLs coloc with 40.4% of immune-trait loci (eQTL alone 26.6% average; sQTL adds 13.8% average); 27.4–50.2% for PP4 cut-offs 0.9–0.5; "60% of GWAS loci remain unexplained" |
| Connally 2022, eLife | 220 coding-variant gene–trait pairs | coloc / JLIM / eCAVIAR / TWAS / annotation | 8% (18/220) of genes by colocalization, 2% TWAS (relevant tissues), 4% annotation+distance |
| Yao 2020, Nat Genet | 42 traits × 48 GTEx tissues | MESC | "only 11 ± 2% of heritability was mediated by assayed gene expression levels" |
| Soskic 2022, Nat Genet | 13 immune diseases × CD4+ T activation sc-eQTL | coloc PP4>0.8 | 471 colocs at 247 loci; 60% of coloc genes detected only upon activation |
| Reales 2026, PLoS Genet | >1.3M Open Targets coloc tests; 14 IMD GWAS × 12 immune eQTL studies | coloc, FDR-calibrated PP4 | 50% of peaks (Open Targets), 34% (IMD); 48% of IMD colocs only in sorted or single-cell data; >50% of colocs in one cell type only; 47% / 37% of colocalizing peaks with multiple genes |
| Rosen 2026, AJHG | Adipose eQTL n=2,256 vs n=420 subset | coloc | Colocalizing GWAS signals "approximately doubles" with the larger eQTL study |
| Sun 2023, Nature (UKB-PPP) | cis-pQTL vs GTEx eQTL | coloc-SuSiE PP.H4≥0.8 | 65% (1,220/1,889) pQTL–eQTL pairs share a causal variant |
| Zhang 2022, Nat Genet (ARIC) | cis-pQTL vs cis-eQTL | coloc PP.H4≥0.8 | 73.9% of sentinel cis-pQTLs are cis-eQTLs; 49.4% of proteins' cis-pQTLs colocalize with cis-eQTLs |
| Ferkingstad 2021, Nat Genet | 35,559 Icelanders, 18,084 pQTLs | LD (not coloc) | 12% of 45,334 GWAS Catalog leads in high LD with a pQTL |

**Caveat.** These rates are not directly comparable. They differ by method (enloc vs coloc), threshold (rcp, PP4 0.8, FDR-calibrated), denominator (loci vs peaks vs genes vs per trait), number of QTL datasets, and trait class. More QTL datasets mechanically increase the per-locus rate, and also the multi-gene, likely co-regulation, rate.

### 1.2 Main explanations and their evidence

**1. Power of QTL studies (the strongest single factor).**
- *Hukku 2021 (AJHG):*
  - "colocalizations of molecular QTLs and causal complex trait associations are widespread. However, only a small proportion can be confidently identified … due to a lack of power."
  - In the bioRxiv version, marginal power was 44% (GWAS) and 62% (eQTL), and only about 10.7% of expected colocalizations were rejected-to-expected in GTEx v8 × 4,091 traits [numbers from PREPRINT v1].
- *Mostafavi 2023:* both assays are "generally low-powered and far from saturation, likely discovering around 10–20% of all causal variants".
- *Rosen 2026:* eQTLs found only at larger N are further from the TSS and in LoF-intolerant genes, i.e. more GWAS-like. Absence of coloc in underpowered studies should not be over-interpreted.
- *Reales 2026:* coloc probability rises with lead-variant MAF, consistent with power limits for lower-frequency variants.

**2. Systematic differences / selection (Mostafavi 2023).**
- 43% of eQTLs vs 22% of GWAS hits lie within 10 kb of a TSS.
- GWAS genes are enriched for pLI > 0.9 (26% vs 21% for controls); eQTL genes are depleted (12% vs 18%).
- "eQTLs with large effects on constrained genes are purged by selection".
- Implication: more eQTL N alone will close the gap only slowly.

**3. Context specificity (cell type, state, development).** See Umans 2021 (review).
- *Alasoo 2018:* only 10 of 22 coloc eQTLs were detected in naive macrophages; each stimulated state added about 30%.
- *Kim-Hellmuth 2017:* 3–18% of eQTLs are response eQTLs.
- *Soskic 2022:* 60% activation-only.
- *Nathan 2022:* 33% of T-cell eQTLs are state-dependent, enriched in autoimmune coloc (psoriasis OR 2.70, MS OR 2.06) but not height (OR 0.87).
- *Reales 2026:* 48% of IMD coloc only with sorted or single-cell data.
- *Wang 2025 (Cell Genomics):* joint sc+bulk modelling colocalized 29.9–32.2% more loci.
- *Counter-evidence:* Natri 2024 found fewer IPF colocs with lung sc-eQTL than with bulk (4–5 vs 7). Mu 2021 found that cell-type-specific eQTLs "contribute very few new colocalizations". Context helps, but small sc-eQTL N offsets the gain.

**4. Other molecular layers.**
- *sQTLs:*
  - Li 2016: splicing QTLs contribute "roughly on a par" with expression QTLs.
  - GTEx: 23% of loci colocalize with sQTL; eQTL and sQTL credible sets overlap only 12%.
  - Mu 2021: sQTLs add 13.8%.
  - Garrido-Martín 2021: sQTL–GWAS OR 2.08.
- *caQTLs:* 4.5–22.6% more colocalized signals than eQTL alone. Combining context, distal effects, multi-causal models and priming raises overlap "from ~20% to ~60%" (Xue 2025 [PREPRINT]).
- *pQTLs:*
  - 65% of pQTL–eQTL pairs share a causal variant (UKB-PPP).
  - No peer-reviewed 2024–2026 head-to-head of the GWAS-locus coloc rate for pQTL vs eQTL was found.

**5. Method conservatism and LD.**
- *Barbeira 2021*, height vs height (GIANT n=253k vs UKB n=337k): 66.2% of credible sets had colocalization probability < 0.01, so "colocalization probability estimates are highly conservative and may miss many causal genes". The authors attribute this to LD/imputation differences, not power.
- *Wallace, Robins & Johnson 2025 [PREPRINT]:* variable per-SNP sample size or imputation quality "propagates to … coloc, yielding confident but incorrect inference" (false H3). This worsens with larger N.
- *Single-causal-variant coloc* under allelic heterogeneity has roughly half the power of fastENLOC (Hukku 2021 preprint). Coloc misattributes tissue specificity under allelic heterogeneity (Arvanitis 2022, CAFEH).
- *Abell 2022:* 17.7% of eQTLs have more than one MPRA-active allele in tight LD, so "one causal variant per signal" is itself an approximation.

**6. Co-regulation and gene-level ambiguity.**
- Multi-gene colocalization in 37–47% of peaks (Reales 2026).
- Tambets 2024 used 793 INTERVAL cis-pQTL proteins as a gene-level gold standard: coloc.susie recall was 46.3% and precision 45.1%, worse than nearest gene (71.9% / 76.9%).
- Liu 2019: 92.7% of common variants nominally associate with some nearby gene's expression.
- **Implication for us:** variant-level "shared causal variant" and gene-level "causal gene" are different estimands. We should benchmark the variant-level question.

**7. Gene prioritization tools (Open Targets L2G; Mountjoy 2021).**
- Gold standard: 445 positives in 4 classes (curated, drug targets, experimental, observational including coloc).
- Average precision 0.65 vs nearest gene 0.37.
- Coloc features alone reach AP 0.36. Removing coloc changes little (3 fewer true positives); coloc mainly *down-weights* wrong genes.
- Cross-validation was split by chromosome (outer 5 folds), a leakage control worth copying.

---

## 2. Statistical power of colocalization

- **Sample size and effect size** (Giambartolomei 2014): with N = 2,000 for trait 2 (eQTL arm N = 966, PVE 10%), the causal variant "needs to explain close to 2% of the variance … (lower percentile for PP4 > 80%)". Untyped causal variants lower PP4 without raising PP3, i.e. a power loss, not a false H3.
- **eCAVIAR (Hormozdiari 2016):** with GWAS N = 5,000 and eQTL heritability 1%, at least about 2,000 eQTL samples were needed.
- **Priors** (Wallace 2020):
  - Defaults are p1 = p2 = 1e-4 and p12 = 1e-5. "p12 = 10^−5 may be overly liberal, with data simulated under H3 having posterior support for H4, particularly for smaller samples … p12 = 5 × 10^−6 may be a more generally robust choice." The Discussion's "5×10^−5" is a typo.
  - Prior(H3) ≈ n(n−1)p1p2 grows with the square of the number of SNPs, so region definition affects H3 vs H4.
  - Recommends prior-sensitivity plots: P(H4) > 0.5 may hold only "if prior beliefs are that H4 is at least as likely as H3".
- **"H3 vs H4 when one trait is weak."** If trait 2 is weak, the posterior mass goes to H1 (trait 1 only). PP4 can still exceed PP3 when trait 2's weak peak happens to sit near trait 1's lead. This is why liberal p12 inflates false H4 at small N (Wallace 2020). H3 needs two *separately* well-supported peaks; with mismatched power the data rarely support it. Zuber 2022 says colocalization "needs strong associations for both traits".
- **Calibration:**
  - *Hukku 2021 (preprint Table 2):* at nominal 5% FDR, coloc with default prior had realized FDR 0.258 (power 0.175); coloc with the true prior 0.045 (power 0.095); fastENLOC with estimated prior 0.034 (power 0.186). Lesson: estimate the prior enrichment from the data.
  - *Reales 2026:* if PP(H4) is well calibrated, expected FDR among calls is mean(1 − PP4). They chose PP4 ≥ 0.86 for 5% FDR, an operational calibration check we can reuse on simulations.
  - *mvSuSiE (Zou 2026):* CAFEH at "study PIP" 0.95 had FDR 0.13, vs 0.0065 for mvSuSiE at lfsr 0.05. Multi-trait methods can be badly miscalibrated.
  - *SharePro (Zhang 2024):* in non-colocalized simulations, coloc, coloc-SuSiE, eCAVIAR and SharePro stayed below 0.2. PWCoCo exceeded 0.8 in 1.2% of non-coloc simulations. A large p12 increases FPR.
  - *Wang F 2022 (AJHG):* coloc FPR > 0.08 when the locus contains other null genes ("mixed null").
- **H3 vs H4 identifiability vs LD** (Chun 2017, JLIM): AUC drops with r² between the distinct causal variants (e.g. AUC 0.883 at r² 0.7–0.8), and FPR rises sharply above r² 0.8.
- **Over-calling H3 (false rejection)** comes from per-SNP N heterogeneity and imputation quality (Wallace 2025 [PREPRINT]), LD differences (Barbeira height self-coloc), and allele mis-alignment.
- **Under-calling H3:** liberal p12, small N, high-r² distinct variants.

---

## 3. Sample overlap

**Theory.** For a SNP with no effect on either trait, with N_s shared samples and phenotypic correlation ρ_p:

Corr(z1, z2) = ρ_z = N_s·ρ_p / √(N1·N2)

- Cross-trait LDSC: overlap affects only the intercept, which estimates ρ_z (Bulik-Sullivan 2015).
- For case-control traits the formula changes; see PLACO (Ray & Chatterjee 2020).
- **Typical magnitude:** for an eQTL nested in a GWAS biobank (e.g. a UKB-based QTL vs a UKB GWAS), N_s = N_QTL, so ρ_z = ρ_p·√(N_QTL/N_GWAS). This is small unless the phenotypes are strongly correlated or N is similar. For GWAS–GWAS overlap (both from UKB), ρ_z ≈ ρ_p, which can be large.

**What correlated noise does to coloc.**
- Coloc multiplies the two per-trait likelihoods, assuming independence.
- Under overlap, noise at the trait-1 peak SNP also pushes trait 2's z at the *same* SNP (and its LD neighbours) in the same direction when ρ_z > 0. This biases toward H4 when trait 2 is weak: the "apparent sharing" concern.
- When both traits are strong, the noise term is negligible next to the signal.
- So the risk concentrates exactly in the weak-trait regime the project cares about.

**Evidence found.**
- *HyPrColoc (Foley 2021, supplement):* "ignoring all correlation … even when there is complete sample overlap … gives reasonable results." Adjusting only the likelihood without the prior gave "significantly smaller" PP and split clusters. [Numbers in supplement, UNVERIFIED]
- *Mitchelmore 2020 (NAR):* "for a quantitative trait the results appear robust to correlated errors". This is an incidental check inside an epromoter paper; numbers are in a supplementary figure. [UNVERIFIED]
- *coloc GitHub (Wallace; issues #123, #137; not peer reviewed):* "theoretically, coloc requires independent samples … once sample size is at least modest (a few hundred), we found this had limited practical impact".
- *ColocBoost [PREPRINT]:* power and FDR stayed comparable without modelling residual correlation; an adjustment is given in its supplementary note.
- *Wang F 2022 (AJHG):* overlap "could lead to an increased false positive rate of colocalization depending on the degree of overlap". Their SS2 test stays controlled with 50–100% of eQTL samples overlapping; coloc itself was not evaluated. SMR/SMR-multi robustness to overlap is "unknown".
- *Fortune 2015:* coloc extension for shared controls, requires genotypes.
- **Bottom line:** no peer-reviewed main-text quantification of coloc.abf / coloc-SuSiE error vs overlap fraction exists.

**Correction strategies in adjacent fields.**
- *MR:* bias from participant overlap is linear in the overlap fraction. With F = 10, the relative bias is 10% at full overlap, 5% at 50% and 3% at 30% (Burgess 2016).
- *Cross-trait LDSC intercept → MTAG:* MTAG uses "bivariate LD score regression to account for (possibly unknown) sample overlap", with intercepts as the off-diagonal of Σ.
- *Null-SNP z correlation:*
  - PLACO decorrelates via R^−1/2 Z; inflation remains when more than half the subjects are shared and traits are strongly correlated.
  - OPERA estimates ρ from SNPs with p(xQTL) > 0.01 and models Σe in an MVN likelihood.
  - mvSuSiE estimates the residual correlation V from the weakest signals.
  - gwas-pw takes the expected correlation as input; Pickrell notes this is conservative because it removes some true shared signal.
- **Recommended for us:**
  - Estimate ρ_z per trait pair from null SNPs genome-wide (|z| < 2 in both traits, outside loci) or from the LDSC intercept.
  - Model the joint likelihood with Cov(z1, z2) = ρ_z·R (bivariate MVN).
  - Evaluate with overlap ∈ {0, 0.25, 0.5, 1} × ρ_p ∈ {0, 0.2, 0.5}, in individual-level simulations.

---

## 4. LD-reference mismatch

| Paper | Setting | Key quantitative result |
|---|---|---|
| Benner 2017 AJHG | FINEMAP with reference LD (Finland, UKB) | Reference of 1,000 is adequate for GWAS N ≤ 10,000; panel size must scale with GWAS N; avoid small panels like 1000G |
| Weissbrod 2020 Nat Genet (PolyFun) | 19 mismatched-LD experiments in UKB | Use in-sample LD, or a reference ≥ 10% of the target N; reduce the maximum number of causal SNPs under mismatch |
| Zou 2022 PLoS Genet (SuSiE-RSS) | In-sample LD (N=50k) vs 500/1,000 reference | All methods degraded; 95% credible sets missed target coverage, worse with the smaller panel. One allele flip gave a false CS with PIP = 1.00. FINEMAP needs λ-regularization; SuSiE-RSS and DAP-G less so. FINEMAP over-estimated the number of causal SNPs in 17% |
| Kanai 2022 Cell Genom (SLALOM) | Meta-analysis, heterogeneous arrays/imputation | FDR "as high as 37%" (expected ≤2%); mismatched imputation panels up to 17% vs 1%. SLALOM AUROC 0.74–0.80. 28% of GWAS Catalog loci and 67% of GBMI loci flagged suspicious |
| Chen 2021 Nat Commun (DENTIST) | COJO/FINEMAP with HRS/ARIC LD vs UKB | COJO FPR 28.4–28.7% → 2.7–3.5% after DENTIST (rare causal; in-sample 0.1–0.2%). FINEMAP correct number of causal variants: 78/50/45% → 87/79/63% (in-sample 91/87/68%) |
| Yang 2023 Nat Genet (CARMA) | Meta-analysis with external LD | Higher power, lower FDR and higher credible-set coverage than SuSiE/FINEMAP (abstract; numbers not extracted) |
| Zhang 2024 bioRxiv (RSparsePro) [PREPRINT] | Varying LD mismatch | Higher power and coverage than SuSiE |
| Hukku 2021 (preprint) | fastENLOC; FIN eQTL vs CEU/GBR/TSI/FIN/YRI GWAS | Type I error controlled in all; power halves for YRI GWAS × FIN eQTL |
| King 2021 BMC Bioinf (POEMColoc) | Ancestry-mismatched reference | "modest" effect; lowers recall more than precision (conservative) |
| Barbeira 2021 | Height GIANT vs UKB | 66% of credible sets had colocalization probability < 0.01 (LD / data heterogeneity causing false non-coloc) |
| SharePro 2024 / ColocBoost 2025 [PREPRINT] | Reference LD | SharePro: "severe LD mismatch can lead to convergence issues"; simulations used in-sample UKB LD. ColocBoost claims "more resilient to LD mismatches … compared to mvSuSiE"; recommends SLALOM-style QC |
| Chen G 2026 Brief Bioinform | Multi-ancestry fine-mapping benchmark | Methods "generally underpowered" under GWAS–reference discrepancies |

**Mechanisms relevant to coloc.**
1. Mismatch creates spurious secondary signals in fine-mapping. This produces false "distinct variant" calls (false H3) in coloc-SuSiE, and can also produce spurious shared credible sets.
2. Allele flips create PIP ≈ 1 artefacts.
3. Missing or poorly imputed SNPs and variable per-SNP N distort the z-score profile (Wallace 2025 [PREPRINT]).
4. Coloc.abf with a single causal variant and no LD is immune to reference mismatch, but is mis-specified under allelic heterogeneity. Wallace 2020's "masking" was designed to avoid errors from small deviations between reference and in-sample LD.

**Gap.** There is no peer-reviewed quantification of coloc-SuSiE PP3/PP4 errors with reference vs in-sample LD.

---

## 5. How colocalization methods have been evaluated

### 5.1 Simulation designs in the literature

| Method (paper) | Genotypes / LD | Causal variants | Effect sizes / N | Overlap? | Mismatch? | Metrics |
|---|---|---|---|---|---|---|
| coloc (Giambartolomei 2014) | 966 liver samples | 1–2 | eQTL PVE 10%; trait N, PVE varied | No | No | Fraction with PP3/PP4 > 0.9 |
| coloc-SuSiE (Wallace 2021) | 1000G EUR, lddetect blocks, 1,000–3,000 SNPs | 1–2 per trait | simGWAS 10k/10k; log OR ~ N(0, 0.2²); kept if min p < 1e-6 | No | No (in-sample LD) | Mean posterior per hypothesis; truth labelled via r² > 0.5 to top SNP |
| eCAVIAR (Hormozdiari 2016) | HAPGEN2 1000G EUR | 1–3 | NCP for 50/80% power; eQTL N 500–3,500 | No | No | ROC, precision, accuracy |
| ENLOC (Wen 2017) | 1000G EUR; GTEx-derived eQTL truth | multiple | tuned to height z distribution; α1 0–5 | No | No | ROC (regional coloc); oracle ceilings |
| fastENLOC (Hukku 2021) | GTEx genotypes (~400) [details from PREPRINT] | multiple | α1 = 4 | No | Yes (ancestry) | Realized FDR, power at 5% FDR |
| JLIM (Chun 2017) | HapGen2 CEU; 97 loci | 1 per trait | 20k/20k OR 1.1; eQTL N 250 | No | No | AUC by r² bin (0–0.4 … 0.8–0.9) |
| HyPrColoc (Foley 2021) | 1000G EUR at CHD loci | 1 per cluster | N 5k–20k; PVE 0.5–2%; 2–100 traits | Yes (supplement) | No | Accuracy, TPR, FPR; runtime |
| CAFEH (Arvanitis 2022) | GTEx 838 genotypes; 1000G EUR/AFR | 1–3 (up to 5) | PVE 0.01–0.2 | No | Partly (genotype source) | Power, FDR, tissue recovery |
| SharePro (Zhang 2024) | UKB, 5 × 1-Mb loci | K_C shared + K_S specific | h² 0.01 / 0.05; N 25k / 1k (non-overlapping) | No | No (in-sample) | P(coloc) > 0.8 power, < 0.2; runtime |
| ColocBoost (Cao 2025) [PREPRINT] | ROSMAP WGS 1,162; 1,287 genes, one per TAD | ≤5 per trait | 5% PVE each; 2–50 traits | Residual correlation tested | Claimed | Strict power/FDR on 95% coloc sets, AUPRC, purity, null type-I error |
| mvSuSiE (Zou 2026) | UKB imputed | 1–5 | Matched to UKB z distribution | Residual V modelled | No (in-sample) | Power–FDR, credible-set coverage, runtime |
| INTACT (Okamoto 2023) | GTEx 500 genotypes | multiple | Mediated PVE 0.05 | No | No | Realized FDR / power at 5% FDR |
| Multi-ancestry coloc (Shen 2026) | 1000G EUR/AFR/EAS; simGWAS, GWASBrewer | 1–2 (r² 0.06–0.88) | OR for 80% power | No | Ancestry by design | Credible-set size, coverage, coloc probabilities |
| POEMColoc (King 2021) | UKB × GTEx (real) | — | — | — | Yes (ancestry) | AUCPR vs full-data coloc (reference method as truth) |

### 5.2 Real-data validation designs used

- **Positive controls:**
  - Drug-target cis-pQTL → biomarker (SharePro: LPL/TG, RSPO3/eBMD).
  - Mendelian/OMIM, rare-variant or coding-variant genes (Barbeira 2021; Connally 2022; PoPS: 1,348 non-coding credible sets near an independent coding variant).
  - CRISPRi enhancer–gene pairs (ABC: 5,755 pairs; ColocBoost).
  - Open Targets gold standards (445 loci).
- **Replication / concordance:**
  - pQTL–eQTL sharing (65% in UKB-PPP).
  - Cross-tissue and cross-dataset replication (eQTL Catalogue: 20.4% of loci found in only one dataset).
  - GTEx primary vs secondary eQTL concordance of mediated effects, using LD-matched low-coloc eGenes as controls.
- **Calibration:**
  - Realized vs expected FDR = mean(1 − PP4) (Reales 2026; Hukku 2021).
  - For fine-mapping: replication failure rate (SuSiE-inf: PIP > 0.9 at 100k but < 0.1 at 366k) and fraction of nonsynonymous variants recovered (SLALOM).
- **Enrichment:** S-LDSC heritability enrichment of colocalized variants (ColocBoost 10.7×; GTEx 11–14× for fine-mapped QTLs).
- **Independent benchmarks:** none from 2023–2026 found outside method authors. Zhang 2026 (GPB, MR-target benchmark: SharePro 79.6% vs coloc 40.3% support) comes from the SharePro group.

### 5.3 Documented pitfalls

1. **In-sample LD optimism.** Most simulations generate and analyse with the same LD (stated explicitly for mvSuSiE, SharePro, coloc-SuSiE). Real-data miscalibration (Kanai; Cui) was missed by such simulations.
2. **Reference method as truth.** E.g. POEMColoc treats coloc P(H4) > 0.9 as truth. This is circular when the new method is coloc-like.
3. **Circular gold standards.**
   - The L2G gold standard includes loci "inferred from observational functional data (e.g., colocalizing molecular QTLs)".
   - OMIM and coding-gene sets are biased toward large-effect, constrained genes, which is exactly where eQTLs are depleted (Mostafavi). This penalizes all eQTL-based methods equally but distorts absolute rates.
4. **Leakage.**
   - L2G (chromosome-split CV), PoPS (leave-one-chromosome-out) and PolyFun (odd/even chromosome cross-fitting "to prevent winner's curse") control it.
   - No coloc benchmark controls LD-block or gene leakage.
   - The same GWAS loci recur across tissues and traits, and the same genes across QTL datasets.
5. **Tuning on test data.** Priors (p12) and PP4 thresholds are often chosen on the evaluation data; Hukku showed prior choice dominates FDR.
6. **Ignoring multiple causal variants and co-regulation.** Single-variant truth labels (Abell 2022: 17.7% of eQTLs multi-allelic in tight LD). Multi-gene coloc in 37–47% of peaks.
7. **Selection on observed significance.** Filtering on min p < threshold (coloc-SuSiE used p < 1e-6) conditions on the noise. This is fine if applied identically across classes and methods, but labels should come from the generative truth.
8. **Power mismatch between classes.** If negatives (distinct) are on average weaker than positives (shared), a method can "win" by tracking signal strength. Classes must be power-matched.
9. **Upstream processing affects coloc.** The RNA-seq quantification pipeline changes coloc and TWAS replicability (Head 2026, Nat Genet). Harmonize processing, or treat it as a nuisance factor.

---

## 6. Implications for the project's framing

- **Evidence supports the three-class framing.** Literature treats "low PP4, low PP3" as uninformative (Giambartolomei 2014; Rosen 2026; Hukku 2021). Hence:
  - Positive class S = shared causal variant.
  - Negative class D = distinct causal variants *with both detectable*.
  - Abstention class W = at least one trait too weak.
- **The real decision is a 2×2:** {both traits powered?} × {same variant?}. Coloc's H0/H1/H2 collapse into W. A good method should spread probability across W when power is lacking, rather than defaulting to H4 (liberal prior) or H3 (LD artefacts).
- **The two failure modes a better method must fix:**
  1. *False H4 under weak trait 2 plus a liberal prior or sample overlap.*
  2. *False H3 under LD mismatch, imputation or per-SNP-N heterogeneity, and allelic heterogeneity.*

  Each needs a dedicated stress test.

---

## 7. Proposed evaluation protocol

### 7.1 Estimand and classes

Per (locus, trait-pair, signal-pair), using the generative truth in simulations:

| Class | Definition | Expected method output |
|---|---|---|
| **S** (shared) | Trait 1 and trait 2 each have a causal variant at the same SNP, and both are "powered" (7.2) | Call shared |
| **D** (distinct, both powered) | Each trait has ≥1 powered causal variant, none shared; stratified by r²(c1, c2) in bins [0, 0.1), [0.1, 0.3), [0.3, 0.5), [0.5, 0.8), [0.8, 0.95), ≥0.95 | Call distinct |
| **W** (weak) | At least one trait has no powered causal variant (classic H0/H1/H2, including sub-threshold true sharing) | Abstain ("insufficient") |
| **G** (grey zone) | Power for a trait between 20% and 80% | Report separately; excluded from primary metric |
| **M** (mixed) | Locus with multiple signals per trait, some shared and some distinct | Evaluated per signal pair (signal-level S/D) and per locus ("any shared") |

**Unidentifiable region.** D with r² ≥ 0.95 is near-unidentifiable at realistic N (JLIM). Exclude it from the primary metric; report it as a curve.

### 7.2 Operational definition of "both had enough power"

Use two definitions and report both. Primary labels come from the oracle; the observed rule is what a method may use.

1. **Oracle (label) definition.** Expected non-centrality of the causal variant j for trait t:

   λ_tj = √(N_t · q²_tj / (1 − q²_tj))

   where q² is the per-variant PVE, taken marginally (i.e. including LD-tagged effects of other causal variants, λ = (Rβ)_j in z-units). The trait is **powered** if P(|z| > z_α) ≥ 0.8:
   - α = 5e-8 for GWAS-like traits. This needs λ ≳ 6.3.
   - α = 1e-5 for molecular QTL traits (≈ cis-window Bonferroni). This needs λ ≳ 5.3.

   The trait is **weak** if power < 0.2, and **grey** otherwise. Fix the thresholds before any experiment and record them in DESIGN.md.
2. **Observed (data-driven) definition.** For real data and as a method-side gate, each trait separately must have:
   - min p below the same α, **and**
   - ≥1 SuSiE credible set (coverage 0.95, purity ≥ 0.5) when fine-mapped alone.

   Report the confusion between oracle and observed power labels. Selection on observed significance is allowed only if applied identically to every method and class.
3. **Power matching.** Within each r² bin, match S and D on the distributions of min(λ1, λ2) and max(λ1, λ2), by stratified sampling or reweighting. This stops methods from exploiting signal strength. Report the λ-stratified results too.

### 7.3 Simulation design (training/validation/test generated per split)

**Genotypes and LD.**
- Real hg38 haplotypes. Inference LD comes from either the same individuals (in-sample) or an independent reference (see mismatch).
- Loci: 1–3 Mb windows centred on genes (cis ±1 Mb) or approximately independent LD blocks. Each locus is drawn only from its split's regions (7.6).

**Architecture.**
- Causal variants per trait: 1 (50%), 2 (30%), 3 (20%).
- Sharing configurations: S, D, W and M.
- Optional infinitesimal background (polygenic h²_local), following the non-sparse miscalibration finding of SuSiE-inf.

**Effect sizes.**

| Trait type | N grid | Per-variant PVE |
|---|---|---|
| GWAS-like | 20k–500k | 1e-4 to 5e-3 |
| QTL-like | 100–30k | 0.5–20% |

- Draw on a log-uniform grid so λ covers weak to strong.
- Include low-frequency variants (MAF 0.005–0.05), since coloc rates are MAF-dependent.

**Sample overlap.**
- Overlap fraction ∈ {0, 0.25, 0.5, 1.0} (of the smaller study) × ρ_p ∈ {0, 0.2, 0.5}.
- Do this with individual-level phenotype simulation on shared genotypes, so correlated noise is exact.
- For bulk speed, use the equivalent MVN: z ~ N((Rλ1, Rλ2), [[R, ρ_z R], [ρ_z R, R]]).
- Validate the MVN generator against the individual-level generator on a subset (a verification script with known answers).

**LD mismatch at inference.**

| Level | Inference LD |
|---|---|
| L0 | In-sample |
| L1 | Same-ancestry external reference, n ≈ 5k |
| L2 | Same-ancestry, n ≈ 500 (1000G-EUR-size) |
| L3 | Different / admixed ancestry |
| L4 | L1 + 0.1–1% allele flips + 5% of variants missing from the reference |
| L5 | Per-SNP N heterogeneity / imputation-quality noise (Wallace 2025 [PREPRINT] mechanism) |

**Seeds.** Use the fixed seed list from the research record. Use ≥ 2,000 loci per cell of the main grid, enough for AUPRC standard errors of about 0.01–0.02. Use block bootstrap over loci for CIs.

### 7.4 Metrics

**Primary (decided before any results):**

1. **AUPRC for S vs D** (both powered; r² < 0.95; power-matched), using each method's score s = P(shared | both have signal), e.g. PP4/(PP3 + PP4) for coloc-type methods.
   - Report under L0/no-overlap ("ideal") and pooled over the realistic stress grid (L1–L4 × overlap). The *pooled realistic* value is the headline.
2. **Runtime**, reported alongside: wall-clock per locus-pair (median and 95th percentile), including LD loading and preprocessing, on the fixed hardware (64 CPU cores / 1 RTX 4090).
   - Plus throughput (pairs per hour, whole machine) and scaling with SNPs per locus (p ∈ {500, 2k, 5k, 10k}) and number of traits.
   - Present accuracy vs runtime as a Pareto front.

**Secondary:**

1. **Three-way performance (S/D/W):**
   - Abstention quality: fraction of W loci receiving a confident S or D call (P > 0.8). Lower is better.
   - Macro-F1 or the 3×3 confusion matrix at pre-registered thresholds.
2. **Calibration:**
   - Reliability diagram and ECE of P(shared) on S∪D.
   - Realized FDR among calls with P ≥ t vs mean(1 − P) (Reales 2026 rule).
   - Brier score.
3. **Operating points:** TPR at 5% realized FDR; FPR (D called S) at PP4 > 0.8. Report FPR separately by r² bin and by overlap level. "Apparent sharing from overlap" is the slope of FPR vs ρ_z.
4. **Robustness deltas:** ΔAUPRC and ΔFPR from L0 to each L-level and overlap level.
5. **Variant-level:**
   - Coverage of the 95% "shared credible set" (contains the true shared variant).
   - Median set size and purity.
6. **Real-data metrics** (7.5): natural-negative FPR, positive-control recall, cross-dataset consistency.

**Baselines (same inputs and splits):**
- coloc.abf with p12 = 1e-5 and 5e-6.
- coloc-SuSiE.
- eCAVIAR / fastENLOC (variant-level).
- HyPrColoc.
- SharePro.
- ColocBoost (pairwise mode).
- PWCoCo, if feasible.
- A naive "lead-SNP r² > 0.8" rule, as a sanity floor.

### 7.5 Real-data validation, including natural negative controls

**A. Natural negatives from conditionally independent signals (class D, real data).**

*Construction:*
1. At a locus where trait A (GWAS or QTL) has ≥ 2 conditionally independent signals (SuSiE CSs or COJO), take an *independent* dataset B for the same or a related trait in which only one of these signals replicates (fine-mapped, CS overlapping signal k).
2. Pairs (A-signal j, B-signal k) with j ≠ k are D; pairs (j = k) are candidate S.
3. Simplest variant: the same molecular trait in two non-overlapping cohorts, e.g. two eQTL Catalogue blood studies, or UKB-PPP vs deCODE/ARIC pQTLs. The secondary signal in cohort 1, conditioned on the primary, vs the primary signal in cohort 2 is a "distinct, both powered" negative. The primary-vs-primary pair is a positive.

*Requirements (pitfalls and guards):*

| Requirement | Why |
|---|---|
| Residual LD between independent signals: require lead r² < 0.1 (in-sample and reference) and stratify by r²; report bins up to 0.5 separately | COJO tolerates collinearity, and "independent" signals at r² 0.2–0.5 are common |
| Conditioning artefacts: perform conditioning with in-sample LD when possible; otherwise flag with SLALOM/DENTIST-style QC and exclude flagged loci | Conditioning with reference LD can create or remove signals (DENTIST; Zou 2022) |
| A single signal may hide ≥ 2 causal variants (Abell 2022), and two "independent" signals could share one causal haplotype | Treat D labels as *noisy*. Report results with and without loci where the two CSs overlap or have any variant pair with r² > 0.5 |
| Power asymmetry: match on |z| of the leads | Secondary signals are weaker than primaries |
| Same-trait, same-sample pairs (signal j vs signal k within one dataset) are fully overlapping; use them only as an explicit overlap stress test, not as the main negative set | Fully overlapping samples |
| Use only datasets whose sample overlap is known (documented in DATA.md) | Label leakage via shared samples: if cohort B is a subset of A, "replication" is not independent |
| Do not define S/D labels with the method under test (e.g. coloc-SuSiE CS overlap) | Circularity. Use a pre-registered, method-agnostic rule: conditional analysis plus lead-SNP identity / r² > 0.8 with a different fine-mapper, and require the label to be stable across two fine-mappers |

**B. Positive controls.**
- Same molecular trait across independent cohorts (primary signals).
- cis-pQTL vs cis-eQTL of the same gene. This is a *partial* positive: only ~49–65% truly share. Use it for relative comparisons, not absolute recall.
- Drug-target cis-pQTL → biomarker pairs (e.g. LPL–TG).
- GWAS vs QTL at loci with an independent fine-mapped coding variant in the same gene (PoPS-style). Use this only at gene level and with caution about co-regulation.

**C. Consistency checks.** Without labels:
- Agreement of calls across replicate QTL datasets.
- Fraction of multi-gene colocalizations.
- S-LDSC enrichment of colocalized variants. This is supporting evidence only: it is circular if priors use annotations.

### 7.6 Data splitting (LD region and gene) and leakage control

1. **Units.** Define approximately independent LD blocks on hg38: lddetect / Berisa–Pickrell blocks lifted to hg38, or recomputed on the hg38 reference. Build a graph whose nodes are blocks; connect blocks spanned by any gene's cis-window (TSS ± 1 Mb) or by any locus window used. Connected components are the split units, so each gene, and all its loci and traits, falls in exactly one split.
2. **Assignment.** Hash components into train / validation / test, about 60/20/20, balancing the number of genes and loci. Optionally enforce whole-chromosome hold-outs for the test set (as in L2G and PoPS), the strictest option. Record the split file with checksum. The same split applies to *all* traits, tissues and simulation draws: a simulated locus inherits the split of the region its genotypes came from.
3. **Usage.**
   - Training: learned priors, any ML components, hyperparameters.
   - Validation: model selection, PP thresholds, p12 / prior choices for baselines too (fair tuning).
   - Test: touched once per final candidate, with results committed.
   - Genome-wide nuisance parameters (e.g. ρ_z from null SNPs, enrichment priors à la fastENLOC) must be estimated from training regions only, or shown to be insensitive.
4. **Trait-level hold-out.** Also report performance on GWAS traits and QTL datasets not seen in training, to detect over-fitting to specific trait architectures.
5. **Leakage audits.** Write a verification script asserting:
   - (a) no gene, locus or LD block appears in two splits;
   - (b) no test locus shares a variant with a training locus;
   - (c) the real-data positive and negative pairs respect splits;
   - (d) no hyperparameter was fitted using test files (log file access).

### 7.7 Cheapest kill tests (screening before full runs)

Apply each candidate to a small, frozen screening set from the *validation* split: about 500 loci per cell for the {S, D(r² < 0.5), D(0.5–0.8), W} × {L0, L2, L4} × {overlap 0, 1 with ρ_p = 0.5} grid. Kill a method if any of the following holds:
- AUPRC(S vs D) at L0 is not better than coloc-SuSiE by more than the bootstrap SE;
- false confident calls on W exceed coloc.abf with p12 = 5e-6;
- FPR at L4 or under full overlap exceeds 2× the best baseline;
- runtime per pair exceeds 10× coloc-SuSiE without a matching accuracy gain.

### 7.8 Reporting

For each method, report:
- primary AUPRC (ideal and realistic);
- runtime (median and 95th percentile);
- three-way confusion matrix;
- calibration plot;
- FPR vs r² curve;
- FPR vs ρ_z curve;
- ΔAUPRC by LD-mismatch level;
- real-data natural-negative FPR and positive-control recall.

All results carry 95% block-bootstrap CIs (resampling split units). Metrics, classes and thresholds are frozen in DESIGN.md before the test split is used, and are never changed afterwards.

---

## 8. Verified reference list

All entries were verified via Crossref and/or PubMed on 2026-10-26: title, first author, year, venue, DOI. [PREPRINT] marks work that is not peer reviewed.

### Colocalization gap / regulatory biology
1. Chun S, et al. Limited statistical evidence for shared genetic effects of eQTLs and autoimmune-disease-associated loci in three major immune-cell types. *Nat Genet* 2017. doi:10.1038/ng.3795
2. Connally NJ, et al. The missing link between genetic association and regulatory function. *eLife* 2022. doi:10.7554/eLife.74970
3. Mostafavi H, et al. Systematic differences in discovery of genetic effects on gene expression and complex traits. *Nat Genet* 2023. doi:10.1038/s41588-023-01529-1
4. Umans BD, Battle A, Gilad Y. Where are the disease-associated eQTLs? *Trends Genet* 2021 (online 2020). doi:10.1016/j.tig.2020.08.009
5. Yao DW, et al. Quantifying genetic effects on disease mediated by assayed gene expression levels. *Nat Genet* 2020. doi:10.1038/s41588-020-0625-2
6. GTEx Consortium. The GTEx Consortium atlas of genetic regulatory effects across human tissues. *Science* 2020. doi:10.1126/science.aaz1776
7. Kim-Hellmuth S, et al. Cell type–specific genetic regulation of gene expression across human tissues. *Science* 2020. doi:10.1126/science.aaz8528
8. Barbeira AN, et al. Exploiting the GTEx resources to decipher the mechanisms at GWAS loci. *Genome Biol* 2021. doi:10.1186/s13059-020-02252-4
9. Reales G, et al. Design and interpretation of eQTL-GWAS colocalisation studies: Lessons from a large-scale evaluation. *PLoS Genet* 2026. doi:10.1371/journal.pgen.1012141
10. Rosen JD, et al. Higher eQTL power reveals signals that boost GWAS colocalization. *Am J Hum Genet* 2026. doi:10.1016/j.ajhg.2026.02.009
11. Mu Z, et al. The impact of cell type and context-dependent regulatory variants on human immune traits. *Genome Biol* 2021. doi:10.1186/s13059-021-02334-x
12. Soskic B, et al. Immune disease risk variants regulate gene expression dynamics during CD4+ T cell activation. *Nat Genet* 2022. doi:10.1038/s41588-022-01066-3
13. Nathan A, et al. Single-cell eQTL models reveal dynamic T cell state dependence of disease loci. *Nature* 2022. doi:10.1038/s41586-022-04713-1
14. Natri HM, et al. Cell-type-specific and disease-associated expression quantitative trait loci in the human lung. *Nat Genet* 2024. doi:10.1038/s41588-024-01702-0
15. Yazar S, et al. Single-cell eQTL mapping identifies cell type–specific genetic control of autoimmune disease. *Science* 2022. doi:10.1126/science.abf3041
16. Perez RK, et al. Single-cell RNA-seq reveals cell type–specific molecular and genetic associations to lupus. *Science* 2022. doi:10.1126/science.abf1970
17. Alasoo K, et al. Shared genetic effects on chromatin and gene expression indicate a role for enhancer priming in immune response. *Nat Genet* 2018. doi:10.1038/s41588-018-0046-7
18. Kim-Hellmuth S, et al. Genetic regulatory effects modified by immune activation contribute to autoimmune disease associations. *Nat Commun* 2017. doi:10.1038/s41467-017-00366-1
19. Strober BJ, et al. Dynamic genetic regulation of gene expression during cellular differentiation. *Science* 2019. doi:10.1126/science.aaw0040
20. Wang L, et al. An atlas of single-cell eQTLs dissects autoimmune disease genes and identifies novel drug classes for treatment. *Cell Genomics* 2025. doi:10.1016/j.xgen.2025.100820
21. Xue A, et al. Genetic regulation of cell type–specific chromatin accessibility shapes immune function and disease risk. *medRxiv* 2025. doi:10.1101/2025.08.27.25334533 [PREPRINT]
22. Mitchel J, et al. A single-cell genetic colocalization test improves power and resolves disease-mediating cell types. *bioRxiv* 2025. doi:10.1101/2025.10.10.681685 [PREPRINT]
23. Kaptijn D, et al. Federated single-cell QTL meta-analysis reveals novel disease mechanisms. *bioRxiv* 2026. doi:10.64898/2026.01.20.700519 [PREPRINT]
24. Li YI, et al. RNA splicing is a primary link between genetic variation and disease. *Science* 2016. doi:10.1126/science.aad9417
25. Garrido-Martín D, et al. Identification and analysis of splicing quantitative trait loci across multiple tissues in the human genome. *Nat Commun* 2021. doi:10.1038/s41467-020-20578-2
26. Sun BB, et al. Plasma proteomic associations with genetics and health in the UK Biobank. *Nature* 2023. doi:10.1038/s41586-023-06592-6
27. Ferkingstad E, et al. Large-scale integration of the plasma proteome with genetics and disease. *Nat Genet* 2021. doi:10.1038/s41588-021-00978-w
28. Zhang J, et al. Plasma proteome analyses in individuals of European and African ancestry identify cis-pQTLs and models for proteome-wide association studies. *Nat Genet* 2022. doi:10.1038/s41588-022-01051-w
29. Tambets R, et al. Extensive co-regulation of neighboring genes complicates the use of eQTLs in target gene prioritization. *HGG Adv* 2024. doi:10.1016/j.xhgg.2024.100348
30. Kerimov N, et al. A compendium of uniformly processed human gene expression and splicing quantitative trait loci. *Nat Genet* 2021. doi:10.1038/s41588-021-00924-w
31. Kerimov N, et al. eQTL Catalogue 2023: New datasets, X chromosome QTLs, and improved detection and visualisation of transcript-level QTLs. *PLoS Genet* 2023. doi:10.1371/journal.pgen.1010932
32. Võsa U, et al. Large-scale cis- and trans-eQTL analyses identify thousands of genetic loci and polygenic scores that regulate blood gene expression. *Nat Genet* 2021. doi:10.1038/s41588-021-00913-z
33. Mountjoy E, et al. An open approach to systematically prioritize causal variants and genes at all published human GWAS trait-associated loci. *Nat Genet* 2021. doi:10.1038/s41588-021-00945-5
34. Ghoussaini M, et al. Open Targets Genetics: systematic identification of trait-associated genes using large-scale genetics and functional genomics. *Nucleic Acids Res* 2021 (online 2020). doi:10.1093/nar/gkaa840
35. Liu B, et al. Abundant associations with gene expression complicate GWAS follow-up. *Nat Genet* 2019. doi:10.1038/s41588-019-0404-0
36. Abell NS, et al. Multiple causal variants underlie genetic associations in humans. *Science* 2022. doi:10.1126/science.abj5117
37. Head ST, et al. Quantification method affects replicability of eQTL analysis, colocalization and TWAS. *Nat Genet* 2026. doi:10.1038/s41588-026-02827-0

### Colocalization and multi-trait methods (power, priors, evaluation)
38. Giambartolomei C, et al. Bayesian test for colocalisation between pairs of genetic association studies using summary statistics. *PLoS Genet* 2014. doi:10.1371/journal.pgen.1004383
39. Wallace C. Eliciting priors and relaxing the single causal variant assumption in colocalisation analyses. *PLoS Genet* 2020. doi:10.1371/journal.pgen.1008720
40. Wallace C. A more accurate method for colocalisation analysis allowing for multiple causal variants. *PLoS Genet* 2021. doi:10.1371/journal.pgen.1009440
41. Wallace C, Robins C, Johnson T. Variable information across SNPs in GWAS data can cause false rejections of colocalisation which can be resolved by proportional colocalisation tests. *bioRxiv* 2025. doi:10.1101/2025.09.08.674910 [PREPRINT]
42. Hukku A, et al. Probabilistic colocalization of genetic variants from complex and molecular traits: promise and limitations. *Am J Hum Genet* 2021. doi:10.1016/j.ajhg.2020.11.012. Preprint (source of Table 2 numbers): doi:10.1101/2020.07.01.182097
43. Hukku A, et al. Analyzing and reconciling colocalization and transcriptome-wide association studies from the perspective of inferential reproducibility. *Am J Hum Genet* 2022. doi:10.1016/j.ajhg.2022.04.005
44. Hormozdiari F, et al. Colocalization of GWAS and eQTL signals detects target genes. *Am J Hum Genet* 2016. doi:10.1016/j.ajhg.2016.10.003
45. Wen X, et al. Integrating molecular QTL data into genome-wide genetic association analysis: Probabilistic assessment of enrichment and colocalization. *PLoS Genet* 2017. doi:10.1371/journal.pgen.1006646
46. Foley CN, et al. A fast and efficient colocalization algorithm for identifying shared genetic risk factors across multiple traits. *Nat Commun* 2021. doi:10.1038/s41467-020-20885-8
47. Giambartolomei C, et al. A Bayesian framework for multiple trait colocalization from summary association statistics. *Bioinformatics* 2018. doi:10.1093/bioinformatics/bty147
48. Zhang W, et al. SharePro: an accurate and efficient genetic colocalization method accounting for multiple causal signals. *Bioinformatics* 2024. doi:10.1093/bioinformatics/btae295
49. Cao X, et al. Integrative multi-omics QTL colocalization maps regulatory architecture in aging human brain (ColocBoost). *medRxiv* 2025. doi:10.1101/2025.04.17.25326042 [PREPRINT]
50. Arvanitis M, et al. Redefining tissue specificity of genetic regulation of gene expression in the presence of allelic heterogeneity (CAFEH). *Am J Hum Genet* 2022. doi:10.1016/j.ajhg.2022.01.002
51. King EA, et al. Estimating colocalization probability from limited summary statistics (POEMColoc). *BMC Bioinformatics* 2021. doi:10.1186/s12859-021-04170-z
52. Robinson JW, et al. An efficient and robust tool for colocalisation: Pair-wise Conditional and Colocalisation (PWCoCo). *bioRxiv* 2022. doi:10.1101/2022.08.08.503158 [PREPRINT]
53. Zhu Z, et al. Integration of summary data from GWAS and eQTL studies predicts complex trait gene targets (SMR/HEIDI). *Nat Genet* 2016. doi:10.1038/ng.3538
54. Deng Y, Pan W. A powerful and versatile colocalization test. *PLoS Comput Biol* 2020. doi:10.1371/journal.pcbi.1007778
55. Wang F, et al. A flexible summary statistics-based colocalization method with application to the mucin cystic fibrosis lung disease modifier locus. *Am J Hum Genet* 2022. doi:10.1016/j.ajhg.2021.12.012
56. Okamoto J, et al. Probabilistic integration of transcriptome-wide association studies and colocalization analysis identifies key molecular pathways of complex traits (INTACT). *Am J Hum Genet* 2023. doi:10.1016/j.ajhg.2022.12.002
57. Zuber V, et al. Combining evidence from Mendelian randomization and colocalization: Review and comparison of approaches. *Am J Hum Genet* 2022. doi:10.1016/j.ajhg.2022.04.001
58. Zhang W, et al. Benchmarking Bayesian colocalization methods in validating Mendelian randomization-identified targets. *Genomics Proteomics Bioinformatics* 2026. doi:10.1093/gpbjnl/qzag087
59. Shen C, et al. Multi-ancestry colocalization approaches. *PLoS Genet* 2026. doi:10.1371/journal.pgen.1012221
60. Zou Y, et al. Fast and flexible joint fine-mapping of multiple traits via the Sum of Single Effects model (mvSuSiE). *Nat Genet* 2026 (online 2025). doi:10.1038/s41588-025-02486-7
61. Gloudemans MJ, et al. Integration of genetic colocalizations with physiological and pharmacological perturbations identifies cardiometabolic disease genes. *Genome Med* 2022. doi:10.1186/s13073-022-01036-8

### Sample overlap
62. Bulik-Sullivan B, et al. An atlas of genetic correlations across human diseases and traits. *Nat Genet* 2015. doi:10.1038/ng.3406
63. Burgess S, Davies NM, Thompson SG. Bias due to participant overlap in two-sample Mendelian randomization. *Genet Epidemiol* 2016. doi:10.1002/gepi.21998
64. Turley P, et al. Multi-trait analysis of genome-wide association summary statistics using MTAG. *Nat Genet* 2018. doi:10.1038/s41588-017-0009-4
65. Ray D, Chatterjee N. A powerful method for pleiotropic analysis under composite null hypothesis identifies novel shared loci between Type 2 Diabetes and Prostate Cancer (PLACO). *PLoS Genet* 2020. doi:10.1371/journal.pgen.1009218
66. Pickrell JK, et al. Detection and interpretation of shared genetic influences on 42 human traits (gwas-pw). *Nat Genet* 2016. doi:10.1038/ng.3570
67. Wu Y, et al. Joint analysis of GWAS and multi-omics QTL summary statistics reveals a large fraction of GWAS signals shared with molecular phenotypes (OPERA). *Cell Genomics* 2023. doi:10.1016/j.xgen.2023.100344
68. Fortune MD, et al. Statistical colocalization of genetic risk variants for related autoimmune diseases in the context of common controls. *Nat Genet* 2015. doi:10.1038/ng.3330
69. Mitchelmore J, et al. Functional effects of variation in transcription factor binding highlight long-range gene regulation by epromoters. *Nucleic Acids Res* 2020. doi:10.1093/nar/gkaa123. Contains an incidental coloc correlated-error check; the numbers are in a supplement and UNVERIFIED.
70. coloc GitHub issues #123, #137 (C. Wallace replies, 2023). https://github.com/chr1swallace/coloc/issues. Not peer reviewed; existence verified via GitHub API by a sub-agent.

### LD-reference mismatch and fine-mapping calibration
71. Benner C, et al. Prospects of fine-mapping trait-associated genomic regions by using summary statistics from genome-wide association studies. *Am J Hum Genet* 2017. doi:10.1016/j.ajhg.2017.08.012
72. Kanai M, et al. Meta-analysis fine-mapping is often miscalibrated at single-variant resolution (SLALOM). *Cell Genomics* 2022. doi:10.1016/j.xgen.2022.100210
73. Zou Y, et al. Fine-mapping from summary data with the "Sum of Single Effects" model (SuSiE-RSS). *PLoS Genet* 2022. doi:10.1371/journal.pgen.1010299
74. Chen W, et al. Improved analyses of GWAS summary statistics by reducing data heterogeneity and errors (DENTIST). *Nat Commun* 2021. doi:10.1038/s41467-021-27438-7
75. Yang Z, et al. CARMA is a new Bayesian model for fine-mapping in genome-wide association meta-analyses. *Nat Genet* 2023. doi:10.1038/s41588-023-01392-0
76. Zhang W, et al. Robust fine-mapping in the presence of linkage disequilibrium mismatch (RSparsePro). *bioRxiv* 2024. doi:10.1101/2024.10.29.620968 [PREPRINT]
77. Weissbrod O, et al. Functionally informed fine-mapping and polygenic localization of complex trait heritability (PolyFun). *Nat Genet* 2020. doi:10.1038/s41588-020-00735-5
78. Cui R, et al. Improving fine-mapping by modeling infinitesimal effects (SuSiE-inf). *Nat Genet* 2024 (online 2023). doi:10.1038/s41588-023-01597-3
79. Chen G, et al. Systematic benchmarking of multi-ancestry fine-mapping under variant heterogeneity and LD mismatch. *Brief Bioinform* 2026. doi:10.1093/bib/bbag574

### Gold standards, LD blocks, gene prioritization
80. Weeks EM, et al. Leveraging polygenic enrichments of gene features to predict genes underlying complex traits and diseases (PoPS). *Nat Genet* 2023. doi:10.1038/s41588-023-01443-6
81. Nasser J, et al. Genome-wide enhancer maps link risk variants to disease genes (ABC). *Nature* 2021. doi:10.1038/s41586-021-03446-x
82. Forgetta V, et al. An effector index to predict target genes at GWAS loci. *Hum Genet* 2022. doi:10.1007/s00439-022-02434-z
83. Berisa T, Pickrell JK. Approximately independent linkage disequilibrium blocks in human populations. *Bioinformatics* 2016 (online 2015). doi:10.1093/bioinformatics/btv546

### Known verification gaps (do not cite these numbers as fact)
- Giambartolomei 2014: the two p12 values in "PP4 drops from 91% to 49%" are images in PMC.
- Hukku 2021: Table 2 FDR and power numbers and the 10.7% ratio come from bioRxiv v1. Confirm against the AJHG version before quoting in a paper.
- HyPrColoc and Mitchelmore sample-overlap results, and gwas-pw's estimator details: supplements not read.
- Yazar 2022 (OneK1K): coloc fraction not extracted.
- CARMA and Benner: detailed numbers beyond the abstract not extracted.
- eQTLGen phase 2 coloc statistics, and Jang 2026 (SingleBrain, Nat Genet, doi:10.1038/s41588-026-02541-x; Crossref-verified only): coloc numbers not extracted.
