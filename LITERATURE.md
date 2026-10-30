# LITERATURE — synthesis

Detailed, citation-verified reviews (written by three independent agents, 2026-10-26):
- [lit/methods.md](lit/methods.md): about 35 colocalization and fine-mapping methods, their handling of overlap, LD mismatch and distinct-vs-weak, speed, and ranked opportunities (about 70 refs).
- [lit/gap_and_evaluation.md](lit/gap_and_evaluation.md): the colocalization gap, power, overlap and LD-mismatch effects, past evaluation designs, and a proposed evaluation protocol (83 refs).
- [lit/data.md](lit/data.md): open datasets on hg38 with probed URLs, sizes and licences, and a recommended data plan (about 35 refs).

This file is the PI's synthesis; numbers are quoted from those files, which hold the citations. Items marked [PREPRINT] there are not peer reviewed.

## 1. Why GWAS and molecular QTLs rarely colocalize

- **Coloc rates depend on the denominator and the threshold.** About 43% of GWAS loci colocalize with an eQTL and 23% with an sQTL in GTEx v8 (enloc); the per-trait median is about 21%. Open Targets reports about 50% at H4 > 0.8, and Chun 2017 about 25%. At gene level, coloc recovers only 8% of 220 genes with coding-variant GWAS signals (Connally 2022). Only about 11% of heritability is mediated by bulk expression (Yao 2020).
- **The main explanations:**
  - **Power.** An eQTL study of 500 people detects between <0.1% and 60% of eQTLs depending on effect size (Rosen 2026). Increasing eQTL N about 5-fold roughly doubled colocalizations.
  - **Systematic differences.** GWAS hits sit further from TSSs, near constrained genes with complex regulatory landscapes, and selection purges large-effect eQTLs at those genes (Mostafavi 2023).
  - **Context.** Many colocalizations appear only in stimulated, sorted or single-cell data.
  - **Method conservatism and miscalibration.** Height colocalized with itself gives an enloc probability below 0.01 for 66% of credible sets. Default priors (p12 = 1e-5) gave a realized FDR of 0.26 at a nominal 0.05 in one simulated setting (Hukku 2021 [PREPRINT]).
- **Consequence for this project:** "no colocalization" is mostly a statement about power. The informative contrast is shared vs. distinct when both traits are well powered. The third outcome, "insufficient", must be explicit.

## 2. Methods landscape (details: lit/methods.md §1–2)

| Family | Representatives | Multi-causal | Overlap | LD mismatch | Distinct vs weak | Speed |
|---|---|---|---|---|---|---|
| Single-variant Bayes | coloc.abf, HyPrColoc, moloc | no | HyPrColoc optional | not needed (no LD) | PP3 vs PP0-2, but misled when there are several causal variants | ~0.1 s per locus |
| Per-trait fine-map, then pair | coloc-SuSiE, eCAVIAR/CLPP, fastENLOC, gpu-coloc | yes | no | in the fine-mapping layer only | weak signals give no credible set, so the pair is not tested | SuSiE 1-5 s per locus; pairing about 2,900 pairs/s (gpu-coloc) |
| Joint multi-trait fine-map | SharePro, CAFEH, mvSuSiE, ColocBoost [PREPRINT], flashfm | yes | mvSuSiE (V), ColocBoost (claimed) | partial (RSparsePro-style) | implicit | 4-40 s per locus |
| Testing / distinct-null | JLIM, SS2, proportional tests (colocPropTest, prop-coloc) | partial | SS2 | sensitive to allele flips | the distinct-variant null is explicit | variable |
| MR/TWAS | SMR/HEIDI, FOCUS, MRLocus | — | — | — | different question | fast |

**No published method combines** multi-causal coloc, a cross-trait error correlation from sample overlap, an LD-mismatch model, and an explicit power-aware "insufficient" call.

## 3. The three stated requirements

1. **Sample overlap.**
   - Null z correlation is ρ = N_s·r_p/√(N1·N2), the LDSC cross-trait intercept, with Cov(e1, e2) = ρR.
   - It is tiny for an eQTL against a biobank GWAS (≤0.05). It can reach 0.35·r_p for a UKB pQTL against a UKB GWAS, and 0.3-0.6 for GWAS against GWAS in one biobank or for eQTLs across tissues from the same donors.
   - Modelling ρ removes spurious "shared noise" and increases power, because conditional variance shrinks to 1−ρ².
   - No peer-reviewed paper quantifies how overlap changes coloc posteriors. This is a gap we fill.
2. **LD mismatch.**
   - Zou 2022: "performance depended more on choice of LD matrix than on method". With out-of-sample panels of 500-1,000 people, 95% credible-set coverage was lost.
   - SLALOM reports FDR up to 37% from mismatch; a single allele flip produced a false credible set with PIP = 1.
   - Remedies exist only in the fine-mapping layer: λ-regularisation, kriging/DENTIST outlier detection, susieR 0.16.x `R_finite` / `R_mismatch` (GitHub/r-universe; documented in vignettes only; CRAN still has 0.14.2), RSparsePro and CARMA.
   - No paper quantifies H3/H4 error as a function of LD mismatch. Another gap.
   - Related failure: "variable information", i.e. different per-SNP N or imputation quality, causes confident false H3/H4 (Wallace 2025 [PREPRINT]).
3. **Distinct vs too weak.** coloc's 5-way posterior separates these in principle, but in practice it does not:
   - PP3 is miscalibrated when there are several causal variants;
   - SuSiE merges or misses signals ("colocalization colliders" in 2.5-7.3% of clusters, Jesse 2026);
   - pipelines often threshold PP4 only.

   JLIM *defines* "distinct" as r² < 0.8 between causal variants (a definitional threshold, not an empirical breakdown point); identifiability really depends on signal strength × √(1−r²) (see DESIGN.md §1). No method reports a predictive-power statement of the form "had the signal been shared, would trait B have detected it?".

## 4. Speed

- The pairing step is solved: gpu-coloc is about 800× faster than R coloc and ran 15.6M pairs in 1.5 h on a laptop.
- The bottleneck is now per-locus fine-mapping (1-5 s), LD I/O and LD QC. There is no published GPU-batched SuSiE-RSS.
- The engineering win is batched GPU inference over thousands of loci with padded LD tensors. An RTX 4090 holds about 1,000 loci of 2,000 SNPs each in float32.

## 5. Evaluation practice and its pitfalls (details: lit/gap_and_evaluation.md §5-7)

- Most benchmarks are run by method authors, simulate and analyse with the same LD, and include neither LD-region/gene splits nor natural negative controls.
- Simulation norms: 1-3 causal variants per locus; real-LD MVN generation of z-scores; AUROC/AUPRC on H4 vs H3; credible-set coverage.
- Real-data validation norms:
  - Positive controls: the same molecular trait in independent cohorts; cis-pQTL vs eQTL as a partial positive (only 49-65% truly share).
  - Negative controls: conditionally independent signals, with guards for residual LD, power asymmetry and circular labels.

## 6. Data (details: lit/data.md)

- **Simulation genotypes:** 1000 Genomes 30x phased GRCh38. 503 unrelated EUR samples; open access.
- **QTL source:** eQTL Catalogue r7 (hg38, CC BY 4.0) with full beta/SE, SuSiE credible sets and per-variant SuSiE log Bayes factors. GTEx's own all-pairs files sit in a pay-per-download bucket.
- **GWAS source:** FinnGen R12 (hg38, open sumstats plus SuSiE with Bayes factors), including quantitative lab traits (LDL N=369k, bilirubin, urate, ferritin) and a public in-sample LD matrix fetchable by region. The in-sample LD makes a real-data in-sample vs reference-LD comparison possible.
- **Reference sets:** Open Targets 26.09 (CC0) offers about 1.4M precomputed coloc/eCAVIAR results and gold standards.
- **Splits:** MacDonald 2022 native-hg38 EUR LD blocks (1,336) and GENCODE v39 (matches eQTL Catalogue).
- **Real-overlap arms:**
  - eQTLs across GTEx tissues from the same donors (via eQTL Catalogue);
  - GWAS vs GWAS within FinnGen;
  - UKB-PPP vs a UKB GWAS needs a Synapse account (OPEN_QUESTIONS.md).

## 7. Opportunities carried into DESIGN.md

1. **Joint bivariate fine-mapping colocalization (main family).** A SuSiE-RSS-type model in which each effect has one variant and a sharing state {1-only, 2-only, both}. It includes overlap ρ by whitening and an LD-error term, and is GPU-batched across loci.
2. **Exact enumeration** over ≤2-3 causal variants per trait among the top-K SNPs on GPU. It serves as a gold standard and a candidate method.
3. **Bold: amortised neural posterior estimation.** Trained on mismatched-LD and overlap simulations; outputs {shared, distinct-both-powered, insufficient} with millisecond inference.
4. **Cross-cutting layers:**
   - a power-aware three-way decision rule;
   - LD-mismatch QC and regularisation;
   - a per-SNP N likelihood.
5. **A benchmark contribution:** coloc error rates vs LD-reference choice and vs overlap ρ, which no paper has quantified.
