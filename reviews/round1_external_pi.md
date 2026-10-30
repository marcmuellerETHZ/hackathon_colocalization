# Review round 1 — external PI (overall strategy)

Reviewer role: senior human-genetics PI, external. Document reviewed: DESIGN.md v1 (2026-10-26), with PROMPT.md, LITERATURE.md and lit/*.md.

**Verdict: major revision, not a rethink.** The architecture is right:
- the estimand (S vs D, both powered, plus an explicit "insufficient" call);
- three families with a bold one;
- a gold-standard enumerator;
- LD-region splits and frozen metrics.

Credibility fails in four places:
1. Too few split units for the CIs that C1 promises.
2. The headline is simulation-only, from a generator that *is* Family A's and B's likelihood, with no prior-shift or misspecification arm.
3. Real-data labels are partly circular, and no real-data success criterion is set.
4. The LD-mismatch baseline is a straw man, because the installed susieR 0.14.2 lacks the current `R_finite`/`R_mismatch` remedies.

Claims C1–C5 are the right *kind* of claims. C2, C3, C4 and C5 need rewording before they can be tested honestly (item 6).

---

## Changes (priority: must / should / could)

### MUST

**1. Split units are too coarse: merge on the analysis window, not ±1 Mb (§3).**
- *Problem.* §3 merges blocks whenever a TSS ± 1 Mb window spans them, which gives 73 components. Test then holds about 15 units, and §8.3 bootstraps over components. A cluster bootstrap with about 15 clusters under-covers. C1 requires "CI excluding zero" against *every* baseline, so the project risks a null result caused by the split design alone.
- *Check.* I recomputed this with `pyrho_EUR_LD_blocks.bed` and GENCODE v39 protein-coding autosomal TSS, 19,049 genes (`.venv/bin/python`). Merging blocks spanned by TSS ± W gives:

  | W | Components | With genes | Median size | Largest |
  |---|---|---|---|---|
  | ±1 Mb | 89 | — | 25 Mb | — |
  | ±500 kb | 208 | 178 | 8 Mb | 68 Mb, 830 genes (4%) |
  | ±250 kb | 387 | — | — | — |

  My ±1 Mb count differs from the design's 73. Commit the script that produced 73. Note that block-gap positions must map to the nearest block: a naive lookup that returns None/0 merged whole chromosomes in my first attempt.
- *Change.*
  - Merge on the window actually used (±500 kb), which gives about 180 gene-bearing units and about 35–40 test units.
  - Draw the ±1 Mb scaling-study windows only where they fit inside one component, or clip them.
  - State the number of test units in §3 and require ≥ 30.
  - Keep the bootstrap over components, but compute *paired* method differences per bootstrap replicate (already partly in §8.3).

**2. Add a held-out, pre-registered misspecification / prior-shift test arm (§2.4, §8.1, §12.3).**
- *Problem.* The generator is exactly the stacked-RSS likelihood that A and B assume. At L0/L1 they are the true model, and only R-mismatch breaks this. NPE (C) learns the §2.4 class mixture and λ distribution as its prior, so training and test share one mixture (30/30/25/15). This flatters C and A/B and will not convince a human-genetics audience.
- *Change.* Add test-only arm T-shift, generated with the test seed, never seen in train or validation. It contains:
  - (a) a shifted class mixture, e.g. S:D:W = 15:15:70, closer to the real colocalization gap;
  - (b) a heavier-tailed λ distribution, trait 1 up to |λ| ≈ 100 (see item 11);
  - (c) 4–5 causal variants per trait in 20% of loci;
  - (d) infinitesimal background in 100% of loci;
  - (e) a second LD world (item 12).
- Pre-register that C1 and C3 must hold *in direction* on T-shift, with effect sizes reported. Calibration (ECE) is reported both within-distribution and under shift.

**3. Real data: make labels non-circular, evaluate every method at locus level, and set one pre-registered real-data success criterion (§2.5 R2, R3, §8).**
- *R3 problem.* As written, R3 pairs cohort A's non-replicated signal with B's signal. At locus level, though, A's summary statistics still contain the replicated signal, so the locus truth is S, not D. Only signal-level methods can be scored, which excludes coloc.abf and the other baselines from the natural-negative test.
- *R3 change.* Use a symmetric conditional construction:
  - for genes with two well-separated signals (lead r² < 0.1 in both reference panels, both |z| > 6) present in *two independent cohorts*, build COJO-style conditional summary statistics;
  - **D pair:** cohort A conditioned on signal 2 vs cohort B conditioned on signal 1;
  - **S pair:** A conditioned on signal 2 vs B conditioned on signal 2, so both carry signal 1.
  - Positives and negatives then go through the same conditioning operation, so conditioning artefacts cannot separate them. Every method sees ordinary locus-level summary statistics.
  - Conditioning error scales with √N·‖ΔR‖. At eQTL N ≈ 200–700 with 1000G reference it is small; quantify it in the generator first (condition with L2 LD and compare with the truth).
- *R2 problem.* R2 positives are selected by "catalogue SuSiE lead identity or r² > 0.8". This conditions on SuSiE agreeing and on easy, lead-concordant cases, which biases R2 towards SuSiE-based and lead-SNP methods.
- *R2 change.* Select positives only on "same gene, same tissue, independent cohorts, both cohorts pass the observed power gate". Treat the label as S-with-noise, and report recall as a lower bound.
- *Success criterion.* §2.5 calls real data "secondary" and sets none. Add: "On R2 ∪ R3 (test-split genes only), at the fixed 0.8 thresholds, the primary candidate has higher S-recall at ≤ equal D-false-S rate than coloc-SuSiE; paired component bootstrap." Without this, reviewers will treat the result as a simulation paper.

**4. Fair LD-mismatch baselines (§4).**
- *Problem.* `env/renv` has susieR **0.14.2** and coloc 5.2.3, so baseline 2 uses only λ-regularisation. susieR 2.x adds `R_finite` (a finite-reference variance term) and `R_mismatch = "eb"/"eb_mix"`. These are the current best practice (lit/methods.md l.25, l.100, l.272; susieR rss_mismatch vignette) and do almost exactly what Layer E's "reference-size-aware variance inflation" proposes.
- *Change.* Add the following, tuned on validation like the others:
  - baseline 2b: coloc on susieR 2.x with `R_finite = n_ref` and `R_mismatch = "eb"`;
  - baseline 2c: coloc-SuSiE after DENTIST/kriging QC.
- Any C1 gain at L2–L4 must be shown against 2b and 2c. Layer E's novelty claim must be stated relative to them.

**5. Exactly one primary candidate touches test for the headline claims (§3 "Usage", §12.2).**
- *Problem.* "Test touched once per frozen final candidate" allows several candidates to be evaluated with the best one reported, which is selection on test.
- *Change.* Before M7, a decision note names *one* primary candidate (plus at most one runtime-variant) for C1–C4. Other survivors are reported as secondary and labelled exploratory. Enforce this in the `FROZEN` check: the evaluation script refuses a second primary.

**6. Re-state the claims so they are testable and not trivially true (§0).**
- **C2.** Beating single-threaded R coloc-SuSiE by 10× with a GPU is engineering that anyone can do; the brief puts the novelty in speed *and* performance.
  - Define the workload: all ~19k protein-coding genes × one GWAS × one tissue at ±500 kb, end-to-end including reference-LD computation or loading.
  - Report speed vs (i) coloc-SuSiE on 64 cores, (ii) engineering baseline G, (iii) gpu-coloc.
  - Claim ≥ 10× over (i) at ≥ accuracy, and report the modelling-vs-engineering split via G.
- **C3.** "≤ the most conservative baseline" may be unattainable without giving up power: the lead-SNP rule or coloc.abf at p12 = 5e-6 may make almost no confident calls. Re-state it as a trade-off: at a confident-call rate on W ≤ that of coloc.abf (p12 = 5e-6), the method has higher S and D recall. Keep ECE ≤ best baseline.
- **C4.** "Does not grow with ρ" is an absence claim.
  - Pre-specify an equivalence margin, e.g. the 95% CI of the false-S slope over ρ ∈ [0, 0.5] within ±0.02 per 0.1 ρ.
  - It must be evaluated with **estimated** ρ̂ (item 8), never oracle ρ. Otherwise C4 is true by construction, because the generator equals the model.
- **C5.** coloc.abf uses no LD and is invariant to the LD reference; allele flips do not change it either, since it uses z². Restrict C5 to LD-using methods. Also add the explicit, possibly embarrassing, comparison of whether coloc.abf beats LD-using methods at L2–L4. Reviewers will ask.

**7. Class definitions do not partition the space, and G is too broad (§1).**
- *Gap.* An S-type locus with a shared variant that is weak in trait 2 (power < 0.2) plus a powered trait-2-specific variant fits no class:
  - not S (the shared variant is not powered in both);
  - not D ("none is shared" fails);
  - not W (both traits have a powered variant);
  - not G (no variant in the grey band).
- *Size of G.* With μ ≈ λ (ignoring LD), the §2.4 λ distributions put about 31–33% of S- and D-type loci in G, and only about 49–50% end up usable as S or D. Grey *extra* trait-specific variants that cannot change the label also exclude loci.
- *Change.*
  - Define classes on the powered causal sets P1 and P2:
    - S if P1 ∩ P2 ≠ ∅;
    - D if P1 and P2 are both non-empty and disjoint, and no variant has power ≥ 0.2 in both traits;
    - W if P1 or P2 is empty and no grey variant is present;
    - otherwise G.
  - Apply G only when a grey variant could change the label.
  - Add a unit test that every simulated locus gets exactly one label.
  - Sample λ conditional on the target class, instead of by rejection, to stop wasting compute.

**8. ρ̂ estimation is biased and noisy as specified (§5 notation, §2.4).**
- *Bias.* Correlation of a bivariate normal truncated to |z1|, |z2| < 2 is 0.077, 0.239 and 0.418 for ρ = 0.1, 0.3 and 0.5 (Monte Carlo with 2M draws). That is a 16–23% downward bias.
- *Noise.* Within one ±500 kb window there are only about 30–300 effectively independent null SNPs, so SE(ρ̂) ≈ 0.05–0.17.
- *Change.*
  - Estimate ρ̂ per *study pair* across all null loci genome-wide, with a truncated-normal correction or the LDSC intercept. This is what a real pipeline can do.
  - In simulation, give methods ρ̂ drawn with that estimator's realistic error, never the true ρ.
  - Report sensitivity to mis-specified ρ̂ (±0.1).

**9. Family A needs an explicit per-effect "null/inactive" state, and Family C's cheapest kill-test is ill-posed (§5).**
- *A.* In IBSS, unused SuSiE effects keep α summing to 1 over diffuse variants. Then P_S = 1 − Π_l(1 − α_l,both-weighted) picks up spurious sharing mass from inactive effects. Add a null state per effect, or weight each effect by its activity probability (derived from its lbf, with prior variance estimated towards 0) and a purity filter, and test it on pure-noise and W toy cases (§7.3 already has the cases).
- *C, kill-test C1.* Requiring NPE posteriors to match ExactColoc within mean |ΔP| < 0.03 fails even for a *perfect* NPE, because the two posteriors use different priors: the §2.4 mixture vs B's p1/p2/p12 Gaussian configuration prior. Replace it with either:
  - (i) a small NPE (≤ 1 GPU-hour) trained on a simulator whose prior is set equal to B's (≤ 2 causal per trait, Gaussian effects, B's configuration prior), then the |ΔP| test; or
  - (ii), the cheaper real-risk test: train at L0/L1, evaluate on L2 and on the second LD world. Kill if the AUPRC drop from L1 to L2 exceeds that of A or coloc-SuSiE by > 2 SE.

  Run (ii) first, because simulation-to-reality shift is what kills NPE.

**10. Define a minimum viable result and a critical path, and cut scope (§4, §2.5, §11).**
- *Problem.* As written, the design has:
  - 9 baselines (+2 from item 4);
  - 3 families + 3 layers + G;
  - 6 LD levels × 4 ρ × 2 signs;
  - 6 real-data arms.

  No time budget is stated (OPEN_QUESTIONS.md Q3). The risk is running out of time with everything 70% done.
- *Change.* Add §11a "Minimum viable result": C1 + C2 + C3 on simulation (L0–L4, ρ grid), plus R2/R3 + R4 on real data, against coloc.abf, coloc-SuSiE (2, 2b), SharePro and ColocBoost.
- Defer until the MVR is in hand:
  - eCAVIAR: dominated, and Open Targets already ships CLPP for real-data comparison;
  - HyPrColoc for 2 traits: close to coloc.abf, keep only if it costs nothing;
  - prop-coloc;
  - L5 and Layer F;
  - R5a;
  - R6, which stays a showcase run once at M7.
- *Critical-path order (fastest credible route):*
  1. Generator + oracle-power tests + splits.
  2. A vertical slice on 200 validation loci: coloc.abf, coloc-SuSiE, ExactColoc with the coloc.abf identity test. This yields an early C5 result and the gold standard.
  3. In parallel, the cheapest kill-tests: an A prototype in numpy/torch on CPU (≤ 1 day, before GPU engineering) and the NPE test 9(ii).
  4. Only then the full screening set and the remaining baselines.
  5. R3 conditional-statistics construction in parallel with step 4, because it has the longest data lead time.

### SHOULD

**11. Extend trait-1 λ to the GWAS regime (§2.4).**
- Lead |z| at major GWAS loci often exceeds 40. LDL at SORT1 in FinnGen R12 (N ≈ 369k) is far above that.
- LD-mismatch artefacts scale with |z|·‖ΔR‖, so a cap at 40 under-represents exactly the regime where mismatch is most harmful.
- Use log-uniform on [3, 100] for trait 1, or fix it via R1 *before* generating validation and test. R1 calibration must be completed and frozen before the screening set is generated, not merely "before any test evaluation".

**12. Answer to open question 13.1: add a cheap second LD world, not msprime (§2.1).**
- Swap the roles: build the target mosaic from pool B and the reference from a 200-person subset of pool A.
- Optionally add a FIN-only target with a non-FIN EUR reference, a realistic within-EUR mismatch that ties directly to R4.
- This probes NPE overfitting to one mosaic's LD texture at near-zero cost. Use it in T-shift (item 2) and kill-test 9(ii).

**13. Fix the Kronecker order in Family B's Σ_alt (§5 B).**
- With z stacked trait-major as [z1; z2] and noise V ⊗ R̃_uu, the mean is (I₂ ⊗ R̃_uu)b. So Σ_alt = Σ_null + (I₂ ⊗ R̃_uu) D (I₂ ⊗ R̃_uu)ᵀ.
- The written (R̃_uu ⊗ I) is the SNP-major ordering and is inconsistent with V ⊗ R̃_uu. That is a likely silent bug.
- Also state that "exact" is conditional on the top-M screen containing all causal variants. Screen recall must be a reported test.

**14. Bound Family B's compute (§5 B, §10).**
- At M = 100 with ≤ 2 causal variants per trait there are 5,051 configurations per trait, i.e. about 2.55 × 10⁷ joint configurations per locus. With ρ ≠ 0 they do not factorise.
- That is feasible in batches (≈ 6 GB for the 8×8 float32 matrices of one locus), but it is about 1 s per locus, not milliseconds.
- The ≤ 3 extended mode at M = 100 is about 2.9 × 10¹⁰ configurations, which is infeasible. Cap M (e.g. ≤ 30) for the extended mode.
- When ρ = 0 and non-shared effects are independent, use the coloc-style factorisation: per-trait BFs combined in O(K²).

**15. Real data: use FinnGen R12 LD with care (R4).**
- The FinnGen R12 matrix sets entries with r² ≤ 0.01 to 0 (lit/data.md l.56), so it is neither exactly in-sample nor PSD.
- Specify the PSD repair (eigenvalue clipping or λ) and report the agreement metric's sensitivity to it.
- R4 measures *robustness* (agreement), not accuracy. Say so in the claims.

**16. Make the kill and stop rules robust to noise (§6.2, §11).**
- "Below coloc-SuSiE by > 1 bootstrap SE" with about 15 validation components is unstable. Use the paired-difference CI, after item 1.
- The stop rule needs:
  - a definition of an "iteration";
  - a compute or calendar budget per milestone, since the time limit is unknown;
  - the rule that a 0.005 AUPRC change below the validation CI half-width counts as "no improvement".

**17. Add GWAS-vs-GWAS replication pairs as a high-z real positive/negative arm (§2.5).**
- Use the same trait in FinnGen R12 and an independent non-FinnGen GWAS (e.g. a GLGC lipid trait, lifted to hg38 by script).
- Primary signals are positives. Conditionally independent signals, built as in item 3, are negatives.
- This exercises GWAS-scale |z| and real Finnish vs non-Finnish LD mismatch, which the QTL-vs-QTL R2/R3 arms cannot.
- Priority is *should* if the liftover and harmonisation pipeline already exists for other reasons, otherwise *could*.

### COULD

**18.** Report the fraction of real GWAS loci each method calls "insufficient". This connects the method to the brief's colocalization-gap framing at no cost.

**19.** For HyPrColoc (P_S only), s = P_S, while all other methods use P_S/(P_S + P_D). Note the incomparability in the tables, or drop HyPrColoc from the primary AUPRC (see item 10).

---

## Errors found

1. **Kronecker order in Σ_alt (§5 B).** The text has (R̃_uu ⊗ I); it should be (I₂ ⊗ R̃_uu) given noise V ⊗ R̃_uu (item 13).
2. **Class partition is incomplete (§1).** Weak-shared plus powered-distinct loci are unlabelled (item 7).
3. **ρ̂ estimator is biased (§5).** The |z| < 2 truncation bias is about 16–23% towards 0 (item 8).
4. **Family C kill-test C1 is ill-posed** because the two posteriors use different priors (item 9).
5. **Family A's P_S formula** lacks effect activity, so inactive effects inflate P_S (item 9).
6. **Component count is not reproducible.** I get 89 at ±1 Mb, not 73, and the generating script is not committed (item 1).
7. **Framing of C5.** coloc.abf is LD-free, so "coloc false-H3/H4 vs LD-reference choice" applies only to coloc-SuSiE and the other LD-using methods.

Verified correct: the oracle power thresholds. For two-sided α = 5e-8, z_c = 5.451, and 80%/20% power give |μ| = 6.293/4.610. For α = 1e-5, z_c = 4.417, giving 5.259/3.576. These match §1. The RSS claim that z_{−u} | z_u does not depend on b_u (conditional mean R_{−u,u}R_uu⁻¹z_u, covariance independent of b) is also correct.

## What I checked

- Split components: Python on `data/raw/annot/pyrho_EUR_LD_blocks.bed` (1,336 blocks, median 1.8 Mb) and `gencode.v39.basic.annotation.gtf.gz` (19,049 autosomal protein-coding genes), with merge rules at ±1 Mb, ±500 kb and ±250 kb.
  - Only 9,637 of 19,049 genes have a ±500 kb window inside a single block, so pure block units without merging would lose half the genes. Merging at ±500 kb is the right compromise.
- Truncated-normal bias of ρ̂ (Monte Carlo) and the power thresholds (scipy).
- Grey-class share under the §2.4 λ distributions, with μ = λ and LD ignored, so it is approximate: about 1/3 of S and D loci are G.
- Installed versions: susieR 0.14.2 and coloc 5.2.3 in `env/renv`, so no `R_finite`/`R_mismatch`. The susieR mismatch vignette documents `R_finite` and `R_mismatch = "eb"` ([rss_mismatch vignette](https://stephenslab.r-universe.dev/susieR/doc/rss_mismatch.html); [summary-stats vignette](https://cran.nics.utk.edu/cran/web/packages/susieR/vignettes/finemapping_summary_statistics.html)).
- A quick literature search ([coloc-SuSiE preprint](https://www.biorxiv.org/content/10.1101/2021.02.23.432421v1.full.pdf)) found no published study of coloc error under LD mismatch × overlap. The C4/C5 gap is plausible, but it is not proven by one search.

Answers to the §13 open questions within my remit:
- 13.1: see item 12.
- 13.2: extend trait-1 λ (item 11) and lock the ranges via R1 before generating the screening set.
- 13.4: catalogue outputs suffice *only* with conditional statistics built from summary statistics plus reference LD. Use the symmetric construction in item 3.
