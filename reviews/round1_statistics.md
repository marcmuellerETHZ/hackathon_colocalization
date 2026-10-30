# Review round 1 — statistics (Bayesian model comparison, fine-mapping, RSS likelihood)

Reviewer: independent statistics agent, 2026-10-26. Scope: DESIGN.md v1 §1, §2.2–2.4, §5, §6, §8 (math, estimand, metrics). Nothing in the project was edited except this file.

**Verdict: sound core, but not ready to freeze. 8 must-change items.** The RSS generator, the overlap covariance, the power thresholds and the exact-BF identity are correct. However, the oracle classes do not form a partition. The ranking score has a gate-induced discontinuity that sets s = 1 on non-signals. Family A's locus aggregation is inflated by null effects. The NPE kill-test compares against a posterior with the wrong prior. "Distinct vs too weak", the brief's core requirement, has no primary metric. Calibration and CI claims are not fair or not reliable as specified.

---

## A. What I checked (with evidence)

| Item | Result |
|---|---|
| Power thresholds (§1) | **Correct.** Two-sided z_α is 5.451 (5e-8) and 4.417 (1e-5). Power 0.8 needs \|μ\| ≥ 6.293 / 5.259; power 0.2 needs 4.610 / 3.576 (scipy, `.venv`). |
| Overlap ρ = N_s r_p/√(N1N2) and Cov(e1,e2) = ρR (§2.2) | **Correct** for quantitative traits. Factor sampling e2 = Gᵀ(ρu1 + √(1−ρ²)u2) gives Var R and cross-covariance ρR. |
| Exact BF identity (§5 B) | **Correct in principle, but the written formula has the wrong Kronecker order** (Error E1). Numeric check: p = 8, ρ = 0.4, R̃ = 0.95R + 0.05I, u = {1,3}, a correlated shared prior. Full-p logBF is 14.18250. The z_u-only logBF with (I₂⊗R̃_uu) is 14.18250 (identical). As written, (R̃_uu⊗I₂) gives 14.18785 (wrong). Under the "inflated residual variance" variant (mean Rb, covariance R + ηI), the full-p logBF is 13.550 and the z_u-only logBF is 13.342, so the identity **breaks** (Error E2). |
| coloc-SuSiE adapter coherence (§6.3) | P_S + P_D ≤ 1 holds: if every pair has PP3 > PP4, then max PP4 ≤ 1 − min PP3. |
| Class partition (§1 + §2.4 mixture, no LD, μ = λ) | 100k draws. Share of each simulated locus type that lands in G: 33% of S-type, 31% of D-type, 55% of M-type. Share that fits **no class**: 3.2% of S-type and 6% of M-type (Error E3). |
| ρ̂ from null SNPs with \|z\| < 2 in both traits (§5 notation) | Truncation attenuates the estimate. 2M bivariate-normal draws: true ρ 0.1 → 0.077, 0.3 → 0.239, 0.5 → 0.418 (about 17–23% low). |
| Number of split units (§3) | Recomputed from `data/raw/annot/pyrho_EUR_LD_blocks.bed` (1,336 blocks) and GENCODE v39 protein-coding TSS. Components: **75** at ±1 Mb (DESIGN says 73), **196** at ±500 kb, **377** at ±250 kb. 20% of 75 is about 15 test clusters. |
| Cluster-bootstrap coverage with few clusters | Percentile bootstrap of a mean, skewed per-cluster effects, 2,000 reps. Nominal 95% coverage was 0.910 at G = 15, 0.934 at G = 40 and 0.934 at G = 250. |
| p12 and AUPRC | For coloc.abf, s = PP4/(PP3+PP4) has odds = p12·ΣBF1jBF2j / (p1p2·Σ_{j≠k}BF1jBF2k). Changing p12 multiplies every locus's odds by the same constant, so **AUPRC is exactly invariant to p12**. Tuning p12 only affects calls and ECE. The prior effect SD (W) does change the ranking. |

---

## B. Errors found

- **E1. Kronecker order in the Family B formula (§5 B).** Σ_null = V⊗R̃_uu implies trait-major stacking z_u = [z1,u; z2,u]. The alternative must then be Σ_alt = V⊗R̃_uu + (I₂⊗R̃_uu) D (I₂⊗R̃_uu)ᵀ, not (R̃_uu⊗I). The mixed form is a silent bug: it is numerically close and wrong (table above).
- **E2. Scope of exactness.** The z_u-only BF equals the full RSS BF only when the *same* PD matrix appears in both the mean and the covariance: z ~ N((I⊗R̃)b, V⊗R̃). It fails for the alternative in §5 A, "η modelled as inflated residual variance" (N(Rb, R+ηI)), and for any reference-size-aware variance (layer E). In those cases B must evaluate the likelihood on the full screened block z_M, which is still cheap with rank-|u| updates of a pre-factored V⊗Σ_MM. It is also "exact" only within (screen M) × (≤ 2 per trait). M loci have up to 3 per trait (§2.4), so B is misspecified on about 15% of loci; this must be reported.
- **E3. Oracle classes are neither exhaustive nor sensibly prioritised (§1).**
  - Gap: no class covers "a shared causal variant exists but is sub-powered in one trait, *and* each trait has a powered distinct variant". It is not D ("none is shared"), not W (both traits are powered) and not S.
  - Over-broad grey zone: G = "any causal variant has power in [0.2, 0.8)" removes a third of S- and D-type loci and half of M-type loci from the primary metric. The cause is usually an irrelevant secondary variant, e.g. an S locus with a strong shared variant plus a grey trait-specific one.
- **E4. Ranking score discontinuity (§1, §6.3).** s = P_S/(P_S+P_D), with P_D *gated* to 0, gives s = 1 for any P_S > 0. Affected cases:
  - SharePro/ColocBoost when one trait lacks a CS: P_S = 0.01 gives s = 1;
  - coloc-SuSiE whenever *any* pair has PP4 ≥ PP3;
  - Family A whenever the power gate fails.

  This puts non-signals at the top of the ranking, creates massive ties and distorts AUPRC for every method with a gate.
- **E5. Family A locus aggregation (§5 A).** P_S = 1 − Π_l(1 − q_l) over L = 10 effects counts null effects. In SuSiE, an effect whose prior variance goes to 0 (or that is diffuse) keeps its *prior* state probabilities. With π12/(π1+π2+π12) = 0.1 and 8 null effects, P_S on a locus with no sharing is 1 − 0.9⁸ = 0.57. Separately, "P_D" is defined as a rule, not a probability, so P_W = 1 − P_S − P_D is undefined.
- **E6. NPE kill-test C1 is mis-specified (§5 C).** NPE trained on simulator labels converges to the posterior under the *simulator's* prior: log-uniform |λ|, random sign, the §2.4 architecture mixture, and oracle classes that depend on μ. ExactColoc uses a Gaussian effect prior and coloc-style configuration priors over hypotheses H0–H4, not over S/D/W. A perfect NPE would fail \|ΔP\| < 0.03.
- **E7. Notation in the generator (§2.2).** "Factors G_std/√N_syn" combined with "u ~ N(0, I/N_syn)" scales the covariance by 1/N_syn twice, giving Cov R/N_syn. Use G_std with u ~ N(0, I/N_syn), or G_std/√N_syn with u ~ N(0, I).
- **E8. Minor.** The component count is 75, not 73, with my reading of the blocks and GTF (check the gene filter). The realism note "λ up to 25 for trait 2" implies a PVE of λ²/(N₂+λ²): 0.56 at N₂ = 500 and 0.86 at N₂ = 100. That is incompatible with the eQTL N range in open question 2 unless λ_max is tied to N₂.

---

## C. Changes

### Must

1. **Make the oracle classes a precedence-ordered partition (§1 table).** Proposed text:
   - (i) **U** if the D condition holds but identifiability is low (see change 10).
   - (ii) **S** if ≥ 1 causal variant is shared *and* powered in both traits.
   - (iii) **D** if each trait has ≥ 1 powered causal variant and no shared causal variant has power ≥ 0.2 in both traits.
   - (iv) **W** if ≥ 1 trait has no causal variant with power ≥ 0.2.
   - (v) **G** for everything else. This includes "powered distinct plus grey or weak shared" and a grey class-determining variant.

   Grey status should apply only to the class-determining variants (the shared variant for S, the powered lead per trait for D), not to "any causal variant". Report the class shares of the generated mixture in DESIGN.md before freezing. My no-LD check suggests the current rule spends about 35% of S/D/M simulation on G.

2. **Decouple the ranking score from the gate (§1 "ranking score", §6.3).** Define s from *ungated* quantities and apply power gates only to the three-way call.
   - coloc.abf: s = PP4/(PP3+PP4).
   - coloc-SuSiE: s = max_k PP4_k/(PP3_k+PP4_k) over CS pairs (fallback: coloc.abf's s).
   - SharePro/ColocBoost: s = P_S (their locus probability).
   - HyPrColoc: s = P_S.
   - Family A/B/C: s = P(S | z, S∪D) from the model.

   Only when nothing is defined is s = 0.5, and the share of ties is reported per method. Use step-wise average precision (ties treated as one threshold).

3. **Family A outputs as coherent probabilities (§5 A "Outputs").**
   - Add an explicit null state, s_l ∈ {0, 1-only, 2-only, both}, or aggregate only over effects with an estimated prior variance > 0 and purity ≥ 0.5.
   - Define, under the mean-field posterior:
     - P_S = 1 − Π_l(1 − q_l^{both,pow});
     - P_D = Π_l(1 − q_l^{both,pow}) · [1 − Π_l(1 − q_l^{1,pow})] · [1 − Π_l(1 − q_l^{2,pow})];
     - P_W = 1 − P_S − P_D.

     Here q_l^{s,pow} is the posterior probability that effect l is in state s *and* its own non-centrality exceeds the oracle power threshold (|b_lt| ≥ τ_t). The latter is computable from the Gaussian SER posterior. This mirrors the oracle estimand directly.
   - Estimate the U_both scale and the π's by empirical Bayes. With a fixed small trait-2 variance in U_both, "both" is barely penalised against "1-only" when z2 ≈ 0, and confident S calls on W follow.

4. **Fix the NPE kill-test reference (§5 B, §5 C kill-test C1).**
   - Give ExactColoc a "simulator-prior" mode. Approximate the log-uniform |λ| by a scale mixture of Gaussians (e.g. 8 grid components), so each BF stays a sum of closed-form Gaussian BFs. Use the §2.4 configuration prior, restricted to ≤ 2 per trait. Output posterior probabilities of the *oracle classes* by integrating the Gaussian effect posterior per configuration over the power thresholds.
   - Run C1 on loci simulated from exactly that prior, at L0 with ≤ 2 causal variants per trait.
   - This also gives a **Bayes-optimal ceiling** for AUPRC at L0, which tells us how much headroom "more power" can have. Report it in M4.

5. **Add a D-vs-W co-primary endpoint, and make C3 non-gameable (§0 C3, §8.1).** The brief's third requirement is "distinct vs too weak", yet the primary metric is S vs D only. Add to §8.1:
   - (a) AUROC of P_D on oracle D ∪ W;
   - (b) at a per-method threshold set on validation so that the confident-call rate on W equals 5% (or equals that of the most conservative baseline), the recall of S and of D on the test set.

   Rewrite C3 as (b). "Fewer confident calls on W" alone is trivially achieved by abstaining.

6. **Make calibration and tuning comparisons fair (§8.2, §12.4).**
   - ECE and Brier must be computed on the *unweighted* simulation distribution: the class prior the simulator implies, which the NPE is trained on. The power-matching reweighting changes P(S | z) by a μ-dependent factor, so no method targets calibration under the reweighted population.
   - Report ECE twice: raw, and after a common one-parameter logit-intercept recalibration fitted on validation for *every* method, baselines included. The fixed priors of Bayesian baselines (p12, π) are otherwise penalised against a simulator-trained NPE for prior mismatch, not for modelling.
   - Pre-register the validation tuning objective for every method, e.g. maximise validation AUPRC, then set the call threshold by the matched-W rule in change 5.
   - Note in §4/§12 that p12 does not affect AUPRC (exactly, for coloc.abf). Tune the prior effect SD instead.

7. **Generator realism: give trait 2 its own in-sample LD, and give baselines meaningful N and SE (§2.2, §2.4, §4).**
   - (a) For small-N trait 2 (an eQTL with N₂ ~ 100–1,000), the z-profile follows *its own* sample LD R₂, not population LD. The per-SNP deviation (R̂₂ − R)λ has SD ≈ λ(1−r²)/√N₂: about 0.9 at λ = 20, N₂ = 500, r = 0. That is comparable to the sampling noise itself, and it is the between-study LD heterogeneity behind false H3 (Barbeira's height self-coloc).
     - Generate z₂ at individual level from an N₂-row subsample G₂ of the synthetic genotypes: z₂ = G₂ᵀ(G₂β₂ + ε₂)/√N₂, standardised. This costs about N₂·p flops per locus.
     - Keep the factor generator for the large-N trait 1. For ρ > 0 with nested samples, use the shared rows' ε.
     - Without this, L0 and L1 are optimistically coherent for joint single-R models (A, B, C) relative to per-trait pipelines.
   - (b) Assign N₁, N₂ per locus from realistic grids and draw λ₂ ≤ c√N₂ (E8). Pass baselines beta = z·SE with SE = 1/√(2N·f(1−f)) (sdY = 1), or z with N and MAF. coloc.abf's prior W = 0.15² is on the β scale; fed raw z with varbeta = 1, it implies a prior SD of 0.15 in z units and all BFs are about 1. A scale-free simulation without N cripples coloc.abf and coloc-SuSiE and would invalidate C1.

8. **Increase the number of independent split units and use a CI method valid for few clusters (§3, §8.3).**
   - With about 15 test components, percentile cluster-bootstrap CIs undercover (0.91 at nominal 0.95 in my check), and "CIs excluding zero" (C1) is weak evidence.
   - Proposed: assign each *block* (1,336) to a split and each gene by its TSS block. Clip every window (simulated and real) at split-run boundaries, exactly as §3 already does for real data. Bootstrap over blocks (about 267 test units).
     - This keeps every gene in one split, and LD across block boundaries is low by construction.
     - Fallback: components at ±500 kb (196 units, about 39 test units).
   - Use a paired cluster bootstrap of the method difference with BCa or studentised intervals. Also report a sign test over units.
   - Pre-register whether the power-matching weights are recomputed inside each bootstrap replicate (recommended: yes).
   - For the screening kill rules (§6.2) the question is a ranking on a fixed set, so a paired locus-level bootstrap SE is appropriate. Say so.

### Should

9. **Fix E1 and E2 in §5 B and state the exactness conditions.**
   - Use the same R̃ in the mean and the covariance (Zou 2022 form), with an η floor (≥ 1e-3), because L2 references (n = 200 < p) make R_uu singular for near-collinear pairs.
   - For the inflated-residual or E-layer variants, use the full-z_M likelihood.
   - B1's coloc.abf identity test must enumerate single-SNP configurations over all p SNPs (not top-M), with z-scale prior variance W/SE_j² and coloc's p1/p2/p12 at full p. The H3 prior mass scales with p(p−1), so a top-M restriction changes the PPs.
   - With ρ = 0, H3 configurations factorise (BF1(j)·BF2(k)), so the full-p sum is closed form. With ρ ≠ 0 they do not, so state how off-screen mass is handled.
   - The "≤ 3 extended mode" is not enumerable: 166,751 configurations per trait at M = 100 gives about 2.8e10 joint configurations. Use pruned or stochastic search there, or drop the mode.

10. **Define U and the D strata by an identifiability index, not by r² alone (§1).**
    - For truth D (c1 in trait 1, c2 in trait 2), the best single-variant S explanation misfits by Mahalanobis distance d = min(|λ1|, |λ2|)·√(1 − r²). This holds because z ~ N(λR_{·j}, R) against the best λ' at k leaves residual λ²(1 − r²).
    - At r² = 0.95, d is 1.41 for λ = 6.3 but 8.9 for λ = 40. A fixed r² ≥ 0.95 cut therefore labels identifiable loci as U and leaves unidentifiable loci with r² < 0.95 in D.
    - Set U := d < 2 (or keep r² ≥ 0.95 as well). Report AUPRC as a curve over d alongside r² bins.

11. **ρ̂ estimation (§5 notation, §5 C inputs, §2.3).**
    - The truncated null-SNP correlation is biased about 20% toward 0 (table A).
    - A per-locus estimate has only tens of LD-pruned SNPs, giving SE ≈ 0.15–0.2.
    - Specify one estimator: either a truncated-bivariate-normal MLE, or the LDSC intercept from genome-wide null SNPs per trait pair.
    - In simulation, produce ρ̂ by the same estimator on a simulated genome-wide null set (with its noise). The headline must use ρ̂, not the true ρ, for every method, the NPE included.

12. **Power matching and prevalence (§2.4, §8.1).**
    - Match on trait-specific (μ1, μ2) (5×5 bins), not on (min μ, max μ). The two traits have different thresholds and ranges, and min/max mixes them.
    - Fix the weighted S prevalence in S∪D at 0.5 and state it.
    - Report AUROC alongside AUPRC, since it is prevalence-free.
    - The shuffled-label control must compare against the *weighted* prevalence.

13. **Pre-register pooled vs macro-averaged AUPRC (§8.1).**
    - Pooling over L1–L5 × ρ penalises methods whose score scale shifts with the condition. That is arguably desirable (one threshold for all conditions), but it differs from the mean of per-cell AUPRC.
    - Declare one as the headline and report both.
    - State the ρ weights (including the sign).

14. **Formalise the power-aware layer (§5 "Cross-cutting D").** "The minimum effect consistent with its observed signal" is undefined across traits, because trait 2's effect units are unrelated to trait 1's. Replace it with an **exclusion probability**:
    - E₂ = P(|λ_{2,j}| < τ₂ for all j ∈ CS₁ | z) under the fitted model, and symmetrically E₁.
    - Call D only if both observed gates pass and E₁, E₂ ≥ 0.95. Otherwise the locus is "insufficient".
    - This is exactly the oracle D-vs-W/G distinction, and it is computable from SER posteriors (A) or configurations (B).

15. **The LD-mismatch layer (§5 E).**
    - η "estimated by ML under the null" is inflated by true signals, which look like mismatch. Estimate it jointly inside IBSS, or on residual z after removing fitted effects.
    - Give f(R, b) explicitly. To first order, the error from an n_ref reference is Var((R̂b)_k) ≈ Σ_j b_j² (1 − r_jk²)²/n_ref, plus covariance terms between causal variants.
    - The error grows with λ: at λ = 40 and n_ref = 200 its SD is about 2.8. A constant η therefore has the wrong functional form for strong signals.

16. **C4 needs an equivalence margin (§0).** "Does not grow with ρ" cannot be shown as a null. Pre-register this: the slope of the false-S rate vs ρ, with its 95% CI inside ±δ (e.g. δ = 0.02 per 0.1 ρ), on D and W loci separately.

17. **Kill rule (b) count noise (§6.2).** With 500 W loci and coloc.abf rates of about 0.5–1%, "2×" means comparing 3 events to 6. Require both > 2× and > +2 percentage points, with a one-sided binomial or bootstrap p < 0.05.

18. **Fix E7 (generator notation) and E8 (λ vs N₂, component count) in §2.2, §2.4 and §3.**

### Could

19. **Power labels.** Marginal μ = (Rλ)_j can mask or inflate a causal variant's own effect: an extra trait-specific variant in LD with opposite sign gives cancellation. Consider requiring both marginal |μ| ≥ τ and own-effect |λ_tj| ≥ τ for "powered".
20. **Specify the per-SNP-N generator (L5).** Nested subsets mean taking the first f_j·N_syn rows of G for variant j, which gives Cov = R·min(f_i, f_j)/√(f_i f_j) to within sub-sample LD noise.
21. **Exclude the MHC** (chr6: 28.5–33.4 Mb) from simulation and the primary real-data metrics, or report it separately. One component can otherwise dominate a 15-unit test split.
22. **Define the NPE soft labels for G** (e.g. the label weight equals the product of the oracle powers). Report the NPE's calibration under at least two W prevalences, since P_W is prior-dependent.
23. **Clarify the D r² definition** as "max r² over cross-trait pairs of class-determining causal variants".

---

## D. Does the estimand separate "distinct, both powered" from "too weak"?

**In the oracle labels, mostly yes. In the evaluation, not yet.**
- The labels hinge on μ thresholds, which are correct, but leave a gap and an over-broad G (change 1).
- The primary endpoint never scores D vs W (change 5).
- The methods' P_D is a rule rather than a posterior for the same event (changes 3 and 14).

With changes 1, 3, 5 and 14, the oracle event "each trait has a powered effect and no shared effect is powered" becomes the same event that A and B report a posterior for and that the NPE is trained on. That makes the three-way claim testable.
