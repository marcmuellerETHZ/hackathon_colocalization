# Review round 2 — statistics

Reviewer: independent statistics agent, 2026-10-26. Scope: DESIGN.md v2 §1, §2.2–2.4, §4.3, §5, §7.1, §8. Read alongside PROMPT.md and reviews/round1_*.md. The only project file edited is this one. Throwaway scripts are in /tmp/claude-190744/-cluster-scratch-mumarc-hackathon-eqtl-gwas/0666a753-70c8-497e-b212-f9558627195a/scratchpad/r2stat/ (bf_check.py, bf_sing.py, pd_check.py, ranges.py, rho_check.py).

**Verdict: minor revision.** The core is now sound. The support-space Bayes factor (BF) is numerically exact. The class partition, the block bootstrap and the ρ̂ estimator are fine. Five must-change items remain:
- **Infeasible ρ grid.** The overlap grid cannot occur for the trait pairs being simulated. This undermines C4 and the headline pooling.
- **Family A P_D.** The formula is not the probability of the event it names. It gives P_D > 0 for a single ambiguous signal.
- **Proportional-test adapter.** Its sign is inverted.
- **NPE reference.** C1 and the "Bayes-optimal ceiling" are not run in a world where B is exact.
- **C3 thresholds.** C3 has two thresholds but only one constraint.

All five can be fixed in the text; none needs a redesign.

---

## 1. Status of round-1 must-change items (statistics)

| R1 item | Status | Reason |
|---|---|---|
| 1. Precedence-ordered partition | **addressed** (minor gaps) | Precedence S > U > D > W > G is exhaustive by construction, and a unit test is planned. Gaps: the class-determining pair is chosen by maximal r², not minimal d; "power ≥ 0.2" uses only μ while "powered" uses μ and λ (see S4). |
| 2. Ungated score | **partially** | The adapter table is ungated. However, the proportional-test s is sign-inverted (M3), and the mvSuSiE s still gates on lfsr < 0.05, which creates ties (C1). |
| 3. Coherent Family A outputs | **partially** | The null state and power filter fix the R1 null-effect inflation. The P_D product formula is wrong (M2, verified by Monte Carlo). The "no shared" factor uses τ (power 0.8), but oracle D uses power 0.2. |
| 4. NPE kill-test reference | **partially** | The simulator-prior mode was added. But the C1 world is not specified so that B is exact (M4): per-trait in-sample LD at L0, an approximate effect prior, the class prior under rejection sampling, background and M-type loci. |
| 5. D-vs-W co-primary; C3 non-gameable | **partially** | The co-primary endpoint was added. The C3 threshold rule is under-determined (M5). |
| 6. Calibration fairness | **mostly** | Raw and recalibrated ECE, and W not p12 tuning, are in place. Still undefined: the ECE event, and the meaning of "unweighted" when cells are filled by rejection sampling (S6). |
| 7. Generator realism | **addressed**, with new issues | Individual-level small-N trait 2, N grids and beta/se/N/MAF inputs are in place. New problems: the ρ grid is infeasible (M1), and λ does not match the generated z noncentrality (S2). |
| 8. Split units and CI | **addressed** | About 267 test blocks, paired BCa, weights recomputed per replicate, locus-level SE for screening. The sign-test statistic is undefined (S7). |
| Should 9 (E1/E2, exactness) | **addressed**; R̃ undefined | The Kronecker order is fixed and the scope is stated. R̃ and the η floor no longer appear anywhere in v2 (S1). |
| Should 10, 11, 13, 15–18 | addressed | ρ̂ check below (S8, minor). |
| Should 12 (power matching) | **partially** | S loci have no r²/d bin, so "within each r²/d bin" is undefined for them (S5). |
| Should 14 (exclusion probability) | **partially** | It uses τ₂ (power 0.8); oracle D excludes shared variants at power 0.2 (see M2). |

## 2. Numerical checks performed

**(a) Support-space BF (§5 B) vs the full-form RSS BF.** The full form uses z ~ N((I₂⊗R̃)β, V⊗R̃) with prior β_active ~ N(0, D). Setup: p = 40, ρ ∈ {0, +0.4, −0.7}. Configurations: D-type (1:j, 2:k), S-type with a correlated prior, S-type with a **rank-1** prior, and 2+2. Three LD cases:
- regular R (cond 3);
- near-singular R_uu (r² = 0.985 inside the configuration);
- rank-deficient R (n = 30 < p) with η = 1e-3, checked against a 60-digit mpmath full form.

Results:
- **All agree to ≤ 2e-13 relative.** The formula is correct.
- **Implementation conditions** (all must be written down):
  - (i) Compute b = (V⁻¹⊗I) z_u and Φ = V⁻¹⊗R̃_uu over the union u for *both* traits, then restrict to the active coordinates. If b is built only from the active z's, the BF is wrong whenever ρ ≠ 0. Example, ρ = 0.4, D-type: logBF 16.48 instead of 32.86.
  - (ii) Write (D⁻¹ + Φ)⁻¹ as (I + DΦ)⁻¹D. A rank-1 shared prior, which is a natural EB outcome, has no D⁻¹. The rewritten form is exact for it (checked).
- **Near-singular R is a test-design issue, not a formula issue.**
  - With an exactly duplicated SNP, the support form is stable down to η = 0: logBF 35.39885 at η = 1e-9 and at η = 0.
  - The **fp64 full form is not**: it is off by 7e-4 relative at η = 1e-6 and gives 14,090 instead of 35.4 at η = 1e-9.
  - So the planned test "support = full form within 1e-8" (§7.1.4) will fail spuriously on near-singular cases unless the reference is computed in extended precision.

**(b) Family A P_D (§5 A).** Monte Carlo over mean-field effect states (4e5 draws) against the design formula:

| Case | True P_D | Design formula |
|---|---|---|
| One effect, ambiguous between 1-only and 2-only (a single signal, i.e. W-like) | 0 | 0.192 |
| L = 1 with split 0.3/0.3/0.3 | 0 | 0.063 |
| An S-ish effect plus weak effects | 0.035 | 0.021 |

The inclusion–exclusion form (M2) matches Monte Carlo in every case.

**(c) Feasible ρ.** For a small-N trait nested in a large-N study:

ρ = N_s·r_p/√(N1·N2) ≤ √(N2/N1).

On the §2.2 grid the maximum is:
- 0.50 at N2 = 5,000, N1 = 20k;
- 0.22 at N2 = 1,000, N1 = 20k;
- 0.11 at N2 = 5,000, N1 = 400k;
- 0.016 at N2 = 100, N1 = 400k.

These maxima assume r_p = 1. ρ = 0.8 is unreachable for every (N1, N2) on the grid, and ρ ≥ 0.3 needs N2 = 5k, N1 = 20k and r_p ≥ 0.6.

**(d) λ ranges.**
- Trait 1: λ goes up to about 90 (N = 400k, PVE 2e-2), not "about 150".
- Trait 2: at N2 = 100 only 6% of drawn effects are powered and 83% are weak. N2 = 100 loci are therefore almost all W or G.

The generator computes z = Gᵀy/(√N σ̂_y), whose mean is √N·(R q). But λ = √(N q²/(1−q²)) is the t-statistic noncentrality. So λ overstates the noncentrality by a factor of 1/√(1−q²). At trait-2 PVE 0.3 that is +19.5%. See S2.

**(e) ρ̂ truncated MLE** on 20k SNPs. It is unbiased when the marginal variances are 1 (0.292 ± 0.005 at ρ = 0.3; 0.503 at 0.5). With a trait-1 z variance of 1.21 and the variances fixed at 1, it gives 0.270 and 0.469 (about 7% low). A 3-parameter MLE that also estimates the variances gives 0.305 and 0.508. A 1% signal with genetic covariance had little effect.

## 3. New problems and changes

### Must

**M1. The ρ grid is infeasible for GWAS × small-N QTL pairs (§2.2, §2.4, §8.1, C4).**
- What is wrong: §2.4 and §8.1 cross ρ ∈ {0, 0.1, 0.3, 0.5, 0.8} with every locus, but §2.2 makes trait 2 small-N (100–5,000) and nested. By check (c), most of the grid is unreachable. Either the generator silently sets a residual correlation c > 1, or it produces ρ values that cannot occur.
- Consequences: C4's slope and the headline pooling would rest on physically impossible cells, and the realism check would never flag it.
- Change:
  - (a) For cells with |ρ| ≥ 0.3, use a **large-N trait 2** (pQTL- or biobank-molecular-like, N2 ∈ {20k, 50k}, factor form for both traits, Cov(e1, e2) = ρR). State that this models e.g. UKB-PPP or lab-trait pairs.
  - (b) For small-N nested pairs, draw ρ = (N_s/√(N1N2))·r_p with r_p ∈ [−0.5, 0.5]. Report the realised ρ distribution.
  - (c) Define C4's slope on the union of cells, with ρ as the regressor.
  - (d) In the nested factor/individual hybrid, state the mapping explicitly. The factor form's effective sample size is N_syn, not N1, so the residual correlation on shared rows is c = ρ·√(N_syn/N_s), and c ≤ 1 must be asserted.

**M2. Family A P_D is not the probability of the oracle D event (§5 A "Locus outputs"; also the mvSuSiE adapter in §4.3 and layer D).**
- What is wrong: the product Π(1−a)·[1−Π(1−b)]·[1−Π(1−c)] treats the three events as independent. But within one effect the states are mutually exclusive. A single signal that is uncertain about which trait it belongs to gets P_D = 0.19 (check b). That is exactly the W-like case the D-vs-W co-primary endpoint scores.
- Change. Define per effect l three mutually exclusive events:
  - A_l = state "both" with |b₁| ≥ w₁ and |b₂| ≥ w₂, where w is the power-0.2 threshold (4.61 / 3.58), matching oracle D's "no shared variant with power ≥ 0.2 in both";
  - B_l = |b₁| ≥ τ₁ and not A_l;
  - C_l = |b₂| ≥ τ₂ and not A_l.

  B_l ∩ C_l is empty because τ ≥ w. Let a_l, b_l, c_l be their posterior probabilities, and s_l = P(both, |b₁| ≥ τ₁, |b₂| ≥ τ₂) ⊆ A_l. Then:
  - P_S = 1 − Π_l(1 − s_l);
  - P_D = Π(1−a) − Π(1−a−b) − Π(1−a−c) + Π(1−a−b−c).

  P_D ≤ Π(1−a) ≤ 1 − P_S, so P_W ≥ 0 holds. Use the same formula in the mvSuSiE adapter.
- Layer D: use w₂ (and w₁) in E₂ (E₁), not τ₂. Otherwise a grey shared effect, e.g. λ₂ = 4.5, is "excluded" and the locus is called D while the oracle says G.

**M3. The proportional-test adapter is sign-inverted (§4.3).**
- What is wrong: p_prop is the p-value of H0 "proportional = shared". A small p is evidence for D. So s must increase with p_prop: s = p_prop, not 1 − p_prop. As written, this baseline's AUPRC falls below the prevalence, which makes an unfair baseline.
- Change: add a known-direction control to the wrapper check: a λ = 30 shared variant at r² = 1 must give s > 0.9.

**M4. Make the C1 world one where B is exactly Bayes-optimal, and relabel the "ceiling" (§5 B simulator-prior mode, §5 C C1, M4/M5).**
- What is wrong: B uses one R̃ in both the mean and the covariance. Under the v2 generator, B is misspecified at L0 in four ways:
  - (i) trait 2's z follows its own in-sample R₂, and at N2 = 100–1,000 that differs materially from trait 1's R;
  - (ii) the effect prior (log-uniform PVE × N grid × strata) is only *approximated* by an 8-component Gaussian scale mixture;
  - (iii) evaluation cells are filled by rejection sampling on the class, which shifts the class prior;
  - (iv) 20% of loci carry polygenic background, and M loci have 3 variants per trait.

  A perfect NPE can therefore disagree with B by more than KL 0.02, and the "Bayes-optimal ceiling" is not a ceiling.
- Change. In the C1 world:
  - both traits use the factor form with the *same* R, which is also the R given to the methods;
  - |λ| is drawn **from the 8-component mixture itself**, conditioned on the per-locus (N1, N2) and stratum;
  - no background, at most 2 causal variants per trait;
  - B's class posteriors are multiplied by the class-prior ratio (target share / natural share) of any rejection sampling, then renormalised.

  Call the number on the main L0 distribution "B reference (approximately Bayes)", not a ceiling. Integrate the oracle-class events, which are regions in up to 4 dimensions involving both |λ| and |μ| = |Rλ|, by Monte Carlo or QMC from each configuration's Gaussian posterior. Report an MC error well below the KL tolerance; ≥ 512 draws per configuration with mass > 1e-6 is a suggestion.

**M5. C3 / matched-W rule: two thresholds, one constraint (§0 C3, §1, §8.2).**
- What is wrong: "confident-call rate on W = 5%" does not fix (t_S, t_D) jointly. A method can trade S calls against D calls, so C3 stays gameable.
- Change:
  - set t_S on validation so that the false-S rate on W is ≤ 2.5%, then t_D so that the false-D rate on W is ≤ 2.5%, with S taking precedence;
  - choose the largest threshold that meets each bound; report achieved rates, because coarse or tied scores cannot hit them exactly;
  - fix the W composition of the validation set used for this, i.e. the shares of "other trait null" vs "sub-powered shared".

### Should

**S1. Specify the B formula unambiguously (§5 notation, §5 B, §7.1.4).**
- Define R̃: e.g. R̃ = (1−η)R + ηI with η ≥ 1e-3 at levels ≥ L1. It is used everywhere but defined nowhere in v2.
- Write BF = |I + DΦ|^{-1/2} exp(½ bᵀ(I + DΦ)⁻¹D b), with b and Φ formed over the union for both traits, then restricted to the active coordinates (check a).
- The §7.1 test must cover ρ ≠ 0, a rank-1 D and 2+2 configurations. It should use fp64 full form only when cond(V⊗R̃) < 1e8, and an mpmath (≥ 50 digits) reference on small p for near-singular cases.

**S2. Make the oracle noncentrality match the generated statistic (§2.2).**
- Either set λ_t = √N_t·β_std, so μ = √N R β matches z = Gᵀy/(√N σ̂_y), or generate the OLS t-statistic.
- Currently the oracle overstates trait-2 noncentrality by up to 19.5% at PVE 0.3, which mislabels powered vs grey for strong small-N eQTLs.
- Cap the total trait PVE βᵀRβ (e.g. ≤ 0.5). M loci with 3 × 0.3 plus background otherwise exceed 1.
- Correct "λ up to about 150" to about 90.

**S3. Near-duplicate collapse removes identifiable D configurations (§5 B screen).**
- Collapsing at a fixed r² > 0.99 makes D impossible for pairs with r² > 0.99. Yet at λ ≥ 20 those pairs have d ≥ 2, so the oracle says D, not U, and biobank-strong loci reach λ ≈ 90.
- The B reference would then be biased toward S in the highest r² bin.
- Collapse only when max|ẑ|·√(1 − r²) < 0.5, or keep within-cluster D configurations with their pair multiplicity.

**S4. Class details (§1).**
- (a) Take the class-determining pair as argmin d over cross-trait powered pairs, not argmax r². Identifiability is driven by the most confusable pair.
- (b) Use one power measure throughout, e.g. min(|μ|, |λ|), for "powered", "power ≥ 0.2" and "weak". Today W and "no shared ≥ 0.2" use μ only, so cancellation can turn a large own effect into "weak".

**S5. Specify power matching for S (§2.4, §8.1).**
- S loci have no r²/d bin.
- Pre-register:
  - the same S pool is reweighted to each bin's D distribution of (μ1, μ2);
  - quantile edges are taken from D in that bin;
  - bins with D but no S are dropped, and the dropped D mass is reported per replicate;
  - matching is done within each (level, ρ) cell before equal-weight pooling.

**S6. Define ECE (§8.2).**
- State the event: P_S vs 1[S] and P_D vs 1[D] over S ∪ D ∪ W, or the 5-class form for C.
- State the distribution: test loci reweighted to the training/natural class prior, undoing the rejection sampling.
- Make the recalibrated ECE the comparison metric. A class-prior shift is exactly a logit-intercept shift.

**S7. Define the sign test's per-block statistic (§3, §8.3)** or drop it. AUPRC is not defined per block when a block has few S/D loci. A per-block paired difference in mean log-loss, or in the rank of s, is one option.

**S8. ρ̂ (§2.4).**
- Estimate both marginal variances in the truncated MLE (3 parameters). Fixing them at 1 attenuates ρ̂ by about 7% under a 1.21 variance inflation (check e).
- For real GWAS × GWAS pairs (R5b), genetic correlation leaks into the truncated estimator. Also report the LDSC intercept there.

### Could

- **C1.** mvSuSiE s: replace the lfsr < 0.05 gate with a continuous product of (1 − lfsr) terms to avoid ties.
- **C2.** Failure scoring for P_D: P_D = 0 on D, 1 on W.
- **C3.** Report D-vs-W AUROC on the hard-W stratum (W with a sub-powered shared variant) as well as the pooled value.

## 4. Items checked and found fine

- The precedence partition is exhaustive.
- P_S + P_D ≤ 1 holds for the coloc adapters.
- B1's coloc.abf identity holds: the z-scale BF with σ_j² = W²/se_j² is exactly Wakefield's ABF.
- Block bootstrap with about 267 units: BCa, with weights recomputed in the replicates and the jackknife.
- C1 needs every baseline's CI to exclude 0 (an intersection-union test). No multiplicity correction is needed.
- The ρ̂ estimator's SE is about 0.005 at 20k SNPs.
- Pooled headline plus macro-averaged values.
- Kill rule (b) count noise is fixed.
