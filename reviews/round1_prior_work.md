# Review round 1: prior work (colocalization and multi-trait fine-mapping)

Reviewer role: prior-work expert. Scope: DESIGN.md v1, LITERATURE.md, lit/methods.md, lit/gap_and_evaluation.md. Date: 2026-10-26.

**Verdict: needs revision.** Families B and C are novel enough. Family A, as a statistical model, is not novel: it is mvSuSiE-RSS with two traits and the canonical (singleton plus shared) prior, which is close to SharePro plus a residual covariance V. A's contribution can only be speed, an LD-mismatch term inside the joint likelihood, and the power-aware decision layer. The design must say this plainly and benchmark A against mvSuSiE-RSS and SharePro, not only against coloc-SuSiE.

The baseline list is missing the two strongest relevant comparators: mvSuSiE-RSS, and susieR's mismatch-aware SuSiE-rRSS. Several adapters are missing or ill-defined. One mathematical issue in A's locus-level P_S would make null loci look shared.

---

## Must-change items

### M1. State A's relation to prior work honestly and benchmark it against the closest models (§5 Family A, §4, §6.2)

**Evidence.**
- mvsusieR exports `mvsusie_rss`, and `create_mixture_prior(R=…)` builds canonical covariance matrices that include the **singletons and the shared effects** (checked in the package NAMESPACE and the reference docs). With two traits this is exactly A's state space {1-only, 2-only, both}, including correlated shared effects.
- mvSuSiE also estimates the residual covariance V from null z, which is A's overlap model.
- SharePro uses one variant selector per effect group with trait-specific Bernoulli activity. That is the same sharing structure, with independent effects and no V.
- CAFEH is the same idea with per-trait activity.

**Change.**
- Replace "a GPU, coloc-oriented, mismatch-aware version of the mvSuSiE / SharePro idea" with a short difference table, A vs mvSuSiE-RSS vs SharePro vs CAFEH, covering:
  - effect prior;
  - V/ρ;
  - LD-error term;
  - null state;
  - decision layer;
  - hardware.
- Declare the claimed contribution as (i) GPU batching, (ii) the η / finite-reference term in the joint likelihood, and (iii) layer D.
- Add an **identity test**: A with η = 0 and layers D/E off must reproduce `mvsusie_rss` PIPs and per-trait lfsr (|ΔPIP| < 1e-2) on about 50 toy loci, using the same prior grid and V.
- Kill-test A1 must compare against **mvSuSiE-RSS and SharePro** as well as coloc-SuSiE. lit/methods.md §6 already said "if not at least as good as SharePro, stop and reuse SharePro's model", and DESIGN dropped this.
- Add an ablation ladder: A0 (≈ mvSuSiE-RSS on GPU) → +η / R_finite → +outlier mask → +D. Report accuracy and runtime at each step, so that the gain is credited to the right component.

### M2. Fix A's locus-level P_S: null effects inflate it (§5 Family A "Outputs")

**Problem.** SuSiE-type models have no "absent" state per effect. Effects that are not used get an estimated prior variance of ≈ 0, so their Bayes factors are ≈ 1 for every state, and their sharing-state posterior falls back to the prior.

With P_S = 1 − Π_l (1 − P(s_l = both)), a pure-null locus gets P_S = 1 − (1 − π12)^L. I checked this numerically in .venv:
- L = 10, π12 = 1/3: 0.97;
- π12 = 0.1: 0.61;
- L = 5, π12 = 1/3: 0.80.

**Change.** Choose one of these and freeze it before screening:
- add an explicit null/inactive state with prior mass (mvSuSiE's `null_weight`);
- compute P_S only over effects with an estimated prior variance > 0 *and* a CS of purity ≥ 0.5 (the SuSiE/coloc-SuSiE convention).

Also note that the product formula treats effects as independent. Its calibration has to be checked against ExactColoc, which family B already allows.

### M3. Add the missing state-of-the-art baselines (§4)

1. **mvSuSiE-RSS** (mvsusieR, pinned commit).
   - Settings: L = 10; canonical prior for 2 traits; V estimated from null z (|z| < 2, as in Zou 2026) and, separately, the oracle ρ; LD per level L0–L5.
   - Runtime is about 41 s per locus with 5k SNPs (mvSuSiE paper). The 12k-locus screening set is therefore ≈ 2–3 h on 64 cores, which is affordable in full.
   - This is A's nearest neighbour and the only published joint fine-mapper that models overlap.
2. **coloc-SuSiE on SuSiE-rRSS**: susieR GitHub / r-universe **0.16.x** (0.16.6, 2026-07-28), `susie_rss(..., R_finite = n_ref, R_mismatch = "eb_mix")`, with its `SER_fallback` diagnostic.
   - This is the current best published practice for LD mismatch.
   - Claim C1 is about performance under mismatch. Without this baseline, C1 compares against a weaker-than-state-of-the-art competitor.
   - The installed susieR is CRAN 0.14.2. I checked that `R_finite`, `R_mismatch` and `lambda` are **not** arguments of its `susie_rss`. Both versions need separate pinned installs.
3. **At least one proportional test, required rather than "if installable"**: colocPropTest v0.9.3 (CRAN) and/or prop.coloc v2.0.0.
   - These are the only methods in the list whose null is "distinct", and they assume that both traits are associated. That is exactly the primary S-vs-D-powered estimand.
   - They are also the published remedy for variable information, the L5 condition.
   - They give p-values, so declare them "ranking-only": they enter AUPRC but have no three-way call.

### M4. Specify every baseline's configuration and give every baseline an adapter (§4, §6.3)

The current adapter table omits eCAVIAR, gpu-coloc, the proportional tests and the lead-SNP rule's score, and several rows are underspecified.

- **coloc-SuSiE (coloc 5.2.3).** Defaults checked in the installed package:
  - `coloc.abf` p12 = 1e-5, but `coloc.bf_bf` p12 = **5e-6**, with `overlap.min = 0.5` and `trim_by_posterior = TRUE`;
  - `runsusie` passes `n = N` and repeats until convergence;
  - `susie_rss` 0.14.2 has no λ argument.

  The phrase "standard λ-regularised R" must therefore be written out:
  - **vanilla** (what coloc/eQTL Catalogue pipelines actually do: no regularisation, `estimate_residual_variance = FALSE` for out-of-sample LD, as the susieR vignette advises);
  - **QC'd**: `estimate_s_rss` → R̃ = (1−s)R + sI, plus `kriging_rss` outlier removal.

  Also note that `trim_by_posterior` silently drops pairs when variants are missing, which affects L4.
- **SharePro** (zhwm/SharePro_coloc, pin the commit; release 5.0.0).
  - Inputs are BETA/SE/N, so specify BETA = z/√N and SE = 1/√N.
  - K = 10, and the documented K = 1 fallback under LD mismatch (added in v2.0.0) should be used when the fit fails.
  - P_S = max over effect groups of `share`.
- **ColocBoost** (CRAN 1.0.10, 2026-09-09).
  - It has documented mismatch modes: one-iteration `M = 1`, recommended under mismatch, and an LD-free mode. Run both the default mode and the mismatch mode, and choose between them on validation, as for the other baselines.
  - Define P_S from its output, e.g. the maximum over two-trait colocalization sets (CoS) of the minimum per-trait `npc_outcome`. A locus with no CoS gets P_S = 0.
- **HyPrColoc.** Defaults checked in the source:
  - `trait.cor` = identity and `sample.overlap` = all ones, so overlap is ignored by default;
  - `prior.c = 0.02`, which makes the effective p12 ≈ 2e-6, not coloc's value;
  - posteriors are rounded to 4 decimals;
  - a 2-trait cluster is only reported if the posterior is ≥ reg × align thresholds (0.5 × 0.5).

  Without changes, many D loci get no posterior at all and fall into "undefined → s = 0.5". The fix:
  - set `reg.thresh = align.thresh = 0` (or call the internal 2-trait posterior directly);
  - for the overlap-aware run, set `trait.cor[1,2] = ρ̂` with `sample.overlap` all ones.
- **eCAVIAR.** P_S = max_j CLPP_j (or Σ_j CLPP_j; pick one now). Add **CLPP from SuSiE PIPs**: it is free given coloc-SuSiE, and Open Targets and gpu-coloc use it.
- **gpu-coloc.** Pin the commit. Its default drops signals with max LBF < 5, and its recommended p12 is 1e-6. Either match coloc-SuSiE's settings exactly, which makes it a pure speed reference, or report it with its own defaults; say which.
- **Ranking score.** For coloc-type methods use the native, widely used ratio s = PP4/(PP3+PP4), taken from the best CS pair for coloc-SuSiE. The current P_D rule for coloc-SuSiE ("min PP3 if every pair has PP3 > PP4, else 0") is a step function: any pair with PP4 ≥ PP3 forces s = 1. That throws away ranking information and penalises the baseline.

### M5. Fix the covariance algebra in Family B (§5 Family B)

**Problem.** With z stacked trait-major, z = (z1; z2), the null covariance is V ⊗ R̃_uu and the mean is (I₂ ⊗ R̃_uu)λ. Then

Σ_alt = V ⊗ R̃_uu + (I₂ ⊗ R̃_uu) D (I₂ ⊗ R̃_uu)ᵀ.

DESIGN writes (R̃_uu ⊗ I), which is the variant-major ordering. That ordering pairs with R̃ ⊗ V, not V ⊗ R̃. The mixed form is wrong for any p_u > 1.

**Change.**
- Write one consistent convention.
- Add a unit test: Σ_alt computed from simulated z equals the empirical covariance.

### M6. Make kill-test B1 well-defined (§5 B, §7.2)

**Problem.** coloc.abf puts its prior on β (W = 0.15² × sdY²). In z-space that is a **MAF- and N-dependent** prior variance, W/V_j = W · 2f(1−f)·N / sdY². With a prior on standardized effects, ExactColoc will not reproduce coloc.abf to 1e-3.

**Change.**
- Specify `type = "quant"`, sdY = 1, varbeta from N and MAF, and the matching per-SNP prior variance in B for the identity test.
- Say how p1, p2 and p12 extend to configurations with 2 variants per trait. For example, prior ∝ Π over included variants, and a "H3+4" prior = p12·p1 or p12·p2.

---

## Should-change items

### S1. Asymmetric-LD arm, the standard real pipeline (§2.3)

In eQTL Catalogue, gpu-coloc and Open Targets practice, the QTL side is fine-mapped with **in-sample** individual-level SuSiE and only the GWAS uses a reference panel. Per-trait methods (coloc-SuSiE, CLPP, gpu-coloc) can exploit this. Joint models (A, SharePro, mvSuSiE) need one R for both traits.

**Change.** Add a level, e.g. L2a: trait 2 uses R_true, trait 1 uses pool B. Otherwise the benchmark overstates the disadvantage of the per-trait baselines and misses the case where joint models are structurally handicapped.

### S2. Screening must cover the D r² 0.8–0.95 stratum (§6.1)

**Problem.** The primary metric includes D up to r² < 0.95, but the screening set stops at 0.8. Shared-selector models are known to merge high-LD distinct variants into one effect:
- CAFEH puts causal variants with r² > 0.7 into one component;
- SuSiE merges variants ("colliders", Jesse 2026).

For A this would show up as false S exactly in the omitted stratum.

**Change.** Add "D r² 0.8–0.95" as a screening cell, or exclude that stratum from the primary metric explicitly.

### S3. p12 tuning is a no-op for the primary metric; give baselines equal recalibration (§12.4, §6.2b)

**Evidence.** For coloc.abf,

PP4/PP3 = (p12/(p1·p2)) · Σ_j ABF1_j·ABF2_j / (Σ_i ABF1_i · Σ_j ABF2_j − Σ_j ABF1_j·ABF2_j).

Changing p12 rescales this ratio by a constant, so the ranking by s is unchanged. I checked this numerically: the ranks across p12 = 1e-5 and 1e-6 were identical on 200 loci. The same holds for HyPrColoc, and per pair for coloc-SuSiE.

So the "p12 grid" only affects thresholded metrics (three-way F1, W confident calls, ECE). Meanwhile C gets temperature scaling on validation.

**Change.** For C3 and the ECE claims, also report every baseline after the same post-hoc recalibration fitted on validation (temperature or Platt scaling on logit s). Otherwise calibration gains may simply reflect that only our method was recalibrated.

### S4. Temper the novelty claims (DESIGN §0 C5; LITERATURE.md §2 last line and §3.2)

**C5.** Partial predecessors exist:
- Hwang et al. 2025 (preprint): same-protein positive controls across populations and platforms. This is mismatch on real data.
- The HyPrColoc supplement: the effect of ignoring overlap.
- SS2: type I error under overlap.
- ColocBoost: overlap in its simulations.
- Zou 2022: LD mismatch in fine-mapping.

Reword C5 as "first systematic, method-comparative quantification…", and cite these.

**"No published method combines…" (LITERATURE.md).** Add that mvSuSiE-RSS already combines multi-causal sharing states with overlap V, and that ColocBoost and SharePro have explicit, if crude, LD-mismatch modes (M = 1 / LD-free; K = 1). The unclaimed part is the combination with an LD-error model in the joint likelihood, a power-aware "insufficient" call, and GPU throughput.

### S5. Speed claim C2 needs the right reference (§0 C2, §7.7)

coloc-SuSiE is a moving target. Report a Pareto front (accuracy vs throughput) that includes:
- coloc.abf: no LD, about 0.1 s, the fastest sane baseline;
- coloc-SuSiE parallelised over 64 cores;
- G (GPU coloc-SuSiE);
- mvSuSiE-RSS.

State C2 against G as well as R coloc-SuSiE. A's speed advantage over G is the honest measure of what the modelling costs or buys in speed.

### S6. Family E vs susieR rRSS (§5 cross-cutting E)

E's "reference-size-aware variance inflation, Var(z|b) ≈ R + (1/n_ref)·f(R, b)" is the published idea behind susieR's `R_finite`.

**Change.**
- Cite it.
- Implement E so that it reduces to `R_finite` in the single-trait case, and add that equivalence as a test.
- Claim novelty only for its use in the bivariate likelihood and the effect on coloc decisions.

### S7. Pin and record versions consistently (§4, §9)

- **coloc.** CRAN is 5.2.3 (2023-10-03); the GitHub release is 6.0.3 (variant-specific `prior_weights`). Pin 5.2.3 for the main runs, since simulation uses uniform causal priors and v6's addition is irrelevant, but record why.
- **Exact versions** for every baseline:
  - susieR 0.14.2 (coloc's engine) and 0.16.6 (rRSS);
  - mvsusieR commit;
  - SharePro commit;
  - colocboost 1.0.10;
  - hyprcoloc commit;
  - gpu-coloc commit;
  - colocPropTest 0.9.3;
  - prop.coloc 2.0.0;
  - CAVIAR/eCAVIAR commit.

  Put these in `env/renv.lock` / a `baselines.lock`.

---

## Could-change items

- **C1. Further baselines** (state the reason for any exclusion in §4):
  - PWCoCo (COJO with reference LD; widely used; in SharePro's benchmark);
  - SS2 / LocusFocus (the only coloc test with an explicit n0·ρ/√(n1·n2) overlap covariance);
  - gwas-pw (overlap-aware, four models);
  - FINEMAP-miss + CLPP for the L5 variable-information arm (Kartau & Pirinen 2025);
  - RSparsePro + coloc.bf_bf for mismatch;
  - CAFEH (largely covered by SharePro and mvSuSiE).

  Exclude JLIM (it needs individual-level data for trait 2 plus permutations) and fastENLOC (a genome-wide enrichment prior, not a per-locus test), with reasons stated.
- **C2. Prior art for Family B.** Cite eCAVIAR/CAVIAR (exhaustive per-trait enumeration), fastPAINTOR multi-trait (Kichaev et al., Bioinformatics 2017;33:248, PMID 27663501: joint multi-trait fine-mapping with LD) and moloc. B's novelty is the exact joint 2-trait posterior with ρ, the "H3+4" hypothesis and GPU batching, not enumeration as such.
- **C3. Prior art for Family C.** No amortised or neural coloc method was found; my quick web/PubMed check agrees. Cite BEATRICE (neural VI fine-mapping) and simulation-based-inference practice as the nearest work. For the GPU-SuSiE claim, note that JAX-based fine-mappers exist (SuShiE, Lu 2025, individual-level), so phrase it as "no published GPU *summary-statistic, batched-over-loci* SuSiE-RSS".
- **C4. Signal level for A, SharePro, mvSuSiE and ColocBoost.** Their native output is per effect group or CS. The §8.4 signal-level metric should use these natively, which favours them fairly. Do not force them only through the locus-level adapter.

---

## Errors and inaccuracies found in the record files

1. **"susieR 2.x `R_finite` / `R_mismatch`"** (LITERATURE.md §3.2; lit/methods.md §0, §1, §4). There is no susieR 2.x release. The features are in susieR **0.16.x** on GitHub / r-universe (0.16.6, 2026-07-28); "SuSiE 2.0" is the name of the McCreight 2025 preprint, whose abstract does not describe rRSS. The CRAN/installed 0.14.2 lacks them (checked `formals(susie_rss)`).
2. **"coloc (CRAN, v6.0.3)"** (lit/methods.md §1). CRAN has 5.2.3 (2023-10-03); 6.0.3 is the GitHub DESCRIPTION.
3. **HyPrColoc's "`sample.overlap` default is a matrix of 1s (complete overlap)… users should be aware"** (lit/methods.md §2.3). Misleading. The default `trait.cor` is the identity, so the default *ignores* overlap; the all-ones overlap matrix only matters once `trait.cor` is set.
4. **"Default priors (p12 = 1e-5) gave a realized FDR of 0.26 at a nominal 0.05"** (LITERATURE.md §1). This comes from the Hukku 2021 **preprint** table (lit/gap_and_evaluation.md l.175), for a specific simulated enrichment. LITERATURE.md does not mark it [PREPRINT]; it should, with that context.
5. **"Separability breaks down above r² ≈ 0.8 (JLIM)"** (LITERATURE.md §3.3). θ = 0.8 is JLIM's *definitional* threshold for "distinct", not an empirical breakdown point. JLIM reports AUC 0.883 at 0.7 < r² < 0.8.
6. **DESIGN §5 B: Kronecker ordering.** See M5.
7. **DESIGN §5 A: P_S formula.** See M2.
8. **DESIGN §4.2: "standard λ-regularised R".** susie_rss 0.14.2 has no λ argument, and the standard coloc pipeline does not regularise. See M4.

## What I checked

- **Installed versions:** coloc 5.2.3 and susieR 0.14.2 (env/renv). mvsusieR, colocboost, hyprcoloc and colocPropTest are **not installed**.
- **coloc defaults:** formals of `coloc.abf`, `coloc.bf_bf`, `coloc.susie` and `runsusie`.
- **susieR 0.14.2:** `susie_rss` has no `R_finite`, `R_mismatch` or `lambda` arguments; `kriging_rss`, `estimate_s_rss` and the internal `susie_rss_lambda` exist.
- **Online sources:**
  - CRAN pages: susieR 0.14.2 (2025-06-05), coloc 5.2.3, colocboost 1.0.10 (2026-09-09);
  - GitHub DESCRIPTIONs: coloc 6.0.3, susieR 0.16.6;
  - susieR rss_mismatch vignette;
  - mvsusieR NAMESPACE (`mvsusie_rss`) and `create_mixture_prior` docs (singletons and shared canonical matrices);
  - SharePro_coloc README (inputs, `share` output, K = 1 mismatch fallback);
  - ColocBoost LD-free vignette (M = 1, LD-free mode, npc outputs);
  - hyprcoloc source (defaults, thresholds, rounding);
  - SuSiE 2.0 preprint abstract;
  - PubMed: fastPAINTOR, PMID 27663501.
- **Numerical checks (.venv):**
  - power thresholds in §1 (6.29 / 5.26 / 4.61 / 3.58) are correct;
  - the coloc.abf ranking by PP4/(PP3+PP4) is invariant to p12 (identical ranks on 200 random loci);
  - the null-effect P_S inflation in M2.
