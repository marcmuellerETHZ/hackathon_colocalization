# S1 Text. Approaches that did not work

This file explains each new approach the agent system built, the idea it borrowed, the test that ended it and what we learned. All verdicts come from pre-registered rules (DESIGN.md v4 §5–§6; DESIGN_P2.md). Kill verdicts were recomputed by an agent that had not produced them (review files in `reviews/`). The exceptions are:
- A1, recomputed only at the end of the project;
- C0, the 74 k LightGBM floor comparison, and B-mm rules (a), (b) and (d), which were not recomputed;
- C1, where the verifier checked the scoring fix on a 600-locus pilot.

**Notation.**
- z_t: the z-scores of trait t in a window;
- R: the LD matrix given to the method; R̃ = (1 − η)R + ηI;
- V = [[1, ρ̂], [ρ̂, 1]]: the null correlation from sample overlap;
- the joint model is z = [z₁; z₂] ~ N((I₂ ⊗ R̃) b, V ⊗ R̃);
- P_S, P_D: the posterior probabilities of the oracle events "shared and powered in both traits" and "distinct, each powered".

The ranking score is s = P_S/(P_S + P_D).

**Screening tests.** Kill rules (DESIGN §6.2), on 12,000 validation loci in six cells: {L0, L2, L4} × {P1 r_p = 0, P3 ρ = 0.5}. An approach is killed if any of these holds:
- **(a)** its AUPRC at L0, ρ = 0 is more than 1 paired SE below the best tier-1 baseline's;
- **(b)** its confident calls on W loci are more than 2× and more than 2 points above coloc.abf's, with one-sided bootstrap p < 0.05;
- **(c)** its false-S rate at L4 or at ρ = 0.5 is more than 2× and more than 2 points above the best baseline's;
- **(d)** it is more than 10× slower than coloc-SuSiE without an accuracy gain.

## 1. Family A: JointSuSiE (joint bivariate SuSiE-RSS)

**Idea and credit.** SuSiE writes a regression as a sum of single effects [Wang 2020]. SuSiE-RSS fits it to summary statistics [Zou 2022]. mvSuSiE extends it to several traits with a mixture prior over effect patterns [Zou 2026]. Family A gives each single effect l a variant j_l and an explicit state s_l ∈ {null, trait 1 only, trait 2 only, both}, with effect b_l ~ N(0, U_{s_l}).
- **Overlap** enters through the noise covariance V ⊗ R̃.
- **LD error** enters through η, estimated inside the IBSS iterations on the residual z.
- **Scales and the state prior** are fitted by empirical Bayes per locus.

**Locus outputs.** These come from inclusion–exclusion over per-effect events:
- a_l = P(effect l is shared and powered in both traits);
- w_l = P(l is shared with ≥ 20% power in both), which blocks D;
- p̃_l^t = P(l is powered in trait t and not blocking).

Then:
- P_S = 1 − Π_l(1 − a_l);
- with B = Π_l(1 − w_l): P_D = B − Π_l(1 − w_l − p̃_l^1) − Π_l(1 − w_l − p̃_l^2) + Π_l(1 − w_l − p̃_l^{1∨2}).

The probabilities use the Gaussian effect posteriors against the oracle power thresholds. Loci are fitted in GPU batches.

**Correctness.** At null weight 0, A reproduced mvsusieR to 8×10⁻¹⁴.

**Kill test A1** (2,000 loci at L0, ρ = 0). A's AUPRC was 0.893 against coloc-SuSiE's 0.978: Δ −0.085, 95% CI [−0.105, −0.064], about 8 paired SE below the bar. This is about 8 paired SE below zero. Its runtime was 0.38× coloc-SuSiE's. An end-of-project recomputation with independent code reproduced both AUPRCs and the gap exactly; the bootstrap CI differed only by Monte Carlo noise.

**Diagnosis.** It was done once, without changing A.
- **Score saturation.** 62.5% of A's loci fell in tied-score groups (s = 0 or 1), against 1.9% for coloc-SuSiE, so A lost ranking resolution.
- **Shared-state attraction.** A called D loci shared more often at moderate LD (r² 0.5–0.8: 25.8% vs 16.6%). This is consistent with IBSS settling into a shared state, despite a multi-start designed to prevent it.
- **Not the cause:** per-trait LD. A was lower on both the per-trait-LD and the single-LD subsets.

We did not test whether the state space, the η regulariser or the mean-field approximation was responsible.

**Lesson.** A joint model adds sharing-state parameters that IBSS must resolve. With exact LD, the simpler route (fine-map each trait, then pair credible sets) was more accurate.

## 2. Family B: ExactColoc (exact enumeration on GPU)

**Idea and credit.** coloc.abf enumerates configurations with at most one causal variant per trait, using Wakefield approximate Bayes factors [Giambartolomei 2014]. B enumerates every configuration with at most two causal variants per trait, on a screened set of M = 50 variants. The screen is the union of each trait's top variants plus conditional leads.

For a configuration with active effects u, B computes the exact Gaussian Bayes factor in effect space:

BF = |I + DΦ|^(−1/2) · exp(½ bᵀ(I + DΦ)⁻¹ D b), with Φ = Xᵀ(V ⊗ R̃)⁻¹X and b = Xᵀ(V ⊗ R̃)⁻¹z.

Here X selects the active (trait, variant) columns of I₂ ⊗ R̃, and D is the prior covariance of the active effects. The prior can be coloc's, Gaussian empirical Bayes, or the simulator's own. This is the RSS likelihood [Zhu 2017] evaluated exactly. It handles overlap through V and near-singular LD through support-space algebra. Configuration sums are done in fp64 on GPU.

**Correctness.**
- coloc mode reproduced coloc.abf to 3.6×10⁻¹⁴ on 200 loci.
- The support-space BF matched the full form to 2×10⁻¹³.

**Kill test B1 and rules (a)–(d)**, primary reading at the matched-W threshold:

| Rule | Result | Verdict |
|---|---|---|
| (a) | 0.977 vs coloc-SuSiE 0.978 | pass |
| (b), evaluated at fixed 0.8 | 1.2% vs coloc.abf 3.9% | pass |
| (d) | 0.126 s vs 23.5 s per locus | pass |
| B1 usefulness (false-S on D loci with r² 0.3–0.8 and multi-causal loci) | 0.217 vs coloc-SuSiE 0.132 | fail |
| (c) false-S at L4 | 0.205 vs coloc.abf 0.049 | kill |
| (c) false-S at ρ = 0.5 | 0.161 vs mismatch-aware coloc-SuSiE 0.061 | kill |

**Why the readings disagree.** The matched-W rule takes the smallest t_S with ≤ 2.5% confident-S calls on W. B puts almost no P_S on weak loci, so its t_S fell to 0.013, and at that threshold it called 22% of mid-LD D loci shared. At a fixed 0.8, B passed every rule and had the lowest false-S rate at ρ = 0.5. The verdict stands, because switching readings after seeing results was not allowed. The finding is a design lesson: an operating point must constrain both error directions.

**On the test set** (descriptive entry), B had:
- the best ideal AUPRC (0.969) and D-vs-W AUROC (0.994);
- a realistic AUPRC of 0.829, weakest at L4 (0.73);
- false-S on D of 22.6% at matched-W and 6.7% at 0.8, with zero confident-S calls on 4,555 W loci at 0.8.

## 3. B-mm: ExactColoc with a reference-panel error likelihood (phase 2)

**Idea and credit.** With a finite reference panel, the LD estimate R̂ is noisy, and the induced error in z is approximately Gaussian. Its variance is Var((R̂b)_k) ≈ Σ_j b_j²(1 − r_jk²)²/n_ref, plus cross terms. This is a delta-method approximation for sample correlations, derived within the project (DESIGN layer E) and motivated by the LD-mismatch analysis of SuSiE-RSS [Zou 2022].

B-mm adds this inflation Δ to the noise covariance:

z ~ N((I ⊗ R̃) b, V ⊗ R̃ + Δ).

It plugs in the exact posterior second moments E[bbᵀ | z], refined twice, and refits (σ, η) by empirical Bayes under the inflation. Each Bayes factor stays exact for this covariance through a Woodbury form over the full z-vector. A configuration-dependent Δ would need about 1.6 million dense 100 × 100 Cholesky factorisations per locus, which is infeasible.

**Correctness** (52 tests pass):
- with n_ref = ∞ it reproduces phase-1 B to 4×10⁻¹¹;
- brute force at p = 10 agrees to 7×10⁻¹²;
- against simulated reference panels, the inflation formula was within 6%; the diagonal-only form was off by 61%.

**Verdict.**
- **New-method gate:** passed. On mismatched screen loci (L2 + L4), AUPRC was 0.861 against phase-1 B's 0.822 (+0.040, block SE 0.006).
- **Candidacy bar:** missed. On the validation subsample B-mm scored 0.860 against mismatch-aware coloc-SuSiE's 0.899 (Δ −0.039 ± 0.009, about 4 SE). This is the robust reason it failed.
- **Rule (c) at L4:** killed under the two-sided operating point OP2. False-S was 10.9% against a kill line of 9.8% (coloc.abf 4.9%), and 14.5% against ColocBoost-m1's 4.5%. This is a point-estimate kill: the bootstrap probability of no kill was 0.12–0.17 against coloc.abf and 0.07–0.08 against ColocBoost-m1.
- **Runtime:** 0.18 s per locus on one GPU.

**Lesson.** Modelling reference size helps an exact method, but a fixed screen of 50 variants with at most two causal variants per trait still loses to SuSiE's adaptive fine-mapping under mismatch.

## 4. Family C: NeuralColoc (amortised neural posterior estimation, the bold family)

**Idea and credit.** Simulation-based inference trains a network to output the posterior directly from data simulated under the prior [Cranmer 2020].
- **Architecture.** A transformer [Vaswani 2017] over the M = 128 screened variants, with an attention bias learned from r_jk, r_jk² and |r_jk|, plus a per-variant stream over all window variants.
- **Readout.** Structured like a sum of Bayes factors: K = 16 quadratic energies in allele-invariant statistics, pooled by log-sum-exp.
- **Invariance.** Variants are re-coded so it is exactly invariant to allele flips.
- **Outputs.** Five-class probabilities.
- **Training.** On train-split simulations at all mismatch levels except L6 and L6c, which were held out.

**Ladder:**
- **C0** (reproduce coloc.abf in its own world): passed, E[KL] 0.00072 (bar 0.01).
- **C1** (reproduce the exact Bayes posterior of B in a world where B's model is correct): passed, E[KL] 0.0056 [0.0040, 0.0087] (bar 0.02) at 14.4 M training loci. The pass depended on a scoring fix: an erroneous class-weight correction was removed and the fix verified independently to 10⁻⁶ before the single evaluation.
- **Floor:** at 74 k records C tied a LightGBM classifier on hand-built features (ΔAUPRC −0.0004).
- **C2** was not run in phase 1. In phase 2 it was trained on 600 k records, a size fixed by a timing pilot before training. Results on the validation subsample:

  | Model | AUPRC |
  |---|---|
  | C | 0.964 |
  | C without LD inputs | 0.965 |
  | LightGBM floor | 0.959 |
  | B | 0.834 |
  | A | 0.675 |

  C beat B, had better calibration, a small region gap (0.005) and beat the floor (+0.005 > 1 SE), but it did not beat its no-LD ablation (−0.001 ± 0.002). **C2 failed.**

**Interpretation.** The independent check found no train/validation overlap (0 shared blocks or windows), and metadata features alone scored at chance. Still, all three learned models were trained and evaluated under the same simulator, priors and class definitions, so the validation numbers measure transfer across genomic regions, not across data-generating models. The no-LD tie suggests the networks learned the simulator's prior over effect sizes and architectures rather than LD structure. We did not audit the 30 global input features in code. The learning curve was still rising (0.970 → 0.984 on validation records from 75 k to 600 k).

**Lesson.** A learned colocalization model needs evaluation on data from a different generator, or on real labels, before its accuracy means anything.

## 5. Families G and G-mm: GPU ports of SuSiE (engineering)

**Idea and credit.** A batched SuSiE-RSS on GPU with coloc's bf_bf pairing. gpu-coloc [Jesse 2026] had already made pairing fast. Fine-mapping per locus was the remaining bottleneck.
- **G** ports susieR 0.14.2. Identity: max |ΔPIP| 6.7×10⁻⁸ in fp64.
- **G-mm** ports the susieR 0.16.6 call that our mismatch-aware wrapper makes (R_finite, eb_mix).

**G-mm identity** (100 L2/L4 loci, 200 fits):
- converged fits: 176 of 176 within 1.6×10⁻⁸;
- status agreement: 200 of 200.

For some of the 24 non-converged fits, the frozen 10⁻³ bar was unattainable even for R against itself. We wrote a rule comparing against R's own run-to-run spread (P4-3), after seeing the status. We amended it once (P4-3a), after seeing the result; both changes are disclosed. Even then one fit failed (ΔPIP 0.47 vs spread 0.021). That fit is chaotic: three G-mm builds that differ only in summation order land at three different points, and its PP.H4 is about 2×10⁻¹¹ in every run. A second amendment was refused. Under the frozen bar alone, 4 fits fail.

**Throughput** (exclusive slot, 2,000 loci, 3 repeats):
- 31,838 loci/h on one GPU against 6,670 for mismatch-aware coloc-SuSiE on 40 memory-limited cores: 4.77×, below the 5× bar;
- 4.34× against coloc-SuSiE 0.14.2 (2 repeats).

About 60% of G-mm's time was serial disk reading by one process. A post-hoc four-process run was slower (13,118 loci/h), because processes time-slice the GPU.

**Lesson.** Matching an iterative R fit exactly is possible for converged fits but not for chaotic ones, so identity bars should be calibrated against the reference's own variability first. Speed is limited by data loading and kernel launch latency before arithmetic.

## 6. Two operating-point rules

- **OP1 (matched-W, phase 1).** The smallest threshold with ≤ 2.5% confident calls on W. It is simple, but it rewards methods that are confident on weak loci and penalises those that are not (see B).
- **OP2 (phase 2).** Adds the condition that ≤ 10% of distinct loci are called shared. The 10% was chosen after seeing phase-1 results and sits just above mismatch-aware coloc-SuSiE's own rate (0.095–0.097), so that method's OP2 equals its OP1. The same 10% set B-mm's and C's thresholds. A different constant could have changed B-mm's rule-(c) verdict but not its candidacy miss.

**References** (full citations in the main text): Wang 2020 [12]; Zou 2022 [8]; Zou 2026 (mvSuSiE) [17]; Giambartolomei 2014 [11]; Zhu & Stephens 2017 [21]; Cranmer 2020 [22]; Vaswani 2017 [23]; Jesse 2026 [20]; Ke 2017 (LightGBM) [24].
