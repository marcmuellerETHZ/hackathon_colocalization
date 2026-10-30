# Review round 1: ML / simulation-based inference (Family C and learned components)

Reviewer role: machine-learning scientist (amortised NPE, set/graph transformers, calibration, distribution shift).
Scope: DESIGN.md v1 §1 (labels), §2.3–2.4 (training data), §5 Family C, §6 screening, §8 calibration metrics, and where the issues spill into A/B.

**Verdict: Family C is a legitimate bold contender, but v1 would not run as written. Its SNP selection loses the trait-2 signal in most strong-GWAS/weak-QTL loci (checked on real LD). The training labels are ambiguous (G precedence and soft labels are undefined). Kill-test C1 compares two different estimands. The ECE comparisons favour whichever method was fitted to the simulation mixture. Five must-changes, six should-changes, four could-changes.**

---

## Checks I ran

1. **Power thresholds (§1).** Recomputed with scipy, counting both tails.
   - α = 5e-8: z_α = 5.451, power 0.8 at μ = 6.293, power 0.2 at μ = 4.610.
   - α = 1e-5: z_α = 4.417, power 0.8 at μ = 5.259, power 0.2 at μ = 3.576.
   - All DESIGN.md numbers are correct.
2. **Top-M selection recall.** Real 1000G-30x EUR genotypes (n = 503, MAF ≥ 0.01, biallelic SNVs), four 1-Mb chr22 windows (p = 1,839–3,739).
   - Setup: trait 1 has a causal variant with λ1; trait 2 has a distinct causal variant with r² < 0.1 to it and λ2; noise follows the genotype-factor generator; 200 replicates per cell.
   - Measured: the fraction of loci where the trait-2 causal variant or any proxy with r² > 0.8 is in the selected set.

   | λ1, λ2 | M | top-M by max(\|z1\|,\|z2\|), recall per window | per-trait union (M/2 + M/2) |
   |---|---|---|---|
   | 10, 6 | 128 | 0.62–0.88 | 1.00 |
   | 20, 6 | 256 | 0.44–0.86 | 0.99–1.00 |
   | 40, 6 | 128 | 0.10–0.24 | 1.00 |
   | 40, 6 | 256 | 0.20–0.46 | 1.00 |
   | 40, 10 | 256 | 0.39–0.72 | 1.00 |

   Why: a SNP has a median of 300–750 neighbours with r² > 0.02 in these windows. Any neighbour with |r| > λ2/λ1 to the GWAS lead outranks the QTL lead. Script: `/tmp/claude-190744/-cluster-scratch-mumarc-hackathon-eqtl-gwas/0666a753-70c8-497e-b212-f9558627195a/scratchpad/topm.py`.
3. **Proxy prevalence.**
   - 67–78% of SNPs have a proxy with r² ≥ 0.95.
   - 11–13% have their best proxy with r² in [0.8, 0.95).
   - Script: `/tmp/claude-190744/-cluster-scratch-mumarc-hackathon-eqtl-gwas/0666a753-70c8-497e-b212-f9558627195a/scratchpad/prox.py`.
4. **ρ̂ estimator (§5 notation: null SNPs with |z| < 2 in both traits).** Monte Carlo, 2·10⁶ draws. Truncation attenuates the sample correlation:

   | true ρ | ρ̂ |
   |---|---|
   | 0.1 | 0.077 |
   | 0.3 | 0.240 |
   | 0.5 | 0.417 |

   That is about 17–23% too low.
5. **Effect-size grid.** Under the §2.4 log-uniform λ ranges:
   - 52% of single-variant pairs are "both powered";
   - 12% (trait 1) and 18% (trait 2) of causal variants fall in the grey zone;
   - so the "any causal variant is grey" rule moves many S, D and M loci into G (see change 2).

## Errors found

- **E1 (factual/design; affects B and C).** Selecting the top M SNPs by max(|z1|,|z2|) systematically drops the weaker trait's distinct signal (check 2). This removes exactly the D loci the project must call. The "LD-proxy expansion" in B does not help, because it expands around SNPs that were already selected.
- **E2 (label definition).** §1 does not define the precedence of G over S and D. A locus with a powered shared variant plus a grey trait-specific extra is either S or G depending on reading. Under the literal "any causal variant", it is G, which silently removes a large share of the S (50% carry an extra), D-with-extras and M loci from the primary metric and from the training labels.
- **E3 (kill-test C1 estimand mismatch).** ExactColoc outputs posteriors over coloc's H0–H4 (+H3+4) under *its* prior (p1, p2, p12, Gaussian D). NPE trained on oracle {S, D, W} labels from the §2.4 mixture outputs posteriors over *different events* under a *different prior*. Mean |ΔP| < 0.03 between them is not a correctness criterion. A perfectly working NPE can fail it, and a broken one can pass it, because a mean is dominated by easy loci with P ≈ 0 or 1.
- **E4 (statistics; affects A and C).** The |z| < 2 null-SNP ρ̂ is biased toward 0 by about 20% (check 4). Under-corrected overlap is exactly the false-sharing mechanism of C4.
- **E5 (minor, B).** If z is stacked trait-major, consistent with Σ_null = V ⊗ R̃_uu, the mean is (I₂ ⊗ R_uu)λ. So the covariance term should be (I₂ ⊗ R_uu) D (I₂ ⊗ R_uu)ᵀ, not (R̃_uu ⊗ I). Also state whether R_uu or R̃_uu enters the mean.

---

## Changes

### 1. [MUST] Per-trait stratified SNP selection (§5 Family C "Network"; same fix in Family B "Screen")

**What.** Build the token set as follows:
- the union of the top M/2 SNPs by |z1| and the top M/2 by |z2|;
- plus, for each trait, the top k (k = 16) by a conditional statistic after removing that trait's lead, (z_j − r_j,lead·z_lead)/√(1 − r²), computed with the reference R;
- deduplicate and pad or mask to M;
- add a per-token flag for why the SNP was selected.

**Pre-registered recall test (M3).**
- Recall of every powered causal variant, or a proxy with r² > 0.8, must be ≥ 0.99.
- Stratify the test by λ1/λ2 ∈ {≤ 1, 1–3, 3–10} and by L-level, and run it on validation windows.

**Why.** See E1 and check 2: the current rule has recall 0.10–0.46 at λ1 = 40, λ2 = 6. The per-trait union has recall ≈ 1.0 at the same budget. NPE posteriors stay valid under any deterministic selection, since they condition on f(x). They are useless, though, if f(x) has discarded the signal.

### 2. [MUST] A clean label partition and hard 4-class training targets (§1 table; §5 C "Training")

**Partition.** Call a causal variant *powered* (≥ 0.8), *grey* ([0.2, 0.8)) or *absent* (< 0.2) per trait. Then:

| Class | Definition |
|---|---|
| S | ∃ a shared variant powered in both traits, regardless of other variants. |
| D | No shared variant is non-absent in both traits; each trait has ≥ 1 powered variant; r² of the closest powered cross-trait pair < 0.95. |
| U | Like D, but that r² ≥ 0.95. |
| W | Some trait has all of its causal variants absent. |
| G | Everything else. |

Write it as one function `oracle_class(θ)` with unit tests on hand-built cases.

**Training targets.**
- Train a 4-way head {S, D, W, G}. Train U as D, or drop U from training, but say which.
- Remove the undefined "soft labels for G". Soft targets are legitimate only if y(θ) is a fixed, pre-registered function, because cross-entropy is then minimised at E[y(θ) | x]. A 4-class head is simpler and lets P_G be reported.
- Adapter: P_S and P_D as output; insufficient = P_W + P_G.

**Why.** See E2. Labels define what C estimates, and must be frozen before any training.

### 3. [MUST] Redefine the kill-tests C0/C1/C2 so they are well-posed and run cheapest first (§5 C "Kill-tests"; §6)

**C0 (hours, no dependency on A or B).** "coloc.abf world".
- Simulate exactly coloc's generative model: ≤ 1 causal variant per trait, the coloc prior, effect prior W = 0.15² in z-units, ρ = 0.
- Train the NPE with H0–H4 indicator targets.
- Under that model, coloc.abf's PPs are the exact Bayes posterior, because a single-causal RSS BF depends only on z_j.
- Pass: expected KL(p_abf ‖ p_NPE) ≤ 0.02 nats, and the 95th percentile of |ΔPP4| < 0.05 on loci with PP4 ∈ [0.05, 0.95], on held-out-component loci.
- This tests the plumbing (selection, aggregation, capacity), not LD use.

**C1 ("B world").**
- Simulate from ExactColoc's prior (≤ 2 per trait, L0, ρ ∈ {0, 0.3}).
- Use targets over B's hypothesis set.
- Compare with B's exact posterior on 5,000 validation-component loci, using the same metrics, with a training budget of ≤ 3 GPU-h.
- Failure means the architecture cannot represent LD-dependent posteriors: stop C.

**Shortcut and floor checks, run alongside C1.**
- (i) A no-LD ablation, i.e. the same network with the attention bias removed. If it matches the full model, the network is not using LD.
- (ii) A gradient-boosted classifier on about 20 hand features: lead–lead r², conditional z of each lead given the other, coloc.abf PPs, max |z|, n_ref, ρ̂.
- If (ii) is within 1 SE of the transformer on C2, the transformer is not needed. The GBM then becomes C's representative, which is still learned and still bold-ish, rather than the transformer.

**C2 (as now).** AUPRC ≥ best of A/B on the realistic validation grid. Add two pre-registered conditions:
- the AUPRC gap between train-component and validation-component loci must be ≤ 0.02;
- ECE is compared per change 4.

**Why.** See E3. The brief also asks for "cheapest test first". C1 as written cannot run until B exists. C0 can run in M5 week 1.

### 4. [MUST] Fair calibration comparison (§8.2 ECE; §0 C3; §5 C "Calibration"; C2)

**What.** Apply one identical post-hoc recalibration step to every method:
- temperature scaling on logit(s), or the 3-class logits;
- fitted on a validation-tune subset, with the power-matching weights of §2.4;
- report ECE both raw and recalibrated.

Also report ECE stratified by L-level × ρ.

**Why.** C is fitted to the simulation mixture, and A/B can tune priors on validation. Baselines like coloc carry fixed priors. Raw-ECE comparisons are therefore partly "who saw the simulation prior", not method quality.

A single temperature cannot fix condition-dependent miscalibration across L-levels, so stratified ECE is the honest number. Fit the temperature with the same power-matching weights used in evaluation. Otherwise C is calibrated on one population and scored on another.

### 5. [MUST] Cover the real input range, or abstain outside it (§2.4 effect sizes; §5 C "Risks")

**Problem.** Training uses λ1 ≤ 40 and λ2 ≤ 25. The R6 showcase loci (bilirubin–UGT1A1, LDL–SORT1 in FinnGen R12) and strong eQTLs exceed this. A neural network's behaviour outside its training support is undefined; Bayesian A/B extrapolate gracefully.

**What.**
- Set the training λ upper bounds from R1: the 99.9th percentile of real lead |z| for each trait type, applied before any test use. The trait-1 bound is likely ≥ 100.
- Add a support check (max |z|, n_ref, p, ρ̂ within the training ranges).
- Outside the support, C returns "fallback" and the pipeline uses B or A. Report the fallback rate.

---

### 6. [SHOULD] Fix the ρ̂ estimator and train on ρ̂, not ρ (§5 notation; §5 C "Training")

**What.**
- Estimate ρ by bivariate-normal MLE truncated to the |z| < 2 box, or use the LDSC cross-trait intercept from genome-wide null SNPs.
- In training, feed the NPE the output of the *same* estimator applied to simulated null SNPs, with realistic noise, instead of true ρ.
- Train on continuous ρ ~ U(−0.6, 0.6), not on the four grid values.

**Why.** See E4. Feeding the estimator's output in training also lets the network learn any residual bias in ρ̂.

### 7. [SHOULD] Make the training prior explicit and separate from the evaluation strata (§2.4)

**Problem.** D's second variant fills r² bins uniformly up to 0.95, so 20% of D loci have r² ∈ [0.8, 0.95). These are close to unidentifiable from S: JLIM puts separability loss above 0.8, and 11–13% of SNPs have their best proxy in that band (check 3). The NPE will learn to hedge P_S on nearly every S locus. That is correct under this prior, but the prior is arbitrary. coloc-type methods implicitly use a prior with ~0 mass there, so three-way-call and ECE comparisons depend on this design choice.

**What.**
- Use a realistic r² prior for training, e.g. the empirical r² distribution of conditionally independent eQTL Catalogue signal pairs, restricted to train components.
- Keep the uniform-bin construction for the stratified *evaluation* sets only.
- Train with balanced classes, and apply logit adjustment log(π_target/π_train) at inference. Cross-entropy classifiers learn likelihood ratios relative to the training class prior, so the class prior can be swapped post hoc. Within-class nuisance priors (λ, number of variants) cannot be swapped this way.
- Pre-register the real-data class prior before R2–R6.

### 8. [SHOULD] Test generalisation to an unseen mismatch mechanism (§2.3; §5 C "Training"; §8.2)

**What.**
- Hold L3 (ancestry mix) out of C's training, and of A/B's validation tuning of η, in a pre-registered secondary run. Report ΔAUPRC and ΔECE on L3.
- Draw n_ref continuously (mosaic subsamples 100–5,000; pool-B subsamples 50–200) instead of using two fixed references, so the network cannot memorise pool-B idiosyncrasies and still behaves sensibly at n_ref = 503 on real data.
- Add *observable* mismatch diagnostics as global inputs: the DENTIST-S/SLALOM statistic of the lead and η̂.

**Why.** If test-time mismatch uses the same mechanisms as training, C's "learned robustness" is in-distribution. Real data (R4) is a different mechanism, so a held-out-mechanism test is the honest proxy.

### 9. [SHOULD] Pin down the architecture with the right inductive biases (§5 C "Network")

**Specifics.**
- 6–8 layers, d = 192–256, 8 heads (about 3–6M parameters), M = 256 tokens.
- A per-head learned attention bias φ_h(r_ij, r²_ij, |pos_i − pos_j|) using *signed* r.
- Token features: z1, z2, the selection flags, N_j/N_max, MAF, conditional z given each trait's lead, single-SNP Wakefield log ABF for each trait, and a missing-in-reference mask.
- Global token: n_ref, ρ̂, p, the mismatch diagnostics.

**Equivariances, via augmentation or canonicalisation.**
- Per-SNP allele flip: (z1j, z2j, R_j·, R_·j) → −.
- Independent sign flips of z1 or z2 as a whole.
- No sinusoidal positional encodings.

**Uncertainty and budget.**
- Use the two training seeds as a deep ensemble, with disagreement as an OOD signal.
- Budget estimate: about 4e9 FLOP per locus for forward and backward together, i.e. ~10⁴ loci/s on a 4090 in bf16. A 12-h run sees about 10⁸ loci, so data generation, not the network, is the bottleneck.
- Report learning curves against the number of *distinct training windows*. This is the quantity that controls LD generalisation, not the number of draws.

**Why.** Softmax attention averages with positive weights. It cannot easily express the signed, subtractive conditioning (z_j − r_jk·z_k) that the exact posterior uses. Giving those statistics as features is cheap and removes a known representational bottleneck.

### 10. [SHOULD] A simulator-vs-reality check for C's inputs (§2.5 R1)

**What.**
- Train a classifier two-sample test (C2ST) to separate simulated from real network input tensors: eQTL Catalogue × FinnGen loci, train components only.
- Pre-register AUC ≤ 0.75 as the bar for trusting C on real data. Above it, revise the R1 grid before test use, or report C's real-data results as unreliable.
- Add a per-locus OOD score (ensemble disagreement plus embedding Mahalanobis distance) to the fallback rule of change 5.

**Why.** This is the standard check for amortised inference under misspecification, and it is cheap.

### 11. [SHOULD] Keep NPE tuning loci disjoint from the screening set (§6.1, §3 "validation")

**What.** Split validation loci into *val-tune*, used for early stopping, architecture choice, temperature and conformal, and *val-screen*, the frozen 500-per-cell set used for kill decisions. Use different seeds and windows; this split is at locus level.

**Why.** Architecture search on the kill-test set makes C's kill decisions optimistic. This applies to the A/B/baseline tuning grids too.

---

### 12. [COULD] Conformal as a secondary decision rule (§1 three-way call)

**What.** Mondrian (class-conditional) split-conformal sets over {S, D, W, G}, with "insufficient" ⇔ the set is not a singleton S or D. Report it alongside the fixed 0.8 rule.

**Caveat.** The coverage guarantee holds only for exchangeable simulated loci, not real data, so do not advertise it as a real-data guarantee.

### 13. [COULD] Calibrate the per-SNP shared-PIP head (§5 C outputs)

**What.**
- Report a reliability curve of PIP against realised causal-shared frequency.
- Report shared-CS coverage. Score it as a miss when the causal variant was not selected, so selection loss is visible.

### 14. [COULD] Seeds for on-the-fly generation (§9)

**What.** Derive all training-stream seeds with `numpy.random.SeedSequence(20261029).spawn(n_workers)`, and likewise for the second replicate seed. Log the worker count, so "no other seeds" stays literally true and runs are reproducible.

### 15. [COULD] Fix the Kronecker order in Family B's Σ_alt (E5)

---

## Can NPE learn components inside A/B? Yes; three candidates, ranked

1. **Amortised mismatch correction.**
   - A small network maps (z_top, R_ref,top, n_ref, diagnostics) to a per-locus η, or to a per-SNP outlier probability: a "learned DENTIST".
   - It is trained on simulations where R_true − R_ref is known.
   - A/B stay exact Bayesian models; only a nuisance parameter is learned. Best risk/return.
2. **Learned recalibration of A/B outputs.**
   - A monotone map of logit P_S, P_D conditioned on *observable* covariates only: n_ref, ρ̂, the mismatch statistic, lead |z|. Never on the simulation L-level, which is unknown on real data.
   - Cheap; improves ECE under mismatch without touching ranking.
3. **Amortised screening for B.** A per-SNP classifier "could be causal", used to choose B's enumeration set. It gives speed only, and its success criterion is recall.

Learned *priors* (π1, π2, π12, U) are better fitted by empirical Bayes on real data (mash-style) than by NPE. On simulation the true prior is known, so a learned prior can only win on real data, where it cannot be validated except through R2/R3.

**Cheapest kill-test, run before training any network: the oracle-ceiling test.**
- Run B (and A) at L2/L3/L4 on val-screen with the *oracle* nuisance plugged in: the η that best whitens R_true vs R_ref, or the true outlier mask, or the true ρ.
- If oracle-corrected B improves AUPRC over η-by-ML B by less than 1 bootstrap SE, and improves ECE by less than 0.01, no learned component can help. Kill it without training.
- This costs one B run per arm, i.e. GPU-minutes.
- For candidate 2, the analogous test is recalibration conditioned on the *true* L-level. If even that does not reduce stratified ECE, kill it.

## Areas needing no change

- The generator algebra (§2.2): Cov(e1) = GᵀG/N_syn = R_true; Cov(e1, e2) = ρR. Correct.
- The overlap formula ρ = N_s·r_p/√(N1·N2). Correct.
- The power thresholds. Correct; see check 1.
- Training C only on train-split LD, with screening and validation on held-out components. Correct.
- Temperature scaling as the primary calibrator. Adequate, given change 4.
