# Review round 2: algorithms and HPC/GPU (DESIGN.md v2)

**Verdict: minor revision.** v2 takes up most of my round-1 changes correctly. There are five must-change items. Each is a local fix to the spec, not a redesign, but each of them would make results wrong or a claim untestable if left as written:

- the (N1, N2, ρ) overlap grid is mostly infeasible, and the nested-overlap rule has the wrong scaling;
- the near-duplicate collapse threshold erases identifiable D pairs;
- the simulator-prior (ceiling / C1) mode of B is about 5 min per locus;
- C2 has no accuracy or amortisation definition;
- baseline timeouts are unspecified, and the baseline CPU budget is about 3–6x too low.

## Checks run this round (throwaway scripts in my /tmp scratchpad, nothing committed)

| # | Check | Result |
|---|---|---|
| V1 | Support-space BF (§5 B) vs full 2p-dim marginal likelihood. Setup: p=40, n=30 (rank-deficient R, a near-duplicate pair), R̃ = R + 0.05I, ρ=0.4, 3-dim support with a correlated shared effect. | Support form = full form to 7e-15. The v2 union-u form agrees too, *if* D is zero-padded and evaluated as D(I+ΦD)⁻¹. As written, (D⁻¹+Φ)⁻¹ is undefined on the padded entries. When the mean uses R and the noise uses R̃, the R̃-only support form is wrong (log BF 25.60 vs 24.73). The general form below is exact (24.7293 = 24.7293). |
| V2 | coloc 5.2.3 `approx.bf.estimates` | r = sd.prior²/(sd.prior²+V). The z-scale prior variance is therefore **W²/SE_j²**, not W/SE_j² as written in §5 B (coloc mode). |
| V3 | GPU generator, factor form: int8 G (N_syn=20,000 × p=4,000, 80 MB), standardisation applied after the GEMM, fp32 | 1.2 ms per 256 draws, about 2×10⁵ loci/s. |
| V4 | GPU generator, individual-level, N2=5,000: row gather + standardise + Gᵀy, 256 loci | 192 ms, about 1.3×10³ loci/s. Peak memory 3.9 GB. |
| V5 | fp32 vs fp64 SER PIP at λ = 10/40/150 with r≈0.9996 proxy groups | max \|ΔPIP\| ≤ 1.1e-5 per SNP. The §7.1 #10 tolerance is fine. |
| V6 | Baseline availability (Q3) | susieR master **0.16.6** has `R_finite` and `R_mismatch ∈ {none, eb, eb_mix, eb_ser_init, …}` (checked in source `R/susie_constructors.R`). Interface confirmed. **Not installed yet:** susieR 0.16 (`env/rlib_susie016` is empty), Rfast, mvsusieR, colocboost, colocPropTest, prop.coloc. Installed: susieR 0.14.2 and coloc 5.2.3 only. |
| V7 | Disk and data | 12.4 M variants in the processed pgens (indels included, MAF ≥ 0.005). Scratch has 91 TB free. GPU is idle (24 GB). |

**Generator feasibility, as asked.**
- At 10⁵–10⁶ loci, the generator costs at most about 13 GPU-minutes (V3/V4). It can share the GPU with NPE training.
- The binding constraint is memory, not FLOPs. The genome-wide mosaic is about 180 GB as int8 (45 GB 2-bit packed), and the train split alone is about 110 GB int8. That fits neither the 24 GB GPU nor the 128 GB cgroup. Windows must therefore be streamed from scratch, with ≥ 256 draws per window load (R1-alg 9, still valid).

## 1. Round-1 must-changes (mine)

| R1 item | Status | Reason |
|---|---|---|
| 1. Fix E1 (generator scaling) and E2 (Kronecker); support-space BF | **Partially** | E1 and E2 are fixed. The formula is correct numerically (V1). But D⁻¹ over the union u is undefined as written, the coloc prior is not squared (V2), and the mismatch case says only "rank-\|u\| updates", with no formula. See change 5. |
| 2. Budget-feasible exact mode | **Partially** | ≤ 2 per trait, M ≤ 100 and the ≤ 3 restriction are in. Three problems remain: the duplicate-collapse threshold r² > 0.99 is wrong (change 2), the proxy-expansion cap has no number, and the new simulator-prior mode is about 10³x more expensive than plain enumeration (change 3). |
| 3. C2 and the runtime protocol | **Partially** | Present: flock, 1-thread BLAS, memory-limited workers, Rfast, synchronize, LD I/O, median/p95 per level, p-scaling, and two workloads. Missing: the definition of "accuracy equal or better", amortisation accounting, the engineering parity of G, host-prep and cold-start rules, the workload's LD source, and which susieR version is the speed baseline (change 4). |
| 4. LD as genotype factor; bucketing by p; η from thin SVD | **Addressed** | §5 A Engineering. |
| 5. Precision policy | **Addressed** | §5 A and §7.1 #10. `NVIDIA_TF32_OVERRIDE=0` is set in `env/activate.sh`. Not stated: the NPE LD-feature transform (atanh r or log(1−r²) before any bf16 cast). Could. |

Round-1 "should" items: 6 is addressed (G first, C1 learning curve, simulator-prior mode). 11 is addressed apart from the square (V2). 7, 8 and 12 are not addressed. 7 and 8 matter for C2 and are folded into change 4.

## 2. New problems in v2 / missed in round 1

### N1. The overlap grid is largely infeasible, and the nested-overlap spec is mis-scaled (§2.2, §2.4, §6.1; threatens C4 and the realistic headline)

**Feasibility.** With N_s ≤ N2, ρ = N_s r_p/√(N1N2) ≤ r_p√(N2/N1).

| (N1, N2) | Max \|ρ\| (r_p = 1) |
|---|---|
| 20k, 5,000 | 0.50 |
| 20k, 1,000 | 0.22 |
| 100k, 5,000 | 0.22 |
| the other 9 combinations | < 0.13 |

So ρ = 0.8 is impossible for every (N1, N2) in the grid, and ρ ∈ {0.3, 0.5} is possible in at most one combination. The §6.1 screening cells at ρ = 0.5, the realistic pooled headline (ρ up to 0.8) and the C4 slope all assume cells the generator cannot produce. If the generator conditions N2 on ρ to make the cells feasible, the C4 slope is confounded with trait-2 power.

**Mis-scaling.** In the factor representation, each of the N_syn rows of u stands for N1/N_syn real individuals. Correlating ε with u at r_p on the shared rows therefore gives ρ_eff = r_p·N_s/√(N_syn·N2). That is √(N1/N_syn) too large: up to 4.5x at N1 = 400k. The correct per-row correlation is **c = r_p·√(N_syn/N1)** with u ~ N(0, I). The §7.1 empirical-ρ test would catch this, but the spec should already be right.

### N2. The duplicate-collapse threshold r² > 0.99 removes identifiable D pairs (§5 B screen, shared by C)

- At λ = 150 (the biobank-strong stratum) and r² = 0.99, d = λ√(1−r²) = 15 ≫ 2. This is an identifiable D pair by the §1 definition, not U.
- Collapsing the two SNPs into one token forces B, and C through the shared screen, to call S.
- The error hits exactly the high-r² D strata that C1 and C3 are stratified on.

The collapse rule must depend on signal strength.

### N3. Simulator-prior mode (ceiling, and the C1 reference) is computationally infeasible as specified (§5 B, §5 C C1, §10 M5)

**Enumeration cost.** Each causal effect carries an 8-component scale mixture, so a configuration with |s| effects has 8^|s| Gaussian terms. The total number of 4×4 solves is (Σ_k n_k·8^k)², with n = (1, M, C(M,2)):

| M | 4×4 solves per locus | GPU time per locus (at about 3×10⁸ solves/s, R1 K5) |
|---|---|---|
| 100 | 1.0×10¹¹ | about 5.5 min |
| 30 | 7.9×10⁸ | about 2.6 s |
| 20 | 1.5×10⁸ | about 0.5 s |

**Class integration.** The oracle classes are joint events over |λ| and μ = R_tλ under a precedence partition (S > U > D > W > G). Their posterior probabilities are not closed-form, so they need Monte Carlo over each configuration's Gaussian posterior.

**Not a ceiling at L0 as specified.** Methods receive trait 1's LD only, while trait 2's in-sample LD differs from R_pop by about 1/√N2 per entry (0.1 at N2 = 100). A "Bayes-optimal ceiling" must use each trait's own R_t. At ρ = 0 the likelihood factorises per trait, so the support form stays exact with a per-trait R_t. At ρ ≠ 0 with R_1 ≠ R_2, the Kronecker structure is lost.

### N4. C2 is untestable as written, and C2(b) is structurally unattainable for A and B (§0, §7.7)

**Untestable.**
- "Accuracy must be equal or better" has no metric, margin or dataset. Workloads (i) and (ii) are genome-scale with no labels.
- "Amortised per-trait fine-mapping allowed" does not say whether the precompute time is counted, or whether externally shipped eQTL Catalogue LBFs may be used at zero cost.

**Unattainable for A and B.**
- In workload (ii) (50 tissues), coloc-SuSiE and G need 20k + 1M single-trait fits. A needs 1M bivariate fits, and each costs at least one G fit. So A/G ≲ 1x.
- B at about 0.1 s per locus is roughly 50x slower than batched G in factor form (about 500 loci/s at p ≈ 3k).
- Only C, or a C-prefilter hybrid, can plausibly reach 10x vs G.
- G also needs the same engineering as A (factor LD, p-bucketing, continuous batching). Otherwise "(b)" measures engineering, not modelling.

### N5. The baseline CPU budget is about 3–6x optimistic, and timeouts are unspecified (§4.3, §10)

**Budget.** Tier 1 has six LD-using slow methods. Under mismatch, susie_rss alone reached 125 s per fit (R1 K7). mvSuSiE-RSS, ColocBoost and eb_mix are slower than standard susie_rss. A realistic figure is about 100–300 core-s per locus across tier 1. That gives:
- M4 (about 52k loci): about 1.5k–4k core-h, i.e. 1.5–4 days wall at about 40 workers, against "≤ 12 h" in §10;
- M7 (about 60k loci plus the shifted arm): similar.

**Timeouts.** §4.3 scores timeouts as the least favourable outcome and blocks claims above 1% failures, but no timeout value is set. Whoever sets it later controls the baselines' scores. Ad-hoc subsampling of slow methods after the fact would be a forking path.

### N6. Smaller issues

- **(should) Generator z vs oracle λ (§2.2).** The small-N z = Gᵀy/(√N σ̂_y) is √N·r. Its mean is ≈ √N·q and its sd is ≈ (1−q²), whereas the oracle uses λ = √N·q/√(1−q²) with unit noise. At trait-2 PVE 0.3 the oracle overstates E[z] by 1.20x, and the noise is 0.7, not 1. Powered labels near τ_2 are then wrong for strong small-N eQTLs. Use the per-SNP marginal OLS t-statistic as z (mean ≈ λ, variance ≈ 1).
- **(should) NPE global feature "LD eigen-spectrum".** At deployment it costs a full-window eigh per locus (132 ms at p = 5k on the GPU, R1 K6). That is 10³x the NPE forward pass, and it counts in C2. Use the spectrum of the screened M×M submatrix, or the thin SVD of the n_ref×p factor.
- **(should) Reference panels L1/L2/L3/L4/L6 are fixed per window.** This includes L4's dropped-variant set (per-locus allele flips stay per locus). LD features can then be precomputed, and storage is per window.
- **(should) Store LD as genotype matrices, not dense LD.** Store int8 genotype matrices per (window, level): pool B is about 0.5 MB per window. Dense float32 LD per locus would take 2–6 TB for about 60k test loci, plus as many files. Every method, R baselines included, builds R inside the timed region. Shard method inputs (about 1k loci per file) to respect the file quota.
- **(should) Proxy-expansion cap.** Give it a number (e.g. ≤ 20% of M). Restrict the screen-recall test to powered and grey causal variants: recall of weak causal variants is low by construction.
- **(could) Tabulate ρ̂.** Draw ρ̂ per (N1, N2, N_s, r_p) cell once (e.g. 10⁴ draws per cell) instead of running a 20k-SNP truncated MLE inside the on-the-fly NPE loop.

## 3. Proposed changes

### Must

**1. Overlap grid and generator (§2.2, §2.4, §6.1, C4).**
- Nested mode: per-row correlation c = r_p·√(N_syn/N1) on the shared rows, with u ~ N(0, I). Assert c ≤ 1.
- Add a **large-N trait-2 overlap stratum**: N2 ∈ {20k, 50k} (pQTL / biobank-trait-like). Both traits use factor form with Cov(e1, e2) = ρR_pop, so every ρ ∈ {0, …, 0.8} is feasible.
- ρ cells with |ρ| ≥ 0.3, including the §6.1 ρ = 0.5 screening cells, draw from that stratum. Small-N QTL cells use ρ ∈ {0, 0.1} only where r_p√(N2/N1) allows it, with N2 sampled conditionally.
- Pre-register the conditional distribution of (N1, N2 | ρ) in a table.
- Estimate the C4 slope **within the large-N stratum**, where N2 does not vary with ρ, so the slope is not confounded with trait-2 power.
- A §7.1 test asserts that every generated cell is feasible.

**2. Collapse rule (§5 B screen, §5 C tokens).**
- Collapse SNPs i and j only if z_max·√(1−r_ij²) < 0.5, where z_max is the larger of the two traits' window max |z|, i.e. r² > 1 − 0.25/z_max². That is r² > 0.9975 at z = 10 and r² > 0.99999 at z = 150. The alternative is the round-1 |r| ≥ 0.9999.
- Add a test: a λ = 150, r² = 0.99 D pair keeps both SNPs.

**3. Make the ceiling and C1 computable (§5 B simulator-prior mode, §5 C C1, §10).**
- Run C1 (and the M4/M5 "ceiling") on **short windows**: p ≤ 30 SNPs, drawn from real LD. B is then exhaustive with no screen, exact and cheap (about 1–3 s per locus). C must already handle variable token counts.
- Class probabilities come from Monte Carlo: about 64 posterior draws per configuration, over the configurations covering 1 − 1e-6 of the posterior mass. Validate against 10⁴ draws on 50 loci.
- The ceiling uses **each trait's own in-sample R_t at ρ = 0**. Call it "the Bayes ceiling under the restricted ≤ 2-per-trait prior", not Bayes-optimal for the full mixture.
- Budget: about 2k loci × 3 s ≈ 2 GPU-h.

**4. Freeze the C2 accounting (§0 C2, §7.7).**
- **(a) Accuracy.** Non-inferiority: the lower 95% CI bound of ΔAUPRC (realistic, test split) is ≥ −0.01, for the *same configuration that is timed*.
- **(b) Amortisation.** All per-trait precompute is inside the timed wall-clock, computed by each method from z + LD. No externally shipped LBFs.
- **(c) Engineering parity.** G uses the same LD representation, p-bucketing and continuous batching as A. Any method may batch the K traits that share a window into one GEMM (R1-alg 7). Max-iterations = 100 for G, A and susieR. Non-convergence is reported per level (R1-alg 8).
- **(d) Workload LD.** Run both a 1000G-EUR reference (n = 503; factor form is valid) and a dense in-sample-like LD (n ≥ p). The factor-form speed-up vanishes with dense LD.
- **(e) Gating.** State whether all 20k genes are processed or only GWAS-gated windows. The same gating applies to every method.
- **(f) CPU baseline version.** C2(a)'s coloc-SuSiE uses the faster of susieR 0.14.2 + Rfast and susieR 0.16.6 (which has a C++ backend), timed in a pilot.
- **(g) What else is timed.** Cold start is reported separately. Host prep, H2D transfer and ρ̂/η estimation are included in the timings.
- **(h) Scope of C2(b).** Record now that A and B cannot meet C2(b) under amortised workloads. Either C2(b) is a claim only if the §6.3 primary candidate is C or a C-hybrid, or it is demoted to a reported ratio. Decide before any result exists.
- **(i) CPU extrapolation.** For workload (ii), the CPU baseline may be timed on a pre-registered 2% random subsample and extrapolated, with a CI.

**5. Pilot-based budget and frozen timeouts (§4.3, §10).**
- At the start of M4, time each tier-1 method on 50 loci per mismatch level.
- Then freeze a per-method timeout (e.g. max(600 s, 20x the pilot median)).
- Then freeze the per-cell test locus counts. Any subsample used for slow methods is the same random subset for all paired comparisons, fixed by seed.
- Fix the §5 B text:
  - write the BF in the support restriction s;
  - coloc-mode prior variance is W²/SE_j²;
  - the mismatch case uses the general form below.

  ```
  W = R̃⁻¹ R_{·u}    (one p×|u| solve per locus; Woodbury when R̃ = factor + ηI)
  Q = R_{·u}ᵀ W
  A_ss = (V⁻¹ ⊗ Q)_ss
  y_s  = ((V⁻¹ ⊗ Wᵀ) z)_s
  log BF = ½ y_sᵀ (D_s⁻¹ + A_ss)⁻¹ y_s − ½ log|I + D_s A_ss|
  ```

  This reduces to the v2 form when R̃ = R, and was verified exact in V1.

### Should

6. Compute the small-N z as the marginal OLS t-statistic (N6).
7. Compute NPE LD features from the screened submatrix or the factor SVD. Fix reference panels per window (N6).
8. Store references as int8 genotypes per (window, level), and shard method inputs (N6).
9. Give the proxy-cap a number, and restrict the recall test to powered and grey causal variants (N6).
10. Install susieR 0.16.6 into `env/rlib_susie016`, plus Rfast, mvsusieR, colocboost, colocPropTest and prop.coloc, at M3. No substitute is needed so far (V6).

### Could

11. Tabulate ρ̂ draws per overlap cell.
12. Set `torch.use_deterministic_algorithms(True)` where supported, and record driver/CUDA/torch versions in the manifest (R1-alg 12).
13. Use atanh or log(1−r²) LD-feature transforms in the NPE.

## Answers to §14 open questions (my scope)

- **Q1 (generator).** Acceptable and fast enough (V3, V4), once N1 is fixed. Window clipping has no compute implication beyond variable p, which bucketing handles.
- **Q2 (BF formula).** Correct (V1). The written form needs the support restriction and the W² fix. The mismatch case needs the general form in change 5.
- **Q3 (baselines).** None is infeasible. The susieR 0.16.6 `R_mismatch = "eb_mix"` interface exists. Six packages remain to be installed.
