# Family B (ExactColoc), M3 report

Author: Family B agent. Date: 2026-10-26. Code: `src/famB/`, tests: `tests/famB/`. README with all decisions:
`src/famB/README.md`.

## 1. What was built (DESIGN v3 §5 B)

| Part | Where | Status |
|---|---|---|
| Support-space BF, BF = \|I+DΦ\|^-½ exp(½ bᵀ(I+DΦ)⁻¹Db), Φ = V⁻¹⊗R̃_uu, b = (V⁻¹⊗I)z_u built over the union for both traits, then restricted to the active (trait, variant) pairs. Evaluated in the symmetric form A = I + LᵀΦL (D = LLᵀ, so rank-1 D works), with a closed-form unrolled 4×4 Cholesky, fp64 | `enum.kernel` | done, tested |
| Enumeration of ≤ 2 causal per trait (also ≤ 1), batched over loci of equal M on GPU (torch), streaming in chunks, fp64 logsumexp per hypothesis category; optional `torch.compile` | `enum.enumerate_batch` | done |
| Screen: per-trait union of top marginal \|z\| (M/2 − M/8 per trait) + lead-removed top SNPs (M/8 per trait, standardised conditional z), round-robin, strength-aware duplicate collapse max\|z\|²(1−r²) < 1 with multiplicity carried into the prior; R̃ = (1−η)R + ηI, η ≥ 1e-3 | `screen.screen` | done, tested on real LD |
| Priors: coloc mode (1 causal per trait, ρ = 0, W²/SE²_j, p1/p2/p12, full p); Gaussian mode with per-locus EB of (σ1, σ2, η) by single-trait marginal likelihood on a grid; simulator-prior mode (type table × K-component scale mixture, 8^k component terms per configuration) | `priors.py` | done |
| Outputs: PP over H0, H1, H2, H3, H4 and H34 ("H3+4" = shared + trait-specific; exact); oracle-event posteriors P_S, P_D, P_U, P_G, P_W, s, E1, E2 by Monte Carlo (terms drawn ∝ posterior mass, one exact draw from each term's Gaussian effect posterior, oracle class per draw) | `events.py`, `method.py` | done |
| Method API (INTERFACES §3/§4): `ExactColoc(mode=...).fit_batch(list[LocusData]) -> list[dict]` | `method.py` | done |
| Adapter to the real L0* world: `priors.sim_prior_from_l0star()` reproduces `src/sim/l0star.py`'s prior exactly | `priors.py` | done, tested |
| Throughput benchmark | `python -m famB.bench` | done |

No stubs of other modules were needed: `eval.io.LocusData`, `sim.oracle` and `sim.l0star` already existed and are
used read-only.

## 2. Tests (all run; 41 passed)

Command (log: `logs/famB_tests_M3.log`):
```
source env/activate.sh; PYTHONPATH=src .venv/bin/python -m pytest tests/famB -v -s -p no:cacheprovider
=> 41 passed in 146.86s
```

| §7.1(4) item / extra | Test | Result |
|---|---|---|
| Support-space BF vs full-form RSS BF, < 1e-8 relative; ρ ∈ {0, .4, −.7}; D-type, S correlated, S rank-1 (±), 2+2 distinct, 2+2 shared, rank-1 2+2, single-trait; p = 40, regular R | `test_bf.py::test_support_equals_full_form_regular` | worst rel. error 6.9e-14, 8.4e-14, 1.6e-14 |
| Same vs 60-digit mpmath full form for near-singular R̃ (r² = 0.985 pair), rank-deficient R (n = 15 < p = 20, η = 1e-3), exact duplicate SNP; all 3 ρ | `test_support_equals_mpmath_near_singular` (9 cases) | all < 1e-8 |
| Negative control: b built from active z's only (the error flagged in round 2) | `test_negative_control_active_only_b_is_wrong` | differs by > 1% (test is sensitive); ours matches |
| coloc.abf identity: coloc mode through the general engine (full p, 1 causal/trait) vs R coloc 5.2.3 `coloc.abf` (env/renv Rscript), 200 loci (p 30–600, null/t1/t2/shared/distinct/weak-shared), p12 ∈ {1e-5, 5e-6} | `test_coloc_identity.py` | max \|ΔPP\| = 3.6e-14 and 3.3e-14 (bar 1e-3) |
| Hand-computed 2-SNP toy (all 81 configurations by closed-form eigen-decomposition; PP per category and PIPs); ρ ≠ 0 single shared variant vs hand 2×2 marginal and conditional-likelihood derivation | `test_toy2snp.py` | agreement < 1e-12 |
| Screen recall ≥ 0.99 of causal (or r² > 0.95 proxy), stratified by λ1/λ2, real 1000G EUR LD | `test_screen_recall_1kg.py` | see §3 (PASS at L0 for every powered stratum) |
| Screen unit tests: λ = 150, r² = 0.99 D pair keeps both SNPs (round-2 review); exact duplicates collapse with multiplicity 4; masked secondary found by the lead-removed list and missed without it (negative control) | `test_screen.py` | pass |
| Oracle logic on constructed draws (S, D, U, D at d = 15, W, G via grey shared, D with weak shared, cancellation, grey, null) and partition on 20k random draws | `test_events.py` | pass |
| Oracle == `sim.oracle.oracle_class` (hash-protected), 5×10⁴ random draws, all 5 classes present | `test_sim_integration.py::test_oracle_matches_sim` | identical labels |
| Our L0* type table == `sim.l0star.config_logprior` for every configuration, p = 7 and 30 | `test_config_prior_matches_l0star` | max diff < 1e-12 |
| Positive control, exact-model toy L0* (own simulator, 800 loci, p = 8, K = 8, ρ ∈ {0, .4}): calibration of all 6 PP categories and S/U/D/W/G/E1/E2, overall and in probability bins | `test_calibration_l0star.py::test_simulator_mode_calibrated` | max \|z\| = 2.42 over 33 checks (bar 4) |
| Same for Gaussian mode (1000 loci, ρ ∈ {0, .4, −.7}, rank-structured shared prior ρ_b = 0.5) | `test_gaussian_mode_calibrated_and_rho_matters` | max \|z\| = 2.6 over 33 checks |
| Negative control: ignoring the overlap correlation ρ on ρ ≠ 0 loci | same | Brier 0.313 (ρ used) vs 0.343 (ρ ignored) |
| Simulator mode on loci drawn by **the sim agent's L0\* generator** (`sim.l0star.sample_locus`, 750 loci, p = 10, AR and genotype LD, ρ ∈ {0, .4}) vs sim's oracle labels | `test_simulator_mode_calibrated_on_sim_l0star` | calibrated (S 50.4 pred / 48 obs; D 42.2/43; W 579.7/574; G 77.8/85; \|z\| ≤ 1.1); log score −0.203 vs prior −0.791 |
| Method controls (§7.1(9)): λ = 30 shared at r² = 1 → s > 0.9 (also with the two traits on two identical SNPs); distinct → s < 0.1, P_D > 0.9; null and trait-1-only → P_W > 0.95; E1/E2 directions; dense == factor LD; output columns; failure reported as status=failed | `test_method.py` | pass (gaussian and coloc modes) |
| Engineering: compiled == eager (rect and mixture paths, < 1e-12); generic path == rectangular path for K = 1; fp32 kernel within 1e-4; sampler regression test | `test_method.py`, `test_bf.py` | pass |

**Bug found and fixed during testing.** In batches that mix a very strong locus (log Z ≈ 3,400) with weak ones,
per-row log-masses underflowed to −inf for some rows; the NaN rows broke the concatenated CDF of the two-stage
sampler, and ~2% of draws landed on terms with posterior mass ≈ e⁻¹⁰⁰ (seen as a 0.02 shared-PIP error). Fixed
(logsumexp row masses, zero-mass rows guarded). The regression test fails on the old code and passes on the new.
The exact PPs were never affected; only Monte-Carlo event posteriors in mixed batches were.

## 3. Screen recall on real LD (pre-registered test)

40 random 1-Mb windows from train-split pyrho blocks on chr15–22 (median p = 3,143 SNVs after §2.1 QC), pool A
(254 NFE) genotypes, z_t = Rλ_t + Xᵀu_t/√n (in-sample, L0). 1,000 loci (35% S ± extra, 45% D with the partner
drawn uniformly over r² bins up to 0.99, 20% 2+2), |λ1| log-uniform [3, 150], |λ2| [3, 60]. "Kept" = causal is a
representative, collapsed into one, or has an r² > 0.95 proxy in the screen (true LD).

| | M = 100, trait 1 | M = 100, trait 2 | M = 50, trait 1 | M = 50, trait 2 |
|---|---|---|---|---|
| powered (\|λ\|, \|μ\| ≥ τ), L0 | **0.999** (n = 1,035) | **0.998** (1,014) | 0.998 | 0.998 |
| \|λ\| 6.29–10 / 10–20 / 20–50 / 50–150, L0 | 0.993 / 1 / 1 / 1 | 1 / 0.996 / 0.997 / 1 | 0.993 / 1 / 1 / 0.997 | 1 / 0.996 / 0.997 / 1 |
| grey \|λ\| 4–6.29, L0 | 0.993 | 0.985 | 0.987 | 0.960 |
| weak \|λ\| 3–4, L0 | 0.90 | 0.84 | 0.82 | 0.81 |
| powered, L2-like (screen with pool-B LD, n_ref = 150) | 0.971 | 0.975 | 0.956 | 0.968 |

The assertion (≥ 0.99 in every powered λ stratum at L0, M ∈ {50, 100}) passes. Weak causal variants are below
0.99 by construction (their |z| is at noise level). **Under a small external reference the screen loses ~3% of
powered causal variants.** These are secondary signals in moderate LD with a strong lead, which only the
lead-removed list finds. With n_ref = 150 and a lead at z ≈ 65, the conditional statistic has noise SD ≈ 5 at
every SNP. A variance-inflated conditional statistic (`n_ref_infl`) did not help (0.971 → 0.971), so it is off
by default.

## 4. Speed

`python -m famB.bench` (synthetic AR(1) LD, p = 2,000, n = 64 loci per run, S/D/W mix, n_mc = 16,384, EB on).
Full pipeline per locus: screen, EB grid, enumeration, MC events. Three repeats, one RTX 4090 shared with other
agents (GPU utilisation before the repeats: 0%, 48%, 11%). Log: `logs/famB_bench_M3.log`.

| Mode | M | Configs per locus | Eager fp64 | Compiled fp64 | Compiled fp32 kernel (fp64 sums) |
|---|---|---|---|---|---|
| Gaussian, ≤ 2/trait | 50 | 1.6 M | 59 / 69 / 73 loci/s | **97 / 91 / 103** | 108 / 119 / 120 |
| Gaussian, ≤ 2/trait | 100 | 25.5 M | 4.7 / 4.5 / 5.5 | **11.1 / 12.0 / 12.6** | 18.3 / 20.3 / 21.4 |
| coloc mode (full p = 2,000, 1/trait) | – | 4.0 M | 35 / 37 / 35 | – | – |

- Real 1000G windows (factor LD, p ≈ 3,100, eager): M = 50 at 0.018–0.023 s/locus, M = 100 at 0.19–0.20 s/locus.
- Simulator-prior mode with the exact `sim.l0star` prior (K = 6 after dropping zero-weight components), compiled:
  p = 20 at 0.13 s/locus, p = 30 at 1.2 s/locus (2.5×10⁸ terms). 2,000 L0\* loci at p = 30 take ≈ 40 GPU-min.
  The toy K = 8 prior: p = 10/20/30 at 0.09/1.2/6.4 s/locus compiled; eager is about 5× slower at p = 20.
- `torch.compile` costs ≈ 20 s once per process. Use it for production runs; tests run eager.

For orientation only: the round-2 estimate for coloc-SuSiE is 1–5 s/locus/core, i.e. ≈ 13–64 loci/s on 64 cores.
Gaussian M = 50 on one GPU (~100 loci/s) is in that range or above. M = 100 (~12 loci/s) is not. This is not a
C2 measurement: it uses different LD and no per-trait amortisation.

## 5. Exploratory sanity on real LD (not the screening set; no claims)

360 loci on 12 train windows (S vs D, D partner r² < 0.8, λ 6–40). L0 uses in-sample LD; L2-like uses pool-B
reference LD. Log: `logs/famB_explore_realLD.log`.
- All modes reach AUROC(s, S vs D) ≥ 0.996. Confident false S (P_S > 0.8 on D) is 0 in all modes.
- S recall at P_S > 0.8: coloc mode 0.81 at both levels. Gaussian M = 100: 0.96 at L0, 0.93 at L2-like.
- EB η: 1e-3 on 100% of L0 loci; above 1e-3 on 100% of L2-like loci (mismatch detected).

These loci are easy. The kill-test B1 (D r² 0.3–0.8 and M strata) has not been run.

## 6. Deviations from DESIGN v3 / interpretations

1. **E1/E2.** "CS₁" is replaced by the posterior set of trait-1 causal variants that are non-weak in trait 1. The
   threshold is w₂ (power 0.2), per round-2 review M2, not τ₂.
2. **P_W** output = 1 − P_S − P_D, absorbing U and G as in Family A. The strict P_W, P_U and P_G are in `extra`.
3. **s** = (P_S + ε)/(P_S + P_D + 2ε) with ε = 0.5/n_mc. An MC-free alternative, `s_pp` =
   (PP4 + PP34)/(PP3 + PP4 + PP34), is in `extra`.
4. **r²** of the class-determining pair is taken from R̃, to match `sim.l0star`'s oracle.
5. **Collapse rule.** max\|z\| is read as the window maximum over both traits (the conservative reading).
6. **Screen prior.** Variants outside the screen get prior mass 0. coloc mode uses the full p, so the identity
   test is unaffected.
7. **No layer-E mismatch model yet.** B's mismatch handling is EB η only. §5 B's "variance-inflated mismatch
   models → full-z_M likelihood (rank-|u| updates)" is not implemented.
8. **ρ̂** is used as given (clipped to ±0.95). rho_se is not used.

## 7. Open issues / what the PI must decide

1. **Default prior for B in screening.** Gaussian-EB is the default. The per-SNP p1/p2/p12 (1e-4/1e-4/5e-6) are
   untuned. Should p12 (or the S:D prior odds) be tuned on validation, as W is for coloc.abf?
2. **Default M.** M = 50 runs at ~100 loci/s with powered recall 0.998 at L0. M = 100 runs at ~12 loci/s with
   0.999. I propose **M = 50** for screening, with M = 100 as an ablation. Under L2-like LD, M = 100 recovers more
   (0.971 vs 0.956).
3. **L0\* window size.** p = 30 costs 1.2 s/locus (exact `sim.l0star` prior). p = 20 costs 0.13 s/locus. Either
   is feasible for ~2k loci. The simulation agent should confirm that its L0\* sets store R̃, so that B runs with
   `eta=0, eta_floor=0`.
4. **Mismatch.** With a small reference, the screen misses ~3% of powered secondaries, and the model has only η.
   If B is to compete at L2/L4, layer E (reference-size variance) needs implementing in the likelihood. Kill-test
   B1 at L2 will tell whether this matters.
5. **Proposed INTERFACES/DESIGN notes** (PI to decide; I did not edit them):
   - §5 B could record that E₂ uses w₂ and the posterior non-weak causal set.
   - The "H3+4" category could be defined explicitly as shared + trait-specific.
   - `runtime_s` for batched GPU methods = CPU prep + an even share of the batch wall-clock.

## 8. Files

- `src/famB/{__init__,constants,enum,events,screen,priors,method,bench}.py`, `src/famB/README.md`
- `tests/famB/{conftest,ref,toy_data,toy_l0star,kg}.py`, `tests/famB/coloc_abf_batch.R`,
  `tests/famB/test_{bf,coloc_identity,toy2snp,events,screen,screen_recall_1kg,calibration_l0star,method,sim_integration}.py`
- Logs (not committed): `logs/famB_tests_M3.log`, `logs/famB_bench_M3.log`, `logs/famB_simmode_timing*.log`,
  `logs/famB_explore_realLD.log`
