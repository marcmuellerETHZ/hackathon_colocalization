# Family B — ExactColoc (owner: Family B agent)

Exact Bayesian enumeration of ≤ 2 causal variants per trait on a screened variant set, on GPU
(DESIGN v3 §5 B). Package `famB` (run with `PYTHONPATH=src`).

| Module | Content |
|---|---|
| `enum.py` | configuration tables, support-space BF kernel (closed-form 4×4 Cholesky, fp64), streaming two-pass enumeration, term sampler |
| `events.py` | Gaussian effect posteriors of sampled terms; oracle classes S/U/D/W/G and exclusion events E1/E2 (DESIGN §1) |
| `screen.py` | per-trait-union screen, lead-removed candidates, strength-aware duplicate collapse |
| `priors.py` | coloc / Gaussian-EB / simulator-prior (scale mixture) priors, class-prior reweighting |
| `method.py` | `ExactColoc` Method (INTERFACES §4) |
| `bench.py` | throughput benchmark (`python -m famB.bench`) |

```python
from famB.method import ExactColoc
rows = ExactColoc(mode="gaussian", M=100).fit_batch(loci)      # loci: list[eval.io.LocusData]
rows = ExactColoc(mode="coloc").fit_batch(loci)                # == coloc.abf (full p, 1 causal/trait)
rows = ExactColoc(mode="simulator", sim_prior_fn=f).fit_batch(loci)  # f(locus) -> priors.SimPrior
```

Output rows follow INTERFACES §3. `extra` (JSON) holds `PP` over H0, H1, H2, H3, H4, H34; `s_pp`;
`P_U`, `P_G`, `P_W_strict`; `eta`, `sigma` (EB picks), `rho_used`, `M`, `p`, `n_terms`, `n_mc`, `zmax`;
max PIPs (exact for single-component priors, Monte Carlo for mixtures); top shared-causal variants
(window index, Monte-Carlo P); timings.

## Model

z = [z1; z2] over the screened union u (M variants) ~ N((I₂⊗R̃)β, V⊗R̃), R̃ = (1−η)R_uu + ηI, V = [[1, ρ̂],[ρ̂, 1]].
For a configuration (S1, S2) with active coordinates s, β_s ~ N(0, D):

BF = |I + DΦ_ss|^{-1/2} exp(½ b_sᵀ (I + DΦ_ss)^{-1} D b_s), Φ = V⁻¹⊗R̃, b = (V⁻¹⊗I) z (built over u for both
traits, then restricted to s). Evaluated in the equivalent symmetric form with D = LLᵀ:
A = I + LᵀΦ_ss L (eigenvalues ≥ 1), log BF = −½ log|A| + ½ yᵀA⁻¹y, y = Lᵀb_s. Exact for rank-1 D and
near-singular R̃ (tests: < 1e-13 vs fp64 full form, < 1e-8 vs 60-digit mpmath full form).

Posterior quantities:
- **PP** over hypothesis categories (exact sums, fp64): H0 none; H1 trait 1 only; H2 trait 2 only;
  H3 both, no shared variant; H4 all causal variants shared (S1 = S2); **H34** ≥ 1 shared variant plus
  ≥ 1 trait-specific variant (the "H3+4" configurations). In coloc mode H34 ≡ 0.
- **Oracle events** (Monte Carlo): `n_mc` terms drawn ∝ posterior mass (stratified sampling: exact
  trait-1 configuration marginal, then trait-2 configuration within row), one exact draw of the effects
  from each term's Gaussian posterior, oracle class evaluated per draw. P_S, P_D, P_U, P_G, P_W_strict, E1, E2.

## Decisions / assumptions (not fixed by DESIGN v3)

1. **Oracle definitions** are a vectorised re-implementation of `src/sim/oracle.py` (DESIGN §1):
   μ_tj = (R̃λ_t)_j (the model's mean map); non-weak ⇔ |μ| ≥ w_t (4.61 / 3.58); powered ⇔ |μ| ≥ τ_t and
   |λ| ≥ τ_t; class-determining pair r² from the model LD R̃ (as `sim.l0star` does). Label-for-label
   agreement with `sim.oracle.oracle_class` on 5×10⁴ random draws is tested
   (`tests/famB/test_sim_integration.py`). Tie-breaking in max r² may differ (sim: smaller d, then ids;
   here: first pair) — only matters for exactly tied r².
2. **E1/E2 (layer D)**: E₂ = P(no trait-1 causal variant that is non-weak in trait 1 has |λ₂| ≥ w₂), i.e. the
   "CS₁" of §5 layer D is replaced by the posterior set of non-weak trait-1 causal variants, and the
   threshold is w₂ (power 0.2), per review R2-stat M2 (not τ₂). E₁ symmetric. Vacuously 1 when the
   other trait has no non-weak causal variant.
3. **Output P_W = 1 − P_S − P_D** (absorbs U and G, consistent with Family A §5 A); the strict W
   probability, P_U and P_G are in `extra`.
4. **s = (P_S + ε)/(P_S + P_D + 2ε)**, ε = 0.5/n_mc (MC resolution); undefined → 0.5 (§4.3 tie rule).
   `extra.s_pp` = (PP4 + PP34)/(PP3 + PP4 + PP34) is an MC-free alternative ranking score.
5. **Configuration prior (coloc & gaussian)**: per-SNP probabilities in the coloc convention (weight
   p1^n1 p2^n2 p12^nsh relative to the empty configuration, no p0 normalisation; needed for the exact
   coloc.abf identity). Gaussian-mode defaults p1 = p2 = 1e-4, p12 = 5e-6 (DESIGN §4.1 coloc-SuSiE value).
   A collapsed representative of multiplicity m_j gets prior weight × m_j per occurrence (counted once if
   shared). Variants outside the screen get prior mass 0 (posteriors are over the screened set; the
   screen keeps all variants with non-negligible BF in practice, see recall test).
6. **Effect scale**: priors are on the z (noncentrality) scale. Gaussian mode: λ_t ~ N(0, σ_t²), the same σ_t
   for all variants of trait t (standardised-effect prior), independent across traits for a shared variant
   unless `rho_b` ≠ 0 (default 0, like coloc). Coloc mode: z-scale variance W²/SE_tj² with W = 0.15
   (sdY = 1).
7. **η floor**: η ≥ 1e-3 (`eta_floor`), except when the LD given is already R̃ (L0* sets): then
   `eta = eta_floor = 0`.
7b. **Empirical Bayes (gaussian mode)**: σ_t and a common η chosen per locus on the grid
   σ ∈ {1.5, 3, 6, 12, 24, 48, 96}, η ∈ {1e-3, 1e-2, 3e-2, 0.1, 0.2} by maximising Σ_t ℓ_t, with
   ℓ_t(σ, η) = log N(z_t,u; 0, R̃(η)) + log Σ_c π_c BF_c − log Σ_c π_c (single-trait marginal likelihood of
   the screened z, ≤ 2 causal, π from per-SNP marginal probability p_t + p12). η floor 1e-3 always.
   This is the only LD-mismatch handling in B so far (layer E not implemented).
8. **ρ**: V uses the locus attribute `rho_hat` (NaN → 0), clipped to ±0.95; coloc mode uses ρ = 0. No
   shrinkage of ρ̂ (rho_se unused) — open item.
9. **Screen**: quotas per trait M/2 = (M/2 − k_cond) marginal + k_cond lead-removed (default k_cond = M/8,
   the "cap" on the proxy/conditional expansion); round-robin m1, m2, c1, c2; a candidate already
   represented does not consume quota (overlap back-filled from deeper in the list). Lead-removed
   statistic standardised: (z_j − r̃_jl z_l)/√(1 − r̃_jl²) (the raw residual λ(1 − r²) would bury
   secondaries in high LD with the lead). Collapse rule max|z|²(1 − r²) < 1 with max|z| over the window and
   both traits (the conservative reading). If p ≤ M: no screen, no collapse (exact; used in L0*).
   Optional `n_ref_infl` (reference-error variance inflation of the lead-removed statistic) gave no
   recall gain at L2-like LD and is off by default.
10. **Simulator-prior mode** (`priors.sim_prior_from_l0star()` gives the exact `src/sim/l0star.py` prior;
    use `ExactColoc(mode="simulator", eta=0, eta_floor=0, sim_prior_fn=lambda _: pr)` on L0* sets because
    they already store R̃ = 0.999R + 0.001I; zero-weight mixture components are dropped, K = 6): configuration prior = type table π(n_sh, n_1only, n_2only), uniform over variant
    choices (log π − log #configs); |λ| per (trait, variant) from a K-component zero-mean Gaussian scale
    mixture, independent across traits and variants (a shared variant's two effects independent).
    `priors.toy_l0star_prior()` (K = 8, sds geometric 1…60, §2.4 mixture restricted to ≤ 2 per trait +
    5% null) is a placeholder until `src/sim` defines L0*. `priors.reweight_classes` applies the
    class-prior correction for rejection-sampled cells.
11. **Precision**: kernel and all sums fp64 by default (`dtype="float32"` available: |Δlog BF| ≤ 1e-4
    relative). TF32 asserted off. `compile=True` uses torch.compile (inductor) — identical to eager to
    1e-12 (test), 2–3× faster at M = 100 after a one-off ~20 s compilation; duck-shape specialisation is
    disabled to avoid recompiles when nB = 2.
12. **Sampling**: rectangular path (single-component priors) = exact trait-1-configuration marginal
    (stratified) then trait-2 configuration by inverse CDF within the row; generic path (mixtures) =
    stratified systematic sampling over all terms. Row log-masses use logsumexp (an earlier underflow
    bug that produced draws of e^-100-mass terms in mixed-strength batches is covered by a regression
    test).
13. **runtime_s** = per-locus CPU prep (screen, LD rows) + the GPU batch wall-clock divided evenly over the
    loci of the batch (batch = all loci with the same M, up to `batch_elems` terms).
13b. **MC seeds (batch invariance)**: the Monte Carlo draws of locus L come from its own torch generator seeded
    with `common.seeds.derive(seed, "famB_mc", seed_key, L.locus_id)` (`seed` must be in `SEEDS`, default
    20261029; `seed_key` distinguishes replicate runs). A locus therefore gets identical posteriors in any batch
    composition or order (tests/famB/test_batch_invariance.py, both sampling paths). Before 2026-10-27 one
    generator was shared by the batch, so draws depended on batching (verifier finding); no result from that code
    is registered.
14. **Failures**: any exception for a locus/batch → `status="failed"` with the error in `extra`; never
    silently dropped.

## Known limitations

- Misspecified for 3 causal variants per trait (by design, §5 B exactness scope) and for background
  polygenicity.
- Mismatch: B's only mismatch handling is EB η. With a small external reference (n = 150) the screen loses
  ~3% of powered causal variants (secondary signals masked by a strong lead cannot be recovered by
  conditioning on a noisy reference; see tests/famB/test_screen_recall_1kg.py output).
- Simulator mode cost grows as C(p,2)²·K⁴: p = 30 with K = 8 is ~0.9×10⁹ terms per locus.

## Tests and performance (M3; details in reports/famB_M3.md)

`PYTHONPATH=src .venv/bin/python -m pytest tests/famB -v` → 41 passed (≈ 2.5 min on the RTX 4090; `-m "not slow"`
skips the R / 1000G / calibration tests). BF vs full form ≤ 1e-13 (fp64) and < 1e-8 vs mpmath; coloc.abf
identity 3.6e-14 on 200 loci; screen recall (powered, L0, real 1000G LD) 0.998–0.999; exact-model calibration
|z| ≤ 2.6. Throughput (p = 2,000, full pipeline incl. EB and MC): M = 50 ≈ 60–70 loci/s eager, ≈ 100 compiled;
M = 100 ≈ 5 eager, ≈ 12 compiled fp64, ≈ 20 compiled fp32.

## B-mm: mismatch-aware ExactColoc (phase 2, DESIGN_P2 §2; `mismatch.py`)

```python
rows = ExactColoc(mode="gaussian", M=50, mismatch=True).fit_batch(loci)   # name exactcoloc_mm_gaussian_M50
```

Model: z ~ N((I2⊗R̃)β, Σ), Σ = V⊗R̃ + Δ̂. Here Δ̂ is the reference-error covariance of (R̂ − R)λ from DESIGN layer E. It is block-diagonal over variants k, with 2×2 trait blocks:
Δ_tu,k = (1/n_ref) Σ_jj' E[λ_tj λ_uj' | z] C^(k)_jj'.
C^(k) is the large-sample covariance of sample correlations that share index k (Olkin–Siotani). Its diagonal is (1 − r_kj²)², and its j ≠ j' entries are the cross terms.

Decisions (fixed before any screen/M4 result):
1. **Locus-level plug-in.** λλ' is replaced by its exact posterior second moment under the current fit. All configurations are summed in a fused fp64 pass (`logz_and_moments`); terms below e^-40 of the running maximum are not accumulated. A configuration-dependent Δ would need a dense 2M×2M factorisation per configuration (about 1.6 M per locus at M = 50), which is infeasible.
2. **Iterations.** `mm_iter = 2` refinements, starting from the phase-1 fit (pass 0).
3. **Inflation terms.** `mm_cross_terms = True` (primary), and cross-trait terms are always included. The diagonal-only form is an ablation (`_diag`). In a simulation of reference panels (n = 150), the formula with cross terms matches the empirical covariance to 6%; the diagonal-only form is off by up to 61% (tests/famB/test_mismatch.py).
4. **n_ref.** Taken from `locus.n_ref` (L0 in-sample: the study N). R̂ entries are the raw reference r.
5. **EB.** (σ_t, η) are re-selected by the phase-1 EB rule with Σ_t(η) = R̃(η) + diag(Δ_tt).
6. **Full-z_M likelihood.** Φ = X'Σ⁻¹X and b = X'Σ⁻¹z, computed in a Woodbury form that never inverts R̃:
   Φ = V⁻¹⊗R̃ − (V⁻¹⊗I)(Δ − ΔΣ⁻¹Δ)(V⁻¹⊗I),
   b = (V⁻¹⊗I)(z − ΔΣ⁻¹z).
   Both enter the unchanged support-space kernel (`LocusTensors.Phi`). With Δ = 0 this is phase-1 B exactly.
7. **Oracle events.** They are computed as in phase 1, with mean map R̃.
8. **Failures.** A non-PD Σ fails that locus only (one-by-one retry inside the batch).
