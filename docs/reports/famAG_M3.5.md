# Family A / G developer — M3.5 report

Owner dirs: `src/famA/`, `src/famG/`, `tests/famA/`, `tests/famG/`. Result files (small) are in `reports/famAG_M3.5/`. Decisions not fixed by DESIGN v4 / D4 / D6 are in `src/famA/README.md` (items 1–16) and `src/famG/README.md` (items 7, 9).
No test-split data was generated or read. Kill test A1 was **not** run. screen_v2 does not exist yet, so the smoke run used 96 screen_v1 validation loci (superseded set; this is not an evaluation).

**Tests:** `source env/activate.sh; PYTHONPATH=src python -m pytest tests/famA tests/famG -q` gives **40 passed** in 198 s (23 at M3, plus 17 new). `tests/eval/test_truth_isolation.py` also passes (7/7).

## 1. η estimation batched on the GPU, and throughput

**What changed**

- **Spectrum for η** (`_JointBucket._eigen`).
  - Genotype-factor LD (n_ref ≤ p): float64 eigen-decomposition of the n_ref × n_ref Gram matrix X̃X̃ᵀ. This is exact; the p − rank null-space directions carry eigenvalue η. Cost per locus at p = 2,000 dropped from ≈ 500 ms (per-locus float64 SVD) to 3 ms (n = 503) or 0.5 ms (n = 150).
  - Dense LD: batched `torch.linalg.eigh`, float64 by default.
  - The η terms match a brute-force dense float64 formula to < 1e-7 relative (test; rank-deficient R, padding, ρ ≠ 0).
- **η Brent as one fused Triton kernel** (`famA/eta_kernel.py`). It equals the torch path (test). Cost: 1.2 s → 0.05 s per fit of 64 loci.
- **EB kernel.** The part of the SER likelihood that does not depend on σ²_t is computed once per Brent run. This means 2 exp per variable instead of 5 exp + 1 log, giving 1.5× on the kernel; results are identical up to fp64 rounding. A float32-exp option was only 1.3–1.45× faster and is off.
- **Continuous batching** (`_fit_stream`). Converged loci are flushed and their rows refilled from the queue, so stragglers (max 33–43 iterations against a median of 4–7) no longer hold a nearly empty batch. Results match single-locus fits (test). In any batched fit, an EB null-check near-tie can amplify fp64 rounding to ≈ 3e-4 in PIP on rare loci; one of 9 toy loci showed this, with and without refill.
- **Fixes:**
  - the output stage now evaluates only events that can contribute (upper-bound pruning, exact for zero-scale effects);
  - `famG/ld.py BatchLD.diag()` is chunked. It allocated 2 GB at once and hit OOM on the shared GPU.

**Throughput** of A (+ρ +η, EB-π, M3.5 power events) at p = 2,000, both traits jointly. LD: 1000G EUR, 25 % mismatch loci. Measured while other agents kept the GPU at **93–99 % utilisation and 13–22 GB used**, so these are lower bounds. Files: `reports/famAG_M3.5/timing_A_*.json`.

| LD | loci per call | fit only, loci/s | with LD read, loci/s |
|---|---|---|---|
| factor (n_ref = 503), η | 64 | 11.3 | 11.1 |
| factor (n_ref = 503), η | 256 | **22.4** | 21.8 |
| dense, η, float64 eig | 256 | 7.2 | 6.9 |
| dense, η, float32 eig (option) | 256 | 13.9 | 11.8 |
| factor, no η | 256 | 26.8 | 25.8 |
| dense, no η | 256 | 27.3 | 22.7 |

The M3 report gave 1.4–2.0 loci/s with η and 3.1–7.5 without, measured on a busy GPU at p = 1,000–5,000. M3.5 at p = 2,000 on the busy GPU before these changes: dense 7.5 and factor 3.0 loci/s with η.

**Target ≥ 20 loci/s with η at p = 2,000:**
- **Met for factor LD**: 22.4 loci/s on a saturated, shared GPU.
- **Not met for dense LD with η.** The bound is the eigen-decomposition. cuSOLVER `syevd` at p = 2,000 takes 61–71 ms per matrix in float64 and 22 ms in float32 on the RTX 4090; FP64 throughput is 1/64 of FP32, and large matrices are solved one after another, not batched. Dense + η is therefore capped at ≈ 15 loci/s (float64) or ≈ 45 loci/s (float32) at p = 2,000, however fast IBSS is.
  - The float32 option floors η at 1e-3, the same floor as B (README 2). It changes P_S / P_D by < 2e-3 on toy loci (test).
  - The same bound applies to factor LD with n_ref ≈ p (L1, n = 2,000), because the Gram matrix is then about as large as R.
  - Remedies not implemented: eig on a CPU worker pool, overlapped with the GPU IBSS; caching one spectrum per shared reference LD (L1/L2/L4 loci in the same window share LD).

## 2. π by empirical Bayes per locus (`pi_mode="eb"`, now the default; `"fixed"` kept)

- **Update.** MAP/EM step after each sweep: π_s ∝ Σ_{used l} q_l(s) + κπ_ref,s, with κ = 4 and π_ref = ¼ each.
  - Effects with both prior scales 0 have q_l(s) = π and zero KL for any π, so they follow π at no cost. Using only "used" effects therefore gives the exact one-step fixed point.
  - The ELBO includes the Dirichlet log-prior. Its monotonicity is tested.
- **Tests:**
  - a strong shared signal raises π_both to ≈ 0.4, the maximum the κ = 4 prior allows with one effect;
  - κ → ∞ reproduces fixed π to < 1e-6.
- **Limitation.** On a single-trait locus, EB sets σ²_2 = 0, which makes 'both' identical to '1-only'. EB then cannot separate them, so only π_2-only and π_0 decrease.
- **Cost.** Median iterations rise from 4 to 7 at p = 2,000.
- κ and π_ref remain to be tuned on validation (D6(g)).

## 3. Power events per DESIGN v4 §1

- **Marginal noncentrality** of effect l at variable j: μ_t = β_t + m_{−l,t,j}, with m_{−l} = R̃E[b_{−l}] = z̃ − r_l.
  - This is a plug-in of the other effects' posterior means. Their posterior variance and the in-sample-versus-reference LD difference are ignored; this is the documented approximation.
- **Event definitions:**
  - powered_t = {|β_t| ≥ τ_t} ∩ {|μ_t| ≥ τ_t};
  - blocking ('power ≥ 0.2') = {|μ_t| ≥ w_t} ∩ {|β_t| ≥ w_t}.
- **Deviation (needs a PI decision).** The oracle's "nonweak" uses μ only. Without an own-effect guard, a near-null 'both'-state effect of the fit (σ² ≈ 0, state probabilities equal to π) at a variable in LD with strong distinct signals would count as an LD-inflated shared variant and block D. `nonweak_own=False` gives the μ-only version.
- **Computation.**
  - Probabilities of the "two-hole" sets come from 1-D CDFs and a conditional Gauss–Legendre integral (`famA/bvn.py`).
  - They agree with an independent disjoint-interval + adaptive-quadrature reference to < 1e-6 over 60 × 5 random cases, and with scipy in 1-D to 1e-12.
- **Thresholds** come from the locus attributes `trait1_type` / `trait2_type`: gwas τ 6.29 / w 4.61, mol τ 5.26 / w 3.58. These equal `sim.params` (tested). Missing attributes fall back to (gwas, mol) and are flagged in `extra`; unknown values raise an error.
  - In the smoke set: 48 gwas/mol, 35 gwas/gwas, 13 mol/mol loci.
  - Test: with λ ≈ 5.8, P_S is clearly higher under (mol, mol) than under (gwas, gwas).
- **E_t** uses the own effect and the weak threshold w of the other trait's type (DESIGN layer D writes λ).
- **Effect of the new events.** On the 8 smoke D loci re-run, P_S / P_D are unchanged to 3 decimals except one locus (P_D 0.053 → 0.020). The A results are dominated by the fit, not by the event definition (§7).

## 4. Layer E — empirically calibrated reference-size variance (ablation, `ref_var=True`)

- **Calibration** (`famA/validation/calibrate_ref_ld_var.py` → `src/famA/ref_ld_var_v1.json` / `.md`, log `logs/famAG/calibrate_ref_ld_var_v1.log`). Pool resampling:
  - NFE = pools A ∪ B (404 individuals);
  - 40 windows of 300 SNVs inside train-split blocks;
  - subsamples n ∈ {100, 150, 254}, 200 draws each;
  - finite-population correction;
  - n·Var(r̂) tabulated by r² bin × min-MAF bin.
- **Calibration result:**
  - normal theory is right for common pairs at r² < 0.5;
  - it is too small by 2.5× (r² 0.7–0.8) up to ≈ 19× (r² 0.95–0.98) for pairs with min MAF ≥ 0.2;
  - it is too small by ≈ 6.7× for rare pairs (min MAF < 0.05) at r² 0.3–0.5;
  - n·Var is stable across n for MAF ≥ 0.05, but not for rare pairs. The method therefore uses the table of the nearest calibrated n.
- **Model.** Per-variable variance inflation of the SER likelihood, d_tj = 1 + Σ_{i ∈ top-32 E[b_t²]} E[b_ti²]·κ(r²_ij, MAF)/n_ref. It is updated after each sweep; cross terms are ignored.
  - With layer E the SER no longer maximises the ELBO; the ELBO is still a valid bound. Convergence is therefore by max |Δα| < 1e-4.
  - The EB step runs in the torch path, so layer E is slow: 0.08 loci/s in the smoke run.
- **Tests:**
  - `update_delta` equals a brute-force computation from the table;
  - on a toy (6 loci; one shared λ = 25 effect; z from a 20k population; LD from a 100-individual reference), spurious credible sets fall from **58 to 6** (η alone: 57; η + E: 23), with P_S > 0.9 kept.

## 5. G coloc.abf fallback aligned with the tier-1 convention

- **Alignment.** The tier-1 wrapper `baselines/methods.py:ColocSusie` uses the same p12 for `coloc.bf_bf` and its coloc.abf fallback, with W = 0.15.
  - G now does the same: `p12_abf=None` means it follows `p12`, so tuning p12 moves both, and `W_abf` defaults to 0.15.
  - G also applies coloc.abf's own `adjust_prior` (p = 1/(nsnps+1) when nsnps·p ≥ 1), which was missing. It only matters for ≥ 10,000 SNPs at p1 = 1e-4. `coloc.bf_bf` does not adjust, as in coloc 5.2.3.
- **Test.** Equal to `baselines.abf.coloc_abf` + `eval.adapters.coloc_abf` to 1e-12 (p = 300 and 12,000; p12 ∈ {5e-6, 1e-5}), including end-to-end through `GPUColocSusie`.

## 6. Identity gates re-run at M3.5 (`reports/famAG_M3.5/identity_*`)

| Gate | Result |
|---|---|
| A vs mvsusieR 0.3.0 (60 loci, `pi_mode="fixed"`, float64) | max \|ΔPIP\| **7.7e-14**, \|Δstate\| 5.3e-14, iterations equal 60/60 |
| G vs susieR 0.14.2 (280 fits, float64) | max \|ΔPIP\| **6.7e-8**, iterations and convergence equal 280/280, CS equal up to α-ties 280/280 |
| G coloc-SuSiE end-to-end (140 loci) | max \|Δs\|, \|ΔP_S\|, \|ΔP_D\| **4.2e-7**, fallback decisions agree 140/140 |

## 7. Smoke run: 96 screen_v1 validation loci (4 per cell × bucket; not an evaluation)

Script: `reports/famAG_M3.5/scripts/smoke_screen.py`. Output: `smoke_screen_v1.tsv`.

**All runs completed.** Failures: A 0; G 1 of 96 ("failed"). Non-converged at max_iter = 100: A 8, A+E 2, G 20.

**Scores on 72 S/D loci and 72 D/W loci:**

| Run | AUROC s, S vs D | mean P_D on D | mean P_S on S | AUROC P_D, D vs W |
|---|---|---|---|---|
| A ρη | 0.845 | 0.29 | 0.77 | 0.85 |
| A ρη + fixed π | 0.849 | 0.29 | 0.78 | 0.83 |
| A ρη + E | 0.888 | 0.35 | 0.78 | 0.87 |
| G | 0.868 | 0.78 | 0.88 | 0.85 |

On W loci, A makes almost no confident calls (mean P_S and P_D < 0.01); G has mean P_D 0.11.

**Findings for the PI, before A1:**

1. **A under-calls D when the distinct variants are in LD.** In ≈ 40 % of D loci the IBSS ends in a local optimum where one effect takes the 'both' state over the LD block.
   - Evidence (`scripts/localopt.py` → `smoke_local_optimum.tsv`): starting with the shared state disabled for 5 sweeps, then fitting the same model, gives a higher ELBO on **19 of 48 D loci** (lower on 13) but on only 2 of 24 S loci.
   - Choosing the better of the two starts by ELBO moves S-vs-D AUROC from 0.832 to 0.846 on these 72 loci.
   - Proposal for M5 (not implemented): multi-start IBSS (default start plus a "no-shared-first" start, or initialisation from the per-trait G fits), keeping the higher ELBO.
2. **A uses one LD matrix for both traits.** At L0, trait 2's in-sample LD (`R2` / `X_ref2`) differs from trait 1's. The SER reduction still holds with per-trait LD (diag R_t = 1), so a per-trait mean map is a feasible extension.
3. **The base and +ρ rungs (η = 0) diverge on rank-deficient factor LD (n_ref = 150 < p).** σ² hits the Brent bound e¹⁵ and the ELBO grows by ≈ 3,000 per iteration, because z has a null-space component that R cannot explain. These rungs need η, or a floor such as B's η ≥ 1e-3, at L2/L4.
4. **The 8 non-converged A loci (with η) creep upward.** All are L2/L4 with η = 0.3–0.8; the ELBO rises by 0.004–0.012 per iteration, just above the absolute tolerance 1e-3. Options: a relative tolerance or an α-based stop.
5. **Throughput on the real set was ≈ 1 locus/s.** Causes:
   - 96 loci spread over p = 2,000–4,000 form about 14 small groups (p-ratio ≤ 1.1);
   - non-converged loci run 100 iterations;
   - 27 dense L0 loci at p ≈ 2,800 need float64 eig.
   - A real benchmark (thousands of loci per group) avoids the first cause; the others are items 4 and §1.

## 8. Proposed changes / decisions needed (I did not edit DESIGN / INTERFACES / RESEARCH_TREE)

1. Accept the own-effect guard on 'power ≥ 0.2' (§3), or choose μ-only (`nonweak_own=False`).
2. Default π: EB (κ = 4) or fixed. Tune κ / π on validation (D6(g)).
3. Dense + η throughput bound (§1): accept, use the float32 eig option (η ≥ 1e-3), or have me add CPU-offloaded / cached spectra.
4. Before A1: authorise the multi-start IBSS and per-trait LD at L0 (§7, items 1–2). These are the main accuracy issues seen.
5. An η floor (1e-3) for the base/+ρ rungs on rank-deficient LD, or report those rungs only at L0.

## 9. Reproduce

```
source env/activate.sh; export PYTHONPATH=src R_PROFILE_USER=/dev/null P=$PWD; W=<scratch>
python -m pytest tests/famA tests/famG -q
python -m famA.validation.calibrate_ref_ld_var --out src/famA/ref_ld_var_v1.json --n-windows 40 --p 300 --draws 200 --n-list 100,150,254
python -m famG.validation.make_loci --out $W/t_f2000 --n-loci 64 --p-set 2000 --factor-frac 1 --mismatch-frac 0.25 --key timing35_f2000   # and t_d2000 (--factor-frac 0, key timing35_d2000)
python -m famA.validation.timing --dirs $W/t_f2000,$W/t_d2000 --out reports/famAG_M3.5/timing_A_eta_p2000.json --reps 1 --eta --replicate 1,4
python -m famA.validation.timing --dirs $W/t_d2000 --out reports/famAG_M3.5/timing_A_eta_p2000_eigfp32.json --reps 1 --eta --eig-dtype float32 --replicate 4
python -m famA.validation.timing --dirs $W/t_f2000,$W/t_d2000 --out reports/famAG_M3.5/timing_A_noeta_p2000.json --reps 1 --replicate 4
# identity gates: as in reports/famAG_M3.md §8, with --out reports/famAG_M3.5/identity_{G,A}
python reports/famAG_M3.5/scripts/smoke_screen.py data/processed/sets/screen_v1.h5 4 reports/famAG_M3.5/smoke_screen_v1.tsv
python reports/famAG_M3.5/scripts/localopt.py
```
