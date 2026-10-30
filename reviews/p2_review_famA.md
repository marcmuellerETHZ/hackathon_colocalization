# Review p2.review_famA: commit 47ceb97 (Family A JointSuSiE M3.5, Family G) and reports/famAG_M3.5.md

Reviewer: independent (did not write this code). Branch `p2-review-famA`. I did not modify any famA/famG code.

**Scope.** I checked the code against:
- DESIGN v4 §1 (trait-type thresholds) and §5 A (states including null; likelihood with V⊗R̃; inclusion–exclusion P_S/P_D/P_W; E_t with weak thresholds);
- DECISIONS P3-5 and P3-6.

**Data.** All loci are from `data/processed/sets/screen_v2`. They are VALIDATION loci only: `split == 'valid'` is asserted per locus, and `LocusSetReader(guard=True)` is used. No test-split data and no truth files were opened.

**Evidence.** Files are in `results/review_famA_v1/`, with a copy in `reviews/p2_review_famA/evidence/`. Scripts are in `reviews/p2_review_famA/`.

## Verdict

**PASS with minor defects.** I found no defect in 47ceb97 that changes P_S, P_D, P_W or E_t, or the G identity.

The inclusion–exclusion P_D, the per-effect power events, the trait-type thresholds and the G vs susieR identity all reproduce independently. They agree to rounding where computed exactly, and to within Monte Carlo error where sampled. The layer-E table was built only from train-split genotypes, reproduces bit-for-bit, and does not touch truth.

Two things remain open:
- The A1 accuracy issues the developer reported (local optimum, single LD at L0, η floor, tolerance) are open. They are being fixed under P3-5 in `p3.famA_fixes`.
- The R-based identity tests cannot run in a normal orch job (defect 2).

## Numerical checks

### (1) Tests through the queue (GPU, 8 GB declared)

| Job | Setting | Result |
|---|---|---|
| `p2rA_pytest_famAG` | plain snapshot of HEAD 334e182 (famA/famG identical to 47ceb97) | **1 failed, 34 passed, 5 skipped** (see defect 2) |
| `p2rA_pytest_famAG_r2` | copy of the snapshot with `env/renv` and `env/rlib_*` linked; `--basetemp` in `$ORCH_OUT` | **40 passed** in 139 s |

The 40 tests include the mvsusieR identity, the susieR 0.14.2 identity, Brent vs R `optimize`, and coloc vs R.

Command: `PYTHONPATH=src python -m pytest tests/famA tests/famG -q -rs`

### (2) Inclusion–exclusion P_D re-implemented (job `p2rA_pd_check`)

**Loci.** 48 loci: 6 per locus-id stratum (W, S, D_r2lt0.5, D_r2_0.5-0.8) in each of L0_P1_r_p=0.0 (gwas/mol) and L0_P3_rho=0.5 (gwas/gwas and mol/mol). The selection is seeded with `rng(VALID_SEED, "p2_review_famA", cell)`.

**Fit.** Rung ρ+η, EB π, default events. 15 loci have P_D > 0.05 and 22 have P_S > 0.05.

**Three independent computations:**
- **(a) Combination.** I recomputed P_S and P_D from A's own per-effect (a, w, p̃¹, p̃²) with my own inclusion–exclusion. I also computed an exact dynamic programme over (blocked, has₁, has₂) that uses no inclusion–exclusion at all. The DP also equals the IE on 1,000 random inputs to 1e-12.
- **(b) SER posterior.** I re-derived each effect's SER posterior from the fitted residual r_l with my own bivariate-normal algebra: BF_s = N(r; 0, U_s+V)/N(r; 0, V), mean U S⁻¹r, covariance U − U S⁻¹U, and α with prior 1/p and π.
- **(c) Monte Carlo.** I ran 400,000 draws per effect of (j, state, β). From them I evaluated the events directly: powered = |β| ≥ τ ∧ |β + m| ≥ τ; blocking = 'both' ∧ (|β| ≥ w ∧ |β + m| ≥ w in both traits), with m = z̃ − r_l. The joint D, S and E_t events use independent effects.

| Quantity | Max abs difference vs A |
|---|---|
| P_D: A vs own IE / exact DP | 1.1e-16 / 2.2e-16 |
| P_D: same check for the +E (layer E) rung / fixed-π rung | 2.2e-16 / 1.7e-16 |
| P_S: A vs own IE | 0 |
| α: A vs my own SER posterior | 2.2e-14 |
| 1,920 per-effect events (a, w, p̃¹, p̃²) vs MC | 1.4e-3, max \|z\| 2.85 |
| P_D vs joint MC | 2.2e-3, max \|z\| 3.7 on one locus |
| P_S vs MC | 1.4e-3 |
| E1 / E2 vs MC | 4.8e-4 / 1.1e-3 |

**Re-check of the z = 3.7 locus** (`..D_r2lt0.5__0021008`), with 10⁷ draws (job `p2rA_pd_check_mc1e7`): P_D A 0.838135 vs MC 0.838188 (z = 0.46). All event |z| values are ≤ 1.6. The earlier deviation was Monte Carlo noise.

**Thresholds.** These follow the trait *type* end to end: P3 loci were fitted with gwas/gwas and mol/mol thresholds from the locus attributes.

### (3) G vs susieR 0.14.2 (job `p2rA_g_identity`)

**Setup.** My own R script (`run_susieR.R`) calls `susie_rss(z, R, n, L = 10)` with all other arguments at their defaults. It ran on 12 validation loci × 2 traits = 24 fits:
- 6 L0_P1 loci (dense LD, p 1,041–3,022);
- 6 L2_P1 loci (factor LD, n_ref = 150 < p, p 1,926–3,978).

The factor loci include 2 fits that do not converge in 100 iterations in R either.

| G LD dtype | max \|ΔPIP\| | max \|Δα\| | niter equal | converged equal | CS equal |
|---|---|---|---|---|---|
| float64 (the identity claim) | **4.0e-8** (dense 3.9e-10) | 1.2e-7 | 24/24 | 24/24 | 24/24 |
| float32 (production default) | **6.2e-4** (dense 2.0e-7) | 7.1e-4 | 23/24 | 24/24 | 24/24 |

The report's claim (float64, max \|ΔPIP\| 6.7e-8) is confirmed. See defect 4 for float32.

### (4) Layer-E variance table: provenance and leakage (job `p2rA_layerE_check`, CPU)

- All 40 calibration windows in `ref_ld_var_v1.json` lie inside **train** blocks of `splits/blocks_v1.tsv` (40/40 train, 0 outside a block).
- The calibration code reads only:
  - 1000G genotypes of NFE pools A ∪ B in those windows;
  - `pools_v1.tsv`;
  - `blocks_v1.tsv`.

  It reads no locus sets, no truth parquet and no R_true. The table is generic (r² bin × min-MAF bin × n) with no locus-specific content.
- At inference, `update_delta` uses only the inference LD (`self.ld`), `maf_ref` and n_ref.
- **Re-running** `calibrate(40, 300, 200, [100, 150, 254])` at the snapshot reproduces the committed table: identical windows and counts, max \|Δκ\| 1.6e-15.

**Conclusion:** no leakage of validation, test or truth data.

## Defects

1. **Docstring contradicts the default event definition (minor, documentation).**
   - The `joint_susie.py` module docstring (line 29) defines nonweak_t = {|μ_t| ≥ w_t} "(marginal only, as the oracle)".
   - The default is `nonweak_own=True`, so blocking also requires own |β_t| ≥ w_t.
   - README item 9 and DECISIONS P3-5 item 5 describe the code correctly; the docstring is wrong.
   - The P3-5 item 5 guard is therefore **already implemented and default in 47ceb97**. `p3.famA_fixes` does not need to re-implement it, only fix the docstring.

2. **R-based identity tests do not run in an orch job (moderate, verification process).**
   - **Cause.** `tests/famA/conftest.py`, `tests/famG/conftest.py`, `famag_testutils.py` and `test_joint_susie.py` locate `env/renv/bin/Rscript` and `env/rlib_{susie016,mvsusie}` relative to the test file. Orch snapshots contain only tracked files plus data/results/logs/.venv links.
   - **Effect in a queue job:**
     - 5 tests **skip silently**: susieR identity ×2, Brent vs R, coloc bf_bf vs R, coloc.abf vs R;
     - `test_identity_vs_mvsusie` **fails** with FileNotFoundError, because it has no `has_r` guard;
     - the job reports "1 failed, 34 passed, 5 skipped".
   - With the env dirs linked, all 40 pass.
   - A queue run that reports "passed" therefore omits every identity gate. The tests also write to `/tmp/pytest-of-*` unless `--basetemp` is set.
   - **Fix:** resolve Rscript and the R libs from `$ORCH_HOME` (or `$PATH` and `R_LIBS`), add `has_r` to the mvsusie test, and fail rather than skip when `ORCH_JOB_ID` is set. Alternatively, orch links the untracked `env/` subdirectories.

3. **Layer-E calibration convention for subsample-monomorphic variants (minor; ablation-only).**
   - **What the code does.** `calibrate_ref_ld_var.py` sets r̂ := 0 when a variant is monomorphic in a subsample (`np.nan_to_num`).
   - **Why this is wrong for inference.** Inference LD never contains such columns. In 5 checked L2 loci, X_ref had 0 zero-variance columns, because the simulator drops them.
   - **Size of the effect.** My independent re-implementation (12 windows, n = 150) excludes monomorphic draws per pair. Rare pairs (min MAF < 0.05) change as follows:

     | r² bin | n·Var(r̂), r̂ := 0 convention | n·Var(r̂), monomorphic draws excluded |
     |---|---|---|
     | 0.2–0.3 | 2.56 | 2.24 |
     | 0.8–0.9 | 1.57 | 1.14 |
     | 0.9–0.95 | 0.91 | 0.33 |
     | ≥ 0.995 | 0.20 | 0.00 |

     The committed table has 0.60 in the ≥ 0.995 cell.
   - **What still stands.** The report's headline finding survives: rare pairs at r² 0.3–0.5 are ≈ 6× above normal theory. The high-r² rare cells, however, are inflated by the convention.
   - **A second, undocumented issue.** The table is binned by pool r² but looked up by the reference r̂², an errors-in-variables mismatch.
   - **Fix:** exclude monomorphic draws (or drop such variants per draw) and document the r̂² lookup.

4. **The G identity is shown only for float64, but production G runs float32 (minor, documentation).**
   - In float32, the max |ΔPIP| vs susieR is 6.2e-4. That passes the DESIGN gate (< 1e-3), but with little margin, on a non-converged L2 locus. One of 24 fits differs in niter (80 vs 82).
   - The report's identity table should state the float32 numbers, since the C2 speed comparison uses float32.

5. **ρ_u is fixed at 0, not estimated (minor, design deviation, documented).**
   - DESIGN §5 A says "U_both allows a correlated effect". README item 5 fixes ρ_u = 0 because the simulator draws signs independently, which is a justification taken from the generator.
   - This is harmless as a neutral prior, but it should be listed as a deviation for real-data use.

6. **Open P3-5 items, already known; status only.** In 47ceb97:
   - single LD for both traits at L0 (`X_ref2` is ignored);
   - single-start IBSS;
   - η lower bound 1e-8 (no 1e-3 floor; the base and +ρ rungs have no η and diverge on rank-deficient LD);
   - absolute ELBO tolerance.

   These are approved fixes in progress under `p3.famA_fixes`. My 48 loci agree with the developer's local-optimum finding: on the D_r2_0.5–0.8 stratum, P_D > 0.5 on only 1 of 6 (P1) and 2 of 6 (P3) loci, with mean P_S 0.53 and 0.38. Locus-id strata are generation buckets, not oracle classes, so this is not an accuracy estimate.

## Conformance summary

| Requirement | Status |
|---|---|
| States {0, 1, 2, both} with explicit null | Confirmed |
| Likelihood z ~ N((I₂⊗R̃)b, V⊗R̃), SER bhat = r_j ~ N(β, V); ELBO η terms | Confirmed (code; η terms tested against brute force) |
| Inclusion–exclusion | Confirmed exactly (defect-free) |
| P_S and P_D disjoint by construction (a ⊂ w) | Confirmed |
| P_W = 1 − P_S − P_D; s = P_S/(P_S + P_D) | Confirmed |
| E_t: own effect, other trait's weak threshold, over CS_t of effects with P(powered) ≥ 0.5 | Confirmed (MC) |
| DESIGN unit test (single either-trait signal gives P_D = 0; two strong signals give P_D = 1) | Present and passing |
| §1 thresholds by trait type (6.29/4.61 GWAS, 5.26/3.58 molecular), from locus attributes | Confirmed |
| P3-6: EB π with κ = 4 is the default; fixed π available | Confirmed |
| P3-6: float32 eig floors η at 1e-3 and has a PIP test (< 5e-3) | Confirmed |
| P3-6: layer E is a separate rung (`_E` in the name) | Confirmed |
| A vs mvsusieR identity | Passes in the test with R available |
| Report numbers I re-ran (tests 40 passed; G float64 identity) | Agree |

## Outside this review's scope (for the PI)

- The locus ids in screen_v2 encode the generation stratum (`__W__`, `__S__`, `__D_r2lt0.5__`). Methods receive `locus_id`. A and G do not use it, but a learned method could. This is for p2.review_sim and famC.
- The `[orch] cmd:` log line shows variables expanded in the submitter's shell environment. For example, `s1=${PIPESTATUS[0]}` was shown as `s1=0`, while the actual execution was correct. The log line is misleading.
