# Review p2.review_sim: simulator commits 3416f2e, 850218b and fb426c4

Reviewer: independent reviewer agent. I did not write this code. Branch `p2-review-sim`, based on master 334e182.

Scope:
- `src/sim/{generator,oracle,params,architecture,cells,testset,train_api,l0star,windows}.py`;
- `src/stats/rho.py`;
- the new tests.

They were checked against:
- DESIGN v4: §1, §2.2–2.4, §5 B (L0\*), §7.3, §8.5 and §8.6;
- RESEARCH_TREE D6;
- DECISIONS P3-1, P3-11 and P3-12.

No test-split data was generated, loaded or viewed. The test-lock checks call only the access predicates.

## Verdict

**Approve with fixes.**
- There is one major defect (1). It is fixed in this branch, with a regression test.
- The other defects are minor. Defect 2 needs the PI to commit the test-set config, and the rest are optional.
- The statistics conform to DESIGN v4 and D6: the oracle partition, the OLS statistics, ρ̂, overlap and reachability, the level rules, the PVE cap, L0\* and the shifted arm.

## Tests run through the queue

| Job | Commit | Command | Result |
|---|---|---|---|
| `p2rs_pytest_sim_stats` (GPU, 8 GB declared, peak 2.1 GB) | 334e182 (master) | `python -m pytest tests/sim tests/stats -rA` | **77 passed, 1 failed**. The failure is `test_testset.py::test_config_file_parses`, FileNotFoundError on `configs/testset_v1.yaml` (defect 2). The job shows exit 0 because my `| tail` dropped pytest's status; the counts are from the log. |
| `p2rs_pytest_sim_stats_r2` (GPU, 8 GB, peak 1.9 GB) | ce7001c (with fix) | same, with `set -o pipefail` | **78 passed, 1 failed** (the same config test). The new `test_worker_access_check_does_not_need_eval_test_on_stack` passes. Recorded as failed (exit 1) because of the config test, then acked. |
| `p2rs_checks` (CPU) | ce7001c | `python reviews/p2_review_sim_checks.py` | ALL OK (A, B, C, D below) |
| `p2rs_checks_c2` (CPU) | ceef79c | `python reviews/p2_review_sim_checks.py C2` | ALL OK |

Logs are in `logs/jobs/<id>.log`.

## Independent checks (`reviews/p2_review_sim_checks.py`)

**A. oracle_class**
- I re-implemented the classes from DESIGN §1 text only. Each class rule (S/U/D/W) is evaluated separately, and the first true rule in S > U > D > W wins, else G.
- Test draws: 200,000 random loci with 0–3 causal variants per trait and shared variants. |λ| and |μ| were concentrated on, and exactly at, the weak and powered thresholds. All four trait-type combinations were used, and 10% of loci had tied r².
- Results:
  - 0 label mismatches;
  - 0 mismatches in class-determining (μ1, μ2), variant ids, r² or d;
  - every draw got exactly one class.
- Thresholds: τ and w are {gwas 6.29/4.61, mol 5.26/3.58}. Exact two-sided power at τ is 0.799 / 0.800, and at w it is 0.200 / 0.201.
- Trait types are required arguments: a missing type raises TypeError and an unknown type raises ValueError.

**B. OLS statistics (D6(a))**
- I ran `noisy_stats` on a CPU toy study (N = 400, 25 SNPs) and compared it with `statsmodels.OLS` per SNP.
- Max relative error:
  - intercept only: β 2.3e-5, SE 2.7e-6, z 6.6e-5;
  - intercept + 2 covariates (L3 PCs): 6.6e-6 / 2.9e-7 / 6.8e-6.
  - These levels are fp32 rounding.
- Factor-form transform z = z_g/√(1 − z_g²/N) against the exact OLS t on N = 50,000 individuals: relative difference 2.0e-5 at t = 0.74, 3.4, 22 and 73. This equals the expected √(N/(N−2)) factor.

**C. ρ̂ (P3-1 / D6(d))**

Genome-wide mode, 300 replicates per value, own seeds:

| ρ | bias | MC SE | SE / SD |
|---|---|---|---|
| 0 | −0.0002 | 0.0005 | 1.03 |
| 0.3 | −0.0025 | 0.0005 | 0.96 |
| 0.5 | −0.0042 | 0.0003 | 0.97 |
| 0.8 | −0.0046 | 0.0001 | 0.92 |
| −0.5 | +0.0036 | 0.0004 | 0.92 |

- All values pass the §2.4 bar of |bias| < 0.01.
- There is a small but statistically clear shrinkage toward 0. Its source is the 1% polygenic SNPs, with independent effects, that pass the |z| < 2 truncation. See note 9.

Per-window mode (C2): 1,500 SNPs with block-AR(1) LD (0.9) and exact R. The plain and residualised estimators were run on the same z (paired), 1,000 replicates per row:

| ρ, scenario | bias plain | bias residualised | paired diff (SE) |
|---|---|---|---|
| 0, null | +0.0031 | +0.0034 | +0.0002 (0.0001) |
| 0, shared concordant λ = 8 | +0.0086 | +0.0040 | −0.0046 (0.0006) |
| 0.3, null | −0.0032 | −0.0032 | −0.0000 (0.0001) |
| 0.3, shared signal | −0.0047 | −0.0073 | −0.0026 (0.0004) |

- MC SE of the biases is about 0.004.
- Residualisation does not change null estimates. It removes about half of the concordant-signal leakage.
- Jackknife SE/SD ratios are 0.97–1.06 (C1, 200 replicates).

**D. Test lock**
- `generate(split="test")` without access is refused.
- With `has_test_access` forced True, `_check_access("test", ...)` is still refused when the caller is not `scripts/eval_test.py`.
- `TrainSampler(split="test")` and `split_windows("test")` are refused.
- Stack check in the eval_test process: parent **True**, but spawned pool children **False**. This is defect 1.

## Defects

### 1. MAJOR: the authorised test run would abort in `sim.testset` pool workers (fixed in ce7001c)

`scripts/eval_test.py` calls `gen(testset, out_dir, seed=...)` with the default `workers=4`. `generate` then runs every cell in a `spawn` pool, and `_worker` calls `_check_access(split, False)`, which requires `_called_from_eval_test()`.

A spawned worker's stack is the multiprocessing bootstrap followed by `sim.testset._worker`. It never contains `scripts/eval_test.py`; check D shows the children returning False. Every worker would therefore raise `TestSplitAccessError`, and eval_test.py would log `aborted` after writing its `started` line. The PI would then have to commit the log and start again near the deadline.

This fails safe, so there was no leak, but it blocks the single test run.

*Patch (committed):* `_check_access(..., in_worker=False)`. Pool workers call it with `in_worker=True`, which requires only `has_test_access()`: an open `started` nonce of eval_test.py at a frozen-tagged HEAD, inherited through the environment. The parent still requires both conditions. The new test `tests/sim/test_testset.py::test_worker_access_check_does_not_need_eval_test_on_stack` checks:
- refusal outside eval_test.py with access;
- that a worker is allowed with access;
- refusal without access and on a dry run;
- the spawn-stack behaviour, using a fake `scripts/eval_test.py` in tmp.

It never calls a generator.

### 2. MINOR (PI action): `configs/testset_v1.yaml` is not on master

`test_config_file_parses` fails because `configs/` exists only in the WIP commit 2cc5e60 (branches `wip-m35-uncommitted` and `p2-review-wip`). With that draft, `main_grid_cells` gives 34 cells, all P3 cells positive-ρ, and `shifted_cells` gives 4 cells. I checked this inline.

The draft still has the pre-P3-12 sizes. Before freezing, the PI must set:
- `per_cell: S 100, D 100, W 100, G_U.max 30`;
- `D_generation_strata.per_bin: 20`, which is informational because GridPlan uses n_D/5;
- `shifted_arm.n_loci: 1600`, with `cells` n = 400 each.

`generate` takes all sizes from the config, so no code change is needed. The staged-relaxation limits scale with n_D (k1 = 100·n_D, k2 = 200·n_D).

### 3. MINOR: `rho_hat_for` with a dense-tensor `src` always residualises

`n_ref = ... else 10**9`, so the D14 rule ("residualise only with accurate LD: L0 or n_ref ≥ 1000") is skipped if a caller passes a dense L2 reference. All current callers pass an `LDSource`, so nothing is affected today. These callers are `generator.finalize`, `train_api` and `famC.simadapter` (via fb426c4).

*Proposed patch:* when `src` is not an `LDSource`, require `level` and an explicit `n_ref` argument, or raise.

### 4. MINOR (outside these commits; for the real-data ρ̂ task)

`src/data/real/extract.py:154` calls `rho_hat_window(z1, z2, R)` with the defaults: no residualisation and position-order pruning. P3-1 says the same estimator is used on simulated and real data. When the real-data ρ̂ is updated:
- pass `residualize_signal` by the D14 rule (in-sample LD, or n_ref ≥ 1000);
- record the pruning order.

### 5. MINOR: S class-determining tie when power saturates (oracle.py; not worth a re-freeze)

`power()` reaches exactly 1.0 in fp64 for |μ| ≥ 13.75 (GWAS) or ≥ 12.71 (molecular). When ≥ 2 shared powered variants are all saturated, the "largest min-power" rule picks the smallest id, not the strongest variant. This only affects which (μ1, μ2) S loci with ≥ 2 shared powered variants (M loci) use for power matching. The S label is unchanged.

*Recommendation:* leave as is, and document in README D8. If it is ever changed, tie-break on min(|μ1|−τ1, |μ2|−τ2).

### 6. MINOR: truth `N_s` for P3 eQTL×eQTL at L3/L6

The donors differ there (`same_rows=False`), but the truth table records `N_s = N`. ρ = 0, so the data are unaffected; only the metadata is wrong.

*Patch:* `N_s = N if pop1 == pop2 else 0` in `draw_pair_spec`.

### 7. MINOR (documentation)

- `stats/rho.py`: the module docstring says "no LD pruning", and `prune()` is labelled "legacy … not used". In fact `rho_hat_window` prunes at r² < 0.1 (README D14, `params.RHO_PRUNE_R2`).
- The `l0star.build_set` docstring says "30-SNP segment"; p = 20 now.
- Test fixtures label chr22:30.0–30.6 Mb as `split="valid"`, but it lies in train block chr22_0008. This is harmless: it is not test split, and nothing is written.

### 8. MINOR (hardening)

The lock sits at two points:
- window listing: `split_windows`;
- writing: `write_locus_set`, which checks split, block ids and test-block coordinate overlap.

`WindowContext` and `draw_candidate` accept arbitrary coordinates, so test-region loci could be simulated in memory by hand, though not written.

`_called_from_eval_test` matches any `scripts/eval_test.py` by name. `has_test_access()` is the effective guard.

*Optional:* an assertion in `WindowContext.__init__` that the window overlaps no test block unless `has_test_access()`.

### Notes (no action)

9. **Genome-wide ρ̂ shrinkage.** The bias reaches −0.0046 at ρ = 0.8, and SE/SD is 0.92 at |ρ| ≥ 0.5. Both are within the pre-registered bar.
10. **L0\* W configurations.** In L0\*, W is "one trait, one causal variant" rather than the §2.4 W mixture (README D13). This is self-consistent because B uses `config_logprior` of the same sampler, so it is not a defect for the B ceiling.
11. **True ρ in Family C batches.** `train_api` batches carry the true `rho` as a truth field next to `rho_hat`. The Family C review should confirm it is never a network input (DESIGN §2.4: methods never receive the true ρ).

## Conformance (checked, no defect)

**§1 classes and thresholds**
- τ and w follow trait type: P1/P2 gwas×mol, P3 eQTL mol×mol, P3 lab gwas×gwas.
- Powered requires both |μ| and |λ| ≥ τ.
- Weak is decided on μ.
- Precedence is S > U > D > W > G.
- U/D pair: maximal r² in R_true, then smaller d; d uses own λ.

**§2.2 generator**
- Individual level: OLS β/SE/t on the ALT dosage, with intercept, plus 2 PCs at L3.
- Factor form: se_ols = se0·√(1−z²/N) (D6(a)).
- λ = √(N q) (D6(b)).
- Overlap modes:
  - two factor traits: u2 = ρu1 + …;
  - nested small-N trait: per-row c = r_p√(N_syn/N1);
  - same donors: ε2 = r_p ε1 + ….
- `test_realised_rho` covers all modes, including N1 > N_syn.
- Reachability |r_p| ≤ 0.9 is asserted. P2 at ρ = 0.3 gives r_p = 0.42.

**§2.3 levels**
- L3/L6/L6c raise on non-zero overlap. The grid and train_api use ρ = 0 there.
- L6 effects are on the allelic scale, with zero effect where FIN is monomorphic.
- L4 drops 5% of variants and flips 0.5% in one trait.

**§2.4**
- The total PVE cap of 0.5 is applied to the realised genetic variance, by redraw; after 50 failed redraws the candidate is dropped.
- P2 is fixed at N1 = 100k, N2 = 50k.
- P3 sub-types are 50:50.
- Training and validation use both ρ signs; the test uses positive ρ.

**§5 B L0\***
- p = 20 (D6(g)), with data generated with R̃ = (1−η)R + ηI, the same R̃ that is stored.
- No background, no M loci, ≤ 2 causal variants per trait.
- 8-component λ mixture; methods get the true ρ (rho_se = 0).
- Natural class shares and their MC SE are stored for the class-prior correction.

**§8.5 / §8.6 test set**
- 34 cells, with L3/L6 only at overlap 0.
- P1 nested; P2 at N1 = 100k, N2 = 50k.
- GridPlan: S/W counts, D uniform over the 5 r² bins with deterministic relaxation (P3-11 item 4), G/U up to their caps.
- Shifted arm:
  - pool-swap world, with references L1swap (pool-B mosaic, n = 2,000) and L2swap (pool-A, n = 254);
  - P1 r_p ∈ {0, 0.6};
  - mix S .15 / D .15 / W .5 / M .2, biobank-strong share 0.5, background share 0.5, up to 4 causal variants per trait;
  - natural fill: K = M = 10 per visit, so the per-visit cap never distorts the class mix;
  - seed rng(20261028, "shifted", …).
- P3-11: entry point `sim.testset:generate(config, out_dir, seed)` with K = 200, M = 3.

## Explicit answers

- **Can `src/sim/oracle.py` be re-frozen?** **Yes.**
  - Its sha256 is `9aa08bc614b7406cc3ec89d0d6b2123ce20eb1bd41bdca7524ddc51e69a016bf`.
  - It has not changed since 3416f2e, and it matches the entry already in the WIP `configs/FROZEN_SHA256`.
  - The implementation matches DESIGN v4 §1 (check A).
  - Defect 5 is cosmetic and does not justify a change.
- **Is `sim.testset:generate` safe** (cannot write test loci outside `scripts/eval_test.py`)? **Yes.**
  - The parent call requires `has_test_access()` and eval_test.py on the call stack.
  - The pool worker, after fix 1, requires `has_test_access()`, which is true only while eval_test.py's `started` nonce is open at a frozen-tagged HEAD.
  - Every other path is closed:
    - `cells.build_set` and `l0star.build_set` with split "test" stop at `split_windows`;
    - any locus whose split, block ids or coordinates touch a test block is refused by `write_locus_set`;
    - `TrainSampler` accepts only "train".
  - Without fix 1, generate is still safe, but the authorised run itself fails.

## Commits in this branch

- `ce7001c`:
  - the `src/sim/testset.py` fix for defect 1;
  - the regression test;
  - `reviews/p2_review_sim_checks.py`.
- `ceef79c`: the paired ρ̂ check (C2).
- This review file.
