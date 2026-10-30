# src/baselines — tier-1 baseline wrappers (owner: baseline agent)

Implements DESIGN v4 §4.1 (tier 1), the adapters of §4.3 and decisions D4 and D6(f). **Since M3.5 every score formula
comes from `src/eval/adapters.py`** (the single, hash-protected implementation); the wrappers call its
method-specific helpers directly and this package holds no adapter code. Every wrapper follows INTERFACES §4:
`Method.fit_batch(list[LocusData]) -> list[dict]` with the INTERFACES §3 columns
(`locus_id, method, s, P_S, P_D, P_W, E1, E2, call, status, runtime_s, extra`; `call` is left `None` for the harness,
`E1/E2` are NaN, `extra` is a strict-JSON string).

```python
from baselines import get_method, all_methods      # PYTHONPATH=src, env/activate.sh sourced
rows = get_method("coloc_susie").fit_batch(loci)    # loci: eval.io.LocusData

# through the run harness (manifest, predictions.parquet, LEDGER), memory-limited workers, per-method timeout:
from baselines.run import run_baseline
r = run_baseline("coloc_susie", "data/processed/sets/screen_v2.h5", ids=None, mem_budget_gb=48, max_cpus=24)
```
CLI: `python -m baselines.run --method coloc_susie --set <set.h5> [--ids-file f] [--mem-budget-gb 48] [--max-cpus 24]`.
`run_baseline` never lets the harness compute metrics (that needs oracle labels, which method code must not read;
`eval.truth` refuses callers from `baselines.*`); metrics are computed by the evaluation side from predictions.

| name | method | runtime env | s | P_S | P_D (ungated) | P_W |
|---|---|---|---|---|---|---|
| `coloc_abf` | local vectorised coloc.abf (`abf.py`), W = 0.15, p12 = 5e-6; `extra.grid` holds PP0–4 for W ∈ {0.1, 0.15, 0.2} × p12 ∈ {5e-6, 1e-5} | Python | PP4/(PP3+PP4) | PP4 | PP3 | PP0+PP1+PP2 |
| `coloc_susie` | susieR 0.14.2 `susie_rss(z, R, n, L=10)` per trait + `coloc.bf_bf` on CS effects (= coloc.susie), p12 = 5e-6; fallback coloc.abf (W 0.15, p12 5e-6) if a trait has no CS | R main lib | max over CS pairs PP4/(PP3+PP4) | max PP4 | (1 − max PP4)·max PP3 | 1 − P_S − P_D |
| `coloc_susie_mm` | as above with susieR 0.16.6 (`env/rlib_susie016`, own R process), `R_finite = n_ref, R_mismatch = "eb_mix"` | R susie016 | same | same | same | same |
| `sharepro` | SharePro_coloc 059caab, default run with its automatic K = 1 fallback (D4), in-process | Python | P_S | 1 − Π_k(1 − share_k) | (1 − P_S)·P(sig1)·P(sig2), P(sig t) = 1 − Π_k(1 − Σ_j γ_jk δ_tjk) (per-effect, D6(f)) | 1 − P_S − P_D |
| `colocboost` | colocboost 1.0.10 pairwise, LD given, `residual_correlation = [[1, ρ̂],[ρ̂, 1]]`, `output_level = 2` | R main lib | P_S | 1 − Π_CoS(1 − cos_npc) | (1 − P_S)·P(sig1)·P(sig2) | 1 − P_S − P_D |
| `colocboost_m1` | as above with `M = 1` (the package's documented LD-mismatch mode) | R main lib | same | same | same | same |
| `mvsusie` | mvsusieR 0.3.0 `mvsusie_rss`, `create_mixture_prior(R = 2)`, residual variance fixed to [[1, ρ̂],[ρ̂, 1]], L = 10 | R mvsusie libs | P_S/(P_S+P_D) | inclusion–exclusion (DESIGN §5) | inclusion–exclusion | 1 − P_S − P_D |
| `prop_coloc` | prop.coloc 2.0.0 (primary proportional test, D4), `n = c(N1, N2)`, `tau = ρ̂` | R main lib | p_full | NaN | NaN | NaN |
| `coloc_prop_test` | colocPropTest 0.9.3 `run_proptests` defaults (secondary) | R main lib | min FDR | NaN | NaN | NaN |
| `lead_snp` | lead-SNP rule | Python | 1{r²(lead1, lead2) ≥ 0.8} | NaN | NaN | NaN |

Files: `methods.py` (wrappers, registry; scores via `eval.adapters`), `run.py` (harness `MethodSpec`s, timeouts, memory model → worker counts),
`pilot.py` (200-locus timing/failure pilot, freezing rules), `benchmarks/` (runtime tables, pilot records,
`frozen_timeouts.json` once frozen), `abf.py`
(coloc.abf with tunable W), `sharepro.py` (in-process SharePro), `rworker.py` + `r/worker.R` (persistent R
workers), `locus.py` (LocusData access, dense LD, eval.io import with a marked stub fallback), `toy.py` (toy loci
from 1000G chr22 for tests/benchmarks), `bench_runtime.py` (runtime/memory pilot).
Tests: `tests/baselines/` (run: `source env/activate.sh && python -m pytest -q tests/baselines`).
Known-direction controls on simulated validation loci: `tests/baselines/pilot/controls.py` (reads oracle labels,
therefore outside this directory).

## R workers
One long-lived `Rscript` per library setting per Python process (`rworker.get_worker(lib)`; keyed by pid, so
harness worker processes each get their own). Requests are JSON lines; arrays travel as raw float64 files in
`/tmp/claude-<uid>/baselines_ipc/`; replies are one `@@RESP` JSON line (17 significant digits, exact double
round-trip). Package chatter (cat/print) is diverted to `logs/baselines/rworker_<lib>_<pid>.log`; R warnings and
the first messages are returned in `extra.r_warnings`. BLAS/OpenMP threads = 1. A request over its timeout kills
the R process group (status `timeout`); the next request respawns the worker (~2 s start-up).
Library settings: `main` (susieR 0.14.2), `susie016` (susieR 0.16.6 first on R_LIBS), `mvsusie` (0.16.6 + mvsusieR);
the wrappers assert the susieR / mvsusieR version returned by R on every call.

## Status values
`ok`; `nonconverged` (SuSiE fit of either trait, mvSuSiE, or SharePro's final run did not converge — the output is
used and flagged, DESIGN §4.3); `failed` (R error, non-finite or inconsistent input, Python exception; `extra.error`);
`timeout`. Failed/timeout rows carry NaN scores; least-favourable scoring is the harness's job.

## Decisions / assumptions (not fixed by DESIGN; PI may overrule)
1. **Summary-statistic scale.** Wrappers assume standard OLS summary statistics: z = beta/se is the t-statistic
   and se the OLS standard error (what real studies publish and what susie_rss(n), SharePro's
   bhat = z/√(z²+n), prop.coloc's a = 1/(n se² + b²) and mvSuSiE's N-adjustment assume). DESIGN §2.2 literally gives
   z = √N·r (correlation scale) with the null se 1/√(2Nf(1−f)); for strong small-N signals this is not
   self-consistent (n·se² + b² = (1 + r²)/var(x) instead of 1/var(x)). Measured on toy loci (λ = 30, N2 = 5000):
   prop.coloc rejects proportionality for 70% of truly shared loci on the correlation scale vs 5% (nominal) with
   OLS statistics. `toy.py` therefore converts to OLS statistics by default (`stats="ols"`:
   se_ols = se·√(1 − z²/N), z_ols = beta/se_ols). **Adopted by the PI (D6(a))**: the generator outputs OLS
   statistics from screen_v2 on; screen_v1 (correlation scale) is superseded and used here only for engineering
   dry runs.
2. **N given to methods that take one N per trait:** the median of the per-SNP n_t (constant unless L5).
3. **ρ̂:** `rho_hat` attribute; NaN/missing → 0. Clipped to ±0.99 (ColocBoost, prop.coloc) and ±0.95 (mvSuSiE,
   positive-definite residual matrix). coloc, coloc-SuSiE, SharePro, colocPropTest have no overlap input.
4. **coloc-SuSiE:** susie_rss called directly (not `coloc::runsusie`, which retries until convergence) with package
   defaults besides L = 10 (coverage 0.95, min_abs_corr 0.5; 0.14.2: max_iter 100; 0.16.6: its RSS default 50,
   PIP convergence when R_finite is active). CS rows of `lbf_variable` → `coloc.bf_bf(p1 = p2 = 1e-4, p12 = 5e-6)`,
   exactly as `coloc.susie`. Fallback coloc.abf uses W = 0.15 (coloc default), p12 = 5e-6, computed in Python
   (validated identical to R). `R_finite = n_ref` is passed at every mismatch level (at L0 n_ref is whatever the
   locus attribute says).
5. **SharePro (D6(f)):** P(trait-t signal) = 1 − Π_k(1 − Σ_j γ_jk δ_tjk) over **all K effects** (SharePro's
   per-effect signal probability; the §4.3 "if reported" clause). Not restricted to the reported effect groups
   (P_S uses those, D4), because an effect that is too diffuse to form a group is still evidence of a signal.
   The PIP fallback 1 − Π_j(1 − PIP_tj) (which underestimates spread signals: 0.66 vs 1.0 on a λ = 30 signal
   spread over 8 SNPs) is kept in `extra.P_sig_pip` / `extra.P_D_pip`; per-effect probabilities in `extra.P_eff`.
6. **ColocBoost:** colocboost reports no per-trait probability; P(trait-t signal) = 1 if trait t is in a CoS or has
   a trait-specific uCoS (`output_level = 2`), else 0 (so its P_D is coarse/tied; the ties are reported, not
   fixed, D6(f)). Both modes (`colocboost`, `colocboost_m1`) are run at every level (D6(f)). Locus P_S = 1 − Π(1 − cos_npc), the
   same aggregation D4 chose for SharePro. `get_robust_colocalization` (post-hoc) is not applied.
7. **mvSuSiE:** z adjusted per trait with its own N (√((N−1)/(z²+N−2))·z, which is what mvsusie_rss does for a
   common N), then `mvsusie_rss` without N (it accepts one N only; N1 ≠ N2 in P1/P2). Prior: package canonical
   mixture for 2 traits (singletons + independent + 0.25…1 correlated shared). For effects that form a CS (others
   q = 0), q_lt from single_effect_lfsr[l, t] goes into the Family A inclusion–exclusion formulas
   (adapters.inclusion_exclusion; tested against exact enumeration). **Decision P3-2: the hard rule
   q = 1{lfsr < 0.05} (DESIGN §4.3, pre-registered) is primary** (`MvSusie(rule="hard")`, the default since p4);
   the soft rule q = 1 − lfsr (D4) is the sensitivity analysis, stored per locus in `extra.soft` (and the hard
   scores also in `extra.hard_lfsr005`), so one run serves both. Runs before p4 (pilots) used the soft rule.
8. **prop.coloc:** alpha = 0.05 (only p_cond depends on it; the default grid of 99 alphas multiplies its runtime);
   s = p_full (D4). `n = c(N1, N2)` always (two-sample formula; with tau = ρ̂ it covers overlap).
9. **colocPropTest:** s = min over tests of the FDR-adjusted p (the package's recommended statistic). If no test is
   run, s = 0.5 with `extra.undefined = true` (DESIGN ties rule).
10. **Lead-SNP rule:** lead = argmax |z| per trait; s = 1 if same SNP or r² ≥ 0.8 in the inference LD.
11. **Input validation:** any non-finite z/beta/se/n or non-positive se → status `failed` (susieR 0.14.2 otherwise
    silently "fits" NaN input).
12. **LD:** factor-form LD is densified as X'X/n_ref in float64 (cached on the locus object); runtime includes it.

13. **Single adapter implementation (D6(f)).** `methods.py` calls `eval.adapters.{coloc_abf, coloc_susie, sharepro,
    colocboost, mvsusie, proportional_test, lead_snp}` directly (the eval agent added the method-specific helpers
    in M3.5); `src/baselines/adapters.py` was deleted, so no formula and no shim exists here. A test asserts this
    (`test_adapters.py`, AST check). No helper was missing.
14. **Harness wrapping (INTERFACES §4).** `run.py` exposes every baseline as an `eval.harness.MethodSpec`
    (`module:Class`, built inside each harness worker so each worker owns one persistent R process); class-level
    `name`s (`ColocSusieMM`, `ColocBoostM1` subclasses) so the harness can name runs without instantiating.
    Per-locus timeout T is enforced by the harness (kills the worker's process group, incl. R); the wrapper's own
    R-call timeout is 0.9·T so a slow R call normally ends as a clean `timeout` row without respawning the worker.
15. **Worker counts by memory (DESIGN §7.7, D6(f)).** Per-worker peak (Python worker + its R process) is modelled as
    a + b·(p/1000)² GB; workers = min(max_cpus, ⌊mem_budget / model(p_max of the run)⌋). Provisional (a, b) from the
    M3 runtime pilot; refitted from the 200-locus pilot (rows carry `extra.mem_mb` = running peak RSS of the
    Python worker and of its R process). Default budget 48 GB / 24 CPUs (the 128 GB job limit is shared by agents).
16. **Pilot sampling and freezing rules** (`pilot.py` docstring; fixed before the pilot ran): 200 ids of a validation
    set, equal per cell, seed derive(20261029, "baselines/pilot200", set); methods run one after another; timeouts
    re-run once at 3× the timeout (censoring check); frozen timeout = max(60 s, 3 × max observed runtime) rounded
    up to 30 s; memory b = 1.25 × LSQ fit; a method whose projected M4 wall-clock (60,000 loci, 110 GB, 60 CPUs) is
    > 12 h runs on the 400-per-cell subsample (`pilot.subsample_ids`, seed derive(20261029, "baselines/subsample400",
    cell), identical for all methods; optional allocation proportional to strata supplied by the evaluation side).
17. **Trait-2 in-sample LD at L0** (`R2`/`X_ref2`, sim README D11) is not read by `eval.io.LocusData`, so every
    wrapper uses trait 1's LD at L0 (DESIGN §2.3 "methods taking one R get trait 1's"). coloc-SuSiE (per-trait
    fits) and ColocBoost (accepts one LD per outcome) could use trait 2's own LD; proposal to the eval agent /
    PI in reports/baseline_M3.5.md.
18. **ColocBoost 1.0.10 bug workaround.** With a residual correlation given, `colocboost()` crashes ("'a' is 0-diml")
    when post-hoc filtering (`get_robust_colocalization`) leaves a CoS with no outcome: the package's placeholder
    weight `matrix(1, nrow)` has no column names, so `get_integrated_weight` solves a 0×0 matrix. Found on 1/24
    screen_v1 dry-run loci (L2, P3 ρ = 0.5). The R worker patches `get_integrated_weight` (assignInNamespace) so that
    ONLY an unnamed weight matrix goes through the package's own `residual_correlation = NULL` branch (= what the
    package does without a correlation); all other calls are unchanged (self-check `cb_patch_check` in
    `test_rworker.py`: named inputs identical, the bug reproduces unpatched, the placeholder equals the NULL branch).
    Uses are counted per locus in `extra.cb_patch_uses`. Alternative (PI may choose): no patch, count as failures.
19. **coloc-SuSiE iterations.** susie_rss is called with its default max_iter (0.14.2: 100; 0.16.6: its default) and
    non-converged fits are used and flagged (`nonconverged`, DESIGN §4.3), instead of `coloc::runsusie`'s
    "repeat with 100× max_iter until converged". Evidence (`conv_check.py`, screen_v1 dry run, all 10 coloc_susie
    and 7 coloc_susie_mm non-converged loci, all at L2/L4): with max_iter 1000 / 10000, susieR 0.14.2 converges on
    none and *stops with an error* ("estimated prior variance is unreasonably large", i.e. LD mismatch) on 5/10 and
    10/10 loci; where it still returns, s is unchanged. 0.16.6 converges on 3/7 at 1000 (s unchanged in order of
    magnitude, all D-like loci stay s < 0.02) and on no further locus at 10000, at ~10× runtime and one 3000-s
    timeout. So runsusie semantics would turn mismatch non-convergence into failures/timeouts without changing the
    answers (`benchmarks/convergence_screen_v1.tsv`).
20. **mvSuSiE with every prior variance estimated 0** returns a scalar NA `single_effect_lfsr` and no CS. This is
    "no active effect" (P_W = 1, s = 0.5 by the ties rule), not a failure. Before the fix the wrapper failed on it
    (7/200 W loci of the screen_v1 pilot); regression test replays the observed R reply. A malformed lfsr *with*
    credible sets still fails loudly.
    **Package defect found in p4 (kept as `failed`, not patched):** mvsusieR 0.3.0's `format_mvsusie_output` sets
    `has_mixture` from effect 1 only (`!is.null(s$pi_V_posterior[[1]])`). When effect 1 (and possibly others) is
    estimated null (V = 0) but later effects form CSs, every lfsr is returned as scalar NA although CSs exist, and
    the wrapper fails loudly ("cannot reshape array of size 1"). Reproduced on screen_v2__L0_P1_r_p=0.0__W__0020802
    (V = 0, 0, 745, 178, 0, ...; pi_V_posterior NULL for effects 1, 2, 5–10; CS = effects 3, 4). Seen on 1/2000
    loci of the L0|P1|r_p=0.0 screen cell. A patch (decide has_mixture from any non-NULL effect) is possible but
    was not applied mid-run; the rate is reported per cell and scored least-favourably (DESIGN §4.3).
21. **prop.coloc failures are kept as failures** (no fallback, not pre-registered): "Unable to compute the standard
    errors" and "object 'solve0' not found" are the same package condition (a singular LD sub-matrix after its
    clumping: the package wraps `solve()` in `try()` and then reads the unset `solve0`), plus "system is
    computationally singular". 11/200 screen_v1 pilot loci, all at L2/L4 (reference LD n = 150). Scored
    least-favourably by the metrics and reported per cell (PI decision needed because > 1% per cell, DESIGN §4.3).
    **Diagnosis (decision P3-20, p4): method limitation, not a wrapper bug.** All 7 screen_v2 pilot failures
    (L2/L4 P1, reference LD from n_ref = 150 individuals) were reproduced outside the wrapper by replaying
    prop.coloc's own clumping (pairwise r² < 0.6, alternating traits) and top-10-per-trait selection on the exact
    LD the wrapper passes: 6/7 have an EXACTLY singular selected LD sub-matrix (smallest eigenvalue −2e-15 … 1e-17,
    cond ~1e300; 3–4 SNPs with x_c = x_a + x_b in the 150 reference genotypes, maf_ref 0.02–0.40), so `solve(A1)`
    fails inside the package's `try()`; the 7th (a W locus, N2 = 100) has a well-conditioned sub-matrix
    (min eig 0.24) but 1 − B'A⁻¹B = −0.33 for trait 2, the package's own "sample size too small" stop (implied
    multivariable R² > 1 with mismatched LD). The wrapper passes the inputs unchanged, as a user would; no fix
    and no retuning (P3-20). Known-answer test: `tests/baselines/test_prop_coloc_limits.py` (the same toy locus
    fails iff one selected SNP is the exact sum of two others in the reference genotypes).

22. **colocPropTest 0.9.3 drop bug (kept as `failed`):** when exactly two unique top SNPs remain after its tagging,
    `pairs[, pairs[1, ] < pairs[2, ]]` drops the 2 x 1 matrix to a vector and the next `pairs[, ok]` stops with
    "incorrect number of dimensions". The package's intended outcome for one pair is `NULL` (fewer than 2 tests →
    "no strongly significant tests"), which the wrapper would score s = 0.5 (undefined). Reproduced in plain R on
    screen_v2__L0_P1_r_p=0.0__D_r2_0.5-0.8__0006929 ("after tagging, unique topsnps: 2"); 3/2000 L0|P1 screen loci.
    Not patched (no pre-registered fallback); counted as failures.

## Known method behaviour found by the controls (not wrapper bugs)
- SharePro false sharing between two strong (λ = 30) distinct signals rises with their LD: 0/15 loci at r² < 0.02,
  2/33 at 0.02–0.05, 8/12 at 0.05–0.1 (s > 0.5). Identical to the official CLI on the same input
  (`test_sharepro_equiv.py`); `test_distinct[sharepro]` is marked xfail only when this happens.
- prop.coloc stops with "Unable to compute the standard errors…" on some strong loci (1/40 toy loci); recorded as
  `failed`.

## Known method behaviour under mismatched LD (bench loci, n_ref = 150, λ = 15)
- mvSuSiE-RSS (no mismatch handling) fits 10 CSs per trait and calls the distinct locus shared (s ≥ 0.99 at all
  three p); its s on null-trait-2 loci is P_S/(P_S+P_D) of two ~0 numbers (DESIGN formula; such loci are W).
- coloc-SuSiE 0.14.2 also fits up to 10 spurious CSs per trait but coloc.bf_bf keeps s ≈ 0 on the distinct loci;
  susieR 0.16.6 eb_mix sets `SER_fallback` / `artifact_flag` and fits 1 CS per trait.
- ColocBoost (default) misses the shared signal at p = 5000 (s = 0); M = 1 does not.

## Runtime and memory (pilot, `bench_runtime.py`; 3 toy loci per cell; machine shared with other agents)
Medians (max) in seconds per locus incl. dense-LD build; peak RSS of the R worker (VmHWM). Full table:
`benchmarks/runtime_M3.tsv` (raw: `benchmarks/runtime_M3*.jsonl`). R worker start-up ≈ 2 s once per process.

| method | p=1000 in / mm | p=3000 in / mm | p=5000 in / mm | R peak RSS p=5000 (GB) | proposed timeout (s) |
|---|---|---|---|---|---|
| coloc_abf | 0.001 / 0.001 | 0.001 / 0.001 | 0.002 / 0.002 | – | 60 |
| lead_snp | <0.001 | <0.001 | 0.001 | – | 60 |
| sharepro | 0.03 / 0.03 | 0.25 / 0.63 (0.82) | 0.85 / 4.5 (7.0) | – (Python ≈ 1.0) | 300 |
| coloc_susie | 0.52 / 1.7 | 2.5 / 11.8 (14.6) | 6.2 (8.7) / 22.8 (24.7) | 1.9 / 2.1 | 600 |
| coloc_susie_mm | 0.74 / 2.6 | 7.1 / 14.7 (21.5) | 30.2 (49.4) / 31.6 (44.1) | 2.5 / 1.9 | 900 |
| colocboost | 0.53 / 3.8 | 1.5 / 18.2 (20.9) | 2.7 / 37.5 (37.6) | 1.0 / 1.3 | 900 |
| colocboost_m1 | 0.12 / 0.12 | 0.51 / 0.53 (0.78) | 0.99 (1.35) / 0.89 (1.15) | 0.9 / 0.9 | 120 |
| mvsusie | 1.8 / 13.1 | 10.3 (39.2) / 37.4 (178, nonconverged) | 22.1 / 76.4 (87.1) | 1.4 / 1.7 | 1800 |
| prop_coloc | 1.0 / 0.2 | 1.5 / 0.5 (1.3) | 1.1 (2.1) / 1.6 (2.0) | 1.0 / 1.0 | 300 |
| coloc_prop_test | 0.5 / 1.0 | 1.3 / 1.1 | 2.4 (2.8) / 2.2 | 1.0 / 1.0 | 300 |

Peak memory grows ~p²: expect ≈ 4× at p = 10k (susieR ≈ 8–10 GB per worker), so susie/mvSuSiE workers must be
limited by memory (≤ 2 GB/core), not by core count. Timeouts: max observed × ~4 (p up to 10k) × ≥ 3 margin;
provisional until the 200-locus validation pilot (DESIGN §4.3).
