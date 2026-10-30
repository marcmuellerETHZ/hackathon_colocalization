# src/eval — evaluation infrastructure (owner: eval agent)

Implements DESIGN v4 §4.3 (the single adapter implementation, D6(f)), §6, §7.2–7.4, §7.7, §8 and INTERFACES §2–4,
with decisions D4 and D6 (M3.5 update; changes vs M3 are listed under "M3.5 changes" below).

| File | Content |
|---|---|
| `io.py` | `LocusData`, HDF5 locus-set reader/writer (INTERFACES §2), test-split guard |
| `truth.py` | truth parquet schema, reader/writer. **Method code must never import it** (static test + runtime caller check) |
| `metrics.py` | the single metrics implementation (hash-protected) |
| `adapters.py` | the SINGLE §4.3 adapter implementation (hash-protected, D6(f)): coloc.abf, coloc-SuSiE, SharePro (per-effect P_D), ColocBoost, mvSuSiE, inclusion–exclusion, proportional test, lead SNP; `src/baselines/adapters.py` only forwards here |
| `strength_control.py` | strength-only control scorer (D6(e), hash-protected): lead-\|z\| tables, K-fold cross-fitting, single tables |
| `harness.py` | `run_method(...)`: process-parallel / GPU-batched execution, per-locus timeouts, machine lock, manifest, predictions, metrics, ledger |
| `reference_methods.py` | `LeadSNPRule` (§4.1 item 8); `StrengthOnly` (D6(e) control, `needs_ld = False`) |
| `../../scripts/run_benchmark.py` | validation benchmark runner: methods × locus sets → harness runs → benchmark.json / summary.md / operating_points.json; `--verify` |
| `../../scripts/verify_run.py` | recompute a run's metrics and check hashes/ledger |
| `../../scripts/eval_test.py` | the only producer/reader of test-split data (§7.3) |
| `../../scripts/check_frozen.py` | verify / refresh `datasets/configs/FROZEN_SHA256` (§7.4) |

Tests: `tests/eval/` (`source env/activate.sh; cd tests/eval; python -m pytest -q -p no:cacheprovider .`).

## Usage

```python
# PYTHONPATH=src, env/activate.sh sourced
from eval.harness import run_method, MethodSpec
r = run_method(MethodSpec("baselines.coloc_abf:ColocABF", {"W": 0.15}), "data/processed/screen_v1.h5",
               n_workers=40, timeout_s=300, metric_config={"t_S": 0.8, "t_D": 0.8}, label="M4 coloc.abf")
# GPU method: mode="batch", n_workers=1, batch_size=256 (loci sorted by p, so batches are p-homogeneous)
# timing run: timing=True (exclusive results/.machine.lock + results/.timing.lock; repeat 3x yourself)
```
Paired comparison with CI (frame with several methods; weights are recomputed inside every replicate):
```python
from eval import metrics as M; from eval.truth import read_truth; from common.seeds import rng
fr = M.build_eval_frame(truth, {"A": predA, "B": predB}, mode="least_favourable")   # and mode="excluded"
ci = M.paired_bootstrap(fr, M.auprc_sd, "A", "B", n_boot=2000, rng=rng(20261028, "C1"), n_jobs=32)
st = M.block_sign_test(fr, "A", "B", endpoint="sd")
```
Cost: one 2,000-replicate BCa paired bootstrap on 42,000 loci / 267 blocks takes ~8 s with `n_jobs=32`.

## Decisions / assumptions (not fixed by DESIGN v4 or decisions D4/D6; PI to confirm)

**I/O**
1. Splits file: `datasets/splits/blocks_v1.tsv` (task brief), falling back to `datasets/splits/split_v1.tsv` (DESIGN §3). Expected
   columns `block_id` (or `block`) and `split`; if `chr,start,end` exist, windows overlapping a test block (half-open
   intervals) are also refused. Read lazily; while the file does not exist only the locus `split` attribute is checked.
2. Test access is granted only when `$COLOC_TEST_ACCESS` holds a nonce with an open `started` entry in
   `results/test_access.log`, the entry's commit is HEAD and HEAD has a `frozen-*` tag. Only `scripts/eval_test.py`
   writes such entries. Positive answers are cached per log mtime. The guard prevents accidents; it is not a security
   boundary (`guard=False` exists for tests and is never used by the harness).
3. HDF5: groups starting with `_` are reserved; `LocusData.ld_key` stores LD once under `/_ld/<key>` and hard-links it
   into each locus (readers see the normal `R`/`X_ref`). Locus order = insertion order (`track_order`).
   Strings are UTF-8; `block_ids` is a string-array attribute; unknown attributes go to `LocusData.extra_attrs`.
4. Truth file of `<dir>/<stem>.h5` is `<dir>/<stem>.truth.parquet`. Required columns: `locus_id, oracle_class, mu1,
   mu2, r2, d, rho, block_id, cell_id, split`; optional `mismatch_level, pairing_type, overlap, control` and list
   columns of causal indices/effects (`truth.TRUTH_SCHEMA`). `block_id` = the LD block containing the window anchor
   (TSS/lead) = bootstrap unit. `mu1, mu2` are the class-determining realised noncentralities; metrics use |mu|.
   `rho` is the realised true rho (C4 slope); `overlap` the grid value of the cell (r_p for P1, rho for P2/P3).

**Failure scoring (§4.3)**
5. Failure of a method at a locus = no row, status `failed`/`timeout`, NaN `s`, or NaN `P_S`/`P_D` when that column is
   produced by the method elsewhere. A column NaN for every non-failed locus is "not applicable" (e.g. proportional
   test: P_D) and never a failure. NaN `E1/E2/P_W` on a non-failed locus are legitimate (E: gate not applied).
   `nonconverged` rows with output are used and counted per cell.
6. Least-favourable values by true class: S → s=0, P_S=0, P_D=1; D → s=1, P_S=1, P_D=0; W/G/U → P_S=1, P_D=1, E=1
   (a confident call). "Excluded" drops the union of failures of all methods in the frame (paired comparisons stay paired).
7. Harness `call` is `NA` for failed loci and for methods without P_S and P_D (proportional test, lead-SNP rule).

**Weights (§2.4, §8.1, §8.6)**
8. Power matching (DESIGN v4 §8.6, D6(e)) is done within each evaluation cell on the class-determining (|μ1|, |μ2|):
   5 × 5 trait-specific quintile bins computed on the POOLED S ∪ D loci of the cell (no r²/d strata); in every 2-D
   bin holding both classes each class gets mass (n_S+n_D)/2, bins holding one class get 0. Each cell's S ∪ D mass
   is normalised to 1 and multiplied by its pooling weight; weighted S prevalence is exactly 0.5 per cell.
   **Known limitation:** 5×5 quantile bins remove most but not all power confounding (toy: a power-only scorer has
   AUROC 0.93 raw and 0.56 matched); hence the strength-only control (item 31) in every benchmark.
9. Pooling weights are computed over the cells present: realistic = equal over the levels in L1–L6 present; pairing
   weights P1 .5 / P2 .25 / P3 .25 renormalised over the types present; equal within type over the overlap values
   present (34-cell grid: L3/L6 have only the overlap-0 cells, so e.g. L3|P1|r_p=0.0 gets 0.1; the explicit weights
   are in datasets/configs/testset_v1.yaml and test-checked). Ideal = L0 cells with overlap 0 (same pairing weights).
   `shifted` = cells of levels L1swap/L2swap (shifted arm), same rule. `all` = equal weight per cell.
   Without level metadata every cell gets equal weight.
10. D-vs-W AUROC: within each cell, D and W each get half the cell's pooling weight, uniform within class; no power matching.

**Operating points and secondary metrics (§8.1–8.2, C3)**
11. C3 thresholds (`matched_w_thresholds`, DESIGN v4 §0): t_S = smallest threshold with weighted confident-S rate on
    W ≤ 2.5%, placed just above a W score (`np.nextafter`; tied W scores are never split, conservative); THEN, with
    t_S fixed and S precedence, t_D = smallest threshold with confident-D rate on W ≤ 2.5%, where a W locus counts as
    a D call only if P_S < t_S, P_D ≥ t_D and the E gate passes (E1, E2 ≥ 0.95 where reported, §1). W weights = the
    D-vs-W weights on W. Thresholds are taken from a validation run (run_benchmark writes operating_points.json).
12. Recall at thresholds: S recall weights are uniform within cell × pooling weight; D recall uses the D-vs-W weights.
13. Calibration: binary events (P_S vs class S, P_D vs class D) over all loci, unweighted, ECE with 10 equal-width bins,
    Brier; recalibration p' = sigmoid(logit(p) + a), a by MLE (p clipped to [1e-6, 1-1e-6]), fitted on validation and
    passed as `recal_a_S/recal_a_D` in the metric config of later runs.
14. Realized FDR: unweighted fraction of S (D) calls whose class is not S (D); nominal = mean(1 − P_S) over S calls.
15. False-S slope (C4): OLS slope of 1[P_S ≥ t_S] on the signed true ρ among D (or W) loci, per 0.1 ρ (option |ρ|);
    its CI comes from `cluster_bootstrap` around `false_s_slope` (not computed in metrics.json).
16. Ties: `s = 0.5` for undefined values (adapters never emit NaN s); reported as share with s = 0.5 and share of S/D
    loci in tied groups.
17. Sign test over blocks: per-block difference of each method's share of the weighted AUROC (the per-locus
    Mann–Whitney contributions sum exactly to the AUROC); two-sided exact binomial test, zero differences dropped.
18. Bootstrap: all resampling draws are made up front from the caller's Generator (results independent of `n_jobs`);
    parallelism uses `fork` workers (Python warns about fork in multi-threaded parents; BLAS threads are 1 under
    env/activate.sh). BCa acceleration from the leave-one-block-out jackknife; NaN replicates are dropped and counted.
19. Shuffled-label control in metrics.json permutes the *scores* within cell (labels, weights fixed), seed 20261029.
19b. D strata (DESIGN v4 §8.6, D6(e)): r² edges [0, .1, .3, .5, .8, .95), d edges [2, 4, 8, 16, ∞), half-open bins;
    D loci outside (r² ≥ 0.95) are reported as "out". `stratified_report` restricts the frame to all non-D loci plus
    D_b and recomputes the power matching (same §8.6 rule) for AUPRC/AUROC S vs D_b, AUROC D_b vs W, and unweighted
    false-S and D-call rates at (t_S, t_D). metrics.json has both r² and d reports at the config thresholds.

**Harness**
20. CPU mode: persistent `spawn` workers, one locus per task, each worker in its own process group so a timeout kills
    R children too. Batch mode: loci sorted by p, `batch_size` per task, timeout = timeout_s × batch length,
    runtime_s = batch wall / batch length. runtime_s always = harness wall time including the HDF5/LD read; a
    method-reported runtime_s is kept in `extra.method_runtime_s`.
21. `git.clean` = no modified tracked files and no untracked files under src/, scripts/, datasets/configs/.
    Env lock hash = sha256 over env/requirements.lock, env/renv.lock, env/R_packages_pinned.tsv.
22. Truth is read only in the parent process after all predictions are written; workers never see it.

**Adapters**
23. coloc.abf: P_W = PP0+PP1+PP2. coloc-SuSiE, SharePro/ColocBoost, inclusion–exclusion: P_W = 1 − P_S − P_D (≥ 0).
24. Lead-SNP rule: s = 1 iff r²(lead1, lead2) ≥ 0.8 in the inference LD given to methods.

**Test lock (§7.3)**
25. `eval_test.py` additionally refuses on a dirty tree, on failed `check_frozen.py`, and when the working-tree access
    log has entries not yet committed. Only a `completed` entry blocks forever; `aborted` attempts are logged.
26. The test-set generator is called through the entry point named in `datasets/configs/testset_v1.yaml`
    (`sim.testset:generate(testset, out_dir, seed) -> [locus-set paths]`, proposal for the simulation agent); the methods
    run are listed in `datasets/configs/test_methods_v1.yaml` (to be written and frozen by the PI).

## M3.5 changes (DESIGN v4, decisions D4/D6)

**C3 test, p-values, contrasts**
27. `c3_test(frame, candidate, baselines, thresholds)`: recalls of S and D at validation thresholds (recall_at
    weights, S precedence, E gate). Non-inferiority to every eligible baseline: one-sided H0 diff ≤ −0.01, p < 0.05
    for both recalls and all baselines (intersection–union, no multiplicity adjustment). Superiority to the best
    eligible baseline per recall (highest point recall on the evaluated frame): one-sided H0 diff ≤ 0, Holm over the
    two recalls, passes if the smaller adjusted p < 0.05. All differences come from ONE paired cluster bootstrap.
    Eligible = baselines with a defined P_D in the frame; run_benchmark also excludes prop_coloc, coloc_prop_test,
    lead_snp and strength_only by name.
28. One-sided bootstrap p-values (`bootstrap_p_value`) invert the BCa interval: p = Φ(u/(1 + a·u) − z0) with
    u = Φ⁻¹(F(null)) − z0, F = share of replicates ≤ null (clipped to [1/2B, 1 − 1/2B]); so p < α ⇔ the one-sided
    (1 − α) BCa lower bound exceeds the null. Percentile p (= F) without jackknife or when 1 + a·u ≤ 0.
    Checked: p at the 90% BCa bound = 0.05 ± 0.004; null rejection 1–11% at α = 0.05 over 200 clustered nulls.
29. Joint bootstraps: `endpoint_vector` computes the co-primaries of all methods on one resample (weights shared);
    `contrast` derives paired differences (BCa from the differenced replicates and jackknife values). It equals
    `paired_bootstrap` exactly for the same seed (test).
30. Bootstrap unit: "block" (claims, §3/§8.3) or "locus" (screening kill rules, §3) via `resampling_units`.

**Strength-only control (D6(e))**
31. Features: lead |z| of each trait (observed data only); table stratum = locus `pairing_type` (pooled table for
    unseen strata). 8 × 8 quantile bins of log lead |z| per stratum; P(c | bin) = (n_c + α π_c)/(n + α), α = 5,
    c ∈ {S, D, W, other}; s = P_S/(P_S + P_D). Validation benchmarks cross-fit with K = 5 folds by sha256(locus_id)
    (simulated loci are independent draws, so locus-level folds do not leak labels; tested: flipping a locus's label
    does not change its own score). A single full-set table (strength_table_full.json) is written for later runs
    (e.g. the test run, via datasets/configs/test_methods_v1.yaml). The control runs through the harness like any method
    (`needs_ld = False`: the harness skips LD reads, tested).

**Harness / results / test lock**
32. Results policy D6(h): `results_policy` records size and decision in manifest["results_policy"]; predictions
    ≥ 5 MB get a run-local `.gitignore` (committed with manifest.json and metrics.json); verify_run checks the
    .gitignore and reports a missing (large, uncommitted) predictions file explicitly.
33. `LocusSetReader.read(id, load_ld=False)` for methods with `needs_ld = False`.
34. eval_test.py refuses a methods file that is not filled in: status ≠ frozen, empty methods, entries without
    target/timeout_s, or a primary candidate that is unset or not listed. The methods file sha256 is recorded in
    every test-run manifest.
35. datasets/configs/FROZEN_SHA256 (proposal): whole DESIGN.md (v4), metrics.py, adapters.py, strength_control.py,
    src/sim/oracle.py, split files, testset_v1.yaml, test_methods_v1.yaml. src/baselines/adapters.py (a forwarding
    shim) is not listed; wrappers are pinned by the frozen-* commit.

**Adapters (D4, D6(f))**
36. `coloc_susie` drops non-finite CS-pair rows itself and falls back to coloc.abf when none remain.
    `locus_prob_with_signals` clips P_S to [0, 1] (NaN stays NaN = failure). `sharepro(shares, effect_prob1,
    effect_prob2)` uses SharePro's per-effect activity probabilities (D6(f)); `colocboost(cos_npc, sig1, sig2)` uses
    binary per-trait indicators (ties reported, D6(f)); `mvsusie(lfsr, cs_index, rule)` with rule "soft"
    (q = 1 − lfsr, D4, primary) or "hard" (q = 1{lfsr < 0.05}, the DESIGN §4.3 wording) — see open item in
    reports/eval_M3.5.md.

**Benchmark runner (scripts/run_benchmark.py)**
37. Deterministic per-cell subsample (`--per-cell N`, seed 20261029 keyed by set stem and cell) identical for every
    method. Reused runs (`--reuse NAME=RUN_DIR`) are matched by their manifest locus-set path and filtered to the
    expected ids (missing ids → failures by outer join). Operating points are fitted on the benchmark set (least-
    favourable frame; labelled "in-sample") unless `--operating-points` gives a validation file. Recalibration
    intercepts likewise. Bootstrap seed derive(20261029, "run_benchmark", "bootstrap", bench_id, mode).
    `--verify` re-verifies every underlying run (verify_run), the strength table hash and expected ids, then
    recomputes benchmark.json and compares (tol 1e-9).

## p6.dryrun_testpipeline changes (single test run; review p5_verify_infra fixes)

38. **Single test run = `src/eval/testrun.py`, driven only by `scripts/eval_test.py`.** Inside ONE open access
    window: real-data arms (P3-14; test: builder entry point `real_data_arms.builder` of the methods file),
    generation of the main grid + shifted arm (main thread; cell-level resume: a cell is done when its part
    truth parquet exists), one harness run per (method x locus set) with a per-locus checkpoint, slow methods on
    the nested slow-method subsample (datasets/configs/testset_v1.yaml `slow_method_subsample`, mvSuSiE 20/20/20 per
    P3-17 = first 20 of the 40-draw), the strength-only control, `run_benchmark.evaluate_all` with GIVEN
    operating points (C1 joint bootstrap, C3 test, per-method id lists), C4 slopes, the DESIGN §2.5 real-data
    criterion, `run_benchmark.verify` + `verify_run`, sha256 of every input. `completed` is written only after
    verification succeeds: test truth is unreadable afterwards (review defect 2).
39. **Resume.** An error or SIGTERM writes `interrupted` (nonce stays open); `eval_test.py --resume NONCE`
    appends `resumed` and continues the same run (finished stages, units and loci are kept; DESIGN §11 M7).
    `--abandon NONCE` (PI) writes `aborted`. A failed unit is retried once; if it still fails the run stops
    as `interrupted` unless `--accept-failed-units` (then reported as not run).
40. **Harness.** `run_method(checkpoint_dir=, run_id=)` resumes from `<run_id>.rows.jsonl` (torn last line
    ignored; segments listed in the manifest); `order="lpt"` (largest p first); batch mode re-queues a crashed /
    timed-out / all-failed batch once as single-locus tasks (`extra.harness_batch_retry`).
41. **Orch snapshots (review defect 1).** `eval.io.code_identity`: inside an orch job ($ORCH_CODE = the code
    dir) the code commit is $ORCH_COMMIT in $ORCH_HOME; `git_info`, `has_test_access` and the eval_test
    preconditions (tag at that commit, `git show <commit>:results/test_access.log`) use it. A snapshot is clean
    by construction.
42. **Access cache (review defect 4)** keyed on the sha256 of the log content, not its mtime.
43. **run_benchmark `--method-ids NAME=FILE` (review defect 6):** a method with its own id list is evaluated
    only in a sub-benchmark `subsample_<sha8>/` restricted to that list (with every method whose list covers
    it); the main benchmark excludes it. `--verify` recomputes every sub-benchmark.
44. **Trait-2 LD (famA request):** `LocusData.R2 / X_ref2`; `LocusSetReader.read(load_ld2=True)`; the harness
    passes `load_ld2 = method.needs_ld2` (all runs: run_benchmark and eval_test use the harness).
45. `read_locus_attrs` caches attribute rows per (file, inode, size, mtime) in-process (35k-locus master file:
    287 s per read) and re-applies the guard on every call.
46. **Generation order (PI decision d):** `testrun.order_cells_by_cost` passes the cells slowest first (valid_grid_v1
    per-cell timings, else level rank L3 > L4 > L6 > L2 > L1 > L0); `sim.testset.generate` keeps the order of
    `cells_filter` (each cell's draws use its own rng keys, so the order changes no locus).
47. **P3-23 real-data ρ sensitivities (PI decision e):** when real arms run, the primary candidate (or
    `real_data_arms.rho_sensitivity.methods`) also runs via `eval.reference_methods.RhoOverride` with the pooled
    per-cohort-pair ρ (`rho_hat_pair`, NaN → 0) and with ρ = 0; reported in real_arms.json as sensitivities
    (`P3_23_sensitivity_of`), excluded from the best-baseline comparison. The test-arm builder must write
    `rho_hat_pair` (as data.real.rho_update does for train/valid).
