# Eval-infrastructure engineer — M3.5 report (DESIGN v4 conformance, decisions D4/D6)

Owner files touched: `src/eval/{adapters,metrics,strength_control,reference_methods,harness,io}.py`, `src/eval/README.md`,
`tests/eval/{test_eval_v4,test_eval_benchmark,test_eval_adapters_config}.py`, `tests/eval/eval_dummy_methods.py`,
`scripts/{run_benchmark,verify_run,eval_test}.py`, `results/eval_dryrun/` (dry-run artefacts, 0.7 MB).
Proposals for the PI (left **uncommitted**, as at M3, because `configs/` is PI-owned): `configs/testset_v1.yaml`,
`configs/test_methods_v1.yaml`, `configs/FROZEN_SHA256`. New decisions/assumptions: `src/eval/README.md` items 8, 9, 11,
19b and 27–37.

## 1. Test-set config (`configs/testset_v1.yaml`) — DESIGN v4 §8.6
- **34 cells**, listed explicitly (`{level}|{pairing}|{param}={value}`, the sim's format): 4 levels × 7 pairing/overlap cells
  plus L3 and L6 with only the overlap-0 cells (P1 r_p = 0, P2 ρ = 0, P3 ρ = 0).
- P2: N1 = 100k, N2 = 50k. P3: positive ρ at test; sub-types eQTL×eQTL (N ∈ {300, 1000}, individual-level, molecular)
  and lab×lab (N = 100k, factor form, GWAS-type). Power thresholds by trait type (§1).
- Per cell: S 400 / D 400 / W 400, G+U ≤ 200, rejection sampling with seed 20261028. D is spread uniformly over the 5 r² bins
  (80 per bin).
- Power matching: 5 × 5 bins on the class-determining (|μ1|, |μ2|), with trait-specific quantiles over the pooled S ∪ D of
  the cell. Strata are not used for matching.
- Reporting strata: r² edges [0, .1, .3, .5, .8, .95); d edges [2, 4, 8, 16, ∞).
- Pooling: realistic over 27 cells and ideal over 3 cells. The explicit weights equal `metrics.cell_pooling_weights`
  (checked by a test).
- C3/C4 specification, controls including the strength-only control, failures, bootstrap.
- Shifted arm in full: 6,000 loci, derive(20261028, "shifted"), L1swap (n_ref 2,000) and L2swap (n_ref 254), P1 with
  r_p ∈ {0, 0.6}, class mix, architecture.
- All PROPOSAL lines are resolved (a test asserts that none remain).

**Still open (listed in the file under `open_items`):**
1. `sim.testset:generate` does not exist yet (simulation agent).
2. P3 mix eQTL×eQTL : lab×lab = 0.5 : 0.5 is the sim default; it needs PI confirmation.
3. Shifted arm: an equal split over r_p, giving 4 cells × 1,500. DESIGN fixes only "half each" by level.
4. D generation strata are uniform over r² bins (d is not crossed). The rare r² bins (0.5–0.95) cost many candidates, about
   1% acceptance in screen_v1.

## 2. Metrics (`src/eval/metrics.py`)
- **Power matching per v4:** within cell, on pooled S ∪ D quantiles, with no r² strata. A test checks that changing D r²
  leaves the weights bit-identical.
- **C3 thresholds:** t_S comes first. Then t_D is set with S precedence: a W locus counts as a confident D only if
  P_S < t_S, P_D ≥ t_D and the E gate passes. Hand case: 40 W loci; t_D drops from just above 0.9 (old independent rule)
  to just above 0.1 (sequential rule).
- **`c3_test`:** non-inferiority to every eligible baseline uses a one-sided margin of 0.01. Superiority to the best
  eligible baseline per recall is Holm-adjusted over the two recalls. Everything comes from one paired cluster bootstrap.
  Known outcomes:
  - identical candidate → non-inferior, not superior, fails;
  - better recall_D → passes (p_holm < 0.05);
  - worse → non-inferiority fails.
- **`bootstrap_p_value`:** the one-sided p is computed by inverting the BCa interval. Checks:
  - the p-value at the 90% BCa bound is 0.05 ± 0.004;
  - null rejection rate is 1–11% over 200 clustered null datasets.
- **Other additions:**
  - `holm`;
  - `contrast` (paired differences from a joint vector bootstrap; equal to `paired_bootstrap` to 1e-12);
  - `endpoint_vector`;
  - bootstrap unit "block" or "locus";
  - `stratified_report` (AUPRC/AUROC per r² and d stratum, re-matched; D-vs-W; false-S and D-call rates; an "out" bin for
    r² ≥ 0.95);
  - per-level AUROCs;
  - per-cell calibration (§8.2 "per mismatch level × ρ");
  - a `shifted` pooling setting.
- **Strength-only control (D6(e)), `strength_control.py` + `reference_methods.StrengthOnly`:** see README item 31.
  - It uses only lead |z| per trait, stratified by pairing type, with 8 × 8 smoothed bins.
  - It is cross-fitted (K = 5) on validation benchmarks. A full table is written for later runs.
  - It runs through the harness with `needs_ld = False`; a test confirms LD is skipped.
  - Tests: strength separates S from D before matching (AUROC > 0.7). After matching it moves toward 0.5. D vs W > 0.97.
    Flipping a locus's label leaves its own cross-fitted score unchanged.

## 3. Adapters, harness, test lock
- **Adapters:** `src/eval/adapters.py` is now the single implementation. New helpers: `sharepro` (per-effect P_D, D6(f)),
  `colocboost`, `mvsusie` (soft per D4, hard per §4.3), `inclusion_exclusion_q`. Baseline-signature aliases are provided.
  `coloc_susie` drops NaN rows itself.
- **Baseline switch:** the baseline agent has already switched (commit b737d8b: `src/baselines/adapters.py` only forwards).
  A test checks that the shim equals the unified helpers.
- **Results policy D6(h):** the manifest records `results_policy`. Predictions ≥ 5 MB get a run-local `.gitignore`.
  `verify_run` checks this and names a missing large file explicitly.
- **Test lock:** `eval_test.py` refuses a methods file that is not filled in. The conditions are:
  - status ≠ frozen;
  - no methods listed;
  - an entry missing target or timeout_s;
  - no primary candidate, or a primary not in the list.

  The methods file's sha256 is recorded in each test-run manifest. `configs/test_methods_v1.yaml` is the template, and the
  PI fills the methods list.
- **FROZEN_SHA256 (proposal):** the whole DESIGN.md (v4), plus:
  - metrics.py, adapters.py and strength_control.py;
  - oracle.py and the split files;
  - both test configs.

  `check_frozen` → `FROZEN OK` at the time of writing. Note: `src/sim/oracle.py` has an **uncommitted change by the sim
  agent**, so its hash changed (d562789c → 9aa08bc6). The PI must re-run `--update` at tag time.

## 4. Benchmark runner (`scripts/run_benchmark.py`)
It takes method(s) × locus set(s) and produces:
- harness runs, or reused runs via `--reuse`;
- the strength control;
- `benchmark.json`, containing:
  - realistic, ideal and shifted co-primaries with BCa CIs from one joint bootstrap;
  - per-cell, per-level and macro tables;
  - D-vs-W AUROC;
  - strata;
  - C3 operating points (in-sample, or given via `--operating-points`);
  - calibration, raw and recalibrated;
  - confusion and FDR;
  - Δ vs the strength control;
  - Δ vs `--candidate`, plus a C3 dry run;
  - failures and runtime;
- `summary.md`, `operating_points.json` and `manifest.json`;
- `--verify`, which re-verifies every run and recomputes the whole benchmark.

Optional features: a per-cell deterministic subsample, and bootstrap unit block or locus.

**Dry run** on the validation screen_v1, 100 per cell = 600 loci (infrastructure only: screen_v1 predates the OLS
statistics of D6(a)):
```
python scripts/run_benchmark.py --locus-set data/processed/sets/screen_v1.h5 --methods results/eval_dryrun/methods_dryrun.yaml \
  --results-dir results/eval_dryrun --label dryrun_screen_v1 --per-cell 100 --n-boot 2000 --unit locus --n-jobs 16 --control-workers 16
python scripts/run_benchmark.py --verify results/eval_dryrun/benchmarks/20261026T161829Z_dryrun_screen_v1   # BENCHMARK OK
```
Results (realistic setting, locus bootstrap):

| Method | AUPRC S-vs-D | AUROC D-vs-W |
|---|---|---|
| coloc.abf | 0.938 [0.886, 0.972] | 0.987 |
| lead-SNP | 0.782 | — |
| strength-only control | 0.431 | **0.922** |

- At the C3 points, coloc.abf has t_S 0.767 and t_D 0.254, with W→S 1.9% and W→D 2.3%.
- Wall time: about 6 minutes, of which about 2 minutes is attribute and feature reading of the 12k-locus master file. The
  2,000-replicate bootstrap for 3 methods takes about 3 s.

**Finding for the PI:** D vs W is largely a strength question. The strength-only control reaches AUROC 0.92 on D ∪ W, so
D-vs-W claims must be read as a margin over this control. This is now in every benchmark. The control's matched S-vs-D
AUPRC (0.43 < 0.5) shows that residual strength structure can work against a strength-only scorer after matching.

## 5. Tests (exact commands)
- `source env/activate.sh; cd tests/eval; python -m pytest -q -p no:cacheprovider .` → **73 passed** (92 s).
  These are the 54 tests from M3 (updated) plus 19 new.
- `python -m pytest -q -p no:cacheprovider tests/baselines tests/famG` → **112 passed, 1 xfailed** (with the new
  adapters).
- `tests/famA`: 1–4 failures in `test_m35.py` / `test_power_events.py`. These files are being edited right now by the
  famA agent (uncommitted `src/famA/*`). famA calls only the unchanged 5-argument `inclusion_exclusion`.

## 6. Proposals / open items for the PI
1. Commit the three `configs/` files after review. Resolve `open_items` 1–4 in testset_v1.yaml.
2. **mvSuSiE "active":** DESIGN §4.3 says "lfsr < 0.05 defines active" (hard), while D4 says q = 1 − lfsr (soft, the
   baseline's primary). Both are available via `adapters.mvsusie(rule=)`. A decision note should name the pre-registered
   one.
3. C3 dry runs on validation use in-sample thresholds. For the frozen thresholds, use `operating_points.json` from the
   final validation benchmark (non-test cells of the 34-cell grid). Copy them into test_methods_v1.yaml, together with the
   strength_only table `strength_table_full.json`.
4. The in-sample logit-intercept recalibration does not always lower ECE (coloc.abf 0.080 → 0.091). It is an MLE, not an
   ECE minimiser, and it is reported as specified.
5. Speed tip for the baseline agent: methods that need no LD (coloc.abf) can declare `needs_ld = False`. The harness then
   skips dense-LD reads (30–60 MB per L0 locus).
