# p5.verify_infra: independent verification of the infrastructure (validation benchmark and single test run)

Verifier: p5.verify_infra. I did not build any of the reviewed code. Branch `p5-verify-infra`, created from master aafcbd1. Master later moved to 3e19da1, but `git diff aafcbd1 3e19da1 -- src tests scripts configs env` is empty, so every finding applies to the current master.

Read before checking: DESIGN v4, DECISIONS P3-1..P3-23, the RESEARCH_TREE decision log D0–D9, INTERFACES, reviews/p2_review_sim.md, reviews/p2_review_wip.md, reports/p2.verify_artifacts.md and reports/p4.realdata_rho.md.

No test-split locus or test real-data pair was generated, loaded or viewed. The scan in check (e) reads only metadata columns. The test-ID lists were counted, never opened for statistics.

All verification code is in `reviews/p5_verify_infra/`. Its JSON outputs are in `reviews/p5_verify_infra/out/`.

## Verdict

**Trust with caveats.** The caveats are serious for the single test run.

**What I verified and found correct:**
- oracle labels;
- metrics, pooling, power matching and C3 thresholds;
- `verify_run.py`;
- failure scoring;
- every §4.3 adapter formula;
- coloc.abf against R;
- the absence of any test-split locus under `data/processed`.

**Validation benchmark (M4).** The infrastructure can be trusted once two things hold:
- the mvSuSiE hard-rule fix is merged (defect 3);
- slow methods are benchmarked on their own id lists (defect 6).

**Single test run: do not run it as it stands.** Defects 1, 2 and 4 must be fixed first, and the D9 dry run must exercise them:
- inside an orch job, provenance and the test lock read the main checkout's HEAD, not the code that runs;
- `eval_test.py` has no slow-method subsample, no real-data arms and no paired evaluation;
- once the run completes, the test truth cannot be read again, so C1/C3 and verify_run cannot be computed afterwards;
- the access cache is broken on the project filesystem.

## 1. Full test suite at master code (job `p5vi_tests_full`, commit f85dc20 = aafcbd1 + my runner only)

**Command:** `bash reviews/p5_verify_infra/run_tests.sh`.
- It runs one `pytest tests/<module>` per module, on the GPU node (10 GB declared, 4.4 GB peak).
- TMPDIR and basetemp are redirected into `$ORCH_OUT`, which is on /cluster/scratch.
- Total time: 18.3 min.

| module | tests | passed | failed | skipped/xfail |
|---|---|---|---|---|
| baselines | 102 | 101 | 0 | 1 xfail (documented SharePro distinct case) |
| data | 69 | 69 | 0 | 0 |
| data_real | 28 | 28 | 0 | 0 |
| eval | 73 | 70 | **2** | 1 skip (`baselines.adapters` shim absent; intended) |
| famA | 28 | 28 | 0 | 0 |
| famB | 41 | 41 | 0 | 0 |
| famC | 15 | 15 | 0 | 0 |
| famG | 12 | 12 | 0 | 0 |
| orch | 10 | 10 | 0 | 0 |
| sim | 59 | 59 | 0 | 0 |
| stats | 20 | 20 | 0 | 0 |
| **total** | **457** | **453** | **2** | **2** |

Baseline tests now pass from a plain orch snapshot. The P3-22 env linking works, and the copy workaround from earlier reviews is no longer needed.

**The two failures:**
1. `test_truth_isolation::test_method_code_never_touches_truth` flags `src/famC/c1.py:167` (`'.truth.parquet'`). This is the known failure; P3-21 orders the fix. My own grep (check f) confirms it is the only truth-file access in method code.
2. `test_eval_io::test_access_grant_requires_frozen_tag_and_open_nonce` fails. **This is a real defect, not the known one** (defect 4).
   - It fails only when the test's tmp directory is on /cluster/scratch, which has 1-second mtime resolution. Measured: `st_mtime_ns % 1e9 == 0`, and two writes in the same second give the same mtime.
   - Earlier reviewers ran it under /tmp (xfs, nanosecond mtime), where it passes.
   - Reproduced in isolation: `--basetemp` under /tmp/claude-… gives 1 passed; under the worktree on /cluster/scratch, 1 failed.
   - `results/test_access.log` itself lives on /cluster/scratch.

So the only pre-existing failure is the c1.py one. The second failure is real and environment-triggered.

## 2. Conformance spot checks

**(a) Oracle classes, 30 validation loci** (`oracle_check.py`, job `p5vi_oracle_check`; 15 from screen_v2 and 15 from valid_grid_v1, stratified S8/D8/W8/G3/U3).

Method:
- I materialised the NFE mosaic target population (N = 20,000) for each window on the CPU.
- I computed r² in R_true in float64 with my own code.
- I reclassified each locus from the stored causal ids, λ and μ, with an oracle written from the DESIGN §1 text. The thresholds were typed in, not imported.

Results:
- **30/30 classes agree.**
- Class-determining |μ1|, |μ2| differ by exactly 0.
- r² of the D/U pair: max |Δ| = 3.1e-7. d: max relative Δ = 1.7e-6. Both are float32 rounding.
- Extra check: for the 29 factor-form traits drawn from the target population without background, I recomputed μ = (R_pop λ) at the causal variants. Max relative error is 2.3e-5, which confirms the "realised μ" definition of §1.

**(b) Metrics on a toy prediction file** (`metrics_check.py`, interactive, 10 s). The toy has 714 loci in 7 cells (L0/L1/L2 × P1/P2/P3), with ties and weights. Each item below was computed by hand by brute force and compared with `src/eval/metrics.py`:

| quantity | max discrepancy |
|---|---|
| step-wise AP (ties = one threshold), weighted | 2e-16; sklearn also agrees |
| weighted AUROC (pairwise, ties ½) | 1e-16; sklearn also agrees |
| realistic and ideal cell-pooling weights (§8.6) | 0 |
| power-matching × pooling weights | 1.7e-18 |
| AUPRC S-vs-D realistic | 2e-16 |
| AUROC S-vs-D realistic | 1e-16 |
| AUROC D-vs-W realistic | 1e-15 |
| C3 t_S, and t_D with S precedence | identical |
| hand case: 40 W loci, 1 allowed → t = nextafter(second-largest score) | identical |

Power-matching invariants hold in every cell:
- S share is 0.5;
- cell mass equals the pooling weight;
- non-S/D loci get weight 0.

The achieved W rates at the C3 thresholds are S 2.08% and D 2.50%, both ≤ 2.5%.

End to end through `run_method` (FlakyMethod on an eval_toy set):
- metrics.json `auprc_sd` matches my recomputation to within 1e-16;
- `verify_run.verify` returns no problems on the untouched run;
- it detects a tampered metric (with the manifest hash fixed);
- it detects a dropped prediction row (id-hash mismatch).

**(c) Adapters** (`adapters_check.py`). I recomputed s, P_S and P_D from `extra` for **all 200 rows** of every screen_v2 pilot200 run with my own §4.3 code. Five random rows per method are listed in the JSON.

- Agreement with the stored columns:
  - coloc_abf, coloc_susie (56 fallbacks), coloc_susie_mm (59 fallbacks), prop_coloc, coloc_prop_test, lead_snp: 0 difference;
  - colocboost and colocboost_m1: ≤ 1.1e-16, and P_sig is consistent with cos_outcomes and ucos_outcomes;
  - sharepro: 2.1e-6, from the 6-dp rounding of the stored P_eff. P_sig uses the per-effect probability, as P3-3 requires.
- mvSuSiE:
  - The inclusion–exclusion matches a brute-force enumeration of effect states to within 1.1e-16.
  - `extra.hard_lfsr005` matches my hard-rule recomputation exactly.
  - **The primary s/P_S/P_D columns are the soft rule.** They differ from the hard rule (P3-2 primary) on 28 of 200 loci, by up to 0.89. This is defect 3.

**(d) coloc.abf in R coloc 5.2.3 against the wrapper** (`coloc_r_check.py` + `.R`).
- Five random pilot loci (p = 1,624–2,975), with beta, varbeta, N, MAF, type quant and sdY = 1, at p12 ∈ {5e-6, 1e-5}.
- Max |ΔPP| over all PP0..PP4 is **3.3e-15**.
- The stored P_S and P_D equal PP4 and PP3 of the W = 0.15, p12 = 5e-6 row.

**(e) Test lock.**
- `python scripts/eval_test.py --check` in my worktree and in the main checkout: **REFUSED** with "HEAD carries no frozen-* tag", plus the template methods file and the configs not yet re-frozen. `has_test_access()` is False.
- `scan_test_split.py` (job `p5vi_scan_testsplit`) is independent of `eval.io`. It reads the test blocks from `splits/blocks_v1.tsv` (267 test blocks). It flags any locus or row whose split is test, whose block id is a test block, or whose window overlaps a test-block interval. Positive and negative controls pass.
- What it scanned:
  - every `*.h5` under data/processed: 67 files, 96,576 locus groups, attributes only;
  - 76 tables with 185,705 rows: every truth parquet and every TSV with split, block or window columns;
  - the famC shard configs.
- Result: **0 flagged** outside the six sanctioned test-ID lists (R2/R3b/R4b `*_test_ids*.tsv`, which hold identifiers and windows only).
- The real candidate TSVs contain only train and valid rows.
- famC shard configs are train or valid.

**(f) Truth isolation.** I searched `src/fam*` and `src/baselines` for `truth`, `read_parquet`, `truth_path_for`, `oracle` and imports from `sim`/`eval`.
- The only truth-file access is `famC/c1.py:167-170` (P3-21).
- famC training labels come from `sim.generator.draw_candidate(...).oracle` inside a `TrainSampler` subclass on train and valid windows. This is the sanctioned simulator path, not a truth file.
- famB's simulator-prior mode imports `sim.l0star` and `sim.params`, as DESIGN §5 B requires.
- The static scanner misses `eval.io.truth_path_for` and the runtime guard does not cover `pd.read_parquet`; see defect 5.

**(g) Failure scoring.**
- Outer join against the expected list: 15 missing rows + 3 `failed` + 3 NaN-s rows give 21 failures, with frame length equal to the expected list.
- Least-favourable values match my substitution (S → s = 0; D → s = 1, P_D = 0; W → P_S = P_D = 1). AUPRC and AUROC(D-vs-W) agree to within 1e-15.
- Excluded mode drops exactly those 21 loci.
- `compute_run_metrics` counts the missing rows (`status_counts.missing = 15`).
- The harness turns a missing method output into a `failed` row (60 of 60 rows).

## 3. Silent failure modes

- **NaNs in predictions.**
  - 40 prediction files and 7,636 rows under results/.
  - ok/nonconverged rows: 0 NaN s, 0 NaN P_S/P_D where the method defines them, 0 inf, 0 values outside [0, 1].
- **Seeds:** see defect 8 (all minor).
- **Writes outside the project:** see defect 7.
- **Data in git:**
  - No raw data and no file ≥ 5 MB in master's history; the largest blob is `results/famC/c0_full/model.pt` at 4.2 MB.
  - The 3 tracked prediction parquets are below 5 MB, per D6(h).
- **Artifacts whose producing commit is not in master:** see defect 9.

## Defects

**1. MAJOR: inside orch jobs, provenance and the test lock use the main checkout's git state, not the executed snapshot.**
- *Evidence:* `PROJECT_ROOT` and `eval_test.ROOT` resolve to `runs/<id>/code`, which is a `git archive` without `.git`. Git therefore walks up to the main checkout. Job `p5vi_gitinfo_probe` (ORCH_COMMIT = 11fc77f) recorded `harness.git_info()` = `{"commit": "3e19da1…", "clean": false, "modified_tracked": [" M logs/redo_chr7.log"]}`.
- *Consequences:*
  - **(a)** Every harness manifest written from an orch job records the wrong commit and `clean: false`. This affects M4 and the test run. `verify_run` does not check the commit.
  - **(b)** `eval_test.py` run as a job (> 1 min, so it must be one) checks the frozen tag, the committed access log and cleanliness against main HEAD. It would currently REFUSE because main is dirty (`logs/redo_chr7.log`).
  - **(c)** If main happens to be clean and tagged, the check passes while the job executes the snapshot of whatever branch was submitted. `_check_access` and the access log then record main HEAD.
- *Fix:*
  - In `git_info`, `eval_test.check_preconditions`, `io._check_access` and `sim.testset` access, use `$ORCH_COMMIT` when set. The snapshot is clean by construction.
  - Require `git tag --points-at $ORCH_COMMIT` to contain `frozen-*`, and read the committed log with `git show $ORCH_COMMIT:results/test_access.log`.
  - Commit or restore `logs/redo_chr7.log` in the main checkout.
  - Add a test that runs `git_info()` from inside a snapshot.

**2. MAJOR (test run): `scripts/eval_test.py` covers only generation plus per-method harness runs.**
- *Evidence* (code read, `scripts/eval_test.py:136-170`):
  - **(i)** `run_method` is called without `ids`. The `subsample` key of `configs/test_methods_v1.yaml` and `testset_v1.yaml: slow_method_subsample` are never read, so mvSuSiE and ColocBoost would run on all main and shifted loci. The WIP review projected 7.2 h for all methods with them subsampled.
  - **(ii)** The real-data test arms of P3-14 (R2 800, R3b 1,237, R4b 600 × 2 LD conditions) are not produced or run at all.
  - **(iii)** No strength-only control is run.
  - **(iv)** There is no joint multi-method evaluation. The C1 paired bootstrap, C3 test and pooled main+shifted results exist only in `run_benchmark.evaluate`.
  - **(v)** After `completed` is written, the nonce is closed. `read_truth` then calls `guard_locus`, which refuses test loci, so `run_benchmark` and `verify_run` cannot read the test truth afterwards. The pre-registered C1/C3 analyses and the §7.6 independent re-check would be impossible without opening a second access.
  - Also, `check_methods_file` does not enforce `metric_config` (t_S, t_D), although the template says it refuses entries without validation thresholds.
- *Fix:*
  - Make `eval_test.py` run the whole pre-registered pipeline inside the open nonce:
    - main + shifted generation;
    - per-method id lists from the subsample spec;
    - the real-data arms;
    - `run_benchmark` with `--operating-points` from M4 and `--strength-table`;
    - `verify_run` and `--verify`.
  - Alternatively, add a sanctioned read-only evaluation access: a `completed` nonce at the same frozen commit allows reading the test predictions and truth, with no generation and no method runs.
  - Enforce thresholds in `check_methods_file`.
  - Make the D9 dry run execute exactly this path on validation blocks.

**3. MAJOR (conformance; fix in flight): the mvSuSiE primary columns use the soft rule.**
- *Evidence:* `src/baselines/methods.py:355` uses `EA.mvsusie(lf, cs, rule="soft")`. Check (c): 28 of 200 pilot loci differ from the P3-2 hard rule, by up to 0.89.
- *Fix:* commit 1375433 on the unmerged branch `p4-baselines` makes `rule="hard"` the default and keeps the soft rule in `extra`. I read that diff but did not test it. Merge it before any M4 mvSuSiE run counts.

**4. MEDIUM: the `has_test_access` positive cache is keyed on the log's `st_mtime_ns`.**
- *Evidence:* the failing test above. /cluster/scratch has 1-second mtime resolution, so after `completed`/`aborted` is appended within the same second the cached grant persists in-process. `io.py:190-200`.
- *Real-world risk:* low, because `eval_test` pops the env variable and the nonce is unique. But the test suite fails on the project filesystem, and DESIGN §7.1 says a result is used only if tests pass at its commit.
- *Fix:* key the cache on `(st_ino, st_size, st_mtime_ns)`, or on a hash of the log, or do not cache at all. Pin the test's tmp directory to the project filesystem so CI sees the real case.

**5. MINOR (known; P3-21): `src/famC/c1.py:167-170` reads `<set>.truth.parquet` for natural-class weighting.**
- *Fix (as ordered):* move this into `scripts/famC_eval/` or `src/eval`.
- *Hardening:* add `truth_path_for` and `\.stats\.json` to the scanner regex. The class shares in stats.json are truth too.

**6. MINOR (M4): `run_benchmark.py` uses one expected-id list for all methods.**
- *Evidence:* `_load_truth_and_preds` and `run_stage`. Joining a slow method run on a subsample with full-set methods would score the other loci as failures in least-favourable mode, and would shrink excluded mode to the subsample for every method.
- `--per-cell` draws uniformly per cell with key `("run_benchmark", "subsample", …)`. It is not the class-stratified P3-13 draw.
- *Fix (in flight):* `scripts/baselines_p4.py bench --ids-file` and `src/baselines/subsample.py` on `p4-baselines`, with artifact `m4_subsample@1`. The PI must run slow-method benchmarks with their own id lists, and compare them on the paired subsample.

**7. MINOR: writes outside the project.**
- *Evidence:*
  - `/tmp/pytest-of-mumarc` (1.2 GB; pytest tmp_path, written by every test job);
  - `/tmp/Rtmp*` (R workers' `tempdir()`);
  - `/tmp/torchinductor_mumarc`.
  - None of these are under `/tmp/claude-*`.
- *Fix:* in `env/activate.sh`, `export TMPDIR=/tmp/claude-$(id -u)/coloc` (after `mkdir -p`) and `TORCHINDUCTOR_CACHE_DIR=$P/env/cache/inductor`. Delete the existing directories.

**8. MINOR: seeds not derived through `common.seeds.derive`.**
- *Evidence:*
  - `famC/c1.py:95` uses `torch.manual_seed(1)`;
  - `famC/compare_dev.py:47` uses `default_rng(20261029)`;
  - `famC/floor.py:146` uses `default_rng(20261030)`;
  - `famC/model.py` initialises from the global torch RNG;
  - `famB/method.py:52,88` uses one `torch.Generator` seeded with the raw value 20261029 for the whole method instance. The Monte Carlo event posteriors (P_S, P_D, E) of a locus therefore depend on batch composition and order, so they are not reproducible per locus across batch sizes or worker splits.
- *Fix:* seed per locus with `derive(20261029, "famB", locus_id)`; derive the others too.

**9. MINOR: artifact hygiene.**
- `ref_ld_var@v2`: producing commit 9daa45a is not in master (pending merge).
- Unregistered entries over 1 GB under data/processed, which reconcile flags:
  - `famC/train_dev` (26 GB);
  - `famC/valid_dev` (2.0 GB);
  - `sets/shifted_dry_v1_shifted` (5.1 GB).
- `real_locus_sets@1` and `@2` share one path. The data was mutated in place; AGENT_RULES now forbids that.
- *Fix:* merge, register or delete, and record that v1 no longer exists on disk.

**10. MINOR (config, before freeze): the subsample spec contradicts P3-17.** `configs/testset_v1.yaml: slow_method_subsample` gives mvSuSiE 40/40/40 per cell, but P3-17 sets mvSuSiE at 20/20/20 in the test run. *Fix:* split the spec per method before `check_frozen --update`.

**11. NOTE: pool-swap world loci exist outside the test arm.**
- `shifted_dry_v1_shifted` holds 400 pool-swap-world loci, on validation blocks.
- DESIGN §2.1 keeps this world test-only "so that no method was developed on it". No code consumer was found.
- *Recommendation:* register it as plumbing-only or delete it, and never use it for tuning.

**12. NOTE: C3 thresholds are fitted on the least-favourable frame.**
- `run_benchmark` writes `operating_points.json` from the least-favourable frame, where failed W loci carry P_S = P_D = 1.
- A method whose W failure share exceeds 2.5% in a setting therefore gets t = nextafter(1), i.e. no calls.
- This is conservative and within the rule, but the PI should know it when freezing thresholds.

**13. NOTE: the "PI-created" tag is not verified.** `eval_test` accepts any `frozen-*` tag, so this is a process control only. *Optional:* use an annotated tag and check its tagger.

## Commands and jobs

| Job | Status | Content |
|---|---|---|
| `p5vi_tests_full` | failed (exit 1 = the 2 failing tests; acked) | full suite |
| `p5vi_scan_testsplit` | done, 5.5 min | check (e) |
| `p5vi_oracle_check` | done, 1.2 min | check (a) |
| `p5vi_gitinfo_probe` | done | defect 1 evidence |

Interactive, each under 15 s:
- `PYTHONPATH=src python reviews/p5_verify_infra/metrics_check.py <scratch>`: 0 failed checks;
- `adapters_check.py <root>`: 0 formula mismatches;
- `coloc_r_check.py <root> <scratch>`: max |ΔPP| 3.3e-15;
- `scripts/eval_test.py --check`: REFUSED.
