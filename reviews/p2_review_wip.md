# Review p2.review_wip: branch wip-m35-uncommitted (2cc5e60)

Reviewer: p2.review_wip (did not write the reviewed content). Branch `p2-review-wip`, started from 2cc5e60.

## Verdict

**Merge, with the changes on this branch.**

| Part | Verdict |
|---|---|
| `src/baselines/pilot.py` (+`by_level`) | **Pass.** Reporting only. |
| screen_v2 pilot and controls outputs | **Pass.** Internally consistent and reproducible from the raw predictions. |
| screen_v1 pilot and controls outputs | **Dropped** from the branch (superseded noise; see §2). |
| `configs/testset_v1.yaml` | **Updated** to P3-11 and P3-12 (§3). |
| `configs/test_methods_v1.yaml` | Methods list left empty. Provisional timeouts and the P3-14 real-data spec added (§3). |
| `configs/FROZEN_SHA256` | **Not touched.** The testset_v1.yaml hash in it is now stale on purpose: the PI runs `check_frozen --update` at the freeze. |

Commits on `p2-review-wip`:
- b0d4265: drop the screen_v1 outputs.
- 20d7045: configs and the config test.
- The commit that adds this review.

## 1. pilot.py change

`by_level(rec, attrs)`:
- groups the merged pilot predictions (censoring re-runs included through `load_predictions`) by mismatch level × pairing;
- reports n, the nonconverged, failed and timeout counts, the runtime median, p95 and max over ok and nonconverged loci (the same runtime population as `summarise`), and the median p.

It is called only from `summarise` and writes `<tag>_by_level.tsv`. It changes no run or freezing rule.

Minor point, no action needed: rows whose locus is missing from attrs would be dropped silently by `groupby`. Every pilot id is present (checked: 200 of 200).

Stale constants, not introduced by this change: `SUBSAMPLE_PER_CELL = 400`, `subsample_ids` (seed 20261029, proportional allocation) and the "projected > 12 h" slow rule in pilot.py now predate decisions:
- P3-12 for the test set: 40/40/40 per cell, seed derive(20261028, 'slow_subsample'), slow methods named explicitly;
- P3-13 for M4.

The baseline or eval agent should implement the P3-12 draw from `configs/testset_v1.yaml: slow_method_subsample` and not reuse `subsample_ids`.

## 2. Pilot outputs: consistency checks

I checked every run record independently (scratch script). For each method in `pilot200_screen_v2_runs.json` and `controls_screen_v2_runs.json`:
- `results/baselines_pilot/<run_dir>/{predictions.parquet, manifest.json}` exist.
- The status counts in the parquet file, the manifest and the runs.json agree, and they sum to the expected count: 200 and 108.
- There are no duplicate locus ids, and the id set equals the ids file.
- The method column is correct.
- No runtime is NaN, negative, or above the configured timeout.
- Rows that are ok have a non-NaN s.
- P_S + P_D + P_W = 1 to within 2e-13, where defined.

**Result: no issues** (40 runs, including the screen_v1 ones).

**Runtimes are plausible.** Σ runtime / n_workers ÷ manifest wall time is between 0.2 and 0.95 for every run, and is ≤ 1 everywhere. The fast methods have low ratios because of worker start-up.

**The summaries reproduce.** Re-running `pilot.summarise` and `pilot.by_level` on the raw predictions reproduces the committed screen_v2 TSVs to within the 4-significant-digit rounding of the files (maximum relative difference 4e-4).

**Pilot sample:**
- 200 validation-split loci (200 of 200 `split == valid`);
- 33–34 per cell over all 6 screen_v2 cells (L0, L2, L4 × P1, P3);
- p from 805 to 5,870 (the set's maximum is 6,252).

**Provenance:**
- Every screen_v2 run was made at e285fe2 or 09f1297. These are ancestors of this branch and differ only in `src/famC`.
- Every manifest is marked dirty. The only dirty code was `src/famC/*` and the uncommitted `by_level` edit to `pilot.py`, which is reporting only.
- No baseline code differed from the committed code.
- The runs used `mem_budget_gb = 40`, 9–24 workers, and no timing lock (`timing: false`), on a shared machine. Per-locus runtimes are therefore inflated by contention, which makes them conservative for timeouts.

**screen_v1 (dropped from the branch):**
- The set is superseded (P3-16, D6a).
- Its pilot ran across 5 commits (4fb84ad → fb426c4), before the mvSuSiE all-null fix (9698815). Its 7 mvSuSiE failures are that bug.
- `controls_screen_v1_runs.json` is incomplete (7 of 10 methods) and has no analysis table.
- The raw run directories stay in `results/baselines_pilot/` (untracked) and the LEDGER.

### Pilot summary: screen_v2, 200 validation loci (numbers that freeze timeouts, DESIGN §4.3)

- Runtimes are in seconds per locus over ok and nonconverged loci, including LD loading.
- "Failed" counts `failed` plus `timeout`. There were no timeouts in any run.
- The timeout uses the rule pre-registered in pilot.py: max(60, 3 × the maximum observed runtime), ceiled to 30 s.

| method | failed (%) | nonconv (%) | L0 med / p95 | L2 med / p95 | L4 med / p95 | nonconv L0/L2/L4 (%) | all med / p95 / max | timeout (provisional, s) |
|---|---|---|---|---|---|---|---|---|
| coloc_abf | 0/200 (0.0) | 0/200 (0.0) | 0.13 / 0.2 | 0.20 / 0.3 | 0.15 / 0.3 | 0/0/0 | 0.15 / 0.3 / 0.4 | 60 |
| lead_snp | 0/200 (0.0) | 0/200 (0.0) | 0.07 / 0.3 | 0.20 / 0.4 | 0.14 / 0.2 | 0/0/0 | 0.13 / 0.3 / 0.5 | 60 |
| sharepro | 0/200 (0.0) | 0/200 (0.0) | 0.47 / 0.9 | 0.52 / 5.0 | 3.31 / 14.0 | 0/0/0 | 0.66 / 11.7 / 19.8 | 60 |
| coloc_susie | 1/200 (0.5) | 56/200 (28.0) | 3.49 / 7.5 | 4.79 / 37.2 | 15.46 / 90.2 | 0/21/63 | 5.45 / 56.6 / 117.4 | 360 |
| coloc_susie_mm | 0/200 (0.0) | 33/200 (16.5) | 10.75 / 33.5 | 6.94 / 21.0 | 12.46 / 57.9 | 1/3/45 | 9.01 / 35.9 / 78.0 | 240 |
| colocboost | 0/200 (0.0) | 0/200 (0.0) | 1.13 / 3.2 | 4.56 / 26.6 | 8.88 / 22.8 | 0/0/0 | 3.13 / 22.1 / 33.6 | 120 |
| colocboost_m1 | 0/200 (0.0) | 0/200 (0.0) | 0.51 / 1.6 | 0.29 / 0.6 | 0.25 / 0.6 | 0/0/0 | 0.31 / 1.5 / 1.7 | 60 |
| mvsusie | 0/200 (0.0) | 58/200 (29.0) | 14.98 / 52.2 | 32.17 / 179.4 | 145.65 / 269.5 | 0/15/72 | 35.48 / 229.4 / 356.2 | 1080 |
| prop_coloc | 7/200 (3.5) | 0/200 (0.0) | 1.10 / 2.3 | 0.91 / 1.3 | 0.98 / 1.2 | 0/0/0 | 0.97 / 1.9 / 2.4 | 60 |
| coloc_prop_test | 0/200 (0.0) | 0/200 (0.0) | 1.49 / 4.9 | 0.90 / 2.3 | 0.79 / 2.0 | 0/0/0 | 1.03 / 4.0 / 5.6 | 60 |

The level × pairing breakdown is in `src/baselines/benchmarks/pilot200_screen_v2_by_level.tsv`.

**Points for the PI:**
- **prop_coloc fails on 3.5% of loci**, all in L2 P1 (5/33 = 15%) and L4 P1 (2/34 = 6%). Both are above the 1% per-cell threshold of §4.3, which blocks claims involving that method in those cells. The cause is a package condition (README decision 21: a singular LD sub-matrix gives "object 'solve0' not found" or "Unable to compute the standard errors").
- **Non-convergence grows strongly with mismatch.** At L4 it is 63% for coloc_susie, 45% for coloc_susie_mm and 72% for mvsusie. DESIGN uses these outputs and flags them, so they do not count as failures.
- **The pilot covers only L0, L2 and L4 with P1 and P3.** There is no L1, L3, L6 or P2, and p ≤ 5,870. That is why the timeouts are marked provisional.

**Known-direction controls** (`controls_screen_v2.tsv`; 12 loci per category × level): the results match the behaviour already documented in `src/baselines/README.md`, and show no new wrapper bug.
- ColocBoost (default) misses strong shared signals at L2/L4: s > 0.9 in 2 of 12 loci at L2 and 5 of 12 at L4.
- mvSuSiE calls strong distinct loci shared under mismatch: s < 0.1 in 0 of 12 at L2.
- SharePro distinct at L2/L4: 7 of 12 and 8 of 12.
- prop_coloc gives p < 0.05 on 100% of distinct loci.

**Projected baseline compute of the test run at P3-12 sizes.** This is an upper bound. I take the pilot's mean runtime and its memory-limited worker count at a 110 GB / 60-CPU budget, and assume:
- 34 cells × (300 + 60 G/U) plus 1,600 shifted loci (13,840 in all);
- for the slow methods, 38 cells × 120 loci.

The methods, run one after another on the whole machine, take about **7.2 h wall-clock**. The largest shares are mvsusie (3.0 h), coloc_susie (2.0 h) and coloc_susie_mm (1.8 h). This exceeds the roughly 5 h test window of D7 (01:30–06:30). It is a budget risk for the PI (decision 2 below).

## 3. Config changes (commit 20d7045)

### configs/testset_v1.yaml

**P3-11 open items, closed and moved to `resolved_items`:**
1. `generator.entry` is `sim.testset:generate`, with `K: 200` and `M: 3`.
2. `eqtl_x_eqtl_share` is 0.5, confirmed.
3. The shifted arm has 4 equal cells.
4. `D_generation_strata` has `relaxation: gridplan_deterministic` and `report_realised_counts: true`.

**P3-12 sizes:**
- Per cell: S 100 and W 100. D 100, at 20 per r² bin; `per_bin` is checked to equal D / 5.
- `G_U.max` is 30, with `cap_applies_to: each_of_G_and_U`, as GridPlan applies it.
- Shifted arm: `n_loci: 1600` and cells of 400 each. `sim.testset.shifted_cells` checks that they sum.

**New `slow_method_subsample` section (P3-12):**
- methods: mvsusie, colocboost, colocboost_m1;
- per cell: 40 S, 40 D and 40 W by oracle class (all of a class if fewer);
- seed: derive(20261028, 'slow_subsample'), with a fully specified draw order;
- `identical_across_methods: true`.

Everything else is unchanged from DESIGN v4 §8.6: cells, pooling weights, power matching, strata, endpoints and the shifted-arm architecture. No consumer reads `K`, `M` or `slow_method_subsample` yet (`generate` takes K and M as kwargs with the same defaults). `scripts/eval_test.py` must still implement the subsample.

### tests/eval/test_eval_adapters_config.py

The test encoded the old DESIGN §8.6 sizes: (400, 400, 400, 200) and shifted 6,000. I updated only those two assertions, to the P3-12 sizes, with a comment citing P3-12. I added checks of the subsample section and that `open_items` is gone. No other assertion changed.

### configs/test_methods_v1.yaml

The `methods` list is still empty: it needs the M4 operating points.

Added:
- **`provisional_timeouts_s`**, the table above. It is marked provisional, with its reasons.
- **`real_data_arms`** (P3-14):
  - R2: a seeded subsample of 800 pairs;
  - R3b: all 1,237 pairs;
  - R4b: 600 loci, each with in-sample and reference LD;
  - methods: the primary candidate, coloc_abf, both coloc-SuSiE variants, sharepro and strength_only;
  - the slow methods are excluded.
- The subsample comment now points to P3-12 instead of §4.3's 400 per cell.

P3-14 does not name the R2 and R4b subsample seeds. I proposed derive(20261028, 'real_R2_subsample') and derive(20261028, 'real_R4b_subsample'), following DESIGN §9 (20261028 = test draws), and marked them for PI confirmation.

I did not open the test ID files. The counts come from P3-14 and `reports/real_data_engineer_M3.md`.

## 4. Tests (via the orch queue)

| Job | Command | Result |
|---|---|---|
| p2rw_tests_baselines_r3 (20d7045) | `pytest tests/baselines -q -rs`, run in a copy of the snapshot with `env/{renv,rlib_*,src}` linked | **101 passed, 1 xfailed** (the documented SharePro distinct xfail) |
| p2rw_tests_eval_cfg (20d7045) | `pytest tests/eval tests/sim/test_testset.py -q -rs` (same setup) | **75 passed, 1 failed, 2 skipped** |

- **The config test passes**, including `test_testset_config_matches_design_v4_8_6` and the `sim` config check (34 cells, the shifted sum).
- **The one failure is pre-existing and unrelated:** `test_truth_isolation::test_method_code_never_touches_truth` flags `src/famC/c1.py:167` (`.truth.parquet`). `src/famC` is identical on master.
- **Skips:**
  - `test_eval_v4.py:309`: `baselines.adapters` does not exist. The FROZEN_SHA256 comment describes it as a shim.
  - `test_testset.py:110`: needs CUDA; the job ran with gpu=no.
- **Infrastructure finding:**
  - The first try, p2rw_tests_baselines_r2, failed with 74 failed and 28 passed. The orch code snapshot (`git archive`) has no `env/renv`, `env/rlib_*` or `env/src`. `baselines/rworker.py` and `sharepro.py` resolve these relative to `__file__`, so **no baseline job can run from a plain orch snapshot**.
  - My workaround copies `$ORCH_CODE` to `$ORCH_OUT/proj`, links the env directories there, and runs from that copy. The famAG agent used the same workaround. The snapshot itself is never written.
  - The first job, p2rw_tests_baselines, was cancelled before it started, because a pipe hid pytest's exit code.
