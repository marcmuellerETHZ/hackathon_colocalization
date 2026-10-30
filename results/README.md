# Results

Everything here is derived output; no raw data. The prediction folders were built from the original project by
`scripts/release_build/build_predictions.py`. Every file is < 50 MB; no Git LFS is used.

| Path | Content |
|---|---|
| `test_v1/` | the single test run: per-locus predictions of every method and arm, run manifests, the simulated benchmark (`benchmarks/sim/`), claims, real-arm results |
| `truth/` | simulation truth tables (metric columns only) for the test set and the M4 validation subsample |
| `phase2_validation/` | phase-2 predictions and summaries on the validation split (no second test run happened) |
| `models/famC/` | NeuralColoc checkpoints (network, no-LD network; 4 MB each) and the LightGBM floor model (13 MB) behind `phase2_validation/famC_C2` |
| `recompute/` | output of `scripts/recompute_benchmark.py` (or `mvp/recompute_benchmark.py`): every simulated test-set number recomputed from `test_v1/` and `truth/`, 0 mismatches |
| all other folders | the rest of the original project's `results/` (screening, validation, kill tests, method families, dry runs, verifications), under their original names; see [Other results](#other-results-the-full-results-tree) |
| `EXCLUDED.tsv`, `REAL_ARM_STRIPPING_ALL.tsv` | what was not copied from the original `results/` and why; what was removed from the real-arm files outside `test_v1/` |

To recompute: `python scripts/recompute_benchmark.py` (needs numpy, scipy, pandas, pyarrow; about 1 min).

## `test_v1/`: the single test run
Test run nonce `ab94c5a1`, frozen code commit `cab2599` (tag `frozen-v1`), 2026-10-27 22:40 to 10-28 03:43 UTC.
This is a copy of the run's bundle, `results/eval_test/test_frozen-v1_ab94c5a1/`, in the original project.

| Path | Content |
|---|---|
| `runs/<run>/predictions.parquet` | one row per locus: `locus_id, method, s, P_S, P_D, P_W, E1, E2, call, status, runtime_s, extra` (INTERFACES §3; `extra` is method-specific JSON) |
| `runs/<run>/manifest.json`, `metrics.json` | the run's provenance (git commit, inputs, sha256 of predictions, segments) and its per-run metrics |
| `runs/LEDGER.tsv` | ledger of the 44 runs |
| `benchmarks/sim/` | the simulated benchmark: `benchmark.json` (all metrics, 2,000-replicate BCa block-bootstrap CIs), `summary.md`, `c4.json` (overlap slopes), `manifest.json`, `operating_points.json`, `expected_ids.txt`, `ids_subsample40.txt` |
| `benchmarks/sim/subsample_8685f328/` | the same on the ColocBoost subsample (40/40/40 per cell, 4,467 loci) |
| `claims.json`, `descriptive.json`, `real_arms.json`, `runtime.json`, `operating_points_used.json`, `bundle.json` | claims C1–C5, three-way calls of the descriptive entries, real-data arm results, runtimes, the frozen operating points used, bundle index |
| `test_access.log` | the test-lock log: `started`, `interrupted` (mvSuSiE dropped, decision D21), `resumed`, `completed`, all under one nonce |
| `REAL_ARM_STRIPPING.tsv` | what was removed from each real-arm prediction file (below), with original and released sha256 |

Run names are `test_frozen-v1_ab94c5a1__<method>__<arm>`. The arms are `sim_main` (11,607 loci, the 34-cell grid),
`sim_shifted` (1,600 loci, 4 shifted cells), and the real-data arms `real_R2` (800 eQTL Catalogue gene × tissue
pairs), `real_R3b` (1,237 FinnGen × eQTL pseudo-negative pairs), `real_R4b_insample` and `real_R4b_reference`
(600 GEUVADIS genes with in-sample vs reference LD).

Not copied: the harness checkpoints (`checkpoints/*.rows.jsonl`, 152 MB, row-by-row duplicates of the
predictions), job logs and pipeline state files. mvSuSiE was not run on the test set (D21); its partial
checkpoint is not included.

Absolute paths inside manifests (`/cluster/scratch/...`) are the original locations. They are kept unchanged
because the manifests are sha256-checked provenance records.

### Real-data arms: licence check and what was removed
The real arms derive from FinnGen R12 (no redistribution of FinnGen data; derived results may be shared) and the
eQTL Catalogue (CC BY 4.0). Every column of all 20 real-arm prediction files was inspected, including every key
of the JSON `extra` column.
- **Top-level columns:** identifiers (`locus_id` = arm, dataset/trait ids, Ensembl gene id), the method name and
  method outputs (`s`, `P_S`, `P_D`, `P_W`, `E1`, `E2`, `call`, `status`, `runtime_s`). None is a summary
  statistic. All are kept.
- **Removed from `extra`:**
  - `strength_only`: `lead_abs_z1`, `lead_abs_z2`, the lead |z| of each trait (raw summary statistics), in 3,237
    rows;
  - `coloc_susie` and `coloc_susie_mm`: `t1.purity_min`, `t2.purity_min`, the minimum absolute LD correlation
    within each credible set (LD values), in 6,472 rows.
- **Kept in `extra`:** method outputs and diagnostics. These are posterior probabilities per credible-set pair
  (`pairs.PP`), the indices of the top variant of each credible set within the locus (`pairs.hit1/hit2`),
  credible-set counts and sizes, maximum PIP, convergence flags, susieR 0.16.6 `R_finite_diagnostics` scalars,
  SharePro effect-group variant indices and sharing probabilities, coloc.abf posteriors over the prior grid, R
  warnings and runtimes.
- No beta, se, z, allele frequency or LD matrix values remain.
- Because of this stripping, the 20 real-arm `predictions.parquet` files no longer match `predictions_sha256` in
  their `manifest.json`. `REAL_ARM_STRIPPING.tsv` lists both hashes. The real-arm truth labels are not shipped:
  they are built from FinnGen and eQTL Catalogue credible sets. So `real_arms.json` is shipped as computed, but
  cannot be recomputed from this repository.

## `truth/`
Simulation truth (derived from open 1000 Genomes data), reduced to the columns the metrics read: `locus_id,
oracle_class, mu1, mu2, r2, d, rho, block_id, cell_id, split, mismatch_level, pairing_type, overlap, control`.
Removed columns: causal-variant indices and ids, effects, generator bookkeeping, ρ̂ and lead |z|.
- `testset_v1.truth.parquet` (11,607 loci) and `testset_v1_shifted.truth.parquet` (1,600): the test set;
- `valid_grid_v1_M4main.truth.parquet` (12,170 loci): the M4 validation subsample used in phase 2.

## `phase2_validation/`: phase 2 (validation split only; no second test run happened)
| Path | Content | Artifact in the original project |
|---|---|---|
| `coloc_susie_mm_M4/<run>/` | mismatch-aware coloc-SuSiE on the 12,170-locus M4 validation subsample, two disjoint half-runs (predictions + manifest) | runs behind `m4_valid_baselines@3` |
| `m4_valid_baselines_v3/` | the M4 benchmark summary of all baselines and the phase-2 operating points (`operating_points_p2.json`: OP1/OP2 per method; coloc_susie_mm t_S 0.8402818292911564, t_D 0.7322585039086077) | `m4_valid_baselines@3` |
| `famB_mm_M4/` | B-mm (mismatch-aware ExactColoc), M4; `m4_Bmm_v2.json` = its M4 evaluation with block-bootstrap contrasts against coloc_susie_mm and phase-1 B (read by `manuscript/figures/make_figures.py`) | `famB_mm_m4_valid@1`; `results/famB/p9_m4_eval/m4_Bmm_v2.json` |
| `famB_M4/` | phase-1 ExactColoc (B) on M4 | `famB_m4_valid@1` |
| `famC_C2/` | NeuralColoc rung C2: `net_m4` (network), `nold_m4` (no-LD network), `floor_m4` (LightGBM floor), `net_trainregion` (training-region check), `c2.json` (verdict: FAIL, does not beat its no-LD version) | `famC_C2@1` |
| `m4_subsample_v1_main_ids.txt` | the 12,170 M4 locus ids | `m4_subsample@1` |

The famC network checkpoints that produced `famC_C2` are in `models/famC/` (this folder).

## Other results: the full results tree
Everything else from the original project's `results/` (validation and screening splits only, apart from the
test run's logs), copied by `scripts/release_build/build_results_all.py` with the same folder names. Reports are
in `../docs/reports/`; independent checks are in `../reviews/`. Run folders hold `predictions.parquet` (INTERFACES
§3), `manifest.json` and per-run logs or metrics. All of it is simulated data from 1000 Genomes, except the
real-arm files in `eval_test_dry/`.

| Path | Content |
|---|---|
| `<timestamp>_<method>_<set>_<hash>/` (178 folders), `LEDGER.tsv` | harness runs at the top level of `results/`, indexed by `LEDGER.tsv` (its `label` column says which study each run belongs to): ExactColoc (B) pilot, screen_v2 and M4 runs (`p5.killtest_B1.md`, `p9.famB_mm.md`); B-mm (`exactcoloc_mm`) runs (`p9.famB_mm.md`); NeuralColoc snapshot and `strength_only` runs of the provisional M4 benchmark (`p4.famC_C1.md`); the famC C2 runs (`p9.famC_C2.md`; their predictions are in `phase2_validation/famC_C2/`) |
| `baseline_checks/` | install checks of each baseline package: version and example output (`../docs/environment/BASELINES_INSTALL.md`) |
| `baselines_pilot/` | first baseline runs on screen_v1 (`baseline_M3.md`, `../reviews/p2_review_wip.md`) |
| `eval_dryrun/` | dry run of the evaluation harness on screen_v1 (`eval_M3.5.md`) |
| `baselines_screen_v2/` | all baselines on the screen_v2 split: runs, collected predictions, locus ids, benchmark (`p4.baselines_screen.md`) |
| `baselines_m4/` | all baselines on the M4 validation subsample: runs, ids, benchmarks `m4_valid_v1`–`v2`, `bench_p9` (`p4.m4_valid.md`, `p9.baselines_m4mm.md`; the final `m4_valid_v3` is in `phase2_validation/`) |
| `baselines_p9/` | phase-2 operating points of the baselines on screen_v2 (`p9.baselines_m4mm.md`) |
| `benchmarks/` | benchmarks of NeuralColoc: the provisional M4 benchmark and the C2 M4 benchmark (`p4.famC_C1.md`, `p9.famC_C2.md`) |
| `famA/` | JointSuSiE (A): screen_v2 and M4 runs, kill test A1 and its inputs, the P3 fix smoke test, reference-LD variance (`p3.famA_fixes.md`, `p5.killtest_A1.md`) |
| `famB/` | ExactColoc (B) and B-mm: L0* reference, screen_v2, kill test B1, phase-2 pilot, ladder, screen and M4 runs, M4 evaluation (`p4.famB_l0star_ref.md`, `p5.killtest_B1.md`, `p9.famB_mm.md`) |
| `famC/` | NeuralColoc (C): rungs C0, C1 and C2 (the C2 models are also in `models/famC/`), dev and snapshot models with their validation probabilities, comparisons, timing pilot (`famC_M3.5.md`, `p4.famC_C1.md`, `p9.famC_C2.md`) |
| `famG_mm/` | mismatch-aware GPU coloc-SuSiE (G-mm): identity checks against susieR 0.16.6 (`identity_v1`, `identity_v2` with per-locus fits, perturbation and precision runs), timing and benchmark rounds (`p9.famG_mm.md`) |
| `eval_test/` | the test run's per-run logs and `pipeline.log` only; the rest of the bundle is `test_v1/` (`p6.dryrun_testpipeline.md`) |
| `eval_test_dry/` | dry runs of the test pipeline on validation loci (`dry_v4`, `dry_mvr1`, `dry_rb1`, `dry_p9_v2`), including the real-data arms; real-arm files stripped as below (`p6.dryrun_testpipeline.md`, `p9.testset_v2.md`) |
| `p2_verify_artifacts/` | regeneration check of the registered locus sets (`p2.verify_artifacts.md`) |
| `review_famA_v1/` | the independent review's checks of JointSuSiE (`../reviews/p2_review_famA.md`) |
| `verify_B1/` | independent recomputation of kill test B1 (`../reviews/p5_verify_B1.md`) |
| `p5_verify_famC/` | independent rescoring of NeuralColoc C1 (`../reviews/p5_verify_famC_scoring.md`) |

### What was not copied (`EXCLUDED.tsv`)
- Files already in this release (same sha256), mostly test-run and phase-2 files; each row names the released copy.
- `eval_test/test_frozen-v1_ab94c5a1/checkpoints/` (159 MB; row-by-row duplicates of the predictions) and `state/`.
- `baselines_screen_v2/collected_v1/predictions.parquet` (68 MB, over the 50 MB limit). It concatenates the
  per-run predictions in `baselines_screen_v2/runs/`, which are released.
- Harness bookkeeping: `.gitignore` files of run folders and empty `.lock` files.

### Real-data arms outside `test_v1/` (`REAL_ARM_STRIPPING_ALL.tsv`)
Only the dry runs in `eval_test_dry/` hold real-arm outputs (R2, R3b, R4b on validation pairs). Their 64
`predictions.parquet` and 64 checkpoint `*.rows.jsonl` files were stripped with the `test_v1` rules
(`strip_extra()` of `scripts/release_build/build_predictions.py`): `lead_abs_z1/2` of `strength_only` (456
rows over both file types) and `t1/t2.purity_min` of `coloc_susie` and `coloc_susie_mm` (782 rows). The build
then checks every key of every `extra` JSON against a list of raw-statistic names (z, beta, se, allele
frequencies, purity, LD) and fails if one is left. JointSuSiE's `rho_used` (one estimated correlation scalar per
locus) and its per-effect fit values are model outputs and are kept. The other files in these folders hold
method outputs, identifiers, counts and hashes, with no summary statistics: manifests, logs, segment logs, the
`state/real.json` id lists and `real_arms.json`. No locus set, LD cache or other file with FinnGen or eQTL
Catalogue statistics exists under `results/`. Those live in `data/`, which is not released. The stripped files no
longer match `predictions_sha256` in their manifests; the TSV lists both hashes.

