# Eval-infrastructure engineer — M3 report

Commit: `b6bb2fd` ("eval: ..."). Owner dirs: `src/eval/`, `tests/eval/`, `scripts/{verify_run,eval_test,check_frozen}.py`.
Uncommitted, for PI review and commit-freeze: `configs/testset_v1.yaml`, `configs/FROZEN_SHA256`.
All decisions not fixed by DESIGN v3 are listed in `src/eval/README.md` ("Decisions / assumptions", items 1–26).

## What was built

| Item (task) | File | Notes |
|---|---|---|
| 1 LocusData + HDF5 I/O | `src/eval/io.py` | INTERFACES §2 reader/writer (dense and factor LD; `ld()`, `ld_rows()`); optional LD dedupe through HDF5 hard links (`/_ld/<key>`); insertion order kept; test-split guard on every read/write |
| 1 truth reader (separate) | `src/eval/truth.py` | schema `TRUTH_SCHEMA` (required: locus_id, oracle_class, mu1, mu2, r2, d, rho, block_id, cell_id, split); runtime refusal when called from `fam*`/`baselines` modules; static AST/regex scan test over `src/fam*/`, `src/baselines/` (Python and R) |
| 2 metrics | `src/eval/metrics.py` | step-wise AUPRC (ties = one threshold), AUROC, power-matching weights, cell pooling weights, D-vs-W AUROC, matched-W thresholds (t_S, t_D separately, E gate), recall, 3-way confusion, ECE/Brier raw + logit-intercept recalibration, realized vs nominal FDR, false-S by r² bin and slope vs ρ, outer-join failure scoring (least_favourable / excluded), cluster bootstrap (BCa, jackknife over blocks, deterministic parallel), paired bootstrap, block sign test, `compute_run_metrics` |
| §4.3 generic adapters | `src/eval/adapters.py` | coloc.abf, coloc-SuSiE (+fallback), signal-probability (SharePro/ColocBoost), inclusion–exclusion (Family A/mvSuSiE), proportional test, lead SNP |
| 3 harness | `src/eval/harness.py` | `run_method`: spawn worker pool (own process groups; timed-out workers killed with R children), batch mode for GPU (p-sorted batches), per-locus timeouts, crash recovery, shared/exclusive flock, manifest (git hash, clean flag, config, seeds, env-lock hash, start/end, counts, input sha256s, machine state), predictions.parquet, metrics.json, append-only LEDGER.tsv |
| 4 verify | `scripts/verify_run.py` | recomputes metrics with the manifest's metric config; checks predictions/truth/metrics sha256, expected-ids hash, counts, calls, ledger entry |
| 5 test lock | `scripts/eval_test.py`, guard in `io.py` | refuses without `frozen-*` tag on HEAD, with a completed entry in the committed `results/test_access.log`, with uncommitted log entries, dirty tree or failed `check_frozen`; writes started/completed/aborted with a nonce; io grants test access only for an open nonce at a frozen HEAD; guard keyed on `splits/blocks_v1.tsv` (block ids and window coordinates) |
| 6 configs | `configs/testset_v1.yaml`, `configs/FROZEN_SHA256`, `scripts/check_frozen.py` | 42 cells, counts, seed, pooling, endpoints, bootstrap, shifted arm per §8.6; FROZEN lists metrics.py, adapters.py, src/sim/oracle.py, splits/blocks_v1.tsv, splits/genes_v1.tsv.gz, DESIGN §1 and §8 (section hashes), testset_v1.yaml; currently `FROZEN OK` |
| 7 reference Method | `src/eval/reference_methods.py` | `LeadSNPRule`: s = 1 iff r²(lead1, lead2) ≥ 0.8 in the inference LD; end-to-end on a synthetic MVN (AR(1) LD) locus set |

## Tests (all run)

Command: `source env/activate.sh; python -m pytest -q -p no:cacheprovider tests/eval` → **54 passed** in 38 s
(also passes from `tests/eval/`; temp files under `/tmp/claude-<uid>/pytest-eval`). Highlights:
- AUPRC/AUROC equal sklearn (weighted, heavy ties) to 1e-12; hand example with ties AP = 0.5833, AUC = 0.625; perfect = 1;
  random ≈ prevalence (±0.01, n = 2e5); integer weights = replication; AUC contributions sum to AUC.
- Power matching: S/D binned weighted distributions identical; a scorer that is a function of the power bin gets exactly
  AUROC 0.5 matched (> 0.6 unmatched); weighted S prevalence = 0.5 per cell; an informative scorer keeps its expected AUROC.
- Pooling weights for the 42-cell grid reproduce §8.6 (L1 P1 cell 0.05, P2 cell 0.025, P3 cell 0.0167; ideal 0.5/0.25/0.25).
- Matched-W thresholds (hand cases incl. ties, weights, gated −inf), calls/recall/confusion/FDR hand example, E gate.
- Calibration: calibrated ECE < 0.01; injected intercept a = 1 recovered (±0.03) and ECE fixed.
- False-S slope recovers 0.02 per 0.1ρ (±0.002); flat gives 0.
- Failure scoring: missing rows, timeouts, NaN s → failures; least-favourable lowers AUPRC; excluded = clean subset;
  union exclusion across methods; nonconverged used; unexpected/duplicate loci raise.
- Bootstrap: BCa 90% coverage on a skewed clustered mean within [0.83, 0.97] (150 sims); row-level bootstrap
  under-covers (< 0.5) while the cluster bootstrap covers (> 0.78) — negative control for the block unit; self-comparison
  gives [0, 0]; seeded reproducibility; serial = 4-process results exactly; replicate with duplicated blocks equals the
  metric on an explicitly duplicated table (weights recomputed).
- Inclusion–exclusion P_D equals brute-force enumeration (50 random 3-effect cases, 1e-12) and the DESIGN §5 unit cases.
- Harness end to end (30 toy loci, 4 workers): outputs, manifest fields, ledger append-only, `verify_run` OK, and tampered
  metrics.json / predictions.parquet are detected; timeouts kill the worker's grandchild process; exceptions, worker
  crash (`os._exit`), missing rows, invalid status → `failed`; GPU batch mode used CUDA with p-homogeneous batches;
  timing run takes the exclusive lock; test-split locus set refused before anything is written.
- Test lock in a throw-away git repo: no tag → refused; tag → allowed; dirty / untracked code / frozen-hash change → refused;
  full `run_test` with a fake generator writes started+completed, workers read test loci during the run, nobody after;
  uncommitted log → refused; committed completed entry → refused under a new frozen tag; aborted run logged.
- Real splits file: guard flags test blocks by id and coordinates (267 test blocks found in `splits/blocks_v1.tsv`).

Performance: `auprc_sd` with recomputed weights on 42,000 loci = 0.043 s; a full 2,000-replicate paired BCa bootstrap
(267 blocks, incl. jackknife) = 8.0 s with `n_jobs=32`.

## Deviations from DESIGN / task

- Splits file name: task says `splits/blocks_v1.tsv`, DESIGN §3 says `splits/split_v1.tsv`; io reads blocks_v1 first,
  falls back to split_v1. The data agent wrote `blocks_v1.tsv` and `genes_v1.tsv.gz` (FROZEN lists those).
- §7.7 names `results/.timing.lock`, §7.5 `flock -s results/.machine.lock`: timing runs take both exclusively.
- §7.7 "repeat 3×" is left to the caller (one harness call = one repeat; label the repeats).
- Bootstrap CIs are not in metrics.json (cost, and they are comparisons between runs); use `paired_bootstrap` /
  `cluster_bootstrap` in the M4/M7 analysis scripts.

## Open issues / what the PI must decide

1. **D strata ("d/r² bins") are not defined in DESIGN.** Proposed r² edges [0, .2, .5, .8, 1] for both test-set
   stratification and power matching; d not crossed by default (12 strata leave ~33 D per stratum). Decide and freeze.
2. **Residual power confounding of 5×5 matching:** with S μ shifted by +4, a power-only scorer has AUROC 0.93 raw and
   0.56 matched. Acceptable for DESIGN's purpose, but a method exploiting power can gain a few points; consider 10×10 bins
   or reporting the power-only scorer as a control in every benchmark.
3. Confirm the metric choices in README items 5–19 (failure definition incl. NaN P_S/P_D, D-vs-W weighting, matched-W
   weights, calibration events, signed-ρ slope, sign test on AUROC contributions).
4. `configs/testset_v1.yaml` PROPOSAL lines: P3 sign and sub-type mix, cell-id format (the simulation agent already uses
   `{level}|{pairing}|{param}={value}`), generator entry point `sim.testset:generate`, shifted-arm level/pairing mix,
   ideal-setting pairing weights. Then commit `configs/testset_v1.yaml` and `configs/FROZEN_SHA256` (re-run
   `python scripts/check_frozen.py --update` right before tagging; data files may still change).
5. **Duplicate adapter code:** `src/baselines/adapters.py` re-implements the §4.3 formulas that `src/eval/adapters.py`
   holds. One copy should be the frozen one (suggest: baselines import `eval.adapters`), and FROZEN_SHA256 must list it.
6. `configs/test_methods_v1.yaml` (methods, timeouts, workers, thresholds from validation) must be written and frozen
   before the test run; `eval_test.py` requires it.
7. Results size: predictions.parquet of large runs may exceed the 5 MB commit limit; decide whether results/<run_id>/ of
   big runs are committed (e.g. via summary tables only) or stored with sha256 in the manifest.
8. DESIGN §7.1 mentions `scripts/run_tests.sh` (not owned by me); tests/eval runs standalone.
9. Parallel bootstrap uses `fork`; Python warns when the parent is multi-threaded. Safe with BLAS threads = 1
   (env/activate.sh), but analysis scripts should not call it from a process with live CUDA contexts.
