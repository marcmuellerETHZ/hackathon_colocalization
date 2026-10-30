# Review round 1: AI-agent failure modes (DESIGN.md v1)

**Verdict: major revision, process only.** The science plan is fine from this angle. §12 names the right risks, but most guards are honour-system statements that agents can't be held to. Specifically:
- test access is "logged", but by nobody in particular;
- the `FROZEN` tag is self-applied;
- the PI "spot-checks with a script" that doesn't exist.

Two infrastructure risks could lose the whole project, and the design text has already drifted from the numbers it reports. Below are 9 must-change items, 8 should-change items and 2 could-change items. I aimed for guards that cost one small script or one rule each.

## Evidence (checked 2026-10-26)

| # | Check | Result |
|---|---|---|
| E1 | `git remote -v` | **Empty: no remote.** The project lives in `/cluster/scratch/mumarc` on Euler. ETH docs (docs.hpc.ethz.ch/hardware/storage) say: "Files older than 15 days will be automatically DELETED without prior notice"; "there is no backup". Personal scratch also has a quota of 1M files / 2.5 TB. |
| E2 | Recomputed §3 components from `data/raw/annot/pyrho_EUR_LD_blocks.bed` and GENCODE v39 protein-coding genes (script: scratchpad `comp.py`) | Using **TSS ± 1 Mb**, as DESIGN.md states, gives **89 components, median 25.0 Mb**. Using **gene body ± 1 Mb** gives **73 components, median 30.2 Mb**. So the "73 components (median 29 Mb)" figure came from a procedure other than the one described, and no split script is committed. This is a real example of prose drifting from code. Largest component: 4.9% of genes. Test split: about 14 components. |
| E3 | Installed versions in `env/renv` | coloc 5.2.3 and susieR 0.14.2, matching the design. susieR 0.14.2's `susie_rss` has λ-regularisation (`susie_rss_lambda`) but **no `R_finite` / `R_mismatch`**. The current stephenslab r-universe susieR is 0.16.6 (2026-10-25), which documents `R_finite` and `R_mismatch="eb"` (vignette rss_mismatch). The lit/methods.md label "susieR 2.x" and the `"eb_mix"` option could not be confirmed. |
| E4 | `env/requirements.lock`, `env/renv.lock` (cited in §9) | **Do not exist.** The venv runs bleeding-edge numpy 2.5.4, pandas 3.1.0 and torch 2.14.1. susieR was built under R 4.4.3 but runs on R 4.4.1. HyPrColoc, ColocBoost, SharePro, gpu-coloc and the other baselines are not installed yet. |
| E5 | Record files | `DATA.md` has an empty table although the annotation files and 1000G data are downloaded with checksums. `RESEARCH_TREE.md` still shows M0 `[~]` and M1 `[ ]` although M1 is committed. The records are already lagging. |
| E6 | 1000G integrity | The download script computes sha256 *after* download (resume with `curl -C -`) and never checks the upstream md5s in `20220804_manifest.txt`. I checked chr21 and chr22 by hand: both match. chr6 was still downloading (no `.tbi`). |
| E7 | §1 power thresholds (scipy) | 6.293 / 4.610 (α = 5e-8) and 5.259 / 3.576 (α = 1e-5). **Correct.** |
| E8 | Hardware | 64 cores and an RTX 4090 (24 GB). This one GPU will be shared by every agent: NPE training, the GPU baselines and timing runs. |

## Must-change

**M1. Protect the project from the scratch purge (§9; new item 0 in §12).**
- Add a git remote now: a private GitHub repo, or a bare repo in `$HOME` or `/cluster/project`. Push after every commit; make it a LESSONS rule and a milestone-gate check.
- Copy the committed `results/` summaries there too.
- Data under `data/raw` will be purged about 15 days after download. Record this in DATA.md, and make the download scripts idempotent and verifying (see S6) so that re-fetching is one command.
- Keep the file count of simulation and baseline outputs under control: one sharded parquet per run, never one file per locus. Otherwise the 1M-file quota becomes a failure mode in the middle of a run.

*Why:* E1. There is a single point of total loss and the clock is already running.

**M2. Make the test lock mechanical and limit test evaluation to one candidate (§3 "Usage", §12.2, §8).**
- (a) Only `scripts/eval_test.py` can generate or load test-split loci or test-component real data. The data loaders and the simulator raise an error on `split="test"` unless that script calls them. The loader itself appends to `results/test_access.log` (timestamp, git hash, caller). The agent does not write the log.
- (b) "Frozen" means an annotated git tag `freeze-<name>` created by the PI. `eval_test.py` checks that HEAD equals the tag, that the tree is clean, and that the log has no earlier entry for that tag.
- (c) Replace "touched once per frozen final candidate" with the following. The primary candidate is named on validation *before* the test run. All frozen candidates and all baselines are evaluated in a single test run and all are reported. Our method may not be swapped after the test is seen.
- (d) Real data:
  - the R1 realism calibration uses train and validation components only;
  - quantitative R2–R5 results are reported on test components at M7;
  - R6 is qualitative and exempt.
- (e) The PI runs `eval_test.py` personally, or has a fresh agent run only that command.

*Why:* the test seed (20261028), the generator and the split are all in the repo, so any agent can regenerate the test set while "just debugging". The most common leak is accidental, and a loader guard removes it.

**M3. Isolate method inputs from the simulation truth (§5 notation, §7 new item 3b).**
- Define one input type, `LocusInput` = (z1, z2, R_input, n_ref, N per SNP, MAF, ρ̂, ld_level). The truth (λ, causal set, R_true, ρ, class) lives in a separate file that method code never opens. Methods and baselines run in subprocesses that receive only the input file.
- Add these tests:
  - for levels L1–L5, `hash(R_input) != hash(R_true)`;
  - ρ̂ is computed by the shared estimator and is not the oracle ρ.
- State in §8 that **the headline uses estimated ρ̂ and η**. Oracle ρ is an ablation only.
- The NPE sampler draws loci only from train components, and a test enforces this.

*Why:* the design says "ρ estimated … or supplied". Two silent errors are plausible here: wiring the wrong LD level, and passing the true ρ. Either would make our method look robust to mismatch and overlap for free.

**M4. Run manifest, ledger and recomputable metrics (§7, §9, §12.6).**
- One context manager, `src/.../runlog.py`, used by every run, writes `results/<run_id>/manifest.json` containing:
  - git hash, a dirty flag (official runs refuse a dirty tree), command and config hash;
  - hashes of the environment lock files and the input file;
  - seed, start and end time, host;
  - `nvidia-smi` and load average at start;
  - n_expected, n_ok, n_failed and n_timeout per method.
- Each run stores **per-locus predictions** (locus_id, method, P_S, P_D, status, runtime) as parquet. Metrics are computed **only** by `src/.../metrics.py`.
- An append-only `results/LEDGER.tsv` records every run, failed or superseded ones included: run_id, purpose, agent, git hash, status, and the path to `metrics.json`.
- `scripts/verify_run.py <run_id>` checks:
  - the manifest exists and the hash is an ancestor of HEAD;
  - output checksums match;
  - completion is at least 99%;
  - the tests passed at that hash, via `results/tests/<hash>.json` written by `run_tests.sh`;
  - recomputing the metrics from the per-locus predictions reproduces `metrics.json`.
- Agent summaries must cite run_ids. The PI ignores claims without one.

*Why:* this is the cheapest single guard against fabricated or partial runs ("passed on 500 loci" when 380 ran), against cherry-picking by dropping failed runs, and against numbers mis-copied from prose.

**M5. Distinguish baseline crashes from undefined outputs, check baselines are sound, and include the strongest baseline (§4, §6.3, §12.4).**
- (a) §6.3 currently says "Ties and undefined values are treated as s = 0.5". Restrict this to outputs that are *undefined by design* (e.g. no credible set). A crash, timeout or non-convergence gets its own status. If any method fails on more than 1% of loci in any cell, the comparison is blocked until the failure is fixed or explicitly reported. Otherwise a baseline that crashes at L4 (missing variants) scores 0.5 and silently loses.
- (b) **Baseline acceptance test** (M4 gate) for each wrapper: reproduce the package's own documented example to the published precision (coloc vignette PPs, the susieR N3finemapping CSs, the SharePro and ColocBoost example outputs). The pinned version, or GitHub commit SHA, goes in the wrapper header.
- (c) **Known-direction checks**, run in every benchmark:
  - coloc-SuSiE AUPRC at L2 is below L0 (Zou 2022);
  - coloc.abf false-S at ρ = 0.5 with no trait-2 signal is above ρ = 0, because z2 then has mean ρμ1;
  - every baseline beats the lead-SNP floor at L0.
  
  If a check fails, the wiring is broken. Fix it before any comparison.
- (d) Add **coloc-SuSiE with susieR's own mismatch model** (`R_finite = n_ref`, `R_mismatch = "eb"`) as a baseline, pinned to a released version. Installed 0.14.2 lacks it (E3). Without this, C1 under mismatch is measured against a straw man.

**M6. Commit the split script and protect frozen files (§3, §12.1).**
- Fix the §3 text or the procedure so they agree (E2: TSS ± 1 Mb gives 89 components, gene body ± 1 Mb gives 73), and state which one is used.
- Commit `scripts/make_splits.py`. `tests/test_splits.py` regenerates the split and compares checksums.
- Add `protected.sha256` covering: `splits/split_v1.tsv`, `src/.../metrics.py`, `labels.py`, `adapters.py`, the screening- and test-set manifests, and the tolerance constants in `tests/`. Add `scripts/check_protected.py` (run by `run_tests.sh`): any change requires a commit message citing a decision-note ID in RESEARCH_TREE.md.

*Why:* "metrics frozen" is only enforceable if a diff is detected automatically. Loosening a test tolerance is the typical silent cheat.

**M7. Make the labels exhaustive and pre-specify the test set and pooling (§1, §6.1, §8).**
- The class definitions overlap and have holes:
  - a locus with a powered shared variant plus a grey-zone extra variant is both S and G;
  - a locus with a weak shared variant plus distinct powered variants in both traits is none of S, D or W.
- Fix one precedence order (e.g. U, G, S, D, W, plus an "other" bucket that is reported). Labels come only from `labels.py` applied to the oracle μ, never from the generating locus type. Commit a cross-tab of locus type × class.
- §6.1 "500 loci per cell" must say that cells are filled by rejection sampling on the oracle class. As a rough guide, log-uniform λ puts about 12–18% of single variants in the grey zone.
- §8 does not define the test set. Specify now:
  - cells and number of loci per cell;
  - pooling weights for the "realistic" headline (equal weight per L × ρ cell?);
  - the power-matching quantiles (pooled over S∪D);
  - the AUPRC implementation (weighted average precision);
  - the bootstrap unit for the kill-rule SE (component-level paired cluster bootstrap, B = 1000).

*Why:* every unspecified choice gets made after an agent has seen validation numbers.

**M8. Run timing under an exclusive-machine lock (§7.7).**
- C2 runtime numbers come only from runs that hold `locks/benchmark.lock` while no other GPU or CPU job is running. The manifest records the `nvidia-smi` process list and load average.
- Use 3 repeats with the median. Report both compute-only and end-to-end timing (LD read included) for every method, timed through the same I/O path.

*Why:* one GPU will be shared by NPE training and other agents (E8). Contaminated timings can go either way and can't be reproduced.

**M9. Prevent environment and working-tree drift (§9).**
- Create the lock files now (`uv pip freeze > env/requirements.lock`; `micromamba env export --explicit` plus an R `installed.packages()` version dump for R) and commit them.
- One designated agent changes environments, through a script, and the lock is re-committed afterwards.
- Development agents work in separate git worktrees or branches. Only the PI merges.

*Why:* E4. Today the §9 lock files don't exist. Parallel agents editing one tree make the recorded git hash meaningless.

## Should-change

**S1. Limit overfitting to the screening set (§6.2).**
- Give each family a screening budget, e.g. at most 3 implementation attempts or at most N GPU-hours. Log every attempt in the ledger. After the budget is spent, the kill decision is final.
- A family that passes after more than one attempt must re-pass on a fresh draw of 500 validation loci (derived seed, see S2).
- Before freezing, confirm the M6 gain on a fresh validation draw, used once.
- Report the number of configurations tried per family.
- "Good-faith implementation" is currently an unlimited escape hatch.

**S2. Allow derived seeds and determinism checks (§9).**
- "No other seeds are used" will be broken: bootstrap, on-the-fly NPE sampling, per-locus draws and R parallel streams all need seeds. Allow only derivation via `np.random.SeedSequence(SEEDS[i]).spawn` keyed by purpose, in `src/.../seeds.py`, and use `RNGkind("L'Ecuyer-CMRG")` in R.
- Run evaluation with `torch.use_deterministic_algorithms(True)`.
- Add a test: running the evaluation twice on 1k loci gives identical metrics.
- Use float64 for all identity tests (A and G vs susieR, B vs coloc.abf). Otherwise float32 failures invite agents to loosen the tolerances.

**S3. Generate report numbers by script (§11 M8).**
- `scripts/make_report_tables.py` writes every table in REPORT.md from `metrics.json` files.
- `scripts/check_report_numbers.py` checks that every numeric claim in the prose cites a `run_id:key` and matches it.

**S4. Make the independent review concrete (§7.8, §12.5).**
- The reviewer gets the run_id and the code path, not the producer's summary.
- It reruns a 5% locus subset from the run's commit in a fresh worktree and compares the per-locus outputs.
- It writes `reviews/<run_id>.md` with a verdict.
- Make the 0.05-AUPRC trigger symmetric: also trigger when a baseline is at or below the lead-SNP floor, or deviates from its known behaviour.
- Define "improvement" as relative to the best baseline in the same cell.

**S5. Run small pilots before large runs (§10).**
- Every run over 1 h starts with a 1% pilot that logs extrapolated time, peak RSS per worker and file count.
- Wrap every job in `timeout`. NPE checkpoints every 30 minutes and writes a heartbeat file.
- Specific risk: 64 R workers each holding a 5,000² float64 LD matrix (200 MB) plus SuSiE copies may exceed 2 GB per core. The pilot measures this, and parallelism is set from the result.

**S6. Add a record-hygiene gate (§9, §11).**
- `scripts/check_records.py` runs at each milestone gate and checks:
  - every file under `data/raw` has a DATA.md row with a matching sha256;
  - RESEARCH_TREE.md has a status entry for the current milestone;
  - the latest commit is pushed (M1).
- `download_1kg30x.sh` must verify against the upstream md5s in `20220804_manifest.txt` (E6) and fail on mismatch.

**S7. Order the kill-tests by dependency (§5, §6).**
- B must pass its correctness tests (B1 identity to coloc.abf, the 2-SNP toy) before it serves as the reference posterior in C's kill-test C1. Otherwise a bug in B kills C.
- Kill rules that refer to coloc-SuSiE apply only after that baseline passes M5(b) and (c).

**S8. Handle NPE replicates and the training sampler (§5 family C).**
- Pre-specify that test results are reported for both replicate seeds (mean and both values), not for the better one.
- Temperature scaling and checkpoint selection use the validation set only. Log these choices in the run manifest.

## Could-change

**C1. Personal PI checks, about 10 minutes per milestone.** These checks go in §12. The PI reads files, not prose:
1. M3: the generator ρ and power test outputs, and `check_protected` / `check_records` passing.
2. M4: the baseline acceptance and known-direction results.
3. Each kill decision: `verify_run.py` passes and the n counts match.
4. Before the test run: `git diff freeze-design..freeze-<name> -- <protected files>` is empty or covered by decision notes.
5. The test run itself (M2e).
6. REPORT: `check_report_numbers.py` passes.

**C2. Pin GitHub-only baselines.** Pin SharePro, ColocBoost and gpu-coloc to commit SHAs, and record the SHAs in DATA.md or the lock files.

## Nothing needed

These parts are sound from the agent-failure angle and I propose no changes:
- §6.2 "kill-test failures never re-run with altered metrics";
- §12.3, the headline excluding L0;
- §12.7, narrow scopes;
- the ≤ 400-word summaries;
- frozen power thresholds whose arithmetic checks out (E7).

Sources: https://docs.hpc.ethz.ch/hardware/storage/ ; https://stephenslab.r-universe.dev/api/packages/susieR ; https://stephenslab.r-universe.dev/susieR/doc/rss_mismatch.html
