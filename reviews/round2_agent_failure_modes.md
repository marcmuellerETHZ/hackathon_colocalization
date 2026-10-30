# Review round 2: AI-agent failure modes (DESIGN.md v2)

**Verdict: minor revision.** The science plan needs no change from this angle. v2 adopted most round-1 process items, but several are written as prose with no mechanism behind them. One silent data corruption has already happened and passed through the pipeline unnoticed (E1). The fixes below are each one small script or one rule. None of them adds a backup remote or touches $HOME.

## Evidence (checked 2026-10-26, read-only; throwaway files only in my /tmp scratchpad)

| # | Check | Result |
|---|---|---|
| E1 | md5 of `data/raw/1kg30x/chr*.vcf.gz` against upstream `20220804_manifest.txt` | **chr7 FAILS.** Its size equals the upstream Content-Length (1,751,799,644 B), but `gzip -t` reports "invalid compressed data--crc error". The other 21 VCFs and all 22 `.tbi` files match. `SHA256SUMS` was computed after download, so it records the corrupt file as the reference. |
| E2 | Per-chromosome output of `prep_1kg_pgen.sh` (`logs/prep_1kg_pgen.log`) | **chr7: 66,497 variants**, against chr8 at 691,021 and chr6 at 856,408. About 90% of chr7 is missing, yet the pgen was written and the script exited "successfully". Cause: `bcftools ... 2>/dev/null \| ...` runs inside an exported function under `xargs bash -c`, and `set -o pipefail` is not inherited there, so the decompression error was discarded. All pools, the mosaic and the chr7 blocks of every split would have been built on 10% of the variants. |
| E3 | `env/requirements.lock`, `env/renv.lock` | **Still absent.** §7.5 says they "are committed", so the design text is ahead of reality for the second round running. |
| E4 | `RESEARCH_TREE.md`, `DATA.md` | The tree still shows M0 `[~]` and M1/M2 `[ ]`, with no M1 or M2 status summaries, although both milestones are committed. DATA.md still has an empty table with 28 GB downloaded. PROMPT requires both. After a context reset the PI would resume from the wrong state. |
| E5 | `env/activate.sh` and R's `tools::R_user_dir` | Caches are redirected, but `R_user_dir(,"data")` resolves to `$HOME/.local/share/R` and `"config"` to `$HOME/.config/R`. `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_STATE_HOME`, `CUDA_CACHE_PATH`, `TRITON_CACHE_DIR` (default `~/.triton`, used by `torch.compile`), `CUPY_CACHE_DIR`, `IPYTHONDIR`, `UV_PYTHON_INSTALL_DIR` and `UV_TOOL_DIR` are unset. Nothing forces an agent to source the file. Git identity comes from GECOS/hostname, not `~/.gitconfig`, so overriding `HOME` is safe for commits. |
| E6 | `env/` | `env/rlib_susie016` is being set up next to the susieR 0.14.2 library. Two versions of the same package make it possible for a wrapper to load the wrong one without noticing. |

## 1. Round-1 must-change items

| R1 item | Status | Reason |
|---|---|---|
| M1 scratch purge / backup remote | **Superseded by the user.** File-count part addressed. | The user declined a backup and does not mind the purge, so this is not re-raised. One parquet per run is in §7.2. The verify-on-download part (S6) was not done, and E1 shows it is needed for integrity, not backup. |
| M2 mechanical test lock | **Partially** | Present: the `eval_test.py`-only generator, the loader guard, the log, and one test run for all candidates (§6.3). R1 on train only is addressed. Missing: the run is not one-shot (a new `frozen-*` tag permits another test run), there is no clean-tree check, the log is not committed immediately, and the test-set composition is unspecified (see M4 below). |
| M3 truth isolation | **Mostly** | ρ̂ in the headline, the `R_true` test at ≥ L1 (§2.3, §7.1-8) and NPE trained on train only are all in. Methods still do not run in a subprocess that sees only the input file. That is acceptable if the §7.1-8 test checks the actual files given to every wrapper. |
| M4 manifest / ledger / verify_run | **Partially** | The manifest, ledger, single `metrics.py` and `verify_run` recompute are in. Missing: `n_expected` and per-locus status, so a dropped locus is invisible (see M3 below). Also missing: the "tests passed at this hash" record, the rule that claims must cite a run_id, and a statement that the ledger is written by code, not by agents. |
| M5 baseline failures and integrity | **Partially** | Failures are now separate from ties, wrapper examples are reproduced, and mismatch-aware coloc-SuSiE is in. The "known-direction controls" are referenced in §4.3 but **absent from §7.1-9**, which lists only the s > 0.9, null and shuffled controls. "Non-convergence" counts as failure, which would wrongly fail susieR at L2/L4 (see M3 below). |
| M6 split script and protected files | **Mostly** | The block-level split with script, SHA256SUMS and audit test resolves the E2 drift. `FROZEN_SHA256` exists but omits test tolerance constants. Nothing runs the check automatically, and the hash file can be edited in the same commit as the protected file (see S1 below). |
| M7 exhaustive labels and test specification | **Partially** | The precedence partition plus unit test is good. Still unspecified: test-set loci per cell, class targets, N1/N2 mix, ρ sign handling, the pool on which the power-matching quantiles are computed, the shifted-arm size and seed, and which pooling is the headline if only the MVR grid (L0–L4 × {0, 0.3, 0.5}) gets run. |
| M8 timing lock | **Partially** | `flock` on `.timing.lock` only excludes other timing runs. NPE training and other agents' GPU jobs do not take the lock, so "otherwise idle" is honour-system and nothing records it. Repeats are not specified. |
| M9 environment and tree drift | **Not / weakened** | The lock files do not exist (E3). The rule "worktrees **or directories**" lets agents share one working tree, which conflicts with the clean-tree requirement: official runs will either always refuse or get the check disabled. |

Round-1 should-items:
- **Not addressed:** S1 (screening budget, configuration counts), S3 (report numbers by script), S5 (pilots), S6 (record gate and md5 check, now proven necessary by E1), S8 (report both NPE replicates).
- **Partially addressed:** S2 (only the NPE seed is derived) and S4 (reviewer reruns a subset, but trigger is one-sided).
- **Addressed or implicit:** S7 (kill-test order) and C2 (pinned commits).

## 2. New or missed problems

1. **Silent truncation of input data has already happened** (E1, E2). The pattern of `2>/dev/null`, pipelines without `pipefail` in subshells, and checksums computed after the fact will recur in the mosaic, LD and harmonisation steps. Nothing in §7 checks that outputs are plausible in size.
2. **Shared working tree combined with an editable install.** If `src` is importable from `.venv` (editable install or PYTHONPATH pointing at the main tree), an agent in a worktree runs the *main tree's* code while the manifest records the worktree's hash. That breaks provenance without any error.
3. **Inner joins and dropped rows.** v2 has no rule that metrics are computed over the *expected* locus list. Loci that are missing, or that have NaN scores, are dropped by a pandas merge or groupby without any warning. This is the easiest route for the ">1% failure blocks claims" rule to be bypassed by accident.
4. **The failure definition harms baselines or blocks C1.** `susie_rss` often warns about non-convergence under mismatch but still returns usable output. Scoring that as a worst-case failure turns the strongest baseline into a straw man. It also trips the 1% block and stalls C1. Separately, scoring failures at the worst case uses the truth label: a GPU method with zero failures gains automatically against a baseline failing at 0.9%.
5. **The test lock is not one-shot.** Any agent can create a `frozen-*` tag. One-shot use is the property that matters, and the tag is not needed to get it.
6. **Precomputed test-block artefacts are not guarded.** M3 builds the mosaic and LD for all 1,336 blocks. The loader guard covers simulated test loci but not these files. Global statistics computed over all blocks (NPE feature normalisers, screen size M, eigen-spectrum summaries) would leak test-region information.
7. **Validation reuse without limit.** The same screening set serves kill tests, M6 development and the stop rule. Neither a fresh confirmation draw nor a count of configurations tried is specified.

## 3. Proposed changes

### Must

**MC1. Fail-loud data pipeline (§2.1, §9; evidence E1, E2).**
- `download_1kg30x.sh` verifies every file against the upstream manifest md5 and exits non-zero on mismatch. DATA.md sha256 values are written only after this passes.
- Re-download chr7 and rebuild `chr7.pgen`.
- In `prep_1kg_pgen.sh` and all future pipelines:
  - put `set -euo pipefail` *inside* functions run via `bash -c`;
  - no `2>/dev/null` on data tools; send stderr to a log;
  - `bcftools`/`gzip -t` integrity check before conversion.
- Add `tests/test_inputs.py`: variants per Mb per chromosome must lie within 0.5–2× of the genome median, and upstream md5 checks must pass. Run it in `run_tests.sh`, so every result inherits it.
- Add a LESSONS rule citing E2.

**MC2. One worktree per writing agent, with an import-path assertion (§7.5).**
- Delete "or directories". Each development agent gets `git worktree add $P/worktrees/<agent> -b <agent>` inside the project. Only the PI merges.
- No editable install of `src` into `.venv`. Entry points set `PYTHONPATH=$(git rev-parse --show-toplevel)/src`.
- `runlog` asserts that `src.__file__` lies under the toplevel whose hash it records, and refuses a dirty tree for official runs.
- R wrappers assert `packageVersion("susieR")` and `.libPaths()[1]` and record both (E6).

**MC3. Completeness-safe metrics and a precise failure definition (§4.3, §7.1-9, §7.2).**
- `metrics.py` takes the committed list of expected loci for the cell, which is generated together with the locus set. It **outer-joins** predictions to that list. A locus that is missing, NaN or non-finite counts as failure. The manifest records `n_expected`, `n_ok`, `n_failed` and `n_timeout` per method and cell.
- **Failure** means no usable output: a crash, a timeout or a non-finite score. Non-convergence that still returns output is *used and flagged*, and the flag rate is reported. This matches how the standard pipeline is used.
- C1, C3 and C4 must hold under both of two scorings:
  1. failures at the least favourable value (v2);
  2. the common set of loci that every method completed.
- Add the R1 known-direction checks to §7.1-9 as blocking checks:
  - coloc-SuSiE AUPRC at L2 is below L0;
  - coloc.abf false-S rises with ρ when trait 2 has no signal;
  - every baseline beats the lead-SNP floor at L0.

**MC4. Make the test lock one-shot and specify the test set now (§6.3, §7.3, §8).**
- `test_access.log` is committed by `eval_test.py` right after writing.
- `eval_test.py` refuses to start if the log already holds **any** completed entry, whatever the tag. The only exception is resuming a crashed run at the same hash. It also refuses a dirty tree. REPORT lists every log entry.
- Write a `configs/testset.yaml` with:
  - cells, loci per cell and class targets;
  - the N1/N2 and effect strata;
  - how ρ signs are handled;
  - the pool used for the power-matching quantiles;
  - the shifted-arm size and seed;
  - the MVR headline pooling (L1–L4 × {0, 0.3, 0.5}), in case the full grid is not run.

  Hash-protect the file before M6 begins.

**MC5. A timing lock that actually excludes other jobs (§7.7; C2 is a pre-registered claim).**
- Every GPU or heavy-CPU job is started through one launcher that holds `flock -s results/.machine.lock`. Timing runs take `flock -x`.
- Timing runs also assert at start and end that `nvidia-smi --query-compute-apps` is empty apart from themselves and that load average is below 2. Both values go in the manifest, and a failed check invalidates the run.
- Use 3 repeats and report the median.

**MC6. The user's $HOME rule, enforced mechanically (§7.5, §12; E5).**
- In `activate.sh`, also export:
  - `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and `XDG_STATE_HOME`;
  - `R_USER_DATA_DIR` and `R_USER_CONFIG_DIR`;
  - `CUDA_CACHE_PATH`, `TRITON_CACHE_DIR`, `CUPY_CACHE_DIR` and `IPYTHONDIR`;
  - `UV_PYTHON_INSTALL_DIR` and `UV_TOOL_DIR`;
  - as a catch-all, `HOME=$P/env/home`, which is safe for git per E5.
- `run_tests.sh`, the Python package `__init__` and the R wrapper preamble refuse to run unless `PROJECT_ENV_ACTIVE=1` (set by `activate.sh`).

**MC7. Bring records and locks in line with the design text before M3 work counts (§7.5, §9, §11; E3, E4).**
- Generate and commit `env/requirements.lock` and `env/renv.lock` now, plus an R `installed.packages()` dump for both library paths.
- Update RESEARCH_TREE with the M1 and M2 status summaries and decision notes, and fill DATA.md.
- Add `scripts/check_records.py` to every milestone gate in §11. It checks that every `data/raw` file has a DATA.md row and that the sha256 matches, that the lock files match the environment, and that the tree has an entry for the current milestone.

### Should

- **S1. Make the frozen definitions robust (§7.4).**
  - Hash code files plus a single `configs/frozen.yaml` (power thresholds, class rules, cells, seeds, *test tolerances*), not sections of DESIGN.md. Typo fixes there would raise false alarms, and agents would learn to rehash.
  - `check_protected.py` runs in `run_tests.sh` and compares against `git show design-frozen:configs/FROZEN_SHA256`, not against the copy in HEAD, which can be edited in the same commit.
- **S2. Limit validation reuse (§6).**
  - Do M6 development on a validation draw separate from the screening set.
  - Confirm the primary candidate once, on a fresh validation draw, before freezing.
  - The ledger counts configurations per family, and REPORT states them.
  - Each family gets at most 3 screening attempts.
- **S3. A seed registry (§9).** `src/seeds.py` derives every stream with `SeedSequence(base).spawn` keyed by purpose: the M4 validation grid (which has no seed in v2), the shifted arm, bootstraps, per-locus draws and the NPE steps. Not Python `hash()`. R uses L'Ecuyer streams. Both NPE replicates are reported, never the better one.
- **S4. Guard test-block artefacts.** Mosaic, LD and gene files for test blocks live under split-named paths behind the same loader guard. All global normalisers and hyperparameters are fitted on train blocks only.
- **S5. Citable claims only.** Agent summaries cite `run_id:key` for every number. `scripts/check_report_numbers.py` verifies REPORT.md against `metrics.json`. E3 and E4 show the prose drifting for the second round running.
- **S6. Pilot before large runs.** Every run longer than 1 h first runs a 1% pilot that logs extrapolated time, peak RSS per worker and file count. All jobs run under `timeout`. This can be a LESSONS rule.

### Could

- **C1. Fixed PI checks, about 10 minutes per gate.** The PI reads files, not prose: `check_records`, `check_protected`, `verify_run` on the kill-decision runs, the baseline known-direction table, and the test-lock log.
- **C2. Purge.** Nothing to add. `verify_run` already checks input hashes, so any file the purge removes causes a loud failure rather than a wrong result.

## 4. Guard audit: keep, harden or simplify

| Guard | Enforcement in v2 | Assessment |
|---|---|---|
| Loader refuses test blocks | mechanical | **Essential**, keep; extend to precomputed artefacts (S4) |
| `eval_test.py` + tag | tag is agent-creatable | **Harden** to one-shot via the committed log (MC4); the tag adds little |
| manifest / `metrics.py` / `verify_run` | mechanical if `runlog` writes them | **Essential**; add `n_expected` and an outer join (MC3); the ledger must be written by `runlog`, never by hand |
| Hashing DESIGN §1/§8 | fragile | **Simplify** to `frozen.yaml` plus code (S1) |
| "Independent re-implementation" for gains > 0.05 | honour-system, expensive | **Replace.** The identity tests (A vs mvsusie, B vs coloc.abf, G vs susieR) already cross-check implementations. Instead, have a reviewer rerun from the run_id in a fresh worktree, and fire the trigger symmetrically: also when a baseline is at or below the lead-SNP floor. |
| "Every key result" reviewed | unbounded | **Scope** to the gates: M3 generator, baseline acceptance, each kill decision, the freeze and the test run |
| Timing `flock` | does not exclude other jobs | **Harden** (MC5) |
| Worktrees "or directories" | honour-system, conflicts with clean-tree | **Harden** (MC2) |
| Env locks | claimed, absent | **Create** (MC7) |
| No writes to $HOME | honour-system, incomplete | **Harden** (MC6) |
| ≤ 400-word summaries, narrow scopes | rule | Fine as is |
