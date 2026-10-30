# STATE at restart (Phase 0 inventory), 2026-10-27 10:10

Evidence was gathered with `ps -u mumarc`, `nvidia-smi`, the logs, `git status`/`git log`, the reports and the set files. No new science was run.

## Processes, GPU, workflow
- **Project processes:** none running (`ps -u mumarc` shows only VS Code / tunnel / Claude). There are no orphans, so nothing was killed. Processes of other users on the node are not ours and were not touched.
- **GPU:** idle (1 MiB used, 0%).
- **Workflow w54efjy22 (M3.5):** stopped when the previous session ended.
  - Finished their tasks: sim, eval, famA.
  - Did not finish: famC (C1 evaluation at 8 M never run) and baselines (pilot ran, but no report and outputs not committed).
  - Never started: realdata step and the independent verifier.

## Git
- HEAD 4b30875. The sim/train_api/oracle code was committed by the agents in 3416f2e (18:06), 850218b (18:13) and fb426c4 (18:26); famA/famG in 47ceb97 (18:09). None of these commits was reviewed independently.
- **Uncommitted:**
  - `src/baselines/pilot.py` (+`by_level` reporting function);
  - `src/baselines/benchmarks/*screen_v1*`, `*screen_v2*` (pilot outputs);
  - `configs/{testset_v1.yaml, test_methods_v1.yaml, FROZEN_SHA256}`;
  - logs (untracked).

## Artifacts (none record the producing commit, a provenance gap)

| Artifact | Built | Code state at build | Status |
|---|---|---|---|
| screen_v1 (sim_v1, 152 GB) | 10-26 ~15:15–15:45 | committed (bbd0804) | **superseded** (pre-OLS, D6a); delete candidate |
| screen_v2 (sim_v2, 12,000 loci, 119 GB) | 10-26 ~18:09–19:06 | started 3 min after 3416f2e; **fb426c4 (ρ̂ level inference) was committed during the run** | **provisional:** verify by deterministic regeneration of a locus sample at HEAD; if not bit-identical, mark stale and regenerate |
| l0star_v1 (2,000 loci, p = 20) | 10-26 17:54 | **before 3416f2e, i.e. uncommitted sim code** | **stale** unless regeneration at HEAD is bit-identical |
| valid_grid_v1 (34 cells, 232 GB) | 10-26 ~19:05–21:21 | after the last sim code commit (fb426c4); unclear whether the tree was clean | **complete:** every cell has S/D/W = 300 and G = 100 (logs/sim/valid_grid_v1.*.log). **No sha256** (SHA256SUMS written 19:08) and the report has placeholders. Verify like screen_v2. |
| shifted_dry_v1 | 10-26 18:39 | dry run (validation) | dry run only |
| famC C0 (results/famC/c0_full, 5 M) | 10-26 17:41 | famC code committed 0aded87 | **C0 kill test passed** (E[KL] 0.00072, q99 \|ΔPP4\| 0.043) |
| famC c1_pilot | 10-26 ~18:20–19:07 | — | **crashed:** CUDA OOM at 1.2 M loci (shared GPU) |
| famC c1_long (8 M, batch 512) | 10-26 19:16–20:36 | sim train API at fb426c4 | **training finished** (model.pt at 20:36, step 15,600 of 15,625). **C1 evaluation at 8 M never ran** (report placeholders `__C1L8__`, `__C1L8W__`). Checkpoint evaluations at 1/2/4 M: E[KL] (class-balanced) 0.220 → 0.178 → 0.143; natural-weighted 0.091 → 0.064 → 0.052; bar 0.02 → **failing so far** |
| baselines pilot200 on screen_v2 (10 methods) | 10-26 18:19–19:47 | baselines code plus an uncommitted `pilot.py` change (reporting only) | complete; outputs uncommitted; no M3.5 report |
| eval dry runs | 10-26 17:26 | committed | done (eval_M3.5 report) |
| real-data locus sets R2/R3b/R4b (train/valid) | M3 | committed (6532bea) | ρ̂ attributes pre-date D6(d), so the M3.5 real-data step is pending |

## Failures found in logs
- `logs/famC/c1_pilot_train.log`, `c1_pilot_train_oom1.log`: CUDA OOM. Five agents shared the GPU.
- `logs/sim/screen_v2_L*.log`: CUDA allocator OOM warnings, recovered by the builder's OOM-retry.
- famB: 29 loci OOM-killed in the C1 pilot reference, re-run OK (famC report).
- `logs/baselines/pilot_screen_v1.log`: tracebacks in the screen_v1 pilot (superseded set).

## Reports with placeholders
- `reports/simulation_engineer_M3.5.md` §4: VALID_GRID_COMMIT, VALID_GRID_LOCI/SIZE/TIME/NOTES.
- `reports/famC_M3.5.md` §5: `__C1L8__`, `__C1L8W__`.

## Pending decisions collected from the reports (resolved in DECISIONS.md, Phase 3)
- ρ̂ refinement and its mode on real data (sim §7.1).
- Re-freeze of the `oracle.py` hash (sim §7.2).
- testset_v1 open items (sim §7.3, eval §6.1).
- mvSuSiE "active" rule: hard lfsr < 0.05 vs soft 1 − lfsr (eval §6.2).
- C3 operating-point source (eval §6.3).
- Family A: own-effect guard, π (EB κ = 4 vs fixed), η precision, **multi-start IBSS and per-trait LD before A1**, η floor (famAG §8).
- SharePro trait-signal formula.
- Family B prior and M.
- Family C: architecture additions C-A1..A4, the C1 path (budget vs relaxed bar), L6/L6c hold-out.
- Test-lock configs (testset_v1.yaml, test_methods_v1.yaml, FROZEN_SHA256, methods list, frozen-* tag).
