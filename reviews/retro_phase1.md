# Retrospective, end of Phase 1 (process only), 2026-10-27 ~10:45

Evidence: git log, STATE_M3.5.md, scripts/reconcile (12 flags), LESSONS.md, DECISIONS.md.

## 1. What we should stop doing

1. **Merging unreviewed science code and building big artifacts on it.** sim/train_api/oracle were committed unreviewed (3416f2e, 850218b, fb426c4); screen_v2 (119 GB) ran while fb426c4 landed mid-run; l0star_v1 was built from uncommitted code. Cost: a full verify-by-regeneration task (p2.verify_artifacts). Rule: *No artifact build starts from an uncommitted or still-changing tree; the job records the commit and refuses a dirty tree.*
2. **Putting data/results in the same run as the code change that invalidates them.** screen_v1 (152 GB) was superseded by D6a. Rule: *Pilot at <=5% size until the generator/statistics are frozen; large builds only after a conformance review.*
3. **Ending sessions with half-done work and no state.** M3.5 stopped with famC eval, baselines outputs and real-data step unfinished, placeholders in two reports (`__C1L8__`, `VALID_GRID_*`, still flagged by reconcile). Rule: *Reports are written by the same task that finishes the run, or the task is not marked done; reconcile placeholders are blockers.*
4. **Letting agents share the GPU ad hoc.** Five agents caused CUDA OOMs (c1_pilot crashed, famB 29 loci OOM-killed). Fixed only by orch and D8. Rule: *All GPU use goes through the queue from day one.*
5. **Registry drift.** Reconcile now shows 5 tasks "running" without live jobs, an unregistered 27.9 GB famC dir, a checksum mismatch from in-place attribute rewrite (real_locus_sets@1), and an unmanaged process. Rule: *Never mutate a registered artifact in place; new version or no write. Agents ack/close tasks before footer.*
6. **Late discovery of data and statistical defects.** Corrupt chr7 VCF (lesson 4), non-OLS z/se (lesson 9), rho-hat bias (P3-23, found only after M3). Rule: *Every data/estimator module ships a known-answer check on real inputs before consumers are built.*
7. **Agent coordination accidents.** SendMessage to a running agent spawned a duplicate (lesson 7); mid-run script edits (lesson 8); a dirty worktree and failed pytest job (p3-famA-fixes, p2rA_pytest_famAG) left with no owner. Rule: *One agent per task id; failed jobs are acked or requeued by the owner before footer.*
8. **Decision volume as a bottleneck.** 23+ P3 decisions plus D0-D8; many "pending" items were the PI's to decide earlier. Rule: *Each report ends with a recommended default; PI accepts by default within one cycle.*

## 2. What worked well

1. **Independent reviewers who compute.** Rounds 1-3 (reviews/*) caught Kronecker order, rho grid, split counts; round 3 gated the freeze (tag design-v4-frozen). Rule: *Reviewers must run numbers, not just read.*
2. **Identity gates against reference implementations** (G vs susieR 6.7e-8; A vs mvsusieR 8e-14; coloc.abf 5e-16). Rule: *Every reimplementation ships an identity test with a stated tolerance.*
3. **Kill tests before scaling** (famC C0 passed at 5M; cap of 100k loci before a kill test). Rule: *Cheapest falsifier first.*
4. **Pre-registration and test lock** (FROZEN_SHA256, test ID sha256, truth-isolation test caught src/famC referencing truth, P3-21). Rule: *Isolation is enforced by a test, not trust.*
5. **Orchestration infrastructure** (orch queue, TASKS/ARTIFACTS/JOBS, reconcile, worktrees; 10/10 toy tests incl. deliberate OOM). Reconcile already surfaces real drift. Rule: *Run reconcile at every cycle and clear flags before dispatch.*
6. **Verification by deterministic regeneration** (screen_v2, valid_grid_v1 bit-identical; l0star_v1 replica byte-identical). Rule: *Seeded, bit-reproducible builders; provenance by regeneration.*
7. **Early pilots finding wrapper bugs** (baselines pilot, prop.coloc 15% failure, ColocBoost crash workaround) and a written lessons file with evidence. Rule: *Run 200-locus pilots with controls before any 50k-locus run.*
8. **Backward-scheduled budget (D7) and explicit decision log with reasons.** Rule: *Re-plan backward from the hard stop at every status summary.*

## 3. Risks for the next 20 h (deadline ~08:15; test run starts ~22:00)

1. **Freeze slip / unfinished prerequisites.** Family A fixes (dirty worktree), C1 (4M E[KL] 0.143 vs bar 0.02, likely fails), sim review and m4_valid have no live jobs. Mitigation: set a 20:00 hard cut, freeze whatever passes, report failed families honestly (kill-test negative is a valid result).
2. **Test run compute (about 7 h) with no rehearsal margin.** Projected baselines 7.2 h before P3-17; one OOM, timeout or wrapper crash on test data cannot be fixed after the lock without violating rules. Mitigation: full dry run of the exact test pipeline on screen_v2/valid by 20:00; checkpoint/resume per cell.
3. **PI context/state loss and registry drift** (12 reconcile flags, placeholders in reports, 5 phantom "running" tasks). Mitigation: clear flags now, update RESEARCH_TREE.md before each wait, and keep REPORT.md skeleton written in advance.
