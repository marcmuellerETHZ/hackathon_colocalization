# Retrospective, end of M5 (process only), 2026-10-27 ~20:15


## 1. What we should stop doing

1. **Pre-registering kill rules without pre-checking their operating point.** B1 failed on matched-W (t_S = 0.013, false-S on D 22%), while at fixed 0.8 it passes and is ~190x faster (D16). Family B died on a threshold artefact; D16 itself calls it a methodological finding. Rule: *Dry-run every pre-registered kill rule on a pilot with a known-good method, and constrain both error directions, before it is locked.*
2. **Reaching the kill tests with the candidate lacking its final scorer/data.** C1 needed a scoring fix (P3-25, 7x KL change) plus a rerun on l0star_v2, landing at 19:25; C2 was then untestable and the "bold" family produced no primary (D15/D16). Rule: *A family enters its kill ladder only when its eval path is verified on a pilot, and all rungs are budgeted against the freeze.*
3. **Two sessions acting as PI.** The second session resumed duplicate agents at 14:34 and broke run_eval.py in famA's worktree (D11); the handover (D12) cost ~1 h. Rule: *Lesson 20 is a hard check: before resuming, `orch scheduler --status` plus a heartbeat file must show no live PI.*
4. **Session-limit pauses without a re-plan.** Two pauses (13:45-14:05, 16:41-19:10) slid selection 20:00 to 21:00 and the freeze 22:00 to 22:30. Rule: *At every pause/resume, regenerate the backward schedule and write it into one place; a stale schedule is a bug.*
5. **Dispatching resource-mismatched jobs.** A1 (24 GB declared, ~3 GB used) and the famB screen starved ~2 h (lesson 19); the 16 GB dry run nearly blocked kill tests (lesson 17). Rule: *Declared RAM comes from a measured pilot; the critical path gets priority at submission.*
6. **Reports tracked with provisional values.** C1 placeholders were replaced only by a later "superseded note" (96009a0), and B1 verdicts were committed as "pending verification" (88b02bc). Rule: *No verdict enters the tree until the verifier has signed; show it as "unverified" in the live table until then.*

## 2. What worked well

1. **Independent verification of every verdict** (p5.verify_famC_scoring recompute to 1e-6; verify_B1 confirmed FAIL/KILL). Rule: *Each kill verdict gets an independent recompute script before it is logged.*
2. **Honouring the no-post-hoc rule** (D13 demote A; D16 refusing to switch B to 0.8). Rule: *Report the alternative reading as sensitivity, never as the verdict.*
3. **Orch queue + reconcile + live-state table** (0 unhandled failures of 10; priorities fixed starvation, D10). Rule: *Live table regenerated at each heartbeat is the single source of state.*
4. **Backward schedule with explicit hard cuts and a fallback** (D7, D9, D16 MVR route). Rule: *Every cut has a pre-declared fallback deliverable.*
5. **Failure policy with honest reporting** (P3-29 L4 failures scored least-favourably and reported as a finding). Rule: *Failures are results; score adversarially, report both ways.*
6. **Recommended-default decisions** (P3-17..29 recorded with reasons). Rule: *Every decision has Decision/Why/default.*

## 3. Top risks for the remaining ~12 h

1. **Test run (~7 h) fails on plumbing after the lock.** Dry run p6dr_dry_v4 was still at 44/120 min; slow methods RAM-bound. Mitigation: do not freeze until the dry run passes end to end; add per-cell checkpoint/resume and a pre-decided drop order (mvSuSiE, ColocBoost first) if it runs past 04:30.
2. **Scope shrinks to "baselines only" and the story is weak.** No candidate survived; claims C1/C3/C4 unsupported. Mitigation: write REPORT.md skeleton during the run with the negative result, C5, runtime and the matched-W finding already drafted.
3. **PI/state loss or another session-limit pause during the run.** Two pauses already occurred; a pause near 02:00 stalls submission. Mitigation: submit the whole test run as one dependency chain up front, with the heartbeat and resume checklist in RESEARCH_TREE.md.
