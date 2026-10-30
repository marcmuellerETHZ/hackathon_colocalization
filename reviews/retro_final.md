# Retrospective, final milestone (process only), 2026-10-28 ~05:15

Reviewer: a fresh agent that produced none of the results. Sources: RESEARCH_TREE D0–D21, LESSONS, the earlier retros, DECISIONS, JOBS.yaml, test_access.log, git history, and REPORT_DRAFT @ e88b579. I read no test-set data.

## 1. Process

**Time against the 22-h budget (D7, 10:15 → 08:15).**
- *Well:* the run stayed ahead of plan.
  - Phases 0–2 were done by 11:50 (planned 13:30).
  - The freeze came at 23:40 (D7 planned 01:00).
  - The test run completed at 04:43, leaving 3.5 h for the report.
  - The report was pre-drafted during the run (0f23cfb).
- *Badly:* about 4.6 h (21%) had no supervision:
  - two pauses, 13:45–14:05 and 16:41–19:10 (D14);
  - an unsupervised gap from 20:25 to 22:15 (D18), during which the dry run had passed at 20:30 but nothing advanced.

  The duplicate-PI handover cost a further ~1 h (D11/D12). In JOBS.yaml, 104 of 228 jobs were cancelled, 38 of them resubmitted for RAM.

**D17 cut and D21 drop.**
- *Well:* the drop itself was clean.
  - The cut-1 file differs from the frozen file only by the mvSuSiE entries, YAML wrapping and `derived_from`. I diffed the two, and the sha256 values match the access log.
  - The cancel and resume took 39 s, under the same nonce, commit and checkpoints.
  - Interrupt-resume had been dry-run beforehand (D18).
- *Badly:* the trigger came late. test_run_v1 had `expected_end` 05:10, already past the 05:00 line, when it was submitted. The cut came 2 h 13 min into the run. Projections missed both ways:
  - mvSuSiE was frozen at 70 s per locus and observed at 170;
  - the post-cut ETA was 06:40 against an actual 04:43.

**Kill-test design.**
- *Well:* no post-hoc rescue (D13, D16), and B1 was verified independently.
- *Badly:*
  - The matched-W rule was never dry-run on a calibrated method (lesson 21).
  - The B1 job failed on plumbing at 19:24.
  - C2 (5–6 h of data generation) never fitted after C1 passed at 19:25.

**M3.5 restart.**
- *Well:* 10/10 orch tests passed, including a deliberate OOM. Regeneration was bit-identical. P3-1..16 cleared the backlog. Of 11 failed jobs, 0 were left unhandled.
- *Badly:* nothing guarded against a second "pick it up again". A second PI resumed duplicates of all 5 agents (D11).

**Coordination.** The duplicate's commit bc7bdba broke run_eval.py in the famA worktree. A1 was decided from that same worktree 40 min later (D13) and never independently recomputed.

**Supervision.** The live-state table (lesson 18) did not prevent the D18 gap.

## 2. Hard rules

| Rule | Verdict | Evidence / doubt |
|---|---|---|
| Fixed seeds | Yes (minor) | `src/common/seeds.py` asserts the list. `torch.manual_seed(1)` at src/famC/c1.py:122 is outside it. |
| No restricted data | Yes | The largest blob ever committed is famC model.pt (4.2 MB). No VCF, pgen or summary statistics are in the history. Test IDs are stored only as sha256. |
| Metrics, labels and tests unchanged | Yes (disclose one) | D13, D16, P3-29 and P3-30 took the less flattering option. P3-25 cut C1 KL about 7× after the 4M pilot failed (0.143). The cut was justified and verified, but REPORT should say this is how C1 crossed its bar. |
| Independent review | **Partial** | **A1 has no independent recompute** (no reviews/ file, no verify task). Test-run verification (p7.verify) was pending while the REPORT numbers were drafted (8506a05). |
| Runtime with accuracy | **Doubtful** | 76× (G) and 294× (B) compare one GPU with one CPU core. Pre-registered C2 is whole-machine throughput against coloc-SuSiE on 64 cores; famAG_M3.md:82 says "G throughput is only ~1× a 64-core susieR". |
| Home untouched | **Doubtful (setup)** | Written before the HOME redirect (9c99882, 14:31 on 10-26): `~/.conda/environments.txt` gained `.../env/renv` (13:19), plus `~/.cache/uv` (13:17), `~/.cache/mamba/proc` and `~/.mamba/pkgs/urls.txt` (14:23). Nothing after that. Not touched by me. |

## 3. Candidate lessons (not in LESSONS.md)

1. **Report speed in the pre-registered unit (whole-machine wall throughput); per-worker ratios are secondary.** Evidence: 76× is one GPU vs one core; famAG_M3.md:82 measured about 1× a 64-core susieR.
2. **Check the overrun drop rule at submission against the job's own expected end, using per-method s/locus measured at test-run concurrency.** Evidence: `expected_end` 05:10 was past the 05:00 line at submission; mvSuSiE ran 2.4× slower than frozen; the cut came 2 h 13 min in (D21).
3. **"Verified" only once the reviews/ file exists; REPORT verification claims come from that index.** Evidence: A1 has no review, yet REPORT §3 says every kill verdict was recomputed. The Summary says "independently verified" while p7.verify was still running.
4. **Before any pause or session end, queue the critical path as gated dependent jobs.** Evidence: the dry run passed at 20:30, but freeze work began at 22:15 (D18); 21% of the budget was unsupervised.
5. **Make the HOME/cache redirect the first commit, before any install.** Evidence: four home paths were written between 13:17 and 14:23, before 9c99882.

## 4. REPORT §8 / §9

**Inaccurate:**
- §8 "one PI session": there were three successive sessions, -7c, -fa and a0 (D12, D18), plus an overlap (D11).
- §8 "every kill verdict … and the test-run result were recomputed" (also §3 line 99 and the Summary): A1 was not recomputed, and p7.verify is pending.
- §8 "Supervision gaps … fixed this": D18 happened after the fix.
- §9 "Two baselines (…, mvSuSiE) use screen operating points": mvSuSiE did not run on test, so only one baseline does.

**Missing:**
- §8: time accounting (about 4.6 h unsupervised, about 1 h handover, 46% of jobs cancelled); D21 as a process event.
- §9: the runtime unit. Qualify Summary finding 3 as per-unit, not whole-machine (C2).
- §9: the C1 pass depends on the P3-25 scoring change.
- §9 (optional): mvSuSiE's frozen t_S > 1 (P3-30), so it could make no confident S calls anyway. The cut cost little.

--- AGENT FOOTER ---
task id: retro-final
branch / commit: retro-final / see PI hand-off
artifacts registered (name@version, checksum): none (review only)
jobs launched (ids, final status): none
jobs still running (ids): none
tests (command -> result): diff frozen vs cut-1 methods -> mvSuSiE + wrapping + derived_from only, sha256 = access log; git blob scan -> max 4.2 MB, no data formats; find /home -newermt 2026-10-26 -> setup writes 13:17–14:23 only
decisions needed (each WITH your recommended default):
- Runtime: default, add whole-machine throughput to Summary/§7/§9; per-locus secondary.
- A1: default, short independent recompute before the final REPORT; else state "A1 not independently verified".
- "Independently verified" wording: default, only after reviews/p7_verify*.md exists.
- P3-25: default, one line in §9.
