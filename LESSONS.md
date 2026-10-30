# LESSONS — process rules for running this project efficiently

Each rule: one line + evidence. Pruned at each milestone (last prune: 2026-10-27 after the Phase 1 retrospective, reviews/retro_phase1.md).

## Jobs, code and artifacts
1. Every job runs from a committed snapshot through the queue, and every artifact records its commit. Never build artifacts from a dirty tree. Evidence: l0star_v1 was built from uncommitted code, and screen_v2's build spanned a sim commit; both needed 1 h of forensic regeneration to verify.
2. All GPU use goes through the queue with declared VRAM. Evidence: 5 agents sharing one GPU in M3.5 produced CUDA OOM deaths (c1_pilot, famB reference, generators) and a stalled project.
3. Never mutate a registered artifact; write a new version. Evidence: real_locus_sets@1 was rewritten in place, so reconcile flags a checksum mismatch and @1 is no longer reproducible.
4. Pilot at ≤ 5% scale until the generator and its consumers pass review. Evidence: screen_v1 (152 GB) was superseded within 2 h by the OLS fix (D6a).
5. Verify by deterministic regeneration, not by trust. Evidence: 72/72 and 102/102 bit-identical regenerations settled screen_v2/valid_grid_v1 provenance in 15 min.

## Verification
6. Reviewers must compute (known-answer checks, re-implementations), not just read. Evidence: Kronecker order, generator scaling, ρ̂ bias, the unreachable ρ grid and the P_S/P_D formulas were all found by computation; the sim reviewer's own call-stack check found a test-run-aborting bug.
7. Check method wrappers against their documented examples and known-direction controls on realistic inputs before benchmarking. Evidence: non-OLS z/se made prop.coloc reject sharing on 70% of shared loci; caught before 50k loci were spent.
8. Verify downloads against upstream checksums, never discard stderr, and set pipefail in subshells. Evidence: the corrupt chr7 VCF was silently converted to 10% of its variants.
9. Test per-chromosome or per-cell pipelines on one small unit before fanning out. Evidence: the chr22 test (1 min) caught two bugs that had failed a 21-job fan-out.

## Coordination
10. Never SendMessage to an agent inside a workflow (it forks a duplicate). Use standalone background agents for anything you may need to talk to. Evidence: the duplicated data agent corrupted its own scan.
11. Never edit a script while it runs; jobs run from snapshots. Evidence: two mid-run edits broke running jobs.
12. Agents end with a recommended default for every decision; the PI accepts unless it objects. Evidence: 23 P3 decisions piled up while agents waited or guessed.
13. A task is not done while reconcile shows its placeholders, failed jobs or dirty tree. Evidence: `__C1L8__` and VALID_GRID_* placeholders survived a session end unnoticed.
14. The PI never ends its turn while jobs run; it loops on reconcile. Evidence: the M3.5 stall went unnoticed for an entire session.

## Environment
15. Keep command-line tools (bcftools/plink2) in their own env, and link untracked env/ installs into job snapshots. Evidence: an R install removed bcftools, and snapshots without env/ broke the baseline tests.
16. Never write outside the project directory (the user's instruction). Keep all caches and HOME inside env/. Evidence: user instruction 2026-10-26.
17. Give critical-path jobs queue priority, and keep dry runs and plumbing jobs light. Evidence: a 16 GB / 100 GB-RAM dry-run job at the queue head would have pushed kill tests past the 20:00 cut; fixed with `orch prio` and by cancelling the job.
18. A Claude Code session end stops all agents, but orch jobs keep running. On every restart, compare `orch status` with the agents and resume the agents (SendMessage). Keep a live-state table (task → running jobs → waiting on) at the top of RESEARCH_TREE.md, refreshed at every heartbeat. Evidence: at 12:45 five agents stopped silently while their jobs ran; the user noticed that the tree did not say what was running.
19. Declare realistic RAM (peak RSS + margin); over-declaration starves the memory-bound queue. Within equal priority the oldest job is head, so give the critical job a strictly higher priority. Evidence: A1 (24 GB declared, about 3 GB used) and famB screen (48 GB declared) waited about 2 h behind backfill; A1 started minutes after it got a unique top priority and the GPU-blocked head started reserving its CPU/RAM share.
20. Only one PI session may be active. On "pick it up again", first check for a live PI and live agents (ListAgents, `orch scheduler --status`, recent commits) before resuming agents. Evidence: a second PI copy resumed duplicates of all 5 agents at 14:34; the famA duplicate broke run_eval.py in a shared worktree (D11).
21. Dry-run every kill rule on a pilot with a known-good, well-calibrated method before pre-registering it, and constrain both error directions. Evidence: B1 killed ExactColoc via the matched-W rule (t_S = 0.013, so 22% false-S on D) although at 0.8 it was best on false-S (D16; reviews/retro_M5.md).
22. A family enters its kill ladder only when its evaluation path is pilot-verified and every rung is budgeted against the freeze. Evidence: C1 needed a scoring fix and a re-run (P3-25), and C2 became untestable.
23. At every pause or resume, regenerate the backward schedule and write it into the tree. Evidence: the D14 pause slipped selection by 1 h with no re-plan until later.
24. Every hard cut gets a pre-declared fallback deliverable. Evidence: D16's MVR route kept the test run meaningful when no candidate survived.

## Final milestone (reviews/retro_final.md)
25. Report speed as whole-machine throughput (the pre-registered unit), with per-worker ratios as secondary figures. Evidence: the report draft headlined G at 76× and B at 294× (one GPU vs one core); against 64-core coloc-SuSiE they are about 1.2× and 4.6×.
26. Check the overrun drop rule when the job is submitted, against the job's own expected end. Evidence: test_run_v1 was submitted with an expected end of 05:10, already past the 05:00 drop line; the mvSuSiE cut (D21) came 2 h 13 min into the run.
27. Call a result "verified" only once its review file in reviews/ exists. Evidence: the report summary said "independently verified" while p7.verify was still running, and A1 (D13) never had a recompute until the end.
28. Before any pause, queue the critical path as gated dependent jobs so that it advances unsupervised. Evidence: the dry run passed at 20:30, but nothing moved until 22:15 (D18).
29. Make the HOME and cache redirect the first commit, before any install. Evidence: installs on 10-26 between 13:17 and 14:23 wrote to the user's home before commit 9c99882.

## Phase 2 (reviews/retro_p2.md)
30. Take decision timestamps from git or the queue, never from memory. Evidence: P4-3 and P4-3a are logged 12 and 30 min later than their commits, and P4-3 is labelled pre-result though it cites the result.
31. Calibrate identity or tolerance bars against the reference's own run-to-run spread before freezing them. Evidence: the frozen 1e-3 bar was unattainable for chaotic non-converged susieR fits, which forced two post-hoc rules (P4-3, P4-3a).
32. Give each track a fixed GPU memory budget at dispatch, set from its measured peak, and keep bulk data generation on CPU when the GPU is shared. Evidence: famC generation held the GPU from 09:56 to 13:56 while most cores sat idle; 24 of 61 GPU jobs were killed and resubmitted; B-mm waited 84 min.
33. Open every summary with one line per track giving its verdict, the rung it failed and the margin. Tag every other number as pre-registered, descriptive, post-hoc or unverified, and show a failed method's favourable metric only after its verdict. Evidence: presenting C's in-simulator 0.96 as "promising" after C2 failed read as moving goalposts to the user.
