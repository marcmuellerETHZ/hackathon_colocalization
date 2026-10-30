# AGENT RULES (binding for every agent; from PROMPT.md + PROMPT_RESTART_M3.5.md)

Project root (main checkout, branch master): `/cluster/scratch/mumarc/hackathon_eqtl_gwas`. Read first:
- DESIGN.md (v4, frozen);
- DECISIONS.md (P3-x, binding);
- RESEARCH_TREE.md decision log (D0–D7);
- INTERFACES.md;
- LESSONS.md;
- scripts/orch/README.md.

## Environment and files
- `source /cluster/scratch/mumarc/hackathon_eqtl_gwas/env/activate.sh` before any command.
- Never write to /home or outside the project root. The only exception is temp files under `/tmp/claude-*`.
- **Work only in your own git worktree.** Create it with `scripts/orch/new_worktree.sh <task-or-agent-name>`, which creates `.worktrees/<name>` on its own branch and symlinks data/, env/, .venv.
  - Commit named paths only (`git add <paths>`), never `git commit -a`.
  - Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - The PI merges your branch. Do not merge into master yourself.
  - Never commit data or files over 5 MB.

## Jobs
- **Every job longer than about 1 minute, and every GPU use** (tests included), goes through the queue: `scripts/orch/orch submit ... -- <cmd>`, run from your worktree with a clean tree.
  - Declare `--gpu yes|no`, `--vram-gb` (peak + 1 GB), `--cores` and `--ram-gb` (2 GB per core).
  - Wait with `scripts/orch/orch wait <job_id>`.
  - Only one GPU job runs at a time unless the declared VRAM sums to < 22 GB.
  - Critical-path jobs (kill tests, candidate selection, the test run) are submitted with `--priority 10`; plumbing and dry runs use the default 0 and stay light (≤ 8 GB VRAM, ≤ 40 GB RAM).
- Put each workload on the device that suits it. Small-matrix simulation and data generation run batched on CPU across many cores; the GPU is for training and large batched linear algebra.
- **You are not done while a job you launched is running.** Wait for it and report the result, or hand its JOBS.yaml id to the PI explicitly.
- **On a failure** (OOM, crash):
  - read the failure_reason in JOBS.yaml;
  - fix the cause (smaller chunks, CPU, fewer workers);
  - re-submit.
  - After two failures of the same job, stop and report with the evidence.
- Never `pkill -f`; cancel with `orch cancel <id>`. Never edit a script while it runs (jobs run from a code snapshot of your committed branch HEAD anyway).

## Science rules
- **Never generate, load or look at test-split loci or test real-data pairs.**
- Never change metrics, labels, controls or tests to make results look better.
- Write tests with known answers. Never claim something works without running it, and report exact commands and numbers.
- **Register every output artifact you produce:** `scripts/orch/orch artifact register --name ... --version ... --path ... --job <job_id> --schema k=v ...`. Consumers read ARTIFACTS.yaml, never other agents' reports.
- **Never mutate a registered artifact in place.** Write a new version at a new path and mark the old one stale.
- **Reports** go in `reports/<task_id>.md`, written from finished results only. No placeholders or TODOs. If a number is missing, say why.
- Update your task with `scripts/orch/orch task set <task_id> --status running|done|failed` (plus `--notes`).

## Final message (≤ 400 words) ends with this footer

```
--- AGENT FOOTER ---
task id:
branch / commit:
artifacts registered (name@version, checksum):
jobs launched (ids, final status):
jobs still running (ids):
tests (command -> result):
decisions needed (each WITH your recommended default; the PI accepts it unless it objects):
```
