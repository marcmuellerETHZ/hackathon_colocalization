# orch: job queue, task and artifact registries, reconcile

Everything is driven by one CLI, `scripts/orch/orch` (Python: `scripts/orch/orch.py`, run with the project `.venv`). The project consistency check is `scripts/reconcile`, which is the same as `orch reconcile`. Tests are in `tests/orch/` (toy jobs through the real daemon, isolated in `/tmp/claude-orchtest-*`).

## Where things live (ORCH_HOME = main checkout)

There is one source of truth: the **main checkout** (`/cluster/scratch/mumarc/hackathon_eqtl_gwas`). `orch` finds it from git's common dir, so it works the same from any worktree. You can override it with `$ORCH_HOME`, which the tests do.

| Path (in ORCH_HOME) | Content |
|---|---|
| `JOBS.yaml`, `TASKS.yaml`, `ARTIFACTS.yaml` | Registries. Every write goes through `orch` under `flock(.orch/registry.lock)` and is atomic (tmp + rename). Read them freely; never hand-edit `JOBS.yaml`. |
| `runs/<job_id>/code/` | `git archive` of the submitting tree's **committed HEAD**. The job runs here. |
| `runs/<job_id>/code/{data,results,logs}` | Symlinks to ORCH_HOME's `data/`, `results/` and `logs/`. Relative paths such as `data/processed/...` and `results/...` therefore read and write the **main checkout's** directories, never the snapshot. |
| `runs/<job_id>/out/` | Private scratch/output dir (`$ORCH_OUT`). |
| `runs/<job_id>/site/sitecustomize.py` | VRAM cap (see below). |
| `runs/<job_id>/{run.sh,exit_code}` | Generated wrapper and the job's exit code. |
| `logs/jobs/<job_id>.log`, `logs/jobs/scheduler.log` | Job stdout+stderr, and the daemon log. |
| `.orch/` | Scheduler pid, lock and heartbeat. |

**Job environment.** `run.sh` sources `env/activate.sh` (absolute paths, so `.venv`, the R env and the tools all work), then exports:
- `PYTHONPATH=runs/<id>/site:runs/<id>/code/src`
- `ORCH_JOB_ID`, `ORCH_HOME`, `ORCH_CODE`, `ORCH_OUT`, `ORCH_DATA` (= `$ORCH_HOME/data`), `ORCH_RESULTS`, `ORCH_CORES`, `ORCH_RAM_GB`, `ORCH_GPU`, `ORCH_COMMIT`
- for GPU jobs: `ORCH_VRAM_GB` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, unless already set
- for gpu=no jobs: `CUDA_VISIBLE_DEVICES=""`

`OMP_NUM_THREADS` stays at activate.sh's default of 1. If you declared N cores, set it in your command (`OMP_NUM_THREADS=$ORCH_CORES ...`) or size your worker pool from `$ORCH_CORES`.

**Outputs** must go to `data/...` or `results/...`, which resolve to the main checkout through the symlinks or `$ORCH_DATA`/`$ORCH_RESULTS`, or to `$ORCH_OUT`. Never write into the code snapshot. Later edits or commits never affect a queued or running job.

**VRAM cap.** For GPU jobs, the per-job `sitecustomize.py` hooks `import torch`. Right after CUDA initialises, it calls `torch.cuda.set_per_process_memory_fraction(vram_gb / total)`. An allocation beyond the declared VRAM then raises `torch.OutOfMemoryError`, the job is recorded `failed` with reason `CUDA OOM: ...`, and the queue moves on. The CUDA context (~0.5 GB) lies outside the cap, so declare peak tensor memory plus about 1 GB. Non-PyTorch GPU code (JAX, CuPy, R) is not capped; declare honestly.

## Jobs

```bash
source env/activate.sh
O=scripts/orch/orch
# commit your code first: submit REFUSES if the tree has uncommitted tracked changes (except logs/ data/
# results/ runs/ reports/ reviews/ lit/, root *.md, registries) or untracked files under src/ scripts/ datasets/configs/ tests/ env/
$O submit --id famC_c1_eval_8M --owner famC --task famC.C1 --gpu yes --vram-gb 10 --cores 8 --ram-gb 32 \
   --expected-min 40 [--timeout-min 120] [--cwd .] -- python -m famC.eval --ckpt results/famC/c1_long/model.pt
$O submit --id sim_valid_L0 --owner sim --gpu no --cores 32 --expected-min 60 -- 'OMP_NUM_THREADS=1 python scripts/x.py --n 32 > /dev/null'
$O status [--running] [--last 20] [ids...] [--yaml]
$O wait <id>...             # blocks; exit 1 if any job is not 'done' (prints failure reasons)
$O cancel <id>... [--reason txt]   # dequeue, or SIGTERM then SIGKILL of the job's process group
$O ack <id>... [--note txt]        # after handling a failed/lost job (reconcile stops flagging it)
```

- **Command.** A single argument after `--` is run as a bash string, so pipes and redirects work. Several arguments are shell-quoted.
- **Job ids** are unique forever. Use a new id for a re-run, e.g. `..._r2`.
- **Scheduling.**
  - FIFO with backfill. A head-of-queue job that has waited more than 30 min blocks backfilling.
  - Budgets: Σcores ≤ 64, ΣRAM ≤ 120 GB (default RAM is 2 GB/core).
  - GPU: one job at a time, unless Σ declared VRAM of the running GPU jobs plus the new one is < 22 GB.
- **Records.** `pid`, `pgid`, `commit`, `start`, `expected_end`, `end`, `exit_code`, `status` (queued/running/done/failed/killed/lost) and `failure_reason`.
  - Failure reasons: CUDA OOM, Killed (137) or host OOM, Python exception (last exception line), R error, timeout, nonzero exit.
  - `peak_rss_gb` (sum over the process group) and `peak_gpu_gb` (nvidia-smi per PID) are sampled every 5 s.
- **Not enforced.** Host RAM is recorded and flagged by reconcile, but nothing stops a job that exceeds it. Size your chunks.

## Scheduler daemon

```bash
scripts/orch/orch scheduler            # start (idempotent; setsid daemon; flock-guarded single instance)
scripts/orch/orch scheduler --status   # exit 0 if running
scripts/orch/orch scheduler --stop     # running jobs keep running and are re-adopted on restart
```

Start it **from the main checkout after merging**, so the daemon runs the merged code. Restarting is safe: jobs live in their own sessions. On restart the daemon re-adopts running jobs through `pid` and `runs/<id>/exit_code`.

## Tasks (TASKS.yaml)

```bash
$O task add famC.C1 --owner famC --depends-on sim.l0star_v2 --inputs l0star@v2 --outputs famC_c1_model@v1 \
    --done-criterion "pre-registered C1 eval at 8M written" --notes "..."
$O task set famC.C1 --status running|done|failed|blocked|abandoned [--add-job ID] [--append-notes txt] [--inputs ...]
$O task show [id] [--status running]
$O task ready [--all]       # planned/blocked tasks whose deps are done and inputs registered & not stale
```

`submit --task T` appends the job id to `T.jobs` automatically.

## Artifacts (ARTIFACTS.yaml)

```bash
$O artifact register --name valid_grid --version v1 --path data/processed/sets/valid_grid_v1 \
    --job <producing job id> | --commit <sha> \
    --schema n_loci=12000 p=500 'rho_grid=[0,0.3,0.6]' seed=20261027 [--schema-json '{...}'] \
    [--status ok|provisional] [--fast|--full] [--notes txt]
$O artifact show [name[@ver]]
$O artifact verify [name[@ver]] [--fast]
$O artifact stale name[@ver] --reason "built from uncommitted sim code"
$O artifact set-status name@ver ok|provisional|stale
```

- **Checksum.** The file sha256, or for a directory the sha256 over sorted `relpath\tsha256` lines.
  - Sets larger than 10 GB default to `fast` mode (size, mtime, and the first and last MB of each file); `checksum_mode` records which mode was used.
  - `fast_checksum` is always stored, so reconcile can re-check any artifact cheaply.
- **Producing commit.** With `--job`, the commit is taken from the job, whose tree was clean by construction. Otherwise it is HEAD, and `dirty_tree` is recorded and flagged.
- **Consumers read ARTIFACTS.yaml, never reports.**

## Reconcile

`scripts/reconcile [--no-checksum] [--no-du]` prints `FLAG` lines (exit 1) and `NOTE` lines (informational). It checks:
- **Git.** Dirty code in the main checkout or any worktree is a FLAG. A branch not merged into master is a NOTE.
- **Jobs.**
  - A `running` job whose PID is dead is marked `lost` (or finalised from `exit_code`) and FLAGged.
  - A failed or lost job is FLAGged until someone runs `orch ack`.
  - Processes from `runs/` not in any running job are FLAGged as orphans.
  - python/R processes in the project not launched via orch are a NOTE.
  - A dead scheduler with queued jobs is a FLAG.
- **GPU.** Compute processes that are not an orch GPU job are FLAGged, as is a job using more than its declared VRAM + 0.75 GB.
- **Artifacts.** Each of these is a FLAG:
  - the path is missing
  - the fast checksum does not match
  - the producing commit is not in master
  - the artifact was registered from a dirty tree
  - an unregistered entry over 1 GB sits under `data/processed`
- **Tasks.** Each of these is a FLAG: a `running` task with no live job, a running or done task whose input is stale or unregistered, a done task whose output is not `ok`, and a dependency on an unknown task.
- **Reports.** A `reports/*.md` file with placeholders (`__X__`, `*_COMMIT|LOCI|SIZE|TIME|NOTES`, TODO/TBD) is a FLAG.

## Worktree workflow (one agent = one worktree = one branch)

```bash
scripts/orch/new_worktree.sh famC_c1      # -> .worktrees/famC_c1 on new branch famC_c1 (from master)
cd .worktrees/famC_c1 && source ../../env/activate.sh
# edit, test (PYTHONPATH=src), commit NAMED paths only (git add src/famC/x.py; git commit), then:
../../scripts/orch/orch submit ...         # or scripts/orch/orch if the branch already contains orch
```

- **Shared directories.** The worktree gets the symlinks `data -> <main>/data` and `.venv -> <main>/.venv`, which are git-excluded and never tracked. `env/activate.sh` uses absolute paths.
- **What a job runs.** A job submitted from a worktree snapshots **that worktree's committed branch HEAD**. Its outputs still land in the main checkout's `data/` and `results/` through the snapshot symlinks.
- **Interactive outputs.** When you run interactively inside a worktree, `results/` is the worktree's own tracked copy. Write shared outputs via `$ORCH_HOME/results/...` or through a job.
- **Merging.** The PI merges branches into master. An artifact counts as clean only once its producing commit is in master; reconcile checks this.
- **Removing a worktree.** Delete the symlinks first, then remove it: `rm .worktrees/X/data .worktrees/X/.venv && git worktree remove .worktrees/X`.

## Agent footer (end every agent report with this, filled from finished results only)

```
--- AGENT FOOTER ---
task id:            <task id(s) in TASKS.yaml, final status>
branch / commit:    <branch> @ <full sha> (merged? no -> PI to merge)
artifacts:          <name@version  path  checksum_mode:sha256[:16]>  (one per line; `orch artifact show`)
jobs finished:      <job ids, status, failure_reason if any>
jobs still running: <JOBS.yaml ids + expected_end, or "none">  (owner hands these to: PI)
tasks updated:      <id: old -> new status>
tests:              <exact command> -> <N passed / M failed>
reconcile:          <scripts/reconcile exit code; flags caused by this agent, or "none">
decisions needed:   <numbered list, or "none">
```
