# Agent job infrastructure

The AI agents shared one node (64 CPU cores, 128 GB RAM, 1 GPU) through a small job queue with registries. It is
not needed to use the benchmark or the pipeline. It is included as part of the record of how the project was run
(REPORT.md §8, LESSONS.md). The code is in `scripts/orch/`, at the same place as in the original project.

| Path | What it is |
|---|---|
| `scripts/orch/orch`, `scripts/orch/orch.py` | CLI and scheduler daemon: `submit` a job with declared cores / RAM / VRAM / priority; each job runs from a `git archive` snapshot of the submitting branch's committed HEAD; the state lives in `JOBS.yaml`, `TASKS.yaml` and `ARTIFACTS.yaml` at the repository root (not shipped) |
| `scripts/orch/reconcile.py`, `scripts/reconcile` | consistency check of git, the registries, processes, the GPU and the reports |
| `scripts/orch/new_worktree.sh` | creates a per-agent git worktree and branch |
| `scripts/orch/sitecustomize_template.py` | per-job Python start-up hook |
| `scripts/orch/README.md` | full documentation, as the agents used it |
| `scripts/pi_livestate.py` | regenerates the live-state table at the top of RESEARCH_TREE.md from the registries |

Changes from the original: the hard-coded project path is replaced by `$ORCH_HOME` (default: the current
directory), and the interpreter is `$ORCH_PYTHON` (default `.venv/bin/python`, falling back to `python3`). The
code directories a job snapshot must have committed are `src/ scripts/ datasets/configs/ tests/ env/`. The
tests are in `tests/orch/`.
