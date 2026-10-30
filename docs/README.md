# Process documents

Part of the agents' own record of the project, copied unchanged from the original project root on 2026-10-29.
All of it was written by AI agents. The main record files (REPORT, RESEARCH_TREE, DESIGN, DATA, LESSONS,
LITERATURE) are at the repository root; reviews are in `../reviews/`.

| File | What it is |
|---|---|
| `DECISIONS.md` | binding post-freeze decisions P3-x / P4-x |
| `DESIGN_P2.md` | pre-registered phase-2 design |
| `AGENT_RULES.md` | rules every agent was given |
| `INTERFACES.md` | data and method interfaces (locus HDF5, truth, predictions) |
| `STATE_M3.5.md` | state snapshot at the M3.5 restart |
| `OPEN_QUESTIONS.md` | the questions the PI agent left for the user (non-blocking) |
| `prompts/` | the user instructions that drove the project, after the first (`../PROMPT.md`) |
| `lit/` | literature notes behind `../LITERATURE.md` |
| `reports/` | each agent task's report |
| `agent_orchestration.md` | the job queue and registries the agents used (code in `../scripts/orch/`) |
| `environment/` | the Python / R / CUDA environment: lock files, pinned R packages, install notes |
| `agent_logs/` | the agents' registries `JOBS.yaml` (every queued job), `TASKS.yaml` (every task) and `ARTIFACTS.yaml` (every registered output, with checksums), and the original `logs/` folder (job, simulation, data, real-arm build and method logs), copied unchanged on 2026-10-30 |

Notes:
- Paths inside the record files refer to the original project layout (`/cluster/scratch/...`, `configs/`,
  `splits/`, `results/...`, `tests/...`). They are kept verbatim. In this repository `configs/` and `splits/` are
  under `datasets/`; the test-run and phase-2 outputs that were released are under `results/` (see
  `results/README.md` for the mapping).
- Four review files were not committed to the original project's git history but were present in its working
  tree: `reviews/retro_M5.md`, `reviews/retro_phase1.md`, `reviews/round3_external_pi.md` and
  `reviews/round3_statistics.md`.
- No record file over 2 MB was copied (none existed). A scan for e-mail addresses, credentials and
  home-directory paths found none needing redaction.
- `agent_logs/` was scanned for e-mail addresses, tokens and other credentials before release; none were found, so
  nothing was redacted. Absolute `/cluster/...` paths are kept. The real-arm logs (`agent_logs/logs/real/`) hold
  counts, checks and download progress only, no summary statistics.
