# Prompts given to the agent

The project was driven by the following user instructions, in this order.

| # | Instruction | Source |
|---|---|---|
| 1 | The initial project prompt | [`PROMPT.md`](../../PROMPT.md) (repository root), verbatim (the agent saved it on day 1, as the prompt asked) |
| 2 | Restart after milestone M3.5, adding operating rules (job queue, registries, supervision) | [`PROMPT_RESTART_M3.5.md`](PROMPT_RESTART_M3.5.md), verbatim. Its first line contains the placeholder "Remaining time budget: [X] hours" exactly as saved by the agent |
| 3 | "Remaining time budget: 22 hours" | not saved to a file; reconstructed from `RESEARCH_TREE.md` (decision D23: "horizon about 22 h, so the report is due by about 07:30 on 10-29") |
| 4 | "Go!": start phase 2, i.e. the follow-ups listed in REPORT.md §10 | not saved to a file; gist from `RESEARCH_TREE.md` D23 ("Phase 2 started on the user's 'Go' (follow-ups from REPORT §10)", 2026-10-28 09:40) |
| 5 | Write a manuscript of the project and assemble this publishable repository | not saved to a file; gist of the final instruction given to the PI agent, which delegated the repository assembly to a sub-agent |
| 6 | Restructure the repository to the layout of the other hackathon projects (github.com/evocellnet/ai-hackathon): the full research record at the root, `mvp/` (the recommended pipeline and the benchmark recompute) and `manuscript/` as separate folders, `datasets/`, `docs/`, `results/`, `reviews/` | not saved to a file; gist of the last instruction, which the PI agent delegated to a sub-agent (2026-10-29) |

Instructions 3–6 were given as chat messages and were not saved verbatim. The wording in this table is the gist
recorded by the agent, not a quotation.

Other short user messages recorded in the research tree, which did not change the project's scope:
- D3 (2026-10-26): do not touch the home directory; scratch purging is not a concern;
- D25 (2026-10-28 11:20): "remount", i.e. resume the PI session after it ended.
