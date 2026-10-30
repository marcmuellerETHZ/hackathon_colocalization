# S2 Text. Agent process record

The full record is in the repository: the record files at the root, the process documents in `docs/`, and the reviews in `reviews/`. Key files:

| File | Content |
|---|---|
| `PROMPT.md` (root), `docs/prompts/PROMPT_RESTART_M3.5.md` | the human briefs, verbatim |
| `DESIGN.md` (v4, root), `docs/DESIGN_P2.md` | pre-registrations, frozen and hash-protected; `reviews/round*_*.md` and `reviews/p9_design_review.md` are the design reviews |
| `RESEARCH_TREE.md` | milestones, decision log D0–D29, status summaries |
| `docs/DECISIONS.md` | detailed decisions P3-1–P3-30 (phase 1) and P4-1–P4-8 (phase 2), with post-hoc items labelled |
| `LESSONS.md` | 33 process rules, each with the incident that produced it |
| `docs/reports/` | one report per task, written by the agent that did it |
| `reviews/` | independent verifications (`p5_verify_*`, `p7_verify_*`, `p9_verify_*`) and retrospectives (`retro_phase1`, `retro_M5`, `retro_final`, `retro_p2`) |
| `docs/STATE_M3.5.md` | the inventory taken at the restart after the overnight stall |

## Timeline (git times)

| Period | Wall time | Content |
|---|---|---|
| 10-26 13:19 → 10-27 ~10:15 | ~21 h | literature (3 agents), design v1–v4 (3 review rounds), data, simulator; overnight stall without supervision |
| 10-27 10:15 → 10-28 05:06 | ~19 h | restart with job queue; screening; freeze; single test run; report |
| 10-28 05:06 → 09:30 | ~4.4 h | idle, waiting for the human |
| 10-28 09:30 → 18:45 | ~9.3 h | phase 2 (validation only) |

## Incidents and the rules they produced (selection)

| Incident | Rule (LESSONS) |
|---|---|
| Five agents shared one GPU without a queue; out-of-memory deaths and an unnoticed stall | all GPU use goes through the queue; the PI never ends its turn while jobs run (2, 14) |
| Two simulation artifacts built from uncommitted code | every job runs from a committed snapshot (1, 5) |
| Messaging an agent inside a workflow forked a duplicate; later two PI sessions ran at once and broke shared code | only one PI session; check live agents before resuming (10, 20) |
| A corrupt chromosome-7 VCF was silently converted to 10% of its variants | verify downloads against upstream checksums; never discard stderr (8) |
| Wrapper inputs not in OLS z/se form made prop.coloc reject sharing on 70% of shared loci | check wrappers on documented examples and known-direction controls (7) |
| The matched-W kill rule killed the fastest well-ranking method | dry-run kill rules on a calibrated method; constrain both error directions (21) |
| The test run was submitted with an expected end already past the overrun line | check the drop rule at submission (26) |
| A report said "independently verified" before the verification existed | call a result verified only once its review file exists (27) |
| A rule was written after its result but labelled as pre-result; decision times were written from memory | take timestamps from git; calibrate identity bars before freezing (30, 31) |
| GPU data generation starved other tracks for hours | fixed GPU budgets per track; bulk generation on CPU (32) |
| Summaries presented a failed method's favourable number; the human objected | open every summary with per-track verdict, failed rung and margin; tag every other number (33) |

## Human interventions

These are listed in Methods ("Agent setup"). The human gave briefs, time budgets and pause instructions, asked status questions, and gave feedback on supervision gaps and unclear summaries. The human asked which follow-ups looked promising and asked for a comparison against mismatch-aware coloc-SuSiE, but did not choose the methods. The human did not write code or edit results.
