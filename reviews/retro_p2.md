# Phase-2 retrospective (10-28, 09:30–18:30)

## 1. Process
**Went well**
- **Kill discipline.** No rule was relaxed to save a candidate.
  - C2's verdict code was committed before any number existed (f9a572c, 09:59).
  - Both relied-on kills were recomputed independently.
  - P4-1 (git 10:07) predates B-mm's first prediction (11:28), though baseline OP2 rates already existed (09:46).
- **The refused second amendment (P4-6) and the declined G-mm mid-run cancellation** were right. The GPU was otherwise idle after 16:08, and the slot gave the only §7.7 verdict (4.77×, a miss). The decline is unrecorded.
- **Restart (D25).** The gap was 10 min, the LESSONS 20 checks ran, and the pending decision was closed at 11:18.

**Went badly**
- **GPU allocation.** famC data generation held the GPU 09:56–13:56 while most of the 64 cores sat idle (AGENT_RULES: generation on CPU).
  - 24 of 61 GPU jobs were killed, mostly to resubmit at different VRAM caps.
  - famB-mm production waited 84 min (two resubmits), and phase-1 B on M4 waited 3.5 h.
- **P4-3 was written after the result it governs.** The entry itself records "At 10:15: 195/200 … 5 outliers non-converged". It exempts those fits from the frozen §2 1e-3 bar, yet is labelled "before … sensitivity result"; under §2 as frozen G-mm had already failed. P4-3a loosened it again.
- **Decision-log times run ahead of git.**
  - P4-3's text says 10:20 and cites a 10:15 result, but it was committed at 10:08.
  - P4-3a says 10:50 and was committed at 10:20.
- **Time.** The outcome was settled at 15:50, about 15 h before the deadline. The cheap open validity check, an audit of C's 30 global inputs (D27), was never scheduled.

## 2. Goalpost audit
| Rule / threshold | Class |
|---|---|
| DESIGN_P2 ladders, candidacy bar, B-mm gate, t_S = 1 → FAIL, recall guard, C checkpoint, G-mm 1e-3 / 5× | Pre-registered before phase-2 data (after v1; disclosed) |
| OP2 10% D-side constant | Pre-registered, chosen after v1, disclosed |
| C2 code; 600k size (pilot 11:21, before training 13:56); B-mm mm_iter / cross terms | Fixed before results |
| P4-1, P4-2, P4-4 definitions | Fixed before candidate results (after the allowed baseline OP2 dry run) |
| P4-3 carve-out | **Post-hoc, labelled as pre-result** |
| P4-3a | Post-hoc, disclosed |
| famC test tolerance 1e-6 → 1e-5 after a failure (b4c4c0b) | Post-hoc; disclosed only in p9.famC_C2.md |
| G-mm stream count tuned on the timed slice | Post-hoc, disclosed |
| 0.14.2 timing at n = 1 as the "pre-declared fallback" | Disclosed, but P4-7 names only a "reduced 60-min set" |

No fully undisclosed change was found.

**§11 flags**
- **Wrong bottom line.** "None … beat the best existing method" is false for C (0.964 vs 0.899, in-simulator). Name each method's failed rung instead.
- **"Every verdict independently recomputed" overstates.**
  - G-mm identity and throughput have no review.
  - coloc_susie_mm's 0.899 (P4-4: "independently") was self-checked by its producer, yet it anchors the candidacy bar and the "validated thresholds" deliverable.
- **G-mm** reports only amended-rule counts; under frozen §2, 4 fits exceed 1e-3.
- **B-mm understated.** The robust kill is the threshold-free candidacy miss (0.860 vs 0.899, about 4 SE), not the fragile rule (c).
- **Stale.**
  - The Summary still says C "could not be finished".
  - Summary item 5 and §10.1 call the two-sided-OP test "most promising". Phase 2 ran it: phase-1 B at OP2, L4, has false-S 0.109, above the 0.098 line.

**Is the 10% constant adequately disclosed?** In DESIGN_P2, yes. In §11, no. It should add:
- 10% sits just above the comparator's own false-S (v1 0.095, M4 0.097), so the comparator's OP2 equals its OP1, while 10% exactly sets B-mm's t_S (10.0%) and C's (9.97%);
- OP2 favoured B-mm (L4 false-S 0.265 → 0.109);
- a tighter constant could have lifted the rule (c) kill, but not the candidacy miss.

## 3. Communication
The user's objections were justified.
- **0.96.** Calling NeuralColoc's 0.96 promising after C2 switched to a criterion it won. Its no-LD version ties; the floor (0.959) recombines coloc/SuSiE outputs. The 0.96 measures fit to the simulator's prior, not a method gain.
- **Confusing summaries.**
  - The 17:54 live-state sentence says both "NO CANDIDATE … v2 NOT generated" and "freeze ≤ 01:00; test v2".
  - The top status block is still phase-1's (no phase-2 summary).
  - Pending checks are reported as "verified".

**Rule.** Every user-facing summary opens with one line per track: the pre-registered verdict, the rung and the margin. Every other number is tagged pre-registered, descriptive, post-hoc or unverified. A metric on which a failed method looks good comes only after its verdict, with why it does not count.

## 4. Proposed LESSONS
1. Take decision times from git or orch, never from memory. Evidence: P4-3 says 10:20 and cites 10:15, but was committed at 10:08.
2. Calibrate identity bars against the reference's own perturbation spread before freezing them. Evidence: the 1e-3 bar was unattainable for chaotic fits (R-vs-R spread up to 1.0), which led to three versions of the rule.
3. At dispatch, give each track a VRAM budget (sum < 22 GB) from the measured peak on the largest batch, and keep bulk generation on CPU. Evidence: 24 of 61 GPU jobs were killed, and famB-mm waited 84 min.
4. "Independent" means a different agent and different code. Evidence: P4-4's "independent" 0.899 was the producer's own check with src/eval.
