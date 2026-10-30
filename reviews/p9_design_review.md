# p9 design review: DESIGN_P2.md (statistician + external PI)

DESIGN_P2.md @ 8914ce3 vs DESIGN v4, REPORT, LESSONS, D13–D23.

**Verdict: APPROVE WITH MUST-CHANGES.**

## Answers
1. **OP2 in rule (c).** Legitimate as a prospective, all-methods amendment. As drafted, though, it is a partial back door to B:
   - if the mismatch term is immaterial, B-mm is just B at OP2;
   - t_S = 1 gives false-S = 0, so rule (c) passes trivially;
   - nothing guards recall;
   - the 10% bound was set after seeing v1 test rates (coloc_susie_mm 0.095, B 0.067 at 0.8).
2. **Test v2** holds out loci, not LD regions: directions and the OP2 constant came from v1 results in the same blocks.
3. **Ambiguous:** OP2 fit set, checkpoint, SE type, "within a CI", P2-C2 comparator, G-mm's role.
4. **Schedule unrealistic:** v1 took 5 h without mvSuSiE; C's plan needs 10–12 h wall. Fallback covers only a freeze slip.

## MUST-CHANGE (replacement wording)

**M1. B-mm guards (§4).**
> "B-mm counts as a new method only if its threshold-free realistic AUPRC on screen L2+L4 exceeds phase-1 B's by > 1 paired SE; otherwise it is B and stays killed. If t_S = 1, rule (c) and B1 FAIL. B1 at OP2 also requires recall-S ≥ comparator − 1 paired SE. Phase-1 B at OP2 is descriptive only and never a candidate. If B-mm passes only at OP2 (OP1 verdicts are recorded too), every mention says 'under the phase-2 amended operating point'."

**M2. Freeze OP2 (§3).**
> "2.5% and 10% are frozen now. No dry run may change them, and the 10% value is disclosed as chosen after v1. Fit sets:
> - kill tests: screen_v2 L2+L4 with §8.6 weights, for all methods, as in phase-1 B1;
> - candidacy and test: M4 realistic cells.
>
> W rates are weighted. The D rate is the unweighted share over all power-matched D loci. t_D is computed as in OP1, after S precedence. Rule (b) stays at OP1."

**M3. SE (§4).**
> "Paired SE is the SD of 2,000 paired cluster-bootstrap replicates over LD blocks (§8.3) of the difference, on the loci scored by both methods, with failures least-favourable. It is used for the candidacy bar, C2 and the floor. Rules (a)–(d) and B1 keep the phase-1 locus-bootstrap SE, with the block SE reported alongside."

**M4. C checkpoint and floor (§2).**
> "Before training, a timing pilot fixes the largest size whose cosine schedule completes in ≤ 4 h GPU. C2 uses only that run's final-step checkpoint. If the run is killed, the final checkpoint of the largest completed size is used. Other sizes are reported and never substituted. The floor and the no-LD ablation are trained on the same records at the same size. The train-region gap uses a pre-generated train-block set with the M4 design."

**M5. Roles (§4, §6).**
> "G-mm may be a secondary candidate for P2-C2 only. The primary is whichever of C and B-mm passes and has the highest M4 realistic AUPRC; a tie (< 1 paired SE) goes to the faster method. B-mm also runs on M4. Claims apply to the primary only. DESIGN v4 C1–C4 are also evaluated for it as frozen (C3 at OP1)."

**M6. Claims (§6).**
- **P2-C2:**
  > "Throughput under §7.7 (exclusive lock, 3 repeats, not test-run runtimes) is ≥ 10× coloc-SuSiE and ≥ 5× coloc_susie_mm, each on 64 memory-limited cores. The one-sided 95% lower bound of ΔAUPRC vs coloc_susie_mm is > −0.01."
- **P2-C3:**
  > "The one-sided 95% upper BCa bounds are ≤ 5% (weighted W→S) and ≤ 20% (unweighted false-S on D). Point estimates are reported against 2.5% and 10%."
- **P2-C1:**
  > "Baselines as listed in test_methods_v2.yaml. If any is dropped, the claim is reported as 'tested against the listed subset'."

**M7. Disclosure (§6).**
> "v1 evaluated all baselines plus B and G (the parents of B-mm and G-mm). v1 results chose the comparator, B-mm, G-mm and the OP2 constant. v2 is held out at the locus level, not the region level. Claims read 'fresh loci in the reused phase-1 test blocks'. P2-C3 for coloc_susie_mm and the C5 replication are not independent."

Add a v1-result → phase-2-choice table and the per-block v1/v2 correlation of baseline differences.

**M8. Lock (§6).**
> "v2 loci are generated only inside eval_test.py, which is locked per testset_id; the v1 log is untouched. The disjointness check runs on generation keys inside the run. The v2 config and the lock code go into FROZEN_SHA256."

**M9. Schedule and fallbacks (§7).**
> "At submission, the conservative projected end (from dry-run timings) plus 1.5 h for evaluation and verification must fall before 06:00. Otherwise apply, in order: exclude mvSuSiE up front, then drop ColocBoost, then the prop tests, then SharePro, then move coloc-SuSiE 0.14.2 to a nested subsample. The primary and coloc_susie_mm are never dropped.
> - If there is no candidate at 22:00, v2 is not generated and is kept unused.
> - A track that has not finished its verified ladder by 22:00 is reported on validation only.
> - Anything unverified by 07:30 is labelled unverified (LESSONS 27)."

## SHOULD-CHANGE
- **S1.** Backward schedule with per-track GPU slots; G-mm's exclusive timing lock conflicts with C training and coloc_susie_mm on M4. Queue the critical path as gated jobs (LESSONS 28).
- **S2.** G-mm must reproduce susieR's failures, not silently succeed.
- **S3.** Dry-run OP2 on baseline validation outputs for feasibility only (LESSONS 21); record t_S values in the freeze note.
- **S4.** Report v2 bootstrap-cluster count (v1: 53) and per-L-level results (user's LD-region split).
