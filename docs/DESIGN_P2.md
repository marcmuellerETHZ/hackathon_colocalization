# DESIGN_P2: Phase-2 pre-registration (addendum to DESIGN v4)

**Status: FROZEN 2026-10-28** (tag design-p2-frozen; sha256 in configs/FROZEN_P2_SHA256).
- Draft at 432bd49. Review round 1: reviews/p9_design_review.md, APPROVE WITH MUST-CHANGES. M1–M9 are adopted verbatim and S1–S4 adopted, with the reviewer's defaults accepted.
- Owner: PI.
- Motivation: REPORT.md §10, items 1–3, after the user's "Go" at 09:30.
- Horizon: about 22 h, with the report due by about 07:30 on 10-29.

## 0. Honesty statement and disclosure (M7)
Phase 2 was chosen after seeing the phase-1 test (v1) results. v1 evaluated all baselines plus B and G, the parents of B-mm and G-mm. v1 results chose the comparator (coloc_susie_mm), the B-mm and G-mm directions, and the OP2 constant (10%).

**v2 is held out at the locus level, not the region level.** Every claim reads "on fresh loci in the reused phase-1 test blocks". P2-C3 for coloc_susie_mm and the C5 replication are not independent of v1.

| v1 result (REPORT.md) | phase-2 choice it drove |
|---|---|
| coloc_susie_mm best realistic AUPRC 0.900; 16 CPU-s per locus | comparator for the candidacy bar and P2-C1; G-mm (GPU port of it) |
| B at fixed 0.8: false-S on D 0.067, W→S 0.000; killed at matched-W; weak at L4 (0.73) | B-mm (mismatch likelihood); OP2 |
| coloc_susie_mm false-S on D 0.095 at OP1 | the OP2 D-side constant 10% (chosen after v1) |
| G about 1.2× whole-machine | G-mm efficiency target ≥ 5× |
| C provisional M4 0.951 (stale simulator) | C full-scale C2 |

The report also gives the per-block correlation, between v1 and v2, of the baseline AUPRC differences.

Phase-1 verdicts are final and are not re-run (DESIGN §6.2). A stays killed. B (phase-1 configuration) stays killed, and is never a candidate.

## 1. Unchanged from DESIGN v4
The following are unchanged; no metric, label, control or test definition changes:
- oracle classes and power thresholds (§1);
- the simulator and the mismatch and overlap grid (§2);
- the splits (§3);
- the co-primary endpoints with their power matching, weights and bootstrap (§8.1, §8.3, §8.6);
- least-favourable failure scoring and the 1% failure block (§4.3);
- the strength-only control;
- the test lock and access log (§7.3), extended per testset in §6.

## 2. Phase-2 methods
- **C (NeuralColoc, full scale) (M4).**
  - The architecture is as in src/famC/README.md, with train-split windows only, L6/L6c held out and seed 20261030.
  - **Before training, a timing pilot fixes the largest size (from 75 k / 150 k / 300 k / 600 k records) whose cosine schedule completes in ≤ 4 h GPU.** C2 uses only that run's final-step checkpoint. If the run is killed, the final checkpoint of the largest completed size is used. Other sizes are reported and never substituted.
  - The LightGBM floor and the no-LD ablation are trained on the same records at the same size.
  - The train-region gap uses a pre-generated train-block set with the M4 design.
- **B-mm (mismatch-aware ExactColoc).**
  - B's exact ≤ 2-causal enumeration with the variance-inflated RSS likelihood of DESIGN §5 layer E: Var((R̂b)_k) adds Σ_j b_j²(1 − r_jk²)²/n_ref, plus η estimated by marginal likelihood. It uses the full-z_M likelihood path.
  - Gaussian EB prior, M = 50.
  - Correctness: with n_ref = ∞ and η = 1e-3 it reproduces phase-1 B's log BF to < 1e-8 on 200 loci; a brute-force full likelihood for p ≤ 12 agrees to < 1e-8.
- **G-mm (GPU mismatch-aware SuSiE).**
  - A GPU-batched port of the susieR 0.16.6 mismatch model as called by our coloc_susie_mm wrapper, with bf_bf pairing.
  - Correctness: max |ΔPIP| < 1e-3 and max |ΔPP.H4| < 1e-3 vs coloc_susie_mm on 100 validation L2/L4 loci in fp64. **It must reproduce susieR's failures and non-convergence, not silently succeed (S2).**
  - It is an engineering entry.

## 3. Operating points (M2)
- **OP1 (primary as DESIGN v4):** matched-W C3, fitted on validation.
- **OP2 (pre-registered amendment; constants frozen now, and no dry run may change them):**
  - t_S = the smallest threshold with weighted confident-S on W ≤ **2.5%** AND unweighted confident-S on D ≤ **10%**. The D share is taken over all power-matched D loci of the fit set.
  - t_D is computed as in OP1, after S precedence.
  - If no threshold satisfies both, t_S = 1.
  - The 10% value is disclosed as chosen after v1.
- **Fit sets:**
  - for kill tests: screen_v2 L2+L4 with §8.6 weights, for all methods, as in phase-1 B1;
  - for candidacy and test: M4 realistic cells.
- Rule (b) stays at OP1.
- An OP2 feasibility dry run on baseline validation outputs is allowed (S3). Its t_S values are recorded in the freeze note, and it cannot change the constants.

## 4. Ladders (validation only; every verdict is independently recomputed before use)
- **Paired SE (M3):** the SD of 2,000 paired cluster-bootstrap replicates over LD blocks (§8.3) of the difference, on loci scored by both methods, with failures scored least-favourably. It is used for the candidacy bar, C2 and the floor. Rules (a)–(d) and B1 keep the phase-1 locus-bootstrap SE, with the block SE reported alongside.
- **C, kill test C2** (DESIGN §5 C, unchanged), on the M4 main validation subsample:
  - realistic AUPRC ≥ the best of A and B (phase-1 M4 numbers), at equal or better recalibrated ECE;
  - train-region vs validation-region gap < 0.02;
  - it beats the floor and the no-LD ablation by more than 1 paired SE.

  Failing C2 makes C a fast pre-filter, not a candidate.
- **B-mm (M1):**
  - B-mm counts as a new method only if its threshold-free realistic AUPRC on screen L2+L4 exceeds phase-1 B's by more than 1 paired SE. Otherwise it is B, and stays killed.
  - Rules (a), (b) and (d) of DESIGN §6.2 apply unchanged.
  - Rule (c) and B1-usefulness are evaluated at OP2. **If t_S = 1, rule (c) and B1 FAIL.**
  - B1 at OP2 also requires recall-S ≥ the comparator's − 1 paired SE.
  - OP1 verdicts are recorded too. If B-mm passes only at OP2, every mention says "under the phase-2 amended operating point".
  - Phase-1 B at OP2 is descriptive only, and never a candidate.
  - B-mm also runs on M4.
- **G-mm:**
  - correctness (§2);
  - throughput under §7.7 (exclusive lock, 3 repeats) ≥ 5× coloc_susie_mm on 64 memory-limited cores.

  Failing either makes G-mm a report-only engineering note.
- **Candidacy bar:** M4 realistic AUPRC ≥ coloc_susie_mm's − 1 paired SE. coloc_susie_mm is measured on M4 in phase 2, which fixes P3-27.

## 5. Roles (M5) and baselines
- **Primary:** whichever of C and B-mm passes its ladder and the candidacy bar and has the highest M4 realistic AUPRC. A tie (difference < 1 paired SE) goes to the faster method (§7.7 throughput).
- **G-mm** may be a secondary candidate, for P2-C2 only.
- Claims apply to the primary only. DESIGN v4 C1–C4 are also evaluated for the primary as frozen, with C3 at OP1.
- **Baselines:**
  - coloc_susie_mm on full M4, with OP1 and OP2 fitted there;
  - every other tier-1 baseline keeps its frozen phase-1 OP1 (P3-30), plus OP2 fitted on M4;
  - **mvSuSiE is excluded from phase 2 up front** (cost).

## 6. Test set v2
- **Loci:** new simulated loci in the same 267 test-split LD blocks, generated by the frozen generator with seed name derive(20261101, 'testset_v2'). The cells, per-cell sizes and shifted arm are identical to configs/testset_v1.yaml.
- **Lock (M8):**
  - v2 loci are generated only inside eval_test.py, which is locked per testset_id; the v1 log is untouched;
  - the disjointness check against v1 runs on generation keys inside the run;
  - the v2 config and the lock code go into FROZEN_P2_SHA256.
- **Real arms:** R2, R3b and R4b are reused as a second look. They are descriptive only.
- **Procedure:** one primary candidate plus ≤ 2 secondary, named in a decision note; test_methods_v2.yaml; tag frozen-v2; a validation dry run; one run with a new nonce.
- **Claims (M6):**
  - **P2-C1:** the primary's realistic AUPRC exceeds every baseline listed in test_methods_v2.yaml, coloc_susie_mm included (paired block-BCa 95% CI of the difference excludes 0). If any baseline is dropped, the claim reads "tested against the listed subset".
  - **P2-C2:** throughput under §7.7 (exclusive lock, 3 repeats, not test-run runtimes) is ≥ 10× coloc-SuSiE and ≥ 5× coloc_susie_mm, each on 64 memory-limited cores. The one-sided 95% lower bound of ΔAUPRC vs coloc_susie_mm is > −0.01.
  - **P2-C3:** at OP2, the one-sided 95% upper BCa bounds are ≤ 5% (weighted W→S) and ≤ 20% (unweighted false-S on D). Point estimates are reported against 2.5% and 10%.
  - **Replication:** phase-1 C5 conclusions on fresh loci (descriptive). Also reported: the v2 bootstrap-cluster count (v1: 53) and per-L-level results (S4).

## 7. Schedule and fallbacks (M9, S1)
- **Backward schedule:**
  - ladders verified by 22:00;
  - selection by 22:30;
  - dry run 22:30–00:30;
  - freeze ≤ 01:00;
  - test run ≤ 04:30 (conservative projected end);
  - evaluation and verification ≤ 06:00;
  - report 07:30.
- **GPU slots:**
  - C training has priority 10 until done;
  - B-mm and G-mm development at priority 5, ≤ 8 GB;
  - G-mm's exclusive timing lock is taken in a gap after C training, or by 19:00 at the latest by pausing C between learning-curve sizes;
  - the critical path is queued as gated jobs (LESSONS 28).
- **Overrun check at submission:** the conservative projected end (from dry-run timings) plus 1.5 h for evaluation and verification must fall before 06:00. Otherwise apply in order:
  1. drop ColocBoost;
  2. drop the prop tests;
  3. drop SharePro;
  4. move coloc-SuSiE 0.14.2 to a nested subsample.

  The primary and coloc_susie_mm are never dropped.
- **No candidate at 22:00:** v2 is not generated and is kept unused.
- **Unfinished ladders:** a track that has not finished its verified ladder by 22:00 is reported on validation only.
- **Unverified results:** anything unverified by 07:30 is labelled unverified (LESSONS 27).
- **Hash-protected:** this file, configs/testset_v2.yaml, configs/test_methods_v2.yaml and the per-testset lock code (configs/FROZEN_P2_SHA256).
