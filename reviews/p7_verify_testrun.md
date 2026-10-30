# p7 independent verification of test run ab94c5a1 (frozen-v1, commit cab2599)

Verifier: p7-verify-testrun (independent; produced none of the run). Date 2026-10-28, 04:54–05:10 CET.
Bundle: `results/eval_test/test_frozen-v1_ab94c5a1/`, read only. Nothing under `results/eval_test/` was written,
and `scripts/eval_test.py` was not run.
Scripts: `reviews/p7_verify_testrun/{integrity.py, recompute.py, realarms.py}`. None of them imports `src/eval`.
`recompute.py` is written from DESIGN v4 §1, §4.3, §8.1 and §8.6, plus the weight definitions in `src/eval/README.md` items 5–12.
As the PI instructed, the verifier read the test truth files (`data/processed/testset_v1/test_frozen-v1_ab94c5a1/`).

## Verdict: VERIFIED WITH ISSUES

Every headline number recomputed independently matches the bundle to ≤ 5.1e-14. This covers the co-primary endpoints
(realistic, ideal and shifted), the C3 rates, the confusion matrices, the real arms and the runtimes.

None of the issues below changes a reported number. They are reporting caveats for REPORT.md:

1. **Failure share > 1% in at least one cell (DESIGN §4.3 blocks claims involving these methods):**
   - coloc_susie: 3 cells, max 2.36%;
   - famG: 2 cells, max 2.06%;
   - sharepro: 1 cell, 1.17%;
   - prop_coloc: 26 cells, max 7.87%;
   - colocboost: 2 cells, max 6.67% (8/120).

   The bundle's own `blocks_claims_gt_1pct` flags agree. There are no claims (D16), but any comparative statement about
   these methods must hold both with failures scored least-favourably and with failures excluded (§4.3).
2. **The shifted-arm power matching is fragile at quintile edges.** The pipeline's quantile levels are
   `np.linspace(0,1,6)[1:-1]`, and the third level is 0.6000000000000001. With n = 51 S ∪ D loci (cell L1swap|P1|r_p=0.0),
   (n−1)·q is an integer, so the bin edge falls on a data value. The float rounding then moves one locus to the other bin.
   - Literal quintiles (0.2, 0.4, 0.6, 0.8) shift the shifted-arm AUPRC by up to 0.0057 (coloc_abf 0.7901 vs 0.7958).
   - Left-closed bins shift it by up to 0.014 (strength_only; AUROC S-vs-D up to 0.022).
   - Realistic and ideal are unaffected: cells have 200 S ∪ D loci, so no edge lands on a data value.

   The design reports the shifted arm only in direction (§8.5), and its cells are tiny (51–63 S ∪ D loci). Its
   magnitudes should be quoted with that caveat. Reproducing with the pipeline's levels gives an exact match.
3. **`state/units.json` still lists `mvsusie__sim_main` with status "running".** A partial checkpoint
   (`checkpoints/...mvsusie__sim_main.rows.jsonl`, 427 rows) remains. The bundle's verify ignores it because mvsusie is
   not in the resume config. REPORT should state that mvSuSiE was NOT RUN (D21) and must not use those rows.
4. **The `summary.md` column "median s/locus" is the median of the per-set medians**, i.e. the mean of the sim_main and
   sim_shifted medians. It is not a per-locus median (coloc_abf: 0.0324 in summary.md vs 0.0233 on sim_main). Quote
   `runtime.json`, which is per set and verified below.
5. **Minor:**
   - Methods with null thresholds (lead_snp, prop_coloc, coloc_prop_test) have manifest t_S = t_D = 0.8, a harness default, where the frozen config has null. It has no effect: every `call` is NA and these methods have no P_S or P_D.
   - The strength_only `table` kwarg is stored as an absolute path, the same file the frozen config names relatively.
   - `bootstrap.n_boot_nan = 2000` counts replicates with any NaN column, from the structurally-NaN AUROC D-vs-W of
     the S-vs-D-only methods. CIs drop NaNs per column (`src/eval/metrics.py` ~l. 844), so this is not a failure.
   - The sharepro R2-recall CI is [0.000, 0.934]. 3.2% of R3b loci have P_S = 1.0 exactly, so in resamples the 5% threshold
     jumps above 1. The CI is uninformative.
   - The brief's "realistic grid of 34 cells" is the full main grid. Realistic is the 27 L1–L6 cells; ideal is the 3 L0 ρ = 0 cells.
   - Bootstrap units: 53 blocks (28 in sim_main), against the ~267 in DESIGN §3.

## Check 1: Integrity. PASS (with issue 1)

| item | result |
|---|---|
| Methods evaluated (44 run dirs) | coloc_abf, coloc_prop_test, coloc_susie, coloc_susie_mm, colocboost, colocboost_m1, famB_exactcoloc_gaussian_M50, famG_coloc_susie_gpu, lead_snp, prop_coloc, sharepro, strength_only = frozen list minus mvsusie. **PASS** |
| metric_config (t_S, t_D, recal_a_S, recal_a_D) vs `configs/test_methods_v1.yaml` | Identical (exact float equality) for every method with thresholds, in every manifest and in `operating_points_used.json`. Only the null-threshold harness default differs (issue 5). Method kwargs are identical. **PASS** |
| `test_methods_v1_d17cut1.yaml` | Parsed and compared: equals the frozen file minus mvsusie (methods and c3_eligible_baselines). Only free-text `*_source`, `design_ref` and `derived_from` differ. sha256: frozen 78585b02…, cut 469e8afb…. **PASS** |
| Git commit | All 44 manifests: `git.commit` = `config.commit` = every checkpoint segment's git = cab2599d5e…, clean tree, nonce ab94c5a1c097…. 3 runs span 2 segments (coloc_susie, colocboost, sharepro sim_main), both segments at cab2599. **PASS** |
| Access log | started → interrupted (signal 15) → resumed → completed. One commit, one nonce (ab94c5a1c097449db8d1d1f4bd18c168), no second nonce. **PASS** |
| Units | 44/44 evaluated units done. Predictions cover every expected locus (0 missing rows, 0 NaN s on ok rows) and counts equal the manifests. The 45th entry is mvsusie, still "running" (issue 3). **PASS** |
| Failures > 1% per cell | 5 methods (issue 1). Full table below. **FLAG** |

Status counts per method × set (failed / timeout / nonconverged; expected n):

| method | sim_main | sim_shifted | real R2 / R3b / R4b-in / R4b-ref |
|---|---|---|---|
| coloc_abf | 0/0/0 (11607) | 0/0/0 (1600) | all 0 |
| coloc_prop_test | 6/0/0 | 0/0/0 | – |
| coloc_susie | 31/0/2331 | 0/0/58 | R2 2/0/137; R3b 0/0/108; R4b-in 0/0/4; R4b-ref 0/0/94 |
| coloc_susie_mm | 0/0/1225 | 0/0/53 | nonconv 34 / 62 / 27 / 32 |
| colocboost (subsample) | 12/1/0 (4080) | 0/0/0 (387) | – |
| colocboost_m1 (subsample) | 0/0/0 (4080) | 0/0/0 (387) | – |
| famB_exactcoloc_gaussian_M50 | 0/0/0 | 0/0/0 | – |
| famG_coloc_susie_gpu | 17/0/2192 | 0/0/54 | – |
| lead_snp | 0/0/0 | 0/0/0 | – |
| prop_coloc | 369/0/0 | 25/0/0 | – |
| sharepro | 9/8/0 | 0/0/0 | all 0 |
| strength_only | 0/0/0 | 0/0/0 | all 0 |

Cells with failure share > 1% (failure = failed, timeout, missing or NaN s):
- **coloc_susie:** L4|P1|r_p=0.6 8/339; L4|P3|rho=0.8 6/344; L4|P3|rho=0.0 4/335.
- **famG:** L4|P1|r_p=0.6 7/339; L4|P3|rho=0.8 5/344.
- **sharepro:** L4|P1|r_p=0.0 4/343.
- **colocboost:** L2|P2|rho=0.0 8/120; L2|P2|rho=0.3 2/120.
- **prop_coloc:** 24 sim_main cells (1.2–7.9%) and both L2swap cells.

Non-converged rows are used and flagged, as §4.3 requires. coloc_susie is non-converged in 20% of sim_main loci.

## Check 2: Pipeline verification. PASS (partly re-run)

- `scripts/eval_test.py` has no read-only verify flag, so it was not run. The bundle's own `state/verify.json` and
  `bundle.json` report `ok: true` with no problems (benchmark, orphan_sim_runs, real_runs, units_not_done).
- `scripts/verify_run.py` on the run dirs (orch jobs `p7_verify_testrun_vr`, `_vr2`, then directly per real dir):
  - **all 20 real-arm runs: OK** (sha256s, coverage, counts, calls vs thresholds, metrics recomputed with tol 1e-9);
  - **sim runs: not runnable outside eval_test.py.** `src/eval/truth.py` raises `TestSplitAccessError` on the test truth
    (the DESIGN §7.3 test lock). The verifier did not bypass the lock; check 3 replaces this for the sim runs.
  - Job `_vr` failed only because the glob included `runs/LEDGER.tsv`.

## Check 3: Independent recomputation. PASS

Definitions used:
- **Failures:** least-favourable (S: s = 0, P_S = 0, P_D = 1; D: s = 1, P_S = 1, P_D = 0; W/G/U: P_S = P_D = 1).
- **Power matching:** per cell, 5 × 5 quintile bins of (|μ1|, |μ2|) over pooled S ∪ D. In bins holding both classes, each
  class gets mass (n_S + n_D)/2; bins holding one class get 0. Each cell is normalised to 1 and multiplied by the pooling weight.
- **Pooling weights:** equal over the levels present, then P1/P2/P3 = .5/.25/.25, equal within type.
- **AUPRC:** step-wise weighted average precision, with ties as one threshold.
- **AUROC D-vs-W:** D and W each get half the cell weight, uniform within class; ties count ½.

Weighted S prevalence is 0.500 in every setting. The maximum |Δ| vs the bundle over 7 methods (plus 4 on the subsample)
× 3 settings × {AUPRC S-vs-D, AUROC S-vs-D, AUROC D-vs-W} is **5.1e-14**. This uses the pipeline's quantile levels;
literal 0.2/0.4/0.6/0.8 levels change only the shifted arm (issue 2).

Headline numbers (least-favourable; reproduced = bundle to the digits shown):

| method | realistic AUPRC S-vs-D (repro / bundle) | realistic AUROC D-vs-W (repro / bundle) | ideal AUPRC | ideal AUROC D-vs-W | unweighted realistic AUPRC S∪D (repro only) |
|---|---|---|---|---|---|
| coloc_abf | 0.85988 / 0.85988 | 0.95678 / 0.95678 | 0.85605 / 0.85605 | 0.95496 / 0.95496 | 0.8632 |
| coloc_susie | 0.71713 / 0.71713 | 0.81471 / 0.81471 | 0.96559 / 0.96559 | 0.92157 / 0.92157 | 0.7304 |
| coloc_susie_mm | 0.89992 / 0.89992 | 0.90599 / 0.90599 | 0.96194 / 0.96194 | 0.92349 / 0.92349 | 0.9058 |
| strength_only (control) | 0.51875 / 0.51875 | 0.99200 / 0.99200 | 0.47065 / 0.47065 | 0.99498 / 0.99498 | 0.5223 |
| sharepro | 0.63475 / 0.63475 | 0.38557 / 0.38557 | 0.71552 / 0.71552 | 0.56481 / 0.56481 | 0.6417 |
| famB_exactcoloc_gaussian_M50 (descr.) | 0.82928 / 0.82928 | 0.94912 / 0.94912 | 0.96866 / 0.96866 | 0.99450 / 0.99450 | 0.8285 |
| famG_coloc_susie_gpu (descr.) | 0.71900 / 0.71900 | 0.81574 / 0.81574 | 0.96559 / 0.96559 | 0.92157 / 0.92157 | 0.7322 |
| colocboost (subsample bench.) | 0.67823 / 0.67823 | 0.74144 / 0.74144 | 0.83798 / 0.83798 | – | 0.6973 |
| colocboost_m1 (subsample bench.) | 0.78664 / 0.78664 | 0.93835 / 0.93835 | 0.82669 / 0.82669 | – | 0.7758 |

The bundle reports no unweighted pooled AUPRC. Its `all` entry is equal-weight-per-cell and power-matched, and that also
matches to < 4e-15 (e.g. coloc_abf 0.852500). For the shifted arm (e.g. coloc_abf 0.79580, coloc_susie 0.89016), see issue 2.

## Check 4: C3 three-way calls at the frozen t_S/t_D. PASS

These are realistic-weighted rates (W and D weights = D-vs-W weights) at the frozen t_S and t_D, with the E gate where reported.

| method | conf-S on W | conf-S on D | conf-D on W | recall S | recall D | bundle (W→S, W→D, recall S, recall D) |
|---|---|---|---|---|---|---|
| coloc_abf (t_S .8312, t_D .5803) | 0.0325 | 0.0690 | 0.02217 | 0.68058 | 0.87342 | 0.0325, 0.02217, 0.68058, 0.87342 |
| coloc_susie (t_S .8785, t_D .8036) | 0.0275 | 0.1860 | 0.02275 | 0.72733 | 0.71350 | 0.0275, 0.02275, 0.72733, 0.71350 |

- The unweighted three-way confusion matrices over all sim loci (main + shifted) match the bundle exactly in all
  15 cells for both methods. For example, coloc_abf D row: D 3072, S 237, insufficient 205.
- The bundle reports conf-S on D only through the confusion matrix (unweighted 237/3514). The weighted realistic value is
  computed here: 0.069 for coloc_abf, 0.186 for coloc_susie.
- At the validation-frozen thresholds, the W→S rate on test exceeds the 2.5% target for both methods (3.25% and 2.75%).
  This is a property of the result, not a pipeline error.
- The same agreement holds for coloc_susie_mm, sharepro, strength_only, famB and famG on the main benchmark, and for
  colocboost and colocboost_m1 on the subsample.

## Check 5: Real arms. PASS

t_R3b is the smallest threshold with an R3b confident-S rate ≤ 5%, placed just above an R3b score, with ties never split.
Failures are least-favourable.

| method | t_R3b | R3b conf-S at t_R3b | R3b conf-S at frozen t_S | R2 S recall at t_R3b (repro / bundle) | R4b call agreement (repro / bundle) |
|---|---|---|---|---|---|
| coloc_abf | 0.01491 | 4.93% | 0.0% | 0.9125 / 0.9125 | 1.000 / 1.000 |
| coloc_susie | 0.05210 | 4.93% | 1.13% | 0.8525 / 0.8525 | 0.7933 / 0.7933 |
| coloc_susie_mm | 0.00744 | 4.93% | 0.0% | 0.9175 / 0.9175 | 0.9267 / 0.9267 |
| sharepro | 0.99814 | 4.93% | 7.11% | 0.8300 / 0.8300 | 0.8467 / 0.8467 |
| strength_only | 0.44751 | 1.78% (tie block) | 78.1% | 0.13375 / 0.13375 | 1.000 / 1.000 |

## Check 6: Runtime. PASS

The median of `runtime_s` in the predictions equals the `runtime.json` `median_s` exactly (|Δ| = 0) for all 44 units.
Examples on sim_main:

| method | median s/locus |
|---|---|
| coloc_abf | 0.02331 |
| coloc_susie | 7.1475 |
| coloc_susie_mm | 12.6286 |
| sharepro | 1.0119 |

The manifests agree. These are harness wall times under the shared load of the test run (`timing: false`), not §7.7
timing-protocol numbers. For the summary.md aggregation, see issue 4.

## numpy "overflow encountered in exp" (metrics.py:172)

- **Source:** line 172 is inside `fit_logit_intercept`. This is the test-time *diagnostic* refit `recal_a_*_fitted_here`, not
  the application of the frozen recalibration. The job log shows 212 warnings.
- **Cause:** the SharePro a_S refit diverges (`recal_a_S_fitted_here` = 1538.1); colocboost and colocboost_m1 a_S fits are
  8.1 and 10.1. These are exactly the degenerate-intercept cases for which P3-30 says no recalibration is applied. The
  bundle reports no `ece_S_recal` / `brier_S_recal` for sharepro, colocboost or colocboost_m1, as it should.
- **Effect:** `exp` overflow gives q → 0, the correct limit. The warnings do not touch any co-primary, C3, real-arm or
  runtime number: all were reproduced above without any recalibration.
- **REPORT:** must not quote `recal_a_S_fitted_here` for these methods.

## Commands

```
source env/activate.sh
python reviews/p7_verify_testrun/integrity.py /cluster/scratch/mumarc/hackathon_eqtl_gwas
QBIN_QS=linspace python reviews/p7_verify_testrun/recompute.py /cluster/scratch/mumarc/hackathon_eqtl_gwas          # exact match
python reviews/p7_verify_testrun/recompute.py /cluster/scratch/mumarc/hackathon_eqtl_gwas                           # literal quintiles: shifted differs (issue 2)
QBIN_QS=linspace SUBSAMPLE=1 python reviews/p7_verify_testrun/recompute.py <root> colocboost colocboost_m1 coloc_abf coloc_susie
python reviews/p7_verify_testrun/realarms.py /cluster/scratch/mumarc/hackathon_eqtl_gwas
orch job p7_verify_testrun_vr2: python scripts/verify_run.py results/eval_test/test_frozen-v1_ab94c5a1/runs/*/   (real OK; sim blocked by test lock)
```
