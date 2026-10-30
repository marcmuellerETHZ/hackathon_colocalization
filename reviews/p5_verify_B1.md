# p5.verify_B1 — independent verification of the Family B kill verdict (B1 + DESIGN §6.2)

Verifier: independent agent (did not write Family B or `scripts/famB_eval/killtest_B1.py`). Date: 2026-10-27.

## Verification setup

| Item | Value |
|---|---|
| Code | `scripts/verify/p5_verify_B1.py` (own numpy/pandas implementation). `src/eval/metrics.py` is imported only in a final cross-check block. |
| Job | `p5_verify_B1` (orch, CPU, priority 11, commit `8fcb22e`), status done |
| Output | `results/verify_B1/verify_B1.json` |
| Inputs | famB_screen_v2@1 `results/famB/screen_v2/M50/predictions.parquet` (method `exactcoloc_gaussian_M50`); baselines_screen_v2@1 `results/baselines_screen_v2/collected_v1/predictions.parquet`; truth from `eval.truth.read_truth` on `data/processed/sets/screen_v2/*.truth.parquet` (12,000 validation loci, 6 cells × {S, D r²<0.5, D r² 0.5–0.8, W} × 500) |

**Definitions used.** These are mine, written from DESIGN v4 before I read the agent's code paths.

- **Failure.** A failure is a status of failed or timeout, or a NaN in s, P_S or P_D. Non-converged runs are used as they are.
- **Least-favourable scoring** (DESIGN §4.3):
  - D loci get P_S = 1, P_D = 0 and s = 1.
  - S loci get P_S = 0 and s = 0.
  - W loci get P_S = P_D = 1, which counts as a confident call.
- **Excluded scoring.** Failed loci are dropped from both sides of every paired comparison.
- **Matched-W t_S.** This is the smallest t with mean(P_S ≥ t) ≤ 2.5% on W. In practice it is the value just above the last W value that has to be excluded. Rates are unweighted.
- **False-S rate.** The share of loci in the stratum with P_S ≥ t_S (S has precedence, so t_D is irrelevant). Each cell gets equal weight.
- **Paired comparisons** use only the loci that both methods were run on. This matters for the subsampled methods colocboost, colocboost_m1 and mvSuSiE.
- **Bootstrap.** Paired, 2,000 replicates, by locus and also by LD block (`block_id`).

## Pre-registered text (quoted)

- **§5 B, Kill-test B1:** "*Usefulness:* on the D r² 0.3–0.8 and M strata (L0, L2), the false-S rate is below coloc-SuSiE's by more than 1 paired SE. Otherwise B is kept as gold standard and ceiling only."
- **§6.2:**
  - "(a) AUPRC(S vs D) at L0, ρ = 0 is below the best tier-1 baseline by more than 1 paired SE";
  - "(b) confident calls on W exceed coloc.abf's (p12 = 5e-6) by > 2× **and** by > 2 percentage points, with one-sided bootstrap p < 0.05";
  - "(c) false-S at L4 or ρ = 0.5 is more than 2× (and +2 pp) that of the best baseline";
  - "(d) runtime is more than 10× coloc-SuSiE without an accuracy gain."
- **§0 C3:** "For each method, on validation: t_S is the smallest threshold with confident-S rate on W ≤ 2.5%; then t_D …".
- **§1:** "Thresholds t_S and t_D are set on validation by the matched-W rule (§8.1). The fixed 0.8 thresholds are also reported."

## 1. Matched-W t_S, and which loci it is fitted on

The DESIGN says only "on validation" and names no subset of W loci. The agent fitted t_S on the W loci of L2 + L4, which is the "realistic" setting. It also used the §8.6 cell weights.

I checked three fitting sets, all with least-favourable scoring:

| t_S | W of L2+L4 (agent's set) | all W (L0+L2+L4, the literal reading) | W of L0 only |
|---|---|---|---|
| **B** | **0.0103** (agent's weighted version: 0.0134) | 0.0157 | 0.0283 |
| coloc.abf (W 0.15, p12 5e-6) | 0.788 | 0.798 | 0.801 |
| coloc-SuSiE | 0.846 (agent 0.862) | 0.825 | 0.811 |
| coloc-SuSiE-mm | 0.818 | 0.818 | 0.813 |
| SharePro | 0.647 | 0.658 | 0.722 |
| colocboost | just above 0 | 0.469 | 0.507 |
| colocboost_m1 | 0.566 | 0.566 | 0.528 |
| mvSuSiE | > 1 (never calls S) | > 1 | > 1 |

Why B's t_S is so low:
- B's P_S on W loci has median 0, 95th percentile 0.0014, 97.5th percentile 0.0157 and 99th percentile 0.061.
- Fitting t_S per cell gives values from 0.0013 to 0.029.
- So B's matched-W t_S is below 0.03 under every fitting choice.

## 2. False-S rates (least-favourable scoring)

B1 strata are at L0 + L2 (all four cells). Dmid is a class-D locus with r² in [0.3, 0.8). Mx is locus_type M with oracle class other than S.

### Matched-W thresholds (agent's fitting set)

| Stratum | B | coloc-SuSiE | coloc-SuSiE-mm | coloc.abf |
|---|---|---|---|---|
| **B1 union Dmid ∪ Mx (L0+L2)** | **0.226** | 0.135 | 0.053 | 0.048 |
| Dmid L0 / L2 | 0.240 / 0.363 | 0.056 / 0.297 | 0.062 / 0.073 | 0.069 / 0.060 |
| Mx L0 / L2 | 0.072 / 0.043 | 0.031 / 0.069 | 0.028 / 0.013 | 0.008 / 0.006 |
| (c) L4, D loci | **0.210** | 0.141 | 0.096 | 0.049 |
| (c) ρ = 0.5 (all P3 cells), D loci | **0.168** | 0.121 | 0.061 | 0.042 |

### Fixed 0.8

| Stratum | B | coloc-SuSiE | coloc-SuSiE-mm | coloc.abf | colocboost_m1 |
|---|---|---|---|---|---|
| B1 union | 0.034 | 0.144 | 0.057 | 0.044 | 0.063 |
| (c) L4 D | 0.066 | 0.145 | 0.100 | 0.048 | 0.040 (B 0.055 on its subsample) |
| (c) ρ = 0.5 D | 0.037 | 0.127 | 0.065 | 0.041 | 0.043 (B 0.030 on its subsample) |

### B1 contrast (coloc-SuSiE − B, ± paired SE)

| Thresholds | Locus SE | Block SE | Result |
|---|---|---|---|
| Matched-W, L2+L4 fit | −0.092 ± 0.008 | ± 0.008 | **FAIL** |
| Matched-W, all-W fit | −0.071 ± 0.008 | — | **FAIL** |
| Matched-W, L0-W fit | −0.042 ± 0.007 | — | **FAIL** |
| Fixed 0.8 | +0.110 ± 0.006 | ± 0.006 | pass |

The FAIL is unchanged:
- with Mx restricted to class D (−0.096);
- with excluded scoring (−0.079);
- with the "observed-value" threshold convention;
- against coloc-SuSiE-mm (−0.173).

### Rule (c), matched-W

| Stratum | B | Comparator | Comparator false-S | Result |
|---|---|---|---|---|
| L4 | 0.210 | best-AUPRC baseline: coloc.abf (my per-stratum AUPRC 0.879) | 0.049 | **KILL** |
| L4 | 0.215 (paired subsample) | lowest non-degenerate baseline: colocboost_m1 | 0.045 | **KILL** |
| ρ = 0.5 | 0.168 | best-AUPRC baseline: coloc-SuSiE-mm (0.922) | 0.061 | **KILL** |
| ρ = 0.5 | — | lowest non-degenerate baseline: coloc.abf | 0.042 | **KILL** |

- The kill also holds with the all-W fit, with excluded scoring, and on D ∪ W.
- It does **not** hold against coloc-SuSiE (standard), SharePro or colocboost. Under the matched-W rule these have high false-S themselves. coloc-SuSiE's L4 claims are also blocked by P3-29.
- On W loci alone, B's matched-W false-S (≈ 0.02) is not inflated.

## 3. Rule (a) and rule (d)

**Rule (a).** Ideal cell L0|P1|r_p=0, S vs D, with my own 5×5 quantile power matching.

| Method | AUPRC |
|---|---|
| B | 0.9759 |
| coloc-SuSiE (best tier-1) | 0.9777 |
| coloc-SuSiE-mm | 0.9711 |
| coloc.abf | 0.871 |
| strength-only control | 0.486 |

- Δ = −0.0018, paired locus SE 0.0049, 95% CI [−0.012, 0.008]. Not < −1 SE, so **not killed**.
- Cross-check with `metrics.auprc_sd`: B 0.9765, coloc-SuSiE 0.9784.

**Rule (b)**, at 0.8 with E-gates. On all 3,000 W loci, B makes confident calls on 1.17% vs coloc.abf 3.87%. Without B's E-gate the rate is 1.3%. **No kill.**

**Rule (d)**, from the predictions' `runtime_s`.

| | B | coloc-SuSiE |
|---|---|---|
| Mean (s/locus) | 0.126 | 23.54 |
| Median (s/locus) | 0.067 | 7.60 |

The ratio is 0.005 on means and 0.009 on medians, so **no kill**.

## 4. Do the verdicts follow from the pre-registered definitions?

**Yes, under the accepted reading** (matched-W primary, fixed 0.8 as sensitivity, rule (b) at 0.8).

- **No implementation detail flips a verdict.** I checked strata membership, failure scoring, the threshold-fitting set, the threshold convention, the locus vs block bootstrap, and weighted vs unweighted thresholds. The agent's point estimates match mine within 0.01.
- **The mvSuSiE threshold > 1 does not drive the rule-(c) kill.** It persists against the non-degenerate comparators listed above.
- **The only thing that flips B1 and (c) is the operating point.** At fixed 0.8, B1 passes and (c) does not kill.
- **Interpretation for the PI.** Under the matched-W rule, B's very low P_S on W loci forces t_S down to about 0.01–0.03. At that threshold its moderate P_S on D loci becomes false-S. This is a real property of B's P_S scale under the pre-registered C3 operating point, not a coding error.

| Rule | Verdict |
|---|---|
| B1 usefulness FAIL | **CONFIRMED** |
| (a) pass | **CONFIRMED** |
| (b) pass | **CONFIRMED** |
| (c) KILL | **CONFIRMED** (requires "best baseline" ≠ coloc-SuSiE standard; holds for lowest-non-degenerate and best-AUPRC readings) |
| (d) pass | **CONFIRMED** |

**Command:**

```
orch submit --id p5_verify_B1 --priority 11 ... -- PYTHONPATH=src python scripts/verify/p5_verify_B1.py --out results/verify_B1/verify_B1.json
```
