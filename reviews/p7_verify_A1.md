# p7 verify A1: independent recomputation of Family A kill test A1

**Verdict: CONFIRMED.** A1 FAILS. Family A is below coloc-SuSiE by a margin far larger than 1 SE.

## Inputs (sha256 checked)
- A predictions `results/famA/killtest_inputs/a1_primary_v1/predictions/famA_joint_susie_rho_eta_ms_ptld.parquet`: `dac7451a…` (matches).
- Baselines `results/baselines_screen_v2/collected_L0P1_v1b/predictions.parquet`: `54239e5e…` (matches).
- Truth: `data/processed/sets/screen_v2/L0_P1_r_p=0.0.truth.parquet` (screen_v2@2, split=valid). It has 500 S, 1,000 D and 500 W loci; S vs D uses 1,500 of them. No test data was touched.

## Method
The script is `reviews/p7_verify_A1/verify_a1.py`, written from DESIGN v4 §2.4, §4.3, §8.1 and §8.6 only. It imports neither `src/eval/metrics.py` nor `src/famA`.
- **Score:** `s`, outer-joined on the expected loci.
- **Failures** (failed, timeout or NaN `s`) are scored least-favourably: s = 0 for S, s = 1 for D. Non-converged loci are used.
- **AUPRC:** weighted step-wise average precision, with tied scores treated as one threshold.
- **Power matching:** 5×5 bins of class-determining |μ1|, |μ2| at quintiles of the pooled S∪D loci. Each bin's target mass is n_S + n_D, and bins without both classes get weight 0. S and D weights are normalised to prevalence 0.5.
- **Bootstrap:** paired locus bootstrap of A minus coloc_susie, 2,000 replicates, seed 20261102. Weights are recomputed in each replicate.

Output: `reviews/p7_verify_A1/output.json`.

## Results (power-matched, pooled target)

| | mine | killtest.json |
|---|---|---|
| A | 0.893247 | 0.893247 |
| coloc_susie | 0.978352 | 0.978352 |
| sharepro | 0.733934 | 0.733934 |
| mvsusie | 0.642432 | 0.642432 |
| Δ (A − coloc_susie) | −0.08511 | −0.08511 |
| paired SE | 0.0101 | 0.0104 |
| 95% CI (percentile) | [−0.1046, −0.0664] | [−0.1052, −0.0643] |
| verdict | FAIL | FAIL |

Point AUPRCs match to every printed digit. SE and CI differ only by bootstrap Monte-Carlo noise (different seed). A needed ≥ 0.968 (best − 1 SE); it is about 8 SE short.

## Robustness
- **Other bin targets** (n_S, n_D, min): A scores 0.897, 0.891 and 0.897, and coloc_susie scores 0.980, 0.978 and 0.980. The gap stays at about −0.08 to −0.09.
- **Unweighted:** A 0.806, coloc_susie 0.960. Δ = −0.154, SE 0.018, CI [−0.188, −0.120]. Same sign, and a larger gap.
- **Failures:** the single mvsusie failure is a W locus, so it does not enter S vs D. The failures-excluded results are identical to the ones above. A, coloc_susie and sharepro have no failures and no NaN scores.
- **Baseline ranking:** coloc-SuSiE is the best of {coloc-SuSiE, SharePro, mvSuSiE-RSS}, as claimed.

Command: `python -I reviews/p7_verify_A1/verify_a1.py <A.parquet> <baselines.parquet> <truth.parquet>`, about 5 s on CPU.
