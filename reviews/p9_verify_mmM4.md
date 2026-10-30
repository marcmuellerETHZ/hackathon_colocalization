# p9_verify_mmM4: coloc_susie_mm M4 endpoints and OP1/OP2

**Verdict: CONFIRMED.** Every number below reproduces to full float precision.

## Inputs (validation only)
- The sha256 of `m4_valid_baselines@3` (5452b2e1…) and of `m4_subsample@1` (b1d02646…) match ARTIFACTS.yaml.
- Half runs `…977229` and `…9ee9eb` hold 6,085 loci each. They are disjoint and their union equals main.txt (12,170).
- Every truth row has split = valid; the script asserts this.

## Method
Script: `reviews/p9_verify_mmM4/verify_mm_m4.py`, output in `out.json`.
- It imports none of metrics.py, op2.py or baselines_p4.py.
- AP and AUROC come from sklearn.
- The §8.6 pooling and the 5×5 power matching are my own code.
- Failures are scored least-favourably.
- Thresholds come from a brute-force search for the smallest feasible value.

## Results
| quantity | claimed | recomputed |
|---|---|---|
| failures / non-converged | 0 / 1,410 | 0 / 1,410 |
| realistic AUPRC S-vs-D | 0.899 [0.887, 0.913] | 0.89885 (bit-equal to benchmark.json); percentile block bootstrap (500 replicates) [0.885, 0.911] |
| ideal AUPRC S-vs-D | 0.977 | 0.97683 |
| AUROC D-vs-W (realistic / ideal) | 0.900 / 0.928 | 0.90044 / 0.92801 |
| OP1 t_S / t_D | 0.84028 / 0.73226 | 0.8402818293 / 0.7322585039 (identical) |
| OP1 W→S / W→D | 0.0248 / 0.0249 | 0.02475 / 0.02492 |
| OP1 recall S / D | 0.782 / 0.803 | 0.7823 / 0.8034 |
| OP2 t_S from the D side alone | 0.82723 | 0.82723 |
| OP2 t_S / t_D | = OP1 | = OP1 (the W side binds) |
| OP2 false-S on matched D | 0.0975 (n = 2,678) | 0.09746 (n = 2,678; n_W = 2,700) |

Unweighted: realistic AUPRC 0.902, AUROC D-vs-W 0.904 (same sign and size).

## Wrapper unchanged
- Jobs ran at df23941; run manifests record e49cd33 (main HEAD then).
- `git diff --stat <c> master -- src/baselines` is empty for both commits.
- The last change to src/baselines is f27cf68 (10-27 11:57), before both runs.

## Note
The reported "failures excluded" 0.896 (with 0 own failures) is consistent with dropping loci failed by any benchmarked method; not recomputed.
