# p9_verify_Bmm: independent check of the B-mm phase-2 ladder (famB_mm_ladder@1)

**Verdict: CONFIRMED.** Rule (c) at OP2, L4, kills B-mm under both P4-1 readings (ii) and (iii). B1 at OP2 fails. The gate passes.

**Method.** Own numpy/pandas script (no `eval.metrics`, `eval.op2` or famB imports), screen_v2 validation data only:

`python reviews/p9_verify_Bmm/verify_Bmm.py --root <project> --out reviews/p9_verify_Bmm/verify_Bmm.json` (8 s, CPU)

Definitions: least-favourable failure scoring; §8.6 weights (L2+L4; P1 2/3, P3 1/3 per level); own 5×5 power matching; rule (c) false-S over all stratum D loci, cells equal-weighted, paired on the comparator's loci.

## Inputs

| Input | sha256 vs registry |
|---|---|
| famB_mm_screen_v2@1, famB_screen_v2@1, baselines_screen_v2@1, op2_screen_v2_baselines@1, famB_mm_ladder@1 | all match (orch full manifest) |
| six screen_v2 `*.truth.parquet` | all match `data/processed/sets/SHA256SUMS` |

## Numbers (claimed → mine)

| Quantity | Ladder | Mine |
|---|---|---|
| Gate: realistic AUPRC, B-mm / B | 0.8614 / 0.8216 | 0.8614 / 0.8216 |
| Gate: Δ, block SE | +0.0398, 0.0057 | +0.0398, block 0.0060 / locus 0.0056 (140 blocks) → **PASS** |
| B-mm OP2 t_S (t_D side) | 0.5084 | 0.5084: W→S 0.13%, D share 10.0%, n_pmD 4000 |
| Baseline OP2 t_S (abf/susie/susie_mm/cb/cb_m1) | 0.778/0.9999/0.820/0.928/0.586 | identical |
| **L4 (ii)**: best stratum AUPRC | coloc.abf (0.880) | coloc.abf (0.880; next coloc_susie_mm 0.851) |
| L4 (ii): false-S B-mm vs comparator | 0.109 vs 0.049 | 0.109 vs 0.049 |
| L4 (ii): kill line max(2c, c+0.02) | 0.098 | 0.098 → **KILL** |
| **L4 (iii)**: comparator | colocboost_m1 | colocboost_m1 (others lower make 0 S calls) |
| L4 (iii): B-mm on paired loci | 0.145 vs 0.045 | 0.145 vs 0.045 (line 0.090) → **KILL** |
| ρ = 0.5 (ii): coloc_susie_mm | 0.063 vs 0.061 | same → no kill |
| ρ = 0.5 (iii): coloc.abf | 0.063 vs 0.042 | same (line 0.085) → no kill |
| B1 at OP2: Dmid ∪ Mx, L0+L2 | 0.0679 vs 0.0663 | 0.0679 vs 0.0663 (n 3245) → **FAIL** |

All point estimates agree to floating-point precision.

## Fragility

Applied as written, §6.2(c) is a point-estimate rule ("> 2× and +2 pp"); unlike (b) it has no bootstrap-p condition.

**Reading (ii).** B-mm calls S on 218 of the 2,000 L4 D loci (P1 0.125, P3 0.093). The kill line allows 196, so B-mm is 22 loci, or 1.1 pp, above it.
- My cell-stratified locus bootstrap of the margin gives SE 0.011 and P(margin ≤ 0) = 0.17. The ladder reports 0.17 (locus) and 0.12 (block).
- The kill would lapse only if B-mm's t_S rose from 0.508 to above about 0.621.

**Reading (iii).** The margin is 5.5 pp, but on only 200 paired D loci: P(no kill) = 0.08.

Each reading is fragile alone; P4-1 makes either binding, and both kill. No implementation choice I varied (threshold scan vs max of one-sided thresholds, pooled vs cell-mean rates, strength_only in or out) changes any number; the fragility is sampling noise only.

**Conclusion.** Gate PASS; rules (a), (b), (d) not recomputed here. **B-mm is KILLED by rule (c) at OP2, L4, under (ii) and (iii): CONFIRMED**, with a modest statistical margin (P ≈ 0.08–0.17 per reading that a resample would not kill).
