# Family C (NeuralColoc) — M3.5 report, addendum: §5 C1 pilot completed and corrected

Author: Family C agent (task p4.famC_C1). Date: 2026-10-27. Code: branch `p4-famC`, commit 76ad32b.
Job: `famC_c1_pilot_eval_r1` (done). Results: `results/famC/c1_pilot_eval_v2/c1_eval_pilot_{main,none}.json`
and per-checkpoint prediction parquets alongside. Log: `logs/famC/c1_pilot_eval_r1.log`.

This addendum replaces §5 ("C1 (world L0*) — readiness and pilot") of `reports/famC_M3.5.md`. That section had
two placeholders (the 8 M column of `c1_long`). Filling them uncovered an evaluation bug that inflated every C1
pilot number in the M3.5 report. Nothing in §§0–4 (C0, architecture, main task) is affected.

## A. What changed

1. **Evaluation bug (fixed).** The C1 trainer (`famC.c1 train`, every version since 4eec69d) uses *unweighted*
   cross-entropy on the natural L0* stream, so the network's softmax is already the natural-mixture posterior.
   The M3.5 evaluation (`famC.c1.evaluate_against_B` → `famC.method.NeuralColoc`) applied the main-task
   correction `logits − log CLASS_W` (README C-T2), which is correct only for models trained with class-weighted
   CE. It multiplied C's W odds by 1/0.12 ≈ 8.3 and its G odds by 2.5 before the KL against B.
   - Fix: `NeuralColoc(class_w=None|"main")`. `famC.c1.predict_l0star` chooses it from the run config: a run's
     config records main-task class weights only for `train_main` runs. Test: `tests/famC/test_c1_stream.py::
     test_method_logit_correction_switch`.
   - Reproduction: re-running the M3.5 pipeline (`--class-w main`) gives the M3.5 numbers to every printed
     digit (balanced 0.220 / 0.178 / 0.143, natural 0.091 / 0.064 / 0.052). So the M3.5 numbers came from the
     wrong correction, not from a different set or seed.
2. **Truth isolation (DECISIONS P3-21).** The truth-reading evaluation moved out of `src/famC` to
   `scripts/famC_eval/c1_eval.py`. `tests/eval/test_truth_isolation.py` now passes (7/7).
3. **Set.** The placeholders sit in the row of the pilot set `l0star_pilot`. That set is 600 loci, S/D/W/G × 150,
   validation windows, seed 20261029, ρ ∈ {0, 0.4}: the same ρ grid c1_long was trained on. They are filled on
   that set, so the row stays comparable. `l0star_v1` was not used, for two reasons. It has no B reference (it
   is stale under P3-16; the B reference is produced on `l0star_v2` only, P3-19). And its ρ grid
   (0, ±0.3, ±0.5, ±0.8) differs from c1_long's training grid.

## B. Corrected §5: C1 (world L0*) — pilot

**Pilot set and reference.**
- `famC.c1_pilot` builds 600 loci with `sim.l0star.build_set` on validation windows (seed 20261029; 44 LD blocks).
- The reference is Family B in simulator mode: the exact `sim_prior_from_l0star`, η = 0, n_mc = 16,384.
- The statistic is KL(B ‖ C) over S, D, W, G, U.
  - "Natural" reweights the class-stratified set to the natural class shares in the set's stats (W 0.782,
    G 0.088, D 0.068, S 0.057; U 0.006 is not in the set). This is the C1 reading of P3-4.
  - "Balanced" is the unweighted mean over the set.
- The 95% intervals are a cluster bootstrap over LD blocks (2,000 replicates).

**Learning curve, corrected evaluation (no logit correction; the valid one for these models):**

| Run | 100 k | 300 k | 1 M | 2 M | 4 M | 8 M |
|---|---|---|---|---|---|---|
| `c1_pilot` (batch 256, cosine over 2 M; died at 1.2 M), balanced | 0.187 | 0.122 | 0.079 | – | – | – |
| same, natural | 0.063 | 0.040 | 0.029 | – | – | – |
| `c1_long` (batch 512, cosine over 8 M), balanced | – | – | 0.081 | 0.066 | 0.045 | **0.020** |
| same, natural | – | – | 0.029 | 0.024 | 0.014 | **0.0064** |
| same, natural, 95% CI | – | – | [0.019, 0.064] | [0.016, 0.053] | [0.009, 0.034] | [0.004, 0.015] |

**The same models under the M3.5 (wrong) correction, for the record:**

| Run | 100 k | 300 k | 1 M | 2 M | 4 M | 8 M |
|---|---|---|---|---|---|---|
| `c1_pilot`, balanced / natural | 0.281 / 0.086 | 0.239 / 0.075 | 0.126 / 0.041 | – | – | – |
| `c1_long`, balanced / natural | – | – | 0.220 / 0.091 | 0.178 / 0.064 | 0.143 / 0.052 | 0.111 / 0.048 |

The as-computed values for the two placeholders (C1L8, C1L8W) are therefore 0.111 (balanced) and 0.048
(natural) under the M3.5 pipeline. The valid values are **0.020** and **0.0064**.

**c1_long at 8 M (corrected), more detail.**

| Measure | Value |
|---|---|
| E[KL] by true class | S 0.043, D 0.017, W 0.0015, G 0.018 |
| per-locus KL, natural-weighted q90 / q99 | 0.011 / 0.119 |
| per-locus KL, balanced q90 / q99 | 0.037 / 0.210 |
| per-locus KL, max | 1.63 |
| max \|ΔP_S\|, max \|ΔP_D\| | 0.79, 0.65 |
| correlation of C's with B's class posteriors | S 0.989, D 0.991, W 0.998, G 0.983 |

Calibration against the oracle labels (natural-weighted; ECE is one-vs-rest with 10 bins):

| | log loss | accuracy | ECE S | ECE D | ECE W | ECE G |
|---|---|---|---|---|---|---|
| C, c1_long 8 M | 0.178 | 0.927 | 0.009 | 0.005 | 0.021 | 0.012 |
| B (Bayes-optimal in L0*) | 0.175 | 0.924 | 0.008 | 0.006 | 0.013 | 0.018 |

**Reading.**
- With the correct inference, C approaches B's posterior steadily with data. At 8 M hard-label loci, on the
  pilot set, it is below the C1 bar under both readings (natural 0.0064, upper CI 0.015; balanced 0.020).
- The tail is not closed. A few loci still differ by up to 0.79 in P_S, and the S class carries the largest
  per-class KL.
- This is a pilot, not the C1 of record:
  - the set is an own draw;
  - c1_long's ρ grid {0, 0.4} is not the l0star_v2 grid;
  - the 8 M checkpoint was examined on this set after the fact.
- The pre-registered C1 is the `c1_v2` run (l0star_v2 spec, ≤ 16 M loci) evaluated once on l0star_v2:
  `reports/p4.famC_C1.md`.

**Commands.**
```
python scripts/famC_eval/c1_eval.py --set data/processed/famC/l0star_pilot/l0star_pilot.h5 \
  --b-preds data/processed/famC/l0star_pilot/famB_sim_preds.parquet --class-w {main|none} \
  --ckpt results/famC/c1_pilot/model_{100096,300032,1000192}.pt results/famC/c1_long/model_{1000448,2000384,4000256}.pt \
         results/famC/c1_long/model.pt \
  --out results/famC/c1_pilot_eval_v2/c1_eval_pilot_{main|none}.json
```
(job `famC_c1_pilot_eval_r1`, commit 76ad32b; `results/famC/c1_long/model.pt` is the 8 M checkpoint: 15,625 steps
× 512 = 8,000,000 loci, cosine over 8 M.)

## C. Consequences for the M3.5 summary and §6.4

- §0 "C1" bullet: the pilot gap to the bar was an artefact of the wrong correction.
  - Corrected natural-weighted E[KL]: 0.029 at 1 M, 0.014 at 4 M and 0.0064 at 8 M.
  - The M3.5 report gave 0.041 (c1_pilot, 1 M) and 0.052 (c1_long, 4 M).
- §6.4 ("C1 at 0.02 looks expensive with hard labels"): no longer supported. Option (i), a larger budget with the
  bar kept, is what P3-9 adopted, and the corrected pilot suggests 8–16 M loci is enough.
- The existing main-task results (§4) use `train_main` models trained with class-weighted CE. For those models
  the correction is right, and they are unchanged.
