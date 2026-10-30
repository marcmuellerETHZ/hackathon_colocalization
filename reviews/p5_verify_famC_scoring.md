# p5.verify_famC_scoring: independent check of the Family C C1 scoring fix

Verifier: p5-verify agent. I did not write Family C's code. Date: 2026-10-27.
Branch `p5-verify-famC` (from `p4-famC` @ 8eec367).

Under review:
- 76ad32b: "no-correction inference for unweighted-CE C1 models".
- 8eec367: `reports/famC_M3.5_addendum.md`.

Scope:
- Pilot set only: `data/processed/famC/l0star_pilot`, 600 loci, S/D/W/G × 150.
- l0star_v2 and its predictions were not opened or evaluated. No test data was touched.

## Verdict

**The fix is correct. The corrected pilot numbers reproduce exactly by an independent path:**
- natural E[KL(B‖C)] = **0.00638**;
- balanced E[KL(B‖C)] = **0.01984**.

**The "before" numbers also reproduce:**
- natural = 0.04842;
- balanced = 0.11117.

I found no other error in the C1 statistic. There are three caveats (§4). They do not change the pilot verdict, but caveat (a) matters for the pre-registered l0star_v2 run.

## 1. Training side: what the softmax estimates

**Loss.** Every version of `src/famC/c1.py` uses plain unweighted cross-entropy: 4eec69d, 0aded87, e285fe2, 09f1297 and 76ad32b.
- The training call is `torch.nn.functional.cross_entropy(o["logits"], y)`.
- There is no `weight=`, no label smoothing and no class-balanced sampling.

**c1_long's training data.** c1_long started at 19:16 on 26 Oct, at the e285fe2 code.
- Hard oracle labels come from `sim.l0star.sample_locus` on a stream of train-split windows.
- `sample_locus` draws from the natural L0* prior. There is no rejection by class.
- ρ is drawn uniformly from {0, 0.4}, the same grid as the pilot set.

**c1_v2.** It uses the same loss. Its `config.json` records `logit_correction: "none"`.

**Augmentation.** Training applies trait-sign augmentation (`fl`). `features.build` flips z_t and sets ρ → s1·s2·ρ. This is a symmetry of the world, so it does not change the class posterior.

**Inference model features.** `features.py` and `model.py` are unchanged between the training code and the evaluation code, except for one added assertion that true ρ is never an input.

**What the softmax therefore estimates.** The minimiser of unweighted CE on the natural stream is P(class | data) under the natural mixture. That is the same target as B's simulator-prior posterior.
- The correct inference transform is the **identity** (plain softmax).
- `logits − log CLASS_W` is right only for `train_main` models. Their CE is weighted with CLASS_W = (1, 1, 0.12, 0.4, 1), so their softmax is proportional to w_c·p_c.
- Applying that correction to the C1 models multiplies the W odds by 8.3 and the G odds by 2.5. That is wrong.

**The `auto` switch picks the right transform for each run type:**

| Run | Config | `auto` gives |
|---|---|---|
| `main_snap1_full`, `main_dev` | has `class_w` | "main" |
| `c1_long`, `c1_pilot` | no `class_w` | none |
| `c1_v2` | `logit_correction` = "none" | none |

**Empirical confirmation.** I averaged each method's posterior over the pilot set with natural weights. A correctly calibrated natural-mixture posterior must return the natural class shares.

| | S | D | W | G |
|---|---|---|---|---|
| Natural shares (renormalised over S/D/W/G) | 0.057 | 0.068 | 0.787 | 0.088 |
| B | 0.052 | 0.067 | 0.784 | 0.093 |
| C, no correction | 0.050 | 0.066 | 0.788 | 0.091 |
| C, old correction | 0.041 | 0.053 | **0.828** | 0.075 |

Only the uncorrected C matches the natural shares. The old correction over-weights W, as the theory predicts.

## 2. Evaluation side (`scripts/famC_eval/c1_eval.py`, `famC.c1.predict_l0star`, `famC.method.NeuralColoc`)

**C posterior.** `softmax(logits)` when `class_w=None`, and `softmax(logits − log CLASS_W)` when `class_w="main"`. The output order is S, D, W, G, U, which is identical to the training label order `L0S_CLASSES`.

**B posterior (pilot).** For the pilot, B comes from `famB_sim_preds.parquet`, produced by `famC.c1_pilot runB`.
- It runs `famB.method.ExactColoc` in simulator mode with `sim_prior_from_l0star()`, η = 0, n_mc = 16,384, and no class reweighting.
- `load_b` maps P_S, P_D and extra.{P_W_strict, P_G, P_U}. The rows sum to 1.000.
- This is not `famB_l0star_ref@1`, which exists only for l0star_v2 (P3-19).
- For the l0star_v2 run, `load_b` reads the `famB_l0star_ref` format (P_S, P_U, P_D, P_W strict, P_G; sum 1, per ARTIFACTS.yaml). The column semantics match.

**KL.** KL(B‖C) is computed per locus.
- It uses the natural log (nats), with 0·ln 0 = 0.
- C is clipped at 1e-12 and renormalised.
- B is clipped to [0, 1] and renormalised.

**Natural weighting.** The weight for class c is nat_c / Σ_{present} nat / set_share_c, and the statistic is Σ w·KL / Σ w.
- The shares come from `l0star_pilot.stats.json` → `natural_class_shares`. I recomputed them from `candidate_class_counts`, and they are identical.
- U is absent from the set (natural share 0.006), so the expectation is over S/D/W/G with the shares renormalised. This is documented in the script.

All of this is correct and matches P3-4.

## 3. Independent recomputation

**Method.** My scripts are in `reviews/p5_verify_famC/`.
- `recompute_c1.py`:
  - builds each locus's batch by hand in the training-stream form, one locus at a time;
  - uses `DenseLD` rather than the method's padded `StackedLD` batch path;
  - takes the **raw float64 logits**;
  - applies my own softmax, KL and class-mean form of the natural expectation;
  - hard-codes CLASS_W for the "old" variant.
- I also re-ran `c1_eval.py` (`--class-w auto|none|main`) on the pilot set for `results/famC/c1_long/model.pt`.
- Both ran in job `p5_verify_famC_recompute_r2` (done; 3.9 min; peak 0.65 GB VRAM).

**Recomputed statistics** (c1_long 8 M):

| Quantity | Mine (raw logits) | c1_eval rerun | Addendum |
|---|---|---|---|
| natural, no correction | 0.0063783 | 0.0063783 (auto = none) | 0.0064 |
| balanced, no correction | 0.019844 | 0.019844 | 0.020 |
| natural, old correction | 0.048416 | 0.048416 | 0.048 |
| balanced, old correction | 0.11117 | 0.11117 | 0.111 |
| by class, no correction | S 0.0428, D 0.0168, W 0.0015, G 0.0183 | identical | S 0.043, D 0.017, W 0.0015, G 0.018 |

**50-locus stratified subset** (13 S, 13 D, 12 W, 12 G, seeded), compared per locus:
- max |ΔP| = 9.8e-7;
- max |ΔKL| = 1.2e-6.

Over all 600 loci: max |ΔP| = 1.05e-6 and max |ΔKL| = 1.3e-6. This is float noise from batched versus single-locus inference.

The subset's own statistics are natural 0.0073 / balanced 0.017 without correction, and 0.072 / 0.115 with the old correction.

**Stored predictions.** `check_stored_preds.py` checked the stored `c1_pilot_eval_v2` predictions on CPU.
- For all 7 checkpoints, P_main = renorm(P_none / CLASS_W) to within 5e-8. The two modes differ by exactly the logit shift.
- My hand KL reproduces every number in both addendum tables (for example 4 M: 0.0142 / 0.0451 none, 0.0523 / 0.1427 main) to within 1e-9.
- The "main" values equal the M3.5 numbers in `results/famC/c1_long/c1_eval.json` (0.0913 / 0.0642 / 0.0523 natural).

The "before" numbers are confirmed as the output of the old correction.

## 4. Other points in the C1 computation

**(a) Most important: the W stratum comes from only 4 LD blocks, and W carries 79% of the natural weight.**
- Distinct blocks per class in the pilot set: S 42, D 39, G 32, W 4.
- Cause: `sim.l0star.build_set` fills classes in visit order, so the 150 W slots fill within the first ~4 segments.
- The point estimate is unbiased in expectation over segments. But the cluster-bootstrap CI rests on very few W clusters, and W's per-class KL is poorly sampled across LD.
- On the pilot this cannot flip the verdict: W contributes 0.0012 of the 0.0064, and even a 5× larger W-KL gives 0.012 < 0.02.
- The same builder made l0star_v2 (500/class), so the pre-registered C1 should report per-class block counts. It should also accept that the W term's uncertainty is under-represented.
- Recommended default: report per-class block counts and the W-term contribution beside the C1 value. Do not change the statistic or the set.

**(b) Shares source.** The natural shares are MC estimates from 2,650 candidates (S SE 0.0045).
- Moving each share by ±2 SE moves the natural statistic within 0.0060–0.0067.
- S is the stopping class (exactly 150 candidates), which introduces a negative-binomial bias of about 0.6% of S's share.
- Both effects are negligible.

**(c) U excluded.** The natural share of U is 0.6%, and U loci are not in the set. Their KL is therefore outside the expectation. The resulting bias is at most 0.006 × (U-locus KL).

**(d) Clipping and rounding.** `predict_l0star` rounds P5 to 8 decimals, and some stored C values are exactly 0.
- The raw minimum of C is 2.1e-9, so the 1e-12 clip never binds against B's mass.
- The raw-versus-rounded KL difference is ≤ 1.3e-6.

**(e) Log base and direction.** KL uses natural log, in the direction KL(B‖C) (B-weighted), as specified.

**(f) B's Monte Carlo error.** The pilot B uses n_mc = 16,384 and has no standard-error columns, so `c1_eval` reports no MC floor.
- The expected inflation of KL from B's noise is about (K−1)/(2·n_eff), roughly 1e-4 for independent draws. That is negligible against the 0.02 bar.
- The l0star_v2 reference (262,144 draws, with SE columns) gets the floor term.

**(g) Pilot caveats.** These are already stated in the addendum:
- the set is Family C's own draw;
- the 8 M checkpoint was examined after the fact;
- the balanced value of 0.0198 sits right at 0.02, though it is the secondary reading.

**(h) Stream/set mismatch outside the scoring.** c1_long's training stream (e285fe2) lacked the p ≥ 200 window filter that `build_set` applies.
- This shifts only the LD distribution of the training inputs. It does not change the posterior target, because C conditions on R.
- It affects C's fit, not the validity of the scoring.

## Commands

```
scripts/orch/orch submit --id p5_verify_famC_recompute_r2 --gpu yes --vram-gb 2.5 --cores 1 --ram-gb 3 -- \
  'python reviews/p5_verify_famC/recompute_c1.py --out results/p5_verify_famC/recompute_c1_long8M_pilot.json;
   for cw in auto none main; do python scripts/famC_eval/c1_eval.py --set data/processed/famC/l0star_pilot/l0star_pilot.h5 \
     --b-preds data/processed/famC/l0star_pilot/famB_sim_preds.parquet --ckpt results/famC/c1_long/model.pt \
     --class-w $cw --out results/p5_verify_famC/c1_eval_pilot_$cw.json; done'
python reviews/p5_verify_famC/check_stored_preds.py reviews/p5_verify_famC/check_stored_preds.out.json   # CPU, <1 min
python reviews/p5_verify_famC/compare_perlocus.py                                                        # CPU, <1 min
```

The artifact is `p5_verify_famC_c1_recompute@1` (`results/p5_verify_famC`, checksum 7a19ff7f…).
