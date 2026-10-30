# Family C (NeuralColoc) — M3.5 report

Author: Family C agent. Date: 2026-10-26. Code: `src/famC/` (its README lists every decision: C-A*, C0-*, C-T*).
Tests: `tests/famC/`. Results: `results/famC/<run>/`. Logs: `logs/famC/`. Commits: 4eec69d, 83923a9, 0aded87,
e285fe2, plus the final commit with this report.

## 0. Summary

- **C0 passes.** In coloc.abf's own world on real 1000G LD (validation-split windows), the network reproduces
  coloc.abf's posteriors. At 5 M training loci: E[KL] = 0.00072 and 99th-percentile |ΔPP4| = 0.043 under uniform
  hypotheses; E[KL] = 0.00029 and |ΔPP4| = 0.012 under coloc's own prior (criteria: < 0.01 and < 0.05). There is no
  train/validation-window gap. Caveat: the tail is worse on H4 loci (99th-percentile |ΔPP4| = 0.083), on loci where
  both traits have |z| ≥ 5 (0.08–0.11), and the worst single locus has |ΔPP4| = 0.63.
- **Architecture built** as DESIGN §5 C asks:
  - Family B's screen, identical to B's in fp64;
  - an LD attention bias;
  - exact allele-flip invariance (tested);
  - global features (ρ̂, N, LD eigen-spectrum, ...);
  - a 5-class head plus a shared-PIP head.

  I added two things: an all-variant stream, and a readout structured like a Bayes factor. The all-variant stream
  halves KL at an equal budget in C0 (screen-only ablation).
- **Training data.** The simulation agent's on-the-fly TRAIN API exists (`sim.train_api`, uncommitted). An adapter
  uses it with L6 held out, and generates shards for training and validation.
  - Per worker the API gives about 10 loci/s, so data generation is the bottleneck of full training.
- **Main task, development data** (dev data, from a simulator still being revised for M3.5; not a kill-test):
  - The network beats its no-LD ablation (paired ΔAUPRC S-vs-D = +0.008 [0.004, 0.011]).
  - It ties the LightGBM floor (coloc.abf + GPU-SuSiE-on-screen features) at matched training data (74 k records):
    ΔAUPRC S-vs-D −0.0004 [−0.0035, 0.0024]; the floor is slightly better on D-vs-W (ΔAUROC −0.0009
    [−0.0016, −0.0004]); the network has the better log loss (0.397 vs 0.427) and ~5× cheaper inference.
    **The floor criterion (C must beat the floor) is not met at this development scale.**
  - It is far above coloc.abf (0.871).
  - With the network trained on 74 k records, the train/validation gap is 0.018 (S-vs-D AUPRC).
  - L6 (held out) is 0.019 below the realistic L1–L4 pool.
- **C1.** The frozen `l0star_v1` set does not exist, so the pre-registered C1 could not be run. A pilot on my own
  L0* validation draw (same builder; B's simulator mode as the reference) gives E[KL(B ‖ C)] = 0.126 at 1 M loci and 0.143 at 4 M in a longer run with a still-high learning rate (class-balanced set; natural-mixture weighted 0.041 / 0.052). That is not yet near the
  0.02 bar, and it is still decreasing with data.

## 1. What was built (DESIGN v4 §5 C)

| Part | Where | Status |
|---|---|---|
| Real-LD window bank (pool A, n = 254; train or valid split, test refused; int8 on the GPU) | `ldbank.py` | done, tested |
| C0 world: coloc.abf's model on real LD (details below) | `c0world.py` | done; target == `baselines.abf.coloc_abf` (< 1e-10); noise covariance == R and signal mean checked |
| Screen: Family B's per-trait-union screen, batched on the GPU, no duplicate collapse (details below) | `features.screen_gpu` | == `famB.screen.screen(collapse=False)` in fp64 on real-LD loci; ≥ 97% overlap with fp32 LD (near-ties) |
| Two-stage features (details below) | `features.py` | allele-flip invariance exact (test: inputs and outputs unchanged) |
| Network (details below); 1.04 M parameters | `model.py` | done |
| C0 trainer, learning curve, error breakdown | `train_c0.py`, `c0_analyze.py` | **C0 pass** |
| Adapter to `sim.train_api.TrainSampler` (details below) | `simadapter.py`, `gen_shards.py` | record z == generator OLS z; label == oracle class (test) |
| Main-task trainer: class-weighted CE (S, D, W, G, U) + shared-PIP CE; natural-mixture correction; monitoring per level | `train_main.py` | dev runs §4 |
| Floor: LightGBM on hand-built features (details below) | `floor.py` | dev runs §4 |
| No-LD ablation: every LD-derived input zeroed, no LD bias | `train_main --variant nold` | §4 |
| Method API `NeuralColoc(ckpt).fit_batch(list[LocusData])` (INTERFACES §3/§4), batched over loci with their own LD | `method.py` | dense == factor; batched == single; failures reported (tests) |
| C1 trainer on `sim.l0star` train segments; evaluation vs B's simulator-mode posteriors; pilot driver | `c1.py`, `c1_pilot.py` | pilot §5 |
| Paired bootstrap between dev runs | `compare_dev.py` | §4 |

Details for the rows above:
- **C0 world:**
  - N_t ~ logU[300, 3×10⁵];
  - se = 1/√(2Nf(1−f));
  - hypotheses H0–H4;
  - β ~ N(0, W²) with W = 0.15;
  - z = λRe_j + X'u/√n;
  - target = coloc.abf PP over the full window (p1 = p2 = 1e-4, p12 = 1e-5), in fp64.
- **Screen:** lists m1, m2, c1, c2; quotas M/2 − M/8 and M/8; M = 128; η = 1e-3; no duplicate collapse.
- **Two-stage features:**
  - `extract` needs the LD: the screen, conditional z, the leads' LD rows, the screen LD and its eigen-spectrum;
  - `build` does not: trait-sign augmentation, allele canonicalisation and scaling.
- **Network:**
  - a transformer (d = 128, 4 layers, 4 heads) over the screened variants, with an LD bias MLP(r, r², |r|) per head;
  - an all-variant stream: a per-variant MLP over every window variant;
  - the readout: K = 16 energies per stream, each quadratic in (z1², z2², z1z2, zc1², zc2², zc1zc2), pooled by
    log-sum-exp;
  - a 5-class head and a shared-PIP head;
  - global features: ρ̂ and its SE, N1, N2, n_ref, p, trait types, max |z|, number of |z| > 5.45, lead–lead r, and
    the effective rank and top-12 eigenvalues of the screen LD.
- **Adapter:** a subclass that replaces only `_locus`. L6 is held out. Records keep the all-variant stream.
  Validation draws come from validation windows with seed 20261029 and include L6.
- **Floor features:** leads; lead r²; conditional z; coloc.abf PP0–4; GPU SuSiE (famG) on the screen with
  coloc.bf_bf; context features.

Tests: `source env/activate.sh; PYTHONPATH=src python -m pytest -q tests/famC` → **15 passed** (41 s, GPU).
- Allele-flip invariance: inputs and outputs unchanged.
- Screen == famB in fp64.
- Batched coloc == local coloc.abf.
- C0 noise covariance == R; signal mean correct.
- Record compaction round-trip.
- Method:
  - dense == factor LD;
  - batched == single;
  - failure reporting.
- Adapter:
  - record z == the generator's z;
  - label == oracle class;
  - PIP target == the shared causal variant or its screened proxy.

## 2. C0 kill-test — PASS

Setup:
- Training: 2,998 train-split windows (seed 20261030), hypotheses uniform. The run is `c0_full`:
  - batches of 32 windows × 8 loci;
  - AdamW, lr 5e-4, cosine schedule over 40 k steps;
  - KL loss + 0.1 × SNP.PP.H4 loss.
- Evaluation: 400 validation windows, seed 20261029, the same draws every time. 20,000 loci per distribution:
  - "uniform": H0–H4 equally likely;
  - "coloc": coloc's own prior odds 1 : p·p1 : p·p2 : p(p−1)p1p2 : p·p12.

The bar must hold on both distributions.

**Learning curve** (`results/famC/c0_full/curve.tsv`; one training seed, derive(20261030, "famC", "C0train", run)).
"Train windows" is the same metric on fresh draws from the training windows.

| training loci | uniform: E[KL] | uniform: q99 \|ΔPP4\| | coloc prior: E[KL] | coloc prior: q99 \|ΔPP4\| | train windows: E[KL] | wall (min) |
|---|---|---|---|---|---|---|
| 50 k | 0.128 | 0.606 | 0.067 | 0.340 | 0.130 | 1 |
| 100 k | 0.048 | 0.375 | 0.024 | 0.093 | 0.049 | 2 |
| 200 k | 0.028 | 0.248 | 0.016 | 0.069 | 0.029 | 3 |
| 500 k | 0.011 | 0.139 | 0.0060 | 0.037 | 0.012 | 9 |
| 1 M | 0.0064 | 0.119 | 0.0031 | 0.028 | 0.0064 | 12 |
| 2 M | 0.0028 | 0.081 | 0.0013 | 0.021 | 0.0026 | 24 |
| **5 M** | **0.00072** | **0.043** | **0.00029** | **0.012** | 0.00069 | 43 |

- **When the criterion was met.** It was first met at the 5 M evaluation, which was in the pre-set list
  (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10 M).
- **Why the run stopped early.** I stopped it at 6.2 M (planned 10 M) to free the GPU. The reported model is the
  5 M checkpoint (`results/famC/c0_full/model.pt`, committed). Stopping early can only make the reported numbers
  worse than a finished run, not better.
- **Code path.** The model was trained with pre-commit code. A re-evaluation with the committed code
  (`famC.c0_analyze`, refactored features and new screen merge) reproduces the metrics to the printed digits:
  uniform E[KL] 0.000724, q99 0.0430. Re-running training would give a statistically equivalent model, not a
  bit-identical one.

**Other 5 M metrics:**
- uniform:
  - 99th-percentile |ΔPP_k| for k = 0..4: 0.024, 0.030, 0.030, 0.035, 0.043;
  - 99th-percentile KL: 0.0087; maximum KL: 0.93; maximum |ΔPP4|: 0.63;
  - argmax agreement: 99.6%.
- coloc prior: maximum |ΔPP4| 0.22; argmax agreement 99.8%.

**Where the error is** (`results/famC/c0_full/analysis.json`), on the uniform distribution:

| Stratum | 99th-percentile \|ΔPP4\| | Share of loci with \|ΔPP4\| > 0.05 |
|---|---|---|
| H0 | 0.002 | |
| H1 | 0.013 | |
| H2 | 0.016 | |
| H3 | 0.027 | |
| H4 | 0.083 | |
| min lead \|z\| [5, 8) | 0.084 | 2.7% |
| min lead \|z\| [8, 15) | 0.107 | 2.0% |
| min lead \|z\| [15, 30) | 0.107 | 3.5% |

The pass is carried by the many easy loci. In the hard region the 99th percentile is about 2× the bar. The error
was still falling when the run stopped (2 M → 5 M: overall 0.081 → 0.043).

**Ablation (decision C-A2, all-variant stream).** `c0_screen` uses the same schedule and was stopped after the
2 M evaluation. It has no all-variant stream: only the transformer over the 128 screened variants.

| At 2 M loci | uniform: E[KL] | uniform: q99 \|ΔPP4\| | coloc prior: E[KL] | coloc prior: q99 \|ΔPP4\| |
|---|---|---|---|---|
| full model | 0.0028 | 0.081 | 0.0013 | 0.021 |
| screen only | 0.0054 | 0.102 | 0.0028 | 0.032 |

The full model has about half the KL. The screen-only model also learns, so screening loses little in this world.

## 3. Throughput (shared RTX 4090; other agents' jobs running throughout, so these are lower bounds)

- C0 training (generation + features + forward/backward, B = 256, p ≈ 2.7 k): 1,400–2,500 loci/s. Peak memory
  9.2 GB (the all-variant stream dominates).
- C0 batched inference (features incl. GPU screen + forward, factor LD in GPU memory): 2,200–6,800 loci/s.
- Main-task training on stored records (B = 128): about 1,100–1,250 loci/s, 4.8 GB.
- Main-task batched inference from stored records: 2,200–4,400 loci/s.
- Method API on `LocusData`, batched (screen_v1 validation loci, p ≈ 2.8 k, random-init weights; disk read
  excluded):

  | LD | ms per locus | loci/s |
  |---|---|---|
  | L0, dense | 7.6 | 130 |
  | L2 / L4, factor | 3.7 | 270 |

  Reading dense L0 LD from HDF5 costs 0.20 s/locus and dominates (factor: 17 ms). Per-locus (unbatched) calls cost
  35–43 ms.
- Simulator → records (`sim.train_api` + adapter):
  - per GPU worker: about 10 loci/s with one noise draw per candidate, 12–13 loci/s with two noise draws
    (decision C-T6);
  - validation with W/G rejection: 6.5 loci/s;
  - 2 workers ≈ 90 k records/h.

  Profile per 128-locus window visit:
  - L3 PC eigh (fp64 4,000 × 4,000 per admixed study): 26%;
  - GPU synchronisations: about 40%;
  - mosaic materialisation: about 10%.
- Floor features: 22–44 ms/locus (dominated by GPU SuSiE on the screen).

## 4. Main task on development data (monitoring, not a kill-test)

**Data:**
- `train_dev`: train windows, seed 20261030, L6 held out, natural class mixture, two noise draws per candidate.
  The runs below use a frozen snapshot, `train_dev_snap1`: 73,728 records (S 7%, D 8%, W 67%, G 13%, U 0.1%).
- `valid_dev`: validation windows, seed 20261029, all levels including L6, W and G down-sampled at generation.
  15,000 loci (S 2,903; D 3,345; W 6,646; G 2,058; U 48).

**Status of these data.** They come from the simulator as it stood at 16:55–17:42, which is uncommitted and
mid-M3.5. I recorded the code sha256s in the shard configs. These numbers are for development only. Final
training must regenerate the data from the committed M3.5 simulator.

**Metrics.** Computed with `eval.metrics` functions, but these are not the pre-registered pooled cells:
- S-vs-D AUPRC of s = P_S/(P_S + P_D), power-matched within level (`power_match_weights`);
- D-vs-W AUROC of P_D;
- log loss under the natural-mixture correction.

Results:

| Run (same validation loci) | S-vs-D AUPRC (pm), all | S-vs-D AUPRC (pm), L1–L4 | S-vs-D AUPRC (pm), L6 (held out) | S-vs-D AUPRC (pm), L0 | D-vs-W AUROC | log loss |
|---|---|---|---|---|---|---|
| coloc.abf (W 0.15, p12 1e-5; reference) | 0.871 | 0.872 | 0.860 | 0.877 | 0.975 | — |
| floor, LightGBM, 32 k train records (`floor_dev`) | 0.974 | 0.973 | 0.953 | 0.995 | 0.998 | 0.444 |
| floor, LightGBM, 74 k train records = the network's data (`floor_snap1`) | 0.977 | 0.975 | 0.958 | 0.994 | 0.998 | 0.427 |
| NeuralColoc, 74 k records, 8 k steps (`main_snap1_full`) | **0.976** | **0.977** | **0.958** | 0.987 | 0.998 | **0.397** |
| no-LD ablation, same data and schedule (`main_snap1_nold`) | 0.968 | 0.969 | 0.953 | 0.983 | 0.998 | 0.410 |

**Full model vs no-LD ablation** (paired locus bootstrap, 500 replicates, power-matching weights recomputed each
replicate; `results/famC/compare_main_snap1_full_vs_main_snap1_nold.json`):

| Endpoint | Difference | 95% CI |
|---|---|---|
| AUPRC S-vs-D, all | +0.0077 | [0.0039, 0.0112] |
| AUPRC S-vs-D, L1–L4 | +0.0083 | [0.0047, 0.0126] |
| AUROC D-vs-W | 0.0000 | [−0.0007, 0.0005] |

**Network vs floor** (matched data; paired bootstrap, `results/famC/compare_main_snap1_full_vs_floor_snap1.json`):

| Endpoint | Network − floor | 95% CI |
|---|---|---|
| AUPRC S-vs-D, all | −0.0004 | [−0.0035, 0.0024] |
| AUPRC S-vs-D, L1–L4 | +0.0017 | [−0.0016, 0.0049] |
| AUROC D-vs-W | −0.0009 | [−0.0016, −0.0004] |

- A tie on the primary endpoint; the floor is marginally better on D-vs-W; the network is better on log loss.
- The floor's top features are SuSiE (fallback flag, PP4) and coloc.abf PP3: it is an ensemble of baselines. Its
  features cost 22–44 ms/locus (GPU SuSiE on the 128-variant screen) vs 4–8 ms/locus for the whole network.
- **Floor kill-criterion: not met at this data scale.** To be re-tested with ≥ 500 k records: the network is
  data-limited (train/validation gap 0.018), while the floor early-stops at 145 trees.

**Gaps:**
- Network, train region vs validation (S-vs-D AUPRC): 0.994 vs 0.976, a gap of 0.018 (the C2 limit is < 0.02).
- An earlier 20 k-step run on fewer data (`main_dev`) overfitted: 0.961 at step 4 k, then 0.954.
- Hence the network is data-limited.
- Per level (matched data, `floor_snap1` vs network): the floor is stronger at L0/L1 (0.994 / 0.982 vs
  0.987 / 0.974); the network is stronger under small-reference mismatch (L2 0.977 vs 0.970, L4 0.974 vs 0.968).

All of this uses one training seed per variant. The bootstrap covers validation-locus noise only, not seed noise.

## 5. C1 (world L0*) — readiness and pilot

> **Superseded (2026-10-27): see `reports/famC_M3.5_addendum.md`.** Every C1 pilot number in this section
> (and in the §0 C1 bullet) was computed with the main-task logit correction applied to models trained with
> unweighted CE, which is wrong for them (fix verified independently, reviews/p5_verify_famC_scoring.md). The
> 8 M column below is filled with the value of that same (wrong) pipeline for comparability; the valid values are
> class-balanced 0.020 and natural-weighted 0.0064 (addendum §B). The pre-registered C1 is reports/p4.famC_C1.md.

**Readiness.**
- `c1.py` trains C on `sim.l0star` segments from train windows: 30 SNPs, R̃, the exact L0* prior, hard
  oracle-class labels, ρ ∈ {0, 0.4}.
- `famC.c1 eval` scores checkpoints against B's simulator-mode posteriors on an L0* locus set (KL(B ‖ C) over
  S, D, W, G, U).
- What is missing: the frozen `l0star_v1` set (simulation agent) and the B predictions on it.
- Once both exist, the pre-registered C1 is:

  ```
  python -m famC.c1 eval --run <run> --set data/processed/sets/l0star_v1.h5 --b-preds <B predictions parquet>
  ```

  This assumes l0star_v1 uses ρ ∈ {0, 0.4} and N 100 k / 1 k. If its ρ grid differs, C must be retrained with the
  matching `rho_values`.

**Pilot** (not the pre-registered test).
- Set: `famC.c1_pilot` builds 600 loci (S, D, W, G 150 each; 53 segments) with `sim.l0star.build_set` on
  validation windows, seed 20261029.
- Reference: B simulator mode (exact `sim_prior_from_l0star`, η = 0, n_mc = 16,384) runs at 0.29 s/locus. 29
  first-pass failures were GPU OOM from contention; they were re-run and all 600 are ok.

| Run (C trained on the L0* train stream) | 100 k | 300 k | 1 M | 2 M | 4 M | 8 M |
|---|---|---|---|---|---|---|
| `c1_pilot` (batch 256, cosine over 2 M; died at 1.2 M), class-balanced set | 0.281 | 0.239 | 0.126 | – | – | – |
| same, natural-mixture weighted | 0.086 | 0.075 | 0.041 | – | – | – |
| `c1_long` (batch 512, cosine over 8 M), class-balanced set | – | – | 0.220 | 0.178 | 0.143 | 0.111 (see addendum) |
| same, natural-mixture weighted | – | – | 0.091 | 0.064 | 0.052 | 0.048 (see addendum) |

Per class (`c1_long`, 4 M): KL on D 0.23, G 0.19, S 0.13, W 0.015. The hard classes carry the KL, and the
class-balanced set weights them about 4× more than the natural mixture does.

At 300 k loci, scored against the oracle labels:

| | Natural-weighted CE | Accuracy |
|---|---|---|
| B | 0.175 | 0.818 |
| C | 0.249 | 0.655 |

Correlation of C's class posteriors with B's: S 0.92, D 0.89, W 0.96, G 0.67.
- C is learning the Bayes posterior but is far from it at ≤ 1 M hard-label samples.
- 0.02 needs about one more order of magnitude of data, or a better-conditioned target (see §6.4).
- The first pilot run (`c1_pilot`) died of a CUDA OOM at 1.2 M loci (shared GPU). Training loops now wait and
  retry on OOM.

## 6. Deviations, open issues, proposals (PI decides; I edited no shared file)

1. **Two architectural additions to §5 C** (README C-A1, C-A2):
   - the BF-structured log-sum-exp readout;
   - the all-variant stream (C0 ablation: 2× lower KL).

   Sign equivariance is exact by allele canonicalisation instead of allele-flip augmentation (C-A3). Augmentation
   is kept for global trait-sign flips.
2. **No-LD ablation leak** (C-A4): the screen membership still comes from LD-aware conditional lists. It is a
   mild leak, so the +0.008 is a lower bound on LD's value.
3. **Data throughput is the bottleneck of full training.** About 10–13 loci/s per GPU worker; most time goes to
   the simulator's candidate draws. Proposals to the simulation agent:
   - (a) L3 PCs: fp32 or cached eigh. The fp64 4 k × 4 k eigh is about 0.27 s per admixed study and 26% of
     generation time.
   - (b) Fewer host–device synchronisations in `draw_candidate`/`observed_stats`.

   With (a)+(b), 4 workers should give > 100 k records/h. I re-use each candidate for 2 noise draws (C-T6).
4. **C1 at 0.02 KL looks expensive with hard labels.** Options (PI):
   - (i) larger budgets (8 M+ loci; the `c1_long` run is in progress, see `results/famC/c1_long/`);
   - (ii) Rao–Blackwellised soft targets (the oracle-class probabilities given the true effects, integrated over
     noise, are not available cheaply; B's posterior would be distillation, which defeats C1);
   - (iii) accept a relaxed bar with a learning-curve extrapolation.

   I recommend (i) and keeping the bar.
5. **Held-out L6.** L6c is also absent from training, because `train_api` lists only L0–L4 and L6. L6 runs
   0.019 below the L1–L4 pool on S-vs-D.
6. **`sim.train_api` was uncommitted.** It changed while I used it (16:46–17:05). My shards record sha256 of
   `src/sim`, `src/stats` and `src/famC` files. `valid_dev0_stale` (first attempt) was discarded because the
   generator changed under it.
7. **GPU sharing.**
   - My jobs exceeded the 12 GB guideline twice for some minutes: C0 at 9–11 GB plus generators.
   - One C0 ablation and one generator were stopped to comply.
   - Lesson proposed for LESSONS.md: run heavy GPU jobs one family at a time, or give each agent a hard
     `PYTORCH_CUDA_ALLOC_CONF`/process cap. Under contention, OOM killed a B run (29 loci) and a C1 run.
8. **Proposed lesson:** `pkill -f <pattern>` run from a shell whose own command line contains the pattern kills
   that shell (it happened twice). Launch queues from script files and kill by PID.

## 7. Next steps (for full training, after the simulation agent commits M3.5)

1. Regenerate `valid_v1` (validation windows, seed 20261029) and `train_v1` from the committed simulator: ≥ 500 k
   records, 4 workers.
2. Retrain the network and the floor on equal data. Report learning curves over data size (C2 needs
   train/validation gap < 0.02).
3. Evaluate through `src/eval` on screen_v2 with `famC.method.NeuralColoc`.
4. Run the pre-registered C1 when l0star_v1 and B's predictions exist.

## 8. Files

- **Code:**
  - `src/famC/{ldbank,c0world,features,model,train_c0,c0_analyze,simadapter,gen_shards,train_main,floor,method,c1,c1_pilot,compare_dev}.py`;
  - `src/famC/run_gen_dev.sh`;
  - `src/famC/README.md`.
- **Tests:** `tests/famC/{test_c0_features,test_method_adapter,test_simadapter}.py`.
- **Results:**
  - `results/famC/c0_full/` (config, curve, metrics, analysis, 5 M checkpoint);
  - `results/famC/c0_screen/`, `c0_snap2M/analysis.json`;
  - `results/famC/{main_dev,main_snap1_full,main_snap1_nold,floor_dev,floor_snap1}/`;
  - `results/famC/compare_*.json`;
  - `results/famC/c1_pilot/`, `results/famC/c1_long/`.
- **Data (git-ignored):**
  - `data/processed/famC/{valid_dev,train_dev,train_dev_snap1,l0star_pilot}/`;
  - `train_dev_snap1` holds hard links to the first 36 `train_dev` shards.
