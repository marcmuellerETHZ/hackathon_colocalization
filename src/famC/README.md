# Family C — NeuralColoc (bold family; owner: Family C agent)

Amortised neural posterior estimation for colocalization (DESIGN v4 §5 C). Package `famC` (`PYTHONPATH=src`,
`source env/activate.sh`). Tests: `python -m pytest -q tests/famC` (GPU).

| Module | Content |
|---|---|
| `ldbank.py` | real 1000G pool-A genotype windows (train / valid split; test refused) on the GPU, sorted by p |
| `c0world.py` | C0 world: coloc.abf's own model on real LD windows + batched fp64 coloc.abf targets |
| `features.py` | GPU screen (= famB screen without collapse), two-stage features `extract` (needs LD) / `build` (augmentation, allele canonicalisation) |
| `model.py` | network: LD-biased transformer over screened variants + all-variant stream + BF-structured log-sum-exp readout; 5-class head + shared-PIP head |
| `train_c0.py` | C0 kill-test training / learning curve |
| `simadapter.py` | adapter to the simulation agent's on-the-fly TRAIN API (`sim.train_api.TrainSampler`), record compaction, shards |
| `gen_shards.py` | generate train / validation record shards (`data/processed/famC/<set>/`) |
| `train_main.py` | main-task training (oracle classes S, D, W, G, U), monitoring metrics |
| `floor.py` | LightGBM floor on hand-built features (incl. coloc.abf and GPU SuSiE summaries) |
| `floor_method.py` | `FloorColoc(model_dir)`: the floor as a harness Method (same record path as NeuralColoc) |
| `method.py` | `NeuralColoc(checkpoint).fit_batch(list[LocusData])` (INTERFACES §3/§4) |
| `c1.py`, `c1_pilot.py` | C1: training in world L0* (sim.l0star, train windows) and truth-free predictions; pilot L0* validation draw. Evaluation vs B: `scripts/famC_eval/c1_eval.py` |
| `c0_analyze.py`, `compare_dev.py` | C0 error breakdown; paired locus bootstrap between development runs |

Commands (all logs in `logs/famC/`, outputs in `results/famC/<run>/`):
```
python -m famC.train_c0 --run c0_full --steps 40000                      # C0, full model
python -m famC.train_c0 --run c0_screen --variant screen --steps 12000   # C0 ablation: no all-variant stream
python -m famC.gen_shards --set valid_v1 --split valid --n-loci 30000 --accept '{"S":1,"D":1,"U":1,"G":0.5,"W":0.3}'
python -m famC.gen_shards --set train_v1 --split train --n-loci N --worker k --n-workers K
python -m famC.train_main --run main_v1 --train-set train_v1 --valid-set valid_v1
python -m famC.floor --run floor_v1 --train-set train_v1 --valid-set valid_v1
# phase 2 (reports/p9.famC_C2.md)
python -m famC.gen_shards ... --device cuda|cpu [--step-offset S]            # CPU path benchmarked: GPU 2.6x faster
python -m famC.train_main ... --stop-after 700 --pilot-sizes N:steps,...     # timing pilot (DESIGN_P2 §2 C)
python -m famC.floor featurize --set train_v1 --worker k --n-workers K --device cpu   # floor features on CPU
python scripts/famC_eval/build_trainregion.py --set trainregion_m4_v1          # train-block M4-design gap set
python scripts/famC_eval/run_harness.py ... [--variant nold | --floor-dir DIR]  # net / ablation / floor (FloorColoc)
python scripts/famC_eval/c2_eval.py ...                                         # C2 verdict (paired block SE)
```

## Decisions / assumptions (not fixed by DESIGN v4 or decisions D4/D6)

### Architecture
- **C-A1 BF-structured readout.** Each stream emits K = 16 per-variant energies quadratic in allele-invariant
  statistics, e_k(j) = a_k + Σ_m c_km q_m(j) with q = (z1², z2², z1z2, zc1², zc2², zc1·zc2)/2 (the form of a
  Gaussian single-variant log BF), coefficients from the variant's embedding; pooled by log-sum-exp (a
  permutation-invariant pooled readout, §5 C). Logits = linear(E) + MLP(tanh(linear(E)/8), pooled embeddings,
  globals). Why: posterior class odds are ratios of sums of BFs; a plain MLP readout cannot represent z² at
  |z| ≈ 100 to the precision needed (KL < 0.01), while the quadratic form extrapolates exactly.
- **C-A2 all-variant stream.** Besides the transformer over the M = 128 screened variants (§5 C), a per-variant
  MLP over *every* window variant (no attention; linear in p) feeds the same readout, so evidence outside the
  screen (e.g. the H0/H1 mass of thousands of null variants in coloc.abf) is not lost. Ablated in C0
  (`--variant screen`).
- **C-A3 sign equivariance** by exact allele canonicalisation (each screened variant re-coded so the trait with
  larger |z| is positive; LD bias uses s_i s_j r_ij), tested to be exactly invariant to allele flips; global
  trait-sign flips (z_t → −z_t, ρ̂ → −ρ̂) by augmentation. DESIGN says "allele-flip augmentation": exact
  invariance makes augmentation over alleles unnecessary.
- **C-A4 no-LD ablation** (`--variant nold`): zero all LD-derived inputs (conditional z, LD with leads, LD bias,
  eigen-spectrum, lead–lead r, r² in the all-variant stream). The screen membership is still the LD-aware screen
  (its conditional lists) — a mild LD leak, stated so the ablation is not over-read.
- Screen: identical to famB's per-trait-union screen with `collapse=False` (M = 128, k_cond = 16, η = 1e-3),
  identity tested in fp64. Global features: ρ̂, its SE, N1, N2, p, n_ref, trait types (GWAS-like / molecular;
  needed because the oracle thresholds depend on trait type), max |z|, counts of |z| > 5.45, lead–lead r,
  effective rank and top-12 eigenvalues of the screen LD.
- Model: d = 128, 4 layers, 4 heads, K = 16, ~1.04 M parameters; fp32 (TF32 off).

### C0 world (`c0world.py`)
- **C0-1** LD: real pool-A genotypes (n = 254) of clipped gene windows (MAF ≥ 0.01 in pool A, the sim world's
  variant rule; variants with constant dosage dropped — one all-heterozygous artefact was found); 2,998 train
  windows (seed 20261030) for training, 400 validation windows (seed 20261029) for evaluation.
- **C0-2** N1, N2 ~ log-uniform [300, 3×10⁵] independently; se = 1/√(2Nf(1−f)); β ~ N(0, W²), W = 0.15,
  λ = β/se; causal positions uniform; z = λ R e_j + X'u/√n (so z ~ N(Rλe_j, R)). Training hypotheses uniform
  over H0–H4; evaluation on two distributions: uniform and coloc's own prior (prior odds 1 : p·p1 : p·p2 :
  p(p−1)p1p2 : p·p12). Target = coloc.abf PP (p1 = p2 = 1e-4, p12 = 1e-5, coloc defaults) over the full window,
  computed in fp64 and tested against `baselines.abf.coloc_abf` (< 1e-10).
- **C0-3** Criterion as DESIGN: E[KL(coloc ‖ net)] < 0.01 and 99th percentile |ΔPP4| < 0.05 on validation
  windows, required on both evaluation distributions.
- **C0-4** Seeds: training draws from one CUDA generator seeded derive(20261030, "famC", "C0train", run);
  evaluation draws derive(20261029, "famC", "C0eval", ...), identical at every evaluation.

### Main task (`simadapter.py`, `train_main.py`)
- **C-T1** Training data: `sim.train_api.TrainSampler` (simulation agent; train-split windows; levels L0–L4,
  **L6 held out**; pairing types and overlap grids of §2.4 with both ρ signs; seed 20261030) with Family C's
  per-locus record step. L6c is not in TrainSampler's level list either, so the held-out FIN mechanism has no
  founder-number proxy in training.
- **C-T2** Class balance: the training data follow the natural mixture (no rejection) and the loss uses class
  weights (S 1, D 1, W 0.12, G 0.4, U 1); outputs are corrected by −log weight to natural-mixture posteriors.
  The validation set (`valid_v1`, validation windows, seed 20261029, all levels incl. L6) keeps candidates with
  probability S 1, D 1, U 1, G 0.5, W 0.3 (ranking metrics are unaffected).
- **C-T3** Storage: records keep z-type columns in fp32 and other per-variant columns and the screen LD in fp16
  (round-trip tested); shards of 2,048 loci in `data/processed/famC/<set>/` (git-ignored data area).
- **C-T4** Shared-PIP target: the class-determining shared variant's token for S loci (or the screened token
  with r² ≥ 0.8 to it in the inference LD), else "none"; loss weight 0.1.
- **C-T5** Monitoring metrics in `train_main` (power-matched S-vs-D AUPRC per level via
  `eval.metrics.power_match_weights`, D-vs-W AUROC) are for development only; claims use the eval harness.
- **C-T6** Noise re-draws: each simulator candidate (study samples, causal architecture, effects, oracle class)
  can yield `--n-noise` loci with independent noise draws (observed statistics, LD at the level, ρ̂). The oracle
  class depends only on the candidate, so these are valid draws from the generator, correlated through the
  candidate; it raises throughput because the candidate draw dominates generation cost.
- **C-T7** Development data (`train_dev`, `valid_dev`) come from the simulator while it was being revised for M3.5
  (uncommitted); shard configs record sha256 of `src/sim`, `src/stats`, `src/famC`. Comparisons use a frozen
  hard-linked snapshot (`train_dev_snap1`, 36 shards) so every run sees the same records. Final training must
  regenerate from the committed simulator.
- **C-T8** Shared GPU: training loops wait 30 s and retry the same batch on CUDA OOM (never skip a batch).
- **C1** Training in L0* (p4.famC_C1, `c1.py` docstring): segments from train windows only (windows with >= 200
  variants, as build_set; `sim.l0star.make_segment` defaults: p = 20, 100 kb, MAF >= 0.05, R~ with eta = 1e-3),
  several segments per window visit, loci from `sample_locus` with rho uniform over `sim.l0star.L0STAR_RHO`
  (the l0star_v2 grid) and the natural class mixture; inputs exactly as `sim.l0star.build_set` writes them
  (N 100k / 1k nominal, se from MAF, LD = R~, n_ref = 1e9, rho-hat = rho, rho_se = 0 - the L0* world gives methods
  the true rho as rho-hat); hard labels; unweighted CE, so NO logit correction at inference (`method.NeuralColoc(
  class_w=None)`; `c1.predict_l0star` picks it from the run config). Earlier runs (c1_pilot, c1_long) used rho in
  {0, 0.4}; their M3.5 evaluations wrongly applied the main-task correction (reports/famC_M3.5_addendum.md).
- **C1 evaluation** reads truth (class shares, oracle labels) and lives in `scripts/famC_eval/c1_eval.py`, outside
  the method package (DECISIONS P3-21). The pilot set (`c1_pilot.py`) is NOT a pre-registered set.
- **True rho.** `features.extract` asserts that a batch carries no `rho` field; the network reads only rho_hat /
  rho_se (DESIGN §2.4). `simadapter` puts the true rho only into the label/monitoring meta.
- Floor features and settings: `floor.py` docstring. SuSiE summaries are computed on the screened variants
  (with their inference LD), not the full window (the records keep only the screen LD).
