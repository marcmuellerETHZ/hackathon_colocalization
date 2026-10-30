# src/sim — simulated world, generator, oracle, cell builder (owner: simulation agent)

Run with `source env/activate.sh` and `PYTHONPATH=src`. Inputs: `datasets/splits/pools_v1.tsv`, `datasets/splits/blocks_v1.tsv`,
`datasets/splits/genes_v1.tsv(.gz)`, `data/processed/variants/chrN.parquet` and the pgen files (all via `src/data`),
Beagle GRCh38 map `data/raw/genmap/chr_in_chrom_field/`.

| Module | Content | DESIGN v4 |
|---|---|---|
| `params.py` | every pinned constant (N_e, theta, N_syn, PVE ranges, grids, bins) | §1, §2 |
| `world.py` | genetic map interpolation, window variant set, source haplotypes / real genotypes | §2.1 |
| `mosaic.py` | Li–Stephens mosaic populations, materialised per window on the GPU | §2.1 |
| `validate_mosaic.py` | r² decay + effective rank vs source pool; output `validation/mosaic_validation*.{json,md}` | §2.1 |
| `architecture.py` | locus mixture S/D/W/M, causal MAF bins, D partner r² bins, PVE draws | §2.4 |
| `oracle.py` | power, `oracle_class` (precedence partition S>U>D>W>G), class-determining (μ1, μ2). **Hash-protected** | §1 |
| `generator.py` | pairing types P1/P2/P3, studies (factor / individual level), overlap, μ, z/β/se, LD per level L0–L6c (+ pool-swap), L4 corruption, ρ̂ | §2.2–2.4 |
| `cells.py` | rejection-sampling cell builder (FixedPlan screening buckets, GridPlan evaluation strata); screening set CLI | §2.4, §6.1 |
| `testset.py` | `generate(config, out_dir, seed)`: the 34-cell main grid + shifted arm (test only via scripts/eval_test.py; validation dry runs) | §7.3, §8.5, §8.6 |
| `train_api.py` | Family C on-the-fly training data (train split, full v4 grid, held-out L6) | §5 C |
| `validate_rho.py` | ρ̂ calibration / bias on realistic loci, estimator variants (`validation/rho_calibration*.{json,parquet}`) | §2.4, D6(d) |
| `pilot_strata.py` | natural class and D-stratum rates per pairing (sets the GridPlan relaxation thresholds) | §8.6 |
| `windows.py` | split windows (anchor = protein-coding TSS, clipped windows from `genes_v1`) | §3 |
| `l0star.py` | world L0* (B simulator-prior mode / C1): config prior (exact log-prior), 8-component λ mixture, builder | §5 B |
| `../stats/rho.py` | ρ̂: truncated bivariate-normal MLE (+SE), per-window and genome-wide modes | §2.4 |

Commands:
```
python -m sim.validate_mosaic --n-windows 40 --ne 500 1000 2000 3000 4000 8000      # pin N_e (pool A)
python -m sim.validate_mosaic --pool FIN --n-windows 40 --ne 250 500 1000 2000 3000  # other pools
python -m sim.cells --set screen_v2 --cells "L0|P1|r_p=0.0" "L0|P3|rho=0.5" ...     # screening set parts
python -m sim.cells --set screen_v2 --link-only                                       # master file + truth
python -m sim.testset --set valid_grid_v1 --seed 20261029 --n-per-class 300 --n-gu 100 --no-shifted --workers 4
python -m sim.l0star --set l0star_v1 --seed 20261029 --per-class 500                  # L0* (validation)
python -m sim.validate_rho --per-group 300                                            # rho-hat calibration
python -m sim.train_api --bench --procs 3 --windows 6 --lpw 64                        # Family C throughput
pytest tests/sim tests/stats                                                          # verification (§7.1)
```

## Locus-set layout (INTERFACES §2 + documented extensions)
* `data/processed/sets/<set>.h5`: one external link per locus to `<set>/<cell>.h5:/<locus_id>`; readers
  (`eval.io`) see ordinary locus groups. One part file per cell keeps writers independent (no per-locus files).
* Locus attributes: the INTERFACES list + `trait1_type`, `trait2_type` ("gwas"/"mol", needed for the
  type-dependent thresholds of DESIGN v4 §1) and, at L0 only, `ld_type2`/`n_ref2` with dataset `R2` or
  `X_ref2` = trait 2's in-sample LD when trait 2 is an individual-level study on different donors
  (DESIGN §2.3 "and of trait 2's study where relevant"; not in INTERFACES, see D11).
* Truth `<set>.truth.parquet` (eval.truth schema) + extra columns: causal variant ids, `locus_type`, `bucket`,
  `pve1/2`, N1/N2/N_s/r_p, populations, `rho_hat`/`rho_se`/`rho_mode`, `p`, `p_study`, L4 `flipped`,
  `lead_abs_z1/2`, `seed_key`. Causal indices refer to the locus' variant arrays (-1 = dropped at L4).
* Stats `<set>.stats*.json`: per-cell candidate class counts (natural class shares of the mixture), runtime.

## Decisions / assumptions (not fixed by DESIGN v4)
* **D1 Mosaic generated on the fly, not stored.** A mosaic population is a pure function of
  (seed 20261026, population key, pool, N_e, θ): per haplotype and chromosome a Poisson switch process in cM
  (drawn once per chromosome), sources uniform over the K pool haplotypes, miscopy flips drawn per (variant
  position) — so a variant's alleles do not depend on the window it is materialised in (tested). Materialising
  a 1-Mb window of 20,000 individuals takes ~1 s on the GPU; storage cost is zero (budget 2 TB / 50k files).
  The mosaic world uses seed 20261026 ("splits and pools") because it is part of the fixed world, shared by
  the screening, validation and test sets.
* **D2 N_e and θ.** θ = 1.4e-4 per site and haplotype (Li–Stephens default θ_W/(2(θ_W+K)) for K = 508), all
  pools. N_e pinned by the rule "largest grid value whose haplotype-r² decay (pool vs full mosaic, 40 random
  1-Mb windows genome-wide, windows with < 1,000 variants skipped) is within 5% (half the DESIGN tolerance)
  of the source pool in every distance bin" → **N_e = 3000 for pool A** (max |rel. err.| 4.8%, all bins
  within 10% for N_e ≤ 4000). Primary comparison is haplotype-level (pool A 508 haplotypes vs 40,000 mosaic
  haplotypes): the mosaic inherits pool A's *sample* LD, so this compares like with like. The dosage-level
  comparison with 254-subsamples double-counts sampling noise and is reported only. Effective rank of 1-Mb
  windows (254-subsample) 61.0 vs pool 61.9. Other pools use the same rule on their own source pool, grid
  {250, 500, 1000, 2000, 3000} (`validation/mosaic_validation_<pool>.json`; pairs restricted to variants with
  MAF ≥ 0.01 in the source pool): **FIN 500** (3.2%), **L6c 500** (2.9%), **ASW/ACB 1000** (4.2%),
  **pool B (pool-swap world) 1000** (3.4%). Smaller pools need fewer switches (rate 4N_e/K per Morgan:
  A 23.6, FIN/L6c 10.1, ASW/ACB 12.7, B 13.3).
* **D3 Variant set per locus.** Window variants = target-pool QC set (pool-A MAF ≥ 0.01); then every study
  sample must have MAF ≥ 0.01 (in-sample for individual-level studies, population AF for factor form) and the
  reference must be polymorphic (L1–L6). Causal variants are drawn inside this analysis set. L4 then drops 5%
  (causal variants can be dropped; truth index -1) and flips the z sign of 0.5% of variants in one random trait.
* **D4 Effect scale (D6(b)) and reported statistics (D6(a)).** PVE q drawn log-uniform; β_std = ±√q (var(y) = 1),
  λ = √N β_std = √(Nq) with q the per-variant PVE (decision D6(b); DESIGN's "q²" is the same quantity). Residual
  variance 1 − var(Xβ); total PVE > 0.5 → redraw the PVEs (up to 50 times, else the candidate is discarded).
  **OLS statistics (D6(a))**: individual-level studies report the actual per-SNP regression of y on the ALT
  dosage with intercept (+ 2 PCs at L3): beta = OLS slope, se = √(RSS/df)/‖g_res‖, z = beta/se (the t-statistic;
  verified against an explicit numpy regression, tests/sim/test_ols_and_v4.py). Factor-form studies: z_gen = μ +
  noise (score scale √N·r), beta = z_gen·se0, se0 = 1/√(2Nf(1−f)), se = se0·√(1 − z_gen²/N), z = beta/se, so
  N·se² + beta² = 1/(2f(1−f)) (sdY = 1). The oracle μ for factor form stays √N·R_pop·β as DESIGN §1 says; at the
  class thresholds (|μ| ≤ 6.3, N ≥ 20k) the OLS transform changes E[z] by < 0.1%. L4 allele flips change the sign
  of beta (and z). Background (20% of loci): λ_bg ~ N(0, 2/p) per variant, **independent between traits**.
* **D5 Thresholds follow trait type** (DESIGN v4 §1): P1/P2 trait 1 GWAS-like / trait 2 molecular; P3 eQTL×eQTL
  both molecular, lab×lab both GWAS-like (also the PVE range follows the type).
* **D6 L3 admixed study.** 85% target mosaic + 15% ASW/ACB mosaic (3,000 synthetic individuals); z adjusted for the
  top 2 PCs of the study sample computed from a genome-wide panel of 4,000 SNPs (MAF ≥ 0.05 in pool A and
  ASW/ACB, ≥ 500 kb apart; `sim/pcs.py`, cached in data/processed/sim/);
  same standardized effects in both groups. L3 is only combined with ρ = 0 (DESIGN v4).
* **D7 Pairing draws.** P1: N1 ∈ {20k,100k,400k}, N2 ∈ {100,300,1k,5k}, nested vs disjoint 50:50 unless the cell
  fixes it; P2: N1 = 100k, N2 = 50k (v4), r_p = ρ√(N1/N2) asserted ≤ 0.9; P3: eQTL×eQTL (N ∈ {300,1000}, same
  donors) vs lab×lab (N = 100k, factor) 50:50, sign of ρ random (validation/training).
* **D8 Class-determining (μ1, μ2)** stored as absolute values; for G (undefined in §1) the strongest causal
  variant per trait, as for W.
* **D9 D partner.** The D partner is drawn uniformly within a uniformly chosen r² bin of DESIGN v4 §8.6
  ([0,.1), [.1,.3), [.3,.5), [.5,.8), [.8,.95)); nearest non-empty bin if empty. Causal MAF bins
  (0.01–0.05, 0.05–0.2, 0.2–0.5) use the target-population MAF; the D partner ignores MAF bins.
  W "sub-powered shared variant": |λ| ~ U(0.5, τ) of the other trait's type (can be weak or grey).
* **D10 Windows.** Anchors are protein-coding genes of the split (clipped windows of `genes_v1`); windows with
  < 200 variants are skipped. Each visit generates K candidates per unfilled cell and accepts ≤ M per bucket.
* **D11 L0 LD.** Factor-form trait 1: dense R_pop (N_syn = 20,000 rows) restricted to the locus variants
  (= R_true); individual-level trait 1: its standardized in-sample genotypes (factor if N < p). Methods that
  take one R get trait 1's; trait 2's in-sample LD is added as `R2`/`X_ref2` (extension).
* **D12 Screening set.** Cells {L0, L2, L4} × {P1 r_p = 0, P3 ρ = 0.5}; buckets S, D r² < 0.5, D 0.5 ≤ r² < 0.8, W;
  500 each; P1 nested/disjoint 50:50 (irrelevant at r_p = 0); P3 sub-type 50:50 and ρ = ±0.5. screen_v2 (M3.5)
  repeats this design with world sim_v2 (OLS statistics, D14 ρ̂, v4 trait types); screen_v1 is superseded (D6(a)).
* **D13 L0\*.** p = **20** SNPs (D6(g); was 30) with MAF ≥ 0.05 from a random 100-kb segment of a split window; R from
  the NFE mosaic; R̃ = (1−η)R + ηI with η = 1e-3, stored as float32 and z generated from exactly the stored matrix
  (z = R̃λ + e, e ~ N(0, V⊗R̃)); no background, no M loci; config prior = §2.4 S/D/W renormalised, uniform D
  partner, W = one trait with one variant; λ mixture σ = {0.5, …, 64}, weights fitted by EM to the main generator's
  λ (P1); n_ref attribute 1e9 (population LD); `rho_hat` = true ρ, `rho_se` = 0 (true ρ given). L0* is a z-world:
  beta/se use nominal N (1e5, 1e3) and z is NOT OLS-transformed (B's model must hold exactly). Set l0star_v1:
  validation, seed 20261029, S/D/W/G 500 each (U too rare to fill: 0.5% naturally), true ρ uniform over
  {0, ±0.3, ±0.5, ±0.8}; stats hold the natural class shares with binomial MC SE for the class-prior correction.
* **D14 ρ̂ (D6(d), refined; validation/rho_calibration.md).** Per-window mode (P1, P3 eQTL×eQTL): (1) mask every SNP
  with max(|z1|,|z2|) > 4 and every SNP with r² > 0.1 to one (LD given to methods); (2) only when the LD is accurate
  (L0 in-sample, or n_ref ≥ 1,000, i.e. L1) the strong-signal SNPs are **residualised** out of both traits
  (`stats.rho.residualize`: conditioning set = signal SNPs chosen greedily with r² < 0.8, ≤ 30; z_res = (z − R_kC
  R_CC⁻¹ z_C)/√(1 − R_kC R_CC⁻¹ R_Ck); the null is preserved exactly); (3) **DESIGN §2.4 pruning kept** (greedy
  r² < 0.1 over all window SNPs, random order); (4) unit-variance truncated MLE on |z| < 2; (5) SE = delete-one-
  block jackknife over 10 contiguous blocks of the used SNPs. Why (measured on 9,000 + 7,500 + 1,600 validation
  loci): without pruning the composite MLE has ~50% larger SD and the jackknife SE is 1.6–2.7× too small (the
  mosaic's 1-Mb windows have effective rank ~60–80, so LD spans the whole window); a free-variance (3-parameter)
  MLE removes a variance-inflation bias but raises the SD by 40–80% (rejected); masking alone leaves a concordant-
  sharing bias of +0.04–0.08 and SEs 2× too small in P3 eQTL; residualisation removes it with accurate LD (P3 L0:
  SD 0.19 → 0.10, concordant-sharing bias +0.081 → +0.011) but makes it worse with the 150-person reference
  (hence the n_ref rule). Genome-wide mode (P2, P3 lab×lab): 20,000 independent SNPs, 1% with λ ~ N(0, 2²)
  independently per trait, same mask, MLE and jackknife (bias ≤ 0.005, SE ratio 0.97–1.02). Pooled mode
  (`rho_hat_pooled`, real data per cohort pair across windows, D6(d)): the windows are the jackknife blocks.
* **D15 Oracle-power test criterion (approved, D6(c)).** |z| ≤ 3 per point and a joint χ² p > 0.01 (per-point 2-SE
  outcomes recorded in `validation/generator_tests.json`).
* **D16 Storage.** LD is stored as a factor X_ref (float32) whenever n_ref < p, else dense float32 R (never fp16).
  Loci whose LD is a deterministic function of (window, level, LD source, variant set) — factor-form studies at L0
  and every reference level — share it within a part file through eval.io `ld_key` HDF5 hard links (readers see
  an ordinary `R`/`X_ref`). L0 dense R of P1 loci cannot be shared (locus-specific variant set from in-sample
  MAF filters).
* **D17 Evaluation-grid plan (`cells.GridPlan`).** S, W fixed counts; D stratified uniformly over strata "as
  feasible": strata from the config (`per_cell.D_generation_strata.by`; datasets/configs/testset_v1.yaml uses the 5 r² bins
  of §8.6; `r2xd` = 5 r² × 4 d bins is available); G and U kept as they arise up to the cap **each**. Deterministic
  staged relaxation in the cell's candidate index k: k < 100·n_D strict per stratum; then per r² bin (any d); from
  200·n_D any D (incl. r² ≥ 0.95). Why: many r² × d strata are structurally empty (d < 4 needs r² ≥ 0.5; d ≥ 16 at
  r² ≥ 0.8 needs λ ≥ 36; `validation/pilot_strata.json`, 20,000 candidates per pairing). K = 200 candidates per
  cell and window visit, M = 3 accepted per bucket and visit (same for the test and validation grids).
* **D18 Shifted arm.** Cells and counts from the config (`shifted_arm.cells`: L1swap/L2swap × P1 nested r_p ∈ {0,
  0.6}, 1,500 each); locus types S .15 / D .15 / W .5 / M .2, biobank-strong share 0.5, background share 0.5, M loci
  2–4 causal per trait (`params.SHIFTED_*`); every candidate kept (no rejection on the oracle class: the shifted
  class mix is part of the shift); random draws rng(20261028, "shifted", set, cell, …). Variant set = the target-
  pool (A) QC set filtered by the in-sample MAF of the studies.
* **D19 Test lock.** `sim.testset.generate` refuses split "test" unless `eval.io.has_test_access()` and the call stack
  contains scripts/eval_test.py; `windows.split_windows("test")` refuses without access; writers go through the
  eval.io guard. Validation runs require `dry_run=True` and split "valid".
* **D20 oracle.py change (hash to be re-frozen by the PI).** Trait types are now required arguments of
  `oracle_class` (no default "gwas"/"mol"); unknown types raise; docstring says v4. Logic unchanged (same labels on
  the 1e5-draw partition test). Reason: a P3 caller could otherwise silently get the P1 thresholds.
* **D21 L3 PCs.** Top-2 PCs via the eigendecomposition of the smaller Gram matrix (fp64; same span as SVD, ~50×
  faster: the SVD took 12 s per study); the fixed L3 factor-form population's PCs are computed once and cached.
* **D22 Validation-only locus types** "N" (no causal) and "Sc" (shared, concordant signs) exist in
  `architecture.py` for the ρ̂ study; they are never in a set's mixture.
* **D23 Family C training API (`train_api.py`).** Train split only (asserted); levels L0–L4 and L6, with L6 held out
  by default (`holdout=("L6",)`); pairing P1/P2/P3 with weights 0.5/0.25/0.25; overlaps P1 r_p ∈ {0, .3, .6}
  (nested/disjoint 50:50), P2 ρ ∈ {0, .1, .2, .3}, P3 ρ ∈ {0, .3, .5, .8} both signs; L3/L6 with 0 overlap; §2.4
  mixture without class rejection; seeds rng(20261030, "train", step, …). Default screen = per trait M/2 − M/8
  SNPs by marginal |z| and M/8 by lead-conditional |z| (Family B's rule, without duplicate collapse); a
  `screen_fn` hook takes Family B's screen.
