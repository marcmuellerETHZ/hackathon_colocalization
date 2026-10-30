# M3 report — simulation engineer (src/sim, src/stats, tests/sim, tests/stats)

Commits: cebb51a (modules + tests), 9009646 (OOM-safe builder; **screen_v1 generated at this commit**), bbd0804
(per-pool N_e, PC panel, level tests, set metadata). Decisions not fixed by DESIGN v4: `src/sim/README.md`
D1–D16 and `src/stats/README.md`.

## 1. What was built

| Item | Where | Status |
|---|---|---|
| Li–Stephens mosaic populations | `src/sim/mosaic.py`, `world.py` | done; on-the-fly GPU generation (no storage), deterministic, window-independent (tested) |
| Mosaic validation + N_e pinning | `src/sim/validate_mosaic.py`, `validation/mosaic_validation*.{json,md}` | done for pools A, FIN, L6c, ASW/ACB, B |
| Generator §2.2/§2.3 | `src/sim/generator.py` | P1 (nested/disjoint), P2 (100k/50k), P3 (eQTL×eQTL individual, lab×lab factor); L0, L1, L2, L3 (2 PCs), L4, L6, L6c, pool-swap L1swap/L2swap |
| Causal architecture §2.4 | `src/sim/architecture.py` | S/D/W/M mixture, MAF bins, D partner r² bins (v4 edges), PVE draws, PVE cap 0.5 |
| Oracle §1 | `src/sim/oracle.py` (hash-protected; unchanged since the PI froze it, sha d562789c…) | trait-type thresholds |
| Rejection-sampling cell builder | `src/sim/cells.py` | INTERFACES HDF5 via `eval.io`, truth via `eval.truth` |
| L0* world | `src/sim/l0star.py` | config prior with exact log-prior, 8-component λ mixture, builder |
| ρ̂ | `src/stats/rho.py` | truncated BVN MLE + SE; per-window and genome-wide modes |
| L3 PC panel | `src/sim/pcs.py` | 4,000 genome-wide SNPs, cached in data/processed/sim |
| Set summary | `src/sim/summarize_set.py` | realism / composition tables |

**Mosaic.** Primary criterion = haplotype r² of the source pool vs the full mosaic (the mosaic inherits the
pool's *sample* LD, so this compares like with like); rule: largest grid N_e with |rel. err.| ≤ 5% in every bin
(40 random 1-Mb windows genome-wide). Pool A (target NFE): **N_e = 3000**, rel. err. −0.5 / −0.9 / −1.8 / −4.8% in
0–10 / 10–50 / 50–100 / 100–500 kb (all bins within 10% up to N_e = 4000); effective (entropy) rank of 1-Mb
windows, 254-individual subsample 61.0 vs pool 61.9 (full 20k mosaic: 82.8). FIN 500, L6c 500, ASW/ACB 1000,
pool B 1000. θ = 1.4e-4. Storage decision: nothing stored; a 1-Mb window × 20,000 individuals materialises in ~1 s
on the GPU.

## 2. Tests (all run; `PYTHONPATH=src pytest tests/sim tests/stats` → **56 passed in 78 s**, log `logs/sim/pytest_final.log`)

| DESIGN §7.1 | Test | Result |
|---|---|---|
| 1 mean/cov factor vs individual | `test_factor_vs_individual_mean_cov` (all 20k rows, 2,000 reps, whitened cov) | max standardized mean diff 3.06 over 155 SNPs; whitened-cov RMS 1.003 (factor), 1.008 (indiv) (expected 1) |
| 1 realised ρ | `test_realised_rho` (600 null reps per mode) | P2 0.3→0.304±0.017 (slope on R 0.300); P3-lab −0.8→−0.803; P1 nested 0.190→0.183, 0.134→0.143; disjoint 0→−0.025±0.016; P3-eQTL 0.5→0.515±0.017 |
| 1 in-sample LD noise | `test_in_sample_ld_noise` (N = 100/300/1000, 300 resamples) | empirical/delta-method variance 1.04 / 0.97 / 1.04 |
| 1 oracle power = detection | `test_oracle_power_equals_detection` (1,000 reps × 6 μ × 2 forms) | 10/12 points within 2 SE; indiv μ=2 z=2.58, factor μ=3.58 z=2.08; joint χ² p = 0.26 / 0.14; 10,000-rep confirmation at those two points: z = 1.23, −1.25. See deviation D15 |
| 2 oracle partition | `test_partition_1e5_draws` + known answers + independent re-implementation | exactly one label each; shares S 0.050, U 0.0002, D 0.101, W 0.730, G 0.120 (random-draw stress distribution, not the mixture) |
| 3 ρ̂ unbiased | `test_genomewide_unbiased` (200 reps, 20k SNPs, 1% polygenic) | bias +0.0005, −0.0021, −0.0038, −0.0046 at ρ = 0, .3, .5, .8 (all < 0.01); SE calibrated (ratio 0.88–1.01) |
| 8 truth isolation | `test_truth_isolation` | only INTERFACES datasets/attrs (+ documented extensions); ρ̂ ≠ true ρ; at ≥ L1 the given LD differs from R_true (checked by rebuilding R_true) |
| 10 TF32 off / precision | `test_tf32_off`, `test_fp32_mu_matches_fp64` | asserted; fp32 vs fp64 μ < 1e-4 |
| other | mosaic determinism/window consistency/AF/switch rate; L4 drop 5% + flip 0.5%; deterministic rebuild; test split refused; all levels incl. L3/L6/L6c/swap/P2; L6 allelic conversion; PCs separate ancestry (|corr| > 0.9); L0* prior normalised + sampler χ² fit, z ~ N(R̃λ, V⊗R̃) | pass |

## 3. Screening set (DESIGN §6.1)

`data/processed/sets/screen_v1.h5` (master with external links) + `screen_v1/` (12 part files, 152 GB) +
`screen_v1.truth.parquet`; sha256 in `data/processed/sets/SHA256SUMS` (copy `src/sim/validation/screen_v1.SHA256SUMS`):
screen_v1.h5 `82d28203…6f99`, truth `93d26442…d191`. Validation split only (3,769 protein-coding anchor windows,
~295 visited), seed 20261027, 12,000 loci = {S, D r²<0.5, D r² 0.5–0.8, W} × 500 × {L0, L2, L4} × {P1 r_p=0, P3 ρ=±0.5}.
No test-split loci were generated (`split_windows("test")` raises). **Runtime 30 min wall** (3 GPU processes;
~58k candidates per cell, 65–95 candidates/s per process; 6.7 accepted loci/s overall).

**Natural class shares of the mixture** (candidates, per cell): S 0.061–0.067, D 0.074–0.091, W 0.71–0.73,
G 0.13, U 0.001–0.002. Bucket acceptance: D r² 0.5–0.8 only ~1%.

**Realism** (`src/sim/validation/screen_v1_summary.md`): lead |z| median [5–95%] — S/D buckets trait 1 ≈ 11–17
[6–72], trait 2 ≈ 10–13 [5.5–41]; W ≈ 3.7–4.1 [2.8–34]. Causal per trait: 1 in ~50–80%, 2–3 otherwise (M type).
D r²<0.5 bucket: r² [0,.1) ≈ 65%, [.1,.3) ≈ 18%, [.3,.5) ≈ 15%; d mostly 4–16 (d ∈ [2,4) only in the r² 0.5–0.8
bucket, ~24%). p median ≈ 2,700 variants per locus.

## 4. Deviations from DESIGN / open issues — PI decisions needed

1. **λ definition (§1 vs §2.2 text):** implemented β_std = ±√q (q = PVE), λ = √(Nq); §1's "β_std² = q_j²" and
   §2.2's "√(Nq²/(1−q²))" read as typos (README D4). Please confirm/patch the text.
2. **Oracle-power test criterion (D15):** per-point ±2 SE over 12 points fails by chance ~43% of the time; test now
   asserts |z| ≤ 3 per point + joint χ² p > 0.01. Needs your approval (I did not change seeds).
3. **Per-window ρ̂ SE is ~2× too small:** in screen_v1 per-window ρ̂ has SD 0.15 (W) to 0.22–0.25 (D/S) vs mean
   reported SE 0.108; bias ≈ 0 (|bias| ≤ 0.007). Methods that shrink ρ̂ by its SE will over-trust it. Options:
   block-jackknife SE, or report an inflation factor. Genome-wide mode is well calibrated (SD 0.007, SE 0.006).
4. **In-sample LD noise for layer E (Family A):** normal-theory Var(r̂) = (1−r²)²/N is wrong for genotypes —
   median per-pair ratio empirical/normal 0.71–0.84, but ~10× too small near r² ≈ 1; the delta-method with
   genotype 4th moments matches (0.97–1.04).
5. **INTERFACES extensions:** attrs `trait1_type`/`trait2_type` (needed for type-dependent thresholds) and, at L0,
   `R2`/`X_ref2` + `ld_type2`/`n_ref2` for trait 2's in-sample LD (§2.3 "where relevant"). Master file uses HDF5
   external links to per-cell parts.
6. **Choices to confirm:** background effects independent between traits; W "sub-powered shared" |λ| ~ U(0.5, τ);
   P1 nested/disjoint 50:50 and P3 eQTL/lab 50:50; D partner uniform within a uniform r² bin, ignoring MAF bins;
   G's class-determining μ = strongest per trait; anchors = protein-coding genes; L0* uses a uniform D partner,
   W = one trait/one variant, n_ref attribute 1e9.
7. **Storage:** L0 dense LD is 30–60 MB/locus (screen_v1 = 152 GB); the 34-cell test set will be several hundred GB
   if L0 cells keep dense R. Fine under the 2 TB budget but I/O-heavy for methods.
8. Not done: an L0* evaluation set was not frozen (builder + dry-run only; suggest seed 20261029 when B/C need it);
   the R1 real-data comparison belongs to the real-data agent (simulated distributions above are ready for it);
   `sim.testset:generate` entry point named in configs/testset_v1.yaml does not exist yet.
