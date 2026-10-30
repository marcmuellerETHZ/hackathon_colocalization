# M3.5 report — simulation engineer (src/sim, src/stats, tests/sim, tests/stats)

Commits: 3416f2e (code + tests), 850218b (READMEs, ρ̂ summary, testset CLI), fb426c4 (LDSource level), 6bb00eb
(screen_v2 / l0star_v1 metadata), no separate metadata commit for valid_grid_v1 (session ended; facts filled by the PI on 2026-10-27 from data/processed/sets/valid_grid_v1.stats.json, the truth table and reports/p2.verify_artifacts.md). All decisions not fixed by DESIGN v4 /
D4 / D6 are in `src/sim/README.md` (D4, D12–D23) and `src/stats/README.md`.

## 1. D6 items

| Item | What was done | Evidence |
|---|---|---|
| (a) OLS statistics | Individual level: beta/se of the actual per-SNP regression of y on the ALT dosage with intercept (+2 PCs at L3), z = beta/se. Factor form: beta = z_gen·se0, se = se0·√(1 − z_gen²/N), z = beta/se. L4 flips flip beta. | `tests/sim/test_ols_and_v4.py`: max rel. error vs explicit numpy OLS < 2e-3 (fp32 GPU) for NFE and ADMIX(+PCs); factor identity N·se² + b² = 1/(2f(1−f)) exact; screen_v2 spot check (300 loci): 0 identity failures, beta/se = z everywhere |
| (b) λ = √(Nq) | Code already used it; docstrings/README/oracle header fixed | test checks |λ| = √(N q) for PVE-specified effects |
| (c) oracle-power criterion | kept (|z| ≤ 3 per point, joint χ² p > 0.01), marked approved | `test_oracle_power_equals_detection` passes (both forms) |
| (d) ρ̂ | implemented, then **refined after verification** (below) | `src/sim/validation/rho_calibration.md` |

### ρ̂: what the verification showed (validation split; 9,000 + 7,500 + 1,600 realistic loci; types N null, S shared random signs, Sc shared concordant signs, D, W; P1 disjoint, P1 nested r_p 0.6, P3 eQTL×eQTL ρ 0.5; L0/L1/L2)
1. **D6(d) literally (mask, no pruning, jackknife)**: SD 50% larger than the pruned estimator and jackknife SE
   1.6–2.7× too small for every block count (10/20/40). Cause: the mosaic's 1-Mb windows have effective rank
   ~60–80, so LD spans the whole window and contiguous blocks are not independent. → **DESIGN §2.4 pruning kept.**
2. Pruned + mask + jackknife (10 blocks): calibrated on null loci (ratio SD/SE 0.95–1.11 for P1, 1.11–1.26 for P3),
   but concordant sharing still biases ρ̂ by **+0.03 to +0.09** and P3 signal loci have SE 1.6–2.0× too small:
   SNPs beyond the r² > 0.1 mask still carry LD-propagated signal.
3. A free-variance (3-parameter) MLE removes a variance-inflation bias but raises SD by 40–80% → rejected.
4. **Residualisation** (project the |z| > 4 signals out of both traits with the given LD; exact null preserved;
   `stats.rho.residualize`) fixes it when the LD is accurate: P3 L0 SD 0.19 → 0.10, Sc bias +0.081 → +0.011,
   P1 L0 Sc bias +0.05 → +0.006; at L1 (n_ref 2,000) Sc bias +0.089 → +0.060, W ratio 1.47 → 1.20; with the
   150-person reference (L2) it makes things worse. → used only at L0 (in-sample LD) or n_ref ≥ 1,000.

**Final estimator** (README D14): mask (D6(d)) → residualise if LD accurate → prune r² < 0.1 → |z| < 2 →
unit-variance truncated MLE → delete-one-block jackknife (10 blocks). Genome-wide mode (P2, P3 lab): bias ≤ 0.005,
SE ratio 0.97–1.02 at ρ ∈ {0, .3, .5, .8} (400 reps). Per window, final vs M3 estimator (bias / SE ratio):

| | null (N) | shared concordant (Sc) | D / W / S |
|---|---|---|---|
| P1, L0 | ±0.011 / 0.99–1.07 | +0.006 / 1.02–1.03 (M3: +0.04–0.06) | ≤ 0.035 / 0.94–1.15 |
| P1, L1–L2 | ≤ 0.008 / 0.95–1.04 | +0.015 to +0.059 / 1.09–1.12 | ≤ 0.027 / 1.04–1.24 |
| P3 eQTL, L0 | 0.001 / 1.21 | +0.011 / 1.26 (M3: +0.081 / 2.16) | ≤ 0.008 / 1.16–1.34 (M3 1.8–2.2) |
| P3 eQTL, L1 | 0.007 / 1.25 | +0.060 / 1.56 | ≤ 0.012 / 1.20–1.50 |
| P3 eQTL, L2 | 0.002 / 1.11 | **+0.069 / 1.78** | ≤ 0.031 / **1.61–1.83** |

**Not met:** SE within 15% for P3 eQTL×eQTL loci with signal at L1/L2 (and P3 null 1.21–1.25); unbiasedness
under concordant sharing at L1/L2 (+0.03 to +0.07). Unbiased under true sharing with random effect signs
(|bias| ≤ 0.031 everywhere). On real data, `rho_hat_pooled` (windows = jackknife blocks) is the remedy (D6(d)
pooled mode); I recommend the real-data arms use it as primary.

## 2. DESIGN v4 conformance
- Trait-type thresholds: oracle logic already per type; **oracle.py changed** (README D20): trait types are now
  required arguments (a P3 caller can no longer get P1 thresholds silently), header v4. Labels unchanged (1e5-draw
  partition test, independent re-implementation). **PI: please re-freeze its hash.**
- P3 eQTL×eQTL: mol/mol thresholds and PVE range [1e-3, 0.3]; lab×lab gwas/gwas (verified per locus in screen_v2).
- P2 N1 = 100k, N2 = 50k, r_p = ρ√2 asserted ≤ 0.9; L3/L6 raise unless ρ = 0; total PVE ≤ 0.5 with redraw (tested).
- L0*: p = 20 (D6(g)), z generated with the stored R̃, no background/M loci, true ρ given; natural class shares with
  binomial MC SE stored (class-prior correction).
- Shifted arm: pool-swap world, L1swap/L2swap × P1 nested r_p ∈ {0, 0.6} (4 × 1,500, from the config), types
  S .15/D .15/W .5/M .2, biobank-strong 50%, background 50%, up to 4 causal/trait, rng(20261028, "shifted", …),
  every candidate kept. Validation dry run (400 loci): oracle S 8.0%, D 6.5%, W 68.8%, G 16.8%; background 0.50,
  strong 0.49 → ~480 S and ~390 D expected in the 6,000-locus test arm.

## 3. Test-set entry point `sim.testset:generate(config, out_dir, seed)`
Reads configs/testset_v1.yaml (eval agent's v4 version): 34 cells (checked against `main_grid.cells`), D strata
from `per_cell.D_generation_strata` (r² bins), G and U up to the cap each, P3 positive ρ on test, both signs on
validation. **Refuses** split "test" unless `eval.io.has_test_access()` *and* scripts/eval_test.py is on the call
stack; `split_windows("test")` also refuses; dry runs only with `split="valid", dry_run=True`
(`tests/sim/test_testset.py`, incl. a chr22 dry run of 2 cells + shifted arm). No test loci were generated.
GridPlan (README D17): strict per-stratum quotas, relaxed deterministically after 100·n_D candidates (r² bin) and
200·n_D (any D), because many r²×d strata are structurally empty (`validation/pilot_strata.json`).

## 4. Locus sets (all validation split; sha256 in data/processed/sets/SHA256SUMS and src/sim/validation/*.SHA256SUMS)

| Set | Seed | Loci | Size | Wall time | Notes |
|---|---|---|---|---|---|
| screen_v2 | 20261027 | 12,000 (6 cells × S, D r²<.5, D r² .5–.8, W × 500) | 119 GB (v1: 152 GB) | 59 min, 3 GPU processes (GPU shared, 100% busy) | master `6cc05e1c…`, truth in SHA256SUMS; natural shares S .063–.067, D .073–.092, W .71–.73, G .13, U .001–.002 |
| l0star_v1 | 20261029 | 2,000 (S/D/W/G × 500), p = 20 | 28 MB | 2 min | h5 `3b660558…`, truth `e3d52800…`; natural S .052, D .066, W .783, G .094, U .005 (MC SE ≤ .004) |
| valid_grid_v1 | 20261029 | 35,086 (34 cells; S/D/W 10,200 each, G 3,400, U 1,086) | 231 GB | 8,090 s wall (2.25 h) | built at code fb426c4. Re-generation of 102 sampled loci (3 per cell) is bit-identical at master 334e182 (p2.verify_artifacts). sha256 in data/processed/sets/SHA256SUMS. Registered valid_grid_v1@1 ok |

screen_v2 realism (`validation/screen_v2_summary.md`): lead |z| S/D buckets trait 1 median 11–16 [5%: 6, 95%: 40–75],
trait 2 10–13; ρ̂ − ρ SD: genome-wide 0.007, per window 0.05–0.07 (P3 L0, residualised) to 0.15–0.22.

## 5. Family C training API (`sim.train_api.TrainSampler`)
Train split only (asserted); full v4 grid with L6 held out by flag; §2.4 mixture without rejection; per locus a
top-M screen (Family B's per-trait union rule) and tensors z1, z2, MAFs, N, n_ref, R block [M×M], ρ̂/SE,
class label, per-SNP causal/shared/proxy-r² labels, truth scalars; deterministic in (seed, step); OOM-tolerant on
the shared GPU (`tests/sim/test_train_api.py`). **Throughput: 8–14 loci/s per process, peak 2.6 GB GPU**, measured
while other agents kept the GPU at 100% (`validation/train_api_bench_1proc.json`; profile: GPU-sync bound, ~2 s
Python per 64 loci). Family C already builds on it (`famC/simadapter.py`); `rho_hat_for` now infers the level
from the LD source, so their calls without `level` get the same residualisation rule.

## 6. Tests
`PYTHONPATH=src pytest tests/sim tests/stats` → **78 passed** (330 s) at 3416f2e; later targeted re-runs pass.

## 7. Requests / proposals for the PI
1. Approve the ρ̂ refinement (pruning kept, residualisation under accurate LD, jackknife 10 blocks) as an amendment
   of D6(d); real data should use pooled mode as primary. Known residual miscalibration for P3 eQTL at L1/L2.
2. Re-freeze `src/sim/oracle.py` hash (D20).
3. configs/testset_v1.yaml: close open item 1 (entry point exists; signature `generate(config, out_dir, seed)`,
   K = 200, M = 3); update `generator.rho_hat` text; open items 2–4 are implemented as the YAML says.
4. screen_v1 (152 GB) is superseded and can be deleted once nobody reads it.
5. INTERFACES §2 extensions (unchanged since M3): `trait1_type`/`trait2_type`, L0 `R2`/`X_ref2`.
