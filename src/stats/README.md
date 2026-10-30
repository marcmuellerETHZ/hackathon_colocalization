# src/stats — statistical helpers (owner: simulation agent)

| Module | Content | DESIGN v4 |
|---|---|---|
| `rho.py` | ρ̂ (decision D6(d), refined): signal mask, optional signal residualisation, LD pruning, truncated bivariate-normal MLE, block-jackknife SE; per-window, genome-wide and pooled (across windows) modes; 3-parameter (free variances) MLE for diagnostics | §2.4 |

Power functions live in `src/sim/oracle.py` (`power(mu, alpha)`), because they are part of the hash-protected oracle.

API:
```python
from stats.rho import rho_hat_window, rho_hat_pooled, rho_hat_genomewide
rho_hat_window(z1, z2, R, order=None, residualize_signal=False)  # per window; R = LD given to methods (dense)
rho_hat_pooled([(z1, z2, R), ...])                              # one cohort pair over many windows (real data)
```
Each returns a dict with `rho_hat`, `rho_se` (jackknife), `n_used`, `n_masked`, `n_blocks`, `mode`.

## Decisions / assumptions
* Estimator (full rationale and validation numbers: `src/sim/README.md` D14, `src/sim/validation/rho_calibration.md`):
  mask = SNPs with max(|z1|, |z2|) > 4 plus SNPs with r² > 0.1 to one of them (D6(d)); pruning at r² < 0.1 over all
  window SNPs (DESIGN §2.4; z-independent, random order); |z| < 2 in both traits; unit marginal variances; SE by
  delete-one-block jackknife over 10 contiguous equal-count blocks of the used SNPs (windows as blocks in pooled
  mode when ≥ 10 windows).
* `residualize_signal=True` (projection of strong signals out of both traits with the given LD) only when the LD is
  accurate (in-sample, or a reference of ≥ 1,000 individuals); with a 150-person reference it increases the error.
  Real-data callers with external reference panels should leave it False.
* The MLE is solved on a fixed grid (step 5e-4 on (−0.995, 0.995)) with parabolic refinement (vectorised over
  jackknife replicates); agreement with the exact Brent solution < 1e-6 (tested).
* Truncation constant c = 2; the square probability is a 96-node Gauss–Legendre integral (error < 1e-12).
  Fewer than 10 usable SNPs → NaN.
* Genome-wide mode simulates 20,000 independent SNPs from the study's true ρ, 1% of them with independent
  λ ~ N(0, 2²) per trait (no genetic correlation), then applies the same estimator (bias ≤ 0.005, SE ratio
  0.97–1.02 at ρ ∈ {0, .3, .5, .8}).
* Known limitation (reported): per-window SEs remain 1.5–1.8× too small for P3 eQTL×eQTL loci with signal under a
  small reference (L2), and concordant sharing biases ρ̂ upwards by +0.03–0.07 there.
