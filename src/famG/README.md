# Family G — GPU coloc-SuSiE engineering baseline (DESIGN §5 "Engineering baseline G")

Owner: Family A/G agent. Tests: `tests/famG/` (`python -m pytest tests/famG`).

| File | Content |
|---|---|
| `susie_rss.py` | Batched GPU SuSiE-RSS reproducing susieR 0.14.2 `susie_rss(z, R, n, L = 10)` (and the `n`-missing path) |
| `ser_kernel.py` | Fused Triton kernel: per-effect prior-variance optimisation (exact port of R's `Brent_fmin`, keep-old rule, null check), one program per locus |
| `brent.py` | Pure-torch batched port of R's `Brent_fmin` (CPU fallback; bit-identical to R `optim(method="Brent")` on test functions) |
| `ld.py` | Batched LD operators: dense R `(B,P,P)` or genotype factor X `(B,n,P)` with R = XᵀX/n_ref |
| `coloc.py` | GPU `combine.abf`, `coloc.bf_bf`, coloc.abf ABFs (coloc 5.2.3) |
| `method.py` | `GPUColocSusie` Method (INTERFACES §4): SuSiE-RSS per trait → bf_bf over CS effects → §4.3 adapter (`eval.adapters.coloc_susie`), coloc.abf fallback |
| `locus.py` | STUB `LocusData` + HDF5 loader (written before `src/eval/io.py` existed; Methods duck-type and also accept `eval.io.LocusData`) |
| `precision.py` | TF32 off + assertion, LD dtype guard (no fp16/bf16) |
| `validation/` | 1000G EUR loci simulator (`make_loci.py`, `genotypes.py`), R reference runners (`run_susie014.R`, `run_coloc_susie014.R`), comparison and timing scripts |

## Reproduce the identity test and timing
```
source env/activate.sh; export PYTHONPATH=src R_PROFILE_USER=/dev/null; W=<work dir>
python -m famG.validation.make_loci --out $W/identity --n-loci 140 --p-set 200,500,1000,2000 --key identity
Rscript src/famG/validation/run_susie014.R $W/identity $W/identity/r014/t1_c0 1 140 1 100 1     # and trait 2 (t2_...)
python -m famG.validation.compare_identity --work $W/identity --rdir $W/identity/r014 --out reports/famAG_M3/identity_G --dtype float64
Rscript src/famG/validation/run_coloc_susie014.R $W/identity $W/identity/coloc014/c0.json 1 140
python -m famG.validation.compare_coloc --work $W/identity --rjson-dir $W/identity/coloc014 --out reports/famAG_M3/identity_G
python -m famG.validation.make_loci --out $W/timing_d5000 --n-loci 24 --p-set 5000 --factor-frac 0 --key timing_d5000
python -m famG.validation.timing --dirs $W/timing_d5000 --out reports/famAG_M3/timing_G_dense_fp32.json
```
Seeds: all simulated loci use `common.seeds.rng(VALID_SEED, "famAG", key)` (20261029 = validation draws).

## Decisions / assumptions
1. **Reference semantics = susieR 0.14.2 `susie_rss` defaults** (the standard pipeline of DESIGN §4.1 item 2): PVE-adjusted z, XtX = (n−1)R, standardize = TRUE, residual variance fixed to 1, scaled_prior_variance 0.2, `estimate_prior_method = "optim"`, `check_null_threshold = 0`, `prior_tol = 1e-9`, `max_iter = 100`, `tol = 1e-3` with 0.14.2's rule (stop when ΔELBO < tol, negative Δ included), `check_prior = TRUE` (reported as status `failed_prior_check` instead of an R error), prior weights with susieR's `log(1/p + sqrt(eps))` quirk, coverage 0.95, min |r| ≥ 0.5.
2. **Purity of credible sets with > 100 members** is computed exactly over all members; susieR samples `n_purity = 100` members with R's RNG (not reproducible outside R). Impure large sets are dropped in both; no disagreement occurred on 280 fits.
3. **Brent**: the prior-variance optimiser is an exact port of R's `Brent_fmin` (same bracket [−30, 15] on log V and tol √ε); the GPU kernel accumulates the log-sum online over variable blocks, so objective values differ from R by ~1e-16 relative. On flat optima this can move V by ~1e-6 relative (PIP effect < 1e-7).
4. **Credible-set ties**: variables in perfect LD have mathematically equal α; floating-point rounding (batch-dependent BLAS) can pick a different member at the coverage boundary than R. The identity report gives both strict CS equality and equality up to α-ties (relative 1e-8).
5. **Bucketing**: loci are sorted by p and grouped so that p_max ≤ 1.10·p_min within a bucket and the LD of the bucket fits a 6 GB budget (no padding to the global p_max). Padded variables get prior weight 0 (exactly zero posterior mass); tested.
6. **Precision**: everything except LD products is float64; LD products run in float32 by default (TF32 off, asserted) or float64 (`ld_dtype`).
7. **Method**: one LD per locus is used for both traits (INTERFACES §2 stores one R / X_ref); n per trait = median of the per-SNP `n1`/`n2`. coloc priors p1 = p2 = 1e-4, p12 = 5e-6 (bf_bf). **coloc.abf fallback (M3.5, aligned with the tier-1 wrapper `src/baselines/methods.py:ColocSusie`)**: it uses the same p12 as bf_bf (`p12_abf=None` → `p12`, so tuning p12 on validation (D6(g)) moves both, as in the wrapper), W = 0.15 (`W_abf`), and coloc.abf's `adjust_prior` (p ← 1/(nsnps+1) when nsnps·p ≥ 1; coloc.bf_bf does not adjust, as in coloc 5.2.3). Equality with `baselines.abf.coloc_abf` and the eval adapter is tested (incl. 12,000 SNPs, p12 ∈ {5e-6, 1e-5}). Fallback beta/se come from the locus (`beta`, `se`), else se = 1/√(2N f(1−f)) with sdY = 1 (DESIGN §2.2). `coloc.bf_bf`'s overlap trimming is a no-op because both traits share all SNPs. `runtime_s` = batch wall time / number of loci (amortised).
8. `LocusData` stub (`locus.py`): kept only for tests; delete once everyone uses `eval.io.LocusData`.
9. `BatchLD.diag()` for factor LD is computed in chunks of ≤ 2 GiB float64 temporaries (M3.5; it allocated B·n·p·8 bytes at once and hit OOM at 256 loci × 503 × 2,000 on the shared GPU).
