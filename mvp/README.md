# mvp: mismatch-aware coloc-SuSiE, and the benchmark recompute

The minimal usable deliverable of the project. It has two parts:
1. `run_coloc_mm.R`: the recommended colocalisation pipeline, with validated calling thresholds, for your own
   GWAS / QTL summary statistics at one locus.
2. `recompute_benchmark.py`: recomputes every simulated test-set number of the benchmark from the shipped
   predictions in `results/`.

## What the method is
**Mismatch-aware coloc-SuSiE** (`coloc_susie_mm` in the benchmark) is an **existing method**, not one the
project developed:
- per trait, susieR 0.16.6 `susie_rss(L = 10, R_finite = n_ref, R_mismatch = "eb_mix")`, i.e. SuSiE fine-mapping
  with susieR's model of the mismatch between the LD reference panel and the study;
- coloc 5.2.3 `coloc.bf_bf` on the credible-set pairs (p1 = p2 = 1e-4, p12 = 5e-6), as in `coloc.susie`;
- a coloc.abf fallback when a trait has no credible set;
- a three-way call: shared (S) if P_S ≥ t_S = 0.8402818292911564; else distinct (D) if
  P_D ≥ t_D = 0.7322585039086077; else insufficient.

The script is a thin, faithful wrapper around susieR and coloc. None of the four method families the project
built (JointSuSiE, ExactColoc and its mismatch-aware version, a GPU coloc-SuSiE, NeuralColoc) beat it.

## Key results
Test set: 13,207 held-out simulated loci, one single-use run. Realistic setting = 27 mismatched-LD cells. AUPRC
for a shared causal variant vs two distinct well-powered variants (S vs D):

| Method | AUPRC S-vs-D, realistic [95% CI] | AUPRC S-vs-D, in-sample LD |
|---|---|---|
| **mismatch-aware coloc-SuSiE** (susieR 0.16.6) | **0.900** [0.884, 0.914] | 0.962 |
| coloc.abf | 0.860 [0.846, 0.874] | 0.856 |
| coloc-SuSiE (susieR 0.14.2) | 0.717 [0.697, 0.741] | 0.966 |

- LD-reference mismatch, not sample overlap, is what breaks fine-mapping-based colocalisation. The mismatch
  model recovers most of the loss.
- At the validated thresholds (M4 validation sample, 12,170 loci): recall S 0.78, recall D 0.80; 2.5% of
  too-weak (W) loci called shared; 9.7% of distinct loci called shared.
- It is slow: about 12–16 CPU-seconds per locus.

Full account: `../REPORT.md` and `../manuscript/`.

## Install
- **R 4.5** (the project used 4.5.3 from conda-forge) with **coloc 5.2.3** (CRAN / conda-forge `r-coloc`) and
  jsonlite.
- **susieR 0.16.6** from GitHub `stephenslab/susieR`, commit `97feba561dda5402fb9d70fabc715cb49030101f`. Install it
  into a **separate library**: the benchmark also used susieR 0.14.2 for standard coloc-SuSiE, and coloc 5.2.3
  was built against 0.14.x.

  ```bash
  git clone https://github.com/stephenslab/susieR && git -C susieR checkout 97feba561dda5402fb9d70fabc715cb49030101f
  mkdir -p rlib_susie016 && R CMD INSTALL --library=rlib_susie016 susieR   # needs Rfast and cpp11armadillo
  ```

  Then run with `R_LIBS=rlib_susie016`, so that this library comes before the default one, where coloc and
  jsonlite live.
- **Version check.** The script stops if `packageVersion("susieR") != "0.16.6"` or
  `packageVersion("coloc") != "5.2.3"`, unless `--allow-other-versions` is given.
- **Recompute and tests:** Python ≥ 3.10 with numpy, scipy, pandas, pyarrow and pytest (pinned versions in
  `../requirements.txt`).
- The full project environment (lock files, every R package with its commit) is in `../docs/environment/`.

## Usage
Run from the repository root.

```bash
R_LIBS=/path/to/rlib_susie016 Rscript mvp/run_coloc_mm.R \
    --sumstats locus.sumstats.tsv.gz \
    --ld-factor locus.ld_factor.txt.gz \
    --n-ref 150 --rho-hat 0.40 \
    --out out/locus
# --ld locus.ld.txt.gz (p x p LD, or .rds) instead of --ld-factor
```

On the shipped example (about 3–5 s per locus):

```bash
R_LIBS=/path/to/rlib_susie016 Rscript mvp/run_coloc_mm.R \
    --sumstats mvp/example/L2_S_locus.sumstats.tsv.gz \
    --ld-factor mvp/example/L2_S_locus.ld_factor.txt.gz --n-ref 150 --rho-hat 0.3959701144286405 \
    --out /tmp/L2_S_locus
# call=S P_S=0.922766 P_D=0.00596514 fallback=FALSE -> /tmp/L2_S_locus.json, /tmp/L2_S_locus.tsv
```

## Outputs
Without `--out`, the JSON result goes to stdout. With `--out PREFIX`:
- `PREFIX.json`: the full result:
  - `P_S`, `P_D`, `P_W` (shared / distinct / too weak), `s` = max PP.H4/(PP.H3+PP.H4);
  - `call` (S / D / insufficient), `status` (ok / nonconverged), `fallback` (and its posteriors);
  - `thresholds`;
  - per-trait SuSiE summaries (`trait1`, `trait2`) and the credible-set `pairs` with their posteriors;
  - `inputs` (including `rho_hat`), `config`, package `versions`, R `warnings` and `elapsed_s`.
- `PREFIX.tsv`: one summary row: P_S, P_D, P_W, s, call, status, fallback, n_cs1, n_cs2, n_variants, n_ref,
  rho_hat.

**Caveat: ρ̂ is accepted but unused.** `--rho-hat` is recorded in the output and has no effect on the result.
Neither coloc nor susie_rss takes a sample-overlap input, and the benchmarked configuration did not use one
either. In the benchmark, sample overlap barely changed this method's error rates once LD was handled (details
under Inputs below).

## Recompute the benchmark
```bash
python mvp/recompute_benchmark.py          # = python scripts/recompute_benchmark.py; about 1 min
# REPRODUCED: 28363 values compared with benchmark.json/descriptive.json and 264 summary.md cells;
#             mismatches 0; max |diff| 1.67e-15
```
It reads the test run's per-locus predictions (`results/test_v1/runs/`) and the simulation truth
(`results/truth/`), and checks each predictions file's sha256 against its run manifest. It then recomputes,
with the project's own `src/eval/metrics.py`:
- realistic, ideal and shifted AUPRC S-vs-D and AUROC D-vs-W;
- every per-cell, per-level and stratified metric;
- the three-way calls at the frozen operating points.

It does this for all methods and both failure-scoring modes, on the main benchmark (13,207 loci) and the
ColocBoost subsample (4,467 loci). Every value is compared with the run's `benchmark.json`, `descriptive.json` and
`summary.md` (tolerance 1e-6), and the output goes to `results/recompute/`. Bootstrap CIs (2,000 block
replicates) are not recomputed; they are in `results/test_v1/benchmarks/sim/summary.md`. Options:
`--methods`, `--modes`, `--out`, `--no-write` (`--help`).

## Tests
```bash
COLOC_BENCH_RSCRIPT=/path/to/Rscript COLOC_BENCH_RLIB_SUSIE016=/path/to/rlib_susie016 python -m pytest mvp/tests
```
- `test_pipeline.py`: the pipeline reproduces the benchmark's stored predictions for the two example loci
  (to 1e-6), and rejects misaligned LD. It is skipped unless Rscript with susieR 0.16.6 and coloc 5.2.3 is found.
- `test_recompute.py`: the wrapper reproduces the benchmark on a subset (coloc.abf and this method,
  least-favourable mode) with 0 mismatches; the script's thresholds equal the validated operating point; the
  headline AUPRCs are those in `results/`.

## Inputs

| Argument | Content |
|---|---|
| `--sumstats` | Tab-separated, optionally gzipped, one row per variant, header required. Columns: `snp` (unique id), `beta1`, `se1`, `n1`, `beta2`, `se2`, `n2`; optional `maf1`, `maf2` (used only by `--estimate-sdy`). Trait 1 and trait 2 must be aligned to the same effect allele, in the same variant order as the LD. |
| `--ld` | p x p LD correlation matrix in the same variant order. Whitespace- or comma-separated text (optionally gzipped), optionally with a header line of SNP ids (checked against `snp`), or an `.rds` matrix. Must be symmetric to 1e-6; it is symmetrised as 0.5 (R + Rᵀ). |
| `--ld-factor` | Instead of `--ld`: an n_ref x p matrix X of standardised reference genotypes, same formats. The LD is R = XᵀX / n_ref, as the benchmark computed it for factor-form loci. |
| `--n-ref` | Size of the LD reference panel. Passed to susieR as `R_finite`. For in-sample LD, give the study's own sample size. |
| `--rho-hat` | Estimated sample-overlap correlation ρ̂. **It is recorded in the output but not used.** The benchmark's `coloc_susie_mm` has no overlap input, because neither coloc nor susie_rss takes one (src/baselines/README.md, decision 3). On the test set, with ρ̂ unused, the pre-registered C4 slopes of the false-S rate on realised ρ (P2/P3 cells) stayed within ±0.02 per 0.1 of ρ for every method, this one included (REPORT.md §5.2; results/test_v1/benchmarks/sim/c4.json). |
| `--estimate-sdy` | Optional. In the coloc.abf fallback, estimate sdY from MAF, N and varbeta (coloc's own estimate) instead of assuming sdY = 1. This is not the benchmark configuration. |
| `--allow-other-versions` | Run even if the susieR or coloc versions differ from 0.16.6 / 5.2.3. A warning is issued, and the results may then differ from the benchmark. |

The summary statistics should be ordinary OLS (or logistic) estimates: z = beta / se is passed to `susie_rss` together with n = the median of the per-variant N. The simulation generated OLS statistics; see src/baselines/README.md, decision 1.

## Exact configuration and its provenance

All file references are relative to the repository root.

| Step | Setting | Source |
|---|---|---|
| Per trait | `susieR::susie_rss(z = beta/se, R, n = median(N), L = 10, R_finite = n_ref, R_mismatch = "eb_mix")`; every other argument at the susieR 0.16.6 default (`max_iter` 50 for RSS, PIP convergence when `R_finite` is active) | src/baselines/methods.py `ColocSusie._fit` (l. 183) and `ColocSusieMM` (l. 214); src/baselines/r/worker.R `fit_susie` (l. 44); src/baselines/locus.py `scalar_n` (l. 103), `dense_ld` (l. 78) |
| Colocalisation | credible-set rows of `lbf_variable` from each fit, then `coloc::coloc.bf_bf(bf1, bf2, p1 = 1e-4, p2 = 1e-4, p12 = 5e-6)`, as in `coloc.susie` | src/baselines/r/worker.R `h_susie_coloc` (l. 72, 88) |
| Scores | rows with any non-finite PP are dropped; P_S = max PP.H4; P_D = (1 − max PP.H4) · max PP.H3; P_W = max(0, 1 − P_S − P_D); s = max over rows of PP.H4/(PP.H3+PP.H4) | src/eval/adapters.py `coloc_susie` (l. 71) |
| Fallback | if either trait has no credible set, or no finite CS-pair row is left: coloc.abf with prior SD W = 0.15 (sdY = 1), p1 = p2 = 1e-4, p12 = 5e-6; then P_S = PP.H4, P_D = PP.H3, P_W = PP.H0+PP.H1+PP.H2 | src/baselines/methods.py l. 202; src/baselines/abf.py; src/eval/adapters.py `coloc_abf` (l. 59) |
| Call | S if P_S ≥ t_S; else D if P_D ≥ t_D; else insufficient. The E-gate does not apply, because the method reports no E1/E2 | src/eval/metrics.py `calls_from_arrays` (l. 561) |
| Thresholds | t_S = 0.8402818292911564, t_D = 0.7322585039086077 | results/phase2_validation/m4_valid_baselines_v3/operating_points_p2.json (results/baselines_m4/m4_valid_v3/ in the original project) |

**Thresholds.** The thresholds were fitted on the M4 validation sample of the simulation (12,170 loci, realistic cells L1–L6). The rule was the pre-registered matched-W operating point: confident-S calls on too-weak (W) loci ≤ 2.5%. Its two-sided OP2 version (also ≤ 10% S calls on distinct loci) gave the same values. At these thresholds, on M4:
- recall S 0.782, recall D 0.803;
- W called S 2.5% (and W called D 2.5%);
- false-S on distinct (D) loci 9.7%.

These values were independently recomputed in reviews/p9_verify_mmM4.md.

## Equivalence with the benchmark

The two example loci come from the project's simulated **validation** split (`mvp/example/*.meta.json`). The benchmark's stored predictions for them are in results/phase2_validation/coloc_susie_mm_M4/. This script reproduces those predictions exactly, with absolute difference 0.0 in P_S, P_D, P_W and s:

| Example | Locus | Cell | Oracle class | Path | P_S | P_D | Call |
|---|---|---|---|---|---|---|---|
| `L2_S_locus` | valid_grid_v1__L2_P3_rho=0.8__S__0009230 | L2, P3, ρ = 0.8 (n_ref 150 external reference) | S | SuSiE + coloc.bf_bf | 0.92277 | 0.00597 | S |
| `L2_W_fallback_locus` | valid_grid_v1__L2_P3_rho=0.8__W__0009204 | L2, P3, ρ = 0.8 | W | coloc.abf fallback (no CS) | 0.00314 | 0.00570 | insufficient |

`mvp/tests/test_pipeline.py` checks this to 1e-6. `mvp/example/make_example.py` regenerates the example inputs from the project's HDF5 locus set; that needs the original project data, which is not shipped.

## Caveats
- **The thresholds are fitted on simulated validation data.** They reproduced the intended rates on the simulated test set (W→S 3.6% with the earlier screen-set thresholds; REPORT.md §5.3). They are not calibrated on real data. Real-data arms tested only relative behaviour.
- **sdY = 1 in the fallback.** The coloc.abf fallback assumes standardised traits, as in the simulation. For traits on another scale, use `--estimate-sdy` or standardise beta and se.
- **ρ̂ is not used** (see Inputs).
- **Runtime.** The median was about 12–16 CPU-s per locus on the benchmark (p ≈ 1,000–5,000 variants), and 15.8 worker-s per locus on average. The LD-mismatch model makes it the slowest of the well-performing methods. coloc.abf is the fast alternative.
- **susieR warnings.** susieR 0.16.6 often warns "Possible summary statistics and R reference mismatch detected" with a small reference panel. The warning is recorded in `warnings`. It is expected, and it is the situation the mismatch model handles.
- **Convergence.** `status` is `nonconverged` when either fit did not converge. The benchmark still used these outputs (11.6% of M4 loci, 1,410 of 12,170).
