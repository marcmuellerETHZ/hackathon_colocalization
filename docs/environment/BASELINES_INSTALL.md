# Baselines: install record (DESIGN.md §4)

Activate first: `source /cluster/scratch/mumarc/hackathon_eqtl_gwas/env/activate.sh; export R_PROFILE_USER=/dev/null`
(nothing is written to $HOME). Locks: `env/requirements.lock` (python), `env/renv.lock` (conda R env),
`env/R_packages_pinned.tsv` (R packages, source, commit, library path). Source clones: `env/src/` (git-ignored).
Example scripts: `scripts/baseline_checks/`; outputs: `results/baseline_checks/`.

**Important env change.** Installing compiled R deps (Rfast, mashr, lme4, ...) via micromamba made the solver upgrade
the R env from 4.4 to **R 4.5.3**. coloc 5.2.3, susieR 0.14.2, mashr 0.2.79 were retained/re-linked and work. Anything
in env/renv built before this was already rebuilt by conda. Re-run `coloc`/`susieR 0.14.2` wrappers on R 4.5.3 only.

## Library layout (select with R_LIBS; R_LIBS comes before the main library)
| Use | R_LIBS value |
|---|---|
| default (susieR 0.14.2, coloc, colocboost, colocPropTest, prop.coloc, Rfast, mashr) | (unset) |
| susieR 0.16.6 | `$P/env/rlib_susie016` |
| mvsusieR (needs susieR >= 0.15.54) | `$P/env/rlib_susie016:$P/env/rlib_mvsusie` |
| hyprcoloc | `$P/env/rlib_hypr_dep:$P/env/rlib_hyprcoloc` |
Never load both susieR versions in one R session; use one process per library setting. (`$P` = project root.)

## Summary
| Method | Version / commit | Library | Example check |
|---|---|---|---|
| coloc.abf, coloc.susie / coloc.bf_bf | coloc 5.2.3 | main | PASS (`coloc_susie014.txt`) |
| susieR (standard) | 0.14.2 | main | PASS (same file) |
| susieR (mismatch-aware) | 0.16.6, 97feba5 | rlib_susie016 | PASS (`susieR_0.16.txt`) |
| mvsusieR | 0.3.0, 2b05b9a | rlib_mvsusie | PASS (`mvsusieR.txt`) |
| colocboost | 1.0.10 (CRAN = GitHub 9f82c2a) | main | PASS (`colocboost.txt`) |
| colocPropTest | 0.9.3 (CRAN; f8dfc43) | main | PASS (`colocPropTest.txt`) |
| prop.coloc | 2.0.0, ash-res/prop-coloc d7f0938 | main | PASS (`prop_coloc.txt`) |
| SharePro_coloc | 059caab (v5.0.0, 2024-06) | env/src + .venv | PASS, reproduces shipped output (`SharePro.txt`) |
| Rfast | 2.1.5.2 | main | PASS |
| msprime 1.4.4, stdpopsim 0.3.0, pyarrow 26.0.0, lightgbm 4.7.0, xgboost 3.4.1, pgenlib 0.95.1 | pip | .venv | PASS (`python_stack.txt`; pgenlib write/read round-trip ok) |
| gpu-coloc (tier 2) | 0.7.1 (PyPI; repo 8d23d6a) | .venv | PASS (`gpu_coloc.txt`) |
| hyprcoloc (tier 2) | 0.0.2, 0348bbd | rlib_hyprcoloc | PASS (`hyprcoloc.txt`) |

## coloc 5.2.3 (and coloc-SuSiE with susieR 0.14.2)
- Install: conda-forge `r-coloc` (CRAN). Library: main.
- `coloc.abf(dataset1=list(beta, varbeta, N, MAF, type="quant"|"cc", snp, position[, sdY]), dataset2=..., p1=1e-4, p2=1e-4, p12=1e-5)`.
  Alternatively `pvalues`+`MAF`+`N`. **The prior effect SD W is hard-coded** in `coloc:::approx.bf.estimates` (0.15*sdY for quant, 0.2 for cc; no argument).
  DESIGN's W-tuning over {0.1,0.15,0.2} therefore needs a local wrapper (re-implement the 3-line ABF with chosen W, then `coloc:::combine.abf`, or pass precomputed lBF via `coloc.bf_bf`). p12 is an argument.
  Output: `$summary` (nsnps, PP.H0..PP.H4.abf), `$results` (per-SNP).
- coloc-SuSiE: `S <- runsusie(D)` (D has `beta, varbeta, N, MAF, LD` [LD matrix named by snp], `snp`, `type`; `runsusie` calls
  `susie_rss`, forwarding `L`, `coverage`, `repeat_until_convergence` ...) then `coloc.susie(S1, S2, p12=5e-6)`;
  output `$summary` rows per CS pair with hit1, hit2, PP.H0-H4.abf, idx1, idx2 (`$summary` is NULL when a trait has no CS -> fallback to coloc.abf).
  Direct `susie_rss(z, R, n, L=10)` then `coloc.bf_bf(bf1, bf2)` using `S$lbf_variable` (rows = effects).
- LD-mismatch: none in coloc itself. Sample overlap: none (assumes independent samples).
- susieR 0.14.2 `susie_rss(z, R, n, bhat, shat, var_y, z_ld_weight, estimate_residual_variance, prior_variance, check_prior, ...)`
  (no R_finite/R_mismatch).

## susieR 0.16.6 (separate library)
- Install: `R CMD INSTALL --library=env/rlib_susie016 env/src/susieR` (GitHub HEAD 97feba5, DESCRIPTION Version 0.16.6; r-universe stephenslab
  reports the same version/commit). Needs Rfast + cpp11armadillo (installed in main lib). Call with `R_LIBS=$P/env/rlib_susie016`.
- Exact `susie_rss` signature found (args): z, R, n, X, bhat, shat, var_y, L, maf, maf_thresh, scaled_prior_variance, residual_variance,
  prior_weights, null_weight, standardize, estimate_residual_variance, estimate_residual_method (MoM|MLE|NIG), z_method (wald|score),
  estimate_prior_variance, estimate_prior_method (optim|EM|simple), prior_variance_grid, mixture_weights, unmappable_effects
  (none|inf|ash|ash_filter_archived), check_null_threshold, prior_tol, residual_variance_lowerbound/upperbound, model_init, s_init,
  coverage, min_abs_corr, max_iter, L_greedy, greedy_lbf_cutoff, tol, convergence_method (elbo|pip), verbose, track_fit, n_purity,
  median_abs_corr, cs_extension_corr, refine, **R_finite** (NULL|FALSE|TRUE|B number), **R_mismatch** (`"none"`, `"eb"`, `"eb_mix"`),
  control (`susie_rss_control()`), alpha0, beta0, init_only, slot_prior. Note `z_ld_weight` of 0.14.2 is gone.
- Mismatch-aware call (DESIGN §4.1 item 3): `susie_rss(z, R, n=n, L=10, R_finite=B_ref, R_mismatch="eb_mix")`.
  Output extras: `fit$R_finite_diagnostics` with B, effective_rank, r_over_B, lambda_bias, B_corrected, Q_art, artifact_flag,
  per_variable_penalty, ser_model, bf_attenuation, R_sensitivity_flag, R_reliability_flag, **SER_fallback**.
  When active it switches to PIP convergence and auto-disables estimate_residual_variance (messages printed).
- Defaults changed vs 0.14.2: `max_iter` set to 50 for RSS (HINT message), default `L = min(10, p)`.
  coloc's `runsusie` has not been verified against 0.16.6 (coloc 5.2.3 was built for 0.14.x) -> wrapper should call `susie_rss`
  directly and `coloc.bf_bf(fit$lbf_variable ...)`.

## mvsusieR 0.3.0
- Install: `R_LIBS=$P/env/rlib_susie016 R CMD INSTALL --library=env/rlib_mvsusie env/src/mvsusieR`. It requires susieR >= 0.15.54, hence the
  separate library; it does NOT work with 0.14.2 (install refused). Deps (mashr, ashr, mvtnorm, abind, ggrepel, cowplot) in main lib.
- `mvsusie_rss(Z, R, N, Bhat, Shat, varY, prior_variance=<mixture prior>, residual_variance=<R x R residual correlation>, estimate_residual_variance=FALSE, L=, ...)`.
  Give Z (p x r z matrix) or (Bhat, Shat). LD `R` (p x p). `N` scalar or per-trait (vignette per_trait_sample_size).
- Prior: `create_mixture_prior(R = r)` (singleton + shared canonical matrices) or `create_mixture_prior(list(matrices=, weights=))`, or mash `fitted_g`.
- Sample overlap / residual correlation: via `residual_variance` (r x r matrix; with estimate_residual_variance=FALSE it is fixed to rho-hat).
  Z-only path estimates cor(Y) from near-null z-scores (|Z|<2).
- Output: `fit$pip`, `fit$sets$cs`, `fit$lfsr` (p x r), `fit$single_effect_lfsr` (L x r), `fit$conditional_lfsr`, `fit$lbf_variable_outcome`, `fit$lbf_outcome`,
  `fit$posterior_mixture_weights`. No direct P(shared): derive "shared in both traits" from per-effect lfsr < 0.05 in both traits (DESIGN adapter).
- No LD-mismatch option in mvsusieR.

## colocboost 1.0.10
- Install: `install.packages("colocboost")` (CRAN); GitHub StatFunGen/colocboost HEAD 9f82c2a has the same version.
- `colocboost(sumstat = list of data.frames per trait with z (or beta, sebeta), n, variant; LD = matrix | X_ref = ref genotypes; dict_sumstatLD = ..., residual_correlation = NULL, ...)`.
  Pairwise: pass a list of 2 traits (or use `focal_outcome_idx` for disease-prioritised). Individual-level: X, Y.
- LD mismatch: `M = 1` (one boosting iteration with LD only for equivalence check; documented "recommended" mismatch approach) or `LD` omitted (LD-free);
  post-hoc `get_robust_colocalization(res, cos_npc_cutoff=, npc_outcome_cutoff=)`. Diagnostics warnings are printed under mismatch.
- Sample overlap: `residual_correlation` (K x K matrix).
- Output: `res$cos_summary` (data.frame: colocalized_outcomes, cos_npc [CoS-level colocalisation probability], min_npc_outcome, purity, top_variable, top_variable_vcp, colocalized_variables_vcp),
  `res$vcp`, `res$cos_details` (`cos_outcomes_npc`, `cos$cos_index`). Returns NULL cos_summary + message "No colocalization results" when none.
- Trait names in output are Y1..Yk (position), not the list names.

## colocPropTest 0.9.3 and prop.coloc 2.0.0
- colocPropTest: CRAN, source build (needs lme4/car chain from conda). `run_proptests(S1, S2, LD, topsnps="auto", r2.thr=0.95, maxtests=10000, nauto=200, nsnps=NULL, adjust_n=FALSE)`
  with coloc-style datasets (beta, varbeta, snp, type; `snp` must match LD dimnames). Statistic: `min(result$fdr)` (reject proportionality if < alpha); also `$p`, `$eta.hat`.
  Null = proportional (shared). **No sample-overlap or LD-mismatch option; assumes independent traits.** In DESIGN's adapter `1 - p` should be formed from min(fdr)
  or the min p x corrections; decide which (see caveats).
- prop.coloc: GitHub **ash-res/prop-coloc** (Patel & Burgess, v2.0.0, arXiv 2402.12171). Chosen over Wallace's colocPropTest because it has sample-overlap (`tau`) input.
  Install: `R CMD INSTALL env/src/prop-coloc`.
  `prop.coloc(b1, se1, b2, se2, n, ld, tau=NULL, pval=NULL, alpha=NULL, clump=NULL, J=NULL, figs=NULL, traits=NULL, seed=100)`.
  `n` one value (one sample) or two (two-sample); `tau` = trait correlation (0 for non-overlapping). Output list: `p_cond` (prop-coloc-cond), `p_full`, `p_eta`, `p_naive`, `eta_cond`, `eta_full`, `top`, `variants`, `convergence`.
  README mentions arg `clump`; `prune` in docs text is `clump` in the function signature.

## SharePro (zhwm/SharePro_coloc @ 059caab, v5.0.0)
- Install: `git clone` into env/src (done); deps numpy/pandas/scipy already in .venv. Run with `python -I env/src/SharePro_coloc/src/SharePro/sharepro_coloc.py` (emits a harmless SyntaxWarning on `'\s+'`).
- CLI: `--z <gwas1.txt> <gwas2.txt>` (tab-delimited columns SNP, BETA, SE, N; matched order/alleles across traits and with LD), `--ld <file>` (whitespace p x p Pearson matrix, no header), `--save <prefix>`,
  `--K 10` (max effects), `--sigma 1e-5` (prior colocalisation prob), `--maxite 100`, `--eps 0.1`, `--ubound 1e6`, `--cthres 0.95`, `--pthres 0.5`.
- Output `<prefix>.sharepro.txt`: columns `cs` (variants in effect group, "/"-separated), `share` (colocalisation probability per effect group), `variantProb`.
- LD mismatch: the only built-in handling is automatic fallback "Set K to 1" when optimisation does not converge (ELBO > ubound or non-convergence; v2.0.0 note). There is no separate flag.
  Therefore "documented LD-mismatch-robust mode" == default run with that fallback; confirm interpretation with PI.
- Sample overlap: not supported. Locus-level P(coloc) must be derived from `share` (e.g. max or 1 - prod(1 - share)); the DESIGN adapter must pick one.
- Analysis code of the paper (zhwm/sharepro_coloc_analysis) not cloned.

## Rfast 2.1.5.2: conda-forge `r-rfast`, main lib; required by susieR >= 0.16.

## Tier 2
- gpu-coloc 0.7.1: `pip install gpu-coloc` (.venv). Input = per-signal log Bayes factors (pickle/feather, variants named `chr[chrom]_[pos]_[ref]_[alt]`),
  steps `gpu-coloc -f` (format) then `gpu-coloc -r --dir1 --dir2 --results --p12 1e-6 --H4 0.8` (`--CPU` for CPU). Output TSV: PP.H3, PP.H4, signal1, lead1, signal2, lead2.
  It is coloc.bf_bf reimplemented; use `-c` for CLPP/CLPA. Repo test.sh run via `scripts/baseline_checks/gpu_coloc_example.sh` -> PP.H4 0.897 for the example pair.
- hyprcoloc 0.0.2 (jrs95/hyprcoloc 0348bbd; not on CRAN). Failed to compile with the conda RcppEigen 3.4 (Eigen IndexedView with double indices);
  fixed by compiling against RcppEigen 0.3.3.9.4 (CRAN archive) in `env/rlib_hypr_dep`. Needs conda r-rmpfr/r-gmp/r-iterpc (installed in main).
  `hyprcoloc(effect.est, effect.se, trait.names, snp.id, ld.matrix, trait.cor, sample.overlap, prior.1=1e-4, prior.c=0.02, prior.12, ...)`; output `results$posterior_prob`, `regional_prob`, `candidate_snp`.
  Note `sample.overlap` default is all-ones (full overlap) -> must be set explicitly for independent studies; `trait.cor` default identity.

## Caveats
1. R changed 4.4 -> 4.5.3 in env/renv (see top).
2. mvsusieR and susieR 0.16 live in separate libraries; wrappers must run in separate R processes from the susieR 0.14.2 pipeline.
3. coloc 5.2.3 `runsusie` was not tested against susieR 0.16.6 (no need: mismatch-aware wrapper calls `susie_rss` + `coloc.bf_bf`).
4. The susieR 0.16.6 example (N3finemapping, B=300 reference panel): the standard and the `eb_mix` fits both found the three causal signals; `eb_mix` produced larger CS for the third signal (more conservative), SER_fallback = FALSE, lambda_bias = 0.
5. Packages from CRAN source (colocboost, colocPropTest) are not captured in renv.lock (conda export lists only conda packages); see R_packages_pinned.tsv.
