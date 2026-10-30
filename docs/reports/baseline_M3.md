# Baseline engineer — M3 report (2026-10-26)

Owner of `src/baselines/`, `tests/baselines/`. Commits: `9468d5a` (wrappers + tests) and the follow-up commit
with timeouts, runtime tables and this report.

## 1. What was built
All tier-1 baselines of DESIGN v3 §4.1 with the §4.3 adapters and D4 decisions, behind the INTERFACES §4 API
(`baselines.get_method(name).fit_batch(loci) -> list[dict]`, rows = INTERFACES §3 columns; `call` left None,
`E1/E2` NaN, `extra` strict JSON with PP0–PP4, CS counts, diagnostics, R warnings).

| name | implementation |
|---|---|
| `coloc_abf` | local vectorised coloc.abf (`abf.py`), W tunable; one run returns PPs for W ∈ {0.1, 0.15, 0.2} × p12 ∈ {5e-6, 1e-5} in `extra.grid` (primary row: W 0.15, p12 5e-6) |
| `coloc_susie` | susieR 0.14.2 `susie_rss(z, R, n, L=10)` + `coloc.bf_bf` on CS effects (exactly `coloc.susie`), p12 5e-6, fallback coloc.abf |
| `coloc_susie_mm` | susieR 0.16.6 (separate R process, env/rlib_susie016), `R_finite = n_ref, R_mismatch = "eb_mix"`, diagnostics (SER_fallback etc.) in extra |
| `sharepro` | official SharePro 059caab imported unchanged and run in-process (same steps as its `main()`, incl. K=1 fallback); P_S = 1 − Π(1 − share_k) |
| `colocboost`, `colocboost_m1` | colocboost 1.0.10 pairwise with LD, residual_correlation from ρ̂; default and `M = 1` (documented mismatch mode) |
| `mvsusie` | mvsusieR 0.3.0 `mvsusie_rss`, canonical singleton+shared mixture prior, residual correlation fixed to ρ̂; q = 1 − single-effect lfsr → Family A inclusion–exclusion (D4) |
| `prop_coloc` (primary), `coloc_prop_test` (secondary) | s = p_full (tau = ρ̂, n = c(N1, N2)); s = min FDR |
| `lead_snp` | s = 1{r²(lead1, lead2) ≥ 0.8} |

**R calls**: persistent `Rscript` workers (`rworker.py`, `r/worker.R`), one per library setting per Python process
(fork-safe), JSON-line protocol, arrays as raw float64 files in `/tmp/claude-<uid>/baselines_ipc`, 17-digit JSON
(exact round-trip), single-threaded BLAS, package chatter diverted to `logs/baselines/`, per-call timeout that kills
the process group and respawns. Wrappers assert the susieR/mvsusieR version returned by R on every call (0.14.2 vs
0.16.6 never mix). R start-up (~2 s) is paid once per process.

## 2. Tests (all run)
Command: `source env/activate.sh && python -m pytest -q -rxs tests/baselines/`
Result: **85 passed, 1 xfailed in 84 s** (xfail = SharePro distinct control rep 1, see §4).

- `test_abf_vs_r.py`: local ABF vs R coloc 5.2.3 `coloc.abf` on 120 loci (p from 1 to 12,000; null/shared/
  distinct/one-trait; |z| up to 40) × W {0.1, 0.15, 0.2} × p12 {5e-6, 1e-5} = 720 comparisons (R prior SD set to W
  via sdY = W/0.15): **max |ΔPP| = 5.3e-16, max |ΔSNP.PP.H4| = 6.1e-16** (requirement 1e-8). Includes coloc's
  `adjust_prior` regime (p ≥ 10^4).
- `test_rworker.py`: library isolation (main: susieR 0.14.2 / coloc 5.2.3 / colocboost 1.0.10 / prop.coloc 2.0.0 /
  colocPropTest 0.9.3; susie016: 0.16.6; mvsusie: 0.16.6 + mvsusieR 0.3.0), column-major array round-trip, R error
  without worker death, timeout → kill → respawn; **wrapper reproduces coloc's documented example**
  (`coloc.susie(runsusie(D3), runsusie(D4))`: 2 CS pairs, max |ΔPP| = 2.5e-9); **mvSuSiE per-trait z pre-adjustment
  equals native `mvsusie_rss(Z, R, N)`** when N1 = N2 (PIP and single-effect lfsr within 1e-6).
- `test_sharepro_equiv.py`: in-process SharePro reproduces the shipped README output and equals the official CLI
  on toy loci (groups and shares identical).
- `test_adapters.py`: adapter formulas with hand-computed answers; inclusion–exclusion P_S/P_D equal exact
  enumeration over 4^L activity states (50 random cases, 1e-12).
- `test_controls.py`: every wrapper on toy loci from a real 1000G chr22 NFE window (404 individuals, p = 300,
  in-sample LD, 2 seeds each): shared λ = 30 → s > 0.9 (p-value methods: s > 0.01; lead-SNP s = 1) **pass for all
  10 methods**; distinct λ = 30 at r² < 0.1 → s < 0.1 **pass for 9/10, SharePro fails 1 of 2** (xfail with reason);
  null trait 2 → P_W > 0.5 or P_D < 0.5 **pass** for all methods with P_*; ρ̂ = 0.5 overlap runs (ColocBoost,
  mvSuSiE, prop.coloc, coloc-SuSiE) pass; factor vs dense LD identical; NaN input → `failed`; timeout → `timeout`
  then respawn ok; SharePro passes the distinct control at r² < 0.01 (3 seeds).
- `test_no_truth_access.py`: no wrapper source mentions truth files.

## 3. Runtime and memory (pilot)
`python -m baselines.bench_runtime` (+ `--mismatch`): 10 methods × p ∈ {1000, 3000, 5000} × 3 loci (shared,
distinct, null2; λ = 15; N1 = 100k, N2 = 1000), each (method, p) in its own process with a fresh R worker; LD in
factor form (runtime includes densifying). "in" = in-sample LD (404 NFE), "mm" = z from 254 individuals, LD from the
other 150 (L2-like). Full table `src/baselines/benchmarks/runtime_M3.tsv`; summary in `src/baselines/README.md`.
Medians (s/locus), in / mm:

| method | p=1000 | p=3000 | p=5000 (max) | R peak RSS p=5000 |
|---|---|---|---|---|
| coloc_abf | 0.001 | 0.001 | 0.002 | – |
| sharepro | 0.03 / 0.03 | 0.25 / 0.63 | 0.85 / 4.5 (7.0) | – |
| coloc_susie | 0.5 / 1.7 | 2.5 / 11.8 | 6.2 / 22.8 (24.7) | 1.9–2.1 GB |
| coloc_susie_mm | 0.7 / 2.6 | 7.1 / 14.7 | 30.2 / 31.6 (49.4) | 1.9–2.5 GB |
| colocboost | 0.5 / 3.8 | 1.5 / 18.2 | 2.7 / 37.5 | 1.0–1.3 GB |
| colocboost_m1 | 0.12 | 0.5 | 0.9–1.0 (1.35) | 0.9 GB |
| mvsusie | 1.8 / 13.1 | 10.3 / 37.4 (178, nonconverged) | 22.1 / 76.4 (87) | 1.4–1.7 GB |
| prop_coloc | 1.0 / 0.2 | 1.5 / 0.5 | 1.1 / 1.6 (2.1) | 1.0 GB |
| coloc_prop_test | 0.5 / 1.0 | 1.3 / 1.1 | 2.4 / 2.2 (2.8) | 1.0 GB |

**Proposed timeouts** (`methods.DEFAULT_TIMEOUT`, = max observed × ~4 for p up to 10k × ≥ 3): coloc_abf 60, lead_snp
60, sharepro 300, prop_coloc 300, coloc_prop_test 300, colocboost_m1 120, coloc_susie 600, coloc_susie_mm 900,
colocboost 900, mvsusie 1800 s. Memory scales ~p²: at p = 10k expect ~8–10 GB per susieR worker → schedule
susie/mvSuSiE workers by memory. These are toy loci with 1–2 signals on a shared machine; the 200-locus pilot
(DESIGN §4.3) should freeze the final values.

## 4. Findings (method behaviour, verified not to be wrapper bugs)
1. **Summary-statistic scale (important for the generator).** DESIGN §2.2 gives z = √N·r with the null
   se = 1/√(2Nf(1−f)). Real studies and the methods use OLS t-statistics and OLS se (n·se² + b² = var(y)/var(x)).
   With the literal DESIGN statistics, prop.coloc rejected proportionality on **70% of truly shared toy loci**
   (λ = 30, N2 = 5000; 20 loci) vs **5%** with OLS-consistent statistics (se_ols = se·√(1 − z²/N), z_ols = b/se_ols).
   SuSiE's n-adjustment and SharePro's bhat = z/√(z²+n) also assume t-statistics. Toy data now use OLS statistics.
2. **SharePro false sharing between strong distinct signals in moderate LD** (λ = 30 both, 60 loci): s > 0.5 in 0/15
   at r² < 0.02, 2/33 at 0.02–0.05, 8/12 at 0.05–0.1. Identical to the official CLI.
3. **mvSuSiE under mismatched LD** (n_ref = 150): 10 spurious CSs, distinct loci called shared (s ≥ 0.99 at all p);
   coloc-SuSiE 0.14.2 also fits many spurious CSs but its s stays ≈ 0; susieR 0.16.6 eb_mix flags SER_fallback.
4. **ColocBoost default** missed the shared signal at p = 5000 under mismatch (s = 0); M = 1 found it.
5. **prop.coloc** occasionally stops with "Unable to compute the standard errors" (1/40 toy loci; 1/9 mismatch
   bench loci) → status `failed`; on loci where trait 2 has no signal it rejects proportionality (expected: S-vs-D only).

## 5. Deviations from DESIGN / D4 and choices (full list: src/baselines/README.md "Decisions / assumptions")
- mvSuSiE: z pre-adjusted per trait with its own N and `mvsusie_rss` called without N (it accepts one N only);
  proven equal to the native path when N1 = N2. Effects not forming a CS get q = 0. q_l1, q_l2 treated as
  independent within an effect (a_l = w_l = q_l1 q_l2).
- ColocBoost: no per-trait probability is reported, so P(trait-t signal) ∈ {0, 1} (trait in a CoS or a uCoS) → its
  P_D is coarse. Locus P_S = 1 − Π(1 − cos_npc) (same aggregation as D4 for SharePro).
- prop.coloc alpha = 0.05 (affects only p_cond, saves runtime). colocPropTest: s = 0.5 + flag if no test is run.
- coloc-SuSiE calls susie_rss directly (no coloc::runsusie retry loop); fallback coloc.abf uses W = 0.15.
- Non-finite input → `failed` (susieR 0.14.2 silently returns a "fit" on NaN z).
- ρ̂ missing → 0; clipped to ±0.99 (ColocBoost, prop.coloc) and ±0.95 (mvSuSiE).

## 6. Decisions needed from the PI
1. **Generator output scale (finding 1):** adopt OLS statistics (beta unchanged, se_ols = se·√(1 − z²/N),
   z = beta/se_ols) in `src/sim` before any benchmark; otherwise prop.coloc/SharePro/SuSiE are handicapped by an
   artefact. (Simulation agent owns the change.)
2. **SharePro P(trait signal):** DESIGN's fallback 1 − Π_j(1 − PIP_tj) underestimates spread signals (0.66 for a
   λ = 30 signal spread over 8 SNPs); the effect-wise 1 − Π_k(1 − Σ_j γ_jk δ_tjk) (in `extra.P_sig_effectwise`)
   gave 1.0. Keep DESIGN (pre-registered) or switch before any validation result?
3. ColocBoost: report both `colocboost` and `colocboost_m1` at every level (proposal), or M = 1 only at ≥ L1?
4. Accept the provisional timeouts until the 200-locus pilot; accept memory-based worker limits for susie/mvSuSiE.
5. Accept the coarse binary per-trait signal indicator for ColocBoost's P_D (ties in D-vs-W AUROC).

## 7. Open issues
- `src/eval/io.py` exists (uncommitted at the time of writing); wrappers duck-type it and fall back to a marked stub
  (`baselines/locus.py`). Wrappers use `locus.R`/`X_ref`, `n_ref`, `rho_hat`, arrays of §2.
- One R worker log file per process in `logs/baselines/` (empty ones are deleted on clean shutdown).
- Not done: tier-2 baselines; harness integration test with real HDF5 locus sets (needs the generator).
