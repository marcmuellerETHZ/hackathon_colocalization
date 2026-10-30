# Review round 2 — external PI (DESIGN.md v2)

**Verdict: minor revision.**

v2 is a credible plan and could be carried out. The estimand, the class partition, the split design, the fair baselines and the test lock are now sound. Four specification problems remain, and each would make a pre-registered claim wrong or impossible to evaluate:
1. The overlap grid cannot be produced by the generator's own N grid.
2. C2 sets a 10× bar against our own GPU baseline, which is probably not attainable.
3. The MVR does not contain what C1, C3 and the real-data criterion need.
4. The C3 operating point is underdetermined.

All four can be fixed in the text in under an hour. None needs a rethink.

Checks run (scratch scripts only, outside the project):
- Block-split window clipping on `pyrho_EUR_LD_blocks.bed` × GENCODE v39, 5 random 60/20/20 assignments.
- The maximum achievable ρ on the §2.2 N grid.
- Algebra of the B support-space BF.
- Installed R baselines.

---

## 1. Round-1 must-change items (my review)

| # | Item | Status | Reason |
|---|---|---|---|
| 1 | Split units | **addressed** | Block-level units (about 267 test blocks) with clipped windows. Better than my component proposal. Script and counts are still to be committed at M3. |
| 2 | Shifted / misspecified test arm | **partially** | §8.5 is pre-registered and well-specified. But it relies on an msprime world 2 that sits outside the MVR and the critical path, while C1/C3 "must hold in direction" on it. See M3 below. |
| 3 | Real-data labels and criterion | **partially** | R2 is fixed: gene identity, not SuSiE agreement. A criterion now exists. The symmetric conditional construction was not adopted, and no reason is logged. R3b selects negatives by disjoint SuSiE CSs, which is circular. The circularity favours coloc-SuSiE, so it is conservative for us and acceptable if it is stated. R3 remains signal-level only. |
| 4 | Fair LD-mismatch baselines | **addressed (pending install)** | susieR 0.16.x with `R_finite` and `R_mismatch` is tier 1. DENTIST-QC'd coloc-SuSiE (2c) was dropped. That is fine unless layer E's DENTIST-S mask is credited as novel, in which case apply the same mask to coloc-SuSiE as a plug-in. **Note:** `env/rlib_susie016` is empty, and mvsusieR, colocboost, colocPropTest and prop.coloc are not yet installed (checked). |
| 5 | One primary candidate | **addressed** | §6.3, with one test run. |
| 6 | Claims well-posed | **partially** | C5 is fixed. C4 has a margin and uses ρ̂, but the ρ range is infeasible (M1). C2 now *requires* ≥ 10× over G (M2). C3's threshold rule is underdetermined (M4). |
| 7 | Class partition | **addressed, one wording error** | See M5. |
| 8 | ρ̂ | **addressed** | Genome-wide truncated MLE with realistic noise, and an unbiasedness test. |
| 9 | A null state; C kill-test | **addressed** | Explicit null state and power-aware q; C1 is now run in B's prior world. |
| 10 | MVR and critical path | **partially** | The MVR is defined but inconsistent with the claims (M3). The critical path is still serial, M4 → M5, with no vertical slice (S1). |

Other round-1 items I own:
- **Addressed:** 11 (biobank stratum λ up to about 150), 13 (Kronecker order), 14 (B budget), 15 (FinnGen PSD), 16 (paired SE; stop rule).
- **Not adopted:** 12 (cheap second LD world); msprime was chosen instead (see M3).
- **Deferred, acceptable:** 17 (GWAS-vs-GWAS arm). It does become relevant to M1.

**Open question 2 (support-space BF): correct as written.**
- With X = I₂⊗R and Σ = V⊗R, XᵀΣ⁻¹ = V⁻¹⊗I. So b = (V⁻¹⊗I)z and Φ = V⁻¹⊗R. The BF depends on z only through z_s, so restricting to the support loses nothing.
- D⁻¹ must be read as restricted to the support indices (Φ_ss, b_s). State this explicitly so that nobody inverts a singular full-size D.

## 2. New errors and problems in v2

**E1. The overlap grid cannot be reached with the eQTL N grid (§2.2, §2.4, §8.1, C4).**
- In nested mode ρ = N_s·r_p/√(N1·N2) ≤ r_p·√(N2/N1). On the §2.2 grid, the maximum ρ at r_p = 1 is:

  | N1 \ N2 | 100 | 300 | 1k | 5k |
  |---|---|---|---|---|
  | 20k | 0.07 | 0.12 | 0.22 | 0.50 |
  | 100k | 0.03 | 0.06 | 0.10 | 0.22 |
  | 400k | 0.02 | 0.03 | 0.05 | 0.11 |

- ρ = 0.8 is impossible in every cell, and ρ = 0.5 is possible in only one cell, at r_p = 1. Factor mode with both traits large-N requires N2 ≥ 20k, which is outside the N2 grid.
- Yet the realistic headline pools ρ ∈ {0, 0.1, 0.3, 0.5, 0.8} with equal weight. Three of those five ρ values therefore describe study designs the generator does not contain, or GWAS-vs-GWAS designs that are not QTL-like.
- Consequences:
  - C1's headline is weighted towards an unrealistic regime.
  - C4 is evaluated mostly where trait 2 is not QTL-like.
- The generator also does not say how trait-1 noise (factor form over N_syn = 20k rows, representing N1 up to 400k) shares "rows" with trait 2. That needs one line, e.g. e1 = √(N_s/N1)·(shared-row component) + √(1 − N_s/N1)·(independent factor noise).

**E2. C2 now requires ≥ 10× over G.**
- R1 asked to *report* the modelling-vs-engineering split via G. v2 makes ≥ 10× over G a pass condition.
- A joint model (A) or enumeration (B, about 1 s/locus) is unlikely to be 10× faster than a batched GPU SuSiE that the project wrote itself.
- In workload (ii), 50 tissues, G amortises per-trait fine-mapping, while joint models cannot. G then wins by construction.
- "Accuracy equal or better" is undefined: no metric, no margin.

**E3. The MVR does not cover the pre-registered claims.** The MVR is L0–L4 × ρ ∈ {0, 0.3, 0.5} with one real arm. But:
- C1's headline needs L6 and five ρ values;
- C1/C3 must hold on the shifted arm, which needs msprime world 2 and is deferred for validation only;
- the real-data criterion needs R2 + R3b + R4/R4b.

If the project stops at the MVR, no claim can be evaluated as registered.

**E4. The C3 operating point is underdetermined.** "Confident-call rate on W = 5%" is one constraint on two thresholds, t_S and t_D. Each method can split it differently, which is a hidden tuning degree of freedom.
- "≥ every baseline" has no statistical criterion.
- Proportional-test and lead-SNP baselines have no P_D, so "every baseline" is undefined for D recall.
- SharePro and ColocBoost get a gated P_D (0 unless both traits are significant). That conflicts with the ungated-score principle and creates massive ties in the D-vs-W co-primary AUROC.

**E5. §1 class table contradiction.** G lists "powered distinct + *weak* shared variant". Rule 3 (D: "no shared causal variant has power ≥ 0.2 in both") already captures weak-shared loci as D, so the precedence order and the prose disagree. §1 is hash-frozen, so fix it before the freeze: G should mean *grey* shared, with power ≥ 0.2 in both but not powered in both.

**E6. Power matching for S is undefined (§2.4, §8.1).** S and D are reweighted "within each r²/d bin", but S loci have no class-determining pair and hence no r² or d. The headline metric depends on this. Pre-register it, for example: S loci are reweighted to match the pooled D (μ1, μ2) distribution, and per-bin AUPRC uses the same S pool reweighted to each bin's D distribution.

**E7. "L0 = ideal" and "Bayes-optimal ceiling at L0" are not true as specified.**
- At L0, a small-N trait 2 follows its own in-sample LD, which is rank ≤ N2 (100–5,000).
- Single-R methods (A, B, NPE inputs) receive trait 1's R_pop.
- With up to 3 causal variants, an infinitesimal background and a log-uniform λ, B's simulator-prior mode is not exact.
- Add an **L0\*** oracle world for the identity tests and the ceiling: both traits in factor form, R_pop given, ≤ 2 causal variants per trait. Call B's L0 number an "approximate ceiling".

**E8. Window clipping.** Across 5 random block assignments:
- 31% of ±500 kb windows are clipped;
- 4% are under 500 kb;
- in about 9% of genes the clip falls within ±100 kb of the TSS, where most cis-eQTL leads lie.

This is harmless in simulation. In the real-data arms it can silently remove the lead, which is the classic false non-coloc. Since nothing is trained on real data, either:
- do not clip real-data windows, and assign genes by TSS block; or
- log and drop real loci whose lead in either trait falls outside the clipped window.

**E9. Test-set size per cell is not pre-registered** (§6.1 covers screening only). This size determines C1's power and must be frozen with §8.

## 3. Proposed changes

### MUST

**M1. Tie overlap to study design (§2.2, §2.4, §8.1, C4).** Define three overlap strata:
- **(a) QTL nested in GWAS:** the §2.2 N grid. ρ is derived (≤ 0.1 at realistic r_p ≤ 0.3).
- **(b) Large-N molecular trait:** pQTL-like, N2 ∈ {20k, 50k}, factor form, ρ ∈ {0.1, 0.3}.
- **(c) GWAS vs GWAS:** both ≥ 20k, factor form, ρ ∈ {0.3, 0.5, 0.8}.

Then:
- The realistic headline (C1) weights strata explicitly, e.g. a 60/20/20 cell weight.
- C4 is evaluated on (b) ∪ (c) at the achievable ρ.
- Add the one-line nested-noise construction to §2.2.
- The screening set's ρ = 0.5 cells come from stratum (c).

**M2. Restate C2 (§0, §7.7).**
- Pass condition: ≥ 10× whole-machine throughput over coloc-SuSiE (64 cores) on workload (i).
- Accuracy is non-inferior: the paired CI lower bound of ΔAUPRC (realistic) is > −0.01.
- G and gpu-coloc are reported as the engineering-vs-modelling decomposition, not as pass bars.
- Workload (ii) is reported. State in advance that joint models cannot amortise per-trait fits, and report the K-pair GEMM batching (R1-alg 7) if implemented.

**M3. Make the MVR cover the claims, or narrow the claims to the MVR (§0, §2.1, §2.5, §8.5).** I recommend expanding the MVR cheaply:
- **World 2:** replace msprime with the near-zero-cost pool swap: target mosaic from pool B, reference from a 150-person subset of pool A. Keep msprime deferred. This puts the shifted arm inside the MVR.
- **MVR grid:** the full realistic grid of C1 (L1–L4, L6 × the M1 strata) plus L0/L0\*.
- **MVR real data:** R1 (realism), R2 + R3b (criterion) and R4b (GEUVADIS, exact in-sample LD).
- **Defer:** R4 (FinnGen LD), R3 (signal-level) and R5b.

**M4. Pin down C3 (§0, §1, §8.2).**
- Set each threshold separately on validation: confident-S rate on W ≤ 2.5% (t_S) and confident-D rate on W ≤ 2.5% (t_D).
- C3 passes if, for every baseline that has a P_D, the paired-bootstrap one-sided 95% lower bound of (candidate − baseline) D recall is > −0.02, with superiority over coloc-SuSiE (mismatch-aware). S recall is reported.
- For the D-vs-W AUROC, give every baseline an **ungated** P_D:
  - coloc PP3;
  - for SharePro and ColocBoost, (1 − P_S) × min over traits of (the trait's max per-trait signal probability).
- Report the tie share.

**M5. Fix the §1 G-row wording (E5) and pre-register S power matching (E6) and test-set size per cell (E9) before hashing §1/§8.** Each is one sentence.

### SHOULD

**S1. Parallelise the critical path (§0, §11).** The brief requires running the cheapest kill-tests before full exploration.
- Run M5 kill-tests on the screening set against coloc.abf, coloc-SuSiE (both) and SharePro only, in parallel with the full M4 grid.
- Decouple A1's accuracy test (a CPU/torch prototype is acceptable) from its runtime clause, which moves to M6.
- B1-correctness and C0 need no baselines and can start right after the generator passes its tests.

**S2. Baseline install fallback (§4, open Q3).** Six of the eight tier-1 baselines are not yet installed.
- Rule: a baseline that cannot reproduce its documented example within one agent-day moves to tier 2, with a decision note.
- The "every baseline" in C1/C3 is the tier-1 list frozen at the end of M4.

**S3. Add the L0\* oracle world (E7)** for the A/B identity tests and the ceiling. Rename B's L0 number to "approximate ceiling".

**S4. Real-data windows (E8).**
- Either do not clip real-data windows, or log and drop loci whose lead falls outside the window.
- Restate the real-data criterion as non-inferiority with a margin: the CI lower bound of ΔS-recall is > −0.05. A bare point-estimate "≥" is not a test.
- State that R3b's SuSiE-CS selection favours coloc-SuSiE.

**S5. Tighten C4's margin.** ±0.02 per 0.1ρ allows +16 pp of false-S over ρ ∈ [0, 0.8], while the operating rate is 2.5–5%. Use a total change in the confident-S rate on W over the achievable ρ range of ≤ 2 pp (95% CI).

### COULD (simplifications: process overhead with little gain)

- **Replace the hashes of DESIGN.md §1/§8** with `git diff freeze-design..frozen-* -- <files>` in `eval_test.py`. Hashing Markdown sections is fragile, and the tag lock already gives this protection. Keep the hashes of code and split files.
- **Keep the guards as one small module.** Manifests, ledger and `verify_run` should be one small module written once in M3, with no extra checker scripts.
- **Cap the slow baselines.** Run mvSuSiE-RSS and ColocBoost on a capped subset (≤ 1k loci per cell), report this, and keep the full grid for coloc.abf, coloc-SuSiE and SharePro.
- **Run B in one mode for M5.** Use coloc mode for correctness and simulator-prior mode for the ceiling; build Gaussian-EB mode only if B survives.
- **Cut the NPE kill ladder from four tests to three.** The GBM floor can be merged into C2 as an extra comparator.

Nothing else in v2 needs changing within my remit.
