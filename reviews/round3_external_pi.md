# Review round 3 — external PI (DESIGN.md v3)

**Verdict: not ready to freeze. Seven must-fix items remain.** All of them are text-level and should take under an hour in total; none needs a rethink. Items 1 and 2 would make test cells impossible to fill or physically impossible. Items 3–7 would leave a pre-registered claim unevaluable or inconsistent across sections.

Checks run (throwaway script under /tmp, nothing written to the project except this file):
- λ ranges per pairing type;
- reachable ρ in P2;
- test-set size versus the compute plan.

## 1. Status of my round-2 must-change items

| R2 item | Status in v3 |
|---|---|
| M1 overlap tied to study design | **Mostly fixed.** The P1/P2/P3 strata, the 0.5/0.25/0.25 weights, C4 restricted to P2 ∪ P3 and the nested per-row correlation r_p·√(N_syn/N1) are all present, and I verified the scaling. New errors in P2 and P3 are listed as must-fix 1 and 2. |
| M2 C2 restated | **Fixed.** The bar is coloc-SuSiE on 64 cores, precompute is included, and G/gpu-coloc are reported only. Two small gaps are listed under should-fix. |
| M3 MVR covers claims | **Fixed in §0/§8.5.** §2.1 contradicts it (must-fix 3). The shifted arm is under-specified (must-fix 5). |
| M4 C3 pinned | **Mostly fixed.** There are separate 2.5% W limits, ungated P_D (§4.3) and a non-inferiority margin. The superiority test, the set of eligible baselines and the precedence of the call are still undefined (must-fix 4). §8.2 still says 5% (must-fix 3). |
| M5 G-row / S power matching / per-cell size | **G-row fixed. Per-cell size fixed (§8.6). S power matching only partly fixed** (must-fix 6). |

## 2. Must-fix

**1. P3 eQTL × eQTL cannot produce S or D loci (§2.2, §2.4, §1).**
- Trait 1 always draws a PVE in [1e-5, 2e-3], or [2e-3, 2e-2] in the biobank stratum, and is judged at τ₁ = 6.29.
- At N1 = N2 = 300 the maximum λ₁ is 0.78 (2.47 in the biobank stratum). At N = 1,000 it is 1.42 (4.52).
- Trait 1 is therefore never powered, so rejection sampling can never fill the 400 S and 400 D per cell required in the P3-eQTL cells (§8.6). Those cells would be W-only, and C4 on P3 would be meaningless.
- Conversely, in P3 lab × lab and in P2, trait 2 uses the QTL range [1e-3, 0.3]. That gives λ₂ ≥ 4.5–10, so trait 2 is never weak, and W-with-weak-trait-2 cannot occur.
- **Fix:** state that the effect range and the oracle α/τ follow the *kind* of trait, not its index:
  - molecular traits use [1e-3, 0.3] and α = 1e-5;
  - GWAS or lab traits use the trait-1 range and α = 5e-8.
- Apply this to P3 eQTL × eQTL (both molecular) and P3 lab × lab (both GWAS-like). §1 is hash-frozen, so write it into §1 before the freeze.

**2. Part of the P2 overlap grid is still unreachable, and the C4 slope can be confounded with N (§2.4, §8.6, C4).**
- In P2, |ρ| ≤ r_p·√(N2/N1).
  - ρ = 0.3 is impossible at (N2, N1) = (20k, 400k): the maximum is 0.22.
  - It needs r_p ≥ 0.85 at (50k, 400k), r_p ≥ 0.67 at (20k, 100k) and r_p ≥ 0.42 at (50k, 100k).
- The factor-form construction "Cov(e1, e2) = ρR_pop exactly" will generate any ρ regardless, so an impossible design passes silently.
- **Fix, in three parts:**
  - (a) In the generator, assert |ρ| ≤ |r_p|·√(min(N1, N2)/max(N1, N2)) with |r_p| ≤ 1. A cap such as r_p ≤ 0.6 would be more realistic for pQTL vs GWAS.
  - (b) Pre-register the (N1, N2) pairs used in each P2 ρ cell, using the *same* (N1, N2) mix for ρ = 0 and ρ = 0.3. For example, N1 = 100k with N2 ∈ {20k, 50k} in both cells.
  - (c) Estimate the C4 slope with fixed effects for pairing type × (N1, N2), so the slope is not confounded with power.

**3. Cross-section inconsistencies that would be frozen as-is.**
- (a) §8.2 says "confident-call rate on W of 5%", but C3 says 2.5% for S and 2.5% for D. §1 points to "§8.1" for the matched-W rule, which actually lives in §8.2.
- (b) §2.1 still defines LD world 2 as msprime "used only in the shifted test arm", which contradicts §0 and §8.5 (pool-swap first, msprime deferred).
- (c) §2.4 gives P3 "both signs", but §8.6 lists only ρ ∈ {0, 0.5, 0.8}. Either add negative-ρ cells or drop "both signs".
- (d) C1–C4 live in §0, which is not in the §7.4 hash list. Add §0 to `FROZEN_SHA256` (or to the git-diff check).

**4. C3 is still not fully evaluable.**
- (a) Define the superiority test: the paired cluster-bootstrap one-sided 95% lower bound of (candidate − baseline) recall is > 0. The comparator is the baseline with the highest validation recall for that endpoint, fixed before the test run, not chosen on test.
- (b) "Every baseline" for S recall means those with P_S. For D recall it means those with P_D. Name the exclusions: proportional test and lead-SNP (§4.3).
- (c) The call needs a precedence rule for when P_S ≥ t_S and P_D ≥ t_D both hold. With the 2.5% thresholds, t can fall below 0.5 for coloc-type outputs. Suggested rule: S first, or argmax.

**5. The shifted arm is under-specified (§8.5, §8.6: "6,000 loci").**
- Pre-register:
  - which LD given to methods: pool-A reference, n = 150, with L4-style corruption in half of loci or none;
  - which pairing types and overlap cells, e.g. the headline P1/P2/P3 weights at ρ = 0;
  - power matching;
  - the seed: a derived seed, not 20261028 reused with overlapping streams.
- Without this, "C1/C3 hold in direction" can be met by choosing the arm's composition after the fact.

**6. S power matching and D bins are still not defined (§2.4, §8.6).**
- §2.4 reweights "within each r²/d bin", but S loci have no r² or d.
- §8.6 stratifies D "uniformly over the d/r² bins", but no bin edges are given anywhere.
- **Fix, one sentence each:**
  - S loci in a cell are reweighted to each D bin's (μ1, μ2) distribution (5×5 quantile bins of the *pooled D* distribution in that cell), and pooled over bins with the D bin weights.
  - Freeze the bin edges in `configs/testset_v1.yaml`, e.g. r² ∈ [0, 0.1), [0.1, 0.3), [0.3, 0.5), [0.5, 0.8), [0.8, 1] × d ∈ [2, 4), [4, ∞).

**7. The test run's compute and one-shot lock are inconsistent (§7.3, §10).**
- §8.6 implies about 56–65k test loci: 42 × (1,200 + up to 200) + 6,000.
- That is the same scale as M4's validation grid, which is budgeted at 36–72 h of CPU, yet M7 is budgeted at about 12 h. At 15–30 s/locus under mismatch, the two coloc-SuSiE variants alone take about 14–27 h on 40 workers.
- With a one-shot lock, an interrupted run would be fatal. **Fix:**
  - set M7 CPU to ≥ 48 h, or pre-register subsampling for the slow methods (as in §4.3);
  - state that `eval_test.py` is resumable under the same run_id, with deterministic per-locus seeds. "One test run" then means one set of predictions per method, not one process invocation.

## 3. Should-fix (not blocking)

- **C2:**
  - name the coloc-SuSiE version that is the speed bar (the faster of 0.14.2 + Rfast and 0.16.6, from a pilot);
  - state the accuracy non-inferiority as "paired-CI lower bound of ΔAUPRC (realistic, test) > −0.01", not as a point estimate.
- **C4:** the margin of ±0.02 per 0.1ρ still allows +16 pp over ρ ∈ [0, 0.8] (round-2 S5). Prefer a total change of ≤ 2 pp over the reachable range.
- **Real-data criterion (§2.5):**
  - it is still a bare point-estimate "≥" (round-2 S4); prefer a CI lower bound > −0.05;
  - it cites R4, which §0 defers. Say "R4b (and R4 if run)".
- **Real-data windows:** they can still clip away the *other* trait's lead (round-2 E8). Log and drop such loci.

Nothing in v3 is over-specified to the point of harm. Apart from the items above, the claims C1–C5, the MVR and §8.6 are now consistent.
