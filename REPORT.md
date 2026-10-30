# REPORT: Speed and power for GWAS–molecular-QTL colocalization under LD mismatch and sample overlap

## Summary
**Goal.** Make GWAS–QTL colocalization faster and more powerful when two things go wrong: the LD reference does not match the study, and the two studies share samples. The method must also tell a shared causal variant (S) apart from two distinct ones (D) and from a signal too weak to call (W).

**What we built and screened** on a pre-registered, LD-block-split simulation (1000G 30x, hg38):
- **A — JointSuSiE:** a GPU joint bivariate SuSiE. Killed: less accurate than coloc-SuSiE.
- **B — ExactColoc:** exact ≤ 2-causal Bayesian enumeration on GPU. Killed by a pre-registered operating-point rule that, as we show, penalised its good behaviour on weak loci.
- **C — NeuralColoc:** amortised neural posterior estimation, the bold family. Passed its first two rungs in phase 1. In phase 2 it was trained at full scale and failed its final rung (C2), because it did not beat its own no-LD version. Phase-1 note: its ladder could not be finished in the time budget.
- **G — GPU coloc-SuSiE:** an engineering baseline.

No new method reached the single test run as a candidate. The test run therefore delivered the pre-registered fallback: a benchmark of ten existing methods and controls on 13,207 held-out simulated loci and three real-data arms, with runtimes. B and G ran as pre-declared descriptive entries.

**Main findings (test set, single run; independent verification in §5.5):**
1. **LD-reference mismatch, not sample overlap, is what breaks colocalization.** coloc-SuSiE's S-vs-D AUPRC falls from 0.96 with in-sample LD to 0.69–0.73 with a 150-sample external reference (L2/L3/L6; L4, at 0.63, is failure-blocked). Supplying an estimated overlap ρ̂ neutralises sample overlap: on weak loci, every method's estimated false-S slope stays within ±0.02 per 0.1 of ρ (main sample).
2. **Mismatch-aware coloc-SuSiE (susieR 0.16.6) is the best existing method under realistic conditions.**
   - AUPRC 0.900 [0.884, 0.914] vs coloc.abf 0.860 and standard coloc-SuSiE 0.717.
   - On real GEUVADIS data its calls change on 7% of genes when in-sample LD is replaced by a reference, vs 21% for coloc-SuSiE.
   - It is also slow: about 16 CPU-s per locus.
3. **GPU engines keep accuracy, but whole-machine speed-ups are modest.**
   - Our GPU coloc-SuSiE (G) matches coloc-SuSiE's accuracy (0.719 vs 0.717).
   - ExactColoc (B) matches its in-sample accuracy (0.969).
   - **On whole-machine throughput** (one GPU vs coloc-SuSiE on all 64 CPU cores of the same node), G is only about 1.2× faster and B about 4.6×. Per worker, i.e. one GPU vs one CPU core, the ratios are 76× and 294×.
   - Neither reaches the pre-registered C2 bar of ≥ 10× whole-machine throughput.
4. **"Distinct vs too weak" is solved by signal strength alone.** A strength-only control reaches 0.99 AUROC and no method beats it. The hard problem is S vs D, where the same control is at chance. Proportional tests (prop.coloc, colocPropTest) barely beat it there: 0.55 and 0.63.
5. **Methodological finding.** A threshold rule that limits only confident calls on weak loci killed the fastest well-ranking method (B): on test it calls 23% of distinct loci shared at that threshold, but only 6.7% at a fixed 0.8. Pre-registered operating points should constrain both error directions. Phase 2 then tested this with a two-sided operating point (§11). It did not rescue the ExactColoc line: B-mm was still killed and missed the candidacy bar.

## 1. Question and approach
Most GWAS signals are non-coding yet rarely colocalize with molecular QTLs, and coloc results conflate "no shared variant" with "not enough power".

We benchmarked existing colocalization methods and built faster and more powerful alternatives that:
- cope with sample overlap and mismatched LD references;
- separate a **shared** causal variant (S) from **two distinct, well-powered** variants (D) and from a signal **too weak to call** (W).

Everything was pre-registered in DESIGN.md (v4, frozen after three review rounds with 14 reviewer reports). That covers estimands, classes, metrics, splits, baselines and kill tests. Each change after the freeze is a decision note (RESEARCH_TREE.md D0–D21, DECISIONS.md P3-1..P3-30).

### 1.1 Estimand and classes
- **Oracle classes** come from simulation truth by a precedence partition (DESIGN §1): S, U (unidentifiable: distinct but d = min|λ|·√(1−r²) < 2), D, W, G (grey zone).
- **Power** is defined on the realized noncentrality, with trait-type thresholds: GWAS-like α = 5e-8, τ = 6.29; molecular α = 1e-5, τ = 5.26.
- **Co-primary endpoints:**
  - AUPRC for S vs D (power-matched; weighted prevalence 0.5);
  - AUROC of P_D for D vs W.

  The headline "realistic" setting pools the 27 mismatched cells (L1–L6) of the 34-cell grid of LD mismatch × sample overlap (DESIGN §8). "Ideal" is L0.
- Runtime is always reported with accuracy.

### 1.2 Data
- **Simulation backbone:** 1000 Genomes 30x phased GRCh38.
  - Pools: NFE target A (254), reference B (150), FIN (99), ASW/ACB (157).
  - Li–Stephens mosaic synthetic populations (N_syn = 20,000) on the Beagle hg38 map.
  - Generator: the small-N trait at individual level, with its own in-sample LD and OLS statistics; the large-N trait in factor form.
- **Mismatch levels:**
  - L0 in-sample;
  - L1 mosaic reference n = 2,000;
  - L2 real external reference n = 150;
  - L3 admixed study vs EUR reference;
  - L4 L2 plus allele flips and missing variants;
  - L6 a Finnish-like GWAS vs NFE QTL.
- **Overlap pairing types:**
  - P1 eQTL × GWAS, nested or disjoint;
  - P2 pQTL-in-biobank, ρ up to 0.3;
  - P3 full overlap, ρ up to 0.8.

  All methods receive the estimated ρ̂, never the true ρ.
- **Real data:** eQTL Catalogue r7 (9 datasets), FinnGen R12 (8 traits, with in-sample LD for lead pairs) and GEUVADIS (in-sample LD from 1000G genotypes). All hg38.
- **Splits:** 1,358 tiled hg38 LD blocks, assigned to train/validation/test at 60/20/20. Each gene goes to its TSS block, and windows are clipped to same-split runs. The test split has 267 blocks and was used once.

## 2. Methods compared
- **Tier-1 baselines:**
  - coloc.abf;
  - coloc-SuSiE (susieR 0.14.2);
  - mismatch-aware coloc-SuSiE (susieR 0.16.6 with R_finite/eb_mix);
  - SharePro;
  - ColocBoost (two modes);
  - mvSuSiE-RSS;
  - prop.coloc and colocPropTest;
  - lead-SNP rule;
  - a strength-only control scorer.
- **Our families:**
  - **A — JointSuSiE:** GPU-batched bivariate SuSiE-RSS with null, 1-only, 2-only and both states; residual covariance V ⊗ R̃ with overlap ρ̂; η LD-mismatch regularisation estimated in IBSS; inclusion–exclusion posteriors of the oracle events; power-aware exclusion probabilities.
  - **B — ExactColoc:** exact enumeration of ≤ 2 causal variants per trait on a per-trait-union screen; a support-space RSS Bayes factor that handles overlap and near-singular LD exactly; GPU-batched; oracle-event posteriors.
  - **C — NeuralColoc (bold):** amortised neural posterior estimation; a transformer over screened SNPs with an LD attention bias, a BF-structured readout and exact allele-flip invariance; trained on train-split simulations with one mismatch mechanism (L6) held out.
  - **G — GPU coloc-SuSiE:** an engineering baseline that separates speed gains from modelling gains.

## 3. Screening (kill tests, validation split; pre-registered rules DESIGN §5–§6)
| Family | Test | Result | Verdict |
|---|---|---|---|
| A — JointSuSiE | A1: AUPRC(S vs D) at L0, ρ = 0 vs the best tier-1 baseline (2,000 loci, power-matched) | 0.893 vs coloc-SuSiE 0.978; Δ −0.085 [−0.105, −0.064]. Runtime 0.38× coloc-SuSiE | **Killed** (D13) |
| B — ExactColoc | B1 correctness (coloc.abf identity) | max \|ΔPP\| 3.6e-14 | pass |
| | B1 usefulness: false-S on D (r² 0.3–0.8) ∪ M at L0 + L2, matched-W thresholds | 0.217 vs coloc-SuSiE 0.132 | **fail** |
| | Rule (a) AUPRC | 0.977 vs 0.978 (Δ −0.002 ± 0.005) | pass |
| | Rule (b) confident calls on W | 1.2% vs coloc.abf 3.9% | pass |
| | Rule (c) false-S at L4 / ρ = 0.5 | 0.21 / 0.16 vs 0.05 / 0.06 | **kill** |
| | Rule (d) runtime | 0.126 s vs 23.5 s per locus (about 190× faster) | pass |
| | *Sensitivity: fixed 0.8 thresholds* | B1 passes (+0.110); no kill (lowest false-S at ρ = 0.5) | — |
| C — NeuralColoc (bold) | C0: reproduce coloc.abf in its own world | E[KL] 0.00072 (bar 0.01); q99 \|ΔPP4\| 0.043 | pass |
| | C1: reproduce the exact Bayes posterior (B, simulator prior) in world L0\* | natural-weighted E[KL] 0.0056 [0.0040, 0.0087] (bar 0.02); log loss 0.182 vs Bayes-optimal 0.177 | **pass** [!] |
| | Floor: beat LightGBM on hand-built features (development scale, 74k records) | ΔAUPRC −0.0004 [−0.0035, 0.0024] | not met |
| | C2 | not run (training data generation takes 5–6 h) | — |

The B1 verdict and the C1 scoring fix were recomputed by agents that did not produce them (reviews/p5_verify_B1.md, reviews/p5_verify_famC_scoring.md); the C1 fix was verified before the single evaluation. The A1 verdict was not independently recomputed at the time. It was recomputed at the end of the project, by an agent with its own code and not the project's metrics module: the AUPRC values (A 0.893, coloc-SuSiE 0.978) and the gap −0.085 were reproduced exactly, and the verdict is confirmed (reviews/p7_verify_A1.md).

**Matched-W finding.** The pre-registered C3 operating point is "the smallest threshold with ≤ 2.5% confident calls on W". It drives B's threshold to 0.013, because B puts essentially no P_S on weak loci, and B then calls 22% of distinct loci shared. At fixed 0.8 B has the lowest false-S rate of all methods. Kill-test operating points should constrain both error directions.

**Provisional (report only).** C's development model on the M4 validation subsample: realistic AUPRC 0.951 [0.940, 0.958], D-vs-W AUROC 0.995, 0.031 s per locus. The model was trained on 74k records from the pre-M3.5 simulator, and nothing was selected on it.

## 4. Validation results
Baselines were measured on the M4 validation subsample: 12,170 loci, 100 S / 100 D / 100 W per cell over the 34 grid cells. Slow methods used 40/40/40 (ColocBoost) or 20/20/20 (mvSuSiE). Failures are scored least-favourably, and CIs are 2,000-replicate BCa bootstraps over LD blocks. Source: reports/p4.m4_valid.md, artifact `m4_valid_baselines@2`.

| Method | Realistic AUPRC S-vs-D | Ideal (L0) AUPRC | Realistic AUROC D-vs-W | Median / p95 s per locus |
|---|---|---|---|---|
| coloc.abf (W 0.2, p12 5e-6) | 0.856 [0.842, 0.869] | 0.859 | 0.954 | 0.012 / 0.089 |
| coloc-SuSiE | 0.721 [0.700, 0.743] | **0.981** | 0.804 | 6.4 / 90 |
| ColocBoost (mode 1) | 0.756 [0.711, 0.779] | 0.842 | 0.923 | 0.54 / 1.5 |
| lead-SNP rule | 0.702 [0.683, 0.719] | 0.719 | — | 0.033 / 0.15 |
| ColocBoost | 0.680 [0.641, 0.724] | 0.861 | 0.753 | 3.0 / 31 |
| SharePro | 0.642 [0.628, 0.656] | 0.731 | 0.384 | 0.79 / 13 |
| colocPropTest | 0.626 [0.608, 0.641] | 0.611 | — | 0.93 / 2.8 |
| prop.coloc | 0.548 [0.527, 0.569] | 0.675 | — | 1.0 / 2.3 |
| strength-only control | 0.497 [0.480, 0.512] | 0.513 | 0.992 | 0.033 / 0.18 |

Mismatch-aware coloc-SuSiE and mvSuSiE did not finish on M4 before the freeze. Their operating points come from the screen set (P3-27). On the screen set's L0, ρ = 0 cell, AUPRC was 0.972 for mismatch-aware coloc-SuSiE and 0.642 for mvSuSiE.

What validation showed:
1. **Fine-mapping methods are best with exact LD and worst-hit by mismatched LD.** coloc-SuSiE is the best method in the ideal cell (0.981) but drops to 0.721 across the realistic grid. coloc.abf, which ignores LD, loses almost nothing (0.859 to 0.856) and is the best realistic baseline.
2. **Power matching works as designed.** The strength-only control scores 0.497 on S-vs-D, i.e. chance, so the S-vs-D endpoint is not driven by signal strength. On D-vs-W the same control scores 0.992, because weak loci are weak by construction. D-vs-W AUROC alone therefore cannot show that a method separates "distinct" from "too weak". The test run reports each method's paired difference to the control.
3. **Several methods fail outright in some cells, and claims there are blocked** (DESIGN §4.3, more than 1% failures per cell):
   - prop.coloc fails on 5–7% of L2/L4 loci: exactly singular LD sub-matrices in an n = 150 reference (P3-20, a method limitation);
   - coloc-SuSiE fails on 1.1–3.9% of L4 loci: susieR stops with "prior variance unreasonably large" (P3-29).
4. **Near-binary outputs make some pre-registered operating points degenerate.**
   - mvSuSiE under the hard lfsr rule never makes a confident S call at matched-W.
   - ColocBoost's t_D sits just above 0.
   - SharePro's and ColocBoost's recalibration intercepts diverge.

   These are reported as properties of the outputs. No method was retuned (P3-30).

## 5. Test-set results (single run)
- **Run:** one test run, nonce ab94c5a1, frozen code cab2599 (tag frozen-v1).
  - **Access log:** started 23:40, interrupted 01:53 (D21: mvSuSiE dropped under the pre-declared D17 overrun rule), resumed 01:53 under the same nonce and code, completed 04:43.
  - **Checks:** the pipeline's own verification passed (every harness run re-verified, every benchmark recomputed). It was then independently verified (§5.5).
- **Test set:** 13,207 simulated loci generated in the 267-block test split.
  - Classes: S 3,513, D 3,514, W 4,555, G 1,236, U 389.
  - Cells: the 34-cell grid (7 ideal L0 cells, 27 realistic L1–L6 cells) plus 4 shifted-arm cells (L1/L2 with the reference and study populations swapped).
  - The bootstrap resamples 53 LD-block clusters (verifier's count), fewer than the DESIGN planning assumption, so CIs are wider than planned.
  - Slow methods used nested subsamples: ColocBoost 40/40/40 per cell (4,467 loci).
- **Scoring:** failures scored least-favourably; 95% BCa CIs from 2,000 bootstrap replicates over LD blocks. Operating points and recalibration intercepts are the frozen validation values (P3-30), never refitted on test.
- **Claims:** C1–C4 read "no candidate (D16)". The deliverable is C5: how existing methods' error rates depend on LD mismatch and sample overlap.

### 5.1 Co-primary endpoints (realistic grid; main sample n = 13,207)
| Method | AUPRC S-vs-D realistic | AUPRC ideal (L0) | AUPRC shifted | AUROC D-vs-W realistic | Δ AUPRC vs strength control |
|---|---|---|---|---|---|
| **coloc-SuSiE, mismatch-aware** (susieR 0.16.6) | **0.900 [0.884, 0.914]** | 0.962 | 0.929 | 0.906 | +0.381 [0.360, 0.404] |
| coloc.abf (W 0.2) | 0.860 [0.846, 0.874] | 0.856 | 0.796 | 0.957 | +0.341 [0.319, 0.364] |
| ExactColoc (family B, descriptive) | 0.829 [0.808, 0.849] | **0.969** | 0.918 | 0.949 | +0.311 |
| coloc-SuSiE (susieR 0.14.2) | 0.717 [0.697, 0.741] | 0.966 | 0.890 | 0.815 | +0.198 |
| GPU coloc-SuSiE (family G, descriptive) | 0.719 [0.700, 0.744] | 0.966 | 0.890 | 0.816 | +0.200 |
| lead-SNP rule | 0.704 [0.675, 0.731] | 0.699 | 0.683 | — | +0.185 |
| SharePro | 0.635 [0.607, 0.658] | 0.716 | 0.703 | 0.386 | +0.116 |
| colocPropTest | 0.632 [0.610, 0.651] | 0.635 | 0.596 | — | +0.114 |
| prop.coloc | 0.552 [0.529, 0.577] | 0.682 | 0.614 | — | +0.033 |
| strength-only control | 0.519 [0.503, 0.537] | 0.471 | 0.569 | 0.992 | control |

ColocBoost was measured on the ColocBoost subsample (4,467 loci), with paired references from the same loci:

| Method | AUPRC realistic | AUPRC ideal | AUROC D-vs-W |
|---|---|---|---|
| ColocBoost (mode 1) | 0.787 [0.756, 0.827] | 0.827 | 0.938 |
| ColocBoost | 0.678 [0.625, 0.715] | 0.838 | 0.741 |
| coloc.abf, same loci | 0.881 [0.866, 0.910] | 0.883 | 0.962 |
| mismatch-aware coloc-SuSiE, same loci | 0.914 [0.884, 0.937] | 0.975 | 0.918 |

Excluding failed loci instead of scoring them least-favourably changes nothing materially. The exception is prop.coloc, which rises to 0.651 because it fails on 369 loci. Every comparative statement below holds under both scorings.

Some methods fail on more than 1% of loci in at least one cell: coloc-SuSiE (max 2.4%, L4), GPU coloc-SuSiE (2.1%), SharePro (1.2%), prop.coloc (26 cells, max 7.9%) and ColocBoost (6.7%). Per-cell claims involving these methods in those cells are blocked (DESIGN §4.3).

**Shifted-arm values are directional only** (DESIGN §8.5). One shifted cell has only 51 S∪D loci, and the verifier showed that plausible alternative power-matching bin conventions move shifted AUPRC values by up to 0.022. Realistic and ideal values are unaffected.

### 5.2 What drives the errors: LD mismatch, not sample overlap (claim C5)
Mean per-cell AUPRC S-vs-D by mismatch level (descriptive; per-cell n ≈ 100/100/100):

| Method | L0 in-sample | L1 large mosaic ref | L2 real ref n = 150 | L3 admixed | L4 L2 + flips/missing | L6 FIN vs NFE |
|---|---|---|---|---|---|---|
| coloc-SuSiE | 0.958 | 0.958 | 0.731 | 0.690 | 0.631 | 0.713 |
| mismatch-aware coloc-SuSiE | 0.960 | 0.962 | 0.925 | 0.902 | 0.834 | 0.866 |
| ExactColoc (B) | 0.973 | 0.965 | 0.812 | 0.883 | 0.729 | 0.806 |
| coloc.abf | 0.851 | 0.870 | 0.869 | 0.876 | 0.859 | 0.830 |
| SharePro | 0.714 | 0.699 | 0.625 | 0.628 | 0.615 | 0.609 |
| prop.coloc | 0.669 | 0.681 | 0.588 | 0.515 | 0.571 | 0.505 |

What the test set shows:
1. **LD-reference mismatch is the dominant source of error for LD-using methods.** coloc-SuSiE drops from 0.96 with in-sample LD to 0.69–0.73 with a small external reference (n = 150; L2, L3, L6). L4 (0.63) is reported but claim-blocked by its 1.1–2.4% failures. The drop is already complete at L2, the realistic "1000G-sized reference" case. A large, well-matched reference (L1, n = 2,000 mosaic) costs nothing.
2. **The mismatch-aware SuSiE extension (R_finite / eb_mix, susieR 0.16.6) recovers most of that loss.** It keeps 0.83–0.93 under mismatch, and it is the best method on the realistic grid: 0.900 vs coloc.abf 0.860. The CIs do not overlap. This comparison was not pre-registered as a paired test; it was descriptive, because no candidate was selected.

   Its gain over coloc.abf comes entirely from the mismatched cells. With in-sample LD, all three fine-mapping methods (coloc-SuSiE, mismatch-aware coloc-SuSiE, B) beat coloc.abf by about 0.1 AUPRC.
3. **coloc.abf is flat across mismatch levels (0.83–0.88), because it uses no LD.** Its errors are the single-causal-variant ones: false-S rises to 26% for D pairs in r² 0.8–0.95 and 36% for barely identifiable pairs (d in [2, 4)).
4. **Sample overlap barely matters once ρ̂ is supplied.** The pre-registered C4-type slopes of the false-S rate per 0.1 of realised ρ in P2/P3 cells are:
   - for every method, among W loci: within ±0.02 (largest +0.003 per 0.1 for coloc.abf and coloc-SuSiE in P3);
   - among D loci: within the margin except coloc-SuSiE, GPU coloc-SuSiE and SharePro in P2 (about −0.02 per 0.1, false-S *decreasing* with ρ) and ExactColoc in P3 (−0.018).

   None shows overlap inflating false sharing.
5. **The proportional tests cannot separate S from D.** prop.coloc and colocPropTest test proportionality, not a shared causal variant. They reach only 0.55–0.63, barely above the strength-only control (+0.03 and +0.11). prop.coloc fails on 3.2% of loci, and on up to 7.9% in L2–L4 cells.
6. **SharePro's P_D is not usable for D-vs-W** (AUROC 0.39, i.e. weak loci get more P_D than distinct ones). It calls S on 56% of D loci at its operating point.
7. **No method beats the strength-only control on D-vs-W AUROC** (best coloc.abf 0.957 vs control 0.992). W loci are weak by construction, so "distinct vs too weak" is solved by signal strength alone. The scientifically hard part is S vs D, and there the control is at chance (0.519).

### 5.3 Three-way calls at the frozen operating points (realistic, main sample)
| Method | recall S (weighted) | recall D (weighted) | false-S on D (unweighted share of D loci) | W called S / D (weighted) |
|---|---|---|---|---|
| mismatch-aware coloc-SuSiE (matched-W) | 0.813 | 0.813 | 0.095 | 0.036 / 0.024 |
| coloc.abf (matched-W) | 0.681 | 0.873 | 0.067 | 0.033 / 0.022 |
| coloc-SuSiE (matched-W) | 0.727 | 0.714 | 0.155 | 0.028 / 0.023 |
| ExactColoc, matched-W (t_S = 0.013) | 0.839 | 0.665 | **0.226** | 0.030 / 0.026 |
| ExactColoc, fixed 0.8 | 0.606 | 0.591 | **0.067** | **0.000** / 0.017 |
| SharePro (matched-W) | 0.949 | 0.256 | 0.564 | 0.034 / 0.025 |

- The validation-fitted matched-W thresholds transfer to test with W→S rates of 2.8–3.6%, slightly above the 2.5% target. That is part of the result, not a refit.
- **The matched-W finding replicates on test data.** ExactColoc at matched-W calls 23% of distinct loci shared. At fixed 0.8 it makes zero confident-S calls on 4,555 weak loci and has a 6.7% false-S rate, but its recall is lower.

### 5.4 Descriptive entries (no claims)
- **ExactColoc (B):**
  - best ideal-condition AUPRC (0.969) and D-vs-W AUROC (0.994);
  - more sensitive to mismatch than mismatch-aware SuSiE (0.73 at L4).
- **GPU coloc-SuSiE (G):** reproduces coloc-SuSiE's accuracy (0.719 vs 0.717 realistic; per-cell AUPRC agrees to about 0.01).

### 5.5 Independent verification
An agent that did not produce the run checked it (reviews/p7_verify_testrun.md, scripts in reviews/p7_verify_testrun/). **Verdict: verified with issues; no reported number changes.**
- **Integrity:**
  - every one of the 44 run manifests is at frozen commit cab2599 with a clean tree;
  - every operating point is identical to the frozen methods file;
  - the resume config is the frozen file minus mvSuSiE;
  - the access log reads started → interrupted → resumed → completed under one nonce.
- **Recomputation:** the verifier's own code (not importing src/eval) reproduced realistic, ideal and shifted AUPRC and AUROC for every method to within 5e-14. It also reproduced the C3 confusion matrices exactly, all real-arm numbers, and every runtime median.
- **The pipeline's own run verifier** passed all 20 real-arm runs. The sim runs could not be re-verified outside the pipeline, because the test lock (correctly) blocks them; the recomputation covers them.
- **Caveats now reflected above:**
  - shifted-arm magnitudes are fragile;
  - per-cell failure blocks;
  - mvSuSiE is NOT RUN, and the partial checkpoint left in the bundle is not used;
  - the `median s/locus` column of the bundle's summary.md is not a per-locus median, so runtimes here come from runtime.json;
  - numpy overflow warnings came from a diagnostic recalibration refit that P3-30 excludes and that is not reported.

## 6. Real data
The test-split real arms were built once, from frozen ID lists, inside the test run. CIs come from cluster bootstraps (2,000 replicates) over LD blocks (R2, R4b) or FinnGen credible sets (R3b).
- **R2** (partial positives): 800 eQTL Catalogue gene × tissue pairs from non-overlapping cohorts.
- **R3b** (pseudo-negatives): 1,237 FinnGen × eQTL pairs with disjoint, unlinked credible sets.
- **R4b:** 600 GEUVADIS genes, run with in-sample vs reference LD.

| Method | S recall on R2 at the threshold giving 5% confident-S on R3b | R4b agreement of three-way calls, in-sample vs reference LD |
|---|---|---|
| mismatch-aware coloc-SuSiE | **0.918 [0.897, 0.941]** | **0.927 [0.905, 0.946]** (among LD-using methods) |
| coloc.abf | 0.913 [0.893, 0.936] | 1.000 (uses no LD; trivially invariant) |
| coloc-SuSiE | 0.853 [0.818, 0.887] | 0.793 [0.755, 0.827] |
| SharePro | 0.830 [0.000, 0.934] | 0.847 [0.809, 0.886] |
| strength-only control | 0.134 [0.113, 0.343] | 1.000 (trivially) |

- Real data agree with the simulation ranking.
  - On R4b, swapping exact in-sample LD for a reference panel changes coloc-SuSiE's call on 21% of genes, against 7% for the mismatch-aware version.
  - On R2/R3b, mismatch-aware coloc-SuSiE and coloc.abf are equally good (0.918 vs 0.913, overlapping CIs), and both beat standard coloc-SuSiE.
- SharePro's R2 CI reaching 0 reflects bootstrap replicates in which its R3b 5% threshold makes no S calls.
- Real-data numbers are relative only. R2 contains some truly distinct pairs, and R3b may contain undetected sharing (§9).

## 7. Runtime
Per-locus compute on the test set's main sample (11,607 loci; worker-seconds = summed per-locus runtime including the LD read). The node was shared (64 cores, 1 GPU) and had no timing lock.

| Method | Hardware | Median s per locus | p95 s | Mean worker-s per locus | Relative to coloc-SuSiE |
|---|---|---|---|---|---|
| coloc.abf | 1 CPU core | 0.023 | 0.26 | 0.078 | 230× faster |
| lead-SNP / strength control | 1 core | 0.024–0.030 | 0.29 | 0.08 | — |
| **ExactColoc (B)** | GPU, batched | 0.056 | 0.12 | **0.061** | 294× per worker; **≈ 4.6× whole machine** |
| **GPU coloc-SuSiE (G)** | GPU, batched | 0.235 | 0.39 | 0.236 | 76× per worker; **≈ 1.2× whole machine** |
| ColocBoost (mode 1) | 1 core | 0.74 | 1.7 | 0.84 | 21× |
| SharePro | 1 core | 1.01 | 19.5 | 3.6 | 5× |
| prop.coloc / colocPropTest | 1 core | 1.2 / 1.4 | 2.6 / 3.4 | 1.2 / 1.6 | 11–15× |
| ColocBoost | 1 core | 3.6 | 36.9 | 10.5 | 1.7× |
| mismatch-aware coloc-SuSiE | 1 core | 12.6 | 41.2 | 15.8 | 1.1× |
| coloc-SuSiE (susieR 0.14.2) | 1 core | 7.1 | 71.9 | 17.9 | 1× |
| mvSuSiE (screen set only; not run on test) | 1 core | 25 | 233 | — | — |

**Speed vs accuracy.**
- "Relative" in the table is per worker: one GPU or one CPU core. The pre-registered measure (C2) is whole-machine throughput against coloc-SuSiE using all 64 cores.
  - coloc-SuSiE on 64 cores processes a locus every 17.9 / 64 ≈ 0.28 s.
  - G (0.236 s per locus on one GPU) is therefore about **1.2×** faster, and B (0.061 s) about **4.6×**. This agrees with the earlier M3 measurement (reports/famAG_M3.md: G about 1× a 64-core susieR).
  - Neither meets the ≥ 10× C2 bar.
  - The GPU methods leave the CPU cores free, so a GPU node can run both. That is a capacity gain, not a speed-up of one scan.
- G gives coloc-SuSiE's accuracy (realistic AUPRC 0.719 vs 0.717; ideal 0.966 vs 0.966).
- ExactColoc gives coloc-SuSiE-level ideal accuracy (0.969). Its whole-machine gain is about 4.6×; per worker it is 294×.
- The most accurate method under realistic LD, mismatch-aware coloc-SuSiE, is also among the slowest: 15.8 worker-s per locus, i.e. about 51 core-hours for a 11,607-locus scan. A GPU port of the mismatch-aware model, combining G's engine with susieR 0.16.6's R_finite model, is the obvious next engineering step.
- Pre-registered claim C2 (≥ 10× whole-machine throughput at non-inferior accuracy) applied only to a selected candidate, so it is reported as "no candidate".

## 8. Process: what the agent team did and what went wrong
The project ran with one PI at a time (three PI sessions in succession, with one accidental overlap, D11) supervising up to 6 Opus/Sonnet agents in separate git worktrees. All compute went through a small job queue (`scripts/orch`) with declared cores, RAM and VRAM, priorities, and code snapshots. State was kept in JOBS.yaml, TASKS.yaml and ARTIFACTS.yaml, plus a reconcile check. Rules learned along the way are in LESSONS.md (29 rules, each with its evidence). The incidents that cost the most time:

- **No orchestration before M3.5.** Five agents shared one GPU with no queue, which caused CUDA out-of-memory deaths and a stall nobody noticed for a whole session. The restart (PROMPT_RESTART_M3.5.md) introduced the queue, the registries and the rule that the PI loops on reconcile while jobs run.
- **Provenance.** Two simulation artifacts were built from uncommitted code. Bit-identical regeneration (72/72 and 102/102 files) settled their provenance, and since then every job runs from a committed snapshot.
- **Duplicate actors.** Messaging an agent that was running inside a workflow forked a duplicate, which corrupted its own data scan. Later two PI sessions ran at once and resumed duplicate agents, and one duplicate broke `run_eval.py` in a shared worktree (D11/D12). The fix is now a rule: only one PI session, and live agents and jobs are checked before anything is resumed.
- **Queue behaviour under load.** A GPU-blocked job at the head of the queue blocked CPU work behind it; the first fix starved the head job instead. Over-declared RAM kept critical jobs waiting about 2 h behind backfill. Priorities, reservations sized to each job's real share, and realistic RAM declarations fixed both.
- **Silent data corruption.** A corrupt chr7 VCF was converted to only 10% of its variants because stderr was discarded. Downloads are now checked against upstream md5s and run with pipefail.
- **Scientific errors caught by computation, not by reading.** Reviewers found these by running known-answer checks:
  - a Kronecker-order error;
  - generator scaling;
  - ρ̂ bias;
  - an unreachable ρ grid;
  - P_S/P_D formula errors;
  - non-OLS z/se inputs that made prop.coloc reject sharing on 70% of shared loci;
  - the famC class-weight scoring bug (fixed and verified to 1e-6 before its single evaluation).
- **Pre-registration that was not dry-run.** The matched-W kill rule was never tried on a calibrated method before it was frozen, and it killed the fastest well-ranking candidate (§3). C1 needed a scoring fix and a re-run, which left no time for C2.
- **Supervision gaps.** The user twice found that the research tree did not show what was running, and once that check-ins had stopped. A live-state table regenerated at every heartbeat and explicit resume procedures improved this, but did not prevent a later unsupervised 20:25–22:15 gap (D18).
- **Time accounting** (reviews/retro_final.md):
  - about 4.6 h of the 22-h budget (21%) ran without active supervision (two session-limit pauses and the D18 gap), and the duplicate-PI handover cost about 1 h more;
  - 104 of 228 queued jobs were cancelled.
- **Late overrun cut (D21).** The test run was submitted with an expected end (05:10) already past the 05:00 drop line, and the drop came only 2 h 13 min into the run, once mvSuSiE was measured at 2.4× its frozen time estimate. The drop itself was clean: same nonce and code, a methods file differing only by the mvSuSiE entry, and logged as interrupted/resumed.

Independent verification (reviews/):
- recomputed or reviewed by an agent that did not produce them: the B1 verdict, the C1 scoring fix, the orchestration infrastructure and the test-run result (§5.5);
- the A1 verdict, only at the end of the project, where it was confirmed (§3).

Home directory: before the HOME redirect (commit 9c99882, 10-26 14:31), installs wrote to the user's home: a line in `~/.conda/environments.txt`, plus cache entries under `~/.cache/uv`, `~/.cache/mamba` and `~/.mamba/pkgs`. Nothing was written there afterwards. Per the user's instruction those files were left untouched.

## 9. Limitations
- Per-window ρ̂ in cis windows absorbs shared cis signal: R2 eQTL×eQTL pairs with no donor overlap give ρ̂ ≈ 0.1. Overlap-aware methods may therefore down-weight true sharing on real eQTL×eQTL data (P3-23).
- The test set was reduced to 100/100/100 loci per cell (D7/P3-12) to fit the time budget, so per-cell results are descriptive.
- **No new method reached the test set as a primary candidate (D16).** A was killed on accuracy, B by the matched-W rule (§3), and C could not finish its ladder: C1 passed, the floor was not met at development scale, and C2 was not run. The test run therefore delivers the pre-registered minimum viable result: a benchmark of existing methods under realistic mismatch and overlap, plus runtimes. B and the GPU coloc-SuSiE are reported only as pre-declared descriptive entries.
- The matched-W operating point, applied to every method, penalises methods that put no P_S on weak loci. Results at the fixed 0.8 operating point are reported alongside but were not the pre-registered primary.
- **mvSuSiE was not run on the test set** (D21, the pre-declared D17 overrun cut 1). At about 170 s per locus it would have ended the run around 07:25, after the report deadline. The run was interrupted and resumed under the same nonce and the same frozen code, with a methods file identical apart from that entry. The access log records `started`, `interrupted` and `resumed`. mvSuSiE's validation and screen results (§4) are its only results.
- Mismatch-aware coloc-SuSiE, the best realistic method, uses operating points fitted on the screen set rather than the M4 validation sample (P3-27), because its M4 runs did not finish before the freeze. The screen set has a different cell mix. Its AUPRC/AUROC are threshold-free and unaffected; only its three-way calls (§5.3) depend on this.
- **Runtime units.** The headline speed-ups of G and B are per worker (one GPU vs one core). On whole-machine throughput they are about 1.2× and 4.6× (§7).
- **C1 depended on a scoring fix (P3-25).** The fix removed an erroneous class-weight correction and reduced C1's KL about 7× after the 4M-record pilot had failed the bar. It was independently verified before the single evaluation, but C1's pass rests on it.
- The simulation uses one population-genetic backbone (1000G 30x) with Li–Stephens mosaics. Real-data arms test relative behaviour only (partial positives, pseudo-negatives, in-sample vs reference LD), not absolute calibration.
- Runtimes were measured on a shared 64-core node with one GPU and no timing lock. Relative runtimes within a run are more reliable than absolute ones.

## 10. Follow-ups and remaining ideas
*Phase-2 status (§11): item 1 was tested on the improved version, mismatch-aware ExactColoc (B-mm). Under the two-sided operating point it was still killed at L4 (false-S 0.109 vs a 0.098 kill line), and it missed the candidacy bar. Item 2 was done; C failed C2. The GPU port in item 4 was done: it failed correctness by 1 fit and reached 4.77× throughput. Its validation-fitted operating points were also done. Items 3 and 5 remain open.*

1. **Re-run B's kill test with an operating point that constrains both error directions**, for example matched false-S on D and matched confident calls on W. At fixed 0.8, B had the lowest false-S rate, matched coloc-SuSiE's AUPRC (0.977 vs 0.978) and was about 190× faster per worker (about 4.6× on whole-machine throughput on test, §7). This is the most promising untested result of the project. It would have to be evaluated on fresh test data, because the current test set has been used.
2. **Finish family C's ladder.** Train at full scale (about 14 M loci from the M3.5 simulator, 5–6 h of data generation), then run C2 (realistic grid, held-out L6 mechanism). The learning curve was still falling at 14.4 M loci (E[KL] 0.0315 at 1 M loci down to 0.0056 at 14.4 M).
3. **Estimate ρ outside the cis window.** Use genome-wide null SNPs, or LD-score-style intercepts, so that shared cis signal does not inflate ρ̂ (P3-23).
4. **Robust fine-mapping under reference mismatch, made fast.** coloc-SuSiE's drop under a mismatched reference is the largest single effect in the benchmark. Mismatch-aware SuSiE recovers most of it (realistic 0.900 vs 0.717, §5), but it remains weakest at L4 (0.83), and it costs 16 CPU-s per locus. Two next steps follow: a GPU port of the R_finite model on G's engine, and validation-fitted operating points for it (P3-27).
5. **Family A** (the joint 4-state model) lost 0.085 AUPRC to coloc-SuSiE. Diagnosing whether the cause is the η regulariser or the IBSS state space was not pursued.

## 11. Phase 2: follow-ups on validation data (10-28, after the user's "Go")
**Verdicts, one line per track** (each verdict independently recomputed):
- **B-mm (mismatch-aware ExactColoc): killed.**
  - It misses the candidacy bar: M4 realistic AUPRC 0.860 vs mismatch-aware coloc-SuSiE 0.899, Δ −0.039 ± 0.009, about 4 SE. That is the robust reason.
  - It is also killed by rule (c) at L4, a point-estimate margin: 10.9% vs a 9.8% kill line.
- **G-mm (GPU mismatch-aware SuSiE): failed.**
  - Correctness rung: 1 of 24 non-converged fits fails even the post-hoc amended rule. Under the frozen 1e-3 bar alone, 4 fits fail.
  - Speed: 4.77×, short of the 5× bar.
- **C (NeuralColoc, full scale): failed rung C2.** It does not beat its own no-LD version: −0.001 ± 0.002, where it needed > +1 SE.

No candidate went to a second test. The fresh test set v2 was never generated, as pre-declared (DESIGN_P2 §7), and stays unused. The practical result of the project is unchanged:
- **mismatch-aware coloc-SuSiE** (susieR 0.16.6, an existing method) is the best method under realistic LD mismatch;
- **coloc.abf** is the fast alternative.

Phase 2 added validated operating thresholds for mismatch-aware coloc-SuSiE.

### 11.1 Design
DESIGN_P2.md was pre-registered before any phase-2 data existed, after one independent review round (reviews/p9_design_review.md: approve with must-changes, all adopted). It was frozen at 09:45 (tag design-p2-frozen).
- **Honesty statement:** phase 2 was chosen after seeing the v1 test results.
- **Phase-1 verdicts stand.** Nothing was re-run with altered metrics.
- **Unchanged from DESIGN v4:** metrics, labels, controls, splits.
- **Added: a two-sided operating point OP2**, applied to every method. It requires ≤ 2.5% confident-S on W and ≤ 10% confident-S on distinct loci. OP2 is used only by the phase-2 kill rules.
  - **Disclosure of the 10% constant.** It was chosen after seeing v1 results, and it sits just above mismatch-aware coloc-SuSiE's own false-S rate (0.095 on v1, 0.097 on M4), so that method's OP2 equals its OP1.
  - The same 10% is exactly what sets B-mm's and C's OP2 thresholds, and OP2 favoured B-mm relative to OP1.
  - A different constant could have changed B-mm's rule-(c) verdict, but not its candidacy miss.
- **Candidacy bar:** validation (M4) realistic AUPRC ≥ mismatch-aware coloc-SuSiE's − 1 paired SE.

### 11.2 Results (validation split only)
Independent recomputations:
- B-mm: reviews/p9_verify_Bmm.md;
- G-mm: reviews/p9_verify_Gmm.md;
- C: reviews/p9_verify_C2.md;
- the mismatch-aware coloc-SuSiE M4 numbers and thresholds: reviews/p9_verify_mmM4.md.
| Track | What it is | Result | Verdict |
|---|---|---|---|
| Mismatch-aware coloc-SuSiE on M4 (baseline) | Completes P3-27: phase 1 had only screen-set thresholds | Realistic AUPRC **0.899 [0.887, 0.913]**, 0 failures, 16 s/locus. Validated thresholds: t_S 0.840, t_D 0.732 (OP2 = OP1; false-S on D 9.7%); at these, recall S 0.782 and D 0.803, W→S 2.5% | Reference (reviews/p9_verify_mmM4.md: every value reproduced exactly) |
| B-mm: mismatch-aware ExactColoc | ExactColoc plus a reference-panel variance-inflated likelihood (DESIGN layer E) | Gate: a real improvement over phase-1 B (screen L2+L4 0.861 vs 0.822, +0.040 ± 0.006). Rules (a), (b), (d) pass. **Rule (c) at L4 under OP2 kills it:** false-S on D 10.9% vs coloc.abf 4.9% (kill line 9.8%), and 14.5% vs ColocBoost-m1 4.5%. B1 at OP2 also fails. **M4 candidacy bar missed by about 4 SE: 0.860 vs 0.899 (Δ −0.039 ± 0.009).** Runtime 0.18 s/locus | **Killed** (reviews/p9_verify_Bmm.md). The rule-(c) kill is a point-estimate kill (bootstrap P(no kill) 0.12–0.17 under the coloc.abf reading); the candidacy miss is robust |
| G-mm: GPU port of mismatch-aware SuSiE | Engineering: same model, batched on GPU | Identity vs R: converged fits 176/176 (max ΔPIP 2e-8, fp64 GPU). Non-converged fits: 23/24 under the amended rule (P4-3a, post-hoc) and 19/24 under P4-3 (also post-hoc, see §11.4). Under the frozen §2 bar of 1e-3 alone, 4 fits exceed it. The one failure is a chaotic, non-converging L4 fit that three G-mm builds also disagree on. Its PP.H4 is about 2e-11, so it changes no call. **Throughput** (§7.7, exclusive slot, 2,000 M4 loci): 31,838 loci/h on 1 GPU vs 6,670 for coloc_susie_mm on 40 memory-limited cores, i.e. **4.77×, short of the 5× bar**; 4.34× vs coloc-SuSiE 0.14.2 (2 complete repeats, median 7,340 loci/h; the third repeat was cut by the job timeout), short of the 10× C2 bar. About 60% of G-mm's time is serial HDF5 reading, not the GPU fit. A post-hoc 4-worker diagnostic was slower (13,118 loci/h) and is not a headline. | **Correctness rung failed as pre-registered** (P4-6), and the speed bar was missed → report-only engineering note (reviews/p9_verify_Gmm.md: confirmed, with the 0.14.2 repeat count corrected) |
| C: NeuralColoc at full scale | 600k training records (size fixed by a timing pilot), final checkpoint | M4 realistic AUPRC 0.964; it beats B (+0.130), has better ECE, region gap 0.005, beats the LightGBM floor (+0.005 > 1 SE). It does **not** beat its own no-LD version (0.965; −0.001 ± 0.002) | **C2 failed** → fast pre-filter only (reviews/p9_verify_C2.md) |

### 11.3 What the learned methods' scores do and do not mean
- The network, its no-LD version and the LightGBM floor all reach about 0.96 on M4, against 0.899 for mismatch-aware coloc-SuSiE.
- The verifier found no leakage:
  - no shared LD blocks between training and validation, and no window overlap even at base-pair level;
  - simulator metadata alone scores at chance (0.50–0.52).
- But the learned methods were trained and validated on the same simulator, priors and class definitions. **M4 therefore measures transfer to new genomic regions, not to real data or to a different data-generating model.**
- The no-LD network matching the full network says the gain comes from learning the simulator's prior, not from better LD modelling.
- These numbers are not evidence that NeuralColoc beats mismatch-aware coloc-SuSiE in practice. The network's 30 global inputs were not audited in code.

### 11.4 Phase-2 process notes
- **Session restart.** The PI session ended at about 11:10 and was resumed (D25). Orch jobs survived. The three method agents were resumed by message after checking that only one PI was live (LESSONS 20).
- **Post-hoc rules (disclosed).** The G-mm identity rule for non-converged fits had two post-hoc changes:
  - P4-3 was itself written after the identity status it cites. It was first labelled pre-result, and is now relabelled post-hoc (reviews/retro_p2.md).
  - P4-3a then amended P4-3 after the full result.

  Even the amended rule failed G-mm, and a second amendment was refused (P4-6). Other post-hoc items, all disclosed: the NeuralColoc floor-test tolerance (1e-6 → 1e-5, an engineering test only), the G-mm stream count tuned on the timing slice, and one post-hoc 4-worker timing diagnostic, run after the pre-registered timing, which was declined as a replacement (P4-8).
- **GPU contention.** NeuralColoc's data generation was placed on the single GPU, overran from about 100 to about 230 min, delaying B-mm's GPU runs by about 1.5 h and G-mm's GPU work by about 3.5 h. Allocation decisions: P4-7; the G-mm validation run was dropped.
- **Pre-results decisions.** B-mm's rule (c) comparator was fixed before any B-mm prediction existed (P4-1: 10:07 vs first prediction 11:28).
