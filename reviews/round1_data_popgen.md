# Review round 1 — data science / population genetics

Reviewer: data / popgen agent (independent of the design author). Date: 2026-10-26. Scope: §2 (data, LD truth, mismatch levels, generator, real-data arms), §3 (splits), harmonisation and licences in DESIGN.md v1.

**Verdict: revise before M3.** The simulation backbone is sound in principle. However, five things need fixing before any data are generated:
- L3 models the GTEx case in the wrong direction.
- The generator cannot give the two traits different in-sample LD, which is the defining feature of eQTL × biobank coloc.
- The ρ̂ estimator as written is biased toward 0.
- R3 as written is not a locus-level D set.
- The split statistics (73 components; 15 test units for CIs) did not reproduce and are too coarse for cluster-bootstrap CIs.

Everything else is should/could. A sixth must: real-data harmonisation and variant QC (E5, change 9).

---

## A. Errors found (with evidence)

**E1. L3 is the wrong way round (must).**
- **What §2.3 L3 does:** it gives the *method* an 85% EUR + 15% AFR reference while the *truth* is pure EUR (pool-A mosaic). That models "a mixed reference for an EUR study", which nobody does.
- **The real GTEx case is the reverse.** The *study* is ancestry-mixed and the method gets an EUR reference. eQTL Catalogue `population_assignments.tsv` (fetched today) gives GTEx 838 donors = 705 EUR, 82 AFR, 10 EAS, 3 SAS, 38 unassigned. GEUVADIS has 445 = 358 EUR + 87 AFR.
- lit/data.md §2(a) already recommends "generate with ~85% EUR + 10% AFR …; analyse with EUR-only". DESIGN.md inverted this.

**E2. ρ̂ from null SNPs truncated at |z| < 2 is biased toward 0 (must).**
- §5 estimates ρ as the correlation of z over SNPs with |z| < 2 in both traits.
- For a bivariate normal truncated to |z1|, |z2| < 2, the sample correlation is (Monte Carlo, 2×10⁶ draws):

| True ρ | Correlation after truncation |
|---|---|
| 0.1 | 0.077 |
| 0.3 | 0.239 |
| 0.5 | 0.418 |

- That is a 16–23% under-estimate. Every method fed ρ̂ would under-correct for overlap, which inflates false-S exactly in the C4 regime.

**E3. Split numbers do not reproduce (must: fix and document).**
- **My recomputation:**
  - inputs: GENCODE v39 basic, `gene_type "protein_coding"`, gene-level TSS, autosomes;
  - blocks: pyrho_EUR (1,336);
  - linking: union-find over blocks touched by TSS ± 1 Mb.
- **Result:** **89 components (85 containing genes)**, median length 25 Mb, not 73 / 29 Mb.
  - Largest components: chr19 (830 genes), chr1 (822), chr17 (757), chr19 (631), chr6/MHC (608).
- **Unassigned genes:** **464 protein-coding genes (2.4%) have a TSS outside every block.** They fall in the 17 inter-block gaps, e.g. chr1:119.6–148.4 Mb (1q21) and chr2:87.9–98.4 Mb; chr17 alone has 83. §3 has no rule for them.
- Whatever definition produced 73 must be written down (script plus parameters) so the number can be reproduced.

**E4. L2 is harsher than labelled (should: relabel/extend).**
- §2.3 calls L2 a "1000G-sized external reference", but 1000G EUR is 503, not 200.
- **Upper bound on the noise.** If R_true were exactly pool-A sample LD, then R_true − R_B would carry the noise of both pools, and the effective reference size would be 1/(1/303 + 1/200) ≈ 120.
- **Measured.** In my mosaic experiment (§D) the mosaic removes most of pool A's sampling noise: ‖R_mosaic − R̂_A‖_F = 112, close to the 123 expected from pool-A noise alone. The measured mismatch ‖R_mosaic − R̂_B‖_F = 164 corresponds to an **effective n ≈ (p/‖·‖_F)² ≈ 170**.
- So L2 is a reference of n ≈ 170, about 3× noisier than the common practice of 1000G-EUR (n = 503) against a large GWAS.
- This does not invalidate L2, but the "realistic" headline (§8.1) would then be weighted toward a harsher-than-typical case.

**E5. Multiallelic sites are not removed by the prep script (must).**
- The NYGC panel stores multiallelic sites as *split biallelic records*. On chr22:20.0–21.0 Mb, 1,081 of 29,874 records share a position with another record.
- `scripts/prep_1kg_pgen.sh` uses `bcftools view -m2 -M2`, which acts per record, so all split records survive.
- After EUR MAF ≥ 0.01, **181 positions in that 1 Mb still carry ≥ 2 records** (about 8% of 4,454 variants). These are in strong negative LD with each other and break chr:pos-keyed joins.
- Also: the script filters MAF ≥ 0.005, while §2.1 says 0.01. Pick one and record it.

---

## B. Assessment of specific design choices

### B1. Li–Stephens mosaic as in-sample LD truth (§2.1)

- **Correct intent:** avoid a rank-≤605 R_true.
- **Pitfall:** what the mosaic actually is depends on one free parameter that §2.1 does not state: the switch (copying-jump) rate.
- **Measured** on chr22:20.0–20.6 Mb, a high-recombination window of 3.05 cM over 0.6 Mb; 2,144 SNVs; full table in §D. Using the LS rate 4N_e·r/K with K = 606:
  - **N_e = 10⁴:** mean copied segment ≈ 0.3 Mb. Mean r² at < 5 kb, 50–100 kb and 100–200 kb is 3%, 15% and 29% below pool A. Long-range r² (200–600 kb) falls from 0.0038 to 0.0004, i.e. pool-A sampling noise of ≈ 1/n is removed. Effective rank (eigenvalues to 99% of the trace) is 573 vs 236 for pool A.
  - **N_e = 10⁵:** short- and mid-range LD is destroyed: −10% at < 5 kb, −41% at 50–100 kb.
  - **N_e = 10³:** closest to pool A, but with ~3 Mb segments the mosaic is close to a bootstrap of pool-A haplotypes in typical-recombination regions.
- So the mosaic does what is intended (removes the low-rank, sample-noise texture), but **the result depends on N_e**, and the trade-off between rank and LD decay is real. The value must be fixed and justified. Because segment length scales with the local map, a single N_e behaves differently in cold and hot regions, so validate per region class.
- **Also:** without a miscopy/mutation term, every mosaic haplotype is locally identical to a pool-A haplotype. Rare-variant LD is then fixed to the 6–30 copies present in pool A.
- **Consequence:** the mosaic is fine as a *population* from which the study and the L1 reference are drawn, provided the design:
  - (i) fixes the switch rate and miscopy rate explicitly and justifies them;
  - (ii) reports the LD-decay match and effective rank against pool A (as §2.1 already proposes) *and against pool B*, which the mosaic never saw;
  - (iii) recognises that L0 and L1 share all mosaic artefacts.
- An independent LD world is worth having for the bold family (C), because NPE can overfit mosaic-specific LD texture. See change 3.

### B2. Pool sizes and composition

- **303/200 is defensible given 503 unrelated EUR.** It is better than 403/100, which would be too noisy a reference.
- **FIN imbalance.** FIN (99/503) is genetically distinct (bottleneck, longer LD). A seeded random split leaves FIN shares unbalanced (SD about ±4 individuals). Stratify the split by sub-population.
- **What the target population is.** As built, the target is a 5-way EUR blend that matches no real cohort.
- **The project's main real GWAS (FinnGen) is FIN; its main QTL source (GTEx) is about 85% NFE.** A FIN-vs-NFE mismatch level is directly relevant to R4 (FinnGen in-sample vs 1000G-EUR/FIN) and is currently missing from simulation. See change 2.

### B3. AFR mix

- About 15% "AFR" is close to GTEx's AFR + unassigned share (120/838 = 14.3%).
- **But GTEx AFR donors are African Americans** (admixed). The right donors are **ASW (61) + ACB (96)**, not all 661 AFR, which include YRI, LWK, GWD, MSL and ESN.
  - Pooling continental-African haplotypes overstates the divergence.
  - A mosaic from a mixed pool gives mixture LD without the long admixture-LD tracts of real admixed genomes. Using ASW/ACB haplotypes directly captures those tracts.

### B4. Single R_true for both traits (§2.2)

- The generator draws both traits from one R_true. In the target application (GTEx/OneK1K eQTL × FinnGen GWAS), **each trait has its own in-sample LD**: mixed-US vs Finnish.
- This is the most common real form of "LD mismatch" for coloc. Even a perfect per-trait LD cannot remove it, and joint models (Family A, B, C) must decide which R to use.
- When samples overlap (ρ ≠ 0), both studies are drawn from one population, so a single R is correct there. The extension is needed only for ρ = 0.

### B5. Variant filtering

- **Indels.** Keep indels in simulation; harmonise them carefully in real data. Evidence from a FinnGen R12 LD tabix probe over 100 kb at chr22:20.0–20.1 Mb:
  - 1000G EUR MAF ≥ 0.01 SNVs: **343/360 (95%)** match a FinnGen LD variant by exact `chr_pos_ref_alt`;
  - indels: **39/106 (37%)** match.
  - So indel joins will silently drop causal or lead variants. Normalise indels before joining (change 9).
- **MAF ≥ 0.01 in pool A** means as few as 6 copies; in pool B (400 haplotypes) a 1% variant is monomorphic with probability ≈ 0.99⁴⁰⁰ ≈ 1.8%.
  - The §2.1 requirement "present in all reference panels" silently makes the R_true variant set depend on the reference.
  - Better: define the variant set on pool A (MAF ≥ 0.01, plus MAF ≥ 0.01 in the mosaic) and let missingness in pool B be *real* L2 missingness. This is more realistic and frees L4's synthetic 5% drop to model something else.
- **Causal MAF bin 0.01–0.05** is one third of causals. Their LD in a 200-person reference is extremely noisy. That is realistic, but report results by causal-MAF bin so it is not hidden in the pooled headline.

### B6. λ range vs mismatch realism (§2.4)

- **Why it matters:** the mismatch-induced residual at SNP j is ≈ λ_c·(R_true − R_ref)_{jc}, with SD ≈ λ_c·(1 − r²)/√n_eff.
- **Worked example:** with n_eff ≈ 170 (L2) and λ = 40, the SD is up to about 3.1: a cloud of spurious |z| ≈ 3 "secondary signals" around every strong signal.
- **Real FinnGen leads** (N ≈ 500k) routinely have λ of 40–150 (e.g. UGT1A1–bilirubin, SORT1–LDL), and that is exactly where reference-LD fine-mapping breaks (Zou 2022 used N = 50k).
- **Trait 1 capped at λ = 40 under-samples the hardest mismatch regime.** See change 7.

### B7. Splits (§3)

- **Too few test clusters.** With 85 gene-bearing components, a 20% test split is about 17 components; under the design's own count it is about 15.
  - A percentile cluster bootstrap over 15–17 very unequal clusters (one test component can hold 4.5% of all genes) gives under-covering CIs.
  - It also makes the test set's LD content depend on which 2–3 giant components land there.
- **Clipping makes most of the merging unnecessary.** §3 already clips real-data windows to the component. For *simulation*, nothing requires gene-centred windows that cross block boundaries, and LD blocks are by construction the approximately independent LD units.
- **Proposal (change 5):**
  - keep the component-level *assignment*, which satisfies "each gene in a single split";
  - draw simulation windows *within single blocks*, or clip them to blocks;
  - bootstrap simulation metrics over **blocks** nested in test components: about 250 test blocks instead of about 15 units.
  - Report the component-level bootstrap as a sensitivity analysis.
- **MHC.** chr6:25–34 Mb (the 608-gene chr6 component includes it) has extreme LD and HLA imputation artefacts. It should be excluded from simulation and from the primary real-data analysis, or reported separately.

### B8. Real-data arms

| Arm | Feasibility with open data | Problem / fix |
|---|---|---|
| R1 | Yes | Add the FinnGen λ distribution (B6), and the per-SNP N spread (eQTL Catalogue `an`; FinnGen is near-constant). |
| R2 | Yes. eQTL Catalogue r7 has blood: GTEx QTD000356, Lepik_2017 QTD000373, TwinsUK QTD000549; LCL: GTEx QTD000221, GEUVADIS QTD000110, TwinsUK QTD000539. These cohorts do not overlap. | Labels use catalogue SuSiE, the same family as coloc-SuSiE. This is acceptable because catalogue SuSiE used **in-sample genotypes** while the methods under test get reference LD, but say so. Require the S label to be stable at lead r² > 0.8 in **both** cohorts' LD. Real positives also contain eQTLs whose causal variant differs between cohorts; treat R2 recall as a lower bound. |
| R3 | **Not at locus level as written.** If cohort A has signals {j, k} and B replicates only k, then the locus pair (A, B) *shares* k. It is class S (or M) at locus level; only the *signal* pair (A-j, B-k) is D. | Conditional summary statistics are not openly published for eQTL Catalogue (only per-effect `lbf_variable1..10`, which are unsigned and need the per-effect prior variance to invert to z). Conditioning with reference LD reintroduces exactly the mismatch under test. **Answer to §13 Q4:** R3 as written is evaluable only for methods that take per-signal LBFs (coloc bf_bf, gpu-coloc, or an LBF-input mode of A/B). Add R3b (change 6). |
| R4 | Yes. Remote tabix on `finngen_r12_chr22_ld.tsv.gz` took 0.9 s for 100 kb (474k rows). | **Contig is `22`, but variant IDs are `chr22_pos_ref_alt`.** All pairs with r² > 0.01 are stored both ways; pairs at \|r\| < 0.1 are absent, i.e. **zero**. Applying that truncation to a full-rank mosaic R over 2,144 SNVs zeroes 84% of entries and gives **674 negative eigenvalues (min −4.39)**. So assembled FinnGen 'in-sample' LD is far from PSD and must be repaired (PSD projection or λ-regularisation, with the same rule for every method). R4 is then 'near-in-sample', not exact; say so. Add a sim level L0t (R_true with \|r\| < 0.1 zeroed and repaired) so that this effect is quantified. Unphased r (PLINK2 `--r-unphased`) differs slightly from the haplotype r used for 1000G. In-sample calls are pseudo-truth for *agreement* only. FinnGen in-sample LD is GWAS-side only. **Missed opportunity:** GEUVADIS (QTD000110, N = 445: 358 EUR + 87 YRI) consists entirely of 1000G individuals, so **exact in-sample QTL LD is computable from the open 1000G 30x genotypes**. This contradicts lit/data.md finding 3 ("no open in-sample LD exists for any eQTL dataset"). It gives a real eQTL arm with true in-sample LD *and* a real admixed (20% YRI) study. Caveat: GEUVADIS samples are in pools A/B, so any real-data reference built from pool B is partly in-sample for GEUVADIS; use a GEUVADIS-excluded reference there. |
| R5 | (a) Feasible with real overlap. GTEx open-access sample attributes give donors per tissue, hence N_s; TwinsUK LCL/blood/skin/fat also come from shared donors. (b) FinnGen trait pairs have full overlap; shared controls induce ρ > 0 even for unrelated binary endpoints. | **There are no D labels**, because cross-tissue eQTLs of the same gene are mostly truly shared. A "false-sharing rate vs ρ̂" is only defined on pairs where trait 2 is null (W: trait-2 min p > 1e-3 in the window, trait 1 strong). Use a difference-in-differences: confident-S rate on W pairs from overlapping pairs (GTEx tissue × GTEx tissue; TwinsUK × TwinsUK) minus the same rate on non-overlapping pairs (GTEx tissue × TwinsUK/Lepik same tissue). For eQTL–eQTL pairs, ρ is **gene-specific** (r_p varies by gene), so a genome-wide ρ̂ is wrong per gene. A per-locus ρ̂ from roughly 100–300 LD-pruned null SNPs has an SE of about 0.06–0.1. Sim must therefore feed methods ρ̂ estimated by the same procedure, not true ρ (change 4). |
| R6 | Yes | Open Targets 26.09 used its own CS and GTEx v10. Compare on matched CS pairs only, and do not use it as truth. |

### B9. hg38 harmonisation

All primary sources are natively GRCh38: 1000G 30x; eQTL Catalogue r7, imputed to 1000G 30x; FinnGen R12; Open Targets 26.09. No liftover is needed, and §2.5's liftover sentence should state that it applies to no current arm.

The real risks are representation, not build:
1. **Contig naming:** `chr22` (1000G), `22` (FinnGen contig, eQTL Catalogue), `chr22_…` (FinnGen IDs), `23` = X in FinnGen.
2. **Indel normalisation** (B5).
3. **Split multiallelics** (E5).
4. **Effect allele:** alt in both FinnGen and eQTL Catalogue. Verify by an allele-frequency concordance check against pool-B frequencies, and flag |ΔAF| > 0.2, or > 0.1 at palindromic A/T and C/G SNPs.
5. **Lead-variant missingness:** a lead missing from the reference or from the other study is a classic cause of false non-coloc. Log it per locus.

### B10. Licences

- The lit/data.md table is adequate. Add these entries to DATA.md (currently empty):
  - **Beagle GRCh38 genetic maps:** derived from HapMap II, freely distributed; record source URL and sha256. The download script already records the sha256.
  - **eQTL Catalogue metadata tables** fetched from GitHub.
  - The **GEUVADIS sample list** if R4b is adopted.
- **FinnGen:** no formal licence. Do not commit any LD slice or sumstats slice, including small test fixtures. Tests must fetch them by script or use synthetic toy data.
- **MacDonald blocks** (no licence asserted): commit only block IDs and coordinates in the split table, which is already planned.

---

## C. Concrete changes

1. **[must] §2.3 L3: flip the direction and use admixed donors.**
   - New L3: trait-2 (QTL) z simulated from a mosaic of 85% pool-A EUR + 15% ASW/ACB haplotypes, or directly from ASW/ACB haplotypes (n ≈ 35 diploids) mixed into the pool.
   - The method receives pool-B EUR LD.
   - Keep the current "mixed reference" only as an optional ablation.

2. **[must] §2.2 generator: support trait-specific in-sample LD (R_1 ≠ R_2) when ρ = 0.**
   - Generate z_t ~ N(R_t λ_t, R_t), using the same causal variant IDs and the per-trait factor-based sampling already planned.
   - Add level **L6 "cross-population study pair"**: trait 1 FIN-like (mosaic from FIN haplotypes, all 99), trait 2 NFE or GTEx-like (L3 world); the method gets pool-B EUR.
   - Stratify the pool-A/B split by sub-population (CEU/GBR/TSI/IBS/FIN).
   - Write in §2.1 whether pool A contains FIN. Recommendation: keep FIN out of pool A, and use FIN as a separate mosaic population for L6 and a FIN-reference sensitivity analysis that mirrors R4.

3. **[should] §2.1 mosaic: pin and validate the parameters; add a second LD world on validation/test only.**
   - State the switch rate (e.g. LS 4N_e·r/K with N_e chosen so that the mosaic's r² decay matches the pool-B decay within 10% per distance bin) and a miscopy rate θ (e.g. Li–Stephens θ formula).
   - Add to the M3 gate: per-distance-bin r² decay and effective rank (eigenvalues to 99% of the trace, plus participation ratio) for the mosaic vs pool A vs **pool B**.
   - Add an msprime/stdpopsim world (HomSap OutOfAfrica_3G09 with the HapMapII_GRCh38 map) on a subset of validation and test cells. It is never used for NPE training. It catches Family C overfitting to mosaic LD texture, and answers §13 Q1.

4. **[must] §5 notation and §2.4: fix the ρ̂ estimator and evaluate with ρ̂, not ρ.**
   - Replace "correlation of SNPs with |z| < 2" with either the bivariate-normal MLE under the truncation, or the cross-trait LDSC intercept (for genome-wide GWAS–GWAS pairs).
   - Add a unit test: recover ρ ∈ {0.1, 0.3, 0.5} within 2 SE on null simulations.
   - Every method that consumes ρ gets the *estimated* ρ̂, using the same per-locus or per-pair estimator on real data.
   - Report the bias and SE of ρ̂ per locus.

5. **[must] §3 and §8.3: fix the split numbers and the CI unit.**
   - Publish the component-building script.
   - Reconcile 73 vs my 89/85, and state the rule for the 464 genes with a TSS in a block gap (proposal: assign to the nearest block).
   - Exclude MHC (chr6:25–34 Mb) from primary analyses.
   - Simulation windows lie inside one LD block (or are clipped to it). Simulation CIs come from a **block bootstrap within test components** (about 250 test blocks), with the component bootstrap as sensitivity.
   - For real-data CIs keep components, but report the number of units and use a BCa or t-bootstrap.
   - Also balance the assignment on simulation-relevant block counts, not only gene counts.

6. **[must] §2.5 R3: redefine it as signal-level for LBF-input methods, and add a locus-level natural-D arm R3b.**
   - **R3b construction:**
     - FinnGen trait × eQTL Catalogue gene pairs where *each* trait has exactly one in-sample SuSiE CS (FinnGen SuSiE uses in-sample LD; catalogue SuSiE uses genotypes);
     - both leads powered (|z| ≥ 6.29 GWAS; ≥ 5.26 QTL);
     - the CSs are disjoint, with max r² between CS members < 0.1 in **both** FinnGen in-sample LD and 1000G EUR (and < 0.1 in GEUVADIS/GTEx-like mixed LD for the QTL side);
     - FINEMAP agreement on FinnGen.
   - Pair it with R2 positives, power-matched on lead |z|.
   - Label honestly: "in-sample-fine-mapping-derived D"; methods are evaluated with reference LD, so the circularity is limited.

7. **[should] §2.4: widen trait-1 λ to log-uniform [3, 150] in a "biobank" stratum**, e.g. 30% of loci, or calibrate on R1 FinnGen leads, so that the mismatch regime that breaks fine-mapping is represented. Report AUPRC by λ_max bin.

8. **[should] §2.3 L2: relabel it "small external reference (measured n_eff ≈ 170 vs the population)"** and add the reference-size axis that lit/data.md proposed.
   - Mosaic-drawn references of n ∈ {200, 500, 2,000}: finite-sample noise only.
   - Pool B n = 200: real haplotypes plus pool noise.
   - Restrict the "realistic" headline pool to {L1, L2, L3, L4, L5, L6} with equal weights, and state the weights.

9. **[must] §2.1 / scripts: variant QC.**
   - (a) After `bcftools norm -m -any -f GRCh38_full_analysis_set…fa`, drop all records at positions with > 1 ALT allele after the MAF filter (or keep only the most frequent ALT), and record the count.
   - (b) Set one MAF threshold for §2.1 and the prep script.
   - (c) Exclude SV records explicitly (the script does so via `-v snps,indels`; say so).
   - (d) **Real-data primary analysis is SNV-only**, with indels as a sensitivity analysis after normalisation of every source against the same FASTA. Matching is by exact `chr:pos:ref:alt`, with an allele-swap check and an AF-concordance check; palindromic SNPs with MAF > 0.4 are dropped.
   - (e) Log the per-locus intersection rate and lead-variant missingness.

10. **[should] §2.5: add R4b, GEUVADIS × FinnGen with exact in-sample LD on *both* sides.**
    - QTL LD from the 445 GEUVADIS individuals in 1000G 30x; GWAS LD from FinnGen R12.
    - It is the only open setting where both traits have true in-sample LD and they differ (B4). It directly tests change 2 on real data.
    - Correct lit/data.md finding 3.

11. **[should] §2.5 R5: restrict the metric to null-trait-2 (W) pairs and use the overlapping-minus-non-overlapping difference** (B8). For eQTL–eQTL pairs, use per-gene ρ̂ with its SE.

12. **[should] §2.4 overlap grid: add ρ = 0.7–0.8** for correlated lab traits within FinnGen (e.g. LDL vs total cholesterol in largely the same individuals, where ρ ≈ r_p ≈ 0.9). Otherwise R5b exceeds the simulated range.

13. **[could] Per-SNP N (L5): implement via sample masks on the mosaic genotypes**, so the correlation N_ij/√(N_i·N_j) is exact by construction rather than approximate. Also add an imputation-quality arm: drop variants with eQTL Catalogue `r2` < 0.8 vs keep them.

14. **[could] Record in DATA.md** the FinnGen LD format quirks, the Beagle map source, the eQTL Catalogue metadata tables, and the GEUVADIS sample list.

---

## D. Checks run

All checks used the scratchpad only; no project files were changed apart from this file.

- **Power thresholds:** recomputed with scipy. z(5e-8) = 5.451 and z(1e-5) = 4.417. 80% power at 6.293 / 5.259; 20% power at 4.610 / 3.576. **The §1 numbers are correct.**
- **n_eff for L2:** the upper bound on noise is 1/(1/303 + 1/200) = 120.5; the measured value is about 170 (mosaic experiment below).
- **ρ̂ truncation bias:** see E2.
- **Split components:** see E3. Script: GENCODE v39 basic + pyrho_EUR blocks in `data/raw/annot`.
- **1000G 30x chr22:20–21 Mb** (bcftools from env/renv):
  - 29,874 records, of which 1,081 positions are duplicated (split multiallelics);
  - EUR (503) MAF ≥ 0.01: 4,454 variants, 181 multi-record positions, 944 indels (21%).
  - So p for ±500 kb is about 4–5k in gene-dense regions, at the top of the "2,000–5,000" range in §2.1; ±1 Mb is about 9k.
- **FinnGen R12 LD:**
  - README confirms PLINK2 `--r-unphased`, r² > 0.01, 3 Mb, 520,210 samples, both orientations stored;
  - a remote tabix query of 100 kb took 0.9 s (474,436 rows; 785 variants);
  - contig `22`, IDs `chr22_pos_ref_alt`;
  - variant match vs 1000G EUR: SNVs 95%, indels 37%.
- **eQTL Catalogue:**
  - population table: GTEx 705 EUR / 82 AFR / 38 unassigned of 838; GEUVADIS 358 EUR / 87 AFR;
  - tabix path table lists GEUVADIS QTD000110, GTEx LCL QTD000221, GTEx blood QTD000356, Lepik blood QTD000373, TwinsUK LCL QTD000539, TwinsUK blood QTD000549.
- **Mosaic experiment** (chr22:20.0–20.6 Mb, 2,144 SNVs, EUR MAF ≥ 0.01, Beagle GRCh38 map, pool A/B = 303/200 by seed 20261026, mosaic 10,000 diploids): results (script and output in the scratchpad, `mosaic.py`). Mean r² by distance bin [0–5k, 5–20k, 20–50k, 50–100k, 100–200k, 200–600k bp]:

| LD | Mean r² by distance bin | Effective rank 99% | Participation ratio | ‖·−R̂_A‖_F | ‖·−R̂_B‖_F |
|---|---|---|---|---|---|
| pool A (303) | .176 .118 .068 .036 .015 .0038 | 236 | 37 | 0 | 191 |
| pool B (200) | .176 .120 .070 .037 .016 .0056 | 170 | 35 | 191 | 0 |
| mosaic N_e = 10³ | .172 .114 .065 .033 .012 .0016 | 415 | 41 | 92 | 171 |
| mosaic N_e = 10⁴ | .170 .111 .062 .031 .010 .0004 | 573 | 44 | 112 | 164 |
| mosaic N_e = 10⁵ | .159 .096 .050 .021 .005 .0001 | 814 | 56 | 152 | 181 |

- L1 (independent mosaic of 2,000, N_e = 10⁴): ‖R_L1 − R_true‖_F = 52, close to the 48 expected (p/√2000).
- FinnGen-style truncation (|r| < 0.1 → 0) of the N_e = 10⁴ R_true: 84% of entries zeroed; minimum eigenvalue −4.39; 674/2,144 eigenvalues negative.
- **Caveats:** one window (a high-recombination one), uniform copying over the K donors, no miscopy term. Indicative only; the M3 gate should repeat this over about 50 windows spanning recombination-rate deciles.
