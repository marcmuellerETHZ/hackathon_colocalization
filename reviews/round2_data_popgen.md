# Review round 2 — data science / population genetics

Reviewer: data/popgen agent (fresh; did not author DESIGN.md). Date: 2026-10-26. Scope: §2.1–2.5 and §3 of DESIGN.md v2, checked against reviews/round1_data_popgen.md. Checks were run in a scratch directory under /tmp. No project file was changed apart from this one.

**Verdict: minor revision.** v2 addresses the round-1 must-changes in substance. Four new or residual problems would make specific results wrong as the text stands:
- **N1 (§3).** The gap-gene rule puts about 5% of test genes into windows that do not contain their own TSS.
- **N2 (§2.2/§2.4).** Most of the overlap grid cannot be realised with the stated N grid, and the nested-overlap construction gives the wrong ρ.
- **N3 (§2.4).** ρ̂ is simulated with genome-wide precision even for molecular-trait pairs, which flatters C4.
- **N4 (§2.2/§2.3, L6).** Cross-population effects are undefined at variants that are monomorphic in FIN.

All four are text and specification fixes costing about an hour each. None needs a redesign.

---

## 1. Round-1 must-change items (data/popgen)

| R1 item | Status | Reason |
|---|---|---|
| E1/C1: flip L3 and use admixed donors | **addressed** | The admixed study now gets an EUR reference, with ASW+ACB as donors. A small realism gap remains (PC adjustment; S5). |
| C2: per-trait in-sample LD; L6; stratified pools; FIN kept out of pool A | **addressed, but L6 is mis-specified** | The generator is individual-level for small N, L6 is defined, and pools are stratified. L6 lacks a per-population allele-frequency model (N4) and confounds FIN-vs-NFE LD with source-panel noise (S1). |
| E2/C4: ρ̂ bias; methods get ρ̂ | **partially** | The truncated MLE, its test, and ρ̂ for every method are in place. ρ̂ precision is unrealistic for molecular-trait pairs (N3). |
| E3/C5: reproducible splits, gap genes, MHC, CI unit | **partially** | Block-level assignment with 267 test blocks reproduces (§3 below). The gap rule is broken (N1). MHC coordinates differ between §2.1 and §3 (S6). |
| C6: R3 → signal-level; add R3b | **addressed** | R3b is feasible with open FinnGen R12 and eQTL Catalogue credible sets. |
| E5/C9: variant QC | **addressed** | SNV-only, split multiallelics removed downstream, MAF ≥ 0.01, harmonisation script, lead-missingness log. The prep script keeps MAF ≥ 0.005 as a superset, which is fine because filtering happens downstream. |
| (should) C7: biobank-strong λ | **partially** | §2.2 says this gives "λ up to about 150". Actually √(4·10⁵ · 0.02/0.98) = **90** (S3). |
| (should) C8: relabel L2; reference-size axis | **partially** | L2 is relabelled. The headline pool now puts the n = 150 pool-B reference in 4 of 5 cells (S2). |
| (should) C10: R4b | **partially** | Added, but without the required GEUVADIS-excluded reference (S4). |
| (should) C11, C12 | addressed / **broken** | R5b metric fixed. ρ = 0.8 was added to the grid but cannot be reached (N2). |

## 2. New or missed problems

### N1 [must] §3: the gap-gene rule plus clipping gives windows that exclude the TSS

**What I checked.** Seeded stratified 60/20/20 assignment of the 1,328 non-MHC pyrho-EUR blocks (seed 20261026, plus 200 other seeds), GENCODE v39 basic, gene TSS by strand, and ±500 kb windows clipped to the maximal run of consecutive same-split blocks.

**The gaps.**
- There are 18 inter-block gaps within chromosomes, plus uncovered chromosome ends.
- 509 protein-coding genes (2.7%) and 2,703 genes of all types have a TSS outside every block.
- Examples:
  - chr17:21.4–32.6 Mb: 83 protein-coding genes, euchromatic 17q11, including NF1 and SUZ12;
  - chr1:119.6–148.4 Mb: 54;
  - chr2:87.9–98.4 Mb: 52;
  - chr20:22.8–32.0 Mb: 49;
  - chr14/15/21 starts: 34, 22 and 17.

**The failure.** v2 sends each gap gene to its nearest block and then clips the window to that block's run. The window therefore lies entirely on one side of the TSS, or misses it:
- **4.7% of test protein-coding genes (180 of 3,866) have a TSS outside their own window.** For train genes it is 2.1%.
- For these genes the promoter-proximal eQTL lead falls outside the window more than 98% of the time.
- The effect is uneven: whichever split draws the chr17 or 1q21 gap blocks loses up to 80+ genes. It also breaks the §3 claim that "every variant belongs to exactly one split", because variants inside gaps belong to no block.

**Fix (in `make_splits.py`).**
- Make the units tile each chromosome:
  - each gap ≥ 1 Mb becomes its own pseudo-block (assigned like any block);
  - smaller gaps are split at the midpoint into the flanking blocks;
  - the first and last blocks are extended to the chromosome ends.
- Then assign genes by containing unit. Add to `test_splits.py` an assertion that every window contains its anchor TSS or lead.

### N2 [must] §2.2/§2.4: overlap grid vs N grid, and the nested-ρ scaling

**The grid is unreachable.**
- §2.2 makes every trait with N ≤ 20k individual-level, and N2 ∈ {100, …, 5,000}. So every simulated pair is "small-N trait nested in a large-N GWAS".
- There ρ = r_p·N_s/√(N1·N2) ≤ √(N2/N1):
  - 0.50 at (N2 = 5k, N1 = 20k);
  - 0.22 at (5k, 100k);
  - 0.016 at (100, 400k).
- So **ρ ∈ {0.3, 0.5, 0.8} is unreachable in nearly all cells, and 0.8 in all of them**, even though C4 and the realistic headline average over them.
- The "two large-N traits" mode cannot occur, because no N2 ≥ 20k exists.

**The scaling is wrong.**
- The large-N trait's noise is e = G_popᵀu/√N_syn, built from N_syn = 20,000 rows that stand for N1 individuals.
- Correlating ε₂ with u at r_p on N_s shared rows gives Cov = r_p·√(N_s²/(N2·N_syn))·R, not r_p·N_s/√(N1·N2)·R. That is √(N1/N_syn) too large: 4.5× at N1 = 400k.
- **Fix:** use r_p′ = r_p·√(N_syn/N1).

**Fix.** Define the ρ grid per pairing type and say which cells exist:
- (a) **GWAS×GWAS or pQTL×GWAS:** add N2 ∈ {20k, 50k} in factor form (UKB-PPP-like), with any ρ;
- (b) **eQTL×eQTL, same donors** (GTEx cross-tissue, TwinsUK; both individual-level): ρ up to r_p;
- (c) **nested eQTL ⊂ GWAS:** ρ ≤ √(N2/N1), using the r_p′ scaling.

C4 slopes should then be reported per pairing type. The existing "ρ verified empirically" test would catch the scaling error, but only after data generation.

### N3 [must] §2.4: ρ̂ precision is that of genome-wide GWAS for all pairs

- ρ̂ is simulated from 20,000 independent SNPs, so SE ≈ 0.007.
- That is right only for GWAS×GWAS. For any pair that involves a molecular trait, the data that exist are the cis window: about 100–300 LD-independent null SNPs per gene (SE ≈ 0.06–0.1).
- For same-donor eQTL×eQTL, ρ is also gene-specific (r_p varies by gene).
- As written, C4 ("under estimated ρ̂") is tested with near-oracle ρ̂ in exactly the regime where it is hardest.

**Fix:**
- Make the number of SNPs depend on pairing type: 20k for (a); per-locus window, n_ind ~ U[100, 300], for (b) and (c).
- On real data use the same per-gene estimator.
- Report C4 at both precisions.

### N4 [must] §2.2/§2.3, L6 (and L3): per-population allele frequency of shared causal variants

**Measured** on chr22:30–31 Mb and chr5:130–131 Mb, among SNVs with NFE MAF ≥ 0.01:
- **2.8–2.9% are monomorphic in FIN (99 individuals);**
- 6.8–8.9% have FIN MAF < 0.01.

**Consequences.**
- In L6 the trait-1 (mosaic-FIN) study has undefined standardized z at these variants.
- A "shared" causal variant drawn from the 0.01–0.05 NFE-MAF bin can have no variance in FIN, yet the generator assigns it a PVE and a λ in both traits.

**Fix:**
- Draw the shared effect on the allelic scale, β, and set λ_t ∝ β·√(2f_t(1−f_t)·N_t) with f_t the frequency in trait t's population.
- Variants monomorphic in a study are missing for that trait. That is real missingness, and it is logged.
- The oracle uses the per-trait λ_t and R_t.

The same applies, more mildly, to L3, where the ASW/ACB admixture shifts f.

## 3. Should / could

**S1 [should] L6 needs a matched-source-size control.**

| Mismatch | chr22 | chr5 |
|---|---|---|
| FIN99 vs pool-B150 | 310 | 225 |
| NFE99 vs pool-B150 (same source size) | 236 | 158 |
| L2-like (A254 vs B150) | 200 | 143 |

Values are Frobenius mismatches.
- Only about half of L6's excess squared mismatch over L2 is FIN-specific. The rest is the 99-person source panel's sampling texture, which the mosaic carries as "truth".
- r² decay of FIN99 and NFE99 is the same up to 100 kb.
- Add L6-null: a trait-1 mosaic from 99 random NFE individuals. Report L6 minus L6-null as the population effect. Otherwise "robust to cross-population mismatch" is not identified.

**S2 [should] §2.3/§8.1: n_ref semantics and headline weight.**
- (i) State whether reference R is computed from phased haplotypes (2n) or dosages (n). Haplotype r has about half the sampling variance. My A254-vs-B150 mismatch (200, chr22, p = 2,808) matches p²(1/508 + 1/300) for haplotypes.
- Recommendation: dosage-based reference LD, as practitioners compute it with PLINK, and n_ref = individuals for `R_finite`/SharePro, applied identically to every method.
- (ii) L2, L3, L4 and L6 all use a 150-person reference, which is smaller than the common 1000G-EUR (503) or a biobank reference. Either add L1 at n_ref = 500 (a mosaic draw) to the realistic pool, or call the headline "small-reference realistic" in C1.

**S3 [should] §2.2.** Correct the claim to λ ≤ 90, or extend the biobank stratum to q² ≤ 0.055 (λ ≈ 150 at 400k), so that it matches the R1 FinnGen leads.

**S4 [should] R4b.** The GEUVADIS individuals are in 1000G and so in pools A and B: 358 EUR = CEU/GBR/TSI/FIN, no IBS. The out-of-sample reference must exclude them, which leaves IBS 107 plus the non-GEUVADIS CEU/GBR/TSI. State this and the resulting n_ref.

**S5 [should] L3 realism.** Real eQTL pipelines adjust for genotype PCs, so the effective in-sample LD is ancestry-adjusted.
- Regress ancestry (an indicator, or the top PCs of the study sample) out of G_t in L3, for both the z computation and the L0 in-sample R.
- Otherwise L3 overstates admixture LD.
- Measured L3-like mismatch vs B150 is only 219 vs 200 (chr22) and 172 vs 143 (chr5), so L3 is a mild level. Report it as such.

**S6 [could] MHC.** §2.1 excludes chr6:28.5–33.4 Mb, but the 8 pyrho blocks that overlap it span chr6:27,258,180–33,988,587. Use the block span in both places, or genes in 27.26–28.5 and 33.4–33.99 Mb have no unit.

**S7 [could] Harmonisation AF check.** FinnGen vs pool-B EUR: 6.6–9.3% of SNVs have |ΔAF| > 0.1, so the palindrome rule over-drops in FIN. Check FinnGen against 1000G FIN (or its own `af_alt` vs gnomAD-FIN), and GTEx against an 85/15 NFE/AFR mix. Only true strand ambiguity should drop a SNP.

**S8 [could] Mosaic validation.** Validate decay and spectrum also against pool B, which the mosaic never saw, over about 50 windows spanning recombination deciles, as round 1 asked. Smoke-test the msprime world on validation regions (no tuning) before its first test use in §8.5.

## 4. Split design check (requested)

Setup:
- 1,336 pyrho-EUR blocks; 8 MHC blocks removed; median block 1.82 Mb (5–95%: 0.68–3.59);
- 80 blocks without a protein-coding gene;
- 60/20/20 stratified by chromosome.

Results:
- **Test units:** 267 blocks, matching §3. Test protein-coding genes: 3,866 (20.5%). Over 200 seeds, median 3,813 (95% range 3,452–4,288). Test runs of consecutive same-split blocks: median 2.3 Mb (10–90%: 0.87–4.6).
- **How many windows reach the clip:** an unclipped ±500 kb window touches more than one block for 45% of genes.

Clipped window length (protein-coding genes, seed 20261026):

| Split | Unclipped (1 Mb) | q10 | q25 | < 500 kb | TSS ≤ 100 kb from a clip edge |
|---|---|---|---|---|---|
| train | 74.7% | 678 kb | 994 kb | 2.3% | 6.5% |
| val | 62.1% | 612 kb | 816 kb | 1.9% | 8.7% |
| test | 56.2% (95% over seeds: 55–63%) | 568 kb | 762 kb | 5.3% | 12.0% |

Illustrative lead loss, assuming the eQTL lead offset is Laplace with scale 20/50/100 kb:

| Genes | Lead outside window |
|---|---|
| non-gap test | 0.8% / 1.9% / 3.9% |
| non-gap train | 0.4% / 1.1% / 2.2% |
| gap test | > 98% (N1) |

**Verdict on clipping: not too destructive once N1 is fixed.** Two residual issues remain.

1. **Train and test windows differ in shape.** Test windows are more often clipped (44% vs 25%) and more often have the anchor near an edge, so p and edge position shift between training and test.
   - Log p and the clip status per locus.
   - Report the headline also restricted to unclipped windows (a sensitivity analysis).
   - For NPE training, sample window offsets so that the train distribution covers the test clip distribution.
2. **For real data, clipping is not needed against leakage** (no model is fitted on real data; R1 realism uses train regions only). Run real arms on clipped windows (primary, as specified) and on unclipped windows (sensitivity), and log the gene-level lead-outside-window rate.

## 5. Checks run (scratch only)

- `splits.py` / `fix.py`: block and gene assignment, clipping, 200-seed distribution, gap census, Laplace lead-loss illustration, and the λ range.
- `l6.py`: 1000G 30x chr22:30–31 Mb (p = 2,808) and chr5:130–131 Mb (p = 2,087), biallelic SNVs with unique position and NFE MAF ≥ 0.01. It computes:
  - FIN monomorphism;
  - Frobenius LD mismatch for FIN99, NFE99, A254, an 85/15 admixed mix, each vs B150;
  - r² decay by distance;
  - AF concordance (FIN and GTEx-like vs pool B).
- Analytic: the feasible ρ range √(N2/N1) and the N_syn scaling of nested overlap.
