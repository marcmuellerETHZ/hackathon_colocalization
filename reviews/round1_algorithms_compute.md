# Review round 1: algorithms and HPC/GPU (DESIGN.md v1)

Reviewer role: algorithms and HPC/GPU engineer. Hardware checked: 1x RTX 4090 (24 GB, idle), 64 CPUs, cgroup memory 128 GiB (2 GB per core), torch 2.14.1+cu130, R 4.4.1 with multithreaded OpenBLAS 0.3.34, susieR 0.14.2, coloc 5.2.3.

**Verdict: feasible, but changes are needed before M3.** The three families fit the hardware. However, the design has two algebra errors (generator scaling, and the Kronecker ordering in ExactColoc). The "≤3 causal" exact mode is computationally infeasible at the stated M. And the C2 speed claim is not well defined: as currently written, measured against a 64-core CPU baseline, the dense padded GPU JointSuSiE would probably miss the 10x target. Each problem has a cheap fix, listed below.

## Checks I ran (scripts in my scratchpad, not committed)

All timings are on this machine.

| # | Check | Result |
|---|---|---|
| K1 | Real LD: 1000G 30x EUR (n=503), chr22:30–31 Mb, biallelic SNVs, MAF ≥ 0.01 | p = 2,883 per 1 Mb. 1,631/2,883 SNPs have a proxy with \|r\| > 0.9999. 11,585 pairs have \|r\| > 0.9999 and 22,949 pairs have r² > 0.95. Collapsing \|r\| > 0.999 duplicates gives 1,586 groups (1.82x fewer SNPs, 0.30x the dense matvec cost). |
| K2 | Eigen-spectrum | n=503 reference: k(99%) = 255, k(99.9%) = 406. Crude dosage mosaic (N=10k, ~0.66 switches/Mb): k(99%) = 472, k(99.9%) = 1,144, min eigenvalue ≈ 0. The mosaic does **not** break perfect LD between tightly linked SNPs, so R_true is near-singular, not "full-rank". |
| K3 | Batched GPU GEMV, fp32, B×p×p times B×p×2 (both traits) | p=2000: 19.8 µs per locus-matvec. p=5000: 133 µs. Bandwidth is ~750–800 GB/s, so the step is memory-bound. At L=10 effects × 100 IBSS iterations this is 7.5 loci/s at p=5000 and 50 loci/s at p=2000. |
| K4 | Factor form R = XᵀX/n on the GPU, p=5000 | n=200: 12.3 µs per matvec (11x faster than dense). n=2000: 107 µs (no gain). |
| K5 | Batched Cholesky + solve + logdet on the GPU | 8×8: 179 M configs/s (fp32), 52 M/s (fp64). 4×4: 347 M/s (fp32). |
| K6 | `torch.linalg.eigh` on the GPU, fp32 | p=256: 2 ms. p=2000: 19 ms. p=5000: 132 ms. 512×(256×256) batched: 29 ms in total. |
| K7 | susieR single-thread (OPENBLAS_NUM_THREADS=1), real LD | p=2883: two `susie_rss` fits (2 causal, z consistent with R) took 2.4–4.0 s; `coloc.susie` with `runsusie` took 2.8 s. p=5766 (block-diagonal duplicate), consistent z: 4.9 s per trait. p=5766 with z **inconsistent** with R (noise not drawn from N(0,R), as under mismatch): **125 s**, peak RSS **3.2 GB**. `eigen(values only)` at p=2883: 1.2 s. susieR prints "consider installing Rfast", and Rfast is not installed. |
| K8 | Precision of high r values | bf16 maps 0.99 → 0.988 and 0.999 → **1.0**. fp16 maps 0.999 → 0.99902. Max error of R·b for a single λ=40 effect, relative to fp32: bf16 0.20, fp16 0.017. torch's TF32 matmul is off by default (verified). |
| K9 | ExactColoc algebra (numpy, random u of size 4, ρ=0.4, shared-effect correlation) | The full 8×8 marginal-likelihood BF equals the support-space form below to 2e-15. The DESIGN's Σ_alt written with (R̃_uu ⊗ I) under Σ_null = V ⊗ R̃_uu gives a different value (log BF 5.77 vs 5.10). |

## Errors found

**E1 (generator scaling, §2.2).** The text sets G = G_std/√N_syn, then e = Gᵀu with u ~ N(0, I/N_syn). This scales twice: Cov(e) = R_true/N_syn, so z is essentially noiseless. Correct to one of:
- G = G_std/√N_syn and u ~ N(0, I_Nsyn); or
- G = G_std and u ~ N(0, I/N_syn).

The verification in §7.1 would catch this, but the frozen spec must be right.

**E2 (Kronecker ordering, §5 Family B).** Σ_null = V ⊗ R̃_uu stacks the z vector trait-major. The mean map must then be (I₂ ⊗ R̃_uu), not (R̃_uu ⊗ I). The written formula gives a wrong BF (K9).

**E3 ("full-rank, smooth R_true", §2.1).** A Li–Stephens mosaic copies template haplotypes. SNP pairs in perfect LD in pool A stay in (near-)perfect LD unless a switch falls between them (rare at ~0.3 kb spacing). K1 and K2 show that about half of the SNPs have a perfect proxy, and the mosaic's minimum eigenvalue is ≈ 0. Consequences:
- R_true is near-singular, so any R_uu Cholesky in Family B fails or is ill-conditioned in fp32;
- the rationale given for the mosaic ("avoid rank-limited LD") is only partly true.

The statistics reviewer should judge the realism. For compute, the fix is change 2.

**E4 (≤3-causal exact mode is infeasible, §5 B).** With M=100, the per-trait configurations of size ≤3 number 1 + 100 + 4,950 + 161,700 = 166,751, so there are 2.8e10 pairs per locus. At the measured ≤347 M configs/s this is more than 80 s per locus (worse for 6×6 systems), i.e. slower than coloc-SuSiE. ≤2 per trait gives 5,051² = 25.5 M pairs per locus, about 0.07–0.15 s on the GPU. That is fine for a gold standard on screening-sized sets, but it is not a speed win over a 64-core coloc-SuSiE.

**E5 (§10 runtime estimate).** "coloc-SuSiE ~5–15 s per locus" holds only for consistent z. Under mismatch (L2–L4), IBSS iteration counts blow up (K7: 4.9 s → 125 s at the same p). Also, 64 workers at p ≈ 5k exceed 128 GB (about 3 GB per worker), so the realistic worker count is about 40. The benchmark wall time for the R baselines on the realistic grid is therefore likely 3–10x the §10 figure.

## Changes

### Must

**1. Fix E1 and E2 in the spec; write Family B in support space (§2.2, §5 B).**
For a configuration (γ1, γ2), let s be the support of λ (|s| = |γ1| + |γ2| ≤ 4). Then:

```
A_ss = (V⁻¹ ⊗ R̃)_ss
y_s  = ((V⁻¹ ⊗ I) z)_s
log BF = ½ y_sᵀ (D_s⁻¹ + A_ss)⁻¹ y_s − ½ log|I + D_s A_ss|
```

Why this form:
- It needs only entries of R̃, never R̃_uu⁻¹, so singular R_uu (E3) is harmless.
- The system is 4×4 instead of 8×8, about 2x the throughput (K5).
- It is exactly equal to the full form (K9) when the mean and the noise use the same R̃. State this condition explicitly: if the mean uses R and the noise uses R̃, the claimed exactness breaks.
- Use fp32 for each configuration and **fp64 for the log-sum-exp** over about 25 M configurations per hypothesis class.

**2. Make the exact mode budget-feasible (§5 B).**
- Default: ≤2 causal per trait, M ≤ 100 (25.5 M configs).
- "Extended ≤3" only with M ≤ 30 (about 2e7 configs), or by pruning: keep each trait's top-K configurations from single-trait BFs covering 1−1e-6 of the mass, then evaluate all K1×K2 pairs exactly. Pruning error must be verified against full enumeration on 200 validation loci.
- Cap the LD-proxy expansion at a hard M. Collapse exact duplicates (\|r\| ≥ 0.9999 in the inference LD) into one token with a log(group size) prior offset. Without this, proxies of the lead consume most of M (K1).
- Add a ρ = 0 fast path: the BF factorizes as BF1(γ1)·BF2(γ2) when there is no shared-effect correlation. The pair enumeration is then O(K1 + K2) BF evaluations plus a 25 M-element outer sum.

**3. Redefine C2 and the runtime protocol (§0 C2, §7.7, §8.1, §10).** As written, "≥10x faster than coloc-SuSiE" can be met trivially against single-thread R and possibly missed against the whole machine. Freeze the following now:
- **Primary speed metric:** whole-machine throughput (locus-pairs/hour) on a fixed *deployment workload*.
  - Distinct LD windows; each locus loads or builds its own LD from the same on-disk format for every method (e.g. reference genotypes, or float32 LD files, on local scratch).
  - No sharing of an LD window across simulated replicates. Simulation-mode batching inflates GPU numbers.
  - Report the mix of mismatch levels (L0/L2/L4 separately), because iteration counts differ by about 25x (K7).
- **CPU baselines:**
  - one process per core with OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1 (OpenBLAS here is pthreaded; otherwise 64×64 threads oversubscribe);
  - worker count set by the 2 GB/core limit and reported;
  - `Rfast` installed for susieR (its own recommendation) so the baseline is not handicapped;
  - SharePro under the same thread settings.
- **GPU methods:**
  - include host-side data preparation, H2D transfer, η/ρ̂ estimation, LD construction and padding;
  - report steady-state and cold-start (CUDA context, `torch.compile`) times separately;
  - report single-locus latency as well as batch throughput.
- **Secondary:** single-thread latency per locus-pair (median/p95).
- **Report an amortised comparator.** Precomputed per-trait SuSiE LBFs (as eQTL Catalogue ships them) plus gpu-coloc pairing. Also report a one-GWAS-vs-K-traits scenario (K = 50 tissues at one locus). In that setting per-trait fine-mapping amortises, and a joint per-pair model loses unless batched (change 7). C2 should state which scenario it claims.
- **Define "genome-scale" numerically.** Proposal: 1 GWAS × all protein-coding genes in 1 tissue (~18k pairs) end-to-end, including LD, in ≤ 15 min on one GPU plus 64 cores. Stretch: × 50 tissues in ≤ 12 h.

**4. LD representation for JointSuSiE and G (§5 A, §2.3).** "Padded B × p_max × p_max" dense tensors are memory-bandwidth-bound (K3): about 7.5 loci/s at p=5000 and 100 iterations, i.e. only a few times faster than about 40 R workers. Change to:
- **L1–L5 (all genotype-based references):** keep LD as the genotype factor X (n_ref × p; R = XᵀX/n) whenever n_ref < p/2. This applies to L2/L3/L4 (n = 200–235): 11x faster matvecs (K4) and 25x less memory, so B ≈ 2,000 loci at p=5000 fit in 24 GB.
- **L0 and L1:** use dense or a truncated eigen-factor, whichever is cheaper (k(99.9%) ≈ 1,100 at p ≈ 2,900, K2, i.e. ~2.6x gain; not worth complexity at first).
- **η estimation by ML:** do it from the thin SVD of X, O(n²p), or a cached eigendecomposition. log|R̃| and zᵀR̃⁻¹z are then O(p) per η grid value. A dense p=5000 eigh costs 132 ms on the GPU (K6), more than the whole IBSS, and 15–40 s single-thread on the CPU.
- **Bucket loci by p** (e.g. 5 quantile buckets) instead of padding to a global p_max. Padding a p ∈ [2,000, 5,000] mix to 5,000 wastes up to 6x memory and bandwidth.

**5. Numeric precision policy, frozen and tested (§5, §7).**
- The IBSS, Exact and η code paths run in fp32 or fp64 only. TF32 is asserted off in a test (`torch.backends.cuda.matmul.allow_tf32 is False`; never call `set_float32_matmul_precision('high')` in those modules).
- No bf16/fp16 storage of R: bf16 maps r = 0.999 to 1.0 and gives a 0.2 error in R·b at λ = 40 (K8). That is exactly the r² 0.8–0.95 vs ≥ 0.95 boundary the D/U classes depend on.
- The NPE may use bf16 autocast for the transformer, but LD features enter as fp32 transforms that spread the near-1 region, e.g. atanh(r) clipped, or log(1−r²), and are then cast.
- Add a test: GPU JointSuSiE fp32 vs an fp64 CPU reference on 50 loci, max |ΔPIP| < 1e-3.

### Should

**6. Kill-test order (§6; "cheapest first").**
- Build G (GPU SuSiE-RSS) first. It shares the IBSS/SER kernel with A, and its identity test against susieR (minutes) is the cheapest correctness anchor for A. A1 then only adds the bivariate SER.
- Then B1 (minutes on the GPU).
- C1 is the most expensive. Run it as a learning curve: train budgets of 15, 60 and 240 min; kill only if the extrapolated |ΔP| stays above 0.03. This prevents killing C for under-training.
- **C1 target mismatch.** An NPE trained on oracle labels converges to P(class | data) under the *simulator's* prior (log-uniform λ, locus mixture, ≤3 causal). Exact's posterior uses its own Gaussian prior and ≤2 causal. They disagree even for a perfect network. For C1, either train on data simulated from Exact's prior, or compare both to the oracle.

**7. Batch pairs that share a locus into one GEMM (§5 A).** For one locus and K traits (GWAS vs K tissues or genes with a common window), the IBSS matvec becomes R·[b₁, b₂⁽¹⁾…b₂⁽ᴷ⁾], a p × (K+1) GEMM. R is read once per sweep and the multiply runs on tensor cores (in fp32, not TF32). This is what makes the K-trait scenario in change 3 competitive with the amortised coloc-SuSiE. It needs the K=1 bivariate model generalised to "1 GWAS vs K independent QTL traits sharing the GWAS fit". Even as K separate bivariate fits sharing the R read, the gain is about K-fold in bandwidth.

**8. Straggler handling in batched IBSS (§5 A, §7.7).**
- Per-locus convergence masks.
- Continuous batching: refill finished slots from the queue instead of waiting for the slowest locus.
- A frozen max-iteration cap (e.g. 100, the same as susieR's default).
- Report the non-convergence rate per mismatch level. Under mismatch some loci take about 25x the iterations (K7), and a batch otherwise runs at its worst member's pace.

**9. Generator throughput and storage (§2.2, §10).**
- Draw ≥ 64–256 replicates per window load, so that GᵀU is a compute-bound GEMM: about 0.3 ms for N_syn=10k, p=3k, 256 draws. Otherwise the per-window genotype load (int8, 30–50 MB) dominates.
- Precompute per-window eigen features (K6: about 10 min for all train windows).
- Generate the mosaic per LD component in streamed chunks: chr1 × 20k haplotypes as int8 is about 16 GB, far above the 2 GB/process limit.
- Store mosaic genotypes 2-bit packed (about 25 GB genome-wide for 10k individuals) on scratch, not in RAM.
- Implement per-SNP N (L5) as K cohort-presence masks, z_j = Σ_k w_jk z_jk with K GEMMs, rather than per-SNP individual subsets (not a single GEMM).
- Note that the ~12k train windows overlap heavily, so "millions of loci" are re-draws over limited LD diversity. Track a train-vs-validation-LD gap for the NPE.

**10. Duplicate-SNP collapsing as an optional, tested variant (§5 A, C).** In the K1 window, collapsing \|r\| ≥ 0.999 duplicates cuts dense matvec cost to 0.30x.
- It is exact for SuSiE only when z also agrees within the group (true at L0). Under mismatch, z differences among reference-duplicates are mismatch evidence, so use group-mean z plus a DENTIST-style flag, and report the variant as a distinct method variant, never silently.
- For the NPE, use one token per duplicate group with a group-size feature. Otherwise 128 tokens cover only about 70 distinct signals.

**11. B1 identity with coloc.abf needs per-SNP prior variance (§5 B, §7.2).** coloc.abf puts prior W on the β scale (sd 0.15 for quantitative traits), so the z-scale prior is W/varbeta_j and differs by SNP. The per-SNP p1/p2/p12 priors must also map one-to-one. Otherwise B1 fails for spec reasons, not bugs. Pin the coloc version used for the identity check (installed: 5.2.3; CRAN is 6.0.3 per lit/methods.md). Use the same coloc version in §4 and in B1.

**12. Reproducibility on the GPU (§9).** Scatter/index_add and some reductions are non-deterministic. Use `torch.use_deterministic_algorithms(True)` where available, and tolerance-based tests (not bit-equality) elsewhere. Record the GPU driver, CUDA and torch versions in the run log.

### Could

**13. FinnGen R12 in-sample LD (R4).**
- Convert the tabix text (all pairs within 3 Mb) once per locus to float32 binary. Text parsing of about 10⁷ pairs per 1 Mb window is tens of seconds, and it must not enter method timings.
- The r² ≤ 0.01 truncation makes R indefinite: apply η or a PSD projection, the same for all methods.
- Sparse LD gives no GPU speed-up at this density. Keep it dense.

**14. NPE size.** A 6-layer, d=256 transformer over 256 tokens costs about 4e9 training FLOPs per locus. At about 30–40 TFLOPS effective bf16 on the 4090, 10 M loci take under 1 h. The "≤ 12 h per run" budget is generous: spend it on 3–4 architecture/size variants on validation rather than one long run. Seeds stay as frozen in §9.

## Realistic compute budget (replaces §10)

| Milestone | Work | Resource | Estimate |
|---|---|---|---|
| M3 | Mosaic generation (streamed per component), reference panels L1–L3, per-window factors/eigen features | 64 CPU + GPU | 3–6 h CPU wall; 25–100 GB scratch; eigen features < 30 min GPU |
| M3 | Generator vs individual-level verification; G identity vs susieR (100 loci); B1 (200 loci) | GPU + few CPU | < 1 h |
| M4 | Baselines on the screening set (24 cells × 500 = 12k loci): coloc.abf, coloc-SuSiE, SharePro, HyPrColoc, lead-SNP | about 40 R/Python workers | coloc-SuSiE 12k × (3–60 s, by level) ≈ 40–120 core-h, i.e. 1–3 h wall per method; ColocBoost and eCAVIAR (228 s/locus) on a 2k subset ≈ 130 core-h, about 3 h |
| M5 | A, G, B on the screening set | GPU | A/G: minutes. B (≤2/trait, M=100): 12k × ~0.1 s ≈ 20 min |
| M5 | C1 learning-curve kill-test | GPU | 15 + 60 + 240 min |
| M6 | NPE variants (3–4) × 2 seeds; A/E ablations | GPU, CPU feeding the generator | 1–2 GPU-days |
| M7 | Test split: realistic grid (5 levels × 4 ρ, about 2k loci per cell ≈ 40k loci) for all methods, plus the runtime harness | CPU-dominated | R baselines 40k × ~15 s ≈ 170 core-h ≈ 4–5 h wall per slow method; our methods < 1 h GPU |

The GPU sits idle during M4 and M7 baseline runs. Schedule NPE training concurrently, giving the generator at most about 8 cores so the CPU baselines keep the rest. Record the core split in the runtime log so throughput comparisons are not contaminated by contention: run timing benchmarks on an otherwise idle machine.
