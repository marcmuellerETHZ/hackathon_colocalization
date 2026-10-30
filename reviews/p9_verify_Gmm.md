# p9 verify: G-mm phase-2 results

**Verdict: CONFIRMED.** The correctness rung FAILED as pre-registered, and throughput is 4.77× coloc_susie_mm, short of 5×. One correction (0.14.2 repeat count, item 3) changes no bar.

Scripts and outputs are in `reviews/p9_verify_Gmm/`. Everything ran on CPU only.

## 1. Checksums
My own re-hash (`checksums.py`) **matches** ARTIFACTS.yaml for famG_mm_identity@1, famG_mm_timing@1 and famG_mm_slice2000@1.

## 2. Identity (`identity.py`)
- **Inputs.** G-vs-R ΔPIP and G status come from `cmp_v7`; G-mm PIP vectors are not stored. R status, R PP.H4 and the 10-seed spreads were recomputed from the raw `r016/` and `pert/` JSONs.
- **Status** (convergence, failure and niter) agrees **200/200**. R has 24 non-converged fits and 0 failures.
- **Converged fits (176).** Max |ΔPIP| is **1.605e-8**. Max |ΔPP.H4| is 2.6e-9, over the 79 loci where both traits converged.
- **Non-converged fits (24).**
  - Strict P4-3: **19/24**.
  - Amended P4-3a: **23/24**. The one failure is **L4_P1_r_p=0.0__W__0007201 t1**: ΔPIP 0.474 vs spread 0.0207.
  - My spreads equal `sens_v7` exactly.
- **Verdict.** As written, P4-3a/P4-6 gives **FAIL**.
- **Minor.** The DECISIONS P4-6 text quotes numbers from an earlier build (ΔPIP 0.69, max 2.0e-8). The report matches the artifact.

## 3. Throughput (`timing.py`)
Harness wall times equal the LEDGER timestamps, and every run is complete.

| Method | Median loci/h |
|---|---|
| G-mm | **31,838** (30,766 / 33,852 / 31,838) |
| coloc_susie_mm | **6,670** (6,613 / 6,887 / 6,670) |

- **G-mm vs coloc_susie_mm: 4.774×.** Confirmed.
- **Correction on coloc-SuSiE 0.14.2:**
  - The timing outputs contain **two complete** 0.14.2 repeats: 944.4 s and 1020.5 s, i.e. 7,624/h and 7,056/h.
  - The timeout cut the **third** repeat. The report and the JOBS.yaml ack wrongly state n = 1.
  - With n = 2, the median is 7,340/h and the ratio **4.34×**, not 4.18×. This is still far short of 10×.
- **Exclusivity.** No other orch job overlaps the slot 16:17:09–18:12:09.

## 4. Spot check (`spot.py`, `spot_pip.sh`)
Locus L2_P1_r_p=0.0__G__0004402 (p = 1,425, both traits converged).
- **Real wrapper vs G-mm.** I ran the wrapper `ColocSusieMM` (fresh R) and G-mm on CPU torch fp64. Both report status ok, and |ΔP_S| = 2.2e-13.
- **Fresh vs stored R.** A fresh `run_susie016_mm.R` run equals the stored r016 bit for bit.
- **G-mm CPU vs fresh R.** Max |ΔPIP| is 1.2e-9.
