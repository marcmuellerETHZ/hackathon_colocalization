# Benchmark dry_p9_v2__sim__subsample_564655fb

- loci: 112 (D 38, S 36, W 38); cells: 38; per-cell subsample: None
- bootstrap: 2000 replicates, resampling unit block (BCa 95%); seed [20261029, 'run_benchmark', 'bootstrap', 'dry_p9_v2__sim']
- strength-only control (D6(e)): table DRY in-sample placeholder
- operating points: given /cluster/scratch/mumarc/hackathon_eqtl_gwas/results/eval_test_dry/dry_p9_v2/DRY_operating_points_in_sample.json
- git 885ec7ec5c clean=True

## Co-primary endpoints (least_favourable; n = 112)

| method | realistic AUPRC S-vs-D | realistic AUROC D-vs-W | ideal AUPRC S-vs-D | ideal AUROC D-vs-W | shifted AUPRC S-vs-D | shifted AUROC D-vs-W | Δ AUPRC vs control (realistic) |
|---|---|---|---|---|---|---|---|
| coloc_abf | NA [NA, NA] | 0.992 [0.992, 0.994] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| coloc_susie_mm | NA [NA, NA] | 0.987 [0.987, 0.987] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| colocboost_m1 | NA [NA, NA] | 0.907 [0.907, 0.908] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| lead_snp | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| strength_only | NA [NA, NA] | 0.994 [0.993, 0.994] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | control |

## Co-primary endpoints (excluded; n = 112)

| method | realistic AUPRC S-vs-D | realistic AUROC D-vs-W | ideal AUPRC S-vs-D | ideal AUROC D-vs-W | shifted AUPRC S-vs-D | shifted AUROC D-vs-W | Δ AUPRC vs control (realistic) |
|---|---|---|---|---|---|---|---|
| coloc_abf | NA [NA, NA] | 0.992 [0.992, 0.994] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| coloc_susie_mm | NA [NA, NA] | 0.987 [0.987, 0.987] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| colocboost_m1 | NA [NA, NA] | 0.907 [0.907, 0.908] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| lead_snp | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| strength_only | NA [NA, NA] | 0.994 [0.993, 0.994] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | control |

## C3 operating points and calibration (least_favourable)

| method | t_S | t_D | W→S | W→D | recall S | recall D | ECE_S raw / recal | ECE_D raw / recal | failures | median s/locus |
|---|---|---|---|---|---|---|---|---|---|---|
| coloc_abf | 0.9311 | 0.2652 | 0.000 | 0.050 | 0.658 | 1.000 | 0.177 / 0.186 | 0.171 / 0.133 | 0 (max cell 0.000) | 0.0057 |
| coloc_susie_mm | 0.9311 | 0.2652 | 0.000 | 0.050 | 0.733 | 0.983 | 0.103 / 0.096 | 0.087 / 0.099 | 0 (max cell 0.000) | 6.2443 |
| colocboost_m1 | 0.6966 | 1.0000 | 0.017 | 0.000 | 0.658 | 0.000 | 0.171 / 0.053 | 0.194 / 0.054 | 0 (max cell 0.000) | 0.4247 |
| lead_snp | inf | inf | 0.000 | 0.000 | 0.000 | 0.000 | NA / NA | NA / NA | 0 (max cell 0.000) | 0.0083 |
| strength_only | 0.2357 | 0.2357 | 0.000 | 0.050 | 0.900 | 0.767 | 0.216 / 0.216 | 0.192 / 0.192 | 0 (max cell 0.000) | 0.0061 |

## Per-cell AUPRC S vs D (least_favourable)

| method | L0|P1|r_p=0.0 | L0|P1|r_p=0.6 | L0|P2|rho=0.0 | L0|P2|rho=0.3 | L0|P3|rho=0.0 | L0|P3|rho=0.5 | L0|P3|rho=0.8 | L1swap|P1|r_p=0.0 | L1swap|P1|r_p=0.6 | L1|P1|r_p=0.0 | L1|P1|r_p=0.6 | L1|P2|rho=0.0 | L1|P2|rho=0.3 | L1|P3|rho=0.0 | L1|P3|rho=0.5 | L1|P3|rho=0.8 | L2swap|P1|r_p=0.0 | L2swap|P1|r_p=0.6 | L2|P1|r_p=0.0 | L2|P1|r_p=0.6 | L2|P2|rho=0.0 | L2|P2|rho=0.3 | L2|P3|rho=0.0 | L2|P3|rho=0.5 | L2|P3|rho=0.8 | L3|P1|r_p=0.0 | L3|P2|rho=0.0 | L3|P3|rho=0.0 | L4|P1|r_p=0.0 | L4|P1|r_p=0.6 | L4|P2|rho=0.0 | L4|P2|rho=0.3 | L4|P3|rho=0.0 | L4|P3|rho=0.5 | L4|P3|rho=0.8 | L6|P1|r_p=0.0 | L6|P2|rho=0.0 | L6|P3|rho=0.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coloc_abf | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| coloc_susie_mm | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| colocboost_m1 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| lead_snp | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| strength_only | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

## Per-cell AUROC D vs W (least_favourable)

| method | L0|P1|r_p=0.0 | L0|P1|r_p=0.6 | L0|P2|rho=0.0 | L0|P2|rho=0.3 | L0|P3|rho=0.0 | L0|P3|rho=0.5 | L0|P3|rho=0.8 | L1swap|P1|r_p=0.0 | L1swap|P1|r_p=0.6 | L1|P1|r_p=0.0 | L1|P1|r_p=0.6 | L1|P2|rho=0.0 | L1|P2|rho=0.3 | L1|P3|rho=0.0 | L1|P3|rho=0.5 | L1|P3|rho=0.8 | L2swap|P1|r_p=0.0 | L2swap|P1|r_p=0.6 | L2|P1|r_p=0.0 | L2|P1|r_p=0.6 | L2|P2|rho=0.0 | L2|P2|rho=0.3 | L2|P3|rho=0.0 | L2|P3|rho=0.5 | L2|P3|rho=0.8 | L3|P1|r_p=0.0 | L3|P2|rho=0.0 | L3|P3|rho=0.0 | L4|P1|r_p=0.0 | L4|P1|r_p=0.6 | L4|P2|rho=0.0 | L4|P2|rho=0.3 | L4|P3|rho=0.0 | L4|P3|rho=0.5 | L4|P3|rho=0.8 | L6|P1|r_p=0.0 | L6|P2|rho=0.0 | L6|P3|rho=0.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coloc_abf | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| coloc_susie_mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| colocboost_m1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 |
| lead_snp | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| strength_only | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## D strata by r2 (least_favourable; AUPRC S vs D_b / false-S rate at t_S)

| method | [0,0.1) | [0.1,0.3) | [0.3,0.5) | [0.5,0.8) | [0.8,0.95) | out |
|---|---|---|---|---|---|---|
| coloc_abf | NA / 0.000 (n=23) | NA / 0.000 (n=14) | NA / 0.000 (n=1) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) |
| coloc_susie_mm | NA / 0.000 (n=23) | NA / 0.071 (n=14) | NA / 0.000 (n=1) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) |
| colocboost_m1 | NA / 0.000 (n=23) | NA / 0.000 (n=14) | NA / 0.000 (n=1) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) |
| lead_snp | NA / 0.000 (n=23) | NA / 0.000 (n=14) | NA / 0.000 (n=1) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) |
| strength_only | NA / 0.217 (n=23) | NA / 0.286 (n=14) | NA / 0.000 (n=1) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) |

## D strata by d (least_favourable; AUPRC S vs D_b / false-S rate at t_S)

| method | [2,4) | [4,8) | [8,16) | [16,inf) | out |
|---|---|---|---|---|---|
| coloc_abf | NA / NA (n=0) | NA / 0.000 (n=17) | NA / 0.000 (n=17) | NA / 0.000 (n=4) | NA / NA (n=0) |
| coloc_susie_mm | NA / NA (n=0) | NA / 0.000 (n=17) | NA / 0.059 (n=17) | NA / 0.000 (n=4) | NA / NA (n=0) |
| colocboost_m1 | NA / NA (n=0) | NA / 0.000 (n=17) | NA / 0.000 (n=17) | NA / 0.000 (n=4) | NA / NA (n=0) |
| lead_snp | NA / NA (n=0) | NA / 0.000 (n=17) | NA / 0.000 (n=17) | NA / 0.000 (n=4) | NA / NA (n=0) |
| strength_only | NA / NA (n=0) | NA / 0.235 (n=17) | NA / 0.235 (n=17) | NA / 0.250 (n=4) | NA / NA (n=0) |

