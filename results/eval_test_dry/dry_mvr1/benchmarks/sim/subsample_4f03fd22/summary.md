# Benchmark dry_mvr1__sim__subsample_4f03fd22

- loci: 106 (D 34, S 34, W 38); cells: 38; per-cell subsample: None
- bootstrap: 2000 replicates, resampling unit block (BCa 95%); seed [20261029, 'run_benchmark', 'bootstrap', 'dry_mvr1__sim']
- strength-only control (D6(e)): table DRY in-sample placeholder
- operating points: given /cluster/scratch/mumarc/hackathon_eqtl_gwas/results/eval_test_dry/dry_mvr1/DRY_operating_points_in_sample.json
- git c32b5114a9 clean=True

## Co-primary endpoints (least_favourable; n = 106)

| method | realistic AUPRC S-vs-D | realistic AUROC D-vs-W | ideal AUPRC S-vs-D | ideal AUROC D-vs-W | shifted AUPRC S-vs-D | shifted AUROC D-vs-W | Δ AUPRC vs control (realistic) |
|---|---|---|---|---|---|---|---|
| coloc_abf | NA [NA, NA] | 0.993 [0.993, 0.993] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| coloc_prop_test | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| coloc_susie | NA [NA, NA] | 0.951 [0.951, 0.951] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| coloc_susie_mm | NA [NA, NA] | 0.998 [0.998, 0.998] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| colocboost | NA [NA, NA] | 0.819 [0.819, 0.819] | NA [NA, NA] | 0.750 [0.750, 0.750] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| colocboost_m1 | NA [NA, NA] | 0.904 [0.904, 0.904] | NA [NA, NA] | 0.750 [0.750, 0.750] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| famB_exactcoloc_gaussian_M50 | NA [NA, NA] | 0.983 [0.983, 0.983] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| famG_coloc_susie_gpu | NA [NA, NA] | 0.951 [0.951, 0.951] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| lead_snp | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| mvsusie | NA [NA, NA] | 0.675 [0.675, 0.675] | NA [NA, NA] | 0.500 [0.500, 0.500] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| prop_coloc | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| sharepro | NA [NA, NA] | 0.635 [0.635, 0.635] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| strength_only | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | NA [NA, NA] | control |

## Co-primary endpoints (excluded; n = 101)

| method | realistic AUPRC S-vs-D | realistic AUROC D-vs-W | ideal AUPRC S-vs-D | ideal AUROC D-vs-W | shifted AUPRC S-vs-D | shifted AUROC D-vs-W | Δ AUPRC vs control (realistic) |
|---|---|---|---|---|---|---|---|
| coloc_abf | NA [NA, NA] | 0.993 [0.993, 0.993] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| coloc_prop_test | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| coloc_susie | NA [NA, NA] | 0.994 [0.994, 0.994] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| coloc_susie_mm | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| colocboost | NA [NA, NA] | 0.834 [0.834, 0.834] | NA [NA, NA] | 0.750 [0.750, 0.750] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| colocboost_m1 | NA [NA, NA] | 0.897 [0.897, 0.897] | NA [NA, NA] | 0.750 [0.750, 0.750] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| famB_exactcoloc_gaussian_M50 | NA [NA, NA] | 0.984 [0.984, 0.984] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| famG_coloc_susie_gpu | NA [NA, NA] | 0.994 [0.994, 0.994] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| lead_snp | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| mvsusie | NA [NA, NA] | 0.687 [0.687, 0.688] | NA [NA, NA] | 0.500 [0.500, 0.500] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| prop_coloc | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| sharepro | NA [NA, NA] | 0.659 [0.659, 0.659] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| strength_only | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | NA [NA, NA] | control |

## C3 operating points and calibration (least_favourable)

| method | t_S | t_D | W→S | W→D | recall S | recall D | ECE_S raw / recal | ECE_D raw / recal | failures | median s/locus |
|---|---|---|---|---|---|---|---|---|---|---|
| coloc_abf | 0.3803 | 0.9371 | 0.025 | 0.000 | 0.800 | 0.933 | 0.110 / 0.135 | 0.141 / 0.069 | 0 (max cell 0.000) | 0.0087 |
| coloc_prop_test | inf | inf | 0.000 | 0.000 | 0.000 | 0.000 | NA / NA | NA / NA | 0 (max cell 0.000) | 1.5399 |
| coloc_susie | 1.0000 | 1.0000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.070 / 0.139 | 0.117 / 0.111 | 1 (max cell 0.333) | 5.2091 |
| coloc_susie_mm | 0.3803 | 0.9989 | 0.025 | 0.000 | 0.892 | 0.950 | 0.066 / 0.130 | 0.102 / 0.083 | 0 (max cell 0.000) | 11.9350 |
| colocboost | 0.3318 | 1.0000 | 0.017 | 0.000 | 0.658 | 0.000 | 0.108 / 0.321 | 0.097 / 0.100 | 0 (max cell 0.000) | 2.7532 |
| colocboost_m1 | 0.2563 | 1.0000 | 0.000 | 0.000 | 0.800 | 0.000 | 0.114 / 0.679 | 0.164 / 0.038 | 0 (max cell 0.000) | 0.9032 |
| famB_exactcoloc_gaussian_M50 | 0.0043 | 0.7968 | 0.000 | 0.025 | 0.792 | 0.817 | 0.108 / 0.095 | 0.093 / 0.109 | 0 (max cell 0.000) | 0.1147 |
| famG_coloc_susie_gpu | 1.0000 | 1.0000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.076 / 0.140 | 0.123 / 0.111 | 1 (max cell 0.333) | 0.3228 |
| lead_snp | inf | inf | 0.000 | 0.000 | 0.000 | 0.000 | NA / NA | NA / NA | 0 (max cell 0.000) | 0.0129 |
| mvsusie | 1.0000 | 0.0000 | 0.000 | 0.000 | 0.000 | 0.350 | 0.189 / 0.321 | 0.179 / 0.679 | 0 (max cell 0.000) | 45.8944 |
| prop_coloc | inf | inf | 0.000 | 0.000 | 0.000 | 0.000 | NA / NA | NA / NA | 4 (max cell 0.333) | 0.9377 |
| sharepro | 0.1888 | 0.2257 | 0.017 | 0.000 | 0.933 | 0.542 | 0.041 / 0.092 | 0.095 / 0.100 | 0 (max cell 0.000) | 0.5499 |
| strength_only | 0.1944 | 0.1944 | 0.000 | 0.000 | 1.000 | 0.792 | 0.293 / 0.293 | 0.292 / 0.292 | 0 (max cell 0.000) | 0.0105 |

## Per-cell AUPRC S vs D (least_favourable)

| method | L0|P1|r_p=0.0 | L0|P1|r_p=0.6 | L0|P2|rho=0.0 | L0|P2|rho=0.3 | L0|P3|rho=0.0 | L0|P3|rho=0.5 | L0|P3|rho=0.8 | L1swap|P1|r_p=0.0 | L1swap|P1|r_p=0.6 | L1|P1|r_p=0.0 | L1|P1|r_p=0.6 | L1|P2|rho=0.0 | L1|P2|rho=0.3 | L1|P3|rho=0.0 | L1|P3|rho=0.5 | L1|P3|rho=0.8 | L2swap|P1|r_p=0.0 | L2swap|P1|r_p=0.6 | L2|P1|r_p=0.0 | L2|P1|r_p=0.6 | L2|P2|rho=0.0 | L2|P2|rho=0.3 | L2|P3|rho=0.0 | L2|P3|rho=0.5 | L2|P3|rho=0.8 | L3|P1|r_p=0.0 | L3|P2|rho=0.0 | L3|P3|rho=0.0 | L4|P1|r_p=0.0 | L4|P1|r_p=0.6 | L4|P2|rho=0.0 | L4|P2|rho=0.3 | L4|P3|rho=0.0 | L4|P3|rho=0.5 | L4|P3|rho=0.8 | L6|P1|r_p=0.0 | L6|P2|rho=0.0 | L6|P3|rho=0.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coloc_abf | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| coloc_prop_test | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| coloc_susie | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| coloc_susie_mm | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| colocboost | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| colocboost_m1 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| famB_exactcoloc_gaussian_M50 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| famG_coloc_susie_gpu | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| lead_snp | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| mvsusie | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| prop_coloc | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| sharepro | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| strength_only | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

## Per-cell AUROC D vs W (least_favourable)

| method | L0|P1|r_p=0.0 | L0|P1|r_p=0.6 | L0|P2|rho=0.0 | L0|P2|rho=0.3 | L0|P3|rho=0.0 | L0|P3|rho=0.5 | L0|P3|rho=0.8 | L1swap|P1|r_p=0.0 | L1swap|P1|r_p=0.6 | L1|P1|r_p=0.0 | L1|P1|r_p=0.6 | L1|P2|rho=0.0 | L1|P2|rho=0.3 | L1|P3|rho=0.0 | L1|P3|rho=0.5 | L1|P3|rho=0.8 | L2swap|P1|r_p=0.0 | L2swap|P1|r_p=0.6 | L2|P1|r_p=0.0 | L2|P1|r_p=0.6 | L2|P2|rho=0.0 | L2|P2|rho=0.3 | L2|P3|rho=0.0 | L2|P3|rho=0.5 | L2|P3|rho=0.8 | L3|P1|r_p=0.0 | L3|P2|rho=0.0 | L3|P3|rho=0.0 | L4|P1|r_p=0.0 | L4|P1|r_p=0.6 | L4|P2|rho=0.0 | L4|P2|rho=0.3 | L4|P3|rho=0.0 | L4|P3|rho=0.5 | L4|P3|rho=0.8 | L6|P1|r_p=0.0 | L6|P2|rho=0.0 | L6|P3|rho=0.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coloc_abf | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| coloc_prop_test | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| coloc_susie | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| coloc_susie_mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| colocboost | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 0.500 | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 0.500 |
| colocboost_m1 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 0.500 | NA | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 0.500 |
| famB_exactcoloc_gaussian_M50 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| famG_coloc_susie_gpu | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| lead_snp | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| mvsusie | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 1.000 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| prop_coloc | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| sharepro | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| strength_only | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## D strata by r2 (least_favourable; AUPRC S vs D_b / false-S rate at t_S)

| method | [0,0.1) | [0.1,0.3) | [0.3,0.5) | [0.5,0.8) | [0.8,0.95) | out |
|---|---|---|---|---|---|---|
| coloc_abf | NA / 0.000 (n=34) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) |
| coloc_prop_test | NA / 0.000 (n=34) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) |
| coloc_susie | NA / 0.000 (n=34) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) |
| coloc_susie_mm | NA / 0.000 (n=34) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) |
| colocboost | NA / 0.029 (n=34) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) |
| colocboost_m1 | NA / 0.000 (n=34) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) |
| famB_exactcoloc_gaussian_M50 | NA / 0.000 (n=34) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) |
| famG_coloc_susie_gpu | NA / 0.000 (n=34) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) |
| lead_snp | NA / 0.000 (n=34) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) |
| mvsusie | NA / 0.000 (n=34) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) |
| prop_coloc | NA / 0.000 (n=34) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) |
| sharepro | NA / 0.059 (n=34) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) |
| strength_only | NA / 0.176 (n=34) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) | NA / NA (n=0) |

## D strata by d (least_favourable; AUPRC S vs D_b / false-S rate at t_S)

| method | [2,4) | [4,8) | [8,16) | [16,inf) | out |
|---|---|---|---|---|---|
| coloc_abf | NA / NA (n=0) | NA / 0.000 (n=15) | NA / 0.000 (n=18) | NA / 0.000 (n=1) | NA / NA (n=0) |
| coloc_prop_test | NA / NA (n=0) | NA / 0.000 (n=15) | NA / 0.000 (n=18) | NA / 0.000 (n=1) | NA / NA (n=0) |
| coloc_susie | NA / NA (n=0) | NA / 0.000 (n=15) | NA / 0.000 (n=18) | NA / 0.000 (n=1) | NA / NA (n=0) |
| coloc_susie_mm | NA / NA (n=0) | NA / 0.000 (n=15) | NA / 0.000 (n=18) | NA / 0.000 (n=1) | NA / NA (n=0) |
| colocboost | NA / NA (n=0) | NA / 0.067 (n=15) | NA / 0.000 (n=18) | NA / 0.000 (n=1) | NA / NA (n=0) |
| colocboost_m1 | NA / NA (n=0) | NA / 0.000 (n=15) | NA / 0.000 (n=18) | NA / 0.000 (n=1) | NA / NA (n=0) |
| famB_exactcoloc_gaussian_M50 | NA / NA (n=0) | NA / 0.000 (n=15) | NA / 0.000 (n=18) | NA / 0.000 (n=1) | NA / NA (n=0) |
| famG_coloc_susie_gpu | NA / NA (n=0) | NA / 0.000 (n=15) | NA / 0.000 (n=18) | NA / 0.000 (n=1) | NA / NA (n=0) |
| lead_snp | NA / NA (n=0) | NA / 0.000 (n=15) | NA / 0.000 (n=18) | NA / 0.000 (n=1) | NA / NA (n=0) |
| mvsusie | NA / NA (n=0) | NA / 0.000 (n=15) | NA / 0.000 (n=18) | NA / 0.000 (n=1) | NA / NA (n=0) |
| prop_coloc | NA / NA (n=0) | NA / 0.000 (n=15) | NA / 0.000 (n=18) | NA / 0.000 (n=1) | NA / NA (n=0) |
| sharepro | NA / NA (n=0) | NA / 0.067 (n=15) | NA / 0.056 (n=18) | NA / 0.000 (n=1) | NA / NA (n=0) |
| strength_only | NA / NA (n=0) | NA / 0.200 (n=15) | NA / 0.167 (n=18) | NA / 0.000 (n=1) | NA / NA (n=0) |

