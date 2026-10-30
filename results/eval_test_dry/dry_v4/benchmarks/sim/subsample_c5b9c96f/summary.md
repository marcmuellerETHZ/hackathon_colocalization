# Benchmark dry_v4__sim__subsample_c5b9c96f

- loci: 111 (D 37, S 36, W 38); cells: 38; per-cell subsample: None
- bootstrap: 2000 replicates, resampling unit block (BCa 95%); seed [20261029, 'run_benchmark', 'bootstrap', 'dry_v4__sim']
- strength-only control (D6(e)): table DRY in-sample placeholder
- operating points: given /cluster/scratch/mumarc/hackathon_eqtl_gwas/results/eval_test_dry/dry_v4/DRY_operating_points_in_sample.json
- git df33fb5346 clean=True

## Co-primary endpoints (least_favourable; n = 111)

| method | realistic AUPRC S-vs-D | realistic AUROC D-vs-W | ideal AUPRC S-vs-D | ideal AUROC D-vs-W | shifted AUPRC S-vs-D | shifted AUROC D-vs-W | Δ AUPRC vs control (realistic) |
|---|---|---|---|---|---|---|---|
| coloc_abf | NA [NA, NA] | 0.983 [0.971, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| coloc_prop_test | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| coloc_susie | NA [NA, NA] | 0.600 [0.572, 0.800] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| coloc_susie_mm | NA [NA, NA] | 0.871 [0.686, 0.953] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| colocboost | NA [NA, NA] | 0.659 [0.646, 0.743] | NA [NA, NA] | 0.875 [0.875, 0.875] | NA [NA, NA] | 0.792 [0.792, 1.000] | NA [NA, NA] |
| colocboost_m1 | NA [NA, NA] | 0.902 [0.893, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 0.583 [0.583, 1.000] | NA [NA, NA] |
| famA_joint_susie_rho_ms_ptld | NA [NA, NA] | 0.805 [0.536, 0.847] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| famB_exactcoloc_gaussian_M50 | NA [NA, NA] | 0.974 [0.943, 0.983] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| lead_snp | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| mvsusie | NA [NA, NA] | 0.586 [0.539, 0.800] | NA [NA, NA] | 0.875 [0.875, 0.875] | NA [NA, NA] | 0.792 [0.792, 1.000] | NA [NA, NA] |
| prop_coloc | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| sharepro | NA [NA, NA] | 0.356 [0.314, 0.358] | NA [NA, NA] | 0.750 [0.750, 0.750] | NA [NA, NA] | 0.917 [0.917, 1.000] | NA [NA, NA] |
| strength_only | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | control |

### Candidate famA_joint_susie_rho_ms_ptld minus method (realistic, least_favourable)

| method | Δ AUPRC [CI] | p (cand > m) | Δ AUROC D-vs-W [CI] | p |
|---|---|---|---|---|
| coloc_abf | NA [NA, NA] | NA | -0.178 [-0.464, -0.123] | 0.9991 |
| coloc_prop_test | NA [NA, NA] | NA | NA [NA, NA] | NA |
| coloc_susie | NA [NA, NA] | NA | 0.206 [-0.264, 0.275] | 0.5063 |
| coloc_susie_mm | NA [NA, NA] | NA | -0.066 [-0.150, -0.066] | 0.9886 |
| colocboost | NA [NA, NA] | NA | 0.146 [-0.207, 0.201] | 0.4786 |
| colocboost_m1 | NA [NA, NA] | NA | -0.097 [-0.464, -0.046] | 0.9988 |
| famB_exactcoloc_gaussian_M50 | NA [NA, NA] | NA | -0.169 [-0.407, -0.136] | 0.9978 |
| lead_snp | NA [NA, NA] | NA | NA [NA, NA] | NA |
| mvsusie | NA [NA, NA] | NA | 0.219 [-0.264, 0.309] | 0.5112 |
| prop_coloc | NA [NA, NA] | NA | NA [NA, NA] | NA |
| sharepro | NA [NA, NA] | NA | 0.450 [0.221, 0.489] | 0.0000 |
| strength_only | NA [NA, NA] | NA | -0.195 [-0.464, -0.153] | 0.9990 |

## Co-primary endpoints (excluded; n = 98)

| method | realistic AUPRC S-vs-D | realistic AUROC D-vs-W | ideal AUPRC S-vs-D | ideal AUROC D-vs-W | shifted AUPRC S-vs-D | shifted AUROC D-vs-W | Δ AUPRC vs control (realistic) |
|---|---|---|---|---|---|---|---|
| coloc_abf | NA [NA, NA] | 0.981 [0.968, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| coloc_prop_test | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| coloc_susie | NA [NA, NA] | 0.613 [0.553, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| coloc_susie_mm | NA [NA, NA] | 0.958 [0.952, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| colocboost | NA [NA, NA] | 0.678 [0.660, 0.833] | NA [NA, NA] | 0.875 [0.875, 0.875] | NA [NA, NA] | 0.792 [0.792, 1.000] | NA [NA, NA] |
| colocboost_m1 | NA [NA, NA] | 0.890 [0.888, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 0.583 [0.583, 1.000] | NA [NA, NA] |
| famA_joint_susie_rho_ms_ptld | NA [NA, NA] | 0.853 [0.552, 0.889] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| famB_exactcoloc_gaussian_M50 | NA [NA, NA] | 0.971 [0.927, 0.983] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| lead_snp | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| mvsusie | NA [NA, NA] | 0.598 [0.541, 0.875] | NA [NA, NA] | 0.875 [0.875, 0.875] | NA [NA, NA] | 0.792 [0.792, 1.000] | NA [NA, NA] |
| prop_coloc | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| sharepro | NA [NA, NA] | 0.387 [0.370, 0.387] | NA [NA, NA] | 0.750 [0.750, 0.750] | NA [NA, NA] | 0.917 [0.917, 1.000] | NA [NA, NA] |
| strength_only | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | control |

### Candidate famA_joint_susie_rho_ms_ptld minus method (realistic, excluded)

| method | Δ AUPRC [CI] | p (cand > m) | Δ AUROC D-vs-W [CI] | p |
|---|---|---|---|---|
| coloc_abf | NA [NA, NA] | NA | -0.128 [-0.448, -0.079] | 0.9813 |
| coloc_prop_test | NA [NA, NA] | NA | NA [NA, NA] | NA |
| coloc_susie | NA [NA, NA] | NA | 0.240 [-0.448, 0.336] | 0.0542 |
| coloc_susie_mm | NA [NA, NA] | NA | -0.104 [-0.448, -0.063] | 0.9764 |
| colocboost | NA [NA, NA] | NA | 0.176 [-0.281, 0.229] | 0.1127 |
| colocboost_m1 | NA [NA, NA] | NA | -0.036 [-0.448, 0.001] | 0.5944 |
| famB_exactcoloc_gaussian_M50 | NA [NA, NA] | NA | -0.117 [-0.375, -0.094] | 0.9921 |
| lead_snp | NA [NA, NA] | NA | NA [NA, NA] | NA |
| mvsusie | NA [NA, NA] | NA | 0.255 [-0.323, 0.348] | 0.1842 |
| prop_coloc | NA [NA, NA] | NA | NA [NA, NA] | NA |
| sharepro | NA [NA, NA] | NA | 0.466 [0.177, 0.519] | 0.0000 |
| strength_only | NA [NA, NA] | NA | -0.147 [-0.448, -0.111] | 0.9900 |

## C3 operating points and calibration (least_favourable)

| method | t_S | t_D | W→S | W→D | recall S | recall D | ECE_S raw / recal | ECE_D raw / recal | failures | median s/locus |
|---|---|---|---|---|---|---|---|---|---|---|
| coloc_abf | 0.8575 | 0.1821 | 0.000 | 0.000 | 0.617 | 0.983 | 0.110 / 0.092 | 0.096 / 0.132 | 0 (max cell 0.000) | 0.0077 |
| coloc_prop_test | inf | inf | 0.000 | 0.000 | 0.000 | 0.000 | NA / NA | NA / NA | 0 (max cell 0.000) | 1.1508 |
| coloc_susie | 0.8905 | 0.1821 | 0.017 | 0.000 | 0.700 | 0.608 | 0.158 / 0.219 | 0.127 / 0.160 | 2 (max cell 0.333) | 3.7306 |
| coloc_susie_mm | 0.8867 | 0.1821 | 0.050 | 0.000 | 0.717 | 0.833 | 0.100 / 0.121 | 0.075 / 0.108 | 2 (max cell 0.333) | 3.5317 |
| colocboost | 0.2177 | 0.7823 | 0.050 | 0.000 | 0.583 | 0.275 | 0.132 / 0.676 | 0.197 / 0.328 | 0 (max cell 0.000) | 2.5622 |
| colocboost_m1 | 0.8216 | 0.2696 | 0.025 | 0.000 | 0.617 | 0.833 | 0.126 / 0.324 | 0.146 / 0.667 | 0 (max cell 0.000) | 0.2508 |
| famA_joint_susie_rho_ms_ptld | 0.0187 | 0.0042 | 0.000 | 0.000 | 0.908 | 0.217 | 0.170 / 0.189 | 0.198 / 0.137 | 0 (max cell 0.000) | 0.4378 |
| famB_exactcoloc_gaussian_M50 | 0.0081 | 0.1490 | 0.075 | 0.017 | 0.775 | 0.783 | 0.103 / 0.094 | 0.094 / 0.128 | 0 (max cell 0.000) | 0.0615 |
| lead_snp | inf | inf | 0.000 | 0.000 | 0.000 | 0.000 | NA / NA | NA / NA | 0 (max cell 0.000) | 0.0098 |
| mvsusie | 1.0000 | 0.0000 | 0.000 | 0.000 | 0.000 | 0.167 | 0.302 / 0.676 | 0.264 / 0.667 | 0 (max cell 0.000) | 33.5046 |
| prop_coloc | inf | inf | 0.000 | 0.000 | 0.000 | 0.000 | NA / NA | NA / NA | 10 (max cell 0.667) | 1.5591 |
| sharepro | 0.7055 | 0.2733 | 0.050 | 0.000 | 0.933 | 0.233 | 0.191 / 0.065 | 0.202 / 0.238 | 0 (max cell 0.000) | 0.3070 |
| strength_only | 0.2252 | 0.2252 | 0.033 | 0.000 | 0.950 | 0.492 | 0.159 / 0.168 | 0.176 / 0.176 | 0 (max cell 0.000) | 0.0079 |

## Per-cell AUPRC S vs D (least_favourable)

| method | L0|P1|r_p=0.0 | L0|P1|r_p=0.6 | L0|P2|rho=0.0 | L0|P2|rho=0.3 | L0|P3|rho=0.0 | L0|P3|rho=0.5 | L0|P3|rho=0.8 | L1swap|P1|r_p=0.0 | L1swap|P1|r_p=0.6 | L1|P1|r_p=0.0 | L1|P1|r_p=0.6 | L1|P2|rho=0.0 | L1|P2|rho=0.3 | L1|P3|rho=0.0 | L1|P3|rho=0.5 | L1|P3|rho=0.8 | L2swap|P1|r_p=0.0 | L2swap|P1|r_p=0.6 | L2|P1|r_p=0.0 | L2|P1|r_p=0.6 | L2|P2|rho=0.0 | L2|P2|rho=0.3 | L2|P3|rho=0.0 | L2|P3|rho=0.5 | L2|P3|rho=0.8 | L3|P1|r_p=0.0 | L3|P2|rho=0.0 | L3|P3|rho=0.0 | L4|P1|r_p=0.0 | L4|P1|r_p=0.6 | L4|P2|rho=0.0 | L4|P2|rho=0.3 | L4|P3|rho=0.0 | L4|P3|rho=0.5 | L4|P3|rho=0.8 | L6|P1|r_p=0.0 | L6|P2|rho=0.0 | L6|P3|rho=0.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coloc_abf | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| coloc_prop_test | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| coloc_susie | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| coloc_susie_mm | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| colocboost | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| colocboost_m1 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| famA_joint_susie_rho_ms_ptld | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| famB_exactcoloc_gaussian_M50 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| lead_snp | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| mvsusie | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| prop_coloc | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| sharepro | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| strength_only | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

## Per-cell AUROC D vs W (least_favourable)

| method | L0|P1|r_p=0.0 | L0|P1|r_p=0.6 | L0|P2|rho=0.0 | L0|P2|rho=0.3 | L0|P3|rho=0.0 | L0|P3|rho=0.5 | L0|P3|rho=0.8 | L1swap|P1|r_p=0.0 | L1swap|P1|r_p=0.6 | L1|P1|r_p=0.0 | L1|P1|r_p=0.6 | L1|P2|rho=0.0 | L1|P2|rho=0.3 | L1|P3|rho=0.0 | L1|P3|rho=0.5 | L1|P3|rho=0.8 | L2swap|P1|r_p=0.0 | L2swap|P1|r_p=0.6 | L2|P1|r_p=0.0 | L2|P1|r_p=0.6 | L2|P2|rho=0.0 | L2|P2|rho=0.3 | L2|P3|rho=0.0 | L2|P3|rho=0.5 | L2|P3|rho=0.8 | L3|P1|r_p=0.0 | L3|P2|rho=0.0 | L3|P3|rho=0.0 | L4|P1|r_p=0.0 | L4|P1|r_p=0.6 | L4|P2|rho=0.0 | L4|P2|rho=0.3 | L4|P3|rho=0.0 | L4|P3|rho=0.5 | L4|P3|rho=0.8 | L6|P1|r_p=0.0 | L6|P2|rho=0.0 | L6|P3|rho=0.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coloc_abf | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| coloc_prop_test | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| coloc_susie | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| coloc_susie_mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.500 | 1.000 |
| colocboost | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 0.500 | 0.000 | 0.500 | 0.000 | 0.500 | 0.500 | 0.500 | 0.500 | 1.000 | 1.000 | 0.500 | 0.500 | 0.500 | 1.000 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 1.000 |
| colocboost_m1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 1.000 | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | NA | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| famA_joint_susie_rho_ms_ptld | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.500 | 1.000 | 0.000 | 1.000 |
| famB_exactcoloc_gaussian_M50 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 |
| lead_snp | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| mvsusie | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 1.000 | 0.500 | 1.000 | 0.000 | 1.000 | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 0.500 | 0.500 | 0.500 | 0.500 | 1.000 | 0.500 | 0.500 | 0.500 | 0.500 | 1.000 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 1.000 |
| prop_coloc | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| sharepro | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| strength_only | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## D strata by r2 (least_favourable; AUPRC S vs D_b / false-S rate at t_S)

| method | [0,0.1) | [0.1,0.3) | [0.3,0.5) | [0.5,0.8) | [0.8,0.95) | out |
|---|---|---|---|---|---|---|
| coloc_abf | NA / 0.000 (n=8) | NA / 0.000 (n=8) | NA / 0.071 (n=14) | NA / 0.000 (n=7) | NA / NA (n=0) | NA / NA (n=0) |
| coloc_prop_test | NA / 0.000 (n=8) | NA / 0.000 (n=8) | NA / 0.000 (n=14) | NA / 0.000 (n=7) | NA / NA (n=0) | NA / NA (n=0) |
| coloc_susie | NA / 0.000 (n=8) | NA / 0.250 (n=8) | NA / 0.357 (n=14) | NA / 0.000 (n=7) | NA / NA (n=0) | NA / NA (n=0) |
| coloc_susie_mm | NA / 0.000 (n=8) | NA / 0.125 (n=8) | NA / 0.143 (n=14) | NA / 0.000 (n=7) | NA / NA (n=0) | NA / NA (n=0) |
| colocboost | NA / 0.000 (n=8) | NA / 0.000 (n=8) | NA / 0.000 (n=14) | NA / 0.429 (n=7) | NA / NA (n=0) | NA / NA (n=0) |
| colocboost_m1 | NA / 0.000 (n=8) | NA / 0.000 (n=8) | NA / 0.071 (n=14) | NA / 0.143 (n=7) | NA / NA (n=0) | NA / NA (n=0) |
| famA_joint_susie_rho_ms_ptld | NA / 0.250 (n=8) | NA / 0.625 (n=8) | NA / 0.643 (n=14) | NA / 0.429 (n=7) | NA / NA (n=0) | NA / NA (n=0) |
| famB_exactcoloc_gaussian_M50 | NA / 0.125 (n=8) | NA / 0.000 (n=8) | NA / 0.071 (n=14) | NA / 0.143 (n=7) | NA / NA (n=0) | NA / NA (n=0) |
| lead_snp | NA / 0.000 (n=8) | NA / 0.000 (n=8) | NA / 0.000 (n=14) | NA / 0.000 (n=7) | NA / NA (n=0) | NA / NA (n=0) |
| mvsusie | NA / 0.000 (n=8) | NA / 0.000 (n=8) | NA / 0.000 (n=14) | NA / 0.000 (n=7) | NA / NA (n=0) | NA / NA (n=0) |
| prop_coloc | NA / 0.000 (n=8) | NA / 0.000 (n=8) | NA / 0.000 (n=14) | NA / 0.000 (n=7) | NA / NA (n=0) | NA / NA (n=0) |
| sharepro | NA / 0.000 (n=8) | NA / 0.375 (n=8) | NA / 0.643 (n=14) | NA / 0.857 (n=7) | NA / NA (n=0) | NA / NA (n=0) |
| strength_only | NA / 0.750 (n=8) | NA / 0.250 (n=8) | NA / 0.571 (n=14) | NA / 0.429 (n=7) | NA / NA (n=0) | NA / NA (n=0) |

## D strata by d (least_favourable; AUPRC S vs D_b / false-S rate at t_S)

| method | [2,4) | [4,8) | [8,16) | [16,inf) | out |
|---|---|---|---|---|---|
| coloc_abf | NA / 0.000 (n=1) | NA / 0.000 (n=23) | NA / 0.091 (n=11) | NA / 0.000 (n=2) | NA / NA (n=0) |
| coloc_prop_test | NA / 0.000 (n=1) | NA / 0.000 (n=23) | NA / 0.000 (n=11) | NA / 0.000 (n=2) | NA / NA (n=0) |
| coloc_susie | NA / 0.000 (n=1) | NA / 0.217 (n=23) | NA / 0.182 (n=11) | NA / 0.000 (n=2) | NA / NA (n=0) |
| coloc_susie_mm | NA / 0.000 (n=1) | NA / 0.087 (n=23) | NA / 0.091 (n=11) | NA / 0.000 (n=2) | NA / NA (n=0) |
| colocboost | NA / 1.000 (n=1) | NA / 0.043 (n=23) | NA / 0.091 (n=11) | NA / 0.000 (n=2) | NA / NA (n=0) |
| colocboost_m1 | NA / 1.000 (n=1) | NA / 0.000 (n=23) | NA / 0.091 (n=11) | NA / 0.000 (n=2) | NA / NA (n=0) |
| famA_joint_susie_rho_ms_ptld | NA / 0.000 (n=1) | NA / 0.565 (n=23) | NA / 0.455 (n=11) | NA / 0.500 (n=2) | NA / NA (n=0) |
| famB_exactcoloc_gaussian_M50 | NA / 0.000 (n=1) | NA / 0.087 (n=23) | NA / 0.091 (n=11) | NA / 0.000 (n=2) | NA / NA (n=0) |
| lead_snp | NA / 0.000 (n=1) | NA / 0.000 (n=23) | NA / 0.000 (n=11) | NA / 0.000 (n=2) | NA / NA (n=0) |
| mvsusie | NA / 0.000 (n=1) | NA / 0.000 (n=23) | NA / 0.000 (n=11) | NA / 0.000 (n=2) | NA / NA (n=0) |
| prop_coloc | NA / 0.000 (n=1) | NA / 0.000 (n=23) | NA / 0.000 (n=11) | NA / 0.000 (n=2) | NA / NA (n=0) |
| sharepro | NA / 1.000 (n=1) | NA / 0.478 (n=23) | NA / 0.364 (n=11) | NA / 1.000 (n=2) | NA / NA (n=0) |
| strength_only | NA / 0.000 (n=1) | NA / 0.348 (n=23) | NA / 0.818 (n=11) | NA / 1.000 (n=2) | NA / NA (n=0) |

## C3 dry run on this set (least_favourable; thresholds given)

- candidate famA_joint_susie_rho_ms_ptld; eligible ['coloc_abf', 'coloc_susie', 'coloc_susie_mm', 'sharepro', 'colocboost', 'colocboost_m1', 'mvsusie']; non-inferior to all: False; superior on one (Holm): False; **pass: False**

## C3 dry run on this set (excluded; thresholds given)

- candidate famA_joint_susie_rho_ms_ptld; eligible ['coloc_abf', 'coloc_susie', 'coloc_susie_mm', 'sharepro', 'colocboost', 'colocboost_m1', 'mvsusie']; non-inferior to all: False; superior on one (Holm): False; **pass: False**

