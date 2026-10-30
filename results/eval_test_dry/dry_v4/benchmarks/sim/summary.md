# Benchmark dry_v4__sim

- loci: 566 (D 140, G 79, S 139, U 26, W 182); cells: 38; per-cell subsample: None
- bootstrap: 2000 replicates, resampling unit block (BCa 95%); seed [20261029, 'run_benchmark', 'bootstrap', 'dry_v4__sim']
- strength-only control (D6(e)): table DRY in-sample placeholder
- operating points: given /cluster/scratch/mumarc/hackathon_eqtl_gwas/results/eval_test_dry/dry_v4/DRY_operating_points_in_sample.json
- git df33fb5346 clean=True

## Co-primary endpoints (least_favourable; n = 566)

| method | realistic AUPRC S-vs-D | realistic AUROC D-vs-W | ideal AUPRC S-vs-D | ideal AUROC D-vs-W | shifted AUPRC S-vs-D | shifted AUROC D-vs-W | Δ AUPRC vs control (realistic) |
|---|---|---|---|---|---|---|---|
| coloc_abf | 1.000 [0.930, 1.000] | 0.993 [0.991, 1.000] | NA [NA, NA] | 0.938 [0.250, 1.000] | NA [NA, NA] | 0.970 [0.949, 1.000] | 0.073 [0.000, 0.357] |
| coloc_prop_test | 0.894 [0.824, 1.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | -0.033 [-0.167, 0.309] |
| coloc_susie | 0.866 [0.580, 1.000] | 0.679 [0.612, 0.970] | NA [NA, NA] | 0.891 [0.000, 1.000] | NA [NA, NA] | 0.978 [0.949, 1.000] | -0.061 [-0.378, 0.145] |
| coloc_susie_mm | 1.000 [0.906, 1.000] | 0.885 [0.557, 0.939] | NA [NA, NA] | 0.875 [0.000, 1.000] | NA [NA, NA] | 0.978 [0.949, 1.000] | 0.073 [-0.015, 0.309] |
| famA_joint_susie_rho_ms_ptld | 0.762 [0.591, 1.000] | 0.817 [0.798, 0.866] | NA [NA, NA] | 0.918 [0.375, 1.000] | NA [NA, NA] | 0.992 [0.986, 1.000] | -0.165 [-0.343, 0.309] |
| famB_exactcoloc_gaussian_M50 | 0.973 [0.824, 1.000] | 0.975 [0.961, 0.982] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 0.978 [0.974, 0.986] | 0.046 [0.000, 0.309] |
| lead_snp | 1.000 [0.930, 1.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | 0.073 [0.000, 0.357] |
| prop_coloc | 0.711 [0.568, 1.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | -0.216 [-0.500, 0.167] |
| sharepro | 0.638 [0.500, 0.667] | 0.353 [0.340, 0.357] | NA [NA, NA] | 0.500 [0.000, 0.583] | NA [NA, NA] | 0.972 [0.969, 0.986] | -0.289 [-0.500, 0.089] |
| strength_only | 0.927 [0.691, 1.000] | 0.992 [0.962, 0.999] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 0.985 [0.981, 0.993] | control |

### Candidate famA_joint_susie_rho_ms_ptld minus method (realistic, least_favourable)

| method | Δ AUPRC [CI] | p (cand > m) | Δ AUROC D-vs-W [CI] | p |
|---|---|---|---|---|
| coloc_abf | -0.238 [-0.409, 0.000] | 0.8964 | -0.176 [-0.193, -0.131] | 1.0000 |
| coloc_prop_test | -0.132 [-0.288, 0.233] | 0.7167 | NA [NA, NA] | NA |
| coloc_susie | -0.104 [-0.104, -0.068] | 0.9967 | 0.138 [-0.103, 0.186] | 0.2830 |
| coloc_susie_mm | -0.238 [-0.358, 0.000] | 0.9902 | -0.068 [-0.126, 0.263] | 0.7473 |
| famB_exactcoloc_gaussian_M50 | -0.211 [-0.375, 0.040] | 0.8638 | -0.158 [-0.174, -0.094] | 1.0000 |
| lead_snp | -0.238 [-0.409, 0.000] | 0.8964 | NA [NA, NA] | NA |
| prop_coloc | 0.051 [-0.100, 0.500] | 0.2100 | NA [NA, NA] | NA |
| sharepro | 0.124 [-0.046, 0.500] | 0.0858 | 0.464 [0.453, 0.510] | 0.0128 |
| strength_only | -0.165 [-0.343, 0.309] | 0.7954 | -0.175 [-0.200, -0.100] | 1.0000 |

## Co-primary endpoints (excluded; n = 513)

| method | realistic AUPRC S-vs-D | realistic AUROC D-vs-W | ideal AUPRC S-vs-D | ideal AUROC D-vs-W | shifted AUPRC S-vs-D | shifted AUROC D-vs-W | Δ AUPRC vs control (realistic) |
|---|---|---|---|---|---|---|---|
| coloc_abf | 1.000 [0.891, 1.000] | 0.992 [0.990, 1.000] | NA [NA, NA] | 0.938 [0.250, 1.000] | NA [NA, NA] | 0.969 [0.946, 1.000] | 0.114 [0.000, 0.500] |
| coloc_prop_test | 1.000 [0.800, 1.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | 0.114 [-0.052, 0.200] |
| coloc_susie | 1.000 [0.669, 1.000] | 0.632 [0.577, 1.000] | NA [NA, NA] | 0.891 [0.000, 1.000] | NA [NA, NA] | 0.977 [0.946, 1.000] | 0.114 [-0.233, 0.371] |
| coloc_susie_mm | 1.000 [0.917, 1.000] | 0.925 [0.921, 0.929] | NA [NA, NA] | 0.875 [0.000, 1.000] | NA [NA, NA] | 0.977 [0.946, 1.000] | 0.114 [0.000, 0.500] |
| famA_joint_susie_rho_ms_ptld | 1.000 [0.701, 1.000] | 0.830 [0.810, 0.866] | NA [NA, NA] | 0.918 [0.375, 1.000] | NA [NA, NA] | 0.992 [0.983, 1.000] | 0.114 [-0.201, 0.500] |
| famB_exactcoloc_gaussian_M50 | 1.000 [0.727, 1.000] | 0.972 [0.955, 0.983] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 0.976 [0.973, 0.983] | 0.114 [0.000, 0.500] |
| lead_snp | 1.000 [0.833, 1.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | 0.114 [0.000, 0.500] |
| prop_coloc | 1.000 [0.850, 1.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | 0.114 [-0.052, 0.500] |
| sharepro | 0.625 [0.500, 0.700] | 0.380 [0.368, 0.398] | NA [NA, NA] | 0.500 [0.000, 0.583] | NA [NA, NA] | 0.969 [0.968, 0.983] | -0.261 [-0.375, 0.200] |
| strength_only | 0.886 [0.500, 1.000] | 0.990 [0.983, 0.999] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 0.984 [0.980, 0.992] | control |

### Candidate famA_joint_susie_rho_ms_ptld minus method (realistic, excluded)

| method | Δ AUPRC [CI] | p (cand > m) | Δ AUROC D-vs-W [CI] | p |
|---|---|---|---|---|
| coloc_abf | 0.000 [-0.299, 0.000] | 0.3756 | -0.162 [-0.189, -0.134] | 1.0000 |
| coloc_prop_test | 0.000 [-0.149, 0.373] | 0.4555 | NA [NA, NA] | NA |
| coloc_susie | 0.000 [-0.045, 0.129] | 0.6931 | 0.198 [-0.189, 0.235] | 0.3187 |
| coloc_susie_mm | 0.000 [-0.299, 0.000] | 0.3756 | -0.095 [-0.113, -0.086] | 0.9166 |
| famB_exactcoloc_gaussian_M50 | 0.000 [-0.299, 0.010] | 0.3827 | -0.142 [-0.172, -0.088] | 1.0000 |
| lead_snp | 0.000 [-0.299, 0.000] | 0.3756 | NA [NA, NA] | NA |
| prop_coloc | 0.000 [-0.149, 0.064] | 0.3782 | NA [NA, NA] | NA |
| sharepro | 0.375 [0.300, 0.500] | 0.0005 | 0.451 [0.419, 0.480] | 0.0002 |
| strength_only | 0.114 [-0.201, 0.500] | 0.1133 | -0.160 [-0.187, -0.118] | 0.9999 |

## C3 operating points and calibration (least_favourable)

| method | t_S | t_D | W→S | W→D | recall S | recall D | ECE_S raw / recal | ECE_D raw / recal | failures | median s/locus |
|---|---|---|---|---|---|---|---|---|---|---|
| coloc_abf | 0.8575 | 0.1821 | 0.023 | 0.023 | 0.648 | 0.979 | 0.157 / 0.129 | 0.145 / 0.132 | 0 (max cell 0.000) | 0.0076 |
| coloc_prop_test | inf | inf | 0.000 | 0.000 | 0.000 | 0.000 | NA / NA | NA / NA | 0 (max cell 0.000) | 1.3709 |
| coloc_susie | 0.8905 | 0.1821 | 0.021 | 0.023 | 0.719 | 0.673 | 0.202 / 0.158 | 0.156 / 0.134 | 2 (max cell 0.067) | 3.3801 |
| coloc_susie_mm | 0.8867 | 0.1821 | 0.023 | 0.023 | 0.740 | 0.867 | 0.152 / 0.093 | 0.106 / 0.098 | 6 (max cell 0.125) | 6.5077 |
| famA_joint_susie_rho_ms_ptld | 0.0187 | 0.0042 | 0.021 | 0.013 | 0.902 | 0.179 | 0.117 / 0.109 | 0.126 / 0.157 | 0 (max cell 0.000) | 0.4607 |
| famB_exactcoloc_gaussian_M50 | 0.0081 | 0.1490 | 0.023 | 0.017 | 0.800 | 0.702 | 0.073 / 0.084 | 0.092 / 0.102 | 0 (max cell 0.000) | 0.0615 |
| lead_snp | inf | inf | 0.000 | 0.000 | 0.000 | 0.000 | NA / NA | NA / NA | 0 (max cell 0.000) | 0.0089 |
| prop_coloc | inf | inf | 0.000 | 0.000 | 0.000 | 0.000 | NA / NA | NA / NA | 46 (max cell 0.500) | 1.2685 |
| sharepro | 0.7055 | 0.2733 | 0.023 | 0.013 | 0.927 | 0.219 | 0.240 / 0.052 | 0.158 / 0.247 | 0 (max cell 0.000) | 0.3502 |
| strength_only | 0.2252 | 0.2252 | 0.025 | 0.013 | 0.958 | 0.463 | 0.135 / 0.132 | 0.129 / 0.129 | 0 (max cell 0.000) | 0.0086 |

## Per-cell AUPRC S vs D (least_favourable)

| method | L0|P1|r_p=0.0 | L0|P1|r_p=0.6 | L0|P2|rho=0.0 | L0|P2|rho=0.3 | L0|P3|rho=0.0 | L0|P3|rho=0.5 | L0|P3|rho=0.8 | L1swap|P1|r_p=0.0 | L1swap|P1|r_p=0.6 | L1|P1|r_p=0.0 | L1|P1|r_p=0.6 | L1|P2|rho=0.0 | L1|P2|rho=0.3 | L1|P3|rho=0.0 | L1|P3|rho=0.5 | L1|P3|rho=0.8 | L2swap|P1|r_p=0.0 | L2swap|P1|r_p=0.6 | L2|P1|r_p=0.0 | L2|P1|r_p=0.6 | L2|P2|rho=0.0 | L2|P2|rho=0.3 | L2|P3|rho=0.0 | L2|P3|rho=0.5 | L2|P3|rho=0.8 | L3|P1|r_p=0.0 | L3|P2|rho=0.0 | L3|P3|rho=0.0 | L4|P1|r_p=0.0 | L4|P1|r_p=0.6 | L4|P2|rho=0.0 | L4|P2|rho=0.3 | L4|P3|rho=0.0 | L4|P3|rho=0.5 | L4|P3|rho=0.8 | L6|P1|r_p=0.0 | L6|P2|rho=0.0 | L6|P3|rho=0.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coloc_abf | NA | 1.000 | NA | NA | NA | NA | 0.500 | NA | NA | NA | NA | NA | 1.000 | 1.000 | NA | NA | NA | NA | NA | NA | 1.000 | NA | NA | NA | 1.000 | 1.000 | NA | 1.000 | NA | 1.000 | 1.000 | NA | NA | NA | NA | NA | NA | NA |
| coloc_prop_test | NA | 1.000 | NA | NA | NA | NA | 0.500 | NA | NA | NA | NA | NA | 1.000 | 1.000 | NA | NA | NA | NA | NA | NA | 1.000 | NA | NA | NA | 0.500 | 1.000 | NA | 1.000 | NA | 1.000 | 1.000 | NA | NA | NA | NA | NA | NA | NA |
| coloc_susie | NA | 1.000 | NA | NA | NA | NA | 1.000 | NA | NA | NA | NA | NA | 1.000 | 1.000 | NA | NA | NA | NA | NA | NA | 0.500 | NA | NA | NA | 1.000 | 1.000 | NA | 1.000 | NA | 1.000 | 1.000 | NA | NA | NA | NA | NA | NA | NA |
| coloc_susie_mm | NA | 1.000 | NA | NA | NA | NA | 1.000 | NA | NA | NA | NA | NA | 1.000 | 1.000 | NA | NA | NA | NA | NA | NA | 1.000 | NA | NA | NA | 1.000 | 1.000 | NA | 1.000 | NA | 1.000 | 1.000 | NA | NA | NA | NA | NA | NA | NA |
| famA_joint_susie_rho_ms_ptld | NA | 1.000 | NA | NA | NA | NA | 1.000 | NA | NA | NA | NA | NA | 1.000 | 1.000 | NA | NA | NA | NA | NA | NA | 0.500 | NA | NA | NA | 1.000 | 1.000 | NA | 1.000 | NA | 1.000 | 1.000 | NA | NA | NA | NA | NA | NA | NA |
| famB_exactcoloc_gaussian_M50 | NA | 1.000 | NA | NA | NA | NA | 1.000 | NA | NA | NA | NA | NA | 1.000 | 1.000 | NA | NA | NA | NA | NA | NA | 1.000 | NA | NA | NA | 0.500 | 1.000 | NA | 1.000 | NA | 1.000 | 1.000 | NA | NA | NA | NA | NA | NA | NA |
| lead_snp | NA | 1.000 | NA | NA | NA | NA | 0.500 | NA | NA | NA | NA | NA | 1.000 | 1.000 | NA | NA | NA | NA | NA | NA | 1.000 | NA | NA | NA | 1.000 | 1.000 | NA | 1.000 | NA | 1.000 | 1.000 | NA | NA | NA | NA | NA | NA | NA |
| prop_coloc | NA | 1.000 | NA | NA | NA | NA | 0.500 | NA | NA | NA | NA | NA | 1.000 | 1.000 | NA | NA | NA | NA | NA | NA | 0.500 | NA | NA | NA | 1.000 | 1.000 | NA | 1.000 | NA | 1.000 | 0.500 | NA | NA | NA | NA | NA | NA | NA |
| sharepro | NA | 1.000 | NA | NA | NA | NA | 1.000 | NA | NA | NA | NA | NA | 0.500 | 1.000 | NA | NA | NA | NA | NA | NA | 0.500 | NA | NA | NA | 1.000 | 0.500 | NA | 1.000 | NA | 1.000 | 0.500 | NA | NA | NA | NA | NA | NA | NA |
| strength_only | NA | 0.500 | NA | NA | NA | NA | 1.000 | NA | NA | NA | NA | NA | 1.000 | 0.500 | NA | NA | NA | NA | NA | NA | 1.000 | NA | NA | NA | 1.000 | 1.000 | NA | 1.000 | NA | 0.500 | 1.000 | NA | NA | NA | NA | NA | NA | NA |

## Per-cell AUROC D vs W (least_favourable)

| method | L0|P1|r_p=0.0 | L0|P1|r_p=0.6 | L0|P2|rho=0.0 | L0|P2|rho=0.3 | L0|P3|rho=0.0 | L0|P3|rho=0.5 | L0|P3|rho=0.8 | L1swap|P1|r_p=0.0 | L1swap|P1|r_p=0.6 | L1|P1|r_p=0.0 | L1|P1|r_p=0.6 | L1|P2|rho=0.0 | L1|P2|rho=0.3 | L1|P3|rho=0.0 | L1|P3|rho=0.5 | L1|P3|rho=0.8 | L2swap|P1|r_p=0.0 | L2swap|P1|r_p=0.6 | L2|P1|r_p=0.0 | L2|P1|r_p=0.6 | L2|P2|rho=0.0 | L2|P2|rho=0.3 | L2|P3|rho=0.0 | L2|P3|rho=0.5 | L2|P3|rho=0.8 | L3|P1|r_p=0.0 | L3|P2|rho=0.0 | L3|P3|rho=0.0 | L4|P1|r_p=0.0 | L4|P1|r_p=0.6 | L4|P2|rho=0.0 | L4|P2|rho=0.3 | L4|P3|rho=0.0 | L4|P3|rho=0.5 | L4|P3|rho=0.8 | L6|P1|r_p=0.0 | L6|P2|rho=0.0 | L6|P3|rho=0.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coloc_abf | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 0.938 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 |
| coloc_prop_test | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| coloc_susie | 0.812 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 | 1.000 | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 0.500 | 0.750 | 0.500 | 0.250 | 1.000 | 0.750 | 0.750 | 0.750 | 0.500 | 0.750 | 0.250 | 0.781 | 1.000 | 0.500 | 0.781 | 0.750 | 0.750 | 0.500 | 0.219 | 0.750 |
| coloc_susie_mm | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 | 1.000 | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 | 0.688 | 0.781 | 0.938 |
| famA_joint_susie_rho_ms_ptld | 0.844 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.938 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 1.000 | 0.750 | 0.344 | 0.375 | 0.938 | 0.688 | 0.938 | 0.375 | 0.938 | 0.844 | 0.250 | 0.875 | 0.938 | 0.312 | 0.875 | 0.812 | 0.719 | 1.000 | 0.562 | 1.000 |
| famB_exactcoloc_gaussian_M50 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.844 | 1.000 | 1.000 | 1.000 | 0.812 | 1.000 | 0.750 | 0.750 | 1.000 | 1.000 | 1.000 |
| lead_snp | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| prop_coloc | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| sharepro | 0.500 | 0.750 | 0.250 | 1.000 | 0.750 | 0.812 | 0.750 | 1.000 | 1.000 | 0.250 | 0.750 | 0.750 | 0.500 | 0.750 | 0.500 | 0.500 | 1.000 | NA | 0.250 | 0.312 | 0.250 | 0.375 | 0.625 | 0.250 | 0.562 | 0.125 | 0.312 | 0.750 | 0.000 | 0.625 | 0.375 | 0.188 | 0.625 | 0.188 | 0.375 | 0.188 | 0.000 | 0.188 |
| strength_only | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.938 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.938 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 1.000 | 1.000 | 1.000 | 0.844 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 |

## D strata by r2 (least_favourable; AUPRC S vs D_b / false-S rate at t_S)

| method | [0,0.1) | [0.1,0.3) | [0.3,0.5) | [0.5,0.8) | [0.8,0.95) | out |
|---|---|---|---|---|---|---|
| coloc_abf | NA / 0.000 (n=37) | NA / 0.000 (n=34) | NA / 0.029 (n=35) | NA / 0.029 (n=34) | NA / NA (n=0) | NA / NA (n=0) |
| coloc_prop_test | NA / 0.000 (n=37) | NA / 0.000 (n=34) | NA / 0.000 (n=35) | NA / 0.000 (n=34) | NA / NA (n=0) | NA / NA (n=0) |
| coloc_susie | NA / 0.081 (n=37) | NA / 0.324 (n=34) | NA / 0.257 (n=35) | NA / 0.235 (n=34) | NA / NA (n=0) | NA / NA (n=0) |
| coloc_susie_mm | NA / 0.027 (n=37) | NA / 0.088 (n=34) | NA / 0.114 (n=35) | NA / 0.088 (n=34) | NA / NA (n=0) | NA / NA (n=0) |
| famA_joint_susie_rho_ms_ptld | NA / 0.108 (n=37) | NA / 0.441 (n=34) | NA / 0.514 (n=35) | NA / 0.588 (n=34) | NA / NA (n=0) | NA / NA (n=0) |
| famB_exactcoloc_gaussian_M50 | NA / 0.027 (n=37) | NA / 0.029 (n=34) | NA / 0.143 (n=35) | NA / 0.235 (n=34) | NA / NA (n=0) | NA / NA (n=0) |
| lead_snp | NA / 0.000 (n=37) | NA / 0.000 (n=34) | NA / 0.000 (n=35) | NA / 0.000 (n=34) | NA / NA (n=0) | NA / NA (n=0) |
| prop_coloc | NA / 0.000 (n=37) | NA / 0.000 (n=34) | NA / 0.000 (n=35) | NA / 0.000 (n=34) | NA / NA (n=0) | NA / NA (n=0) |
| sharepro | NA / 0.135 (n=37) | NA / 0.353 (n=34) | NA / 0.657 (n=35) | NA / 0.912 (n=34) | NA / NA (n=0) | NA / NA (n=0) |
| strength_only | NA / 0.622 (n=37) | NA / 0.471 (n=34) | NA / 0.629 (n=35) | NA / 0.529 (n=34) | NA / NA (n=0) | NA / NA (n=0) |

## D strata by d (least_favourable; AUPRC S vs D_b / false-S rate at t_S)

| method | [2,4) | [4,8) | [8,16) | [16,inf) | out |
|---|---|---|---|---|---|
| coloc_abf | NA / 0.167 (n=6) | 0.867 / 0.000 (n=83) | NA / 0.024 (n=41) | NA / 0.000 (n=10) | NA / NA (n=0) |
| coloc_prop_test | NA / 0.000 (n=6) | 0.867 / 0.000 (n=83) | NA / 0.000 (n=41) | NA / 0.000 (n=10) | NA / NA (n=0) |
| coloc_susie | NA / 0.333 (n=6) | 0.639 / 0.229 (n=83) | NA / 0.195 (n=41) | NA / 0.200 (n=10) | NA / NA (n=0) |
| coloc_susie_mm | NA / 0.167 (n=6) | 1.000 / 0.084 (n=83) | NA / 0.073 (n=41) | NA / 0.000 (n=10) | NA / NA (n=0) |
| famA_joint_susie_rho_ms_ptld | NA / 0.667 (n=6) | 0.589 / 0.446 (n=83) | NA / 0.268 (n=41) | NA / 0.500 (n=10) | NA / NA (n=0) |
| famB_exactcoloc_gaussian_M50 | NA / 0.500 (n=6) | 1.000 / 0.096 (n=83) | NA / 0.073 (n=41) | NA / 0.100 (n=10) | NA / NA (n=0) |
| lead_snp | NA / 0.000 (n=6) | 0.833 / 0.000 (n=83) | NA / 0.000 (n=41) | NA / 0.000 (n=10) | NA / NA (n=0) |
| prop_coloc | NA / 0.000 (n=6) | 0.667 / 0.000 (n=83) | NA / 0.000 (n=41) | NA / 0.000 (n=10) | NA / NA (n=0) |
| sharepro | NA / 1.000 (n=6) | 0.600 / 0.470 (n=83) | NA / 0.439 (n=41) | NA / 0.800 (n=10) | NA / NA (n=0) |
| strength_only | NA / 0.333 (n=6) | 0.806 / 0.446 (n=83) | NA / 0.756 (n=41) | NA / 0.900 (n=10) | NA / NA (n=0) |

## C3 dry run on this set (least_favourable; thresholds given)

- candidate famA_joint_susie_rho_ms_ptld; eligible ['coloc_abf', 'coloc_susie', 'coloc_susie_mm', 'sharepro']; non-inferior to all: False; superior on one (Holm): False; **pass: False**

## C3 dry run on this set (excluded; thresholds given)

- candidate famA_joint_susie_rho_ms_ptld; eligible ['coloc_abf', 'coloc_susie', 'coloc_susie_mm', 'sharepro']; non-inferior to all: False; superior on one (Holm): False; **pass: False**

