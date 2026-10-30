# Benchmark dry_v4__sim__subsample_9a475bf8

- loci: 219 (D 72, S 71, W 76); cells: 38; per-cell subsample: None
- bootstrap: 2000 replicates, resampling unit block (BCa 95%); seed [20261029, 'run_benchmark', 'bootstrap', 'dry_v4__sim']
- strength-only control (D6(e)): table DRY in-sample placeholder
- operating points: given /cluster/scratch/mumarc/hackathon_eqtl_gwas/results/eval_test_dry/dry_v4/DRY_operating_points_in_sample.json
- git df33fb5346 clean=True

## Co-primary endpoints (least_favourable; n = 219)

| method | realistic AUPRC S-vs-D | realistic AUROC D-vs-W | ideal AUPRC S-vs-D | ideal AUROC D-vs-W | shifted AUPRC S-vs-D | shifted AUROC D-vs-W | Δ AUPRC vs control (realistic) |
|---|---|---|---|---|---|---|---|
| coloc_abf | NA [NA, NA] | 0.990 [0.980, 1.000] | NA [NA, NA] | 0.875 [0.800, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| coloc_prop_test | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| coloc_susie | NA [NA, NA] | 0.592 [0.541, 0.833] | NA [NA, NA] | 0.750 [0.600, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| coloc_susie_mm | NA [NA, NA] | 0.910 [0.769, 0.940] | NA [NA, NA] | 0.750 [0.600, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| colocboost | NA [NA, NA] | 0.670 [0.656, 0.678] | NA [NA, NA] | 0.875 [0.875, 0.875] | NA [NA, NA] | 0.812 [0.800, 0.833] | NA [NA, NA] |
| colocboost_m1 | NA [NA, NA] | 0.923 [0.886, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 0.625 [0.600, 0.667] | NA [NA, NA] |
| famA_joint_susie_rho_ms_ptld | NA [NA, NA] | 0.837 [0.835, 0.840] | NA [NA, NA] | 0.812 [0.700, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| famB_exactcoloc_gaussian_M50 | NA [NA, NA] | 0.980 [0.917, 0.985] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| lead_snp | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| prop_coloc | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| sharepro | NA [NA, NA] | 0.371 [0.368, 0.399] | NA [NA, NA] | 0.500 [0.350, 0.750] | NA [NA, NA] | 0.958 [0.933, 1.000] | NA [NA, NA] |
| strength_only | NA [NA, NA] | 0.996 [0.995, 0.999] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | control |

### Candidate famA_joint_susie_rho_ms_ptld minus method (realistic, least_favourable)

| method | Δ AUPRC [CI] | p (cand > m) | Δ AUROC D-vs-W [CI] | p |
|---|---|---|---|---|
| coloc_abf | NA [NA, NA] | NA | -0.153 [-0.304, -0.147] | 0.9979 |
| coloc_prop_test | NA [NA, NA] | NA | NA [NA, NA] | NA |
| coloc_susie | NA [NA, NA] | NA | 0.245 [-0.123, 0.292] | 0.2731 |
| coloc_susie_mm | NA [NA, NA] | NA | -0.073 [-0.100, -0.054] | 0.9938 |
| colocboost | NA [NA, NA] | NA | 0.168 [0.052, 0.181] | 0.0000 |
| colocboost_m1 | NA [NA, NA] | NA | -0.086 [-0.304, -0.053] | 0.9944 |
| famB_exactcoloc_gaussian_M50 | NA [NA, NA] | NA | -0.142 [-0.149, -0.138] | 0.8419 |
| lead_snp | NA [NA, NA] | NA | NA [NA, NA] | NA |
| prop_coloc | NA [NA, NA] | NA | NA [NA, NA] | NA |
| sharepro | NA [NA, NA] | NA | 0.466 [0.356, 0.470] | 0.0000 |
| strength_only | NA [NA, NA] | NA | -0.159 [-0.163, -0.156] | 0.8012 |

## Co-primary endpoints (excluded; n = 198)

| method | realistic AUPRC S-vs-D | realistic AUROC D-vs-W | ideal AUPRC S-vs-D | ideal AUROC D-vs-W | shifted AUPRC S-vs-D | shifted AUROC D-vs-W | Δ AUPRC vs control (realistic) |
|---|---|---|---|---|---|---|---|
| coloc_abf | NA [NA, NA] | 0.989 [0.980, 1.000] | NA [NA, NA] | 0.875 [0.800, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| coloc_prop_test | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| coloc_susie | NA [NA, NA] | 0.591 [0.546, 1.000] | NA [NA, NA] | 0.750 [0.600, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| coloc_susie_mm | NA [NA, NA] | 0.925 [0.915, 1.000] | NA [NA, NA] | 0.750 [0.600, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| colocboost | NA [NA, NA] | 0.629 [0.629, 0.630] | NA [NA, NA] | 0.875 [0.875, 0.875] | NA [NA, NA] | 0.812 [0.800, 0.833] | NA [NA, NA] |
| colocboost_m1 | NA [NA, NA] | 0.911 [0.887, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 0.625 [0.600, 0.667] | NA [NA, NA] |
| famA_joint_susie_rho_ms_ptld | NA [NA, NA] | 0.846 [0.758, 0.852] | NA [NA, NA] | 0.812 [0.700, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| famB_exactcoloc_gaussian_M50 | NA [NA, NA] | 0.980 [0.903, 0.986] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] |
| lead_snp | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| prop_coloc | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] |
| sharepro | NA [NA, NA] | 0.388 [0.370, 0.492] | NA [NA, NA] | 0.500 [0.350, 0.750] | NA [NA, NA] | 0.958 [0.933, 1.000] | NA [NA, NA] |
| strength_only | NA [NA, NA] | 0.996 [0.995, 0.998] | NA [NA, NA] | 1.000 [1.000, 1.000] | NA [NA, NA] | 1.000 [1.000, 1.000] | control |

### Candidate famA_joint_susie_rho_ms_ptld minus method (realistic, excluded)

| method | Δ AUPRC [CI] | p (cand > m) | Δ AUROC D-vs-W [CI] | p |
|---|---|---|---|---|
| coloc_abf | NA [NA, NA] | NA | -0.142 [-0.269, -0.133] | 0.9967 |
| coloc_prop_test | NA [NA, NA] | NA | NA [NA, NA] | NA |
| coloc_susie | NA [NA, NA] | NA | 0.256 [-0.269, 0.300] | 0.2893 |
| coloc_susie_mm | NA [NA, NA] | NA | -0.079 [-0.269, -0.068] | 0.9974 |
| colocboost | NA [NA, NA] | NA | 0.218 [0.095, 0.222] | 0.0000 |
| colocboost_m1 | NA [NA, NA] | NA | -0.064 [-0.269, -0.040] | 0.9953 |
| famB_exactcoloc_gaussian_M50 | NA [NA, NA] | NA | -0.133 [-0.136, -0.131] | 0.8378 |
| lead_snp | NA [NA, NA] | NA | NA [NA, NA] | NA |
| prop_coloc | NA [NA, NA] | NA | NA [NA, NA] | NA |
| sharepro | NA [NA, NA] | NA | 0.459 [0.239, 0.481] | 0.0000 |
| strength_only | NA [NA, NA] | NA | -0.149 [-0.242, -0.146] | 0.9556 |

## C3 operating points and calibration (least_favourable)

| method | t_S | t_D | W→S | W→D | recall S | recall D | ECE_S raw / recal | ECE_D raw / recal | failures | median s/locus |
|---|---|---|---|---|---|---|---|---|---|---|
| coloc_abf | 0.8575 | 0.1821 | 0.025 | 0.038 | 0.663 | 0.992 | 0.116 / 0.111 | 0.117 / 0.153 | 0 (max cell 0.000) | 0.0077 |
| coloc_prop_test | inf | inf | 0.000 | 0.000 | 0.000 | 0.000 | NA / NA | NA / NA | 0 (max cell 0.000) | 1.3810 |
| coloc_susie | 0.8905 | 0.1821 | 0.033 | 0.038 | 0.750 | 0.592 | 0.164 / 0.206 | 0.153 / 0.170 | 2 (max cell 0.167) | 3.6373 |
| coloc_susie_mm | 0.8867 | 0.1821 | 0.025 | 0.038 | 0.750 | 0.892 | 0.084 / 0.106 | 0.068 / 0.102 | 2 (max cell 0.167) | 6.3781 |
| colocboost | 0.2177 | 0.7823 | 0.025 | 0.025 | 0.529 | 0.333 | 0.150 / 0.676 | 0.205 / 0.313 | 0 (max cell 0.000) | 2.3736 |
| colocboost_m1 | 0.8216 | 0.2696 | 0.013 | 0.025 | 0.663 | 0.875 | 0.144 / 0.324 | 0.158 / 0.671 | 0 (max cell 0.000) | 0.2661 |
| famA_joint_susie_rho_ms_ptld | 0.0187 | 0.0042 | 0.025 | 0.025 | 0.908 | 0.183 | 0.108 / 0.143 | 0.170 / 0.131 | 0 (max cell 0.000) | 0.4378 |
| famB_exactcoloc_gaussian_M50 | 0.0081 | 0.1490 | 0.038 | 0.025 | 0.767 | 0.704 | 0.096 / 0.081 | 0.083 / 0.119 | 0 (max cell 0.000) | 0.0615 |
| lead_snp | inf | inf | 0.000 | 0.000 | 0.000 | 0.000 | NA / NA | NA / NA | 0 (max cell 0.000) | 0.0097 |
| prop_coloc | inf | inf | 0.000 | 0.000 | 0.000 | 0.000 | NA / NA | NA / NA | 18 (max cell 0.667) | 1.5287 |
| sharepro | 0.7055 | 0.2733 | 0.025 | 0.025 | 0.938 | 0.233 | 0.191 / 0.055 | 0.198 / 0.245 | 0 (max cell 0.000) | 0.3271 |
| strength_only | 0.2252 | 0.2252 | 0.050 | 0.013 | 0.975 | 0.571 | 0.169 / 0.176 | 0.174 / 0.174 | 0 (max cell 0.000) | 0.0085 |

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
| prop_coloc | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| sharepro | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| strength_only | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

## Per-cell AUROC D vs W (least_favourable)

| method | L0|P1|r_p=0.0 | L0|P1|r_p=0.6 | L0|P2|rho=0.0 | L0|P2|rho=0.3 | L0|P3|rho=0.0 | L0|P3|rho=0.5 | L0|P3|rho=0.8 | L1swap|P1|r_p=0.0 | L1swap|P1|r_p=0.6 | L1|P1|r_p=0.0 | L1|P1|r_p=0.6 | L1|P2|rho=0.0 | L1|P2|rho=0.3 | L1|P3|rho=0.0 | L1|P3|rho=0.5 | L1|P3|rho=0.8 | L2swap|P1|r_p=0.0 | L2swap|P1|r_p=0.6 | L2|P1|r_p=0.0 | L2|P1|r_p=0.6 | L2|P2|rho=0.0 | L2|P2|rho=0.3 | L2|P3|rho=0.0 | L2|P3|rho=0.5 | L2|P3|rho=0.8 | L3|P1|r_p=0.0 | L3|P2|rho=0.0 | L3|P3|rho=0.0 | L4|P1|r_p=0.0 | L4|P1|r_p=0.6 | L4|P2|rho=0.0 | L4|P2|rho=0.3 | L4|P3|rho=0.0 | L4|P3|rho=0.5 | L4|P3|rho=0.8 | L6|P1|r_p=0.0 | L6|P2|rho=0.0 | L6|P3|rho=0.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coloc_abf | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| coloc_prop_test | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| coloc_susie | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 0.000 | 1.000 | 0.500 | 0.000 | 1.000 | 0.500 | 1.000 | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 | 1.000 | 0.500 | 0.625 | 0.500 | 0.500 | 0.000 | 0.375 | 0.500 |
| coloc_susie_mm | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 0.750 | 1.000 |
| colocboost | 1.000 | 1.000 | 0.750 | 1.000 | 0.750 | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 0.500 | 0.250 | 0.500 | 0.250 | 0.500 | 0.750 | 0.500 | 0.500 | 0.750 | 1.000 | 0.500 | 0.750 | 0.500 | 0.750 | 0.500 | 0.500 | 0.750 | 0.750 | 0.250 | 0.750 |
| colocboost_m1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.500 | 1.000 | 0.750 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.250 | NA | 1.000 | 0.625 | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 0.750 | 0.750 |
| famA_joint_susie_rho_ms_ptld | 0.625 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 1.000 | 0.750 | 0.500 | 0.750 | 1.000 | 0.750 | 0.750 | 0.500 | 1.000 | 1.000 | 0.000 | 1.000 | 0.750 | 0.375 | 0.500 | 0.750 | 0.625 | 1.000 | 0.625 | 1.000 |
| famB_exactcoloc_gaussian_M50 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 0.625 | 1.000 | 1.000 | 1.000 |
| lead_snp | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| prop_coloc | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| sharepro | 0.500 | 0.500 | 0.500 | 1.000 | 0.500 | 1.000 | 0.500 | 1.000 | 1.000 | 0.500 | 0.500 | 0.500 | 0.500 | 1.000 | 0.500 | 1.000 | 1.000 | NA | 0.000 | 0.500 | 0.500 | 0.250 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.750 | 0.000 | 0.250 | 0.250 | 0.000 | 0.500 | 0.250 | 0.000 | 0.000 |
| strength_only | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | NA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## D strata by r2 (least_favourable; AUPRC S vs D_b / false-S rate at t_S)

| method | [0,0.1) | [0.1,0.3) | [0.3,0.5) | [0.5,0.8) | [0.8,0.95) | out |
|---|---|---|---|---|---|---|
| coloc_abf | NA / 0.000 (n=18) | NA / 0.000 (n=18) | NA / 0.050 (n=20) | NA / 0.062 (n=16) | NA / NA (n=0) | NA / NA (n=0) |
| coloc_prop_test | NA / 0.000 (n=18) | NA / 0.000 (n=18) | NA / 0.000 (n=20) | NA / 0.000 (n=16) | NA / NA (n=0) | NA / NA (n=0) |
| coloc_susie | NA / 0.111 (n=18) | NA / 0.389 (n=18) | NA / 0.250 (n=20) | NA / 0.312 (n=16) | NA / NA (n=0) | NA / NA (n=0) |
| coloc_susie_mm | NA / 0.000 (n=18) | NA / 0.111 (n=18) | NA / 0.100 (n=20) | NA / 0.062 (n=16) | NA / NA (n=0) | NA / NA (n=0) |
| colocboost | NA / 0.000 (n=18) | NA / 0.000 (n=18) | NA / 0.050 (n=20) | NA / 0.438 (n=16) | NA / NA (n=0) | NA / NA (n=0) |
| colocboost_m1 | NA / 0.000 (n=18) | NA / 0.000 (n=18) | NA / 0.050 (n=20) | NA / 0.188 (n=16) | NA / NA (n=0) | NA / NA (n=0) |
| famA_joint_susie_rho_ms_ptld | NA / 0.167 (n=18) | NA / 0.500 (n=18) | NA / 0.550 (n=20) | NA / 0.562 (n=16) | NA / NA (n=0) | NA / NA (n=0) |
| famB_exactcoloc_gaussian_M50 | NA / 0.056 (n=18) | NA / 0.000 (n=18) | NA / 0.100 (n=20) | NA / 0.188 (n=16) | NA / NA (n=0) | NA / NA (n=0) |
| lead_snp | NA / 0.000 (n=18) | NA / 0.000 (n=18) | NA / 0.000 (n=20) | NA / 0.000 (n=16) | NA / NA (n=0) | NA / NA (n=0) |
| prop_coloc | NA / 0.000 (n=18) | NA / 0.000 (n=18) | NA / 0.000 (n=20) | NA / 0.000 (n=16) | NA / NA (n=0) | NA / NA (n=0) |
| sharepro | NA / 0.111 (n=18) | NA / 0.389 (n=18) | NA / 0.600 (n=20) | NA / 0.875 (n=16) | NA / NA (n=0) | NA / NA (n=0) |
| strength_only | NA / 0.556 (n=18) | NA / 0.333 (n=18) | NA / 0.650 (n=20) | NA / 0.500 (n=16) | NA / NA (n=0) | NA / NA (n=0) |

## D strata by d (least_favourable; AUPRC S vs D_b / false-S rate at t_S)

| method | [2,4) | [4,8) | [8,16) | [16,inf) | out |
|---|---|---|---|---|---|
| coloc_abf | NA / 0.333 (n=3) | NA / 0.000 (n=44) | NA / 0.045 (n=22) | NA / 0.000 (n=3) | NA / NA (n=0) |
| coloc_prop_test | NA / 0.000 (n=3) | NA / 0.000 (n=44) | NA / 0.000 (n=22) | NA / 0.000 (n=3) | NA / NA (n=0) |
| coloc_susie | NA / 0.667 (n=3) | NA / 0.273 (n=44) | NA / 0.227 (n=22) | NA / 0.000 (n=3) | NA / NA (n=0) |
| coloc_susie_mm | NA / 0.333 (n=3) | NA / 0.068 (n=44) | NA / 0.045 (n=22) | NA / 0.000 (n=3) | NA / NA (n=0) |
| colocboost | NA / 1.000 (n=3) | NA / 0.045 (n=44) | NA / 0.136 (n=22) | NA / 0.000 (n=3) | NA / NA (n=0) |
| colocboost_m1 | NA / 1.000 (n=3) | NA / 0.000 (n=44) | NA / 0.045 (n=22) | NA / 0.000 (n=3) | NA / NA (n=0) |
| famA_joint_susie_rho_ms_ptld | NA / 0.667 (n=3) | NA / 0.523 (n=44) | NA / 0.273 (n=22) | NA / 0.333 (n=3) | NA / NA (n=0) |
| famB_exactcoloc_gaussian_M50 | NA / 0.333 (n=3) | NA / 0.091 (n=44) | NA / 0.045 (n=22) | NA / 0.000 (n=3) | NA / NA (n=0) |
| lead_snp | NA / 0.000 (n=3) | NA / 0.000 (n=44) | NA / 0.000 (n=22) | NA / 0.000 (n=3) | NA / NA (n=0) |
| prop_coloc | NA / 0.000 (n=3) | NA / 0.000 (n=44) | NA / 0.000 (n=22) | NA / 0.000 (n=3) | NA / NA (n=0) |
| sharepro | NA / 1.000 (n=3) | NA / 0.455 (n=44) | NA / 0.455 (n=22) | NA / 0.667 (n=3) | NA / NA (n=0) |
| strength_only | NA / 0.333 (n=3) | NA / 0.409 (n=44) | NA / 0.727 (n=22) | NA / 0.667 (n=3) | NA / NA (n=0) |

## C3 dry run on this set (least_favourable; thresholds given)

- candidate famA_joint_susie_rho_ms_ptld; eligible ['coloc_abf', 'coloc_susie', 'coloc_susie_mm', 'sharepro', 'colocboost', 'colocboost_m1']; non-inferior to all: False; superior on one (Holm): False; **pass: False**

## C3 dry run on this set (excluded; thresholds given)

- candidate famA_joint_susie_rho_ms_ptld; eligible ['coloc_abf', 'coloc_susie', 'coloc_susie_mm', 'sharepro', 'colocboost', 'colocboost_m1']; non-inferior to all: False; superior on one (Holm): False; **pass: False**

