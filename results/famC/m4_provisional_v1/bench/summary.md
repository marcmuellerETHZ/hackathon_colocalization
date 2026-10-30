# Benchmark 20261027T184412Z_famC_m4_provisional

- loci: 12170 (D 3400, G 1020, S 3400, U 950, W 3400); cells: 34; per-cell subsample: None
- bootstrap: 2000 replicates, resampling unit block (BCa 95%); seed [20261029, 'run_benchmark', 'bootstrap', '20261027T184412Z_famC_m4_provisional']
- strength-only control (D6(e)): table crossfit_k5
- operating points: fitted on this set (C3 rule, in-sample)
- git 2dd9d8a897 clean=True

## Co-primary endpoints (least_favourable; n = 12170)

| method | realistic AUPRC S-vs-D | realistic AUROC D-vs-W | ideal AUPRC S-vs-D | ideal AUROC D-vs-W | Δ AUPRC vs control (realistic) |
|---|---|---|---|---|---|
| neuralcoloc_main_snap1_full | 0.951 [0.940, 0.958] | 0.995 [0.994, 0.997] | 0.981 [0.961, 0.990] | 0.999 [0.996, 1.000] | 0.454 [0.436, 0.473] |
| strength_only | 0.497 [0.480, 0.512] | 0.992 [0.990, 0.994] | 0.513 [0.463, 0.558] | 0.991 [0.977, 0.995] | control |

## Co-primary endpoints (excluded; n = 12170)

| method | realistic AUPRC S-vs-D | realistic AUROC D-vs-W | ideal AUPRC S-vs-D | ideal AUROC D-vs-W | Δ AUPRC vs control (realistic) |
|---|---|---|---|---|---|
| neuralcoloc_main_snap1_full | 0.951 [0.941, 0.958] | 0.995 [0.994, 0.997] | 0.981 [0.965, 0.990] | 0.999 [0.996, 1.000] | 0.454 [0.434, 0.471] |
| strength_only | 0.497 [0.480, 0.513] | 0.992 [0.990, 0.994] | 0.513 [0.465, 0.560] | 0.991 [0.979, 0.995] | control |

## C3 operating points and calibration (least_favourable)

| method | t_S | t_D | W→S | W→D | recall S | recall D | ECE_S raw / recal | ECE_D raw / recal | failures | median s/locus |
|---|---|---|---|---|---|---|---|---|---|---|
| neuralcoloc_main_snap1_full | 0.0115 | 0.0233 | 0.025 | 0.025 | 0.986 | 0.595 | 0.049 / 0.046 | 0.035 / 0.022 | 0 (max cell 0.000) | 0.0309 |
| strength_only | 0.1940 | 0.1996 | 0.024 | 0.025 | 0.942 | 0.025 | 0.025 / 0.025 | 0.030 / 0.029 | 0 (max cell 0.000) | 0.0359 |

## Per-cell AUPRC S vs D (least_favourable)

| method | L0|P1|r_p=0.0 | L0|P1|r_p=0.6 | L0|P2|rho=0.0 | L0|P2|rho=0.3 | L0|P3|rho=0.0 | L0|P3|rho=0.5 | L0|P3|rho=0.8 | L1|P1|r_p=0.0 | L1|P1|r_p=0.6 | L1|P2|rho=0.0 | L1|P2|rho=0.3 | L1|P3|rho=0.0 | L1|P3|rho=0.5 | L1|P3|rho=0.8 | L2|P1|r_p=0.0 | L2|P1|r_p=0.6 | L2|P2|rho=0.0 | L2|P2|rho=0.3 | L2|P3|rho=0.0 | L2|P3|rho=0.5 | L2|P3|rho=0.8 | L3|P1|r_p=0.0 | L3|P2|rho=0.0 | L3|P3|rho=0.0 | L4|P1|r_p=0.0 | L4|P1|r_p=0.6 | L4|P2|rho=0.0 | L4|P2|rho=0.3 | L4|P3|rho=0.0 | L4|P3|rho=0.5 | L4|P3|rho=0.8 | L6|P1|r_p=0.0 | L6|P2|rho=0.0 | L6|P3|rho=0.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| neuralcoloc_main_snap1_full | 0.973 | 0.977 | 0.990 | 0.965 | 0.989 | 0.981 | 0.970 | 0.973 | 0.981 | 0.977 | 0.966 | 0.968 | 0.949 | 0.988 | 0.962 | 0.956 | 0.946 | 0.965 | 0.965 | 0.970 | 0.977 | 0.971 | 0.945 | 0.971 | 0.959 | 0.949 | 0.959 | 0.954 | 0.952 | 0.933 | 0.968 | 0.864 | 0.925 | 0.943 |
| strength_only | 0.531 | 0.466 | 0.481 | 0.506 | 0.516 | 0.492 | 0.453 | 0.456 | 0.472 | 0.524 | 0.482 | 0.491 | 0.456 | 0.510 | 0.529 | 0.539 | 0.467 | 0.543 | 0.512 | 0.513 | 0.497 | 0.512 | 0.482 | 0.500 | 0.503 | 0.539 | 0.554 | 0.506 | 0.513 | 0.506 | 0.527 | 0.510 | 0.494 | 0.525 |

## Per-cell AUROC D vs W (least_favourable)

| method | L0|P1|r_p=0.0 | L0|P1|r_p=0.6 | L0|P2|rho=0.0 | L0|P2|rho=0.3 | L0|P3|rho=0.0 | L0|P3|rho=0.5 | L0|P3|rho=0.8 | L1|P1|r_p=0.0 | L1|P1|r_p=0.6 | L1|P2|rho=0.0 | L1|P2|rho=0.3 | L1|P3|rho=0.0 | L1|P3|rho=0.5 | L1|P3|rho=0.8 | L2|P1|r_p=0.0 | L2|P1|r_p=0.6 | L2|P2|rho=0.0 | L2|P2|rho=0.3 | L2|P3|rho=0.0 | L2|P3|rho=0.5 | L2|P3|rho=0.8 | L3|P1|r_p=0.0 | L3|P2|rho=0.0 | L3|P3|rho=0.0 | L4|P1|r_p=0.0 | L4|P1|r_p=0.6 | L4|P2|rho=0.0 | L4|P2|rho=0.3 | L4|P3|rho=0.0 | L4|P3|rho=0.5 | L4|P3|rho=0.8 | L6|P1|r_p=0.0 | L6|P2|rho=0.0 | L6|P3|rho=0.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| neuralcoloc_main_snap1_full | 0.999 | 0.997 | 0.998 | 0.998 | 1.000 | 0.998 | 0.998 | 0.997 | 0.997 | 0.997 | 0.997 | 0.995 | 0.998 | 0.997 | 0.998 | 0.998 | 0.998 | 0.997 | 0.996 | 0.997 | 0.999 | 0.996 | 0.992 | 0.999 | 0.999 | 0.993 | 0.996 | 0.996 | 0.994 | 0.992 | 1.000 | 0.991 | 0.988 | 0.994 |
| strength_only | 0.992 | 0.989 | 0.986 | 0.995 | 0.993 | 0.988 | 0.991 | 0.997 | 0.998 | 0.977 | 0.987 | 0.988 | 0.993 | 0.994 | 0.995 | 0.996 | 0.982 | 0.988 | 0.998 | 0.998 | 0.997 | 0.993 | 0.987 | 0.995 | 0.995 | 0.983 | 0.990 | 0.996 | 0.993 | 0.998 | 0.993 | 0.991 | 0.985 | 0.994 |

## D strata by r2 (least_favourable; AUPRC S vs D_b / false-S rate at t_S)

| method | [0,0.1) | [0.1,0.3) | [0.3,0.5) | [0.5,0.8) | [0.8,0.95) | out |
|---|---|---|---|---|---|---|
| neuralcoloc_main_snap1_full | 0.991 / 0.128 (n=695) | 0.982 / 0.193 (n=667) | 0.988 / 0.217 (n=700) | 0.949 / 0.463 (n=663) | 0.866 / 0.873 (n=675) | NA / NA (n=0) |
| strength_only | 0.507 / 0.931 (n=695) | 0.499 / 0.933 (n=667) | 0.491 / 0.940 (n=700) | 0.499 / 0.935 (n=663) | 0.491 / 0.956 (n=675) | NA / NA (n=0) |

## D strata by d (least_favourable; AUPRC S vs D_b / false-S rate at t_S)

| method | [2,4) | [4,8) | [8,16) | [16,inf) | out |
|---|---|---|---|---|---|
| neuralcoloc_main_snap1_full | 0.836 / 0.865 (n=533) | 0.955 / 0.313 (n=1587) | 0.984 / 0.228 (n=1020) | 0.989 / 0.288 (n=260) | NA / NA (n=0) |
| strength_only | 0.483 / 0.893 (n=533) | 0.507 / 0.907 (n=1587) | 0.505 / 0.996 (n=1020) | 0.463 / 1.000 (n=260) | NA / NA (n=0) |

