# Benchmark 20261027T102443Z_p4_screen_v2_control

- loci: 12000 (D 6000, S 3000, W 3000); cells: 6; per-cell subsample: None
- bootstrap: 2000 replicates, resampling unit locus (BCa 95%); seed [20261029, 'run_benchmark', 'bootstrap', '20261027T102443Z_p4_screen_v2_control']
- strength-only control (D6(e)): table crossfit_k5
- operating points: fitted on this set (C3 rule, in-sample)
- git 93ecabb3da clean=True

## Co-primary endpoints (least_favourable; n = 12000)

| method | realistic AUPRC S-vs-D | realistic AUROC D-vs-W | ideal AUPRC S-vs-D | ideal AUROC D-vs-W | Δ AUPRC vs control (realistic) |
|---|---|---|---|---|---|
| strength_only | 0.517 [0.507, 0.531] | 0.978 [0.974, 0.982] | 0.486 [0.464, 0.503] | 0.978 [0.969, 0.984] | control |

## Co-primary endpoints (excluded; n = 12000)

| method | realistic AUPRC S-vs-D | realistic AUROC D-vs-W | ideal AUPRC S-vs-D | ideal AUROC D-vs-W | Δ AUPRC vs control (realistic) |
|---|---|---|---|---|---|
| strength_only | 0.517 [0.507, 0.530] | 0.978 [0.974, 0.982] | 0.486 [0.465, 0.503] | 0.978 [0.970, 0.984] | control |

## C3 operating points and calibration (least_favourable)

| method | t_S | t_D | W→S | W→D | recall S | recall D | ECE_S raw / recal | ECE_D raw / recal | failures | median s/locus |
|---|---|---|---|---|---|---|---|---|---|---|
| strength_only | 0.3105 | 0.5923 | 0.023 | 0.025 | 0.609 | 0.335 | 0.021 / 0.021 | 0.028 / 0.028 | 0 (max cell 0.000) | 0.0327 |

## Per-cell AUPRC S vs D (least_favourable)

| method | L0|P1|r_p=0.0 | L0|P3|rho=0.5 | L2|P1|r_p=0.0 | L2|P3|rho=0.5 | L4|P1|r_p=0.0 | L4|P3|rho=0.5 |
|---|---|---|---|---|---|---|
| strength_only | 0.486 | 0.521 | 0.510 | 0.535 | 0.518 | 0.512 |

## Per-cell AUROC D vs W (least_favourable)

| method | L0|P1|r_p=0.0 | L0|P3|rho=0.5 | L2|P1|r_p=0.0 | L2|P3|rho=0.5 | L4|P1|r_p=0.0 | L4|P3|rho=0.5 |
|---|---|---|---|---|---|---|
| strength_only | 0.978 | 0.975 | 0.981 | 0.972 | 0.977 | 0.983 |

## D strata by r2 (least_favourable; AUPRC S vs D_b / false-S rate at t_S)

| method | [0,0.1) | [0.1,0.3) | [0.3,0.5) | [0.5,0.8) | [0.8,0.95) | out |
|---|---|---|---|---|---|---|
| strength_only | 0.517 / 0.512 (n=2016) | 0.544 / 0.611 (n=529) | 0.503 / 0.591 (n=455) | 0.514 / 0.566 (n=3000) | NA / NA (n=0) | NA / NA (n=0) |

## D strata by d (least_favourable; AUPRC S vs D_b / false-S rate at t_S)

| method | [2,4) | [4,8) | [8,16) | [16,inf) | out |
|---|---|---|---|---|---|
| strength_only | 0.527 / 0.332 (n=717) | 0.518 / 0.490 (n=3341) | 0.509 / 0.721 (n=1672) | 0.495 / 0.900 (n=270) | NA / NA (n=0) |

