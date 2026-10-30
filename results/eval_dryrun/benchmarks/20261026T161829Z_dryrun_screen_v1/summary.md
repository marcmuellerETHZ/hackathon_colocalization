# Benchmark 20261026T161829Z_dryrun_screen_v1

- loci: 600 (D 284, S 150, W 166); cells: 6; per-cell subsample: 100
- bootstrap: 2000 replicates, resampling unit locus (BCa 95%); seed [20261029, 'run_benchmark', 'bootstrap', '20261026T161829Z_dryrun_screen_v1']
- strength-only control (D6(e)): table crossfit_k5
- operating points: fitted on this set (C3 rule, in-sample)
- git 83923a989f clean=False

## Co-primary endpoints (least_favourable; n = 600)

| method | realistic AUPRC S-vs-D | realistic AUROC D-vs-W | ideal AUPRC S-vs-D | ideal AUROC D-vs-W | Δ AUPRC vs control (realistic) |
|---|---|---|---|---|---|
| coloc_abf | 0.938 [0.886, 0.972] | 0.987 [0.969, 0.995] | 0.940 [0.828, 0.986] | 1.000 [1.000, 1.000] | 0.507 [0.431, 0.587] |
| lead_snp | 0.782 [0.694, 0.883] | NA [NA, NA] | 0.907 [0.814, 1.000] | NA [NA, NA] | 0.351 [0.246, 0.499] |
| strength_only | 0.431 [0.368, 0.482] | 0.922 [0.880, 0.952] | 0.487 [0.376, 0.633] | 0.942 [0.867, 0.978] | control |

## Co-primary endpoints (excluded; n = 600)

| method | realistic AUPRC S-vs-D | realistic AUROC D-vs-W | ideal AUPRC S-vs-D | ideal AUROC D-vs-W | Δ AUPRC vs control (realistic) |
|---|---|---|---|---|---|
| coloc_abf | 0.938 [0.885, 0.974] | 0.987 [0.970, 0.995] | 0.940 [0.841, 0.986] | 1.000 [1.000, 1.000] | 0.507 [0.439, 0.595] |
| lead_snp | 0.782 [0.694, 0.884] | NA [NA, NA] | 0.907 [0.806, 1.000] | NA [NA, NA] | 0.351 [0.247, 0.504] |
| strength_only | 0.431 [0.366, 0.482] | 0.922 [0.880, 0.952] | 0.487 [0.380, 0.630] | 0.942 [0.865, 0.981] | control |

## C3 operating points and calibration (least_favourable)

| method | t_S | t_D | W→S | W→D | recall S | recall D | ECE_S raw / recal | ECE_D raw / recal | failures | median s/locus |
|---|---|---|---|---|---|---|---|---|---|---|
| coloc_abf | 0.7674 | 0.2544 | 0.019 | 0.023 | 0.733 | 0.966 | 0.080 / 0.091 | 0.067 / 0.097 | 0 (max cell 0.000) | 0.1418 |
| lead_snp | inf | inf | 0.000 | 0.000 | 0.000 | 0.000 | NA / NA | NA / NA | 0 (max cell 0.000) | 0.0249 |
| strength_only | 0.4777 | 0.5440 | 0.023 | 0.024 | 0.051 | 0.393 | 0.066 / 0.065 | 0.063 / 0.090 | 0 (max cell 0.000) | 0.0963 |

## Per-cell AUPRC S vs D (least_favourable)

| method | L0|P1|r_p=0.0 | L0|P3|rho=0.5 | L2|P1|r_p=0.0 | L2|P3|rho=0.5 | L4|P1|r_p=0.0 | L4|P3|rho=0.5 |
|---|---|---|---|---|---|---|
| coloc_abf | 0.940 | 0.783 | 0.947 | 0.873 | 0.959 | 0.957 |
| lead_snp | 0.907 | 0.688 | 0.782 | 0.686 | 0.806 | 0.837 |
| strength_only | 0.487 | 0.409 | 0.412 | 0.518 | 0.459 | 0.437 |

## Per-cell AUROC D vs W (least_favourable)

| method | L0|P1|r_p=0.0 | L0|P3|rho=0.5 | L2|P1|r_p=0.0 | L2|P3|rho=0.5 | L4|P1|r_p=0.0 | L4|P3|rho=0.5 |
|---|---|---|---|---|---|---|
| coloc_abf | 1.000 | 0.950 | 0.974 | 0.977 | 1.000 | 0.994 |
| lead_snp | NA | NA | NA | NA | NA | NA |
| strength_only | 0.942 | 0.936 | 0.873 | 0.949 | 0.940 | 0.963 |

## D strata by r2 (least_favourable; AUPRC S vs D_b / false-S rate at t_S)

| method | [0,0.1) | [0.1,0.3) | [0.3,0.5) | [0.5,0.8) | [0.8,0.95) | out |
|---|---|---|---|---|---|---|
| coloc_abf | 0.984 / 0.011 (n=91) | 0.990 / 0.000 (n=22) | 0.962 / 0.000 (n=26) | 0.906 / 0.062 (n=145) | NA / NA (n=0) | NA / NA (n=0) |
| lead_snp | 0.890 / 0.000 (n=91) | 0.919 / 0.000 (n=22) | 0.956 / 0.000 (n=26) | 0.677 / 0.000 (n=145) | NA / NA (n=0) | NA / NA (n=0) |
| strength_only | 0.487 / 0.033 (n=91) | 0.419 / 0.045 (n=22) | 0.390 / 0.000 (n=26) | 0.496 / 0.048 (n=145) | NA / NA (n=0) | NA / NA (n=0) |

## D strata by d (least_favourable; AUPRC S vs D_b / false-S rate at t_S)

| method | [2,4) | [4,8) | [8,16) | [16,inf) | out |
|---|---|---|---|---|---|
| coloc_abf | 0.876 / 0.241 (n=29) | 0.923 / 0.019 (n=157) | 0.996 / 0.000 (n=86) | 1.000 / 0.000 (n=12) | NA / NA (n=0) |
| lead_snp | 0.505 / 0.000 (n=29) | 0.771 / 0.000 (n=157) | 0.947 / 0.000 (n=86) | 0.917 / 0.000 (n=12) | NA / NA (n=0) |
| strength_only | 0.713 / 0.000 (n=29) | 0.448 / 0.064 (n=157) | 0.512 / 0.012 (n=86) | 0.500 / 0.000 (n=12) | NA / NA (n=0) |

