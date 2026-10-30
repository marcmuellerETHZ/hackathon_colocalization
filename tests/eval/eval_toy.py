"""Tiny synthetic locus sets with known answers (simple MVN; NOT the project generator).

LD: AR(1), r_ij = a^|i-j|. Classes by construction:
  S: one shared causal variant c (lambda1, lambda2) -> z_t ~ N(R e_c lambda_t, R)
  D: c1 != c2 far apart (r² ≈ 0)
  W: trait 1 signal only, trait 2 null
Truth mu_t = (R lambda_t)_c at the class-determining variants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eval.io import LocusData, write_locus_set
from eval.truth import write_truth


def ar1(p: int, a: float = 0.8) -> np.ndarray:
    i = np.arange(p)
    return a ** np.abs(i[:, None] - i[None, :])


def make_locus(lid, cls, rng, p=40, lam=9.0, ld_type="dense", n_ref=300, cell="L1|P1|r_p=0.0", block="toyb0",
               level="L1", pairing="P1", split="validation"):
    R = ar1(p)
    L = np.linalg.cholesky(R + 1e-10 * np.eye(p))
    lam1, lam2 = np.zeros(p), np.zeros(p)
    if cls == "S":
        c = int(rng.integers(5, p - 5))
        lam1[c] = lam * rng.choice([-1, 1])
        lam2[c] = lam * rng.choice([-1, 1])
        c1 = c2 = c
    elif cls == "D":
        c1 = int(rng.integers(0, 8))
        c2 = int(rng.integers(p - 8, p))
        lam1[c1] = lam * rng.choice([-1, 1])
        lam2[c2] = lam * rng.choice([-1, 1])
    else:  # W
        c1 = c2 = int(rng.integers(5, p - 5))
        lam1[c1] = lam
    m1, m2 = R @ lam1, R @ lam2
    z1 = m1 + L @ rng.standard_normal(p)
    z2 = m2 + L @ rng.standard_normal(p)
    maf = rng.uniform(0.05, 0.5, p)
    n1, n2 = 50000.0, 1000.0
    se1 = 1 / np.sqrt(2 * n1 * maf * (1 - maf))
    se2 = 1 / np.sqrt(2 * n2 * maf * (1 - maf))
    kw = dict(locus_id=lid, variant_id=np.array([f"chr1:{1000 + 10 * j}:A:G" for j in range(p)], dtype=object),
              pos=np.arange(1000, 1000 + 10 * p, 10), z1=z1, z2=z2, beta1=z1 * se1, se1=se1, beta2=z2 * se2, se2=se2,
              n1=np.full(p, n1), n2=np.full(p, n2), maf1=maf, maf2=maf, maf_ref=maf, n_ref=n_ref,
              rho_hat=0.0, rho_se=0.05, split=split, block_ids=(block,), mismatch_level=level, pairing_type=pairing,
              cell_id=cell, window_chr="chrToy", window_start=1000, window_end=1000 + 10 * p)
    if ld_type == "dense":
        loc = LocusData(ld_type="dense", R=R.astype(np.float32), **kw)
    else:
        X = (rng.standard_normal((n_ref, p)) @ L.T)
        X = (X - X.mean(0)) / X.std(0)
        loc = LocusData(ld_type="factor", X_ref=X.astype(np.float32), **kw)
    r2 = float(R[c1, c2] ** 2) if cls == "D" else np.nan
    truth = dict(locus_id=lid, oracle_class=cls, mu1=float(m1[c1]), mu2=float(m2[c2]) if cls != "W" else 0.0,
                 r2=r2, d=(lam * np.sqrt(1 - r2)) if cls == "D" else np.nan, rho=0.0, block_id=block, cell_id=cell,
                 split=split, mismatch_level=level, pairing_type=pairing, overlap=0.0, control="")
    return loc, truth


def make_locus_set(path, n_per_class=10, seed=1, ld_type="dense", n_blocks=6, p=40):
    rng = np.random.default_rng(seed)
    locs, truths = [], []
    k = 0
    for cls in ("S", "D", "W"):
        for i in range(n_per_class):
            loc, t = make_locus(f"loc{k:04d}", cls, rng, p=p, ld_type=ld_type, block=f"toyb{k % n_blocks}")
            locs.append(loc)
            truths.append(t)
            k += 1
    write_locus_set(path, locs)
    tp = str(path)[:-3] + ".truth.parquet"
    write_truth(pd.DataFrame(truths), tp)
    return locs, pd.DataFrame(truths), tp
