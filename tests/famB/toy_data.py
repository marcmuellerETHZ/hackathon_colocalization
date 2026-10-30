"""Toy locus builders for Family B tests (known answers)."""
from __future__ import annotations

import math

import numpy as np

from eval.io import LocusData


def ar_genotypes(rng, n, p, ar=0.8):
    X = rng.standard_normal((n, p))
    for j in range(1, p):
        X[:, j] = ar * X[:, j - 1] + math.sqrt(1 - ar * ar) * X[:, j]
    return (X - X.mean(0)) / X.std(0)


def make_locus(lid, X, lam1, lam2, rho=0.0, N1=1e5, N2=1e3, rng=None, maf=None, factor=False, z_override=None):
    """z_t = R lam_t + X' u_t / sqrt(n), corr(u1, u2) = rho  (exact in-sample RSS model, L0-like)."""
    n, p = X.shape
    R = X.T @ X / n
    if z_override is None:
        u1 = rng.standard_normal(n)
        u2 = rho * u1 + math.sqrt(1 - rho * rho) * rng.standard_normal(n)
        z1 = R @ lam1 + X.T @ u1 / math.sqrt(n)
        z2 = R @ lam2 + X.T @ u2 / math.sqrt(n)
    else:
        z1, z2 = z_override
    f = maf if maf is not None else rng.uniform(0.02, 0.5, p)
    se1 = 1.0 / np.sqrt(2 * N1 * f * (1 - f))
    se2 = 1.0 / np.sqrt(2 * N2 * f * (1 - f))
    kw = dict(ld_type="factor", n_ref=n, X_ref=X.astype(np.float32)) if factor else \
        dict(ld_type="dense", n_ref=n, R=R.astype(np.float32))
    return LocusData(locus_id=lid, variant_id=np.array([f"chr1:{j + 1}:A:C" for j in range(p)]),
                     pos=np.arange(1, p + 1, dtype=np.int64), z1=z1, z2=z2, beta1=z1 * se1, se1=se1,
                     beta2=z2 * se2, se2=se2, n1=np.full(p, N1), n2=np.full(p, N2), maf1=f, maf2=f, maf_ref=f,
                     rho_hat=rho, **kw)
