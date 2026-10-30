"""Screen unit tests (DESIGN Sec. 5 B): strength-aware collapse, multiplicity, lead-removed candidates."""
import math

import numpy as np

from famB.screen import LDAccess, screen


def _X(rng, n, p, ar=0.5):
    X = rng.standard_normal((n, p))
    for j in range(1, p):
        X[:, j] = ar * X[:, j - 1] + math.sqrt(1 - ar * ar) * X[:, j]
    return X


def _std(X):
    return (X - X.mean(0)) / X.std(0)


def test_identity_when_p_le_M():
    rng = np.random.default_rng(1)
    X = _std(_X(rng, 100, 30))
    sc = screen(rng.standard_normal(30), rng.standard_normal(30), LDAccess(X=X, n_ref=100), M=30)
    assert np.array_equal(sc.sel, np.arange(30)) and np.all(sc.mult == 1)


def test_strong_D_pair_r2_099_kept():
    """Review R2-alg change 2: lambda = 150, r^2 = 0.99 D pair keeps both SNPs."""
    rng = np.random.default_rng(2)
    n, p = 5000, 300
    X = _X(rng, n, p, ar=0.3)
    X[:, 101] = X[:, 100] + 0.1005 * rng.standard_normal(n)
    X = _std(X)
    R = X.T @ X / n
    assert 0.985 < R[100, 101] ** 2 < 0.995
    z1 = R[:, 100] * 150 + rng.standard_normal(p)
    z2 = R[:, 101] * 150 + rng.standard_normal(p)
    sc = screen(z1, z2, LDAccess(R=R), M=40)
    assert 100 in sc.sel and 101 in sc.sel


def test_exact_duplicates_collapse_with_multiplicity():
    rng = np.random.default_rng(3)
    n, p = 1000, 200
    X = _X(rng, n, p, ar=0.3)
    for k in (51, 52, 53):
        X[:, k] = X[:, 50]  # three exact copies of SNP 50
    X = _std(X)
    R = X.T @ X / n
    z1 = R[:, 50] * 20 + rng.standard_normal(p) * 0.0
    z2 = R[:, 120] * 10
    sc = screen(z1, z2, LDAccess(X=X, n_ref=n), M=30)
    reps = [j for j in (50, 51, 52, 53) if j in sc.sel]
    assert len(reps) == 1
    gi = int(np.where(sc.sel == reps[0])[0][0])
    assert sc.mult[gi] == 4
    assert all(sc.group[j] == gi for j in (50, 51, 52, 53))
    # weak signal: the collapse threshold is looser, but distinct SNPs at r^2 ~ 0.25 are never collapsed
    assert len(sc.sel) == 30


def test_lead_removed_finds_masked_secondary():
    """A secondary signal in moderate LD with a strong lead, behind many lead proxies, is found by the
    lead-removed list; negative control: without it (k_cond = 0) it is missed."""
    rng = np.random.default_rng(4)
    n, p = 5000, 400
    X = _X(rng, n, p, ar=0.3)
    base = X[:, 200].copy()
    for k in range(150, 250):  # 100 proxies of the lead with r^2 ~ 0.7 - 0.95
        if k != 200:
            X[:, k] = base + rng.uniform(0.25, 0.65) * rng.standard_normal(n)
    X[:, 300] = 0.55 * base + rng.standard_normal(n) * 0.83  # secondary, r ~ 0.55 with lead
    X = _std(X)
    R = X.T @ X / n
    lam = np.zeros(p); lam[200] = 60.0; lam[300] = 9.0
    z1 = R @ lam + rng.standard_normal(p) * 0.3
    z2 = rng.standard_normal(p)
    M = 20
    sc = screen(z1, z2, LDAccess(R=R), M=M, k_cond=3)
    assert 200 in sc.sel and 300 in sc.sel
    sc0 = screen(z1, z2, LDAccess(R=R), M=M, k_cond=0)
    assert 300 not in sc0.sel
