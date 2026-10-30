"""Tests for src/stats/rho.py (DESIGN v4 §2.4, §7.1 item 3)."""
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from stats.rho import prune, rho_hat_genomewide, rho_hat_window, square_prob, truncated_mle

SEED = 20261026


@pytest.mark.parametrize("rho", [-0.9, -0.3, 0.0, 0.4, 0.8, 0.95])
def test_square_prob_matches_scipy(rho):
    ref = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([2, 2], lower_limit=[-2, -2])
    assert abs(square_prob(rho) - ref) < 1e-6


def test_truncated_mle_exact_bivariate_normal():
    """Positive control: pure bivariate normal, 2e5 SNPs -> estimate within 3 SE and SE calibrated."""
    g = np.random.default_rng(SEED)
    for rho in (-0.6, 0.0, 0.7):
        z = g.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=200_000)
        r, se, n = truncated_mle(z[:, 0], z[:, 1])
        assert abs(r - rho) < 3 * se, (rho, r, se)
        assert 0.0005 < se < 0.01


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.5, 0.8])
def test_genomewide_unbiased(rho):
    """DESIGN §7.1.3: |bias| < 0.01 at rho in {0, .3, .5, .8} with genome-wide SNP counts (20,000 SNPs,
    null + 1% polygenic signal). 200 replicates; also the mean reported SE matches the empirical SD."""
    g = np.random.default_rng([SEED, int(rho * 100)])
    est = [rho_hat_genomewide(rho, g) for _ in range(200)]
    r = np.array([e["rho_hat"] for e in est])
    se = np.array([e["rho_se"] for e in est])
    bias = r.mean() - rho
    print(f"rho={rho}: bias={bias:+.4f} (MC SE {r.std() / np.sqrt(len(r)):.4f}), sd={r.std():.4f}, mean SE={se.mean():.4f}")
    assert abs(bias) < 0.01
    assert 0.8 < se.mean() / r.std() < 1.25


def test_negative_control_independent_traits():
    """rho = 0 and strong independent signals in a few SNPs: estimate stays near 0."""
    g = np.random.default_rng(SEED + 1)
    z1, z2 = g.standard_normal(50_000), g.standard_normal(50_000)
    z1[:100] += 8.0
    z2[100:200] -= 8.0
    r, se, n = truncated_mle(z1, z2)
    assert abs(r) < 3 * se


def test_prune_known_answer():
    # three blocks of perfectly correlated SNPs + one independent SNP
    R = np.eye(7)
    for blk in ([0, 1, 2], [3, 4], [5]):
        for i in blk:
            for j in blk:
                R[i, j] = 1.0
    R[0, 6] = R[6, 0] = 0.2   # r² = 0.04 < 0.1: kept
    keep = prune(R, 0.1)
    assert keep.tolist() == [0, 3, 5, 6]
    keep = prune(R, 0.1, order=np.array([2, 1, 0, 4, 3, 5, 6]))
    assert keep.tolist() == [2, 4, 5, 6]


def test_window_mode_on_toy_ld():
    """Per-window mode: AR(1)-type LD, 3000 SNPs; pruned estimate unbiased over replicates."""
    g = np.random.default_rng(SEED + 2)
    p, a = 3000, 0.9
    idx = np.arange(p)
    R = a ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(R + 1e-9 * np.eye(p))
    est = []
    for _ in range(60):
        e1 = L @ g.standard_normal(p)
        e2 = 0.5 * e1 + np.sqrt(1 - 0.25) * (L @ g.standard_normal(p))
        est.append(rho_hat_window(e1, e2, R)["rho_hat"])
    est = np.array(est)
    assert abs(est.mean() - 0.5) < 3 * est.std() / np.sqrt(len(est)) + 0.01


# ----------------------------------------------------------------------------------------- D6(d) estimator
from stats.rho import (block_stats, estimate, jackknife_from_blocks, mle3_from_stats, mle_from_stats,
                       rho_hat_pooled, signal_mask, truncated_mle_brent)


def test_grid_mle_matches_brent():
    g = np.random.default_rng(SEED + 10)
    for rho in (-0.9, -0.4, 0.0, 0.3, 0.85):
        z = g.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=5000)
        rb, _, _ = truncated_mle_brent(z[:, 0], z[:, 1])
        rg, _, _ = truncated_mle(z[:, 0], z[:, 1])
        assert abs(rb - rg) < 1e-6, (rho, rb, rg)


def test_mle3_robust_to_variance_inflation():
    """Free marginal variances: inflated null variances (weak LD-tagged signal) do not bias rho-hat, while the
    unit-variance model is biased towards 0 (README D14, the reason for the 3-parameter model)."""
    g = np.random.default_rng(SEED + 11)
    rho, s1, s2 = 0.5, 1.3, 1.15
    z = g.multivariate_normal([0, 0], [[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]], size=400_000)
    m = (np.abs(z) < 2).all(1)
    a, b = z[m, 0], z[m, 1]
    r3, e1, e2 = mle3_from_stats(m.sum(), a @ a, b @ b, a @ b)
    r1 = float(mle_from_stats(m.sum(), a @ a + b @ b, a @ b))
    assert abs(r3 - rho) < 0.01 and abs(e1 - s1) < 0.01 and abs(e2 - s2) < 0.01, (r3, e1, e2)
    assert r1 < rho - 0.05, r1


def test_signal_mask_known_answer():
    p = 8
    R = np.eye(p)
    R[0, 1] = R[1, 0] = 0.5        # r² 0.25 > 0.1: masked with the signal SNP 0
    R[0, 2] = R[2, 0] = 0.3        # r² 0.09 < 0.1: kept
    R[5, 6] = R[6, 5] = 0.9
    z1 = np.zeros(p)
    z2 = np.zeros(p)
    z1[0] = 5.0                     # signal in trait 1
    z2[6] = -4.5                    # signal in trait 2
    m = signal_mask(z1, z2, R)
    assert m.tolist() == [True, True, False, False, False, True, True, False]
    m2 = signal_mask(z1, z2, ("factor", np.linalg.cholesky(R).T * np.sqrt(p)), 0.1)   # X'X/p = R
    assert m2.tolist() == m.tolist()
    assert not signal_mask(z1 * 0, z2 * 0, R).any()


def test_jackknife_se_calibrated_independent_blocks():
    """Block jackknife SE matches the spread over replicates when blocks are independent (AR(1) LD)."""
    g = np.random.default_rng(SEED + 12)
    p, a = 2000, 0.8
    idx = np.arange(p)
    R = a ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(R + 1e-9 * np.eye(p))
    est, se = [], []
    for _ in range(300):
        e1 = L @ g.standard_normal(p)
        e2 = 0.3 * e1 + np.sqrt(1 - 0.09) * (L @ g.standard_normal(p))
        use = (np.abs(e1) < 2) & (np.abs(e2) < 2)
        n, sqq, s12 = block_stats(e1, e2, use, 20)
        r, s = jackknife_from_blocks(n, sqq, s12)
        est.append(r)
        se.append(s)
    est, se = np.array(est), np.array(se)
    ratio = est.std() / np.sqrt((se ** 2).mean())
    assert abs(est.mean() - 0.3) < 0.01
    assert 0.85 < ratio < 1.15, ratio


def test_pooled_mode_windows_as_blocks():
    g = np.random.default_rng(SEED + 13)
    wins = []
    for _ in range(40):
        p = 300
        z1 = g.standard_normal(p)
        z2 = 0.2 * z1 + np.sqrt(1 - 0.04) * g.standard_normal(p)
        z1[:3] += 10.0                   # a signal, masked with its LD (identity LD here: itself only)
        wins.append((z1, z2, np.eye(p)))
    out = rho_hat_pooled(wins)
    assert out["n_blocks"] == 40 and 120 <= out["n_masked"] <= 130   # + chance |z| > 4 null SNPs
    assert abs(out["rho_hat"] - 0.2) < 3 * out["rho_se"]


def test_residualize_removes_concordant_leakage_and_keeps_null():
    """Low-effective-rank LD (like the mosaic windows): a strong concordant shared signal leaks into |z| < 2 SNPs
    beyond the r² > 0.1 mask and biases rho-hat upwards; the residualised estimator (exact LD) removes it, and
    on null data it stays unbiased (README D14)."""
    g = np.random.default_rng(SEED + 14)
    p, k = 600, 25
    F = g.standard_normal((p, k))
    R = 0.5 * (F @ F.T) / k + 0.5 * np.eye(p)
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)
    L = np.linalg.cholesky(R)
    res = {"plain": [], "resid": [], "null_plain": [], "null_resid": []}
    for _ in range(300):
        e1 = L @ g.standard_normal(p)
        e2 = 0.3 * e1 + np.sqrt(1 - 0.09) * (L @ g.standard_normal(p))
        res["null_plain"].append(rho_hat_window(e1, e2, R)["rho_hat"])
        res["null_resid"].append(rho_hat_window(e1, e2, R, residualize_signal=True)["rho_hat"])
        lam = np.zeros(p)
        lam[g.integers(p)] = 12.0
        mu = R @ lam
        z1, z2 = mu + e1, 0.8 * mu + e2                       # concordant sharing
        res["plain"].append(rho_hat_window(z1, z2, R)["rho_hat"])
        res["resid"].append(rho_hat_window(z1, z2, R, residualize_signal=True)["rho_hat"])
    m = {k_: float(np.nanmean(v)) for k_, v in res.items()}
    se = {k_: float(np.nanstd(v) / np.sqrt(np.isfinite(v).sum())) for k_, v in res.items()}
    assert all(np.isfinite(v).mean() > 0.95 for v in res.values())
    print(m, se)
    assert m["plain"] - 0.3 > 4 * se["plain"] + 0.02, (m, se)     # the leakage is real
    for key in ("resid", "null_plain", "null_resid"):
        assert abs(m[key] - 0.3) < 3 * se[key] + 0.01, (key, m, se)
