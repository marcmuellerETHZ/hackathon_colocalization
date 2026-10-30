"""Local vectorised coloc.abf (src/baselines/abf.py) vs R coloc 5.2.3 coloc.abf, to 1e-8, on many loci.

R's prior SD is hard-coded to 0.15 * sdY; passing sdY = W / 0.15 makes it exactly W, so every W in the tuning grid
{0.1, 0.15, 0.2} is checked against R itself. p12 in {5e-6, 1e-5}; p >= 10^4 exercises coloc's adjust_prior.
"""
import numpy as np
import pytest

from baselines.abf import coloc_abf, coloc_abf_grid
from baselines.rworker import RWorker
from common.seeds import VALID_SEED, rng

W_GRID = (0.1, 0.15, 0.2)
P12_GRID = (5e-6, 1e-5)


def _random_loci(n_loci=120):
    g = rng(VALID_SEED, "tests/baselines/abf")
    loci = []
    for i in range(n_loci):
        p = int(g.choice([1, 2, 10, 50, 200, 1000, 3000])) if i < n_loci - 4 else int(g.choice([10_000, 12_000]))
        maf = g.uniform(0.01, 0.5, p)
        N1, N2 = g.choice([2e4, 1e5, 4e5]), g.choice([100, 300, 1000, 5000, 5e4])
        se1 = 1 / np.sqrt(2 * N1 * maf * (1 - maf))
        se2 = 1 / np.sqrt(2 * N2 * maf * (1 - maf))
        z1, z2 = g.standard_normal(p), g.standard_normal(p)
        kind = i % 4  # 0 null, 1 shared, 2 distinct, 3 one-trait
        j, k = g.integers(0, p, 2)
        lam1, lam2 = g.uniform(2, 40), g.uniform(2, 40)
        if kind == 1:
            z1[j] += lam1
            z2[j] += lam2
        elif kind == 2:
            z1[j] += lam1
            z2[k] += lam2
        elif kind == 3:
            z1[j] += lam1
        loci.append((z1, se1, z2, se2))
    return loci


@pytest.fixture(scope="module")
def worker():
    w = RWorker("main")
    yield w
    w.close()


def test_abf_matches_r(worker):
    loci = _random_loci()
    max_pp = max_snp = 0.0
    n_cmp = 0
    for z1, se1, z2, se2 in loci:
        grid = coloc_abf_grid(z1, se1, z2, se2, W_GRID, P12_GRID)
        for W in W_GRID:
            for p12 in P12_GRID:
                arrays = {"beta1": z1 * se1, "varbeta1": se1 ** 2, "beta2": z2 * se2, "varbeta2": se2 ** 2}
                r = worker.call("coloc_abf", {"sdY": W / 0.15, "p1": 1e-4, "p2": 1e-4, "p12": p12}, arrays)["result"]
                pp_r = np.asarray(r["pp"], dtype=float)
                snp_r = np.atleast_1d(np.asarray(r["snp_pp_h4"], dtype=float))
                pp, snp = coloc_abf(z1, se1, z2, se2, W=W, p12=p12, return_snp_pp=True)
                d = np.nanmax(np.abs(pp - pp_r))
                assert np.array_equal(np.isnan(pp), np.isnan(pp_r))
                assert d < 1e-8, (W, p12, len(z1), pp, pp_r)
                np.testing.assert_allclose(grid[f"W={W:g},p12={p12:g}"], pp, rtol=0, atol=1e-14)
                ds = np.max(np.abs(snp - snp_r))
                assert ds < 1e-8
                max_pp, max_snp = max(max_pp, d), max(max_snp, ds)
                n_cmp += 1
    print(f"\n[abf vs R] {len(loci)} loci x {len(W_GRID) * len(P12_GRID)} settings = {n_cmp} comparisons; "
          f"max |dPP| = {max_pp:.3e}, max |dSNP.PP.H4| = {max_snp:.3e}")
    assert n_cmp == len(loci) * 6
