"""prop.coloc failure diagnosis (decision P3-20): a method limitation, not a wrapper bug.

On screen_v2, every prop_coloc failure at L2/L4 (reference LD from n_ref = 150 individuals) was traced to
(a) an EXACT linear dependency among 3-4 of the <= 20 SNPs prop.coloc selects after its pairwise-r^2 clumping
(genotype columns with x_c = x_a + x_b in the small reference panel; eigenvalue ~1e-15 of the selected LD
sub-matrix), which makes `solve(A1)` fail inside the package's `try()` ("object 'solve0' not found" /
"Unable to compute the standard errors"), or (b) the package's own variance check 1 - B'A^-1 B < 0 for a trait
with tiny N under mismatched LD. The wrapper passes the LD unchanged, as a user would.

Known-answer test of (a): the same toy locus fails when (and only when) one selected SNP is the exact sum of two
others in the reference genotypes, and succeeds when that column is replaced by an independent SNP.
"""
import json

import numpy as np

from baselines.methods import PropColoc
from baselines.toy import make_toy_locus
from common.seeds import VALID_SEED, rng


def _panel(dependent: bool):
    g = rng(VALID_SEED, "tests/baselines/prop_coloc_singular")
    n = 150
    G = g.binomial(2, 0.3, size=(n, 8)).astype(float)
    G[:, 0] = 0.0
    G[:12, 0] = 1.0          # rare SNP a: carriers 0..11
    G[:, 1] = 0.0
    G[12:24, 1] = 1.0        # rare SNP b: carriers 12..23 (disjoint from a)
    if dependent:
        G[:, 2] = G[:, 0] + G[:, 1]   # c = a + b exactly (pairwise r^2 with a or b ~ 0.47 < clump 0.6)
    maf = G.mean(0) / 2
    maf = np.minimum(maf, 1 - maf)
    X = (G - G.mean(0)) / G.std(0)
    ids = np.array([f"chr22:{1000 + i}:A:G" for i in range(8)])
    pos = np.arange(1000, 1008, dtype=np.int64)
    return X, maf, ids, pos


def _fit(dependent: bool):
    X, maf, ids, pos = _panel(dependent)
    R = X.T @ X / X.shape[0]
    assert np.max(np.triu(R, 1) ** 2) < 0.6  # nothing is removed by prop.coloc's clumping
    g = rng(VALID_SEED, "tests/baselines/prop_coloc_singular", "z")
    loc = make_toy_locus(X, maf, ids, pos, {3: 8.0}, {3: 6.0}, g, N1=100_000, N2=5_000, locus_id="prop_sing")
    return np.linalg.eigvalsh(R)[0], PropColoc().fit_one(loc)


def test_prop_coloc_fails_exactly_on_singular_selected_ld():
    eig0, r = _fit(dependent=True)
    assert abs(eig0) < 1e-10
    assert r["status"] == "failed"
    err = json.loads(r["extra"])["error"]
    assert ("solve0" in err) or ("Unable to compute the standard errors" in err) or ("singular" in err), err


def test_prop_coloc_ok_without_the_dependency():
    eig0, r = _fit(dependent=False)
    assert eig0 > 1e-3
    assert r["status"] == "ok", r["extra"]
    assert 0.0 <= r["s"] <= 1.0
