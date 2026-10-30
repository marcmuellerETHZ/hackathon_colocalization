"""Monte Carlo posteriors of a locus must not depend on the batch it is evaluated in (per-locus keyed draws,
derive(seed, "famB_mc", seed_key, locus_id)). Covers the rectangular (gaussian) and generic (mixture) paths."""
import json

import numpy as np
import pytest

from common.seeds import rng as seeded_rng
from famB.method import ExactColoc
from famB.priors import toy_l0star_prior
from toy_data import ar_genotypes, make_locus

COLS = ("s", "P_S", "P_D", "P_W", "E1", "E2")
EXTRA = ("P_U", "P_G", "P_W_strict")  # Monte Carlo: bitwise identical
EXACT = ("max_pip1", "max_pip2")  # exact sums in the gaussian path: GPU reduction order may differ by ~1e-16


def _loci(n, p, key):
    g = seeded_rng(20261029, "famB_test_batch_invariance", key)
    X = ar_genotypes(g, 400, p)
    out = []
    for i in range(n):
        lam1, lam2 = np.zeros(p), np.zeros(p)
        a, b = g.choice(p, 2, replace=False)
        lam1[a] = g.uniform(3, 12) * g.choice([-1, 1])
        lam2[b if i % 2 else a] = g.uniform(3, 10) * g.choice([-1, 1])
        out.append(make_locus(f"loc{i:03d}", X, lam1, lam2, rho=0.3 * (i % 3 == 0), rng=g))
    return out


def _by_id(rows):
    return {r["locus_id"]: r for r in rows}


def _assert_same(r1, r2):
    for c in COLS:
        assert r1[c] == r2[c] or (np.isnan(r1[c]) and np.isnan(r2[c])), (c, r1[c], r2[c])
    e1, e2 = json.loads(r1["extra"]), json.loads(r2["extra"])
    for c in EXTRA:
        assert e1[c] == e2[c], (c, e1[c], e2[c])
    for c in EXACT:
        assert abs(e1[c] - e2[c]) <= 1e-12, (c, e1[c], e2[c])
    for h, v in e1["PP"].items():
        assert abs(v - e2["PP"][h]) <= 1e-12, (h, v, e2["PP"][h])


@pytest.mark.parametrize("mode", ["gaussian", "simulator"])
def test_same_locus_same_posteriors_in_any_batch(mode, device):
    p = 30 if mode == "gaussian" else 7
    loci = _loci(6, p, mode)
    kw = dict(M=20, n_mc=4096) if mode == "gaussian" else \
        dict(M=20, n_mc=4096, eta=0.0, eta_floor=0.0, sim_prior_fn=lambda _l: toy_l0star_prior(K=3))
    ref = _by_id(ExactColoc(mode=mode, device=device, **kw).fit_batch(loci))
    # different composition, order and batch sizes; fresh instance (no shared generator state)
    rev = _by_id(ExactColoc(mode=mode, device=device, **kw).fit_batch(loci[::-1]))
    m = ExactColoc(mode=mode, device=device, **kw)
    single = {}
    for lo in (loci[3], loci[0]):
        single.update(_by_id(m.fit_batch([lo])))
    pair = _by_id(ExactColoc(mode=mode, device=device, **kw).fit_batch([loci[5], loci[3], loci[1]]))
    for other in (rev, single, pair):
        for lid, r in other.items():
            assert r["status"] == "ok"
            _assert_same(ref[lid], r)
    # negative control: a different seed_key changes the draws
    alt = _by_id(ExactColoc(mode=mode, device=device, seed_key="other", **kw).fit_batch(loci))
    assert any(alt[k]["P_S"] != ref[k]["P_S"] or alt[k]["P_D"] != ref[k]["P_D"] for k in ref)


def test_seed_must_come_from_seed_list():
    with pytest.raises(ValueError):
        ExactColoc(mode="gaussian", seed=12345, device="cpu")
