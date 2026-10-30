"""Tests for the L0* world (src/sim/l0star.py)."""
import itertools
import math

import numpy as np
import pytest

from sim import l0star
from sim.l0star import Segment, config_logprior, sample_config, sample_locus


def _all_configs(p):
    sets = [()] + [(i,) for i in range(p)] + list(itertools.combinations(range(p), 2))
    for c1 in sets:
        for c2 in sets:
            yield c1, c2


def test_config_prior_normalised_and_matches_sampler():
    p = 5
    tot = sum(math.exp(config_logprior(c1, c2, p)) for c1, c2 in _all_configs(p)
              if config_logprior(c1, c2, p) > -math.inf)
    assert abs(tot - 1) < 1e-12
    g = np.random.default_rng(20261026)
    n = 200_000
    cnt = {}
    for _ in range(n):
        _, c1, c2 = sample_config(g, p)
        cnt[(tuple(c1), tuple(c2))] = cnt.get((tuple(c1), tuple(c2)), 0) + 1
    chi2, df = 0.0, 0
    for (c1, c2), k in cnt.items():
        e = n * math.exp(config_logprior(c1, c2, p))
        assert e > 0, (c1, c2)
        chi2 += (k - e) ** 2 / e
        df += 1
    # every sampled config has positive prior and the frequencies fit (chi² / df close to 1)
    assert chi2 / df < 1.3, chi2 / df
    # no M-type configs, at most 2 per trait
    assert all(len(c1) <= 2 and len(c2) <= 2 and not (len(c1) == 2 and len(c2) == 2) for c1, c2 in cnt)


def test_lambda_mixture_and_data_model():
    for t in (1, 2):
        w, s = l0star.LAMBDA_MIX[t]
        assert len(w) == 8 and abs(w.sum() - 1) < 1e-12 and np.all(w >= 0)
    p = 6
    R = 0.7 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    Rt = (1 - l0star.ETA) * R + l0star.ETA * np.eye(p)
    seg = Segment("toy", np.arange(p), np.arange(p).astype(str), np.full(p, 0.3), R, Rt, np.linalg.cholesky(Rt),
                  ("toy", 0, p), "toy")
    g = np.random.default_rng(1)
    rho = 0.4
    E1, E2 = [], []
    for _ in range(20000):
        loc = sample_locus(seg, g, rho)
        assert np.allclose(loc["mu1"], Rt @ loc["lam1"]) and np.allclose(loc["mu2"], Rt @ loc["lam2"])
        E1.append(loc["z1"] - loc["mu1"])
        E2.append(loc["z2"] - loc["mu2"])
    E = np.hstack([np.array(E1), np.array(E2)])
    C = np.cov(E.T)
    V = np.array([[1, rho], [rho, 1]])
    assert np.abs(C - np.kron(V, Rt)).max() < 0.05
