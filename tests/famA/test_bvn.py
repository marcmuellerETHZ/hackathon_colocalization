"""Bivariate-normal rectangle probabilities (power events) vs independent adaptive quadrature."""
import numpy as np
import torch
from scipy.integrate import quad
from scipy.stats import norm

from famA.bvn import marginal, rect


def _ref(m1, m2, s11, s22, s12, r1, t1, r2, t2):
    sd1 = np.sqrt(s11); b = s12 / s11; cs = np.sqrt(max(s22 - s12 * b, 0))

    def g(x):
        cm = m2 + b * (x - m1)
        po = norm.cdf(-t2, cm, cs) + norm.sf(t2, cm, cs)
        return norm.pdf(x, m1, sd1) * (po if r2 == "out" else 1 - po)
    I1 = [(-np.inf, -t1), (t1, np.inf)] if r1 == "out" else [(-t1, t1)]
    tot = 0.0
    for a, bb in I1:
        lo, hi = max(a, m1 - 12 * sd1), min(bb, m1 + 12 * sd1)
        if lo < hi:
            tot += quad(g, lo, hi, epsabs=1e-14, epsrel=1e-12, limit=500)[0]
    return tot


def test_rect_vs_quadrature():
    rng = np.random.default_rng(20261029)
    worst = 0.0
    for k in range(120):
        m1, m2 = rng.normal(0, 8, 2)
        s11, s22 = np.exp(rng.uniform(np.log(0.01), np.log(5), 2))
        s12 = rng.uniform(-0.95, 0.95) * np.sqrt(s11 * s22)
        for r1, r2 in [("out", "out"), ("out", "in"), ("in", "out"), ("in", "in")]:
            t1, t2 = (6.29, 3.58) if k % 2 else (4.61, 5.26)
            a = rect(*[torch.tensor(v, dtype=torch.float64) for v in (m1, m2, s11, s22, s12)], r1, t1, r2, t2).item()
            worst = max(worst, abs(a - _ref(m1, m2, s11, s22, s12, r1, t1, r2, t2)))
    assert worst < 1e-6, worst


def test_degenerate_and_independent():
    m = torch.tensor([7.0, 0.0, -9.0], dtype=torch.float64)
    # zero variance -> indicator
    assert marginal(m, torch.zeros(3, dtype=torch.float64), "out", 6.29).tolist() == [1.0, 0.0, 1.0]
    # independent case = product of marginals
    s = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    p = rect(m, m + 1, s, s, torch.zeros(3, dtype=torch.float64), "out", 4.61, "in", 3.58)
    q = marginal(m, s.sqrt(), "out", 4.61) * marginal(m + 1, s.sqrt(), "in", 3.58)
    assert torch.allclose(p, q, atol=1e-9)
