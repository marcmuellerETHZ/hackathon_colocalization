"""Local, vectorised re-implementation of coloc 5.2.3 `coloc.abf` (beta/varbeta input, quantitative trait).

coloc hard-codes the prior effect SD (0.15 * sdY for "quant") in `coloc:::approx.bf.estimates`; DESIGN §4.1
tunes W over {0.1, 0.15, 0.2}, so the ABF is re-implemented here with W as an argument. Validated against
R coloc 5.2.3 to < 1e-8 (tests/baselines/test_abf_vs_r.py; the R reference uses sdY = W / 0.15, which makes
R's prior SD equal to W exactly).

Formulas (coloc 5.2.3 source):
    r     = W^2 / (W^2 + V),  V = se^2 (sdY = 1, DESIGN §2.2)
    lABF  = 0.5 * (log(1 - r) + r z^2)
    lH0 = 0, lH1 = log p1 + logsum(l1), lH2 = log p2 + logsum(l2),
    lH3 = log p1 + log p2 + logdiff(logsum(l1) + logsum(l2), logsum(l1 + l2)),
    lH4 = log p12 + logsum(l1 + l2);  PP = softmax(lH)
    adjust_prior: if nsnps * p >= 1 then p = 1 / (nsnps + 1)   (applied to p1, p2, p12 separately)
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

DEFAULT_P1 = 1e-4
DEFAULT_P2 = 1e-4


def labf(z: np.ndarray, se: np.ndarray, W: float) -> np.ndarray:
    """Wakefield log approximate Bayes factor per SNP (coloc's approx.bf.estimates with sd.prior = W)."""
    z = np.asarray(z, dtype=np.float64)
    V = np.asarray(se, dtype=np.float64) ** 2
    r = W * W / (W * W + V)
    return 0.5 * (np.log1p(-r) + r * z * z)


def _adjust_prior(p: float, nsnps: int) -> float:
    return 1.0 / (nsnps + 1) if nsnps * p >= 1 else p


def _logdiff(x: float, y: float) -> float:
    m = max(x, y)
    with np.errstate(divide="ignore", invalid="ignore"):  # log(0) = -inf -> PP3 = 0, as in R
        return m + np.log(np.exp(x - m) - np.exp(y - m))


def combine_abf(l1: np.ndarray, l2: np.ndarray, p1: float, p2: float, p12: float) -> np.ndarray:
    """PP.H0..PP.H4 (coloc:::combine.abf, priors already adjusted)."""
    ls1, ls2, ls12 = logsumexp(l1), logsumexp(l2), logsumexp(l1 + l2)
    lH = np.array([
        0.0,
        np.log(p1) + ls1,
        np.log(p2) + ls2,
        np.log(p1) + np.log(p2) + _logdiff(ls1 + ls2, ls12),
        np.log(p12) + ls12,
    ])
    return np.exp(lH - logsumexp(lH))


def coloc_abf(z1, se1, z2, se2, W: float = 0.15, p1: float = DEFAULT_P1, p2: float = DEFAULT_P2,
              p12: float = 1e-5, return_snp_pp: bool = False):
    """coloc.abf on z / se (beta = z * se). Returns PP vector (5,), and optionally SNP.PP.H4 (p,)."""
    l1 = labf(z1, se1, W)
    l2 = labf(z2, se2, W)
    n = l1.shape[0]
    pp = combine_abf(l1, l2, _adjust_prior(p1, n), _adjust_prior(p2, n), _adjust_prior(p12, n))
    if return_snp_pp:
        s = l1 + l2
        return pp, np.exp(s - logsumexp(s))
    return pp


def coloc_abf_grid(z1, se1, z2, se2, W_grid=(0.1, 0.15, 0.2), p12_grid=(5e-6, 1e-5),
                   p1: float = DEFAULT_P1, p2: float = DEFAULT_P2) -> dict:
    """All (W, p12) combinations for one locus; lABF computed once per W. Keys 'W=<W>,p12=<p12>' -> PP list."""
    out = {}
    n = len(np.asarray(z1))
    a1, a2 = _adjust_prior(p1, n), _adjust_prior(p2, n)
    for W in W_grid:
        l1, l2 = labf(z1, se1, W), labf(z2, se2, W)
        ls1, ls2, ls12 = logsumexp(l1), logsumexp(l2), logsumexp(l1 + l2)
        lh3 = np.log(a1) + np.log(a2) + _logdiff(ls1 + ls2, ls12)
        for p12 in p12_grid:
            lH = np.array([0.0, np.log(a1) + ls1, np.log(a2) + ls2, lh3, np.log(_adjust_prior(p12, n)) + ls12])
            out[f"W={W:g},p12={p12:g}"] = np.exp(lH - logsumexp(lH))
    return out
