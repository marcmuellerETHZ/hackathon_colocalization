"""Rectangle-type probabilities of a bivariate normal for the power events of Family A (float64 torch).

P(X1 in A1, X2 in A2) for (X1, X2) ~ N((m1, m2), [[s11, s12], [s12, s22]]) where each A_t is either
'out' = {|x| >= t} or 'in' = {|x| < t}. Computed by conditioning on X1 and Gauss-Legendre quadrature in x
over each piece of A1 clipped to m1 +- 10 sd1. Symmetric
regions allow flipping signs so that m1, m2 >= 0. Degenerate cases (s11 = 0 or s22 = 0) are exact.
Validated against scipy.stats.multivariate_normal in tests/famA/test_bvn.py.
"""
from __future__ import annotations

import numpy as np
import torch

_GL_CACHE: dict = {}


def _gl(n: int, device):
    key = (n, str(device))
    if key not in _GL_CACHE:
        x, w = np.polynomial.legendre.leggauss(n)
        _GL_CACHE[key] = (torch.as_tensor(x, dtype=torch.float64, device=device),
                          torch.as_tensor(w, dtype=torch.float64, device=device))
    return _GL_CACHE[key]


def ndtr(x):
    return torch.special.ndtr(x)


def marginal(m, s, region: str, t: float):
    """P(X in region) for X ~ N(m, s^2), s >= 0 (s = 0 -> indicator)."""
    pos = s > 0
    ss = torch.where(pos, s, torch.ones_like(s))
    am = m.abs()
    out_p = ndtr((-t - am) / ss) + ndtr((am - t) / ss)
    in_p = ndtr((t - am) / ss) - ndtr((-t - am) / ss)
    p = out_p if region == "out" else in_p
    ind = (am >= t) if region == "out" else (am < t)
    return torch.where(pos, p, ind.to(m.dtype))


def rect(m1, m2, s11, s22, s12, r1: str, t1: float, r2: str, t2: float, nodes: int = 64):
    """Elementwise P(X1 in r1(t1), X2 in r2(t2)); all inputs broadcastable float64 tensors."""
    m1, m2, s11, s22, s12 = torch.broadcast_tensors(m1, m2, s11, s22, s12)
    dev = m1.device
    # flip signs so means are non-negative (regions are symmetric)
    f1 = m1 < 0
    f2 = m2 < 0
    m1 = m1.abs()
    m2 = m2.abs()
    s12 = torch.where(f1 ^ f2, -s12, s12)
    s11 = s11.clamp_min(0.0)
    s22 = s22.clamp_min(0.0)
    sd1 = torch.sqrt(s11)
    deg = sd1 <= 1e-150
    sd1s = torch.where(deg, torch.ones_like(sd1), sd1)
    # degenerate X1: P = 1[m1 in A1] * P(X2 in A2)
    p_deg = marginal(m1, sd1 * 0, r1, t1) * marginal(m2, torch.sqrt(s22), r2, t2)
    # conditional X2 | X1 = x ~ N(m2 + b (x - m1), cv)
    b = s12 / torch.where(deg, torch.ones_like(s11), s11)
    cv = (s22 - s12 * b).clamp_min(0.0)
    csd = torch.sqrt(cv)
    xg, wg = _gl(nodes, dev)
    K = 10.0
    inf = float("inf")

    def piece(lo, hi):
        """int_{lo}^{hi} phi(x; m1, sd1) P(X2 in A2 | x) dx over the clipped interval [lo, hi] ∩ m1 ± K sd1."""
        a = torch.maximum(lo, m1 - K * sd1s)
        bnd = torch.minimum(hi, m1 + K * sd1s)
        width = (bnd - a).clamp_min(0.0)
        x = a[..., None] + (xg + 1.0) * 0.5 * width[..., None]
        dens = torch.exp(-0.5 * ((x - m1[..., None]) / sd1s[..., None]) ** 2) / (sd1s[..., None] * 2.5066282746310002)
        cm = m2[..., None] + b[..., None] * (x - m1[..., None])
        g = marginal(cm, csd[..., None].expand(x.shape), r2, t2)
        return 0.5 * width * (dens * g * wg).sum(-1)

    ones = torch.ones_like(m1)
    if r1 == "out":
        p = piece(-inf * ones, -t1 * ones) + piece(t1 * ones, inf * ones)
    else:
        p = piece(-t1 * ones, t1 * ones)
    return torch.where(deg, p_deg, p).clamp(0.0, 1.0)


# ------------------------------------------------------------------------------------------------------
# M3.5: sets with two "holes" (own-effect AND marginal-noncentrality power conditions, DESIGN v4 §1)
#
# A coordinate set is one of
#   ("out", lo1, hi1, lo2, hi2):  R minus ([lo1, hi1] ∪ [lo2, hi2])   e.g. {|β| >= τ} ∩ {|β + m| >= τ}
#   ("in2", lo1, hi1, lo2, hi2):  [lo1, hi1] ∪ [lo2, hi2]             e.g. {|β| < w} ∪ {|β + m| < w}
#   ("in",  lo, hi):              [lo, hi]                             e.g. {|β + m| < w}
# with tensor (broadcastable) bounds. Probabilities of the union use P(J1) + P(J2) - P(J1 ∩ J2).
# ------------------------------------------------------------------------------------------------------
def interval_prob(mu, sd, lo, hi):
    """P(lo <= X <= hi) for X ~ N(mu, sd^2); sd = 0 -> indicator; empty interval (lo > hi) -> 0."""
    pos = sd > 0
    ss = torch.where(pos, sd, torch.ones_like(sd))
    p = (ndtr((hi - mu) / ss) - ndtr((lo - mu) / ss)).clamp_min(0.0)
    ind = ((mu >= lo) & (mu <= hi)).to(mu.dtype)
    return torch.where(lo > hi, torch.zeros_like(p), torch.where(pos, p, ind))


def _isect(lo1, hi1, lo2, hi2):
    return torch.maximum(lo1, lo2), torch.minimum(hi1, hi2)


def set_prob(mu, sd, spec):
    """P(X in set) for X ~ N(mu, sd^2), set as described above."""
    if spec[0] == "in":
        return interval_prob(mu, sd, spec[1], spec[2])
    kind, lo1, hi1, lo2, hi2 = spec
    li, hi_ = _isect(lo1, hi1, lo2, hi2)
    u = interval_prob(mu, sd, lo1, hi1) + interval_prob(mu, sd, lo2, hi2) - interval_prob(mu, sd, li, hi_)
    p = u if kind == "in2" else 1.0 - u
    return p.clamp(0.0, 1.0)


def powered_spec(m, tau, marginal_cond: bool = True):
    """{|β| >= τ} ∩ {|β + m| >= τ} (own effect and marginal noncentrality μ = β + m), or {|β| >= τ}."""
    if not marginal_cond:
        m = torch.zeros_like(m)
    return ("out", -tau + 0 * m, tau + 0 * m, -tau - m, tau - m)


def nonweak_spec(m, w, own: bool = True):
    """'power >= 0.2': {|β + m| >= w} (marginal, as the oracle) and, with own = True, also {|β| >= w}
    (own-effect guard, so that near-null effects of the fit cannot count as LD-inflated shared variants)."""
    if own:
        return ("out", -w + 0 * m, w + 0 * m, -w - m, w - m)
    return ("out", -w - m, w - m, -w - m, w - m)


def weak_spec(m, w, own: bool = True):
    """Complement of nonweak_spec: {|β + m| < w} ∪ {|β| < w} (own = True) or {|β + m| < w}."""
    if own:
        return ("in2", -w + 0 * m, w + 0 * m, -w - m, w - m)
    return ("in", -w - m, w - m)


def _spec_slice(spec, sl):
    return (spec[0],) + tuple(v[sl] for v in spec[1:])


def bvn_set_prob(m1, m2, s11, s22, s12, spec1, spec2, nodes: int = 64):
    """P(X1 in spec1, X2 in spec2) for (X1, X2) ~ N((m1, m2), [[s11, s12], [s12, s22]]), elementwise over 1-D
    tensors. Conditioning on X1: P = ∫_{A1} φ(x) P(X2 in A2 | x) dx; for an 'out' set A1 = R minus holes,
    ∫_{A1} = P(X2 in A2) - ∫_{J1} - ∫_{J2} + ∫_{J1 ∩ J2} ('in2': ∫_{J1} + ∫_{J2} - ∫_{J1 ∩ J2}); every remaining integral is over a finite interval
    (clipped to m1 ± 10 sd1) and uses Gauss–Legendre quadrature. s11 = 0 is exact (indicator)."""
    s11 = s11.clamp_min(0.0)
    s22 = s22.clamp_min(0.0)
    sd1 = torch.sqrt(s11)
    deg = sd1 <= 1e-150
    sd1s = torch.where(deg, torch.ones_like(sd1), sd1)
    b = s12 / torch.where(deg, torch.ones_like(s11), s11)
    csd = torch.sqrt((s22 - s12 * b).clamp_min(0.0))
    p2 = set_prob(m2, torch.sqrt(s22), spec2)
    p_deg = set_prob(m1, torch.zeros_like(m1), spec1) * p2
    xg, wg = _gl(nodes, m1.device)
    K = 10.0

    def integral(lo, hi):
        a = torch.maximum(lo, m1 - K * sd1s)
        c = torch.minimum(hi, m1 + K * sd1s)
        width = (c - a).clamp_min(0.0)
        x = a[..., None] + (xg + 1.0) * 0.5 * width[..., None]
        dens = torch.exp(-0.5 * ((x - m1[..., None]) / sd1s[..., None]) ** 2) / (sd1s[..., None] * 2.5066282746310002)
        cm = m2[..., None] + b[..., None] * (x - m1[..., None])
        sp = (spec2[0],) + tuple(v[..., None] for v in spec2[1:])
        g = set_prob(cm, csd[..., None].expand(x.shape), sp)
        return 0.5 * width * (dens * g * wg).sum(-1)

    if spec1[0] == "in":
        p = integral(spec1[1], spec1[2])
    else:
        kind, lo1, hi1, lo2, hi2 = spec1
        li, hi_ = _isect(lo1, hi1, lo2, hi2)
        u = integral(lo1, hi1) + integral(lo2, hi2) - torch.where(li <= hi_, integral(li, hi_), torch.zeros_like(p2))
        p = u if kind == "in2" else p2 - u
    return torch.where(deg, p_deg, p).clamp(0.0, 1.0)
