"""Batched port of R's Brent_fmin (src/library/stats/src/optimize.c), used by optimize()/optim(method="Brent").

susieR's optimize_prior_variance calls optim(..., method="Brent", lower=-30, upper=15), which is
optimize(fn, lower, upper, tol = sqrt(.Machine$double.eps)); the starting value `par` is ignored.
This module reproduces the exact sequence of evaluation points of the C code, vectorised over a
batch of independent 1-D problems (torch, float64). Non-finite objective values are replaced by
DBL_MAX, as R's fcn1 wrapper does.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

DBL_EPS = 2.220446049250313e-16
DBL_MAX = 1.7976931348623157e308
R_OPTIM_RELTOL = math.sqrt(DBL_EPS)  # optim()'s default reltol, passed as optimize(tol=)


def _clean(f: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(f), f, torch.full_like(f, DBL_MAX))


def brent_fmin(f: Callable[[torch.Tensor], torch.Tensor], ax: float | torch.Tensor, bx: float | torch.Tensor,
               batch: int, device, tol: float = R_OPTIM_RELTOL, max_iter: int = 1000) -> torch.Tensor:
    """Minimise f over [ax, bx] for each of `batch` problems. f maps a (batch,) float64 tensor of
    abscissae to (batch,) objective values. Returns the (batch,) minimisers (R's `x`)."""
    dt = torch.float64
    c = (3.0 - math.sqrt(5.0)) * 0.5
    eps = math.sqrt(DBL_EPS)
    a = torch.as_tensor(ax, dtype=dt, device=device).expand(batch).clone()
    b = torch.as_tensor(bx, dtype=dt, device=device).expand(batch).clone()
    v = a + c * (b - a)
    w = v.clone()
    x = v.clone()
    d = torch.zeros(batch, dtype=dt, device=device)
    e = torch.zeros(batch, dtype=dt, device=device)
    fx = _clean(f(x))
    fv = fx.clone()
    fw = fx.clone()
    tol3 = tol / 3.0
    active = torch.ones(batch, dtype=torch.bool, device=device)
    for _ in range(max_iter):
        xm = (a + b) * 0.5
        tol1 = eps * x.abs() + tol3
        t2 = tol1 * 2.0
        stop = (x - xm).abs() <= t2 - (b - a) * 0.5
        active = active & ~stop
        if not bool(active.any()):
            break
        # parabola fit
        fit = e.abs() > tol1
        r_ = (x - w) * (fx - fv)
        q_ = (x - v) * (fx - fw)
        p_ = (x - v) * q_ - (x - w) * r_
        q_ = (q_ - r_) * 2.0
        p_ = torch.where(q_ > 0.0, -p_, p_)
        q_ = q_.abs()
        p = torch.where(fit, p_, torch.zeros_like(p_))
        q = torch.where(fit, q_, torch.zeros_like(q_))
        r = torch.where(fit, e, torch.zeros_like(e))
        e = torch.where(fit, d, e)
        golden = (p.abs() >= (q * 0.5 * r).abs()) | (p <= q * (a - x)) | (p >= q * (b - x))
        # golden-section step
        e_g = torch.where(x < xm, b - x, a - x)
        d_g = c * e_g
        # parabolic step
        d_p = p / torch.where(golden, torch.ones_like(q), q)
        u_p = x + d_p
        near = ((u_p - a) < t2) | ((b - u_p) < t2)
        d_near = torch.where(x >= xm, -tol1, tol1)
        d_p = torch.where(near, d_near, d_p)
        e = torch.where(golden, e_g, e)
        d = torch.where(golden, d_g, d_p)
        u = torch.where(d.abs() >= tol1, x + d, torch.where(d > 0.0, x + tol1, x - tol1))
        fu = _clean(f(u))
        # updates (only for active problems)
        le = fu <= fx
        u_lt_x = u < x
        # branch fu <= fx
        b1 = torch.where(u_lt_x, x, b)
        a1 = torch.where(u_lt_x, a, x)
        # branch fu > fx
        a2 = torch.where(u_lt_x, u, a)
        b2 = torch.where(u_lt_x, b, u)
        c1 = (fu <= fw) | (w == x)
        c2 = (~c1) & ((fu <= fv) | (v == x) | (v == w))
        nv = torch.where(le, w, torch.where(c1, w, torch.where(c2, u, v)))
        nfv = torch.where(le, fw, torch.where(c1, fw, torch.where(c2, fu, fv)))
        nw = torch.where(le, x, torch.where(c1, u, w))
        nfw = torch.where(le, fx, torch.where(c1, fu, fw))
        nx = torch.where(le, u, x)
        nfx = torch.where(le, fu, fx)
        na = torch.where(le, a1, a2)
        nb = torch.where(le, b1, b2)
        act = active
        a = torch.where(act, na, a)
        b = torch.where(act, nb, b)
        v = torch.where(act, nv, v)
        fv = torch.where(act, nfv, fv)
        w = torch.where(act, nw, w)
        fw = torch.where(act, nfw, fw)
        x = torch.where(act, nx, x)
        fx = torch.where(act, nfx, fx)
        # d, e for inactive problems are irrelevant (they never update again)
    return x
