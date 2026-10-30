"""Fused Triton kernel for susieR's per-effect prior-variance optimisation (one program per locus).

Does, for every locus in a batch and entirely on the GPU (float64):
  lV* = Brent_fmin(-loglik(exp(lV)), -30, 15, tol = sqrt(eps))   (exact port of R's Brent_fmin)
  if -loglik(exp(lV*)) > -loglik(V_old): lV* = log(V_old)         (susieR keep-old rule)
  V = exp(lV*); if loglik(0) + thr >= loglik(V): V = 0             (susieR null check)
loglik(V) = log sum_j exp(lbf_j(V) + lpw_j), lbf_j = dnorm(bhat_j; 0, sqrt(V + s2_j)) - dnorm(bhat_j; 0, sqrt(s2_j))
(log-scale densities). The log-sum is accumulated online over blocks of variables (rounding differs from
R's two-pass max/sum by ~1e-16 relative). Replaces ~40 x ~60 small kernel launches per SER by one launch.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

DBL_MAX = 1.7976931348623157e308


@triton.jit
def _loglik(b_ptr, s_ptr, w_ptr, row, P, V, BLOCK: tl.constexpr):
    C: tl.constexpr = 0.918938533204672741780329736406
    m = -float("inf")
    m = m.to(tl.float64)
    s = tl.zeros((), dtype=tl.float64)
    for start in range(0, P, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        msk = offs < P
        bh = tl.load(b_ptr + row + offs, mask=msk, other=0.0)
        s2 = tl.load(s_ptr + row + offs, mask=msk, other=1.0)
        lw = tl.load(w_ptr + row + offs, mask=msk, other=-float("inf"))
        sd1 = tl.sqrt(V + s2)
        x1 = bh / sd1
        sd0 = tl.sqrt(s2)
        x0 = bh / sd0
        lbf = -(C + 0.5 * x1 * x1 + tl.log(sd1)) + (C + 0.5 * x0 * x0 + tl.log(sd0))
        lpo = lbf + lw
        bm = tl.max(lpo, 0)
        nm = tl.maximum(m, bm)
        nms = tl.where(nm == -float("inf"), 0.0, nm)
        s = s * tl.exp(m - nms) + tl.sum(tl.exp(lpo - nms), 0)
        m = nm
    return m + tl.log(s)


@triton.jit
def _negll_clean(b_ptr, s_ptr, w_ptr, row, P, lV, BLOCK: tl.constexpr):
    f = -_loglik(b_ptr, s_ptr, w_ptr, row, P, tl.exp(lV), BLOCK)
    ok = (f == f) & (f != float("inf")) & (f != -float("inf"))
    return tl.where(ok, f, 1.7976931348623157e308)


@triton.jit
def _optim_V_kernel(b_ptr, s_ptr, w_ptr, vold_ptr, vout_ptr, act_ptr, P, stride, thr,
                    TOL: tl.constexpr, MAX_IT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid.to(tl.int64) * stride
    act = tl.load(act_ptr + pid)
    Vold = tl.load(vold_ptr + pid)
    if act != 0:
        c = 0.3819660112501051  # (3 - sqrt(5)) / 2
        eps = 1.4901161193847656e-08  # sqrt(DBL_EPSILON)
        tol3 = TOL / 3.0
        a = tl.full((), -30.0, tl.float64)
        b = tl.full((), 15.0, tl.float64)
        v = a + c * (b - a)
        w = v
        x = v
        d = tl.zeros((), dtype=tl.float64)
        e = tl.zeros((), dtype=tl.float64)
        fx = _negll_clean(b_ptr, s_ptr, w_ptr, row, P, x, BLOCK)
        fv = fx
        fw = fx
        done = 0
        for it in range(MAX_IT):
            if done == 0:
                xm = (a + b) * 0.5
                tol1 = eps * tl.abs(x) + tol3
                t2 = tol1 * 2.0
                if tl.abs(x - xm) <= t2 - (b - a) * 0.5:
                    done = 1
                else:
                    p = tl.zeros((), dtype=tl.float64)
                    q = tl.zeros((), dtype=tl.float64)
                    r = tl.zeros((), dtype=tl.float64)
                    if tl.abs(e) > tol1:
                        r = (x - w) * (fx - fv)
                        q = (x - v) * (fx - fw)
                        p = (x - v) * q - (x - w) * r
                        q = (q - r) * 2.0
                        if q > 0.0:
                            p = -p
                        else:
                            q = -q
                        r = e
                        e = d
                    if (tl.abs(p) >= tl.abs(q * 0.5 * r)) | (p <= q * (a - x)) | (p >= q * (b - x)):
                        if x < xm:
                            e = b - x
                        else:
                            e = a - x
                        d = c * e
                    else:
                        d = p / q
                        u0 = x + d
                        if ((u0 - a) < t2) | ((b - u0) < t2):
                            d = tol1
                            if x >= xm:
                                d = -d
                    u = x
                    if tl.abs(d) >= tol1:
                        u = x + d
                    elif d > 0.0:
                        u = x + tol1
                    else:
                        u = x - tol1
                    fu = _negll_clean(b_ptr, s_ptr, w_ptr, row, P, u, BLOCK)
                    if fu <= fx:
                        if u < x:
                            b = x
                        else:
                            a = x
                        v = w
                        fv = fw
                        w = x
                        fw = fx
                        x = u
                        fx = fu
                    else:
                        if u < x:
                            a = u
                        else:
                            b = u
                        if (fu <= fw) | (w == x):
                            v = w
                            fv = fw
                            w = u
                            fw = fu
                        elif (fu <= fv) | (v == x) | (v == w):
                            v = u
                            fv = fu
        lV = x
        f_opt = -_loglik(b_ptr, s_ptr, w_ptr, row, P, tl.exp(lV), BLOCK)
        lVold = tl.log(Vold)
        f_old = -_loglik(b_ptr, s_ptr, w_ptr, row, P, tl.exp(lVold), BLOCK)
        if f_opt > f_old:
            lV = lVold
        Vn = tl.exp(lV)
        l0 = _loglik(b_ptr, s_ptr, w_ptr, row, P, Vn * 0.0, BLOCK)
        l1 = _loglik(b_ptr, s_ptr, w_ptr, row, P, Vn, BLOCK)
        if l0 + thr >= l1:
            Vn = Vn * 0.0
        tl.store(vout_ptr + pid, Vn)
    else:
        tl.store(vout_ptr + pid, Vold)


@triton.jit
def _null_check_kernel(b_ptr, s_ptr, w_ptr, v_ptr, vout_ptr, P, stride, thr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid.to(tl.int64) * stride
    V = tl.load(v_ptr + pid)
    l0 = _loglik(b_ptr, s_ptr, w_ptr, row, P, V * 0.0, BLOCK)
    l1 = _loglik(b_ptr, s_ptr, w_ptr, row, P, V, BLOCK)
    tl.store(vout_ptr + pid, tl.where(l0 + thr >= l1, V * 0.0, V))


def optim_prior_variance(betahat: torch.Tensor, shat2: torch.Tensor, lpw: torch.Tensor, Vold: torch.Tensor,
                         active: torch.Tensor | None = None, check_null_threshold: float = 0.0,
                         block: int = 512) -> torch.Tensor:
    """(B,P) float64 inputs on CUDA -> (B,) optimised prior variances (susieR 0.14.2 'optim' path)."""
    B, P = betahat.shape
    bh = betahat.contiguous(); s2 = shat2.contiguous(); lw = lpw.contiguous()
    act = (torch.ones(B, dtype=torch.int32, device=bh.device) if active is None else active.to(torch.int32)).contiguous()
    out = torch.empty(B, dtype=torch.float64, device=bh.device)
    _optim_V_kernel[(B,)](bh, s2, lw, Vold.contiguous(), out, act, P, P, float(check_null_threshold),
                          TOL=math.sqrt(2.220446049250313e-16), MAX_IT=200, BLOCK=block)
    return out


def null_check(betahat, shat2, lpw, V, check_null_threshold=0.0, block=512):
    B, P = betahat.shape
    out = torch.empty(B, dtype=torch.float64, device=betahat.device)
    _null_check_kernel[(B,)](betahat.contiguous(), shat2.contiguous(), lpw.contiguous(), V.contiguous(), out, P, P,
                             float(check_null_threshold), BLOCK=block)
    return out
