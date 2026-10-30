"""Fused Triton kernel: per-effect empirical-Bayes update of one trait's prior scale in JointSuSiE.

For each locus (one program), with the other trait's scale fixed, it runs the exact R Brent_fmin on
f(x) = -loglik(σ²_t = exp(x)) over x in [-30, 15] (tol sqrt(eps)), applies susieR's keep-old rule and the
null check (σ²_t <- 0 if loglik(0) + thr >= loglik(σ²_t)), all in float64. loglik is the single-effect
log marginal likelihood log Σ_j (1/p) [π0 + Σ_s π_s BF_js] of the 4-state bivariate model (see
famA/joint_susie.py); it is accumulated online over blocks of variables. The part that does not depend on σ²_t
(null state and the other trait's single state) is computed once per Brent run, so each evaluation needs two
exp per variable (M3.5 speed-up; identical up to fp64 rounding). Equivalent to the pure-torch path
(JointConfig.use_triton = False) up to floating-point rounding (tests/famA/test_joint_susie.py).
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _single_lbf_coef(s, rho):
    """Single-trait state log BF = A + B u², u = x - ρ y (trait 1) or y - ρ x (trait 2) (Sherman–Morrison form of
    log N(bhat; 0, V + s e e') - log N(bhat; 0, V))."""
    dS = 1.0 - rho * rho
    A = -0.5 * tl.log(1.0 + s / dS)
    B = 0.5 * s / (dS * (dS + s))
    return A, B


@triton.jit
def _lse_acc(m, acc, v, FAST: tl.constexpr):
    """Online log-sum-exp: fold the block of values v into (m, acc) (sum = exp(m) * acc). FAST: the per-element
    exp(v - max) <= 1 is evaluated in float32 (relative error ~1e-7 per term) and accumulated in float64."""
    bm = tl.max(v, 0)
    nm = tl.maximum(m, bm)
    nms = tl.where(nm == -float("inf"), 0.0, nm)
    if FAST:
        ev = tl.exp((v - nms).to(tl.float32)).to(tl.float64)
    else:
        ev = tl.exp(v - nms)
    acc = acc * tl.exp(m - nms) + tl.sum(ev, 0)
    return nm, acc


@triton.jit
def _lse2(a, b):
    m = tl.maximum(a, b)
    ms = tl.where(m == -float("inf"), 0.0, m)
    return ms + tl.log(tl.exp(a - ms) + tl.exp(b - ms))


@triton.jit
def _const_part(r1_ptr, r2_ptr, w_ptr, row, P, other, rho, lp0, lpo, T: tl.constexpr, BLOCK: tl.constexpr,
                FAST: tl.constexpr):
    """log Σ_j w_j (π0 + π_o BF_o,j): the part of the SER likelihood that does not depend on σ²_T
    (o = the other trait's single-trait state, with its scale fixed at `other`)."""
    A, Bc = _single_lbf_coef(other, rho)
    m = (-float("inf")) * tl.full((), 1.0, tl.float64)
    acc = tl.zeros((), dtype=tl.float64)
    for start in range(0, P, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        msk = offs < P
        x = tl.load(r1_ptr + row + offs, mask=msk, other=0.0)
        y = tl.load(r2_ptr + row + offs, mask=msk, other=0.0)
        lw = tl.load(w_ptr + row + offs, mask=msk, other=-float("inf"))
        if T == 0:
            u = y - rho * x
        else:
            u = x - rho * y
        lo = A + Bc * u * u
        v = lw + _lse2(lp0 + 0.0 * x, lpo + lo)
        m, acc = _lse_acc(m, acc, v, FAST)
    return m + tl.log(acc)


@triton.jit
def _ll_coord(r1_ptr, r2_ptr, w_ptr, row, P, val, other, rho, rho_u, lpc, lpt, lp3, logC,
              T: tl.constexpr, BLOCK: tl.constexpr, FAST: tl.constexpr):
    """SER log-likelihood log Σ_j w_j [π0 + Σ_s π_s BF_s,j] at σ²_T = val, σ²_other = other: the constant part
    logC (precomputed) plus two online log-sum-exp accumulators (single state T, shared state), i.e. two exp per
    variable instead of five exp and one log."""
    dS = 1.0 - rho * rho
    if T == 0:
        s1 = val
        s2 = other
    else:
        s1 = other
        s2 = val
    A, Bc = _single_lbf_coef(val, rho)
    c = rho_u * tl.sqrt(s1 * s2)
    a11 = 1.0 + s1
    b22 = 1.0 + s2
    m12 = rho + c
    det3 = a11 * b22 - m12 * m12
    ld3 = -0.5 * tl.log(det3 / dS)
    mt = (-float("inf")) * tl.full((), 1.0, tl.float64)
    at = tl.zeros((), dtype=tl.float64)
    m3 = (-float("inf")) * tl.full((), 1.0, tl.float64)
    a3 = tl.zeros((), dtype=tl.float64)
    for start in range(0, P, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        msk = offs < P
        x = tl.load(r1_ptr + row + offs, mask=msk, other=0.0)
        y = tl.load(r2_ptr + row + offs, mask=msk, other=0.0)
        lw = tl.load(w_ptr + row + offs, mask=msk, other=-float("inf"))
        if T == 0:
            u = x - rho * y
        else:
            u = y - rho * x
        lt = A + Bc * u * u
        xx = x * x
        xy = x * y
        yy = y * y
        qS = (xx - 2.0 * rho * xy + yy) / dS
        q3 = (b22 * xx - 2.0 * m12 * xy + a11 * yy) / det3
        l3 = ld3 - 0.5 * (q3 - qS)
        mt, at = _lse_acc(mt, at, lw + lpt + lt, FAST)
        m3, a3 = _lse_acc(m3, a3, lw + lp3 + l3, FAST)
    return _lse2(_lse2(logC, mt + tl.log(at)), m3 + tl.log(a3))


@triton.jit
def _negll(r1_ptr, r2_ptr, w_ptr, row, P, lx, other, rho, rho_u, lpc, lpt, lp3, logC, T: tl.constexpr, BLOCK: tl.constexpr,
           FAST: tl.constexpr):
    f = -_ll_coord(r1_ptr, r2_ptr, w_ptr, row, P, tl.exp(lx), other, rho, rho_u, lpc, lpt, lp3, logC, T, BLOCK, FAST)
    ok = (f == f) & (f != float("inf")) & (f != -float("inf"))
    return tl.where(ok, f, 1.7976931348623157e308)


@triton.jit
def _eb_kernel(r1_ptr, r2_ptr, w_ptr, s2_ptr, rho_ptr, lp_ptr, act_ptr, out_ptr, P, stride, rho_u, thr,
               T: tl.constexpr, TOL: tl.constexpr, MAX_IT: tl.constexpr, BLOCK: tl.constexpr, FAST: tl.constexpr):
    pid = tl.program_id(0)
    row = pid.to(tl.int64) * stride
    old = tl.load(s2_ptr + pid * 2 + T)
    other = tl.load(s2_ptr + pid * 2 + (1 - T))
    rho = tl.load(rho_ptr + pid)
    lp0 = tl.load(lp_ptr + pid * 4 + 0)
    lp1 = tl.load(lp_ptr + pid * 4 + 1)
    lp2 = tl.load(lp_ptr + pid * 4 + 2)
    lp3 = tl.load(lp_ptr + pid * 4 + 3)
    act = tl.load(act_ptr + pid)
    if T == 0:
        lpt = lp1
        lpo = lp2
    else:
        lpt = lp2
        lpo = lp1
    if act != 0:
        logC = _const_part(r1_ptr, r2_ptr, w_ptr, row, P, other, rho, lp0, lpo, T, BLOCK, FAST)
        c = 0.3819660112501051
        eps = 1.4901161193847656e-08
        tol3 = TOL / 3.0
        a = tl.full((), -30.0, tl.float64)
        b = tl.full((), 15.0, tl.float64)
        v = a + c * (b - a)
        w = v
        x = v
        d = tl.zeros((), dtype=tl.float64)
        e = tl.zeros((), dtype=tl.float64)
        fx = _negll(r1_ptr, r2_ptr, w_ptr, row, P, x, other, rho, rho_u, lp0, lpt, lp3, logC, T, BLOCK, FAST)
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
                    fu = _negll(r1_ptr, r2_ptr, w_ptr, row, P, u, other, rho, rho_u, lp0, lpt, lp3, logC, T, BLOCK, FAST)
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
        lx = x
        f_opt = -_ll_coord(r1_ptr, r2_ptr, w_ptr, row, P, tl.exp(lx), other, rho, rho_u, lp0, lpt, lp3, logC, T, BLOCK, FAST)
        lold = tl.log(old)
        f_old = -_ll_coord(r1_ptr, r2_ptr, w_ptr, row, P, tl.exp(lold), other, rho, rho_u, lp0, lpt, lp3, logC, T, BLOCK, FAST)
        if f_opt > f_old:
            lx = lold
        val = tl.exp(lx)
        l0 = _ll_coord(r1_ptr, r2_ptr, w_ptr, row, P, val * 0.0, other, rho, rho_u, lp0, lpt, lp3, logC, T, BLOCK, FAST)
        l1 = _ll_coord(r1_ptr, r2_ptr, w_ptr, row, P, val, other, rho, rho_u, lp0, lpt, lp3, logC, T, BLOCK, FAST)
        if l0 + thr >= l1:
            val = val * 0.0
        tl.store(out_ptr + pid, val)
    else:
        tl.store(out_ptr + pid, old)


def eb_coordinate(r: torch.Tensor, lpw: torch.Tensor, s2: torch.Tensor, rho: torch.Tensor, logpi: torch.Tensor,
                  active: torch.Tensor, t: int, rho_u: float = 0.0, thr: float = 0.0, block: int = 512,
                  fast_exp: bool = False, num_warps: int = 4) -> torch.Tensor:
    """r (B,2,P), lpw (B,P), s2 (B,2), rho (B,), logpi (B,4), active (B,) -> new σ²_t (B,).
    block 512 / 4 warps was fastest on the RTX 4090 at p = 2000 (M3.5 timing). fast_exp: float32 exp of the
    per-variable terms (see _lse_acc); off by default."""
    B, _, P = r.shape
    r1 = r[:, 0].contiguous()
    r2 = r[:, 1].contiguous()
    out = torch.empty(B, dtype=torch.float64, device=r.device)
    _eb_kernel[(B,)](r1, r2, lpw.contiguous(), s2.contiguous(), rho.contiguous(), logpi.contiguous(),
                     active.to(torch.int32).contiguous(), out, P, P, float(rho_u), float(thr),
                     T=t, TOL=math.sqrt(2.220446049250313e-16), MAX_IT=200, BLOCK=block, FAST=bool(fast_exp),
                     num_warps=num_warps)
    return out
