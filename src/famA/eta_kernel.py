"""Fused Triton kernel for the η update of JointSuSiE (M3.5): one program per locus runs R's Brent_fmin on

    g(x) = -( -½ [p log(1-ρ²) + 2 log|R̃|] - ½ z̃ᵀ(V^{-1} ⊗ R̃^{-1}) z̃ - p log 2π  -  ½ [(1-η) A_R + η A_I] ),  η = e^x,

over x in [log η_lo, log η_hi] (tol sqrt(eps)), with log|R̃| and the quadratic form evaluated in the eigenbasis of
R (eig_lam, eig_q, eig_mask, plus the null-space count and mass), then applies the keep-old rule
(η_new only if g(x_new) <= g(log η_old)). This is exactly `_JointBucket.update_eta` (pure torch, famG.brent)
in one launch instead of ~2,000 small kernels per update; tests/famA/test_joint_susie.py checks equality.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LOG2PI = math.log(2 * math.pi)


@triton.jit
def _g(lam_ptr, q_ptr, m_ptr, row, K, x, qnull, nnull, pr, rho, A_R, A_I, BLOCK: tl.constexpr):
    e = tl.exp(x)
    ld = tl.zeros((), dtype=tl.float64)
    qd = tl.zeros((), dtype=tl.float64)
    for start in range(0, K, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        msk = offs < K
        lam = tl.load(lam_ptr + row + offs, mask=msk, other=0.0)
        q = tl.load(q_ptr + row + offs, mask=msk, other=0.0)
        mm = tl.load(m_ptr + row + offs, mask=msk, other=0)
        use = msk & (mm != 0)
        lt = (1.0 - e) * lam + e
        lt = tl.where(use, lt, 1.0)
        ld += tl.sum(tl.where(use, tl.log(lt), 0.0), 0)
        qd += tl.sum(tl.where(use, q / lt, 0.0), 0)
    logdetR = ld + nnull * tl.log(e)
    quad = qd + qnull / e
    terms = -0.5 * (pr * tl.log(1.0 - rho * rho) + 2.0 * logdetR) - 0.5 * quad - pr * 1.8378770664093453
    f = -(terms - 0.5 * ((1.0 - e) * A_R + e * A_I))
    ok = (f == f) & (f != float("inf")) & (f != -float("inf"))
    return tl.where(ok, f, 1.7976931348623157e308)


@triton.jit
def _eta_kernel(lam_ptr, q_ptr, m_ptr, qnull_ptr, nnull_ptr, p_ptr, rho_ptr, ar_ptr, ai_ptr, old_ptr, act_ptr, out_ptr,
                K, lo, hi, TOL: tl.constexpr, MAX_IT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid.to(tl.int64) * K
    qnull = tl.load(qnull_ptr + pid)
    nnull = tl.load(nnull_ptr + pid)
    pr = tl.load(p_ptr + pid)
    rho = tl.load(rho_ptr + pid)
    A_R = tl.load(ar_ptr + pid)
    A_I = tl.load(ai_ptr + pid)
    old = tl.load(old_ptr + pid)
    act = tl.load(act_ptr + pid)
    if act != 0:
        c = 0.3819660112501051
        eps = 1.4901161193847656e-08
        tol3 = TOL / 3.0
        a = lo + 0.0 * qnull
        b = hi + 0.0 * qnull
        v = a + c * (b - a)
        w = v
        x = v
        d = tl.zeros((), dtype=tl.float64)
        e = tl.zeros((), dtype=tl.float64)
        fx = _g(lam_ptr, q_ptr, m_ptr, row, K, x, qnull, nnull, pr, rho, A_R, A_I, BLOCK)
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
                    fu = _g(lam_ptr, q_ptr, m_ptr, row, K, u, qnull, nnull, pr, rho, A_R, A_I, BLOCK)
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
        f_new = _g(lam_ptr, q_ptr, m_ptr, row, K, x, qnull, nnull, pr, rho, A_R, A_I, BLOCK)
        f_old = _g(lam_ptr, q_ptr, m_ptr, row, K, tl.log(old), qnull, nnull, pr, rho, A_R, A_I, BLOCK)
        res = old
        if f_new <= f_old:
            res = tl.exp(x)
        tl.store(out_ptr + pid, res)
    else:
        tl.store(out_ptr + pid, old)


def eta_update(lam, q, mask, qnull, nnull, p_real, rho, A_R, A_I, eta_old, active, bounds, block: int = 512):
    """All (B,K) / (B,) float64 CUDA tensors (mask bool, active bool); eta_old already clamped to bounds[0].
    Returns the new η (B,) (old value for inactive loci)."""
    B, K = lam.shape
    out = torch.empty(B, dtype=torch.float64, device=lam.device)
    _eta_kernel[(B,)](lam.contiguous(), q.contiguous(), mask.to(torch.int8).contiguous(), qnull.contiguous(),
                      nnull.contiguous(), p_real.to(torch.float64).contiguous(), rho.contiguous(), A_R.contiguous(),
                      A_I.contiguous(), eta_old.contiguous(), active.to(torch.int32).contiguous(), out, K,
                      math.log(bounds[0]), math.log(bounds[1]), TOL=math.sqrt(2.220446049250313e-16), MAX_IT=1000,
                      BLOCK=block)
    return out
