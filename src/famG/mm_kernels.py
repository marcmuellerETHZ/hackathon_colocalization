"""Fused Triton kernels for G-mm's three per-row 1-D optimisations (one program per fit row, float64).

Each kernel runs an exact port of R's Brent_fmin (src/library/stats/src/optimize.c; same as famG.brent.brent_fmin
and famG.ser_kernel) on a row-specific interval, with the objective accumulated over the row's variables in blocks:
  MODE 0  SER prior variance  [single_effect_regression.R optimize_scalar_prior_variance, optim(method = "Brent")]
          f(lV) = -log sum_j exp(lbf_j(exp lV) + lpw_j), lbf = gaussian_ser_lbf; interval [-30, 15], tol sqrt(eps)
  MODE 1  eb_mix signal variance  [rss_mismatch.R estimate_eb_mix_vsig]
          f(lV) = -sum_j logsumexp(log0_j, log(pi) + dnorm(rz_j, 0, sqrt(exp lV), log = TRUE)); tol 1e-6
  MODE 2  lambda_bias MLE  [rss_mismatch.R estimate_lambda_bias]
          f(lam) = 0.5 sum_{j in keep} log(base_j + lam s_j) + r2_j / (base_j + lam s_j); interval [0, upper], tol 1e-8
Non-finite objective values are replaced by DBL_MAX (R's fcn1). Only the minimiser x is returned; the callers
re-evaluate the objective in torch exactly as R does (keep-old rule, null check, nll(0) check).
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _obj(p0, p1, p2, p3, row, n, sc, x, MODE: tl.constexpr, BLOCK: tl.constexpr):
    C: tl.constexpr = 0.918938533204672741780329736406
    if MODE == 0:
        V = tl.exp(x)
        m = -float("inf")
        m = m.to(tl.float64)
        s = tl.zeros((), dtype=tl.float64)
        for start in range(0, n, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            msk = offs < n
            bh = tl.load(p0 + row + offs, mask=msk, other=0.0)
            s2 = tl.load(p1 + row + offs, mask=msk, other=1.0)
            lw = tl.load(p2 + row + offs, mask=msk, other=-float("inf"))
            s2 = tl.maximum(s2, 2.220446049250313e-16)
            lbf = -0.5 * tl.log(1.0 + V / s2) + 0.5 * bh * bh * V / (s2 * (V + s2))
            lpo = lbf + lw
            bm = tl.max(lpo, 0)
            nm = tl.maximum(m, bm)
            nms = tl.where(nm == -float("inf"), 0.0, nm)
            s = s * tl.exp(m - nms) + tl.sum(tl.exp(lpo - nms), 0)
            m = nm
        f = -(m + tl.log(s))
    elif MODE == 1:
        sd = tl.sqrt(tl.exp(x))
        lsd = tl.log(sd)
        acc = tl.zeros((BLOCK,), dtype=tl.float64)
        for start in range(0, n, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            msk = offs < n
            rz = tl.load(p0 + row + offs, mask=msk, other=0.0)
            l0 = tl.load(p1 + row + offs, mask=msk, other=0.0)
            xs = rz / sd
            l1 = sc - (C + 0.5 * xs * xs + lsd)
            mm = tl.maximum(l0, l1)
            # R: m + log(exp(log0 - m) + exp(log1 - m)); one of the two terms is exp(0) = 1 exactly
            t = mm + tl.log(1.0 + tl.exp(tl.minimum(l0, l1) - mm))
            acc += tl.where(msk, t, 0.0)
        f = -tl.sum(acc, 0)
    else:
        acc = tl.zeros((BLOCK,), dtype=tl.float64)
        for start in range(0, n, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            msk = offs < n
            r2 = tl.load(p0 + row + offs, mask=msk, other=0.0)
            sv = tl.load(p1 + row + offs, mask=msk, other=0.0)
            bs = tl.load(p2 + row + offs, mask=msk, other=1.0)
            kp = tl.load(p3 + row + offs, mask=msk, other=0.0)
            tau = bs + x * sv
            t = tl.log(tau) + r2 / tau
            acc += tl.where(msk & (kp != 0.0), t, 0.0)
        f = 0.5 * tl.sum(acc, 0)
    ok = (f == f) & (f != float("inf")) & (f != -float("inf"))
    return tl.where(ok, f, 1.7976931348623157e308)


@triton.jit
def _brent_kernel(p0, p1, p2, p3, a_ptr, b_ptr, sc_ptr, n_ptr, out_ptr, stride, tol,
                  MODE: tl.constexpr, MAX_IT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid.to(tl.int64) * stride
    n = tl.load(n_ptr + pid)
    sc = tl.load(sc_ptr + pid)
    c = 0.3819660112501051  # (3 - sqrt(5)) / 2
    eps = 1.4901161193847656e-08  # sqrt(DBL_EPSILON)
    tol3 = tol / 3.0
    a = tl.load(a_ptr + pid)
    b = tl.load(b_ptr + pid)
    if n == 0:                      # skipped row (e.g. lambda == 0: no eb_mix gate in susieR)
        tl.store(out_ptr + pid, a)
        return
    v = a + c * (b - a)
    w = v
    x = v
    d = tl.zeros((), dtype=tl.float64)
    e = tl.zeros((), dtype=tl.float64)
    fx = _obj(p0, p1, p2, p3, row, n, sc, x, MODE, BLOCK)
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
                fu = _obj(p0, p1, p2, p3, row, n, sc, u, MODE, BLOCK)
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
    tl.store(out_ptr + pid, x)


BLOCK = 1024
NUM_WARPS = 8


def _launch(mode, arrs, a, b, sc, n, tol, block=None):
    block = block or BLOCK
    rows, P = arrs[0].shape
    arrs = [x.contiguous() for x in arrs] + [arrs[0]] * (4 - len(arrs))
    out = torch.empty(rows, dtype=torch.float64, device=arrs[0].device)
    if rows == 0:
        return out
    _brent_kernel[(rows,)](arrs[0], arrs[1], arrs[2], arrs[3], a.contiguous().to(torch.float64),
                           b.contiguous().to(torch.float64), sc.contiguous().to(torch.float64),
                           n.contiguous().to(torch.int32), out, P, float(tol),
                           MODE=mode, MAX_IT=300, BLOCK=block, num_warps=NUM_WARPS)
    return out


def brent_ser(betahat, shat2, lpw, n=None):
    rows, P = betahat.shape
    dev = betahat.device
    a = torch.full((rows,), -30.0, dtype=torch.float64, device=dev)
    b = torch.full((rows,), 15.0, dtype=torch.float64, device=dev)
    n = torch.full((rows,), P, dtype=torch.int32, device=dev) if n is None else n
    return _launch(0, [betahat, shat2, lpw], a, b, torch.zeros(rows, dtype=torch.float64, device=dev), n,
                   math.sqrt(2.220446049250313e-16))


def brent_vsig(rz, log0, log_pi, n, lo, hi):
    return _launch(1, [rz, log0], lo, hi, log_pi, n, 1e-6)


def brent_lambda(r2, s, base, keep, upper, n=None):
    rows, P = r2.shape
    dev = r2.device
    n = torch.full((rows,), P, dtype=torch.int32, device=dev) if n is None else n
    return _launch(2, [r2, s, base, keep.to(torch.float64)], torch.zeros_like(upper), upper,
                   torch.zeros(rows, dtype=torch.float64, device=dev), n, 1e-8)


@triton.jit
def _loglik3_kernel(p0, p1, p2, v_ptr, out_ptr, stride, n_ptr, BLOCK: tl.constexpr):
    """loglik(V_k) = log sum_j exp(lbf_j(V_k) + lpw_j) for the 3 values V_k of a row in one pass."""
    pid = tl.program_id(0)
    row = pid.to(tl.int64) * stride
    n = tl.load(n_ptr + pid)
    V0 = tl.load(v_ptr + pid * 3)
    V1 = tl.load(v_ptr + pid * 3 + 1)
    V2 = tl.load(v_ptr + pid * 3 + 2)
    ninf = -float("inf")
    m0 = tl.full((), ninf, tl.float64)
    m1 = tl.full((), ninf, tl.float64)
    m2 = tl.full((), ninf, tl.float64)
    s0 = tl.zeros((), dtype=tl.float64)
    s1 = tl.zeros((), dtype=tl.float64)
    s2_ = tl.zeros((), dtype=tl.float64)
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        msk = offs < n
        bh = tl.load(p0 + row + offs, mask=msk, other=0.0)
        s2 = tl.load(p1 + row + offs, mask=msk, other=1.0)
        lw = tl.load(p2 + row + offs, mask=msk, other=-float("inf"))
        s2 = tl.maximum(s2, 2.220446049250313e-16)
        b2 = 0.5 * bh * bh
        l0 = -0.5 * tl.log(1.0 + V0 / s2) + b2 * V0 / (s2 * (V0 + s2)) + lw
        l1 = -0.5 * tl.log(1.0 + V1 / s2) + b2 * V1 / (s2 * (V1 + s2)) + lw
        l2 = -0.5 * tl.log(1.0 + V2 / s2) + b2 * V2 / (s2 * (V2 + s2)) + lw
        n0 = tl.maximum(m0, tl.max(l0, 0))
        n1 = tl.maximum(m1, tl.max(l1, 0))
        n2 = tl.maximum(m2, tl.max(l2, 0))
        z0 = tl.where(n0 == ninf, 0.0, n0)
        z1 = tl.where(n1 == ninf, 0.0, n1)
        z2 = tl.where(n2 == ninf, 0.0, n2)
        s0 = s0 * tl.exp(m0 - z0) + tl.sum(tl.exp(l0 - z0), 0)
        s1 = s1 * tl.exp(m1 - z1) + tl.sum(tl.exp(l1 - z1), 0)
        s2_ = s2_ * tl.exp(m2 - z2) + tl.sum(tl.exp(l2 - z2), 0)
        m0, m1, m2 = n0, n1, n2
    tl.store(out_ptr + pid * 3, m0 + tl.log(s0))
    tl.store(out_ptr + pid * 3 + 1, m1 + tl.log(s1))
    tl.store(out_ptr + pid * 3 + 2, m2 + tl.log(s2_))


def loglik3(betahat, shat2, lpw, V3, n=None):
    """(rows,P) inputs, V3 (rows,3) -> (rows,3) SER log-likelihoods (lbf_model) at the three prior variances.
    n (rows,) int: real row lengths (padded entries have lpw = -inf and contribute exactly 0)."""
    rows, P = betahat.shape
    out = torch.empty((rows, 3), dtype=torch.float64, device=betahat.device)
    if n is None:
        n = torch.full((rows,), P, dtype=torch.int32, device=betahat.device)
    if rows:
        _loglik3_kernel[(rows,)](betahat.contiguous(), shat2.contiguous(), lpw.contiguous(), V3.contiguous(), out,
                                 P, n.contiguous().to(torch.int32), BLOCK=BLOCK, num_warps=NUM_WARPS)
    return out


# ---------------------------------------------------------------------------------------------- batched GEMV
# The LD products are batched matrix-VECTOR products (2 trait rows per locus). cuBLAS/cutlass fp64 GEMMs pad the
# 2-row operand to a full tile (measured: 41% of fp64 GPU time); these memory-bound kernels read each LD element once.
@triton.jit
def _vtm_kernel(v_ptr, m_ptr, o_ptr, I, J, K: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr):
    """o[b,k,j] = sum_i v[b,k,i] M[b,i,j]  (M (B,I,J) contiguous, v (B,K,I), o (B,K,J)); K in {1, 2}."""
    b = tl.program_id(0).to(tl.int64)
    jb = tl.program_id(1)
    offj = jb * BJ + tl.arange(0, BJ)
    mj = offj < J
    acc0 = tl.zeros((BJ,), dtype=tl.float64)
    acc1 = tl.zeros((BJ,), dtype=tl.float64)
    mbase = m_ptr + b * I * J
    vbase = v_ptr + b * K * I
    for i0 in range(0, I, BI):
        offi = i0 + tl.arange(0, BI)
        mi = offi < I
        tile = tl.load(mbase + offi[:, None] * J + offj[None, :], mask=mi[:, None] & mj[None, :], other=0.0)
        tile = tile.to(tl.float64)
        v0 = tl.load(vbase + offi, mask=mi, other=0.0)
        acc0 += tl.sum(tile * v0[:, None], 0)
        if K == 2:
            v1 = tl.load(vbase + I + offi, mask=mi, other=0.0)
            acc1 += tl.sum(tile * v1[:, None], 0)
    obase = o_ptr + b * K * J
    tl.store(obase + offj, acc0, mask=mj)
    if K == 2:
        tl.store(obase + J + offj, acc1, mask=mj)


@triton.jit
def _mv_kernel(v_ptr, m_ptr, o_ptr, I, J, K: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr):
    """o[b,k,i] = sum_j M[b,i,j] v[b,k,j]  (M (B,I,J), v (B,K,J), o (B,K,I)); K in {1, 2}."""
    b = tl.program_id(0).to(tl.int64)
    ib = tl.program_id(1)
    offi = ib * BI + tl.arange(0, BI)
    mi = offi < I
    acc0 = tl.zeros((BI,), dtype=tl.float64)
    acc1 = tl.zeros((BI,), dtype=tl.float64)
    mbase = m_ptr + b * I * J
    vbase = v_ptr + b * K * J
    for j0 in range(0, J, BJ):
        offj = j0 + tl.arange(0, BJ)
        mj = offj < J
        tile = tl.load(mbase + offi[:, None] * J + offj[None, :], mask=mi[:, None] & mj[None, :], other=0.0)
        tile = tile.to(tl.float64)
        v0 = tl.load(vbase + offj, mask=mj, other=0.0)
        acc0 += tl.sum(tile * v0[None, :], 1)
        if K == 2:
            v1 = tl.load(vbase + J + offj, mask=mj, other=0.0)
            acc1 += tl.sum(tile * v1[None, :], 1)
    obase = o_ptr + b * K * I
    tl.store(obase + offi, acc0, mask=mi)
    if K == 2:
        tl.store(obase + I + offi, acc1, mask=mi)


def vtm(v: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """(B,K,I) x (B,I,J) -> (B,K,J) float64, K <= 2 (larger K: chunks of 2)."""
    B, Kk, I = v.shape
    J = M.shape[2]
    M = M.contiguous()
    out = torch.empty((B, Kk, J), dtype=torch.float64, device=v.device)
    for k0 in range(0, Kk, 2):
        vk = v[:, k0:k0 + 2].contiguous().to(torch.float64)
        kk = vk.shape[1]
        o = torch.empty((B, kk, J), dtype=torch.float64, device=v.device)
        _vtm_kernel[(B, triton.cdiv(J, 128))](vk, M, o, I, J, K=kk, BI=32, BJ=128, num_warps=4)
        out[:, k0:k0 + kk] = o
    return out


def mv(v: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """M v: (B,K,J) with M (B,I,J) -> (B,K,I) float64, K <= 2 (larger K: chunks of 2)."""
    B, Kk, J = v.shape
    I = M.shape[1]
    M = M.contiguous()
    out = torch.empty((B, Kk, I), dtype=torch.float64, device=v.device)
    for k0 in range(0, Kk, 2):
        vk = v[:, k0:k0 + 2].contiguous().to(torch.float64)
        kk = vk.shape[1]
        o = torch.empty((B, kk, I), dtype=torch.float64, device=v.device)
        _mv_kernel[(B, triton.cdiv(I, 16))](vk, M, o, I, J, K=kk, BI=16, BJ=256, num_warps=4)
        out[:, k0:k0 + kk] = o
    return out
