"""coloc pairing on GPU: combine.abf, coloc.bf_bf (coloc 5.2.3) and coloc.abf's approximate Bayes factors.

All in float64 torch (device-agnostic), batched over loci and over all (CS1, CS2) effect pairs.
Validated against R coloc 5.2.3 in tests/famG/test_coloc.py.

coloc.bf_bf(bf1, bf2, p1, p2, p12) with the same SNP set in both traits (always true here): the
overlap trimming (prop1/prop2 >= overlap.min) is a no-op because every SNP is shared, so each pair
(i, j) of rows gets combine.abf(bf1[i,], bf2[j,], p1, p2, p12).
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _logsum(x: torch.Tensor, dim=-1) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    return (m + torch.log(torch.exp(x - m).sum(dim=dim, keepdim=True))).squeeze(dim)


def _logdiff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m = torch.maximum(x, y)
    return m + torch.log(torch.exp(x - m) - torch.exp(y - m))


def combine_abf_from_sums(ls1, ls2, ls12, p1, p2, p12):
    """PP.H0..H4 from logsum(l1), logsum(l2), logsum(l1 + l2) (any broadcastable shape) -> (..., 5)."""
    lH0 = torch.zeros_like(ls12)
    lH1 = math.log(p1) + ls1
    lH2 = math.log(p2) + ls2
    lH3 = math.log(p1) + math.log(p2) + _logdiff(ls1 + ls2, ls12)
    lH4 = math.log(p12) + ls12
    allh = torch.stack(torch.broadcast_tensors(lH0, lH1, lH2, lH3, lH4), dim=-1)
    den = _logsum(allh, dim=-1)
    return torch.exp(allh - den[..., None])


def combine_abf(l1, l2, p1=1e-4, p2=1e-4, p12=1e-5):
    """coloc::combine.abf for vectors (..., p) -> (..., 5)."""
    l1 = torch.as_tensor(l1, dtype=torch.float64)
    l2 = torch.as_tensor(l2, dtype=torch.float64, device=l1.device)
    return combine_abf_from_sums(_logsum(l1), _logsum(l2), _logsum(l1 + l2), p1, p2, p12)


def bf_bf_batch(bf1_list, bf2_list, p1=1e-4, p2=1e-4, p12=5e-6, device=None, chunk=256):
    """coloc.bf_bf for many loci. bf*_list: lists of (k_i, p_i) arrays of log BFs (rows = CS effects).
    Returns list of (k1_i, k2_i, 5) numpy arrays of PP.H0..H4 (empty arrays when a trait has no rows)."""
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    out = [None] * len(bf1_list)
    order = [i for i in range(len(bf1_list)) if len(bf1_list[i]) and len(bf2_list[i])]
    oset = set(order)
    for i in range(len(bf1_list)):
        if i not in oset:
            out[i] = np.zeros((len(bf1_list[i]), len(bf2_list[i]), 5))
    for c0 in range(0, len(order), chunk):
        idx = order[c0:c0 + chunk]
        K1 = max(len(bf1_list[i]) for i in idx)
        K2 = max(len(bf2_list[i]) for i in idx)
        P = max(np.asarray(bf1_list[i]).shape[1] for i in idx)
        B = len(idx)
        neg = -math.inf
        A = torch.full((B, K1, P), neg, dtype=torch.float64, device=device)
        C = torch.full((B, K2, P), neg, dtype=torch.float64, device=device)
        for b, i in enumerate(idx):
            a = torch.as_tensor(np.asarray(bf1_list[i], dtype=np.float64), device=device)
            c = torch.as_tensor(np.asarray(bf2_list[i], dtype=np.float64), device=device)
            A[b, : a.shape[0], : a.shape[1]] = a
            C[b, : c.shape[0], : c.shape[1]] = c
        ls1 = _logsum(A)                                       # (B,K1)
        ls2 = _logsum(C)                                       # (B,K2)
        ls12 = _logsum(A[:, :, None, :] + C[:, None, :, :])    # (B,K1,K2)
        pp = combine_abf_from_sums(ls1[:, :, None], ls2[:, None, :], ls12, p1, p2, p12)
        pp = pp.cpu().numpy()
        for b, i in enumerate(idx):
            k1, k2 = len(bf1_list[i]), len(bf2_list[i])
            r = pp[b, :k1, :k2]
            bad = ~np.isfinite(r).all(-1)
            r[bad] = np.array([1.0, 0, 0, 0, 0])               # coloc: all-NA SNP.PP.H4 -> (1,0,0,0,0)
            out[i] = r
    return out


def abf_lbf(beta, varbeta, sdY=1.0, W=0.15):
    """coloc:::approx.bf.estimates for a quantitative trait: lABF = 0.5 (log(1 - r) + r z^2), r = W^2/(W^2 + V),
    with prior SD W * sdY (coloc hard-codes W = 0.15)."""
    beta = np.asarray(beta, dtype=np.float64)
    V = np.asarray(varbeta, dtype=np.float64)
    z = beta / np.sqrt(V)
    sd_prior = W * sdY
    r = sd_prior ** 2 / (sd_prior ** 2 + V)
    return 0.5 * (np.log(1 - r) + r * z * z)


def adjust_prior(p: float, nsnps: int) -> float:
    """coloc 5.2.3 adjust_prior (applied by coloc.abf, not by coloc.bf_bf): p = 1/(nsnps + 1) if nsnps * p >= 1."""
    return 1.0 / (nsnps + 1) if nsnps * p >= 1 else p


def coloc_abf(beta1, se1, beta2, se2, p1=1e-4, p2=1e-4, p12=1e-5, sdY1=1.0, sdY2=1.0, W=0.15):
    """coloc.abf posterior (PP.H0..H4) from beta/se, sdY given (DESIGN §2.2: sdY = 1), including coloc.abf's
    adjust_prior on p1, p2, p12 (M3.5; matters only when p * nsnps >= 1, e.g. p1 = 1e-4 with >= 10,000 SNPs)."""
    l1 = abf_lbf(beta1, np.asarray(se1) ** 2, sdY1, W)
    l2 = abf_lbf(beta2, np.asarray(se2) ** 2, sdY2, W)
    n = len(l1)
    return combine_abf(torch.as_tensor(l1), torch.as_tensor(l2), adjust_prior(p1, n), adjust_prior(p2, n),
                       adjust_prior(p12, n)).numpy()
