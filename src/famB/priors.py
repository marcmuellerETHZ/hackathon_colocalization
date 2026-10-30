"""Priors for ExactColoc (DESIGN v3 Sec. 5 B "Priors").

Three modes:
  coloc      single causal per trait (Kc = 1), rho = 0, z-scale prior variance W^2 / SE_tj^2, per-SNP prior
             probabilities p1, p2, p12 at full p (no screen). Reproduces coloc.abf exactly (tests).
  gaussian   <= 2 causal per trait on the screen; per-SNP probabilities p1, p2, p12 (coloc convention, no
             p0 normalisation); effect lambda_t ~ N(0, sigma_t^2) on the z scale with (sigma_t, eta) chosen
             per locus by empirical Bayes (maximum marginal likelihood of each trait's screened z on a grid);
             shared-effect prior correlation rho_b (default 0).
  simulator  <= 2 causal per trait; configuration prior = type table pi(n_shared, n_1only, n_2only), uniform
             over variant choices; |lambda| prior = K-component scale mixture of zero-mean Gaussians per
             trait (independent across traits and variants). Used in world L0* (exact Bayes posterior).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .enum import LocusTensors, TermSet, enumerate_batch

NEG_INF = -math.inf


def type_table_per_snp(p1: float, p2: float, p12: float, Kc: int) -> np.ndarray:
    """log prior by (n_sh, n_1only, n_2only) for independent per-SNP states (coloc convention:
    config weight p1^n1 p2^n2 p12^nsh relative to the empty configuration)."""
    t = np.full((3, 3, 3), NEG_INF)
    for nsh in range(3):
        for n1 in range(3):
            for n2 in range(3):
                if nsh + n1 <= Kc and nsh + n2 <= Kc:
                    t[nsh, n1, n2] = n1 * math.log(p1) + n2 * math.log(p2) + (nsh * math.log(p12) if nsh else 0.0)
    return t


def n_configs_of_type(M: int, nsh: int, n1: int, n2: int) -> float:
    """Number of (S1, S2) with the given type among M variants (multinomial)."""
    k = nsh + n1 + n2
    if k > M:
        return 0.0
    return math.factorial(M) / (math.factorial(nsh) * math.factorial(n1) * math.factorial(n2) * math.factorial(M - k))


def type_table_exchangeable(type_prob: np.ndarray, M: int) -> np.ndarray:
    """log prior per configuration for a type-probability table (sums to 1 over valid types) with uniform
    choice of the variants: log pi(type) - log #configs(type)."""
    tp = np.asarray(type_prob, dtype=np.float64)
    t = np.full((3, 3, 3), NEG_INF)
    for nsh in range(3):
        for n1 in range(3):
            for n2 in range(3):
                n = n_configs_of_type(M, nsh, n1, n2)
                if tp[nsh, n1, n2] > 0 and n > 0 and nsh + n1 <= 2 and nsh + n2 <= 2:
                    t[nsh, n1, n2] = math.log(tp[nsh, n1, n2]) - math.log(n)
    return t


@dataclass
class SimPrior:
    """Simulator-prior specification (one per locus; may depend on N1, N2 and stratum)."""

    type_prob: np.ndarray  # [3,3,3] probabilities of (n_sh, n_1only, n_2only); invalid entries 0
    mix_w: np.ndarray  # [2, K] mixture weights per trait (rows sum to 1)
    mix_sd: np.ndarray  # [2, K] component sds of lambda (z scale)

    def validate(self):
        tp = np.asarray(self.type_prob)
        assert tp.shape == (3, 3, 3) and np.all(tp >= 0) and abs(tp.sum() - 1) < 1e-9
        for nsh in range(3):
            for n1 in range(3):
                for n2 in range(3):
                    if nsh + n1 > 2 or nsh + n2 > 2:
                        assert tp[nsh, n1, n2] == 0, "types beyond 2 causal per trait"
        w = np.asarray(self.mix_w)
        assert np.allclose(w.sum(1), 1) and np.all(np.asarray(self.mix_sd) > 0)


def toy_l0star_prior(K: int = 8) -> SimPrior:
    """Toy L0*-like prior (until src/sim provides the real one): DESIGN Sec. 2.4 mixture restricted to
    <= 2 causal per trait, plus 5% null loci. Scale mixture with K geometric sds from 1 to 60."""
    tp = np.zeros((3, 3, 3))
    tp[0, 0, 0] = 0.05
    # S 30%: 1 shared, +1 trait-specific in 50%
    tp[1, 0, 0] += 0.30 * 0.5
    tp[1, 1, 0] += 0.30 * 0.25
    tp[1, 0, 1] += 0.30 * 0.25
    # D 30%: 1 + 1, extras in 30%
    tp[0, 1, 1] += 0.30 * 0.7
    tp[0, 2, 1] += 0.30 * 0.15
    tp[0, 1, 2] += 0.30 * 0.15
    # W 20%: one trait only
    tp[0, 1, 0] += 0.10
    tp[0, 0, 1] += 0.10
    # M 15%: mixed 2-per-trait
    tp[1, 1, 1] += 0.05
    tp[0, 2, 2] += 0.05
    tp[2, 0, 0] += 0.05
    tp /= tp.sum()
    sd = np.geomspace(1.0, 60.0, K)
    w = np.full(K, 1.0 / K)
    return SimPrior(type_prob=tp, mix_w=np.stack([w, w]), mix_sd=np.stack([sd, sd]))


def sim_prior_from_l0star(drop_zero_components: bool = True) -> SimPrior:
    """SimPrior equal to the L0* world of src/sim/l0star.py (owner: simulation agent): type table from
    TYPE_PRIOR / S_EXTRA / D_EXTRA (uniform positions) and the per-trait 8-component scale mixture
    LAMBDA_MIX. Components with zero weight in both traits are dropped (exact; saves (8/K)^4 compute)."""
    from sim import l0star, params

    tp = np.zeros((3, 3, 3))
    pS, pD, pW = l0star.TYPE_PRIOR["S"], l0star.TYPE_PRIOR["D"], l0star.TYPE_PRIOR["W"]
    tp[1, 0, 0] = pS * (1 - params.S_EXTRA)
    tp[1, 1, 0] = tp[1, 0, 1] = pS * params.S_EXTRA / 2
    tp[0, 1, 1] = pD * (1 - params.D_EXTRA)
    tp[0, 2, 1] = tp[0, 1, 2] = pD * params.D_EXTRA / 2
    tp[0, 1, 0] = tp[0, 0, 1] = pW / 2
    w1, s1 = l0star.LAMBDA_MIX[1]
    w2, s2 = l0star.LAMBDA_MIX[2]
    assert np.allclose(s1, s2), "B assumes a common sd grid for both traits"
    w = np.stack([w1, w2]).astype(np.float64)
    sd = np.stack([s1, s2]).astype(np.float64)
    if drop_zero_components:
        keep = w.max(0) > 0
        w, sd = w[:, keep], sd[:, keep]
    return SimPrior(type_prob=tp / tp.sum(), mix_w=w / w.sum(1, keepdims=True), mix_sd=sd)


def reweight_classes(post: dict, ratio: dict) -> dict:
    """Class-prior correction for rejection-sampled evaluation cells (DESIGN Sec. 5 B, R2-stat M4):
    multiply each class posterior by target_share / natural_share and renormalise over S,U,D,W,G."""
    keys = ("S", "U", "D", "W", "G")
    num = {k: np.asarray(post[k], dtype=np.float64) * ratio.get(k, 1.0) for k in keys}
    tot = sum(num.values())
    return {k: num[k] / tot for k in keys}


# ---------------------------------------------------------------------------------------------------
# Empirical Bayes for Gaussian mode
# ---------------------------------------------------------------------------------------------------
@dataclass
class EBGrid:
    sigma: tuple = (1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0)
    eta: tuple = (1e-3, 1e-2, 3e-2, 0.1, 0.2)


def _log_sum_prior_single(logm: np.ndarray, logp: float, Kc: int) -> float:
    """log of sum over single-trait configs (<= Kc) of prod_{j in S} m_j p  (empty config weight 1)."""
    m = np.exp(logm)
    s1 = m.sum()
    tot = 1.0 + math.exp(logp) * s1
    if Kc >= 2:
        s2 = 0.5 * (s1 * s1 - (m * m).sum())
        tot += math.exp(2 * logp) * s2
    return math.log(tot)


def eb_select(zu: np.ndarray, Ruu: np.ndarray, logm: np.ndarray, p_marg: tuple, Kc: int, grid: EBGrid,
              device, dtype=torch.float64, termset_cache: dict | None = None):
    """Per-locus EB: choose sigma_t per trait and a common eta maximising
        l_t(sigma, eta) = log N(z_tu; 0, Rt(eta)) + log sum_c pi_c BF_c(sigma, eta) - log sum_c pi_c
    (single-trait marginal likelihood of the screened z; pi from per-SNP marginal probability p_t + p12).
    zu [2, M], Ruu [M, M] raw LD. Returns (sigma1, sigma2, eta, table of l)."""
    M = zu.shape[1]
    ne, ns = len(grid.eta), len(grid.sigma)
    key = ("single", M, Kc)
    if termset_cache is not None and key in termset_cache:
        ts = termset_cache[key]
    else:
        ts = TermSet(M, Kc, K=1, single_trait=True, device=device)
        if termset_cache is not None:
            termset_cache[key] = ts
    R = torch.as_tensor(Ruu, dtype=torch.float64, device=device)
    eye = torch.eye(M, dtype=torch.float64, device=device)
    etas = torch.as_tensor(grid.eta, dtype=torch.float64, device=device)
    Rts = (1 - etas)[:, None, None] * R[None] + etas[:, None, None] * eye[None]  # [ne, M, M]
    Lc = torch.linalg.cholesky(Rts)
    z = torch.as_tensor(zu, dtype=torch.float64, device=device)  # [2, M]
    sol = torch.cholesky_solve(z.T[None].expand(ne, -1, -1).contiguous(), Lc)  # [ne, M, 2]
    quad = (z.T[None] * sol).sum(1)  # [ne, 2]
    logdet = 2 * torch.log(torch.diagonal(Lc, dim1=1, dim2=2)).sum(1)  # [ne]
    null = -0.5 * quad - 0.5 * logdet[:, None] - 0.5 * M * math.log(2 * math.pi)  # [ne, 2]
    # batch over (trait, eta, sigma)
    nB = 2 * ne * ns
    tt = torch.arange(2, device=device).repeat_interleave(ne * ns)
    ei = torch.arange(ne, device=device).repeat_interleave(ns).repeat(2)
    si = torch.arange(ns, device=device).repeat(2 * ne)
    sig = torch.as_tensor(grid.sigma, dtype=torch.float64, device=device)[si]
    b = torch.zeros(nB, 2, M, dtype=torch.float64, device=device)
    b[:, 0, :] = z[tt]
    sd = torch.ones(nB, 2, M, 1, dtype=torch.float64, device=device)
    sd[:, 0, :, 0] = sig[:, None]
    lpt = torch.full((nB, 3, 3, 3), NEG_INF, dtype=torch.float64, device=device)
    logp = torch.log(torch.as_tensor(p_marg, dtype=torch.float64, device=device))[tt]
    for n1 in range(Kc + 1):
        lpt[:, 0, n1, 0] = n1 * logp
    lm = torch.as_tensor(logm, dtype=torch.float64, device=device)
    T = LocusTensors(Rt=Rts[ei], b=b, Vinv=eye[:2, :2][None].expand(nB, -1, -1).contiguous(), sd=sd,
                     logwk=torch.zeros(nB, 2, 1, dtype=torch.float64, device=device),
                     rho_b=torch.zeros(nB, dtype=torch.float64, device=device), logm=lm[None].expand(nB, -1),
                     logprior_type=lpt, eta=etas[ei])
    res = enumerate_batch(T, ts, n_samples=0, want_pip=False, dtype=dtype)
    logZ = res["logZ"].view(2, ne, ns)
    lsp = torch.as_tensor([_log_sum_prior_single(logm, math.log(p), Kc) for p in p_marg],
                          dtype=torch.float64, device=device)
    ell = null.T[:, :, None] + logZ - lsp[:, None, None]  # [2, ne, ns]
    best_s = ell.max(dim=2)
    tot = best_s.values.sum(0)  # [ne]
    e_star = int(torch.argmax(tot))
    s1 = float(grid.sigma[int(best_s.indices[0, e_star])])
    s2 = float(grid.sigma[int(best_s.indices[1, e_star])])
    return s1, s2, float(grid.eta[e_star]), ell.cpu().numpy()
