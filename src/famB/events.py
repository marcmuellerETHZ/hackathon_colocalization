"""Monte-Carlo oracle-event posteriors for ExactColoc (DESIGN v3 Sec. 1, Sec. 5 A/B, layer D).

Given terms sampled proportional to their posterior mass (enum.enumerate_batch), draw one effect vector per
sampled term from that term's Gaussian effect posterior
    beta_s | z, term ~ N(C y_L, C),  C = L A^-1 L',  y_L = A^-1 L' b_s,  A = I + L' Phi_ss L,
evaluate the oracle class of the drawn (S1, S2, lambda) and average. The estimator is unbiased for the
posterior class probabilities (stratified resampling + one exact Gaussian draw per sampled term).

Oracle definitions (DESIGN Sec. 1; vectorised re-implementation of src/sim/oracle.py, cross-checked in
tests/famB/test_sim_integration.py):
  lambda_tj  own effect of causal j in trait t (z scale);  mu_tj = (Rt lambda_t)_j  (expected z)
  powered_t(j) = |mu_tj| >= tau_t and |lambda_tj| >= tau_t
  nonweak_t(j) = |mu_tj| >= w_t                    (power >= 0.2)
  S : some j in S1 n S2 powered in both traits
  U : not S, each trait has a powered variant, class-determining pair (cross-trait powered pair with max
      r^2) has d = min(|lambda_1c1|, |lambda_2c2|) sqrt(1 - r^2) < 2
  D : not S/U, each trait has a powered variant, no j in S1 n S2 nonweak in both traits
  W : not S/U/D, some trait has no nonweak causal variant
  G : otherwise
  E2 (layer D) : no trait-1 causal j that is nonweak in trait 1 has |lambda_2j| >= w_2 (E1 symmetric)
r^2 uses the model LD Rt (identical to src/sim/l0star.py, whose oracle uses R_tilde; tested against
sim.oracle.oracle_class).
"""
from __future__ import annotations

import torch

from .constants import D_MIN, TAU1, TAU2, WEAK1, WEAK2
from .enum import TX, L_entries, LocusTensors, TermSet, rand_rows, structure


def posterior_draws(T: LocusTensors, ts: TermSet, samples: torch.Tensor, generator=None):
    """samples [nB, N] term ids -> (bidx [S], V [S,4], m [S,4], lam [S,4]) with S = nB*N (fp64)."""
    dev = T.b.device
    nB, N = samples.shape
    flat = samples.reshape(-1)
    bidx = torch.arange(nB, device=dev).repeat_interleave(N)
    V, m, kx = ts.decode(flat)
    dt = torch.float64
    _, _, _, _, flags = structure(V, m)
    mf = m.to(dt)
    sdx = [T.sd[bidx, TX[x], V[:, x], kx[:, x]].to(dt) * mf[:, x] for x in range(4)]
    Le = L_entries(sdx, [f.to(dt) for f in flags], T.rho_b.to(dt)[bidx])
    S = flat.shape[0]
    L = torch.zeros(S, 4, 4, dtype=dt, device=dev)
    L[:, 0, 0] = Le["L00"]; L[:, 1, 1] = Le["L11"]; L[:, 2, 0] = Le["L20"]; L[:, 2, 1] = Le["L21"]
    L[:, 2, 2] = Le["L22"]; L[:, 3, 0] = Le["L30"]; L[:, 3, 1] = Le["L31"]; L[:, 3, 3] = Le["L33"]
    Phi = torch.empty(S, 4, 4, dtype=dt, device=dev)
    for x in range(4):
        for y in range(4):
            if T.Phi is not None:  # B-mm: general X' Sigma^-1 X
                Phi[:, x, y] = T.Phi[bidx, TX[x] * T.M + V[:, x], TX[y] * T.M + V[:, y]].to(dt)
                continue
            if T.Rt is None:
                r = (V[:, x] == V[:, y]).to(dt)
            else:
                r = T.Rt[bidx, V[:, x], V[:, y]].to(dt)
            Phi[:, x, y] = T.Vinv[bidx, TX[x], TX[y]].to(dt) * r
    bs = torch.stack([T.b[bidx, TX[x], V[:, x]].to(dt) for x in range(4)], dim=-1)
    A = torch.eye(4, dtype=dt, device=dev) + L.mT @ Phi @ L
    Kc = torch.linalg.cholesky(A)
    y = (L.mT @ bs[..., None])
    mean = L @ torch.cholesky_solve(y, Kc)
    eps = rand_rows(nB, (N, 4, 1), dev, generator, normal=True).reshape(S, 4, 1).to(dt)  # per-locus rows
    noise = L @ torch.linalg.solve_triangular(Kc.mT, eps, upper=True)
    lam = (mean + noise)[..., 0] * mf
    return bidx, V, m, lam


def oracle_events(T: LocusTensors, bidx, V, m, lam, r_raw_fn=None, thresholds=None):
    """Per-draw oracle class + exclusion events. Returns dict of bool tensors [S].

    r_raw_fn(bidx, vi, vj) -> raw LD r between screened variants (default: from T.Rt with eta removed)."""
    tau1, tau2, w1, w2 = thresholds or (TAU1, TAU2, WEAK1, WEAK2)
    dt = torch.float64

    def Rt(x, y):
        if T.Rt is None:
            return (V[:, x] == V[:, y]).to(dt)
        return T.Rt[bidx, V[:, x], V[:, y]].to(dt)

    def rraw(vi, vj):
        if r_raw_fn is not None:
            return r_raw_fn(bidx, vi, vj)
        if T.Rt is None:
            return (vi == vj).to(dt)
        # r^2 of the class-determining pair from the model LD Rt (as src/sim/l0star.py's oracle does)
        return T.Rt[bidx, vi, vj].to(dt)

    # expected z of each causal variant (mean map Rt, consistent with the likelihood)
    mu0 = Rt(0, 0) * lam[:, 0] + Rt(0, 1) * lam[:, 1]
    mu1 = Rt(1, 0) * lam[:, 0] + Rt(1, 1) * lam[:, 1]
    mu2 = Rt(2, 2) * lam[:, 2] + Rt(2, 3) * lam[:, 3]
    mu3 = Rt(3, 2) * lam[:, 2] + Rt(3, 3) * lam[:, 3]
    mu = torch.stack([mu0, mu1, mu2, mu3], dim=-1).abs()
    la = lam.abs()
    tau = torch.tensor([tau1, tau1, tau2, tau2], dtype=dt, device=lam.device)
    wk = torch.tensor([w1, w1, w2, w2], dtype=dt, device=lam.device)
    powered = m & (mu >= tau) & (la >= tau)
    nonweak = m & (mu >= wk)
    # cross-trait pairs (i in {0,1}, k in {2,3})
    pairs = [(0, 2), (0, 3), (1, 2), (1, 3)]
    shared = {(i, k): m[:, i] & m[:, k] & (V[:, i] == V[:, k]) for i, k in pairs}
    S_ev = torch.zeros_like(m[:, 0])
    block = torch.zeros_like(S_ev)
    excl2_viol = torch.zeros_like(S_ev)  # violates E2
    excl1_viol = torch.zeros_like(S_ev)
    for i, k in pairs:
        sh = shared[(i, k)]
        S_ev = S_ev | (sh & powered[:, i] & powered[:, k])
        block = block | (sh & nonweak[:, i] & nonweak[:, k])
        excl2_viol = excl2_viol | (sh & nonweak[:, i] & (la[:, k] >= w2))
        excl1_viol = excl1_viol | (sh & nonweak[:, k] & (la[:, i] >= w1))
    has1 = powered[:, 0] | powered[:, 1]
    has2 = powered[:, 2] | powered[:, 3]
    # class-determining pair: cross-trait powered pair with maximal r^2
    best_r2 = torch.full_like(mu0, -1.0)
    best_d = torch.full_like(mu0, float("inf"))
    for i, k in pairs:
        ok = powered[:, i] & powered[:, k] & (V[:, i] != V[:, k])
        r2 = rraw(V[:, i], V[:, k]) ** 2
        d = torch.minimum(la[:, i], la[:, k]) * torch.sqrt(torch.clamp(1.0 - r2, min=0.0))
        better = ok & (r2 > best_r2)
        best_r2 = torch.where(better, r2, best_r2)
        best_d = torch.where(better, d, best_d)
    U_ev = ~S_ev & has1 & has2 & (best_d < D_MIN)
    D_ev = ~S_ev & ~U_ev & has1 & has2 & ~block
    any_nw1 = nonweak[:, 0] | nonweak[:, 1]
    any_nw2 = nonweak[:, 2] | nonweak[:, 3]
    W_ev = ~S_ev & ~U_ev & ~D_ev & (~any_nw1 | ~any_nw2)
    G_ev = ~S_ev & ~U_ev & ~D_ev & ~W_ev
    return dict(S=S_ev, U=U_ev, D=D_ev, W=W_ev, G=G_ev, E1=~excl1_viol, E2=~excl2_viol)


def event_posteriors(T: LocusTensors, ts: TermSet, samples: torch.Tensor, generator=None, r_raw_fn=None,
                     thresholds=None):
    """Average events over draws -> dict name -> [nB] fp64 posterior probabilities (+ MC n)."""
    nB, N = samples.shape
    bidx, V, m, lam = posterior_draws(T, ts, samples, generator)
    ev = oracle_events(T, bidx, V, m, lam, r_raw_fn=r_raw_fn, thresholds=thresholds)
    return {k: v.to(torch.float64).view(nB, N).mean(dim=1) for k, v in ev.items()}
