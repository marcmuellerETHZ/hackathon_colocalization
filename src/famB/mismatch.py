"""B-mm: LD-mismatch likelihood for ExactColoc (DESIGN §5 layer E; DESIGN_P2 §2 "B-mm").

Model. With the reference LD R_hat (n_ref individuals) in place of the study LD R, the RSS model
    z_t = R lambda_t + e_t,  e ~ N(0, V (x) R)
becomes z_t = R_hat lambda_t + (R - R_hat) lambda_t + e_t. The reference-error term (R - R_hat) lambda adds,
for variant k and traits t, u,
    Cov((E lambda_t)_k, (E lambda_u)_k) = (1/n_ref) sum_{j,j'} lambda_tj lambda_uj' C^(k)_{jj'}
with the large-sample covariance of sample correlations sharing index k (Olkin & Siotani 1976):
    C^(k)_{jj'} = r_jj' (1 - r_kj^2 - r_kj'^2) - 1/2 r_kj r_kj' (1 - r_kj^2 - r_kj'^2 - r_jj'^2),
    C^(k)_{jj}  = (1 - r_kj^2)^2                                   (DESIGN layer E diagonal term),
the j != j' entries being the "cross terms" (cross_terms=False keeps the diagonal term only). The same
reference error enters both traits, so the inflation also has a cross-trait part (t != u). Only the
per-variant (co)variance is inflated (DESIGN: Var((R_hat b)_k)); covariances between different k are not
added. Delta is therefore block-diagonal over k with 2x2 (trait x trait) blocks.

Plug-in (empirical Bayes, as susieR's R_finite plugs in its current fit): lambda lambda' is replaced by its
posterior second moment E[lambda lambda' | z] under the current fit (exact over all enumerated configurations,
`second_moments`), giving a per-locus Delta_hat; then
    z ~ N((I2 (x) Rt) beta, Sigma),   Sigma = V (x) Rt + Delta_hat,   Rt = (1 - eta) R_hat + eta I,
and the Bayes factor of every configuration is the exact support-space BF of this Gaussian model computed
from the FULL z_M likelihood (DESIGN §5 B exactness scope "variance-inflated mismatch models"):
    Phi = X' Sigma^-1 X,  b = X' Sigma^-1 z,  X = I2 (x) Rt.
Stable form (no inverse of the near-singular Rt; Sigma0 = V (x) Rt, X' Sigma0^-1 = V^-1 (x) I):
    Phi = V^-1 (x) Rt - (V^-1 (x) I)(Delta - Delta Sigma^-1 Delta)(V^-1 (x) I)
    b   = (V^-1 (x) I)(z - Delta Sigma^-1 z)
With Delta = 0 (n_ref = inf) both reduce exactly to phase-1 B (Phi = V^-1 (x) Rt, b = (V^-1 (x) I) z).
(sigma_t, eta) are re-selected by the phase-1 EB rule with each trait's marginal likelihood under
Sigma_t = Rt(eta) + diag(Delta_tt) (`eb_select_mm`). The plug-in is iterated `mm_iter` times starting from the
phase-1 fit (pass 0).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .enum import TX, L_entries, LocusTensors, TermSet, enumerate_batch, eval_terms, structure
from .priors import NEG_INF, EBGrid, _log_sum_prior_single

F64 = torch.float64


# ---------------------------------------------------------------------------------------------------------
# exact posterior second moments E[lambda lambda' | z] over all configurations (rectangular path, K = 1)
# ---------------------------------------------------------------------------------------------------------
def term_moments(T: LocusTensors, V, m, kx):
    """Per-term Gaussian effect posterior: mean [nB, P, 4] and covariance [nB, P, 4, 4] of beta_s
    (beta_s | z, term ~ N(L A^-1 L' b_s, L A^-1 L'), A = I + L' Phi_ss L), zero on inactive coordinates."""
    dt = F64
    nB, P = T.nB, V.shape[0]
    dev = T.b.device
    _, _, _, _, flags = structure(V, m)
    mf = m.to(dt)
    sdx = [T.sd[:, TX[x], V[:, x], kx[:, x]].to(dt) * mf[:, x][None] for x in range(4)]
    Le = L_entries(sdx, [f.to(dt)[None] for f in flags], T.rho_b.to(dt)[:, None])
    L = torch.zeros(nB, P, 4, 4, dtype=dt, device=dev)
    for (i, j), k in (((0, 0), "L00"), ((1, 1), "L11"), ((2, 0), "L20"), ((2, 1), "L21"), ((2, 2), "L22"),
                      ((3, 0), "L30"), ((3, 1), "L31"), ((3, 3), "L33")):
        L[:, :, i, j] = Le[k]
    Phi = torch.empty(nB, P, 4, 4, dtype=dt, device=dev)
    M = T.M
    for x in range(4):
        for y in range(4):
            if T.Phi is not None:
                Phi[:, :, x, y] = T.Phi[:, TX[x] * M + V[:, x], TX[y] * M + V[:, y]]
            else:
                Phi[:, :, x, y] = T.Vinv[:, TX[x], TX[y]][:, None] * T.Rt[:, V[:, x], V[:, y]]
    bs = torch.stack([T.b[:, TX[x], V[:, x]] for x in range(4)], dim=-1)  # [nB, P, 4]
    A = torch.eye(4, dtype=dt, device=dev) + L.mT @ Phi @ L
    Kc = torch.linalg.cholesky(A)
    Ainv = torch.cholesky_inverse(Kc)
    C = L @ Ainv @ L.mT
    mean = (C @ bs[..., None])[..., 0]
    return mean, C


def second_moments(T: LocusTensors, ts: TermSet, logZ: torch.Tensor, chunk_elems: int = 1 << 19,
                   min_logw_rel: float = -60.0) -> torch.Tensor:
    """E[lambda lambda' | z] [nB, 2M, 2M] (index t * M + v) = sum over terms of w_c (mu_c mu_c' + C_c).
    Terms with log posterior mass < logZ + min_logw_rel (mass < e^-60 = 9e-27 relative) are skipped
    (contribution below fp64 resolution of the total); everything else is exact."""
    if ts.offsets is not None:
        raise NotImplementedError("B-mm uses single-component (Gaussian) priors")
    dev = T.b.device
    nB, M = T.nB, T.M
    M2 = 2 * M
    S = torch.zeros(nB * M2 * M2, dtype=F64, device=dev)
    per = max(1, chunk_elems // max(1, nB))
    base = (torch.arange(nB, device=dev) * M2 * M2)[:, None]
    for s in range(0, ts.n_terms, per):
        t = torch.arange(s, min(ts.n_terms, s + per), device=dev, dtype=torch.int64)
        V, m, kx = ts.decode(t)
        logw, _ = eval_terms(T, ts, t)[:2]
        rel = logw - logZ[:, None]
        keep = (rel > min_logw_rel).any(dim=0)
        if not bool(keep.any()):
            continue
        V, m, kx, rel = V[keep], m[keep], kx[keep], rel[:, keep]
        w = torch.exp(rel)  # [nB, P]
        mean, C = term_moments(T, V, m, kx)
        M2t = (C + mean[..., :, None] * mean[..., None, :]) * w[..., None, None]  # [nB, P, 4, 4]
        gi = [TX[x] * M + V[:, x] for x in range(4)]
        for x in range(4):
            for y in range(4):
                if x != y:
                    ok = (m[:, x] & m[:, y])[None]
                else:
                    ok = m[:, x][None]
                idx = base + (gi[x] * M2 + gi[y])[None]
                S.index_add_(0, idx.reshape(-1), torch.where(ok, M2t[:, :, x, y], 0.0).reshape(-1))
    S = S.view(nB, M2, M2)
    return 0.5 * (S + S.mT)


# ---------------------------------------------------------------------------------------------------------
# inflation Delta and the full-z_M likelihood quantities
# ---------------------------------------------------------------------------------------------------------
def inflation_blocks(Rraw: torch.Tensor, S: torch.Tensor, n_ref: torch.Tensor, cross_terms: bool = True):
    """Rraw [nB, M, M] reference LD (raw r), S [nB, 2M, 2M] second moments, n_ref [nB] (inf allowed).
    Returns D [nB, 2, 2, M] (Delta_{t u, k}) with every 2x2 block projected to PSD, and the number of
    blocks that needed projection."""
    nB, M, _ = Rraw.shape
    R = Rraw.clamp(-1.0, 1.0)
    a = R * R
    S5 = S.view(nB, 2, M, 2, M)
    if cross_terms:
        aki = a[:, :, :, None]  # r_kj^2 over (k, j, .)
        akj = a[:, :, None, :]  # r_kj'^2 over (k, ., j')
        C = R[:, None] * (1 - aki - akj) - 0.5 * R[:, :, :, None] * R[:, :, None, :] * (1 - aki - akj - a[:, None])
        D = torch.einsum("bkij,btiuj->btuk", C, S5)
    else:
        w = (1 - a) ** 2  # [nB, k, j]
        Sd = torch.diagonal(S5.permute(0, 1, 3, 2, 4), dim1=3, dim2=4)  # [nB, t, u, j]
        D = torch.einsum("bkj,btuj->btuk", w, Sd)
    D = D / n_ref.to(F64)[:, None, None, None]
    D = 0.5 * (D + D.transpose(1, 2))
    d11 = D[:, 0, 0].clamp(min=0.0)
    d22 = D[:, 1, 1].clamp(min=0.0)
    lim = torch.sqrt(d11 * d22)
    d12 = torch.maximum(torch.minimum(D[:, 0, 1], lim), -lim)
    n_proj = int(((D[:, 0, 0] < 0) | (D[:, 1, 1] < 0) | (D[:, 0, 1].abs() > lim)).sum())
    out = torch.stack([torch.stack([d11, d12], 1), torch.stack([d12, d22], 1)], 1)
    return out, n_proj


def _kron2(A2: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """[nB, 2, 2] (x) [nB, M, M] -> [nB, 2M, 2M] with index (t, k) -> t * M + k."""
    nB, M = B.shape[0], B.shape[1]
    return (A2[:, :, None, :, None] * B[:, None, :, None, :]).reshape(nB, 2 * M, 2 * M)


def delta_full(D: torch.Tensor) -> torch.Tensor:
    """D [nB, 2, 2, M] -> Delta [nB, 2M, 2M] (block-diagonal over variants k)."""
    nB, M = D.shape[0], D.shape[3]
    return torch.diag_embed(D).permute(0, 1, 3, 2, 4).reshape(nB, 2 * M, 2 * M)


def phi_b(Rt: torch.Tensor, Vmat: torch.Tensor, Vinv: torch.Tensor, zu: torch.Tensor, D: torch.Tensor):
    """Full-z_M likelihood quantities of Sigma = V (x) Rt + Delta. Returns Phi [nB, 2M, 2M], b [nB, 2, M],
    null log-likelihood log N(z; 0, Sigma) [nB]."""
    nB, M = Rt.shape[0], Rt.shape[1]
    eye = torch.eye(M, dtype=F64, device=Rt.device)[None].expand(nB, -1, -1)
    Delta = delta_full(D)
    Sigma = _kron2(Vmat, Rt) + Delta
    Lc, info = torch.linalg.cholesky_ex(Sigma)
    if bool((info > 0).any()):
        bad = torch.nonzero(info > 0).flatten().tolist()
        raise RuntimeError(f"B-mm: Sigma not positive definite for batch rows {bad}")
    Kv = _kron2(Vinv, eye)
    zv = zu.reshape(nB, 2 * M, 1)
    SiD = torch.cholesky_solve(Delta, Lc)
    corr = Delta - Delta @ SiD
    corr = 0.5 * (corr + corr.mT)
    Phi = _kron2(Vinv, Rt) - Kv @ corr @ Kv
    Siz = torch.cholesky_solve(zv, Lc)
    b = (Kv @ (zv - Delta @ Siz)).reshape(nB, 2, M)
    logdet = 2 * torch.log(torch.diagonal(Lc, dim1=1, dim2=2)).sum(1)
    null = -0.5 * (zv * Siz).sum((1, 2)) - 0.5 * logdet - M * math.log(2 * math.pi)
    return Phi, b, null


# ---------------------------------------------------------------------------------------------------------
# EB (sigma_t, eta) under the inflated single-trait likelihood
# ---------------------------------------------------------------------------------------------------------
def eb_select_mm(zu: np.ndarray, Ruu: np.ndarray, logm: np.ndarray, p_marg: tuple, Kc: int, grid: EBGrid,
                 device, delta: np.ndarray | None, termset_cache: dict | None = None):
    """Phase-1 EB rule (priors.eb_select) with Sigma_t(eta) = Rt(eta) + diag(delta_t), delta [2, M] (None = 0):
        l_t(sigma, eta) = log N(z_t; 0, Sigma_t) + log sum_c pi_c BF_c(sigma, eta) - log sum_c pi_c,
    BF from the full-z likelihood (Phi_t = Rt - (Delta - Delta Sigma^-1 Delta), b_t = z - Delta Sigma^-1 z).
    Returns (sigma1, sigma2, eta, l table [2, ne, ns])."""
    M = zu.shape[1]
    ne, ns = len(grid.eta), len(grid.sigma)
    key = ("single", M, Kc)
    if termset_cache is not None and key in termset_cache:
        ts = termset_cache[key]
    else:
        ts = TermSet(M, Kc, K=1, single_trait=True, device=device)
        if termset_cache is not None:
            termset_cache[key] = ts
    R = torch.as_tensor(Ruu, dtype=F64, device=device)
    eye = torch.eye(M, dtype=F64, device=device)
    etas = torch.as_tensor(grid.eta, dtype=F64, device=device)
    Rts = (1 - etas)[:, None, None] * R[None] + etas[:, None, None] * eye[None]  # [ne, M, M]
    dl = torch.zeros(2, M, dtype=F64, device=device) if delta is None else torch.as_tensor(delta, dtype=F64,
                                                                                            device=device)
    z = torch.as_tensor(zu, dtype=F64, device=device)  # [2, M]
    Sig = Rts[None] + torch.diag_embed(dl)[:, None]  # [2, ne, M, M]
    Lc, info = torch.linalg.cholesky_ex(Sig)
    if bool((info > 0).any()):
        raise RuntimeError("B-mm EB: Sigma_t(eta) not positive definite")
    zz = z[:, None, :, None].expand(2, ne, M, 1).contiguous()
    Siz = torch.cholesky_solve(zz, Lc)  # [2, ne, M, 1]
    quad = (zz * Siz).sum((2, 3))  # [2, ne]
    logdet = 2 * torch.log(torch.diagonal(Lc, dim1=2, dim2=3)).sum(2)  # [2, ne]
    null = -0.5 * quad - 0.5 * logdet - 0.5 * M * math.log(2 * math.pi)  # [2, ne]
    Dg = torch.diag_embed(dl)[:, None].expand(2, ne, M, M)
    SiD = torch.cholesky_solve(Dg.contiguous(), Lc)
    corr = Dg - Dg @ SiD
    Phit = Rts[None] - 0.5 * (corr + corr.mT)  # [2, ne, M, M]
    bt = (zz - Dg @ Siz)[..., 0]  # [2, ne, M]
    nB = 2 * ne * ns
    tt = torch.arange(2, device=device).repeat_interleave(ne * ns)
    ei = torch.arange(ne, device=device).repeat_interleave(ns).repeat(2)
    si = torch.arange(ns, device=device).repeat(2 * ne)
    sig = torch.as_tensor(grid.sigma, dtype=F64, device=device)[si]
    b = torch.zeros(nB, 2, M, dtype=F64, device=device)
    b[:, 0, :] = bt[tt, ei]
    Phi = torch.zeros(nB, 2 * M, 2 * M, dtype=F64, device=device)
    Phi[:, :M, :M] = Phit[tt, ei]
    Phi[:, M:, M:] = eye
    sd = torch.ones(nB, 2, M, 1, dtype=F64, device=device)
    sd[:, 0, :, 0] = sig[:, None]
    lpt = torch.full((nB, 3, 3, 3), NEG_INF, dtype=F64, device=device)
    logp = torch.log(torch.as_tensor(p_marg, dtype=F64, device=device))[tt]
    for n1 in range(Kc + 1):
        lpt[:, 0, n1, 0] = n1 * logp
    lm = torch.as_tensor(logm, dtype=F64, device=device)
    T = LocusTensors(Rt=Rts[ei], b=b, Vinv=eye[:2, :2][None].expand(nB, -1, -1).contiguous(), sd=sd,
                     logwk=torch.zeros(nB, 2, 1, dtype=F64, device=device),
                     rho_b=torch.zeros(nB, dtype=F64, device=device), logm=lm[None].expand(nB, -1),
                     logprior_type=lpt, eta=etas[ei], Phi=Phi)
    res = enumerate_batch(T, ts, n_samples=0, want_pip=False)
    logZ = res["logZ"].view(2, ne, ns)
    lsp = torch.as_tensor([_log_sum_prior_single(logm, math.log(p), Kc) for p in p_marg], dtype=F64, device=device)
    ell = null[:, :, None] + logZ - lsp[:, None, None]  # [2, ne, ns]
    best_s = ell.max(dim=2)
    tot = best_s.values.sum(0)
    e_star = int(torch.argmax(tot))
    s1 = float(grid.sigma[int(best_s.indices[0, e_star])])
    s2 = float(grid.sigma[int(best_s.indices[1, e_star])])
    return s1, s2, float(grid.eta[e_star]), ell.cpu().numpy()


# ---------------------------------------------------------------------------------------------------------
# fused single pass: logZ and E[lambda lambda' | z] with closed-form 4x4 algebra (fast path)
# ---------------------------------------------------------------------------------------------------------
def _kernel_moments(T: LocusTensors, V, m, kx):
    """Same algebra as enum.kernel (log prior + log BF), plus the term's posterior second moments
    E[beta_c beta_d | z, term] for c <= d (10 entries), all elementwise on [nB, P] (no batched linalg).
    Posterior: mean = G' w, Cov = G' G with G = l^-1 L', w = l^-1 L' b_s, l = chol(I + L' Phi_ss L)."""
    from .enum import kernel
    dt = F64
    logw, _ = kernel(T, V, m, kx, dtype=dt)
    _, _, _, _, flags = structure(V, m)
    mf = m.to(dt)
    sdx = [T.sd[:, TX[x], V[:, x], kx[:, x]].to(dt) * mf[:, x] for x in range(4)]
    Le = L_entries(sdx, [f.to(dt) for f in flags], T.rho_b.to(dt)[:, None])
    Lm = [[None] * 4 for _ in range(4)]  # Lm[r][c] = L[r][c] (lower), None = 0
    for (r, c), k in (((0, 0), "L00"), ((1, 1), "L11"), ((2, 0), "L20"), ((2, 1), "L21"), ((2, 2), "L22"),
                      ((3, 0), "L30"), ((3, 1), "L31"), ((3, 3), "L33")):
        Lm[r][c] = Le[k]
    M = T.M
    if T.Phi is not None:
        gi = [TX[x] * M + V[:, x] for x in range(4)]
        Ph = lambda x, y: T.Phi[:, gi[x], gi[y]]  # noqa: E731
    else:
        Ph = lambda x, y: T.Vinv[:, TX[x], TX[y]][:, None] * T.Rt[:, V[:, x], V[:, y]]  # noqa: E731
    P_ = {}
    for x in range(4):
        for y in range(x, 4):
            P_[(x, y)] = P_[(y, x)] = Ph(x, y)
    # H = Phi L ; A = I + L' H
    H = {}
    for x in range(4):
        for j in range(4):
            acc = None
            for k in range(4):
                if Lm[k][j] is not None:
                    t = P_[(x, k)] * Lm[k][j]
                    acc = t if acc is None else acc + t
            H[(x, j)] = acc
    A = {}
    for i in range(4):
        for j in range(i, 4):
            acc = None
            for k in range(4):
                if Lm[k][i] is not None:
                    t = Lm[k][i] * H[(k, j)]
                    acc = t if acc is None else acc + t
            A[(i, j)] = acc + (1.0 if i == j else 0.0)
    bs = [T.b[:, TX[x], V[:, x]].to(dt) for x in range(4)]
    y = []
    for i in range(4):
        acc = None
        for k in range(4):
            if Lm[k][i] is not None:
                t = Lm[k][i] * bs[k]
                acc = t if acc is None else acc + t
        y.append(acc)
    # Cholesky l of A
    l = {}
    for i in range(4):
        for j in range(i + 1):
            s = A[(j, i)]
            for k in range(j):
                s = s - l[(i, k)] * l[(j, k)]
            l[(i, j)] = torch.sqrt(s) if i == j else s / l[(j, j)]
    # li = l^-1
    li = {}
    for i in range(4):
        li[(i, i)] = 1.0 / l[(i, i)]
        for j in range(i):
            s = None
            for k in range(j, i):
                t = l[(i, k)] * li[(k, j)]
                s = t if s is None else s + t
            li[(i, j)] = -s * li[(i, i)]
    # w = l^-1 y ; G = l^-1 L'  (G[i][c] = sum_k li[i][k] L[c][k])
    w = []
    for i in range(4):
        acc = None
        for k in range(i + 1):
            t = li[(i, k)] * y[k]
            acc = t if acc is None else acc + t
        w.append(acc)
    G = {}
    for i in range(4):
        for c in range(4):
            acc = None
            for k in range(i + 1):
                if Lm[c][k] is not None:
                    t = li[(i, k)] * Lm[c][k]
                    acc = t if acc is None else acc + t
            G[(i, c)] = acc if acc is not None else torch.zeros_like(w[0])
    mean = []
    for c in range(4):
        acc = G[(0, c)] * w[0]
        for i in range(1, 4):
            acc = acc + G[(i, c)] * w[i]
        mean.append(acc)
    mom = {}
    for c in range(4):
        for d in range(c, 4):
            acc = mean[c] * mean[d]
            for i in range(4):
                acc = acc + G[(i, c)] * G[(i, d)]
            mom[(c, d)] = acc
    return logw, mom


def logz_and_moments(T: LocusTensors, ts: TermSet, chunk_elems: int = 1 << 21, min_logw_rel: float = -40.0):
    """One streaming pass: logZ [nB] and E[lambda lambda' | z] [nB, 2M, 2M]. Terms whose log mass is below the
    running maximum + min_logw_rel are not accumulated into the moments (each such term has relative mass
    < e^-40 = 4e-18; at most n_terms of them: total omitted mass < 1e-11 for M = 50); logZ is exact."""
    if ts.offsets is not None:
        raise NotImplementedError("B-mm uses single-component (Gaussian) priors")
    dev = T.b.device
    nB, M = T.nB, T.M
    M2 = 2 * M
    S = torch.zeros(nB, M2 * M2, dtype=F64, device=dev)
    run = torch.full((nB,), -math.inf, dtype=F64, device=dev)
    Zs = torch.zeros(nB, dtype=F64, device=dev)  # sum exp(logw - run)
    per = max(1, chunk_elems // max(1, nB))
    pairs = [(c, d) for c in range(4) for d in range(c, 4)]
    for s in range(0, ts.n_terms, per):
        t = torch.arange(s, min(ts.n_terms, s + per), device=dev, dtype=torch.int64)
        V, m, kx = ts.decode(t)
        logw, mom = _kernel_moments(T, V, m, kx)
        cmax = logw.max(dim=1).values
        new = torch.maximum(run, cmax)
        fac = torch.where(torch.isfinite(run), torch.exp(run - new), torch.zeros_like(run))
        Zs = Zs * fac
        S *= fac[:, None]
        run = new
        rel = logw - run[:, None]
        wt = torch.exp(rel)
        Zs = Zs + wt.sum(1)
        keep = (rel > min_logw_rel).any(dim=0)
        if not bool(keep.any()):
            continue
        kidx = torch.nonzero(keep).flatten()
        wt = wt[:, kidx]
        Vk, mk = V[kidx], m[kidx]
        gi = [TX[x] * M + Vk[:, x] for x in range(4)]
        for c, d in pairs:
            ok = mk[:, c] & mk[:, d]
            if not bool(ok.any()):
                continue
            val = torch.where(ok[None], mom[(c, d)][:, kidx] * wt, 0.0)
            S.index_add_(1, gi[c] * M2 + gi[d], val)
            if c != d:
                S.index_add_(1, gi[d] * M2 + gi[c], val)
    logZ = run + torch.log(Zs)
    S = S / Zs[:, None]
    S = S.view(nB, M2, M2)
    return logZ, 0.5 * (S + S.mT)
