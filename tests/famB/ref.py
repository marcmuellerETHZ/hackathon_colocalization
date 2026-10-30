"""Independent reference implementations for Family B tests (not used by src/famB)."""
from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import torch

from famB.enum import LocusTensors


def coords_of(S1, S2):
    return [(0, j) for j in S1] + [(1, j) for j in S2]


def prior_cov(S1, S2, sd, rho_b):
    """D over active coords; sd [2, p]."""
    cs = coords_of(S1, S2)
    D = np.zeros((len(cs), len(cs)))
    for a, (t, j) in enumerate(cs):
        for c, (u, k) in enumerate(cs):
            if j == k:
                D[a, c] = sd[t, j] * sd[u, k] * (1.0 if t == u else rho_b)
    return cs, D


def full_form_logbf(z, Rt, rho, S1, S2, sd, rho_b):
    """Full 2p-dimensional RSS marginal likelihood ratio in fp64:
    z ~ N(X_s beta_s, V (x) Rt), X = I2 (x) Rt, beta_s ~ N(0, D)."""
    p = Rt.shape[0]
    V = np.array([[1.0, rho], [rho, 1.0]])
    S0 = np.kron(V, Rt)
    X = np.kron(np.eye(2), Rt)
    cs, D = prior_cov(S1, S2, sd, rho_b)
    Xs = X[:, [t * p + j for t, j in cs]]
    S1m = S0 + Xs @ D @ Xs.T

    def lp(S):
        c = np.linalg.cholesky(S)
        a = np.linalg.solve(c, z)
        return -0.5 * a @ a - np.log(np.diag(c)).sum()

    return lp(S1m) - lp(S0)


def full_form_logbf_mp(z, Rt, rho, S1, S2, sd, rho_b, dps=60):
    """Same in mpmath extended precision (reference for near-singular / rank-deficient Rt)."""
    with mp.workdps(dps):
        p = Rt.shape[0]
        V = np.array([[1.0, rho], [rho, 1.0]])
        S0 = np.kron(V, Rt)
        X = np.kron(np.eye(2), Rt)
        cs, D = prior_cov(S1, S2, sd, rho_b)
        Xs = X[:, [t * p + j for t, j in cs]]
        S0m = mp.matrix(S0.tolist())
        Xm = mp.matrix(Xs.tolist())
        Dm = mp.matrix(D.tolist())
        S1m = S0m + Xm * Dm * Xm.T
        zm = mp.matrix(z.tolist())

        def lp(S):
            sol = mp.lu_solve(S, zm)
            q = (zm.T * sol)[0]
            return -q / 2 - mp.log(mp.det(S)) / 2

        return float(lp(S1m) - lp(S0m))


def active_only_logbf(z, Rt, rho, S1, S2, sd, rho_b):
    """WRONG variant (negative control): likelihood of the active z's only."""
    p = Rt.shape[0]
    cs, D = prior_cov(S1, S2, sd, rho_b)
    idx = [t * p + j for t, j in cs]
    V = np.array([[1.0, rho], [rho, 1.0]])
    S0 = np.kron(V, Rt)[np.ix_(idx, idx)]
    X = np.kron(np.eye(2), Rt)[np.ix_(idx, idx)]
    zs = z[idx]

    def lp(S):
        c = np.linalg.cholesky(S)
        a = np.linalg.solve(c, zs)
        return -0.5 * a @ a - np.log(np.diag(c)).sum()

    return lp(S0 + X @ D @ X.T) - lp(S0)


def tensors_for(z, Rt, rho, sd, rho_b, device, logprior_type=None, logm=None, eta=1e-3):
    """LocusTensors (nB = 1) for a full-p model (no screen)."""
    p = Rt.shape[0]
    V = np.array([[1.0, rho], [rho, 1.0]])
    Vi = np.linalg.inv(V)
    zz = np.stack([z[:p], z[p:]])
    b = Vi @ zz
    f = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=device)
    lpt = np.zeros((3, 3, 3)) if logprior_type is None else logprior_type
    return LocusTensors(Rt=f(Rt)[None], b=f(b)[None], Vinv=f(Vi)[None], sd=f(sd)[None, :, :, None],
                        logwk=torch.zeros(1, 2, 1, dtype=torch.float64, device=device),
                        rho_b=torch.tensor([rho_b], dtype=torch.float64, device=device),
                        logm=f(np.zeros(p) if logm is None else logm)[None], logprior_type=f(lpt)[None],
                        eta=torch.tensor([eta], dtype=torch.float64, device=device))


def term_arrays(configs, device):
    """configs: list of (S1, S2) -> V, m, kx tensors for enum.kernel."""
    V, m = [], []
    for S1, S2 in configs:
        S1 = sorted(S1)
        S2 = sorted(S2)
        row = [S1[0] if len(S1) > 0 else -1, S1[1] if len(S1) > 1 else -1,
               S2[0] if len(S2) > 0 else -1, S2[1] if len(S2) > 1 else -1]
        V.append([max(0, v) for v in row])
        m.append([v >= 0 for v in row])
    V = torch.tensor(V, dtype=torch.int64, device=device)
    m = torch.tensor(m, dtype=torch.bool, device=device)
    return V, m, torch.zeros_like(V)


def genotype_ld(rng, n, p, dup_pairs=(), ar=0.7):
    X = rng.standard_normal((n, p))
    for j in range(1, p):
        X[:, j] = ar * X[:, j - 1] + math.sqrt(1 - ar * ar) * X[:, j]
    for i, j, noise in dup_pairs:
        X[:, j] = X[:, i] + noise * rng.standard_normal(n)
    X = (X - X.mean(0)) / X.std(0)
    return X.T @ X / n
