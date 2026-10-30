"""DESIGN Sec. 7.1(4): hand-computed 2-SNP toy example.

Two SNPs with LD r, rho = 0 and independent effects (rho_b = 0), so BF(S1, S2) = BF_1(S1) BF_2(S2) with
closed forms (eigen-decomposition of R = [[1, r], [r, 1]], eigenvalues 1 +- r, eigenvectors (1, +-1)/sqrt2):
  BF_t({})    = 1
  BF_t({j})   = (1 + s^2)^-1/2 exp( z_j^2 s^2 / (2 (1 + s^2)) )
  BF_t({1,2}) = [(1 + s^2 (1+r)) (1 + s^2 (1-r))]^-1/2
                exp( (z1+z2)^2 / (4 (1 + r + 1/s^2)) + (z1-z2)^2 / (4 (1 - r + 1/s^2)) )
plus a rho != 0 single-shared-variant case computed from the conditional likelihood
  z1 - rho z2 | z2 ~ N(lambda_1 ..., .)  (independent derivation of the 2x2 support form).
Configuration prior: per-SNP p1, p2, p12 (coloc convention). All 9 x 9 configurations enumerated by hand.
"""
import itertools
import math

import numpy as np
import torch

from famB.enum import TermSet, enumerate_batch, kernel
from famB.priors import type_table_per_snp
from ref import tensors_for, term_arrays


def bf_single_trait(z, r, s, S):
    S = tuple(sorted(S))
    if len(S) == 0:
        return 1.0
    if len(S) == 1:
        zj = z[S[0]]
        return (1 + s * s) ** -0.5 * math.exp(zj * zj * s * s / (2 * (1 + s * s)))
    det = (1 + s * s * (1 + r)) * (1 + s * s * (1 - r))
    q = (z[0] + z[1]) ** 2 / (4 * (1 + r + 1 / s ** 2)) + (z[0] - z[1]) ** 2 / (4 * (1 - r + 1 / s ** 2))
    return det ** -0.5 * math.exp(q)


def test_hand_computed_two_snp(device):
    r = 0.5
    R = np.array([[1.0, r], [r, 1.0]])
    z1 = np.array([5.0, 1.0])
    z2 = np.array([4.0, 0.5])
    s1, s2 = 3.0, 2.0
    p1, p2, p12 = 1e-3, 2e-3, 5e-4
    sets = [(), (0,), (1,), (0, 1)]
    # --- by hand
    W = {}
    for S1, S2 in itertools.product(sets, sets):
        nsh = len(set(S1) & set(S2))
        n1, n2 = len(S1) - nsh, len(S2) - nsh
        prior = p1 ** n1 * p2 ** n2 * p12 ** nsh
        W[(S1, S2)] = prior * bf_single_trait(z1, r, s1, S1) * bf_single_trait(z2, r, s2, S2)
    Z = sum(W.values())

    def cat(S1, S2):
        sh = set(S1) & set(S2)
        if not S1 and not S2: return "H0"
        if S1 and not S2: return "H1"
        if S2 and not S1: return "H2"
        if not sh: return "H3"
        if set(S1) == set(S2): return "H4"
        return "H34"

    PPh = {h: 0.0 for h in ("H0", "H1", "H2", "H3", "H4", "H34")}
    for k, w in W.items():
        PPh[cat(*k)] += w / Z
    pip1 = [sum(w for (S1, _), w in W.items() if j in S1) / Z for j in (0, 1)]
    # --- engine (eta = 0 so that Rt = R exactly as in the hand formulas)
    sd = np.stack([np.full(2, s1), np.full(2, s2)])
    T = tensors_for(np.r_[z1, z2], R, 0.0, sd, 0.0, device, logprior_type=type_table_per_snp(p1, p2, p12, 2), eta=0.0)
    ts = TermSet(2, 2, device=device)
    res = enumerate_batch(T, ts, n_samples=0)
    names = ("H0", "H1", "H2", "H3", "H4", "H34")
    for c, h in enumerate(names):
        assert abs(float(res["PP"][0, c]) - PPh[h]) < 1e-12, (h, float(res["PP"][0, c]), PPh[h])
    for j in (0, 1):
        assert abs(float(res["pip"][0, 0, j]) - pip1[j]) < 1e-12
    # spot values printed for the record
    print({h: round(v, 6) for h, v in PPh.items()})


def test_hand_rho_nonzero_shared(device):
    """rho != 0: single variant active in both traits, D = diag(s1^2, s2^2), versus the 2-dim Gaussian
    marginal of (z1j, z2j) written out by hand, and the single-trait-1 BF via the conditional likelihood."""
    rho, s1, s2 = 0.6, 4.0, 3.0
    zj = np.array([6.0, 4.5])
    R = np.eye(1)
    # (z1, z2) ~ N(0, V + diag(s^2)) vs N(0, V)
    V = np.array([[1, rho], [rho, 1]])

    def lnorm(cov):
        det = cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2
        inv = np.array([[cov[1, 1], -cov[0, 1]], [-cov[0, 1], cov[0, 0]]]) / det
        return -0.5 * zj @ inv @ zj - 0.5 * math.log(det)

    hand_shared = lnorm(V + np.diag([s1 * s1, s2 * s2])) - lnorm(V)
    # trait-1-only config: z1 - rho z2 ~ N(lambda, 1 - rho^2) given z2 (z2 has no signal)
    x = zj[0] - rho * zj[1]
    v0 = 1 - rho * rho
    hand_t1 = (-0.5 * x * x / (v0 + s1 * s1) - 0.5 * math.log(v0 + s1 * s1)) - (-0.5 * x * x / v0 - 0.5 * math.log(v0))
    sd = np.stack([np.full(1, s1), np.full(1, s2)])
    T = tensors_for(zj, R, rho, sd, 0.0, device, eta=0.0)
    Vt, m, kx = term_arrays([((0,), (0,)), ((0,), ())], device)
    lw, _ = kernel(T, Vt, m, kx)
    assert abs(float(lw[0, 0]) - hand_shared) < 1e-12
    assert abs(float(lw[0, 1]) - hand_t1) < 1e-12
