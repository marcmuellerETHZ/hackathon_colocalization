"""Toy L0*-like world (until src/sim provides the real L0* generator; DESIGN Sec. 5 B simulator-prior mode).

Exactly the model ExactColoc assumes, so its posteriors must be calibrated here:
  * short windows (p <= 30), one LD matrix R for both traits (factor form, the R given to the method),
    Rt = (1 - eta) R + eta I used in mean and covariance;
  * configuration (S1, S2) drawn from a type table pi(n_sh, n_1only, n_2only) with uniform variant choice
    (simulator mode), or from the per-SNP state prior restricted to <= 2 per trait (gaussian mode);
  * effects: K-component scale mixture per (trait, variant) (simulator mode) or N(0, D) with shared
    correlation rho_b (gaussian mode);
  * z ~ N((I2 (x) Rt) beta, V (x) Rt), V = [[1, rho], [rho, 1]], rho known (rho_hat = rho).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from toy_data import ar_genotypes, make_locus


def _draw_z(rng, Rt, rho, lam1, lam2):
    p = Rt.shape[0]
    C = np.linalg.cholesky(Rt)
    e1 = C @ rng.standard_normal(p)
    e2 = rho * e1 + math.sqrt(1 - rho * rho) * (C @ rng.standard_normal(p))
    return Rt @ lam1 + e1, Rt @ lam2 + e2


def _choose_config(rng, p, nsh, n1, n2):
    v = rng.choice(p, nsh + n1 + n2, replace=False)
    sh, o1, o2 = v[:nsh], v[nsh:nsh + n1], v[nsh + n1:]
    return sorted(np.r_[sh, o1].astype(int)), sorted(np.r_[sh, o2].astype(int))


def simulate_simprior(rng, prior, n_loci, p=10, rho_choices=(0.0, 0.4), eta=1e-3, n_ref=400):
    """Returns (loci, truths); truth = dict(S1, S2, lam1, lam2, R, rho)."""
    tp = np.asarray(prior.type_prob)
    types = [(a, b, c) for a in range(3) for b in range(3) for c in range(3) if tp[a, b, c] > 0]
    probs = np.array([tp[t] for t in types]); probs /= probs.sum()
    loci, truths = [], []
    for i in range(n_loci):
        X = ar_genotypes(rng, n_ref, p, ar=float(rng.uniform(0.3, 0.97)))
        R = X.T @ X / n_ref
        Rt = (1 - eta) * R + eta * np.eye(p)
        nsh, n1, n2 = types[rng.choice(len(types), p=probs)]
        S1, S2 = _choose_config(rng, p, nsh, n1, n2)
        lam1 = np.zeros(p); lam2 = np.zeros(p)
        for t, S, lam in ((0, S1, lam1), (1, S2, lam2)):
            for j in S:
                k = rng.choice(len(prior.mix_w[t]), p=prior.mix_w[t])
                lam[j] = rng.normal(0, prior.mix_sd[t][k])
        rho = float(rng.choice(rho_choices))
        z1, z2 = _draw_z(rng, Rt, rho, lam1, lam2)
        lo = make_locus(f"T{i:05d}", X, lam1, lam2, rho=rho, rng=rng, factor=True, z_override=(z1, z2))
        loci.append(lo)
        truths.append(dict(S1=S1, S2=S2, lam1=lam1, lam2=lam2, R=R, rho=rho, eta=eta))
    return loci, truths


def simulate_gaussian(rng, n_loci, p=8, p1=0.04, p2=0.04, p12=0.03, sig=(5.0, 4.0), rho_b=0.5,
                      rho_choices=(0.0, 0.4, -0.7), eta=1e-3, n_ref=400):
    """Configurations from the per-SNP state prior restricted to <= 2 per trait (exact enumeration)."""
    sets = [()] + [(j,) for j in range(p)] + list(itertools.combinations(range(p), 2))
    cfgs, w = [], []
    for S1 in sets:
        for S2 in sets:
            nsh = len(set(S1) & set(S2))
            w.append(p1 ** (len(S1) - nsh) * p2 ** (len(S2) - nsh) * p12 ** nsh)
            cfgs.append((S1, S2))
    w = np.array(w) / np.sum(w)
    loci, truths = [], []
    for i in range(n_loci):
        X = ar_genotypes(rng, n_ref, p, ar=float(rng.uniform(0.3, 0.97)))
        R = X.T @ X / n_ref
        Rt = (1 - eta) * R + eta * np.eye(p)
        S1, S2 = cfgs[rng.choice(len(cfgs), p=w)]
        lam1 = np.zeros(p); lam2 = np.zeros(p)
        for j in set(S1) | set(S2):
            if j in S1 and j in S2:
                c = np.array([[sig[0] ** 2, rho_b * sig[0] * sig[1]], [rho_b * sig[0] * sig[1], sig[1] ** 2]])
                lam1[j], lam2[j] = rng.multivariate_normal([0, 0], c)
            elif j in S1:
                lam1[j] = rng.normal(0, sig[0])
            else:
                lam2[j] = rng.normal(0, sig[1])
        rho = float(rng.choice(rho_choices))
        z1, z2 = _draw_z(rng, Rt, rho, lam1, lam2)
        loci.append(make_locus(f"G{i:05d}", X, lam1, lam2, rho=rho, rng=rng, factor=True, z_override=(z1, z2)))
        truths.append(dict(S1=list(S1), S2=list(S2), lam1=lam1, lam2=lam2, R=R, rho=rho, eta=eta))
    return loci, truths


def true_category(S1, S2):
    sh = set(S1) & set(S2)
    if not S1 and not S2: return "H0"
    if S1 and not S2: return "H1"
    if S2 and not S1: return "H2"
    if not sh: return "H3"
    if set(S1) == set(S2): return "H4"
    return "H34"
