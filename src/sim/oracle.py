"""Oracle power and the precedence-ordered oracle-class partition (DESIGN v4 §1). HASH-PROTECTED (§7.4).

Inputs per locus (all on the *realised* study samples):
  c_t    causal variant identifiers of trait t (any hashable ints; shared variants appear in both)
  lam_t  own effects lambda_tj = sqrt(N_t) * beta_std,j = +-sqrt(N_t q_j), q = per-variant PVE (D6(b))
  mu_t   realised marginal noncentralities mu_tj = E[z_tj | study-t genotypes, beta_t] at c_t
  r2     function (a, b) -> r² in R_true (pooled target-population LD)
  kind_t "gwas" (alpha 5e-8, tau 6.29, w 4.61) or "mol" (alpha 1e-5, tau 5.26, w 3.58); thresholds follow
         the trait TYPE (DESIGN v4 §1): P1/P2 gwas x mol, P3 eQTL x eQTL mol x mol, P3 lab x lab gwas x gwas.
         REQUIRED arguments since v4 (no default), so a P3 caller cannot silently get the P1 thresholds.

Definitions (§1): powered_t(j) <=> |mu_tj| >= tau_t and |lam_tj| >= tau_t; nonweak_t(j) <=> |mu_tj| >= w_t
(power >= 0.2). Precedence: S > U > D > W > G.
  S  some shared variant is powered in both traits;
  U  each trait has a powered variant, none shared-and-powered, class-determining pair has d < 2;
  D  each trait has a powered variant and no shared variant is nonweak in both traits;
  W  some trait has no nonweak causal variant;
  G  everything else.
Class-determining pair for D/U: the cross-trait pair (a in powered_1, b in powered_2) with maximal r²
(ties: smaller d, then smaller ids); d = min(|lam_1a|, |lam_2b|) * sqrt(1 - r²).
Class-determining (mu1, mu2), stored as ABSOLUTE values:
  S: the shared powered variant with the largest min-power; D/U: the pair; W and G: the strongest causal
  variant (max |mu|) per trait, 0 if the trait has none (G is not defined in §1: README D8).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.stats import norm

from sim.params import ALPHA, D_MIN, TAU, WEAK

CLASSES = ("S", "U", "D", "W", "G")


def power(mu, alpha: float) -> np.ndarray:
    """P(|z| > z_{alpha/2}) for z ~ N(mu, 1) (two-sided)."""
    za = norm.isf(alpha / 2)
    m = np.abs(np.asarray(mu, dtype=np.float64))
    return norm.sf(za - m) + norm.cdf(-za - m)


@dataclass
class OracleResult:
    cls: str
    mu1: float          # class-determining |mu| trait 1
    mu2: float
    r2: float           # r² of class-determining pair (D/U), NaN otherwise
    d: float            # identifiability index (D/U), NaN otherwise
    cd1: int | None     # class-determining variant trait 1 (None if none)
    cd2: int | None
    n_powered1: int
    n_powered2: int


def oracle_class(c1: Sequence[int], lam1: Sequence[float], mu1: Sequence[float],
                 c2: Sequence[int], lam2: Sequence[float], mu2: Sequence[float],
                 r2: Callable[[int, int], float], kind1: str, kind2: str) -> OracleResult:
    if kind1 not in TAU or kind2 not in TAU:
        raise ValueError(f"unknown trait types {kind1!r}, {kind2!r} (expected 'gwas' or 'mol')")
    tau1, tau2, w1, w2 = TAU[kind1], TAU[kind2], WEAK[kind1], WEAK[kind2]
    a1, a2 = ALPHA[kind1], ALPHA[kind2]
    L1 = {int(j): (abs(float(l)), abs(float(m))) for j, l, m in zip(c1, lam1, mu1)}
    L2 = {int(j): (abs(float(l)), abs(float(m))) for j, l, m in zip(c2, lam2, mu2)}
    assert len(L1) == len(c1) and len(L2) == len(c2), "duplicate causal ids within a trait"
    pw1 = {j for j, (l, m) in L1.items() if m >= tau1 and l >= tau1}
    pw2 = {j for j, (l, m) in L2.items() if m >= tau2 and l >= tau2}
    nw1 = {j for j, (l, m) in L1.items() if m >= w1}
    nw2 = {j for j, (l, m) in L2.items() if m >= w2}
    shared = set(L1) & set(L2)

    def strongest(L):
        if not L:
            return None, 0.0
        j = max(sorted(L), key=lambda k: L[k][1])
        return j, L[j][1]

    # 1. S
    sp = sorted(shared & pw1 & pw2)
    if sp:
        j = max(sp, key=lambda k: min(power(L1[k][1], a1), power(L2[k][1], a2)))
        return OracleResult("S", L1[j][1], L2[j][1], np.nan, np.nan, j, j, len(pw1), len(pw2))
    if pw1 and pw2:
        best = None
        for a in sorted(pw1):
            for b in sorted(pw2):
                if a == b:
                    continue  # cannot happen after S, kept for safety
                rr = float(np.clip(r2(a, b), 0.0, 1.0))
                d = min(L1[a][0], L2[b][0]) * np.sqrt(max(0.0, 1.0 - rr))
                key = (-rr, d, a, b)
                if best is None or key < best[0]:
                    best = (key, a, b, rr, d)
        _, a, b, rr, d = best
        # 2. U
        if d < D_MIN:
            return OracleResult("U", L1[a][1], L2[b][1], rr, d, a, b, len(pw1), len(pw2))
        # 3. D (no shared variant nonweak in both traits)
        if not (shared & nw1 & nw2):
            return OracleResult("D", L1[a][1], L2[b][1], rr, d, a, b, len(pw1), len(pw2))
    j1, m1 = strongest(L1)
    j2, m2 = strongest(L2)
    # 4. W
    if not nw1 or not nw2:
        return OracleResult("W", m1, m2, np.nan, np.nan, j1, j2, len(pw1), len(pw2))
    # 5. G
    return OracleResult("G", m1, m2, np.nan, np.nan, j1, j2, len(pw1), len(pw2))


def r2_bin(r2: float, bins) -> int:
    """Index of the half-open bin [lo, hi) containing r2 (last bin closed); -1 if NaN."""
    if not np.isfinite(r2):
        return -1
    for k, (lo, hi) in enumerate(bins):
        if lo <= r2 < hi or (k == len(bins) - 1 and r2 == hi):
            return k
    return -1
