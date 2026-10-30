"""Per-trait-union screen for ExactColoc (DESIGN v3 Sec. 5 B "Screen").

Candidate lists (priority order within each list):
  m_t : variants by marginal |z_t|                                         (t = 1, 2)
  c_t : variants by the lead-removed (conditional) statistic of trait t:
          zc_tj = (z_tj - Rt_jl z_tl) / sqrt(1 - Rt_jl^2),   l = argmax |z_t|,  Rt = (1-eta) R + eta I
        i.e. z after removing the lead's LD-projected effect, standardised to unit null variance
        (the raw residual lambda_j (1 - r^2) would bury a secondary signal in high LD with the lead).
Quotas: each trait contributes M/2 new representatives: M/2 - k_cond from m_t and k_cond from c_t
(default k_cond = M/8, i.e. 25% of the trait's budget; "cap of the proxy expansion"). Lists are visited
round-robin (m1, m2, c1, c2); a candidate already represented does not consume quota, so overlap between
the traits' lists is back-filled from deeper in the same list. Total = min(M, number of groups).

Strength-aware duplicate collapse: candidate j joins representative i (no new slot) iff
    zmax^2 (1 - r_ij^2) < 1,   zmax = max over the window and both traits of |z|
(unidentifiable at the observed signal strength: d = zmax sqrt(1 - r^2) < 1). After selection every
non-selected window variant satisfying the rule with some representative is assigned to the most
correlated one; m_i = 1 + group size is the multiplicity used in the configuration prior.
If p <= M no screen and no collapse is applied (exact).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class LDAccess:
    """Row access to the inference LD R (dense [p,p] or factor X_ref [n,p] with R = X'X/n)."""

    def __init__(self, R=None, X=None, n_ref=None):
        if (R is None) == (X is None):
            raise ValueError("give exactly one of R (dense) or X (factor)")
        self.R = None if R is None else np.asarray(R)
        self.X = None if X is None else np.asarray(X, dtype=np.float64)
        self.n = None if X is None else float(n_ref if n_ref is not None else self.X.shape[0])
        self.p = (self.R if self.R is not None else self.X).shape[1]

    @classmethod
    def from_locus(cls, locus):
        if locus.ld_type == "dense":
            return cls(R=locus.R)
        return cls(X=locus.X_ref, n_ref=locus.n_ref)

    def rows(self, idx) -> np.ndarray:
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        if self.R is not None:
            return np.asarray(self.R[idx], dtype=np.float64)
        return (self.X[:, idx].T @ self.X) / self.n

    def sub(self, idx) -> np.ndarray:
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        if self.R is not None:
            return np.asarray(self.R[np.ix_(idx, idx)], dtype=np.float64)
        Xi = self.X[:, idx]
        return (Xi.T @ Xi) / self.n


@dataclass
class ScreenResult:
    sel: np.ndarray  # [M'] window indices of representatives
    mult: np.ndarray  # [M'] multiplicity (>= 1)
    group: np.ndarray  # [p] index into sel of the group a window variant belongs to, -1 if none
    zmax: float
    source: np.ndarray  # [M'] which list added it: 0 m1, 1 m2, 2 c1, 3 c2, 4 all (p <= M)


def conditional_z(z: np.ndarray, r_lead: np.ndarray, lead: int, eta: float, n_ref_infl: float | None = None) -> np.ndarray:
    """Lead-removed statistic. With n_ref_infl = n (reference size) the null variance is inflated by the
    LD-estimation error of the reference, Var(r_hat - r) ~ (1 - r^2)^2 / n, times z_lead^2."""
    rt = (1.0 - eta) * r_lead
    rt[lead] = 1.0
    var = np.clip(1.0 - rt * rt, 1e-12, None)
    if n_ref_infl:
        var = var + z[lead] ** 2 * (1.0 - rt * rt) ** 2 / float(n_ref_infl)
    zc = (z - rt * z[lead]) / np.sqrt(var)
    zc[lead] = 0.0
    return zc


def screen(z1: np.ndarray, z2: np.ndarray, ld: LDAccess, M: int = 100, k_cond: int | None = None,
           eta: float = 1e-3, collapse: bool = True, n_ref_infl: float | None = None) -> ScreenResult:
    z1 = np.asarray(z1, np.float64)
    z2 = np.asarray(z2, np.float64)
    p = len(z1)
    zmax = float(max(np.max(np.abs(z1)), np.max(np.abs(z2))))
    if p <= M:
        return ScreenResult(sel=np.arange(p), mult=np.ones(p), group=np.arange(p), zmax=zmax,
                            source=np.full(p, 4))
    if k_cond is None:
        k_cond = max(1, M // 8)
    half = M // 2
    quota = np.array([half - k_cond, M - half - k_cond, k_cond, k_cond])
    l1, l2 = int(np.argmax(np.abs(z1))), int(np.argmax(np.abs(z2)))
    lead_rows = ld.rows([l1, l2])
    zc1 = conditional_z(z1, lead_rows[0].copy(), l1, eta, n_ref_infl)
    zc2 = conditional_z(z2, lead_rows[1].copy(), l2, eta, n_ref_infl)
    lists = [np.argsort(-np.abs(z1), kind="stable"), np.argsort(-np.abs(z2), kind="stable"),
             np.argsort(-np.abs(zc1), kind="stable"), np.argsort(-np.abs(zc2), kind="stable")]
    thr = 1.0 / zmax ** 2 if (collapse and zmax > 0) else 0.0  # collapse iff 1 - r^2 < thr
    pos = [0, 0, 0, 0]
    taken = np.zeros(4, dtype=np.int64)
    sel: list[int] = []
    src: list[int] = []
    rows_sel = np.zeros((M, p))  # LD rows of the representatives (needed for collapse + multiplicity)
    covered = np.zeros(p, dtype=bool)  # already a representative or collapsed into one
    # row cache for candidates: fetch in blocks
    while len(sel) < M:
        progressed = False
        for li in range(4):
            if taken[li] >= quota[li] or len(sel) >= M:
                continue
            L = lists[li]
            while pos[li] < p and covered[L[pos[li]]]:
                pos[li] += 1
            if pos[li] >= p:
                continue
            j = int(L[pos[li]])
            pos[li] += 1
            progressed = True
            if thr > 0 and sel:
                r = rows_sel[: len(sel), j]
                if np.any(1.0 - r * r < thr):
                    covered[j] = True
                    continue
            rows_sel[len(sel)] = ld.rows([j])[0]
            sel.append(j)
            src.append(li)
            covered[j] = True
            taken[li] += 1
        if not progressed:
            # quotas exhausted for some lists (or lists exhausted): reassign remaining budget to any list
            if np.all([pos[i] >= p for i in range(4)]):
                break
            quota = quota + (M - len(sel))  # let all lists continue round-robin
    sel_a = np.asarray(sel, dtype=np.int64)
    Rs = rows_sel[: len(sel)]
    group = np.full(p, -1, dtype=np.int64)
    group[sel_a] = np.arange(len(sel_a))
    mult = np.ones(len(sel_a))
    if thr > 0:
        r2 = Rs * Rs  # [M', p]
        cand = (1.0 - r2) < thr
        cand[:, sel_a] = False
        hit = cand.any(axis=0)
        best = np.argmax(np.where(cand, r2, -1.0), axis=0)
        group[hit] = best[hit]
        mult = 1.0 + np.bincount(best[hit], minlength=len(sel_a)).astype(np.float64)
    return ScreenResult(sel=sel_a, mult=mult, group=group, zmax=zmax, source=np.asarray(src))
