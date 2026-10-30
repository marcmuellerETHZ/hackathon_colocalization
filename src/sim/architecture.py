"""Causal architecture: the locus mixture of DESIGN §2.4 (the training prior) and effect-size draws.

An architecture is drawn on the locus' analysis variant set (indices 0..p-1) and returns causal sets and
effect *specifications*; the generator turns PVEs into lambda on the realised samples.

Effect spec per (trait, causal variant):  ("pve", q, sign)  -> beta_std = sign * sqrt(q), lambda = sqrt(N q)
                                          (q = per-variant PVE; decision D6(b) notation)
                                          ("lam", lam)      -> lambda given directly (sub-powered W partner)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim import params


@dataclass
class Architecture:
    ltype: str                         # generated locus type S/D/W/M
    c1: list[int]
    c2: list[int]
    eff1: dict = field(default_factory=dict)   # variant -> spec
    eff2: dict = field(default_factory=dict)
    strong1: bool = False              # biobank-strong stratum used for trait 1 (GWAS kind)
    strong2: bool = False
    bg: bool = False                   # infinitesimal background
    target_r2_bin: int = -1            # D: requested r² bin for the partner


def maf_bins(maf: np.ndarray) -> list[np.ndarray]:
    return [np.where((maf >= lo) & (maf < hi if hi < 0.5 else maf <= hi))[0] for lo, hi in params.CAUSAL_MAF_BINS]


def pick_by_maf(g: np.random.Generator, bins: list[np.ndarray], exclude: set) -> int:
    """Causal MAF uniform over MAF bins: choose a non-empty bin uniformly, then a variant uniformly."""
    cand = [np.setdiff1d(b, list(exclude)) if exclude else b for b in bins]
    nonempty = [c for c in cand if len(c)]
    if not nonempty:
        raise ValueError("no variant available for a causal draw")
    c = nonempty[g.integers(len(nonempty))]
    return int(c[g.integers(len(c))])


def draw_pve(g: np.random.Generator, kind: str, strong: bool) -> float:
    main, hi = params.PVE_RANGE[kind]
    lo_, hi_ = hi if (strong and hi is not None) else main
    return float(np.exp(g.uniform(np.log(lo_), np.log(hi_))))


def draw_architecture(g: np.random.Generator, maf: np.ndarray, r2_row, kind1: str, kind2: str,
                      ltype: str | None = None, p_strong: float = params.P_STRONG,
                      bg_share: float = params.BG_SHARE, m_max: int = params.M_MAX_CAUSAL) -> Architecture:
    """maf: target-population MAF of the analysis variants; r2_row(j) -> r² of j with all variants (R_true).
    p_strong / bg_share / m_max: biobank-strong share, background share, max causal per trait in M loci
    (defaults = training prior DESIGN §2.4; the shifted arm §8.5 uses 0.5 / 0.5 / 4)."""
    types = list(params.LOCUS_TYPES)
    if ltype is None:
        ltype = types[g.choice(len(types), p=np.array([params.LOCUS_TYPES[t] for t in types]))]
    bins = maf_bins(maf)
    strong1 = kind1 == "gwas" and g.random() < p_strong
    strong2 = kind2 == "gwas" and g.random() < p_strong
    a = Architecture(ltype, [], [], strong1=strong1, strong2=strong2, bg=bool(g.random() < bg_share))

    def pve(t):
        kind, strong = (kind1, strong1) if t == 1 else (kind2, strong2)
        return ("pve", draw_pve(g, kind, strong), float(g.choice([-1.0, 1.0])))

    def add(t, j, spec=None):
        (a.c1 if t == 1 else a.c2).append(j)
        (a.eff1 if t == 1 else a.eff2)[j] = spec if spec is not None else pve(t)

    used: set[int] = set()
    if ltype == "S":
        j = pick_by_maf(g, bins, used); used.add(j)
        add(1, j); add(2, j)
        if g.random() < params.S_EXTRA:
            k = pick_by_maf(g, bins, used); used.add(k)
            add(int(g.integers(1, 3)), k)
    elif ltype == "D":
        j = pick_by_maf(g, bins, used); used.add(j)
        r2 = np.asarray(r2_row(j), dtype=np.float64)
        b = int(g.integers(len(params.R2_BINS)))
        a.target_r2_bin = b
        k = _partner(g, r2, b, used)
        used.add(k)
        if g.random() < 0.5:
            add(1, j); add(2, k)
        else:
            add(1, k); add(2, j)
        if g.random() < params.D_EXTRA:
            e = pick_by_maf(g, bins, used); used.add(e)
            add(int(g.integers(1, 3)), e)
    elif ltype == "W":
        t = int(g.integers(1, 3))
        j = pick_by_maf(g, bins, used); used.add(j)
        add(t, j)
        if g.random() < params.W_SHARED_SUB:
            o = 3 - t
            kind = kind1 if o == 1 else kind2
            lam = float(g.uniform(0.5, params.TAU[kind])) * float(g.choice([-1.0, 1.0]))
            add(o, j, ("lam", lam))
    elif ltype == "Sc":
        # validation only: shared variant with concordant effect signs in the two traits (worst case for
        # rho-hat leakage, e.g. the same eQTL in two cohorts)
        j = pick_by_maf(g, bins, used); used.add(j)
        add(1, j)
        s1 = a.eff1[j][2]
        sp = pve(2)
        add(2, j, (sp[0], sp[1], s1))
    elif ltype == "N":
        pass   # null locus (no causal variant; background allowed). Validation only, not in the mixture.
    elif ltype == "M":
        k1, k2 = int(g.integers(2, m_max + 1)), int(g.integers(2, m_max + 1))
        s = int(g.integers(0, min(k1, k2) + 1))
        for _ in range(s):
            j = pick_by_maf(g, bins, used); used.add(j)
            add(1, j); add(2, j)
        for _ in range(k1 - s):
            j = pick_by_maf(g, bins, used); used.add(j)
            add(1, j)
        for _ in range(k2 - s):
            j = pick_by_maf(g, bins, used); used.add(j)
            add(2, j)
    else:
        raise ValueError(ltype)
    return a


def _partner(g: np.random.Generator, r2: np.ndarray, b: int, used: set) -> int:
    """Variant whose r² with the first causal falls in R2_BINS[b]; nearest non-empty bin if empty."""
    p = len(r2)
    ok = np.ones(p, bool)
    ok[list(used)] = False
    order = sorted(range(len(params.R2_BINS)), key=lambda k: (abs(k - b), k))
    for k in order:
        lo, hi = params.R2_BINS[k]
        m = ok & (r2 >= lo) & ((r2 < hi) if k < len(params.R2_BINS) - 1 else (r2 <= hi + 1e-9))
        idx = np.where(m)[0]
        if len(idx):
            return int(idx[g.integers(len(idx))])
    raise ValueError("no partner variant")
