"""Tests for src/sim/oracle.py (DESIGN v4 §1, §7.1 item 2)."""
import json
from pathlib import Path

import numpy as np
import pytest

from sim.oracle import CLASSES, oracle_class, power
from sim.params import ALPHA, TAU, WEAK

OUT = Path(__file__).resolve().parents[2] / "src/sim/validation"


def test_thresholds_frozen_values():
    assert TAU == {"gwas": 6.29, "mol": 5.26}
    assert WEAK == {"gwas": 4.61, "mol": 3.58}
    for k in ("gwas", "mol"):
        assert abs(power(TAU[k], ALPHA[k]) - 0.8) < 2e-3
        assert abs(power(WEAK[k], ALPHA[k]) - 0.2) < 2e-3
        assert power(-TAU[k], ALPHA[k]) == power(TAU[k], ALPHA[k])


R0 = lambda a, b: 0.0  # noqa: E731


def test_known_answers():
    # S: shared variant powered in both
    assert oracle_class([1], [10], [10], [1], [8], [8], R0, "gwas", "mol").cls == "S"
    # S needs own-effect lambda >= tau as well (cancellation guard): mu big, lambda small -> not powered
    assert oracle_class([1], [3], [10], [1], [8], [8], R0, "gwas", "mol").cls == "G"   # nonweak in both via mu -> grey
    # D: distinct powered, r2 = 0.3, d = 8 * sqrt(.7) > 2
    o = oracle_class([1], [8], [8], [2], [9], [9], lambda a, b: 0.3, "gwas", "mol")
    assert o.cls == "D" and abs(o.r2 - 0.3) < 1e-12 and abs(o.d - 8 * np.sqrt(0.7)) < 1e-12
    assert (o.mu1, o.mu2) == (8.0, 9.0)
    # U: distinct powered but nearly perfect LD: d = 7 * sqrt(1 - 0.95) = 1.57 < 2
    assert oracle_class([1], [7], [7], [2], [7], [7], lambda a, b: 0.95, "gwas", "mol").cls == "U"
    # W: trait 2 has no causal variant
    o = oracle_class([1], [20], [20], [], [], [], R0, "gwas", "mol")
    assert o.cls == "W" and o.mu1 == 20 and o.mu2 == 0.0
    # W: trait 2 causal is weak (|mu| < 3.58)
    assert oracle_class([1], [20], [20], [1], [3], [3], R0, "gwas", "mol").cls == "W"
    # G: trait 2 grey (3.58 <= mu < 5.26)
    assert oracle_class([1], [20], [20], [1], [4.5], [4.5], R0, "gwas", "mol").cls == "G"
    # D blocked by a shared variant nonweak in both -> G
    assert oracle_class([1, 3], [8, 5], [8, 5], [2, 3], [9, 4], [9, 4], R0, "gwas", "mol").cls == "G"
    # ... but a shared variant weak in one trait does not block D (R2-PI)
    assert oracle_class([1, 3], [8, 5], [8, 5], [2, 3], [9, 2], [9, 2], R0, "gwas", "mol").cls == "D"
    # trait-type thresholds: mu = 5.5 is powered for 'mol' (5.26) but grey for 'gwas' (6.29)
    assert oracle_class([1], [5.5], [5.5], [1], [5.5], [5.5], R0, "mol", "mol").cls == "S"
    assert oracle_class([1], [5.5], [5.5], [1], [5.5], [5.5], R0, "gwas", "gwas").cls == "G"
    # class-determining pair = max r²
    r2 = {(1, 2): 0.1, (1, 4): 0.6, (3, 2): 0.2, (3, 4): 0.0}
    o = oracle_class([1, 3], [10, 10], [10, 10], [2, 4], [10, 10], [10, 10], lambda a, b: r2[(a, b)], "gwas", "mol")
    assert o.cls == "D" and (o.cd1, o.cd2) == (1, 4)
    # v4: trait types are required (no silent P1 default)
    with pytest.raises(TypeError):
        oracle_class([1], [10], [10], [1], [8], [8], R0)
    with pytest.raises(ValueError):
        oracle_class([1], [10], [10], [1], [8], [8], R0, "gwas", "eqtl")


def _reference_class(c1, l1, m1, c2, l2, m2, r2f, k1, k2):
    """Independent re-implementation of §1 (set logic written differently) for cross-checking."""
    t1, t2, w1, w2 = TAU[k1], TAU[k2], WEAK[k1], WEAK[k2]
    P1 = [j for j, l, m in zip(c1, l1, m1) if abs(m) >= t1 and abs(l) >= t1]
    P2 = [j for j, l, m in zip(c2, l2, m2) if abs(m) >= t2 and abs(l) >= t2]
    N1 = [j for j, m in zip(c1, m1) if abs(m) >= w1]
    N2 = [j for j, m in zip(c2, m2) if abs(m) >= w2]
    if any(j in P2 for j in P1):
        return "S"
    if P1 and P2:
        pairs = [(r2f(a, b), a, b) for a in P1 for b in P2]
        rmax = max(p[0] for p in pairs)
        cands = [p for p in pairs if p[0] == rmax]
        dmin = min(min(abs(l1[c1.index(a)]), abs(l2[c2.index(b)])) * np.sqrt(1 - r) for r, a, b in cands)
        if dmin < 2:
            return "U"
        if not any(j in N2 for j in N1):
            return "D"
    if not N1 or not N2:
        return "W"
    return "G"


def test_partition_1e5_draws():
    """DESIGN §7.1.2: exactly one label per locus on 1e5 random draws; agreement with an independent
    re-implementation; class shares written to src/sim/validation/oracle_partition.json."""
    g = np.random.default_rng(20261026)
    counts = {c: 0 for c in CLASSES}
    n = 100_000
    for i in range(n):
        p = 8
        k1, k2 = g.integers(0, 4), g.integers(0, 4)
        c1 = list(g.choice(p, k1, replace=False))
        c2 = list(g.choice(p, k2, replace=False))
        scale = np.exp(g.uniform(np.log(0.5), np.log(40)))
        l1 = list(g.normal(0, scale, k1))
        l2 = list(g.normal(0, scale, k2))
        m1 = [l + g.normal(0, 3) for l in l1]   # LD perturbs mu around lambda (incl. cancellation)
        m2 = [l + g.normal(0, 3) for l in l2]
        R2 = g.uniform(0, 1, (p, p)) ** 2
        R2 = (R2 + R2.T) / 2
        np.fill_diagonal(R2, 1.0)
        kinds = [("gwas", "mol"), ("mol", "mol"), ("gwas", "gwas")][i % 3]
        o = oracle_class(c1, l1, m1, c2, l2, m2, lambda a, b: R2[a, b], *kinds)
        assert o.cls in CLASSES
        assert o.cls == _reference_class(c1, l1, m1, c2, l2, m2, lambda a, b: R2[a, b], *kinds)
        if o.cls in ("D", "U"):
            assert np.isfinite(o.r2) and np.isfinite(o.d) and o.cd1 != o.cd2
        else:
            assert np.isnan(o.d)
        counts[o.cls] += 1
    assert sum(counts.values()) == n
    shares = {k: v / n for k, v in counts.items()}
    print("oracle class shares on 1e5 random draws:", shares)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "oracle_partition.json").write_text(json.dumps({"n": n, "counts": counts, "shares": shares}, indent=1))
    assert all(v > 0 for v in counts.values())
