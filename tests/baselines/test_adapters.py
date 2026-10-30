"""Adapters used by the baseline wrappers (DESIGN §4.3, §5; D4, D6(f)).

Since D6(f) the formulas exist only in src/eval/adapters.py (owner: eval agent) and the wrappers call its
method-specific helpers directly (no copy and no shim in src/baselines). These tests (1) assert that, and
(2) re-check the helpers the wrappers rely on with hand-computed answers and exact enumeration (independent of
the eval agent's own tests).
"""
import ast
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from eval import adapters as EA

SRC = Path(__file__).resolve().parents[2] / "src" / "baselines"


def test_no_adapter_copy_in_baselines():
    assert importlib.util.find_spec("baselines.adapters") is None
    tree = ast.parse((SRC / "methods.py").read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "eval"]
    assert any(a.name == "adapters" for n in imports for a in n.names)
    src = (SRC / "methods.py").read_text()
    for token in ("np.prod", "PP4/(PP3", "1.0 - P_S", "(1 - P_S)"):
        assert token not in src, f"formula fragment {token!r} in methods.py"
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "EA"}
    assert {"coloc_abf", "coloc_susie", "sharepro", "colocboost", "mvsusie", "proportional_test",
            "lead_snp"} <= used


def test_coloc_abf_and_susie():
    r = EA.coloc_abf([0.1, 0.1, 0.1, 0.2, 0.5])
    assert r["P_S"] == 0.5 and r["P_D"] == 0.2 and math.isclose(r["P_W"], 0.3) and math.isclose(r["s"], 0.5 / 0.7)
    assert EA.coloc_abf([1, 0, 0, 0, 0])["s"] == 0.5
    r = EA.coloc_susie([[0, 0, 0, 0.9, 0.1], [0, 0, 0, 0.2, 0.8]])
    assert r["P_S"] == 0.8 and math.isclose(r["P_D"], 0.2 * 0.9) and math.isclose(r["s"], 0.8)
    assert EA.coloc_susie([[np.nan] * 5, [0, 0, 0, 0.5, 0.5]])["P_S"] == 0.5  # NaN rows dropped
    fb = [0.1, 0.1, 0.1, 0.2, 0.5]
    assert EA.coloc_susie([[np.nan] * 5], fallback_pp=fb) == EA.coloc_abf(fb)
    assert EA.coloc_susie(None, fallback_pp=fb) == EA.coloc_abf(fb)
    with pytest.raises(ValueError):
        EA.coloc_susie([[np.nan] * 5])


def test_sharepro_and_colocboost_rows():
    # SharePro: P_S = 1 - prod(1 - share); P(sig t) = 1 - prod_k(1 - effect_prob_tk) (D6(f))
    r = EA.sharepro([0.2, 0.5], [0.9, 0.5], [0.4])
    PS, s1, s2 = 1 - 0.8 * 0.5, 1 - 0.1 * 0.5, 0.4
    assert math.isclose(r["P_S"], PS) and math.isclose(r["P_D"], (1 - PS) * s1 * s2)
    assert math.isclose(r["P_W"], 1 - PS - (1 - PS) * s1 * s2) and r["s"] == r["P_S"]
    r = EA.sharepro([], [0.0] * 10, [0.0] * 10)
    assert r["P_S"] == 0 and r["P_D"] == 0 and r["P_W"] == 1
    # ColocBoost: binary per-trait signal indicators
    r = EA.colocboost([0.9], 1.0, 1.0)
    assert math.isclose(r["P_S"], 0.9) and math.isclose(r["P_D"], 0.1)
    assert EA.colocboost([], 1.0, 0.0)["P_D"] == 0 and EA.colocboost([], 1.0, 1.0)["P_D"] == 1


def test_mvsusie_known_answers():
    lf = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    # effect 1 trait-1 only, effect 2 trait-2 only: P_D = 1
    r = EA.mvsusie(lf, [1, 2])
    assert math.isclose(r["P_D"], 1.0) and r["P_S"] == 0 and r["s"] == 0
    # + a shared effect blocks D
    r = EA.mvsusie(lf, [1, 2, 3])
    assert r["P_S"] == 1 and r["P_D"] == 0
    # effects outside a CS are ignored
    assert EA.mvsusie(lf, [])["P_W"] == 1 and EA.mvsusie(lf, [])["s"] == 0.5
    # one effect in trait 1 only: W
    r = EA.mvsusie(lf[:1], [1])
    assert r["P_W"] == 1


def test_inclusion_exclusion_q_matches_enumeration():
    """P_S, P_D of the mvSuSiE mapping equal exact enumeration over independent activity indicators."""
    g = np.random.default_rng(20261029)
    for _ in range(50):
        L = int(g.integers(1, 5))
        q1, q2 = g.random(L), g.random(L)
        ps = pd_ = 0.0
        for a in range(4 ** L):
            st = [(a >> (2 * l)) & 3 for l in range(L)]  # bit0: active in trait 1, bit1: active in trait 2
            pr = 1.0
            for l, s in enumerate(st):
                pr *= (q1[l] if s & 1 else 1 - q1[l]) * (q2[l] if s & 2 else 1 - q2[l])
            if any(s == 3 for s in st):
                ps += pr
            elif any(s == 1 for s in st) and any(s == 2 for s in st):
                pd_ += pr
        r = EA.inclusion_exclusion_q(q1, q2)
        assert math.isclose(r["P_S"], ps, abs_tol=1e-12)
        assert math.isclose(r["P_D"], pd_, abs_tol=1e-12)


def test_s_vs_d_only_rows():
    assert EA.proportional_test(0.3)["s"] == 0.3 and math.isnan(EA.proportional_test(0.3)["P_D"])
    assert EA.proportional_test(float("nan"))["s"] == 0.5
    assert EA.lead_snp(True)["s"] == 1.0 and EA.lead_snp(False)["s"] == 0.0
