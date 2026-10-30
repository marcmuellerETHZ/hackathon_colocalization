"""Every tier-1 wrapper on known-direction controls built from a real 1000G chr22 NFE LD window (p = 300).

shared   : lambda = 30 at the same SNP in both traits             -> s > 0.9
distinct : lambda = 30 at two SNPs with r^2 < 0.1                  -> s < 0.1
null2    : lambda = 30 in trait 1, nothing in trait 2              -> high P_W or low P_D (P_W > 0.5 or P_D < 0.5)
Proportional tests (s = p-value of the proportional = shared null): shared -> s > 0.01 (a p-value is uniform
under its null, so s > 0.9 is not expected); they have no P_D/P_W. Lead-SNP rule: binary s, no P_*.
"""
import json
import math

import numpy as np
import pytest

from baselines.methods import ColocSusie, all_methods
from baselines.toy import make_toy_locus, pick_pair
from common.seeds import VALID_SEED, rng

METHODS = list(all_methods())
PVALUE_METHODS = {"prop_coloc", "coloc_prop_test"}
NO_PROB = PVALUE_METHODS | {"lead_snp"}
COLUMNS = ["locus_id", "method", "s", "P_S", "P_D", "P_W", "E1", "E2", "call", "status", "runtime_s", "extra"]
_RESULTS = {}


def _run(name, locus):
    key = (name, locus.locus_id)
    if key not in _RESULTS:
        _RESULTS[key] = all_methods()[name]().fit_batch([locus])[0]
    return _RESULTS[key]


@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("rep", [0, 1])
def test_shared(name, rep, controls):
    r = _run(name, controls[("shared", rep)])
    assert r["status"] == "ok", r["extra"]
    assert list(r) == COLUMNS
    json.loads(r["extra"])
    if name in PVALUE_METHODS:
        assert r["s"] > 0.01, r
    else:
        assert r["s"] > 0.9, r


@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("rep", [0, 1])
def test_distinct(name, rep, controls):
    r = _run(name, controls[("distinct", rep)])
    assert r["status"] == "ok", r["extra"]
    if name == "sharepro" and r["s"] >= 0.1:
        # Known method behaviour, not a wrapper bug: identical to the official CLI on these loci
        # (test_sharepro_equiv.py). SharePro's false sharing rises for r^2 0.05-0.1 at lambda = 30 (README).
        pytest.xfail(f"SharePro false sharing on distinct control (s = {r['s']:.3g}); matches CLI")
    assert r["s"] < 0.1, r


@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("rep", [0, 1])
def test_null_trait2(name, rep, controls):
    r = _run(name, controls[("null2", rep)])
    assert r["status"] == "ok", r["extra"]
    if name in NO_PROB:
        assert math.isnan(r["P_D"]) and math.isnan(r["P_W"])
        return
    assert r["P_W"] > 0.5 or r["P_D"] < 0.5, r
    for k in ("P_S", "P_D", "P_W"):
        assert -1e-12 <= r[k] <= 1 + 1e-12
    assert abs(r["P_S"] + r["P_D"] + r["P_W"] - 1) < 1e-9


@pytest.mark.parametrize("name", ["sharepro"])
def test_sharepro_low_r2_distinct(name, window300):
    """SharePro must pass the distinct control when the two causal variants are nearly independent (r^2 < 0.01)."""
    X, maf, ids, pos = window300
    R = X.T @ X / X.shape[0]
    for rep in range(3):
        g = rng(VALID_SEED, "tests/baselines/lowr2", rep)
        a, b = pick_pair(R, g, r2_max=0.01)
        r = all_methods()[name]().fit_one(make_toy_locus(X, maf, ids, pos, {a: 30.0}, {b: 30.0}, g))
        assert r["status"] == "ok" and r["s"] < 0.1, r


@pytest.mark.parametrize("name", ["colocboost", "mvsusie", "prop_coloc", "coloc_susie"])
def test_sample_overlap_runs(name, window300):
    """rho = 0.5 with rho_hat given: the overlap-aware wrappers run and still call the shared control."""
    X, maf, ids, pos = window300
    g = rng(VALID_SEED, "tests/baselines/overlap")
    L = make_toy_locus(X, maf, ids, pos, {150: 30.0}, {150: 30.0}, g, rho=0.5, locus_id="overlap")
    r = all_methods()[name]().fit_one(L)
    assert r["status"] == "ok", r["extra"]
    ex = json.loads(r["extra"])
    if "rho_used" in ex:
        assert ex["rho_used"] == pytest.approx(0.5)
    if name == "prop_coloc":
        assert ex["tau"] == pytest.approx(0.5) and r["s"] > 0.01
    else:
        assert r["s"] > 0.9


def test_factor_ld_equals_dense(window300):
    X, maf, ids, pos = window300
    out = []
    for ld_type in ("dense", "factor"):
        g = rng(VALID_SEED, "tests/baselines/factor")
        L = make_toy_locus(X, maf, ids, pos, {100: 8.0}, {104: 8.0}, g, ld_type=ld_type, locus_id=ld_type)
        out.append(ColocSusie().fit_one(L))
    assert out[0]["status"] == out[1]["status"] == "ok"
    assert abs(out[0]["s"] - out[1]["s"]) < 1e-3 and abs(out[0]["P_D"] - out[1]["P_D"]) < 1e-3


@pytest.mark.parametrize("name", ["coloc_susie", "sharepro", "coloc_abf"])
def test_failure_is_recorded(name, controls):
    import copy

    L = copy.deepcopy(controls[("shared", 0)])
    L.z1 = L.z1.copy()
    L.z1[3] = np.nan
    L.beta1 = L.beta1.copy()
    L.beta1[3] = np.nan
    r = all_methods()[name]().fit_one(L)
    assert r["status"] == "failed" and math.isnan(r["s"]) and "non-finite" in json.loads(r["extra"])["error"]


def test_timeout_is_recorded(controls):
    r = ColocSusie(timeout=0.01).fit_one(controls[("shared", 0)])
    assert r["status"] == "timeout" and math.isnan(r["s"])
    r2 = ColocSusie().fit_one(controls[("shared", 0)])  # worker respawns
    assert r2["status"] == "ok"


def test_mvsusie_all_null_locus(window300, monkeypatch):
    """When mvsusie estimates every prior variance 0 it returns a scalar NA lfsr and no CS (seen on 7/200 W loci of
    the screen_v1 pilot, which failed before the fix). The wrapper must report P_W = 1, not fail. The R reply
    observed on screen_v1__L0_P3_rho=0.5__W__0022203 is replayed (a toy null locus does not trigger it)."""
    from baselines.methods import MvSusie

    X, maf, ids, pos = window300
    g = rng(VALID_SEED, "tests/baselines/mvsusie_null")
    loc = make_toy_locus(X, maf, ids, pos, {}, {}, g, locus_id="null_both")
    reply = {"result": {"mvsusieR": "0.3.0", "susieR": "0.16.6", "single_effect_lfsr": [[None]], "cs_index": [],
                        "converged": True, "niter": 2, "V_est": [0.0] * 10, "max_pip": 0.0}}
    m = MvSusie()
    monkeypatch.setattr(m, "_r", lambda fn, args, arrays: reply)
    r = m.fit_one(loc)
    assert r["status"] == "ok", r["extra"]
    assert r["P_W"] == 1.0 and r["P_S"] == 0.0 and r["P_D"] == 0.0 and r["s"] == 0.5
    # a malformed lfsr WITH credible sets must still fail loudly
    reply["result"]["cs_index"] = [1]
    assert m.fit_one(loc)["status"] == "failed"


def test_mvsusie_hard_rule_is_primary(window300, monkeypatch):
    """Decision P3-2: the hard lfsr < 0.05 rule gives the row's s/P_S/P_D/P_W; the soft rule (q = 1 - lfsr) is in
    extra["soft"]. Known answer on a replayed reply with two CS effects:
    effect 1 lfsr (0.01, 0.30) -> hard q = (1, 0); effect 2 lfsr (0.02, 0.04) -> hard q = (1, 1).
    Hard: effect 2 is active in both traits -> P_S = 1, P_D = 0, s = 1."""
    from baselines.methods import MvSusie
    from eval import adapters as EA

    X, maf, ids, pos = window300
    g = rng(VALID_SEED, "tests/baselines/mvsusie_hard")
    loc = make_toy_locus(X, maf, ids, pos, {}, {}, g, locus_id="replay")
    lf = [[0.01, 0.30], [0.02, 0.04], [0.9, 0.9]]
    reply = {"result": {"mvsusieR": "0.3.0", "susieR": "0.16.6", "single_effect_lfsr": lf, "cs_index": [1, 2],
                        "converged": True, "niter": 5}}
    m = MvSusie()
    assert m.rule == "hard"
    monkeypatch.setattr(m, "_r", lambda fn, args, arrays: reply)
    r = m.fit_one(loc)
    assert r["status"] == "ok", r["extra"]
    assert r["P_S"] == pytest.approx(1.0) and r["P_D"] == pytest.approx(0.0) and r["s"] == pytest.approx(1.0)
    ex = json.loads(r["extra"])
    soft = EA.mvsusie(np.array(lf), [1, 2], rule="soft")
    for k in ("s", "P_S", "P_D", "P_W"):
        assert ex["soft"][k] == pytest.approx(soft[k], abs=1e-12)
        assert ex["hard_lfsr005"][k] == pytest.approx(r[k], abs=1e-12)
    # soft by hand: q1 = (0.99, 0.70), q2 = (0.98, 0.96); P_S = 1 - (1 - .99*.70)(1 - .98*.96)
    assert ex["soft"]["P_S"] == pytest.approx(1 - (1 - 0.99 * 0.70) * (1 - 0.98 * 0.96), abs=1e-12)
    ms = MvSusie(rule="soft")
    monkeypatch.setattr(ms, "_r", lambda fn, args, arrays: reply)
    assert ms.fit_one(loc)["P_S"] == pytest.approx(soft["P_S"], abs=1e-12)
    with pytest.raises(ValueError):
        MvSusie(rule="other")
