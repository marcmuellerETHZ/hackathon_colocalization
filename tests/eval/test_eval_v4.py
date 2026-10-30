"""Known-answer tests for the DESIGN v4 / D6 changes: C3 thresholds with S precedence, C3 test, bootstrap
p-values / Holm / paired contrasts, power matching without strata, D strata reporting, shifted-arm pooling,
the unified adapters, the strength-only control, and the D6(h) results policy."""
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import eval.io as eio
from eval import adapters as A
from eval import metrics as M
from eval import strength_control as SC
from eval.harness import MethodSpec, results_policy, run_method
from eval_toy import make_locus_set

import verify_run


@pytest.fixture(autouse=True)
def no_splits(monkeypatch):
    monkeypatch.setattr(eio, "_splits_file", lambda: None)
    monkeypatch.delenv(eio.ACCESS_ENV, raising=False)


def _truth(rows):
    t = pd.DataFrame(rows)
    for c, v in (("mu1", 8.0), ("mu2", 8.0), ("r2", np.nan), ("d", np.nan), ("rho", 0.0), ("block_id", "b0"),
                 ("cell_id", "L1|P1|r_p=0.0"), ("split", "validation"), ("mismatch_level", "L1"), ("pairing_type", "P1"),
                 ("overlap", 0.0), ("control", "")):
        if c not in t:
            t[c] = v
    return t


def _preds(truth, method="m", **cols):
    df = pd.DataFrame({"locus_id": truth["locus_id"], "method": method, "status": "ok"})
    for c in ("s", "P_S", "P_D", "P_W", "E1", "E2"):
        df[c] = cols.get(c, np.nan)
    if np.isnan(df["s"]).all():
        df["s"] = 0.5
    return df


# ==================================================================================================
# C3 thresholds: t_S first, then t_D with S precedence
# ==================================================================================================
def test_c3_thresholds_s_precedence_hand_case():
    # 40 W loci (weight 1/40 = 0.025 each, so exactly one may be called): locus 0 has P_S 0.95 and P_D 0.99,
    # locus 1 has P_D 0.9; others P_S 0, P_D 0.1. Plus D and S loci so all classes exist.
    nW = 40
    rows = [dict(locus_id=f"w{i}", oracle_class="W") for i in range(nW)]
    rows += [dict(locus_id=f"d{i}", oracle_class="D", r2=0.05, d=6.0) for i in range(10)]
    rows += [dict(locus_id=f"s{i}", oracle_class="S") for i in range(10)]
    t = _truth(rows)
    PS = np.r_[0.95, np.zeros(nW - 1), np.zeros(10), np.full(10, 0.99)]
    PD = np.r_[0.99, 0.9, np.full(nW - 2, 0.1), np.full(10, 0.95), np.full(10, 0.0)]
    fr = M.build_eval_frame(t, _preds(t, P_S=PS, P_D=PD, s=PS))
    mw = M.matched_w_thresholds(fr, "m", "realistic")
    assert mw["t_S"] == np.nextafter(0.0, 1)       # locus 0 (0.95) is the one W called S
    # with S precedence locus 0 cannot be called D, so locus 1 (0.9) may be: t_D just above 0.1
    assert mw["t_D"] == np.nextafter(0.1, 1)
    assert mw["rate_S_on_W"] == pytest.approx(0.025) and mw["rate_D_on_W"] == pytest.approx(0.025)
    r = M.recall_at(fr, "m", mw["t_S"], mw["t_D"], "realistic")
    assert r["W_called_S"] <= 0.025 + 1e-12 and r["W_called_D"] <= 0.025 + 1e-12
    assert r["recall_D"] == 1.0 and r["recall_S"] == 1.0
    # without precedence (old rule) t_D would have had to sit above 0.9; check the sequential rule differs
    old = M.matched_threshold(np.r_[0.99, 0.9, np.full(nW - 2, 0.1)], alpha=0.025)
    assert old["t"] == np.nextafter(0.9, 1) and old["t"] > mw["t_D"]


def test_c3_thresholds_e_gate_and_missing_ps():
    rows = [dict(locus_id=f"w{i}", oracle_class="W") for i in range(40)]
    rows += [dict(locus_id=f"d{i}", oracle_class="D", r2=0.0, d=5.0) for i in range(5)]
    t = _truth(rows)
    PD = np.r_[0.99, 0.98, np.full(38, 0.1), np.full(5, 0.97)]
    E = np.r_[0.5, 1.0, np.ones(38), np.ones(5)]  # W locus 0 fails the E gate -> cannot be called D
    fr = M.build_eval_frame(t, _preds(t, P_D=PD, E1=E, E2=E))
    mw = M.matched_w_thresholds(fr, "m", "realistic")
    assert "t_S" not in mw and mw["t_D"] == np.nextafter(0.1, 1)  # one callable W (0.98) allowed


# ==================================================================================================
# bootstrap p-values, Holm, contrasts
# ==================================================================================================
def test_holm_known():
    assert M.holm([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert M.holm([0.04, 0.01]) == pytest.approx([0.04, 0.02])
    assert M.holm([0.03, 0.03]) == pytest.approx([0.06, 0.06])
    assert M.holm([0.6, 0.01]) == pytest.approx([0.6, 0.02])


def test_bootstrap_p_value_duality_with_bca_bound():
    rng = np.random.default_rng(11)
    x = rng.exponential(1.0, 300)  # skewed: BCa acceleration non-zero
    res = M.cluster_bootstrap(lambda idx: x[idx].mean(), np.arange(300), n_boot=4000,
                              rng=np.random.default_rng(1), level=0.90, return_replicates=True)
    # the one-sided 95% lower bound is the two-sided 90% lower bound: p at that null equals 0.05
    assert M.bootstrap_p_value(res, res["ci_low"]) == pytest.approx(0.05, abs=0.004)
    assert M.bootstrap_p_value(res, res["ci_high"], alternative="less") == pytest.approx(0.05, abs=0.004)
    # null at the estimate: p ~ 0.5; far below: tiny; far above: ~1
    assert M.bootstrap_p_value(res, res["estimate"]) == pytest.approx(0.5, abs=0.05)
    assert M.bootstrap_p_value(res, 0.0) < 1e-3
    assert M.bootstrap_p_value(res, 3.0) > 0.999
    # percentile version: p = share of replicates <= null
    rp = M.cluster_bootstrap(lambda idx: x[idx].mean(), np.arange(300), n_boot=2000, rng=np.random.default_rng(2),
                             bca=False, return_replicates=True)
    nul = float(np.quantile(rp["replicates"][:, 0], 0.1))
    assert M.bootstrap_p_value(rp, nul) == pytest.approx(np.mean(rp["replicates"][:, 0] < nul), abs=1e-3)


def test_bootstrap_p_value_level_under_null():
    """Calibration: under H0 (true mean 0) the one-sided p < 0.05 rejects ~5% (clustered data)."""
    rej = 0
    for k in range(200):
        g = np.random.default_rng(100 + k)
        clusters = np.repeat(np.arange(60), 5)
        x = np.repeat(g.normal(0, 1, 60), 5) + g.normal(0, 0.5, 300)
        res = M.cluster_bootstrap(lambda idx: x[idx].mean(), clusters, n_boot=400, rng=g, return_replicates=True)
        rej += M.bootstrap_p_value(res, 0.0) < 0.05
    assert 0.01 <= rej / 200 <= 0.11, rej


def test_contrast_equals_paired_bootstrap():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(600):
        c = ["S", "D", "W"][i % 3]
        rows.append(dict(locus_id=f"l{i}", oracle_class=c, mu1=rng.uniform(6, 20), mu2=rng.uniform(6, 20),
                         r2=0.2 if c == "D" else np.nan, d=5.0 if c == "D" else np.nan, block_id=f"b{i % 40}"))
    t = _truth(rows)
    y = (t["oracle_class"] == "S").to_numpy()
    fa = _preds(t, "a", s=y + rng.normal(0, 0.7, len(t)))
    fb = _preds(t, "b", s=y + rng.normal(0, 1.0, len(t)))
    fr = M.build_eval_frame(t, {"a": fa, "b": fb})
    pb = M.paired_bootstrap(fr, lambda f, m: M.auprc_sd(f, m, "realistic"), "a", "b", n_boot=300,
                            rng=np.random.default_rng(9))
    jt = M.cluster_bootstrap(lambda idx: np.array([M.auprc_sd(fr.take(idx), m, "realistic") for m in ("a", "b")]),
                             fr.block, n_boot=300, rng=np.random.default_rng(9), return_replicates=True)
    c = M.contrast(jt, 0, 1)
    for k in ("estimate", "ci_low", "ci_high"):
        assert c[k] == pytest.approx(pb[k], abs=1e-12)
    # endpoint_vector equals the per-method endpoint functions
    v, lab = M.endpoint_vector(fr, ["a", "b"], ["realistic"])
    pos = {l: i for i, l in enumerate(lab)}
    assert v[pos[("a", "realistic", "auprc_sd")]] == pytest.approx(M.auprc_sd(fr, "a", "realistic"), abs=1e-14)
    assert v[pos[("b", "realistic", "auroc_sd")]] == pytest.approx(M.auroc_sd(fr, "b", "realistic"), abs=1e-14)
    assert math.isnan(v[pos[("a", "realistic", "auroc_dw")]])  # no P_D


# ==================================================================================================
# C3 test
# ==================================================================================================
def _c3_frame(rng, n=900, recall_shift=0.0):
    rows, k = [], 0
    for b in range(60):
        for cls in ("S", "D", "W") * 5:
            rows.append(dict(locus_id=f"l{k}", oracle_class=cls, block_id=f"b{b}",
                             r2=0.1 if cls == "D" else np.nan, d=6.0 if cls == "D" else np.nan))
            k += 1
    t = _truth(rows)
    cls = t["oracle_class"].to_numpy()
    base_PS = np.where(cls == "S", rng.uniform(0.3, 1, len(t)), rng.uniform(0, 0.6, len(t)))
    base_PD = np.where(cls == "D", rng.uniform(0.3, 1, len(t)), rng.uniform(0, 0.6, len(t)))
    return t, base_PS, base_PD


def test_c3_test_known_outcomes():
    rng = np.random.default_rng(4)
    t, PS, PD = _c3_frame(rng)
    cls = t["oracle_class"].to_numpy()
    PD_better = np.where(cls == "D", np.maximum(PD, 0.75), PD)  # recall_D clearly higher at the same t_D
    PD_worse = np.where(cls == "D", PD * 0.6, PD)
    preds = {"cand_same": _preds(t, "cand_same", P_S=PS, P_D=PD, s=PS),
             "base": _preds(t, "base", P_S=PS, P_D=PD, s=PS),
             "cand_better": _preds(t, "cand_better", P_S=PS, P_D=PD_better, s=PS),
             "cand_worse": _preds(t, "cand_worse", P_S=PS, P_D=PD_worse, s=PS),
             "prop": _preds(t, "prop", s=PS)}
    fr = M.build_eval_frame(t, preds)
    thr = {m: {"t_S": 0.62, "t_D": 0.62} for m in preds}
    same = M.c3_test(fr, "cand_same", ["base", "prop"], thr, "realistic", n_boot=300, rng=np.random.default_rng(1))
    assert same["eligible"] == ["base"] and same["ineligible"] == ["prop"]
    assert same["noninferior_all"] and not same["superior_one"] and not same["pass"]
    assert same["vs"]["base"]["recall_D"]["diff"] == 0.0
    better = M.c3_test(fr, "cand_better", ["base"], thr, "realistic", n_boot=300, rng=np.random.default_rng(1))
    assert better["pass"] and better["superiority"]["recall_D"]["p_holm"] < 0.05
    assert better["superiority"]["recall_D"]["best_baseline"] == "base"
    worse = M.c3_test(fr, "cand_worse", ["base"], thr, "realistic", n_boot=300, rng=np.random.default_rng(1))
    assert not worse["noninferior_all"] and not worse["pass"]
    assert worse["vs"]["base"]["recall_D"]["p_noninferiority"] > 0.5


# ==================================================================================================
# power matching (no strata) and stratified reporting
# ==================================================================================================
def test_power_matching_ignores_strata_and_strata_report():
    rng = np.random.default_rng(8)
    rows = []
    for i in range(900):
        c = ["S", "D", "W"][i % 3]
        r2 = rng.choice([0.05, 0.2, 0.4, 0.6, 0.9, 0.97]) if c == "D" else np.nan
        rows.append(dict(locus_id=f"l{i}", oracle_class=c, mu1=rng.uniform(6, 30), mu2=rng.uniform(5, 25), r2=r2,
                         d=rng.choice([3.0, 5.0, 10.0, 20.0]) if c == "D" else np.nan, block_id=f"b{i % 30}"))
    t = _truth(rows)
    s = (t["oracle_class"] == "S").to_numpy() + rng.normal(0, 1, len(t))
    fr = M.build_eval_frame(t, _preds(t, s=s, P_S=1 / (1 + np.exp(-s)), P_D=rng.uniform(0, 1, len(t))))
    w1 = M.sd_weights(fr, "realistic")
    t2 = t.copy()
    t2.loc[t2.oracle_class == "D", "r2"] = rng.uniform(0, 0.9, (t2.oracle_class == "D").sum())
    fr2 = M.build_eval_frame(t2, _preds(t2, s=s))
    np.testing.assert_array_equal(w1, M.sd_weights(fr2, "realistic"))  # r² does not enter the matching
    # quantiles over pooled S ∪ D of the cell: S and D binned masses equal in the single stratum
    isS, isD = fr.cls == "S", fr.cls == "D"
    U = isS | isD
    k = np.zeros(len(fr), int)
    k[U] = M._qbins(fr.mu1[U]) * 5 + M._qbins(fr.mu2[U])
    np.testing.assert_allclose(np.bincount(k[isS], w1[isS], 25), np.bincount(k[isD], w1[isD], 25), atol=1e-12)
    # D strata per v4: half-open bins plus "out" (r² >= 0.95)
    rep = M.stratified_report(fr, "m", "r2", M.DEFAULT_R2_EDGES, 0.8, 0.8, "realistic")
    assert list(rep) == ["[0,0.1)", "[0.1,0.3)", "[0.3,0.5)", "[0.5,0.8)", "[0.8,0.95)", "out"]
    nD = {lab: int((isD & np.isclose(fr.r2, v)).sum()) for lab, v in
          zip(rep, [0.05, 0.2, 0.4, 0.6, 0.9, 0.97])}
    assert {k_: rep[k_]["n"] for k_ in rep} == nD
    assert all(0.5 < rep[b]["auprc_sd"] <= 1 for b in list(rep)[:-1])
    repd = M.stratified_report(fr, "m", "d", M.DEFAULT_D_EDGES, 0.8, 0.8, "realistic")
    assert list(repd) == ["[2,4)", "[4,8)", "[8,16)", "[16,inf)", "out"] and repd["out"]["n"] == 0
    assert M.bin_index([0.0, 0.0999, 0.1, 0.95, np.nan, -1], M.DEFAULT_R2_EDGES).tolist() == [0, 0, 1, -1, -1, -1]


def test_shifted_and_v4_grid_pooling():
    rows = []
    cells = [("L1swap", "P1", 0.0), ("L1swap", "P1", 0.6), ("L2swap", "P1", 0.0), ("L2swap", "P1", 0.6),
             ("L1", "P1", 0.0), ("L3", "P1", 0.0), ("L3", "P2", 0.0), ("L3", "P3", 0.0)]
    for L, P, ov in cells:
        for c in ("S", "D", "W"):
            rows.append(dict(locus_id=f"{L}{P}{ov}{c}", oracle_class=c, cell_id=f"{L}|{P}|{ov}", mismatch_level=L,
                             pairing_type=P, overlap=ov, r2=0.1 if c == "D" else np.nan, d=5.0 if c == "D" else np.nan))
    fr = M.build_eval_frame(_truth(rows), _preds(_truth(rows), s=0.5))
    sh = M.cell_pooling_weights(fr, "shifted")
    assert sh == pytest.approx({f"{L}|P1|{ov}": 0.25 for L in ("L1swap", "L2swap") for ov in (0.0, 0.6)})
    re = M.cell_pooling_weights(fr, "realistic")
    assert not any("swap" in c for c in re)
    assert re["L3|P1|0.0"] == pytest.approx(0.5 * 0.5) and re["L3|P2|0.0"] == pytest.approx(0.5 * 0.25)


# ==================================================================================================
# adapters (single implementation, D6(f))
# ==================================================================================================
def test_adapter_aliases_and_new_helpers():
    assert A.coloc_abf_scores([0.1, 0.1, 0.1, 0.2, 0.5]) == A.coloc_abf([0.1, 0.1, 0.1, 0.2, 0.5])
    r = A.coloc_susie_scores([[np.nan] * 5, [0, 0, 0, 0.5, 0.5]])
    assert r["P_S"] == 0.5 and r["P_D"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        A.coloc_susie_scores([[np.nan] * 5])
    fb = A.coloc_susie([[np.nan] * 5], fallback_pp=[0.1, 0.1, 0.1, 0.3, 0.4])  # all-NaN rows -> coloc.abf fallback
    assert fb == A.coloc_abf([0.1, 0.1, 0.1, 0.3, 0.4])
    assert A.one_minus_prod([0.5, 0.5]) == pytest.approx(0.75) and A.one_minus_prod([]) == 0.0
    r = A.locus_prob_scores(0.3, 1.0, 0.5)
    assert r["s"] == 0.3 and r["P_D"] == pytest.approx(0.35) and r["P_W"] == pytest.approx(0.35)
    assert A.locus_prob_scores(1.2, 1, 1)["P_S"] == 1.0  # clipped
    nan = A.locus_prob_scores(float("nan"), 1, 1)
    assert nan["s"] == 0.5 and math.isnan(nan["P_S"]) and math.isnan(nan["P_D"])
    # SharePro per-effect signal probability (D6(f)): a lambda=30 signal spread over 8 SNPs in one effect
    sp = A.sharepro(shares=[0.0], effect_prob1=[1.0], effect_prob2=[0.0, 0.5])
    assert sp["P_S"] == 0.0 and sp["P_D"] == pytest.approx(0.5)
    sp = A.sharepro(shares=[0.2, 0.5], effect_prob1=[0.9, 0.1], effect_prob2=[0.8])
    PS = 1 - 0.8 * 0.5
    assert sp["P_S"] == pytest.approx(PS) and sp["P_D"] == pytest.approx((1 - PS) * (1 - 0.1 * 0.9) * 0.8)
    cb = A.colocboost([0.9, 0.5], True, False)
    assert cb["P_S"] == pytest.approx(0.95) and cb["P_D"] == 0.0
    assert A.colocboost([], True, True)["P_D"] == 1.0 and A.colocboost([], True, True)["s"] == 0.0


def test_inclusion_exclusion_q_and_mvsusie():
    r = A.inclusion_exclusion_q([1.0, 0.0], [0.0, 1.0])
    assert r["P_D"] == pytest.approx(1.0) and r["P_S"] == 0 and r["s"] == 0
    assert A.inclusion_exclusion_q([1.0, 0.0, 1.0], [0.0, 1.0, 1.0])["P_D"] == 0
    e = A.inclusion_exclusion_q([], [])
    assert (e["s"], e["P_S"], e["P_D"], e["P_W"]) == (0.5, 0.0, 0.0, 1.0)
    g = np.random.default_rng(20261029)
    for _ in range(30):  # equals the 5-argument form and brute-force enumeration
        L = int(g.integers(1, 5))
        q1, q2 = g.random(L), g.random(L)
        ps = pd_ = 0.0
        for a in range(4 ** L):
            st = [(a >> (2 * l)) & 3 for l in range(L)]
            pr = np.prod([(q1[l] if s_ & 1 else 1 - q1[l]) * (q2[l] if s_ & 2 else 1 - q2[l]) for l, s_ in enumerate(st)])
            if any(s_ == 3 for s_ in st):
                ps += pr
            elif any(s_ == 1 for s_ in st) and any(s_ == 2 for s_ in st):
                pd_ += pr
        r = A.inclusion_exclusion_q(q1, q2)
        assert r["P_S"] == pytest.approx(ps, abs=1e-12) and r["P_D"] == pytest.approx(pd_, abs=1e-12)
    lf = np.array([[0.01, 0.9], [0.5, 0.02], [0.0, 0.0]])
    soft = A.mvsusie(lf, [1, 2])  # effect 3 not in a CS -> q = 0
    ref = A.inclusion_exclusion_q([0.99, 0.5, 0], [0.1, 0.98, 0])
    assert soft == ref
    hard = A.mvsusie(lf, [1, 2], rule="hard")
    assert hard == A.inclusion_exclusion_q([1, 0, 0], [0, 1, 0]) and hard["P_D"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        A.mvsusie(lf, [4])


def test_adapters_agree_with_baseline_copy_if_present():
    """src/baselines/adapters.py is a thin shim that forwards to eval.adapters (D6(f)); its outputs must equal the
    unified helpers (in particular its mvSuSiE event mapping equals inclusion_exclusion_q)."""
    B = pytest.importorskip("baselines.adapters")
    g = np.random.default_rng(5)
    for _ in range(200):
        pp = g.dirichlet(np.ones(5))
        assert A.coloc_abf_scores(pp)["s"] == pytest.approx(B.coloc_abf_scores(pp)["s"], abs=1e-15)
        rows = g.dirichlet(np.ones(5), size=int(g.integers(1, 4)))
        a, b = A.coloc_susie_scores(rows), B.coloc_susie_scores(rows)
        for k in ("s", "P_S", "P_D", "P_W"):
            assert a[k] == pytest.approx(b[k], abs=1e-14)
        x = g.random(int(g.integers(0, 6)))
        assert A.one_minus_prod(x) == pytest.approx(B.signal_prob(x), abs=1e-14)
        ps, s1, s2 = g.random(3)
        a, b = A.locus_prob_scores(ps, s1, s2), B.locus_prob_scores(ps, s1, s2)
        for k in ("s", "P_S", "P_D", "P_W"):
            assert a[k] == pytest.approx(b[k], abs=1e-14)
        L = int(g.integers(1, 5))
        q1, q2 = g.random(L), g.random(L)
        a, b = A.inclusion_exclusion_q(q1, q2), B.inclusion_exclusion(q1, q2)
        for k in ("s", "P_S", "P_D", "P_W"):
            assert a[k] == pytest.approx(b[k], abs=1e-12)


# ==================================================================================================
# strength-only control (D6(e))
# ==================================================================================================
def _strength_data(rng, n=3000, shift=3.0):
    """S stronger than D by `shift` in both traits (pure power confounding); W weak in trait 2."""
    cls = rng.choice(["S", "D", "W"], n)
    z1 = rng.gamma(3, 3, n) + 5 + np.where(cls == "S", shift, 0)
    z2 = np.where(cls == "W", rng.uniform(0.5, 3.5, n), rng.gamma(3, 2.5, n) + 4 + np.where(cls == "S", shift, 0))
    feats = pd.DataFrame({"locus_id": [f"l{i}" for i in range(n)], "lead_abs_z1": z1, "lead_abs_z2": z2,
                          "stratum": "P1"})
    return feats, cls


def test_strength_control_known_answers():
    rng = np.random.default_rng(12)
    feats, cls = _strength_data(rng)
    tab = SC.fit_crossfit(feats, cls, k=5)
    pr = SC.predict_any(tab, feats)
    assert list(pr["locus_id"]) == list(feats["locus_id"])
    sd = np.isin(cls, ("S", "D"))
    raw = M.roc_auc(cls[sd] == "S", pr["s"].to_numpy()[sd])
    assert raw > 0.7  # strength alone separates S from D here (positive control)
    dw = np.isin(cls, ("D", "W"))
    assert M.roc_auc(cls[dw] == "D", pr["P_D"].to_numpy()[dw]) > 0.97  # D vs W is mostly strength
    # after power matching on the (here observed = oracle) strengths, the control is near chance
    t = _truth([dict(locus_id=l, oracle_class=c, mu1=a, mu2=b, r2=0.1 if c == "D" else np.nan,
                     d=5.0 if c == "D" else np.nan, block_id=f"b{i % 50}")
                for i, (l, c, a, b) in enumerate(zip(feats.locus_id, cls, feats.lead_abs_z1, feats.lead_abs_z2))])
    fr = M.build_eval_frame(t, _preds(t, s=pr["s"].to_numpy(), P_S=pr["P_S"].to_numpy(), P_D=pr["P_D"].to_numpy()))
    matched = M.auroc_sd(fr, "m", "realistic")
    assert abs(matched - 0.5) < 0.6 * (raw - 0.5), (raw, matched)
    # probabilities are coherent
    assert np.allclose(pr[["P_S", "P_D", "P_W"]].sum(axis=1) <= 1 + 1e-12, True)
    assert np.allclose(pr["s"], pr["P_S"] / (pr["P_S"] + pr["P_D"]))


def test_strength_crossfit_never_uses_own_label_and_roundtrip(tmp_path):
    rng = np.random.default_rng(13)
    feats, cls = _strength_data(rng, n=800)
    tab = SC.fit_crossfit(feats, cls, k=5)
    cls2 = cls.copy()
    cls2[0] = "S" if cls[0] != "S" else "D"   # flip one label: its own (cross-fitted) score must not change
    tab2 = SC.fit_crossfit(feats, cls2, k=5)
    p1, p2 = SC.predict_any(tab, feats.iloc[[0]]), SC.predict_any(tab2, feats.iloc[[0]])
    assert p1["s"].iloc[0] == p2["s"].iloc[0]
    # but an in-sample (single) table does change
    full1, full2 = SC.StrengthTable.fit(feats, cls), SC.StrengthTable.fit(feats, cls2)
    assert full1.predict(feats.iloc[[0]])["s"].iloc[0] != full2.predict(feats.iloc[[0]])["s"].iloc[0]
    sha = SC.dump(tab, tmp_path / "t.json")
    assert len(sha) == 64
    back = SC.load(tmp_path / "t.json")
    pd.testing.assert_frame_equal(SC.predict_any(back, feats), SC.predict_any(tab, feats))
    # unseen stratum -> pooled table
    f2 = feats.iloc[:5].assign(stratum="P9")
    assert (SC.predict_any(SC.StrengthTable.fit(feats, cls).to_dict(), f2)["table_stratum"] == SC.POOLED).all()


def test_strength_only_method_via_harness_without_ld(tmp_path):
    f = tmp_path / "toy.h5"
    locs, truth, tp = make_locus_set(f, n_per_class=8, seed=5)
    feats = SC.read_features(f)
    assert np.allclose(feats["lead_abs_z1"], [np.abs(l.z1).max() for l in locs])
    tab = SC.StrengthTable.fit(feats, truth["oracle_class"].to_numpy(), n_bins=3)
    SC.dump(tab.to_dict(), tmp_path / "tab.json")
    r = run_method(MethodSpec("eval.reference_methods:StrengthOnly", {"table": str(tmp_path / "tab.json")}), f,
                   results_dir=tmp_path / "res", n_workers=2, timeout_s=60)
    assert r.counts["ok"] == 24
    pred = pd.read_parquet(r.run_dir / "predictions.parquet")
    exp = tab.predict(feats)
    np.testing.assert_allclose(pred["s"], exp["s"], atol=1e-15)
    np.testing.assert_allclose(pred["P_D"], exp["P_D"], atol=1e-15)
    # needs_ld = False really skips LD reads (the method asserts it)
    r2 = run_method(MethodSpec("eval_dummy_methods:NoLDCheck"), f, results_dir=tmp_path / "res", n_workers=2,
                    timeout_s=60)
    assert r2.counts["ok"] == 24


# ==================================================================================================
# results policy D6(h)
# ==================================================================================================
def test_results_policy_large_predictions(tmp_path):
    f = tmp_path / "toy.h5"
    make_locus_set(f, n_per_class=4, seed=6)
    r = run_method(MethodSpec("eval.reference_methods:LeadSNPRule"), f, results_dir=tmp_path / "res", n_workers=1,
                   timeout_s=60)
    man = json.loads((r.run_dir / "manifest.json").read_text())
    assert man["results_policy"]["predictions_committed"] is True and not (r.run_dir / ".gitignore").exists()
    assert verify_run.verify(r.run_dir) == []
    # pretend the file is large: the policy writes a run-local .gitignore
    pol = results_policy(r.run_dir, limit=10)
    assert pol["predictions_committed"] is False
    assert "predictions.parquet" in (r.run_dir / ".gitignore").read_text()
    man["results_policy"] = pol
    (r.run_dir / "manifest.json").write_text(json.dumps(man))
    assert verify_run.verify(r.run_dir) == []
    (r.run_dir / ".gitignore").unlink()
    assert any("D6(h)" in p for p in verify_run.verify(r.run_dir))
    (r.run_dir / "predictions.parquet").unlink()
    assert any("not present" in p for p in verify_run.verify(r.run_dir))
