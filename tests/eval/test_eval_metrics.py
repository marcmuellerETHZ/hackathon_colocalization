"""Known-answer tests for src/eval/metrics.py (positive and negative controls)."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from eval import metrics as M


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def synth_truth(rng, cells=(("L1", "P1", 0.0),), n=(300, 300, 300), n_blocks=40, mu_shift=0.0):
    """Truth table: per cell nS S, nD D, nW W loci; S have larger mu by `mu_shift` (power confounding)."""
    rows = []
    k = 0
    for (L, P, ov) in cells:
        cid = f"{L}|{P}|{ov}"
        for cls, cnt in zip(("S", "D", "W"), n):
            for _ in range(cnt):
                shift = mu_shift if cls == "S" else 0.0
                mu1 = 7 + shift + rng.gamma(2, 2) if cls != "W" else rng.uniform(0, 3)
                mu2 = 6 + shift + rng.gamma(2, 2) if cls != "W" else rng.uniform(0, 3)
                r2 = rng.uniform(0, 1) if cls == "D" else np.nan
                rows.append(dict(locus_id=f"l{k}", oracle_class=cls, mu1=mu1, mu2=mu2, r2=r2,
                                 d=8 * np.sqrt(1 - r2) if cls == "D" else np.nan, rho=ov, block_id=f"b{rng.integers(n_blocks)}",
                                 cell_id=cid, split="validation", mismatch_level=L, pairing_type=P, overlap=ov, control=""))
                k += 1
    return pd.DataFrame(rows)


def preds_from(truth, s, P_S=None, P_D=None, method="m", status=None, **extra):
    df = pd.DataFrame({"locus_id": truth["locus_id"], "method": method, "s": s,
                       "P_S": np.nan if P_S is None else P_S, "P_D": np.nan if P_D is None else P_D,
                       "P_W": np.nan, "E1": np.nan, "E2": np.nan, "status": "ok" if status is None else status})
    for k, v in extra.items():
        df[k] = v
    return df


# --------------------------------------------------------------------------------------------------
# elementary metrics
# --------------------------------------------------------------------------------------------------
def test_ap_hand_example_with_ties():
    y = [1, 0, 1, 0]
    s = [0.9, 0.9, 0.5, 0.1]
    # threshold 0.9: P=1/2, R=1/2 ; threshold 0.5: P=2/3, R=1
    assert M.average_precision(y, s) == pytest.approx(0.5 * 0.5 + 0.5 * 2 / 3)
    assert M.roc_auc(y, s) == pytest.approx(2.5 / 4)


def test_ap_auc_match_sklearn_weighted_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 300
        y = rng.random(n) < 0.3
        s = np.round(rng.random(n) + 0.5 * y, 1)  # many ties
        w = rng.uniform(0.1, 3, n)
        assert M.average_precision(y, s, w) == pytest.approx(average_precision_score(y, s, sample_weight=w), abs=1e-12)
        assert M.roc_auc(y, s, w) == pytest.approx(roc_auc_score(y, s, sample_weight=w), abs=1e-12)


def test_integer_weights_equal_replication():
    rng = np.random.default_rng(1)
    y = rng.random(100) < 0.4
    s = rng.random(100)
    w = rng.integers(1, 4, 100)
    rep = np.repeat(np.arange(100), w)
    assert M.average_precision(y, s, w) == pytest.approx(M.average_precision(y[rep], s[rep]))
    assert M.roc_auc(y, s, w) == pytest.approx(M.roc_auc(y[rep], s[rep]))


def test_perfect_reversed_and_random_scorers():
    rng = np.random.default_rng(2)
    y = rng.random(200000) < 0.3
    assert M.average_precision(y, y.astype(float)) == 1.0
    assert M.roc_auc(y, y.astype(float)) == 1.0
    assert M.roc_auc(y, -y.astype(float)) == 0.0
    assert M.average_precision(y, np.full(len(y), 0.5)) == pytest.approx(y.mean())  # all tied = prevalence
    assert M.average_precision(y, rng.random(len(y))) == pytest.approx(y.mean(), abs=0.01)
    assert M.roc_auc(y, rng.random(len(y))) == pytest.approx(0.5, abs=0.01)
    assert np.isnan(M.average_precision(np.zeros(5, bool), np.arange(5.0)))
    assert np.isnan(M.roc_auc(np.ones(5, bool), np.arange(5.0)))


def test_auc_contributions_sum_to_auc():
    rng = np.random.default_rng(3)
    y = rng.random(500) < 0.5
    s = np.round(rng.random(500) + 0.3 * y, 2)
    w = rng.uniform(0.2, 2, 500)
    assert M.auc_contributions(y, s, w).sum() == pytest.approx(M.roc_auc(y, s, w), abs=1e-12)


def test_brier_ece_and_recalibration():
    assert M.brier([0.2, 0.8], [0, 1]) == pytest.approx(0.04)
    assert M.ece([0.05, 0.95], [0, 1], n_bins=10) == pytest.approx(0.05)
    rng = np.random.default_rng(4)
    p = rng.uniform(0.01, 0.99, 200000)
    y = rng.random(len(p)) < p
    assert M.ece(p, y) < 0.01  # calibrated -> ECE ~ 0 (negative control)
    a_true = 1.0
    y2 = rng.random(len(p)) < M.apply_logit_intercept(p, a_true)
    assert M.ece(p, y2) > 0.08  # miscalibrated (positive control)
    a = M.fit_logit_intercept(p, y2)
    assert a == pytest.approx(a_true, abs=0.03)
    assert M.ece(M.apply_logit_intercept(p, a), y2) < 0.01


def test_sign_test():
    r = M.sign_test([1, 1, 1, -1, 0, np.nan])
    assert (r["n_pos"], r["n_neg"], r["n_tie"]) == (3, 1, 1)
    assert r["p_value"] == pytest.approx(stats.binomtest(3, 4, 0.5).pvalue)


def test_matched_threshold_hand():
    s = np.arange(40) / 40.0  # 40 W loci
    r = M.matched_threshold(s, alpha=0.025)  # at most 1/40 may be called
    assert r["rate"] == pytest.approx(1 / 40)
    assert 38 / 40 < r["t"] <= 39 / 40
    assert (s >= r["t"]).sum() == 1
    # weights: the top locus has weight 2 of total 41 -> rate 2/41 > 0.025 -> must exclude it too
    w = np.ones(40)
    w[-1] = 2
    r2 = M.matched_threshold(s, w, alpha=0.025)
    assert (s >= r2["t"]).sum() == 0 and r2["rate"] == 0
    # ties: 5 W loci tied at the top (12.5%) cannot be split -> threshold above them
    s3 = np.r_[np.zeros(35), np.ones(5)]
    r3 = M.matched_threshold(s3, alpha=0.025)
    assert r3["t"] > 1 and r3["rate"] == 0
    # -inf (gated) scores are never called
    assert M.matched_threshold(np.full(10, -np.inf))["rate"] == 0


# --------------------------------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------------------------------
def test_power_matching_removes_power_confounding():
    rng = np.random.default_rng(5)
    truth = synth_truth(rng, cells=(("L1", "P1", 0.0), ("L2", "P1", 0.0)), n=(800, 800, 100), mu_shift=4.0)
    power_score = truth["mu1"] + truth["mu2"]  # knows only power, nothing about sharing
    fr = M.build_eval_frame(truth, preds_from(truth, power_score))
    sd = np.isin(fr.cls, ("S", "D"))
    raw = M.roc_auc(fr.cls[sd] == "S", fr.scores["m"]["s"][sd])
    assert raw > 0.75  # positive control: unmatched, power alone separates S from D
    w = M.sd_weights(fr, "realistic")
    m = w > 0
    matched = M.roc_auc(fr.cls[m] == "S", fr.scores["m"]["s"][m], w[m])
    # 5x5 quantile matching removes most, not all, power confounding (residual within-bin variation): see README
    assert abs(matched - 0.5) < 0.3 * (raw - 0.5), (raw, matched)
    # weighted S prevalence exactly 0.5, in every cell
    assert M.weighted_rate(fr.cls[m] == "S", w[m]) == pytest.approx(0.5)
    for c in np.unique(fr.cell):
        ic = m & (fr.cell == c)
        assert w[ic & (fr.cls == "S")].sum() == pytest.approx(w[ic & (fr.cls == "D")].sum())
    # an informative scorer independent of power keeps its discrimination (negative control)
    good = (truth["oracle_class"] == "S").astype(float) + rng.normal(0, 0.5, len(truth))
    fr2 = M.build_eval_frame(truth, preds_from(truth, good))
    assert M.auroc_sd(fr2, "m") == pytest.approx(stats.norm.cdf(1 / (0.5 * np.sqrt(2))), abs=0.04)


def test_power_matching_equalises_binned_distributions():
    rng = np.random.default_rng(6)
    n = 400
    cls = np.array(["S"] * n + ["D"] * n)
    mu1 = np.r_[rng.gamma(3, 3, n) + 3, rng.gamma(2, 3, n)]
    mu2 = np.r_[rng.gamma(3, 3, n) + 3, rng.gamma(2, 3, n)]
    strat = np.zeros(2 * n, int)
    w = M.power_match_weights(cls, mu1, mu2, strat, np.zeros(2 * n))
    k = M._qbins(mu1) * 5 + M._qbins(mu2)
    wS = np.bincount(k[:n], w[:n], 25)
    wD = np.bincount(k[n:], w[n:], 25)
    np.testing.assert_allclose(wS, wD, atol=1e-12)
    assert w.sum() == pytest.approx(1.0)
    # a scorer that is a function of the power bin only has matched AUROC exactly 0.5 (known answer)
    assert M.roc_auc(cls == "S", k.astype(float), w) == pytest.approx(0.5, abs=1e-12)
    assert M.roc_auc(cls == "S", k.astype(float)) > 0.6  # but not without the weights


def test_cell_pooling_weights_testset_grid():
    rows = []
    po = [("P1", 0.0), ("P1", 0.6), ("P2", 0.0), ("P2", 0.3), ("P3", 0.0), ("P3", 0.5), ("P3", 0.8)]
    for L in ("L0", "L1", "L2", "L3", "L4", "L6"):
        for P, ov in po:
            for cls in ("S", "D", "W"):
                rows.append(dict(locus_id=f"{L}{P}{ov}{cls}", oracle_class=cls, mu1=8, mu2=8, r2=0.1, d=5, rho=ov,
                                 block_id="b", cell_id=f"{L}|{P}|{ov}", split="validation", mismatch_level=L,
                                 pairing_type=P, overlap=ov))
    truth = pd.DataFrame(rows)
    fr = M.build_eval_frame(truth, preds_from(truth, 0.5))
    cw = M.cell_pooling_weights(fr, "realistic")
    assert len(cw) == 35 and sum(cw.values()) == pytest.approx(1.0)
    assert cw["L1|P1|0.0"] == pytest.approx(0.2 * 0.5 / 2)
    assert cw["L6|P2|0.3"] == pytest.approx(0.2 * 0.25 / 2)
    assert cw["L3|P3|0.8"] == pytest.approx(0.2 * 0.25 / 3)
    assert not any(k.startswith("L0") for k in cw)
    ideal = M.cell_pooling_weights(fr, "ideal")
    assert ideal == pytest.approx({"L0|P1|0.0": 0.5, "L0|P2|0.0": 0.25, "L0|P3|0.0": 0.25})
    assert len(M.cell_pooling_weights(fr, "all")) == 42


# --------------------------------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------------------------------
def test_endpoints_perfect_and_random():
    rng = np.random.default_rng(7)
    truth = synth_truth(rng, n=(400, 400, 400))
    isS, isD = (truth["oracle_class"] == "S").to_numpy(), (truth["oracle_class"] == "D").to_numpy()
    perfect = preds_from(truth, isS.astype(float), P_S=isS.astype(float), P_D=isD.astype(float))
    fr = M.build_eval_frame(truth, perfect)
    assert M.auprc_sd(fr, "m") == 1.0 and M.auroc_sd(fr, "m") == 1.0 and M.auroc_dw(fr, "m") == 1.0
    rnd = preds_from(truth, rng.random(len(truth)), P_S=rng.random(len(truth)), P_D=rng.random(len(truth)))
    fr = M.build_eval_frame(truth, rnd)
    assert M.auprc_sd(fr, "m") == pytest.approx(0.5, abs=0.06)  # weighted prevalence 0.5
    assert M.auroc_dw(fr, "m") == pytest.approx(0.5, abs=0.06)
    # proportional-test-like method: P_D undefined -> auroc_dw NaN, not a failure
    fr = M.build_eval_frame(truth, preds_from(truth, isS.astype(float)))
    assert np.isnan(M.auroc_dw(fr, "m")) and not fr.failed["m"].any()


def test_calls_recall_confusion_fdr_hand():
    truth = pd.DataFrame(dict(locus_id=list("abcdef"), oracle_class=list("SSDDWW"), mu1=8.0, mu2=8.0,
                              r2=[np.nan, np.nan, .1, .1, np.nan, np.nan], d=[np.nan, np.nan, 5, 5, np.nan, np.nan],
                              rho=0.0, block_id="b", cell_id="c", split="validation"))
    PS = [0.9, 0.5, 0.85, 0.1, 0.2, 0.1]
    PD = [0.0, 0.3, 0.1, 0.9, 0.95, 0.1]
    pr = preds_from(truth, PS, P_S=PS, P_D=PD)
    pr["E1"] = [np.nan, np.nan, np.nan, 0.99, 0.5, np.nan]  # W locus e has P_D .95 but fails the E gate
    fr = M.build_eval_frame(truth, pr)
    calls = M.three_way_calls(fr, "m", 0.8, 0.8)
    assert list(calls) == ["S", "insufficient", "S", "D", "insufficient", "insufficient"]
    r = M.recall_at(fr, "m", 0.8, 0.8, "all")
    assert r["recall_S"] == pytest.approx(0.5) and r["recall_D"] == pytest.approx(0.5)
    assert r["W_called_S"] == 0 and r["W_called_D"] == 0
    conf = M.confusion(fr, "m", 0.8, 0.8)
    assert conf == {"S": {"S": 1, "insufficient": 1}, "D": {"D": 1, "S": 1}, "W": {"insufficient": 2}}
    f = M.realized_fdr(fr, "m", 0.8, 0.8)
    assert f["n_called_S"] == 2 and f["fdr_S_realized"] == pytest.approx(0.5)
    assert f["fdr_S_nominal"] == pytest.approx(np.mean([0.1, 0.15]))
    assert f["fdr_D_realized"] == 0 and f["fdr_D_nominal"] == pytest.approx(0.1)
    mw = M.matched_w_thresholds(fr, "m", "all")  # 2 W loci: rate must be 0 -> above max W score
    assert mw["t_S"] > 0.2 and mw["rate_S_on_W"] == 0
    assert mw["t_D"] > 0.1 and mw["t_D"] <= 0.9  # gated W locus (0.95) does not constrain t_D
    assert M.three_way_calls(fr, "m", mw["t_S"], mw["t_D"])[4] == "insufficient"


def test_false_s_slope_known():
    rng = np.random.default_rng(8)
    n = 200000
    rho = rng.uniform(-0.8, 0.8, n)
    pfs = 0.1 + 0.2 * rho  # slope 0.02 per 0.1 rho
    PS = np.where(rng.random(n) < pfs, 0.95, 0.05)
    truth = pd.DataFrame(dict(locus_id=[f"l{i}" for i in range(n)], oracle_class="D", mu1=8.0, mu2=8.0, r2=0.1, d=5.0,
                              rho=rho, block_id=[f"b{i % 300}" for i in range(n)], cell_id="c", split="validation"))
    fr = M.build_eval_frame(truth, preds_from(truth, PS, P_S=PS, P_D=1 - PS))
    assert M.false_s_slope(fr, "m", "D", 0.8) == pytest.approx(0.02, abs=0.002)
    flat = np.where(rng.random(n) < 0.1, 0.95, 0.05)
    fr0 = M.build_eval_frame(truth, preds_from(truth, flat, P_S=flat, P_D=1 - flat))
    assert M.false_s_slope(fr0, "m", "D", 0.8) == pytest.approx(0.0, abs=0.002)


# --------------------------------------------------------------------------------------------------
# failure scoring (outer join)
# --------------------------------------------------------------------------------------------------
def test_failure_scoring_least_favourable_and_excluded():
    rng = np.random.default_rng(9)
    truth = synth_truth(rng, n=(200, 200, 200))
    isS = (truth["oracle_class"] == "S").to_numpy()
    s = isS.astype(float)  # perfect
    pr = preds_from(truth, s, P_S=s, P_D=(~isS).astype(float))
    drop = rng.choice(len(truth), 30, replace=False)
    keep = np.setdiff1d(np.arange(len(truth)), drop[:10])
    pr.loc[drop[10:20], "status"] = "timeout"
    pr.loc[drop[20:30], "s"] = np.nan  # NaN s on an ok row = failure
    pr = pr.iloc[keep]  # 10 loci without any row
    lf = M.build_eval_frame(truth, pr, "least_favourable")
    ex = M.build_eval_frame(truth, pr, "excluded")
    assert lf.failed["m"].sum() == 30 and len(lf) == len(truth)
    assert len(ex) == len(truth) - 30
    assert M.auprc_sd(ex, "m") == 1.0
    n_fail_sd = np.isin(truth["oracle_class"].to_numpy()[drop], ("S", "D")).sum()
    assert n_fail_sd > 0 and M.auprc_sd(lf, "m") < 1.0  # failures cost
    # least favourable: failed S ranked below every D, failed D above every S
    fS = lf.failed["m"] & (lf.cls == "S")
    fD = lf.failed["m"] & (lf.cls == "D")
    assert (lf.scores["m"]["s"][fS] == 0).all() and (lf.scores["m"]["s"][fD] == 1).all()
    assert (lf.scores["m"]["P_S"][lf.failed["m"] & (lf.cls == "W")] == 1).all()
    # unexpected loci are an error; duplicate rows are an error
    bad = pd.concat([pr, pr.iloc[:1]])
    with pytest.raises(ValueError):
        M.build_eval_frame(truth, bad)
    extra = pr.copy()
    extra.loc[extra.index[0], "locus_id"] = "nope"
    with pytest.raises(ValueError):
        M.build_eval_frame(truth, extra)


def test_nonconverged_used_and_paired_exclusion():
    rng = np.random.default_rng(10)
    truth = synth_truth(rng, n=(100, 100, 50))
    isS = (truth["oracle_class"] == "S").to_numpy().astype(float)
    a = preds_from(truth, isS, method="a")
    a.loc[0:4, "status"] = "nonconverged"
    b = preds_from(truth, isS, method="b")
    b.loc[10:14, "status"] = "failed"
    fr = M.build_eval_frame(truth, {"a": a, "b": b}, "excluded")
    assert len(fr) == len(truth) - 5  # union of failures dropped for both methods
    assert fr.status["a"][:5].tolist() == ["nonconverged"] * 5  # used and flagged
    lf = M.build_eval_frame(truth, {"a": a, "b": b}, "least_favourable")
    assert not lf.failed["a"].any() and lf.failed["b"].sum() == 5


# --------------------------------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------------------------------
def test_bca_coverage_clustered_mean():
    """Coverage of the 90% BCa interval for a clustered mean with a skewed cluster effect (known truth)."""
    rng = np.random.default_rng(11)
    K, m, true_mean = 40, 5, 1.0
    hits, n_sim = 0, 150
    for _ in range(n_sim):
        eff = rng.exponential(1.0, K)  # mean 1, skewed
        x = np.repeat(eff, m) + rng.normal(0, 0.5, K * m)
        cl = np.repeat(np.arange(K), m)
        r = M.cluster_bootstrap(lambda idx: x[idx].mean(), cl, n_boot=400, rng=rng, level=0.9)
        hits += r["ci_low"] <= true_mean <= r["ci_high"]
    cov = hits / n_sim
    assert 0.83 <= cov <= 0.97, cov  # nominal 0.90; binomial SE ~ 0.025


def test_bootstrap_ignoring_clusters_would_undercover():
    """Negative control for the cluster unit: an i.i.d. (row-level) bootstrap undercovers with strong clustering."""
    rng = np.random.default_rng(12)
    K, m = 30, 20
    hits_c = hits_r = 0
    for _ in range(80):
        eff = rng.normal(0, 1, K)
        x = np.repeat(eff, m) + rng.normal(0, 0.2, K * m)
        cl = np.repeat(np.arange(K), m)
        rc = M.cluster_bootstrap(lambda idx: x[idx].mean(), cl, n_boot=300, rng=rng, level=0.9, bca=False)
        rr = M.cluster_bootstrap(lambda idx: x[idx].mean(), np.arange(K * m), n_boot=300, rng=rng, level=0.9, bca=False)
        hits_c += rc["ci_low"] <= 0 <= rc["ci_high"]
        hits_r += rr["ci_low"] <= 0 <= rr["ci_high"]
    assert hits_c / 80 > 0.78 and hits_r / 80 < 0.5


def test_paired_bootstrap_self_zero_and_reproducible():
    rng = np.random.default_rng(13)
    truth = synth_truth(rng, n=(150, 150, 50), n_blocks=30)
    isS = (truth["oracle_class"] == "S").to_numpy()
    good = isS + rng.normal(0, 0.6, len(truth))
    bad = isS + rng.normal(0, 2.0, len(truth))
    fr = M.build_eval_frame(truth, {"a": preds_from(truth, good, method="a"), "b": preds_from(truth, bad, method="b"),
                                    "a2": preds_from(truth, good, method="a2")})
    r0 = M.paired_bootstrap(fr, M.auprc_sd, "a", "a2", n_boot=200, rng=np.random.default_rng(1))
    assert r0["estimate"] == 0 and r0["ci_low"] == 0 and r0["ci_high"] == 0
    r1 = M.paired_bootstrap(fr, M.auprc_sd, "a", "b", n_boot=300, rng=np.random.default_rng(2))
    r1b = M.paired_bootstrap(fr, M.auprc_sd, "a", "b", n_boot=300, rng=np.random.default_rng(2))
    assert r1["estimate"] > 0 and r1["ci_low"] > 0 and r1["ci_low"] < r1["estimate"] < r1["ci_high"]
    assert r1 == r1b or (r1["ci_low"] == r1b["ci_low"] and r1["ci_high"] == r1b["ci_high"])
    st = M.block_sign_test(fr, "a", "b")
    assert st["n_pos"] > st["n_neg"] and st["p_value"] < 0.05 and st["n_blocks"] == 30
    st0 = M.block_sign_test(fr, "a", "a2")
    assert st0["n_tie"] == 30


def test_bootstrap_recomputes_weights_per_replicate():
    """A replicate with duplicated clusters must equal the metric computed on an explicitly duplicated frame."""
    rng = np.random.default_rng(14)
    truth = synth_truth(rng, n=(100, 100, 20), n_blocks=10, mu_shift=2.0)
    s = (truth["oracle_class"] == "S") + rng.normal(0, 1, len(truth))
    fr = M.build_eval_frame(truth, preds_from(truth, s))
    blocks = np.unique(fr.block)
    pick = np.r_[blocks[:5], blocks[:5]]  # 5 blocks twice
    idx = np.concatenate([np.nonzero(fr.block == b)[0] for b in pick])
    rep = fr.take(idx)
    # recompute from a truth table that literally contains duplicated rows with fresh ids
    t2 = truth.iloc[idx].copy()
    t2["locus_id"] = [f"r{i}" for i in range(len(t2))]
    p2 = preds_from(t2, np.asarray(s)[idx])
    assert M.auprc_sd(rep, "m") == pytest.approx(M.auprc_sd(M.build_eval_frame(t2, p2), "m"), abs=1e-12)
    assert M.auprc_sd(rep, "m") != pytest.approx(M.auprc_sd(fr, "m"))


# --------------------------------------------------------------------------------------------------
# run-level metrics
# --------------------------------------------------------------------------------------------------
def test_compute_run_metrics_json_and_controls():
    rng = np.random.default_rng(15)
    truth = synth_truth(rng, cells=(("L0", "P1", 0.0), ("L1", "P1", 0.0), ("L2", "P3", 0.5)), n=(200, 200, 200))
    isS = (truth["oracle_class"] == "S").to_numpy()
    s = np.clip(isS * 0.6 + rng.random(len(truth)) * 0.5, 0, 1)
    pr = preds_from(truth, s, P_S=s, P_D=(1 - s) * (truth["oracle_class"] != "W"))
    m1 = M.compute_run_metrics(truth, pr)
    m2 = M.compute_run_metrics(truth, pr)
    assert M.dumps(m1) == M.dumps(m2)  # deterministic
    json.loads(M.dumps(m1))
    mm = m1["methods"]["m"]["least_favourable"]
    assert set(mm) >= {"realistic", "ideal", "all", "per_cell", "macro", "per_level", "matched_w", "calibration"}
    assert mm["shuffled_label_control"]["auprc_shuffled"] == pytest.approx(0.5, abs=0.07)
    assert mm["shuffled_label_control"]["weighted_prevalence"] == pytest.approx(0.5)
    assert mm["matched_w"]["rate_S_on_W"] <= 0.025 + 1e-12
    assert len(mm["per_cell"]) == 3


def test_bootstrap_parallel_equals_serial():
    rng = np.random.default_rng(16)
    truth = synth_truth(rng, n=(80, 80, 20), n_blocks=15)
    isS = (truth["oracle_class"] == "S").to_numpy()
    fr = M.build_eval_frame(truth, {"a": preds_from(truth, isS + rng.normal(0, 1, len(truth)), method="a"),
                                    "b": preds_from(truth, isS + rng.normal(0, 2, len(truth)), method="b")})
    r1 = M.paired_bootstrap(fr, M.auprc_sd, "a", "b", n_boot=103, rng=np.random.default_rng(5), n_jobs=1)
    r4 = M.paired_bootstrap(fr, M.auprc_sd, "a", "b", n_boot=103, rng=np.random.default_rng(5), n_jobs=4)
    for k in ("estimate", "ci_low", "ci_high", "se", "z0", "acceleration"):
        assert r1[k] == r4[k]
