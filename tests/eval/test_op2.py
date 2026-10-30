"""Known-answer tests for eval.metrics.op2_thresholds (DESIGN_P2 §3 OP2; matched_w_thresholds unchanged)."""
import numpy as np
import pandas as pd
import pytest

from eval import metrics as M
from eval import op2 as O2


def _truth(nS, nD, nW, mu_D=None):
    n = nS + nD + nW
    cls = ["S"] * nS + ["D"] * nD + ["W"] * nW
    mu = np.full(n, 8.0)
    if mu_D is not None:
        mu[nS:nS + nD] = mu_D
    return pd.DataFrame(dict(locus_id=[f"l{i}" for i in range(n)], oracle_class=cls, mu1=mu, mu2=mu,
                             r2=[0.2 if c == "D" else np.nan for c in cls], d=[5.0 if c == "D" else np.nan for c in cls],
                             rho=0.0, block_id=[f"b{i % 7}" for i in range(n)], cell_id="L1|P1|r_p=0.0",
                             split="validation", mismatch_level="L1", pairing_type="P1", overlap=0.0, control=""))


def _frame(truth, PS, PD, E1=None):
    pr = pd.DataFrame({"locus_id": truth["locus_id"], "method": "m", "s": PS, "P_S": PS, "P_D": PD, "P_W": np.nan,
                       "E1": np.nan if E1 is None else E1, "E2": np.nan, "status": "ok"})
    return M.build_eval_frame(truth, pr)


def test_w_side_binds_equals_op1():
    t = _truth(40, 40, 40)
    PS = np.r_[np.ones(40), np.zeros(40), np.arange(40) / 40]  # D never called; W like test_matched_threshold_hand
    fr = _frame(t, PS, np.zeros(120))
    op1 = M.matched_w_thresholds(fr, "m", "realistic")
    op2 = O2.op2_thresholds(fr, "m", "realistic")
    assert op2["n_D_matched"] == 40 and op2["n_W"] == 40
    assert op2["op2_feasible"] and op2["t_S"] == op1["t_S"] == op2["t_S_call"]
    assert 38 / 40 < op2["t_S"] <= 39 / 40
    assert op2["rate_S_on_W"] == pytest.approx(1 / 40) and op2["rate_S_on_D"] == 0
    assert op2["t_D"] == op1["t_D"]


def test_d_side_binds():
    t = _truth(40, 40, 40)
    PS = np.r_[np.ones(40), np.arange(40) / 40, np.zeros(40)]  # W all 0; D 0, 1/40, ..., 39/40
    fr = _frame(t, PS, np.zeros(120))
    op1 = M.matched_w_thresholds(fr, "m", "realistic")
    op2 = O2.op2_thresholds(fr, "m", "realistic")
    assert op1["t_S"] == np.nextafter(0, 1)  # OP1: just above the W scores (all 0) -> 39 of 40 D called S
    assert (fr.scores["m"]["P_S"][40:80] >= op1["t_S"]).sum() == 39
    assert op2["op2_feasible"]
    assert 35 / 40 < op2["t_S"] <= 36 / 40  # at most 4 of 40 D (10%) called S
    assert op2["rate_S_on_D"] == pytest.approx(0.10) and op2["rate_S_on_W"] == 0
    assert (fr.scores["m"]["P_S"][40:80] >= op2["t_S"]).sum() == 4


def test_infeasible_gives_t_s_1_and_no_s_calls():
    t = _truth(40, 40, 40)
    PS = np.r_[np.ones(40), np.r_[np.ones(5), np.zeros(35)], np.zeros(40)]  # 5/40 = 12.5% of D at P_S = 1.0
    PD = np.r_[np.zeros(80), np.arange(40) / 40]
    fr = _frame(t, PS, PD)
    op2 = O2.op2_thresholds(fr, "m", "realistic")
    assert op2["op2_feasible"] is False and op2["t_S"] == 1.0 and op2["t_S_call"] == np.inf
    assert op2["rate_S_on_D"] == 0 and op2["rate_S_on_W"] == 0
    assert 38 / 40 < op2["t_D"] <= 39 / 40  # t_D from W P_D alone (no S precedence binds)
    # exactly at the boundary: 4/40 D at 1.0 is feasible with t_S = 1.0 itself
    PS2 = np.r_[np.ones(40), np.r_[np.ones(4), np.zeros(36)], np.zeros(40)]
    op2b = O2.op2_thresholds(_frame(t, PS2, PD), "m", "realistic")
    assert op2b["op2_feasible"] is True and op2b["t_S"] == op2b["t_S_call"] <= 1.0
    assert op2b["rate_S_on_D"] == pytest.approx(0.10)


def test_t_d_after_op2_s_precedence():
    """A W locus called S at OP1 but not at OP2 becomes eligible for D and can raise t_D."""
    t = _truth(40, 40, 40)
    PS_W = np.r_[np.zeros(39), 0.5]  # W locus 39: P_S .5 (1/40 = 2.5% -> allowed at OP1 W side)
    PD_W = np.r_[np.arange(39) / 100, 0.99]
    PS_D = np.r_[np.full(8, 0.5), np.zeros(32)]  # 8/40 = 20% of D at 0.5 -> OP2 must exceed 0.5
    PS = np.r_[np.ones(40), PS_D, PS_W]
    PD = np.r_[np.zeros(80), PD_W]
    fr = _frame(t, PS, PD)
    op1 = M.matched_w_thresholds(fr, "m", "realistic")
    op2 = O2.op2_thresholds(fr, "m", "realistic")
    assert op1["t_S"] <= 0.5 and op2["t_S"] > 0.5
    # OP1: locus 39 is S (precedence) -> D-eligible W top is 0.38 (1 of 40 allowed) -> t_D just above 0.37
    assert 0.37 < op1["t_D"] <= 0.38
    # OP2: locus 39 (P_D .99) is D-eligible and takes the single allowed W call -> t_D just above 0.38
    assert 0.38 < op2["t_D"] <= 0.99 and op2["rate_D_on_W"] == pytest.approx(1 / 40)


def test_only_power_matched_d_count():
    """D loci in power bins without S loci get S-vs-D weight 0 and are not in the D share."""
    rng = np.random.default_rng(3)
    nS, nD, nW = 60, 60, 40
    t = _truth(nS, nD, nW)
    t.loc[:nS - 1, ["mu1", "mu2"]] = rng.uniform(6, 10, size=(nS, 2))
    muD = np.r_[rng.uniform(6, 10, size=(40, 2)), np.full((20, 2), 50.0)]  # 20 D far above every S: unmatched
    t.loc[nS:nS + nD - 1, ["mu1", "mu2"]] = muD
    PS = np.r_[np.ones(nS), np.zeros(40), np.ones(20), np.zeros(nW)]  # only the unmatched D carry P_S = 1
    fr = _frame(t, PS, np.zeros(len(t)))
    wsd = M.sd_weights(fr, "realistic")
    matched = (fr.cls == "D") & (wsd > 0)
    assert not matched[nS + 40:nS + 60].any()  # the 20 high-mu D are unmatched (their bins hold no S)
    op2 = O2.op2_thresholds(fr, "m", "realistic")
    assert op2["n_D_matched"] == int(matched.sum())
    # matched D never reach P_S 1 -> no D constraint: OP2 = OP1 (just above the W scores, all 0)
    assert op2["op2_feasible"] and op2["t_S"] == M.matched_w_thresholds(fr, "m", "realistic")["t_S"]
    assert op2["t_S_D_side"] <= op2["t_S_W_side"] and op2["rate_S_on_D"] == 0


def test_bruteforce_smallest_feasible_and_limits():
    rng = np.random.default_rng(11)
    t = _truth(50, 80, 120)
    for rep in range(20):
        PS = np.r_[rng.uniform(0, 1, 50), rng.beta(1, 3 + rep % 4, 80), rng.beta(1, 6, 120)].round(3)
        PD = rng.uniform(0, 1, 250)
        fr = _frame(t, PS, PD)
        op2 = O2.op2_thresholds(fr, "m", "realistic")
        w = M.dw_weights(fr, "realistic")
        isW, isD = fr.cls == "W", fr.cls == "D"
        ps = fr.scores["m"]["P_S"]

        def ok(th):
            c = ps >= th
            return (w[isW] * c[isW]).sum() / w[isW].sum() <= 0.025 + 1e-12 and c[isD].mean() <= 0.10 + 1e-12

        vals = np.unique(ps)
        cands = np.r_[vals, np.nextafter(vals, np.inf)]
        feas = [c for c in cands if ok(c) and c <= 1.0]
        assert op2["op2_feasible"] == bool(feas)
        if feas:
            best = min(feas)
            # same call set as the brute-force minimum (thresholds may differ only between equal scores)
            assert np.array_equal(ps >= op2["t_S"], ps >= best)
            assert ok(op2["t_S"])
        # d_limit = 1 reduces OP2 to OP1 exactly
        op2_1 = O2.op2_thresholds(fr, "m", "realistic", d_limit=1.0)
        op1 = M.matched_w_thresholds(fr, "m", "realistic")
        assert op2_1["t_S"] == op1["t_S"] and op2_1["t_D"] == op1["t_D"]


def test_no_p_s_method_gets_only_t_d():
    t = _truth(10, 10, 40)
    fr = _frame(t, np.r_[np.ones(20), np.zeros(40)], np.r_[np.zeros(20), np.arange(40) / 40])
    fr.scores["m"]["P_S"][:] = np.nan
    op2 = O2.op2_thresholds(fr, "m", "realistic")
    assert "t_S" not in op2 and 38 / 40 < op2["t_D"] <= 39 / 40
