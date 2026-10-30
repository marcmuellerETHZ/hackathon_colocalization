"""Generic adapters (DESIGN §4.3) and the frozen test-set config (§8.6)."""
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from eval import adapters as A

ROOT = Path(__file__).resolve().parents[2]


def test_coloc_abf_adapter():
    r = A.coloc_abf([0.1, 0.1, 0.1, 0.2, 0.5])
    assert r["s"] == pytest.approx(0.5 / 0.7) and r["P_S"] == 0.5 and r["P_D"] == 0.2 and r["P_W"] == pytest.approx(0.3)
    assert A.coloc_abf([1, 0, 0, 0, 0])["s"] == 0.5  # undefined -> tie value
    assert math.isnan(r["E1"])


def test_coloc_susie_adapter_and_fallback():
    pp = np.array([[0, 0, 0, 0.9, 0.1], [0, 0, 0, 0.3, 0.7]])
    r = A.coloc_susie(pp)
    assert r["s"] == pytest.approx(0.7) and r["P_S"] == pytest.approx(0.7)
    assert r["P_D"] == pytest.approx((1 - 0.7) * 0.9)
    fb = A.coloc_susie(None, fallback_pp=[0.1, 0.1, 0.1, 0.2, 0.5])
    assert fb == A.coloc_abf([0.1, 0.1, 0.1, 0.2, 0.5])
    with pytest.raises(ValueError):
        A.coloc_susie([], None)


def test_signal_prob_and_sharepro_adapter():
    assert A.signal_prob([0.5, 0.5]) == pytest.approx(0.75)
    assert A.signal_prob([]) == 0.0
    assert A.signal_prob([1.0, 0.2]) == pytest.approx(1.0)
    r = A.locus_prob_with_signals(0.2, pips1=[0.5, 0.5], pips2=[0.9])
    assert r["s"] == 0.2 and r["P_D"] == pytest.approx(0.8 * 0.75 * 0.9)
    r2 = A.locus_prob_with_signals(0.2, p_signal1=1.0, p_signal2=0.5)
    assert r2["P_D"] == pytest.approx(0.4)


def test_inclusion_exclusion_known_answers():
    # DESIGN §5 unit test: a single signal that could belong to either trait -> P_D = 0
    r = A.inclusion_exclusion(a=[0], w=[0], p1t=[0.5], p2t=[0.5], p12t=[1.0])
    assert r["P_D"] == pytest.approx(0.0, abs=1e-12)
    # two independent strong single-trait signals -> P_D ~ 1
    r = A.inclusion_exclusion(a=[0, 0], w=[0, 0], p1t=[0.99, 0], p2t=[0, 0.99], p12t=[0.99, 0.99])
    assert r["P_D"] == pytest.approx(0.99 * 0.99)
    # a shared powered effect -> P_S high, blocks D
    r = A.inclusion_exclusion(a=[0.95, 0], w=[0.97, 0], p1t=[0, 0.9], p2t=[0, 0], p12t=[0, 0.9])
    assert r["P_S"] == pytest.approx(0.95) and r["P_D"] == pytest.approx(0.0, abs=1e-12)
    # brute-force check against enumeration of independent effect states
    rng = np.random.default_rng(0)
    for _ in range(50):
        L = 3
        # per effect: categorical over {blocking(w), only1, only2, both-powered-not-blocking, none}
        probs = rng.dirichlet(np.ones(5), L)
        w, o1, o2, b12 = probs[:, 0], probs[:, 1], probs[:, 2], probs[:, 3]
        p1t, p2t, p12t = o1 + b12, o2 + b12, o1 + o2 + b12
        # exact P(no blocking & some l powered in 1 & some l powered in 2) by enumeration
        import itertools
        tot = 0.0
        for states in itertools.product(range(5), repeat=L):
            pr = np.prod([probs[l, s] for l, s in enumerate(states)])
            if 0 in states:
                continue
            has1 = any(s in (1, 3) for s in states)
            has2 = any(s in (2, 3) for s in states)
            tot += pr * (has1 and has2)
        r = A.inclusion_exclusion(a=np.zeros(L), w=w, p1t=p1t, p2t=p2t, p12t=p12t)
        assert r["P_D"] == pytest.approx(tot, abs=1e-12)


def test_prop_and_lead_adapters():
    r = A.proportional_test(0.3)
    assert r["s"] == 0.3 and math.isnan(r["P_S"]) and math.isnan(r["P_D"])
    assert A.proportional_test(float("nan"))["s"] == 0.5
    assert A.lead_snp(True)["s"] == 1.0 and A.lead_snp(False)["s"] == 0.0


def test_testset_config_matches_design_v4_8_6():
    import pandas as pd

    from eval import metrics as M
    c = yaml.safe_load((ROOT / "datasets" / "configs" / "testset_v1.yaml").read_text())
    g = c["main_grid"]
    assert g["mismatch_levels"] == ["L0", "L1", "L2", "L3", "L4", "L6"]
    po = [(x["pairing"], x["param"], x["value"]) for x in g["pairing_overlap"]]
    assert po == [("P1", "r_p", 0.0), ("P1", "r_p", 0.6), ("P2", "rho", 0.0), ("P2", "rho", 0.3),
                  ("P3", "rho", 0.0), ("P3", "rho", 0.5), ("P3", "rho", 0.8)]
    # v4 §8.6: L3 and L6 only with the overlap-0 cells -> 4 x 7 + 2 x 3 = 34 cells
    exp = [f"{L}|{P}|{par}={v}" for L in g["mismatch_levels"] for P, par, v in po
           if not (L in ("L3", "L6") and v != 0)]
    assert g["cells"] == exp and g["n_cells"] == len(exp) == 34
    assert g["cell_exclusions"]["levels"] == ["L3", "L6"]
    assert g["pairing_types"]["P2"]["N1"] == 100000 and g["pairing_types"]["P2"]["N2"] == 50000
    assert g["pairing_types"]["P3"]["rho_sign"] == "positive"
    thr = g["power_thresholds"]
    assert (thr["GWAS"]["tau"], thr["GWAS"]["weak"], thr["molecular"]["tau"], thr["molecular"]["weak"]) == (6.29, 4.61, 5.26, 3.58)
    pc = c["per_cell"]
    # DECISIONS.md P3-12 (forced by D7, decided before any test-split data existed) replaces DESIGN §8.6's
    # 400/400/400 and G/U <= 200 by 100/100/100 and G, U <= 30 each; everything else stays per §8.6.
    assert (pc["S"], pc["D"], pc["W"], pc["G_U"]["max"]) == (100, 100, 100, 30)
    assert pc["D_generation_strata"]["per_bin"] * (len(pc["D_generation_strata"]["r2_edges"]) - 1) == pc["D"]
    assert c["seed"] == 20261028
    pm = c["power_matching"]
    assert pm["bins"] == [5, 5] and pm["quantiles_over"] == "pooled_S_and_D_of_the_cell"
    assert pm["strata_used_for_matching"] is False and pm["weighted_S_prevalence"] == 0.5
    assert tuple(c["reporting_strata"]["r2_edges"]) == M.DEFAULT_R2_EDGES == (0.0, 0.1, 0.3, 0.5, 0.8, 0.95)
    assert tuple(c["reporting_strata"]["d_edges"]) == M.DEFAULT_D_EDGES == (2, 4, 8, 16, math.inf)
    assert M.N_QUANTILE_BINS == 5
    # pooling weights in the YAML equal what metrics.py computes on the 34-cell grid
    rows = []
    for cid in g["cells"]:
        L, P, pv = cid.split("|")
        for k in ("S", "D", "W"):
            rows.append(dict(locus_id=f"{cid}{k}", oracle_class=k, mu1=8.0, mu2=8.0, r2=0.1, d=5.0,
                             rho=float(pv.split("=")[1]), block_id="b", cell_id=cid, split="validation",
                             mismatch_level=L, pairing_type=P, overlap=float(pv.split("=")[1])))
    t = pd.DataFrame(rows)
    fr = M.build_eval_frame(t, pd.DataFrame({"locus_id": t.locus_id, "method": "m", "s": 0.5, "status": "ok"}))
    p = c["pooling"]
    assert p["realistic"]["levels"] == ["L1", "L2", "L3", "L4", "L6"]
    assert p["realistic"]["pairing_weights"] == {"P1": 0.5, "P2": 0.25, "P3": 0.25}
    assert M.cell_pooling_weights(fr, "realistic") == pytest.approx(p["realistic"]["cell_weights"])
    assert len(p["realistic"]["cell_weights"]) == 27 and sum(p["realistic"]["cell_weights"].values()) == pytest.approx(1)
    assert M.cell_pooling_weights(fr, "ideal") == pytest.approx(p["ideal"]["cell_weights"])
    c3 = c["endpoints"]["C3"]
    assert c3["noninferiority"]["margin"] == M.NI_MARGIN == 0.01 and c3["superiority"]["multiplicity"] == "holm_over_the_two_recalls"
    mw = c["endpoints"]["secondary"]["matched_w"]
    assert mw["confident_S_on_W_max"] == mw["confident_D_on_W_max"] == M.W_LIMIT == 0.025 and mw["s_precedence"]
    b = c["uncertainty"]["bootstrap"]
    assert (b["unit"], b["ci"], b["replicates"]) == ("test_ld_block", "BCa", 2000)
    sh = c["shifted_arm"]
    # P3-12: 1,600 shifted loci = 4 cells x 400 (DESIGN §8.6 had 6,000); P3-11.3: equal split over level x r_p
    assert sh["n_loci"] == 1600 == sum(x["n"] for x in sh["cells"]) and [x["n"] for x in sh["cells"]] == [400] * 4
    ss = c["slow_method_subsample"]  # P3-12
    assert ss["per_cell"] == {"S": 40, "D": 40, "W": 40} and ss["seed"] == "derive(20261028, 'slow_subsample')"
    assert set(ss["methods"]) == {"mvsusie", "colocboost", "colocboost_m1"} and ss["identical_across_methods"]
    assert "open_items" not in c
    assert sh["class_mix"] == {"S": 0.15, "D": 0.15, "W": 0.5, "M": 0.2}
    assert set(sh["levels"]) == set(M.SHIFTED_LEVELS) and sh["levels"]["L2swap"]["n_ref"] == 254
    assert sh["levels"]["L1swap"]["n_ref"] == 2000 and sh["pairing"]["r_p"] == [0.0, 0.6]
    assert "PROPOSAL" not in (ROOT / "datasets" / "configs" / "testset_v1.yaml").read_text()


def test_frozen_test_methods_file_d16():
    """datasets/configs/test_methods_v1.yaml as frozen for frozen-v1 (D16 MVR route, P3-30 operating points)."""
    import eval_test
    from eval.testrun import check_methods_runnable, parse_methods, resolve_subsamples
    f = ROOT / "datasets" / "configs" / "test_methods_v1.yaml"
    d = yaml.safe_load(f.read_text())
    assert d["status"] == "frozen" and d["candidates"]["primary"] is None and d["candidates"]["decision"] == "D16"
    assert eval_test.check_methods_file(f) == []
    ts = yaml.safe_load((ROOT / "datasets" / "configs" / "testset_v1.yaml").read_text())
    probs = check_methods_runnable(d, ROOT, "test", ts)
    # the frozen strength table lives under results/ (main checkout, git-ignored by D6(h)); in a worktree whose
    # results/ lacks it, the only accepted problem is its absence
    assert all("frozen strength table" in p for p in probs), probs
    ms, ra = parse_methods(d)
    by = {m.name: m for m in ms}
    assert resolve_subsamples(ms, ts) == [] and by["mvsusie"].subsample_per_class == 20
    assert by["colocboost"].subsample_per_class == by["colocboost_m1"].subsample_per_class == 40
    assert {n for n, m in by.items() if m.descriptive} == {"famB_exactcoloc_gaussian_M50", "famG_coloc_susie_gpu"}
    assert not set(d["c3_eligible_baselines"]) & {"famB_exactcoloc_gaussian_M50", "famG_coloc_susie_gpu"}
    assert by["coloc_abf"].kwargs["W"] == 0.2 and by["coloc_abf"].kwargs["p12"] == 5e-6
    for n in ("sharepro", "colocboost", "colocboost_m1"):                   # P3-30: no recalibration of a_S
        assert by[n].metric_config["recal_a_S"] is None
    assert by["mvsusie"].metric_config["recal_a_S"] is None and by["mvsusie"].metric_config["recal_a_D"] is None
    for n, m in by.items():
        if n not in ("lead_snp", "prop_coloc", "coloc_prop_test"):
            assert m.metric_config["t_S"] is not None and m.metric_config["t_D"] is not None, n
    assert ra["builder"] == "data.real.extract:build_test_arm"
    assert {by[n].real for n in ("coloc_abf", "coloc_susie", "coloc_susie_mm", "sharepro", "strength_only")} == {True}
    assert not by["mvsusie"].real and not by["famB_exactcoloc_gaussian_M50"].real


def test_test_methods_template_is_refused_until_frozen(tmp_path):
    import eval_test
    t = tmp_path / "template.yaml"
    t.write_text(yaml.safe_dump({"status": "template", "candidates": {"primary": None}, "methods": []}))
    probs = eval_test.check_methods_file(t)
    assert any("status" in p for p in probs) and any("no methods" in p for p in probs)
    assert any("primary" in p for p in probs)
    f = tmp_path / "m.yaml"
    f.write_text(yaml.safe_dump({"status": "frozen", "candidates": {"primary": "a"},
                                 "methods": [{"name": "a", "target": "x:Y", "timeout_s": 5}]}))
    assert eval_test.check_methods_file(f) == []
    f.write_text(yaml.safe_dump({"status": "frozen", "candidates": {"primary": "b"},
                                 "methods": [{"name": "a", "target": "x:Y"}]}))
    probs = eval_test.check_methods_file(f)
    assert any("timeout_s" in p for p in probs) and any("not in the methods list" in p for p in probs)
