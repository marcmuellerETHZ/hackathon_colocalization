#!/usr/bin/env python
"""p5.verify_infra checks (b) and (g): metrics by hand on toy predictions vs src/eval/metrics.py and
scripts/verify_run.py. Own implementations (brute force) of: step-wise AP (ties = one threshold), weighted
AUROC (pairwise, ties 1/2), cell pooling weights (DESIGN v4 §8.6), power-matching weights (5x5 quantile bins per
cell, rule documented in metrics.power_match_weights), C3 matched-W thresholds (smallest t with weighted W rate
<= 2.5%, S precedence), and failure scoring (outer join over the expected list, least-favourable values).
Usage: PYTHONPATH=src python reviews/p5_verify_infra/metrics_check.py <scratch_dir>
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests" / "eval"), str(ROOT / "scripts")]
from eval import metrics as M  # noqa: E402

RES = {}


def check(name, a, b, tol=1e-12):
    ok = (a == b) if isinstance(a, str) else (np.isnan(a) and np.isnan(b)) or abs(a - b) <= tol * max(1, abs(b))
    RES[name] = {"mine": a, "code": b, "ok": bool(ok)}
    return ok


# ------------------------------------------------------------------ own implementations
def my_ap(y, s, w):
    y, s, w = np.asarray(y, bool), np.asarray(s, float), np.asarray(w, float)
    k = w > 0
    y, s, w = y[k], s[k], w[k]
    P = (w * y).sum()
    ap, prev_r = 0.0, 0.0
    for t in sorted(set(s), reverse=True):
        sel = s >= t
        tp = (w * y * sel).sum()
        prec = tp / (w * sel).sum()
        r = tp / P
        ap += (r - prev_r) * prec
        prev_r = r
    return ap


def my_auc(y, s, w):
    y, s, w = np.asarray(y, bool), np.asarray(s, float), np.asarray(w, float)
    pi, ni = np.nonzero(y & (w > 0))[0], np.nonzero(~y & (w > 0))[0]
    num = 0.0
    for i in pi:
        num += w[i] * (w[ni] * ((s[i] > s[ni]) + 0.5 * (s[i] == s[ni]))).sum()
    return num / (w[pi].sum() * w[ni].sum())


def my_cell_weights(cells: pd.DataFrame, setting="realistic"):
    """DESIGN v4 §8.6: equal over levels present (L1-L6 realistic; L0 & overlap 0 ideal), pairing P1 .5 P2 .25
    P3 .25 renormalised over the types present at that level, equal within type."""
    if setting == "realistic":
        c = cells[cells.level.isin(["L1", "L2", "L3", "L4", "L5", "L6"])]
    else:
        c = cells[(cells.level == "L0") & (cells.overlap == 0)]
    pw = {"P1": .5, "P2": .25, "P3": .25}
    out = {}
    levels = sorted(c.level.unique())
    for L in levels:
        cl = c[c.level == L]
        tot = sum(pw[p] for p in cl.pairing.unique())
        for p in cl.pairing.unique():
            cp = cl[cl.pairing == p]
            for cid in cp.cell:
                out[cid] = 1 / len(levels) * pw[p] / tot / len(cp)
    return out


def my_pm_weights(cls, mu1, mu2):
    """One cell. Bins: np.quantile(x, [.2,.4,.6,.8]) of pooled S u D, bin = #edges <= x. Each 2-D bin with both
    classes gets mass (nS+nD)/2 per class, spread evenly; single-class bins 0; then normalise total to 1."""
    sd = np.isin(cls, ["S", "D"])
    w = np.zeros(len(cls))
    idx = np.nonzero(sd)[0]
    e1 = np.quantile(mu1[idx], [.2, .4, .6, .8])
    e2 = np.quantile(mu2[idx], [.2, .4, .6, .8])
    b = {i: (int((e1 <= mu1[i]).sum()), int((e2 <= mu2[i]).sum())) for i in idx}
    for key in set(b.values()):
        mem = [i for i in idx if b[i] == key]
        S = [i for i in mem if cls[i] == "S"]
        D = [i for i in mem if cls[i] == "D"]
        if S and D:
            m = (len(S) + len(D)) / 2
            for i in S:
                w[i] = m / len(S)
            for i in D:
                w[i] = m / len(D)
    return w / w.sum() if w.sum() > 0 else w


def my_threshold(scores_w, w, alpha=0.025):
    s = np.asarray(scores_w, float)
    w = np.asarray(w, float)
    cand = np.unique(np.r_[s[np.isfinite(s)], np.nextafter(s[np.isfinite(s)], np.inf)])
    ok = [t for t in cand if w[s >= t].sum() / w.sum() <= alpha + 1e-12]
    return min(ok) if ok else np.inf


# ------------------------------------------------------------------ toy truth + predictions
def toy(seed=7):
    g = np.random.default_rng(seed)
    cells = pd.DataFrame([("L1|P1|r_p=0.0", "L1", "P1", 0.0), ("L1|P3|rho=0.5", "L1", "P3", 0.5),
                          ("L2|P1|r_p=0.0", "L2", "P1", 0.0), ("L2|P1|r_p=0.6", "L2", "P1", 0.6),
                          ("L2|P2|rho=0.3", "L2", "P2", 0.3), ("L0|P1|r_p=0.0", "L0", "P1", 0.0),
                          ("L0|P3|rho=0.5", "L0", "P3", 0.5)], columns=["cell", "level", "pairing", "overlap"])
    rows = []
    for c in cells.itertuples():
        for cls, n in (("S", 30), ("D", 25), ("W", 40), ("G", 4), ("U", 3)):
            for i in range(n):
                rows.append(dict(locus_id=f"{c.cell}#{cls}{i}", oracle_class=cls, mu1=g.uniform(5, 30),
                                 mu2=g.uniform(4, 20) * (1.3 if cls == "S" else 1.0), r2=g.uniform(0, .9),
                                 d=g.uniform(2, 20), rho=c.overlap, block_id=f"b{g.integers(0, 12)}", cell_id=c.cell,
                                 split="valid", mismatch_level=c.level, pairing_type=c.pairing, overlap=c.overlap,
                                 control=""))
    truth = pd.DataFrame(rows)
    n = len(truth)
    y = truth.oracle_class.to_numpy()
    s = np.round(np.clip(0.5 + 0.25 * ((y == "S") * 1.0 - (y == "D")) + g.normal(0, .25, n), 0, 1), 2)  # ties
    PS = np.round(np.clip(s * g.uniform(.5, 1, n), 0, 1), 3)
    PD = np.round(np.clip((1 - s) * g.uniform(.2, 1, n) * (y != "W") + (y == "W") * g.uniform(0, .6, n), 0, 1), 3)
    pred = pd.DataFrame(dict(locus_id=truth.locus_id, method="toy", s=s, P_S=PS, P_D=PD, P_W=1 - PS - PD,
                             E1=np.nan, E2=np.nan, status="ok"))
    return truth, pred, cells


def main(scratch: Path):
    truth, pred, cells = toy()
    y = truth.oracle_class.to_numpy()
    # ---------------- (b1) elementary metrics with ties and weights
    g = np.random.default_rng(3)
    yy = g.random(300) < .4
    ss = np.round(g.random(300), 1)
    ww = g.uniform(.1, 2, 300)
    check("AP weighted ties: mine vs metrics", my_ap(yy, ss, ww), M.average_precision(yy, ss, ww))
    check("AP weighted ties: sklearn vs metrics", float(average_precision_score(yy, ss, sample_weight=ww)),
          M.average_precision(yy, ss, ww))
    check("AUROC weighted ties: mine vs metrics", my_auc(yy, ss, ww), M.roc_auc(yy, ss, ww))
    check("AUROC weighted ties: sklearn vs metrics", float(roc_auc_score(yy, ss, sample_weight=ww)), M.roc_auc(yy, ss, ww))
    # ---------------- (b2) cell pooling + power matching
    fr = M.build_eval_frame(truth, {"toy": pred})
    for setting in ("realistic", "ideal"):
        mine = my_cell_weights(cells, setting)
        code = M.cell_pooling_weights(fr, setting)
        check(f"cell pooling {setting}: same cells", ",".join(sorted(mine)), ",".join(sorted(code)))
        check(f"cell pooling {setting}: max |diff|", max(abs(mine[k] - code.get(k, np.nan)) for k in mine), 0.0)
    w_mine = np.zeros(len(truth))
    cw = my_cell_weights(cells, "realistic")
    for cid, wc in cw.items():
        m = truth.cell_id.to_numpy() == cid
        w_mine[m] = my_pm_weights(y[m], truth.mu1.to_numpy()[m], truth.mu2.to_numpy()[m]) * wc
    w_code = M.sd_weights(fr, "realistic")
    check("power-matching x pooling weights realistic: max |diff|", float(np.max(np.abs(w_mine - w_code))), 0.0)
    for cid in cw:
        m = truth.cell_id.to_numpy() == cid
        RES.setdefault("pm_invariants", {})[cid] = {
            "S_share": float(w_code[m & (y == "S")].sum() / w_code[m].sum()),
            "cell_mass_over_pool_w": float(w_code[m].sum() / cw[cid]),
            "nonSD_weight": float(w_code[m & ~np.isin(y, ["S", "D"])].sum())}
    inv = RES["pm_invariants"]
    RES["pm_invariants_ok"] = all(abs(v["S_share"] - .5) < 1e-12 and abs(v["cell_mass_over_pool_w"] - 1) < 1e-12
                                  and v["nonSD_weight"] == 0 for v in inv.values())
    sdm = w_mine > 0
    check("AUPRC S vs D realistic: mine vs metrics.auprc_sd",
          my_ap(y[sdm] == "S", pred.s.to_numpy()[sdm], w_mine[sdm]), M.auprc_sd(fr, "toy", "realistic"))
    check("AUROC S vs D realistic: mine vs metrics.auroc_sd",
          my_auc(y[sdm] == "S", pred.s.to_numpy()[sdm], w_mine[sdm]), M.auroc_sd(fr, "toy", "realistic"))
    # D vs W: D and W each get half of the cell pooling weight, uniform within class
    wdw = np.zeros(len(truth))
    for cid, wc in cw.items():
        for k in ("D", "W"):
            m = (truth.cell_id.to_numpy() == cid) & (y == k)
            wdw[m] = wc * .5 / m.sum()
    dwm = wdw > 0
    check("AUROC D vs W realistic: mine vs metrics.auroc_dw",
          my_auc(y[dwm] == "D", pred.P_D.to_numpy()[dwm], wdw[dwm]), M.auroc_dw(fr, "toy", "realistic"))
    # ---------------- (b3) C3 thresholds
    isW = (y == "W") & dwm
    tS = my_threshold(pred.P_S.to_numpy()[isW], wdw[isW])
    callS = pred.P_S.to_numpy() >= tS
    effD = np.where(callS, -np.inf, pred.P_D.to_numpy())
    tD = my_threshold(effD[isW], wdw[isW])
    mw = M.matched_w_thresholds(fr, "toy", "realistic")
    check("C3 t_S: mine vs metrics", tS, mw["t_S"], 0)
    check("C3 t_D (S precedence): mine vs metrics", tD, mw["t_D"], 0)
    # hand-sized case: 40 W loci equal weight, distinct scores -> 1/40 = 2.5% allowed -> t = nextafter(2nd largest)
    sc = np.linspace(0.01, 0.99, 40)
    check("C3 hand case (40 W, 1 allowed)", float(np.nextafter(sc[-2], np.inf)), M.matched_threshold(sc)["t"], 0)
    calls = M.three_way_calls(fr, "toy", mw["t_S"], mw["t_D"])
    rW = {k: float((wdw[isW] * (calls[isW] == k)).sum() / wdw[isW].sum()) for k in ("S", "D")}
    RES["C3 achieved W rates"] = {"S": rW["S"], "D": rW["D"], "ok": rW["S"] <= .025 + 1e-12 and rW["D"] <= .025 + 1e-12}
    # ---------------- (g) failure scoring: outer join, missing / failed / NaN rows
    p2 = pred.copy()
    drop = truth.index[(y == "S")][:5].tolist() + truth.index[(y == "D")][:5].tolist() + \
        truth.index[(y == "W")][:5].tolist()
    p2.loc[truth.index[(y == "S")][5:8], "status"] = "failed"
    p2.loc[truth.index[(y == "D")][5:8], "s"] = np.nan
    p2 = p2.drop(index=drop)
    lf = M.build_eval_frame(truth, {"toy": p2}, "least_favourable")
    ex = M.build_eval_frame(truth, {"toy": p2}, "excluded")
    exp_failed = 15 + 3 + 3
    check("outer join: frame length == expected list", float(len(lf)), float(len(truth)), 0)
    check("failed count (5+5+5 missing, 3 failed, 3 NaN s)", float(lf.failed["toy"].sum()), float(exp_failed), 0)
    check("excluded frame length", float(len(ex)), float(len(truth) - exp_failed), 0)
    # my LF substitution
    s_lf = pred.s.to_numpy().copy()
    PS_lf, PD_lf = pred.P_S.to_numpy().copy(), pred.P_D.to_numpy().copy()
    bad = np.zeros(len(truth), bool)
    bad[drop] = True
    bad[truth.index[(y == "S")][5:8]] = True
    bad[truth.index[(y == "D")][5:8]] = True
    lfv = {"S": (0.0, 0.0, 1.0), "D": (1.0, 1.0, 0.0), "W": (.5, 1.0, 1.0), "G": (.5, 1.0, 1.0), "U": (.5, 1.0, 1.0)}
    for i in np.nonzero(bad)[0]:
        s_lf[i], PS_lf[i], PD_lf[i] = lfv[y[i]]
    check("LF AUPRC: mine vs metrics", my_ap(y[sdm] == "S", s_lf[sdm], w_mine[sdm]), M.auprc_sd(lf, "toy", "realistic"))
    check("LF AUROC D-vs-W: mine vs metrics", my_auc(y[dwm] == "D", PD_lf[dwm], wdw[dwm]), M.auroc_dw(lf, "toy", "realistic"))
    RES["LF lowers AUPRC vs complete"] = {"complete": M.auprc_sd(fr, "toy", "realistic"),
                                          "lf": M.auprc_sd(lf, "toy", "realistic"),
                                          "ok": M.auprc_sd(lf, "toy", "realistic") < M.auprc_sd(fr, "toy", "realistic")}
    rm = M.compute_run_metrics(truth, p2)
    check("compute_run_metrics n_failed counts missing rows", float(rm["methods"]["toy"]["failures"]["n_failed"]),
          float(exp_failed), 0)
    check("compute_run_metrics status 'missing' count", float(rm["methods"]["toy"]["failures"]["status_counts"].get("missing", 0)),
          15.0, 0)
    # ---------------- (b4) end-to-end through the harness + verify_run
    import eval_toy
    from eval.harness import MethodSpec, run_method
    import verify_run
    d = scratch / "e2e"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    ls = d / "toyset.h5"
    _, ttruth, tp = eval_toy.make_locus_set(ls, n_per_class=20, seed=11)
    ids = ttruth.locus_id.tolist()
    spec = MethodSpec("eval_dummy_methods:FlakyMethod",
                      {"fail_ids": ids[:2], "missing_ids": ids[25:27], "nc_ids": ids[45:47]})
    rr = run_method(spec, ls, results_dir=d / "results", mode="cpu", n_workers=2, timeout_s=30,
                    log=lambda *a: None)
    pr = pd.read_parquet(rr.run_dir / "predictions.parquet")
    RES["e2e rows == expected (missing outputs become rows)"] = {"rows": len(pr), "expected": len(ids),
                                                                 "ok": len(pr) == len(ids)}
    RES["e2e statuses"] = pr.status.value_counts().to_dict()
    met = json.loads((rr.run_dir / "metrics.json").read_text())
    # my recomputation of the headline from predictions + truth (single cell L1|P1, realistic)
    tt = ttruth.set_index("locus_id").loc[pr.locus_id]
    yt = tt.oracle_class.to_numpy()
    sv = pr.s.to_numpy().copy()
    failm = ~pr.status.isin(["ok", "nonconverged"]).to_numpy() | np.isnan(sv)
    for i in np.nonzero(failm)[0]:
        sv[i] = lfv[yt[i]][0]
    wpm = my_pm_weights(yt, np.abs(tt.mu1.to_numpy()), np.abs(tt.mu2.to_numpy()))
    m = wpm > 0
    check("e2e metrics.json auprc_sd realistic vs mine",
          my_ap(yt[m] == "S", sv[m], wpm[m]), met["methods"]["flaky"]["least_favourable"]["realistic"]["auprc_sd"], 1e-12)
    check("e2e n_failed (2 raised + 2 missing)", float(met["methods"]["flaky"]["failures"]["n_failed"]), 4.0, 0)
    probs = verify_run.verify(rr.run_dir)
    RES["verify_run on untouched run"] = {"problems": probs, "ok": probs == []}
    # positive controls: tamper metrics.json (and fix its manifest hash) -> mismatch must be reported
    from eval.harness import sha256_file
    met["methods"]["flaky"]["least_favourable"]["realistic"]["auprc_sd"] += 0.01
    (rr.run_dir / "metrics.json").write_text(json.dumps(met))
    man = json.loads((rr.run_dir / "manifest.json").read_text())
    man["metrics_sha256"] = sha256_file(rr.run_dir / "metrics.json")
    (rr.run_dir / "manifest.json").write_text(json.dumps(man))
    probs2 = verify_run.verify(rr.run_dir)
    RES["verify_run detects tampered metric"] = {"problems": probs2[:3], "ok": any("auprc_sd" in p for p in probs2)}
    # drop a prediction row (and re-hash) -> verify_run must flag the expected-id mismatch
    pr.drop(index=[5]).to_parquet(rr.run_dir / "predictions.parquet", index=False)
    man["predictions_sha256"] = sha256_file(rr.run_dir / "predictions.parquet")
    (rr.run_dir / "manifest.json").write_text(json.dumps(man))
    probs3 = verify_run.verify(rr.run_dir)
    RES["verify_run detects a dropped prediction row"] = {"problems": probs3[:3],
                                                          "ok": any("expected ids" in p for p in probs3)}
    n_bad = sum(1 for v in RES.values() if isinstance(v, dict) and "ok" in v and not v["ok"])
    n_bad += 0 if RES["pm_invariants_ok"] else 1
    RES["n_failed_checks"] = n_bad
    print(json.dumps(RES, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
