#!/usr/bin/env python
"""p7 verify, checks 3 and 4: independent recomputation of the co-primary endpoints and C3 rates of test bundle
ab94c5a1 from predictions + truth. Does NOT import src/eval (written from DESIGN v4 §1, §4.3, §8.1, §8.6 and the
definitions in src/eval/README.md items 5-12).

Usage: python reviews/p7_verify_testrun/recompute.py <project_root> [methods...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(sys.argv[1])
METHODS = sys.argv[2:] or ["coloc_abf", "coloc_susie", "coloc_susie_mm", "strength_only", "sharepro",
                            "famB_exactcoloc_gaussian_M50", "famG_coloc_susie_gpu"]
B = ROOT / "results/eval_test/test_frozen-v1_ab94c5a1"
SETS = ROOT / "data/processed/testset_v1/test_frozen-v1_ab94c5a1/sets"
TAG = "test_frozen-v1_ab94c5a1"
PAIR_W = {"P1": 0.5, "P2": 0.25, "P3": 0.25}
SIDE = __import__("os").environ.get("QBIN_SIDE", "right")  # edge-inclusion convention for quintile bins
# QBIN_QS=linspace reproduces np.linspace(0,1,6)[1:-1] (0.6000000000000001), which matters only when (n-1)*q is an integer
QS = np.linspace(0, 1, 6)[1:-1] if __import__("os").environ.get("QBIN_QS") == "linspace" else np.array([0.2, 0.4, 0.6, 0.8])
SUB = __import__("os").environ.get("SUBSAMPLE") == "1"
REALISTIC = ["L1", "L2", "L3", "L4", "L6"]
frozen = {m["name"]: m for m in yaml.safe_load((ROOT / "configs/test_methods_v1.yaml").read_text())["methods"]}


# ---------------------------------------------------------------- weights
def pooling_weights(cells: pd.DataFrame, levels):
    """cells: one row per cell (cell_id, mismatch_level, pairing_type, overlap). DESIGN §8.6."""
    c = cells[cells.mismatch_level.isin(levels)]
    w = {}
    lv = sorted(c.mismatch_level.unique())
    for L in lv:
        cl = c[c.mismatch_level == L]
        types = sorted(cl.pairing_type.unique())
        tot = sum(PAIR_W[t] for t in types)
        for t in types:
            ct = cl[cl.pairing_type == t]
            for cid in ct.cell_id:
                w[cid] = (1 / len(lv)) * (PAIR_W[t] / tot) / len(ct)
    return w


def quintile_bin(x):
    edges = np.quantile(x, QS)
    return np.searchsorted(edges, x, side=SIDE)


def sd_weights(df, cw):
    """Power-matched S/D weights. Per cell: 5x5 bins on (|mu1|,|mu2|) quintiles of pooled S u D; in a bin holding
    both classes each class gets mass (nS+nD)/2 (uniform within class), else 0; cell mass -> 1 x pooling weight."""
    out = pd.Series(0.0, index=df.index)
    for cid, g in df.groupby("cell_id"):
        if cid not in cw:
            continue
        b = quintile_bin(g.mu1.abs().to_numpy()) * 5 + quintile_bin(g.mu2.abs().to_numpy())
        y = (g.oracle_class == "S").to_numpy()
        w = np.zeros(len(g))
        for bb in np.unique(b):
            m = b == bb
            nS, nD = int((y & m).sum()), int((~y & m).sum())
            if nS and nD:
                w[m & y] = (nS + nD) / 2 / nS
                w[m & ~y] = (nS + nD) / 2 / nD
        if w.sum() > 0:
            out.loc[g.index] = w / w.sum() * cw[cid]
    return out


def dw_weights(df, cw):
    """D and W each half the cell's pooling weight, uniform within class (README item 10)."""
    out = pd.Series(0.0, index=df.index)
    for (cid, cls), g in df.groupby(["cell_id", "oracle_class"]):
        if cid in cw:
            out.loc[g.index] = cw[cid] / 2 / len(g)
    return out


# ---------------------------------------------------------------- metrics
def wap(score, y, w):
    """Weighted step-wise average precision; tied scores form one threshold."""
    s = pd.DataFrame({"s": score, "tp": w * y, "fp": w * (1 - y)}).groupby("s", sort=True)[["tp", "fp"]].sum()[::-1]
    tp, fp = s.tp.cumsum().to_numpy(), s.fp.cumsum().to_numpy()
    P = tp[-1]
    prec = tp / np.maximum(tp + fp, 1e-300)
    rec = tp / P
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def wauc(score, y, w):
    """Weighted AUROC (ties count 1/2)."""
    s = pd.DataFrame({"s": score, "p": w * y, "n": w * (1 - y)}).groupby("s", sort=True)[["p", "n"]].sum()
    n_below = s.n.cumsum().to_numpy() - s.n.to_numpy()
    return float(np.sum(s.p.to_numpy() * (n_below + 0.5 * s.n.to_numpy())) / (s.p.sum() * s.n.sum()))


# ---------------------------------------------------------------- data
def load(method, setkey, truthfile):
    t = pd.read_parquet(SETS / truthfile, columns=["locus_id", "oracle_class", "mu1", "mu2", "cell_id",
                                                    "mismatch_level", "pairing_type", "overlap", "block_id"])
    if SUB:  # slow-method subsample benchmark (ColocBoost 40/40/40 per cell)
        ids = set((B / "benchmarks/sim/ids_subsample40.txt").read_text().split())
        t = t[t.locus_id.isin(ids)]
    p = pd.read_parquet(B / "runs" / f"{TAG}__{method}__{setkey}" / "predictions.parquet",
                        columns=["locus_id", "s", "P_S", "P_D", "E1", "E2", "status"])
    df = t.merge(p, on="locus_id", how="left")
    has_ps, has_pd = p.P_S.notna().any(), p.P_D.notna().any()
    fail = df.status.isna() | df.status.isin(["failed", "timeout"]) | df.s.isna()
    if has_ps:
        fail |= df.P_S.isna()
    if has_pd:
        fail |= df.P_D.isna()
    df["fail"] = fail
    # least favourable scoring (DESIGN §4.3)
    lf = df.copy()
    c = lf.oracle_class
    for col, sv, dv, ov in [("s", 0.0, 1.0, np.nan), ("P_S", 0.0, 1.0, 1.0), ("P_D", 1.0, 0.0, 1.0)]:
        v = np.where(c == "S", sv, np.where(c == "D", dv, ov))
        lf.loc[fail, col] = v[fail.to_numpy()]
    for col in ("E1", "E2"):
        lf.loc[fail, col] = 1.0
    return lf


def endpoints(df, levels):
    cells = df.drop_duplicates("cell_id")[["cell_id", "mismatch_level", "pairing_type", "overlap"]]
    if levels == "ideal":
        cw = pooling_weights(cells[(cells.mismatch_level == "L0") & (cells.overlap == 0)], ["L0"])
    else:
        cw = pooling_weights(cells, levels)
    sd = df[df.oracle_class.isin(["S", "D"]) & df.cell_id.isin(cw)].copy()
    wsd = sd_weights(sd, cw)
    ysd = (sd.oracle_class == "S").astype(float).to_numpy()
    dw = df[df.oracle_class.isin(["D", "W"]) & df.cell_id.isin(cw)].copy()
    wdw = dw_weights(dw, cw)
    ydw = (dw.oracle_class == "D").astype(float).to_numpy()
    res = dict(auprc_sd=wap(sd.s.to_numpy(), ysd, wsd.to_numpy()),
               auroc_sd=wauc(sd.s.to_numpy(), ysd, wsd.to_numpy()),
               auprc_sd_unweighted=wap(sd.s.to_numpy(), ysd, np.ones(len(sd))),
               auroc_dw=wauc(dw.P_D.to_numpy(), ydw, wdw.to_numpy()) if dw.P_D.notna().all() else None,
               auroc_dw_unweighted=wauc(dw.P_D.to_numpy(), ydw, np.ones(len(dw))) if dw.P_D.notna().all() else None,
               weighted_S_prev=float((wsd * ysd).sum() / wsd.sum()), n_sd=len(sd), n_dw=len(dw))
    return res, cw, dw, wdw


def c3(df, method, cw, dw, wdw):
    mc = frozen[method]["metric_config"]
    tS, tD = mc["t_S"], mc["t_D"]
    if tS is None:
        return None
    egate = (df.E1.isna() | (df.E1 >= 0.95)) & (df.E2.isna() | (df.E2 >= 0.95))
    callS = df.P_S >= tS
    callD = (~callS) & (df.P_D >= tD) & egate
    out = {}
    for cls in ("W", "D", "S"):
        m = (df.oracle_class == cls) & df.cell_id.isin(cw)
        if cls in ("W", "D"):
            w = wdw.reindex(df.index[m]).to_numpy()
        else:  # S recall: uniform within cell x pooling weight
            w = df[m].groupby("cell_id").locus_id.transform(lambda g: 1 / len(g)).to_numpy() * \
                df[m].cell_id.map(cw).to_numpy()
        out[f"confS_on_{cls}_weighted"] = float((callS[m] * w).sum() / w.sum())
        out[f"confD_on_{cls}_weighted"] = float((callD[m] * w).sum() / w.sum())
        out[f"confS_on_{cls}_unweighted_allcells"] = float(callS[df.oracle_class == cls].mean())
    # confusion (unweighted, all cells of this set)
    call = pd.Series(np.where(callS, "S", np.where(callD, "D", "insufficient")), index=df.index)
    out["confusion_D_row_this_set"] = {k: int(v) for k, v in call[df.oracle_class == "D"].value_counts().items()}
    return out


bench = json.loads((B / ("benchmarks/sim/subsample_8685f328/benchmark.json" if SUB else "benchmarks/sim/benchmark.json")).read_text())
report = {}
for method in METHODS:
    main = load(method, "sim_main", "testset_v1.truth.parquet")
    shf = load(method, "sim_shifted", "testset_v1_shifted.truth.parquet")
    both = pd.concat([main, shf], ignore_index=True)
    r = {}
    for setting, df, lv in [("realistic", main, REALISTIC), ("ideal", main, "ideal"),
                            ("shifted", shf, ["L1swap", "L2swap"])]:
        e, cw, dw, wdw = endpoints(df, lv)
        bb = bench["methods"][method]["least_favourable"][setting]
        e["bundle"] = bb
        e["diff_auprc_sd"] = e["auprc_sd"] - bb["auprc_sd"]
        e["diff_auroc_sd"] = e["auroc_sd"] - bb["auroc_sd"]
        e["diff_auroc_dw"] = None if e["auroc_dw"] is None or bb["auroc_dw"] is None else e["auroc_dw"] - bb["auroc_dw"]
        if setting == "realistic":
            e["c3"] = c3(df, method, cw, dw, wdw)
            e["c3_bundle_at_config"] = bench["methods"][method]["least_favourable"]["at_config_thresholds"]
            e["c3_bundle_confusion_D"] = bench["methods"][method]["least_favourable"]["confusion_config_thresholds"].get("D")
            e["n_fail_main"] = int(main.fail.sum())
        r[setting] = e
    # unweighted pooled over everything (main + shifted) vs bundle "all"? (bundle 'all' = equal weight per cell)
    cells = both.drop_duplicates("cell_id")
    cw_all = {c: 1 / len(cells) for c in cells.cell_id}
    sd = both[both.oracle_class.isin(["S", "D"])]
    wsd = sd_weights(sd, cw_all)
    dwf = both[both.oracle_class.isin(["D", "W"])]
    r["all_equal_cell"] = dict(auprc_sd=wap(sd.s.to_numpy(), (sd.oracle_class == "S").to_numpy() * 1.0, wsd.to_numpy()),
                               auroc_dw=wauc(dwf.P_D.to_numpy(), (dwf.oracle_class == "D").to_numpy() * 1.0,
                                             dw_weights(dwf, cw_all).to_numpy()),
                               bundle=bench["methods"][method]["least_favourable"]["all"])
    report[method] = r
    rr = r["realistic"]
    print(f"{method:30s} realistic AUPRC {rr['auprc_sd']:.6f} (bundle {rr['bundle']['auprc_sd']:.6f}, d {rr['diff_auprc_sd']:+.2e}) "
          f"AUROC_DW {rr['auroc_dw']:.6f} (bundle {rr['bundle']['auroc_dw']:.6f}, d {rr['diff_auroc_dw']:+.2e}) "
          f"unweighted AUPRC {rr['auprc_sd_unweighted']:.4f}", flush=True)

print(json.dumps(report, indent=1, default=str))
