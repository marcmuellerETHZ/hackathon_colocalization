"""Independent recomputation of coloc_susie_mm M4 (m4_subsample@1 main, valid_grid_v1) endpoints, OP1 and OP2.

Own implementation: does NOT import src/eval/metrics.py, src/eval/op2.py or scripts/baselines_p4.py.
AP / AUROC via sklearn (sample_weight); weights and thresholds re-implemented from DESIGN v4 §4.3/§8.1/§8.6/C3
and DESIGN_P2 §3. Validation split only (asserted).
Usage: python reviews/p9_verify_mmM4/verify_mm_m4.py <project_root>
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
RUNS = ["results/baselines_m4/runs/20261027T192144Z_coloc_susie_mm_valid_grid_v1_977229",
        "results/baselines_m4/runs/20261027T192145Z_coloc_susie_mm_valid_grid_v1_9ee9eb"]
REAL = {"L1", "L2", "L3", "L4", "L5", "L6"}
PW = {"P1": 0.5, "P2": 0.25, "P3": 0.25}
WLIM, DLIM = 0.025, 0.10

ids = pd.read_csv(ROOT / "results/baselines_m4/m4_subsample_v1/main.txt", header=None)[0].astype(str)
assert len(ids) == 12170 and ids.is_unique
tr = pd.read_parquet(ROOT / "data/processed/sets/valid_grid_v1.truth.parquet")
assert (tr["split"] == "valid").all()
tr = tr[tr["locus_id"].isin(set(ids))].reset_index(drop=True)
assert len(tr) == 12170
parts = [pd.read_parquet(ROOT / r / "predictions.parquet") for r in RUNS]
assert not set(parts[0].locus_id) & set(parts[1].locus_id), "halves overlap"
pr = pd.concat(parts, ignore_index=True)
assert set(pr.locus_id) == set(ids)
df = tr.merge(pr[["locus_id", "s", "P_S", "P_D", "E1", "E2", "status"]], on="locus_id", how="left")
cls = df["oracle_class"].to_numpy()
fail = (df["status"].isna() | df["status"].isin(["failed", "timeout"]) | df[["s", "P_S", "P_D"]].isna().any(axis=1)).to_numpy()
# least-favourable failure scoring (DESIGN §4.3)
LF = {"S": (0, 0, 1), "D": (1, 1, 0)}
for i in np.nonzero(fail)[0]:
    s, ps, pd_ = LF.get(cls[i], (0.5, 1, 1))
    df.loc[i, ["s", "P_S", "P_D"]] = [s, ps, pd_]
n_fail, n_nonconv = int(fail.sum()), int((df["status"] == "nonconverged").sum())
assert df[["E1", "E2"]].isna().all().all()  # no E gate for this method

cell, lev, pair, ov = (df[c].to_numpy() for c in ("cell_id", "mismatch_level", "pairing_type", "overlap"))

def cell_w(setting):
    c = df.drop_duplicates("cell_id")
    if setting == "realistic":
        c = c[c.mismatch_level.isin(REAL)]
    else:
        c = c[(c.mismatch_level == "L0") & (c.overlap.astype(float) == 0)]
    out, levels = {}, sorted(c.mismatch_level.unique())
    for L in levels:
        cl = c[c.mismatch_level == L]
        tot = sum(PW[p] for p in cl.pairing_type.unique())
        for p, g in cl.groupby("pairing_type"):
            for cid in g.cell_id:
                out[cid] = PW[p] / tot / len(g) / len(levels)
    return out

def qbin(x):
    e = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(e, x, side="right")

def sd_w(setting):
    cw, w = cell_w(setting), np.zeros(len(df))
    m1, m2 = df["mu1"].abs().to_numpy(), df["mu2"].abs().to_numpy()
    for cid, wc in cw.items():
        idx = np.nonzero((cell == cid) & np.isin(cls, ["S", "D"]))[0]
        k = qbin(m1[idx]) * 5 + qbin(m2[idx])
        isS = cls[idx] == "S"
        loc = np.zeros(len(idx))
        for b in np.unique(k):
            inb = k == b
            nS, nD = (inb & isS).sum(), (inb & ~isS).sum()
            if nS and nD:
                half = (nS + nD) / 2
                loc[inb & isS] = half / nS
                loc[inb & ~isS] = half / nD
        if loc.sum() > 0:
            w[idx] = wc * loc / loc.sum()
    return w

def dw_w(setting):
    cw, w = cell_w(setting), np.zeros(len(df))
    for cid, wc in cw.items():
        for k in ("D", "W"):
            m = (cell == cid) & (cls == k)
            if m.any():
                w[m] = wc * 0.5 / m.sum()
    return w

s, PS, PD = (df[c].to_numpy(float) for c in ("s", "P_S", "P_D"))
res = {"n_loci": len(df), "failures": n_fail, "nonconverged": n_nonconv}
for setting in ("realistic", "ideal"):
    w = sd_w(setting); m = w > 0
    res[f"{setting}_AUPRC_SvsD"] = average_precision_score(cls[m] == "S", s[m], sample_weight=w[m])
    res[f"{setting}_AUPRC_SvsD_unweighted"] = average_precision_score(cls[m] == "S", s[m])
    w2 = dw_w(setting); m2 = w2 > 0
    res[f"{setting}_AUROC_DvsW"] = roc_auc_score(cls[m2] == "D", PD[m2], sample_weight=w2[m2])
    res[f"{setting}_AUROC_DvsW_unweighted"] = roc_auc_score(cls[m2] == "D", PD[m2])

# ---- operating points on realistic (M4 L1-L6) ----
wdw = dw_w("realistic"); isW = (cls == "W") & (wdw > 0)
wW = wdw[isW] / wdw[isW].sum()
matchedD = (cls == "D") & (sd_w("realistic") > 0)

def rate_W(score, t):  # weighted share of W loci with score >= t
    return float(wW[score[isW] >= t].sum())

def rate_D(t):  # unweighted share of power-matched D with P_S >= t
    return float((PS[matchedD] >= t).mean())

def cands(vals):
    v = np.unique(vals[np.isfinite(vals)])
    return np.sort(np.r_[v, np.nextafter(v, np.inf)])

def smallest(ok, vals):  # brute force: smallest candidate t satisfying ok(t)
    for t in cands(vals):
        if ok(t):
            return float(t)
    return float("inf")

def fit_tD(tS):
    eff = np.where(PS >= tS, -np.inf, PD)
    return smallest(lambda t: rate_W(eff, t) <= WLIM + 1e-12, eff[isW]), eff

def summary(tS, tD, eff):
    callS, callD = PS >= tS, eff >= tD
    wc = cell_w("realistic")
    def rec(k, call):
        num = den = 0.0
        for cid, c in wc.items():
            m = (cell == cid) & (cls == k)
            num += c * call[m].mean(); den += c
        return num / den
    return dict(t_S=tS, t_D=tD, W_to_S=rate_W(PS, tS), W_to_D=rate_W(eff, tD), falseS_matchedD=rate_D(tS),
                recall_S=rec("S", callS), recall_D=rec("D", callD & ~callS))

tS1 = smallest(lambda t: rate_W(PS, t) <= WLIM + 1e-12, PS[isW])
tD1, eff1 = fit_tD(tS1)
res["OP1"] = summary(tS1, tD1, eff1)
allv = np.r_[PS[isW], PS[matchedD], 1.0]
tS2 = smallest(lambda t: rate_W(PS, t) <= WLIM + 1e-12 and rate_D(t) <= DLIM + 1e-12, allv)
res["OP2_D_side_alone"] = smallest(lambda t: rate_D(t) <= DLIM + 1e-12, PS[matchedD])
tD2, eff2 = fit_tD(tS2)
res["OP2"] = summary(tS2, tD2, eff2)
res["n_W_realistic"], res["n_D_matched"] = int(isW.sum()), int(matchedD.sum())

# ---- percentile block bootstrap (500 reps, LD block) for realistic AUPRC S-vs-D (sanity only, not BCa) ----
rng = np.random.default_rng(1)
blocks = df["block_id"].to_numpy(); ub = np.unique(blocks); bidx = {b: np.nonzero(blocks == b)[0] for b in ub}
if "--boot" in sys.argv:
    full = df.copy(); vals = []
    for _ in range(500):
        take = np.concatenate([bidx[b] for b in rng.choice(ub, len(ub))])
        df = full.iloc[take].reset_index(drop=True)
        cls, cell = df["oracle_class"].to_numpy(), df["cell_id"].to_numpy()
        w = sd_w("realistic"); m = w > 0
        vals.append(average_precision_score(cls[m] == "S", df["s"].to_numpy()[m], sample_weight=w[m]))
    res["realistic_AUPRC_SvsD_pctCI95"] = list(np.percentile(vals, [2.5, 97.5]))
print(json.dumps(res, indent=1, default=float))
