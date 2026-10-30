#!/usr/bin/env python
"""p5.verify_infra check (a): recompute oracle classes of 30 random VALIDATION loci with independent code.

For each sampled locus (15 from screen_v2, 15 from valid_grid_v1; stratified over the classes present):
  * the target population (NFE mosaic, N_syn = 20,000) is materialised on the CPU for the locus window with
    sim.mosaic (data, not logic) and r²(a, b) in R_true is computed in float64 from its dosages with own code;
  * the oracle class is recomputed from the stored causal ids, own effects lambda and realised mu, with an
    implementation written from DESIGN v4 §1 text (thresholds typed in from DESIGN, not imported);
  * class, class-determining |mu1|/|mu2|, r² and d are compared with the truth file;
  * for factor-form traits drawn from the target population without background, mu is recomputed as
    (R_pop lambda) at the causal variants (DESIGN §1: "for factor-form traits it equals (R_pop λ_t)_j").
Writes $ORCH_OUT/oracle_check.json; exit 1 on any class mismatch.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(os.environ.get("ORCH_HOME", Path(__file__).resolve().parents[2]))
SETS = ROOT / "data" / "processed" / "sets"

# DESIGN v4 §1 (typed from the text)
ALPHA = {"gwas": 5e-8, "mol": 1e-5}
TAU = {"gwas": 6.29, "mol": 5.26}
WEAK = {"gwas": 4.61, "mol": 3.58}


def pw(mu, kind):
    z = norm.isf(ALPHA[kind] / 2)
    m = abs(mu)
    return norm.sf(z - m) + norm.cdf(-z - m)


def my_oracle(c1, lam1, mu1, c2, lam2, mu2, r2, k1, k2):
    T1 = {c: (abs(l), abs(m)) for c, l, m in zip(c1, lam1, mu1)}
    T2 = {c: (abs(l), abs(m)) for c, l, m in zip(c2, lam2, mu2)}
    P1 = {c for c, (l, m) in T1.items() if l >= TAU[k1] and m >= TAU[k1]}
    P2 = {c for c, (l, m) in T2.items() if l >= TAU[k2] and m >= TAU[k2]}
    N1 = {c for c, (l, m) in T1.items() if m >= WEAK[k1]}
    N2 = {c for c, (l, m) in T2.items() if m >= WEAK[k2]}
    shared = set(T1) & set(T2)
    sp = shared & P1 & P2
    if sp:  # S: shared powered variant with the largest min-power (ties: larger min |mu| as a reporting aid)
        j = max(sp, key=lambda c: (min(pw(T1[c][1], k1), pw(T2[c][1], k2)), -c))
        return dict(cls="S", mu1=T1[j][1], mu2=T2[j][1], r2=np.nan, d=np.nan, n_sp=len(sp))
    if P1 and P2:
        pairs = []
        for a in P1:
            for b in P2:
                rr = min(max(r2(a, b), 0.0), 1.0)
                d = min(T1[a][0], T2[b][0]) * math.sqrt(1 - rr)
                pairs.append((rr, d, a, b))
        rmax = max(p[0] for p in pairs)
        top = [p for p in pairs if p[0] >= rmax - 1e-6]                 # tolerant tie on r² (float32 upstream)
        rr, d, a, b = min(top, key=lambda p: (p[1], p[2], p[3]))
        if d < 2:
            return dict(cls="U", mu1=T1[a][1], mu2=T2[b][1], r2=rr, d=d, n_top=len(top))
        if not (shared & N1 & N2):
            return dict(cls="D", mu1=T1[a][1], mu2=T2[b][1], r2=rr, d=d, n_top=len(top))
    m1 = max((v[1] for v in T1.values()), default=0.0)
    m2 = max((v[1] for v in T2.values()), default=0.0)
    if not N1 or not N2:
        return dict(cls="W", mu1=m1, mu2=m2, r2=np.nan, d=np.nan)
    return dict(cls="G", mu1=m1, mu2=m2, r2=np.nan, d=np.nan)


def sample(truth: pd.DataFrame, n: int, g) -> pd.DataFrame:
    classes = sorted(truth["oracle_class"].unique())
    per = {c: n // len(classes) for c in classes}
    for c in classes[: n - sum(per.values())]:
        per[c] += 1
    out = []
    for c in classes:
        sub = truth[(truth["oracle_class"] == c) & (truth["control"].astype(str) == "")]
        out.append(sub.iloc[g.choice(len(sub), size=min(per[c], len(sub)), replace=False)])
    return pd.concat(out)


def main():
    sys.path.insert(0, str(ROOT / "src"))
    from common.seeds import rng
    from sim.mosaic import Mosaic

    g = rng(20261029, "p5.verify_infra", "oracle_check")
    cols = ["locus_id", "oracle_class", "mu1", "mu2", "r2", "d", "split", "mismatch_level", "control", "causal1",
            "causal2", "effect1", "effect2", "mu1_all", "mu2_all", "causal1_vid", "causal2_vid", "trait1_type",
            "trait2_type", "pop1", "pop2", "form1", "form2", "background", "window_chr", "window_start",
            "window_end", "N1", "N2", "set"]
    picks = []
    for name in ("screen_v2", "valid_grid_v1"):
        t = pd.read_parquet(SETS / f"{name}.truth.parquet", columns=cols)
        assert (t["split"] == "valid").all(), f"{name}: non-validation rows"
        picks.append(sample(t, 15, g))
    S = pd.concat(picks).reset_index(drop=True)
    pop = Mosaic.named("NFE")
    rows, n_bad = [], 0
    mu_checks = []
    for r in S.itertuples():
        assert r.mismatch_level not in ("L1swap", "L2swap")
        h, v = pop.haplotypes(r.window_chr, int(r.window_start), int(r.window_end), device="cpu")
        vid = {x: i for i, x in enumerate(v["variant_id"].to_numpy())}
        c1 = [vid[x] for x in r.causal1_vid]
        c2 = [vid[x] for x in r.causal2_vid]
        cols_ = sorted(set(c1) | set(c2))
        G = (h[0::2, cols_] + h[1::2, cols_]).double().numpy()
        Z = (G - G.mean(0)) / np.where(G.std(0) > 0, G.std(0), 1.0)
        C = Z.T @ Z / Z.shape[0]
        ix = {c: i for i, c in enumerate(cols_)}
        r2 = lambda a, b: float(C[ix[a], ix[b]] ** 2)
        mine = my_oracle(c1, r.effect1, r.mu1_all, c2, r.effect2, r.mu2_all, r2, r.trait1_type, r.trait2_type)
        ok_cls = mine["cls"] == r.oracle_class
        n_bad += not ok_cls
        row = {"locus_id": r.locus_id, "level": r.mismatch_level, "stored": r.oracle_class, "mine": mine["cls"],
               "class_ok": ok_cls, "n_causal": [len(c1), len(c2)],
               "dmu1": abs(mine["mu1"] - abs(r.mu1)), "dmu2": abs(mine["mu2"] - abs(r.mu2))}
        if mine["cls"] in ("D", "U") and ok_cls:
            row["dr2"] = abs(mine["r2"] - r.r2)
            row["dd_rel"] = abs(mine["d"] - r.d) / max(abs(r.d), 1e-9)
        rows.append(row)
        # mu = R_pop lambda for factor-form traits from the target population, no background
        for t, (form, popn, c, lam, mu) in enumerate(((r.form1, r.pop1, c1, r.effect1, r.mu1_all),
                                                       (r.form2, r.pop2, c2, r.effect2, r.mu2_all)), start=1):
            if form == "factor" and popn == "NFE" and not bool(r.background) and len(c):
                Rc = np.array([[C[ix[a], ix[b]] for b in c] for a in c])
                pred = Rc @ np.asarray(lam, float)
                mu_checks.append({"locus_id": r.locus_id, "trait": t,
                                  "max_rel_err": float(np.max(np.abs(pred - np.asarray(mu, float)) /
                                                              np.maximum(np.abs(mu), 1.0)))})
    out = {"n": len(rows), "n_class_mismatch": n_bad,
           "class_counts": pd.Series([x["stored"] for x in rows]).value_counts().to_dict(),
           "max_dmu": max(max(x["dmu1"], x["dmu2"]) for x in rows),
           "max_dr2": max((x.get("dr2", 0.0) for x in rows), default=0.0),
           "max_dd_rel": max((x.get("dd_rel", 0.0) for x in rows), default=0.0),
           "mu_factor_checks": len(mu_checks),
           "mu_factor_max_rel_err": max((m["max_rel_err"] for m in mu_checks), default=None),
           "rows": rows, "mu_rows": mu_checks}
    s = json.dumps(out, indent=1, default=float)
    if os.environ.get("ORCH_OUT"):
        Path(os.environ["ORCH_OUT"], "oracle_check.json").write_text(s)
    print(s)
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
