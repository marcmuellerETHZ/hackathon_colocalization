#!/usr/bin/env python
"""p7 verify, check 5 (+ C3 confusion totals of check 4): real-arm spot check from run predictions (no src/eval import).

t_R3b = smallest threshold whose confident-S rate on R3b is <= 5%, placed just above an R3b score (never splitting
ties); S recall on R2 at t_R3b; R4b call agreement in-sample vs reference LD at the frozen t_S/t_D (E gate).
Failures least favourable (R3b: P_S = 1; R2: P_S = 0; R4b: disagreement).

Usage: python reviews/p7_verify_testrun/realarms.py <project_root>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(sys.argv[1])
B = ROOT / "results/eval_test/test_frozen-v1_ab94c5a1"
TAG = "test_frozen-v1_ab94c5a1"
frozen = {m["name"]: m for m in yaml.safe_load((ROOT / "configs/test_methods_v1.yaml").read_text())["methods"]}
ra = json.loads((B / "real_arms.json").read_text())


def pred(m, s):
    p = pd.read_parquet(B / "runs" / f"{TAG}__{m}__{s}" / "predictions.parquet")
    p["fail"] = p.status.isin(["failed", "timeout"]) | p.P_S.isna()
    return p


def thr_at(scores, alpha):
    """smallest t with mean(scores >= t) <= alpha, t just above a score value."""
    v = np.sort(np.unique(scores))
    n = len(scores)
    for x in v:  # candidate t = nextafter(x, inf): rate = share strictly above x
        if (scores > x).sum() / n <= alpha:
            return float(np.nextafter(x, np.inf))
    return float(np.nextafter(v[-1], np.inf))


def calls(p, mc):
    tS, tD = mc["t_S"], mc["t_D"]
    eg = (p.E1.isna() | (p.E1 >= 0.95)) & (p.E2.isna() | (p.E2 >= 0.95))
    cS = p.P_S >= tS
    cD = (~cS) & (p.P_D >= tD) & eg
    return pd.Series(np.where(cS, "S", np.where(cD, "D", "I")), index=p.locus_id)


out = {}
for m in ["coloc_abf", "coloc_susie", "coloc_susie_mm", "sharepro", "strength_only"]:
    mc = frozen[m]["metric_config"]
    r3 = pred(m, "real_R3b")
    ps3 = np.where(r3.fail, 1.0, r3.P_S)
    t = thr_at(ps3, 0.05)
    r2 = pred(m, "real_R2")
    ps2 = np.where(r2.fail, 0.0, r2.P_S)
    a, b = pred(m, "real_R4b_insample"), pred(m, "real_R4b_reference")
    ca, cb = calls(a, mc), calls(b, mc)
    fa = a.set_index("locus_id").fail | b.set_index("locus_id").fail.reindex(a.locus_id).fillna(True).to_numpy()
    agree = ((ca == cb.reindex(ca.index)) & ~fa.reindex(ca.index)).mean()
    out[m] = dict(t_R3b_5pct=t, R3b_confS_rate_at_t=float((ps3 >= t).mean()),
                  R3b_confS_rate_at_frozen_tS=float((ps3 >= mc["t_S"]).mean()),
                  R2_recall_S_at_t=float((ps2 >= t).mean()),
                  bundle_R2_recall=ra["methods"][m]["recall_S_R2_at_5pct_R3b"]["est"],
                  R4b_agreement=float(agree), bundle_R4b=ra["methods"][m]["R4b_call_agreement"]["est"],
                  n=(len(r3), len(r2), len(a), len(b)))
    out[m]["diff_R2"] = out[m]["R2_recall_S_at_t"] - out[m]["bundle_R2_recall"]
    out[m]["diff_R4b"] = out[m]["R4b_agreement"] - out[m]["bundle_R4b"]
print(json.dumps(out, indent=1))
