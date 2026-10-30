#!/usr/bin/env python
"""p5.verify_infra check (c): recompute s, P_S, P_D of the screen_v2 baseline pilot predictions from their stored
`extra` fields with the DESIGN v4 §4.3 formulas (+ DECISIONS P3-2: mvSuSiE hard lfsr < 0.05 rule is PRIMARY;
P3-3: SharePro per-effect signal probability). Own code; nothing imported from src/eval/adapters.py.

Every row of every method's pilot200 run is recomputed (the brief asks for >= 5; all 200 are cheap); 5 random
rows per method are printed in full. mvSuSiE: the inclusion-exclusion P_D is ALSO checked by brute-force
enumeration of per-effect activity states. Output JSON on stdout.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
RUNS = json.loads((ROOT / "src/baselines/benchmarks/pilot200_screen_v2_runs.json").read_text())["runs"]


def ratio(a, b):
    return a / (a + b) if (a + b) > 0 and math.isfinite(a + b) else 0.5


def one_minus_prod(x):
    x = np.clip(np.asarray(x, float), 0, 1)
    return float(1 - np.prod(1 - x)) if len(x) else 0.0


def coloc_abf(e):
    pp = e["PP"]
    return ratio(pp[4], pp[3]), pp[4], pp[3]


def coloc_susie(e):
    if e.get("fallback"):
        pp = e["fallback_PP"]
        return ratio(pp[4], pp[3]), pp[4], pp[3]
    P = np.array([[np.nan if v is None else v for v in r] for r in e["pairs"]["PP"]], float)
    P = P[np.all(np.isfinite(P), axis=1)]
    PS = P[:, 4].max()
    return max(ratio(r[4], r[3]) for r in P), PS, (1 - PS) * P[:, 3].max()


def sharepro(e):
    PS = one_minus_prod(e["share"])
    sig = [one_minus_prod(e["P_eff"][t]) for t in range(2)]           # per-effect signal prob. (P3-3), 6-dp rounded
    return PS, PS, (1 - PS) * sig[0] * sig[1], {"P_sig_stored": e["P_sig"], "P_sig_from_P_eff": sig}


def colocboost(e):
    PS = one_minus_prod(e["cos_npc"] or [])
    inc = set()
    co = e.get("cos_outcomes")
    for o in (co if isinstance(co, list) else ([co] if co else [])):
        for tok in str(o).replace(" ", "").split(";"):
            if tok.startswith("Y"):
                inc.add(int(tok[1:]))
    uc = set(int(x) for x in (e.get("ucos_outcomes") or []))
    sig = [1.0 if (t in inc or t in uc) else 0.0 for t in (1, 2)]
    return PS, PS, (1 - PS) * sig[0] * sig[1], {"P_sig_stored": e["P_sig"], "P_sig_mine": sig}


def mv_events(q1, q2):
    """Inclusion-exclusion (DESIGN §5 with activities independent within an effect)."""
    q1, q2 = np.asarray(q1, float), np.asarray(q2, float)
    if len(q1) == 0:
        return 0.5, 0.0, 0.0
    w = q1 * q2
    p1, p2 = q1 * (1 - q2), q2 * (1 - q1)
    PS = 1 - np.prod(1 - w)
    PD = np.prod(1 - w) - np.prod(1 - w - p1) - np.prod(1 - w - p2) + np.prod(1 - w - p1 - p2)
    PD = min(max(PD, 0.0), 1.0)
    return ratio(PS, PD), PS, PD


def mv_brute(q1, q2):
    """P(no effect active in both, >=1 active only in 1, >=1 active only in 2) by enumeration."""
    L = len(q1)
    PS = PD = 0.0
    for st in itertools.product(range(4), repeat=L):   # 0 none, 1 only-1, 2 only-2, 3 both
        p = 1.0
        for l, s in enumerate(st):
            a, b = q1[l], q2[l]
            p *= [(1 - a) * (1 - b), a * (1 - b), b * (1 - a), a * b][s]
        if 3 in st:
            PS += p
        elif 1 in st and 2 in st:
            PD += p
    return PS, PD


def mvsusie(e, rule):
    lf = np.array([[np.nan if v is None else v for v in r] for r in e["single_effect_lfsr"]], float).reshape(-1, 2)
    q = (lf < 0.05).astype(float) if rule == "hard" else 1 - lf
    return mv_events(q[:, 0], q[:, 1]), q


def main():
    out, n_bad = {}, 0
    g = np.random.default_rng(20261029)
    for m, info in RUNS.items():
        p = pd.read_parquet(ROOT / info["run_dir"] / "predictions.parquet")
        ok = p[p.status.isin(["ok", "nonconverged"])]
        diffs, notes, show = [], {}, []
        nfb = 0
        for r in ok.itertuples():
            e = json.loads(r.extra)
            if m == "coloc_abf":
                s, PS, PD = coloc_abf(e)
            elif m in ("coloc_susie", "coloc_susie_mm"):
                nfb += bool(e.get("fallback"))
                s, PS, PD = coloc_susie(e)
            elif m == "sharepro":
                s, PS, PD, x = sharepro(e)
                notes.setdefault("max|P_sig stored - from P_eff|", 0.0)
                notes["max|P_sig stored - from P_eff|"] = max(notes["max|P_sig stored - from P_eff|"], max(
                    abs(a - b) for a, b in zip(x["P_sig_stored"], x["P_sig_from_P_eff"])))
            elif m.startswith("colocboost"):
                s, PS, PD, x = colocboost(e)
                notes["P_sig mismatches"] = notes.get("P_sig mismatches", 0) + (x["P_sig_stored"] != x["P_sig_mine"])
            elif m == "mvsusie":
                (sh, PSh, PDh), q = mvsusie(e, "hard")
                (ss, PSs, PDs), qs = mvsusie(e, "soft")
                if len(qs) and len(qs) <= 7:
                    bPS, bPD = mv_brute(qs[:, 0], qs[:, 1])
                    notes["max|brute - incl.excl.| (soft)"] = max(notes.get("max|brute - incl.excl.| (soft)", 0.0),
                                                                    abs(bPS - PSs), abs(bPD - PDs))
                hard_stored = e["hard_lfsr005"]
                notes["max|hard recomputed - extra.hard_lfsr005|"] = max(
                    notes.get("max|hard recomputed - extra.hard_lfsr005|", 0.0),
                    abs(sh - hard_stored["s"]), abs(PSh - hard_stored["P_S"]), abs(PDh - hard_stored["P_D"]))
                d_soft = max(abs(ss - r.s), abs(PSs - r.P_S), abs(PDs - r.P_D))
                d_hard = max(abs(sh - r.s), abs(PSh - r.P_S), abs(PDh - r.P_D))
                notes["main columns == soft rule (max diff)"] = max(notes.get("main columns == soft rule (max diff)", 0), d_soft)
                notes["main columns vs hard rule (P3-2 primary): max diff"] = max(
                    notes.get("main columns vs hard rule (P3-2 primary): max diff", 0), d_hard)
                notes["n rows where hard != soft (>1e-3)"] = notes.get("n rows where hard != soft (>1e-3)", 0) + (d_hard > 1e-3)
                s, PS, PD = ss, PSs, PDs   # compare stored columns to what they claim to be (soft), report hard above
            elif m == "prop_coloc":
                s, PS, PD = e["p_full"], np.nan, np.nan
            elif m == "coloc_prop_test":
                s = 0.5 if not e.get("n_tests") else e["min_fdr"]
                PS = PD = np.nan
            elif m == "lead_snp":
                s = 1.0 if (e["lead1"] == e["lead2"] or e["r2"] >= 0.8) else 0.0
                PS = PD = np.nan
            d = max(abs(s - r.s), 0 if np.isnan(PS) and np.isnan(r.P_S) else abs(PS - r.P_S),
                    0 if np.isnan(PD) and np.isnan(r.P_D) else abs(PD - r.P_D))
            diffs.append(d)
        pick = g.choice(len(ok), size=min(5, len(ok)), replace=False)
        for i in pick:
            r = ok.iloc[i]
            show.append({"locus_id": r.locus_id, "s": r.s, "P_S": r.P_S, "P_D": r.P_D, "status": r.status})
        mx = float(np.nanmax(diffs)) if diffs else float("nan")
        bad = not (mx < 1e-9)
        if m == "sharepro":
            bad = not (mx < 1e-5)     # P_eff stored rounded to 6 dp
        n_bad += bad
        out[m] = {"n_rows": len(p), "n_recomputed": len(ok), "n_failed_rows": int(len(p) - len(ok)),
                  "max_abs_diff_stored_vs_recomputed": mx, "ok": not bad, "n_fallback": nfb, "notes": notes,
                  "five_random_rows": show,
                  "nan_in_s_for_ok_rows": int(ok.s.isna().sum()),
                  "nan_in_PS_PD_for_ok_rows": int((ok.P_S.isna() | ok.P_D.isna()).sum())}
    out["n_methods_with_mismatch"] = n_bad
    print(json.dumps(out, indent=1, default=float))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
