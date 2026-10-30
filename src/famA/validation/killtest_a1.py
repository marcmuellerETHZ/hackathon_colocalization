"""Kill test A1 and screening kill rules (a)-(d) for Family A (DESIGN v4 §5 A, §6.2). Metrics: src/eval/metrics.py only.

python -m famA.validation.killtest_a1 --a-dir results/famA/famA_screen_v2_v2 --a-method famA_joint_susie_rho_eta_ms_ptld \
    --baseline-dir results/baselines_screen_v2/runs --out results/famA/killtest_a1_v1 [--cells ...] [--n-boot 2000]

Readings (fixed before looking at A1; written to the report):
  * A1 / rule (a): cell L0|P1|r_p=0.0, AUPRC(S vs D) with the §8.6 power-matched weights (setting "ideal"),
    least-favourable failures; best = baseline with the highest point AUPRC; paired locus-level bootstrap
    (unit "locus", 2,000 replicates, seed derive(20261029, 'killtest_A1', ...)); SE = sd of the paired difference
    replicates. PASS iff AUPRC_A >= AUPRC_best - 1 SE. A1 compares with max(coloc-SuSiE, SharePro, mvSuSiE-RSS);
    rule (a) with the best tier-1 baseline that has an S-vs-D score.
  * (b) confident calls on W (fixed 0.8 thresholds, S precedence, E gate where reported) pooled over the screening
    cells, unweighted; vs coloc.abf (W = 0.15, p12 = 5e-6). FAIL iff rate_A > 2 rate_abf AND rate_A > rate_abf + 0.02
    AND one-sided p < 0.05 for H0: rate_A - max(2 rate_abf, rate_abf + 0.02) <= 0 (paired locus bootstrap).
  * (c) false-S (P_S >= 0.8) on D loci, unweighted, pooled over the L4 cells and over the rho = 0.5 cells
    separately; vs the best tier-1 baseline of rule (a)'s definition evaluated on the screening cells (highest
    AUPRC macro over cells). FAIL iff on either subset rate_A > 2 rate_best AND rate_A > rate_best + 0.02.
  * (d) median per-locus harness runtime (same cells) > 10x coloc-SuSiE's without an accuracy gain (A's
    AUPRC macro over cells not above coloc-SuSiE's).
Paired comparisons use the loci both methods were run on (slow baselines ran a 200-per-cell subsample outside A1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.seeds import rng as seeded_rng
from eval import metrics as M
from eval.truth import read_truth

TRUTH = "data/processed/sets/screen_v2.truth.parquet"
A1_CELL = "L0|P1|r_p=0.0"
A1_BASELINES = ("coloc_susie", "sharepro", "mvsusie")
TIER1 = ("coloc_abf_W0.15_p12_5e-06", "coloc_abf_W0.1_p12_5e-06", "coloc_abf_W0.2_p12_5e-06", "coloc_susie",
         "coloc_susie_mm", "sharepro", "colocboost", "mvsusie")
ABF = "coloc_abf_W0.15_p12_5e-06"
T = M.FIXED_THRESHOLD


def load_collected(path: Path) -> dict[str, pd.DataFrame]:
    """method -> predictions from a collected baseline artifact file (predictions.parquet with a method column)."""
    p = pd.read_parquet(path)
    out = {}
    for m, d in p.groupby("method"):
        if d["locus_id"].duplicated().any():
            raise SystemExit(f"{path}: duplicate locus ids for {m}")
        out[str(m)] = d.reset_index(drop=True)
    return out


def load_runs(root: Path) -> dict[str, pd.DataFrame]:
    """method -> predictions concatenated over every harness run under root (manifest method_name)."""
    out: dict[str, list] = {}
    for man in sorted(root.rglob("manifest.json")):
        pp = man.parent / "predictions.parquet"
        if not pp.exists():
            continue
        mj = json.loads(man.read_text())
        if "method_name" not in mj:
            continue
        p = pd.read_parquet(pp)
        p["method"] = mj["method_name"]
        p["_run"] = mj["run_id"]
        out.setdefault(mj["method_name"], []).append(p)
    res = {}
    for m, parts in out.items():
        p = pd.concat(parts, ignore_index=True)
        # several runs on the same loci (re-runs): keep the latest run
        p = p.sort_values("_run").drop_duplicates("locus_id", keep="last")
        res[m] = p.drop(columns="_run")
    return res


def frame_for(truth, preds: dict, methods, mode="least_favourable"):
    common = set(truth["locus_id"])
    for m in methods:
        common &= set(preds[m]["locus_id"])
    t = truth[truth["locus_id"].isin(common)]
    return M.build_eval_frame(t, {m: preds[m][preds[m]["locus_id"].isin(common)] for m in methods}, mode)


def confident_W(fr, m):
    c = M.three_way_calls(fr, m, T, T)
    w = fr.cls == "W"
    return float(np.mean(np.isin(c[w], ["S", "D"]))) if w.any() else float("nan")


def false_S(fr, m):
    d = fr.cls == "D"
    return float(np.mean(np.nan_to_num(fr.scores[m]["P_S"][d], nan=-np.inf) >= T)) if d.any() else float("nan")


def boot(fr, stat, key, n_boot, n_jobs):
    g = seeded_rng(20261029, "killtest_A1", *key)
    return M.cluster_bootstrap(stat, M.resampling_units(fr, "locus"), n_boot=n_boot, rng=g, n_jobs=n_jobs,
                               return_replicates=True)


def auprc_cell(fr, m):
    st = "ideal" if set(fr.level) == {"L0"} and np.all(fr.overlap == 0) else "all"
    return M.auprc_sd(fr, m, st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", required=True, action="append", help="famA collected artifact dir(s)")
    ap.add_argument("--a-method", required=True)
    ap.add_argument("--extra-a-methods", default="", help="comma-separated further famA methods (reported only)")
    ap.add_argument("--baseline-dir", default=None, help="harness run dirs (dry runs only)")
    ap.add_argument("--baseline-pred", action="append", default=[],
                    help="registered collected baseline predictions.parquet (method column); repeatable, later "
                         "files add methods/loci")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", default="", help="restrict to these cells (comma-separated; default all six)")
    ap.add_argument("--skip-a1", action="store_true", help="do not evaluate the A1 cell (dry run on other cells)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    truth = read_truth(TRUTH)
    if a.cells:
        truth = truth[truth["cell_id"].isin(a.cells.split(","))]
    if a.skip_a1:
        truth = truth[truth["cell_id"] != A1_CELL]
    preds = load_runs(Path(a.baseline_dir)) if a.baseline_dir else {}
    for bp in a.baseline_pred:
        for m, p in load_collected(Path(bp)).items():
            preds[m] = (pd.concat([preds[m], p]).drop_duplicates("locus_id", keep="last") if m in preds else p)
    res_inputs = {"baseline_pred": a.baseline_pred, "baseline_dir": a.baseline_dir, "a_dir": a.a_dir}
    A = a.a_method
    for d in a.a_dir:
        for pp in sorted((Path(d) / "predictions").glob("*.parquet")):
            m = pp.stem
            if m == A or m in a.extra_a_methods.split(","):
                p = pd.read_parquet(pp)
                preds[m] = pd.concat([preds[m], p]).drop_duplicates("locus_id", keep="last") if m in preds else p
    preds = {m: p[p["locus_id"].isin(set(truth["locus_id"]))] for m, p in preds.items()}
    preds = {m: p for m, p in preds.items() if len(p)}
    tier1 = [m for m in TIER1 if m in preds]
    res = {"a_method": A, "baselines_present": sorted(preds), "tier1_present": tier1,
           "n_rows": {m: int(len(p)) for m, p in preds.items()}, "readings": __doc__, "inputs": res_inputs}
    cells = sorted(truth["cell_id"].unique())

    # ---------------- per-cell table (point estimates; each method on its own run loci, LF failures)
    rows = []
    for c in cells:
        tc = truth[truth["cell_id"] == c]
        for m in [A] + [x for x in a.extra_a_methods.split(",") if x in preds] + tier1:
            if m not in preds:
                continue
            fr = frame_for(tc, preds, [m])
            pm = preds[m][preds[m]["locus_id"].isin(set(tc["locus_id"]))]
            rows.append({"cell": c, "method": m, "n": len(fr), "auprc_sd": auprc_cell(fr, m),
                         "auroc_sd": M.auroc_sd(fr, m, "ideal" if c.startswith("L0") and c.endswith("=0.0") else "all"),
                         "confident_W": confident_W(fr, m), "false_S_D": false_S(fr, m),
                         "n_failed": int(fr.failed[m].sum()), "runtime_median_s": float(pm["runtime_s"].median()),
                         "runtime_mean_s": float(pm["runtime_s"].mean())})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "per_cell.tsv", sep="\t", index=False)
    res["per_cell"] = rows

    # ---------------- A1 and rule (a)
    if A1_CELL in cells:
        tc = truth[truth["cell_id"] == A1_CELL]
        for tag, pool in (("A1", [m for m in A1_BASELINES if m in preds]), ("rule_a", tier1)):
            if not pool:
                res[tag] = {"error": "no baselines present"}
                continue
            fr = frame_for(tc, preds, [A] + pool)
            pts = {m: auprc_cell(fr, m) for m in [A] + pool}
            best = max(pool, key=lambda m: pts[m])
            b = boot(fr, lambda idx, best=best: auprc_cell(fr.take(idx), A) - auprc_cell(fr.take(idx), best),
                     (tag, A, best), a.n_boot, a.n_jobs)
            se = float(np.nanstd(np.asarray(b["replicates"]).ravel(), ddof=1))
            diff = pts[A] - pts[best]
            # excluded-mode sensitivity
            fx = frame_for(tc, preds, [A] + pool, mode="excluded")
            res[tag] = {"cell": A1_CELL, "n_loci": len(fr), "auprc": pts, "best_baseline": best, "diff": diff,
                        "paired_se": se, "ci95": [b["ci_low"], b["ci_high"]],
                        "verdict": "PASS" if diff >= -se else "FAIL",
                        "excluded_mode": {"n_loci": len(fx), "auprc": {m: auprc_cell(fx, m) for m in [A] + pool}},
                        "failures": {m: int(fr.failed[m].sum()) for m in [A] + pool}}
        # runtime vs coloc-SuSiE on the A1 cell
        if "coloc_susie" in preds:
            ids = set(tc["locus_id"])
            ra = preds[A][preds[A]["locus_id"].isin(ids)]["runtime_s"]
            rc = preds["coloc_susie"][preds["coloc_susie"]["locus_id"].isin(ids)]["runtime_s"]
            res.setdefault("A1", {})["runtime"] = {
                "A_median_s": float(ra.median()), "A_mean_s": float(ra.mean()),
                "coloc_susie_median_s": float(rc.median()), "coloc_susie_mean_s": float(rc.mean()),
                "verdict": "PASS" if ra.median() <= rc.median() else "FAIL"}

    # ---------------- rule (b): confident calls on W vs coloc.abf, pooled over cells
    if ABF in preds:
        fr = frame_for(truth, preds, [A, ABF])
        ra, rb = confident_W(fr, A), confident_W(fr, ABF)

        def stat_b(idx):
            f = fr.take(idx)
            x, y = confident_W(f, A), confident_W(f, ABF)
            return np.array([x - max(2 * y, y + 0.02), x - y])
        bb = boot(fr, stat_b, ("rule_b", A, ABF), a.n_boot, a.n_jobs)
        p_margin = M.bootstrap_p_value(bb, 0.0, j=0, alternative="greater")
        p_diff = M.bootstrap_p_value(bb, 0.0, j=1, alternative="greater")
        fail = ra > 2 * rb and ra > rb + 0.02 and p_margin < 0.05
        res["rule_b"] = {"cells": sorted(set(fr.cell)), "n_W": int((fr.cls == "W").sum()), "rate_A": ra, "rate_abf": rb, "p_margin": p_margin,
                         "p_diff_gt0": p_diff, "per_cell": {c: {"A": confident_W(fr.mask(fr.cell == c), A),
                                                                 "abf": confident_W(fr.mask(fr.cell == c), ABF)}
                                                             for c in sorted(set(fr.cell))},
                         "verdict": "FAIL" if fail else "PASS"}

    # ---------------- best tier-1 baseline across screening cells (macro AUPRC over cells, own loci)
    macro = tab[tab["method"].isin(tier1)].groupby("method")["auprc_sd"].mean()
    best_t1 = macro.idxmax() if len(macro) else None
    a_macro = float(tab[tab["method"] == A]["auprc_sd"].mean())
    res["macro_auprc"] = {**{m: float(v) for m, v in macro.items()}, A: a_macro, "best_tier1": best_t1}

    # ---------------- rule (c): false-S at L4 and at rho = 0.5
    if best_t1 is not None:
        subsets = {"L4": [c for c in cells if c.startswith("L4")], "rho0.5": [c for c in cells if c.endswith("rho=0.5")]}
        rc = {}
        fail_c = False
        for name, cs in subsets.items():
            if not cs:
                continue
            tc = truth[truth["cell_id"].isin(cs)]
            per = {}
            for m in tier1:
                fr = frame_for(tc, preds, [A, m])
                per[m] = {"n_D": int((fr.cls == "D").sum()), "A": false_S(fr, A), "base": false_S(fr, m)}
            x = per[best_t1]
            f = x["A"] > 2 * x["base"] and x["A"] > x["base"] + 0.02
            fail_c |= f
            rc[name] = {"cells": cs, "best_tier1": best_t1, "A": x["A"], "best": x["base"], "n_D": x["n_D"],
                        "fails": bool(f), "all_tier1": per}
        evaluated = [k for k in ("L4", "rho0.5") if k in rc]
        rc["subsets_evaluated"] = evaluated
        rc["verdict"] = ("not evaluated (no L4 / rho=0.5 cells in this run)" if not evaluated
                         else ("FAIL" if fail_c else "PASS"))
        res["rule_c"] = rc

    # ---------------- rule (d): runtime vs coloc-SuSiE across the cells
    if "coloc_susie" in preds:
        ids = set(truth["locus_id"])
        ra = preds[A][preds[A]["locus_id"].isin(ids)]["runtime_s"]
        rcs = preds["coloc_susie"][preds["coloc_susie"]["locus_id"].isin(ids)]["runtime_s"]
        cs_macro = float(tab[tab["method"] == "coloc_susie"]["auprc_sd"].mean())
        ratio = float(ra.median() / rcs.median())
        res["rule_d"] = {"cells": cells, "A_median_s": float(ra.median()), "coloc_susie_median_s": float(rcs.median()),
                         "ratio": ratio, "A_macro_auprc": a_macro, "coloc_susie_macro_auprc": cs_macro,
                         "verdict": "FAIL" if (ratio > 10 and not a_macro > cs_macro) else "PASS"}
    (out / "killtest.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_cell", "readings")}, indent=1, default=float))
    print(tab.to_string())


if __name__ == "__main__":
    main()
