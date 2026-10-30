"""Kill-test B1 and the DESIGN §6.2 kill rules (a)-(d) for Family B on screen_v2 (task p5.killtest_B1).

    PYTHONPATH=src python scripts/famB_eval/killtest_B1.py --b results/famB/screen_v2_v1/M50/predictions.parquet \
        --baselines <baselines_screen_v2@1 path> --sets data/processed/sets/screen_v2/*.h5 --out <json>

All metrics come from src/eval/metrics.py (the single implementation): build_eval_frame (failure scoring),
matched_w_thresholds (C3 operating points), three_way_calls, auprc_sd, cluster_bootstrap / contrast /
bootstrap_p_value. Bootstrap unit "locus" (screening kill rules, eval README item 30), 2,000 replicates,
percentile intervals and bootstrap SE (bca=False: the leave-one-locus-out jackknife BCa needs would cost
12,000 extra evaluations per statistic; the rules use the paired SE).

Pre-specified definitions (fixed before any baseline result on screen_v2 was seen; reports/p5.killtest_B1.md):
  * Thresholds: each method's own matched-W C3 thresholds (t_S, then t_D; W rates <= 2.5%) fitted on its
    screen_v2 predictions in the "realistic" setting (run_benchmark's convention, screen_v2 cells L2 and L4);
    fixed 0.8 thresholds as sensitivity. Rule (b) only at the fixed 0.8 thresholds (at matched-W thresholds every
    method has <= 2.5% + 2.5% confident calls on W by construction, so (b) would be vacuous).
  * false-S rate = share of loci in a stratum called S (P_S >= t_S), cells of the stratum weighted equally.
  * B1 usefulness strata: Dmid = class D with r² in [0.3, 0.8); Mx = locus_type M with oracle class != S.
    Primary: the union Dmid ∪ Mx over the L0 and L2 cells; PASS iff false-S(coloc-SuSiE standard) -
    false-S(B) > 1 paired SE. Per stratum x level and vs coloc-SuSiE mismatch-aware: secondary.
  * Rule (a): AUPRC(S vs D) "ideal" (L0, overlap 0). best tier-1 baseline = highest ideal AUPRC; KILL iff
    AUPRC(B) - AUPRC(best) < -1 paired SE.
  * Rule (b): confident calls (S or D) on all W loci at 0.8; KILL iff rate(B) > 2 rate(coloc.abf) and
    rate(B) - rate(coloc.abf) > 0.02 and the one-sided bootstrap p of delta = rate(B) - max(2 r_abf, r_abf + .02)
    > 0 is < 0.05.
  * Rule (c): false-S on D loci, strata L4 (both L4 cells) and rho = 0.5 (the three P3 cells); comparator =
    the tier-1 baseline with the LOWEST false-S rate in the stratum (the strictest reading of "best baseline");
    KILL iff false-S(B) > 2 x comparator and > comparator + 0.02 (point estimates; one-sided p reported).
    Also reported vs the best-AUPRC baseline of that stratum.
  * Rule (d): mean harness runtime_s per locus over screen_v2, B vs coloc-SuSiE (standard); KILL iff ratio > 10
    and B has no accuracy gain (ideal and realistic AUPRC not higher).
  * Subsampled baselines (slow methods): every paired comparison uses the loci both methods were run on.
  * coloc.abf enters as ONE baseline: the variant with p12 = 5e-6 and the W chosen by the baselines agent
    (--abf-name; W = 0.15 if none is recorded). The other W / p12 variants are not separate baselines.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.seeds import rng  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval.io import truth_path_for  # noqa: E402
from eval.truth import read_truth  # noqa: E402

B = "exactcoloc_gaussian_M50"
CS, CS_MM, ABF = "coloc_susie", "coloc_susie_mm", "coloc_abf"
NOT_BASELINES = ("strength_only",)
N_BOOT = 2000


def load_preds_dir(path: Path) -> dict[str, pd.DataFrame]:
    """Predictions of several methods from a benchmark dir (manifest.json 'runs'), a runs index, a parquet with a
    method column, or a directory tree of harness run dirs."""
    path = Path(path)
    out: dict[str, list] = {}
    if path.is_dir() and (path / "index.json").exists() and (path / "predictions.parquet").exists():
        # collected baselines artifact (baselines_screen_v2@1): one parquet + index.json with its sha256
        from eval.harness import sha256_file
        ix = json.loads((path / "index.json").read_text())
        if ix.get("predictions_sha256") and sha256_file(path / "predictions.parquet") != ix["predictions_sha256"]:
            raise ValueError(f"{path}: predictions.parquet sha256 differs from index.json")
        path = path / "predictions.parquet"
    man = path / "manifest.json"
    if man.exists() and "runs" in json.loads(man.read_text()):
        for m, by_set in json.loads(man.read_text())["runs"].items():
            for d in by_set.values():
                out.setdefault(m, []).append(pd.read_parquet(Path(d) / "predictions.parquet"))
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
        for m, g in df.groupby("method"):
            out.setdefault(str(m), []).append(g)
    else:
        for f in sorted(path.rglob("predictions.parquet")):
            df = pd.read_parquet(f)
            for m, g in df.groupby("method"):
                out.setdefault(str(m), []).append(g)
    res = {}
    for m, parts in out.items():
        df = pd.concat(parts, ignore_index=True)
        if df["locus_id"].duplicated().any():
            raise ValueError(f"{m}: duplicated loci across inputs")
        res[m] = df
    return res


def cell_equal_rate(call_is_s, mask, cell):
    m = np.asarray(mask, bool)
    if not m.any():
        return float("nan")
    cells = pd.Series(cell[m])
    w = 1.0 / cells.map(cells.value_counts()).to_numpy(float)
    return float(np.sum(w * call_is_s[m]) / np.sum(w))


def boot_vector(stat, units, key):
    return M.cluster_bootstrap(stat, units, n_boot=N_BOOT, rng=rng(20261029, "killtest_B1", key), n_jobs=NJ,
                               bca=False, return_replicates=True)


def contrast_summary(res, ia, ib):
    c = M.contrast(res, ia, ib)
    return {"diff": c["estimate"], "se": c["se"], "ci95": [c["ci_low"], c["ci_high"]],
            "p_greater": M.bootstrap_p_value(c, 0.0, alternative="greater"),
            "p_less": M.bootstrap_p_value(c, 0.0, alternative="less")}


def main(argv=None):
    global NJ
    global N_BOOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", required=True, help="B predictions parquet (famB_screen_v2@1)")
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--abf-name", default="coloc_abf_W0.15_p12_5e-06",
                    help="the coloc.abf variant used as the tier-1 baseline (renamed coloc_abf)")
    a = ap.parse_args(argv)
    NJ, N_BOOT = a.n_jobs, a.n_boot

    truth = pd.concat([read_truth(truth_path_for(s)) for s in a.sets], ignore_index=True)
    ltype = dict(zip(truth["locus_id"].astype(str), truth["locus_type"].astype(str)))
    preds = load_preds_dir(Path(a.baselines))
    if a.abf_name in preds:
        preds[ABF] = preds.pop(a.abf_name).assign(method=ABF)
    for m in [m for m in preds if m.startswith("coloc_abf_")]:
        preds.pop(m)  # other W / p12 variants: not separate baselines
    bdf = pd.read_parquet(a.b)
    preds[B] = bdf
    methods = sorted(preds)
    baselines = [m for m in methods if m != B and m not in NOT_BASELINES]
    ids = {m: set(preds[m]["locus_id"].astype(str)) for m in methods}
    out = {"abf_variant": a.abf_name, "methods": methods, "baselines": baselines, "n_loci": {m: len(ids[m]) for m in methods},
           "n_truth": len(truth), "B_status": bdf["status"].value_counts().to_dict()}
    assert ids[B] == set(truth["locus_id"].astype(str)), "B must cover every screen_v2 locus"

    def frame(ms, mode):
        common = set.intersection(*[ids[m] for m in ms])
        tr = truth[truth["locus_id"].astype(str).isin(common)]
        return M.build_eval_frame(tr, {m: preds[m][preds[m]["locus_id"].astype(str).isin(common)] for m in ms},
                                  mode)

    # ---- operating points (each method on its own loci; least-favourable failure scoring)
    ops = {}
    for m in methods:
        fr = frame([m], "least_favourable")
        mw = M.matched_w_thresholds(fr, m, "realistic")
        ops[m] = {"t_S": float(mw.get("t_S", np.inf)), "t_D": float(mw.get("t_D", np.inf)),
                  "rate_S_on_W": mw.get("rate_S_on_W"), "rate_D_on_W": mw.get("rate_D_on_W"),
                  "has_P_S": bool(not np.isnan(fr.scores[m]["P_S"]).all()),
                  "failures": int(fr.failed[m].sum()), "n": len(fr)}
    out["operating_points_matchedW_realistic"] = ops
    thr = {"matchedW": {m: (ops[m]["t_S"], ops[m]["t_D"]) for m in methods},
           "fixed0.8": {m: (0.8, 0.8) for m in methods}}

    for mode in ("least_favourable", "excluded"):
        R = out.setdefault(mode, {})

        # ---- B1 usefulness: false-S on Dmid and Mx at L0 / L2, vs coloc-SuSiE (standard; mismatch-aware 2nd)
        b1 = R.setdefault("B1_usefulness", {})
        for comp in (CS, CS_MM):
            if comp not in methods:
                b1[comp] = "not available"
                continue
            fr = frame([B, comp], mode)
            lt = np.array([ltype[i] for i in fr.locus_id])
            dmid = (fr.cls == "D") & (fr.r2 >= 0.3) & (fr.r2 < 0.8)
            mx = (lt == "M") & (fr.cls != "S")
            lev = {L: fr.level == L for L in ("L0", "L2")}
            strata = {"primary_union_L0_L2": (dmid | mx) & (lev["L0"] | lev["L2"])}
            for sn, sm in (("Dmid", dmid), ("Mx", mx)):
                for L in ("L0", "L2"):
                    strata[f"{sn}_{L}"] = sm & lev[L]
            for tn, tt in thr.items():
                callS = {m: (M.three_way_calls(fr, m, *tt[m]) == "S").astype(float) for m in (B, comp)}
                names = list(strata)

                def stat(idx, names=names, callS=callS, fr=fr):
                    v = []
                    for sn in names:
                        msk = strata[sn][idx]
                        v += [cell_equal_rate(callS[B][idx], msk, fr.cell[idx]),
                              cell_equal_rate(callS[comp][idx], msk, fr.cell[idx])]
                    return np.array(v)

                res = boot_vector(stat, M.resampling_units(fr, "locus"), ("B1", mode, comp, tn))
                rows = {}
                for k, sn in enumerate(names):
                    c = contrast_summary(res, 2 * k + 1, 2 * k)  # comparator - B (> 0: B better)
                    rows[sn] = {"n": int(strata[sn].sum()), "false_S_B": float(res["estimate"][2 * k]),
                                f"false_S_{comp}": float(res["estimate"][2 * k + 1]),
                                "comp_minus_B": c, "pass_gt_1SE": bool(c["diff"] > c["se"])}
                b1.setdefault(comp, {})[tn] = rows

        # ---- rule (a): ideal AUPRC vs the best tier-1 baseline
        ra = {}
        for m in methods:
            fr = frame([m], mode)
            ra[m] = M.auprc_sd(fr, m, "ideal")
        best = max(baselines, key=lambda m: -np.inf if np.isnan(ra[m]) else ra[m])
        fr = frame([B, best], mode)
        fr = fr.mask((fr.level == "L0") & (np.nan_to_num(fr.overlap, nan=1) == 0))
        res = boot_vector(lambda idx, fr=fr: np.array([M.auprc_sd(fr.take(idx), B, "ideal"),
                                                       M.auprc_sd(fr.take(idx), best, "ideal")]),
                          M.resampling_units(fr, "locus"), ("a", mode))
        c = contrast_summary(res, 0, 1)
        R["rule_a"] = {"auprc_ideal_own_loci": ra, "best_baseline": best, "paired": c,
                       "kill": bool(c["diff"] < -c["se"])}

        # ---- rule (b): confident calls on W at 0.8 vs coloc.abf
        if ABF in methods:
            fr = frame([B, ABF], mode)
            isW = fr.cls == "W"
            conf = {m: np.isin(M.three_way_calls(fr, m, 0.8, 0.8), ["S", "D"]).astype(float) for m in (B, ABF)}

            def stat_b(idx, fr=fr, conf=conf, isW=isW):
                rb = cell_equal_rate(conf[B][idx], isW[idx], fr.cell[idx])
                ra_ = cell_equal_rate(conf[ABF][idx], isW[idx], fr.cell[idx])
                return np.array([rb, ra_, rb - max(2 * ra_, ra_ + 0.02)])

            res = boot_vector(stat_b, M.resampling_units(fr, "locus"), ("b", mode))
            est = res["estimate"]
            p = M.bootstrap_p_value(res, 0.0, j=2, alternative="greater")
            R["rule_b"] = {"conf_on_W_B": float(est[0]), "conf_on_W_coloc_abf": float(est[1]), "delta": float(est[2]),
                           "p_one_sided": p, "kill": bool(est[2] > 0 and p < 0.05),
                           "per_class_calls_B": pd.Series(M.three_way_calls(fr, B, 0.8, 0.8)[isW]).value_counts().to_dict()}

        # ---- rule (c): false-S on D at L4 and at rho = 0.5
        rc = {}
        for sname in ("L4", "rho0.5"):
            rc[sname] = {}
            for tn, tt in thr.items():
                rates, auprc = {}, {}
                for m in methods:
                    if not ops[m]["has_P_S"]:
                        continue
                    fr = frame([m], mode)
                    sm = (fr.level == "L4") if sname == "L4" else (fr.pairing == "P3") & (np.abs(fr.overlap - 0.5) < 1e-9)
                    cS = (M.three_way_calls(fr, m, *tt[m]) == "S").astype(float)
                    rates[m] = cell_equal_rate(cS, sm & (fr.cls == "D"), fr.cell)
                    sub = fr.mask(sm)
                    cw = {c_: 1.0 for c_ in np.unique(sub.cell)}
                    auprc[m] = float(np.mean([M.auprc_sd(sub, m, f"cell:{c_}") for c_ in cw]))
                bl = [m for m in rates if m in baselines]
                comp = min(bl, key=lambda m: rates[m])
                comp_auprc = max(bl, key=lambda m: auprc[m])
                entry = {"false_S_on_D": rates, "mean_cell_auprc_sd": auprc, "comparator_lowest_falseS": comp,
                         "comparator_best_auprc": comp_auprc}
                for role, cm in (("lowest_falseS", comp), ("best_auprc", comp_auprc)):
                    fr = frame([B, cm], mode)
                    sm = (fr.level == "L4") if sname == "L4" else (fr.pairing == "P3") & (np.abs(fr.overlap - 0.5) < 1e-9)
                    msk = sm & (fr.cls == "D")
                    cS = {m: (M.three_way_calls(fr, m, *tt[m]) == "S").astype(float) for m in (B, cm)}

                    def stat_c(idx, fr=fr, cS=cS, msk=msk, cm=cm):
                        rb = cell_equal_rate(cS[B][idx], msk[idx], fr.cell[idx])
                        rcm = cell_equal_rate(cS[cm][idx], msk[idx], fr.cell[idx])
                        return np.array([rb, rcm, rb - max(2 * rcm, rcm + 0.02)])

                    res = boot_vector(stat_c, M.resampling_units(fr, "locus"), ("c", mode, sname, tn, role))
                    est = res["estimate"]
                    entry[role] = {"comparator": cm, "false_S_B": float(est[0]), "false_S_comp": float(est[1]),
                                   "delta": float(est[2]),
                                   "p_one_sided": M.bootstrap_p_value(res, 0.0, j=2, alternative="greater"),
                                   "kill": bool(est[0] > 2 * est[1] and est[0] > est[1] + 0.02)}
                rc[sname][tn] = entry
        R["rule_c"] = rc

    # ---- rule (d): runtime
    rt = {m: float(np.nanmean(pd.to_numeric(preds[m]["runtime_s"], errors="coerce"))) for m in methods}
    rd = {"mean_runtime_s": rt}
    if CS in methods:
        rd["ratio_B_over_coloc_susie"] = rt[B] / rt[CS]
        lf = out["least_favourable"]
        fr = frame([B, CS], "least_favourable")
        gain = {st: M.auprc_sd(fr, B, st) - M.auprc_sd(fr, CS, st) for st in ("ideal", "realistic")}
        rd["auprc_gain_vs_coloc_susie"] = gain
        rd["kill"] = bool(rd["ratio_B_over_coloc_susie"] > 10 and max(gain.values()) <= 0)
        del lf
    out["rule_d"] = rd
    Path(a.out).write_text(M.dumps(M._clean(out)))
    print(M.dumps(M._clean(out)))


if __name__ == "__main__":
    main()
