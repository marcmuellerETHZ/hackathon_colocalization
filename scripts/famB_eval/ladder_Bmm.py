"""Phase-2 ladder for B-mm on screen_v2 (DESIGN_P2 §3, §4 as frozen at cb6bdd0 + PI binding notes of 10-28).

    PYTHONPATH=src python scripts/famB_eval/ladder_Bmm.py --bmm <B-mm predictions.parquet> \
        --b1 results/famB/screen_v2/M50/predictions.parquet --baselines results/baselines_screen_v2/collected_v1 \
        --sets data/processed/sets/screen_v2/*.h5 --out results/famB/p9_ladder/ladder_Bmm.json

All metrics come from src/eval/metrics.py (build_eval_frame, auprc_sd, matched_w_thresholds, three_way_calls,
cluster_bootstrap, contrast, resampling_units). Helpers (prediction loading, cell-equal rates) are imported from
the phase-1 kill-test script (scripts/famB_eval/killtest_B1.py) so the phase-1 definitions are reused verbatim.

Order and definitions:
 0. NEW-METHOD GATE (computed first): threshold-free realistic AUPRC(S vs D) on screen_v2 (= L2+L4 cells, §8.6
    weights), B-mm minus phase-1 B (exactcoloc_gaussian_M50, famB_screen_v2@1); PASS iff diff > 1 paired SE,
    SE = SD of 2,000 paired cluster-bootstrap replicates over LD blocks (failures least-favourable).
    Locus-bootstrap SE reported alongside. If it fails, B-mm "is B and stays killed"; the rest is descriptive.
 1. Operating points, fitted for EVERY method on screen_v2 L2+L4 ("realistic", §8.6 weights), least-favourable:
    OP1 = matched-W C3 (metrics.matched_w_thresholds).
    OP2 = t_S: smallest t with weighted confident-S on W <= 2.5% AND unweighted share of P_S >= t over all
          power-matched D loci of the fit set (D loci with S-vs-D weight > 0) <= 10%; t_D as OP1 after S precedence;
          if no t <= 1 satisfies both, t_S = 1 (flag t_S_is_1: rule (c) and B1 then FAIL).
 2. Rule (a) (unchanged; phase-1 code path): ideal AUPRC vs the best tier-1 baseline, KILL iff diff < -1 locus SE.
 3. Rule (b) (unchanged, phase-1 reading at fixed 0.8; OP1 matched-W value reported alongside).
 4. Rule (c) at OP2 (primary) and OP1 (recorded): false-S on D at L4 and at rho = 0.5, comparators as phase 1
    (lowest false-S tier-1 baseline = pre-registered reading; best-AUPRC baseline; lowest non-degenerate = lowest
    among baselines that make >= 1 S call in the stratum). KILL iff > 2x and > +2 pp of the comparator.
 5. B1 usefulness at OP2 (primary) and OP1 (recorded): false-S on Dmid u Mx at L0+L2 below coloc-SuSiE's by more
    than 1 locus SE, AND recall-S (S loci of the L0+L2 cells, cells equal) >= coloc-SuSiE's - 1 locus SE.
    vs coloc-SuSiE-mm secondary.
 6. Rule (d): mean harness runtime_s vs coloc-SuSiE (phase-1 rule).
Block-bootstrap SEs are reported next to every locus SE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "famB_eval"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import killtest_B1 as K1  # noqa: E402
from common.seeds import rng  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval.io import truth_path_for  # noqa: E402
from eval.truth import read_truth  # noqa: E402

BMM = "exactcoloc_mm_gaussian_M50"
B1 = "exactcoloc_gaussian_M50"
CS, CS_MM, ABF = "coloc_susie", "coloc_susie_mm", "coloc_abf"
NOT_BASELINES = ("strength_only",)
D_LIMIT = 0.10
N_BOOT = 2000
NJ = 16


def op2_thresholds(fr: M.EvalFrame, m: str, setting: str = "realistic") -> dict:
    """DESIGN_P2 §3 OP2 on frame fr (fit set), method m."""
    cfg = M.MetricConfig()
    sc = fr.scores[m]
    out = {}
    if np.isnan(sc["P_S"]).all():
        return {"t_S": float("inf"), "t_D": M.matched_w_thresholds(fr, m, setting).get("t_D", float("inf")),
                "no_P_S": True}
    w = M.dw_weights(fr, setting, cfg.pairing_weights)
    isW = (fr.cls == "W") & (w > 0)
    rW = M.matched_threshold(sc["P_S"][isW], w[isW], cfg.w_limit)
    wsd = M.sd_weights(fr, setting, cfg.pairing_weights)
    isD = (fr.cls == "D") & (wsd > 0)
    rD = M.matched_threshold(sc["P_S"][isD], None, D_LIMIT)
    t = max(rW["t"], rD["t"])
    t_is_1 = bool(t > 1.0)
    if t_is_1:  # infeasible: t_S = 1 reported, and NO S calls are made (t used for calls = inf)
        t = float("inf")
    callS = np.nan_to_num(sc["P_S"], nan=-np.inf) >= t
    out.update(t_S=float(t), t_S_reported=1.0 if t_is_1 else float(t), t_S_W=float(rW["t"]), t_S_D10=float(rD["t"]),
               t_S_is_1=t_is_1,
               rate_S_on_W=float(np.sum(w[isW] * callS[isW]) / np.sum(w[isW])),
               share_S_on_pmD=float(callS[isD].mean()), n_pmD=int(isD.sum()))
    if not np.isnan(sc["P_D"]).all():
        eff = np.where(M._gate(fr, m, cfg.e_gate) & ~callS, sc["P_D"], -np.inf)
        eff = np.where(np.isnan(eff), -np.inf, eff)
        r = M.matched_threshold(eff[isW], w[isW], cfg.w_limit)
        out.update(t_D=r["t"], rate_D_on_W=r["rate"])
    else:
        out["t_D"] = float("inf")
    return out


def boot(stat, fr, unit, key):
    return M.cluster_bootstrap(stat, M.resampling_units(fr, unit), n_boot=N_BOOT, rng=rng(20261029, "ladder_Bmm", unit, key),
                               n_jobs=NJ, bca=False, return_replicates=True)


def both_se(stat, fr, key, ia, ib, primary):
    """Contrast ia - ib with locus and block bootstrap; `primary` names the unit whose SE the rule uses."""
    out = {}
    for unit in ("locus", "block"):
        res = boot(stat, fr, unit, key)
        out[unit] = K1.contrast_summary(res, ia, ib)
        out[unit]["estimates"] = [float(x) for x in res["estimate"]]
    out["primary_unit"] = primary
    return out


def main(argv=None):
    global N_BOOT, NJ
    ap = argparse.ArgumentParser()
    ap.add_argument("--bmm", required=True)
    ap.add_argument("--b1", required=True, help="phase-1 B predictions (famB_screen_v2@1 M50)")
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--abf-name", default="coloc_abf_W0.15_p12_5e-06")
    ap.add_argument("--gate-only", action="store_true")
    a = ap.parse_args(argv)
    N_BOOT, NJ = a.n_boot, a.n_jobs
    K1.NJ, K1.N_BOOT = a.n_jobs, a.n_boot

    truth = pd.concat([read_truth(truth_path_for(s)) for s in a.sets], ignore_index=True)
    ltype = dict(zip(truth["locus_id"].astype(str), truth["locus_type"].astype(str)))
    preds = K1.load_preds_dir(Path(a.baselines))
    if a.abf_name in preds:
        preds[ABF] = preds.pop(a.abf_name).assign(method=ABF)
    for m in [m for m in preds if m.startswith("coloc_abf_")]:
        preds.pop(m)
    preds[BMM] = pd.read_parquet(a.bmm)
    preds[B1] = pd.read_parquet(a.b1)
    assert set(preds[BMM]["method"]) == {BMM} and set(preds[B1]["method"]) == {B1}
    methods = sorted(preds)
    baselines = [m for m in methods if m not in (BMM, B1) and m not in NOT_BASELINES]
    ids = {m: set(preds[m]["locus_id"].astype(str)) for m in methods}
    alltruth = set(truth["locus_id"].astype(str))
    assert ids[BMM] == alltruth and ids[B1] == alltruth, "B-mm and phase-1 B must cover every screen_v2 locus"
    out = {"inputs": {"bmm": a.bmm, "b1": a.b1, "baselines": a.baselines, "abf_variant": a.abf_name},
           "methods": methods, "baselines": baselines, "n_loci": {m: len(ids[m]) for m in methods},
           "status_Bmm": preds[BMM]["status"].value_counts().to_dict(), "n_boot": N_BOOT}

    def frame(ms, mode="least_favourable"):
        common = set.intersection(*[ids[m] for m in ms])
        tr = truth[truth["locus_id"].astype(str).isin(common)]
        return M.build_eval_frame(tr, {m: preds[m][preds[m]["locus_id"].astype(str).isin(common)] for m in ms}, mode)

    # ---------------------------------------------------------------- 0. new-method gate
    fr = frame([BMM, B1])
    g = both_se(lambda idx, fr=fr: np.array([M.auprc_sd(fr.take(idx), BMM, "realistic"),
                                             M.auprc_sd(fr.take(idx), B1, "realistic")]), fr, ("gate",), 0, 1, "block")
    gb = g["block"]
    out["new_method_gate"] = {"auprc_realistic_Bmm": g["locus"]["estimates"][0], "auprc_realistic_B1": g["locus"]["estimates"][1],
                              "auroc_realistic_Bmm": M.auroc_sd(fr, BMM, "realistic"),
                              "auroc_realistic_B1": M.auroc_sd(fr, B1, "realistic"),
                              "per_level": {L: {m: M.auprc_sd(fr, m, f"level:{L}") for m in (BMM, B1)} for L in ("L2", "L4")},
                              "contrast": g, "pass": bool(gb["diff"] > gb["se"])}
    print(json.dumps(M._clean(out["new_method_gate"])), flush=True)
    if a.gate_only:
        Path(a.out).write_text(M.dumps(M._clean(out)))
        return

    # ---------------------------------------------------------------- 1. operating points (own loci)
    ops = {}
    for m in methods:
        f1 = frame([m])
        op1 = M.matched_w_thresholds(f1, m, "realistic")
        ops[m] = {"OP1": {"t_S": float(op1.get("t_S", np.inf)), "t_D": float(op1.get("t_D", np.inf)),
                          "rate_S_on_W": op1.get("rate_S_on_W"), "rate_D_on_W": op1.get("rate_D_on_W")},
                  "OP2": op2_thresholds(f1, m, "realistic"),
                  "has_P_S": bool(not np.isnan(f1.scores[m]["P_S"]).all()),
                  "failures": int(f1.failed[m].sum()), "n": len(f1)}
    out["operating_points"] = ops
    thr = {op: {m: (ops[m][op]["t_S"], ops[m][op]["t_D"]) for m in methods} for op in ("OP1", "OP2")}
    thr["fixed0.8"] = {m: (0.8, 0.8) for m in methods}
    out["Bmm_OP2_t_S_is_1"] = bool(ops[BMM]["OP2"].get("t_S_is_1", False))

    # ---------------------------------------------------------------- 2. rule (a)
    ra = {m: M.auprc_sd(frame([m]), m, "ideal") for m in methods}
    best = max(baselines, key=lambda m: -np.inf if np.isnan(ra[m]) else ra[m])
    fr = frame([BMM, best])
    fr = fr.mask((fr.level == "L0") & (np.nan_to_num(fr.overlap, nan=1) == 0))
    c = both_se(lambda idx, fr=fr: np.array([M.auprc_sd(fr.take(idx), BMM, "ideal"),
                                             M.auprc_sd(fr.take(idx), best, "ideal")]), fr, ("a",), 0, 1, "locus")
    out["rule_a"] = {"auprc_ideal_own_loci": ra, "best_baseline": best, "paired": c,
                     "kill": bool(c["locus"]["diff"] < -c["locus"]["se"])}

    # ---------------------------------------------------------------- 3. rule (b)
    rb = {}
    fr = frame([BMM, ABF])
    isW = fr.cls == "W"
    for tn in ("fixed0.8", "OP1"):
        conf = {m: np.isin(M.three_way_calls(fr, m, *thr[tn][m]), ["S", "D"]).astype(float) for m in (BMM, ABF)}

        def stat_b(idx, fr=fr, conf=conf, isW=isW):
            r1 = K1.cell_equal_rate(conf[BMM][idx], isW[idx], fr.cell[idx])
            r0 = K1.cell_equal_rate(conf[ABF][idx], isW[idx], fr.cell[idx])
            return np.array([r1, r0, r1 - max(2 * r0, r0 + 0.02)])

        e = {}
        for unit in ("locus", "block"):
            res = boot(stat_b, fr, unit, ("b", tn))
            e[unit] = {"p_one_sided": M.bootstrap_p_value(res, 0.0, j=2, alternative="greater"),
                       "se_delta": float(np.nanstd(res["replicates"][:, 2], ddof=1)) if "replicates" in res else None}
            est = res["estimate"]
        rb[tn] = {"conf_on_W_Bmm": float(est[0]), "conf_on_W_coloc_abf": float(est[1]), "delta": float(est[2]),
                  "boot": e, "kill": bool(est[2] > 0 and e["locus"]["p_one_sided"] < 0.05)}
    rb["primary"] = "fixed0.8 (phase-1 reading, unchanged); OP1 recorded"
    out["rule_b"] = rb

    # ---------------------------------------------------------------- 4. rule (c)
    def smask(fr, sname):
        return (fr.level == "L4") if sname == "L4" else (fr.pairing == "P3") & (np.abs(fr.overlap - 0.5) < 1e-9)

    rc = {}
    for sname in ("L4", "rho0.5"):
        rc[sname] = {}
        for tn in ("OP2", "OP1"):
            rates, auprc, ncalls = {}, {}, {}
            for m in methods:
                if not ops[m]["has_P_S"]:
                    continue
                f1 = frame([m])
                sm = smask(f1, sname)
                cS = (M.three_way_calls(f1, m, *thr[tn][m]) == "S").astype(float)
                rates[m] = K1.cell_equal_rate(cS, sm & (f1.cls == "D"), f1.cell)
                ncalls[m] = int(cS[sm].sum())
                sub = f1.mask(sm)
                auprc[m] = float(np.mean([M.auprc_sd(sub, m, f"cell:{c_}") for c_ in np.unique(sub.cell)]))
            bl = [m for m in rates if m in baselines]
            comps = {"lowest_falseS": min(bl, key=lambda m: rates[m]),
                     "best_auprc": max(bl, key=lambda m: auprc[m])}
            nd = [m for m in bl if ncalls[m] > 0]
            if nd:
                comps["lowest_nondegenerate"] = min(nd, key=lambda m: rates[m])
            entry = {"false_S_on_D": rates, "n_S_calls_in_stratum": ncalls, "mean_cell_auprc_sd": auprc,
                     "comparators": comps}
            for role, cm in comps.items():
                fr = frame([BMM, cm])
                msk = smask(fr, sname) & (fr.cls == "D")
                cS = {m: (M.three_way_calls(fr, m, *thr[tn][m]) == "S").astype(float) for m in (BMM, cm)}

                def stat_c(idx, fr=fr, cS=cS, msk=msk, cm=cm):
                    r1 = K1.cell_equal_rate(cS[BMM][idx], msk[idx], fr.cell[idx])
                    r0 = K1.cell_equal_rate(cS[cm][idx], msk[idx], fr.cell[idx])
                    return np.array([r1, r0, r1 - max(2 * r0, r0 + 0.02)])

                ps = {}
                for unit in ("locus", "block"):
                    res = boot(stat_c, fr, unit, ("c", sname, tn, role))
                    ps[unit] = M.bootstrap_p_value(res, 0.0, j=2, alternative="greater")
                est = res["estimate"]
                kill = bool(est[0] > 2 * est[1] and est[0] > est[1] + 0.02)
                if tn == "OP2" and ops[BMM]["OP2"].get("t_S_is_1"):
                    kill = True
                entry[role] = {"comparator": cm, "false_S_Bmm": float(est[0]), "false_S_comp": float(est[1]),
                               "delta": float(est[2]), "p_one_sided": ps, "kill": kill}
            rc[sname][tn] = entry
    out["rule_c"] = rc

    # ---------------------------------------------------------------- 5. B1 usefulness
    b1 = {}
    for comp in (CS, CS_MM):
        if comp not in methods:
            b1[comp] = "not available"
            continue
        fr = frame([BMM, comp])
        lt = np.array([ltype[i] for i in fr.locus_id])
        lev = (fr.level == "L0") | (fr.level == "L2")
        dmid = (fr.cls == "D") & (fr.r2 >= 0.3) & (fr.r2 < 0.8)
        mx = (lt == "M") & (fr.cls != "S")
        strata = {"primary_union_L0_L2": (dmid | mx) & lev}
        for sn, sm in (("Dmid", dmid), ("Mx", mx)):
            for L in ("L0", "L2"):
                strata[f"{sn}_{L}"] = sm & (fr.level == L)
        sS = (fr.cls == "S") & lev
        names = list(strata)
        for tn in ("OP2", "OP1", "fixed0.8"):
            callS = {m: (M.three_way_calls(fr, m, *thr[tn][m]) == "S").astype(float) for m in (BMM, comp)}

            def stat(idx, names=names, callS=callS, fr=fr, strata=strata, sS=sS):
                v = []
                for sn in names:
                    msk = strata[sn][idx]
                    v += [K1.cell_equal_rate(callS[BMM][idx], msk, fr.cell[idx]),
                          K1.cell_equal_rate(callS[comp][idx], msk, fr.cell[idx])]
                v += [K1.cell_equal_rate(callS[BMM][idx], sS[idx], fr.cell[idx]),
                      K1.cell_equal_rate(callS[comp][idx], sS[idx], fr.cell[idx])]
                return np.array(v)

            res = {u: boot(stat, fr, u, ("B1", comp, tn)) for u in ("locus", "block")}
            rows = {}
            for k, sn in enumerate(names):
                cl = K1.contrast_summary(res["locus"], 2 * k + 1, 2 * k)  # comp - Bmm (> 0: B-mm better)
                cb = K1.contrast_summary(res["block"], 2 * k + 1, 2 * k)
                rows[sn] = {"n": int(strata[sn].sum()), "false_S_Bmm": float(res["locus"]["estimate"][2 * k]),
                            f"false_S_{comp}": float(res["locus"]["estimate"][2 * k + 1]),
                            "comp_minus_Bmm": cl, "comp_minus_Bmm_block": cb, "pass_gt_1SE": bool(cl["diff"] > cl["se"])}
            k = len(names)
            rl = K1.contrast_summary(res["locus"], 2 * k, 2 * k + 1)  # Bmm - comp recall
            rbk = K1.contrast_summary(res["block"], 2 * k, 2 * k + 1)
            rows["recall_S_L0_L2"] = {"n": int(sS.sum()), "recall_Bmm": float(res["locus"]["estimate"][2 * k]),
                                      f"recall_{comp}": float(res["locus"]["estimate"][2 * k + 1]),
                                      "Bmm_minus_comp": rl, "Bmm_minus_comp_block": rbk,
                                      "pass_ge_minus_1SE": bool(rl["diff"] >= -rl["se"])}
            p = rows["primary_union_L0_L2"]["pass_gt_1SE"]
            if tn == "OP2":
                p = p and rows["recall_S_L0_L2"]["pass_ge_minus_1SE"] and not ops[BMM]["OP2"].get("t_S_is_1")
            rows["PASS"] = bool(p)
            b1.setdefault(comp, {})[tn] = rows
    out["B1_usefulness"] = b1

    # ---------------------------------------------------------------- 6. rule (d)
    rt = {m: float(np.nanmean(pd.to_numeric(preds[m]["runtime_s"], errors="coerce"))) for m in methods}
    rtm = {m: float(np.nanmedian(pd.to_numeric(preds[m]["runtime_s"], errors="coerce"))) for m in methods}
    fr = frame([BMM, CS])
    gain = {st: M.auprc_sd(fr, BMM, st) - M.auprc_sd(fr, CS, st) for st in ("ideal", "realistic")}
    ratio = rt[BMM] / rt[CS]
    out["rule_d"] = {"mean_runtime_s": rt, "median_runtime_s": rtm, "ratio_Bmm_over_coloc_susie": ratio,
                     "auprc_gain_vs_coloc_susie": gain, "kill": bool(ratio > 10 and max(gain.values()) <= 0)}

    # ---------------------------------------------------------------- summary
    s = {"new_method_gate_pass": out["new_method_gate"]["pass"], "rule_a_kill": out["rule_a"]["kill"],
         "rule_b_kill": out["rule_b"]["fixed0.8"]["kill"], "rule_d_kill": out["rule_d"]["kill"]}
    for tn in ("OP2", "OP1"):
        s[f"rule_c_kill_{tn}_prereg_lowest_falseS"] = any(rc[sn][tn]["lowest_falseS"]["kill"] for sn in rc)
        s[f"rule_c_kill_{tn}_best_auprc"] = any(rc[sn][tn]["best_auprc"]["kill"] for sn in rc)
        s[f"rule_c_kill_{tn}_lowest_nondegenerate"] = any(rc[sn][tn].get("lowest_nondegenerate", {}).get("kill", False)
                                                         for sn in rc)
        # DECISIONS P4-1 (PI, fixed before any B-mm prediction existed): binding = kill under (ii) OR (iii)
        s[f"rule_c_kill_{tn}_BINDING_P4-1"] = (s[f"rule_c_kill_{tn}_best_auprc"]
                                               or s[f"rule_c_kill_{tn}_lowest_nondegenerate"])
        if isinstance(b1.get(CS), dict):
            s[f"B1_usefulness_pass_{tn}"] = b1[CS][tn]["PASS"]
    out["summary"] = s
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(M.dumps(M._clean(out)))
    print(json.dumps(M._clean(s), indent=1))


if __name__ == "__main__":
    main()
