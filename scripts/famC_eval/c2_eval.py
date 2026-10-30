"""Family C kill test C2 (DESIGN v4 §5 C; DESIGN_P2 §2 C, §3, §4) on the M4 main validation subsample.

    PYTHONPATH=src python scripts/famC_eval/c2_eval.py --ids results/baselines_m4/m4_subsample_v1/main.txt \
        --set data/processed/sets/valid_grid_v1.h5 --net results/famC/c2_v1/net_m4 \
        --pred nold=results/famC/c2_v1/nold_m4 --pred floor=results/famC/c2_v1/floor_m4 \
        --pred A=<famA_m4_valid@1 run dir> --pred B=<famB_m4_valid@1 predictions> \
        --trainregion-set data/processed/famC/sets/trainregion_m4_v1.h5 --net-trainregion results/famC/c2_v1/net_tr \
        --out results/famC/c2_v1/c2.json

Prediction sources (--net, --net-trainregion, --pred NAME=SRC): a predictions parquet, a harness run dir
(predictions.parquet + manifest.json; sha256 checked), or a directory with runs.json (scripts/famC_eval/run_harness.py
output: predictions.parquet there; or famA-style {"runs": {method: [{run_dir}]}}). Each source must hold ONE method.

Definitions (all metrics from src/eval/metrics.py; failures scored least-favourably, §4.3):
  * endpoints per method on its own M4 main loci: realistic AUPRC S-vs-D (L1-L6, §8.6 weights), ideal AUPRC (L0),
    realistic AUROC D-vs-W, AUPRC per level, ECE of P_S / P_D raw and recalibrated (one logit intercept fitted on
    the same validation frame, as run_benchmark does on validation runs), runtime_s per locus;
  * paired contrasts net - X on the loci scored by both: 2,000 cluster-bootstrap replicates over LD blocks
    (seed rng(20261029, "famC_C2", X)); paired SE = SD of the replicate differences (DESIGN_P2 §4 M3); the
    recalibration intercept is refitted within each replicate;
  * region gap = realistic AUPRC(net, trainregion set) - realistic AUPRC(net, M4 main); SE from the two independent
    block bootstraps;
  * OP1 (matched-W C3, metrics.matched_w_thresholds) and OP2 (eval.op2.op2_thresholds, DESIGN_P2 §3) of every
    method, fitted on the M4 realistic cells, with recalls and unweighted false-S on power-matched D.
C2 verdict (pre-declared in task p9.famC_C2 notes before any C2 number was seen):
  (i)   realistic AUPRC(net) >= that of the better of A and B (point estimate of the paired difference >= 0), and
        recalibrated ECE_S and ECE_D of net <= the comparator's + 1 paired SE ("equal or better");
  (ii)  region gap < 0.02;
  (iii) net - floor and net - nold realistic AUPRC each > 1 paired SE.
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

NET = "net"


def load_source(src: str) -> pd.DataFrame:
    from eval.harness import sha256_file
    p = Path(src)
    if p.is_file():
        return pd.read_parquet(p)
    if (p / "manifest.json").exists() and (p / "predictions.parquet").exists():
        man = json.loads((p / "manifest.json").read_text())
        if man.get("predictions_sha256") and sha256_file(p / "predictions.parquet") != man["predictions_sha256"]:
            raise SystemExit(f"{p}: predictions sha256 differs from manifest")
        return pd.read_parquet(p / "predictions.parquet")
    if (p / "runs.json").exists():
        rj = json.loads((p / "runs.json").read_text())
        if (p / "predictions.parquet").exists():
            return pd.read_parquet(p / "predictions.parquet")
        parts = []
        for m, runs in rj["runs"].items():
            for r in runs:
                parts.append(load_source(r["run_dir"]))
        return pd.concat(parts, ignore_index=True)
    raise SystemExit(f"cannot read predictions from {src}")


def recal_ece(fr: M.EvalFrame, m: str, ev: str) -> float:
    col = "P_S" if ev == "S" else "P_D"
    p = fr.scores[m][col]
    ok = ~np.isnan(p)
    if not ok.any():
        return float("nan")
    y = (fr.cls[ok] == ev).astype(float)
    a = M.fit_logit_intercept(p[ok], y)
    if not np.isfinite(a):
        return float("nan")
    return M.ece(M.apply_logit_intercept(p[ok], a), y)


def endpoints(fr: M.EvalFrame, m: str, preds: pd.DataFrame) -> dict:
    lv = sorted(set(str(x) for x in fr.level if str(x)))
    cal = M.calibration(fr, m)
    rt = pd.to_numeric(preds["runtime_s"], errors="coerce")
    return {"n": int(len(fr)), "n_failed": int(fr.failed[m].sum()),
            "status": {str(k): int(v) for k, v in preds["status"].value_counts().items()},
            "auprc_realistic": M.auprc_sd(fr, m, "realistic"), "auprc_ideal": M.auprc_sd(fr, m, "ideal"),
            "auroc_sd_realistic": M.auroc_sd(fr, m, "realistic"), "auroc_dw_realistic": M.auroc_dw(fr, m, "realistic"),
            "auroc_dw_ideal": M.auroc_dw(fr, m, "ideal"),
            "auprc_per_level": {L: M.auprc_sd(fr, m, f"level:{L}") for L in lv},
            "ece_S_raw": cal.get("ece_S"), "ece_D_raw": cal.get("ece_D"),
            "ece_S_recal": recal_ece(fr, m, "S"), "ece_D_recal": recal_ece(fr, m, "D"),
            "runtime_mean_s": float(np.nanmean(rt)), "runtime_median_s": float(np.nanmedian(rt)),
            "runtime_p95_s": float(np.nanpercentile(rt, 95))}


def op_points(fr: M.EvalFrame, m: str) -> dict:
    from eval.op2 import op2_thresholds
    op1 = M.matched_w_thresholds(fr, m, "realistic")
    op2 = op2_thresholds(fr, m, "realistic")
    pmD = (fr.cls == "D") & (M.sd_weights(fr, "realistic") > 0)
    out = {}
    for name, t in (("OP1", (op1.get("t_S", np.inf), op1.get("t_D", np.inf))),
                    ("OP2", (op2.get("t_S_call", np.inf), op2.get("t_D", np.inf))), ("fixed0.8", (0.8, 0.8))):
        calls = M.three_way_calls(fr, m, float(t[0]), float(t[1]))
        out[name] = {"t_S": float(t[0]), "t_D": float(t[1]), **M.recall_at(fr, m, float(t[0]), float(t[1]), "realistic"),
                     "false_S_share_pmD_realistic_unweighted": float(np.mean(calls[pmD] == "S")) if pmD.any() else None}
    out["OP1"]["detail"], out["OP2"]["detail"] = op1, op2
    return out


def boot_vec(fr: M.EvalFrame, ms: list[str]):
    def stat(idx):
        f = fr.take(idx)
        v = []
        for m in ms:
            v += [M.auprc_sd(f, m, "realistic"), M.auroc_dw(f, m, "realistic"), recal_ece(f, m, "S"),
                  recal_ece(f, m, "D")]
        return np.array(v)
    return stat


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--pred", action="append", default=[], help="NAME=SRC (repeatable)")
    ap.add_argument("--trainregion-set", default=None)
    ap.add_argument("--net-trainregion", default=None)
    ap.add_argument("--comparators", nargs="*", default=["A", "B"])
    ap.add_argument("--floors", nargs="*", default=["floor", "nold"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=16)
    a = ap.parse_args(argv)
    ids = set(Path(a.ids).read_text().split())
    truth = read_truth(truth_path_for(Path(a.set)))
    assert not truth["split"].astype(str).eq("test").any()
    truth = truth[truth["locus_id"].astype(str).isin(ids)]
    assert len(truth) == len(ids), (len(truth), len(ids))
    preds, srcs = {}, {}
    for name, src in [(NET, a.net)] + [tuple(x.split("=", 1)) for x in a.pred]:
        df = load_source(src)
        ms = df["method"].unique()
        if len(ms) != 1:
            raise SystemExit(f"{src}: {len(ms)} methods ({ms[:5]}); one expected")
        df = df[df["locus_id"].astype(str).isin(ids)].copy()
        if df["locus_id"].duplicated().any():
            raise SystemExit(f"{name}: duplicate locus rows")
        srcs[name] = {"src": src, "method": str(ms[0]), "n_loci_in_main": int(len(df))}
        df["method"] = name
        preds[name] = df
    have = {m: set(preds[m]["locus_id"].astype(str)) for m in preds}
    if have[NET] != ids:
        raise SystemExit(f"net covers {len(have[NET])} of {len(ids)} main ids")
    out = {"ids": a.ids, "set": a.set, "n_main": len(ids), "sources": srcs, "n_boot": a.n_boot, "unit": "block",
           "failure_mode": "least_favourable"}

    def frame(ms, tr=truth, pr=preds):
        common = set.intersection(*[set(pr[m]["locus_id"].astype(str)) for m in ms])
        t = tr[tr["locus_id"].astype(str).isin(common)]
        return M.build_eval_frame(t, {m: pr[m][pr[m]["locus_id"].astype(str).isin(common)] for m in ms},
                                  "least_favourable")

    ep, ops = {}, {}
    for m in preds:
        fr = frame([m])
        ep[m] = endpoints(fr, m, preds[m])
        ops[m] = op_points(fr, m)
        print(m, json.dumps({k: ep[m][k] for k in ("n", "n_failed", "auprc_realistic", "auroc_dw_realistic",
                                                    "ece_S_recal", "ece_D_recal", "runtime_median_s")}), flush=True)
    out["endpoints_own_loci"] = ep
    out["operating_points_M4_realistic"] = ops

    names = ("auprc_realistic", "auroc_dw_realistic", "ece_S_recal", "ece_D_recal")
    con = {}
    for comp in [m for m in preds if m != NET]:
        fr = frame([NET, comp])
        res = M.cluster_bootstrap(boot_vec(fr, [NET, comp]), M.resampling_units(fr, "block"), n_boot=a.n_boot,
                                  rng=rng(20261029, "famC_C2", comp), n_jobs=a.n_jobs, bca=False,
                                  return_replicates=True)
        est = np.atleast_1d(res["estimate"])
        c = {"n_paired": int(len(fr)), "n_clusters": res["n_clusters"], "n_boot_nan": res["n_boot_nan"]}
        for k, nm in enumerate(names):
            d = M.contrast(res, k, len(names) + k)
            c[nm] = {"net": float(est[k]), comp: float(est[len(names) + k]), "diff_net_minus": d["estimate"],
                     "se": d["se"], "ci95_percentile": [d["ci_low"], d["ci_high"]]}
        con[comp] = c
        print("contrast", comp, json.dumps({nm: c[nm] for nm in names}, default=float), flush=True)
    out["paired_contrasts"] = con

    gap = None
    if a.trainregion_set and a.net_trainregion:
        ttr = read_truth(truth_path_for(Path(a.trainregion_set)))
        assert set(ttr["split"].astype(str)) == {"train"}, set(ttr["split"].astype(str))
        ptr = load_source(a.net_trainregion)
        ptr = ptr[ptr["locus_id"].astype(str).isin(set(ttr["locus_id"].astype(str)))].copy()
        ptr["method"] = NET
        if len(ptr) != len(ttr):
            raise SystemExit(f"net covers {len(ptr)} of {len(ttr)} trainregion loci")
        frt = M.build_eval_frame(ttr, {NET: ptr}, "least_favourable")
        frv = frame([NET])
        st = lambda f: (lambda idx: np.array([M.auprc_sd(f.take(idx), NET, "realistic")]))
        rt = M.cluster_bootstrap(st(frt), M.resampling_units(frt, "block"), n_boot=a.n_boot,
                                 rng=rng(20261029, "famC_C2", "gap_train"), n_jobs=a.n_jobs, bca=False)
        rv = M.cluster_bootstrap(st(frv), M.resampling_units(frv, "block"), n_boot=a.n_boot,
                                 rng=rng(20261029, "famC_C2", "gap_valid"), n_jobs=a.n_jobs, bca=False)
        at, av = float(np.atleast_1d(rt["estimate"])[0]), float(np.atleast_1d(rv["estimate"])[0])
        se = float(np.hypot(np.atleast_1d(rt["se"])[0], np.atleast_1d(rv["se"])[0]))
        gap = {"auprc_realistic_trainregion": at, "auprc_realistic_m4": av, "gap": at - av, "se": se,
               "n_trainregion": int(len(frt)), "n_blocks_trainregion": rt["n_clusters"],
               "trainregion_endpoints": endpoints(frt, NET, ptr), "pass": bool(at - av < 0.02)}
        print("gap", json.dumps({k: v for k, v in gap.items() if k != "trainregion_endpoints"}), flush=True)
    out["region_gap"] = gap

    v = {}
    comps = [c for c in a.comparators if c in con]
    v["comparators_present"] = comps
    v["comparators_missing"] = [c for c in a.comparators if c not in con]
    if comps:
        best = max(comps, key=lambda c: con[c]["auprc_realistic"][c])
        cb = con[best]
        v["best_of_A_B"] = best
        v["i_auprc"] = bool(cb["auprc_realistic"]["diff_net_minus"] >= 0)
        v["i_ece"] = bool(all(cb[k]["diff_net_minus"] <= cb[k]["se"] for k in ("ece_S_recal", "ece_D_recal")))
    v["ii_gap"] = None if gap is None else gap["pass"]
    for f in a.floors:
        if f in con:
            c = con[f]["auprc_realistic"]
            v[f"iii_beats_{f}"] = bool(c["diff_net_minus"] > c["se"])
    req = ["i_auprc", "i_ece", "ii_gap"] + [f"iii_beats_{f}" for f in a.floors]
    complete = not v["comparators_missing"] and all(v.get(k) is not None for k in req)
    v["complete"] = complete
    v["C2_pass"] = bool(complete and all(v[k] for k in req)) if complete else None
    out["verdict"] = v
    print("VERDICT", json.dumps(v), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(M.dumps(M._clean(out)))


if __name__ == "__main__":
    main()
