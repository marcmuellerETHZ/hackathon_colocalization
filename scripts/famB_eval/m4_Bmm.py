"""B-mm on the M4 main validation subsample (m4_subsample@1): candidacy bar (DESIGN_P2 §4) and OP2 on M4.

    PYTHONPATH=src python scripts/famB_eval/m4_Bmm.py --bmm <B-mm M4 predictions.parquet> \
        [--b1 <phase-1 B M4 predictions.parquet>] [--comp <predictions (dir/parquet) holding coloc_susie_mm> ...] \
        --ids results/baselines_m4/m4_subsample_v1/main.txt --sets data/processed/sets/valid_grid_v1/*.h5 --out <json>

Definitions (DESIGN v4 §8 / DESIGN_P2 §3-§4; metrics from src/eval/metrics.py, least-favourable failure scoring):
  * endpoints on the M4 main loci: realistic AUPRC(S vs D) (L1-L6, §8.6 weights), ideal AUPRC, AUROC D-vs-W;
  * candidacy bar: realistic AUPRC(B-mm) >= realistic AUPRC(coloc_susie_mm) - 1 paired SE, SE = SD of 2,000 paired
    cluster-bootstrap replicates over LD blocks, on loci scored by both;
  * OP1 (matched-W) and OP2 (ladder_Bmm.op2_thresholds) fitted on the M4 realistic cells; recall-S/D, W->S/D
    (metrics.recall_at) and the unweighted false-S share over power-matched realistic D loci at each point;
  * runtime: harness runtime_s per locus (mean, median).
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
from ladder_Bmm import op2_thresholds  # noqa: E402

BMM, B1, CSMM = "exactcoloc_mm_gaussian_M50", "exactcoloc_gaussian_M50", "coloc_susie_mm"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bmm", required=True)
    ap.add_argument("--b1", default=None)
    ap.add_argument("--comp", nargs="*", default=[], help="prediction sources for comparators (any K1.load_preds_dir form)")
    ap.add_argument("--comp-methods", nargs="*", default=[CSMM, "coloc_susie", "coloc_abf_W0.2_p12_5e-06"])
    ap.add_argument("--ids", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=16)
    a = ap.parse_args(argv)
    ids = set(Path(a.ids).read_text().split())
    truth = pd.concat([read_truth(truth_path_for(s)) for s in a.sets], ignore_index=True)
    truth = truth[truth["locus_id"].astype(str).isin(ids)]
    assert len(truth) == len(ids), (len(truth), len(ids))
    preds = {BMM: pd.read_parquet(a.bmm)}
    if a.b1:
        preds[B1] = pd.read_parquet(a.b1)
    parts: dict = {}
    for src in a.comp:  # several sources of one method (e.g. disjoint half runs) are concatenated
        for m, df in K1.load_preds_dir(Path(src)).items():
            if m in a.comp_methods and m not in preds:
                parts.setdefault(m, []).append(df)
    for m, ps in parts.items():
        df = pd.concat(ps, ignore_index=True)
        if df["locus_id"].duplicated().any():
            raise ValueError(f"{m}: duplicated loci across --comp sources")
        preds[m] = df
    for m in list(preds):
        preds[m] = preds[m][preds[m]["locus_id"].astype(str).isin(ids)]
    have = {m: set(preds[m]["locus_id"].astype(str)) for m in preds}
    assert have[BMM] == ids, "B-mm must cover the whole M4 main subsample"
    out = {"n_main": len(ids), "methods": sorted(preds), "coverage": {m: len(v) for m, v in have.items()},
           "status": {m: preds[m]["status"].value_counts().to_dict() for m in preds}}

    def frame(ms):
        common = set.intersection(*[have[m] for m in ms])
        tr = truth[truth["locus_id"].astype(str).isin(common)]
        return M.build_eval_frame(tr, {m: preds[m][preds[m]["locus_id"].astype(str).isin(common)] for m in ms},
                                  "least_favourable")

    ep, opp = {}, {}
    for m in preds:
        fr = frame([m])
        ep[m] = {"n": len(fr), "failures": int(fr.failed[m].sum()),
                 "auprc_realistic": M.auprc_sd(fr, m, "realistic"), "auprc_ideal": M.auprc_sd(fr, m, "ideal"),
                 "auroc_sd_realistic": M.auroc_sd(fr, m, "realistic"), "auroc_dw_realistic": M.auroc_dw(fr, m, "realistic"),
                 "per_level": {L: M.auprc_sd(fr, m, f"level:{L}") for L in sorted(set(fr.level)) if L != ""},
                 "runtime_mean_s": float(np.nanmean(pd.to_numeric(preds[m]["runtime_s"], errors="coerce"))),
                 "runtime_median_s": float(np.nanmedian(pd.to_numeric(preds[m]["runtime_s"], errors="coerce")))}
        op1 = M.matched_w_thresholds(fr, m, "realistic")
        op2 = op2_thresholds(fr, m, "realistic")
        wsd = M.sd_weights(fr, "realistic")
        pmD = (fr.cls == "D") & (wsd > 0)
        o = {}
        for name, t in (("OP1", (op1.get("t_S", np.inf), op1.get("t_D", np.inf))), ("OP2", (op2["t_S"], op2["t_D"])),
                        ("fixed0.8", (0.8, 0.8))):
            calls = M.three_way_calls(fr, m, *t)
            o[name] = {"t_S": float(t[0]), "t_D": float(t[1]), **M.recall_at(fr, m, *t, setting="realistic"),
                       "false_S_share_pmD_realistic": float(np.mean(calls[pmD] == "S"))}
        o["OP2"]["detail"] = op2
        opp[m] = o
    out["endpoints"] = ep
    out["operating_points_M4_realistic"] = opp

    con = {}
    for comp in [m for m in (CSMM, B1) if m in preds]:
        fr = frame([BMM, comp])
        res = M.cluster_bootstrap(lambda idx, fr=fr, comp=comp: np.array([M.auprc_sd(fr.take(idx), BMM, "realistic"),
                                                                           M.auprc_sd(fr.take(idx), comp, "realistic")]),
                                  M.resampling_units(fr, "block"), n_boot=a.n_boot,
                                  rng=rng(20261029, "m4_Bmm", comp), n_jobs=a.n_jobs, bca=False, return_replicates=True)
        c = K1.contrast_summary(res, 0, 1)
        con[comp] = {"n_paired": len(fr), "auprc_Bmm": float(res["estimate"][0]), f"auprc_{comp}": float(res["estimate"][1]),
                     "Bmm_minus_comp": c}
        if comp == CSMM:
            con[comp]["candidacy_bar_pass"] = bool(c["diff"] >= -c["se"])
    out["contrasts_block_bootstrap"] = con
    if CSMM not in preds:
        out["candidacy_bar"] = "coloc_susie_mm M4 predictions not supplied: bar not evaluated"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(M.dumps(M._clean(out)))
    print(M.dumps(M._clean({k: out[k] for k in ("coverage", "endpoints", "contrasts_block_bootstrap")})))


if __name__ == "__main__":
    main()
