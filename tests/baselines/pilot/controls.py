"""Known-direction controls of the baseline wrappers on simulated validation loci (DESIGN §7.1 item 9; lesson 9).

This script reads oracle labels, so it lives OUTSIDE src/baselines (method code never reads them). It
  select   picks control loci from a validation locus set by oracle class (seed derive(20261029, ...)):
             shared_strong   : class S, one causal variant per trait, the same variant, min(|mu1|, |mu2|) >= 15
             distinct_strong : class D, one causal variant per trait, r^2 < 0.1, min(|mu1|, |mu2|) >= 10
             null_trait      : class W, one trait without any causal variant, the other |mu| >= 10
           N_PER per category per cell (fewer if the cell has fewer), written as an id list for baselines.pilot run;
  analyse  joins the pilot predictions with the labels and reports, per method x category x mismatch level:
             shared_strong   : share with s > 0.9 (LD-using methods and coloc.abf; DESIGN control); p-value methods:
                               share with p > 0.05 (descriptive: p is uniform under its null)
             distinct_strong : share with s < 0.1 (descriptive); proportional tests: share with p < 0.05,
                               DESIGN control requires >= 0.8
             null_trait      : share with P_W > 0.5 (DESIGN "null loci must give high P_W")
Usage:
  python tests/baselines/pilot/controls.py select --set S.h5 --out ids.txt
  python tests/baselines/pilot/controls.py analyse --runs src/baselines/benchmarks/<tag>_runs.json --out table.tsv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from common.seeds import VALID_SEED, rng  # noqa: E402
from eval.io import truth_path_for  # noqa: E402
from eval.truth import read_truth  # noqa: E402

N_PER = 6
PVALUE = {"prop_coloc", "coloc_prop_test"}
NO_PROB = PVALUE | {"lead_snp"}


def categorise(t: pd.DataFrame) -> pd.Series:
    n1, n2 = t["causal1"].map(len), t["causal2"].map(len)
    one = (n1 == 1) & (n2 == 1)
    same = one & t.apply(lambda r: len(r.causal1) == 1 and len(r.causal2) == 1 and r.causal1[0] == r.causal2[0],
                         axis=1)
    mn = np.minimum(t["mu1"].abs(), t["mu2"].abs())
    mx = np.maximum(t["mu1"].abs(), t["mu2"].abs())
    cat = pd.Series("", index=t.index)
    cat[(t["oracle_class"] == "S") & same & (mn >= 15)] = "shared_strong"
    cat[(t["oracle_class"] == "D") & one & (t["r2"] < 0.1) & (mn >= 10)] = "distinct_strong"
    cat[(t["oracle_class"] == "W") & ((n1 == 0) | (n2 == 0)) & (mx >= 10)] = "null_trait"
    return cat


def select(locus_set: str, out: str, n_per: int = N_PER) -> None:
    t = read_truth(truth_path_for(locus_set))
    if (t["split"].astype(str) != "valid").any():
        raise SystemExit("controls use validation loci only")
    t["cat"] = categorise(t)
    ids = []
    for (cell, cat), d in t[t["cat"] != ""].groupby(["cell_id", "cat"], sort=True):
        g = rng(VALID_SEED, "baselines/controls", Path(locus_set).stem, cell, cat)
        d = d.sort_values("locus_id")
        k = min(n_per, len(d))
        ids += d["locus_id"].to_numpy()[np.sort(g.choice(len(d), size=k, replace=False))].tolist()
        print(f"{cell:20s} {cat:16s} {k:3d} of {len(d)}")
    Path(out).write_text("\n".join(ids) + "\n")
    print(f"{len(ids)} control ids -> {out}")


def analyse(runs: str, out: str) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "src"))
    from baselines.pilot import load_predictions

    rec = json.loads(Path(runs).read_text())
    t = read_truth(truth_path_for(rec["locus_set"]))
    t["cat"] = categorise(t)
    t = t.set_index("locus_id")
    rows = []
    for m in rec["runs"]:
        p = load_predictions(rec, m).join(t[["cat", "mismatch_level", "oracle_class"]], on="locus_id")
        p = p[p["cat"] != ""]
        for (cat, lev), d in p.groupby(["cat", "mismatch_level"]):
            okd = d[d["status"].isin(["ok", "nonconverged"])]
            s = okd["s"].astype(float)
            r = {"method": m, "category": cat, "level": lev, "n": len(d), "n_fail": len(d) - len(okd)}
            if cat == "shared_strong":
                r["criterion"] = "p>0.05" if m in PVALUE else "s>0.9"
                r["pass_rate"] = float((s > 0.05).mean() if m in PVALUE else (s > 0.9).mean())
            elif cat == "distinct_strong":
                r["criterion"] = "p<0.05" if m in PVALUE else "s<0.1"
                r["pass_rate"] = float((s < 0.05).mean() if m in PVALUE else (s < 0.1).mean())
            else:
                if m in NO_PROB:
                    continue
                r["criterion"] = "P_W>0.5"
                r["pass_rate"] = float((okd["P_W"].astype(float) > 0.5).mean())
            r["median_s"] = float(s.median()) if len(s) else np.nan
            rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(out, sep="\t", index=False, float_format="%.3g")
    piv = tab.pivot_table(index=["method"], columns=["category", "level"], values="pass_rate")
    pd.set_option("display.width", 250)
    print(piv.round(2).to_string())
    return tab


def main(argv=None):
    if os.environ.get("COLOC_ENV_ACTIVE") != "1":
        raise SystemExit("source env/activate.sh first")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("select")
    a1.add_argument("--set", required=True)
    a1.add_argument("--out", required=True)
    a1.add_argument("--n-per", type=int, default=N_PER)
    a2 = sub.add_parser("analyse")
    a2.add_argument("--runs", required=True)
    a2.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "select":
        select(a.set, a.out, a.n_per)
    else:
        analyse(a.runs, a.out)


if __name__ == "__main__":
    main()
