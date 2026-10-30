"""Trajectory/sensitivity table for G-mm identity mismatches: for each (locus, trait) and max_iter = k compare
  G-mm (fp64) vs susieR, and susieR vs susieR with z perturbed at rounding level (run_traj_mm.R outputs).
Writes <out>/traj.tsv with max |dPIP| per (id, trait, k) for G-vs-R and R-vs-Rpert."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from famG.susie_mm import MMConfig, susie_mm_batch
from famG.validation.mm_loci import load_exported


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--rjson", required=True, help="dir with <id>__t<trait>__p<pert>.json")
    ap.add_argument("--pairs", required=True, help="comma list id:trait")
    ap.add_argument("--ks", default="1,2,3,5,10,20,30,40,50")
    ap.add_argument("--pert", default="1e-14")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ks = [int(k) for k in a.ks.split(",")]
    pairs = [(x.rsplit(":", 1)[0], int(x.rsplit(":", 1)[1])) for x in a.pairs.split(",")]
    ids = sorted(set(i for i, _ in pairs))
    loci = load_exported(Path(a.work), ids)
    by = {d["id"]: d for d in loci}
    G = {}
    for k in ks:
        fits = susie_mm_batch(loci, MMConfig(ld_dtype=torch.float64, use_triton=a.device != "cpu", max_iter=k),
                              device=torch.device(a.device))
        for d, f in zip(loci, fits):
            G[(d["id"], k)] = f
    rows = []
    for i, t in pairs:
        R0 = {f["k"]: f for f in json.load(open(Path(a.rjson) / f"{i}__t{t}__p0.json"))["fits"]}
        R1 = {f["k"]: f for f in json.load(open(Path(a.rjson) / f"{i}__t{t}__p{a.pert}.json"))["fits"]}
        for k in ks:
            g = G[(i, k)][t - 1]
            r0, r1 = np.asarray(R0[k]["pip"]), np.asarray(R1[k]["pip"])
            rows.append(dict(id=i, trait=t, k=k, dpip_G_R=float(np.abs(g["pip"] - r0).max()),
                             dpip_R_Rpert=float(np.abs(r1 - r0).max()), dpip_G_Rpert=float(np.abs(g["pip"] - r1).max()),
                             lam_R=R0[k]["lambda"], lam_G=g["lambda_bias"], lam_Rpert=R1[k]["lambda"]))
    df = pd.DataFrame(rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.out) / "traj.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string())


if __name__ == "__main__":
    main()
