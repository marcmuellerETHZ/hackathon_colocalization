"""DECISIONS P4-3 check for non-converged R fits: G-mm matches a non-converged susieR fit iff (a) G-mm is also
non-converged at the same max_iter and (b) |dPIP(G, R)| and |d PP.H4(G, R)| are within the R-vs-R spread, i.e. the
max over seeds of |dPIP(R_s, R)| and |d PP.H4(R_s, R)|, R_s = susieR on z * (1 + 1e-14 u_s) (run_susie016_mm.R with
MM_PERT/MM_SEED). PP.H4 of a locus = max over CS pairs of PP.H4 (what the §4.3 adapter's P_S uses); coloc.abf
fallback PP.H4 (W = 0.15, p12 = 5e-6, as the wrapper) when a run has no CS pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.abf import coloc_abf
from eval.io import LocusSetReader


def _h4(js, lid, cache):
    pr = js.get("pairs")
    if pr is not None:
        return float(np.asarray(pr["PP"], float).reshape(-1, 5)[:, 4].max())
    if lid not in cache:
        cell = lid.split("__")[1]
        with LocusSetReader(f"data/processed/sets/valid_grid_v1/{cell}.h5") as rd:
            lc = rd.read(lid, load_ld=False)
        cache[lid] = float(coloc_abf(lc.z1, lc.se1, lc.z2, lc.se2, W=0.15, p12=5e-6)[4])
    return cache[lid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r0", required=True)
    ap.add_argument("--pert", required=True, help="dir with s<seed>/ subdirs of perturbed R outputs")
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--gfits", required=True, help="compare_mm fits_*.tsv")
    ap.add_argument("--gpairs", required=True, help="compare_mm pairs_*.tsv")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    ids = Path(a.ids).read_text().split()
    gf = pd.read_csv(a.gfits, sep="\t").set_index(["locus_id", "trait"])
    gp = pd.read_csv(a.gpairs, sep="\t").set_index("locus_id")
    cache, rows = {}, []
    for lid in ids:
        R0 = json.load(open(Path(a.r0) / f"{lid}.json"))
        Rs = [json.load(open(Path(a.pert) / f"s{s}" / f"{lid}.json")) for s in seeds]
        h0 = _h4(R0, lid, cache)
        sp_h4 = max(abs(_h4(r, lid, cache) - h0) for r in Rs)
        g_h4 = float(gp.loc[lid, "maxH4_g"]) if bool(gp.loc[lid, "g_pairs"]) else _h4({"pairs": None}, lid, cache)
        d_h4 = abs(g_h4 - h0)
        for t in (1, 2):
            r0 = R0[f"t{t}"]
            if r0.get("converged", True):
                continue
            pip0 = np.asarray(r0["pip"])
            sp_pip = max(float(np.abs(np.asarray(r[f"t{t}"]["pip"]) - pip0).max()) for r in Rs)
            g = gf.loc[(lid, t)]
            rows.append(dict(locus_id=lid, trait=t, niter_r=r0["niter"], niter_g=int(g.niter_g),
                             conv_g=bool(g.conv_g), dpip_G_R=float(g.max_dpip), spread_pip=sp_pip,
                             dH4_G_R=d_h4, spread_H4=sp_h4,
                             a_ok=(not bool(g.conv_g)) and int(g.niter_g) == r0["niter"],
                             b_ok=(float(g.max_dpip) <= sp_pip) and (d_h4 <= sp_h4),
                             b_ok_or_1e3=(float(g.max_dpip) <= max(sp_pip, 1e-3)) and (d_h4 <= max(sp_h4, 1e-3))))
    df = pd.DataFrame(rows)
    df["match"] = df.a_ok & df.b_ok                 # P4-3 strict
    df["match_or_1e3"] = df.a_ok & df.b_ok_or_1e3   # also accepting differences below the converged-fit bar
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.out) / "sensitivity.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string())
    print(json.dumps({"n_nonconv_R_fits": len(df), "n_match": int(df.match.sum()),
                      "n_match_or_1e3": int(df.match_or_1e3.sum()),
                      "fails": df[~df.match][["locus_id", "trait"]].values.tolist()}, indent=1))


if __name__ == "__main__":
    main()
