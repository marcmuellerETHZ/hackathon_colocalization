"""Reviewer check of the layer-E variance table src/famA/ref_ld_var_v1.json (task p2.review_famA).

1. Every calibration window recorded in the JSON lies inside a TRAIN block of splits/blocks_v1.tsv.
2. Provenance: re-run famA.validation.calibrate_ref_ld_var.calibrate with the documented arguments and compare
   with the committed table (bitwise / max abs diff).
3. Independent re-implementation on the first --n-win recorded windows at n = 150 (L2 reference size):
   n·Var(r̂) by (r² bin, min-MAF bin) with (a) the committed convention (subsample-monomorphic r̂ := 0) and
   (b) monomorphic draws excluded per pair (inference LD never contains a monomorphic reference column: the
   simulator drops them). Quantifies how much of the rare-pair inflation is the r̂ := 0 convention.
Writes $ORCH_OUT/layerE_check.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common.seeds import VALID_SEED, rng as seeded_rng
from famA.validation import calibrate_ref_ld_var as C
from famG.validation.genotypes import chrom_variants, read_genotypes

TABLE = Path("src/famA/ref_ld_var_v1.json")


def main():
    out = Path(os.environ.get("ORCH_OUT", ".")); out.mkdir(parents=True, exist_ok=True)
    tab = json.loads(TABLE.read_text())
    blocks = pd.read_csv("splits/blocks_v1.tsv", sep="\t").set_index("block_id")
    res = {}
    # 1. windows inside train blocks
    bad = []
    for w in tab["windows"]:
        b = blocks.loc[w["block"]]
        ok = (b["split"] == "train") and (str(b["chr"]).replace("chr", "") == str(w["chrom"])) \
            and (b["start"] <= w["start_pos"]) and (w["end_pos"] < b["end"])
        if not ok:
            bad.append(w)
    res["n_windows"] = len(tab["windows"]); res["windows_not_in_train_block"] = bad
    res["split_of_window_blocks"] = blocks.loc[[w["block"] for w in tab["windows"]], "split"].value_counts().to_dict()
    # 2. provenance re-run
    new = C.calibrate(int(len(tab["windows"])), int(tab["p"]), int(tab["draws"]), [int(n) for n in tab["n_list"]])
    res["rerun_windows_equal"] = new["windows"] == tab["windows"]
    res["rerun_counts_equal"] = new["counts"] == tab["counts"]
    dk = 0.0
    for n in tab["n_list"]:
        a = np.array(tab["kappa_by_n"][str(n)], float); b = np.array(new["kappa_by_n"][str(n)], float)
        both = np.isfinite(a) & np.isfinite(b)
        assert (np.isfinite(a) == np.isfinite(b)).all()
        dk = max(dk, float(np.abs(a[both] - b[both]).max()))
    res["rerun_max_abs_diff_kappa_by_n"] = dk
    # 3. independent re-implementation, n = 150
    sidx = C.nfe_sample_index(); N = len(sidx)
    g = seeded_rng(VALID_SEED, "p2_review_famA", "layerE")
    n, draws, nwin = 150, 200, int(os.environ.get("N_WIN", 12))
    nb, nm = len(C.R2_EDGES) - 1, len(C.MAF_EDGES) - 1
    acc0 = np.zeros((nb, nm)); acc1 = np.zeros((nb, nm)); cnt = np.zeros((nb, nm)); mono_frac = np.zeros((nb, nm))
    for w in tab["windows"][:nwin]:
        t = chrom_variants(int(w["chrom"]))
        sub = t[(t.POS >= w["start_pos"]) & (t.POS <= w["end_pos"])]
        G = read_genotypes(int(w["chrom"]), sub.vidx.to_numpy(), sidx)
        f = G.mean(0) / 2; maf = np.minimum(f, 1 - f)
        keep = np.nonzero(maf >= 0.01)[0][: int(tab["p"])]
        G, maf = G[:, keep], maf[keep]
        p = G.shape[1]
        Xc = G - G.mean(0); Xs = Xc / Xc.std(0)
        R = Xs.T @ Xs / N
        iu = np.triu_indices(p, 1)
        r2 = R[iu] ** 2; mm = np.minimum(maf[iu[0]], maf[iu[1]])
        bi = np.clip(np.searchsorted(C.R2_EDGES, r2, side="right") - 1, 0, nb - 1)
        mi = np.clip(np.searchsorted(C.MAF_EDGES, mm, side="right") - 1, 0, nm - 1)
        flat = bi * nm + mi
        s0 = np.zeros(len(r2)); q0 = np.zeros(len(r2))       # r̂ := 0 when monomorphic
        s1 = np.zeros(len(r2)); q1 = np.zeros(len(r2)); k1 = np.zeros(len(r2))   # monomorphic draws excluded
        for _ in range(draws):
            rows = g.choice(N, size=n, replace=False)
            X = G[rows]; Xc = X - X.mean(0); sd = Xc.std(0)
            poly = sd > 0
            Xs = Xc / np.where(poly, sd, 1.0)
            Rh = (Xs.T @ Xs / n)[iu]
            ok = poly[iu[0]] & poly[iu[1]]
            Rh0 = np.where(ok, Rh, 0.0)
            s0 += Rh0; q0 += Rh0 * Rh0
            s1 += np.where(ok, Rh, 0.0); q1 += np.where(ok, Rh * Rh, 0.0); k1 += ok
        v0 = (q0 - s0 * s0 / draws) / (draws - 1)
        k1c = np.maximum(k1, 2)
        v1 = (q1 - s1 * s1 / k1c) / (k1c - 1)
        v1 = np.where(k1 >= 2, v1, np.nan)
        fpc = (N - 1) / (N - n)
        good = np.isfinite(v1)
        cnt += np.bincount(flat[good], minlength=nb * nm).reshape(nb, nm)
        acc0 += np.bincount(flat[good], weights=n * fpc * v0[good], minlength=nb * nm).reshape(nb, nm)
        acc1 += np.bincount(flat[good], weights=n * fpc * v1[good], minlength=nb * nm).reshape(nb, nm)
        mono_frac += np.bincount(flat[good], weights=1 - k1[good] / draws, minlength=nb * nm).reshape(nb, nm)
        print("window", w["block"], p, flush=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        res["indep_n150"] = dict(r2_edges=C.R2_EDGES, maf_edges=C.MAF_EDGES, n_windows=nwin, draws=draws,
                                 counts=cnt.astype(int).tolist(),
                                 kappa_mono_as_zero=np.round(acc0 / cnt, 4).tolist(),
                                 kappa_mono_excluded=np.round(acc1 / cnt, 4).tolist(),
                                 mean_frac_draws_monomorphic=np.round(mono_frac / cnt, 4).tolist(),
                                 committed_kappa_n150=np.round(np.array(tab["kappa_by_n"]["150"], float), 4).tolist())
    (out / "layerE_check.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
