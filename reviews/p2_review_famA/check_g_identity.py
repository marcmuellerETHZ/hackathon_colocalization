"""Independent reviewer check of the G (famG.susie_rss) vs susieR 0.14.2 identity claim (task p2.review_famA).

12 screen_v2 VALIDATION loci (6 from L0_P1 dense in-sample LD, 6 from L2_P1 factor LD with n_ref = 150 < p),
both traits -> 24 fits of susie_rss(z, R, n = median N, L = 10). G is run with float64 LD (the identity setting)
and with the default float32 LD. The R reference is my own script (reviews/p2_review_famA/run_susieR.R), fed the
same z and R = X'X / n_ref (float64) or the stored dense R.
Writes $ORCH_OUT/g_identity.tsv and g_identity_summary.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.seeds import VALID_SEED, rng as seeded_rng
from eval.io import LocusSetReader
from famG.susie_rss import SusieRSSConfig, susie_rss_batch

SETDIR = Path("data/processed/sets/screen_v2")
CELLS = [("L0_P1_r_p=0.0", 6), ("L2_P1_r_p=0.0", 6)]


def main():
    out = Path(os.environ.get("ORCH_OUT", "."))
    wd = out / "g"; wd.mkdir(parents=True, exist_ok=True)
    fits = []
    for cell, k in CELLS:
        path = SETDIR / f"{cell}.h5"
        with LocusSetReader(path) as r:
            ids = r.ids()
            g = seeded_rng(VALID_SEED, "p2_review_famA", "G", cell)
            sel = sorted(ids[i] for i in g.permutation(len(ids))[:k])
            for lid in sel:
                lc = r.read(lid)
                assert lc.split == "valid"
                if lc.ld_type == "factor":
                    X = np.asarray(lc.X_ref, np.float64)
                    R = X.T @ X / float(lc.n_ref)
                    ld, kind = lc.X_ref, "factor"
                else:
                    R = np.asarray(lc.R, np.float64)
                    ld, kind = lc.R, "dense"
                for t, z, n in ((1, lc.z1, lc.n1), (2, lc.z2, lc.n2)):
                    fits.append(dict(fit_id=f"{lid}|t{t}", z=np.asarray(z, float), R=R, ld=ld, kind=kind,
                                     n=float(np.median(n)), n_ref=int(lc.n_ref)))
    # write inputs for R
    with open(wd / "R.bin", "wb") as fR, open(wd / "z.bin", "wb") as fz:
        offR = offz = 0
        idx = []
        for f in fits:
            p = len(f["z"])
            np.asarray(f["R"], np.float64).tofile(fR); np.asarray(f["z"], np.float64).tofile(fz)
            idx.append(dict(fit_id=f["fit_id"], p=p, n=f["n"], R_offset=offR, z_offset=offz))
            offR += p * p; offz += p
    pd.DataFrame(idx).to_csv(wd / "index.tsv", sep="\t", index=False)
    t0 = time.time()
    rs = subprocess.run([os.environ.get("RSCRIPT", "Rscript"), str(Path(__file__).with_name("run_susieR.R")), str(wd)],
                        capture_output=True, text=True)
    print(rs.stdout[-3000:], rs.stderr[-3000:], flush=True)
    if rs.returncode:
        raise SystemExit("R failed")
    r_time = time.time() - t0
    rj = json.loads((wd / "r_out.json").read_text())
    rbin = np.fromfile(wd / "r_out.bin", dtype=np.float64)
    # G fits
    gres = {}
    for tag, dt in (("f64", torch.float64), ("f32", torch.float32)):
        cfg = SusieRSSConfig(ld_dtype=dt, mem_budget_bytes=2.5e9)
        t1 = time.time()
        res = [None] * len(fits)
        for kind in ("dense", "factor"):
            ii = [i for i, f in enumerate(fits) if f["kind"] == kind]
            rr = susie_rss_batch([fits[i]["z"] for i in ii], [fits[i]["ld"] for i in ii], [fits[i]["n"] for i in ii],
                                 kind, cfg, n_refs=[fits[i]["n_ref"] for i in ii] if kind == "factor" else None)
            for i, x in zip(ii, rr):
                res[i] = x
        gres[tag] = (res, time.time() - t1)
    rows = []
    off = 0
    for i, f in enumerate(fits):
        p = len(f["z"]); Rr = rj[f["fit_id"]]
        L = int(Rr["L"]) if "L" in Rr else 10
        pip_r = rbin[off:off + p]; alpha_r = rbin[off + p: off + p + L * p].reshape(L, p); off += p + L * p
        row = dict(fit_id=f["fit_id"], kind=f["kind"], p=p, n=f["n"], r_error=Rr.get("error"), niter_r=Rr.get("niter"),
                   conv_r=Rr.get("converged"), n_cs_r=len(Rr.get("cs") or []))
        cs_r = sorted(tuple(sorted(c if isinstance(c, list) else [c])) for c in (Rr.get("cs") or []))
        for tag in ("f64", "f32"):
            g = gres[tag][0][i]
            row[f"max_dpip_{tag}"] = float(np.abs(g["pip"] - pip_r).max())
            row[f"max_dalpha_{tag}"] = float(np.abs(g["alpha"] - alpha_r).max()) if g["alpha"].shape == alpha_r.shape else np.nan
            row[f"max_dV_rel_{tag}"] = float((np.abs(g["V"] - np.asarray(Rr["V"])) / np.maximum(np.abs(np.asarray(Rr["V"])), 1e-12)).max())
            row[f"niter_{tag}"] = g["niter"]; row[f"conv_{tag}"] = g["converged"]; row[f"status_{tag}"] = g["status"]
            cs_g = sorted(tuple(sorted(int(x) for x in c)) for c in (g["sets"].get("cs") or []))
            row[f"cs_equal_{tag}"] = cs_g == cs_r
            row[f"n_cs_{tag}"] = len(cs_g)
            row[f"delbo_{tag}"] = float(g["elbo"][-1] - np.asarray(Rr["elbo"])[-1])
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "g_identity.tsv", sep="\t", index=False)
    s = dict(n_fits=len(df), n_loci=len(df) // 2, r_errors=int(df.r_error.notna().sum()), r_time_s=r_time,
             g_time_s_f64=gres["f64"][1], g_time_s_f32=gres["f32"][1])
    for tag in ("f64", "f32"):
        s[f"max_dpip_{tag}"] = float(df[f"max_dpip_{tag}"].max())
        s[f"n_dpip_ge_1e-3_{tag}"] = int((df[f"max_dpip_{tag}"] >= 1e-3).sum())
        s[f"max_dalpha_{tag}"] = float(df[f"max_dalpha_{tag}"].max())
        s[f"niter_equal_{tag}"] = int((df[f"niter_{tag}"] == df.niter_r).sum())
        s[f"conv_equal_{tag}"] = int((df[f"conv_{tag}"] == df.conv_r).sum())
        s[f"cs_equal_{tag}"] = int(df[f"cs_equal_{tag}"].sum())
        for kind in ("dense", "factor"):
            s[f"max_dpip_{tag}_{kind}"] = float(df.loc[df.kind == kind, f"max_dpip_{tag}"].max())
    (out / "g_identity_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    sys.exit(main())
