"""G-mm vs susieR 0.16.6 (run_susie016_mm.R) identity comparison: PIP, alpha, V, lambda, iterations, convergence,
failures, credible sets and coloc.bf_bf PP.H4 per CS pair (same pairing code as GPUColocSusieMM)."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from famG.coloc import bf_bf_batch
from famG.susie_mm import MMConfig, susie_mm_batch
from famG.validation.mm_loci import load_exported


def _arr(x):
    return np.asarray(x if isinstance(x, list) else [x], dtype=float)


def _cs_lists(r):
    cs = r.get("cs") or []
    if cs and not isinstance(cs[0], list):
        cs = [cs] if len(r.get("cs_index") if isinstance(r.get("cs_index"), list) else [r.get("cs_index")]) == 1 else cs
    return [sorted(int(v) - 1 for v in (c if isinstance(c, list) else [c])) for c in cs]


def run(work: Path, rdir: Path, ids: list[str], dtype: torch.dtype, device, use_triton: bool, mem_gb: float = 5.0):
    loci = load_exported(work, ids)
    cfg = MMConfig(ld_dtype=dtype, use_triton=use_triton, mem_budget_bytes=mem_gb * 1e9)
    t0 = time.time()
    fits = susie_mm_batch(loci, cfg, device=device)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    el = time.time() - t0
    bf1 = [f[0]["lbf_variable"][f[0]["sets"]["cs_index"]] if f[0]["sets"]["cs_index"] and f[0]["status"] == "ok"
           else np.zeros((0, len(f[0]["pip"]))) for f in fits]
    bf2 = [f[1]["lbf_variable"][f[1]["sets"]["cs_index"]] if f[1]["sets"]["cs_index"] and f[1]["status"] == "ok"
           else np.zeros((0, len(f[1]["pip"]))) for f in fits]
    pps = bf_bf_batch(bf1, bf2, 1e-4, 1e-4, 5e-6, device=device)
    rows, fit_rows = [], []
    for k, lid in enumerate(ids):
        R = json.load(open(rdir / f"{lid}.json"))
        for t in (1, 2):
            g, r = fits[k][t - 1], R[f"t{t}"]
            r_fail = r.get("error") is not None
            g_fail = g["status"] != "ok"
            row = dict(locus_id=lid, trait=t, p=len(g["pip"]), r_failed=r_fail, g_failed=g_fail,
                       r_error=r.get("error"))
            if not r_fail and not g_fail:
                pip_r = _arr(r["pip"])
                A_r = np.asarray(r["alpha"], float).reshape(-1, len(pip_r)) if np.ndim(r["alpha"]) else None
                cs_r = _cs_lists(r)
                cs_g = [sorted(c) for c in g["sets"]["cs"]]
                row.update(
                    max_dpip=float(np.abs(g["pip"] - pip_r).max()),
                    max_dalpha=float(np.abs(g["alpha"] - A_r).max()),
                    max_dV_rel=float(np.max(np.abs(g["V"] - _arr(r["V"])) / np.maximum(1e-12, np.abs(_arr(r["V"]))))),
                    niter_r=int(r["niter"]), niter_g=g["niter"], conv_r=bool(r["converged"]), conv_g=g["converged"],
                    lam_r=float(r["lambda_bias"]), lam_g=g["lambda_bias"],
                    lam0_r=float(r["lambda_init"]) if r.get("lambda_init") is not None else math.nan,
                    lam0_g=g["lambda_init"],
                    coh_r=float(r["ld_coherence"]) if r.get("ld_coherence") is not None else math.nan,
                    coh_g=g["ld_coherence"],
                    ncs_r=len(cs_r), ncs_g=len(cs_g),
                    cs_equal=sorted(map(tuple, cs_r)) == sorted(map(tuple, cs_g)),
                    reason_r=r.get("convergence_reason"))
            fit_rows.append(row)
        # coloc pairs: max PP.H4 over pairs and pairwise |dPP| (pairs matched by CS effect index)
        pr = R.get("pairs")
        pg = pps[k]
        lrow = dict(locus_id=lid, r_pairs=pr is not None, g_pairs=pg.size > 0)
        if pr is not None and pg.size > 0:
            PPr = np.asarray(pr["PP"], float).reshape(-1, 5)
            i1 = _arr(pr["idx1"]).astype(int)
            i2 = _arr(pr["idx2"]).astype(int)
            c1 = fits[k][0]["sets"]["cs_index"]
            c2 = fits[k][1]["sets"]["cs_index"]
            d = []
            for q in range(len(PPr)):
                l1 = int(R["t1"]["cs_index"][i1[q] - 1] if isinstance(R["t1"]["cs_index"], list) else R["t1"]["cs_index"]) - 1
                l2 = int(R["t2"]["cs_index"][i2[q] - 1] if isinstance(R["t2"]["cs_index"], list) else R["t2"]["cs_index"]) - 1
                if l1 in c1 and l2 in c2:
                    d.append(np.abs(pg[c1.index(l1), c2.index(l2)] - PPr[q]).max())
                else:
                    d.append(math.inf)
            lrow.update(max_dPP_pair=float(max(d)) if d else math.nan,
                        maxH4_r=float(PPr[:, 4].max()), maxH4_g=float(pg.reshape(-1, 5)[:, 4].max()),
                        dmaxH4=abs(float(PPr[:, 4].max()) - float(pg.reshape(-1, 5)[:, 4].max())))
        rows.append(lrow)
    return pd.DataFrame(fit_rows), pd.DataFrame(rows), el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--rdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-triton", action="store_true")
    ap.add_argument("--mem-gb", type=float, default=5.0)
    a = ap.parse_args()
    work, rdir, out = Path(a.work), Path(a.rdir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = [l for l in (work / "ids.txt").read_text().split() if (rdir / f"{l}.json").exists()]
    fr, lr, el = run(work, rdir, ids, getattr(torch, a.dtype), torch.device(a.device), not a.no_triton, a.mem_gb)
    tag = f"{a.dtype}_{a.device}{'_notriton' if a.no_triton else ''}"
    fr.to_csv(out / f"fits_{tag}.tsv", sep="\t", index=False)
    lr.to_csv(out / f"pairs_{tag}.tsv", sep="\t", index=False)
    both = fr[~fr.r_failed & ~fr.g_failed]
    summ = dict(
        n_loci=len(ids), n_fits=len(fr), dtype=a.dtype, device=a.device, triton=not a.no_triton, g_seconds=el,
        fail_r=int(fr.r_failed.sum()), fail_g=int(fr.g_failed.sum()),
        fail_agree=int((fr.r_failed == fr.g_failed).sum()),
        max_dpip=float(both.max_dpip.max()) if len(both) else None,
        median_dpip=float(both.max_dpip.median()) if len(both) else None,
        n_dpip_ge_1e3=int((both.max_dpip >= 1e-3).sum()) if len(both) else None,
        max_dalpha=float(both.max_dalpha.max()) if len(both) else None,
        conv_agree=int((both.conv_r == both.conv_g).sum()), n_both_ok=len(both),
        nonconv_r=int((~both.conv_r).sum()), nonconv_g=int((~both.conv_g).sum()),
        niter_agree=int((both.niter_r == both.niter_g).sum()),
        max_dlambda=float((both.lam_r - both.lam_g).abs().max()) if len(both) else None,
        cs_equal=int(both.cs_equal.sum()),
        pairs_presence_agree=int((lr.r_pairs == lr.g_pairs).sum()),
        max_dPP_pair=float(lr.max_dPP_pair.max()) if "max_dPP_pair" in lr else None,
        max_dmaxH4=float(lr.dmaxH4.max()) if "dmaxH4" in lr else None,
    )
    json.dump(summ, open(out / f"summary_{tag}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
