"""Identity test A: JointSuSiE (eta = 0, rho = 0, fixed prior) vs mvsusieR 0.3.0 mvsusie_rss with the same prior.

Usage: python -m famA.validation.compare_mvsusie --work DIR --rdir DIR/mvs --out OUTDIR [--dtype float64]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from famA.joint_susie import JointConfig, joint_susie_batch
from famG.validation.make_loci import LocusStore

U1, U2 = 25.0, 25.0
PI = (0.0, 0.375, 0.375, 0.25)


def load_r(rdir: Path):
    out = {}
    for jf in sorted(rdir.glob("c*.json")):
        js = json.loads(jf.read_text())
        bf = np.fromfile(jf.with_suffix(".bin"), dtype=np.float64)
        off = 0
        for lid, rec in js.items():
            p = int(rec["p"])
            if "error" in rec:
                off += p + 10 * p + 10 * p * 4
                out[lid] = rec
                continue
            L, K1 = int(rec["L"]), int(rec["K1"])
            rec = dict(rec)
            rec["pip"] = bf[off:off + p]; off += p
            rec["alpha"] = bf[off:off + L * p].reshape(L, p); off += L * p
            rec["pmw"] = bf[off:off + L * p * K1].reshape(L, p, K1); off += L * p * K1
            out[lid] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--rdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--pi", default=",".join(map(str, PI)))
    a = ap.parse_args()
    pi = tuple(float(x) for x in a.pi.split(","))
    store = LocusStore(Path(a.work))
    rres = load_r(Path(a.rdir))
    ids = [i for i in range(len(store)) if store.row(i).locus_id in rres]
    cfg = JointConfig(prior="fixed", U_fixed=((U1, 0, 0), (0, 0, U2), (U1, 0, U2)), pi=pi, pi_mode="fixed", convergence="susie016", eta_floor=0.0, multistart=False,
                      tol=1e-4, z_adjust=True, estimate_eta=False, ld_dtype=getattr(torch, a.dtype))
    z1 = [store.z(i, 1) for i in ids]; z2 = [store.z(i, 2) for i in ids]
    lds = [store.R(i).astype(np.float64 if a.dtype == "float64" else np.float32) for i in ids]
    n = [float(store.row(i).n) for i in ids]
    torch.cuda.synchronize(); t0 = time.time()
    res = joint_susie_batch(z1, z2, lds, n, n, [0.0] * len(ids), "dense", cfg)
    torch.cuda.synchronize(); t_gpu = time.time() - t0
    rows = []
    for k, i in enumerate(ids):
        lid = store.row(i).locus_id
        q, g = rres[lid], res[k]
        if "error" in q:
            rows.append(dict(locus_id=lid, status="R_error", err=q["error"]))
            continue
        st_r = (q["alpha"][:, :, None] * q["pmw"]).sum(1)          # (L, K[+1]) state probabilities
        if st_r.shape[1] == 3:                                      # mvsusie drops the null column if null_weight = 0
            st_r = np.concatenate([np.zeros((st_r.shape[0], 1)), st_r], 1)
        rows.append(dict(locus_id=lid, p=int(store.row(i).p), scenario=store.row(i).scenario, status="ok",
                         max_dpip=float(np.abs(g["pip"] - q["pip"]).max()),
                         max_dalpha=float(np.abs(g["alpha"] - q["alpha"]).max()),
                         max_dstate=float(np.abs(g["state_prob"] - st_r).max()),
                         niter_gpu=g["niter"], niter_r=int(q["niter"]), conv_gpu=g["converged"], conv_r=bool(q["converged"]),
                         r_time=float(q["time"])))
    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    summ = dict(n=len(df), n_compared=len(ok), max_dpip=float(ok.max_dpip.max()), max_dalpha=float(ok.max_dalpha.max()),
                max_dstate=float(ok.max_dstate.max()), median_dpip=float(ok.max_dpip.median()),
                n_dpip_ge_1em2=int((ok.max_dpip >= 1e-2).sum()), n_dstate_ge_1em2=int((ok.max_dstate >= 1e-2).sum()),
                niter_equal_frac=float((ok.niter_gpu == ok.niter_r).mean()),
                converged_equal_frac=float((ok.conv_gpu == ok.conv_r).mean()),
                gpu_s=t_gpu, r_s_single_thread=float(ok.r_time.sum()), dtype=a.dtype, pi=pi)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"mvsusie_per_locus_{a.dtype}.tsv", sep="\t", index=False)
    (out / f"mvsusie_summary_{a.dtype}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
