"""GPU SuSiE-RSS throughput at fixed p (DESIGN §7.7 style: torch.cuda.synchronize, LD read from disk included
in the 'with_io' number). Usage:
  python -m famG.validation.timing --dirs D1,D2,... --out FILE.json [--reps 3] [--replicate 1,4] [--dtype float32]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from famG.susie_rss import SusieRSSConfig, susie_rss_batch
from famG.validation.make_loci import LocusStore


def load_inputs(store: LocusStore, dtype=np.float32):
    zs, lds, ns, kinds = [], [], [], []
    for i in range(len(store)):
        kind, ld = store.ld_input(i, dense_dtype=dtype)
        zs.append(store.z(i, 1)); lds.append(ld); kinds.append(kind); ns.append(float(store.row(i).n))
    return zs, lds, ns, kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--replicate", default="1")
    ap.add_argument("--dtype", default="float32")
    a = ap.parse_args()
    cfg = SusieRSSConfig(ld_dtype=getattr(torch, a.dtype))
    from famA.validation.timing import gpu_util
    res = {"loadavg_start": os.getloadavg(), "gpu": torch.cuda.get_device_name(0), "dtype": a.dtype,
           "gpu_util_start": gpu_util(), "runs": []}
    # warm-up (Triton compile, cuBLAS init)
    st0 = LocusStore(Path(a.dirs.split(",")[0]))
    z, l, n, k = load_inputs(st0)
    susie_rss_batch(z[:4], l[:4], n[:4], k[:4], cfg)
    for d in a.dirs.split(","):
        st = LocusStore(Path(d))
        p = int(st.row(0).p)
        for rep_factor in [int(x) for x in a.replicate.split(",")]:
            for rep in range(a.reps):
                torch.cuda.synchronize()
                t0 = time.time()
                z, l, n, k = load_inputs(st)            # LD read from disk (page cache after first rep)
                t_io = time.time() - t0
                z, l, n, k = z * rep_factor, l * rep_factor, n * rep_factor, k * rep_factor
                torch.cuda.synchronize()
                t1 = time.time()
                out = susie_rss_batch(z, l, n, k, cfg)
                torch.cuda.synchronize()
                t_fit = time.time() - t1
                nl = len(z)
                niter = [o["niter"] for o in out]
                r = dict(dir=d, p=p, ld_kind=k[0], n_loci=nl, rep=rep, replicate=rep_factor, t_io=t_io, t_fit=t_fit,
                         loci_per_s_fit=nl / t_fit, loci_per_s_with_io=nl / (t_fit + t_io * rep_factor),
                         median_niter=float(np.median(niter)), max_niter=int(max(niter)),
                         peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9, gpu_util=gpu_util())
                torch.cuda.reset_peak_memory_stats()
                print(json.dumps(r), flush=True)
                res["runs"].append(r)
    res["loadavg_end"] = os.getloadavg()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
