"""Development benchmark for G-mm (not the §7.7 timing): read N slice loci into memory, then time the GPU fit
(torch.cuda.synchronize) and print a kernel profile. Reports fit-only loci/s, iterations and I/O time separately."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from eval.io import LocusSetReader
from famG.method_mm import GPUColocSusieMM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="results/famG_mm/slice2000_v1/ids.txt")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--mem-gb", type=float, default=5.0)
    ap.add_argument("--sweep", default="", help="comma list BLOCK:NUM_WARPS:bucket_ratio:streams, e.g. 1024:8:1.5:1,1024:8:1.5:4")
    a = ap.parse_args()
    allids = Path(a.ids).read_text().split()
    ids = allids[:: max(1, len(allids) // a.n)][: a.n]          # evenly strided over the (cell-sorted) slice
    t0 = time.time()
    with LocusSetReader("data/processed/sets/valid_grid_v1.h5") as rd:
        loci = [rd.read(i) for i in ids]
    t_io = time.time() - t0
    import famG.mm_kernels as K
    configs = [tuple(c.split(":")) for c in a.sweep.split(",")] if a.sweep else [(None, None, None, "1")]
    allres = []
    for blk, nw, br, nst in configs:
        if blk:
            K.BLOCK, K.NUM_WARPS = int(blk), int(nw)
        m = GPUColocSusieMM(ld_dtype=a.dtype, device=torch.device("cuda"), max_batch=a.batch, mem_budget_gb=a.mem_gb,
                            streams=int(nst))
        if br:
            m.cfg.bucket_ratio = float(br)
        m.fit_batch(loci[: min(8, len(loci))])          # warm-up (Triton compile)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        res = {"n": len(loci), "io_s": t_io, "block": K.BLOCK, "num_warps": K.NUM_WARPS,
               "bucket_ratio": m.cfg.bucket_ratio, "streams": int(nst), "dtype": a.dtype, "mem_budget_gb": a.mem_gb}
        torch.cuda.synchronize()
        t0 = time.time()
        rows = []
        for k in range(0, len(loci), a.batch):
            rows += m.fit_batch(loci[k:k + a.batch])
        torch.cuda.synchronize()
        res["fit_s"] = time.time() - t0
        res["fit_loci_per_s"] = len(loci) / res["fit_s"]
        st = [r["status"] for r in rows]
        res["status"] = {s_: st.count(s_) for s_ in set(st)}
        nit = [max(json.loads(r["extra"])["niter"]) for r in rows]
        res["niter_max_mean"] = float(np.mean(nit))
        res["max_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        print(json.dumps(res), flush=True)
        allres.append(res)
    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            m.fit_batch(loci[: a.batch])
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    if a.out:
        Path(a.out).write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
