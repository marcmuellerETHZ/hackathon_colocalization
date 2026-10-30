"""JointSuSiE GPU throughput at fixed p (both traits jointly; LD read from disk included in 'with_io').
Usage: python -m famA.validation.timing --dirs D1,D2 --out FILE.json [--reps 2] [--eta] [--dtype float32]
           [--replicate 1,4] [--eig-dtype float64] [--pi-mode eb] [--no-marginal] [--ref-var]
--replicate k fits k copies of the store's loci in one call (larger GPU batches, as in a genome-scale run)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from famA.joint_susie import JointConfig, joint_susie_batch
from famG.validation.make_loci import LocusStore


def gpu_util():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--eta", action="store_true")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--eig-dtype", default="float64")
    ap.add_argument("--pi-mode", default="eb")
    ap.add_argument("--no-marginal", action="store_true")
    ap.add_argument("--ref-var", action="store_true")
    ap.add_argument("--replicate", default="1")
    a = ap.parse_args()
    cfg = JointConfig(estimate_eta=a.eta, ld_dtype=getattr(torch, a.dtype), eig_dtype=getattr(torch, a.eig_dtype),
                      pi_mode=a.pi_mode, power_marginal=not a.no_marginal, ref_var=a.ref_var,
                      convergence="alpha" if a.ref_var else "susie014")
    res = {"loadavg_start": os.getloadavg(), "gpu_util_start": gpu_util(), "eta": a.eta, "cfg": repr(cfg), "runs": []}
    for d in a.dirs.split(","):
        st = LocusStore(Path(d))
        for k in [int(x) for x in a.replicate.split(",")]:
            for rep in range(a.reps):
                t0 = time.time()
                z1 = [st.z(i, 1) for i in range(len(st))]; z2 = [st.z(i, 2) for i in range(len(st))]
                lds, kinds = zip(*[st.ld_input(i)[::-1] for i in range(len(st))])
                kinds = list(kinds); lds = list(lds)
                n1 = [float(st.row(i).n) for i in range(len(st))]; n2 = [float(st.row(i).n2) for i in range(len(st))]
                nref = [float(st.row(i).n_ref) if kinds[i] == "factor" else 503.0 for i in range(len(st))]
                t_io = time.time() - t0
                z1, z2, lds, kinds, n1, n2, nref = (x * k for x in (z1, z2, lds, kinds, n1, n2, nref))
                if rep == 0:
                    joint_susie_batch(z1[:2], z2[:2], lds[:2], n1[:2], n2[:2], None, kinds[:2], cfg,
                                      ref_sizes=nref[:2])   # warm-up
                torch.cuda.synchronize(); t1 = time.time()
                out = joint_susie_batch(z1, z2, lds, n1, n2, None, kinds, cfg, ref_sizes=nref)
                torch.cuda.synchronize(); t_fit = time.time() - t1
                r = dict(dir=d, p=int(st.row(0).p), ld_kind=kinds[0], n_loci=len(z1), replicate=k, rep=rep,
                         t_io=t_io * k, t_fit=t_fit, loci_per_s_fit=len(z1) / t_fit,
                         loci_per_s_with_io=len(z1) / (t_fit + t_io * k),
                         median_niter=float(np.median([o["niter"] for o in out])),
                         max_niter=int(max(o["niter"] for o in out)), gpu_util=gpu_util(), loadavg=os.getloadavg()[0])
                print(json.dumps(r), flush=True)
                res["runs"].append(r)
    res["loadavg_end"] = os.getloadavg()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
