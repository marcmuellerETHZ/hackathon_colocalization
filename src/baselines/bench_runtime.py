"""Per-locus runtime and peak memory of each baseline wrapper at p ~ 1000, 3000, 5000 (M3 pilot, not the frozen
timing protocol of DESIGN §7.7, which needs the exclusive machine lock and 3 repeats).

Usage (env active, PYTHONPATH=src):
    python -m baselines.bench_runtime --out logs/baselines/runtime_M3.jsonl [--methods a,b] [--sizes 1000,3000,5000]
Each (method, p) runs in its own subprocess (fresh R worker, so VmHWM is that cell's peak), on three toy loci
(shared, distinct, null trait 2; lambda = 15, N1 = 100k, N2 = 1000) from chr22 NFE windows with MAF >= 0.01,
LD in factor form (n_ref = 404), so runtime includes building the dense LD.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]


def _loci(p: int, mismatch: bool = False):
    """mismatch=False: z and LD from the same 404 NFE individuals (in-sample).
    mismatch=True : z generated from the first 254 NFE individuals, LD from the other 150 (an L2-like
    small external reference; SNPs monomorphic in either half are dropped, so p is slightly below the target)."""
    from common.seeds import VALID_SEED, rng

    from .toy import load_window, make_toy_locus, pick_pair

    X, maf, ids, pos = load_window(p, start_bp=25_000_000, maf_min=0.01)
    g = rng(VALID_SEED, "baselines/bench_runtime", p, "mm") if mismatch else rng(VALID_SEED, "baselines/bench_runtime", p)
    kw = dict(N1=100_000, N2=1_000, ld_type="factor")
    tag = "mm" if mismatch else "in"
    if mismatch:
        perm = g.permutation(X.shape[0])
        XA, XB = X[perm[:254]], X[perm[254:]]
        keep = (XA.std(0) > 1e-6) & (XB.std(0) > 1e-6)  # drop SNPs monomorphic in either half
        XA, XB, maf, ids, pos = XA[:, keep], XB[:, keep], maf[keep], ids[keep], pos[keep]
        XA = (XA - XA.mean(0)) / XA.std(0)
        XB = (XB - XB.mean(0)) / XB.std(0)
    else:
        XA = XB = X
    R = XA.T @ XA / XA.shape[0]
    a, b = pick_pair(R, g, r2_max=0.1, margin=50)
    out = []
    for kind, l1, l2 in (("shared", {a: 15.0}, {a: 15.0}), ("distinct", {a: 15.0}, {b: 15.0}),
                         ("null2", {a: 15.0}, {})):
        L = make_toy_locus(XA, maf, ids, pos, l1, l2, g, locus_id=f"p{p}_{tag}_{kind}", **kw)
        if mismatch:
            object.__setattr__(L, "X_ref", XB.astype(np.float32))
            object.__setattr__(L, "n_ref", XB.shape[0])
            object.__setattr__(L, "mismatch_level", "L2-like")
        out.append(L)
    return out


def run_cell(method: str, p: int, mismatch: bool = False) -> dict:
    from .methods import get_method
    from .rworker import get_worker, shutdown_all

    m = get_method(method)
    m.timeout = 3600.0
    loci = _loci(p, mismatch)
    startup = None
    if m.r_lib is not None:
        t0 = time.perf_counter()
        get_worker(m.r_lib).call("info")
        startup = time.perf_counter() - t0
    rows = []
    for L in loci:
        t0 = time.perf_counter()
        from .locus import dense_ld

        dense_ld(L)
        t_ld = time.perf_counter() - t0
        object.__setattr__(L, "_baselines_R64", None)  # the method rebuilds it inside its timed call
        r = m.fit_one(L)
        rows.append(dict(locus_id=L.locus_id, status=r["status"], runtime_s=r["runtime_s"], ld_build_s=t_ld,
                         s=r["s"], P_D=r["P_D"], P_W=r["P_W"]))
    r_peak = get_worker(m.r_lib).peak_rss_kb() if m.r_lib else 0
    shutdown_all()
    return dict(method=method, p=p, mismatch=mismatch, rows=rows, r_startup_s=startup, r_peak_rss_mb=r_peak / 1024,
                py_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT / "logs" / "baselines" / "runtime_M3.jsonl"))
    ap.add_argument("--methods", default="")
    ap.add_argument("--sizes", default="1000,3000,5000")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--cell", default="", help="internal: method:p[:mm]")
    ap.add_argument("--mismatch", action="store_true", help="L2-like LD (z from 254, LD from 150 individuals)")
    a = ap.parse_args()
    if a.cell:
        parts = a.cell.split(":")
        print("@@CELL " + json.dumps(run_cell(parts[0], int(parts[1]), len(parts) > 2)), flush=True)
        return
    from .methods import all_methods

    methods = a.methods.split(",") if a.methods else list(all_methods())
    sizes = [int(x) for x in a.sizes.split(",")]
    cells = [(m, p) for p in sizes for m in methods]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def go(cell):
        m, p = cell
        cmd = [sys.executable, "-m", "baselines.bench_runtime", "--cell", f"{m}:{p}" + (":mm" if a.mismatch else "")]
        r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ))
        line = [x for x in r.stdout.splitlines() if x.startswith("@@CELL ")]
        if r.returncode != 0 or not line:
            res = dict(method=m, p=p, error=r.stderr[-2000:], returncode=r.returncode)
        else:
            res = json.loads(line[0][7:])
        with open(out, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(json.dumps(res)[:400], flush=True)
        return res

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(go, cells))


if __name__ == "__main__":
    main()
