"""DESIGN §7.7 throughput runs for G-mm vs the CPU coloc-SuSiE baselines on a fixed M4 slice.

Each repeat is one harness run (eval.harness.run_method, timing=True: exclusive results/.timing.lock + machine lock;
LD read from disk per locus by the harness workers, i.e. LD I/O included; GPU work synchronised by the harness
before a batch is reported). Whole-machine throughput = n_loci / harness wall time.

  slice:  python -m famG.validation.timing_mm slice --out results/famG_mm/slice2000_v1 --n 2000
  timing: python -m famG.validation.timing_mm run --ids results/famG_mm/slice2000_v1/ids.txt \
              --plan gmm:3,mm:3,cs:1,cs:2 --out results/famG_mm/timing_v1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from common.seeds import rng as seeded_rng

LOCUS_SET = "data/processed/sets/valid_grid_v1.h5"
M4_MAIN = "results/baselines_m4/m4_subsample_v1/main.txt"

METHODS = {
    # name: (MethodSpec target, kwargs, mode, n_workers, batch_size, timeout_s)
    # CPU baselines: memory-limited worker count per DESIGN §7.7 (<= 2 GB/core -> 128 GB; measured peak per worker
    # R 2.7 GB + python 0.5 GB -> floor(128 / 3.2) = 40 workers); wrapper default per-locus timeouts.
    "gmm": ("famG.method_mm:GPUColocSusieMM", {"ld_dtype": "float64", "mem_budget_gb": 5.0}, "batch", 3, 256, 600.0),
    "mm": ("baselines.methods:ColocSusieMM", {}, "cpu", 40, 1, 900.0),
    "cs": ("baselines.methods:ColocSusie", {}, "cpu", 40, 1, 600.0),
}


def make_slice(out: Path, n: int):
    ids = Path(M4_MAIN).read_text().split()
    r = seeded_rng(20261029, "famG_mm", "slice", n)
    sel = sorted(ids[k] for k in r.choice(len(ids), size=n, replace=False))
    out.mkdir(parents=True, exist_ok=True)
    (out / "ids.txt").write_text("\n".join(sel) + "\n")
    print(f"{len(sel)} ids -> {out / 'ids.txt'}")


def _read_chunk(ids):
    from eval.io import LocusSetReader
    nb = 0
    with LocusSetReader(LOCUS_SET) as rd:
        for i in ids:
            lc = rd.read(i)
            nb += (lc.R if lc.ld_type == "dense" else lc.X_ref).nbytes
    return nb


def prewarm(ids: list[str], workers: int = 16) -> dict:
    """Read every locus once (untimed) so that all timed repeats see the same warm page cache."""
    import multiprocessing as mp
    t0 = time.time()
    chunks = [ids[k::workers] for k in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        nb = sum(pool.map(_read_chunk, chunks))
    return {"prewarm_s": time.time() - t0, "prewarm_gb": nb / 1e9, "workers": workers}


def run(ids_file: Path, plan: list[tuple[str, int]], out: Path, overrides: dict, do_prewarm: bool = True):
    from eval.harness import MethodSpec, run_method
    ids = ids_file.read_text().split()
    if do_prewarm:
        pw = prewarm(ids)
        print(json.dumps(pw), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "timing_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for m, repeats in plan:
        target, kw, mode, nw, bs, to = METHODS[m]
        ov = overrides.get(m, {})
        nw, bs = int(ov.get("n_workers", nw)), int(ov.get("batch_size", bs))
        kw = {**kw, **ov.get("kwargs", {})}
        for _ in range(repeats):
            rep = len(summary.get(m, []))
            t0 = time.time()
            res = run_method(MethodSpec(target, kw), LOCUS_SET, results_dir=out, ids=ids, mode=mode, n_workers=nw,
                             batch_size=bs, timeout_s=to, timing=True, hash_inputs=False,
                             label=f"famG_mm timing {m} rep {rep}", order="lpt" if mode == "cpu" else "given")
            wall = time.time() - t0
            rd = Path(res.run_dir) if hasattr(res, "run_dir") else None
            man = json.loads((rd / "manifest.json").read_text()) if rd and (rd / "manifest.json").exists() else {}
            hw = man.get("wall_s", wall)
            rec = {"method": m, "target": target, "kwargs": kw, "mode": mode, "n_workers": nw, "batch_size": bs,
                   "rep": rep, "n_loci": len(ids), "harness_wall_s": hw, "driver_wall_s": wall,
                   "loci_per_hour": len(ids) / hw * 3600.0, "counts": man.get("counts"), "run_dir": str(rd),
                   "machine_state": man.get("machine_state")}
            summary.setdefault(m, []).append(rec)
            summary_path.write_text(json.dumps(summary, indent=1, default=str))
            print(json.dumps(rec, default=str), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("slice")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=2000)
    r = sub.add_parser("run")
    r.add_argument("--ids", required=True)
    r.add_argument("--plan", default="gmm:3,mm:3,cs:1,cs:2",
                   help="ordered method:repeats list (the default runs the PI fallback set first)")
    r.add_argument("--out", required=True)
    r.add_argument("--overrides", default="{}", help="JSON {method: {n_workers, batch_size, kwargs}}")
    r.add_argument("--no-prewarm", action="store_true")
    a = ap.parse_args()
    if a.cmd == "slice":
        make_slice(Path(a.out), a.n)
    else:
        plan = [(x.split(":")[0], int(x.split(":")[1])) for x in a.plan.split(",")]
        run(Path(a.ids), plan, Path(a.out), json.loads(a.overrides), not a.no_prewarm)


if __name__ == "__main__":
    main()
