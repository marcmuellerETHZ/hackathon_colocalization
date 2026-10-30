"""Run the actual coloc_susie_mm baseline wrapper (src/baselines/methods.py:ColocSusieMM, one R worker per process)
on a list of loci; writes one parquet row per locus (s, P_S, P_D, P_W, status, extra). Used to check that the
identity-test R runner (run_susie016_mm.R) reproduces the wrapper exactly (status, CS-pair PPs, fallback)."""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import pandas as pd


def _work(args):
    f, ids = args
    from baselines.methods import ColocSusieMM
    from eval.io import LocusSetReader
    m = ColocSusieMM()
    out = []
    with LocusSetReader(f) as rd:
        for i in ids:
            out.append(m.fit_one(rd.read(i)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-dir", default="data/processed/sets/valid_grid_v1")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    ids = Path(a.ids).read_text().split()
    tasks = []
    for i in ids:
        cell = i.split("__")[1]
        tasks.append((str(Path(a.set_dir) / f"{cell}.h5"), [i]))
    with mp.get_context("spawn").Pool(a.workers) as pool:
        rows = [r for chunk in pool.imap_unordered(_work, tasks) for r in chunk]
    pd.DataFrame(rows).to_parquet(a.out)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
