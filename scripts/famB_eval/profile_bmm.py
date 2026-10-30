"""Stage timing of B-mm on a few validation loci (runtime pilot only; no accuracy is computed).

    PYTHONPATH=src python scripts/famB_eval/profile_bmm.py --set data/processed/sets/valid_grid_v1/L2_P1_r_p=0.0.h5 --n 48
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

import famB.method as FM  # noqa: E402
import famB.mismatch as MM  # noqa: E402
from eval.io import iter_loci, list_loci  # noqa: E402

T = defaultdict(float)


def timed(name, f):
    def g(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = f(*a, **k)
        torch.cuda.synchronize()
        T[name] += time.perf_counter() - t
        return r
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--mm-iter", type=int, default=2)
    a = ap.parse_args()
    ids = list_loci(a.set)
    ids = ids[:: max(1, len(ids) // a.n)][: a.n]
    loci = list(iter_loci(a.set, ids))
    FM.eb_select = timed("eb_select_phase1", FM.eb_select)
    MM.second_moments = timed("second_moments", MM.second_moments)
    MM.logz_and_moments = timed("logz_and_moments", MM.logz_and_moments)
    MM.eb_select_mm = timed("eb_select_mm", MM.eb_select_mm)
    MM.phi_b = timed("phi_b", MM.phi_b)
    MM.enumerate_batch = timed("enum_inside_eb_mm", MM.enumerate_batch)
    FM.enumerate_batch = timed("enumerate_iter_and_final", FM.enumerate_batch)
    FM.event_posteriors = timed("events", FM.event_posteriors)
    for mm in (False, True):
        T.clear()
        m = FM.ExactColoc(mode="gaussian", M=50, compile=a.compile, mismatch=mm, mm_iter=a.mm_iter)
        m.fit_batch(loci[:4])  # warm-up
        T.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = m.fit_batch(loci)
        torch.cuda.synchronize()
        tot = time.perf_counter() - t0
        print(m.name, f"total {tot:.2f}s for {len(loci)} loci = {tot / len(loci):.3f} s/locus;",
              {k: round(v, 2) for k, v in T.items()}, "status", {s: sum(x["status"] == s for x in r) for s in ("ok", "failed")},
              "peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)


if __name__ == "__main__":
    main()
