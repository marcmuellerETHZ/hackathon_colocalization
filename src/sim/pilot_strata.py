"""Pilot: natural oracle-class and D-stratum (r² x d bins, DESIGN v4 §8.6) rates of the candidate mixture per
pairing type, on validation windows. Used to set the staged-relaxation thresholds of the grid builder
(src/sim/cells.py GridPlan; README D17). Candidates only (no noise / LD), so it is cheap.

  python -m sim.pilot_strata --n 20000 --out src/sim/validation/pilot_strata.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common.seeds import VALID_SEED, rng
from sim import params
from sim.generator import CellSpec, WindowContext, assert_no_tf32, draw_candidate
from sim.windows import split_windows

CELLS = [CellSpec("L1|P1|r_p=0.0", "L1", "P1", 0.0, p1_mode="nested"),
         CellSpec("L1|P2|rho=0.0", "L1", "P2", 0.0),
         CellSpec("L1|P3|rho=0.0|eqtl", "L1", "P3", 0.0, p3_mode="eqtl"),
         CellSpec("L1|P3|rho=0.0|lab", "L1", "P3", 0.0, p3_mode="lab")]


def stratum(orc):
    r2e = [b[0] for b in params.R2_BINS] + [params.R2_BINS[-1][1]]
    de = [b[0] for b in params.D_BINS] + [np.inf]
    rb = int(np.searchsorted(r2e, orc.r2, side="right") - 1)
    db = int(np.searchsorted(de, orc.d, side="right") - 1)
    return (rb if rb < len(params.R2_BINS) else "ge0.95"), db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--per-window", type=int, default=100)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "validation/pilot_strata.json"))
    a = ap.parse_args()
    assert_no_tf32()
    wins = split_windows("valid")
    order = rng(VALID_SEED, "pilot_strata", "windows").permutation(len(wins))
    res = {}
    t0 = time.time()
    counts = {c.cell_id: {"n": 0, "cls": {}, "D": {}} for c in CELLS}
    v = 0
    while min(x["n"] for x in counts.values()) < a.n:
        w = wins.iloc[order[v]]
        v += 1
        ctx = WindowContext(w["chr"], int(w["window_start"]), int(w["window_end"]), split="valid")
        if ctx.p < 200:
            continue
        for c in CELLS:
            st = counts[c.cell_id]
            for k in range(a.per_window):
                cand = draw_candidate(ctx, c, rng(VALID_SEED, "pilot_strata", c.cell_id, v, k))
                st["n"] += 1
                if cand is None:
                    continue
                o = cand.oracle
                st["cls"][o.cls] = st["cls"].get(o.cls, 0) + 1
                if o.cls == "D":
                    key = "r%s_d%s" % stratum(o)
                    st["D"][key] = st["D"].get(key, 0) + 1
        ctx.release()
        print(f"[{time.time()-t0:.0f}s] window {v} " + " ".join(f"{k}:{x['n']}" for k, x in counts.items()), flush=True)
    for k, st in counts.items():
        n = st["n"]
        res[k] = {"n": n, "class_rate": {c: v / n for c, v in sorted(st["cls"].items())},
                  "D_stratum_rate": {c: v / n for c, v in sorted(st["D"].items())},
                  "D_stratum_count": dict(sorted(st["D"].items()))}
    res["_runtime_s"] = time.time() - t0
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
