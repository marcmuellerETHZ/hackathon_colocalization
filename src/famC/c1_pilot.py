"""C1 PILOT (not the pre-registered C1, which needs the simulation agent's frozen l0star_v1 set): build a small
L0* draw on VALIDATION windows with the simulation agent's own builder (`sim.l0star.build_set`, seed 20261029),
run Family B's simulator mode on it (the Bayes-optimal posterior in L0*), and score C checkpoints with
`scripts/famC_eval/c1_eval.py` (evaluation reads truth, so it lives outside src/famC; DECISIONS P3-21).

    python -m famC.c1_pilot build --per-class 150
    python -m famC.c1_pilot runB
    python scripts/famC_eval/c1_eval.py --set data/processed/famC/l0star_pilot/l0star_pilot.h5 \
        --b-preds data/processed/famC/l0star_pilot/famB_sim_preds.parquet --ckpt <checkpoints> --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from common.seeds import VALID_SEED
from famC.simadapter import DATA_DIR

OUT = DATA_DIR / "l0star_pilot"
RHO_VALUES = (0.0, 0.4)


def build(per_class: int):
    from sim.l0star import build_set
    st = build_set("l0star_pilot", "valid", VALID_SEED, {"S": per_class, "D": per_class, "W": per_class,
                                                          "G": per_class}, OUT, rho_values=RHO_VALUES)
    print(json.dumps(st, indent=1, default=str))


def run_b(n_mc: int = 16384, retry_failed: bool = False):
    from eval.io import iter_loci
    from famB.method import ExactColoc
    from famB.priors import sim_prior_from_l0star
    loci = list(iter_loci(OUT / "l0star_pilot.h5"))
    prev = None
    if retry_failed:
        prev = pd.read_parquet(OUT / "famB_sim_preds.parquet")
        bad = set(prev.loc[prev["status"] != "ok", "locus_id"])
        loci = [lc for lc in loci if lc.locus_id in bad]
        print(f"retrying {len(loci)} failed loci", flush=True)
    prior = sim_prior_from_l0star()
    m = ExactColoc(mode="simulator", sim_prior_fn=lambda lc: prior, eta=0.0, eta_floor=0.0, n_mc=n_mc, compile=True)
    t0 = time.time()
    rows = []
    for i in range(0, len(loci), 64):
        rows += m.fit_batch(loci[i:i + 64])
        print(f"{len(rows)}/{len(loci)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    if prev is not None:
        df = pd.concat([prev[~prev["locus_id"].isin(df["locus_id"])], df], ignore_index=True)
    df.to_parquet(OUT / "famB_sim_preds.parquet")
    print(df["status"].value_counts().to_dict(), f"{(time.time() - t0) / len(loci):.3f} s/locus")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "runB"])
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.per_class)
    else:
        run_b(retry_failed=a.retry_failed)


if __name__ == "__main__":
    main()
