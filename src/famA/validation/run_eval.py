"""Family A ladder runs through the eval harness (INTERFACES §3 predictions; task p5.famA_screen_m4).

python -m famA.validation.run_eval --group screen_L24 --rungs primary,base,+rho --label famA_screen_v2

Groups (validation split only):
  screen_L24    screen_v2 cells at L2/L4 (4 cells x 2000)
  screen_L0P3   screen_v2 L0|P3|rho=0.5
  screen_L0P1   screen_v2 L0|P1|r_p=0.0 = kill-test A1 cell: run WITHOUT truth (no metrics.json)
  m4_L1p        m4_subsample@1 main.txt ids at L1..L6
  m4_L0P23      m4_subsample@1 main.txt ids at L0, pairing P2/P3 (no trait-2 LD in these cells)
  m4_L0P1       m4_subsample@1 main.txt ids at L0|P1 (trait-2 LD; run without truth until A1: same population
                as the kill-test cell)
Rungs: 'primary' = rho+eta+multistart+perTraitLD (P3-5/P3-6, PI), LADDER rungs, or '+'-joined components.
L0 groups refuse to run unless the shared reader can load the trait-2 LD (eval.io load_ld2): otherwise per-trait LD
would silently fall back to one LD.
Each harness run lands in <results-dir>/<run_id>; one JSON line per run is appended to <index>.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eval.harness import MethodSpec, run_method
from famA.method import ladder_name

PRIMARY = "rho+eta+multistart+perTraitLD"
SCREEN = "data/processed/sets/screen_v2.h5"
GRID = "data/processed/sets/valid_grid_v1.h5"
M4_IDS = "results/baselines_m4/m4_subsample_v1/main.txt"
A1_CELL = "L0_P1_r_p=0.0"


def cell_of(locus_id: str) -> str:
    return locus_id.split("__")[1]


def level_of(locus_id: str) -> str:
    return cell_of(locus_id).split("_")[0]


def group_ids(group: str) -> tuple[str, list[str], bool]:
    """(locus set, ids, with_truth)."""
    if group.startswith("screen_"):
        import h5py
        with h5py.File(SCREEN, "r") as f:
            ids = [k for k in f.keys() if not k.startswith("_")]
        sel = {"screen_L24": lambda i: level_of(i) in ("L2", "L4"),
               "screen_L0P3": lambda i: cell_of(i) == "L0_P3_rho=0.5",
               "screen_L0P1": lambda i: cell_of(i) == A1_CELL}[group]
        return SCREEN, [i for i in ids if sel(i)], group != "screen_L0P1"
    ids = Path(M4_IDS).read_text().split()
    if group == "m4_L1p":
        return GRID, [i for i in ids if level_of(i) != "L0"], True
    if group == "m4_L0P23":
        return GRID, [i for i in ids if level_of(i) == "L0" and "_P1_" not in cell_of(i)], True
    if group == "m4_L0P1":
        return GRID, [i for i in ids if level_of(i) == "L0" and "_P1_" in cell_of(i)], False
    raise ValueError(group)


def ladder_subsample(ids: list[str], frac: float) -> list[str]:
    """Seeded per-cell subsample (sorted ids per cell; same ids for every rung and job)."""
    from common.seeds import rng
    keep = set()
    by_cell = {}
    for i in ids:
        by_cell.setdefault(cell_of(i), []).append(i)
    for c, cid in sorted(by_cell.items()):
        cid = sorted(cid)
        k = int(round(frac * len(cid)))
        g = rng(20261029, "famA_ladder_subsample", c)
        keep.update(np.asarray(cid)[np.sort(g.choice(len(cid), size=k, replace=False))].tolist())
    return [i for i in ids if i in keep]


def reader_loads_ld2() -> bool:
    from eval.io import LocusSetReader
    return "load_ld2" in inspect.signature(LocusSetReader.read).parameters


def n_with_ld2(locus_set: str, ids: list[str]) -> int:
    """Loci that carry a trait-2 LD (attribute ld_type2; L0 sets)."""
    import h5py
    with h5py.File(locus_set, "r") as f:
        return sum("ld_type2" in f[i].attrs for i in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, help="comma-separated groups")
    ap.add_argument("--rungs", required=True, help="comma-separated; 'primary' = " + PRIMARY)
    ap.add_argument("--label", required=True)
    ap.add_argument("--results-dir", default="results/famA/runs")
    ap.add_argument("--index", default=None, help="JSONL index (default results/famA/<label>/runs.jsonl)")
    ap.add_argument("--batch-size", type=int, default=256, help="harness batch for factor-only groups (screen_L24)")
    ap.add_argument("--batch-size-dense", type=int, default=40,
                    help="harness batch for groups with dense LD (L0, M4 L1): <= 40 x 150 MB per batch in RAM")
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--mem-budget-gb", type=float, default=0.0,
                    help="GPU working-set budget of the continuous batching (engineering only; results equal "
                         "up to fp64 rounding, README 13); 0 = JointConfig default (6 GB)")
    ap.add_argument("--subsample-frac", type=float, default=0.0,
                    help="ladder ablations (PI triage): seeded per-cell subsample, rng(20261029, "
                         "'famA_ladder_subsample', cell), identical for every rung; 0 = all loci")
    ap.add_argument("--retry-failed", action="append", default=[],
                    help="harness run dir(s): rerun only the loci with status failed/timeout there (infrastructure "
                         "retry, e.g. CUDA OOM); the group is recorded as <group>__retry")
    ap.add_argument("--eig-chunk", type=int, default=0,
                    help="loci per batched eigh (engineering; bounds GPU memory; 0 = JointConfig default 4)")
    ap.add_argument("--limit", type=int, default=0, help="first N ids only (plumbing check)")
    a = ap.parse_args()
    groups = {}
    for g in a.group.split(","):
        locus_set, ids, with_truth = group_ids(g)
        if g != "screen_L0P1":
            assert not any(i.startswith("screen_v2__") and cell_of(i) == A1_CELL for i in ids)
        if a.retry_failed:
            bad = set()
            for d in a.retry_failed:
                pr = pd.read_parquet(Path(d) / "predictions.parquet")
                bad |= set(pr.loc[pr["status"].isin(["failed", "timeout"]), "locus_id"].astype(str))
            ids = [i for i in ids if i in bad]
        if a.subsample_frac:
            ids = ladder_subsample(ids, a.subsample_frac)
        if a.limit:
            ids = ids[: a.limit]
        n_ld2 = n_with_ld2(locus_set, ids) if "L0" in g else 0
        if n_ld2 and not reader_loads_ld2():
            raise SystemExit(f"{g}: {n_ld2} loci carry a trait-2 LD but eval.io has no load_ld2: per-trait LD "
                             "would silently use one LD; not running")
        if a.retry_failed:
            g = g + "__retry"
        if not ids:
            print(f"[run_eval] group {g}: no loci, skipped", flush=True)
            continue
        groups[g] = (locus_set, ids, with_truth, n_ld2)
        print(f"[run_eval] group {g}: {len(ids)} loci from {locus_set}; truth={'yes' if with_truth else 'NO'}; "
              f"{n_ld2} with a trait-2 LD", flush=True)
    index = Path(a.index or f"results/famA/{a.label}/runs.jsonl")
    index.parent.mkdir(parents=True, exist_ok=True)
    Path(a.results_dir).mkdir(parents=True, exist_ok=True)
    for rung in a.rungs.split(","):
        comp = PRIMARY if rung == "primary" else rung
        name = ladder_name(comp)
        for g, (locus_set, ids, with_truth, n_ld2) in groups.items():
            t0 = time.time()
            kw = {"rung": comp}
            if a.eig_chunk:
                kw["eig_chunk"] = a.eig_chunk
            if a.mem_budget_gb:
                kw["mem_budget_bytes"] = a.mem_budget_gb * 1e9
            r = run_method(MethodSpec("famA.method:LadderMethod", kw, name), locus_set,
                           results_dir=a.results_dir, mode="batch", n_workers=1,
                           batch_size=a.batch_size if g.startswith("screen_L24") else a.batch_size_dense,
                           timeout_s=a.timeout_s, ids=ids, truth_path="auto" if with_truth else None,
                           label=f"{a.label} {g} {rung}" + (f" sub{a.subsample_frac}" if a.subsample_frac else ""),
                           config={"task": "p5.famA_screen_m4", "group": g, "rung": rung, "components": comp, "subsample_frac": a.subsample_frac,
                                   "orch_job": os.environ.get("ORCH_JOB_ID"),
                                   "orch_commit": os.environ.get("ORCH_COMMIT")})
            pred = pd.read_parquet(r.run_dir / "predictions.parquet")
            ex = pred["extra"].map(lambda s: json.loads(s) if isinstance(s, str) else {})
            rec = {"group": g, "rung": rung, "components": comp, "method": name, "run_id": r.run_id,
                   "run_dir": str(Path(r.run_dir).resolve()), "locus_set": locus_set, "n": len(ids),
                   "counts": r.counts, "with_truth": with_truth, "n_with_ld2": n_ld2,
                   "wall_s": round(time.time() - t0, 1),
                   "n_per_trait_ld": int(ex.map(lambda e: bool(e.get("per_trait_ld"))).sum()),
                   "n_noshare_start": int(ex.map(lambda e: e.get("start") == "noshare").sum()),
                   "runtime_median_s": float(pred["runtime_s"].median()),
                   "orch_job": os.environ.get("ORCH_JOB_ID"), "orch_commit": os.environ.get("ORCH_COMMIT")}
            with open(index, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print("[run_eval] " + json.dumps(rec), flush=True)

if __name__ == "__main__":
    main()
