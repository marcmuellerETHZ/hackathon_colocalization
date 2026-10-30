"""Train-region locus set with the M4 design (DESIGN_P2 §2 C: "the train-region gap uses a pre-generated train-block
set with the M4 design").

The 34 main-grid cells of datasets/configs/testset_v1.yaml (sim.testset.main_grid_cells, validation conventions: both rho
signs), per cell 100 S / 100 D / 100 W with D stratified over the r² bins and G, U kept as they arise up to 30 each
(the config's per_cell = the M4 subsample design, m4_subsample@1 main.txt), drawn by the frozen generator
(sim.cells.build_set, GridPlan as sim.testset.generate) on TRAIN-split windows with seed 20261030 (DESIGN §9 training
draws) under the key prefix ("famC_trainregion",), so no draw coincides with the training records
(rng(20261030, "train", step, ...)) or with any validation set.

Used only to measure C's train-region vs validation-region gap through the same harness and benchmark as the M4
validation subsample. Never used for training, tuning or checkpoint selection.

    python scripts/famC_eval/build_trainregion.py --set trainregion_m4_v1 --workers 3
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SEED = 20261030
PREFIX = ("famC_trainregion",)


def _worker(args):
    set_name, cell, plan, out_dir, K, M, log_path, max_visits = args
    import torch
    from sim.cells import build_set
    torch.cuda.set_device(0)
    lf = open(log_path, "a")

    def log(s):
        lf.write(s + "\n")
        lf.flush()
    st = build_set(set_name, [cell], None, None, "train", SEED, Path(out_dir), K=K, M=M, plan=plan, log=log,
                   key_prefix=PREFIX, max_visits=max_visits)
    lf.close()
    return st


def main():
    import yaml
    from sim import params
    from sim.cells import GridPlan, link_set
    from sim.testset import K1_PER_D, K2_PER_D, _merge_stats, d_strata_by, main_grid_cells
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "datasets/configs/testset_v1.yaml"))
    ap.add_argument("--set", default="trainregion_m4_v1")
    ap.add_argument("--out", default=str(ROOT / "data/processed/famC/sets"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--cells", nargs="*", default=None, help="smoke tests only")
    ap.add_argument("--max-visits", type=int, default=None, help="smoke tests only")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    pc = cfg["per_cell"]
    nS, nD, nW = int(pc["S"]), int(pc["D"]), int(pc["W"])
    nGU = int(pc["G_U"]["max"])
    cells = main_grid_cells(cfg, "train")
    assert len(cells) == 34, len(cells)
    if a.cells:
        cells = [c for c in cells if c.cell_id in set(a.cells)]
    plan = GridPlan(nS, nD, nW, nGU, nGU, K1_PER_D * nD, K2_PER_D * nD, by=d_strata_by(cfg))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logd = ROOT / "logs" / "famC"
    logd.mkdir(parents=True, exist_ok=True)
    jobs = [(a.set, c, plan, str(out), a.K, a.M,
             str(logd / f"{a.set}.{re.sub(r'[^A-Za-z0-9._=-]+', '_', c.cell_id)}.log"), a.max_visits) for c in cells]
    t0 = time.time()
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as pool:
        stats = pool.map(_worker, jobs, chunksize=1)
    m, _ = link_set(a.set, out)
    meta = {"config": a.config, "split": "train", "seed": SEED, "key_prefix": list(PREFIX), "design": "M4 (per_cell)",
            "per_cell": {"S": nS, "D": nD, "W": nW, "G_U_max": nGU}, "world_version": params.WORLD_VERSION,
            "commit": os.environ.get("ORCH_COMMIT"), "job": os.environ.get("ORCH_JOB_ID")}
    merged = _merge_stats(stats, meta, time.time() - t0)
    (out / f"{a.set}.stats.json").write_text(json.dumps(merged, indent=1))
    print(json.dumps({"set": str(m), "runtime_s": time.time() - t0}), flush=True)


if __name__ == "__main__":
    main()
