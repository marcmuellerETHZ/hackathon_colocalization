"""Generate Family C training / validation records in shards (data/processed/famC/<set>/).

    python -m famC.gen_shards --set train_v1 --split train --n-loci 200000 --worker 0 --n-workers 4
    python -m famC.gen_shards --set valid_v1 --split valid --n-loci 20000 --worker 0 --n-workers 1

Train: sim.train_api window visits step = worker + n_workers * i (seed 20261030, DESIGN §9), held-out level L6.
Valid: validation-split windows, seed 20261029, all levels including the held-out L6.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from common.seeds import REPLICATE_SEED, VALID_SEED
from famC.simadapter import ACCEPT, DATA_DIR, make_sampler_class, make_valid_sampler_class, write_shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--split", choices=["train", "valid"], required=True)
    ap.add_argument("--n-loci", type=int, required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--lpw", type=int, default=64)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--shard-size", type=int, default=2048)
    ap.add_argument("--n-noise", type=int, default=1, help="noise draws per candidate (README C-T6)")
    ap.add_argument("--device", default="cuda", help="cuda or cpu (CPU workers: one process per core, OMP 1)")
    ap.add_argument("--step-offset", type=int, default=0,
                    help="first window-visit step; a disjoint offset gives fresh draws from the same windows "
                         "(e.g. the held-out train-region set)")
    ap.add_argument("--out-dir", default=None, help="override output directory (benchmarks)")
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads for CPU workers")
    ap.add_argument("--accept", default="none",
                    help='"none" (natural class mixture), "default" (simadapter.ACCEPT) or a JSON dict class->prob')
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    accept = None if a.accept == "none" else (ACCEPT if a.accept == "default" else json.loads(a.accept))
    if a.split == "train":
        S = make_sampler_class()
        s = S(M=a.M, seed=REPLICATE_SEED, holdout=("L6",), loci_per_window=a.lpw, accept=accept, n_noise=a.n_noise,
              device=a.device)
    else:
        S = make_valid_sampler_class()
        s = S(M=a.M, seed=VALID_SEED, holdout=(), loci_per_window=a.lpw, accept=accept, n_noise=a.n_noise,
              device=a.device)
    out = Path(a.out_dir) if a.out_dir else DATA_DIR / a.set
    out.mkdir(parents=True, exist_ok=True)
    import hashlib
    import subprocess
    root = DATA_DIR.parents[2]
    code = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
            for d in ("src/sim", "src/stats", "src/famC") for f in sorted((root / d).glob("*.py"))}
    import os
    git = os.environ.get("ORCH_COMMIT") or subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    (out / f"config_w{a.worker}.json").write_text(json.dumps(vars(a) | {
        "accept": accept, "levels": s.levels, "orch_job": os.environ.get("ORCH_JOB_ID"), "start": time.strftime("%F %T"), "git_head": git,
        "code_sha256": code}, indent=1))   # the simulator may be uncommitted: record exact file hashes
    # worker k visits windows k, k + K, k + 2K, ...
    orig = s.window_loci
    s.step = a.step_offset + a.worker

    def stepped(step):
        r = orig(step)
        s.step += a.n_workers - 1   # write_shards adds 1
        return r
    s.window_loci = stepped
    n = write_shards(s, out, a.n_loci, a.shard_size, tag=f"w{a.worker}")
    print("DONE", n, json.dumps(s.stats), flush=True)


if __name__ == "__main__":
    main()
