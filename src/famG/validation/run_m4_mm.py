"""G-mm on the M4 main validation subsample (m4_subsample@1 main.txt, valid_grid_v1) through the eval harness
(predictions + manifest + metrics + ledger), resumable via checkpoint_dir."""
from __future__ import annotations

import argparse
from pathlib import Path

from eval.harness import MethodSpec, run_method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="results/baselines_m4/m4_subsample_v1/main.txt")
    ap.add_argument("--out", default="results/famG_mm/m4_v1")
    ap.add_argument("--run-id", default="famG_mm_m4_main_v1")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--mem-gb", type=float, default=2.5)
    a = ap.parse_args()
    ids = Path(a.ids).read_text().split()
    out = Path(a.out)
    res = run_method(MethodSpec("famG.method_mm:GPUColocSusieMM", {"ld_dtype": a.dtype, "mem_budget_gb": a.mem_gb}),
                     "data/processed/sets/valid_grid_v1.h5", results_dir=out, run_id=a.run_id, ids=ids, mode="batch",
                     n_workers=a.workers, batch_size=a.batch, timeout_s=600.0, checkpoint_dir=out / "ckpt",
                     label="p9 famG G-mm M4 main")
    print(res.run_dir, res.counts)


if __name__ == "__main__":
    main()
