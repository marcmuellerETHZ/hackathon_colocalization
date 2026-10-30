"""Run Family C (NeuralColoc, a main-task checkpoint) through the eval harness on one or more locus sets.

    PYTHONPATH=src python scripts/famC_eval/run_harness.py --sets data/processed/sets/valid_grid_v1/*.h5 \
        --ids-file results/baselines_m4/m4_subsample_v1/main.txt --checkpoint results/famC/main_snap1_full/model.pt \
        --out results/famC/m4_provisional_v1 --label famC_m4_provisional

One harness run (eval.harness.run_method, mode "batch") per locus set; run directories go to the global results/
(manifest, predictions, metrics, LEDGER line). `<out>/predictions.parquet` concatenates the runs' predictions and
`<out>/runs.json` lists every run (run_id, run_dir, predictions sha256, counts, wall time). Locus selection uses
only locus ids, never truth. Method: famC.method.NeuralColoc(checkpoint, class_w="main") (main-task models are
trained with class-weighted CE; README C-T2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from eval.harness import MethodSpec, run_method, sha256_file  # noqa: E402
from eval.io import list_loci  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--name", default="neuralcoloc_main_snap1_full")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--n-workers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--batch-size-dense", type=int, default=32)
    ap.add_argument("--timeout-s", type=float, default=60.0)
    ap.add_argument("--label", default="famC")
    ap.add_argument("--variant", default="full", choices=["full", "nold"], help="network variant (nold = ablation)")
    ap.add_argument("--floor-dir", default=None, help="run the LightGBM floor (famC.floor_method) instead; "
                    "--checkpoint is then ignored")
    ap.add_argument("--floor-threads", type=int, default=1)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results_dir = Path(os.environ.get("ORCH_RESULTS", ROOT / "results"))
    if a.floor_dir:
        kw = {"model_dir": str(Path(a.floor_dir).resolve()), "device": "cpu", "batch": 64,
              "threads": a.floor_threads}
        spec = MethodSpec("famC.floor_method:FloorColoc", kw, a.name)
    else:
        kw = {"checkpoint": str(Path(a.checkpoint).resolve()), "class_w": "main", "batch": 32, "variant": a.variant}
        spec = MethodSpec("famC.method:NeuralColoc", kw, a.name)
    keep = set(Path(a.ids_file).read_text().split()) if a.ids_file else None
    runs, preds = [], []
    for s in a.sets:
        s = Path(s)
        ids = list_loci(s)
        if keep is not None:
            ids = [i for i in ids if i in keep]
        if not ids:
            continue
        bs = a.batch_size_dense if s.stem.startswith("L0_") else a.batch_size
        t0 = time.time()
        r = run_method(spec, s, results_dir=results_dir, mode="batch", n_workers=a.n_workers, batch_size=bs,
                       timeout_s=a.timeout_s, ids=ids, label=f"{a.label} {s.stem}",
                       config={"famC": kw, "job": os.environ.get("ORCH_JOB_ID"), "ids_file": a.ids_file},
                       hash_inputs=False)
        wall = time.time() - t0
        p = pd.read_parquet(r.run_dir / "predictions.parquet")
        preds.append(p)
        runs.append({"set": str(s), "run_id": r.run_id, "run_dir": str(r.run_dir), "n": len(ids), "counts": r.counts,
                     "wall_s": wall, "loci_per_s": len(ids) / wall,
                     "predictions_sha256": sha256_file(r.run_dir / "predictions.parquet")})
        print(json.dumps(runs[-1]), flush=True)
    allp = pd.concat(preds, ignore_index=True)
    allp.to_parquet(out / "predictions.parquet", index=False)
    if keep is not None:
        missing = keep - set(allp["locus_id"].astype(str))
        if missing:
            raise RuntimeError(f"{len(missing)} ids of {a.ids_file} not in the given sets, e.g. {sorted(missing)[:3]}")
    meta = {"method": a.name, "kwargs": kw, "runs": runs, "n_loci": len(allp), "ids_file": a.ids_file,
            "status_counts": allp["status"].value_counts().to_dict(),
            "commit": os.environ.get("ORCH_COMMIT"), "job": os.environ.get("ORCH_JOB_ID")}
    (out / "runs.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "runs"}), flush=True)


if __name__ == "__main__":
    main()
