"""Run Family B (ExactColoc, Gaussian-EB) through the eval harness on one or more locus sets.

    PYTHONPATH=src python scripts/famB_eval/run_harness.py --sets data/processed/sets/screen_v2/*.h5 \
        --M 50 --out results/famB/screen_v2_v1/M50 [--bucket D_r2_0.5-0.8] [--full-sets 'L0_P1_r_p=0.0'] \
        [--every 10] [--n-workers 2] [--batch-size 128]

One harness run (eval.harness.run_method, mode "batch") per locus set; run directories go to the global
results/ (manifest, predictions, metrics, LEDGER line). `<out>/predictions.parquet` concatenates the runs'
predictions and `<out>/runs.json` lists every run (run_id, run_dir, predictions sha256, counts, wall time).
Locus selection uses only locus ids (bucket token of the id), never truth.
Config (DECISIONS P3-7): mode gaussian, EB (sigma1, sigma2, eta), p1 = p2 = 1e-4, p12 = 5e-6, M = 50 primary.
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


def bucket_of(locus_id: str) -> str:
    return locus_id.split("__")[2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--p12", type=float, default=5e-6)
    ap.add_argument("--n-mc", type=int, default=16384)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bucket", default=None, help="only loci of this bucket (id token) ...")
    ap.add_argument("--full-sets", nargs="*", default=[], help="... except in these set stems (all loci)")
    ap.add_argument("--every", type=int, default=1, help="every k-th locus (pilots)")
    ap.add_argument("--ids-file", default=None, help="only loci listed in this file (one id per line)")
    ap.add_argument("--n-workers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--batch-size-dense", type=int, default=48, help="batch size for sets with dense LD (L0)")
    ap.add_argument("--timeout-s", type=float, default=60.0, help="per locus")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--label", default="famB")
    ap.add_argument("--mismatch", action="store_true", help="B-mm (DESIGN_P2 §2): variance-inflated RSS likelihood")
    ap.add_argument("--mm-iter", type=int, default=2)
    ap.add_argument("--mm-diag-only", action="store_true", help="ablation: no cross terms in the inflation")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results_dir = Path(os.environ.get("ORCH_RESULTS", ROOT / "results"))
    kw = {"mode": "gaussian", "M": a.M, "p12": a.p12, "n_mc": a.n_mc, "compile": not a.no_compile}
    name = f"exactcoloc_gaussian_M{a.M}" + ("" if a.p12 == 5e-6 else f"_p12_{a.p12:g}")
    if a.mismatch:
        kw.update(mismatch=True, mm_iter=a.mm_iter, mm_cross_terms=not a.mm_diag_only)
        name = (f"exactcoloc_mm_gaussian_M{a.M}" + ("" if a.mm_iter == 2 else f"_it{a.mm_iter}")
                + ("_diag" if a.mm_diag_only else "") + ("" if a.p12 == 5e-6 else f"_p12_{a.p12:g}"))
    spec = MethodSpec("famB.method:ExactColoc", kw, name)
    keep = set(Path(a.ids_file).read_text().split()) if a.ids_file else None
    runs, preds = [], []
    for s in a.sets:
        s = Path(s)
        ids = list_loci(s)
        if a.bucket and s.stem not in a.full_sets:
            ids = [i for i in ids if bucket_of(i) == a.bucket]
        if keep is not None:
            ids = [i for i in ids if i in keep]
        ids = ids[::a.every]
        if not ids:
            continue
        dense = s.stem.startswith("L0_")
        bs = a.batch_size_dense if dense else a.batch_size
        t0 = time.time()
        r = run_method(spec, s, results_dir=results_dir, mode="batch", n_workers=a.n_workers, batch_size=bs,
                       timeout_s=a.timeout_s, ids=ids, label=f"{a.label} {s.stem}",
                       config={"famB": kw, "job": os.environ.get("ORCH_JOB_ID"), "bucket": a.bucket, "ids_file": a.ids_file,
                               "every": a.every}, hash_inputs=False)
        wall = time.time() - t0
        p = pd.read_parquet(r.run_dir / "predictions.parquet")
        preds.append(p)
        runs.append({"set": str(s), "run_id": r.run_id, "run_dir": str(r.run_dir), "n": len(ids),
                     "counts": r.counts, "wall_s": wall, "loci_per_s": len(ids) / wall,
                     "predictions_sha256": sha256_file(r.run_dir / "predictions.parquet")})
        print(json.dumps(runs[-1]), flush=True)
    allp = pd.concat(preds, ignore_index=True)
    allp.to_parquet(out / "predictions.parquet", index=False)
    if keep is not None:
        missing = keep - set(allp["locus_id"].astype(str))
        if missing:
            raise RuntimeError(f"{len(missing)} ids of {a.ids_file} not in the given sets, e.g. {sorted(missing)[:3]}")
    meta = {"method": name, "kwargs": kw, "runs": runs, "n_loci": len(allp), "ids_file": a.ids_file,
            "status_counts": allp["status"].value_counts().to_dict(),
            "commit": os.environ.get("ORCH_COMMIT"), "job": os.environ.get("ORCH_JOB_ID")}
    (out / "runs.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "runs"}), flush=True)


if __name__ == "__main__":
    main()
