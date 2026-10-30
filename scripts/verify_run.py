#!/usr/bin/env python
"""Recompute a run's metrics from predictions + truth + manifest and compare with its metrics.json (DESIGN §7.2).

Usage: python scripts/verify_run.py results/<run_id> [results/<run_id> ...] [--tol 1e-9]
Exit code 0 if every run verifies, 1 otherwise. Checks:
  * predictions.parquet / truth / metrics.json sha256 match the manifest;
  * results policy D6(h): predictions >= 5 MB carry a run-local .gitignore (and are not committed);
  * predictions cover exactly the expected loci (ids_sha256), counts match the manifest;
  * the `call` column matches the manifest thresholds;
  * metrics recomputed with src/eval/metrics.py (manifest metric_config) equal metrics.json within tol;
  * the run is listed in <results>/LEDGER.tsv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from eval import metrics as M  # noqa: E402
from eval.harness import STATUSES, fill_calls, sha256_file  # noqa: E402
from eval.truth import read_truth  # noqa: E402


def _compare(a, b, tol, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present in only one")
            else:
                _compare(a[k], b[k], tol, f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: list length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _compare(x, y, tol, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol):
            out.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def verify(run_dir: str | Path, tol: float = 1e-9) -> list[str]:
    run_dir = Path(run_dir)
    problems = []
    man = json.loads((run_dir / "manifest.json").read_text())
    pred_path = run_dir / "predictions.parquet"
    pol = man.get("results_policy")
    if pol is not None and not pol.get("predictions_committed", True):
        gi = run_dir / ".gitignore"
        if not gi.exists() or "predictions.parquet" not in gi.read_text():
            problems.append("D6(h): predictions >= 5 MB but no run-local .gitignore excluding predictions.parquet")
    if not pred_path.exists():
        problems.append("predictions.parquet not present (D6(h): large predictions are kept only under results/; "
                        f"restore the file with sha256 {man['predictions_sha256']} to verify)")
        return problems
    if sha256_file(pred_path) != man["predictions_sha256"]:
        problems.append("predictions.parquet sha256 differs from manifest")
    pred = pd.read_parquet(pred_path)
    ids_hash = hashlib.sha256("\n".join(pred["locus_id"].astype(str)).encode()).hexdigest()
    if ids_hash != man["locus_set"]["ids_sha256"]:
        problems.append("predicted locus ids (order) differ from the manifest's expected ids")
    counts = {s: int((pred["status"] == s).sum()) for s in STATUSES}
    counts["expected"] = len(pred)
    if counts != man["counts"]:
        problems.append(f"status counts {counts} != manifest {man['counts']}")
    mc = man["metric_config"]
    calls = fill_calls(pred, mc["t_S"], mc["t_D"], mc["e_gate"])
    if not (calls.astype(str).to_numpy() == pred["call"].astype(str).to_numpy()).all():
        problems.append("call column inconsistent with manifest thresholds")
    mpath = run_dir / "metrics.json"
    if man.get("truth"):
        tpath = Path(man["truth"]["path"])
        if man["truth"].get("sha256") and sha256_file(tpath) != man["truth"]["sha256"]:
            problems.append("truth file sha256 differs from manifest")
        if not mpath.exists():
            problems.append("metrics.json missing although the run had truth")
        else:
            if man.get("metrics_sha256") and sha256_file(mpath) != man["metrics_sha256"]:
                problems.append("metrics.json sha256 differs from manifest")
            truth = read_truth(tpath)
            truth = truth[truth["locus_id"].astype(str).isin(set(pred["locus_id"].astype(str)))]
            recomputed = M.compute_run_metrics(truth, pred, M.MetricConfig.from_dict(mc))
            reported = json.loads(mpath.read_text())
            diffs = _compare(recomputed, reported, tol)
            problems += [f"metric mismatch {d}" for d in diffs[:50]]
    ledger = run_dir.parent / "LEDGER.tsv"
    if not ledger.exists() or man["run_id"] not in {l.split("\t")[0] for l in ledger.read_text().splitlines()}:
        problems.append("run not listed in LEDGER.tsv")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args(argv)
    if os.environ.get("COLOC_ENV_ACTIVE") != "1":
        print("source env/activate.sh first", file=sys.stderr)
        return 2
    bad = 0
    for d in a.run_dirs:
        probs = verify(d, a.tol)
        if probs:
            bad += 1
            print(f"FAIL {d}")
            for p in probs:
                print(f"  - {p}")
        else:
            print(f"OK   {d}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
