"""Collect Family A harness runs (run_eval.py index files) into one registered artifact directory.

python -m famA.validation.collect --index results/famA/famA_screen_v2/runs.jsonl [--index ...] \
    --out results/famA/famA_screen_v2_v1 [--exclude-group screen_L0P1]

Writes (never overwrites an existing directory):
  predictions/<method>.parquet   INTERFACES §3 rows of every collected run of that method (concatenated over groups)
  runs.json                      method -> [{group, run_id, run_dir, predictions_sha256, n, counts, cells}]
  coverage.tsv                   method x cell: n rows, n ok / nonconverged / failed / timeout
No truth is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def cell_of(locus_id: str) -> str:
    return locus_id.split("__")[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-group", action="append", default=[])
    ap.add_argument("--include-group", action="append", default=[], help="keep only these groups (default all)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists (never mutate a registered artifact)")
    recs = []
    for ix in a.index:
        recs += [json.loads(l) for l in Path(ix).read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r["group"] not in a.exclude_group
            and (not a.include_group or r["group"] in a.include_group)]
    # keep the LAST run per (method, group) (a re-run supersedes; superseded runs are listed as such)
    last = {}
    for r in recs:
        last[(r["method"], r["group"])] = r
    (out / "predictions").mkdir(parents=True)
    runs, cov = {}, []
    by_method = {}
    for (m, g), r in sorted(last.items()):
        d = Path(r["run_dir"])
        man = json.loads((d / "manifest.json").read_text())
        pp = d / "predictions.parquet"
        h = sha256(pp)
        if h != man["predictions_sha256"]:
            raise SystemExit(f"{d}: predictions sha256 differs from manifest")
        p = pd.read_parquet(pp)
        p["cell"] = p["locus_id"].map(cell_of)
        p["_retry"] = g.endswith("__retry")
        p["_group"] = g
        by_method.setdefault(m, []).append(p)
        runs.setdefault(m, []).append({"group": g, "rung": r["rung"], "components": r["components"],
                                       "run_id": r["run_id"], "run_dir": str(d), "predictions_sha256": h,
                                       "locus_set": man["locus_set"]["path"], "n": int(len(p)),
                                       "counts": man["counts"], "orch_job": r.get("orch_job"),
                                       "orch_commit": r.get("orch_commit"), "wall_s": man.get("wall_s"),
                                       "n_with_ld2": r.get("n_with_ld2"), "n_per_trait_ld": r.get("n_per_trait_ld"),
                                       "cells": sorted(p["cell"].unique().tolist())})
    superseded = [r for r in recs if last[(r["method"], r["group"])] is not r]
    for m, parts in by_method.items():
        # infrastructure retries (group '<g>__retry', e.g. after CUDA OOM) replace the earlier rows of their loci
        parts = sorted(parts, key=lambda d: d["_retry"].iloc[0])
        p = pd.concat(parts, ignore_index=True).drop_duplicates("locus_id", keep="last").drop(columns="_retry")
        if p["locus_id"].duplicated().any():
            raise SystemExit(f"{m}: duplicate locus ids across groups")
        for c, dc in p.groupby("cell"):
            row = {"method": m, "cell": c, "n": len(dc)}
            row.update({s: int((dc["status"] == s).sum()) for s in ("ok", "nonconverged", "failed", "timeout")})
            cov.append(row)
        p.drop(columns=["cell", "_group"]).to_parquet(out / "predictions" / f"{m}.parquet", index=False)
    (out / "runs.json").write_text(json.dumps({"runs": runs, "superseded": superseded}, indent=1, sort_keys=True))
    pd.DataFrame(cov).sort_values(["method", "cell"]).to_csv(out / "coverage.tsv", sep="\t", index=False)
    print(pd.DataFrame(cov).pivot_table(index="cell", columns="method", values="n", aggfunc="sum").to_string())


if __name__ == "__main__":
    main()
