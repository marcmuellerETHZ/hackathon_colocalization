"""Summarise bench_runtime JSONL files into a TSV / markdown table.

python -m baselines.bench_summary logs/baselines/runtime_M3.jsonl [more.jsonl] --tsv out.tsv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def summarise(paths) -> pd.DataFrame:
    recs = []
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if "rows" not in r:
                recs.append(dict(method=r["method"], p=r["p"], ld="?", status="cell_error"))
                continue
            rt = np.array([x["runtime_s"] for x in r["rows"]])
            recs.append(dict(method=r["method"], p=r["p"], ld="mismatch" if r.get("mismatch") else "in-sample",
                             n_loci=len(rt), median_s=float(np.median(rt)), max_s=float(rt.max()),
                             ld_build_s=float(np.median([x["ld_build_s"] for x in r["rows"]])),
                             r_startup_s=r.get("r_startup_s"), r_peak_rss_mb=r.get("r_peak_rss_mb"),
                             py_peak_rss_mb=r.get("py_peak_rss_mb"),
                             status=",".join(sorted({x["status"] for x in r["rows"]})),
                             s=";".join(f"{x['s']:.3g}" for x in r["rows"])))
    return pd.DataFrame(recs).sort_values(["ld", "method", "p"]).reset_index(drop=True)


def to_markdown(df: pd.DataFrame) -> str:
    def f(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.3g}"
        return "" if v is None else str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--tsv", default="")
    a = ap.parse_args()
    df = summarise(a.paths)
    if a.tsv:
        df.to_csv(a.tsv, sep="\t", index=False, float_format="%.4g")
    print(to_markdown(df))


if __name__ == "__main__":
    main()
