#!/usr/bin/env python
"""p5.verify_infra check (e): no test-split locus anywhere under data/processed.

Independent of src/eval/io.py: test blocks are read directly from splits/blocks_v1.tsv (split == "test").
A locus/row is flagged if ANY of:
  * its `split` attribute/column is "test";
  * any of its block ids is a test block;
  * its window [start, end) overlaps a test block interval [start, end) on the same chromosome.
Metadata only: HDF5 group attributes (no datasets), parquet/TSV id/split/block/window columns, famC shard configs.
Never reads z, beta, LD or any test-pair statistic (candidate TSVs: only split/block/window columns are read).

Output: JSON summary on stdout (and $ORCH_OUT/scan_test_split.json if set). Exit 1 if anything is flagged.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ORCH_HOME", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data" / "processed"

blk = pd.read_csv(ROOT / "splits" / "blocks_v1.tsv", sep="\t")
TEST = blk[blk["split"] == "test"]
TEST_IDS = set(TEST["block_id"].astype(str))
TEST_IV = defaultdict(list)
for r in TEST.itertuples():
    TEST_IV[str(r.chr)].append((int(r.start), int(r.end)))
TEST_IV = {c: np.array(sorted(v)) for c, v in TEST_IV.items()}


def overlaps_test(chrom, start, end) -> bool:
    iv = TEST_IV.get(str(chrom))
    if iv is None or start is None or end is None or start < 0:
        return False
    return bool(np.any((iv[:, 0] < int(end)) & (iv[:, 1] > int(start))))


def _s(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def is_test(split, block_ids, chrom, start, end) -> list[str]:
    why = []
    if _s(split).lower() == "test":
        why.append("split")
    bids = [] if block_ids is None else [_s(b) for b in np.atleast_1d(block_ids)]
    if any(b in TEST_IDS for b in bids):
        why.append("block")
    if chrom is not None and overlaps_test(_s(chrom), start, end):
        why.append("window")
    return why


def scan_h5(path: Path, rep: dict):
    n = n_link = n_noattr = 0
    flagged = []
    splits = Counter()
    with h5py.File(path, "r") as h:
        for k in h.keys():
            ln = h.get(k, getlink=True)
            if isinstance(ln, h5py.ExternalLink):
                n_link += 1           # master file: the part file is scanned on its own
                continue
            if k.startswith("_"):
                continue              # shared LD store (_ld)
            g = h[k]
            if not isinstance(g, h5py.Group):
                continue
            a = g.attrs
            if "split" not in a and "block_ids" not in a and "window_chr" not in a:
                n_noattr += 1
                continue
            n += 1
            splits[_s(a.get("split", ""))] += 1
            why = is_test(a.get("split", ""), a.get("block_ids"), a.get("window_chr"),
                          a.get("window_start"), a.get("window_end"))
            if why:
                flagged.append((k, why))
    rep[str(path.relative_to(ROOT))] = {"kind": "h5", "n_loci": n, "n_external_links": n_link,
                                        "n_groups_without_locus_attrs": n_noattr, "splits": dict(splits),
                                        "n_flagged": len(flagged), "flagged_examples": flagged[:5]}


def scan_table(path: Path, rep: dict):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        cols = pq.ParquetFile(path).schema_arrow.names
        want = [c for c in ("locus_id", "split", "block_id", "block_ids", "window_chr", "window_start",
                            "window_end", "chr", "win_lo", "win_hi") if c in cols]
        if not want or not ({"split", "block_id", "block_ids", "window_chr", "chr"} & set(want)):
            rep[str(path.relative_to(ROOT))] = {"kind": "parquet", "note": "no split/block/window columns",
                                                "columns": cols[:12]}
            return
        df = pd.read_parquet(path, columns=want)
    else:
        head = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
        want = [c for c in ("locus_id", "split", "block_id", "chr", "win_lo", "win_hi") if c in head]
        if not ({"split", "block_id", "chr"} & set(want)):
            rep[str(path.relative_to(ROOT))] = {"kind": "tsv", "note": "no split/block/window columns"}
            return
        df = pd.read_csv(path, sep="\t", usecols=want)
    chrom = df["window_chr"] if "window_chr" in df else (df["chr"] if "chr" in df else None)
    st = df["window_start"] if "window_start" in df else (df["win_lo"] if "win_lo" in df else None)
    en = df["window_end"] if "window_end" in df else (df["win_hi"] if "win_hi" in df else None)
    bcol = "block_ids" if "block_ids" in df else ("block_id" if "block_id" in df else None)
    flagged = []
    for i in range(len(df)):
        why = is_test(df["split"].iat[i] if "split" in df else "",
                      df[bcol].iat[i] if bcol else None,
                      chrom.iat[i] if chrom is not None else None,
                      int(st.iat[i]) if st is not None else None, int(en.iat[i]) if en is not None else None)
        if why:
            flagged.append((str(df["locus_id"].iat[i]) if "locus_id" in df else i, why))
    rep[str(path.relative_to(ROOT))] = {"kind": path.suffix[1:], "n_rows": len(df),
                                        "splits": dict(Counter(df["split"].astype(str))) if "split" in df else None,
                                        "n_flagged": len(flagged), "flagged_examples": flagged[:3]}


def scan_famc_configs(rep: dict):
    for cfg in sorted((DATA / "famC").rglob("config_*.json")):
        d = json.loads(cfg.read_text())
        rep[str(cfg.relative_to(ROOT))] = {"kind": "famC_shard_config", "split": d.get("split"),
                                           "set": d.get("set"), "levels": d.get("levels"),
                                           "n_flagged": int(str(d.get("split", "")).lower() == "test")}


def main():
    rep: dict = {}
    for p in sorted(DATA.rglob("*.h5")):
        try:
            scan_h5(p, rep)
        except OSError as e:
            rep[str(p.relative_to(ROOT))] = {"kind": "h5", "error": str(e)}
    for p in sorted(list(DATA.rglob("*.truth.parquet")) + list(DATA.rglob("*.tsv"))):
        if p.name in ("SHA256SUMS",) or "extract_log" in p.name:
            continue
        try:
            scan_table(p, rep)
        except Exception as e:  # noqa: BLE001
            rep[str(p.relative_to(ROOT))] = {"kind": "table", "error": f"{type(e).__name__}: {e}"}
    scan_famc_configs(rep)
    # positive control: a real test block must be flagged by every rule
    b = TEST.iloc[0]
    ctl = {"split": is_test("test", [], None, None, None), "block": is_test("", [b["block_id"]], None, None, None),
           "window": is_test("", [], b["chr"], int(b["start"]) + 1, int(b["start"]) + 2),
           "negative": is_test("valid", ["chrX_none"], "chr1", -5, -1)}
    tot = sum(v.get("n_flagged", 0) for v in rep.values())
    known_test_lists = {k: v for k, v in rep.items() if "test_ids" in k}
    other = {k: v for k, v in rep.items() if "test_ids" not in k and v.get("n_flagged", 0)}
    out = {"n_test_blocks": len(TEST_IDS), "positive_control": ctl, "files": rep,
           "total_flagged": tot, "flagged_outside_test_id_lists": {k: v["n_flagged"] for k, v in other.items()},
           "test_id_lists": {k: v.get("n_rows") for k, v in known_test_lists.items()},
           "errors": {k: v["error"] for k, v in rep.items() if "error" in v}}
    s = json.dumps(out, indent=1, default=str)
    if os.environ.get("ORCH_OUT"):
        Path(os.environ["ORCH_OUT"], "scan_test_split.json").write_text(s)
    print(s)
    return 1 if other else 0


if __name__ == "__main__":
    sys.exit(main())
