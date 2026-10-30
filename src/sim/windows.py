"""Simulation windows from the data agent's split files (DESIGN §3).

Windows are the clipped gene windows of datasets/splits/genes_v1.tsv (TSS +- 500 kb clipped to the maximal run of
same-split blocks containing the TSS; BED convention: a variant at pos is inside iff start < pos <= end).
Simulated loci are anchored on protein-coding genes of the requested split (README D10); a window is used
only if it holds >= MIN_WINDOW_VARIANTS QC'd variants with target-pool MAF >= 0.01.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "datasets" / "splits"
MIN_WINDOW_VARIANTS = 200
SPLIT_ALIASES = {"validation": "valid", "valid": "valid", "train": "train", "test": "test"}


@lru_cache(maxsize=1)
def blocks() -> pd.DataFrame:
    return pd.read_csv(SPLITS_DIR / "blocks_v1.tsv", sep="\t")


@lru_cache(maxsize=1)
def genes() -> pd.DataFrame:
    f = SPLITS_DIR / "genes_v1.tsv"
    if not f.exists():
        f = SPLITS_DIR / "genes_v1.tsv.gz"
    return pd.read_csv(f, sep="\t")


def block_ids_in(chrom: str, start: int, end: int) -> list[str]:
    b = blocks()
    b = b[(b["chr"] == chrom) & (b["start"] < end) & (b["end"] > start)]
    return b["block_id"].tolist()


def split_windows(split: str, gene_type: str | None = "protein_coding") -> pd.DataFrame:
    """One row per anchor gene of the split: chr, anchor (TSS), window_start, window_end, block_id, block_ids."""
    if split == "test":
        from eval.io import TestSplitAccessError, has_test_access
        if not has_test_access():
            raise TestSplitAccessError("test-split windows are produced only by scripts/eval_test.py (DESIGN §7.3)")
    s = SPLIT_ALIASES[split]
    g = genes()
    g = g[(g["split"] == s) & (g["window_start"] >= 0) & (g["window_end"] > g["window_start"])]
    if gene_type is not None:
        g = g[g["gene_type"] == gene_type]
    g = g.sort_values(["chr", "tss", "gene_id"], key=lambda c: c.map(_chr_key) if c.name == "chr" else c)
    out = g[["gene_id", "chr", "tss", "window_start", "window_end", "block_id", "split"]].rename(
        columns={"tss": "anchor"}).reset_index(drop=True)
    b = blocks().set_index("block_id")
    assert (b.loc[out["block_id"], "split"].to_numpy() == s).all(), "anchor block split mismatch"
    out["block_ids"] = [tuple(block_ids_in(c, a, e)) for c, a, e in
                        zip(out["chr"], out["window_start"], out["window_end"])]
    # every block touched by the window must be in the same split (DESIGN §3 audit)
    for ids in out["block_ids"]:
        assert (b.loc[list(ids), "split"] == s).all(), ids
    return out


def _chr_key(c):
    return int(str(c).replace("chr", ""))
