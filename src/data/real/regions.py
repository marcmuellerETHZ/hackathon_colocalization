"""Genes, split assignment and analysis windows for real loci (DESIGN §3) — thin adapter over the data
agent's split files (datasets/splits/blocks_v1.tsv, datasets/splits/genes_v1.tsv[.gz]; src/data/make_splits.py).

Conventions (from make_splits.py): blocks and windows are BED-like (0-based start, exclusive end); a 1-based
position x lies in [start, end) iff start < x <= end. Splits are {train, valid, test, excluded}.
A gene's window is its TSS +- 500 kb clipped to the maximal run of same-split blocks containing the TSS
(genes_v1.tsv columns window_start/window_end). Genes in "excluded" (MHC) blocks are dropped.
Gene ids are compared without version suffix (eQTL Catalogue uses unversioned Ensembl ids).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from data.real.config import SPLITS

SPLIT_NAMES = ("train", "valid", "test")


class SplitsUnavailable(FileNotFoundError):
    pass


def _need(path: Path) -> Path:
    if not path.exists():
        raise SplitsUnavailable(f"{path} not found (produced by the data agent: python -m data.make_splits)")
    return path


@lru_cache(maxsize=1)
def load_blocks() -> pd.DataFrame:
    from data.make_splits import load_blocks as _lb
    return _lb(_need(SPLITS / "blocks_v1.tsv"))


@lru_cache(maxsize=1)
def gene_table() -> pd.DataFrame:
    """gene_id (unversioned), gene_name, gene_type, chr, tss, strand, block_id, split, window_start, window_end
    (BED), plus win_lo/win_hi = 1-based inclusive bounds (win_lo = window_start + 1, win_hi = window_end)."""
    f = SPLITS / "genes_v1.tsv.gz"
    g = pd.read_csv(_need(f if f.exists() else SPLITS / "genes_v1.tsv"), sep="\t", dtype={"chr": str})
    g = g[g["split"].isin(SPLIT_NAMES)].copy()
    g["gene_id_versioned"] = g["gene_id"]
    g["gene_id"] = g["gene_id"].str.split(".").str[0]
    if g["gene_id"].duplicated().any():
        dup = g[g["gene_id"].duplicated(keep=False)]["gene_id_versioned"].tolist()[:5]
        raise ValueError(f"duplicated unversioned gene ids in genes_v1.tsv, e.g. {dup}")
    g["win_lo"] = g["window_start"].astype(np.int64) + 1
    g["win_hi"] = g["window_end"].astype(np.int64)
    return g.reset_index(drop=True)


def blocks_in_window(chrom: str, win_lo: int, win_hi: int) -> list[str]:
    """Block ids overlapping the 1-based inclusive window [win_lo, win_hi]."""
    b = load_blocks()
    b = b[b["chr"] == chrom]
    return b[(b["start"] < win_hi) & (b["end"] >= win_lo)]["block_id"].astype(str).tolist()


def in_window(pos, win_lo: int, win_hi: int) -> np.ndarray:
    pos = np.asarray(pos)
    return (pos >= win_lo) & (pos <= win_hi)
