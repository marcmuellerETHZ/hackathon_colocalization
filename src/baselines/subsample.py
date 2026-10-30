"""Pre-registered, seeded locus subsamples for the p4 validation baseline runs (pure functions; no file access).

The caller (evaluation side, scripts/baselines_p4.py) supplies a table of locus ids with their cell and stratum
(oracle class / generation bucket); this module opens no file at all.

screen_v2 (task p4.baselines_screen): slow methods (mvSuSiE, ColocBoost both modes) run on ALL loci of the A1 cell
L0|P1|r_p=0.0 and on 200 loci per other cell, stratified by generation bucket (equal allocation, 50 per bucket for
4 buckets), seed derive(20261027, 'screen_slow_subsample').

valid_grid_v1 M4 (decision P3-13, P3-17): per cell 100 S / 100 D / 100 W plus G and U up to 30 each ("main",
every method); slow methods 40 S / 40 D / 40 W per cell ("slow"); mvSuSiE 20 / 20 / 20 ("mvsusie");
seed derive(20261029, 'm4_subsample'). The three samples are NESTED (mvsusie c slow c main), so every paired
comparison on a smaller sample uses the same loci for all methods.

Draw (both): for each (cell, stratum) the ids are sorted as strings and permuted with
rng(seed, key, cell, stratum).permutation(n); a sample of size k takes the first min(k, n) of that permutation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.seeds import rng

SCREEN_SEED_KEY = (20261027, "screen_slow_subsample")
M4_SEED_KEY = (20261029, "m4_subsample")
SCREEN_FULL_CELLS = ("L0|P1|r_p=0.0",)


def _ordered(ids, seed_key, cell, stratum) -> list[str]:
    ids = sorted(map(str, ids))
    perm = rng(seed_key[0], seed_key[1], str(cell), str(stratum)).permutation(len(ids))
    return [ids[i] for i in perm]


def _check(df: pd.DataFrame, cols) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"missing columns {miss}")
    if df["locus_id"].duplicated().any():
        raise ValueError("duplicate locus ids")


def screen_slow_subsample(df: pd.DataFrame, n_per_cell: int = 200, full_cells=SCREEN_FULL_CELLS,
                          seed_key=SCREEN_SEED_KEY) -> list[str]:
    """df: locus_id, cell_id, bucket. All loci of `full_cells`, n_per_cell (equal allocation over the cell's
    buckets in sorted order, remainder to the first buckets; a bucket smaller than its share gives all) per other
    cell. Returns the sorted id list."""
    _check(df, ("locus_id", "cell_id", "bucket"))
    keep: list[str] = []
    for cell, dc in df.groupby("cell_id", sort=True):
        if cell in full_cells:
            keep += dc["locus_id"].astype(str).tolist()
            continue
        buckets = sorted(dc["bucket"].astype(str).unique())
        base, rem = divmod(n_per_cell, len(buckets))
        for i, b in enumerate(buckets):
            k = base + (1 if i < rem else 0)
            keep += _ordered(dc.loc[dc["bucket"].astype(str) == b, "locus_id"], seed_key, cell, b)[:k]
    return sorted(keep)


def m4_subsample(df: pd.DataFrame, per_class: int = 100, slow: int = 40, mvsusie: int = 20, gu_max: int = 30,
                 seed_key=M4_SEED_KEY) -> dict[str, list[str]]:
    """df: locus_id, cell_id, oracle_class. Returns {"main", "slow", "mvsusie"} sorted id lists (nested)."""
    _check(df, ("locus_id", "cell_id", "oracle_class"))
    if not mvsusie <= slow <= per_class:
        raise ValueError("need mvsusie <= slow <= per_class (nested samples)")
    out = {"main": [], "slow": [], "mvsusie": []}
    for cell, dc in df.groupby("cell_id", sort=True):
        for cls, dk in dc.groupby("oracle_class", sort=True):
            order = _ordered(dk["locus_id"], seed_key, cell, cls)
            if cls in ("S", "D", "W"):
                out["main"] += order[:per_class]
                out["slow"] += order[:slow]
                out["mvsusie"] += order[:mvsusie]
            else:  # G, U (and any other class): kept as they arise, up to gu_max each; fast methods only
                out["main"] += order[:gu_max]
    return {k: sorted(v) for k, v in out.items()}
