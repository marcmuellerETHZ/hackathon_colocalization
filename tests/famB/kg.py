"""Real 1000G (30x, GRCh38) EUR-NFE LD windows for Family B screen-recall tests (read-only use of
data/processed/1kg pgen files, datasets/splits/pools_v1.tsv and datasets/splits/blocks_v1.tsv; train-split blocks only).

QC as DESIGN Sec. 2.1: biallelic SNVs, positions with > 1 record removed, MAF >= 0.01 in the target pool,
MHC (chr6:28.5-33.4 Mb) excluded.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pgenlib

ROOT = Path(__file__).resolve().parents[2]
KG = ROOT / "data/processed/1kg"


@lru_cache(maxsize=32)
def _pvar(chrom: str) -> pd.DataFrame:
    df = pd.read_csv(KG / f"{chrom}.pvar", sep="\t", comment=None, skiprows=_n_header(chrom),
                     usecols=[0, 1, 2, 3, 4], names=["chr", "pos", "id", "ref", "alt"], dtype={"pos": np.int64})
    df["idx"] = np.arange(len(df))
    return df


def _n_header(chrom):
    n = 0
    with open(KG / f"{chrom}.pvar") as f:
        for line in f:
            if line.startswith("#"):
                n += 1
            else:
                break
    return n


@lru_cache(maxsize=32)
def _psam(chrom: str) -> list[str]:
    return pd.read_csv(KG / f"{chrom}.psam", sep="\t")["#IID"].tolist()


def pool_samples(pool: str) -> list[str]:
    p = pd.read_csv(ROOT / "datasets/splits/pools_v1.tsv", sep="\t")
    return p.loc[p["pool"] == pool, "sample_id"].tolist()


def train_windows(chroms, n, rng, width=1_000_000, min_len=300_000):
    b = pd.read_csv(ROOT / "datasets/splits/blocks_v1.tsv", sep="\t")
    b = b[(b["split"] == "train") & (b["source"] == "pyrho") & b["chr"].isin(chroms) &
          ((b["end"] - b["start"]) >= min_len)]
    rows = b.iloc[rng.choice(len(b), size=min(n, len(b)), replace=False)]
    out = []
    for _, r in rows.iterrows():
        mid = (r["start"] + r["end"]) // 2
        s, e = max(r["start"], mid - width // 2), min(r["end"], mid + width // 2)
        out.append((r["chr"], int(s), int(e), r["block_id"]))
    return out


def load_window(chrom, start, end, target_pool="A", ref_pool=None, maf_min=0.01):
    """Returns dict(pos, ids, X (standardized target dosages [n, p]), maf, Xref (if ref_pool))."""
    pv = _pvar(chrom)
    w = pv[(pv["pos"] >= start) & (pv["pos"] < end)]
    if chrom == "chr6":
        w = w[~((w["pos"] >= 28_500_000) & (w["pos"] <= 33_400_000))]
    dup = w["pos"].duplicated(keep=False)
    snv = (w["ref"].str.len() == 1) & (w["alt"].str.len() == 1)
    w = w[snv & ~dup]
    if len(w) == 0:
        raise ValueError(f"no SNVs in {chrom}:{start}-{end}")
    ids = _psam(chrom)
    pos_of = {s: i for i, s in enumerate(ids)}

    def read(pool):
        sm = np.array(sorted(pos_of[s] for s in pool_samples(pool) if s in pos_of), dtype=np.uint32)
        r = pgenlib.PgenReader(str(KG / f"{chrom}.pgen").encode(), sample_subset=sm)
        v0, v1 = int(w["idx"].min()), int(w["idx"].max()) + 1
        buf = np.empty((v1 - v0, len(sm)), dtype=np.int8)
        r.read_range(v0, v1, buf)
        r.close()
        G = buf[(w["idx"].to_numpy() - v0)].T.astype(np.float64)  # [n, p]
        if np.any(G < 0):
            raise ValueError("missing genotypes in 1000G 30x pgen (unexpected)")
        return G

    G = read(target_pool)
    f = G.mean(0) / 2
    maf = np.minimum(f, 1 - f)
    keep = maf >= maf_min
    out = {}
    if ref_pool is not None:
        Gr = read(ref_pool)
        sd_r = Gr.std(0)
        keep &= sd_r > 0  # monomorphic in the reference: cannot form LD (dropped, as a reference panel would)
    G = G[:, keep]
    X = (G - G.mean(0)) / G.std(0)
    out.update(pos=w["pos"].to_numpy()[keep], ids=w["id"].to_numpy()[keep], X=X, maf=maf[keep])
    if ref_pool is not None:
        Gr = Gr[:, keep]
        out["Xref"] = (Gr - Gr.mean(0)) / Gr.std(0)
    return out
