"""Access to the real-data inputs of the simulated world: pools, QC'd variants, source haplotypes,
real reference genotypes and the Beagle GRCh38 genetic map.

Everything goes through the data agent's loaders (src/data/genotypes.py, src/data/pools.py); this module
only adds the genetic map and small caches.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from data.genotypes import load_genotypes, load_haplotypes, variant_table
from data.pools import load_pools
from sim import params

ROOT = Path(__file__).resolve().parents[2]
GENMAP_DIR = ROOT / "data/raw/genmap/chr_in_chrom_field"
MHC = ("chr6", 28_500_000, 33_400_000)


def chrom_name(c) -> str:
    c = str(c)
    return c if c.startswith("chr") else f"chr{c}"


@lru_cache(maxsize=1)
def pools() -> dict[str, list[str]]:
    return load_pools(ROOT / "datasets/splits/pools_v1.tsv")


@lru_cache(maxsize=24)
def genetic_map(chrom: str) -> tuple[np.ndarray, np.ndarray]:
    """(bp, cM) arrays of the Beagle GRCh38 plink map, strictly increasing in bp."""
    chrom = chrom_name(chrom)
    f = GENMAP_DIR / f"plink.chr{chrom}.GRCh38.map"
    df = pd.read_csv(f, sep=r"\s+", header=None, names=["chr", "id", "cm", "bp"])
    assert (df["chr"] == chrom).all()
    df = df.sort_values("bp").drop_duplicates("bp")
    bp = df["bp"].to_numpy(np.int64)
    cm = np.maximum.accumulate(df["cm"].to_numpy(np.float64))  # liftover leaves tiny non-monotonicities
    return bp, cm


def interp_cm(chrom: str, pos: np.ndarray) -> np.ndarray:
    """Genetic position (cM) by linear interpolation; constant extrapolation beyond the map ends."""
    bp, cm = genetic_map(chrom)
    return np.interp(np.asarray(pos, dtype=np.float64), bp, cm)


def chrom_cm_length(chrom: str) -> float:
    return float(genetic_map(chrom)[1][-1])


def window_variants(chrom, start: int, end: int) -> pd.DataFrame:
    """QC'd SNVs with MAF >= MIN_MAF in the target pool, start <= pos <= end (DESIGN §2.1)."""
    v = variant_table(chrom_name(chrom), int(start), int(end), maf_pool=params.TARGET_POOL,
                      min_maf=params.MIN_MAF)
    v = v.copy()
    v["cm"] = interp_cm(chrom_name(chrom), v["pos"].to_numpy())
    return v


def source_haplotypes(pool: str, chrom, start: int, end: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Phased haplotypes (int8 [2n, p]) of a pool at the window's variant set."""
    H, v = load_haplotypes(chrom_name(chrom), int(start), int(end), pools()[pool],
                           maf_pool=params.TARGET_POOL, min_maf=params.MIN_MAF)
    v = v.copy()
    v["cm"] = interp_cm(chrom_name(chrom), v["pos"].to_numpy())
    return H, v


def pool_genotypes(pool: str, chrom, start: int, end: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Real ALT dosages (int8 [n, p]) of a pool at the window's variant set (e.g. pool-B reference)."""
    G, v = load_genotypes(chrom_name(chrom), int(start), int(end), pools()[pool],
                          maf_pool=params.TARGET_POOL, min_maf=params.MIN_MAF)
    return G, v
