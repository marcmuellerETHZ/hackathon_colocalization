"""1000G EUR genotype windows for Family A/G validation (identity tests, timing).

Reads data/processed/1kg/chrNN.{pgen,pvar,psam} with pgenlib, restricted to the samples in
data/processed/1kg/samples_eur.txt. Variant QC mirrors DESIGN §2.1 for these tests only (biallelic
SNVs, positions with one record, MAF >= 0.01 in the sample used, MHC excluded); the project-wide QC
lives in src/data/variants.py (data agent) and is not used here.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pgenlib

PROJECT = Path(__file__).resolve().parents[3]
KG = PROJECT / "data" / "processed" / "1kg"
MHC = (28_500_000, 33_400_000)


def _read_pvar(path: Path) -> pd.DataFrame:
    skip = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("##"):
                skip += 1
            else:
                break
    df = pd.read_csv(path, sep="\t", skiprows=skip, usecols=[0, 1, 2, 3, 4], dtype={0: str, 1: np.int64, 2: str, 3: str, 4: str})
    df.columns = ["CHROM", "POS", "ID", "REF", "ALT"]
    return df


@lru_cache(maxsize=4)
def chrom_variants(chrom: int) -> pd.DataFrame:
    """Filtered variant table with column 'vidx' = row index in the pgen."""
    df = _read_pvar(KG / f"chr{chrom}.pvar")
    df["vidx"] = np.arange(len(df), dtype=np.int64)
    snv = (df.REF.str.len() == 1) & (df.ALT.str.len() == 1) & df.REF.isin(list("ACGT")) & df.ALT.isin(list("ACGT"))
    single = ~df.POS.duplicated(keep=False)
    keep = snv & single
    if chrom == 6:
        keep &= ~((df.POS >= MHC[0]) & (df.POS <= MHC[1]))
    return df.loc[keep].reset_index(drop=True)


@lru_cache(maxsize=1)
def eur_sample_index() -> np.ndarray:
    psam = pd.read_csv(KG / "chr22.psam", sep="\t")
    iid = psam.iloc[:, 0].astype(str).tolist()
    eur = set(pd.read_csv(KG / "samples_eur.txt", header=None)[0].astype(str))
    idx = np.array([i for i, s in enumerate(iid) if s in eur], dtype=np.uint32)
    return idx


def read_genotypes(chrom: int, vidx: np.ndarray, sample_idx: np.ndarray | None = None) -> np.ndarray:
    """Alt-allele dosages (n_samples, n_variants) as float64 for pgen rows vidx (sorted)."""
    sample_idx = eur_sample_index() if sample_idx is None else np.asarray(sample_idx, dtype=np.uint32)
    reader = pgenlib.PgenReader(str(KG / f"chr{chrom}.pgen").encode(), sample_subset=sample_idx)
    try:
        v = np.ascontiguousarray(np.asarray(vidx, dtype=np.uint32))
        buf = np.empty((len(v), len(sample_idx)), dtype=np.int8)
        reader.read_list(v, buf)
    finally:
        reader.close()
    G = buf.T.astype(np.float64)
    if (G < 0).any():
        raise ValueError(f"missing genotypes in chr{chrom} window")
    return G


def window(chrom: int, start: int, p: int, maf_min: float = 0.01, sample_idx=None, oversample: float = 2.0):
    """p consecutive QC'd variants (MAF >= maf_min in the EUR sample) starting at filtered index `start`.

    Returns (G standardized (n,p) float64 with population sd, variant table)."""
    tab = chrom_variants(chrom)
    take = int(p * oversample) + 50
    sub = tab.iloc[start:start + take]
    G = read_genotypes(chrom, sub.vidx.to_numpy(), sample_idx)
    f = G.mean(0) / 2.0
    maf = np.minimum(f, 1 - f)
    ok = np.nonzero(maf >= maf_min)[0]
    if len(ok) < p:
        raise ValueError("window too short after MAF filter")
    ok = ok[:p]
    G = G[:, ok]
    sub = sub.iloc[ok].reset_index(drop=True)
    sub["maf"] = maf[ok]
    G = (G - G.mean(0)) / G.std(0)
    return G, sub
