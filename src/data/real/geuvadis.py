"""GEUVADIS individuals for R4b (exact in-sample QTL LD) and their genotype PCs.

Sample list: the eQTL Catalogue does not publish the 1000G IDs of its GEUVADIS donors. We use
E-GEUV-1 (ArrayExpress) individuals that are in the 1000G phase-3 unrelated set (2,504). This gives 445
individuals = 358 EUR (CEU 89, FIN 92, GBR 86, TSI 91) + 87 YRI, exactly the eQTL Catalogue population
table for GEUVADIS (445; 358 EUR, 87 AFR) and its 445 genotype-QC-passed samples (Zenodo 4678936).
The identity of each individual is therefore inferred, not published (documented in the README).

PCs: eQTL Catalogue (qtlmap) adjusts for 6 genotype PCs (and 6 expression PCs). In-sample LD that matches
the z-scores is the LD of genotypes residualised on the covariates; we residualise on 6 genotype PCs
computed here from the 445 individuals (thinned common SNVs: one SNV with in-sample MAF >= 0.05 per
100 kb, autosomes). Expression PCs are not available and are ignored (documented).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from data.real.config import OUT, RAW_EQTL, ROOT

GEUV_DIR = OUT / "geuvadis"
N_PCS = 6
THIN_BP = 100_000


def derive_samples() -> pd.DataFrame:
    s = pd.read_csv(RAW_EQTL / "meta/E-GEUV-1.sdrf.txt", sep="\t", dtype=str)
    s = s[["Source Name", "Characteristics[ancestry category]"]].drop_duplicates()
    s.columns = ["sample_id", "geuvadis_ancestry"]
    p3 = pd.read_csv(ROOT / "data/raw/1kg30x/phase3_2504.panel", sep="\t", dtype=str)
    m = s.merge(p3[["sample", "pop", "super_pop"]].rename(columns={"sample": "sample_id"}), on="sample_id")
    return m.sort_values("sample_id").reset_index(drop=True)


@lru_cache(maxsize=1)
def samples() -> list[str]:
    f = GEUV_DIR / "samples.tsv"
    if not f.exists():
        GEUV_DIR.mkdir(parents=True, exist_ok=True)
        derive_samples().to_csv(f, sep="\t", index=False)
    return pd.read_csv(f, sep="\t")["sample_id"].tolist()


@lru_cache(maxsize=1)
def nfe_minus_geuvadis() -> list[str]:
    """R4b reference panel: NFE (pools A u B; CEU, GBR, IBS, TSI unrelated) minus all GEUVADIS individuals."""
    from data.real.ld import pools
    p = pools()
    nfe = set(p["A"]) | set(p["B"])
    return sorted(nfe - set(samples()))


def compute_pcs() -> pd.DataFrame:
    from data.genotypes import load_genotypes, variant_table
    sm = samples()
    cols = []
    for c in range(1, 23):
        vt = variant_table(f"chr{c}", 0, 10**10)
        if vt.empty:
            raise RuntimeError(f"no variant table for chr{c}")
        start, end = int(vt["pos"].iloc[0]), int(vt["pos"].iloc[-1])
        G, v = load_genotypes(f"chr{c}", start - 1, end, sm)  # BED convention
        af = G.mean(axis=0) / 2.0
        maf = np.minimum(af, 1 - af)
        ok = np.flatnonzero(maf >= 0.05)
        bins = v["pos"].to_numpy()[ok] // THIN_BP
        _, first = np.unique(bins, return_index=True)
        cols.append(G[:, ok[first]].astype(np.float64))
    X = np.concatenate(cols, axis=1)
    X = (X - X.mean(0)) / X.std(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    pcs = pd.DataFrame(U[:, :N_PCS] * S[:N_PCS], columns=[f"PC{i + 1}" for i in range(N_PCS)])
    pcs.insert(0, "sample_id", sm)
    pcs.attrs["n_snvs"] = X.shape[1]
    return pcs


@lru_cache(maxsize=1)
def pcs() -> np.ndarray:
    f = GEUV_DIR / "pcs.tsv"
    if not f.exists():
        GEUV_DIR.mkdir(parents=True, exist_ok=True)
        p = compute_pcs()
        p.to_csv(f, sep="\t", index=False)
    p = pd.read_csv(f, sep="\t")
    assert p["sample_id"].tolist() == samples()
    return p[[f"PC{i + 1}" for i in range(N_PCS)]].to_numpy()


def residualise(G: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Residuals of each genotype column on [1, C] (least squares)."""
    A = np.column_stack([np.ones(len(C)), C])
    coef, *_ = np.linalg.lstsq(A, np.asarray(G, dtype=np.float64), rcond=None)
    return np.asarray(G, dtype=np.float64) - A @ coef
