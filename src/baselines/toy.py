"""Toy loci for wrapper tests and runtime measurements (not used in benchmark runs).

Genotypes: 1000G 30x chr22 (data/processed/1kg/chr22.pgen), NFE individuals (CEU, GBR, TSI, IBS; panel.tsv),
biallelic SNVs with MAF >= maf_min among them, first `p` such SNVs from `start_bp`. Standardized dosages X
(n x p); R = X'X / n is used both as the "population" LD for generating z and as the inference LD (an L0-like
setting: the controls test the wrappers, not robustness to mismatch).

z_t = R lam_t + e_t, e_t = X' u_t / sqrt(n), corr(u_1, u_2) = rho   (DESIGN §2.2 factor form, small n)
se_tj = 1 / sqrt(2 N_t f_j (1 - f_j)), beta = z * se   (sdY = 1)   [stats="corr": DESIGN §2.2 literally]

stats="ols" (default): z above is read as sqrt(N) * r (correlation scale, as the DESIGN §2.2 individual-level
z = G'y / (sqrt(N) sd(y))); it is converted to the OLS summary statistics that real studies report and that the
methods assume: beta = z * se (unchanged, = OLS beta), se_ols = se * sqrt(1 - z^2/N), z_ols = beta / se_ols
(= the t-statistic). See README "Summary-statistic scale" for why this matters (prop.coloc, SharePro, SuSiE n).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .locus import make_locus

PROJECT = Path(__file__).resolve().parents[2]
KG = PROJECT / "data" / "processed" / "1kg"
NFE = ("CEU", "GBR", "TSI", "IBS")


@lru_cache(maxsize=2)
def _pvar(chrom: str = "chr22") -> pd.DataFrame:
    df = pd.read_csv(KG / f"{chrom}.pvar", sep="\t", comment=None, skiprows=_n_header(KG / f"{chrom}.pvar"),
                     usecols=[0, 1, 2, 3, 4], names=["chrom", "pos", "id", "ref", "alt"], dtype={"pos": np.int64})
    df["idx"] = np.arange(len(df))
    snv = (df.ref.str.len() == 1) & (df.alt.str.len() == 1)
    dup = df.pos.duplicated(keep=False)
    return df[snv & ~dup].reset_index(drop=True)


def _n_header(path: Path) -> int:
    n = 0
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                n += 1
            else:
                break
    return n


@lru_cache(maxsize=2)
def _samples(chrom: str = "chr22") -> np.ndarray:
    psam = pd.read_csv(KG / f"{chrom}.psam", sep="\t")
    panel = pd.read_csv(KG / "panel.tsv", sep="\t", header=None, names=["iid", "pop", "spop", "sex"])
    keep = set(panel.loc[panel["pop"].isin(NFE), "iid"])
    idx = np.flatnonzero(psam.iloc[:, 0].isin(keep).to_numpy())
    return idx.astype(np.uint32)


@lru_cache(maxsize=8)
def load_window(p: int, start_bp: int = 30_000_000, maf_min: float = 0.05, chrom: str = "chr22"):
    """Returns (X standardized float64 n x p, maf (p,), variant ids, positions)."""
    import pgenlib

    pv = _pvar(chrom)
    samp = _samples(chrom)
    cand = pv[pv.pos >= start_bp]
    reader = pgenlib.PgenReader(str(KG / f"{chrom}.pgen").encode(), sample_subset=samp)
    n = len(samp)
    cols, mafs, ids, poss = [], [], [], []
    buf = np.empty(n, dtype=np.int32)
    for row in cand.itertuples(index=False):
        reader.read(int(row.idx), buf)
        g = buf.astype(np.float64)
        if (g < 0).any():
            continue
        f = g.mean() / 2
        maf = min(f, 1 - f)
        if maf < maf_min:
            continue
        cols.append((g - g.mean()) / g.std())
        mafs.append(maf)
        ids.append(row.id)
        poss.append(row.pos)
        if len(cols) == p:
            break
    reader.close()
    X = np.column_stack(cols)
    return X, np.array(mafs), np.array(ids, dtype=object), np.array(poss, dtype=np.int64)


def make_toy_locus(X, maf, ids, pos, lam1: dict, lam2: dict, rng: np.random.Generator, N1=100_000, N2=5_000,
                   rho=0.0, rho_hat=None, locus_id="toy", ld_type="dense", stats="ols"):
    """lam_t: {snp index: lambda}. Returns a LocusData (dense or factor LD from X)."""
    n, p = X.shape
    R = X.T @ X / n
    l1 = np.zeros(p)
    l2 = np.zeros(p)
    for j, v in lam1.items():
        l1[j] = v
    for j, v in lam2.items():
        l2[j] = v
    u1 = rng.standard_normal(n)
    u2 = rho * u1 + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    z1 = R @ l1 + X.T @ u1 / np.sqrt(n)
    z2 = R @ l2 + X.T @ u2 / np.sqrt(n)
    se1 = 1 / np.sqrt(2 * N1 * maf * (1 - maf))
    se2 = 1 / np.sqrt(2 * N2 * maf * (1 - maf))
    b1, b2 = z1 * se1, z2 * se2
    if stats == "ols":
        se1 = se1 * np.sqrt(1 - z1**2 / N1)
        se2 = se2 * np.sqrt(1 - z2**2 / N2)
        z1, z2 = b1 / se1, b2 / se2
    elif stats != "corr":
        raise ValueError(stats)
    kw = dict(locus_id=locus_id, variant_id=ids.copy(), pos=pos.copy(), z1=z1, z2=z2, beta1=b1, se1=se1,
              beta2=b2, se2=se2, n1=np.full(p, float(N1)), n2=np.full(p, float(N2)), maf1=maf.copy(),
              maf2=maf.copy(), maf_ref=maf.copy(), ld_type=ld_type, n_ref=n,
              rho_hat=float(rho if rho_hat is None else rho_hat), rho_se=0.05, mismatch_level="L0",
              pairing_type="toy", cell_id="toy", window_chr="chr22", window_start=int(pos[0]),
              window_end=int(pos[-1]) + 1, split="validation")
    if ld_type == "dense":
        kw["R"] = R.astype(np.float32)
    else:
        kw["X_ref"] = X.astype(np.float32)
    return make_locus(**kw)


def pick_pair(R: np.ndarray, rng: np.random.Generator, r2_max: float = 0.1, margin: int = 20):
    """Two SNP indices with r^2 < r2_max (away from the window edges)."""
    p = R.shape[0]
    for _ in range(10_000):
        a, b = rng.integers(margin, p - margin, size=2)
        if a != b and R[a, b] ** 2 < r2_max:
            return int(a), int(b)
    raise RuntimeError("no pair found")
