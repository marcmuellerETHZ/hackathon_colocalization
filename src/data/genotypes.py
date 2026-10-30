"""Genotype / haplotype loader for QC'd SNVs (DESIGN §2.1), via pgenlib.

    from data.genotypes import load_genotypes, load_haplotypes
    G, v = load_genotypes("chr22", 20_000_000, 21_000_000, samples)   # G int8 [n_samples, p], ALT counts 0/1/2
    H, v = load_haplotypes("chr22", 20_000_000, 21_000_000, samples)  # H int8 [2*n_samples, p], ALT 0/1;
                                                                      # rows 2i, 2i+1 = haplotypes of samples[i]
`v` is the variant table (data/processed/variants/chrN.parquet rows, positional index reset), columns
chr, pos, variant_id, ref, alt, pgen_idx, af_<pool>. Rows of G/H follow the order of `samples`.
Region convention = BED, as in datasets/splits/blocks_v1.tsv and the window_start/window_end of
datasets/splits/genes_v1.tsv: a variant at 1-based pos is included iff start < pos <= end. (So a gene window can
be passed as-is and never picks up a variant from a neighbouring block of another split.) Optional `min_maf`/`maf_pool` apply the population-specific
MAF filter (DESIGN §2.1), e.g. maf_pool="A", min_maf=0.01.

The pgen files are converted from the phased 1000G 30x VCF with plink2 --make-pgen, which keeps phase;
load_haplotypes asserts that every heterozygous call is phased (tests/data/test_genotypes.py also
checks the haplotypes against the VCF).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd
import pgenlib

from data.variants import PGEN_DIR, OUT_DIR, read_psam


def _chr(chrom) -> str:
    chrom = str(chrom)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


@lru_cache(maxsize=4)
def _variants(chrom: str) -> pd.DataFrame:
    return pd.read_parquet(OUT_DIR / f"{chrom}.parquet")


@lru_cache(maxsize=1)
def _psam(chrom: str) -> tuple[str, ...]:
    return tuple(read_psam(PGEN_DIR / f"{chrom}.psam"))


def variant_table(chrom, start: int, end: int, maf_pool: str | None = None,
                  min_maf: float | None = None) -> pd.DataFrame:
    chrom = _chr(chrom)
    v = _variants(chrom)
    pos = v["pos"].to_numpy()
    lo = np.searchsorted(pos, start, side="right")  # first pos > start
    hi = np.searchsorted(pos, end, side="right")    # last pos <= end
    v = v.iloc[lo:hi]
    if min_maf is not None:
        if maf_pool is None:
            raise ValueError("min_maf needs maf_pool")
        af = v[f"af_{maf_pool}"].to_numpy()
        v = v[np.minimum(af, 1 - af) >= min_maf - 1e-9]
    return v.reset_index(drop=True)


def _sample_subset(chrom: str, samples: Sequence[str]) -> tuple[np.ndarray, np.ndarray, int]:
    psam = _psam(chrom)
    pos = {s: i for i, s in enumerate(psam)}
    missing = [s for s in samples if s not in pos]
    if missing:
        raise KeyError(f"{len(missing)} samples not in {chrom}.psam, e.g. {missing[:3]}")
    if len(set(samples)) != len(samples):
        raise ValueError("duplicate sample IDs")
    idx = np.array([pos[s] for s in samples], dtype=np.uint32)
    order = np.argsort(idx, kind="stable")
    subset = idx[order]                      # pgenlib needs a sorted subset
    back = np.empty_like(order)
    back[order] = np.arange(len(order))      # requested row i = subset column back[i]
    return subset, back, len(psam)


def _reader(chrom: str, subset: np.ndarray, n_raw: int) -> pgenlib.PgenReader:
    return pgenlib.PgenReader(bytes(str(PGEN_DIR / f"{chrom}.pgen"), "utf8"), raw_sample_ct=n_raw,
                              sample_subset=subset)


def load_genotypes(chrom, start: int, end: int, samples: Sequence[str], maf_pool: str | None = None,
                   min_maf: float | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """ALT dosage matrix int8 [n_samples, p] (0/1/2) and the variant table."""
    chrom = _chr(chrom)
    v = variant_table(chrom, start, end, maf_pool, min_maf)
    subset, back, n_raw = _sample_subset(chrom, samples)
    g = np.empty((len(v), len(subset)), dtype=np.int8)
    if len(v):
        r = _reader(chrom, subset, n_raw)
        r.read_list(v["pgen_idx"].to_numpy(np.uint32), g)
        r.close()
        if (g < 0).any():
            raise ValueError(f"missing genotypes in {chrom}:{start}-{end}")
    return np.ascontiguousarray(g[:, back].T), v


def load_haplotypes(chrom, start: int, end: int, samples: Sequence[str], maf_pool: str | None = None,
                    min_maf: float | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Phased ALT indicators int8 [2*n_samples, p]; rows 2i, 2i+1 are the two haplotypes of samples[i]
    (in VCF GT order a|b)."""
    chrom = _chr(chrom)
    v = variant_table(chrom, start, end, maf_pool, min_maf)
    subset, back, n_raw = _sample_subset(chrom, samples)
    n = len(subset)
    al = np.empty((len(v), 2 * n), dtype=np.int32)
    pp = np.empty((len(v), n), dtype=np.uint8)
    if len(v):
        r = _reader(chrom, subset, n_raw)
        r.read_alleles_and_phasepresent_list(v["pgen_idx"].to_numpy(np.uint32), al, pp)
        r.close()
        if (al < 0).any():
            raise ValueError(f"missing alleles in {chrom}:{start}-{end}")
        het = al[:, 0::2] != al[:, 1::2]
        if not pp[het].all():
            raise ValueError(f"unphased heterozygous calls in {chrom}:{start}-{end}")
    # columns of al: sample-major (s0h0, s0h1, s1h0, ...) in subset order -> reorder samples
    al = al.reshape(len(v), n, 2)[:, back, :].reshape(len(v), 2 * n)
    return np.ascontiguousarray(al.T.astype(np.int8)), v
