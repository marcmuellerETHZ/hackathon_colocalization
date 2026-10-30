"""LD look-ups for pair building: reference-panel r^2 (1000G pools) and FinnGen R12 in-sample r^2.

FinnGen R12 LD (PLINK2 --r-unphased, 520,210 samples, pairs within 3 Mb with r^2 > 0.01) is queried
remotely by tabix (local copy of the .tbi, nothing downloaded besides the queried BGZF blocks).
A pair absent from the file while variant 1 is present means r^2 <= 0.01 (reported as 0.0, status
'below_0.01'); if variant 1 has no rows at all its status is 'v1_absent' and r^2 is NaN.
Results are cached in data/processed/real/cache/finngen_ld_pairs.tsv.
"""
from __future__ import annotations

import fcntl
import io
import subprocess
from functools import lru_cache

import numpy as np
import pandas as pd

from data.real.config import FINNGEN_LD_URL, OUT, RAW_FG
from data.real.harmonise import variant_key

CACHE_FILE = OUT / "cache" / "finngen_ld_pairs.tsv"


def _unordered(v: str) -> str:
    c, p, a, b = variant_key(v).split(":")
    return f"{c}:{p}:" + ":".join(sorted([a, b]))


def finngen_rows(v1: str) -> pd.DataFrame:
    """All FinnGen LD rows for variant v1 (as variant1)."""
    c, p, _, _ = variant_key(v1).split(":")
    cb = c[3:]
    url = FINNGEN_LD_URL.format(c=cb) + "##idx##" + str(RAW_FG / "ld_matrix" / f"finngen_r12_chr{cb}_ld.tsv.gz.tbi")
    res = subprocess.run(["tabix", url, f"{cb}:{p}-{p}"], check=True, capture_output=True, text=True)
    if res.stderr.strip():
        raise RuntimeError(f"tabix FinnGen LD {v1}: {res.stderr.strip()}")
    if not res.stdout:
        return pd.DataFrame(columns=["chr", "pos", "variant1", "variant2", "r", "r2"])
    df = pd.read_csv(io.StringIO(res.stdout), sep="\t", header=None, names=["chr", "pos", "variant1", "variant2", "r", "r2"],
                     dtype={"chr": str})
    return df


def finngen_r2(v1: str, v2: str) -> tuple[float, str]:
    """(r^2, status) for one pair in FinnGen R12 LD. status: 'found' | 'below_0.01' | 'v1_absent'.
    Matching ignores allele order (r^2 is symmetric under allele swap); v1 is tried in both orientations."""
    rows = finngen_rows(v1)
    sel = rows[rows["variant1"].map(_unordered) == _unordered(v1)]
    if sel.empty:
        return float("nan"), "v1_absent"
    hit = sel[sel["variant2"].map(_unordered) == _unordered(v2)]
    if len(hit):
        return float(hit["r2"].iloc[0]), "found"
    return 0.0, "below_0.01"


def _cache_load() -> dict:
    if not CACHE_FILE.exists():
        return {}
    df = pd.read_csv(CACHE_FILE, sep="\t", dtype={"v1": str, "v2": str, "status": str})
    return {(a, b): (r, s) for a, b, r, s in zip(df["v1"], df["v2"], df["r2"], df["status"])}


def finngen_r2_cached(pairs: list[tuple[str, str]], threads: int = 8) -> list[tuple[float, str]]:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    cache = _cache_load()
    keys = [(_unordered(a), _unordered(b)) for a, b in pairs]
    todo = sorted({k for k in keys if k not in cache})
    new = []
    if todo:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(threads) as ex:  # remote tabix calls are I/O bound
            for k, res in zip(todo, ex.map(lambda kk: finngen_r2(*kk), todo)):
                cache[k] = res
                new.append((k[0], k[1], *res))
    out = [cache[k] for k in keys]
    if new:
        with open(CACHE_FILE, "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            if fh.tell() == 0:
                fh.write("v1\tv2\tr2\tstatus\n")
            for row in new:
                fh.write("\t".join(map(str, row)) + "\n")
            fcntl.flock(fh, fcntl.LOCK_UN)
    return out


# --------------------------------------------------------------------------------------------------
# reference panel (1000G 30x; data agent's QC'd variant tables and genotype loader)
# --------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def pools() -> dict[str, list[str]]:
    from data.pools import load_pools
    return load_pools()


def standardize(G: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """G int [n, p] ALT counts -> (X float64 [n,p] with mean 0 and population variance 1, af, sd).
    Monomorphic columns are returned as zeros (callers filter them by MAF first)."""
    G = np.asarray(G, dtype=np.float64)
    mu = G.mean(axis=0)
    sd = G.std(axis=0)
    X = np.zeros_like(G)
    ok = sd > 0
    X[:, ok] = (G[:, ok] - mu[ok]) / sd[ok]
    return X, mu / 2.0, sd


def panel_r2(chrom: str, positions_pairs: list[tuple[str, str]], samples: list[str]) -> list[float]:
    """r^2 in a reference panel (sample list) between variant pairs given as ids (any orientation).
    NaN when a variant is absent from the QC'd panel or monomorphic in it."""
    from data.genotypes import load_genotypes
    ids = sorted({v for p in positions_pairs for v in p}, key=lambda v: int(variant_key(v).split(":")[1]))
    if not ids:
        return []
    poss = [int(variant_key(v).split(":")[1]) for v in ids]
    G, vt = load_genotypes(chrom, min(poss) - 1, max(poss), samples)  # BED convention
    key2col = {_unordered(v): j for j, v in enumerate(vt["variant_id"].tolist())}
    X, _, sd = standardize(G)
    n = X.shape[0]
    out = []
    for a, b in positions_pairs:
        ia, ib = key2col.get(_unordered(a)), key2col.get(_unordered(b))
        if ia is None or ib is None or sd[ia] == 0 or sd[ib] == 0:
            out.append(float("nan"))
            continue
        r = float(X[:, ia] @ X[:, ib] / n)
        out.append(r * r)
    return out
