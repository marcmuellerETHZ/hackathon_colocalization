"""Sample pools (DESIGN §2.1).

Pools (1000G 30x, phase-3 unrelated samples present in data/processed/1kg/*.psam):
  A        254 NFE (CEU, GBR, IBS, TSI), target-population haplotypes (mosaic source)
  B        150 NFE, external LD reference
  FIN       99 FIN (all), cross-population GWAS world (L6)
  ASW_ACB  157 ASW + ACB (all), admixed component (L3)
  L6c       99 NFE drawn from pool A (= NFE not in pool B), control for L6

A and B partition the 404 NFE samples, stratified by subpopulation (largest-remainder
allocation of the 150 B slots, ties broken by subpopulation name). Randomness: SPLIT_SEED
(20261026) via src/common/seeds.py, one child generator per (purpose, subpopulation).

Output: long TSV (one row per sample x pool; a sample may appear in A and L6c):
  sample_id  pop  superpop  pool

Usage:  python -m data.pools [--panel ...] [--psam ...] [--out ...] [--copy-to datasets/splits/pools_v1.tsv]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from common.seeds import SPLIT_SEED, rng

ROOT = Path(__file__).resolve().parents[2]
NFE_POPS = ("CEU", "GBR", "IBS", "TSI")
ADMIX_POPS = ("ASW", "ACB")
N_POOL_B = 150
N_L6C = 99
POOL_ORDER = ("A", "B", "FIN", "ASW_ACB", "L6c")
VERSION = "pools_v1"


def largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """Allocate `total` slots proportionally to `counts` (Hamilton method).
    Ties in the fractional remainder are broken by key name (alphabetical), deterministic."""
    n = sum(counts.values())
    assert 0 <= total <= n
    quota = {k: total * v / n for k, v in counts.items()}
    alloc = {k: int(np.floor(q)) for k, q in quota.items()}
    rest = total - sum(alloc.values())
    order = sorted(counts, key=lambda k: (-(quota[k] - alloc[k]), k))
    for k in order[:rest]:
        alloc[k] += 1
    assert sum(alloc.values()) == total and all(alloc[k] <= counts[k] for k in counts)
    return alloc


def stratified_draw(df: pd.DataFrame, n: int, purpose: str, seed: int = SPLIT_SEED) -> pd.Index:
    """Draw n rows of df (index = sample_id, column pop) stratified by pop. Deterministic:
    samples are sorted by ID before permuting with rng(seed, VERSION, purpose, pop)."""
    counts = df.groupby("pop").size().to_dict()
    alloc = largest_remainder(counts, n)
    chosen = []
    for pop in sorted(alloc):
        ids = np.array(sorted(df.index[df["pop"] == pop]))
        perm = rng(seed, VERSION, purpose, pop).permutation(len(ids))
        chosen.extend(ids[perm[: alloc[pop]]])
    return pd.Index(sorted(chosen))


def make_pools(panel: pd.DataFrame, psam_ids: list[str], seed: int = SPLIT_SEED) -> pd.DataFrame:
    panel = panel[panel["sample_id"].isin(set(psam_ids))].set_index("sample_id")
    nfe = panel[panel["pop"].isin(NFE_POPS)]
    b = stratified_draw(nfe, N_POOL_B, "poolB", seed)
    a = nfe.index.difference(b)
    l6c = stratified_draw(nfe.loc[a], N_L6C, "L6c", seed)
    fin = panel.index[panel["pop"] == "FIN"]
    adm = panel.index[panel["pop"].isin(ADMIX_POPS)]
    members = {"A": a, "B": b, "FIN": fin, "ASW_ACB": adm, "L6c": l6c}
    rows = []
    for pool in POOL_ORDER:
        sub = panel.loc[sorted(members[pool]), ["pop", "superpop"]].reset_index()
        sub["pool"] = pool
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)[["sample_id", "pop", "superpop", "pool"]]
    return out


def read_panel(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=["sample_id", "pop", "superpop", "sex"])


def read_psam_ids(path: Path) -> list[str]:
    return pd.read_csv(path, sep="\t")["#IID"].tolist()


def load_pools(path: Path | str = ROOT / "datasets" / "splits" / "pools_v1.tsv") -> dict[str, list[str]]:
    """Pool name -> sorted list of sample IDs."""
    df = pd.read_csv(path, sep="\t")
    return {p: sorted(g["sample_id"]) for p, g in df.groupby("pool", sort=False)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=ROOT / "data/processed/1kg/panel.tsv", type=Path)
    ap.add_argument("--psam", default=ROOT / "data/processed/1kg/chr22.psam", type=Path)
    ap.add_argument("--out", default=ROOT / "data/processed/pools/pools_v1.tsv", type=Path)
    ap.add_argument("--copy-to", default=ROOT / "datasets/splits/pools_v1.tsv", type=Path)
    args = ap.parse_args()
    pools = make_pools(read_panel(args.panel), read_psam_ids(args.psam))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pools.to_csv(args.out, sep="\t", index=False)
    if args.copy_to:
        args.copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.out, args.copy_to)
    print(pools.groupby(["pool", "pop"], sort=False).size().to_string())


if __name__ == "__main__":
    main()
