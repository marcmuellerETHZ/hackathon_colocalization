"""Freeze the real-data test ID lists (DECISIONS P3-14; test lock DESIGN §7.3). IDs only: no data is read.

  python -m data.real.freeze_test_ids            # write the P3-14 subsamples + datasets/configs/real_test_ids_v1.sha256
  python -m data.real.freeze_test_ids --check    # verify every listed file against the frozen sha256

Inputs: data/processed/real/pairs/{R2,R3b,R4b}_test_ids.tsv (identifier rows written by pairs.py).
Subsamples (P3-14): R2 800 pairs, R3b all pairs, R4b 600 loci (each later run with in-sample and reference LD).
Selection: numpy Generator seeded with common.seeds.derive(TEST_SEED=20261028, "real_test_subsample_v1", arm),
`choice(n, k, replace=False)`, rows kept in their original file order. Output rows are the same identifier
columns as the input (`<arm>_test_ids_sub_v1.tsv`). The sha256 file is in `sha256sum` format (paths relative to
the project root), so `sha256sum -c datasets/configs/real_test_ids_v1.sha256` also verifies it.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common.seeds import TEST_SEED, rng
from data.real.config import OUT, ROOT

PAIRS = OUT / "pairs"
SHA_FILE = ROOT / "datasets" / "configs" / "real_test_ids_v1.sha256"
SUBSAMPLE = {"R2": 800, "R3b": None, "R4b": 600}   # None = all test pairs (P3-14)
ID_COLS = ["locus_id", "arm", "dataset1", "dataset2", "gene_id", "split", "chr", "win_lo", "win_hi"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def subsample(ids: pd.DataFrame, arm: str, k: int | None) -> pd.DataFrame:
    """Seeded subsample of k rows (all rows if k is None or k >= n), original order kept."""
    if (ids["split"] != "test").any() or (ids["arm"] != arm).any():
        raise ValueError(f"{arm}: test ID list must contain only split == 'test' rows of arm {arm}")
    if ids["locus_id"].duplicated().any():
        raise ValueError(f"{arm}: duplicate locus_id in the test ID list")
    n = len(ids)
    if k is None or k >= n:
        return ids.reset_index(drop=True)
    idx = np.sort(rng(TEST_SEED, "real_test_subsample_v1", arm).choice(n, size=k, replace=False))
    return ids.iloc[idx].reset_index(drop=True)


def freeze(pairs_dir: Path = PAIRS, sha_file: Path = SHA_FILE, root: Path = ROOT, force: bool = False) -> dict:
    if sha_file.exists() and not force:
        raise FileExistsError(f"{sha_file} exists: the test ID lists are frozen (use --check)")
    lines, out = [], {}
    for arm, k in SUBSAMPLE.items():
        src = pairs_dir / f"{arm}_test_ids.tsv"
        ids = pd.read_csv(src, sep="\t", dtype=str, usecols=ID_COLS)[ID_COLS]  # identifier columns only
        sub = subsample(ids, arm, k)
        dst = pairs_dir / f"{arm}_test_ids_sub_v1.tsv"
        sub.to_csv(dst, sep="\t", index=False)
        for f in (src, dst):
            lines.append(f"{sha256(f)}  {os.path.relpath(f, root)}")  # symlinked data/: no resolve
        out[arm] = {"n_test": len(ids), "n_sub": len(sub), "k_requested": k}
    sha_file.parent.mkdir(parents=True, exist_ok=True)
    sha_file.write_text("\n".join(lines) + "\n")
    return out


def check(sha_file: Path = SHA_FILE, root: Path = ROOT) -> list[str]:
    """Return the list of mismatching/missing files (empty = all frozen hashes match)."""
    bad = []
    for line in sha_file.read_text().splitlines():
        if not line.strip():
            continue
        h, rel = line.split(None, 1)
        f = root / rel.strip()
        if not f.exists() or sha256(f) != h:
            bad.append(rel.strip())
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen sha256 file")
    a = ap.parse_args()
    if a.check:
        bad = check()
        print("real test IDs frozen sha256:", "OK" if not bad else f"MISMATCH {bad}")
        raise SystemExit(1 if bad else 0)
    print(freeze(force=a.force))


if __name__ == "__main__":
    main()
