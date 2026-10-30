"""Export validation loci for the G-mm identity test (R reference runner input) and load them for G-mm.

Each exported locus: <dir>/<id>/meta.json, z1.bin, z2.bin, R.bin (float64; R exactly as the coloc_susie_mm wrapper
builds it: src/baselines/locus.py:dense_ld, i.e. float64 X'X/n_ref or float64(R), symmetrised).
Optional window truncation (--max-p) keeps a contiguous block of variants (known-answer development loci only).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.seeds import rng as seeded_rng
from eval.io import LocusSetReader, read_locus_attrs


def dense_R(lc) -> np.ndarray:
    if lc.ld_type == "dense":
        R = np.asarray(lc.R, dtype=np.float64)
    else:
        X = np.asarray(lc.X_ref, dtype=np.float64)
        R = X.T @ X / float(lc.n_ref)
    return 0.5 * (R + R.T)


def locus_dict(lc, max_p: int | None = None, start: int = 0, dense: bool = False) -> dict:
    """G-mm input dict from an eval.io.LocusData (z per trait, n per trait = median, n_ref, LD as stored)."""
    sl = slice(start, start + max_p) if max_p else slice(None)
    z = [np.asarray(lc.z1, float)[sl], np.asarray(lc.z2, float)[sl]]
    n = [float(np.median(np.asarray(lc.n1, float))), float(np.median(np.asarray(lc.n2, float)))]
    d = {"id": lc.locus_id, "z": z, "n": n, "n_ref": int(lc.n_ref)}
    if dense or max_p:
        Rf = dense_R(lc)
        d["R"] = Rf[sl, sl] if max_p else Rf
    elif lc.ld_type == "dense":
        d["R"] = lc.R
    else:
        d["X"] = lc.X_ref
    return d


def export(d: dict, out: Path) -> None:
    o = out / d["id"]
    o.mkdir(parents=True, exist_ok=True)
    R = d["R"]
    p = len(d["z"][0])
    assert R.shape == (p, p) and np.isfinite(R).all(), d["id"]
    np.asarray(d["z"][0], "<f8").tofile(o / "z1.bin")
    np.asarray(d["z"][1], "<f8").tofile(o / "z2.bin")
    np.asfortranarray(np.asarray(R, "<f8")).T.tofile(o / "R.bin")   # column-major
    json.dump({"p": p, "n1": d["n"][0], "n2": d["n"][1], "n_ref": d["n_ref"]}, open(o / "meta.json", "w"))


def load_exported(dirpath: Path, ids: list[str]) -> list[dict]:
    out = []
    for i in ids:
        o = Path(dirpath) / i
        m = json.load(open(o / "meta.json"))
        p = m["p"]
        R = np.fromfile(o / "R.bin", "<f8").reshape(p, p).T
        out.append({"id": i, "z": [np.fromfile(o / "z1.bin", "<f8"), np.fromfile(o / "z2.bin", "<f8")],
                    "n": [m["n1"], m["n2"]], "n_ref": m["n_ref"], "R": R})
    return out


def pick_ids(set_dir: Path, cells: list[str], n_per_cell: int, key: str) -> list[tuple[str, str]]:
    """Seeded draw of n_per_cell loci per cell file, stratified over oracle-class prefixes in the id."""
    rng = seeded_rng(20261029, "famG_mm", key)
    picked = []
    for c in cells:
        f = set_dir / f"{c}.h5"
        ids = sorted(read_locus_attrs(f)["locus_id"].tolist())
        sel = rng.choice(len(ids), size=n_per_cell, replace=False)
        picked += [(str(f), ids[k]) for k in sorted(sel)]
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-dir", default="data/processed/sets/valid_grid_v1")
    ap.add_argument("--cells", default="L2_P1_r_p=0.0,L2_P3_rho=0.5,L4_P1_r_p=0.0,L4_P3_rho=0.5")
    ap.add_argument("--n-per-cell", type=int, default=25)
    ap.add_argument("--max-p", type=int, default=0)
    ap.add_argument("--key", default="identity")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    picked = pick_ids(Path(a.set_dir), a.cells.split(","), a.n_per_cell, a.key)
    ids = []
    by_file = {}
    for f, i in picked:
        by_file.setdefault(f, []).append(i)
    for f, lst in by_file.items():
        with LocusSetReader(f) as rd:
            for i in lst:
                lc = rd.read(i)
                d = locus_dict(lc, max_p=a.max_p or None, dense=True)
                export(d, out)
                ids.append(i)
    (out / "ids.txt").write_text("\n".join(ids) + "\n")
    print(f"exported {len(ids)} loci to {out}")


if __name__ == "__main__":
    main()
