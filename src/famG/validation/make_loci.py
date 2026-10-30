"""Simulate validation loci from real 1000G EUR LD windows (identity tests and timing for Families G/A).

z_t = R lambda_t + e_t with e_t = G^T u_t / sqrt(n) (G: standardized EUR genotypes, so Cov(e) = R exactly).
'mismatch' loci draw z from the LD of a random half of the sample and give the full-sample LD to the
method (realistic in-sample-vs-reference noise; used to probe non-convergence agreement).

Outputs (one directory, never one file per locus):
  index.tsv   one row per locus (offsets into the binary files)
  R.bin       float64 LD matrices given to the methods (dense: float64(float32(R)); factor: X32^T X32 / n)
  X.bin       float32 standardized genotype factors (factor loci only), row-major (n_ref, p)
  z.bin       float64 z-scores, two traits per locus (z1 then z2)
  maf.bin     float64 in-sample (EUR) minor allele frequencies, one per variant (offset = z_offset / 2)
  truth.json  causal indices and lambdas (never read by the methods)

Usage: python -m famG.validation.make_loci --out DIR --n-loci 140 --p-set 200,500,1000,2000 \
           --factor-frac 0.25 --mismatch-frac 0.15 --key identity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.seeds import VALID_SEED, rng as seeded_rng
from famG.validation.genotypes import chrom_variants, window


def simulate_effects(rng, p, n_causal, lam_range=(4.0, 25.0)):
    idx = rng.choice(p, size=n_causal, replace=False) if n_causal else np.array([], dtype=int)
    lam = np.exp(rng.uniform(np.log(lam_range[0]), np.log(lam_range[1]), size=n_causal)) * rng.choice([-1, 1], size=n_causal)
    return idx, lam


def make(out: Path, n_loci: int, p_set: list[int], factor_frac: float, mismatch_frac: float, key: str,
         two_traits: bool = True, dense_f64: bool = False, n_choices=(2000, 20000, 200000), lam_range=(4.0, 25.0),
         k_probs=(0.1, 0.4, 0.3, 0.2)):
    out.mkdir(parents=True, exist_ok=True)
    rng = seeded_rng(VALID_SEED, "famAG", key)
    rows, truth = [], {}
    fR = open(out / "R.bin", "wb")
    fX = open(out / "X.bin", "wb")
    fz = open(out / "z.bin", "wb")
    fm = open(out / "maf.bin", "wb")
    offR = offX = offz = 0
    for k in range(n_loci):
        p = int(p_set[k % len(p_set)])
        while True:
            chrom = int(rng.integers(1, 23))
            tab = chrom_variants(chrom)
            start = int(rng.integers(0, max(1, len(tab) - 3 * p - 100)))
            try:
                G, sub = window(chrom, start, p)
                break
            except ValueError:
                continue
        n = G.shape[0]
        kind = "factor" if rng.random() < factor_frac else "dense"
        scen = "mismatch" if rng.random() < mismatch_frac else "consistent"
        if scen == "mismatch":
            half = np.sort(rng.choice(n, size=n // 2, replace=False))
            Gs = G[half]
            sd = Gs.std(0)
            sd[sd == 0] = 1.0
            Gs = (Gs - Gs.mean(0)) / sd
        else:
            Gs = G
        Rs = Gs.T @ Gs / Gs.shape[0]
        zs, tr = [], {}
        for t in (1, 2) if two_traits else (1,):
            K = int(rng.choice(len(k_probs), p=k_probs))
            idx, lam = simulate_effects(rng, p, K, lam_range)
            lam_vec = np.zeros(p)
            lam_vec[idx] = lam
            u = rng.standard_normal(Gs.shape[0])
            z = Rs @ lam_vec + Gs.T @ u / np.sqrt(Gs.shape[0])
            zs.append(z)
            tr[f"causal{t}"] = idx.tolist()
            tr[f"lambda{t}"] = lam.tolist()
        if not two_traits:
            zs.append(np.zeros(p))
        if kind == "dense" and dense_f64:
            R_given = G.T @ G / n          # exact float64 (PSD to rounding); used by the Family A identity test
            x_off, n_ref = -1, n
        elif kind == "dense":
            R32 = (G.T @ G / n).astype(np.float32)
            R_given = R32.astype(np.float64)
            x_off, n_ref = -1, n
        else:
            X32 = G.astype(np.float32)
            Xd = X32.astype(np.float64)
            R_given = Xd.T @ Xd / n
            np.ascontiguousarray(X32).tofile(fX)
            x_off, n_ref = offX, n
            offX += X32.size
        R_given = 0.5 * (R_given + R_given.T)
        np.ascontiguousarray(R_given).tofile(fR)
        np.concatenate(zs).astype(np.float64).tofile(fz)
        sub.maf.to_numpy(np.float64).tofile(fm)
        lid = f"{key}_{k:04d}"
        rows.append(dict(locus_id=lid, chrom=chrom, pos_start=int(sub.POS.iloc[0]), pos_end=int(sub.POS.iloc[-1]),
                         p=p, n=float(rng.choice(n_choices)), n2=float(rng.choice(n_choices)), ld_kind=kind, scenario=scen,
                         R_offset=offR, X_offset=x_off, n_ref=n_ref, z_offset=offz))
        truth[lid] = tr
        offR += p * p
        offz += 2 * p
    fR.close(); fX.close(); fz.close(); fm.close()
    pd.DataFrame(rows).to_csv(out / "index.tsv", sep="\t", index=False)
    (out / "truth.json").write_text(json.dumps(truth))
    return pd.DataFrame(rows)


class LocusStore:
    """Reader for a directory written by make()."""

    def __init__(self, d: Path):
        self.d = Path(d)
        self.index = pd.read_csv(self.d / "index.tsv", sep="\t")
        self._R = np.memmap(self.d / "R.bin", dtype=np.float64, mode="r")
        self._z = np.memmap(self.d / "z.bin", dtype=np.float64, mode="r")
        self._maf = np.memmap(self.d / "maf.bin", dtype=np.float64, mode="r") if (self.d / "maf.bin").exists() else None
        xs = (self.d / "X.bin").stat().st_size
        self._X = np.memmap(self.d / "X.bin", dtype=np.float32, mode="r") if xs > 0 else None

    def __len__(self):
        return len(self.index)

    def row(self, i):
        return self.index.iloc[i]

    def R(self, i) -> np.ndarray:
        r = self.row(i)
        p = int(r.p)
        return np.array(self._R[int(r.R_offset): int(r.R_offset) + p * p]).reshape(p, p)

    def X(self, i) -> np.ndarray:
        r = self.row(i)
        p, n = int(r.p), int(r.n_ref)
        return np.array(self._X[int(r.X_offset): int(r.X_offset) + n * p]).reshape(n, p)

    def z(self, i, t=1) -> np.ndarray:
        r = self.row(i)
        p = int(r.p)
        o = int(r.z_offset) + (t - 1) * p
        return np.array(self._z[o: o + p])

    def maf(self, i) -> np.ndarray:
        r = self.row(i)
        o = int(r.z_offset) // 2
        return np.array(self._maf[o: o + int(r.p)])

    def ld_input(self, i, dense_dtype=np.float32):
        """(kind, array) as given to the GPU: dense R cast to float32 (lossless here) or X float32."""
        r = self.row(i)
        if r.ld_kind == "factor":
            return "factor", self.X(i)
        return "dense", self.R(i).astype(dense_dtype)   # float32 cast is lossless unless written with --dense-f64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-loci", type=int, default=140)
    ap.add_argument("--p-set", default="200,500,1000,2000")
    ap.add_argument("--factor-frac", type=float, default=0.25)
    ap.add_argument("--mismatch-frac", type=float, default=0.15)
    ap.add_argument("--key", default="identity")
    ap.add_argument("--dense-f64", action="store_true")
    a = ap.parse_args()
    df = make(Path(a.out), a.n_loci, [int(x) for x in a.p_set.split(",")], a.factor_frac, a.mismatch_frac, a.key,
              dense_f64=a.dense_f64)
    print(df.groupby(["ld_kind", "scenario"]).size())


if __name__ == "__main__":
    main()
