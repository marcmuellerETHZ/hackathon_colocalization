"""Synthetic toy loci with known answers (no 1000G dependency) in the famG/validation store format."""
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
RSCRIPT = PROJECT / "env" / "renv" / "bin" / "Rscript"


def run_r(code: str, r_libs: str | None = None, timeout=600) -> str:
    env = dict(os.environ)
    env["R_PROFILE_USER"] = "/dev/null"
    if r_libs:
        env["R_LIBS"] = r_libs
    else:
        env.pop("R_LIBS", None)
    res = subprocess.run([str(RSCRIPT), "-e", code], capture_output=True, text=True, env=env, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"R failed: {res.stderr[-3000:]}")
    return res.stdout


def ar_genotypes(rng, n, p, phi=0.8):
    X = rng.standard_normal((n, p))
    for j in range(1, p):
        X[:, j] = phi * X[:, j - 1] + np.sqrt(1 - phi ** 2) * X[:, j]
    return (X - X.mean(0)) / X.std(0)


def toy_locus(rng, n_ind, p, causal1=(), lam1=(), causal2=(), lam2=(), rho=0.0, X=None):
    X = ar_genotypes(rng, n_ind, p) if X is None else X
    R = X.T @ X / n_ind
    l1 = np.zeros(p); l1[list(causal1)] = lam1
    l2 = np.zeros(p); l2[list(causal2)] = lam2
    u1 = rng.standard_normal(n_ind)
    u2 = rho * u1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n_ind)
    z1 = R @ l1 + X.T @ u1 / np.sqrt(n_ind)
    z2 = R @ l2 + X.T @ u2 / np.sqrt(n_ind)
    return X, R, z1, z2


def write_store(out: Path, loci: list[dict]):
    """loci: dicts with R (float64, p x p), z1, z2, n, n2, maf (optional)."""
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    offR = offz = 0
    with open(out / "R.bin", "wb") as fR, open(out / "z.bin", "wb") as fz, open(out / "maf.bin", "wb") as fm:
        for k, lc in enumerate(loci):
            p = len(lc["z1"])
            np.ascontiguousarray(lc["R"], dtype=np.float64).tofile(fR)
            np.concatenate([lc["z1"], lc["z2"]]).astype(np.float64).tofile(fz)
            np.asarray(lc.get("maf", np.full(p, 0.3)), dtype=np.float64).tofile(fm)
            rows.append(dict(locus_id=f"toy_{k:03d}", chrom=0, pos_start=0, pos_end=p, p=p, n=float(lc["n"]),
                             n2=float(lc["n2"]), ld_kind="dense", scenario="toy", R_offset=offR, X_offset=-1,
                             n_ref=0, z_offset=offz))
            offR += p * p
            offz += 2 * p
    (out / "X.bin").write_bytes(b"")
    pd.DataFrame(rows).to_csv(out / "index.tsv", sep="\t", index=False)
