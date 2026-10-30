"""Throughput benchmark for ExactColoc (loci/s at M = 50 / 100).

    PYTHONPATH=src python -m famB.bench --p 2000 --n 64 --M 50 100 [--compile] [--dtype float64]

Loci are synthetic L0-like (AR(1) genotype LD, n = 500, z = R lam + X'u/sqrt(n)) with S/D/W mixtures;
reported time is the wall-clock of fit_batch (screen + EB + enumeration + MC events), after one warm-up batch,
with torch.cuda.synchronize.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from common.seeds import VALID_SEED, rng as seeded_rng
from eval.io import LocusData

from .method import ExactColoc


def synthetic_loci(n, p, seed_key=0, n_ref=500):
    g = seeded_rng(VALID_SEED, "famB.bench", seed_key)
    out = []
    for i in range(n):
        X = g.standard_normal((n_ref, p))
        ar = float(g.uniform(0.5, 0.95))
        for j in range(1, p):
            X[:, j] = ar * X[:, j - 1] + math.sqrt(1 - ar * ar) * X[:, j]
        X = (X - X.mean(0)) / X.std(0)
        R = X.T @ X / n_ref
        lam1 = np.zeros(p); lam2 = np.zeros(p)
        j, k = g.choice(p, 2, replace=False)
        kind = i % 3
        lam1[j] = g.uniform(5, 30)
        if kind == 0: lam2[j] = g.uniform(4, 20)
        if kind == 1: lam2[k] = g.uniform(4, 20)
        z1 = R @ lam1 + X.T @ g.standard_normal(n_ref) / math.sqrt(n_ref)
        z2 = R @ lam2 + X.T @ g.standard_normal(n_ref) / math.sqrt(n_ref)
        f = g.uniform(0.05, 0.5, p)
        se1 = 1 / np.sqrt(2 * 1e5 * f * (1 - f)); se2 = 1 / np.sqrt(2 * 1e3 * f * (1 - f))
        out.append(LocusData(locus_id=f"B{i}", variant_id=np.array([f"chr1:{q + 1}:A:C" for q in range(p)]),
                             pos=np.arange(1, p + 1), z1=z1, z2=z2, beta1=z1 * se1, se1=se1, beta2=z2 * se2, se2=se2,
                             n1=np.full(p, 1e5), n2=np.full(p, 1e3), maf1=f, maf2=f, maf_ref=f, ld_type="dense",
                             n_ref=n_ref, R=R.astype(np.float32), rho_hat=0.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=2000)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--M", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--mode", default="gaussian")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--n_mc", type=int, default=16384)
    ap.add_argument("--eb", type=int, default=1)
    a = ap.parse_args()
    loci = synthetic_loci(a.n, a.p)
    res = []
    for M in a.M:
        m = ExactColoc(mode=a.mode, M=M, compile=a.compile, dtype=a.dtype, n_mc=a.n_mc, eb=bool(a.eb))
        m.fit_batch(synthetic_loci(2, a.p, seed_key=1))  # warm-up (CUDA context, compile)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = m.fit_batch(loci)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ok = sum(o["status"] == "ok" for o in out)
        prep = float(np.mean([json.loads(o["extra"])["t_prep"] for o in out if o["status"] == "ok"]))
        r = dict(M=M, p=a.p, n=a.n, mode=a.mode, compile=a.compile, dtype=a.dtype, eb=a.eb, n_mc=a.n_mc,
                 wall_s=round(dt, 3), loci_per_s=round(a.n / dt, 3), ok=ok, mean_cpu_prep_s=round(prep, 4),
                 gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")
        print(json.dumps(r), flush=True)
        res.append(r)


if __name__ == "__main__":
    main()
