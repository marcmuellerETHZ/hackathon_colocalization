"""coloc pairing on GPU vs R coloc 5.2.3 (coloc.bf_bf, coloc.abf), plus known-answer controls."""
import json
import tempfile

import numpy as np
import pytest
import torch

from famag_testutils import run_r
from famG.coloc import bf_bf_batch, coloc_abf, combine_abf


def _rand_bf(rng, k, p, strong=None):
    bf = rng.normal(0, 1, size=(k, p))
    if strong is not None:
        for r, j in enumerate(strong[:k]):
            bf[r, j] += 15 + 10 * rng.random()
    return bf


def test_bf_bf_matches_r(has_r):
    rng = np.random.default_rng(20261029)
    cases = []
    for t in range(12):
        p = int(rng.integers(5, 300))
        k1, k2 = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        s1 = rng.integers(0, p, size=k1).tolist()
        s2 = (s1 if t % 3 == 0 else rng.integers(0, p, size=k2).tolist())
        cases.append((_rand_bf(rng, k1, p, s1), _rand_bf(rng, k2, p, s2)))
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        fn = f"{d}/bf.json"
        json.dump([{"bf1": a.tolist(), "bf2": b.tolist()} for a, b in cases], open(fn, "w"))
        out = run_r(f"""
suppressPackageStartupMessages({{library(coloc); library(jsonlite)}})
cs <- fromJSON("{fn}", simplifyMatrix = TRUE)
res <- list()
for (i in seq_len(nrow(cs))) {{
  b1 <- cs$bf1[[i]]; b2 <- cs$bf2[[i]]
  if (!is.matrix(b1)) b1 <- matrix(b1, nrow = 1)
  if (!is.matrix(b2)) b2 <- matrix(b2, nrow = 1)
  colnames(b1) <- colnames(b2) <- paste0("s", seq_len(ncol(b1)))
  r <- suppressWarnings(coloc.bf_bf(b1, b2, p12 = 5e-6))$summary
  res[[i]] <- as.matrix(r[, c("idx1", "idx2", "PP.H0.abf", "PP.H1.abf", "PP.H2.abf", "PP.H3.abf", "PP.H4.abf")])
}}
cat("@@JSON@@", toJSON(res, digits = NA))
""")
    rres = json.loads(out.split("@@JSON@@")[-1])
    ours = bf_bf_batch([a for a, _ in cases], [b for _, b in cases], p12=5e-6)
    maxd = 0.0
    for i, r in enumerate(rres):
        r = np.asarray(r, dtype=float)
        for row in r:
            i1, i2 = int(row[0]) - 1, int(row[1]) - 1
            maxd = max(maxd, np.abs(ours[i][i1, i2] - row[2:]).max())
    assert maxd < 1e-10, maxd


def test_coloc_abf_matches_r(has_r):
    rng = np.random.default_rng(20261029)
    p = 200
    maf = rng.uniform(0.02, 0.5, p)
    N1, N2 = 50000, 800
    se1 = 1 / np.sqrt(2 * N1 * maf * (1 - maf))
    se2 = 1 / np.sqrt(2 * N2 * maf * (1 - maf))
    z1 = rng.normal(size=p); z1[50] += 9
    z2 = rng.normal(size=p); z2[50] += 7
    b1, b2 = z1 * se1, z2 * se2
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        fn = f"{d}/abf.json"
        json.dump(dict(b1=b1.tolist(), b2=b2.tolist(), v1=(se1 ** 2).tolist(), v2=(se2 ** 2).tolist(), maf=maf.tolist()), open(fn, "w"))
        out = run_r(f"""
suppressPackageStartupMessages({{library(coloc); library(jsonlite)}})
x <- fromJSON("{fn}")
snp <- paste0("s", seq_along(x$b1))
r <- coloc.abf(list(beta = x$b1, varbeta = x$v1, N = {N1}, MAF = x$maf, type = "quant", sdY = 1, snp = snp),
               list(beta = x$b2, varbeta = x$v2, N = {N2}, MAF = x$maf, type = "quant", sdY = 1, snp = snp),
               p12 = 5e-6)
cat("@@JSON@@", toJSON(as.numeric(r$summary[2:6]), digits = NA))
""")
    rpp = np.asarray(json.loads(out.split("@@JSON@@")[-1]))
    pp = coloc_abf(b1, se1, b2, se2, p12=5e-6)
    assert np.abs(pp - rpp).max() < 1e-10


def test_combine_abf_known_answers():
    p = 100
    l = np.zeros(p); l[3] = 30.0
    # same strong SNP in both -> H4
    pp = combine_abf(torch.tensor(l), torch.tensor(l), 1e-4, 1e-4, 5e-6).numpy()
    assert pp[4] > 0.99
    # different strong SNPs -> H3
    l2 = np.zeros(p); l2[60] = 30.0
    pp = combine_abf(torch.tensor(l), torch.tensor(l2), 1e-4, 1e-4, 5e-6).numpy()
    assert pp[3] > 0.99
    # nothing -> H0
    pp = combine_abf(torch.zeros(p, dtype=torch.float64), torch.zeros(p, dtype=torch.float64)).numpy()
    assert pp[0] > 0.97
    assert abs(pp.sum() - 1) < 1e-12
