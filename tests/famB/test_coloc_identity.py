"""Kill-test B1 correctness / DESIGN Sec. 7.1(4): coloc mode (full-p single-SNP enumeration through the
general engine) reproduces R coloc 5.2.3 coloc.abf to < 1e-3 on 200 loci."""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from famB.method import ExactColoc
from toy_data import ar_genotypes, make_locus

ROOT = Path(__file__).resolve().parents[2]
RSCRIPT = ROOT / "env/renv/bin/Rscript"
HERE = Path(__file__).resolve().parent


def _loci(n_loci=200, seed=20261026):
    rng = np.random.default_rng(seed)
    loci = []
    kinds = ["null", "t1", "t2", "shared", "distinct", "weak_shared"]
    for i in range(n_loci):
        p = int(rng.integers(30, 600))
        X = ar_genotypes(rng, 300, p, ar=float(rng.uniform(0.3, 0.95)))
        kind = kinds[i % len(kinds)]
        lam1 = np.zeros(p); lam2 = np.zeros(p)
        j, k = rng.choice(p, 2, replace=False)
        a1, a2 = rng.uniform(3, 15) * rng.choice([-1, 1]), rng.uniform(3, 12) * rng.choice([-1, 1])
        if kind == "t1": lam1[j] = a1
        if kind == "t2": lam2[k] = a2
        if kind == "shared": lam1[j] = a1; lam2[j] = a2
        if kind == "distinct": lam1[j] = a1; lam2[k] = a2
        if kind == "weak_shared": lam1[j] = a1 / 3; lam2[j] = a2 / 3
        N1 = float(rng.choice([2e4, 1e5, 4e5])); N2 = float(rng.choice([300, 1000, 5000]))
        loci.append(make_locus(f"L{i:03d}", X, lam1, lam2, rho=0.0, N1=N1, N2=N2, rng=rng))
    return loci


def _r_coloc(loci, tmp, p1, p2, p12):
    rows = []
    for lo in loci:
        rows.append(pd.DataFrame(dict(locus=lo.locus_id, snp=lo.variant_id, beta1=lo.beta1, varbeta1=lo.se1 ** 2,
                                      beta2=lo.beta2, varbeta2=lo.se2 ** 2, MAF=lo.maf1, N1=lo.n1, N2=lo.n2)))
    fin = tmp / "in.tsv"; fout = tmp / "out.tsv"
    pd.concat(rows).to_csv(fin, sep="\t", index=False, float_format="%.17g")
    env = dict(os.environ, R_PROFILE_USER="/dev/null")
    r = subprocess.run([str(RSCRIPT), str(HERE / "coloc_abf_batch.R"), str(fin), str(fout), str(p1), str(p2), str(p12)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    return pd.read_csv(fout, sep="\t").set_index("locus")


@pytest.mark.slow
@pytest.mark.parametrize("p12", [1e-5, 5e-6])
def test_coloc_mode_matches_R_coloc_abf(p12, tmp_path):
    loci = _loci()
    ref = _r_coloc(loci, tmp_path, 1e-4, 1e-4, p12)
    m = ExactColoc(mode="coloc", W=0.15, p1=1e-4, p2=1e-4, p12=p12, n_mc=2048)
    out = m.fit_batch(loci)
    worst = 0.0
    for lo, o in zip(loci, out):
        assert o["status"] == "ok", o["extra"]
        e = json.loads(o["extra"])
        assert e["PP"]["H34"] == 0.0
        assert int(ref.loc[lo.locus_id, "nsnps"]) == lo.p
        for h in ("H0", "H1", "H2", "H3", "H4"):
            d = abs(e["PP"][h] - float(ref.loc[lo.locus_id, h]))
            worst = max(worst, d)
    print(f"p12={p12}: max |dPP| over 200 loci x 5 hypotheses = {worst:.3e}")
    assert worst < 1e-3
