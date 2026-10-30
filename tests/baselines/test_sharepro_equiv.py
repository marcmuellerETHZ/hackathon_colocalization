"""In-process SharePro (src/baselines/sharepro.py) == official CLI (env/src/SharePro_coloc @ 059caab).

(1) shipped README example (RSPO3 pQTL x eBMD): in-process shares equal the shipped doc/res.sharepro.txt;
(2) a toy locus: in-process groups and shares equal those of the CLI run on the same data written to files.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baselines.sharepro import SHAREPRO_PY, run_sharepro

SP = SHAREPRO_PY.parents[2]


@pytest.fixture
def tmpdir_claude():
    root = Path(f"/tmp/claude-{os.getuid()}")
    root.mkdir(exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="sharepro_test_", dir=root))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _fmt(eff, eff_share, snps):
    return {"/".join(snps[j] for j in eff[k]): float(f"{eff_share[k]:.4}") for k in eff}


def test_shipped_example():
    z1 = pd.read_csv(SP / "dat" / "BMD_bse.txt", sep="\t")
    z2 = pd.read_csv(SP / "dat" / "RSPO3_bse.txt", sep="\t")
    ld = pd.read_csv(SP / "dat" / "RSPO3.ld", sep=r"\s+", header=None).values
    out = run_sharepro([z1.BETA.values, z2.BETA.values], [z1.SE.values, z2.SE.values], [z1.N.values, z2.N.values],
                       ld, K=10)
    got = _fmt(out["eff"], out["eff_share"], list(z1.SNP))
    ref = pd.read_csv(SP / "doc" / "res.sharepro.txt", sep="\t")
    exp = dict(zip(ref.cs, ref.share.astype(float)))
    assert got == exp


def test_cli_equivalence_toy(controls, tmpdir_claude):
    for key in [("shared", 0), ("distinct", 0), ("null2", 0)]:
        L = controls[key]
        snps = [f"v{i}" for i in range(len(L.z1))]
        for t, (b, s, n) in enumerate([(L.beta1, L.se1, L.n1), (L.beta2, L.se2, L.n2)]):
            pd.DataFrame({"SNP": snps, "BETA": b, "SE": s, "N": n}).to_csv(tmpdir_claude / f"z{t}.txt", sep="\t",
                                                                         index=False, float_format="%.17g")
        R = np.asarray(L.R, dtype=np.float64)
        np.savetxt(tmpdir_claude / "ld.txt", R, fmt="%.17g")
        r = subprocess.run([sys.executable, "-I", "-W", "ignore::SyntaxWarning", str(SHAREPRO_PY), "--z",
                            str(tmpdir_claude / "z0.txt"), str(tmpdir_claude / "z1.txt"), "--ld",
                            str(tmpdir_claude / "ld.txt"), "--save", str(tmpdir_claude / "res"), "--K", "10"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        cli = pd.read_csv(tmpdir_claude / "res.sharepro.txt", sep="\t")
        out = run_sharepro([L.beta1, L.beta2], [L.se1, L.se2], [L.n1, L.n2], R, K=10)
        got = _fmt(out["eff"], out["eff_share"], snps)
        exp = dict(zip(cli.cs, cli.share.astype(float)))
        assert got == exp, (key, got, exp)
