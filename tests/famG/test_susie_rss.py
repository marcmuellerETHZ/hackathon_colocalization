"""Family G: batched GPU SuSiE-RSS -- identity with susieR 0.14.2 on toy loci, and internal consistency checks."""
import os
import subprocess

import numpy as np
import pytest
import torch

from famag_testutils import PROJECT, RSCRIPT
from famag_testutils import ar_genotypes, toy_locus, write_store
from famG.brent import brent_fmin
from famG.precision import assert_no_tf32
from famG.susie_rss import SusieRSSConfig, susie_rss_batch
from famG.validation.compare_identity import load_r_results
from famG.validation.make_loci import LocusStore


@pytest.fixture(scope="module")
def toy_store(tmp_path_factory):
    rng = np.random.default_rng(20261029)
    loci = []
    specs = [((), (), 200), ((40,), (9.0,), 150), ((10, 120), (7.0, -11.0), 250), ((60, 61, 200), (6.0, 6.0, -8.0), 300),
             ((5,), (4.0,), 120), ((100,), (20.0,), 400)]
    for c, lam, p in specs:
        X, R, z1, z2 = toy_locus(rng, 800, p, c, lam, c, lam)
        loci.append(dict(R=R.astype(np.float32).astype(np.float64), z1=z1, z2=z2, n=20000.0, n2=1000.0))
    d = tmp_path_factory.mktemp("toyG")
    write_store(d, loci)
    return d


def _run_r(store_dir, n_loci):
    env = dict(os.environ, R_PROFILE_USER="/dev/null")
    env.pop("R_LIBS", None)
    (store_dir / "r").mkdir(exist_ok=True)
    for t in (1, 2):
        subprocess.run([str(RSCRIPT), str(PROJECT / "src/famG/validation/run_susie014.R"), str(store_dir),
                        str(store_dir / "r" / f"t{t}_c0"), "1", str(n_loci), str(t), "100", "1"],
                       check=True, env=env, capture_output=True, timeout=600)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_identity_vs_susieR_0142(toy_store, has_r, cuda, dtype):
    st = LocusStore(toy_store)
    if not (toy_store / "r" / "t1_c0.json").exists():
        _run_r(toy_store, len(st))
    for t in (1, 2):
        rres = load_r_results(toy_store / "r", st, t)
        zs = [st.z(i, t) for i in range(len(st))]
        lds = [st.R(i).astype(np.float32 if dtype == torch.float32 else np.float64) for i in range(len(st))]
        ns = [float(st.row(i).n if t == 1 else st.row(i).n2) for i in range(len(st))]
        res = susie_rss_batch(zs, lds, ns, "dense", SusieRSSConfig(ld_dtype=dtype))
        for i in range(len(st)):
            q = rres[st.row(i).locus_id]
            tol = 1e-6 if dtype == torch.float64 else 1e-3
            assert np.abs(res[i]["pip"] - q["pip"]).max() < tol
            assert res[i]["niter"] == int(q["niter"])
            assert sorted(map(tuple, res[i]["sets"]["cs"])) == sorted(tuple(np.atleast_1d(c).tolist()) for c in q["cs"])


def test_known_answers(cuda):
    rng = np.random.default_rng(1)
    X = ar_genotypes(rng, 1000, 200)
    _, R, z_sig, _ = toy_locus(rng, 1000, 200, (77,), (15.0,), X=X)
    _, _, z_null, _ = toy_locus(rng, 1000, 200, X=X)
    res = susie_rss_batch([z_sig, z_null], [R.astype(np.float32)] * 2, [50000.0] * 2)
    assert res[0]["pip"][77] > 0.5 and 77 in res[0]["sets"]["cs"][0]           # positive control
    assert len(res[1]["sets"]["cs"]) == 0 and res[1]["pip"].max() < 0.2        # negative control


def test_padding_and_factor_equivalence(cuda):
    """Bucketed (padded) fits equal stand-alone fits; factor LD X equals dense R = X'X/n."""
    rng = np.random.default_rng(2)
    loci = []
    for p in (180, 190, 200):
        X = ar_genotypes(rng, 300, p).astype(np.float32)
        _, _, z, _ = toy_locus(rng, 300, p, (p // 2,), (10.0,), X=X.astype(np.float64))
        loci.append((X, z))
    cfg = SusieRSSConfig(ld_dtype=torch.float64, bucket_ratio=1.2)
    joint = susie_rss_batch([z for _, z in loci], [X for X, _ in loci], [5000.0] * 3, "factor", cfg)
    for k, (X, z) in enumerate(loci):
        alone = susie_rss_batch([z], [X], [5000.0], "factor", cfg)[0]
        Xd = X.astype(np.float64)
        dense = susie_rss_batch([z], [Xd.T @ Xd / Xd.shape[0]], [5000.0], "dense", cfg)[0]
        assert np.abs(joint[k]["pip"] - alone["pip"]).max() < 1e-7   # Brent near flat optimum: ~1e-9 rounding
        assert np.abs(dense["pip"] - alone["pip"]).max() < 1e-8


def test_triton_vs_torch_brent(cuda):
    rng = np.random.default_rng(3)
    X = ar_genotypes(rng, 500, 150)
    zs = [toy_locus(rng, 500, 150, (20, 90), (8.0, -6.0), X=X)[2] for _ in range(5)]
    R = X.T @ X / 500
    a = susie_rss_batch(zs, [R] * 5, [10000.0] * 5, "dense", SusieRSSConfig(ld_dtype=torch.float64, use_triton=True))
    b = susie_rss_batch(zs, [R] * 5, [10000.0] * 5, "dense", SusieRSSConfig(ld_dtype=torch.float64, use_triton=False))
    for x, y in zip(a, b):
        assert np.abs(x["pip"] - y["pip"]).max() < 1e-8
        assert np.allclose(x["V"], y["V"], rtol=1e-5, atol=1e-12)


def test_brent_matches_r_optimize(has_r):
    from famag_testutils import run_r
    fs = [lambda x: (x - 1.234) ** 2, lambda x: torch.sin(x) + 0.1 * x, lambda x: torch.abs(x + 3.3) + 0.01 * x ** 2]
    xs = brent_fmin(lambda x: torch.stack([fs[i](x[i]) for i in range(3)]), -30.0, 15.0, 3, "cpu")
    out = run_r('fs <- list(function(x) (x-1.234)^2, function(x) sin(x)+0.1*x, function(x) abs(x+3.3)+0.01*x^2);'
                'cat(sapply(fs, function(f) sprintf("%.17g", optim(0, f, method="Brent", lower=-30, upper=15)$par)))')
    ref = np.array([float(v) for v in out.split()])
    assert np.array_equal(xs.numpy(), ref)


def test_tf32_disabled(cuda):
    assert_no_tf32()
    assert torch.backends.cuda.matmul.allow_tf32 is False
