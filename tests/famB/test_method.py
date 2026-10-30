"""Method API (INTERFACES Sec. 3/4) and known-direction controls (DESIGN Sec. 7.1(9))."""
import json

import numpy as np
import pytest
import torch

from famB.enum import TermSet, enumerate_batch, get_evaluator
from famB.method import ExactColoc
from toy_data import ar_genotypes, make_locus

COLS = {"locus_id", "method", "s", "P_S", "P_D", "P_W", "E1", "E2", "call", "status", "runtime_s", "extra"}


def _controls(rng, factor=False):
    n, p = 500, 300
    X = ar_genotypes(rng, n, p, ar=0.6)
    X[:, 101] = X[:, 100]  # exact duplicate: r^2 = 1
    X = (X - X.mean(0)) / X.std(0)
    z0 = np.zeros(p)
    l_sh = z0.copy(); l_sh[100] = 30.0
    l_d1 = z0.copy(); l_d1[50] = 30.0
    l_d2 = z0.copy(); l_d2[250] = 30.0
    return [
        make_locus("shared_r2_1", X, l_sh, l_sh.copy(), rng=rng, factor=factor),
        make_locus("shared_dup_other", X, l_sh, np.where(np.arange(p) == 101, 30.0, 0.0), rng=rng, factor=factor),
        make_locus("distinct", X, l_d1, l_d2, rng=rng, factor=factor),
        make_locus("null", X, z0, z0, rng=rng, factor=factor),
        make_locus("t1_only", X, l_d1, z0, rng=rng, factor=factor),
    ]


@pytest.mark.parametrize("mode", ["gaussian", "coloc"])
@pytest.mark.parametrize("factor", [False, True])
def test_controls(mode, factor):
    rng = np.random.default_rng(20261026)
    loci = _controls(rng, factor)
    out = ExactColoc(mode=mode, M=50, n_mc=4096).fit_batch(loci)
    r = {o["locus_id"]: o for o in out}
    for o in out:
        assert set(o) == COLS and o["status"] == "ok", o
        json.loads(o["extra"])
        assert 0 <= o["P_S"] <= 1 and 0 <= o["P_D"] <= 1 and abs(o["P_S"] + o["P_D"] + o["P_W"] - 1) < 1e-9
        assert o["runtime_s"] > 0
    assert r["shared_r2_1"]["s"] > 0.9 and r["shared_r2_1"]["P_S"] > 0.9
    assert r["shared_dup_other"]["s"] > 0.9  # unidentifiable duplicate: must not be called distinct
    assert r["distinct"]["s"] < 0.1 and r["distinct"]["P_D"] > 0.9
    assert r["null"]["P_W"] > 0.95
    assert r["t1_only"]["P_W"] > 0.95
    assert r["distinct"]["E1"] > 0.9 and r["distinct"]["E2"] > 0.9 and r["shared_r2_1"]["E2"] < 0.1


def test_dense_equals_factor():
    rng = np.random.default_rng(1)
    a = _controls(rng, factor=False)
    rng = np.random.default_rng(1)
    b = _controls(rng, factor=True)
    oa = ExactColoc(mode="gaussian", M=50, n_mc=1024, seed=20261029).fit_batch(a)
    ob = ExactColoc(mode="gaussian", M=50, n_mc=1024, seed=20261029).fit_batch(b)
    for x, y in zip(oa, ob):
        ea, eb = json.loads(x["extra"]), json.loads(y["extra"])
        for h in ea["PP"]:
            assert abs(ea["PP"][h] - eb["PP"][h]) < 1e-5  # float32 storage of R vs X_ref


def test_compiled_equals_eager(device):
    if device.type != "cuda":
        pytest.skip("compile path exercised on GPU only")
    rng = np.random.default_rng(3)
    loci = _controls(rng)
    m_e = ExactColoc(mode="gaussian", M=40, n_mc=512)
    m_c = ExactColoc(mode="gaussian", M=40, n_mc=512, compile=True)
    items = [m_e._prepare(lo) for lo in loci]
    for d in items:
        d["sigma"], d["eta"] = (8.0, 6.0), 1e-3
    T = m_e._tensors(items)
    ts = TermSet(40, 2, device=T.b.device)
    r1 = enumerate_batch(T, ts, evaluator=get_evaluator(False))
    r2 = enumerate_batch(T, ts, evaluator=get_evaluator(True))
    assert torch.allclose(r1["logZ_cat"], r2["logZ_cat"], rtol=1e-12, atol=1e-9)
    assert torch.allclose(r1["pip"], r2["pip"], rtol=1e-10, atol=1e-12)


def test_failure_is_reported_not_raised():
    rng = np.random.default_rng(4)
    lo = _controls(rng)[0]
    lo.R = lo.R[:10, :10]  # corrupt LD
    out = ExactColoc(mode="gaussian", M=50, n_mc=256).fit_batch([lo])
    assert out[0]["status"] == "failed" and "error" in json.loads(out[0]["extra"])


def test_no_truth_file_access():
    import famB
    import pathlib
    src = pathlib.Path(famB.__file__).parent
    for f in src.glob("*.py"):
        txt = f.read_text()
        assert "truth" not in txt.lower(), f


def test_compiled_equals_eager_mixture(device):
    """Generic (mixture, K > 1) path: compiled == eager; and the generic path == rectangular path for K = 1."""
    if device.type != "cuda":
        pytest.skip("compile path exercised on GPU only")
    import sys
    from famB.priors import toy_l0star_prior
    from toy_l0star import simulate_simprior
    rng = np.random.default_rng(5)
    loci, _ = simulate_simprior(rng, toy_l0star_prior(), 3, p=7)
    m = ExactColoc(mode="simulator", M=30, n_mc=256)
    items = [m._prepare(lo) for lo in loci]
    for d in items:
        d["eta"] = 1e-3
    T = m._tensors(items)
    ts = TermSet(7, 2, K=T.K, device=T.b.device)
    r1 = enumerate_batch(T, ts, evaluator=get_evaluator(False))
    r2 = enumerate_batch(T, ts, evaluator=get_evaluator(True))
    assert torch.allclose(r1["logZ_cat"], r2["logZ_cat"], rtol=1e-12, atol=1e-9)
    # K = 1 through both code paths (generic path forced via a mixture with one component)
    T1 = m._tensors(items)
    T1.sd = T1.sd[..., :1].contiguous() * 0 + 7.0
    T1.logwk = T1.logwk[..., :1].contiguous() * 0
    ts_rect = TermSet(7, 2, K=1, device=T.b.device)
    ts_gen = TermSet(7, 2, K=1, device=T.b.device)
    ts_gen.offsets = torch.arange(ts_gen.n_terms + 1, device=T.b.device)  # forces the generic path
    a = enumerate_batch(T1, ts_rect, n_samples=20000)
    b = enumerate_batch(T1, ts_gen, n_samples=20000)
    assert torch.allclose(a["logZ_cat"], b["logZ_cat"], rtol=1e-12, atol=1e-9)
    assert torch.max(torch.abs(a["pip"] - b["pip"])) < 0.02  # exact vs Monte Carlo (n = 20000)
    assert torch.max(torch.abs(a["spip"] - b["spip"])) < 0.02  # MC vs MC


def test_sampler_never_draws_negligible_terms(device):
    """Regression (found 2026-10-26): rows whose mass underflowed for one locus of a batch broke the
    concatenated CDF of the two-stage sampler and produced draws of terms with weight ~e^-100.
    Batch mixing a very strong locus (logZ ~ 3e3) with weak ones; every sampled term must carry
    non-negligible posterior mass, and the sampled H-category frequencies must match the exact PPs."""
    from famB.enum import eval_terms, structure
    from famB.priors import toy_l0star_prior
    from toy_l0star import simulate_simprior
    rng = np.random.default_rng(5)
    loci, _ = simulate_simprior(rng, toy_l0star_prior(), 3, p=7)
    m = ExactColoc(mode="simulator", M=30, n_mc=256)
    items = [m._prepare(lo) for lo in loci]
    for d in items:
        d["eta"] = 1e-3
    T = m._tensors(items)
    T.sd = T.sd[..., :1].contiguous() * 0 + 7.0
    T.logwk = T.logwk[..., :1].contiguous() * 0
    ts = TermSet(7, 2, K=1, device=T.b.device)
    g = torch.Generator(device=T.b.device); g.manual_seed(20261029)
    res = enumerate_batch(T, ts, n_samples=50000, generator=g)
    t = torch.arange(ts.n_terms, device=T.b.device)
    lw, cat, _, _ = eval_terms(T, ts, t)
    wl = lw - torch.logsumexp(lw, dim=1, keepdim=True)
    for b in range(3):
        assert float(wl[b, res["samples"][b]].min()) > -40.0
        freq = torch.bincount(cat[res["samples"][b]], minlength=6).double() / 50000
        assert torch.max(torch.abs(freq - res["PP"][b])) < 0.01
