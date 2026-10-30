"""M3.5 Family A additions: batched η spectrum (Gram / dense), fused Triton η update, per-locus EB of π,
layer E (empirical reference-size variance). Toy data with known answers."""
import math

import numpy as np
import pytest
import torch

from famA import joint_susie as js
from famA.joint_susie import JointConfig, joint_susie_batch
from famG.ld import build_batch_ld
from famag_testutils import ar_genotypes, toy_locus

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="CUDA required")


def _bucket(Xs, zs, kind, rhos, cfg, ld_dtype=torch.float64, nref=None):
    ps = [len(z[0]) for z in zs]
    P = max(ps)
    B = len(zs)
    z = torch.zeros((B, 2, P), dtype=torch.float64, device="cuda")
    valid = torch.zeros((B, P), dtype=torch.bool, device="cuda")
    for i, (a, b) in enumerate(zs):
        z[i, 0, :len(a)] = torch.tensor(a); z[i, 1, :len(a)] = torch.tensor(b); valid[i, :len(a)] = True
    lds = [X.T @ X / X.shape[0] for X in Xs] if kind == "dense" else Xs
    ld = build_batch_ld(lds, kind, P, ld_dtype, "cuda")
    rho = torch.tensor(rhos, dtype=torch.float64, device="cuda")
    nr = None if nref is None else torch.tensor(nref, dtype=torch.float64, device="cuda")
    return js._JointBucket(z, valid, torch.tensor(ps, device="cuda"), None, ld, rho, cfg, nref_e=nr), lds


@needs_cuda
@pytest.mark.parametrize("kind", ["dense", "factor"])
def test_eta_terms_vs_bruteforce(kind):
    """−½log|V⊗R̃| − ½zᵀ(V⊗R̃)⁻¹z − p log2π from the batched spectrum equals the dense float64 formula, incl.
    rank-deficient R (n < p), padding (unequal p) and rho != 0."""
    rng = np.random.default_rng(1)
    ps, ns = [150, 180, 170], [100, 120, 400]
    Xs = [ar_genotypes(rng, n, p, phi=0.7) for p, n in zip(ps, ns)]
    zs = [(X.T @ rng.standard_normal(X.shape[0]) / np.sqrt(X.shape[0]) + 0.3 * rng.standard_normal(X.shape[1]),
           rng.standard_normal(X.shape[1])) for X in Xs]
    rhos = [0.0, 0.4, -0.3]
    st, lds = _bucket(Xs, zs, kind, rhos, JointConfig(estimate_eta=True))
    for eta in (1e-6, 1e-3, 0.05, 0.5):
        got = st._eta_terms(torch.full((3,), eta, dtype=torch.float64, device="cuda")).cpu().numpy()
        for i in range(3):
            R = lds[i] if kind == "dense" else lds[i].T @ lds[i] / lds[i].shape[0]
            d = np.sqrt(np.diag(R)); R = R / d[:, None] / d[None, :]
            zz = np.concatenate([zs[i][0] / d, zs[i][1] / d])
            V = np.array([[1, rhos[i]], [rhos[i], 1]])
            S = np.kron(V, (1 - eta) * R + eta * np.eye(ps[i]))
            ref = -0.5 * np.linalg.slogdet(S)[1] - 0.5 * zz @ np.linalg.solve(S, zz) - ps[i] * math.log(2 * math.pi)
            assert abs(got[i] - ref) < 1e-7 * abs(ref), (kind, eta, i, got[i], ref)


@needs_cuda
def test_eta_triton_matches_torch():
    rng = np.random.default_rng(9)
    p, n = 300, 150
    Xin = ar_genotypes(rng, n, p)
    Xo = ar_genotypes(rng, n, p)
    loci = [toy_locus(rng, n, p, (50,), (10.0,), (50,), (8.0,), X=X) for X in (Xin, Xo, Xo)]
    args = ([l[2] for l in loci], [l[3] for l in loci], [Xin.astype(np.float32)] * 3, [1e5] * 3, [1e5] * 3,
            [0.0, 0.2, -0.1], "factor")
    a = joint_susie_batch(*args, JointConfig(estimate_eta=True, use_triton=True), n_refs=[n] * 3)
    b = joint_susie_batch(*args, JointConfig(estimate_eta=True, use_triton=False), n_refs=[n] * 3)
    for x, y in zip(a, b):
        assert abs(x["eta"] - y["eta"]) <= 1e-7 * max(y["eta"], 1e-8)
        assert np.abs(x["pip"] - y["pip"]).max() < 1e-7
        assert x["niter"] == y["niter"]


@needs_cuda
def test_eig_float32_option_close():
    """The float32 dense eigendecomposition (speed option) gives the same η / P_S / P_D up to small error."""
    rng = np.random.default_rng(4)
    p, n = 250, 400
    X = ar_genotypes(rng, n, p)
    Xo = ar_genotypes(rng, n, p)
    loci = [toy_locus(rng, n, p, (40,), (9.0,), (40 if k else 200,), (8.0,), X=(X if k % 2 else Xo)) for k in range(4)]
    R = (X.T @ X / n).astype(np.float32)
    args = ([l[2] for l in loci], [l[3] for l in loci], [R] * 4, [1e5] * 4, [1e5] * 4, [0.0] * 4, "dense")
    a = joint_susie_batch(*args, JointConfig(estimate_eta=True, eig_dtype=torch.float64))
    b = joint_susie_batch(*args, JointConfig(estimate_eta=True, eig_dtype=torch.float32))
    for x, y in zip(a, b):
        assert y["eta"] >= 1e-3 - 1e-15                            # float32 spectrum: η floored at 1e-3
        if x["eta"] > 1e-2:                                         # identifiable η: same estimate
            assert abs(math.log(x["eta"]) - math.log(y["eta"])) < 0.05
        assert abs(x["P_S"] - y["P_S"]) < 2e-3 and abs(x["P_D"] - y["P_D"]) < 2e-3
        assert np.abs(x["pip"] - y["pip"]).max() < 5e-3


@needs_cuda
def test_pi_eb():
    """EB of π: a locus with one strong shared signal moves π_both up; a locus with one trait-1-only signal moves
    π_2-only down; κ -> ∞ reproduces the fixed-π fit."""
    rng = np.random.default_rng(21)
    p, n = 200, 2000
    X = ar_genotypes(rng, n, p)
    sh = toy_locus(rng, n, p, (60,), (12.0,), (60,), (11.0,), X=X)
    on = toy_locus(rng, n, p, (60,), (12.0,), X=X)
    args = ([sh[2], on[2]], [sh[3], on[3]], [sh[1], on[1]], [1e5] * 2, [1e5] * 2, [0.0] * 2, "dense")
    eb = joint_susie_batch(*args, JointConfig(pi_mode="eb", pi_prior_strength=4.0))
    # one used effect, κ = 4, π_ref = 1/4: the fixed point is π_s = (q(s) + 1)/5, so π_both -> ~0.4
    assert eb[0]["pi"][3] > 0.35 and eb[0]["pi"][3] > max(eb[0]["pi"][:3])
    # trait-1-only signal: σ²_2 -> 0 by EB makes 'both' identical to '1-only' (split by π), so EB moves mass away
    # from the null and 2-only states only: π_2 -> 1/5
    assert eb[1]["pi"][1] > 0.27 and abs(eb[1]["pi"][1] - eb[1]["pi"][3]) < 1e-6 and eb[1]["pi"][2] < 0.21
    assert abs(eb[0]["pi"].sum() - 1) < 1e-12
    big = joint_susie_batch(*args, JointConfig(pi_mode="eb", pi_prior_strength=1e12))
    fx = joint_susie_batch(*args, JointConfig(pi_mode="fixed"))
    for x, y in zip(big, fx):
        assert np.abs(x["pip"] - y["pip"]).max() < 1e-6 and abs(x["P_S"] - y["P_S"]) < 1e-6
    # ELBO (MAP objective) never decreases by more than rounding in EB mode
    for o in eb:
        assert np.all(np.diff(o["elbo"]) > -1e-6 * np.abs(o["elbo"][1:]))


def test_ref_var_table():
    r2e, mafe, n_list, K, pooled = js.load_ref_var_table(str(js.REF_VAR_TABLE))
    assert K.shape == (len(n_list), len(r2e) - 1, len(mafe) - 1) and np.all(np.isfinite(K)) and np.all(K >= 0)
    # common variants (min MAF >= 0.2), r² ~ 0: n·Var(r̂) ~ 1 (normal theory); high r²: far above (1-r²)²
    assert 0.9 < K[1, 0, 2] < 1.1
    assert K[1, 9, 2] > 3 * (1 - 0.925) ** 2


@needs_cuda
def test_layer_E_delta_bruteforce():
    """update_delta equals a direct computation from the table on a small locus."""
    rng = np.random.default_rng(2)
    p, n = 60, 150
    X = ar_genotypes(rng, n, p)
    z1, z2 = rng.standard_normal(p), rng.standard_normal(p)
    cfg = JointConfig(ref_var=True, convergence="alpha", ref_var_topk=p)
    zs = [(z1, z2)]
    st, lds = _bucket([X], zs, "factor", [0.0], cfg, nref=[float(n)])
    E2 = torch.zeros_like(st.E2)
    E2[0, 0, 0, 10] = 25.0; E2[0, 1, 0, 30] = 16.0; E2[0, 0, 1, 11] = 4.0
    st.E2 = E2
    st.update_delta()
    r2e, mafe, n_list, K, pooled = js.load_ref_var_table(str(js.REF_VAR_TABLE))
    nb = np.abs(np.log(n_list) - np.log(n)).argmin()
    R = X.T @ X / n
    e2 = E2.sum(1)[0].cpu().numpy()
    ref = np.ones((2, p))
    for t in range(2):
        for k in range(p):
            s = 0.0
            for i in np.nonzero(e2[t])[0]:
                if i == k:
                    continue
                b = min(np.searchsorted(r2e, R[i, k] ** 2, side="right") - 1, len(r2e) - 2)
                s += e2[t, i] * pooled[nb, b] / n
            ref[t, k] += s
    assert np.abs(st.dvar[0].cpu().numpy() - ref).max() < 1e-9


@needs_cuda
def test_layer_E_reduces_spurious_signals():
    """Strong signals, z from the in-sample LD of one sample, LD from a small reference (n = 100): the
    reference-size variance term reduces the number of credible sets (spurious secondary signals) and the
    distinct-signal probability P_D on a truly shared locus."""
    rng = np.random.default_rng(5)
    p, n_ref = 300, 100
    Xpop = ar_genotypes(rng, 20000, p, phi=0.9)
    ref = Xpop[rng.choice(20000, n_ref, replace=False)]
    ref = (ref - ref.mean(0)) / ref.std(0)
    R = Xpop.T @ Xpop / 20000
    out_E, out_0 = [], []
    for k in range(6):
        lam = np.zeros(p); lam[100 + 10 * k] = 25.0
        z1 = R @ lam + Xpop.T @ rng.standard_normal(20000) / np.sqrt(20000)
        z2 = R @ lam + Xpop.T @ rng.standard_normal(20000) / np.sqrt(20000)
        common = ([z1], [z2], [ref.astype(np.float32)], [1e6], [1e6], [0.0], "factor")
        out_0.append(joint_susie_batch(*common, JointConfig(convergence="alpha"), n_refs=[n_ref])[0])
        out_E.append(joint_susie_batch(*common, JointConfig(ref_var=True, convergence="alpha"), n_refs=[n_ref],
                                       ref_sizes=[n_ref])[0])
    n0 = sum(len(o["sets"]["cs"]) for o in out_0)
    nE = sum(len(o["sets"]["cs"]) for o in out_E)
    assert nE < n0, (nE, n0)
    assert np.mean([o["P_D"] for o in out_E]) <= np.mean([o["P_D"] for o in out_0]) + 1e-9
    assert all(o["P_S"] > 0.9 for o in out_E)


@needs_cuda
@pytest.mark.parametrize("kind", ["dense", "factor"])
def test_continuous_batching_equals_single_fits(kind):
    """A working set of 3 rows refilled from a queue of 9 loci (mixed iteration counts, η estimated) gives the
    same per-locus results as fitting every locus alone."""
    rng = np.random.default_rng(12)
    p, n = 120, 300
    Xs = [ar_genotypes(rng, n, p, phi=0.6 + 0.03 * k) for k in range(3)]
    loci = []
    for k in range(9):
        c1 = [(20,), (20, 80), ()][k % 3]
        c2 = [(20,), (50,), (90,)][(k // 3) % 3]
        loci.append(toy_locus(rng, n, p, c1, [9.0, -7.0][:len(c1)], c2, [8.0], X=Xs[k % 3]))
    lds = [(l[1] if kind == "dense" else l[0]) for l in loci]
    args = ([l[2] for l in loci], [l[3] for l in loci], lds, [1e5] * 9, [2e4] * 9, list(np.linspace(-0.3, 0.3, 9)), kind)
    cfg = dict(estimate_eta=True, ld_dtype=torch.float64)
    many = joint_susie_batch(*args, JointConfig(max_batch=3, refill_fraction=0.34, **cfg))
    assert len({o["niter"] for o in many}) > 1
    dp = []
    for i, o in enumerate(many):
        one = joint_susie_batch(*[[a[i]] if isinstance(a, list) else a for a in args], JointConfig(**cfg))[0]
        assert o["niter"] == one["niter"]
        dp.append(np.abs(o["pip"] - one["pip"]).max())
        # η sits on a flat ELBO (Brent tolerance), so rounding moves it slightly: same up to 1e-4 relative,
        # or both at the noise floor (< 1e-6) where it has no effect on the fit
        assert abs(o["eta"] - one["eta"]) <= 1e-4 * one["eta"] or max(o["eta"], one["eta"]) < 1e-6
    # float64 throughout; batch-size dependent BLAS reductions differ at ~1e-16 and are amplified only where a
    # discrete EB decision (null check / keep-old) is a near tie: one locus here moves by ~3e-4 in every batched
    # fit, with or without refill (M3.5 report); all others agree to < 1e-7.
    assert max(dp) < 1e-3 and sorted(dp)[-2] < 1e-7, dp
