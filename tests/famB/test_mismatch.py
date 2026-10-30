"""B-mm (famB.mismatch, DESIGN_P2 §2) known-answer tests.

1. n_ref = inf, eta = 1e-3: B-mm's full-z_M path reproduces phase-1 B's log BF (every term) to < 1e-8 on 200
   validation loci (valid_grid_v1, all 34 cells; NOT screen_v2, NOT test).
2. Brute force (p = 10 <= 12): every piece recomputed with dense 2p-dimensional Gaussian algebra in numpy
   (explicit loops for the inflation formula): posterior second moments, Delta, the EB objective table and every
   configuration's log BF of the final inflated model agree to < 1e-8.
3. The inflation formula (with cross terms) matches the Monte-Carlo covariance of (R_hat - R) b over simulated
   reference panels (n_ref = 150).
4. End to end (fit_batch, EB on): n_ref = inf gives phase-1 B's PP and event posteriors.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from famB.enum import TermSet, eval_terms
from famB.method import ExactColoc
from famB.mismatch import eb_select_mm, inflation_blocks, second_moments
from toy_data import ar_genotypes, make_locus

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parents[2]
VALID = ROOT / "data/processed/sets/valid_grid_v1"
F64 = torch.float64


def _prepared(m: ExactColoc, loci):
    items = [m._prepare(lo) for lo in loci]
    for d in items:
        d["sigma"], d["eta"] = tuple(m.cfg.sigma), m.cfg.eta
    return items


def _max_term_diff(T1, T2, ts, chunk=1 << 18):
    out = 0.0
    per = max(1, chunk // T1.nB)
    for s in range(0, ts.n_terms, per):
        t = torch.arange(s, min(ts.n_terms, s + per), device=T1.b.device)
        a = eval_terms(T1, ts, t)[0]
        b = eval_terms(T2, ts, t)[0]
        fin = torch.isfinite(a)
        assert torch.equal(fin, torch.isfinite(b))
        out = max(out, float((a[fin] - b[fin]).abs().max()))
    return out


# ------------------------------------------------------------------------------------------------- 1
@pytest.mark.slow
@pytest.mark.skipif(not VALID.exists(), reason="validation sets not available")
def test_nref_inf_reproduces_phase1_logbf_200_validation_loci():
    from eval.io import iter_loci, list_loci

    files = sorted(VALID.glob("*.h5"))
    assert len(files) >= 30
    per_cell = math.ceil(200 / len(files))
    n_done, worst = 0, 0.0
    for f in files:
        ids = list_loci(f)
        take = ids[:: max(1, len(ids) // per_cell)][:per_cell][: 200 - n_done]
        if not take:
            break
        loci = list(iter_loci(f, take))
        for sig in ((6.0, 6.0), (3.0, 24.0)):
            kw = dict(mode="gaussian", M=50, eb=False, eta=1e-3, sigma=sig, device=DEV)
            m1 = ExactColoc(**kw)
            m2 = ExactColoc(mismatch=True, mm_n_ref=math.inf, mm_iter=2, **kw)
            it1 = _prepared(m1, loci)
            it2 = _prepared(m2, loci)
            groups = {}
            for a, b in zip(it1, it2):
                assert np.array_equal(a["sel"], b["sel"])
                groups.setdefault(len(a["sel"]), []).append((a, b))
            for M, pairs in groups.items():
                g1 = [a for a, _ in pairs]
                g2 = [b for _, b in pairs]
                T1 = m1._tensors(g1)
                T2 = m2._mm_fit(g2, m2._tensors(g2))
                assert T1.Phi is None and T2.Phi is not None
                ts = TermSet(M, 2, device=DEV)
                worst = max(worst, _max_term_diff(T1, T2, ts))
        n_done += len(take)
    assert n_done == 200
    print(f"max |d log(prior x BF)| over all terms of 200 loci: {worst:.3e}")
    assert worst < 1e-8


# ------------------------------------------------------------------------------------------------- 2
def _bf_moments(z, Sigma, Rt, sd, p1, p2, p12, configs):
    """Brute force over configurations (S1, S2): log prior, log BF (dense 2p Gaussian), E[lambda lambda']."""
    p = Rt.shape[0]
    Si = np.linalg.inv(Sigma)
    _, ld0 = np.linalg.slogdet(Sigma)
    ll0 = -0.5 * z @ Si @ z - 0.5 * ld0
    lw, mom = [], []
    for S1, S2 in configs:
        coords = [(0, j) for j in S1] + [(1, j) for j in S2]
        nsh = len(set(S1) & set(S2))
        lp = (len(S1) - nsh) * math.log(p1) + (len(S2) - nsh) * math.log(p2) + nsh * math.log(p12)
        if not coords:
            lw.append(lp)
            mom.append(np.zeros((2 * p, 2 * p)))
            continue
        X = np.zeros((2 * p, len(coords)))
        for c, (t, j) in enumerate(coords):
            X[t * p:(t + 1) * p, c] = Rt[:, j]
        D = np.diag([sd[t] ** 2 for t, _ in coords])
        C1 = Sigma + X @ D @ X.T
        _, ld1 = np.linalg.slogdet(C1)
        ll1 = -0.5 * z @ np.linalg.solve(C1, z) - 0.5 * ld1
        lw.append(lp + ll1 - ll0)
        P = np.linalg.inv(np.linalg.inv(D) + X.T @ Si @ X)
        mu = P @ X.T @ Si @ z
        Mm = P + np.outer(mu, mu)
        full = np.zeros((2 * p, 2 * p))
        idx = [t * p + j for t, j in coords]
        full[np.ix_(idx, idx)] = Mm
        mom.append(full)
    lw = np.array(lw)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    S = sum(wi * mi for wi, mi in zip(w, mom))
    return lw, S


def _delta_loops(R, S, n_ref, cross=True):
    p = R.shape[0]
    D = np.zeros((2, 2, p))
    for t in range(2):
        for u in range(2):
            for k in range(p):
                acc = 0.0
                for j in range(p):
                    for jj in range(p):
                        if j == jj:
                            c = (1 - R[k, j] ** 2) ** 2
                        elif cross:
                            c = (R[j, jj] * (1 - R[k, j] ** 2 - R[k, jj] ** 2)
                                 - 0.5 * R[k, j] * R[k, jj] * (1 - R[k, j] ** 2 - R[k, jj] ** 2 - R[j, jj] ** 2))
                        else:
                            c = 0.0
                        acc += S[t * p + j, u * p + jj] * c
                D[t, u, k] = acc / n_ref
    return D


def _mismatch_locus(rng, lid, rho, kind, p=10):
    Xs = ar_genotypes(rng, 4000, p, ar=0.7)  # study
    Xr = ar_genotypes(rng, 150, p, ar=0.7)  # reference (independent draw: mismatch)
    lam1 = np.zeros(p); lam2 = np.zeros(p)
    j, k = rng.choice(p, 2, replace=False)
    lam1[j] = rng.uniform(6, 12) * rng.choice([-1, 1])
    if kind == "S":
        lam2[j] = rng.uniform(5, 10) * rng.choice([-1, 1])
    else:
        lam2[k] = rng.uniform(5, 10) * rng.choice([-1, 1])
    lo = make_locus(lid, Xs, lam1, lam2, rho=rho, rng=rng)
    ref = make_locus(lid, Xr, lam1, lam2, rho=rho, rng=rng, z_override=(lo.z1, lo.z2))
    object.__setattr__(ref, "rho_hat", rho)
    return ref  # z from the study, LD (dense, n_ref = 150) from the reference


@pytest.mark.parametrize("rho,kind,cross", [(0.0, "S", True), (0.4, "D", True), (-0.6, "S", True),
                                            (0.4, "S", False)])
def test_bruteforce_full_likelihood_small_p(rho, kind, cross):
    rng = np.random.default_rng([20261028, int(round(10 * rho)) + 10, ord(kind), int(cross)])
    p = 10
    lo = _mismatch_locus(rng, f"bf_{kind}_{rho}", rho, kind, p)
    m = ExactColoc(mode="gaussian", M=12, mismatch=True, mm_iter=1, mm_cross_terms=cross, device=DEV)
    cfg = m.cfg
    d = m._prepare(lo)
    assert len(d["sel"]) == p and d["n_ref"] == 150
    from famB.priors import eb_select
    s1, s2, eta0, _ = eb_select(d["zu"], d["Ruu"], np.log(d["mult"]), (cfg.p1 + cfg.p12, cfg.p2 + cfg.p12), 2,
                                cfg.eb_grid, m.device)
    d["sigma"], d["eta"] = (s1, s2), max(eta0, cfg.eta_floor)
    T0 = m._tensors([d])
    ts = TermSet(p, 2, device=DEV)
    from famB.enum import enumerate_batch
    res0 = enumerate_batch(T0, ts, n_samples=0, want_pip=False)
    S_code = second_moments(T0, ts, res0["logZ"])[0].cpu().numpy()
    # brute force pass 0 (phase-1 model)
    R = d["Ruu"].astype(np.float64)
    V = np.array([[1, rho], [rho, 1.0]])
    z = d["zu"].reshape(-1)
    singles = [()] + [(j,) for j in range(p)] + list(itertools.combinations(range(p), 2))
    configs = [(a, b) for a in singles for b in singles]
    Rt0 = (1 - d["eta"]) * R + d["eta"] * np.eye(p)
    lw0, S_bf = _bf_moments(z, np.kron(V, Rt0), Rt0, d["sigma"], cfg.p1, cfg.p2, cfg.p12, configs)
    assert np.abs(S_code - S_bf).max() < 1e-8 * max(1.0, np.abs(S_bf).max())
    from famB.mismatch import logz_and_moments
    lz_f, S_f = logz_and_moments(T0, ts, chunk_elems=997)  # fused fast path (production), odd chunking
    assert abs(float(lz_f[0]) - float(res0["logZ"][0])) < 1e-9
    assert np.abs(S_f[0].cpu().numpy() - S_bf).max() < 1e-8 * max(1.0, np.abs(S_bf).max())
    D_bf = _delta_loops(R, S_bf, 150.0, cross)
    D_code, _ = inflation_blocks(torch.as_tensor(R[None], device=DEV), torch.as_tensor(S_code[None], device=DEV),
                                 torch.tensor([150.0], device=DEV, dtype=F64), cross)
    D_code = D_code[0].cpu().numpy()
    assert np.abs(D_code - D_bf).max() < 1e-10  # (no PSD projection expected here)
    # EB objective table under the inflated single-trait likelihood
    dl = np.stack([D_bf[0, 0], D_bf[1, 1]])
    _, _, _, ell = eb_select_mm(d["zu"], d["Ruu"], np.log(d["mult"]), (cfg.p1 + cfg.p12, cfg.p2 + cfg.p12), 2,
                                cfg.eb_grid, m.device, dl)
    for t in range(2):
        pm = (cfg.p1 if t == 0 else cfg.p2) + cfg.p12
        for ie, eta in enumerate(cfg.eb_grid.eta):
            Rt = (1 - eta) * R + eta * np.eye(p)
            Sig = Rt + np.diag(dl[t])
            _, ld = np.linalg.slogdet(Sig)
            null = -0.5 * d["zu"][t] @ np.linalg.solve(Sig, d["zu"][t]) - 0.5 * ld - 0.5 * p * math.log(2 * math.pi)
            for isg, sg in enumerate(cfg.eb_grid.sigma):
                tot, totp = [], []
                for c in singles:
                    lp = len(c) * math.log(pm)
                    if c:
                        X = Rt[:, list(c)]
                        C1 = Sig + sg ** 2 * X @ X.T
                        _, ld1 = np.linalg.slogdet(C1)
                        l1 = -0.5 * d["zu"][t] @ np.linalg.solve(C1, d["zu"][t]) - 0.5 * ld1 - 0.5 * p * math.log(2 * math.pi)
                    else:
                        l1 = null
                    tot.append(lp + l1)
                    totp.append(lp)
                ref = np.logaddexp.reduce(tot) - np.logaddexp.reduce(totp)
                assert abs(ell[t, ie, isg] - ref) < 1e-8, (t, eta, sg, ell[t, ie, isg], ref)
    # final model of the method (one plug-in refinement) vs brute force with Delta_bf
    T1 = m._mm_fit([d], m._tensors([d]))
    eta1, sig1 = d["eta"], d["sigma"]
    Rt1 = (1 - eta1) * R + eta1 * np.eye(p)
    Delta = np.zeros((2 * p, 2 * p))
    for t in range(2):
        for u in range(2):
            Delta[t * p + np.arange(p), u * p + np.arange(p)] = D_bf[t, u]
    lw1, _ = _bf_moments(z, np.kron(V, Rt1) + Delta, Rt1, sig1, cfg.p1, cfg.p2, cfg.p12, configs)
    t = torch.arange(ts.n_terms, device=DEV)
    lw_code = eval_terms(T1, ts, t)[0][0].cpu().numpy()
    Vt, mt, _ = ts.decode(t)
    Vt, mt = Vt.cpu().numpy(), mt.cpu().numpy()
    pos = {c: i for i, c in enumerate(configs)}
    worst = 0.0
    for i in range(ts.n_terms):
        S1 = tuple(int(v) for v, ok in zip(Vt[i, :2], mt[i, :2]) if ok)
        S2 = tuple(int(v) for v, ok in zip(Vt[i, 2:], mt[i, 2:]) if ok)
        worst = max(worst, abs(lw_code[i] - lw1[pos[(S1, S2)]]))
    print(f"rho={rho} kind={kind} cross={cross}: max |d log w| = {worst:.2e}, eta={eta1}, sigma={sig1}")
    assert worst < 1e-8


# ------------------------------------------------------------------------------------------------- 3
def test_inflation_formula_matches_reference_panel_simulation():
    rng = np.random.default_rng(20261029)
    p, n, reps = 6, 150, 6000
    idx = np.arange(p)
    R = 0.65 ** np.abs(idx[:, None] - idx[None, :])
    Lc = np.linalg.cholesky(R)
    b1 = np.zeros(p); b2 = np.zeros(p)
    b1[1], b1[3] = 8.0, -5.0
    b2[1], b2[4] = 6.0, 4.0
    v = np.zeros((reps, 2, p))
    for r in range(reps):
        X = rng.standard_normal((n, p)) @ Lc.T
        X = (X - X.mean(0)) / X.std(0)
        E = X.T @ X / n - R
        v[r, 0] = E @ b1
        v[r, 1] = E @ b2
    emp = np.zeros((2, 2, p))
    vc = v - v.mean(0)
    for t in range(2):
        for u in range(2):
            emp[t, u] = (vc[:, t] * vc[:, u]).mean(0)
    bb = np.concatenate([b1, b2])
    S = np.outer(bb, bb)
    Dx, _ = inflation_blocks(torch.as_tensor(R[None]), torch.as_tensor(S[None]), torch.tensor([float(n)], dtype=F64))
    Dd, _ = inflation_blocks(torch.as_tensor(R[None]), torch.as_tensor(S[None]), torch.tensor([float(n)], dtype=F64),
                             cross_terms=False)
    Dx, Dd = Dx[0].numpy(), Dd[0].numpy()
    den = np.sqrt(np.einsum("tk,uk->tuk", np.stack([emp[0, 0], emp[1, 1]]), np.stack([emp[0, 0], emp[1, 1]])))
    err_x = np.abs(Dx - emp) / den
    err_d = np.abs(Dd - emp) / den
    print("max rel err with cross terms", err_x.max(), "diagonal only", err_d.max())
    assert err_x.max() < 0.12
    assert err_d.max() > err_x.max()


# ------------------------------------------------------------------------------------------------- 4
def test_end_to_end_nref_inf_equals_phase1():
    rng = np.random.default_rng(7)
    loci = []
    for i in range(12):
        p = int(rng.integers(40, 120))
        X = ar_genotypes(rng, 300, p, ar=0.8)
        lam1 = np.zeros(p); lam2 = np.zeros(p)
        j, k = rng.choice(p, 2, replace=False)
        lam1[j] = rng.uniform(5, 12)
        (lam2.__setitem__(j, rng.uniform(4, 9)) if i % 2 else lam2.__setitem__(k, rng.uniform(4, 9)))
        loci.append(make_locus(f"E{i}", X, lam1, lam2, rho=0.3 * (i % 3 - 1), rng=rng))
    a = ExactColoc(mode="gaussian", M=30, n_mc=4096, device=DEV).fit_batch(loci)
    b = ExactColoc(mode="gaussian", M=30, n_mc=4096, device=DEV, mismatch=True, mm_n_ref=math.inf).fit_batch(loci)
    import json
    for ra, rb in zip(a, b):
        assert ra["status"] == rb["status"] == "ok"
        pa, pb = json.loads(ra["extra"])["PP"], json.loads(rb["extra"])["PP"]
        assert max(abs(pa[k] - pb[k]) for k in pa) < 1e-10
        assert abs(ra["P_S"] - rb["P_S"]) < 1e-12 and abs(ra["P_D"] - rb["P_D"]) < 1e-12
