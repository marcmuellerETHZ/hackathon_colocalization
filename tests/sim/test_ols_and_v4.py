"""D6(a) OLS statistics, D6(b) lambda, DESIGN v4 conformance of the generator (trait types, P2 sizes,
population levels with rho = 0, PVE cap, shifted-arm architecture). GPU + chr22 data."""
import math

import numpy as np
import pytest
import torch

from common.seeds import rng
from sim import params
from sim.generator import (CellSpec, TraitSpec, WindowContext, _genetic_signal, _mu, draw_candidate, finalize,
                           make_factor_study, make_indiv_study, noisy_stats)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def ctx():
    c = WindowContext("chr22", 30_000_000, 30_300_000, split="valid", block_ids=("x",), anchor_block="x")
    yield c
    c.release()


def _ols_reference(G: np.ndarray, y: np.ndarray, C: np.ndarray):
    """Per-SNP OLS of y on [C, g_j] (C includes the intercept): slope, SE, t (numpy fp64)."""
    n = len(y)
    Q, _ = np.linalg.qr(C)
    yr = y - Q @ (Q.T @ y)
    Gr = G - Q @ (Q.T @ G)
    gg = (Gr * Gr).sum(0)
    b = Gr.T @ yr / gg
    rss = (yr @ yr) - b * b * gg
    df = n - C.shape[1] - 1
    se = np.sqrt(rss / df / gg)
    return b, se, b / se


@pytest.mark.parametrize("pop", ["NFE", "ADMIX"])
def test_individual_level_stats_are_ols(ctx, pop):
    """beta/se/z equal an explicit per-SNP regression of y on the ALT dosage (+ intercept, + 2 PCs at L3)."""
    g = np.random.default_rng(1)
    st = make_indiv_study(ctx, TraitSpec("mol", 500, "indiv", pop), g)
    beta = torch.zeros(ctx.p, device=ctx.device)
    j = int(np.argmin(np.abs(st.maf - 0.3)))
    beta[j] = 0.25
    s = _genetic_signal(st, beta)
    mu = _mu(st, beta, s)
    out = noisy_stats(st, beta, mu, g)
    var_s = float((s.double() ** 2).mean() - s.double().mean() ** 2)
    y = (s.double() + math.sqrt(1 - var_s) * torch.as_tensor(out["eps"], device=ctx.device)).cpu().numpy()
    G = torch.cat([P.G[torch.as_tensor(r, device=ctx.device)] for P, r in st.rows], 0).double().cpu().numpy()
    C = np.ones((len(y), 1)) if st.Q is None else st.Q.double().cpu().numpy()
    ok = st.maf >= 0.01
    b, se, t = _ols_reference(G[:, ok], y, C)
    for name, ref in (("beta", b), ("se", se), ("z", t)):
        got = out[name].cpu().numpy()[ok]
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-3)
        assert np.max(rel) < 2e-3, (name, np.max(rel))
    # the noise-free t (mu) is close to the realised t at the causal variant for a strong effect
    assert abs(float(out["z"][j]) - float(mu[j])) < 6


def test_factor_form_ols_transform(ctx):
    """Factor form: beta = z_gen se0, se_ols = se0 sqrt(1 - z_gen²/N), z = beta/se_ols (D6(a))."""
    g = np.random.default_rng(2)
    N = 20_000
    st = make_factor_study(ctx, TraitSpec("gwas", N, "factor", "NFE"))
    beta = torch.zeros(ctx.p, device=ctx.device)
    j = int(np.argmin(np.abs(st.maf - 0.3)))
    beta[j] = 60.0 / math.sqrt(N)        # lambda = 60: z²/N = 0.18, the transform matters
    s = _genetic_signal(st, beta)
    mu = _mu(st, beta, s)
    out = noisy_stats(st, beta, mu, g)
    f = st.af
    se0 = 1 / np.sqrt(2 * N * f * (1 - f))
    b, se, z = (out[k].cpu().numpy() for k in ("beta", "se", "z"))
    zg = b / se0
    assert np.allclose(se, se0 * np.sqrt(1 - zg ** 2 / N), rtol=1e-10)
    assert np.allclose(z, b / se, rtol=1e-12)
    assert abs(z[j] / zg[j] - 1 / math.sqrt(1 - zg[j] ** 2 / N)) < 1e-10
    # OLS identity n se² + b² = var(y)/var(x) with var(y) = 1 (sdY = 1)
    assert np.allclose(N * se ** 2 + b ** 2, 1 / (2 * f * (1 - f)), rtol=1e-9)


CASES = [("L1", "P1", 0.3, ("gwas", "mol")), ("L1", "P2", 0.2, ("gwas", "mol")),
         ("L2", "P3", 0.5, None)]


@pytest.mark.parametrize("level,pairing,ov,kinds", CASES)
def test_trait_types_sizes_and_lambda(ctx, level, pairing, ov, kinds):
    """v4 §1: trait types by pairing (P3 eQTL x eQTL mol/mol, lab x lab gwas/gwas); P2 N1 = 100k, N2 = 50k;
    lambda = sqrt(N q) for PVE-specified effects (D6(b)); total PVE <= 0.5."""
    cell = CellSpec("x", level, pairing, ov)
    seen = set()
    for k in range(12):
        c = draw_candidate(ctx, cell, rng(20261029, "v4conf", level, pairing, k))
        if c is None:
            continue
        loc, tr, _ = finalize(ctx, c, rng(20261029, "v4conf_fin", level, pairing, k), f"t{k}")
        k1, k2 = tr["trait1_type"], tr["trait2_type"]
        if pairing == "P3":
            assert (k1, k2) == (("mol", "mol") if tr["pairing_sub"] == "eqtl" else ("gwas", "gwas"))
            seen.add(tr["pairing_sub"])
            assert abs(abs(tr["rho"]) - ov) < 1e-12
        else:
            assert (k1, k2) == kinds
        if pairing == "P2":
            assert (tr["N1"], tr["N2"]) == (100_000, 50_000) and abs(tr["r_p"] - ov * math.sqrt(2)) < 1e-12
        assert tr["pve1"] <= params.MAX_TOTAL_PVE + 1e-9 and tr["pve2"] <= params.MAX_TOTAL_PVE + 1e-9
        for t, eff, N in ((1, c.arch.eff1, c.ps.t1.N), (2, c.arch.eff2, c.ps.t2.N)):
            lam = c.lam1 if t == 1 else c.lam2
            cs = c.arch.c1 if t == 1 else c.arch.c2
            for i, jw in enumerate(cs):
                spec = eff[jw]
                if spec[0] == "pve" and not c.arch.bg and c.ps.t1.pop not in ("FIN", "L6c"):
                    assert abs(abs(lam[i]) - math.sqrt(N * spec[1])) < 1e-3 * max(1, abs(lam[i]))
                    lo, hi = params.PVE_RANGE[k1 if t == 1 else k2][0][0], 0.02 if (k1 if t == 1 else k2) == "gwas" else 0.3
                    assert lo <= spec[1] <= hi
    if pairing == "P3":
        assert seen == {"eqtl", "lab"}


def test_population_levels_only_rho0():
    from sim.generator import draw_pair_spec
    for lvl in ("L3", "L6"):
        for pr, ov in (("P1", 0.6), ("P2", 0.3), ("P3", 0.5)):
            with pytest.raises(ValueError):
                draw_pair_spec(CellSpec("x", lvl, pr, ov), np.random.default_rng(0))


def test_shifted_arm_architecture(ctx):
    """Shifted arm (§8.5): up to 4 causal variants per trait in M loci, biobank-strong and background ~50%."""
    cell = CellSpec("shifted", "L1swap", "P1", 0.0, p1_mode="nested", ltype_weights=params.SHIFTED_LTYPES,
                    arch=params.SHIFTED_ARCH)
    n_m4, strong, bg, n = 0, 0, 0, 0
    types = {}
    for k in range(80):
        c = draw_candidate(ctx, cell, rng(20261029, "shifted_test", k))
        if c is None:
            continue
        n += 1
        types[c.arch.ltype] = types.get(c.arch.ltype, 0) + 1
        strong += c.arch.strong1
        bg += c.arch.bg
        if c.arch.ltype == "M":
            assert 2 <= len(c.arch.c1) <= 4 and 2 <= len(c.arch.c2) <= 4
            n_m4 += max(len(c.arch.c1), len(c.arch.c2)) == 4
    assert n >= 70
    assert n_m4 > 0
    assert 0.3 < strong / n < 0.7 and 0.3 < bg / n < 0.7
    assert types.get("W", 0) > types.get("S", 0)
