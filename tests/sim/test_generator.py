"""Generator verification (DESIGN v4 §7.1 item 1, 8, 10). Needs the GPU and data/processed (chr22).

  * factor form vs individual level: same mean and covariance of z;
  * realised null cross-trait correlation equals the target rho for every overlap mode;
  * in-sample LD noise has the predicted (delta-method) magnitude;
  * oracle power equals the empirical detection rate (1,000 replicates per mu, +-2 SE);
  * TF32 is off.
"""
import json
import math
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

from sim import generator as gen
from sim.generator import PairSpec, TraitSpec, WindowContext, _genetic_signal, _mu, _noisy_z
from sim.oracle import power
from sim.params import ALPHA
from scipy.stats import norm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
OUT = Path(__file__).resolve().parents[2] / "src/sim/validation"
SEED = 20261026


@pytest.fixture(scope="module")
def ctx():
    gen.assert_no_tf32()
    c = WindowContext("chr22", 30_000_000, 30_080_000)
    assert 100 < c.p < 600, c.p
    return c


def _rng(*k):
    return np.random.default_rng([SEED, *k])


def _record(name, payload):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "generator_tests.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    d[name] = payload
    f.write_text(json.dumps(d, indent=1, default=float))


def test_tf32_off():
    gen.assert_no_tf32()
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    import os
    assert os.environ.get("NVIDIA_TF32_OVERRIDE") == "0", "source env/activate.sh"


def test_factor_vs_individual_mean_cov(ctx):
    """With all N_syn rows, the individual-level study's in-sample LD equals R_pop, so factor-form and
    individual-level z must have the same mean (R lambda) and covariance (R)."""
    P = ctx.pop("NFE")
    g = _rng(1)
    sf = gen.make_factor_study(ctx, TraitSpec("gwas", P.N, "factor", "NFE"))
    si = gen.make_indiv_study(ctx, TraitSpec("gwas", P.N, "indiv", "NFE"), g)
    maf = np.minimum(P.af, 1 - P.af)
    j = int(np.argmin(np.abs(maf - 0.3)))
    beta = torch.zeros(ctx.p, device=ctx.device)
    beta[j] = 5.0 / math.sqrt(P.N)                       # lambda = 5
    mf = _mu(sf, beta, _genetic_signal(sf, beta))
    mi = _mu(si, beta, _genetic_signal(si, beta))
    assert torch.allclose(mf, mi, atol=0.02), (mf - mi).abs().max()
    reps = 2000
    Zf = torch.stack([_noisy_z(sf, beta, mf, g)[0] for _ in range(reps)]).double()
    Zi = torch.stack([_noisy_z(si, beta, mi, g)[0] for _ in range(reps)]).double()
    R = (P.std.T @ P.std / P.N).double()
    # means
    se = math.sqrt(2.0 / reps)
    dmean = ((Zf.mean(0) - Zi.mean(0)) / se).abs().max().item()
    # covariances: whiten with R^{-1/2} on an LD-pruned subset (r² < 0.8, well conditioned), so that
    # entries of the sample covariance are ~independent; standardized deviations from I have RMS ~ 1.
    from stats.rho import prune
    sub = torch.as_tensor(prune(R.cpu().numpy(), 0.8), device=ctx.device)
    ev, V = torch.linalg.eigh(R[sub][:, sub])
    assert ev.min() > 1e-3, ev.min()
    Wh = V @ torch.diag(ev ** -0.5) @ V.T
    q = len(sub)
    iu = torch.triu_indices(q, q, 0)
    var_entry = (torch.eye(q, device=ctx.device, dtype=torch.float64) + 1) / reps   # 2/n diag, 1/n off-diag

    def rms(Z):
        W = (Z[:, sub] - Z[:, sub].mean(0)) @ Wh
        C = W.T @ W / (reps - 1)
        I = torch.eye(q, device=ctx.device, dtype=torch.float64)
        return float(torch.sqrt((((C - I) ** 2) / var_entry)[iu[0], iu[1]].mean()))

    rf, ri = rms(Zf), rms(Zi)
    _record("factor_vs_indiv", {"p": ctx.p, "p_whitened": q, "reps": reps, "max_std_mean_diff": dmean,
                                "cov_rms_std_factor": rf, "cov_rms_std_indiv": ri, "mean_lambda5_factor": float(mf[j]),
                                "mean_lambda5_indiv_emp": float(Zi.mean(0)[j])})
    assert dmean < 4.5
    assert 0.9 < rf < 1.1 and 0.9 < ri < 1.1, (rf, ri)


OVERLAP_CASES = [
    # name, PairSpec builder (N1, N2, r_p, mode), target rho
    ("P2_factor_rho0.3", "factor2", 100_000, 50_000, None, 0.3),
    ("P3_lab_rho-0.8", "factor2", 100_000, 100_000, None, -0.8),
    ("P1_nested_N1=20k_N2=2k_rp0.6", "nested", 20_000, 2_000, 0.6, 0.6 * math.sqrt(2_000 / 20_000)),
    ("P1_nested_N1=100k_N2=5k_rp0.6", "nested", 100_000, 5_000, 0.6, 0.6 * math.sqrt(5_000 / 100_000)),
    ("P1_disjoint_rp0.6", "disjoint", 100_000, 1_000, 0.0, 0.0),
    ("P3_eqtl_N=1000_rp0.5", "eqtl", 1_000, 1_000, 0.5, 0.5),
]


@pytest.mark.parametrize("name,mode,N1,N2,r_p,rho", OVERLAP_CASES)
def test_realised_rho(ctx, name, mode, N1, N2, r_p, rho):
    """Null loci: E[z1_j z2_k] = rho R_jk; check the diagonal mean and the slope on R over all pairs."""
    g = _rng(2, zlib.crc32(name.encode()))
    if mode == "factor2":
        ps = PairSpec("P2", "factor2", TraitSpec("gwas", N1, "factor", "NFE"), TraitSpec("mol", N2, "factor", "NFE"),
                      rho * math.sqrt(N1 / N2), rho, N2, False)
    elif mode in ("nested", "disjoint"):
        rr = r_p if mode == "nested" else 0.0
        ps = PairSpec("P1", mode, TraitSpec("gwas", N1, "factor", "NFE"), TraitSpec("mol", N2, "indiv", "NFE"),
                      rr, N2 * rr / math.sqrt(N1 * N2) if mode == "nested" else 0.0, N2 if mode == "nested" else 0, False)
    else:
        ps = PairSpec("P3", "eqtl", TraitSpec("mol", N1, "indiv", "NFE"), TraitSpec("mol", N2, "indiv", "NFE"),
                      r_p, r_p, N1, True)
    assert abs(ps.rho - rho) < 1e-12
    zero = torch.zeros(ctx.p, device=ctx.device)
    reps = 600
    diag, prods = [], torch.zeros(ctx.p, ctx.p, device=ctx.device, dtype=torch.float64)
    for r in range(reps):
        s1, s2 = gen.make_studies(ctx, ps, g)
        z1, e1, u1 = _noisy_z(s1, zero, zero, g)
        z2, _, _ = _noisy_z(s2, zero, zero, g, partner=(ps, s1, e1, u1))
        z1, z2 = z1.double(), z2.double()
        diag.append(float((z1 * z2).mean()))
        prods += torch.outer(z1, z2)
    diag = np.array(diag)
    est, se = diag.mean(), diag.std() / math.sqrt(reps)
    P = ctx.pop("NFE")
    R = (P.std.T @ P.std / P.N).double()
    C = prods / reps
    slope = float((C * R).sum() / (R * R).sum())
    _record(f"realised_rho::{name}", {"target": rho, "diag_mean": est, "diag_se": se, "slope_on_R": slope, "reps": reps})
    assert abs(est - rho) < max(4 * se, 0.01), (est, se, rho)
    assert abs(slope - rho) < max(4 * se, 0.015), (slope, rho)


def test_in_sample_ld_noise(ctx):
    """R_t from N_t study rows vs R_pop: empirical Var(r_t - r_pop) over 300 resamples equals the delta-method
    prediction from population moments (with finite-population correction); normal theory (1-r²)²/N reported."""
    P = ctx.pop("NFE")
    Z = P.std.double()
    g = _rng(3)
    maf = np.minimum(P.af, 1 - P.af)
    ok = np.where(maf > 0.05)[0]
    pairs = np.array([(a, b) for a, b in zip(g.choice(ok, 60), g.choice(ok, 60)) if a != b])
    a_t = torch.as_tensor(pairs[:, 0], device=ctx.device)
    b_t = torch.as_tensor(pairs[:, 1], device=ctx.device)
    x, y = Z[:, a_t], Z[:, b_t]
    rho = (x * y).mean(0)
    m22, m31, m13 = (x ** 2 * y ** 2).mean(0), (x ** 3 * y).mean(0), (x * y ** 3).mean(0)
    m40, m04 = (x ** 4).mean(0), (y ** 4).mean(0)
    v_delta = (1 + rho ** 2 / 2) * m22 - rho * (m31 + m13) + rho ** 2 / 4 * (m40 + m04)
    res = {}
    for N in (100, 300, 1000):
        rs = []
        for _ in range(300):
            st = gen.make_indiv_study(ctx, TraitSpec("mol", N, "indiv", "NFE"), g)
            X = st.X
            rs.append(((X[:, a_t] * X[:, b_t]).mean(0)).double())
        rs = torch.stack(rs)
        emp = ((rs - rho) ** 2).mean(0)
        fpc = 1 - N / P.N
        ratio_delta = float((emp / (v_delta * fpc / N)).mean())
        rn = emp / ((1 - rho ** 2) ** 2 * fpc / N)
        ratio_normal = float(rn.mean())
        res[N] = {"ratio_emp_over_delta": ratio_delta, "ratio_emp_over_normal_theory": ratio_normal,
                  "median_ratio_emp_over_normal_theory": float(rn.median()),
                  "ratio_of_means_emp_over_normal_theory": float(emp.sum() / ((1 - rho ** 2) ** 2 * fpc / N).sum())}
        assert 0.85 < ratio_delta < 1.15, (N, ratio_delta)
    _record("in_sample_ld_noise", res)


@pytest.mark.parametrize("form", ["indiv", "factor"])
def test_oracle_power_equals_detection(ctx, form):
    """1,000 replicates per mu; empirical P(|z| > z_alpha) at the causal variant within +-2 SE of power(mu)."""
    P = ctx.pop("NFE")
    g = _rng(4, form == "factor")
    if form == "indiv":
        spec, kind = TraitSpec("mol", 1000, "indiv", "NFE"), "mol"
        st = gen.make_indiv_study(ctx, spec, g)
    else:
        spec, kind = TraitSpec("gwas", 100_000, "factor", "NFE"), "gwas"
        st = gen.make_factor_study(ctx, spec)
    alpha = ALPHA[kind]
    za = norm.isf(alpha / 2)
    maf = np.minimum(P.af, 1 - P.af)
    j = int(np.argmin(np.abs(maf - 0.25)))
    out = []
    for target in (2.0, 3.58, 4.61, 5.26, 6.29, 8.0):
        beta = torch.zeros(ctx.p, device=ctx.device)
        beta[j] = 1.0
        mu1 = float(_mu(st, beta * 1e-3, _genetic_signal(st, beta * 1e-3))[j]) / 1e-3   # d mu / d beta (approx.)
        b = target / mu1
        for _ in range(3):   # refine for the non-linear t-statistic
            beta[j] = b
            m = float(_mu(st, beta, _genetic_signal(st, beta))[j])
            b *= target / m
        beta[j] = b
        mu = _mu(st, beta, _genetic_signal(st, beta))
        m = float(mu[j])
        hits = sum(abs(float(_noisy_z(st, beta, mu, g)[0][j])) > za for _ in range(1000))
        emp, pw = hits / 1000, float(power(m, alpha))
        se = math.sqrt(max(pw * (1 - pw), 1e-4) / 1000)
        out.append({"mu": m, "power": pw, "empirical": emp, "se": se, "z": (emp - pw) / se})
    from scipy.stats import chi2
    within_2se = [abs(o["z"]) <= 2 for o in out]
    p_joint = float(chi2.sf(sum(o["z"] ** 2 for o in out), len(out)))
    _record(f"oracle_power::{form}", {"points": out, "within_2se": within_2se, "joint_chi2_p": p_joint})
    # DESIGN §7.1.1 says +-2 SE per mu. Applied to 6 mu x 2 forms that rule fails by chance with ~43%
    # probability, so the assertion is multiplicity-aware: every point within 3 SE and joint chi² p > 0.01;
    # the per-point 2-SE outcomes are recorded (README D15; deviation reported to the PI).
    for o in out:
        assert abs(o["z"]) <= 3, o
    assert p_joint > 0.01, p_joint


def test_fp32_mu_matches_fp64(ctx):
    """DESIGN §7.1.10 (generator side): realised noncentralities computed in fp32 on the GPU agree with an
    fp64 recomputation to < 1e-4 (absolute, on the z scale) for factor and individual-level studies."""
    P = ctx.pop("NFE")
    g = _rng(6)
    beta = torch.zeros(ctx.p, device=ctx.device)
    idx = g.choice(ctx.p, 3, replace=False)
    beta[torch.as_tensor(idx, device=ctx.device)] = torch.tensor([0.03, -0.02, 0.025], device=ctx.device)
    sf = gen.make_factor_study(ctx, TraitSpec("gwas", 100_000, "factor", "NFE"))
    m32 = _mu(sf, beta, _genetic_signal(sf, beta)).double()
    Z = P.std.double()
    m64 = math.sqrt(100_000) * (Z.T @ (Z @ beta.double())) / P.N
    assert (m32 - m64).abs().max() < 1e-4 * max(1.0, float(m64.abs().max()))
    si = gen.make_indiv_study(ctx, TraitSpec("mol", 1000, "indiv", "NFE"), g)
    b2 = beta * 3
    m32 = _mu(si, b2, _genetic_signal(si, b2)).double()
    X = si.X.double()
    s = X @ b2.double()
    r = (X.T @ (s - s.mean())) / (torch.linalg.vector_norm(X, dim=0) * math.sqrt(float(((s - s.mean()) ** 2).sum()) + (1 - float(s.var(unbiased=False))) * (1000 - 1)))
    m64 = r * math.sqrt(1000 - 2) / torch.sqrt(1 - r * r)
    assert (m32 - m64).abs().max() < 1e-4 * max(1.0, float(m64.abs().max()))
