"""M3.5 power events (DESIGN v4 §1): own-effect AND marginal-noncentrality sets, trait-type thresholds.

Reference: an independent formulation that decomposes every coordinate set into disjoint intervals and integrates
the conditional bivariate normal with scipy's adaptive quadrature (no inclusion–exclusion over holes)."""
import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import norm

from famA.bvn import bvn_set_prob, nonweak_spec, powered_spec, set_prob, weak_spec
from famA.joint_susie import TYPE_THRESHOLDS, JointConfig, joint_susie_batch, thresholds_for_types
from famag_testutils import ar_genotypes, toy_locus

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="CUDA required")


def test_type_thresholds_match_oracle():
    from sim.params import TAU, WEAK
    for t in ("gwas", "mol"):
        assert TYPE_THRESHOLDS[t] == (TAU[t], WEAK[t])
    thr = thresholds_for_types([("gwas", "mol"), ("mol", "mol"), ("gwas", "gwas")], JointConfig())
    assert thr.tolist() == [[6.29, 5.26, 4.61, 3.58], [5.26, 5.26, 3.58, 3.58], [6.29, 6.29, 4.61, 4.61]]
    with pytest.raises(ValueError):
        thresholds_for_types([("gwas", "eqtl")], JointConfig())


def _disjoint(spec):
    """Disjoint closed intervals of a set spec (floats)."""
    kind = spec[0]
    if kind == "in":
        return [(spec[1], spec[2])] if spec[1] <= spec[2] else []
    a = sorted([(spec[1], spec[2]), (spec[3], spec[4])])
    merged = [list(a[0])]
    for lo, hi in a[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    if kind == "in2":
        return [tuple(x) for x in merged]
    out, prev = [], -np.inf
    for lo, hi in merged:
        out.append((prev, lo))
        prev = hi
    out.append((prev, np.inf))
    return out


def _ref(m1, m2, s11, s22, s12, spec1, spec2):
    sd1 = np.sqrt(s11); b = s12 / s11; cs = np.sqrt(max(s22 - s12 * b, 0))
    I2 = _disjoint(spec2)

    def g(x):
        cm = m2 + b * (x - m1)
        return norm.pdf(x, m1, sd1) * sum(norm.cdf(hi, cm, cs) - norm.cdf(lo, cm, cs) for lo, hi in I2)
    tot = 0.0
    for a, c in _disjoint(spec1):
        lo, hi = max(a, m1 - 12 * sd1), min(c, m1 + 12 * sd1)
        if lo < hi:
            tot += quad(g, lo, hi, epsabs=1e-14, epsrel=1e-12, limit=500, points=[x for x in (a, c) if lo < x < hi] or None)[0]
    return tot


def _spec_float(spec):
    return (spec[0],) + tuple(float(v) for v in spec[1:])


def test_set_prob_1d_vs_scipy():
    rng = np.random.default_rng(20261029)
    for _ in range(200):
        mu, sd, m = rng.normal(0, 7), np.exp(rng.uniform(-3, 1.5)), rng.normal(0, 6)
        for spec in (powered_spec(torch.tensor(m), 6.29), nonweak_spec(torch.tensor(m), 4.61),
                     weak_spec(torch.tensor(m), 4.61), weak_spec(torch.tensor(m), 3.58, own=False)):
            sp = _spec_float(spec)
            ref = sum(norm.cdf(hi, mu, sd) - norm.cdf(lo, mu, sd) for lo, hi in _disjoint(sp))
            got = set_prob(torch.tensor(mu, dtype=torch.float64), torch.tensor(sd, dtype=torch.float64), spec).item()
            assert abs(got - ref) < 1e-12, (sp, mu, sd, got, ref)


def test_bvn_set_prob_vs_quadrature():
    rng = np.random.default_rng(20261028)
    worst = 0.0
    t = lambda x: torch.tensor([x], dtype=torch.float64)
    for k in range(60):
        m1, m2 = rng.normal(0, 7, 2)
        s11, s22 = np.exp(rng.uniform(np.log(0.05), np.log(5), 2))
        s12 = rng.uniform(-0.9, 0.9) * np.sqrt(s11 * s22)
        g1, g2 = rng.normal(0, 6, 2)                        # marginal offsets m of the other effects
        tau1, tau2, w1, w2 = (6.29, 5.26, 4.61, 3.58) if k % 2 else (5.26, 5.26, 3.58, 3.58)
        pairs = [(powered_spec(t(g1), tau1), powered_spec(t(g2), tau2)),
                 (nonweak_spec(t(g1), w1), nonweak_spec(t(g2), w2)),
                 (powered_spec(t(g1), tau1), weak_spec(t(g2), w2)),
                 (weak_spec(t(g1), w1), powered_spec(t(g2), tau2)),
                 (weak_spec(t(g1), w1, own=False), nonweak_spec(t(g2), w2, own=False))]
        for s1, s2 in pairs:
            got = bvn_set_prob(t(m1), t(m2), t(s11), t(s22), t(s12), s1, s2).item()
            ref = _ref(m1, m2, s11, s22, s12, _spec_float(s1), _spec_float(s2))
            worst = max(worst, abs(got - ref))
    assert worst < 1e-6, worst


def test_event_logic():
    """powered ⊂ nonweak; the p̃ events are disjoint from blocking; marginal cancellation removes power."""
    t = lambda x: torch.tensor([x], dtype=torch.float64)
    m1, m2, s11, s22, s12 = t(7.0), t(6.0), t(1.0), t(1.5), t(0.3)
    g1, g2 = t(-2.0), t(1.0)
    P = lambda a, b: bvn_set_prob(m1, m2, s11, s22, s12, a, b).item()
    a = P(powered_spec(g1, 6.29), powered_spec(g2, 5.26))
    w = P(nonweak_spec(g1, 4.61), nonweak_spec(g2, 3.58))
    p1 = P(powered_spec(g1, 6.29), weak_spec(g2, 3.58))
    nw1 = set_prob(m1, s11.sqrt(), nonweak_spec(g1, 4.61)).item()
    assert a <= w + 1e-12
    assert abs(w + P(nonweak_spec(g1, 4.61), weak_spec(g2, 3.58)) - nw1) < 1e-9     # partition of nonweak_1
    assert p1 <= nw1 - w + 1e-9
    # own effect 8 cancelled by an LD partner (m = -8): marginal ~0, so not powered although |β| >= τ
    assert set_prob(t(8.0), t(1e-3), powered_spec(t(-8.0), 6.29)).item() < 1e-12
    assert set_prob(t(8.0), t(1e-3), powered_spec(t(-8.0), 6.29, marginal_cond=False)).item() > 1 - 1e-12


@needs_cuda
def test_trait_type_thresholds_change_power():
    """A shared effect of noncentrality ~5.8 is powered for molecular traits (τ 5.26) but not for GWAS-like
    traits (τ 6.29): P_S(mol, mol) > P_S(gwas, gwas) on the same data."""
    rng = np.random.default_rng(23)
    p, n = 200, 3000
    X = ar_genotypes(rng, n, p)
    loci = [toy_locus(rng, n, p, (80,), (5.8,), (80,), (5.8,), X=X) for _ in range(4)]
    args = ([l[2] for l in loci] * 2, [l[3] for l in loci] * 2, [l[1] for l in loci] * 2, [1e5] * 8, [1e5] * 8,
            [0.0] * 8, "dense", JointConfig())
    out = joint_susie_batch(*args, types=[("mol", "mol")] * 4 + [("gwas", "gwas")] * 4)
    ps_mol = np.mean([o["P_S"] for o in out[:4]])
    ps_gwas = np.mean([o["P_S"] for o in out[4:]])
    assert ps_mol > ps_gwas + 0.1, (ps_mol, ps_gwas)
    assert out[0]["thresholds"].tolist() == [5.26, 5.26, 3.58, 3.58]


@needs_cuda
def test_marginal_condition_cancellation_toy():
    """Two opposite-sign causal variants in strong LD for trait 1 (marginals partly cancel) and the same
    variants for trait 2: with the marginal condition P_S can only decrease relative to own-effect-only events."""
    rng = np.random.default_rng(31)
    p, n = 150, 3000
    X = ar_genotypes(rng, n, p, phi=0.95)
    R = X.T @ X / n
    lam = np.zeros(p); lam[60] = 9.0; lam[61] = -8.0
    z1 = R @ lam + X.T @ rng.standard_normal(n) / np.sqrt(n)
    z2 = R @ lam + X.T @ rng.standard_normal(n) / np.sqrt(n)
    common = ([z1], [z2], [R], [1e5], [1e5], [0.0], "dense")
    a = joint_susie_batch(*common, JointConfig(power_marginal=True))[0]
    b = joint_susie_batch(*common, JointConfig(power_marginal=False, nonweak_own=False))[0]
    assert a["P_S"] <= b["P_S"] + 1e-9
