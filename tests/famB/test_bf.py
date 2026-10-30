"""DESIGN Sec. 7.1(4): support-space BF == full-form RSS BF (< 1e-8 relative), incl. rho in {0, .4, -.7},
rank-1 shared prior, 2+2 configurations; mpmath reference for near-singular / rank-deficient Rt."""
import numpy as np
import pytest
import torch

from famB.enum import kernel
from ref import (active_only_logbf, full_form_logbf, full_form_logbf_mp, genotype_ld, tensors_for, term_arrays)

RHOS = (0.0, 0.4, -0.7)


def _configs(p):
    j, k, a, c = 3, 11, 7, 15
    return {
        "D (1:j, 2:k)": ((j,), (k,), 0.0),
        "S correlated": ((j,), (j,), 0.6),
        "S rank-1": ((j,), (j,), 1.0),
        "S rank-1 neg": ((j,), (j,), -1.0),
        "2+2 distinct": ((j, a), (k, c), 0.0),
        "2+2 one shared corr": ((j, a), (j, c), 0.5),
        "2+2 both shared rank-1": ((j, a), (j, a), 1.0),
        "1 only": ((a,), (), 0.0),
        "2 only trait2": ((), (k, c), 0.0),
    }


def _signal(rng, Rt, rho, p):
    lam1 = np.zeros(p); lam2 = np.zeros(p)
    lam1[3] = 7.0; lam1[7] = -4.0; lam2[3] = 5.0; lam2[11] = 6.0
    V = np.array([[1, rho], [rho, 1]])
    C = np.kron(V, Rt)
    e = np.linalg.cholesky(C + 1e-12 * np.eye(2 * p)) @ rng.standard_normal(2 * p)
    return np.r_[Rt @ lam1, Rt @ lam2] + e


def _support(z, Rt, rho, S1, S2, sd, rb, device):
    T = tensors_for(z, Rt, rho, sd, rb, device)
    V, m, kx = term_arrays([(S1, S2)], device)
    lw, _ = kernel(T, V, m, kx)
    return float(lw[0, 0])


@pytest.mark.parametrize("rho", RHOS)
def test_support_equals_full_form_regular(rho, device):
    rng = np.random.default_rng(20261026)
    p = 40
    R = genotype_ld(rng, 2000, p)
    eta = 1e-3
    Rt = (1 - eta) * R + eta * np.eye(p)
    assert np.linalg.cond(np.kron(np.array([[1, rho], [rho, 1]]), Rt)) < 1e8
    z = _signal(rng, Rt, rho, p)
    sd = np.stack([np.full(p, 5.0), np.full(p, 3.0)])
    worst = 0.0
    for name, (S1, S2, rb) in _configs(p).items():
        ref = full_form_logbf(z, Rt, rho, S1, S2, sd, rb)
        got = _support(z, Rt, rho, S1, S2, sd, rb, device)
        rel = abs(got - ref) / max(1.0, abs(ref))
        worst = max(worst, rel)
        assert rel < 1e-8, (name, rho, got, ref)
    print(f"rho={rho}: worst rel err {worst:.2e}")


@pytest.mark.parametrize("rho", RHOS)
@pytest.mark.parametrize("case", ["near_singular_r2_0.985", "rank_deficient_n15", "exact_duplicate"])
def test_support_equals_mpmath_near_singular(rho, case, device):
    rng = np.random.default_rng(20261027)
    p = 20
    if case == "near_singular_r2_0.985":
        R = genotype_ld(rng, 3000, p, dup_pairs=[(3, 4, 0.124)])
        assert abs(R[3, 4]) ** 2 > 0.98
        eta = 1e-3
        cfgs = {"D j,j+1": ((3,), (4,), 0.0), "2+2 dup": ((3, 4), (4, 11), 0.3), "S rank-1": ((3,), (3,), 1.0)}
    elif case == "rank_deficient_n15":
        R = genotype_ld(rng, 15, p)
        assert np.linalg.matrix_rank(R) < p
        eta = 1e-3
        cfgs = {"D": ((3,), (11,), 0.0), "2+2": ((3, 7), (11, 15), 0.0), "S rank-1": ((3,), (3,), 1.0)}
    else:
        R = genotype_ld(rng, 500, p, dup_pairs=[(3, 4, 0.0)])
        eta = 1e-3
        cfgs = {"dup D": ((3,), (4,), 0.0), "dup 2": ((3, 4), (3,), 0.5)}
    Rt = (1 - eta) * R + eta * np.eye(p)
    z = _signal(rng, Rt, rho, p)
    sd = np.stack([np.full(p, 6.0), np.full(p, 4.0)])
    for name, (S1, S2, rb) in cfgs.items():
        ref = full_form_logbf_mp(z, Rt, rho, S1, S2, sd, rb, dps=60)
        got = _support(z, Rt, rho, S1, S2, sd, rb, device)
        rel = abs(got - ref) / max(1.0, abs(ref))
        assert rel < 1e-8, (case, name, rho, got, ref)


def test_negative_control_active_only_b_is_wrong(device):
    """Building b from the active z's only must NOT match (sensitivity of the test above)."""
    rng = np.random.default_rng(20261028)
    p = 40
    R = genotype_ld(rng, 2000, p)
    Rt = 0.999 * R + 1e-3 * np.eye(p)
    rho = 0.4
    z = _signal(rng, Rt, rho, p)
    sd = np.stack([np.full(p, 5.0), np.full(p, 3.0)])
    ref = full_form_logbf(z, Rt, rho, (3,), (11,), sd, 0.0)
    wrong = active_only_logbf(z, Rt, rho, (3,), (11,), sd, 0.0)
    got = _support(z, Rt, rho, (3,), (11,), sd, 0.0, device)
    assert abs(got - ref) / abs(ref) < 1e-8
    assert abs(wrong - ref) / abs(ref) > 1e-2, (wrong, ref)


def test_fp32_kernel_close(device):
    """fp32 kernel (optional speed mode) agrees with fp64 to ~1e-5 relative on regular loci."""
    rng = np.random.default_rng(20261029)
    p = 30
    R = genotype_ld(rng, 2000, p)
    Rt = 0.999 * R + 1e-3 * np.eye(p)
    z = _signal(rng, Rt, 0.4, p)
    sd = np.stack([np.full(p, 5.0), np.full(p, 3.0)])
    T = tensors_for(z, Rt, 0.4, sd, 0.5, device)
    V, m, kx = term_arrays([c[:2] for c in _configs(p).values()], device)
    a, _ = kernel(T, V, m, kx, dtype=torch.float64)
    b, _ = kernel(T, V, m, kx, dtype=torch.float32)
    assert torch.max(torch.abs(a - b) / torch.clamp(torch.abs(a), min=1.0)) < 1e-4
