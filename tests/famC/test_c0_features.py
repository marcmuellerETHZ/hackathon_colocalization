"""Family C: C0 world, batched coloc.abf targets, GPU screen and allele-flip invariance (known answers)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from baselines.abf import coloc_abf
from famB.screen import LDAccess, screen
from famC.c0world import coloc_abf_torch, coloc_prior_probs, sample_c0
from famC.features import DenseLD, FactorLD, featurize, screen_gpu
from famC.ldbank import LDBank, torch_gen
from famC.model import NeuralColoc

SEED = 20261030
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def bank():
    return LDBank("train", 6, SEED, device="cuda")


@pytest.fixture(scope="module")
def batch(bank):
    g = torch_gen(SEED, "test", "batch", device="cuda")
    gc = torch_gen(SEED, "test", "batchc")
    return sample_c0(bank, g, gc, n_win=4, per_win=8, hyp_probs="uniform")


def test_bank_is_train_and_refuses_test(bank):
    assert bank.split == "train" and len(bank) == 6
    with pytest.raises(PermissionError):
        LDBank("test", 1, SEED)


def test_coloc_torch_matches_local_abf(batch):
    """Batched fp64 coloc.abf == baselines.abf.coloc_abf (itself validated vs R coloc 5.2.3 to < 1e-8)."""
    pp, snp4 = batch["pp"].cpu().numpy(), batch["snp_pp4"].cpu().numpy()
    mask = batch["mask"].cpu().numpy()
    worst = 0.0
    for b in range(pp.shape[0]):
        m = mask[b]
        z1, z2 = batch["z1"][b].cpu().numpy()[m], batch["z2"][b].cpu().numpy()[m]
        s1, s2 = batch["se1"][b].cpu().numpy()[m], batch["se2"][b].cpu().numpy()[m]
        ref, rsnp = coloc_abf(z1, s1, z2, s2, W=0.15, p12=1e-5, return_snp_pp=True)
        worst = max(worst, np.abs(ref - pp[b]).max(), np.abs(rsnp - snp4[b][m]).max())
    assert worst < 1e-10, worst


def test_coloc_prior_probs_known_answer():
    p = torch.tensor([3000, 20000])
    pr = coloc_prior_probs(p).numpy()
    odds = np.array([1, 0.3, 0.3, 3000 * 2999 * 1e-8, 0.03])
    assert np.allclose(pr[0], odds / odds.sum())
    # p * p1 >= 1 -> adjusted to 1/(p+1) (coloc adjust_prior)
    a = 1 / 20001
    odds = np.array([1, 20000 * a, 20000 * a, 20000 * 19999 * a * a, 20000 * 1e-5])
    assert np.allclose(pr[1], odds / odds.sum())
    assert np.allclose(pr.sum(1), 1)


def test_c0_noise_covariance_is_R(bank):
    """z under H0 ~ N(0, R): empirical covariance over many draws matches R (window 0)."""
    g = torch_gen(SEED, "test", "cov", device="cuda")
    gc = torch_gen(SEED, "test", "covc")
    zs = []
    for _ in range(40):
        b = sample_c0(bank, g, gc, n_win=1, per_win=256, hyp_probs=[1, 0, 0, 0, 0], widx=[0])
        zs.append(b["z1"])
    Z = torch.cat(zs).double()[:, :200]
    X, _ = bank.standardized([0])
    R = (X[0].T @ X[0] / X.shape[1]).double()[:200, :200]
    C = Z.T @ Z / Z.shape[0]
    # sd of an empirical covariance entry with n draws is <= sqrt((1 + r^2)/n)
    n = Z.shape[0]
    err = ((C - R).abs() / torch.sqrt((1 + R * R) / n)).max().item()
    assert err < 5.5, err
    assert abs(Z.mean().item()) < 0.02


def test_c0_signal_mean(bank):
    """E[z_1 | H1, causal j, lambda] = lambda R[:, j]: average of noise-only draws plus the deterministic mean."""
    g = torch_gen(SEED, "test", "mean", device="cuda")
    gc = torch_gen(SEED, "test", "meanc")
    b = sample_c0(bank, g, gc, n_win=1, per_win=64, hyp_probs=[0, 1, 0, 0, 0], widx=[1])
    X, _ = bank.standardized([1])
    R = (X[0].T @ X[0] / X.shape[1]).double()
    resid = b["z1"] - b["lam1"][:, None] * R[b["causal1"]]
    # residual = noise ~ N(0, R): per-SNP unit variance
    v = resid[:, b["mask"][0]].var().item()
    assert 0.9 < v < 1.1, v
    assert (b["causal2"] == -1).all() and (b["lam2"] == 0).all()


def _famb_sel(z1, z2, ld_np, M):
    r = screen(z1, z2, ld_np, M=M, collapse=False)
    return r.sel, r.source


def test_screen_matches_famB_factor(batch):
    """Exact identity in fp64 (production uses fp32 LD rows; near-ties in |zc| may then order differently)."""
    ld = FactorLD(batch["X"].double(), 8)
    sc = screen_gpu(batch["z1"], batch["z2"], batch["mask"], ld, M=64)
    X = batch["X"].double().cpu().numpy()
    sc32 = screen_gpu(batch["z1"], batch["z2"], batch["mask"], FactorLD(batch["X"], 8), M=64)
    overlap = np.mean([len(set(sc.sel[b].tolist()) & set(sc32.sel[b].tolist())) / 64 for b in range(sc.sel.shape[0])])
    assert overlap > 0.97, overlap
    for b in range(batch["z1"].shape[0]):
        w = b // 8
        m = batch["mask"][b].cpu().numpy()
        Xw = X[w][:, m]
        sel, src = _famb_sel(batch["z1"][b].cpu().numpy()[m], batch["z2"][b].cpu().numpy()[m],
                             LDAccess(X=Xw, n_ref=Xw.shape[0]), 64)
        assert np.array_equal(sel, sc.sel[b].cpu().numpy()), b
        assert np.array_equal(src, sc.source[b].cpu().numpy()), b


def test_screen_dense_equals_factor_and_small_p(batch):
    X = batch["X"][:1].double()
    R = (X[0].T @ X[0] / X.shape[1])[None]
    z1, z2, mask = batch["z1"][:1], batch["z2"][:1], batch["mask"][:1]
    a = screen_gpu(z1, z2, mask, DenseLD(R), M=64)
    b = screen_gpu(z1, z2, mask, FactorLD(X, 1), M=64)
    assert torch.equal(a.sel, b.sel)
    # p <= M: everything selected, source 4
    small = mask.clone()
    small[:, 40:] = False
    c = screen_gpu(z1, z2, small, DenseLD(R), M=64)
    assert c.valid.sum().item() == 40 and set(c.sel[0, :40].tolist()) == set(range(40))
    assert (c.source[0, :40] == 4).all() and (c.source[0, 40:] == -1).all()


def test_allele_flip_invariance(batch):
    """Flipping the coding allele of random variants (z sign, genotype column sign) leaves every model input and
    the network output unchanged (exact sign-equivariance by canonicalisation)."""
    torch.manual_seed(0)
    b = {k: v for k, v in batch.items()}
    f0 = featurize(b, FactorLD(b["X"], 8), M=64)
    P = b["z1"].shape[1]
    s = torch.where(torch.rand(P, device="cuda") < 0.5, -1.0, 1.0)
    b2 = dict(b)
    b2["z1"] = b["z1"] * s
    b2["z2"] = b["z2"] * s
    X2 = b["X"] * s.float()
    f1 = featurize(b2, FactorLD(X2, 8), M=64)
    assert torch.equal(f0["sel"], f1["sel"])
    for k in ("tok", "Rmm", "q_tok", "q_all", "all_feat", "glob"):
        assert torch.allclose(f0[k], f1[k], atol=1e-5), k
    net = NeuralColoc(d=32, layers=1, H=2, K=4).cuda().eval()
    with torch.no_grad():
        assert torch.allclose(net(f0)["logits"], net(f1)["logits"], atol=1e-4)


def test_trait_flip_feature_effect(batch):
    """Global trait flips: z1 -> -z1 flips rho-hat's sign in the features (augmentation contract)."""
    b = dict(batch)
    b["rho_hat"] = torch.full_like(b["rho_hat"], 0.3)
    B = b["z1"].shape[0]
    fl = torch.ones(B, 2, dtype=torch.float64, device="cuda")
    fl[:, 0] = -1
    f = featurize(b, FactorLD(b["X"], 8), M=64, flip_traits=fl)
    assert torch.allclose(f["glob"][:, 0], torch.full((B,), -0.3, device="cuda"))


def test_no_ld_ablation_has_no_ld(batch):
    f = featurize(batch, None, M=64, use_ld=False)
    assert f["Rmm"] is None and float(f["all_feat"][..., 7:9].abs().max()) == 0
    net = NeuralColoc(d=32, layers=1, H=2, K=4, use_ld=False).cuda()
    out = net(f)
    assert out["logits"].shape == (batch["z1"].shape[0], 5)
