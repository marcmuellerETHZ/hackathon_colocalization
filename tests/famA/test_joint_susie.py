"""Family A (JointSuSiE): DESIGN §5 A / §7.1 item 5 unit tests with known answers, plus identity tests."""
import itertools
import os
import subprocess

import numpy as np
import pytest
import torch

from famag_testutils import ar_genotypes, toy_locus, write_store
from eval import adapters
from famA.joint_susie import JointConfig, joint_susie_batch
from famA.method import JointSusieMethod
from famG.locus import LocusData
from famG.susie_rss import SusieRSSConfig, susie_rss_batch
from famG.validation.make_loci import LocusStore

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="CUDA required")


# ---------------- inclusion–exclusion formula ----------------------------------------------------------
def _brute(a, w, p1, p2):
    """Exact P_S, P_D by enumerating per-effect disjoint events {A (⊂ W), W\\A, P1, P2, none}."""
    L = len(a)
    PS = PD = 0.0
    cats = ["A", "W", "P1", "P2", "N"]
    for combo in itertools.product(range(5), repeat=L):
        pr = 1.0
        for l, c in enumerate(combo):
            pr *= [a[l], w[l] - a[l], p1[l], p2[l], 1 - w[l] - p1[l] - p2[l]][c]
        names = [cats[c] for c in combo]
        if "A" in names:
            PS += pr
        if "A" not in names and "W" not in names and "P1" in names and "P2" in names:
            PD += pr
    return PS, PD


def test_inclusion_exclusion_matches_enumeration():
    rng = np.random.default_rng(20261029)
    for _ in range(50):
        L = 3
        x = rng.dirichlet(np.ones(5), size=L)          # A, W\A, P1, P2, none
        a, w, p1, p2 = x[:, 0], x[:, 0] + x[:, 1], x[:, 2], x[:, 3]
        sc = adapters.inclusion_exclusion(a, w, p1, p2, p1 + p2)
        PS, PD = _brute(a, w, p1, p2)
        assert abs(sc["P_S"] - PS) < 1e-12 and abs(sc["P_D"] - PD) < 1e-12


def test_single_signal_either_trait_gives_PD_zero():
    # one effect, powered in trait 1 or trait 2 (never both): P_D must be exactly 0
    sc = adapters.inclusion_exclusion([0.0], [0.0], [0.5], [0.5], [1.0])
    assert sc["P_D"] == 0.0
    # two strong independent single-trait effects: P_D = 1
    sc = adapters.inclusion_exclusion([0, 0], [0, 0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0])
    assert abs(sc["P_D"] - 1.0) < 1e-12


# ---------------- end-to-end toy loci --------------------------------------------------------------------
@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(7)
    p, n = 300, 2000
    X = ar_genotypes(rng, n, p)
    cases = {
        "shared": toy_locus(rng, n, p, (100,), (12.0,), (100,), (10.0,), X=X),
        "identical_z": None,
        "distinct": toy_locus(rng, n, p, (40,), (12.0,), (250,), (12.0,), X=X),
        "null": toy_locus(rng, n, p, X=X),
        "one_trait": toy_locus(rng, n, p, (150,), (12.0,), X=X),
        "weak_both": toy_locus(rng, n, p, (60,), (2.5,), (200,), (2.0,), X=X),
    }
    _, R, z1, _ = cases["shared"]
    cases["identical_z"] = (X, R, z1, z1.copy())
    names = list(cases)
    res = joint_susie_batch([cases[k][2] for k in names], [cases[k][3] for k in names],
                            [cases[k][1].astype(np.float32) for k in names], [1e5] * len(names), [1e5] * len(names),
                            [0.0] * len(names), "dense", JointConfig())
    return dict(zip(names, res)), X


@needs_cuda
def test_known_answers(toy):
    r, _ = toy
    assert r["shared"]["P_S"] > 0.95
    assert r["identical_z"]["P_S"] > 0.95                      # identical z -> high P_S
    assert r["distinct"]["P_D"] > 0.95                          # two r2~0 strong signals -> high P_D
    assert r["distinct"]["E1"] > 0.95 and r["distinct"]["E2"] > 0.95
    assert r["null"]["P_W"] > 0.95                              # noise -> P_W ~ 1
    assert r["one_trait"]["P_D"] < 0.05 and r["one_trait"]["P_W"] > 0.9   # single-trait signal is not D
    assert r["weak_both"]["P_S"] < 0.05 and r["weak_both"]["P_D"] < 0.05  # too weak to call
    assert r["shared"]["E2"] < 0.05                             # trait 2 not excluded at trait-1 CS


def test_state_bayes_factors_with_rho_exact():
    """SER state log-BFs (with residual correlation rho and correlated U_both) equal exact bivariate-normal
    likelihood ratios N(bhat; 0, V + U_s) / N(bhat; 0, V)."""
    from scipy.stats import multivariate_normal as mvn
    from famA.joint_susie import _state_lbf, _U_from_scales
    rng = np.random.default_rng(3)
    for rho in (0.0, 0.5, -0.8):
        r = torch.tensor(rng.normal(0, 4, size=(1, 2, 7)))
        s2 = torch.tensor([[30.0, 12.0]], dtype=torch.float64)
        U = _U_from_scales(s2, 0.4)
        lbf, _ = _state_lbf(r, torch.tensor([rho], dtype=torch.float64), U)
        V = np.array([[1, rho], [rho, 1]])
        Us = [np.diag([30.0, 0]), np.diag([0, 12.0]), np.array([[30.0, 0.4 * np.sqrt(360)], [0.4 * np.sqrt(360), 12.0]])]
        for k, Uk in enumerate(Us):
            for j in range(7):
                x = r[0, :, j].numpy()
                ref = mvn(np.zeros(2), V + Uk).logpdf(x) - mvn(np.zeros(2), V).logpdf(x)
                assert abs(lbf[0, k, j].item() - ref) < 1e-9


@needs_cuda
def test_overlap_null_calibration():
    """Null loci with strongly correlated noise (sample overlap, rho = 0.8): the rho-aware fit calls nothing."""
    rng = np.random.default_rng(11)
    p, n = 200, 2000
    X = ar_genotypes(rng, n, p)
    zz = [toy_locus(rng, n, p, rho=0.8, X=X) for _ in range(8)]
    out = joint_susie_batch([z[2] for z in zz], [z[3] for z in zz], [z[1] for z in zz], [1e5] * 8, [1e5] * 8,
                            [0.8] * 8, "dense", JointConfig())
    assert max(o["P_S"] for o in out) < 0.05 and min(o["P_W"] for o in out) > 0.9


@needs_cuda
def test_reduces_to_susie_rss_with_one_trait():
    """pi = (0, 1, 0, 0), no z adjustment: JointSuSiE = susie_rss(z1, R) without n (z scale)."""
    rng = np.random.default_rng(5)
    p, n = 250, 1500
    X = ar_genotypes(rng, n, p)
    zs, Rs = [], []
    for c in [(30,), (30, 180), ()]:
        _, R, z1, _ = toy_locus(rng, n, p, c, [9.0] * len(c), X=X)
        zs.append(z1); Rs.append(R)
    a = joint_susie_batch(zs, [np.zeros(p)] * 3, Rs, None, None, [0.0] * 3, "dense",
                          JointConfig(pi=(0.0, 1.0, 0.0, 0.0), z_adjust=False, ld_dtype=torch.float64,
                                     eta_floor=0.0, multistart=False, convergence="susie014"))   # identity: M3.5 settings
    g = susie_rss_batch(zs, Rs, [None] * 3, "dense", SusieRSSConfig(ld_dtype=torch.float64))
    for x, y in zip(a, g):
        assert np.abs(x["pip"] - y["pip"]).max() < 1e-6
        assert np.allclose(x["s2"][:, 0], y["V"], rtol=1e-4, atol=1e-8)
        assert x["niter"] == y["niter"]


@needs_cuda
def test_eta_detects_reference_mismatch():
    """eta ~ 0 with the in-sample genotype factor; eta > 0 when z comes from a different sample."""
    rng = np.random.default_rng(9)
    p, n = 300, 150
    Xin = ar_genotypes(rng, n, p)
    Xother = ar_genotypes(rng, n, p)
    _, _, z1, z2 = toy_locus(rng, n, p, (50,), (10.0,), (50,), (8.0,), X=Xin)
    _, _, w1, w2 = toy_locus(rng, n, p, (50,), (10.0,), (50,), (8.0,), X=Xother)
    cfg = JointConfig(estimate_eta=True)
    out = joint_susie_batch([z1, w1], [z2, w2], [Xin.astype(np.float32)] * 2, [1e5] * 2, [1e5] * 2, [0.0] * 2, "factor",
                            cfg, n_refs=[n, n])
    assert out[0]["eta"] <= cfg.eta_floor * (1 + 1e-9)        # in-sample LD: η at the P3-5 floor (1e-3)
    assert out[1]["eta"] > 10 * max(out[0]["eta"], 1e-6) and out[1]["eta"] > 0.01
    # without the floor (M3.5 behaviour) the in-sample η goes below 1e-3
    out0 = joint_susie_batch([z1], [z2], [Xin.astype(np.float32)], [1e5], [1e5], [0.0], "factor",
                             JointConfig(estimate_eta=True, eta_floor=0.0), n_refs=[n])
    assert out0[0]["eta"] < 1e-3


@needs_cuda
def test_triton_eb_matches_torch():
    rng = np.random.default_rng(13)
    p, n = 200, 1000
    X = ar_genotypes(rng, n, p)
    zz = [toy_locus(rng, n, p, (20,), (9.0,), (20 if k % 2 else 150,), (7.0,), rho=0.3, X=X) for k in range(4)]
    args = ([z[2] for z in zz], [z[3] for z in zz], [z[1] for z in zz], [1e5] * 4, [1e4] * 4, [0.3] * 4, "dense")
    a = joint_susie_batch(*args, JointConfig(use_triton=True, ld_dtype=torch.float64))
    b = joint_susie_batch(*args, JointConfig(use_triton=False, ld_dtype=torch.float64))
    for x, y in zip(a, b):
        assert np.abs(x["pip"] - y["pip"]).max() < 1e-7
        assert abs(x["P_S"] - y["P_S"]) < 1e-7 and abs(x["P_D"] - y["P_D"]) < 1e-7


@needs_cuda
def test_identity_vs_mvsusie(tmp_path, has_r):
    """eta = 0, rho = 0, fixed prior (no null weight): PIP and state posteriors equal mvsusie_rss."""
    from famA.validation.compare_mvsusie import load_r
    rng = np.random.default_rng(17)
    loci = []
    for k, (c1, c2) in enumerate([((30,), (30,)), ((30,), (120,)), ((10, 90), (90,)), ((), (60,))]):
        X, R, z1, z2 = toy_locus(rng, 1000, 150, c1, [8.0] * len(c1), c2, [7.0] * len(c2))
        loci.append(dict(R=R, z1=z1, z2=z2, n=5000.0, n2=5000.0))
    write_store(tmp_path, loci)
    env = dict(os.environ, R_PROFILE_USER="/dev/null",
               R_LIBS=f"{PROJECT}/env/rlib_susie016:{PROJECT}/env/rlib_mvsusie")
    (tmp_path / "mvs").mkdir()
    subprocess.run([f"{PROJECT}/env/renv/bin/Rscript", f"{PROJECT}/src/famA/validation/run_mvsusie.R", str(tmp_path),
                    str(tmp_path / "mvs" / "c0"), "1,2,3,4", "25", "25", "0", "0.375", "0.375", "0.25", "1e-4"],
                   check=True, env=env, capture_output=True, timeout=600)
    rr = load_r(tmp_path / "mvs")
    st = LocusStore(tmp_path)
    cfg = JointConfig(prior="fixed", U_fixed=((25, 0, 0), (0, 0, 25), (25, 0, 25)), pi=(0, 0.375, 0.375, 0.25), pi_mode="fixed", eta_floor=0.0, multistart=False,
                      convergence="susie016", tol=1e-4, ld_dtype=torch.float64)
    res = joint_susie_batch([st.z(i, 1) for i in range(4)], [st.z(i, 2) for i in range(4)], [st.R(i) for i in range(4)],
                            [5000.0] * 4, [5000.0] * 4, [0.0] * 4, "dense", cfg)
    for i in range(4):
        q = rr[st.row(i).locus_id]
        stp = (q["alpha"][:, :, None] * q["pmw"]).sum(1)
        stp = np.concatenate([np.zeros((stp.shape[0], 1)), stp], 1)
        assert np.abs(res[i]["pip"] - q["pip"]).max() < 1e-8
        assert np.abs(res[i]["state_prob"] - stp).max() < 1e-8
        assert res[i]["niter"] == int(q["niter"])


@needs_cuda
def test_method_interface(toy):
    r, X = toy
    rng = np.random.default_rng(3)
    p = X.shape[1]
    _, R, z1, z2 = toy_locus(rng, X.shape[0], p, (10,), (11.0,), (10,), (11.0,), X=X)
    loci = [LocusData(locus_id="a", z1=z1, z2=z2, n1=np.full(p, 1e5), n2=np.full(p, 1e4), ld_type="dense",
                      R=R.astype(np.float32), rho_hat=0.1),
            LocusData(locus_id="b", z1=z1, z2=z2, n1=np.full(p, 1e5), n2=np.full(p, 1e4), ld_type="factor",
                      X_ref=X.astype(np.float32), n_ref=X.shape[0], rho_hat=float("nan"))]
    out = JointSusieMethod().fit_batch(loci)
    for o in out:
        assert set(["locus_id", "method", "s", "P_S", "P_D", "P_W", "E1", "E2", "status", "runtime_s", "extra"]) <= set(o)
        assert o["status"] == "ok" and o["P_S"] > 0.9
        assert abs(o["P_S"] + o["P_D"] + o["P_W"] - 1) < 1e-9
