"""P3-5 / P3-6 Family A fixes: multi-start IBSS, η floor, relative stopping, per-trait LD at L0, ladder configs.
Toy data with known answers (GPU)."""
import math

import numpy as np
import pytest
import torch

from famA import joint_susie as js
from famA.joint_susie import JointConfig, joint_susie_batch
from famA.method import LADDER, JointSusieMethod, ladder_config, ladder_method
from famG.ld import build_batch_ld
from famag_testutils import ar_genotypes, toy_locus

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="CUDA required")


def _d_toys(seed=0, k=6, p=300, n=3000):
    """D-like loci: distinct causal variants of the two traits in LD (AR φ = 0.9, 2–4 variants apart), plus two
    S loci; the regime where the default start can end in a shared-state local optimum."""
    rng = np.random.default_rng(seed)
    z1s, z2s, Rs = [], [], []
    for i in range(k):
        X = ar_genotypes(rng, n, p, phi=0.9)
        c = 100 + 20 * i
        if i < k - 2:
            _, R, z1, z2 = toy_locus(rng, n, p, (c,), (8.0,), (c + 2 + (i % 3),), (7.0,), X=X)
        else:
            _, R, z1, z2 = toy_locus(rng, n, p, (c,), (8.0,), (c,), (7.0,), X=X)
        z1s.append(z1); z2s.append(z2); Rs.append(R)
    return z1s, z2s, Rs, n


@needs_cuda
def test_multistart_never_lower_elbo():
    z1s, z2s, Rs, n = _d_toys()
    k = len(z1s)
    args = (z1s, z2s, Rs, [n] * k, [n] * k, [0.0] * k, "dense")
    one = joint_susie_batch(*args, JointConfig(multistart=False, estimate_eta=True))
    ms = joint_susie_batch(*args, JointConfig(multistart=True, estimate_eta=True))
    for a, b in zip(one, ms):
        ea = a["elbo"][-1]
        assert b["elbo"][-1] >= ea - 1e-6 * abs(ea)                       # never lower than the single start
        assert abs(b["elbo_starts"][0] - ea) <= 1e-6 * abs(ea)            # default start = single-start fit
        assert b["elbo"][-1] == pytest.approx(max(b["elbo_starts"]), rel=0, abs=0)
        assert b["start"] in ("default", "noshare")
        if b["start"] == "noshare":
            assert len(b["elbo"]) > JointConfig().noshare_sweeps


@needs_cuda
def test_noshare_phase_disables_shared_state():
    """During the first noshare_sweeps sweeps the 'both' state has zero posterior mass; afterwards it is free."""
    rng = np.random.default_rng(2)
    p, n = 200, 2000
    _, R, z1, z2 = toy_locus(rng, n, p, (80,), (9.0,), (80,), (9.0,))
    B = 2
    z = torch.tensor(np.stack([np.stack([z1, z2])] * B), dtype=torch.float64, device="cuda")
    valid = torch.ones((B, p), dtype=torch.bool, device="cuda")
    ld = build_batch_ld([R] * B, "dense", p, torch.float64, "cuda")
    st = js._JointBucket(z, valid, torch.tensor([p] * B, device="cuda"), None, ld,
                         torch.zeros(B, dtype=torch.float64, device="cuda"), JointConfig(multistart=False),
                         noshare=[0, 3])
    for it in range(3):
        st.sweep(); st.niter += 1
        assert float(st.smass[1, :, 3].max()) == 0.0
        assert float(st.smass[0, :, 3].max()) > 0.5                      # shared signal: 'both' in default start
    st.sweep(); st.niter += 1
    assert float(st.smass[1, :, 3].max()) > 0.1                          # shared state re-enabled after the phase


def _dup_ref_toy(lam=15.0):
    """Reference (n_ref = 40) with two identical columns 50/51 that are only weakly correlated in the study:
    with η = 0 the pair (+m at 50, −m at 51) has zero R-norm, so the ELBO is unbounded (σ² runs to the Brent bound)."""
    rng = np.random.default_rng(3)
    p, n_ref, n = 100, 40, 3000
    _, _, z1, z2 = toy_locus(rng, n, p, (50,), (lam,), (50,), (lam * 0.8,), X=ar_genotypes(rng, n, p, phi=0.3))
    X = ar_genotypes(rng, n_ref, p, phi=0.7)
    X[:, 51] = X[:, 50]
    X = ((X - X.mean(0)) / X.std(0)).astype(np.float32)
    return z1, z2, X, n, n_ref


@needs_cuda
def test_eta_floor_bounds_rank_deficient_divergence():
    z1, z2, X, n, n_ref = _dup_ref_toy()
    common = ([z1], [z2], [X], [n], [n], [0.0], "factor")
    kw = dict(multistart=False, estimate_eta=False, convergence="susie014", max_iter=200)
    r0 = joint_susie_batch(*common, JointConfig(eta_floor=0.0, **kw), n_refs=[n_ref])[0]
    r1 = joint_susie_batch(*common, JointConfig(eta_floor=1e-3, **kw), n_refs=[n_ref])[0]
    assert r1["eta"] == pytest.approx(1e-3)
    # η = 0: σ² reaches the Brent bound e^15 and the ELBO keeps growing by > 100 per sweep (divergence)
    assert math.log(r0["s2"].max()) > 14.9 and not r0["converged"]
    assert r0["elbo"][-1] - r0["elbo"][-2] > 100
    # η floor 1e-3: proper likelihood, the ELBO increments shrink and σ² stays below the bound
    d1 = np.diff(r1["elbo"])
    assert np.all(np.isfinite(r1["elbo"]))
    assert math.log(r1["s2"].max()) < 14.9
    assert d1[-1] < 0.1 * d1[20]
    assert r1["elbo"][-1] < 0.5 * r0["elbo"][-1]
    # with η estimated (the +eta rung) the same locus converges
    r2 = joint_susie_batch(*common, JointConfig(estimate_eta=True, multistart=False), n_refs=[n_ref])[0]
    assert r2["converged"] and r2["eta"] >= 1e-3 and math.log(r2["s2"].max()) < 14


@needs_cuda
def test_rel_tolerance_stops_no_later_and_same_answer():
    z1s, z2s, Rs, n = _d_toys(seed=4, k=4)
    args = (z1s, z2s, Rs, [n] * 4, [n] * 4, [0.0] * 4, "dense")
    a = joint_susie_batch(*args, JointConfig(convergence="susie014", multistart=False, estimate_eta=True))
    b = joint_susie_batch(*args, JointConfig(convergence="rel", multistart=False, estimate_eta=True))
    for x, y in zip(a, b):
        assert y["niter"] <= x["niter"]
        assert np.abs(x["pip"] - y["pip"]).max() < 1e-2
        assert abs(x["P_S"] - y["P_S"]) < 1e-2 and abs(x["P_D"] - y["P_D"]) < 1e-2


@needs_cuda
@pytest.mark.parametrize("kind", ["dense", "factor"])
def test_per_trait_ld_reduces_to_single_R(kind):
    """lds2 = lds gives the one-LD fit (with η estimated and ρ ≠ 0; torch η path on both sides)."""
    rng = np.random.default_rng(7)
    p, n, n_ref = 150, 2000, 120
    z1s, z2s, lds = [], [], []
    for c in [(30, 90), (60,), ()]:
        _, R, z1, z2 = toy_locus(rng, n, p, c, [8.0] * len(c), c[:1], [7.0] * len(c[:1]), rho=0.4)
        Xr = ar_genotypes(rng, n_ref, p)
        Xr = ((Xr - Xr.mean(0)) / Xr.std(0)).astype(np.float32)
        z1s.append(z1); z2s.append(z2)
        lds.append(Xr if kind == "factor" else (Xr.T @ Xr / n_ref).astype(np.float64))
    args = (z1s, z2s, lds, [n] * 3, [n] * 3, [0.35] * 3, kind)
    cfg = JointConfig(estimate_eta=True, use_triton=False, ld_dtype=torch.float64, multistart=False)
    nr = [n_ref] * 3 if kind == "factor" else None
    a = joint_susie_batch(*args, cfg, n_refs=nr)
    b = joint_susie_batch(*args, cfg, n_refs=nr, lds2=lds, n_refs2=nr)
    for x, y in zip(a, b):
        assert not x["per_trait_ld"] and y["per_trait_ld"]
        assert x["niter"] == y["niter"]
        assert y["eta"] == pytest.approx(x["eta"], rel=1e-6, abs=1e-10)
        assert np.abs(x["pip"] - y["pip"]).max() < 1e-6
        assert abs(x["P_S"] - y["P_S"]) < 1e-6 and abs(x["P_D"] - y["P_D"]) < 1e-6
        assert np.allclose(x["elbo"], y["elbo"], rtol=1e-8, atol=1e-6)


@needs_cuda
def test_per_trait_eta_terms_vs_bruteforce_rho0():
    """ρ = 0, R_1 ≠ R_2 (exact case): η terms = −½log|R̃_1| − ½log|R̃_2| − ½Σ_t z_tᵀR̃_t⁻¹z_t − p log 2π."""
    rng = np.random.default_rng(9)
    p = 120
    X1, X2 = ar_genotypes(rng, 300, p), ar_genotypes(rng, 90, p, phi=0.6)    # R_2 rank-deficient (90 < p)
    R1, R2 = X1.T @ X1 / 300, X2.T @ X2 / 90
    d1, d2 = np.sqrt(np.diag(R1)), np.sqrt(np.diag(R2))
    R1s, R2s = R1 / d1[:, None] / d1[None], R2 / d2[:, None] / d2[None]
    z1, z2 = rng.standard_normal(p) * 2, rng.standard_normal(p) * 2
    z = torch.tensor(np.stack([np.stack([z1, z2])]), dtype=torch.float64, device="cuda")
    valid = torch.ones((1, p), dtype=torch.bool, device="cuda")
    st = js._JointBucket(z, valid, torch.tensor([p], device="cuda"), None,
                         build_batch_ld([R1], "dense", p, torch.float64, "cuda"),
                         torch.zeros(1, dtype=torch.float64, device="cuda"), JointConfig(estimate_eta=True),
                         ld2=build_batch_ld([R2], "dense", p, torch.float64, "cuda"))
    zz1, zz2 = z1 / d1, z2 / d2
    for eta in (1e-3, 0.05, 0.4):
        got = float(st._eta_terms(torch.tensor([eta], dtype=torch.float64, device="cuda")))
        ref = -p * math.log(2 * math.pi)
        for Rs, zz in ((R1s, zz1), (R2s, zz2)):
            Rt = (1 - eta) * Rs + eta * np.eye(p)
            ref += -0.5 * np.linalg.slogdet(Rt)[1] - 0.5 * zz @ np.linalg.solve(Rt, zz)
        assert got == pytest.approx(ref, rel=1e-9, abs=1e-6)


def test_ladder_configs():
    assert LADDER == ("base", "+rho", "+eta", "+E", "+multistart", "+perTraitLD")
    names = [ladder_method(r).name for r in LADDER]
    assert names == ["famA_joint_susie", "famA_joint_susie_rho", "famA_joint_susie_rho_eta",
                     "famA_joint_susie_rho_eta_E", "famA_joint_susie_rho_eta_E_ms",
                     "famA_joint_susie_rho_eta_E_ms_ptld"]
    assert len(set(names)) == len(names)
    for r in LADDER:
        cfg, _ = ladder_config(r)
        assert cfg.eta_floor == 1e-3 and cfg.nonweak_own and cfg.pi_mode == "eb" and cfg.pi_prior_strength == 4.0
        assert cfg.convergence == ("alpha" if cfg.ref_var else "rel")
    cfg, use_rho = ladder_config("base")
    assert not (use_rho or cfg.estimate_eta or cfg.ref_var or cfg.multistart or cfg.per_trait_ld)
    cfg, use_rho = ladder_config("rho+eta+multistart+perTraitLD")         # ladder without layer E
    assert use_rho and cfg.estimate_eta and cfg.multistart and cfg.per_trait_ld and not cfg.ref_var
    assert ladder_method("+eta", pi_mode="fixed").name == "famA_joint_susie_rho_eta_pifix"
    assert ladder_method("+eta", nonweak_own=False).name == "famA_joint_susie_rho_eta_numu"
    assert ladder_config("+eta", pi_mode="fixed")[0].pi == (0.25, 0.25, 0.25, 0.25)
    with pytest.raises(ValueError):
        ladder_config("rho+foo")
    # library defaults = the P3-5/P3-6 method (own-effect guard, EB π, η floor, multistart, per-trait LD)
    c = JointConfig()
    assert c.nonweak_own and c.pi_mode == "eb" and c.eta_floor == 1e-3 and c.multistart and c.per_trait_ld
    assert JointSusieMethod().name.startswith("famA_joint_susie_rho")


def test_ref_var_table_v2_excludes_monomorphic():
    """v2 (default) = v1's windows and draws with subsample-monomorphic draws excluded: common pairs (min MAF >= 0.2,
    never monomorphic in a subsample) are unchanged; the rare high-r² cells inflated by v1's r̂ := 0 drop."""
    import json
    from pathlib import Path
    assert Path(js.REF_VAR_TABLE).name == "ref_ld_var_v2.json"
    v1 = json.loads((Path(js.__file__).parent / "ref_ld_var_v1.json").read_text())
    v2 = json.loads(Path(js.REF_VAR_TABLE).read_text())
    assert v2["exclude_monomorphic"] and v2["windows"] == v1["windows"]
    K1, K2 = np.array(v1["kappa"]), np.array(v2["kappa"])
    assert np.allclose(K1[:, 2], K2[:, 2], rtol=1e-9, atol=1e-12)
    b = v2["r2_edges"].index(0.9)                        # r² bin [0.9, 0.95)
    assert K2[b, 0] < 0.5 * K1[b, 0]
    assert K2[-1, 0] < 1e-6 < K1[-1, 0]                  # r² >= 0.995, rare: v1 0.89 (artefact), v2 ~0
