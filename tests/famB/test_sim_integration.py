"""Integration with the simulation agent's modules (read-only use of src/sim):
  * famB.events.oracle_events == sim.oracle.oracle_class on random draws (partition + labels);
  * famB.priors.sim_prior_from_l0star reproduces sim.l0star.config_logprior exactly;
  * simulator-prior mode is calibrated on loci drawn by sim.l0star.sample_locus (the real L0* generator,
    toy p = 10 segments with AR(1)-type and random genotype LD), against sim's oracle labels.
"""
import itertools
import json
import math

import numpy as np
import pytest
import torch

sim_oracle = pytest.importorskip("sim.oracle")
l0star = pytest.importorskip("sim.l0star")

from eval.io import LocusData
from famB.enum import LocusTensors
from famB.events import oracle_events
from famB.method import ExactColoc
from famB.priors import sim_prior_from_l0star, type_table_exchangeable


def _T_from_Rt(Rt, device):
    p = Rt.shape[0]
    f = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    return LocusTensors(Rt=f(Rt)[None], b=torch.zeros(1, 2, p, dtype=torch.float64, device=device),
                        Vinv=f(np.eye(2))[None], sd=torch.ones(1, 2, p, 1, dtype=torch.float64, device=device),
                        logwk=torch.zeros(1, 2, 1, dtype=torch.float64, device=device),
                        rho_b=torch.zeros(1, dtype=torch.float64, device=device),
                        logm=torch.zeros(1, p, dtype=torch.float64, device=device),
                        logprior_type=torch.zeros(1, 3, 3, 3, dtype=torch.float64, device=device),
                        eta=torch.tensor([1e-3], dtype=torch.float64, device=device))


def test_oracle_matches_sim(device):
    rng = np.random.default_rng(20261026)
    n_draw = 50_000
    p = 8
    X = rng.standard_normal((300, p))
    X[:, 1] = X[:, 0] + 0.15 * rng.standard_normal(300)
    X[:, 3] = X[:, 2] + 0.5 * rng.standard_normal(300)
    X = (X - X.mean(0)) / X.std(0)
    Rt = 0.999 * (X.T @ X / 300) + 1e-3 * np.eye(p)
    T = _T_from_Rt(Rt, device)
    V, m, lam, sims = [], [], [], []
    for _ in range(n_draw):
        c1 = sorted(rng.choice(p, rng.integers(0, 3), replace=False).tolist())
        c2 = sorted(rng.choice(p, rng.integers(0, 3), replace=False).tolist())
        l1 = np.zeros(p); l2 = np.zeros(p)
        l1[c1] = rng.normal(0, 9, len(c1)); l2[c2] = rng.normal(0, 7, len(c2))
        mu1, mu2 = Rt @ l1, Rt @ l2
        o = sim_oracle.oracle_class(c1, l1[c1], mu1[c1], c2, l2[c2], mu2[c2], lambda a, b: Rt[a, b] ** 2, "gwas", "mol")
        sims.append(o.cls)
        row = (c1 + [-1, -1])[:2] + (c2 + [-1, -1])[:2]
        V.append([max(0, v) for v in row]); m.append([v >= 0 for v in row])
        lam.append([l1[v] if v >= 0 else 0.0 for v in row[:2]] + [l2[v] if v >= 0 else 0.0 for v in row[2:]])
    ev = oracle_events(T, torch.zeros(n_draw, dtype=torch.int64, device=device), torch.tensor(V, device=device),
                       torch.tensor(m, device=device), torch.tensor(lam, dtype=torch.float64, device=device))
    mine = np.array(["SUDWG"[int(i)] for i in torch.stack([ev[k] for k in "SUDWG"], 1).double().argmax(1).cpu()])
    sims = np.array(sims)
    shares = {c: float((sims == c).mean()) for c in "SUDWG"}
    print("class shares", shares)
    assert all(v > 0.003 for v in shares.values())
    assert np.array_equal(mine, sims), np.unique(list(zip(mine[mine != sims], sims[mine != sims])), axis=0)[:10]


@pytest.mark.parametrize("p", [7, 30])
def test_config_prior_matches_l0star(p):
    pr = sim_prior_from_l0star()
    pr.validate()
    tab = type_table_exchangeable(pr.type_prob, p)
    sets = [()] + [(j,) for j in range(p)] + list(itertools.combinations(range(p), 2))
    worst = 0.0
    for S1 in sets:
        for S2 in sets:
            ref = l0star.config_logprior(list(S1), list(S2), p)
            nsh = len(set(S1) & set(S2))
            got = tab[nsh, len(S1) - nsh, len(S2) - nsh]
            if math.isinf(ref):
                assert math.isinf(got) and got < 0, (S1, S2, got)
            else:
                worst = max(worst, abs(got - ref))
    assert worst < 1e-12, worst


def _segment(rng, p, kind):
    if kind == "ar":
        r = 0.9
        R = r ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    else:
        X = rng.standard_normal((400, p))
        for j in range(1, p):
            X[:, j] = 0.85 * X[:, j - 1] + math.sqrt(1 - 0.85 ** 2) * X[:, j]
        X = (X - X.mean(0)) / X.std(0)
        R = X.T @ X / 400
    Rt = ((1 - l0star.ETA) * R + l0star.ETA * np.eye(p)).astype(np.float32).astype(np.float64)
    return l0star.Segment("toy", np.arange(1, p + 1), np.array([f"chr1:{j + 1}:A:C" for j in range(p)]),
                          np.full(p, 0.3), R, Rt, np.linalg.cholesky(Rt), ("chr1", 0, p), "toy")


def _locus(seg, d, i):
    p = len(seg.pos)
    one = np.ones(p)
    return LocusData(locus_id=f"S{i:05d}", variant_id=seg.variant_id, pos=seg.pos, z1=d["z1"], z2=d["z2"],
                     beta1=d["z1"] * 0.01, se1=one * 0.01, beta2=d["z2"] * 0.03, se2=one * 0.03, n1=one * 1e5,
                     n2=one * 1e3, maf1=seg.maf, maf2=seg.maf, maf_ref=seg.maf, ld_type="dense", n_ref=10_000,
                     R=seg.Rt.astype(np.float32), rho_hat=d["rho"])


@pytest.mark.slow
def test_simulator_mode_calibrated_on_sim_l0star(device):
    g = np.random.default_rng(20261029)
    p = 10
    loci, labels = [], []
    for s in range(30):
        seg = _segment(g, p, "ar" if s % 2 else "geno")
        for k in range(25):
            d = l0star.sample_locus(seg, g, rho=float(g.choice([0.0, 0.4])))
            loci.append(_locus(seg, d, len(loci)))
            labels.append(d["oracle"].cls)
    labels = np.array(labels)
    pr = sim_prior_from_l0star()
    m = ExactColoc(mode="simulator", M=30, n_mc=4096, eta=0.0, eta_floor=0.0, sim_prior_fn=lambda _l: pr)
    out = m.fit_batch(loci)
    assert all(o["status"] == "ok" for o in out)
    ext = [json.loads(o["extra"]) for o in out]
    P = {"S": np.array([o["P_S"] for o in out]), "D": np.array([o["P_D"] for o in out]),
         "W": np.array([e["P_W_strict"] for e in ext]), "U": np.array([e["P_U"] for e in ext]),
         "G": np.array([e["P_G"] for e in ext])}
    rep = {}
    for c, Pc in P.items():
        y = labels == c
        var = np.sum(Pc * (1 - Pc)) * (1 + 1 / 4096)
        z = (y.sum() - Pc.sum()) / math.sqrt(max(var, 1e-12))
        rep[c] = (round(float(Pc.sum()), 1), int(y.sum()), round(float(z), 2))
        assert abs(z) < 4, (c, rep[c])
    # log score of the 5-class posterior vs the prior class shares (skill > 0)
    Pm = np.stack([P[c] for c in "SUDWG"], 1).clip(1e-4, 1)
    prior = np.array([(labels == c).mean() for c in "SUDWG"]).clip(1e-4, 1)
    idx = np.array(["SUDWG".index(c) for c in labels])
    ll_post = np.mean(np.log(Pm[np.arange(len(idx)), idx]))
    ll_prior = np.mean(np.log(prior[idx]))
    print("L0* (sim generator) calibration", rep, f"mean log score posterior {ll_post:.3f} vs prior {ll_prior:.3f}")
    assert ll_post > ll_prior
