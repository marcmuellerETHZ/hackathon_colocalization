"""Toy-L0* positive control: when data are generated from exactly the model ExactColoc assumes, its exact
posteriors (hypothesis categories) and Monte-Carlo oracle-class / exclusion posteriors are calibrated:
for every event, sum_i P_i matches the realised count within 4 SD (Var = sum P(1-P) + MC variance).
Negative control: dropping the known sample-overlap correlation (rho -> 0) on rho != 0 data worsens the
Brier score of the hypothesis posteriors."""
import json

import numpy as np
import pytest
import torch

from famB.constants import CAT_NAMES
from famB.enum import LocusTensors
from famB.events import oracle_events
from famB.method import ExactColoc
from famB.priors import toy_l0star_prior
from toy_l0star import simulate_gaussian, simulate_simprior, true_category


def _true_events(truths, device):
    out = {k: [] for k in ("S", "U", "D", "W", "G", "E1", "E2")}
    for tr in truths:
        p = tr["R"].shape[0]
        eta = tr["eta"]
        Rt = (1 - eta) * tr["R"] + eta * np.eye(p)
        f = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
        T = LocusTensors(Rt=f(Rt)[None], b=torch.zeros(1, 2, p, dtype=torch.float64, device=device), Vinv=f(np.eye(2))[None],
                         sd=torch.ones(1, 2, p, 1, dtype=torch.float64, device=device),
                         logwk=torch.zeros(1, 2, 1, dtype=torch.float64, device=device),
                         rho_b=torch.zeros(1, dtype=torch.float64, device=device),
                         logm=torch.zeros(1, p, dtype=torch.float64, device=device),
                         logprior_type=torch.zeros(1, 3, 3, 3, dtype=torch.float64, device=device),
                         eta=torch.tensor([eta], dtype=torch.float64, device=device))
        S1, S2 = sorted(tr["S1"]), sorted(tr["S2"])
        row = (S1 + [-1, -1])[:2] + (S2 + [-1, -1])[:2]
        lam = [tr["lam1"][v] if (v >= 0 and i < 2) else (tr["lam2"][v] if v >= 0 else 0.0) for i, v in enumerate(row)]
        V = torch.tensor([[max(0, v) for v in row]], device=device)
        m = torch.tensor([[v >= 0 for v in row]], device=device)
        ev = oracle_events(T, torch.zeros(1, dtype=torch.int64, device=device), V, m,
                           torch.tensor([lam], dtype=torch.float64, device=device))
        for k in out:
            out[k].append(bool(ev[k][0]))
    return {k: np.array(v) for k, v in out.items()}


def _zscore(P, y, n_mc=None):
    var = np.sum(P * (1 - P))
    if n_mc:
        var += np.sum(P * (1 - P)) / n_mc
    return (y.sum() - P.sum()) / np.sqrt(max(var, 1e-12))


def _check(out, loci, truths, device, n_mc, label, z_max=4.0):
    ext = [json.loads(o["extra"]) for o in out]
    assert all(o["status"] == "ok" for o in out)
    report = {}
    cats = np.array([true_category(t["S1"], t["S2"]) for t in truths])
    for h in CAT_NAMES:
        P = np.array([e["PP"][h] for e in ext])
        z = _zscore(P, cats == h)
        report[h] = (round(float(P.sum()), 1), int((cats == h).sum()), round(float(z), 2))
        assert abs(z) < z_max, (label, h, report[h])
    te = _true_events(truths, device)
    pe = {"S": [o["P_S"] for o in out], "D": [o["P_D"] for o in out], "W": [e["P_W_strict"] for e in ext],
          "U": [e["P_U"] for e in ext], "G": [e["P_G"] for e in ext], "E1": [o["E1"] for o in out],
          "E2": [o["E2"] for o in out]}
    for k, P in pe.items():
        P = np.array(P)
        z = _zscore(P, te[k], n_mc)
        report[k] = (round(float(P.sum()), 1), int(te[k].sum()), round(float(z), 2))
        assert abs(z) < z_max, (label, k, report[k])
        # reliability: within bins of the predicted probability (bins with >= 20 loci)
        for lo_, hi_ in ((0.0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.01)):
            sel = (P >= lo_) & (P < hi_)
            if sel.sum() >= 20:
                zb = _zscore(P[sel], te[k][sel], n_mc)
                report[f"{k}[{lo_},{hi_})"] = (int(sel.sum()), round(float(P[sel].mean()), 3),
                                               round(float(te[k][sel].mean()), 3), round(float(zb), 2))
                assert abs(zb) < z_max, (label, k, lo_, hi_, report[f"{k}[{lo_},{hi_})"])
    print(label, report)
    return report


@pytest.mark.slow
def test_simulator_mode_calibrated(device):
    rng = np.random.default_rng(20261029)
    prior = toy_l0star_prior()
    loci, truths = simulate_simprior(rng, prior, 800, p=8, rho_choices=(0.0, 0.4))
    m = ExactColoc(mode="simulator", M=30, n_mc=4096, eta=1e-3, sim_prior_fn=lambda _l: prior)
    out = m.fit_batch(loci)
    _check(out, loci, truths, device, 4096, "simulator")


@pytest.mark.slow
def test_gaussian_mode_calibrated_and_rho_matters(device):
    rng = np.random.default_rng(20261030)
    kw = dict(p1=0.04, p2=0.04, p12=0.03, sig=(9.0, 7.0), rho_b=0.5)
    loci, truths = simulate_gaussian(rng, 1000, p=8, rho_choices=(0.0, 0.4, -0.7), **kw)
    common = dict(M=30, n_mc=4096, eb=False, sigma=kw["sig"], eta=1e-3, p1=kw["p1"], p2=kw["p2"], p12=kw["p12"],
                  rho_b=kw["rho_b"])
    out = ExactColoc(mode="gaussian", **common).fit_batch(loci)
    _check(out, loci, truths, device, 4096, "gaussian")
    # negative control: ignore the overlap correlation
    out0 = ExactColoc(mode="gaussian", use_rho_hat=False, **common).fit_batch(loci)
    cats = np.array([true_category(t["S1"], t["S2"]) for t in truths])
    nz = np.array([t["rho"] != 0 for t in truths])

    def brier(o):
        P = np.array([[json.loads(r["extra"])["PP"][h] for h in CAT_NAMES] for r in o])
        Y = np.array([[c == h for h in CAT_NAMES] for c in cats], dtype=float)
        return float(np.mean(np.sum((P - Y) ** 2, axis=1)[nz]))

    b1, b0 = brier(out), brier(out0)
    print(f"Brier (rho != 0 loci): correct rho {b1:.4f}  vs rho ignored {b0:.4f}")
    assert b1 < b0
