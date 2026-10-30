"""Oracle-event logic (DESIGN Sec. 1) on constructed effect draws with known classes."""
import numpy as np
import torch

from famB.enum import LocusTensors
from famB.events import oracle_events


def _T(R, device, eta=0.0):
    p = R.shape[0]
    f = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    Rt = (1 - eta) * R + eta * np.eye(p)
    return LocusTensors(Rt=f(Rt)[None], b=torch.zeros(1, 2, p, dtype=torch.float64, device=device),
                        Vinv=f(np.eye(2))[None], sd=torch.ones(1, 2, p, 1, dtype=torch.float64, device=device),
                        logwk=torch.zeros(1, 2, 1, dtype=torch.float64, device=device),
                        rho_b=torch.zeros(1, dtype=torch.float64, device=device),
                        logm=torch.zeros(1, p, dtype=torch.float64, device=device),
                        logprior_type=torch.zeros(1, 3, 3, 3, dtype=torch.float64, device=device),
                        eta=torch.tensor([eta], dtype=torch.float64, device=device))


def _ev(T, draws, device):
    """draws: list of (S1 dict var->lam, S2 dict var->lam)."""
    V, m, lam = [], [], []
    for d1, d2 in draws:
        k1, k2 = sorted(d1), sorted(d2)
        row = (k1 + [-1, -1])[:2] + (k2 + [-1, -1])[:2]
        lv = ([d1[k] for k in k1] + [0, 0])[:2] + ([d2[k] for k in k2] + [0, 0])[:2]
        V.append([max(0, v) for v in row]); m.append([v >= 0 for v in row]); lam.append(lv)
    V = torch.tensor(V, device=device); m = torch.tensor(m, device=device)
    lam = torch.tensor(lam, dtype=torch.float64, device=device)
    bidx = torch.zeros(len(draws), dtype=torch.int64, device=device)
    return oracle_events(T, bidx, V, m, lam)


def _cls(ev, i):
    hits = [k for k in ("S", "U", "D", "W", "G") if bool(ev[k][i])]
    assert len(hits) == 1, hits
    return hits[0]


def test_known_classes(device):
    p = 6
    R = np.eye(p)
    R[0, 1] = R[1, 0] = np.sqrt(0.99)  # r^2 = 0.99 pair
    R[2, 3] = R[3, 2] = 0.9
    T = _T(R, device)
    draws = [
        ({4: 10.0}, {4: 8.0}),                # 0 S
        ({4: 10.0}, {5: 8.0}),                # 1 D, r = 0
        ({0: 10.0}, {1: 8.0}),                # 2 U: d = 8 * 0.1 = 0.8
        ({0: 150.0}, {1: 150.0}),             # 3 D: d = 15
        ({4: 10.0}, {}),                      # 4 W
        ({4: 10.0}, {4: 4.5, 5: 8.0}),        # 5 G: shared grey (nonweak both, not powered in 2)
        ({4: 10.0}, {4: 2.0, 5: 8.0}),        # 6 D: weak shared does not block
        ({2: 7.0, 3: -7.0}, {5: 8.0}),        # 7 cancellation: mu_2 = 7 - 6.3 = 0.7 -> trait 1 not powered; |mu| < 4.61 -> W
        ({4: 6.0}, {5: 8.0}),                 # 8 trait 1 grey (6.0 < 6.29, >= 4.61) -> G
        ({4: 3.0}, {4: 3.0}),                 # 9 W (both weak)
        ({}, {}),                             # 10 W (null)
    ]
    ev = _ev(T, draws, device)
    want = ["S", "D", "U", "D", "W", "G", "D", "W", "G", "W", "W"]
    got = [_cls(ev, i) for i in range(len(draws))]
    assert got == want, list(zip(got, want))
    # exclusion: E2 false iff a trait-1 (nonweak) causal has |lambda_2| >= w2
    assert not bool(ev["E2"][0]) and bool(ev["E2"][1]) and not bool(ev["E2"][5]) and bool(ev["E2"][6])
    assert not bool(ev["E1"][0]) and bool(ev["E1"][1])


def test_partition_random(device):
    rng = np.random.default_rng(20261030)
    p = 8
    X = rng.standard_normal((200, p)); X[:, 1] = X[:, 0] + 0.2 * rng.standard_normal(200)
    X = (X - X.mean(0)) / X.std(0)
    T = _T(X.T @ X / 200, device, eta=1e-3)
    draws = []
    for _ in range(20000):
        s1 = rng.choice(p, rng.integers(0, 3), replace=False)
        s2 = rng.choice(p, rng.integers(0, 3), replace=False)
        draws.append(({int(j): float(rng.normal(0, 8)) for j in s1}, {int(j): float(rng.normal(0, 8)) for j in s2}))
    ev = _ev(T, draws, device)
    tot = sum(ev[k].to(torch.int64) for k in ("S", "U", "D", "W", "G"))
    assert torch.all(tot == 1)
    shares = {k: float(ev[k].double().mean()) for k in ("S", "U", "D", "W", "G")}
    print(shares)
    assert all(v > 0 for v in shares.values())
