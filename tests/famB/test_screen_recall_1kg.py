"""Pre-registered screen recall test (DESIGN Sec. 5 B / 7.1(4)): >= 0.99 of causal variants (or an r^2 > 0.95
proxy) retained, stratified by lambda_1 / lambda_2, on loci simulated from real 1000G EUR (NFE pool A,
n = 254) LD windows of train-split blocks (L0: in-sample LD, z_t = R lam_t + X'u_t / sqrt(n)).

Assertion: recall >= 0.99 in every *powered* stratum (|lambda| and |mu| >= tau_t) at M = 100 and M = 50.
Weak/grey strata and an L2-like arm (screen with pool-B reference LD, n = 150) are reported, not asserted.
"""
import math

import numpy as np
import pandas as pd
import pytest

from famB.constants import TAU1, TAU2, WEAK1, WEAK2
from famB.screen import LDAccess, screen
from kg import load_window, train_windows

from common.seeds import VALID_SEED, rng as seeded_rng

BINS = [3, 4, 6.29, 10, 20, 50, 151]


def _pick_partner(rng, R, j, lo, hi):
    r2 = R[j] ** 2
    cand = np.where((r2 >= lo) & (r2 < hi))[0]
    cand = cand[cand != j]
    if len(cand) == 0:
        return None
    return int(rng.choice(cand))


def _simulate(rng, X, n_loci):
    n, p = X.shape
    R = X.T @ X / n
    out = []
    for _ in range(n_loci):
        kind = rng.choice(["S", "D", "M"], p=[0.35, 0.45, 0.2])
        lam1 = np.zeros(p); lam2 = np.zeros(p)
        draw1 = lambda: math.exp(rng.uniform(math.log(3), math.log(150))) * rng.choice([-1, 1])
        draw2 = lambda: math.exp(rng.uniform(math.log(3), math.log(60))) * rng.choice([-1, 1])
        c = int(rng.integers(p))
        if kind == "S":
            lam1[c] = draw1(); lam2[c] = draw2()
            if rng.random() < 0.5:
                k = _pick_partner(rng, R, c, 0.0, 0.95)
                if k is not None:
                    (lam1 if rng.random() < 0.5 else lam2)[k] = draw1() if rng.random() < 0.5 else draw2()
        elif kind == "D":
            lo, hi = [(0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 0.99)][rng.integers(5)]
            k = _pick_partner(rng, R, c, lo, hi)
            if k is None:
                k = int(rng.integers(p))
            lam1[c] = draw1(); lam2[k] = draw2()
        else:
            a, b, cc, d = rng.choice(p, 4, replace=False)
            lam1[a] = draw1(); lam1[b] = draw1(); lam2[cc] = draw2(); lam2[d] = draw2()
        u1 = rng.standard_normal(n); u2 = rng.standard_normal(n)
        z1 = R @ lam1 + X.T @ u1 / math.sqrt(n)
        z2 = R @ lam2 + X.T @ u2 / math.sqrt(n)
        out.append((z1, z2, lam1, lam2, R @ lam1, R @ lam2))
    return R, out


def _retained(sc, Rsel_rows, j):
    if sc.group[j] >= 0:
        return True
    return bool(np.max(Rsel_rows[:, j] ** 2) > 0.95)


@pytest.mark.slow
def test_screen_recall_real_ld():
    g = seeded_rng(VALID_SEED, "famB.screen_recall")
    wins = train_windows([f"chr{c}" for c in range(15, 23)], 40, g)
    recs = []
    for wi, (chrom, s, e, bid) in enumerate(wins):
        d = load_window(chrom, s, e, target_pool="A", ref_pool="B")
        X, Xr = d["X"], d["Xref"]
        R, loci = _simulate(g, X, 25)
        for arm, ld in (("L0", LDAccess(X=X, n_ref=X.shape[0])), ("L2like", LDAccess(X=Xr, n_ref=Xr.shape[0]))):
            for M in (100, 50):
                for z1, z2, l1, l2, mu1, mu2 in loci:
                    sc = screen(z1, z2, ld, M=M)
                    Rs = R[sc.sel]  # retention judged in the true (in-sample) LD
                    for t, lam, mu, tau in ((1, l1, mu1, TAU1), (2, l2, mu2, TAU2)):
                        for j in np.nonzero(lam)[0]:
                            recs.append(dict(arm=arm, M=M, trait=t, lam=abs(lam[j]), mu=abs(mu[j]),
                                             powered=bool(abs(lam[j]) >= tau and abs(mu[j]) >= tau),
                                             kept=_retained(sc, Rs, j), window=bid, p=X.shape[1]))
    df = pd.DataFrame(recs)
    df["lam_bin"] = pd.cut(df["lam"], BINS, right=False).astype(str)
    tab = df.groupby(["arm", "M", "trait", "lam_bin"])["kept"].agg(["mean", "size"]).reset_index()
    pw = df[df.powered].groupby(["arm", "M", "trait"])["kept"].agg(["mean", "size"]).reset_index()
    pd.set_option("display.width", 200)
    print("\nwindows:", len(wins), "median p:", int(df.drop_duplicates("window")["p"].median()))
    print(tab.to_string(index=False))
    print("powered causal variants:\n", pw.to_string(index=False))
    miss = df[(df.arm == "L0") & df.powered & ~df.kept]
    if len(miss):
        print("missed powered (L0):\n", miss.head(20).to_string(index=False))
    for M in (100, 50):
        for t in (1, 2):
            sub = df[(df.arm == "L0") & (df.M == M) & (df.trait == t) & df.powered]
            # every powered lambda stratum separately
            for lb, grp in sub.groupby("lam_bin"):
                assert grp["kept"].mean() >= 0.99, (M, t, lb, grp["kept"].mean(), len(grp))
