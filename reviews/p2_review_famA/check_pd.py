"""Independent reviewer check of Family A locus outputs (task p2.review_famA, commit 47ceb97).

On screen_v2 L0 VALIDATION loci (6 per locus-id stratum per L0 cell, seeded selection):
  1. fit A (famA.joint_susie, rung rho+eta, EB pi, default events) and capture the per-effect fitted
     quantities (residual r_l, scales, pi, rho, z, thresholds) by wrapping _JointBucket.outputs (read only);
  2. P_D from A's per-effect (a, w, p1t, p2t) by (i) my own inclusion-exclusion and (ii) an exact dynamic
     programme over (blocked, has1, has2) -- no inclusion-exclusion at all; compare to A's P_D / P_S;
  3. re-derive each effect's SER posterior from r_l with my own bivariate-normal algebra (alpha, state
     weights, Gaussian posterior of beta), then Monte Carlo the power events (own AND marginal mu = beta + m,
     trait-type tau / w, own-effect guard for 'power >= 0.2') per effect, and the joint D / S events across
     independent effects; compare with A's a, w, p1t, p2t, P_D, P_S, E1, E2;
  4. the same IE / DP check (2) for rungs +E (layer E) and fixed pi.
Writes $ORCH_OUT (or --out) / pd_check.tsv, pd_effects.tsv, summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common.seeds import VALID_SEED, rng as seeded_rng
from eval.io import LocusSetReader
from famA import joint_susie as JS
from famA.method import _ld_input, _scalar_n, locus_types

CELLS = ["L0_P1_r_p=0.0", "L0_P3_rho=0.5"]
SETDIR = Path("data/processed/sets/screen_v2")


def pick_ids(path, per_stratum, key):
    with LocusSetReader(path) as r:
        ids = r.ids()
    g = seeded_rng(VALID_SEED, "p2_review_famA", key)
    out = []
    for s in sorted({i.split("__")[2] for i in ids}):
        cand = sorted(i for i in ids if i.split("__")[2] == s)
        sel = g.permutation(len(cand))[:per_stratum]
        out += [cand[k] for k in sorted(sel)]
    return out


# ---------------- (2) combination of per-effect probabilities -----------------------------------------
def ie_pd(a, w, p1, p2):
    one = np.ones_like(w)
    PD = np.prod(one - w) - np.prod(one - w - p1) - np.prod(one - w - p2) + np.prod(one - w - p1 - p2)
    PS = 1 - np.prod(1 - a)
    return PS, PD


def dp_pd(w, p1, p2):
    """Exact P(no effect blocking, >=1 effect powered-in-1-not-blocking, >=1 powered-in-2-not-blocking) for
    independent effects with disjoint per-effect categories {block, p1, p2, other} (DP over 8 states)."""
    st = np.zeros((2, 2, 2)); st[0, 0, 0] = 1.0
    for wl, al, bl in zip(w, p1, p2):
        other = 1.0 - wl - al - bl
        new = np.zeros_like(st)
        for b in (0, 1):
            for h1 in (0, 1):
                for h2 in (0, 1):
                    m = st[b, h1, h2]
                    if m == 0:
                        continue
                    new[1, h1, h2] += m * wl
                    new[b, 1, h2] += m * al
                    new[b, h1, 1] += m * bl
                    new[b, h1, h2] += m * other
        st = new
    return st[0, 1, 1]


# ---------------- (3) independent SER posterior and Monte Carlo ----------------------------------------
def ser_posterior(r, rho, s2, logpi, p):
    """r (2,p). Returns alpha (p,), state weights (4,p), means (3,2,p), covs (3,2,2) per state (same for all j)."""
    V = np.array([[1.0, rho], [rho, 1.0]])
    Us = [np.diag([s2[0], 0.0]), np.diag([0.0, s2[1]]), np.diag([s2[0], s2[1]])]
    Vi = np.linalg.inv(V)
    qV = np.einsum("ip,ij,jp->p", r, Vi, r)
    ldV = np.log(np.linalg.det(V))
    lbf = [np.zeros(p)]
    means, covs = [], []
    for U in Us:
        S = U + V
        Si = np.linalg.inv(S)
        lbf.append(-0.5 * (np.log(np.linalg.det(S)) - ldV) - 0.5 * (np.einsum("ip,ij,jp->p", r, Si, r) - qV))
        K = U @ Si
        means.append(K @ r)
        covs.append(U - K @ U)
    L = np.stack(lbf) + logpi[:, None]                     # (4,p)
    mx = L.max(0)
    lj = mx + np.log(np.exp(L - mx).sum(0))
    wst = np.exp(L - lj)
    lpo = lj + np.log(1.0 / p)
    alpha = np.exp(lpo - lpo.max()); alpha /= alpha.sum()
    return alpha, wst, np.stack(means), np.stack(covs)


def mc_effect(g, alpha, wst, means, covs, m, thr, M, s2):
    """Draw M samples of (j, state, beta) from q(b_l); returns boolean arrays a, w, p1, p2 and (j, state, beta)."""
    tau1, tau2, w1, w2 = thr
    if max(s2) <= 0:      # beta = 0 a.s.: no power event possible
        z = np.zeros(M, bool)
        return z, z, z, z, None
    j = g.choice(len(alpha), size=M, p=alpha)
    u = g.random(M)
    cw = np.cumsum(wst[:, j], 0)                          # (4,M)
    s = (u[None] > cw).sum(0)                              # 0..3
    s = np.minimum(s, 3)
    beta = np.zeros((M, 2))
    for k in (1, 2, 3):
        idx = np.nonzero(s == k)[0]
        if idx.size == 0:
            continue
        C = covs[k - 1]
        mu = means[k - 1][:, j[idx]].T                     # (n,2)
        # covariance may be singular (single-trait states); sample via eigen-decomposition
        ev, evec = np.linalg.eigh(C)
        ev = np.clip(ev, 0, None)
        e = g.standard_normal((idx.size, 2))
        beta[idx] = mu + (e * np.sqrt(ev)) @ evec.T
        if k == 1:
            beta[idx, 1] = 0.0
        if k == 2:
            beta[idx, 0] = 0.0
    mu1 = beta[:, 0] + m[0, j]
    mu2 = beta[:, 1] + m[1, j]
    pw1 = (np.abs(beta[:, 0]) >= tau1) & (np.abs(mu1) >= tau1)
    pw2 = (np.abs(beta[:, 1]) >= tau2) & (np.abs(mu2) >= tau2)
    nw1 = (np.abs(beta[:, 0]) >= w1) & (np.abs(mu1) >= w1)
    nw2 = (np.abs(beta[:, 1]) >= w2) & (np.abs(mu2) >= w2)
    both = s == 3
    blk = both & nw1 & nw2
    a = both & pw1 & pw2
    p1 = pw1 & ~blk
    p2 = pw2 & ~blk
    return a, blk, p1, p2, (j, s, beta)


def run(cfg, loci, capture):
    z1s, z2s, lds, kinds, n1s, n2s, rhos, nrefs, types, refn, mafs = ([] for _ in range(11))
    for lc in loci:
        kind, ld = _ld_input(lc)
        z1s.append(np.asarray(lc.z1, float)); z2s.append(np.asarray(lc.z2, float)); lds.append(ld); kinds.append(kind)
        n1s.append(_scalar_n(lc.n1)); n2s.append(_scalar_n(lc.n2))
        rh = float(lc.rho_hat); rhos.append(rh if math.isfinite(rh) else 0.0)
        nrefs.append(lc.n_ref if kind == "factor" else None)
        types.append(locus_types(lc)[0]); refn.append(float(lc.n_ref)); mafs.append(lc.maf_ref)
    orig = JS._JointBucket.outputs
    if capture:
        def patched(self, rows):
            outs = orig(self, rows)
            for k, row in enumerate(rows):
                p = int(self.p_real[row])
                outs[k]["_cap"] = dict(z=self.z[row, :, :p].cpu().numpy(), r=self.rlast[row, :, :, :p].cpu().numpy(),
                                       s2=self.s2[row].cpu().numpy(), rho=float(self.rho[row]),
                                       logpi=self.logpi[row].cpu().numpy(), thr=self.thr[row].cpu().numpy(),
                                       lpw=self.lpw[row, :p].cpu().numpy())
            return outs
        JS._JointBucket.outputs = patched
    try:
        t0 = time.time()
        res = JS.joint_susie_batch(z1s, z2s, lds, n1s, n2s, rhos, kinds, cfg, n_refs=nrefs, types=types,
                                   ref_sizes=refn if cfg.ref_var else None, mafs=mafs if cfg.ref_var else None)
        dt = time.time() - t0
    finally:
        JS._JointBucket.outputs = orig
    return res, types, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("ORCH_OUT", "."))
    ap.add_argument("--per-stratum", type=int, default=6)
    ap.add_argument("--mc", type=int, default=400_000)
    ap.add_argument("--ids", default="", help="comma-separated locus ids (overrides the seeded selection)")
    ap.add_argument("--no-rungs", action="store_true", help="skip the +E / fixed-pi rungs")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    loci = []
    for c in CELLS:
        path = SETDIR / f"{c}.h5"
        ids = pick_ids(path, a.per_stratum, c) if not a.ids else [i for i in a.ids.split(",") if f"__{c}__" in i]
        with LocusSetReader(path) as r:            # guard=True: refuses test-split loci
            for i in ids:
                lc = r.read(i)
                assert lc.split == "valid", (i, lc.split)
                loci.append(lc)
    print(f"{len(loci)} validation loci", flush=True)

    rows, eff_rows, summ = [], [], {"n_loci": len(loci), "cells": CELLS, "commit_reviewed": "47ceb97"}
    # ---- main rung: rho + eta, EB pi, default events (capture) ---------------------------------------
    cfg = JS.JointConfig(estimate_eta=True, mem_budget_bytes=2.5e9)
    res, types, dt = run(cfg, loci, capture=True)
    summ["fit_s_rho_eta"] = dt
    g = seeded_rng(VALID_SEED, "p2_review_famA", "mc")
    for lc, r, ty in zip(loci, res, types):
        cap = r["_cap"]
        a_, w_, p1_, p2_ = (np.asarray(r[k], float) for k in ("a", "w", "p1t", "p2t"))
        PS_ie, PD_ie = ie_pd(a_, w_, p1_, p2_)
        PD_dp = dp_pd(w_, p1_, p2_)
        p = len(lc.z1)
        L = cap["r"].shape[0]
        logpi = cap["logpi"]
        thr = cap["thr"]
        assert np.allclose(cap["lpw"], np.log(1.0 / p))
        m_all = cap["z"][None] - cap["r"]                 # (L,2,p) marginal of the other effects
        samples = []
        max_dalpha = 0.0
        for l in range(L):
            al, wst, means, covs = ser_posterior(cap["r"][l], cap["rho"], cap["s2"][l], logpi, p)
            max_dalpha = max(max_dalpha, float(np.abs(al - r["alpha"][l]).max()))
            ev = mc_effect(g, al, wst, means, covs, m_all[l], thr, a.mc, cap["s2"][l])
            samples.append(ev)
            se = lambda q: math.sqrt(max(q * (1 - q), 1e-12) / a.mc)
            for name, mcv, av in (("a", ev[0].mean(), a_[l]), ("w", ev[1].mean(), w_[l]),
                                  ("p1t", ev[2].mean(), p1_[l]), ("p2t", ev[3].mean(), p2_[l])):
                eff_rows.append(dict(locus_id=lc.locus_id, l=l, event=name, A=av, MC=mcv,
                                     z=(mcv - av) / se(max(av, 1.0 / a.mc))))
        blk_any = np.any([s[1] for s in samples], 0)
        h1 = np.any([s[2] for s in samples], 0)
        h2 = np.any([s[3] for s in samples], 0)
        D_mc = float((~blk_any & h1 & h2).mean())
        S_mc = float(np.any([s[0] for s in samples], 0).mean())
        # exclusion probability E2 / E1 via own effect >= w of the other trait on A-style CS (pw_t >= 0.5)
        E_mc = []
        for t, other in ((0, 1), (1, 0)):
            pw = np.asarray(r["pw1"] if t == 0 else r["pw2"])
            mask = np.zeros(p, bool)
            for l in range(L):
                if pw[l] >= cfg.powered_cs_min:
                    srt = np.argsort(-r["alpha"][l], kind="stable")
                    n_in = min(int((np.cumsum(r["alpha"][l][srt]) < cfg.coverage).sum()) + 1, p)
                    mask[srt[:n_in]] = True
            if not mask.any():
                E_mc.append(math.nan); continue
            hit = np.zeros(a.mc, bool)
            wthr = thr[2 + other]
            for s in samples:
                if s[4] is None:
                    continue
                j, st, beta = s[4]
                has = (st == 3) | (st == (2 if other == 1 else 1))
                hit |= has & mask[j] & (np.abs(beta[:, other]) >= wthr)
            E_mc.append(float((~hit).mean()))
        rows.append(dict(locus_id=lc.locus_id, cell=lc.cell_id, stratum=lc.locus_id.split("__")[2],
                         types="/".join(ty), eta=r["eta"], niter=r["niter"], converged=r["converged"],
                         P_S_A=r["P_S"], P_D_A=r["P_D"], P_S_ie=PS_ie, P_D_ie=PD_ie, P_D_dp=PD_dp,
                         P_D_mc=D_mc, P_S_mc=S_mc, E1_A=r["E1"], E2_A=r["E2"], E2_mc=E_mc[0], E1_mc=E_mc[1],
                         max_dalpha_indep=max_dalpha))
        print(rows[-1], flush=True)
    import pandas as pd
    df = pd.DataFrame(rows); ef = pd.DataFrame(eff_rows)
    df.to_csv(out / "pd_check.tsv", sep="\t", index=False)
    ef.to_csv(out / "pd_effects.tsv", sep="\t", index=False)
    mcse = lambda q: np.sqrt(np.clip(q * (1 - q), 1e-12, None) / a.mc)
    summ.update(
        max_abs_PD_A_vs_ie=float((df.P_D_A - df.P_D_ie.clip(0, 1)).abs().max()),
        max_abs_PD_A_vs_dp=float((df.P_D_A - df.P_D_dp).abs().max()),
        max_abs_PS_A_vs_ie=float((df.P_S_A - df.P_S_ie).abs().max()),
        max_abs_PD_A_vs_mc=float((df.P_D_A - df.P_D_mc).abs().max()),
        max_z_PD_A_vs_mc=float(((df.P_D_A - df.P_D_mc).abs() / mcse(df.P_D_A.clip(1.0 / a.mc, 1))).max()),
        max_abs_PS_A_vs_mc=float((df.P_S_A - df.P_S_mc).abs().max()),
        max_abs_E1_A_vs_mc=float((df.E1_A - df.E1_mc).abs().max()), max_abs_E2_A_vs_mc=float((df.E2_A - df.E2_mc).abs().max()),
        max_dalpha_indep=float(df.max_dalpha_indep.max()),
        effect_events_max_abs=float((ef.A - ef.MC).abs().max()), effect_events_max_absz=float(ef.z.abs().max()),
        n_effect_events=len(ef), n_loci_PD_gt_0_05=int((df.P_D_A > 0.05).sum()), n_loci_PS_gt_0_05=int((df.P_S_A > 0.05).sum()),
        mc_draws=a.mc,
    )
    # ---- other rungs: IE / DP combination only ---------------------------------------------------------
    for tag, c2 in () if a.no_rungs else (("rho_eta_E", JS.JointConfig(estimate_eta=True, ref_var=True, convergence="alpha", mem_budget_bytes=2.5e9)),
                    ("rho_eta_pifix", JS.JointConfig(estimate_eta=True, pi_mode="fixed", mem_budget_bytes=2.5e9))):
        try:
            res2, _, dt2 = run(c2, loci, capture=False)
            d_ie, d_dp = [], []
            for r in res2:
                a_, w_, p1_, p2_ = (np.asarray(r[k], float) for k in ("a", "w", "p1t", "p2t"))
                PS_ie, PD_ie = ie_pd(a_, w_, p1_, p2_)
                d_ie.append(abs(r["P_D"] - min(max(PD_ie, 0), 1))); d_dp.append(abs(r["P_D"] - dp_pd(w_, p1_, p2_)))
            summ[f"{tag}_max_abs_PD_vs_ie"] = float(max(d_ie)); summ[f"{tag}_max_abs_PD_vs_dp"] = float(max(d_dp))
            summ[f"{tag}_fit_s"] = dt2
            summ[f"{tag}_n_PD_gt_0_05"] = int(sum(r["P_D"] > 0.05 for r in res2))
        except Exception as e:  # report, never hide
            summ[f"{tag}_error"] = repr(e)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    sys.exit(main())
