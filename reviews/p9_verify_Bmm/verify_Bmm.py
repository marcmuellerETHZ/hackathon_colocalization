"""Independent recomputation of the B-mm phase-2 ladder (famB_mm_ladder@1): OP2 thresholds, rule (c) at OP2
(readings ii and iii of DECISIONS P4-1), B1 at OP2, and the new-method gate (DESIGN_P2 §4 M1).

Own numpy/pandas implementation; does NOT import src/eval/metrics.py, src/eval/op2.py or any famB ladder code.
Validation (screen_v2) data only.

Usage (from project root, env activated):
    python reviews/p9_verify_Bmm/verify_Bmm.py --root <project root> --out reviews/p9_verify_Bmm/verify_Bmm.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CELLS = ["L0_P1_r_p=0.0", "L0_P3_rho=0.5", "L2_P1_r_p=0.0", "L2_P3_rho=0.5", "L4_P1_r_p=0.0", "L4_P3_rho=0.5"]
W_LIM, D_LIM = 0.025, 0.10
ABF = "coloc_abf_W0.15_p12_5e-06"
BASELINES = ["coloc_abf", "coloc_susie", "coloc_susie_mm", "colocboost", "colocboost_m1", "mvsusie", "sharepro",
             "prop_coloc", "coloc_prop_test", "lead_snp"]  # tier-1 baselines; strength_only is a control
CONTROL = ["strength_only"]
BMM, B1M = "exactcoloc_mm_gaussian_M50", "exactcoloc_gaussian_M50"


# ------------------------------------------------------------------ checksums (orch 'full' manifest definition)
def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_manifest(root: Path) -> str:
    if root.is_file():
        return sha_file(root)
    files = []
    for dp, dns, fns in os.walk(root, followlinks=True):
        dns.sort()
        files += [Path(dp) / f for f in sorted(fns)]
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.relative_to(root)}\t{sha_file(f)}\n".encode())
    return h.hexdigest()


# ------------------------------------------------------------------ data
def load(root: Path):
    sets = root / "data/processed/sets/screen_v2"
    truth = pd.concat([pd.read_parquet(sets / f"{c}.truth.parquet") for c in CELLS], ignore_index=True)
    assert (truth["split"] == "valid").all()
    truth["mu1a"], truth["mu2a"] = truth["mu1"].abs(), truth["mu2"].abs()
    preds = {}
    bm = pd.read_parquet(root / "results/famB/p9_screen_v2/Bmm_M50/predictions.parquet")
    b1 = pd.read_parquet(root / "results/famB/screen_v2/M50/predictions.parquet")
    bl = pd.read_parquet(root / "results/baselines_screen_v2/collected_v1/predictions.parquet")
    preds[BMM] = bm[bm.method == BMM]
    preds[B1M] = b1[b1.method == B1M]
    for m in BASELINES + CONTROL:
        src = ABF if m == "coloc_abf" else m
        preds[m] = bl[bl.method == src]
    return truth, preds


def scored(truth, df):
    """Join predictions onto truth; least-favourable failure scoring (DESIGN §4.3). Returns frame restricted to
    the loci the method was run on (rows present), with s/P_S/P_D after LF replacement and an 'has_PS' flag."""
    d = truth.merge(df[["locus_id", "s", "P_S", "P_D", "status"]], on="locus_id", how="inner")
    has_ps = d.loc[~d.status.isin(["failed", "timeout"]), "P_S"].notna().any()
    has_pd = d.loc[~d.status.isin(["failed", "timeout"]), "P_D"].notna().any()
    fail = d.status.isin(["failed", "timeout"]) | d.s.isna()
    if has_ps:
        fail |= d.P_S.isna()
    if has_pd:
        fail |= d.P_D.isna()
    lf = {"S": (0.0, 0.0, 1.0), "D": (1.0, 1.0, 0.0), "W": (0.5, 1.0, 1.0)}
    for k, (s, ps, pd_) in lf.items():
        m = fail & (d.oracle_class == k)
        d.loc[m, "s"] = s
        if has_ps:
            d.loc[m, "P_S"] = ps
        if has_pd:
            d.loc[m, "P_D"] = pd_
    if not has_ps:
        d["P_S"] = -np.inf  # never calls S
    d["failed"] = fail
    return d, bool(has_ps), int(fail.sum())


def cell_weights(cells):
    """DESIGN §8.6 pooling over the cells present: equal per level; P1 .5 / P3 .25 renormalised within level."""
    pw = {"P1": 0.5, "P2": 0.25, "P3": 0.25}
    lev = sorted({c[:2] for c in cells})
    out = {}
    for L in lev:
        cs = [c for c in cells if c[:2] == L]
        types = sorted({c.split("_")[1] for c in cs})
        tot = sum(pw[t] for t in types)
        for c in cs:
            t = c.split("_")[1]
            n_t = sum(1 for x in cs if x.split("_")[1] == t)
            out[c] = (1 / len(lev)) * (pw[t] / tot) / n_t
    return out


def qbin(x, nq=5):
    e = np.quantile(x, np.linspace(0, 1, nq + 1)[1:-1])
    return np.searchsorted(e, x, side="right")


def power_match(d):
    """Per cell: 5x5 trait-specific quantile bins of (|mu1|,|mu2|) over pooled S∪D; in a bin with both classes each
    class gets mass (nS+nD)/2 spread evenly; bins with one class get 0; cell S∪D mass normalised to 1."""
    w = pd.Series(0.0, index=d.index)
    for c, g in d.groupby("cell_key"):
        g = g[g.oracle_class.isin(["S", "D"])]
        b = qbin(g.mu1a.to_numpy()) * 5 + qbin(g.mu2a.to_numpy())
        isS = (g.oracle_class == "S").to_numpy()
        nS = np.bincount(b[isS], minlength=25).astype(float)
        nD = np.bincount(b[~isS], minlength=25).astype(float)
        m = np.where((nS > 0) & (nD > 0), (nS + nD) / 2, 0.0)
        ww = np.where(isS, m[b] / np.maximum(nS[b], 1), m[b] / np.maximum(nD[b], 1))
        ww = ww / ww.sum()
        w.loc[g.index] = ww
    return w.to_numpy()


def smallest_t(score, w, lim):
    """Smallest t such that sum(w[score>=t])/sum(w) <= lim (scan over the values and just above them)."""
    s = np.asarray(score, float)
    w = np.ones(len(s)) if w is None else np.asarray(w, float)
    if not np.isfinite(s).any():
        return -np.inf
    o = np.argsort(-s, kind="mergesort")
    s_sorted, cw = s[o], np.cumsum(w[o]) / w.sum()
    vals = np.unique(s[np.isfinite(s)])
    # rate at t=v: mass of scores >= v
    for t in np.r_[vals[:1], np.nextafter(vals, np.inf)]:
        r = w[s >= t].sum() / w.sum()
        if r <= lim + 1e-12:
            return float(t)
    return np.inf


def op2(d, has_ps):
    """OP2 t_S on the realistic fit set (L2+L4 cells present in d)."""
    fit = d[d.level.isin(["L2", "L4"])].copy()
    cw = cell_weights(sorted(fit.cell_key.unique()))
    fit["pm"] = power_match(fit)
    isW = (fit.oracle_class == "W").to_numpy()
    nW = fit[isW].groupby("cell_key").size()
    wW = fit.loc[isW, "cell_key"].map(lambda c: cw[c] * 0.5 / nW[c]).to_numpy()
    isD = ((fit.oracle_class == "D") & (fit.pm > 0)).to_numpy()
    if not has_ps:
        return dict(t_S=np.inf, feasible=False, n_pmD=int(isD.sum()))
    ps = fit.P_S.to_numpy()
    tW = smallest_t(ps[isW], wW, W_LIM)
    tD = smallest_t(ps[isD], None, D_LIM)
    t = max(tW, tD)
    # direct joint check by scanning (independent of the max() shortcut)
    cand = np.unique(np.r_[ps, np.nextafter(ps, np.inf)])
    tj = np.inf
    for tt in cand[cand >= min(tW, tD)]:
        rW = wW[ps[isW] >= tt].sum() / wW.sum()
        rD = (ps[isD] >= tt).mean()
        if rW <= W_LIM + 1e-12 and rD <= D_LIM + 1e-12:
            tj = tt
            break
    feas = t <= 1.0
    return dict(t_S=float(t) if feas else 1.0, t_S_call=float(t) if feas else np.inf, t_joint_scan=float(tj),
                t_W_side=tW, t_D_side=tD, feasible=bool(feas), n_pmD=int(isD.sum()), n_W=int(isW.sum()),
                rate_W=float(wW[ps[isW] >= t].sum() / wW.sum()), rate_D=float((ps[isD] >= t).mean()))


def ap(y, s, w):
    """Weighted step-wise average precision, ties grouped as one threshold."""
    y, s, w = map(np.asarray, (y, s, w))
    k = w > 0
    y, s, w = y[k], s[k], w[k]
    o = np.argsort(-s, kind="mergesort")
    y, s, w = y[o], s[o], w[o]
    last = np.r_[np.diff(s) != 0, True]
    tp = np.cumsum(w * y)[last]
    fp = np.cumsum(w * (1 - y))[last]
    P = tp[-1]
    rec = tp / P
    prec = tp / (tp + fp)
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def realistic_auprc(d, idx=None):
    fit = d[d.level.isin(["L2", "L4"]) & d.oracle_class.isin(["S", "D"])].copy()
    cw = cell_weights(sorted(fit.cell_key.unique()))
    fit["w"] = power_match(fit) * fit.cell_key.map(cw).to_numpy()
    return fit


def false_S(d, t_call, mask):
    sub = d[mask & (d.oracle_class == "D")]
    hit = (sub.P_S >= t_call).astype(float)
    per_cell = hit.groupby(sub.cell_key).mean()
    return float(per_cell.mean()), float(hit.mean()), len(sub)


def boot_margin(bm_hit, cp_hit, cells, B=2000, seed=1):
    """Locus bootstrap (stratified by cell) of the kill margin min(b-2c, b-c-0.02)."""
    rng = np.random.default_rng(seed)
    out = []
    groups = [np.nonzero(cells == c)[0] for c in np.unique(cells)]
    for _ in range(B):
        bs, cs = [], []
        for g in groups:
            i = rng.choice(g, len(g))
            bs.append(bm_hit[i].mean()); cs.append(cp_hit[i].mean())
        b, c = np.mean(bs), np.mean(cs)
        out.append(min(b - 2 * c, b - c - 0.02))
    out = np.array(out)
    return dict(se=float(out.std(ddof=1)), p_not_kill=float((out <= 0).mean()))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--root", default=".")
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--n-boot", type=int, default=2000)
    a = ap_.parse_args()
    root = Path(a.root).resolve()
    res = {}

    # 1. checksums
    reg = {f"{x['name']}@{x['version']}": x for x in yaml.safe_load(open(root / "ARTIFACTS.yaml"))["artifacts"]}
    ck = {}
    for name in ["famB_mm_screen_v2@1", "famB_screen_v2@1", "baselines_screen_v2@1", "op2_screen_v2_baselines@1",
                 "famB_mm_ladder@1"]:
        x = reg[name]
        got = sha_manifest(root / x["path"])
        ck[name] = dict(registered=x["checksum"], computed=got, match=got == x["checksum"])
    sums = dict(l.split()[::-1] for l in open(root / "data/processed/sets/SHA256SUMS") if "screen_v2/" in l)
    for c in CELLS:
        k = f"screen_v2/{c}.truth.parquet"
        got = sha_file(root / "data/processed/sets" / k)
        ck[k] = dict(registered=sums[k], computed=got, match=got == sums[k])
    res["checksums"] = ck

    truth, preds = load(root)
    truth["cell_key"] = truth["mismatch_level"] + "_" + truth["cell_id"].str.split("|").str[1] + "_" + \
        truth["cell_id"].str.split("|").str[2]
    truth["level"] = truth["mismatch_level"]
    truth["is_rho05"] = truth["cell_key"].str.contains("rho=0.5")

    frames, thr = {}, {}
    for m, df in preds.items():
        d, has_ps, nf = scored(truth, df)
        frames[m] = d
        thr[m] = dict(op2(d, has_ps), has_PS=has_ps, n_failed=nf, n=len(d))
    res["op2"] = thr
    # registry comparison
    reg_op2 = json.load(open(root / "results/baselines_p9/op2_screen_v2_v1/op2_screen_v2.json"))["methods"]
    lad = json.load(open(root / "results/famB/p9_ladder/ladder_Bmm.json"))
    cmp = {}
    for m in thr:
        src = ABF if m == "coloc_abf" else m
        r = reg_op2.get(src, {}).get("OP2", {})
        cmp[m] = dict(mine=thr[m].get("t_S"), registry=r.get("t_S"),
                      ladder=lad["operating_points"].get(m, {}).get("OP2", {}).get("t_S_reported"))
    res["op2_compare"] = cmp

    # 2. rule (c) at OP2
    rc = {}
    for stratum, maskf in [("L4", lambda d: d.level == "L4"), ("rho0.5", lambda d: d.is_rho05)]:
        tab = {}
        for m, d in frames.items():
            t = thr[m].get("t_S_call", np.inf)
            mk = maskf(d)
            fs_cell, fs_pool, nD = false_S(d, t, mk)
            nScalls = int(((d.P_S >= t) & mk).sum())
            # per-stratum AUPRC: mean over stratum cells of power-matched S-vs-D AP on s
            sub = d[mk & d.oracle_class.isin(["S", "D"])].copy()
            sub["pm"] = power_match(sub)
            au = np.mean([ap((g.oracle_class == "S").astype(float), g.s, g.pm) for _, g in sub.groupby("cell_key")])
            tab[m] = dict(false_S=fs_cell, false_S_pooled=fs_pool, n_D=nD, n_S_calls=nScalls, auprc=float(au))
        base = [m for m in BASELINES if m in tab]
        best_auprc = max(base, key=lambda m: tab[m]["auprc"])
        nondeg = [m for m in base if tab[m]["n_S_calls"] >= 1]
        lowest_nd = min(nondeg, key=lambda m: tab[m]["false_S"])
        lowest_all = min(base, key=lambda m: tab[m]["false_S"])
        verdicts = {}
        for reading, comp in [("ii_best_auprc", best_auprc), ("iii_lowest_nondegenerate", lowest_nd),
                              ("i_lowest_all_descriptive", lowest_all)]:
            dc = frames[comp][maskf(frames[comp]) & (frames[comp].oracle_class == "D")]
            db = frames[BMM].set_index("locus_id").loc[dc.locus_id]
            tb, tc = thr[BMM]["t_S_call"], thr[comp].get("t_S_call", np.inf)
            bh = (db.P_S.to_numpy() >= tb).astype(float)
            chh = (dc.P_S.to_numpy() >= tc).astype(float)
            cells = dc.cell_key.to_numpy()
            b = float(pd.Series(bh).groupby(cells).mean().mean())
            c = float(pd.Series(chh).groupby(cells).mean().mean())
            kill = bool(b > 2 * c and b - c > 0.02)
            verdicts[reading] = dict(comparator=comp, false_S_Bmm_paired=b, false_S_comp=c, kill_line=max(2 * c, c + 0.02),
                                     margin=b - max(2 * c, c + 0.02), kill=kill, n_D=len(dc),
                                     boot=boot_margin(bh, chh, cells, a.n_boot))
        rc[stratum] = dict(table=tab, verdicts=verdicts,
                           binding_kill=bool(verdicts["ii_best_auprc"]["kill"] or verdicts["iii_lowest_nondegenerate"]["kill"]))
    res["rule_c_OP2"] = rc
    # fragility (descriptive): B-mm L4 D false-S per cell, counts, and the t_S at which (ii) would stop killing
    d = frames[BMM]
    L4D = d[(d.level == "L4") & (d.oracle_class == "D")]
    ps = np.sort(L4D.P_S.to_numpy())[::-1]
    line = rc["L4"]["verdicts"]["ii_best_auprc"]["kill_line"]
    k_allowed = int(np.floor(line * len(ps) + 1e-9))  # max # S calls on L4 D without exceeding the kill line
    res["fragility_L4_ii"] = dict(
        per_cell={c: float((g.P_S >= thr[BMM]["t_S_call"]).mean()) for c, g in L4D.groupby("cell_key")},
        n_called=int((ps >= thr[BMM]["t_S_call"]).sum()), n_D=len(ps), n_allowed_at_line=k_allowed,
        t_S_needed_to_escape=float(ps[k_allowed]) if k_allowed < len(ps) else None)

    # 3. B1 at OP2: false-S on Dmid (D, r2 in [0.3,0.8)) ∪ Mx (locus_type M, class != S), L0+L2
    b1 = {}
    for m in [BMM, "coloc_susie"]:
        d = frames[m]
        mk = d.level.isin(["L0", "L2"]) & (((d.oracle_class == "D") & (d.r2 >= 0.3) & (d.r2 < 0.8)) |
                                           ((d.locus_type == "M") & (d.oracle_class != "S")))
        sub = d[mk]
        hit = (sub.P_S >= thr[m]["t_S_call"]).astype(float)
        b1[m] = dict(pooled=float(hit.mean()), cell_mean=float(hit.groupby(sub.cell_key).mean().mean()), n=len(sub))
    res["B1_OP2"] = b1

    # 4. gate: realistic AUPRC (S vs D, L2+L4, §8.6 weights) on s
    fb = realistic_auprc(frames[BMM]).set_index("locus_id")
    f1 = realistic_auprc(frames[B1M]).set_index("locus_id")
    f1 = f1.loc[fb.index]
    y = (fb.oracle_class == "S").astype(float).to_numpy()
    w = fb.w.to_numpy()
    g_bm, g_b1 = ap(y, fb.s.to_numpy(), w), ap(y, f1.s.to_numpy(), w)
    rng = np.random.default_rng(7)
    n = len(fb)
    loc = []
    for _ in range(a.n_boot):
        i = rng.integers(0, n, n)
        loc.append(ap(y[i], fb.s.to_numpy()[i], w[i]) - ap(y[i], f1.s.to_numpy()[i], w[i]))
    blocks = fb.block_id.to_numpy()
    ub = np.unique(blocks)
    bidx = {b: np.nonzero(blocks == b)[0] for b in ub}
    blk = []
    for _ in range(a.n_boot):
        i = np.concatenate([bidx[b] for b in rng.choice(ub, len(ub))])
        blk.append(ap(y[i], fb.s.to_numpy()[i], w[i]) - ap(y[i], f1.s.to_numpy()[i], w[i]))
    res["gate"] = dict(auprc_Bmm=g_bm, auprc_B=g_b1, diff=g_bm - g_b1, se_locus=float(np.std(loc, ddof=1)),
                       se_block=float(np.std(blk, ddof=1)), n_blocks=int(len(ub)),
                       unweighted_auprc_Bmm=ap(y, fb.s.to_numpy(), np.ones(n)),
                       unweighted_auprc_B=ap(y, f1.s.to_numpy(), np.ones(n)))
    res["gate"]["pass"] = bool(res["gate"]["diff"] > max(res["gate"]["se_locus"], res["gate"]["se_block"]))

    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
