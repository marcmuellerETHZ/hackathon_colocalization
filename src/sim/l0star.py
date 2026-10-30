"""World L0* (DESIGN v4 §5 Family B, simulator-prior mode; C1 kill-test): a world in which Family B's
simulator-prior model is exactly correct, so B's oracle-class posterior is the Bayes-optimal ceiling.

Specification (DESIGN v4 §5 B, R3-stat MF5):
  * short windows: p <= 30 SNPs (default p = 20, decision D6(g)), chosen at random among MAF >= 0.05 variants of a random 100-kb
    segment inside a window of the requested split (README D13);
  * the same LD for both traits: R_tilde = (1 - eta) R + eta I, R = target-population (NFE mosaic) LD, eta = 1e-3;
    data are generated WITH R_tilde (the matrix given to methods):  z = R_tilde lambda + e,
    e ~ N(0, V (x) R_tilde), V = [[1, rho], [rho, 1]];  mu = R_tilde lambda;
  * no background, no M loci, <= 2 causal variants per trait;
  * configuration prior: DESIGN §2.4 types S/D/W renormalised (0.30, 0.30, 0.25)/0.85 with the §2.4 extras
    (S: +1 trait-specific variant w.p. 0.5; D: +1 w.p. 0.3) and uniform positions (README D13: uniform D partner,
    W = exactly one trait with one causal variant); exact log-prior in `config_logprior`;
  * effects: every (trait, causal variant) own effect lambda is drawn i.i.d. from an 8-component zero-mean
    Gaussian scale mixture LAMBDA_MIX[trait] (fitted once to the main generator's lambda draws, `fit_lambda_mixture`);
  * true rho given to methods (attribute rho_hat = rho, rho_se = 0 in L0* sets only);
  * oracle classes with the trait types (default trait 1 GWAS-like, trait 2 molecular).
Class-balanced sets record the natural class shares of this prior (CLASS_PRIOR, MC) so posteriors can be
corrected for rejection sampling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim import params
from sim.oracle import oracle_class

ETA = 1e-3
P_DEFAULT = 20   # D6(g): L0* at p = 20 (B simulator-prior mode 0.13 s/locus vs 1.2 s at p = 30)
TYPE_PRIOR = {k: params.LOCUS_TYPES[k] / 0.85 for k in ("S", "D", "W")}
SIGMA_GRID = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
# Mixture weights fitted by fit_lambda_mixture(seed=20261026, n=200000) to |lambda| of the main generator
# (P1 pairing: trait 1 GWAS-like with N1 in {20k,100k,400k}, trait 2 molecular with N2 in {100,...,5000}).
LAMBDA_MIX = {
    1: None,   # filled below from _FITTED (kept explicit so B can import the exact numbers)
    2: None,
}
_FITTED = {   # fit_lambda_mixture(seed=20261026, n=200000, iters=500), rounded to 5 decimals (renormalised)
    1: [0.0, 0.0, 0.0229, 0.37072, 0.22708, 0.15883, 0.21486, 0.00561],
    2: [0.0, 0.0, 0.15196, 0.48091, 0.1038, 0.26334, 0.0, 0.0],
}


def _set_mix(fitted):
    for t in (1, 2):
        w = np.asarray(fitted[t], dtype=float)
        LAMBDA_MIX[t] = (w / w.sum(), SIGMA_GRID.copy()) if w.sum() > 0 else None


_set_mix(_FITTED)


# ------------------------------------------------------------------------------------------- prior
def fit_lambda_mixture(seed: int = 20261026, n: int = 400_000, iters: int = 500) -> dict:
    """EM for the weights of a zero-mean Gaussian scale mixture with fixed SIGMA_GRID, fitted to signed
    lambda = +-sqrt(N q) drawn as in the main generator (P1 pairing)."""
    from common.seeds import rng as _rng
    from sim.architecture import draw_pve
    g = _rng(seed, "l0star", "lambda_fit")
    out = {}
    for t, kind, Ns in ((1, "gwas", params.N1_GRID), (2, "mol", params.N2_GRID_P1)):
        lam = np.empty(n)
        for i in range(n):
            strong = kind == "gwas" and g.random() < params.P_STRONG
            q = draw_pve(g, kind, strong)
            lam[i] = math.sqrt(Ns[g.integers(len(Ns))] * q) * (1 if g.random() < 0.5 else -1)
        w = np.full(len(SIGMA_GRID), 1 / len(SIGMA_GRID))
        dens = np.exp(-0.5 * (lam[:, None] / SIGMA_GRID) ** 2) / (SIGMA_GRID * math.sqrt(2 * math.pi))
        for _ in range(iters):
            r = dens * w
            r /= r.sum(1, keepdims=True)
            w = r.mean(0)
        out[t] = w.tolist()
    return out


def draw_lambda(g: np.random.Generator, trait: int, size: int) -> np.ndarray:
    w, s = LAMBDA_MIX[trait]
    k = g.choice(len(w), size=size, p=w)
    return g.normal(0.0, s[k])


def lambda_logprior(lam: np.ndarray, trait: int) -> np.ndarray:
    w, s = LAMBDA_MIX[trait]
    lam = np.asarray(lam, dtype=float)[..., None]
    return np.log((w * np.exp(-0.5 * (lam / s) ** 2) / (s * math.sqrt(2 * math.pi))).sum(-1))


def sample_config(g: np.random.Generator, p: int) -> tuple[str, list[int], list[int]]:
    types = list(TYPE_PRIOR)
    t = types[g.choice(3, p=[TYPE_PRIOR[k] for k in types])]
    if t == "S":
        j = int(g.integers(p))
        c1, c2 = [j], [j]
        if g.random() < params.S_EXTRA:
            k = int(g.choice([x for x in range(p) if x != j]))
            (c1 if g.random() < 0.5 else c2).append(k)
    elif t == "D":
        j = int(g.integers(p))
        k = int(g.choice([x for x in range(p) if x != j]))
        c1, c2 = ([j], [k]) if g.random() < 0.5 else ([k], [j])
        if g.random() < params.D_EXTRA:
            e = int(g.choice([x for x in range(p) if x not in (j, k)]))
            (c1 if g.random() < 0.5 else c2).append(e)
    else:
        j = int(g.integers(p))
        c1, c2 = ([j], []) if g.random() < 0.5 else ([], [j])
    return t, sorted(c1), sorted(c2)


def config_logprior(c1, c2, p: int) -> float:
    """Exact log P(C1, C2) of `sample_config` (unordered causal sets); -inf if impossible."""
    s1, s2 = set(c1), set(c2)
    sh = s1 & s2
    n1, n2 = len(s1), len(s2)
    if n1 > 2 or n2 > 2:
        return -math.inf
    pS, pD, pW = TYPE_PRIOR["S"], TYPE_PRIOR["D"], TYPE_PRIOR["W"]
    if len(sh) == 1:
        if (n1, n2) == (1, 1):
            return math.log(pS * (1 - params.S_EXTRA) / p)
        if (n1, n2) in ((2, 1), (1, 2)):
            return math.log(pS * params.S_EXTRA / (p * (p - 1) * 2))
        return -math.inf
    if sh:
        return -math.inf
    if n1 >= 1 and n2 >= 1:
        if (n1, n2) == (1, 1):
            return math.log(pD * (1 - params.D_EXTRA) / (p * (p - 1)))
        if (n1, n2) in ((2, 1), (1, 2)):
            return math.log(pD * params.D_EXTRA / (p * (p - 1) * (p - 2)))
        return -math.inf
    if (n1, n2) in ((1, 0), (0, 1)):
        return math.log(pW * 0.5 / p)
    return -math.inf


# ------------------------------------------------------------------------------------------- data
@dataclass
class Segment:
    chrom: str
    pos: np.ndarray
    variant_id: np.ndarray
    maf: np.ndarray
    R: np.ndarray          # target-population LD (float64)
    Rt: np.ndarray         # R_tilde = (1 - eta) R + eta I
    L: np.ndarray          # cholesky(R_tilde)
    window: tuple
    block_id: str


def make_segment(ctx, g: np.random.Generator, p: int = P_DEFAULT, seg_len: int = 100_000,
                 min_maf: float = 0.05) -> Segment | None:
    """p SNPs from a random 100-kb segment of a WindowContext; R from the NFE mosaic population."""
    v = ctx.v
    P = ctx.pop("NFE")
    maf = np.minimum(P.af, 1 - P.af)
    lo, hi = ctx.start, max(ctx.start + 1, ctx.end - seg_len)
    a = int(g.integers(lo, hi))
    idx = np.where((v["pos"].to_numpy() > a) & (v["pos"].to_numpy() <= a + seg_len) & (maf >= min_maf))[0]
    if len(idx) < p:
        return None
    idx = np.sort(g.choice(idx, p, replace=False))
    import torch
    Z = P.std[:, torch.as_tensor(idx, device=ctx.device)].double()
    R = (Z.T @ Z / P.N).cpu().numpy()
    Rt = ((1 - ETA) * R + ETA * np.eye(p)).astype(np.float32).astype(np.float64)   # exactly the stored matrix
    return Segment(ctx.chrom, v["pos"].to_numpy()[idx], v["variant_id"].to_numpy()[idx], maf[idx], R, Rt,
                   np.linalg.cholesky(Rt), (ctx.chrom, ctx.start, ctx.end), ctx.anchor_block)


def sample_locus(seg: Segment, g: np.random.Generator, rho: float = 0.0, kinds=("gwas", "mol")) -> dict:
    p = len(seg.pos)
    ltype, c1, c2 = sample_config(g, p)
    lam1 = np.zeros(p)
    lam2 = np.zeros(p)
    if c1:
        lam1[c1] = draw_lambda(g, 1, len(c1))
    if c2:
        lam2[c2] = draw_lambda(g, 2, len(c2))
    mu1, mu2 = seg.Rt @ lam1, seg.Rt @ lam2
    e1 = seg.L @ g.standard_normal(p)
    e2 = rho * e1 + math.sqrt(1 - rho * rho) * (seg.L @ g.standard_normal(p))
    o = oracle_class(c1, lam1[c1], mu1[c1], c2, lam2[c2], mu2[c2], lambda a, b: seg.Rt[a, b] ** 2, *kinds)
    return {"ltype": ltype, "c1": c1, "c2": c2, "lam1": lam1, "lam2": lam2, "mu1": mu1, "mu2": mu2,
            "z1": mu1 + e1, "z2": mu2 + e2, "oracle": o, "rho": rho}


def class_prior_mc(n: int = 100_000, p: int = P_DEFAULT, seed: int = 20261026, kinds=("gwas", "mol")) -> dict:
    """Natural oracle-class shares of the L0* prior with independent SNPs (R = I) and with a fixed AR(1) LD;
    the class of a locus depends on R only through mu and r², so shares are reported per LD model."""
    from common.seeds import rng as _rng
    out = {}
    for name, R in (("R=I", np.eye(p)), ("AR1(0.8)", 0.8 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p))))):
        Rt = (1 - ETA) * R + ETA * np.eye(p)
        seg = Segment("toy", np.arange(p), np.arange(p).astype(str), np.full(p, 0.3), R, Rt, np.linalg.cholesky(Rt),
                      ("toy", 0, p), "toy")
        g = _rng(seed, "l0star", "class_prior", name)
        cnt = {}
        for _ in range(n):
            c = sample_locus(seg, g, 0.0, kinds)["oracle"].cls
            cnt[c] = cnt.get(c, 0) + 1
        out[name] = {k: v / n for k, v in sorted(cnt.items())}
    return out


# ------------------------------------------------------------------------------------------- set builder
def build_set(set_name: str, split: str, seed: int, per_class: dict, out_dir, rho_values=(0.0,),
              loci_per_segment: int = 50, n_nominal=(100_000, 1_000), kinds=("gwas", "mol"), device="cuda",
              log=print) -> dict:
    """Class-balanced L0* locus set (INTERFACES §2, dense R = R_tilde, rho_hat = true rho, rho_se = 0).
    Windows of `split` are visited in seeded order; one 30-SNP segment per window; candidate loci are accepted
    by oracle class until per_class counts are reached (exact rejection sampling; natural class shares of the
    candidates are stored in the stats so posteriors can be corrected)."""
    import json
    from pathlib import Path

    import pandas as pd

    from common.seeds import rng as _rng
    from eval.io import LocusData, write_locus_set
    from eval.truth import write_truth
    from sim.generator import WindowContext
    from sim.windows import split_windows

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5 = out_dir / f"{set_name}.h5"
    if h5.exists():
        h5.unlink()
    wins = split_windows(split)
    order = list(_rng(seed, set_name, "windows").permutation(len(wins)))
    filled = {k: 0 for k in per_class}
    cand_counts: dict = {}
    truth, pending = [], []
    k = 0
    visits = 0
    while any(filled[c] < per_class[c] for c in per_class):
        if not order:
            raise RuntimeError("ran out of windows")
        w = wins.iloc[order.pop(0)]
        ctx = WindowContext(w["chr"], int(w["window_start"]), int(w["window_end"]), split=str(w["split"]),
                            block_ids=tuple(w["block_ids"]), anchor_block=str(w["block_id"]),
                            anchor=int(w["anchor"]), device=device)
        gs = _rng(seed, set_name, "segment", visits)
        visits += 1
        seg = make_segment(ctx, gs) if ctx.p >= 200 else None
        ctx.release()
        if seg is None:
            continue
        for _ in range(loci_per_segment):
            g = _rng(seed, set_name, "locus", k)
            rho = float(rho_values[int(g.integers(len(rho_values)))])
            loc = sample_locus(seg, g, rho, kinds)
            cls = loc["oracle"].cls
            cand_counts[cls] = cand_counts.get(cls, 0) + 1
            kk = k
            k += 1
            if cls not in per_class or filled[cls] >= per_class[cls]:
                continue
            filled[cls] += 1
            lid = f"{set_name}__{cls}__{kk:07d}"
            p = len(seg.pos)
            se1 = 1 / np.sqrt(2 * n_nominal[0] * seg.maf * (1 - seg.maf))
            se2 = 1 / np.sqrt(2 * n_nominal[1] * seg.maf * (1 - seg.maf))
            pending.append(LocusData(
                locus_id=lid, variant_id=seg.variant_id.astype(object), pos=seg.pos.astype(np.int64),
                z1=loc["z1"], z2=loc["z2"], beta1=loc["z1"] * se1, se1=se1, beta2=loc["z2"] * se2, se2=se2,
                n1=np.full(p, float(n_nominal[0])), n2=np.full(p, float(n_nominal[1])), maf1=seg.maf, maf2=seg.maf,
                maf_ref=seg.maf, ld_type="dense", n_ref=int(1e9), R=seg.Rt.astype(np.float32),
                rho_hat=rho, rho_se=0.0, split=str(w["split"]), block_ids=tuple(w["block_ids"]),
                mismatch_level="L0star", pairing_type="L0star", cell_id=f"L0star|rho={rho}",
                window_chr=seg.chrom, window_start=int(w["window_start"]), window_end=int(w["window_end"]),
                extra_attrs={"eta": ETA, "trait1_type": kinds[0], "trait2_type": kinds[1]}))
            o = loc["oracle"]
            truth.append({"locus_id": lid, "oracle_class": cls, "mu1": o.mu1, "mu2": o.mu2, "r2": o.r2, "d": o.d,
                          "rho": rho, "block_id": seg.block_id, "cell_id": f"L0star|rho={rho}",
                          "split": str(w["split"]), "mismatch_level": "L0star", "pairing_type": "L0star",
                          "overlap": rho, "control": "", "causal1": loc["c1"], "causal2": loc["c2"],
                          "effect1": loc["lam1"][loc["c1"]].tolist(), "effect2": loc["lam2"][loc["c2"]].tolist(),
                          "mu1_all": loc["mu1"][loc["c1"]].tolist(), "mu2_all": loc["mu2"][loc["c2"]].tolist(),
                          "locus_type": loc["ltype"], "cd1": -1 if o.cd1 is None else o.cd1,
                          "cd2": -1 if o.cd2 is None else o.cd2,
                          "lead_abs_z1": float(np.abs(loc["z1"]).max()), "lead_abs_z2": float(np.abs(loc["z2"]).max())})
        if pending:
            write_locus_set(h5, pending, mode="a")
            pending = []
    write_truth(pd.DataFrame(truth), out_dir / f"{set_name}.truth.parquet")
    n = sum(cand_counts.values())
    shares = {c: v / n for c, v in cand_counts.items()}
    stats = {"set": set_name, "seed": seed, "split": split, "per_class": per_class, "segments_visited": visits,
             "candidates": n, "candidate_class_counts": cand_counts,
             "natural_class_shares": shares,
             # Monte Carlo SE of the natural shares (binomial), for the class-prior correction (R3-stat 5)
             "natural_class_shares_mc_se": {c: math.sqrt(v * (1 - v) / n) for c, v in shares.items()},
             "eta": ETA, "lambda_mix": {t: {"w": LAMBDA_MIX[t][0].tolist(), "sigma": LAMBDA_MIX[t][1].tolist()}
                                         for t in (1, 2)},
             "type_prior": TYPE_PRIOR, "kinds": kinds, "rho_values": list(rho_values)}
    (out_dir / f"{set_name}.stats.json").write_text(json.dumps(stats, indent=1))
    log(json.dumps({k_: v for k_, v in stats.items() if k_ not in ("lambda_mix",)}))
    return stats


L0STAR_RHO = (0.0, 0.3, -0.3, 0.5, -0.5, 0.8, -0.8)   # README D13: true rho per locus, uniform over these values


def main():
    import argparse
    import os
    from pathlib import Path
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    ap = argparse.ArgumentParser(description="Build an L0* locus set (validation split only)")
    ap.add_argument("--set", default="l0star_v1")
    ap.add_argument("--split", default="valid", choices=["valid"])
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--per-class", type=int, default=500, help="S, D, W, G each (U is too rare to fill)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "data/processed/sets"))
    a = ap.parse_args()
    import time
    t0 = time.time()
    st = build_set(a.set, a.split, a.seed, {c: a.per_class for c in ("S", "D", "W", "G")}, a.out,
                   rho_values=L0STAR_RHO, log=lambda s: print(s, flush=True))
    st["runtime_s"] = time.time() - t0
    import json
    (Path(a.out) / f"{a.set}.stats.json").write_text(json.dumps(st, indent=1, default=float))


if __name__ == "__main__":
    main()
