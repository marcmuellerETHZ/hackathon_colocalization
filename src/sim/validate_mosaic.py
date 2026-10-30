"""Validate the Li-Stephens mosaic against its source pool (DESIGN §2.1) and pin N_e.

For random 1-Mb windows (seeded) and a grid of N_e:
* r² decay: mean r² over random SNP pairs in distance bins 0-10, 10-50, 50-100, 100-500 kb.
  PRIMARY (the pinning criterion): haplotype r² of pool A (508 haplotypes) vs haplotype r² of the full
  mosaic (40,000 haplotypes). The mosaic's population LD is pool A's *sample* haplotype LD times the
  probability of no copying switch, so this compares like with like (both contain pool A's ~1/508
  sampling-noise floor, the mosaic adds only ~1/40,000).
  SECONDARY: dosage r² of pool A (254 individuals) vs 254-individual subsamples of the mosaic (5 reps).
  This double-counts sampling noise in the mosaic arm (pool noise inherited + new subsample noise), so it is
  biased upward at long range; reported, not used for pinning.
* effective rank of the 1-Mb window LD matrix: entropy rank exp(H(λ/Σλ)) and the number of eigenvalues
  explaining 90% / 99% of the trace, for pool A, a 254-subsample of the mosaic and the full mosaic.

Usage: python -m sim.validate_mosaic [--ne 500 1000 2000 4000 8000 16000] [--n-windows 24]
Writes src/sim/validation/mosaic_validation.json (+ .md table).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common.seeds import SPLIT_SEED, rng
from data.variants import OUT_DIR as VAR_DIR
from sim import params
from sim.mosaic import Mosaic
from sim.world import chrom_name, source_haplotypes

BINS = ((0, 10_000), (10_000, 50_000), (50_000, 100_000), (100_000, 500_000))
OUT = Path(__file__).resolve().parent / "validation"


def available_chroms() -> list[str]:
    return sorted({p.name.split(".")[0] for p in VAR_DIR.glob("chr*.parquet")}, key=lambda c: int(c[3:]))


def pick_windows(n: int, chroms: list[str], g: np.random.Generator) -> list[tuple[str, int, int]]:
    import pandas as pd
    out = []
    lens, allpos = {}, {}
    for c in chroms:
        pos = pd.read_parquet(VAR_DIR / f"{c}.parquet", columns=["pos"])["pos"].to_numpy()
        lens[c] = (int(pos[0]), int(pos[-1]))
        allpos[c] = pos
    w = np.array([lens[c][1] - lens[c][0] for c in chroms], dtype=float)
    while len(out) < n:
        c = chroms[g.choice(len(chroms), p=w / w.sum())]
        a = int(g.integers(lens[c][0], lens[c][1] - 1_000_000))
        if c == "chr6" and a < 33_400_000 and a + 1_000_000 > 28_500_000:
            continue
        n_var = np.searchsorted(allpos[c], a + 1_000_000) - np.searchsorted(allpos[c], a)
        if n_var < 1000:   # skip assembly gaps / centromeres (README D2)
            continue
        out.append((c, a, a + 1_000_000))
    return out


def sample_pairs(pos: np.ndarray, g: np.random.Generator, n_per_bin: int = 20000):
    p = len(pos)
    pairs = {}
    for lo, hi in BINS:
        i = g.integers(0, p, size=4 * n_per_bin)
        # partner: pick a random offset distance in [lo, hi) then nearest variant to the right
        d = g.uniform(lo, hi, size=len(i))
        j = np.searchsorted(pos, pos[i] + d)
        ok = (j < p) & (j != i)
        i, j = i[ok], j[ok]
        dist = pos[j] - pos[i]
        ok = (dist >= lo) & (dist < hi) & (dist > 0)
        pairs[(lo, hi)] = (i[ok][:n_per_bin], j[ok][:n_per_bin])
    return pairs


def std_cols(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    X = X.float()
    m = X.mean(0)
    s = X.std(0, unbiased=False)
    ok = s > 0
    Z = (X - m) / torch.where(ok, s, torch.ones_like(s))
    return Z, ok


def pair_r2(X: torch.Tensor, pairs) -> dict:
    Z, ok = std_cols(X)
    n = Z.shape[0]
    out = {}
    for b, (i, j) in pairs.items():
        it = torch.as_tensor(i, device=Z.device)
        jt = torch.as_tensor(j, device=Z.device)
        valid = ok[it] & ok[jt]
        r = (Z[:, it] * Z[:, jt]).sum(0) / n
        out[b] = r[valid] ** 2
    return out


def eff_rank(X: torch.Tensor) -> dict:
    Z, ok = std_cols(X)
    Z = Z[:, ok]
    n, p = Z.shape
    if n < p:
        ev = torch.linalg.eigvalsh((Z @ Z.T / n).double())
    else:
        ev = torch.linalg.eigvalsh((Z.T @ Z / n).double())
    ev = torch.clamp(ev, min=0).flip(0).cpu().numpy()
    w = ev / ev.sum()
    w = w[w > 0]
    cum = np.cumsum(ev) / ev.sum()
    return {"entropy_rank": float(np.exp(-(w * np.log(w)).sum())),
            "n90": int(np.searchsorted(cum, 0.90) + 1), "n99": int(np.searchsorted(cum, 0.99) + 1), "p": int(p)}


def run(ne_grid, n_windows, n_sub=5, device="cuda", pool="A"):
    g = rng(SPLIT_SEED, "validate_mosaic", "windows")
    chroms = available_chroms()
    wins = pick_windows(n_windows, chroms, g)
    res = {"pool": pool, "chroms_available": chroms, "windows": wins, "ne_grid": list(ne_grid), "theta": params.THETA,
           "bins": BINS, "per_window": []}
    for (c, a, b) in wins:
        t0 = time.time()
        Hs, v = source_haplotypes(pool, c, a, b)
        pos = v["pos"].to_numpy()
        gw = rng(SPLIT_SEED, "validate_mosaic", c, a)
        pairs = sample_pairs(pos, gw)
        # compare like with like: only variants with MAF >= 0.01 in the SOURCE pool (studies apply the same
        # in-sample filter; otherwise miscopy turns pool-monomorphic sites into ultra-rare mosaic variants
        # with r² ~ 0 that exist only in the mosaic arm). No-op for pool A (variant set = pool-A MAF >= 0.01).
        af_pool = Hs.mean(0)
        okv = np.minimum(af_pool, 1 - af_pool) >= params.MIN_MAF
        pairs = {b: (i[okv[i] & okv[j]], j[okv[i] & okv[j]]) for b, (i, j) in pairs.items()}
        Hs_t = torch.as_tensor(Hs, device=device)
        Gs = Hs_t[0::2] + Hs_t[1::2]
        rec = {"window": (c, a, b), "p": len(v), "pool": {}, "mosaic": {}}
        r2p = pair_r2(Gs, pairs)
        r2h = pair_r2(Hs_t, pairs)
        rec["pool"]["dos_r2"] = {f"{lo}-{hi}": float(r2p[(lo, hi)].mean()) for lo, hi in BINS}
        rec["pool"]["hap_r2"] = {f"{lo}-{hi}": float(r2h[(lo, hi)].mean()) for lo, hi in BINS}
        rec["pool"]["rank"] = eff_rank(Gs)
        for ne in ne_grid:
            m = Mosaic(pool, params.N_SYN, f"validate_{pool}", ne=ne)
            h = m.haplotypes_from_source(c, Hs, v, device)
            G = h[0::2] + h[1::2]
            sub = []
            for r in range(n_sub):
                idx = torch.as_tensor(rng(SPLIT_SEED, "validate_mosaic", "sub", c, a, r).choice(
                    G.shape[0], Gs.shape[0], replace=False), device=device)
                rr = pair_r2(G[idx], pairs)
                sub.append({f"{lo}-{hi}": float(rr[(lo, hi)].mean()) for lo, hi in BINS})
            rfull = pair_r2(h, pairs)
            idx = torch.as_tensor(rng(SPLIT_SEED, "validate_mosaic", "sub", c, a, 0).choice(
                G.shape[0], Gs.shape[0], replace=False), device=device)
            rec["mosaic"][str(ne)] = {
                "dos_r2_sub": {k: float(np.mean([s[k] for s in sub])) for k in sub[0]},
                "hap_r2_full": {f"{lo}-{hi}": float(rfull[(lo, hi)].mean()) for lo, hi in BINS},
                "rank_sub": eff_rank(G[idx]),
                "rank_full": eff_rank(G),
                "mean_switches_per_hap_per_window": float(
                    (4 * ne / (m.K * 100)) * (v["cm"].iloc[-1] - v["cm"].iloc[0])),
            }
            del h, G
            torch.cuda.empty_cache()
        res["per_window"].append(rec)
        print(f"{c}:{a}-{b} p={len(v)} {time.time()-t0:.1f}s", flush=True)
    # summary: pooled over windows (mean of per-window means), relative error per bin
    summ = {}
    for ne in ne_grid:
        s = {}
        for lo, hi in BINS:
            k = f"{lo}-{hi}"
            pool_d = np.mean([w["pool"]["dos_r2"][k] for w in res["per_window"]])
            mos_d = np.mean([w["mosaic"][str(ne)]["dos_r2_sub"][k] for w in res["per_window"]])
            pool_h = np.mean([w["pool"]["hap_r2"][k] for w in res["per_window"]])
            mos_h = np.mean([w["mosaic"][str(ne)]["hap_r2_full"][k] for w in res["per_window"]])
            s[k] = {"pool_dos": pool_d, "mosaic_dos_sub": mos_d, "rel_err_dos": mos_d / pool_d - 1,
                    "pool_hap": pool_h, "mosaic_hap_full": mos_h, "rel_err_hap": mos_h / pool_h - 1}
        rk = {}
        for which in ("entropy_rank", "n90", "n99"):
            rk[which] = {
                "pool": float(np.mean([w["pool"]["rank"][which] for w in res["per_window"]])),
                "mosaic_sub": float(np.mean([w["mosaic"][str(ne)]["rank_sub"][which] for w in res["per_window"]])),
                "mosaic_full": float(np.mean([w["mosaic"][str(ne)]["rank_full"][which] for w in res["per_window"]])),
            }
        s["rank"] = rk
        s["max_abs_rel_err_dos"] = max(abs(s[f"{lo}-{hi}"]["rel_err_dos"]) for lo, hi in BINS)
        s["max_abs_rel_err_hap"] = max(abs(s[f"{lo}-{hi}"]["rel_err_hap"]) for lo, hi in BINS)
        summ[str(ne)] = s
    res["summary"] = summ
    return res


def to_md(res) -> str:
    bins = [f"{lo}-{hi}" for lo, hi in BINS]
    head = " | ".join(f"{lo//1000}-{hi//1000} kb" for lo, hi in BINS)
    lines = [f"Pool {res.get('pool', 'A')}; {len(res['per_window'])} windows; mean r² over random pairs.", "",
             "PRIMARY: haplotype r², pool vs full mosaic (relative error)", "",
             f"| N_e | {head} | max abs rel err |", "|" + "---|" * (len(BINS) + 2)]
    for ne, s in res["summary"].items():
        cells = [f"{s[b]['pool_hap']:.4f}/{s[b]['mosaic_hap_full']:.4f} ({s[b]['rel_err_hap']:+.3f})" for b in bins]
        lines.append(f"| {ne} | " + " | ".join(cells) + f" | {s['max_abs_rel_err_hap']:.3f} |")
    lines += ["", "SECONDARY: dosage r², pool (n individuals) vs n-individual mosaic subsamples; effective rank", "",
              f"| N_e | {head} | max abs rel err | entropy rank pool / mosaic-sub / mosaic-20k | n99 pool / sub / 20k |",
              "|" + "---|" * (len(BINS) + 4)]
    for ne, s in res["summary"].items():
        cells = [f"{s[b]['pool_dos']:.4f}/{s[b]['mosaic_dos_sub']:.4f} ({s[b]['rel_err_dos']:+.3f})" for b in bins]
        rk, n99 = s["rank"]["entropy_rank"], s["rank"]["n99"]
        lines.append(f"| {ne} | " + " | ".join(cells) + f" | {s['max_abs_rel_err_dos']:.3f} | "
                     f"{rk['pool']:.1f} / {rk['mosaic_sub']:.1f} / {rk['mosaic_full']:.1f} | "
                     f"{n99['pool']:.0f} / {n99['mosaic_sub']:.0f} / {n99['mosaic_full']:.0f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ne", type=float, nargs="+", default=[500, 1000, 2000, 4000, 8000, 16000])
    ap.add_argument("--n-windows", type=int, default=24)
    ap.add_argument("--pool", default="A")
    ap.add_argument("--md-only", action="store_true", help="re-render the .md table from an existing json")
    ap.add_argument("--out", default=str(OUT / "mosaic_validation.json"))
    a = ap.parse_args()
    out = Path(a.out)
    if a.md_only:
        out.with_suffix(".md").write_text(to_md(json.loads(out.read_text())))
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    res = run(a.ne, a.n_windows, pool=a.pool)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    out.with_suffix(".md").write_text(to_md(res))
    print(to_md(res))


if __name__ == "__main__":
    main()
