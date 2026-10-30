"""Empirical calibration of the reference-size LD variance for the Family A layer E (DESIGN §5 layer E).

Why: the normal-theory formula Var(r̂) = (1 - r²)²/n is wrong for genotypes (reports/simulation_engineer_M3.md,
decision 4: median ratio 0.71–0.84, ~10× too small near r² = 1). Layer E therefore uses an empirical table
    n · Var(r̂_jk)  as a function of (r²_jk bin, min-MAF bin of the pair),
estimated by resampling individuals of a real pool:
  * population = the NFE pool (pools A ∪ B of data/processed/pools/pools_v1.tsv, 404 individuals);
  * windows of p consecutive QC'd SNVs (MAF >= 0.01 in NFE), entirely inside TRAIN-split LD blocks;
  * for each reference size n, draw subsamples of n individuals without replacement, compute the
    within-subsample correlation r̂ (standardised dosages, as reference LD is computed), and the per-pair
    variance across draws about its mean; the finite-population correction (N - 1)/(N - n) converts it to
    the variance for n individuals drawn from an infinite population;
  * pairs are binned by the full-pool r² and by min(MAF_j, MAF_k) of the pool.
  * v2 (P3, review p2_review_famA defect 3): subsample draws in which either variant of a pair is monomorphic
    are EXCLUDED for that pair (v1 set r̂ := 0 there, which never occurs at inference: the simulator drops
    zero-variance reference columns); the per-pair variance uses that pair's k usable draws (pairs with k < 2 are
    dropped for that n; `counts_by_n`). Note: bins use the full-pool r², while layer E looks the table up by the
    REFERENCE r̂² (errors-in-variables mismatch, documented, not corrected).
Output JSON: {"r2_edges", "maf_edges", "n_list", "kappa": n·Var per bin pooled over n, "kappa_by_n",
"normal_theory": mean (1 - r²)² per bin, "counts"}, plus a Markdown summary.

Usage: python -m famA.validation.calibrate_ref_ld_var --out src/famA/ref_ld_var_v1.json \
           --n-windows 40 --p 300 --draws 200 --n-list 100,150,254
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.seeds import SPLIT_SEED, rng as seeded_rng
from famG.validation.genotypes import PROJECT, chrom_variants, read_genotypes

R2_EDGES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995, 1.0 + 1e-12]
MAF_EDGES = [0.0, 0.05, 0.2, 0.5 + 1e-12]


def nfe_sample_index() -> np.ndarray:
    pools = pd.read_csv(PROJECT / "data" / "processed" / "pools" / "pools_v1.tsv", sep="\t")
    nfe = set(pools.loc[pools["pool"].isin(["A", "B"]), "sample_id"].astype(str))
    psam = pd.read_csv(PROJECT / "data" / "processed" / "1kg" / "chr22.psam", sep="\t")
    iid = psam.iloc[:, 0].astype(str).tolist()
    idx = np.array([i for i, s in enumerate(iid) if s in nfe], dtype=np.uint32)
    if len(idx) != len(nfe):
        raise RuntimeError(f"NFE samples found in psam: {len(idx)} != {len(nfe)}")
    return idx


def train_blocks() -> pd.DataFrame:
    b = pd.read_csv(PROJECT / "datasets" / "splits" / "blocks_v1.tsv", sep="\t")
    return b[b["split"] == "train"].reset_index(drop=True)


def corr(G: np.ndarray) -> np.ndarray:
    X = G - G.mean(0)
    sd = X.std(0)
    X = X / np.where(sd > 0, sd, 1.0)
    return X.T @ X / X.shape[0]


def calibrate(n_windows: int, p: int, draws: int, n_list: list[int], seed_key: str = "ref_ld_var_v1",
              version: str = "ref_ld_var_v2", exclude_monomorphic: bool = True):
    g = seeded_rng(SPLIT_SEED, "famA", seed_key)
    sidx = nfe_sample_index()
    N = len(sidx)
    blocks = train_blocks()
    nb, nm = len(R2_EDGES) - 1, len(MAF_EDGES) - 1
    acc = {n: np.zeros((nb, nm)) for n in n_list}
    cnt_n = {n: np.zeros((nb, nm)) for n in n_list}
    cnt = np.zeros((nb, nm))
    nt = np.zeros((nb, nm))
    used = []
    tries = 0
    while len(used) < n_windows:
        tries += 1
        if tries > 50 * n_windows:
            raise RuntimeError("could not find enough train-split windows")
        blk = blocks.iloc[int(g.integers(len(blocks)))]
        chrom = int(str(blk["chr"]).replace("chr", ""))
        tab = chrom_variants(chrom)
        inb = tab[(tab.POS >= blk["start"]) & (tab.POS < blk["end"])]
        if len(inb) < 3 * p:
            continue
        st = int(g.integers(0, len(inb) - 3 * p))
        sub = inb.iloc[st:st + 3 * p]
        G = read_genotypes(chrom, sub.vidx.to_numpy(), sidx)
        f = G.mean(0) / 2
        maf = np.minimum(f, 1 - f)
        ok = np.nonzero(maf >= 0.01)[0][:p]
        if len(ok) < p:
            continue
        G, maf = G[:, ok], maf[ok]
        R = corr(G)
        iu = np.triu_indices(p, 1)
        r2 = R[iu] ** 2
        mm = np.minimum(maf[iu[0]], maf[iu[1]])
        bi = np.clip(np.searchsorted(R2_EDGES, r2, side="right") - 1, 0, nb - 1)
        mi = np.clip(np.searchsorted(MAF_EDGES, mm, side="right") - 1, 0, nm - 1)
        flat = bi * nm + mi
        cnt += np.bincount(flat, minlength=nb * nm).reshape(nb, nm)
        nt += np.bincount(flat, weights=(1 - r2) ** 2, minlength=nb * nm).reshape(nb, nm)
        for n in n_list:
            s1 = np.zeros(len(r2))
            s2 = np.zeros(len(r2))
            k = np.zeros(len(r2))
            for _ in range(draws):
                rows = g.choice(N, size=n, replace=False)
                Gs = G[rows]
                Rh = corr(Gs)[iu]              # monomorphic columns give r̂ = 0 here
                if exclude_monomorphic:
                    poly = Gs.std(0) > 0
                    use = (poly[iu[0]] & poly[iu[1]]).astype(float)
                else:
                    use = np.ones(len(r2))
                s1 += Rh * use
                s2 += Rh * Rh * use
                k += use
            ok_p = k >= 2
            kk = np.where(ok_p, k, 2.0)
            var = np.where(ok_p, (s2 - s1 * s1 / kk) / (kk - 1), 0.0)
            var_inf = var * (N - 1) / (N - n)  # finite-population correction
            acc[n] += np.bincount(flat, weights=n * var_inf * ok_p, minlength=nb * nm).reshape(nb, nm)
            cnt_n[n] += np.bincount(flat, weights=ok_p.astype(float), minlength=nb * nm).reshape(nb, nm)
        used.append(dict(block=blk["block_id"], chrom=chrom, start_pos=int(sub.POS.iloc[ok[0]]),
                         end_pos=int(sub.POS.iloc[ok[-1]])))
        print(f"window {len(used)}/{n_windows} {blk['block_id']}", flush=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        kbn = {str(n): (acc[n] / cnt_n[n]).tolist() for n in n_list}
        kappa = (sum(acc.values()) / sum(cnt_n.values()))
        normal = nt / cnt
    return dict(version=version, seed_key=seed_key, exclude_monomorphic=bool(exclude_monomorphic),
                counts_by_n={str(n): cnt_n[n].astype(int).tolist() for n in n_list},
                lookup_note="binned by full-pool r2; layer E looks up by reference r_hat^2 (errors-in-variables)",
                population="NFE (pools A+B, 1000G 30x)", N_pool=int(N),
                r2_edges=R2_EDGES, maf_edges=MAF_EDGES, n_list=n_list, draws=draws, p=p, windows=used,
                kappa=np.nan_to_num(kappa, nan=-1).tolist(), kappa_by_n=kbn,
                normal_theory=np.nan_to_num(normal, nan=-1).tolist(), counts=cnt.astype(int).tolist(),
                definition="kappa[b][m] = mean over pairs in bin of n * Var(r_hat_n) (FPC-corrected); "
                           "Var(r_hat) at reference size n_ref = kappa / n_ref")


def summary_md(res: dict) -> str:
    lines = ["| r² bin | " + " | ".join(f"min MAF [{a:g},{b:.2g})" for a, b in zip(res["maf_edges"][:-1], res["maf_edges"][1:]))
             + " | normal theory (1−r²)² (pooled) | pairs |", "|---" * (len(res["maf_edges"]) + 2) + "|"]
    K = np.array(res["kappa"]); NT = np.array(res["normal_theory"]); C = np.array(res["counts"])
    for b in range(K.shape[0]):
        lo, hi = res["r2_edges"][b], min(res["r2_edges"][b + 1], 1.0)
        ntp = (NT[b] * C[b]).sum() / max(C[b].sum(), 1)
        lines.append(f"| [{lo:g},{hi:g}) | " + " | ".join(f"{K[b, m]:.3g}" for m in range(K.shape[1]))
                     + f" | {ntp:.3g} | {int(C[b].sum())} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-windows", type=int, default=40)
    ap.add_argument("--p", type=int, default=300)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--n-list", default="100,150,254")
    ap.add_argument("--seed-key", default="ref_ld_var_v1")      # v2 reuses v1's windows and draws (same seed)
    ap.add_argument("--version", default="ref_ld_var_v2")
    ap.add_argument("--keep-monomorphic", action="store_true")  # v1 convention (r̂ := 0)
    a = ap.parse_args()
    res = calibrate(a.n_windows, a.p, a.draws, [int(x) for x in a.n_list.split(",")], a.seed_key, a.version,
                    not a.keep_monomorphic)
    out = Path(a.out)
    out.write_text(json.dumps(res, indent=1))
    out.with_suffix(".md").write_text("n·Var(r̂) (empirical, FPC-corrected) by r² bin and min-MAF bin\n\n" + summary_md(res) + "\n")
    print(summary_md(res))


if __name__ == "__main__":
    main()
