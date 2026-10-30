"""Independent recompute of the famC C2 kill test (task p9.verify_C2).

Written from DESIGN v4 §8.1/§8.6 and DESIGN_P2 §4 only; does NOT import src/eval or src/famC.
Realistic AUPRC (S vs D): levels L1-L6 of the M4 main ids; cell weight = 1/#levels x pairing weight
(P1 .5, P2 .25, P3 .25, renormalised over types present in the level) / #cells of that type in the level;
within each cell S and D are power matched on (|mu1|,|mu2|) in 5x5 trait-specific quantile bins over the
pooled S u D loci of the cell (bin k with both classes gets mass (nS_k+nD_k)/2 per class, spread evenly;
single-class bins weight 0); each cell's S u D mass normalised to 1, then times the cell weight.
Failures (missing row, status failed/timeout, NaN s) scored least favourably: S -> s=0, D -> s=1.
Paired SE: SD over 2,000 cluster-bootstrap replicates (LD blocks, all 80 blocks of main) of the AUPRC
difference, weights (incl. quantile bins) recomputed on every replicate's resampled rows.

usage: python reviews/p9_verify_C2/recompute_c2.py > reviews/p9_verify_C2/recompute_c2.out
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/cluster/scratch/mumarc/hackathon_eqtl_gwas")
TRUTH = ROOT / "data/processed/sets/valid_grid_v1.truth.parquet"
IDS = ROOT / "results/baselines_m4/m4_subsample_v1/main.txt"
PRED = {
    "net": ROOT / "results/famC/c2_v1/net_m4/predictions.parquet",
    "nold": ROOT / "results/famC/c2_v1/nold_m4/predictions.parquet",
    "floor": ROOT / "results/famC/c2_v1/floor_m4/predictions.parquet",
    "B": ROOT / "results/famB/p9_m4_valid/B1_M50/predictions.parquet",
    "A": ROOT / "results/famA/famA_m4_valid_v1/predictions/famA_joint_susie_rho_eta_ms_ptld.parquet",
}
ART_DIRS = {  # registered name@version -> (path, mode, registered checksum)
    "famC_C2@1": ("results/famC/c2_v1", "f2245e6f92cc59112ba5b42e33cd52558ec765466c315ad0c78b7490d0561cd2"),
    "famC_C2_model@1": ("results/famC/p9_main_600k", "13162e8f5128e9d391e5ec26c921be49b96c46571c429cfa88559c73b425b1f0"),
    "famC_nold_600k@1": ("results/famC/p9_nold_600k", "85141ba58a9fcbbe82bb90eb8e6b731f88f7f634d5885d96af5df01e0dfd68b7"),
    "famC_floor_600k@1": ("results/famC/p9_floor_600k", "89d9c92bf130dd63751af231569e5d7a9404383ceea93bb03e03221d8bdb38f1"),
    "famB_m4_valid@1": ("results/famB/p9_m4_valid/B1_M50", "f1de8d6694d530f0bdc045520f3a6142e3103297d0af04f9621bde6569bd2336"),
    "famA_m4_valid@1": ("results/famA/famA_m4_valid_v1", "1c2a6c2454977cf8c78819e21addaf69df613462475b901c22904d26948398ba"),
}
PW = {"P1": 0.5, "P2": 0.25, "P3": 0.25}
REAL = {"L1", "L2", "L3", "L4", "L5", "L6"}


def sha(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 23), b""):
            h.update(c)
    return h.hexdigest()


def dir_manifest(root):  # same manifest definition as the registry (relpath \t sha256 per sorted file)
    root = Path(root)
    files = []
    for dp, dns, fns in os.walk(root, followlinks=True):
        dns.sort()
        files += [Path(dp) / f for f in sorted(fns)]
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.relative_to(root)}\t{sha(f)}\n".encode())
    return h.hexdigest()


def ap(y, s, w):
    """weighted step AP, tied scores form one threshold."""
    k = w > 0
    y, s, w = y[k], s[k], w[k]
    o = np.argsort(-s, kind="stable")
    y, s, w = y[o], s[o], w[o]
    tp = np.cumsum(w * y)
    fp = np.cumsum(w * (1 - y))
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tp, fp = tp[last], fp[last]
    rec = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * tp / (tp + fp)))


def qbin(x):
    e = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(e, x, side="right")


def weights(T, rows):
    """realistic S-vs-D weights for (possibly duplicated) row indices `rows` into T."""
    lev, pt, cell = T.level[rows], T.pair[rows], T.cell[rows]
    isS, isD = T.isS[rows], T.isD[rows]
    w = np.zeros(len(rows))
    real = np.isin(lev, list(REAL))
    cells = pd.unique(cell[real])
    meta = {c: (lev[cell == c][0], pt[cell == c][0]) for c in cells}
    levels = sorted({m[0] for m in meta.values()})
    for c in cells:
        L, P = meta[c]
        types = sorted({m[1] for m in meta.values() if m[0] == L})
        tot = sum(PW[t] for t in types)
        ncell = sum(1 for m in meta.values() if m == (L, P))
        cw = (1 / len(levels)) * (PW[P] / tot) / ncell
        idx = np.flatnonzero(cell == c)
        S, D = idx[isS[idx]], idx[isD[idx]]
        if len(S) == 0 or len(D) == 0:
            continue
        U = np.r_[S, D]
        m1, m2 = T.mu1[rows[U]], T.mu2[rows[U]]
        k = qbin(m1) * 5 + qbin(m2)
        kS, kD = k[: len(S)], k[len(S):]
        nS = np.bincount(kS, minlength=25).astype(float)
        nD = np.bincount(kD, minlength=25).astype(float)
        m = np.where((nS > 0) & (nD > 0), (nS + nD) / 2, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            pS = np.where(nS > 0, m / nS, 0.0)
            pD = np.where(nD > 0, m / nD, 0.0)
        ww = np.r_[pS[kS], pD[kD]]
        if ww.sum() <= 0:
            continue
        w[U] = ww / ww.sum() * cw
    return w


class T:  # namespace filled in main()
    pass


def main():
    out = {}
    # ---- provenance: checksums
    out["pred_sha256"] = {k: sha(v) for k, v in PRED.items()}
    out["artifact_manifest"] = {}
    for name, (p, ref) in ART_DIRS.items():
        ck = dir_manifest(ROOT / p)
        out["artifact_manifest"][name] = {"recomputed": ck, "registered": ref, "match": ck == ref}
    for k in ("net", "nold", "floor"):
        rj = json.loads((PRED[k].parent / "runs.json").read_text())
        out.setdefault("runs_json_sha_match", {})[k] = rj["runs"][0]["predictions_sha256"] == out["pred_sha256"][k]
        out.setdefault("runs_json_ckpt", {})[k] = rj["kwargs"].get("checkpoint", rj["kwargs"])

    ids = [l.strip() for l in IDS.read_text().splitlines() if l.strip()]
    tr = pd.read_parquet(TRUTH).set_index("locus_id").loc[ids].reset_index()
    T.level = tr["mismatch_level"].astype(str).to_numpy()
    T.pair = tr["pairing_type"].astype(str).to_numpy()
    T.cell = tr["cell_id"].astype(str).to_numpy()
    T.isS = (tr["oracle_class"] == "S").to_numpy()
    T.isD = (tr["oracle_class"] == "D").to_numpy()
    T.mu1 = np.abs(tr["mu1"].to_numpy(float))
    T.mu2 = np.abs(tr["mu2"].to_numpy(float))
    block = tr["block_id"].astype(str).to_numpy()
    y = T.isS.astype(float)
    n = len(tr)
    S = {}
    out["failures"] = {}
    for k, f in PRED.items():
        p = pd.read_parquet(f).set_index("locus_id")
        assert not p.index.duplicated().any()
        p = p.reindex(tr["locus_id"])
        st = p["status"].astype(object).fillna("missing").astype(str).to_numpy()
        s = p["s"].to_numpy(float)
        fail = (st == "missing") | np.isin(st, ["failed", "timeout"]) | np.isnan(s)
        s = s.copy()
        s[fail & T.isS] = 0.0
        s[fail & T.isD] = 1.0
        S[k] = s
        real_sd = np.isin(T.level, list(REAL)) & (T.isS | T.isD)
        out["failures"][k] = {"n_rows": int((st != "missing").sum()), "n_fail_all": int(fail.sum()),
                              "n_fail_realistic_SD": int((fail & real_sd).sum()),
                              "status": pd.Series(st).value_counts().to_dict()}
    all_rows = np.arange(n)
    W = weights(T, all_rows)
    out["n_realistic_SD_weighted_loci"] = int((W > 0).sum())
    out["auprc_realistic"] = {k: ap(y, S[k], W) for k in S}
    sd = np.isin(T.level, list(REAL)) & (T.isS | T.isD)
    out["auprc_realistic_unweighted"] = {k: ap(y[sd], S[k][sd], np.ones(sd.sum())) for k in S}
    try:
        from sklearn.metrics import average_precision_score
        out["sklearn_check_weighted"] = {k: float(average_precision_score(y[W > 0], S[k][W > 0], sample_weight=W[W > 0]))
                                         for k in S}
    except Exception as e:  # pragma: no cover
        out["sklearn_check_weighted"] = repr(e)

    # ---- paired cluster bootstrap over LD blocks (all main loci; A only on its own 9677 loci not needed here)
    ub, inv = np.unique(block, return_inverse=True)
    groups = [np.flatnonzero(inv == b) for b in range(len(ub))]
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 777)
    K = len(groups)
    pairs = [("net", "nold"), ("net", "floor"), ("net", "B")]
    reps = {p: [] for p in pairs}
    for b in range(2000):
        pick = rng.integers(0, K, K)
        rows = np.concatenate([groups[j] for j in pick])
        w = weights(T, rows)
        yy = y[rows]
        a = {m: ap(yy, S[m][rows], w) for m in ("net", "nold", "floor", "B")}
        for p in pairs:
            reps[p].append(a[p[0]] - a[p[1]])
    out["n_blocks"] = K
    out["paired"] = {}
    for p in pairs:
        r = np.array(reps[p])
        d = out["auprc_realistic"][p[0]] - out["auprc_realistic"][p[1]]
        se = float(np.nanstd(r, ddof=1))
        out["paired"][f"{p[0]}-{p[1]}"] = {"diff": d, "se": se, "n_nan": int(np.isnan(r).sum()),
                                           "ci95_pct": [float(np.nanquantile(r, .025)), float(np.nanquantile(r, .975))],
                                           "beats_by_gt_1SE": bool(d > se)}
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
