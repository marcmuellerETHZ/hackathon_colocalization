"""Diagnostics for the v2 real-data rho-hat (p4.realdata_rho): why is rho-hat > 0 in eQTL x eQTL pairs of
donor-disjoint cohorts? Train + valid only (reads z1, z2, pos; no LD).

  python scripts/real/rho_diagnostics.py --workers 32

(a) Distance profile: per window, SNPs with |z| < 2 in both traits, binned by distance to the nearest signal SNP
    (max(|z1|,|z2|) > 4); pooled unit-variance truncated MLE per bin (stats.rho.mle_from_stats), jackknife over
    windows. Leakage of shared signal through LD predicts rho-hat decreasing with distance; a genuine null
    correlation (sample overlap) predicts a flat profile.
(b) Window rho-hat (rho_v2.parquet) by quintile of signal strength min(|lead1_z|, |lead2_z|) (candidates).
(c) Window rho-hat by the sign of z1*z2 at the strongest joint SNP (argmax min(|z1|,|z2|)): concordant vs
    discordant shared signal.
Writes data/processed/real/locus_sets/rho_v2_diagnostics.json.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from data.real.rho_update import LOCUS_DIR, PAIRS, locus_file
from stats.rho import jackknife_from_blocks

BINS = [0, 50_000, 100_000, 200_000, 400_000, 10 ** 9]
LABELS = ["0-50kb", "50-100kb", "100-200kb", "200-400kb", ">400kb", "no_signal"]


def _job(args):
    from eval.io import LocusSetReader
    path, ids = args
    out = []
    with LocusSetReader(path) as r:
        for lid in ids:
            L = r.read(lid, load_ld=False)
            if L.split not in ("train", "valid"):
                raise PermissionError(lid)
            z1, z2, pos = L.z1, L.z2, L.pos
            sig = np.where(np.maximum(np.abs(z1), np.abs(z2)) > 4)[0]
            use = (np.abs(z1) < 2) & (np.abs(z2) < 2)
            if len(sig):
                d = np.min(np.abs(pos[:, None] - pos[sig][None, :]), axis=1)
                b = np.digitize(d, BINS[1:-1])
            else:
                b = np.full(len(z1), len(LABELS) - 1)
            rec = {"locus_id": lid, "split": L.split, "d1": L.extra_attrs["dataset1"], "d2": L.extra_attrs["dataset2"]}
            for k in range(len(LABELS)):
                m = use & (b == k)
                rec[f"n_{k}"] = float(m.sum())
                rec[f"q_{k}"] = float(np.sum(z1[m] ** 2 + z2[m] ** 2))
                rec[f"s_{k}"] = float(np.sum(z1[m] * z2[m]))
            j = int(np.argmax(np.minimum(np.abs(z1), np.abs(z2))))
            rec["joint_min_z"] = float(min(abs(z1[j]), abs(z2[j])))
            rec["joint_sign"] = int(np.sign(z1[j] * z2[j]))
            out.append(rec)
    return out


def profile(df: pd.DataFrame) -> list[dict]:
    res = []
    for k, lab in enumerate(LABELS):
        n, q, s = (df[f"{x}_{k}"].to_numpy() for x in ("n", "q", "s"))
        keep = n > 0
        if n.sum() < 10 or keep.sum() < 2:
            res.append({"bin": lab, "n_snp": int(n.sum()), "rho_hat": None, "se": None})
            continue
        r, se = jackknife_from_blocks(n[keep], q[keep], s[keep])
        res.append({"bin": lab, "n_snp": int(n.sum()), "n_windows": int(keep.sum()), "rho_hat": round(r, 4),
                    "se": round(se, 4)})
    return res


def main() -> None:
    from eval.io import list_loci
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    files = [("R2", "poolB"), ("R3b", "poolB"), ("R4b", "geuv_insample_pcadj")]
    jobs, meta = [], []
    for arm, mode in files:
        for split in ("train", "valid"):
            p = locus_file(arm, split, mode)
            ids = list_loci(p)
            for i in range(0, len(ids), 100):
                jobs.append((str(p), ids[i:i + 100]))
                meta.append(arm)
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for arm, res in zip(meta, ex.map(_job, jobs)):
            for r in res:
                r["arm"] = arm
            rows += res
    df = pd.DataFrame(rows)
    df["pairing"] = np.where(df["d1"].str.startswith("FG_"), "gwas_x_eqtl", "eqtl_x_eqtl")
    out = {"distance_profile": {}, "by_strength": {}, "by_joint_sign": {}}
    for (arm, pairing), g in df.groupby(["arm", "pairing"]):
        out["distance_profile"][f"{arm}/{pairing}"] = profile(g)
    v2 = pd.read_parquet(LOCUS_DIR / "rho_v2.parquet")
    v2 = v2[v2["mode"].isin(["poolB", "geuv_insample_pcadj"])]
    v2 = v2.merge(df[["arm", "locus_id", "joint_min_z", "joint_sign", "pairing"]], on=["arm", "locus_id"])
    for arm, g in v2.groupby("arm"):
        c = pd.read_csv(PAIRS / f"{arm}_candidates.tsv", sep="\t", usecols=["locus_id", "lead1_z", "lead2_z", "split"])
        c = c[c["split"].isin(["train", "valid"])]
        g = g.merge(c[["locus_id", "lead1_z", "lead2_z"]], on="locus_id")
        for pairing, h in g.groupby("pairing"):
            h = h.copy()
            h["strength"] = np.minimum(h["lead1_z"].abs(), h["lead2_z"].abs())
            h["q"] = pd.qcut(h["strength"], 5, labels=False, duplicates="drop")
            out["by_strength"][f"{arm}/{pairing}"] = [
                {"quintile": int(q), "strength_median": round(float(x["strength"].median()), 2), "n": int(len(x)),
                 "rho_hat_median": round(float(x["rho_hat"].median()), 4),
                 "rho_hat_mean": round(float(x["rho_hat"].mean()), 4)} for q, x in h.groupby("q")]
            out["by_joint_sign"][f"{arm}/{pairing}"] = [
                {"joint_sign": int(sgn), "n": int(len(x)), "joint_min_z_median": round(float(x["joint_min_z"].median()), 2),
                 "rho_hat_median": round(float(x["rho_hat"].median()), 4),
                 "rho_hat_mean": round(float(x["rho_hat"].mean()), 4)}
                for sgn, x in h[h["joint_min_z"] > 4].groupby("joint_sign")]
    (LOCUS_DIR / "rho_v2_diagnostics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
