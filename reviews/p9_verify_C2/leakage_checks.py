"""Leakage sanity checks for C2 (context only; no verdict change). Does not import src/eval or src/famC.

1. Split overlap: blocks touched by M4-main loci windows vs blocks touched by any train-split window
   (splits/genes_v1.tsv.gz, protein-coding anchors, as used by the train sampler) and block split labels.
2. Univariate 'design cue' AUPRCs (realistic weights, same as recompute_c2.py) of observable per-locus
   metadata: N1, N2, N_s, n_ref, p, rho_hat, rho_se, trait types; best of both directions.
3. An LD-free, training-free statistic: Pearson corr(z1, z2) over SNPs with max(|z1|,|z2|) >= 3 (z from
   valid_grid_v1.h5), to see whether S-vs-D is separable without LD or learning.
4. LightGBM floor feature importance (gain) from its registered metrics.json.

usage: python reviews/p9_verify_C2/leakage_checks.py > reviews/p9_verify_C2/leakage_checks.out
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from recompute_c2 import ROOT, IDS, TRUTH, PRED, REAL, T, ap, weights  # noqa: E402


def main():
    out = {}
    ids = [l.strip() for l in IDS.read_text().splitlines() if l.strip()]
    tr = pd.read_parquet(TRUTH).set_index("locus_id").loc[ids].reset_index()
    blocks = pd.read_csv(ROOT / "splits/blocks_v1.tsv", sep="\t")
    genes = pd.read_csv(ROOT / "splits/genes_v1.tsv.gz", sep="\t")

    def touched(chrom, s, e):
        b = blocks[(blocks.chr == chrom) & (blocks.start < e) & (blocks.end > s)]
        return set(b.block_id)

    m4_blocks = set()
    for c, s, e in tr[["window_chr", "window_start", "window_end"]].drop_duplicates().itertuples(index=False):
        m4_blocks |= touched(c, s, e)
    m4_anchor = set(tr.block_id)
    g = genes[(genes.split == "train") & (genes.gene_type == "protein_coding") & (genes.window_end > genes.window_start)]
    tr_blocks = set()
    for c, s, e in g[["chr", "window_start", "window_end"]].itertuples(index=False):
        tr_blocks |= touched(c, s, e)
    spl = blocks.set_index("block_id").split
    out["split_overlap"] = {
        "m4_anchor_blocks": len(m4_anchor), "m4_touched_blocks": len(m4_blocks),
        "m4_touched_split_counts": pd.Series([spl[b] for b in m4_blocks]).value_counts().to_dict(),
        "m4_truth_split_col": tr["split"].value_counts().to_dict(),
        "train_window_blocks": len(tr_blocks),
        "train_window_split_counts": pd.Series([spl[b] for b in tr_blocks]).value_counts().to_dict(),
        "overlap_blocks": sorted(m4_blocks & tr_blocks),
        "m4_windows_overlapping_train_window_bp": 0,
    }
    # base-pair overlap of windows (stronger than block overlap)
    nov = 0
    gw = {c: d[["window_start", "window_end"]].to_numpy() for c, d in g.groupby("chr")}
    for c, s, e in tr[["window_chr", "window_start", "window_end"]].drop_duplicates().itertuples(index=False):
        a = gw.get(c)
        if a is not None and ((a[:, 0] < e) & (a[:, 1] > s)).any():
            nov += 1
    out["split_overlap"]["m4_windows_overlapping_train_window_bp"] = nov

    # ---- cue AUPRCs
    T.level = tr["mismatch_level"].astype(str).to_numpy()
    T.pair = tr["pairing_type"].astype(str).to_numpy()
    T.cell = tr["cell_id"].astype(str).to_numpy()
    T.isS = (tr["oracle_class"] == "S").to_numpy()
    T.isD = (tr["oracle_class"] == "D").to_numpy()
    T.mu1 = np.abs(tr["mu1"].to_numpy(float))
    T.mu2 = np.abs(tr["mu2"].to_numpy(float))
    W = weights(T, np.arange(len(tr)))
    y = T.isS.astype(float)
    m = W > 0
    f = h5py.File(ROOT / "data/processed/sets/valid_grid_v1.h5", "r")
    nref = np.array([float(f[l].attrs.get("n_ref", np.nan)) if m[i] else np.nan for i, l in enumerate(tr.locus_id)])
    cues = {c: tr[c].to_numpy(float) for c in ["N1", "N2", "N_s", "p", "rho_hat", "rho_se", "lead_abs_z1", "lead_abs_z2"]}
    cues["n_ref"] = nref
    cues["type_pair"] = pd.factorize(tr.trait1_type.astype(str) + tr.trait2_type.astype(str))[0].astype(float)
    rnd = np.random.default_rng(1).random(len(tr))
    out["cue_auprc_realistic"] = {}
    for k, v in cues.items():
        v = np.nan_to_num(v, nan=np.nanmedian(v[m]))
        v = v + 1e-9 * rnd  # break exact ties identically in both directions
        out["cue_auprc_realistic"][k] = max(ap(y[m], v[m], W[m]), ap(y[m], -v[m], W[m]))
    out["cue_auprc_realistic"]["_random_baseline"] = ap(y[m], rnd[m], W[m])

    # ---- LD-free, training-free statistic
    corr = np.full(len(tr), np.nan)
    for i in np.flatnonzero(m):
        gg = f[tr.locus_id[i]]
        z1, z2 = gg["z1"][()], gg["z2"][()]
        k = np.maximum(np.abs(z1), np.abs(z2)) >= 3
        if k.sum() >= 3:
            corr[i] = np.corrcoef(np.abs(z1[k]), np.abs(z2[k]))[0, 1]
    c = np.nan_to_num(corr, nan=-1.0)
    out["ldfree_abs_z_corr_auprc_realistic"] = ap(y[m], c[m], W[m])
    out["ldfree_n_nan"] = int(np.isnan(corr[m]).sum())
    # same statistic per level
    out["ldfree_by_level"] = {L: ap(y[m & (T.level == L)], c[m & (T.level == L)], W[m & (T.level == L)])
                              for L in sorted(set(T.level[m]))}

    # ---- floor importance
    mj = json.loads((ROOT / "results/famC/p9_floor_600k/metrics.json").read_text())
    imp = pd.Series(mj["importance"]).sort_values(ascending=False)
    out["floor_gain_share_top12"] = (imp / imp.sum()).round(4).head(12).to_dict()
    meta_feats = ["rho_hat", "rho_se", "log_n1", "log_n2", "log_nref", "log_p", "type1", "type2"]
    out["floor_gain_share_meta_feats"] = float((imp[meta_feats] / imp.sum()).sum())
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
