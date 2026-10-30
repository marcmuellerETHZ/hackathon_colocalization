"""Candidate locus pairs for the real-data arms R2, R3b, R4b and realism summaries R1 (DESIGN §2.5).

Unit: a gene window (TSS +- 500 kb clipped to the same-split block run; datasets/splits/genes_v1.tsv). Every pair
inherits the split of its gene. Selection uses only gene-level / credible-set metadata (eQTL Catalogue
SuSiE credible sets, FinnGen SuSiE credible sets), never data of the pair's partner beyond what the rules
below state, and never SuSiE lead agreement between the two traits.

Observed power gate for one trait in a window (DESIGN §1): >= 1 pure 95% credible set (purity min |r| >= 0.5)
whose lead variant (max PIP) lies in the window and has p < alpha (1e-5 molecular, 5e-8 GWAS-like).
The lead is the strongest in-CS variant, so this implies min p < alpha in the window.

R2  (partial positives): same gene, same tissue / cell type, two eQTL cohorts with no shared donors;
    both pass the molecular gate. Gene-level selection only.
R3b (pseudo-negatives, locus-level D): FinnGen trait x eQTL gene; the eQTL gene has exactly one credible
    set (eQTL Catalogue SuSiE, any purity) and it passes the gate; exactly one FinnGen credible set (any
    purity) has a member in the window, it is pure, its lead is in the window with p < 5e-8; the two CSs
    share no variant; lead r^2 < 0.1 in pool-B 1000G LD and in FinnGen R12 in-sample LD.
R4b (exact in-sample QTL LD): GEUVADIS genes passing the gate, paired with (a) every FinnGen trait with a
    pure CS (lead p < 5e-8) whose lead is in the window and (b) TwinsUK LCL when it passes the gate.
R1  (realism): per dataset / trait distributions on train-split genes / CSs only.

Outputs (data/processed/real/pairs/): <arm>_candidates.tsv (train + valid, full columns), <arm>_test_ids.tsv
(test: identifiers only), r1_*.tsv, counts.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

from data.real import sources
from data.real.config import (ALPHA_GWAS, ALPHA_QTL, CS_MIN_R2, OUT, R3B_MAX_LEAD_R2, SAME_DONOR_POOLS,
                              assert_env, eqtl_datasets, finngen_traits)
from data.real.harmonise import variant_key
from data.real.regions import gene_table

PAIRS = OUT / "pairs"
GEUVADIS = "QTD000110"
TWINSUK_LCL = "QTD000539"
TEST_ID_COLS = ["locus_id", "arm", "dataset1", "dataset2", "gene_id", "split", "chr", "win_lo", "win_hi"]


def _site(v: str) -> str:
    c, p, a, b = variant_key(v).split(":")
    return f"{c}:{p}:" + ":".join(sorted((a, b)))


# --------------------------------------------------------------------------------------------------
# per-dataset gene gate
# --------------------------------------------------------------------------------------------------
def eqtl_gene_cs(ds_id: str, genes: pd.DataFrame) -> pd.DataFrame:
    """Per gene (genes in genes_v1 with a credible set): CS counts and the gate in the gene window."""
    ds = eqtl_datasets()[ds_id]
    cs = sources.cs_summary(sources.eqtl_cs_variants(ds), ["gene_id", "cs_id"])
    g = genes[["gene_id", "gene_name", "chr", "tss", "split", "block_id", "win_lo", "win_hi"]]
    cs = cs.merge(g, on="gene_id", how="inner", suffixes=("", "_gene"))
    assert (cs["chr"] == cs["chr_gene"]).all(), f"{ds_id}: CS on another chromosome than its gene"
    cs["pure"] = cs["cs_min_r2"] >= CS_MIN_R2
    cs["lead_in_window"] = (cs["lead_pos"] >= cs["win_lo"]) & (cs["lead_pos"] <= cs["win_hi"])
    cs["gate_cs"] = cs["pure"] & cs["lead_in_window"] & (cs["lead_p"] < ALPHA_QTL)
    agg = cs.groupby("gene_id").agg(
        n_cs=("cs_id", "size"), n_cs_pure=("pure", "sum"), n_cs_gate=("gate_cs", "sum"))
    best = cs[cs["gate_cs"]].sort_values("lead_p").groupby("gene_id").head(1).set_index("gene_id")
    out = g.set_index("gene_id").join(agg, how="inner")
    out["gate"] = out["n_cs_gate"] >= 1
    for c in ("lead_variant", "lead_snv", "snv_by_pip", "lead_p", "lead_z", "cs_id", "members", "n_members", "cs_min_r2"):
        out[f"top_{c}"] = best[c]
    out["dataset"] = ds_id
    return out.reset_index()


def finngen_cs_table(pheno: str) -> pd.DataFrame:
    s = sources.finngen_cs_summary(finngen_traits()[pheno])
    s["pure"] = ~s["low_purity"]
    return s


# --------------------------------------------------------------------------------------------------
# R2
# --------------------------------------------------------------------------------------------------
def build_r2(gates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    groups = defaultdict(list)
    for ds in eqtl_datasets().values():
        if ds.r2_group:
            groups[ds.r2_group].append(ds.dataset_id)
    rows = []
    for grp, members in groups.items():
        for a, b in itertools.combinations(members, 2):
            if any({a, b} <= s for s in SAME_DONOR_POOLS):
                continue
            ga, gb = gates[a], gates[b]
            m = ga[ga["gate"]].merge(gb[gb["gate"]], on="gene_id", suffixes=("_1", "_2"))
            for r in m.itertuples():
                rows.append({
                    "locus_id": f"R2__{a}__{b}__{r.gene_id}", "arm": "R2", "r2_group": grp,
                    "dataset1": a, "dataset2": b, "gene_id": r.gene_id, "gene_name": r.gene_name_1,
                    "split": r.split_1, "chr": r.chr_1, "tss": r.tss_1, "block_id": r.block_id_1,
                    "win_lo": r.win_lo_1, "win_hi": r.win_hi_1,
                    "lead1": r.top_lead_variant_1, "lead1_p": r.top_lead_p_1, "lead1_z": r.top_lead_z_1,
                    "lead2": r.top_lead_variant_2, "lead2_p": r.top_lead_p_2, "lead2_z": r.top_lead_z_2,
                    "n_cs1": r.n_cs_1, "n_cs2": r.n_cs_2,
                    # informational only (NOT used for selection, R1-PI 3):
                    "same_top_lead": _site(r.top_lead_variant_1) == _site(r.top_lead_variant_2),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# R3b
# --------------------------------------------------------------------------------------------------
def _fg_cs_in_windows(fg: pd.DataFrame, fg_members: pd.DataFrame, genes: pd.DataFrame) -> pd.DataFrame:
    """For each gene window: number of FinnGen CSs with >= 1 member in the window (any purity)."""
    out = []
    for c, gg in genes.groupby("chr"):
        mem = fg_members[fg_members["chr"] == c]
        if mem.empty:
            continue
        pos = mem["pos"].to_numpy()
        key = (mem["region"] + "#" + mem["cs"].astype(str)).to_numpy()
        order = np.argsort(pos)
        pos, key = pos[order], key[order]
        for r in gg.itertuples():
            lo, hi = np.searchsorted(pos, r.win_lo, "left"), np.searchsorted(pos, r.win_hi, "right")
            ks = sorted(set(key[lo:hi]))
            if ks:
                out.append({"gene_id": r.gene_id, "fg_cs_keys": ks})
    return pd.DataFrame(out, columns=["gene_id", "fg_cs_keys"])


def r3b_prefilter(gates: dict[str, pd.DataFrame], genes: pd.DataFrame) -> pd.DataFrame:
    """R3b candidates before the LD criteria."""
    rows = []
    for pheno, t in finngen_traits().items():
        fg = finngen_cs_table(pheno)
        fg["key"] = fg["region"] + "#" + fg["cs"].astype(str)
        fgi = fg.set_index("key")
        mem = sources.finngen_cs_variants(t)
        win = _fg_cs_in_windows(fg, mem, genes)
        win = win[win["fg_cs_keys"].map(len) == 1]
        win["key"] = win["fg_cs_keys"].map(lambda k: k[0])
        for ds_id, g in gates.items():
            e = g[(g["n_cs"] == 1) & g["gate"]]
            m = e.merge(win[["gene_id", "key"]], on="gene_id")
            for r in m.itertuples():
                f = fgi.loc[r.key]
                if not (f["pure"] and r.win_lo <= f["lead_pos"] <= r.win_hi and f["lead_p"] < ALPHA_GWAS):
                    continue
                s1 = {_site(v) for v in f["members"].split(";")}
                s2 = {_site(v) for v in r.top_members.split(";")}
                if s1 & s2:
                    continue
                rows.append({
                    "locus_id": f"R3b__FG_{pheno}__{ds_id}__{r.gene_id}", "arm": "R3b",
                    "dataset1": f"FG_{pheno}", "dataset2": ds_id, "gene_id": r.gene_id, "gene_name": r.gene_name,
                    "split": r.split, "chr": r.chr, "tss": r.tss, "block_id": r.block_id,
                    "win_lo": r.win_lo, "win_hi": r.win_hi,
                    "lead1": f["lead_variant"], "lead1_p": f["lead_p"], "lead1_z": f["lead_z"],
                    "lead2": r.top_lead_variant, "lead2_p": r.top_lead_p, "lead2_z": r.top_lead_z,
                    "lead1_snv": f["lead_snv"], "lead2_snv": r.top_lead_snv,
                    "cs1_snv_by_pip": f["snv_by_pip"], "cs2_snv_by_pip": r.top_snv_by_pip,
                    "fg_region": f["region"], "fg_cs": int(f["cs"]), "cs1_size": int(f["n_members"]),
                    "cs2_size": int(r.top_n_members), "cs1_members": f["members"], "cs2_members": r.top_members,
                })
    return pd.DataFrame(rows)


def _poolb_lead_r2(args):
    """Pool-B r^2 between the two CS 'LD leads': the highest-PIP SNV member of each CS that is present
    (and polymorphic) in the QC'd 1000G panel. Returns (r2, ld_lead1, ld_lead2); NaN if none present."""
    chrom, l1, l2 = args
    if not (isinstance(l1, str) and isinstance(l2, str)):
        return float("nan"), "", ""  # a CS without any SNV member
    from data.genotypes import load_genotypes
    from data.real.ld import pools, standardize
    m1, m2 = l1.split(";"), l2.split(";")
    poss = [int(variant_key(v).split(":")[1]) for v in m1 + m2]
    G, vt = load_genotypes(chrom, min(poss) - 1, max(poss), pools()["B"])  # BED convention
    k2c = {_site(v): j for j, v in enumerate(vt["variant_id"])}
    X, _, sd = standardize(G)
    def first(ms):
        for v in ms:
            j = k2c.get(_site(v))
            if j is not None and sd[j] > 0:
                return v, j
        return "", None
    v1, j1 = first(m1)
    v2, j2 = first(m2)
    if j1 is None or j2 is None:
        return float("nan"), v1, v2
    r = float(X[:, j1] @ X[:, j2] / X.shape[0])
    return r * r, v1, v2


def _cs_max_r2(args):
    """Max r^2 (pool B) over all cross pairs of CS members present in the panel (label-noise guard)."""
    chrom, m1, m2 = args
    from data.genotypes import load_genotypes
    from data.real.ld import pools, standardize
    ids = m1 + m2
    poss = [int(variant_key(v).split(":")[1]) for v in ids]
    G, vt = load_genotypes(chrom, min(poss) - 1, max(poss), pools()["B"])  # BED convention
    k2c = {_site(v): j for j, v in enumerate(vt["variant_id"])}
    X, _, sd = standardize(G)
    i1 = [k2c[_site(v)] for v in m1 if _site(v) in k2c and sd[k2c[_site(v)]] > 0]
    i2 = [k2c[_site(v)] for v in m2 if _site(v) in k2c and sd[k2c[_site(v)]] > 0]
    if not i1 or not i2:
        return float("nan")
    r = X[:, i1].T @ X[:, i2] / X.shape[0]
    return float((r ** 2).max())


def _finngen_lookup(pairs):
    from data.real.ld import finngen_r2_cached
    return finngen_r2_cached(pairs)


def r3b_ld_filter(pre: pd.DataFrame, workers: int = 16, poolb_fn=_poolb_lead_r2, csmax_fn=_cs_max_r2,
                  finngen_fn=_finngen_lookup) -> pd.DataFrame:
    """Adds lead_r2_poolB, max_cs_r2_poolB, lead_r2_finngen, finngen_ld_status, pass_ld.
    LD leads = highest-PIP SNV member of each CS present in the pool-B panel (ld_lead1/ld_lead2).
    pass_ld = LD-lead r^2 < 0.1 in pool B AND in FinnGen LD (NaN = not computable -> fail).
    workers = 0 runs serially (tests inject toy functions)."""
    if pre.empty:
        return pre.assign(lead_r2_poolB=[], lead_r2_finngen=[], finngen_ld_status=[], max_cs_r2_poolB=[], pass_ld=[])
    pre = pre.copy()
    a1 = [(r.chr, r.cs1_snv_by_pip, r.cs2_snv_by_pip) for r in pre.itertuples()]
    a2 = [(r.chr, r.cs1_members.split(";"), r.cs2_members.split(";")) for r in pre.itertuples()]
    if workers:
        with ProcessPoolExecutor(workers) as ex:
            lr = list(ex.map(poolb_fn, a1, chunksize=8))
            pre["max_cs_r2_poolB"] = list(ex.map(csmax_fn, a2, chunksize=8))
    else:
        lr = [poolb_fn(x) for x in a1]
        pre["max_cs_r2_poolB"] = [csmax_fn(x) for x in a2]
    pre["lead_r2_poolB"] = [x[0] for x in lr]
    pre["ld_lead1"] = [x[1] for x in lr]
    pre["ld_lead2"] = [x[2] for x in lr]
    cand = (pre["lead_r2_poolB"] < R3B_MAX_LEAD_R2).to_numpy()  # NaN -> False
    fg = [(float("nan"), "not_queried")] * len(pre)
    idx = np.flatnonzero(cand)
    res = finngen_fn([(pre["ld_lead1"].iloc[i], pre["ld_lead2"].iloc[i]) for i in idx])
    for i, v in zip(idx, res):
        fg[i] = v
    pre["lead_r2_finngen"] = [x[0] for x in fg]
    pre["finngen_ld_status"] = [x[1] for x in fg]
    pre["pass_ld"] = cand & (pre["lead_r2_finngen"] < R3B_MAX_LEAD_R2).to_numpy()
    return pre


# --------------------------------------------------------------------------------------------------
# R4b
# --------------------------------------------------------------------------------------------------
def build_r4b(gates: dict[str, pd.DataFrame], genes: pd.DataFrame) -> pd.DataFrame:
    g = gates[GEUVADIS]
    g = g[g["gate"]]
    rows = []
    for pheno in finngen_traits():
        fg = finngen_cs_table(pheno)
        fg = fg[fg["pure"] & (fg["lead_p"] < ALPHA_GWAS)]
        for c, gg in g.groupby("chr"):
            f = fg[fg["chr"] == c]
            if f.empty:
                continue
            for r in gg.itertuples():
                hit = f[(f["lead_pos"] >= r.win_lo) & (f["lead_pos"] <= r.win_hi)].sort_values("lead_p")
                if hit.empty:
                    continue
                h = hit.iloc[0]
                rows.append({"locus_id": f"R4b__FG_{pheno}__{GEUVADIS}__{r.gene_id}", "arm": "R4b",
                             "dataset1": f"FG_{pheno}", "dataset2": GEUVADIS, "gene_id": r.gene_id,
                             "gene_name": r.gene_name, "split": r.split, "chr": r.chr, "tss": r.tss,
                             "block_id": r.block_id, "win_lo": r.win_lo, "win_hi": r.win_hi,
                             "lead1": h["lead_variant"], "lead1_p": h["lead_p"], "lead1_z": h["lead_z"],
                             "lead2": r.top_lead_variant, "lead2_p": r.top_lead_p, "lead2_z": r.top_lead_z,
                             "n_fg_cs_in_window": len(hit)})
    t = gates[TWINSUK_LCL]
    m = g.merge(t[t["gate"]], on="gene_id", suffixes=("_1", "_2"))
    for r in m.itertuples():
        rows.append({"locus_id": f"R4b__{TWINSUK_LCL}__{GEUVADIS}__{r.gene_id}", "arm": "R4b",
                     "dataset1": TWINSUK_LCL, "dataset2": GEUVADIS, "gene_id": r.gene_id, "gene_name": r.gene_name_1,
                     "split": r.split_1, "chr": r.chr_1, "tss": r.tss_1, "block_id": r.block_id_1,
                     "win_lo": r.win_lo_1, "win_hi": r.win_hi_1,
                     "lead1": r.top_lead_variant_2, "lead1_p": r.top_lead_p_2, "lead1_z": r.top_lead_z_2,
                     "lead2": r.top_lead_variant_1, "lead2_p": r.top_lead_p_1, "lead2_z": r.top_lead_z_1})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# R1 realism summaries (train split only)
# --------------------------------------------------------------------------------------------------
def _q(x) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    qs = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(len(x)), "mean": float(x.mean()), **{f"q{int(q * 100):02d}": float(v) for q, v in
                                                         zip([0.05, 0.25, 0.5, 0.75, 0.95], qs)}, "max": float(x.max())}


def z_from_p(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-320, 1.0)
    return stats.norm.isf(p / 2.0)


def r1_summaries(gates: dict[str, pd.DataFrame], genes: pd.DataFrame) -> dict:
    train_genes = genes[genes["split"] == "train"]
    out = {"eqtl": {}, "finngen": {}}
    for ds_id, ds in eqtl_datasets().items():
        perm = sources.eqtl_permuted(ds).merge(train_genes[["gene_id", "win_lo", "win_hi"]], on="gene_id")
        g = gates[ds_id]
        g = g[g["split"] == "train"]
        n_tested = len(perm)
        ncs = g.set_index("gene_id")["n_cs"].reindex(perm["gene_id"]).fillna(0).to_numpy()
        lead_abs_z = z_from_p(perm["lead_p"])
        sig = perm["lead_p"].to_numpy() < ALPHA_QTL
        out["eqtl"][ds_id] = {
            "label": ds.label, "n_genes_tested_train": n_tested,
            "frac_lead_p_lt_1e-5": float(sig.mean()) if n_tested else float("nan"),
            "frac_gate_pass": float(g["gate"].sum() / n_tested) if n_tested else float("nan"),
            "lead_abs_z_all_tested": _q(lead_abs_z),
            "lead_abs_z_gate_genes": _q(np.abs(g.loc[g["gate"], "top_lead_z"])),
            "n_cs_per_tested_gene": {str(k): int(v) for k, v in zip(*np.unique(ncs.astype(int), return_counts=True))},
            "n_cs_per_gate_gene": {str(k): int(v) for k, v in
                                   zip(*np.unique(g.loc[g["gate"], "n_cs"].astype(int), return_counts=True))},
            "cs_size_top": _q(g.loc[g["gate"], "top_n_members"]),
        }
    # FinnGen CSs are assigned to the split of the block containing their lead (same rule as genes)
    from data.make_splits import locate
    from data.real.regions import load_blocks
    b = load_blocks()
    for pheno in finngen_traits():
        fg = finngen_cs_table(pheno)
        sp = []
        for c, d in fg.groupby("chr"):
            if c not in set(b["chr"]):  # chrX etc.: no blocks (autosomes only)
                sp.append(pd.Series("none", index=d.index))
                continue
            ii = locate(b, c, d["lead_pos"].to_numpy())
            sp.append(pd.Series(np.where(ii >= 0, b["split"].to_numpy()[np.maximum(ii, 0)], "none"), index=d.index))
        fg["split"] = pd.concat(sp)
        t = fg[fg["split"] == "train"]
        per_region = t.groupby("region")["cs"].size()
        out["finngen"][pheno] = {
            "n_cs_train": int(len(t)), "n_cs_pure_train": int(t["pure"].sum()),
            "lead_abs_z_pure_cs": _q(np.abs(t.loc[t["pure"], "lead_z"])),
            "frac_pure_cs_lead_p_lt_5e-8": float((t.loc[t["pure"], "lead_p"] < ALPHA_GWAS).mean()) if len(t) else float("nan"),
            "n_cs_per_region": {str(k): int(v) for k, v in zip(*np.unique(per_region.to_numpy(), return_counts=True))},
            "cs_size_pure": _q(t.loc[t["pure"], "n_members"]),
        }
    return out


# --------------------------------------------------------------------------------------------------
def _gate_job(ds_id):
    return ds_id, eqtl_gene_cs(ds_id, gene_table())


def write_arm(df: pd.DataFrame, arm: str) -> dict:
    PAIRS.mkdir(parents=True, exist_ok=True)
    tv = df[df["split"].isin(["train", "valid"])]
    te = df[df["split"] == "test"]
    tv.to_csv(PAIRS / f"{arm}_candidates.tsv", sep="\t", index=False)
    te[[c for c in TEST_ID_COLS if c in te.columns]].to_csv(PAIRS / f"{arm}_test_ids.tsv", sep="\t", index=False)
    return {s: int((df["split"] == s).sum()) for s in ("train", "valid", "test")}


def main() -> None:
    assert_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    genes = gene_table()
    with ProcessPoolExecutor(min(a.workers, len(eqtl_datasets()))) as ex:
        gates = dict(ex.map(_gate_job, list(eqtl_datasets())))
    PAIRS.mkdir(parents=True, exist_ok=True)
    gsum = {d: {"n_genes_with_cs": int(len(g)), **{f"gate_{s}": int(((g["split"] == s) & g["gate"]).sum())
                                                     for s in ("train", "valid", "test")}} for d, g in gates.items()}
    counts = {"gate": gsum}
    r2 = build_r2(gates)
    counts["R2"] = write_arm(r2, "R2")
    counts["R2_by_pair"] = {f"{k[0]}x{k[1]}": {s: int(v) for s, v in grp["split"].value_counts().items()}
                            for k, grp in r2.groupby(["dataset1", "dataset2"])}
    pre = r3b_prefilter(gates, genes)
    counts["R3b_prefilter"] = {s: int((pre["split"] == s).sum()) for s in ("train", "valid", "test")}
    r3 = r3b_ld_filter(pre, a.workers)
    r3p = r3[r3["pass_ld"]].drop(columns=["pass_ld"])
    counts["R3b"] = write_arm(r3p, "R3b")
    counts["R3b_ld_status"] = {k: int(v) for k, v in r3["finngen_ld_status"].value_counts().items()}
    counts["R3b_poolB_nan"] = int(r3["lead_r2_poolB"].isna().sum())
    r4 = build_r4b(gates, genes)
    counts["R4b"] = write_arm(r4, "R4b")
    counts["R1"] = r1_summaries(gates, genes)
    (PAIRS / "counts.json").write_text(json.dumps(counts, indent=1, default=float))
    print(json.dumps({k: v for k, v in counts.items() if k != "R1"}, indent=1, default=float))


if __name__ == "__main__":
    main()
