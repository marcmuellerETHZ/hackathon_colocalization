"""P3-1 rho-hat for the real locus sets (real_locus_sets@2), train and valid splits only.

  python -m data.real.rho_update --workers 32

For every locus of every R2/R3b/R4b train/valid file (each R4b LD mode separately):
  * PRIMARY (P3-1): per-window rho-hat = `stats.rho.rho_hat_window` (README D14 of src/sim) with the LD stored in
    that file (= the LD given to methods), the same settings as the simulator's per-window mode
    (`sim.generator.rho_hat_for`): mask max|z| > 4 + r² > 0.1, pruning r² < 0.1 in a random order
    (seed derive(VALID_SEED, "real_rho_prune_order", locus_id), so the R4b modes of one pair share the order),
    |z| < 2, unit-variance MLE, 10-block jackknife. `residualize_signal=False` everywhere: the simulator's
    rule is "L0 in-sample (both traits) or n_ref >= 1000"; R2/R3b use pool B (n = 150), R4b reference LD
    n = 138, and R4b in-sample LD (n = 445) is in-sample for trait 2 only.
    -> attrs rho_hat, rho_se (+ rho_n_used, rho_n_masked, rho_n_pruned, rho_method).
  * SENSITIVITY (P3-1): per cohort pair (arm, LD mode, dataset1, dataset2), the pooled estimate of
    `stats.rho.rho_hat_pooled` over all TRAIN windows of that pair (signal mask per window with its LD, no
    pruning, |z| < 2, unit-variance MLE). Computed from per-window sufficient statistics (identical point
    estimate; tested against rho_hat_pooled). Jackknife unit: clusters of windows that share an LD block
    (windows of neighbouring genes overlap and, for FinnGen traits, share z), so delete-one-cluster;
    the windows-as-blocks SE of rho_hat_pooled is stored too (rho_se_pair_win). The train estimate is
    written to the pair's train AND valid loci.
    -> attrs rho_hat_pair, rho_se_pair, rho_se_pair_win, rho_pair_n_windows, rho_pair_n_clusters.
  * Locus metadata: trait1_type/trait2_type in the simulator's convention ("gwas"/"mol", src/sim/params.py
    ALPHA keys); ld_role (R4b: geuv_insample_pcadj = "primary_insample", geuv_insample_raw =
    "sensitivity_insample", reference panels = "reference"); boot_unit (R3b: FinnGen credible-set id
    "<trait>|<fg_region>|cs<fg_cs>"; R2/R4b: the candidate's LD block id, DESIGN default) and fg_cs_id (R3b).
Writes attributes in place (HDF5 r+) plus `locus_sets/rho_v2.parquet` (one row per file x locus, with the
previous rho_hat_v1/rho_se_v1), `locus_sets/rho_v2_pairs.tsv` and `locus_sets/rho_v2_summary.json`.
Refuses any locus whose split is not train/valid or that touches a test block (eval.io guard).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from common.seeds import VALID_SEED, rng
from data.real.config import OUT, assert_env
from stats.rho import C_TRUNC, MIN_N, PRUNE_R2, jackknife_from_blocks, mle_from_stats, rho_hat_window, signal_mask

LOCUS_DIR = OUT / "locus_sets"
PAIRS = OUT / "pairs"
ARM_MODES = {"R2": ["poolB"], "R3b": ["poolB"],
             "R4b": ["nfe_minus_geuv", "geuv_insample_pcadj", "geuv_insample_raw"]}
LD_ROLE = {"poolB": "reference", "nfe_minus_geuv": "reference",
           "geuv_insample_pcadj": "primary_insample", "geuv_insample_raw": "sensitivity_insample"}
RHO_METHOD = "P3-1 per-window rho_hat_window (sim D14 settings, residualize_signal=False)"


def trait_type(source: str) -> str:
    """Simulator convention (src/sim/params.py ALPHA keys): FinnGen = GWAS-like, eQTL = molecular."""
    return "gwas" if str(source).startswith("FG_") else "mol"


def locus_file(arm: str, split: str, mode: str, locus_dir: Path = LOCUS_DIR) -> Path:
    return locus_dir / (f"{arm}_{split}.h5" if len(ARM_MODES[arm]) == 1 else f"{arm}_{split}.{mode}.h5")


def prune_order(locus_id: str, p: int) -> np.ndarray:
    return rng(VALID_SEED, "real_rho_prune_order", locus_id).permutation(p)


def window_rho(z1, z2, R, locus_id: str, c: float = C_TRUNC) -> dict:
    """Per-window rho-hat (primary) + the window's pooled-mode sufficient statistics (mask, no pruning)."""
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    rh = rho_hat_window(z1, z2, R, PRUNE_R2, c, prune_order(locus_id, len(z1)), residualize_signal=False)
    mask = signal_mask(z1, z2, R)
    use = (~mask) & (np.abs(z1) < c) & (np.abs(z2) < c) & np.isfinite(z1) & np.isfinite(z2)
    return {"rho_hat": float(rh["rho_hat"]), "rho_se": float(rh["rho_se"]), "rho_n_used": int(rh["n_used"]),
            "rho_n_masked": int(rh["n_masked"]), "rho_n_pruned": int(rh["n_pruned"]),
            "pool_n": float(use.sum()), "pool_sqq": float(np.sum(z1[use] ** 2 + z2[use] ** 2)),
            "pool_s12": float(np.sum(z1[use] * z2[use]))}


def ld_clusters(block_lists) -> np.ndarray:
    """Cluster label per window: windows sharing any LD block id are in one cluster (union-find)."""
    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, bl in enumerate(block_lists):
        w = ("w", k)
        find(w)
        for b in bl:
            ra, rb = find(w), find(("b", b))
            if ra != rb:
                parent[ra] = rb
    roots = [find(("w", k)) for k in range(len(block_lists))]
    _, lab = np.unique([repr(r) for r in roots], return_inverse=True)
    return lab


def pooled_from_stats(n, sqq, s12, labels=None, c: float = C_TRUNC, min_n: int = MIN_N,
                      min_windows_as_blocks: int = 10) -> dict:
    """rho_hat_pooled from per-window sufficient statistics. Jackknife over `labels` (clusters of windows) and
    over windows (= rho_hat_pooled's SE when >= min_windows_as_blocks windows)."""
    n, sqq, s12 = (np.asarray(a, dtype=np.float64) for a in (n, sqq, s12))
    out = {"n_windows": int(len(n)), "n_used": int(n.sum())}
    if n.sum() < min_n:
        return {**out, "rho_hat": float("nan"), "rho_se": float("nan"), "rho_se_win": float("nan"), "n_clusters": 0}
    full = float(mle_from_stats(n.sum(), sqq.sum(), s12.sum(), c))
    _, se_win = jackknife_from_blocks(n, sqq, s12, c) if len(n) >= min_windows_as_blocks else (None, float("nan"))
    labels = np.arange(len(n)) if labels is None else np.asarray(labels)
    _, lab = np.unique(labels, return_inverse=True)
    nb = int(lab.max()) + 1
    agg = [np.bincount(lab, weights=a, minlength=nb) for a in (n, sqq, s12)]
    _, se = jackknife_from_blocks(*agg, c)
    return {**out, "rho_hat": full, "rho_se": float(se), "rho_se_win": float(se_win), "n_clusters": nb}


# ------------------------------------------------------------------------------------------- pass 1 (read only)
def _chunk_job(args) -> list[dict]:
    from eval.io import LocusSetReader
    path, ids = args
    rows = []
    with LocusSetReader(path, guard=True) as r:
        for lid in ids:
            L = r.read(lid)
            if L.split not in ("train", "valid"):
                raise PermissionError(f"{lid}: split {L.split!r} (only train/valid)")
            res = window_rho(L.z1, L.z2, L.ld(), lid)
            a = L.extra_attrs
            rows.append({"locus_id": lid, "split": L.split, "dataset1": a["dataset1"], "dataset2": a["dataset2"],
                         "block_ids": "|".join(L.block_ids), "p": L.p, "n_ref": int(L.n_ref),
                         "rho_hat_file": float(L.rho_hat), "rho_method_file": str(a.get("rho_method", "")), **res})
    return rows


def compute(files: list[tuple[str, str, str, Path]], workers: int, chunk: int = 50) -> pd.DataFrame:
    from eval.io import list_loci
    jobs, meta = [], []
    for arm, split, mode, path in files:
        ids = list_loci(path)
        for i in range(0, len(ids), chunk):
            jobs.append((str(path), ids[i:i + chunk]))
            meta.append((arm, split, mode, path.name))
    frames = []
    with ProcessPoolExecutor(workers) as ex:
        for (arm, split, mode, fname), rows in zip(meta, ex.map(_chunk_job, jobs)):
            d = pd.DataFrame(rows)
            d.insert(0, "file", fname)
            d.insert(0, "mode", mode)
            d.insert(0, "arm", arm)
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def v1_values(arm: str, split: str, locus_dir: Path = LOCUS_DIR) -> pd.DataFrame:
    """The M3 (v1) rho-hat per locus from <arm>_<split>.extract_log.tsv (shared by all LD modes of a pair):
    the 'before' values, independent of whether the HDF5 attributes were already rewritten."""
    lg = pd.read_csv(locus_dir / f"{arm}_{split}.extract_log.tsv", sep="\t", usecols=["locus_id", "status", "rho_hat", "rho_se"])
    lg = lg[lg["status"] == "ok"].drop_duplicates("locus_id")
    return lg.rename(columns={"rho_hat": "rho_hat_v1", "rho_se": "rho_se_v1"})[["locus_id", "rho_hat_v1", "rho_se_v1"]]


def pair_estimates(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled estimate per (arm, mode, dataset1, dataset2) from its TRAIN windows."""
    out = []
    for (arm, mode, d1, d2), g in df[df["split"] == "train"].groupby(["arm", "mode", "dataset1", "dataset2"]):
        lab = ld_clusters([b.split("|") if b else [] for b in g["block_ids"]])
        res = pooled_from_stats(g["pool_n"], g["pool_sqq"], g["pool_s12"], lab)
        out.append({"arm": arm, "mode": mode, "dataset1": d1, "dataset2": d2, **res})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------- pass 2 (write attrs)
def locus_meta(arm: str, pairs_dir: Path = PAIRS) -> pd.DataFrame:
    cols = ["locus_id", "dataset1", "block_id", "split"] + (["fg_region", "fg_cs"] if arm == "R3b" else [])
    c = pd.read_csv(pairs_dir / f"{arm}_candidates.tsv", sep="\t", dtype=str, usecols=cols)
    c = c[c["split"].isin(["train", "valid"])]
    if arm == "R3b":
        c["fg_cs_id"] = c["dataset1"] + "|" + c["fg_region"] + "|cs" + c["fg_cs"]
        c["boot_unit"] = c["fg_cs_id"]
    else:
        c["boot_unit"] = c["block_id"]
    return c.set_index("locus_id")


def write_attrs(path: Path, mode: str, rows: pd.DataFrame, pairs: pd.DataFrame, meta: pd.DataFrame) -> int:
    from eval.io import guard_locus
    pr = pairs.set_index(["dataset1", "dataset2"])
    rows = rows.set_index("locus_id")
    n = 0
    with h5py.File(path, "r+") as f:
        for lid in f.keys():
            if lid.startswith("_"):
                continue
            g = f[lid]
            split = g.attrs["split"]
            split = split.decode() if isinstance(split, bytes) else str(split)
            blocks = tuple(x.decode() if isinstance(x, bytes) else str(x) for x in np.atleast_1d(g.attrs["block_ids"]))
            guard_locus(lid, split, blocks, str(g.attrs["window_chr"]), int(g.attrs["window_start"]),
                        int(g.attrs["window_end"]))
            if split not in ("train", "valid"):
                raise PermissionError(f"{lid}: split {split!r}")
            r = rows.loc[lid]
            d1, d2 = str(g.attrs["dataset1"]), str(g.attrs["dataset2"])
            pe = pr.loc[(d1, d2)] if (d1, d2) in pr.index else None
            g.attrs["rho_hat"] = float(r["rho_hat"])
            g.attrs["rho_se"] = float(r["rho_se"])
            g.attrs["rho_n_used"] = int(r["rho_n_used"])
            g.attrs["rho_n_masked"] = int(r["rho_n_masked"])
            g.attrs["rho_n_pruned"] = int(r["rho_n_pruned"])
            g.attrs["rho_method"] = RHO_METHOD
            g.attrs["rho_hat_pair"] = float(pe["rho_hat"]) if pe is not None else float("nan")
            g.attrs["rho_se_pair"] = float(pe["rho_se"]) if pe is not None else float("nan")
            g.attrs["rho_se_pair_win"] = float(pe["rho_se_win"]) if pe is not None else float("nan")
            g.attrs["rho_pair_n_windows"] = int(pe["n_windows"]) if pe is not None else 0
            g.attrs["rho_pair_n_clusters"] = int(pe["n_clusters"]) if pe is not None else 0
            g.attrs["trait1_type"] = trait_type(d1)
            g.attrs["trait2_type"] = trait_type(d2)
            g.attrs["ld_role"] = LD_ROLE[mode]
            m = meta.loc[lid]
            g.attrs["boot_unit"] = str(m["boot_unit"])
            if "fg_cs_id" in meta.columns:
                g.attrs["fg_cs_id"] = str(m["fg_cs_id"])
            n += 1
    return n


def _q(x: pd.Series) -> dict:
    x = x.dropna()
    return {"n": int(len(x)), "median": float(x.median()), "q25": float(x.quantile(.25)),
            "q75": float(x.quantile(.75)), "iqr": float(x.quantile(.75) - x.quantile(.25)), "mean": float(x.mean()),
            "sd": float(x.std())}


def summarise(df: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    s = {"per_arm_mode": {}, "per_arm_mode_split": {}, "per_cohort_pair": [], "pairs": pairs.to_dict("records")}
    for (arm, mode), g in df.groupby(["arm", "mode"]):
        s["per_arm_mode"][f"{arm}/{mode}"] = {
            "before_rho_hat_v1": _q(g["rho_hat_v1"]), "after_rho_hat": _q(g["rho_hat"]),
            "after_rho_se": _q(g["rho_se"]), "before_rho_se_v1": _q(g["rho_se_v1"]),
            "after_rho_hat_pair": _q(g["rho_hat_pair"]), "n_nan_after": int(g["rho_hat"].isna().sum()),
            "n_nan_before": int(g["rho_hat_v1"].isna().sum()),
            "z_window_mean": float((g["rho_hat"] / g["rho_se"]).replace([np.inf, -np.inf], np.nan).mean())}
        for split, h in g.groupby("split"):
            s["per_arm_mode_split"][f"{arm}/{mode}/{split}"] = {"before": _q(h["rho_hat_v1"]), "after": _q(h["rho_hat"])}
    for (arm, mode, d1, d2), g in df.groupby(["arm", "mode", "dataset1", "dataset2"]):
        s["per_cohort_pair"].append({"arm": arm, "mode": mode, "dataset1": d1, "dataset2": d2, "n": int(len(g)),
                                     "median_before": float(g["rho_hat_v1"].median()),
                                     "median_after": float(g["rho_hat"].median()),
                                     "rho_hat_pair": float(g["rho_hat_pair"].iloc[0]),
                                     "rho_se_pair": float(g["rho_se_pair"].iloc[0])})
    return s


def run(arms=("R2", "R3b", "R4b"), splits=("train", "valid"), workers: int = 32, locus_dir: Path = LOCUS_DIR,
        pairs_dir: Path = PAIRS, write: bool = True) -> dict:
    for s in splits:
        if s not in ("train", "valid"):
            raise PermissionError("only train/valid locus sets (test lock, DESIGN §7.3)")
    files = [(a, s, m, locus_file(a, s, m, locus_dir)) for a in arms for m in ARM_MODES[a] for s in splits]
    t0 = time.time()
    df = compute(files, workers)
    t1 = time.time()
    v1 = pd.concat([v1_values(a, s, locus_dir).assign(arm=a, split=s) for a in arms for s in splits])
    df = df.merge(v1, on=["arm", "split", "locus_id"], how="left", validate="many_to_one")
    fresh = df["rho_method_file"] == ""          # attributes not yet rewritten: file value must equal the log's
    v1_mismatch = int((~np.isclose(df.loc[fresh, "rho_hat_file"], df.loc[fresh, "rho_hat_v1"], equal_nan=True)).sum())
    pairs = pair_estimates(df)
    key = ["arm", "mode", "dataset1", "dataset2"]
    df = df.merge(pairs[key + ["rho_hat", "rho_se", "rho_se_win", "n_windows", "n_clusters"]].rename(columns={
        "rho_hat": "rho_hat_pair", "rho_se": "rho_se_pair", "rho_se_win": "rho_se_pair_win",
        "n_windows": "rho_pair_n_windows", "n_clusters": "rho_pair_n_clusters"}), on=key, how="left")
    n_written = {}
    if write:
        metas = {a: locus_meta(a, pairs_dir) for a in arms}
        for a, s, m, path in files:
            sub = df[(df["arm"] == a) & (df["mode"] == m) & (df["split"] == s)]
            n_written[path.name] = write_attrs(path, m, sub, pairs[(pairs["arm"] == a) & (pairs["mode"] == m)], metas[a])
            assert n_written[path.name] == len(sub), (path, n_written[path.name], len(sub))
        df.to_parquet(locus_dir / "rho_v2.parquet", index=False)
        pairs.to_csv(locus_dir / "rho_v2_pairs.tsv", sep="\t", index=False)
    summ = summarise(df, pairs)
    summ.update(n_written=n_written, n_rows=int(len(df)), n_fresh=int(fresh.sum()), v1_file_vs_log_mismatch=v1_mismatch, compute_s=round(t1 - t0, 1), total_s=round(time.time() - t0, 1),
                rho_method=RHO_METHOD)
    if write:
        (locus_dir / "rho_v2_summary.json").write_text(json.dumps(summ, indent=1))
    return summ


def main() -> None:
    assert_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["R2", "R3b", "R4b"])
    ap.add_argument("--splits", nargs="+", default=["train", "valid"])
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    s = run(a.arms, a.splits, a.workers)
    print(json.dumps({k: s[k] for k in ("per_arm_mode", "n_written", "compute_s", "total_s")}, indent=1))


if __name__ == "__main__":
    main()
