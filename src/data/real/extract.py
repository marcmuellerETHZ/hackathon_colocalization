"""Extract harmonised real locus pairs into INTERFACES §2 locus sets (train and valid splits ONLY).

For each candidate pair (pairs.py output) and LD mode:
  1. reference variants + genotypes in the gene window from the QC'd 1000G tables (data.genotypes);
  2. both traits' summary statistics in the window (tabix), harmonised to the reference (harmonise.py,
     AF concordance always against pool B, DESIGN §2.5);
  3. variant set = SNVs present in both traits and in the reference with MAF >= 0.01 in every LD panel
     used for this arm (so the R4b in-sample and reference files share one variant set);
  4. z = beta/se after alignment; standardized beta/se as in DESIGN §2.2 (sdY = 1):
     se_std = 1/sqrt(2 n f (1-f)) with f the study's in-sample ALT frequency (pool-B AF if missing), beta_std = z se_std;
  5. LD in factor form: X_ref = standardized dosages of the LD panel (population SD; R = X'X/n_ref).
Writes <arm>_<split>[.<mode>].h5 via src/eval/io.py, a per-variant sidecar parquet (source beta/se/AF,
flags) and a labels file <stem>.truth.parquet (arm label; methods never open it).
Test-split pairs are refused (and io.py's guard refuses them independently).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.real import sources
from data.real.config import MIN_MAF_REF, OUT, assert_env, eqtl_datasets, finngen_sample_sizes, finngen_traits
from data.real.harmonise import harmonise, lead_missingness
from data.real.regions import blocks_in_window
# trait types in the simulator's convention ("gwas"/"mol") and the P3-1 rho-hat: one definition in rho_update
from data.real.rho_update import LD_ROLE, RHO_METHOD, trait_type, window_rho  # noqa: F401 (trait_type re-exported)

LOCUS_DIR = OUT / "locus_sets"
PAIRS = OUT / "pairs"

# LD modes: (panel name, mismatch_level attribute)
MODES = {
    "poolB": "real_ref_poolB",
    "nfe_minus_geuv": "real_ref_nfe_minus_geuvadis",
    "geuv_insample_pcadj": "real_insample_geuvadis_pcadj",
    "geuv_insample_raw": "real_insample_geuvadis_raw",
}
ARM_MODES = {"R2": ["poolB"], "R3b": ["poolB"],
             "R4b": ["nfe_minus_geuv", "geuv_insample_pcadj", "geuv_insample_raw"]}
ARM_LABEL = {"R2": "S_partial_positive", "R3b": "D_pseudo_negative", "R4b": "unlabelled_agreement"}


def panel_samples(mode: str) -> list[str]:
    from data.real import geuvadis
    from data.real.ld import pools
    if mode == "poolB":
        return pools()["B"]
    if mode == "nfe_minus_geuv":
        return geuvadis.nfe_minus_geuvadis()
    if mode.startswith("geuv_insample"):
        return geuvadis.samples()
    raise KeyError(mode)




def load_trait(source: str, gene_id: str, chrom: str, lo: int, hi: int) -> pd.DataFrame:
    if source.startswith("FG_"):
        pheno = source[3:]
        df = sources.finngen_region(finngen_traits()[pheno], chrom, lo, hi)
        ns = finngen_sample_sizes().set_index("pheno")
        df["n"] = float(ns.at[pheno, "n_eff"])
        return df
    return sources.eqtl_gene_region(eqtl_datasets()[source], gene_id, chrom, lo, hi)


@dataclass
class Extracted:
    locus: object
    sidecar: pd.DataFrame
    log: dict


def ld_key(mode: str, variant_ids) -> str:
    """Loci of one file with the same panel and identical variant list share one stored X_ref (io.py hard link)."""
    import hashlib
    h = hashlib.sha1(("|".join([mode, *map(str, variant_ids)])).encode()).hexdigest()
    return f"{mode}_{h[:20]}"


def _std_beta_se(z: np.ndarray, n: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    se = 1.0 / np.sqrt(2.0 * n * f * (1.0 - f))
    return z * se, se


def extract_pair(row: dict, modes: list[str], allow_test: bool = False) -> list[Extracted]:
    """One candidate pair -> one LocusData per LD mode (same variant set across modes).
    Test-split rows only with allow_test=True AND an open eval_test.py nonce (worker check, `_check_test_access`)."""
    from data.genotypes import load_genotypes
    from data.real import geuvadis
    from data.real.ld import pools, standardize
    from eval.io import LocusData

    if row["split"] == "test" and allow_test:
        _check_test_access(in_worker=True)
    elif row["split"] not in ("train", "valid"):
        raise PermissionError(f"{row['locus_id']}: split {row['split']} is not extractable (test lock)")
    chrom, lo, hi = row["chr"], int(row["win_lo"]), int(row["win_hi"])
    # reference variants/genotypes per panel (variant tables identical; samples differ)
    geno, vt = {}, None
    for m in modes:
        G, v = load_genotypes(chrom, lo - 1, hi, panel_samples(m))  # BED: lo-1 < pos <= hi
        if vt is None:
            vt = v
        else:
            assert (v["variant_id"].to_numpy() == vt["variant_id"].to_numpy()).all()
        geno[m] = G
    ref = vt[["chr", "pos", "ref", "alt", "variant_id", "af_B"]]
    t1, log1 = harmonise(load_trait(row["dataset1"], row["gene_id"], chrom, lo, hi), ref, "af_B", "trait1")
    t2, log2 = harmonise(load_trait(row["dataset2"], row["gene_id"], chrom, lo, hi), ref, "af_B", "trait2")
    maf_ok = np.ones(len(vt), dtype=bool)
    for m in modes:
        af = geno[m].mean(axis=0) / 2.0
        maf_ok &= np.minimum(af, 1 - af) >= MIN_MAF_REF
    keep_ids = set(vt["variant_id"][maf_ok]) & set(t1["variant_id"]) & set(t2["variant_id"])
    sel = vt["variant_id"].isin(keep_ids).to_numpy()
    v = vt[sel].reset_index(drop=True)
    a = t1.set_index("variant_id").loc[v["variant_id"]].reset_index()
    b = t2.set_index("variant_id").loc[v["variant_id"]].reset_index()
    p = len(v)
    log = {"locus_id": row["locus_id"], "p": p, "n_ref_window": int(len(vt)), "n_ref_maf_ok": int(maf_ok.sum()),
           **{f"t1_{k}": val for k, val in log1.items() if k != "label"},
           **{f"t2_{k}": val for k, val in log2.items() if k != "label"},
           **lead_missingness({"trait1": row.get("lead1"), "trait2": row.get("lead2")}, set(v["variant_id"]))}
    if p < 2:
        log["status"] = "too_few_variants"
        return [Extracted(None, pd.DataFrame(), log)]
    af_b = v["af_B"].to_numpy(float)
    out = []
    f1 = a["af"].to_numpy(float)
    f2 = b["af"].to_numpy(float)
    bad1 = ~np.isfinite(f1) | (f1 <= 0) | (f1 >= 1)
    bad2 = ~np.isfinite(f2) | (f2 <= 0) | (f2 >= 1)
    log["t1_af_fallback_poolB"], log["t2_af_fallback_poolB"] = int(bad1.sum()), int(bad2.sum())
    f1 = np.where(bad1, af_b, f1)
    f2 = np.where(bad2, af_b, f2)
    z1, z2 = a["z"].to_numpy(float), b["z"].to_numpy(float)
    n1, n2 = a["n"].to_numpy(float), b["n"].to_numpy(float)
    beta1, se1 = _std_beta_se(z1, n1, f1)
    beta2, se2 = _std_beta_se(z2, n2, f2)
    side = pd.DataFrame({"locus_id": row["locus_id"], "variant_id": v["variant_id"],
                         "beta1_src": a["beta"], "se1_src": a["se"], "p1_src": a["pvalue"], "af1": a["af"],
                         "daf1": a["daf"], "af_flag1": a["af_flag"], "flipped1": a["flipped"],
                         "beta2_src": b["beta"], "se2_src": b["se"], "p2_src": b["pvalue"], "af2": b["af"],
                         "daf2": b["daf"], "af_flag2": b["af_flag"], "flipped2": b["flipped"], "af_poolB": af_b})
    block_ids = blocks_in_window(chrom, lo, hi)
    # rho-hat: P3-1 per-window estimator with each mode's own LD (= the LD given to methods), rho_update.py;
    # the per-cohort-pair pooled attributes (rho_hat_pair, ...) need all train windows: `python -m data.real.rho_update`
    if row["arm"] == "R3b":
        boot = {"fg_cs_id": f"{row['dataset1']}|{row['fg_region']}|cs{row['fg_cs']}"}
        boot["boot_unit"] = boot["fg_cs_id"]
    else:
        boot = {"boot_unit": str(row.get("block_id", ""))}
    for m in modes:
        G = geno[m][:, sel]
        if m == "geuv_insample_pcadj":
            G = geuvadis.residualise(G, geuvadis.pcs())
        X, af_panel, sd = standardize(G)
        if (sd <= 0).any():
            raise AssertionError(f"{row['locus_id']}: monomorphic column after MAF filter ({m})")
        Xf = X.astype(np.float32).astype(np.float64)  # exactly the stored LD
        rh = window_rho(z1, z2, Xf.T @ Xf / Xf.shape[0], row["locus_id"])
        if m == modes[0]:
            log.update({"rho_hat": rh["rho_hat"], "rho_se": rh["rho_se"], "rho_n_used": rh["rho_n_used"]})
        if m == "geuv_insample_pcadj":
            af_panel = geno[m][:, sel].mean(axis=0) / 2.0  # MAF from raw dosages
        maf_ref = np.minimum(af_panel, 1 - af_panel)
        extra = {"arm": row["arm"], "gene_id": row["gene_id"], "gene_name": str(row.get("gene_name", "")),
                 "dataset1": row["dataset1"], "dataset2": row["dataset2"],
                 "trait1_type": trait_type(row["dataset1"]), "trait2_type": trait_type(row["dataset2"]),
                 "lead1": str(row.get("lead1", "")), "lead2": str(row.get("lead2", "")),
                 "lead1_present": bool(log["lead_present_trait1"]), "lead2_present": bool(log["lead_present_trait2"]),
                 "ld_panel": m, "n_af_flag1": int(a["af_flag"].sum()), "n_af_flag2": int(b["af_flag"].sum()),
                 "ld_role": LD_ROLE[m], **boot, "rho_method": RHO_METHOD,
                 **{k: int(rh[k]) for k in ("rho_n_used", "rho_n_masked", "rho_n_pruned")}}
        loc = LocusData(
            locus_id=row["locus_id"], variant_id=v["variant_id"].to_numpy(object), pos=v["pos"].to_numpy(np.int64),
            z1=z1, z2=z2, beta1=beta1, se1=se1, beta2=beta2, se2=se2, n1=n1, n2=n2,
            maf1=np.minimum(f1, 1 - f1), maf2=np.minimum(f2, 1 - f2), maf_ref=maf_ref,
            ld_type="factor", n_ref=int(X.shape[0]), X_ref=X.astype(np.float32),
            rho_hat=float(rh["rho_hat"]), rho_se=float(rh["rho_se"]),  # per-window (P3-1 primary), this mode's LD
            split=row["split"], block_ids=tuple(block_ids), mismatch_level=MODES[m],
            pairing_type="real_gwas_x_eqtl" if trait_type(row["dataset1"]) == "gwas" else "real_eqtl_x_eqtl",
            cell_id=row["arm"], window_chr=chrom, window_start=lo - 1, window_end=hi, extra_attrs=extra,
            ld_key=ld_key(m, v["variant_id"]))
        loc.validate()
        out.append(Extracted(loc, side if m == modes[0] else pd.DataFrame(), {**log, "mode": m, "status": "ok"}))
    return out


def _job(args):
    row, modes, *rest = args
    allow_test = bool(rest[0]) if rest else False
    t0 = time.time()
    try:
        if row["split"] == "test":   # access failures are never swallowed into the per-locus error log
            _check_test_access(in_worker=True)
        res = extract_pair(row, modes, allow_test=allow_test)
    except PermissionError:
        raise
    except Exception as e:  # recorded, reported, never silently dropped
        return [Extracted(None, pd.DataFrame(), {"locus_id": row["locus_id"], "status": f"error: {type(e).__name__}: {e}"})], 0.0
    return res, time.time() - t0


def run_arm(arm: str, split: str, workers: int = 32, limit: int | None = None, compression: str | None = "gzip") -> dict:
    from eval.io import truth_path_for, write_locus_set
    if split not in ("train", "valid"):
        raise PermissionError("only train/valid locus sets may be extracted (test lock, DESIGN §7.3)")
    cand = pd.read_csv(PAIRS / f"{arm}_candidates.tsv", sep="\t", dtype={"chr": str})
    cand = cand[cand["split"] == split]
    if limit:
        cand = cand.head(limit)
    modes = ARM_MODES[arm]
    LOCUS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {m: LOCUS_DIR / (f"{arm}_{split}.h5" if len(modes) == 1 else f"{arm}_{split}.{m}.h5") for m in modes}
    for pth in paths.values():
        if pth.exists():
            pth.unlink()
    logs, sides, n_written = [], [], {m: 0 for m in modes}
    rows = [r._asdict() for r in cand.itertuples(index=False)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for res, dt in ex.map(_job, [(r, modes) for r in rows], chunksize=2):
            for e in res:
                logs.append({**e.log, "extract_s": dt})
                if e.locus is None:
                    continue
                m = e.log["mode"]
                n_written[m] += write_locus_set(paths[m], [e.locus], mode="a", compression=compression)
                if len(e.sidecar):
                    sides.append(e.sidecar)
    lg = pd.DataFrame(logs)
    stem = f"{arm}_{split}"
    lg.to_csv(LOCUS_DIR / f"{stem}.extract_log.tsv", sep="\t", index=False)
    if sides:
        pd.concat(sides, ignore_index=True).to_parquet(LOCUS_DIR / f"{stem}.variants.parquet", index=False)
    ok_ids = set(lg.loc[lg["status"] == "ok", "locus_id"])
    truth = cand[cand["locus_id"].isin(ok_ids)].copy()
    truth["label"] = ARM_LABEL[arm]
    for pth in paths.values():
        truth.to_parquet(truth_path_for(pth), index=False)
    summary = {"arm": arm, "split": split, "n_candidates": int(len(cand)), "n_written": n_written,
               "status_counts": {k: int(v) for k, v in lg.drop_duplicates("locus_id")["status"].str.split(":").str[0].value_counts().items()},
               "wall_s": round(time.time() - t0, 1),
               "p_median": float(lg.loc[lg["status"] == "ok", "p"].median()) if len(ok_ids) else float("nan"),
               "files": {m: str(pth.relative_to(OUT.parents[1])) for m, pth in paths.items()}}
    (LOCUS_DIR / f"{stem}.summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# ================================================================================================
# Test arms (P3-14 / P3-26): the frozen ID lists, built inside the authorised test run only
# ================================================================================================
# eval_test real-set key -> (arm, LD mode). R4b in-sample = PC-adjusted GEUVADIS (D6(i) primary), reference = NFE
# minus GEUVADIS; the raw in-sample file (sensitivity) is built too (identical to train/valid) and returned under
# an extra key that src/eval/testrun.py ignores.
REAL_SET_MODES = {"real_R2": ("R2", "poolB"), "real_R3b": ("R3b", "poolB"),
                  "real_R4b_insample": ("R4b", "geuv_insample_pcadj"),
                  "real_R4b_reference": ("R4b", "nfe_minus_geuv"),
                  "real_R4b_insample_raw": ("R4b", "geuv_insample_raw")}
FROZEN_SUB_LIST = {"R2": "R2_test_ids_sub_v1.tsv", "R3b": "R3b_test_ids_sub_v1.tsv", "R4b": "R4b_test_ids_sub_v1.tsv"}
PAIR_ATTRS = {"rho_hat_pair": float, "rho_se_pair": float, "rho_se_pair_win": float, "rho_pair_n_windows": int,
              "rho_pair_n_clusters": int}


class TestArmError(RuntimeError):
    pass


def _called_from_eval_test() -> bool:
    import inspect
    from pathlib import Path
    for fr in inspect.stack():
        f = Path(fr.filename).resolve()
        if f.name == "eval_test.py" and f.parent.name == "scripts":
            return True
    return False


def _check_test_access(in_worker: bool = False) -> None:
    """Same pattern as sim.testset: the parent needs eval.io.has_test_access() (open nonce of an authorised
    scripts/eval_test.py run at a frozen-* tag) AND scripts/eval_test.py on the call stack; a pool worker (whose
    stack never contains eval_test.py under spawn) needs has_test_access() (nonce inherited via the env)."""
    from eval.io import TestSplitAccessError, has_test_access
    if not (has_test_access() and (in_worker or _called_from_eval_test())):
        raise TestSplitAccessError("real-data test arms are built only inside scripts/eval_test.py (DESIGN §7.3)")


def donor_disjoint(d1: str, d2: str) -> bool:
    """P3-23(b): documented donor-disjoint cohort pair (README 'Sample-overlap status'): distinct datasets that
    are not both in one same-donor pool (config.SAME_DONOR_POOLS). FinnGen x eQTL Catalogue cohorts: disjoint
    (distinct recruitment). Every pair of R2 / R3b / R4b is disjoint by construction (pairs.build_r2 skips
    same-donor pools)."""
    from data.real.config import SAME_DONOR_POOLS
    return d1 != d2 and not any({d1, d2} <= p for p in SAME_DONOR_POOLS)


def pair_rho_table(locus_dir=LOCUS_DIR) -> pd.DataFrame:
    """Per (arm, mode, dataset1, dataset2): the TRAIN pooled rho attributes exactly as rho_update wrote them to
    train/valid loci (from locus_sets/rho_v2.parquet, real_locus_sets@2; exact float64)."""
    d = pd.read_parquet(locus_dir / "rho_v2.parquet", columns=["arm", "mode", "dataset1", "dataset2", *PAIR_ATTRS])
    key = ["arm", "mode", "dataset1", "dataset2"]
    u = d.drop_duplicates()
    if u.duplicated(key).any():
        raise TestArmError("rho_v2.parquet: pooled rho attributes not unique per cohort pair")
    return u.set_index(key)


def _pair_attrs(tab: pd.DataFrame, arm: str, mode: str, d1: str, d2: str) -> dict:
    k = (arm, mode, d1, d2)
    if k in tab.index:   # the values rho_update.write_attrs wrote
        r = tab.loc[k]
        return {a: t(r[a]) for a, t in PAIR_ATTRS.items()}
    return {"rho_hat_pair": float("nan"), "rho_se_pair": float("nan"), "rho_se_pair_win": float("nan"),
            "rho_pair_n_windows": 0, "rho_pair_n_clusters": 0}


def _sha256(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _id_rows(arm: str, id_list_path, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(id list as strings, extraction rows in id-list order). mode "test": the frozen sub list (sha256 checked
    against configs/real_test_ids_v1.sha256) + data.real.test_meta (sha256 checked); mode "valid": an ID list of
    validation rows in the same format + the rows of <arm>_candidates.tsv (exactly what run_arm extracts)."""
    from pathlib import Path
    from data.real.freeze_test_ids import ID_COLS, SHA_FILE
    from data.real.test_meta import frozen_hashes, load_meta
    p = Path(id_list_path)
    ids = pd.read_csv(p, sep="\t", dtype=str)
    if list(ids.columns[:len(ID_COLS)]) != ID_COLS:
        raise TestArmError(f"{p}: columns {list(ids.columns)} (expected the frozen ID-list format {ID_COLS})")
    if (ids["arm"] != arm).any() or ids["locus_id"].duplicated().any():
        raise TestArmError(f"{p}: rows of another arm or duplicate locus ids")
    if mode == "test":
        fz = frozen_hashes(SHA_FILE)
        if fz.get(FROZEN_SUB_LIST[arm]) != _sha256(p):
            raise TestArmError(f"{p}: sha256 is not the frozen {FROZEN_SUB_LIST[arm]} hash (P3-14/P3-24)")
        if (ids["split"] != "test").any():
            raise TestArmError(f"{p}: non-test rows in a test ID list")
        meta = load_meta(arm)
    elif mode == "valid":
        if (ids["split"] != "valid").any():
            raise TestArmError(f"{p}: validation mode takes split == 'valid' rows only")
        meta = pd.read_csv(PAIRS / f"{arm}_candidates.tsv", sep="\t", dtype={"chr": str})
        meta = meta[meta["split"] == "valid"]
    else:
        raise ValueError(f"mode {mode!r}: 'test' or 'valid'")
    meta = meta.set_index("locus_id", drop=False)
    missing = set(ids["locus_id"]) - set(meta.index)
    if missing:
        raise TestArmError(f"{arm}: {len(missing)} ids of the list have no extraction metadata")
    rows = meta.loc[ids["locus_id"]].reset_index(drop=True)
    if not rows[ID_COLS].astype(str).reset_index(drop=True).equals(ids[ID_COLS].reset_index(drop=True)):
        raise TestArmError(f"{arm}: metadata ID columns differ from the ID list")
    return ids, rows


def build_test_arm(arm, id_list_path=None, out_dir=None, workers: int = 16, mode: str = "test",
                   compression: str | None = "gzip", locus_dir=None, **kw):
    """Build the locus set(s) of one real-data arm for a frozen ID list, IDENTICALLY to the train/valid sets
    (extract_pair: same harmonisation, variant set, LD panels/modes, per-window P3-1 rho_hat with each mode's LD,
    trait types, ld_role, boot_unit / fg_cs_id), plus the attributes rho_update writes (rho_hat_pair & co = the
    pair's TRAIN pooled estimate from real_locus_sets@2) and the P3-23(b) attributes donor_disjoint and
    rho_sens_zero (0.0 for donor-disjoint pairs, else NaN).

      arm:          "R2" | "R3b" | "R4b"
      id_list_path: an ID list in the frozen format (data.real.freeze_test_ids.ID_COLS); mode "test": the frozen
                    pairs/<arm>_test_ids_sub_v1.tsv (sha256 verified); mode "valid": validation rows
      out_dir:      output directory (never the real_locus_sets@2 directory); files <arm>_<split>[.<ld mode>].h5,
                    *.truth.parquet, *.variants.parquet, *.extract_log.tsv, *.summary.json
      mode "test" refuses unless eval.io.has_test_access() and the caller is scripts/eval_test.py (parent check);
      each worker re-checks has_test_access() (worker check).
    Returns {"arm", "split", "files": {ld mode: path}, "ids": written locus ids in list order, ...}.

    Entry-point form (src/eval/testrun.py calls `builder(real_spec, out_dir, workers=...)`): when `arm` is the
    methods file's real_data_arms mapping, this dispatches to build_test_arms(arm, id_list_path, workers=...)."""
    from collections.abc import Mapping
    if isinstance(arm, Mapping):
        return build_test_arms(arm, id_list_path if out_dir is None else out_dir, workers=workers, mode=mode, **kw)
    import multiprocessing as mp
    from pathlib import Path
    from eval.io import truth_path_for, write_locus_set
    if arm not in ARM_MODES:
        raise ValueError(f"arm {arm!r}")
    if mode == "test":
        _check_test_access(in_worker=False)          # parent check
    out_dir = Path(out_dir)
    locus_dir = Path(locus_dir) if locus_dir is not None else LOCUS_DIR
    if out_dir.resolve() == OUT.resolve() or OUT.resolve() in out_dir.resolve().parents:
        raise TestArmError(f"out_dir must not be inside {OUT} (real_locus_sets@2; registered artifacts are immutable)")
    ids, rows_df = _id_rows(arm, id_list_path, mode)
    split = "test" if mode == "test" else "valid"
    modes = ARM_MODES[arm]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {m: out_dir / (f"{arm}_{split}.h5" if len(modes) == 1 else f"{arm}_{split}.{m}.h5") for m in modes}
    for pth in paths.values():
        if pth.exists():
            pth.unlink()
    rho_tab = pair_rho_table(locus_dir)
    rows = [r._asdict() for r in rows_df.itertuples(index=False)]
    logs, sides, n_written, ok_ids = [], [], {m: 0 for m in modes}, {m: [] for m in modes}
    t0 = time.time()
    ctx = mp.get_context("spawn")   # the test run is multi-threaded: never fork it
    with ProcessPoolExecutor(max(1, int(workers)), mp_context=ctx) as ex:
        for res, dt in ex.map(_job, [(r, modes, mode == "test") for r in rows], chunksize=2):
            for e in res:
                logs.append({**e.log, "extract_s": dt})
                if e.locus is None:
                    continue
                m = e.log["mode"]
                a = e.locus.extra_attrs
                dj = donor_disjoint(a["dataset1"], a["dataset2"])
                a.update(_pair_attrs(rho_tab, arm, m, a["dataset1"], a["dataset2"]))
                a.update(donor_disjoint=bool(dj), rho_sens_zero=0.0 if dj else float("nan"))
                n_written[m] += write_locus_set(paths[m], [e.locus], mode="a", compression=compression)
                ok_ids[m].append(e.locus.locus_id)
                if len(e.sidecar):
                    sides.append(e.sidecar)
    if len({tuple(v) for v in ok_ids.values()}) != 1:
        raise TestArmError(f"{arm}: LD modes wrote different locus lists")
    lg = pd.DataFrame(logs)
    stem = f"{arm}_{split}"
    lg.to_csv(out_dir / f"{stem}.extract_log.tsv", sep="\t", index=False)
    if sides:
        pd.concat(sides, ignore_index=True).to_parquet(out_dir / f"{stem}.variants.parquet", index=False)
    written = ok_ids[modes[0]]
    truth = rows_df[rows_df["locus_id"].isin(set(written))].copy()
    truth["label"] = ARM_LABEL[arm]
    for pth in paths.values():
        truth.to_parquet(truth_path_for(pth), index=False)
    st = lg.drop_duplicates("locus_id")["status"].astype(str).str.split(":").str[0].value_counts()
    summary = {"arm": arm, "split": split, "mode": mode, "id_list": str(id_list_path),
               "id_list_sha256": _sha256(id_list_path), "n_requested": int(len(ids)), "n_written": n_written,
               "status_counts": {k: int(v) for k, v in st.items()},
               "failed_ids": sorted(set(ids["locus_id"]) - set(written)),
               "rho_pair_source": str(locus_dir / "rho_v2.parquet"), "wall_s": round(time.time() - t0, 1),
               "files": {m: str(p) for m, p in paths.items()}}
    (out_dir / f"{stem}.summary.json").write_text(json.dumps(summary, indent=1))
    return {**summary, "ids": list(written), "files": {m: str(p) for m, p in paths.items()}}


def build_test_arms(spec: dict, out_dir, workers: int = 16, mode: str = "test", id_lists: dict | None = None,
                    arms=("R2", "R3b", "R4b"), **kw) -> dict:
    """Entry point of the test run (configs/test_methods_v1.yaml real_data_arms.builder; src/eval/testrun.py
    calls builder(real_spec, out_dir, workers=...)). Builds R2, R3b, R4b from the frozen sub lists
    (pairs/<arm>_test_ids_sub_v1.tsv; P3-14 sizes, P3-24 seeds) and returns
    {real_R2 | real_R3b | real_R4b_insample | real_R4b_reference | real_R4b_insample_raw: {"path", "ids", ...}}.
    spec[<arm>]["n"], when given, must equal the frozen list's size. mode "valid" (dry checks) needs `id_lists`."""
    from pathlib import Path
    out_dir = Path(out_dir)
    if mode == "test":
        _check_test_access(in_worker=False)
    if mode == "valid" and not id_lists:
        raise ValueError("validation mode needs id_lists={arm: path}")
    res, out = {}, {}
    for arm in arms:
        lst = Path(id_lists[arm]) if id_lists else PAIRS / FROZEN_SUB_LIST[arm]
        n_spec = ((spec or {}).get(arm) or {}).get("n")
        n_list = sum(1 for _ in open(lst)) - 1
        if mode == "test" and n_spec is not None and int(n_spec) != n_list:
            raise TestArmError(f"{arm}: methods file n={n_spec} but the frozen list has {n_list} loci")
        res[arm] = build_test_arm(arm, lst, out_dir, workers=workers, mode=mode, **kw)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, (arm, m) in REAL_SET_MODES.items():
        if arm in res:
            r = res[arm]
            out[key] = {"path": r["files"][m], "ids": r["ids"], "arm": arm, "ld_mode": m,
                        "n_requested": r["n_requested"], "failed_ids": r["failed_ids"]}
    (out_dir / f"real_arms_{'test' if mode == 'test' else 'valid'}.summary.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "ids"} | {"n_ids": len(v["ids"])}
                    for k, v in out.items()}, indent=1))
    return out


def main() -> None:
    assert_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["R2", "R3b", "R4b"])
    ap.add_argument("--splits", nargs="+", default=["train", "valid"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    for arm in a.arms:
        for s in a.splits:
            print(json.dumps(run_arm(arm, s, a.workers, a.limit)), flush=True)


if __name__ == "__main__":
    main()
