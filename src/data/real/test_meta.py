"""Extraction metadata for the frozen real-data test ID lists (P3-14 / P3-24), derived WITHOUT test data.

  python -m data.real.test_meta --workers 9          # derive, verify, write (refuses to overwrite)
  python -m data.real.test_meta --check              # verify the written files against datasets/configs/real_test_meta_v1.sha256

Why: the frozen lists `pairs/<arm>_test_ids_sub_v1.tsv` hold identifier columns only (locus_id, arm, datasets,
gene, split, chr, window). `extract.extract_pair` also needs, exactly as for train/valid, the columns
  gene_name, block_id (R2/R4b boot_unit), lead1, lead2 (lead-present flags)        -- all arms
  fg_region, fg_cs (R3b boot_unit = FinnGen credible-set id "<trait>|<fg_region>|cs<fg_cs>")  -- R3b.
M3's `pairs.write_arm` kept full columns only for train/valid (`<arm>_candidates.tsv`); the test rows were
written as identifiers only. So the columns are re-derived here by re-running the SAME gene-level /
credible-set selection code (`pairs.eqtl_gene_cs`, `build_r2`, `r3b_prefilter`, `build_r4b`) that produced the
ID lists. That code reads gene tables and credible-set METADATA (eQTL Catalogue SuSiE credible sets, FinnGen
SuSiE credible sets) for all splits -- the same inputs the M3 freeze read -- and never any summary statistics
or LD of a test window. The R3b LD filter (`pairs.r3b_ld_filter`, which used lead LD) is NOT re-run: the R3b
test rows are the pre-LD-filter rows whose locus_id is in the frozen list.

Verification (all must hold, else nothing is written):
  1. determinism on train/valid: the rebuilt train+valid rows equal `<arm>_candidates.tsv` (R2, R4b: the whole
     file byte-identical; R3b: every pre-LD-filter column of the candidate rows identical);
  2. test rows vs the frozen hashes (datasets/configs/real_test_ids_v1.sha256): R2/R4b: the rebuilt full test list,
     serialised exactly as `pairs.write_arm` did, has the frozen sha256 of `<arm>_test_ids.tsv`, and the P3-24
     subsample drawn from it (`freeze_test_ids.subsample`) has the frozen sha256 of `<arm>_test_ids_sub_v1.tsv`;
     R3b: the rebuilt pre-filter test rows restricted to the frozen list's locus_ids, serialised the same way,
     have the frozen sha256 of `R3b_test_ids.tsv` (all frozen ids found, same order) and of the sub list;
  3. the written metadata file's ID columns equal the frozen sub list (string-exact, same order).
Outputs (data/processed/real_testmeta_v1/, artifact real_test_meta@1): `<arm>_test_meta_v1.tsv` (rows = the frozen sub list, ID columns + EXTRA_COLS; no p-values,
z-scores, CS members or LD), `test_meta_v1.verify.json` (hashes and counts only) and
`test_meta_v1.sha256`, committed as `datasets/configs/real_test_meta_v1.sha256` (which `load_meta` checks). Nothing about test rows is printed except counts and hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from data.real.config import OUT, ROOT
from data.real.freeze_test_ids import ID_COLS, SHA_FILE, SUBSAMPLE, subsample

PAIRS = OUT / "pairs"
# own directory, registered as real_test_meta@1 (never inside the registered real_locus_sets@2 tree)
META_DIR = ROOT / "data" / "processed" / "real_testmeta_v1"
META_SHA_FILE = ROOT / "datasets" / "configs" / "real_test_meta_v1.sha256"   # committed copy (frozen with the code)
META_SHA_OUT = META_DIR / "test_meta_v1.sha256"                    # written by derive (jobs never write datasets/configs/)
EXTRA_COLS = {"R2": ["gene_name", "tss", "block_id", "lead1", "lead2"],
              "R3b": ["gene_name", "tss", "block_id", "lead1", "lead2", "fg_region", "fg_cs"],
              "R4b": ["gene_name", "tss", "block_id", "lead1", "lead2"]}
# R3b columns produced before the LD filter (pairs.r3b_prefilter); the rest of R3b_candidates.tsv is LD output
R3B_PRE_COLS = ["locus_id", "arm", "dataset1", "dataset2", "gene_id", "gene_name", "split", "chr", "tss", "block_id",
                "win_lo", "win_hi", "lead1", "lead1_p", "lead1_z", "lead2", "lead2_p", "lead2_z", "lead1_snv",
                "lead2_snv", "cs1_snv_by_pip", "cs2_snv_by_pip", "fg_region", "fg_cs", "cs1_size", "cs2_size",
                "cs1_members", "cs2_members"]


def meta_path(arm: str, meta_dir: Path = META_DIR) -> Path:
    return meta_dir / f"{arm}_test_meta_v1.tsv"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tsv_bytes(df: pd.DataFrame) -> bytes:
    """Serialisation used by pairs.write_arm and freeze_test_ids (to_csv, tab, no index)."""
    return df.to_csv(sep="\t", index=False).encode()


def frozen_hashes(sha_file: Path = SHA_FILE) -> dict[str, str]:
    """{file name: sha256} of datasets/configs/real_test_ids_v1.sha256."""
    out = {}
    for line in sha_file.read_text().splitlines():
        if line.strip():
            h, rel = line.split(None, 1)
            out[Path(rel.strip()).name] = h
    return out


def _gate_job(ds_id):
    from data.real.pairs import eqtl_gene_cs
    from data.real.regions import gene_table
    return ds_id, eqtl_gene_cs(ds_id, gene_table())


def rebuild_tables(workers: int = 9) -> dict[str, pd.DataFrame]:
    """The M3 selection code (pairs.main without the R3b LD filter): R2, R3b (pre-LD), R4b, all splits."""
    from data.real.config import eqtl_datasets
    from data.real.pairs import build_r2, build_r4b, r3b_prefilter
    from data.real.regions import gene_table
    genes = gene_table()
    with ProcessPoolExecutor(min(workers, len(eqtl_datasets()))) as ex:
        gates = dict(ex.map(_gate_job, list(eqtl_datasets())))
    return {"R2": build_r2(gates), "R3b": r3b_prefilter(gates, genes), "R4b": build_r4b(gates, genes)}


def _str_frame(b: bytes, cols=None) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b), sep="\t", dtype=str, keep_default_na=False, usecols=cols)


def verify_and_select(arm: str, table: pd.DataFrame, pairs_dir: Path = PAIRS, sha_file: Path = SHA_FILE) -> tuple[pd.DataFrame, dict]:
    """Checks 1-3 of the module docstring for one arm; returns (metadata rows of the frozen sub list, report).
    Raises AssertionError on any mismatch. The report holds counts and hashes only."""
    fz = frozen_hashes(sha_file)
    rep: dict = {"arm": arm}
    tv = table[table["split"].isin(["train", "valid"])]
    te = table[table["split"] == "test"]
    cand_file = pairs_dir / f"{arm}_candidates.tsv"
    # 1. train/valid determinism
    if arm == "R3b":
        cand = pd.read_csv(cand_file, sep="\t", dtype=str, keep_default_na=False, usecols=R3B_PRE_COLS)[R3B_PRE_COLS]
        reb = _str_frame(tsv_bytes(tv[tv["locus_id"].isin(set(cand["locus_id"]))][R3B_PRE_COLS]))
        same = cand.reset_index(drop=True).equals(reb.reset_index(drop=True))
        rep["trainvalid_check"] = {"n_candidates": int(len(cand)), "n_rebuilt_prefilter_trainvalid": int(len(tv)),
                                   "prefilter_columns_identical": bool(same)}
        assert same, f"{arm}: rebuilt pre-filter train/valid rows differ from {cand_file.name}"
    else:
        b = tsv_bytes(tv)
        same = sha256_bytes(b) == sha256_bytes(cand_file.read_bytes())
        rep["trainvalid_check"] = {"n_candidates": int(len(tv)), "candidates_file_byte_identical": bool(same),
                                   "sha256": sha256_bytes(b)}
        assert same, f"{arm}: rebuilt train/valid rows are not byte-identical to {cand_file.name}"
    # 2. test rows vs the frozen hashes
    full_name, sub_name = f"{arm}_test_ids.tsv", f"{arm}_test_ids_sub_v1.tsv"
    if arm == "R3b":
        frozen_ids = pd.read_csv(pairs_dir / full_name, sep="\t", dtype=str, usecols=["locus_id"])["locus_id"]
        te = te[te["locus_id"].isin(set(frozen_ids))]
        assert len(te) == len(frozen_ids), f"{arm}: {len(frozen_ids) - len(te)} frozen ids not rebuilt"
        rep["n_test_prefilter"] = int((table["split"] == "test").sum())
    full_b = tsv_bytes(te[ID_COLS])
    rep["full"] = {"n": int(len(te)), "sha256": sha256_bytes(full_b), "frozen": fz[full_name]}
    assert rep["full"]["sha256"] == fz[full_name], f"{arm}: rebuilt full test ID list hash != frozen"
    ids = _str_frame(full_b, ID_COLS)[ID_COLS]
    sub = subsample(ids, arm, SUBSAMPLE[arm])
    sub_b = tsv_bytes(sub)
    rep["sub"] = {"n": int(len(sub)), "sha256": sha256_bytes(sub_b), "frozen": fz[sub_name]}
    assert rep["sub"]["sha256"] == fz[sub_name], f"{arm}: rebuilt P3-24 subsample hash != frozen"
    on_disk = pairs_dir / sub_name
    assert sha256_bytes(on_disk.read_bytes()) == fz[sub_name], f"{on_disk} does not match its frozen hash"
    # 3. metadata rows of the frozen sub list, in its order
    te_s = _str_frame(tsv_bytes(te[ID_COLS + EXTRA_COLS[arm]]))
    te_s = te_s.set_index("locus_id", drop=False)
    frozen_sub = pd.read_csv(on_disk, sep="\t", dtype=str, keep_default_na=False)
    meta = te_s.loc[frozen_sub["locus_id"]].reset_index(drop=True)
    assert meta[ID_COLS].equals(frozen_sub[ID_COLS]), f"{arm}: metadata ID columns != frozen sub list"
    rep["meta"] = {"n": int(len(meta)), "columns": list(meta.columns)}
    return meta, rep


def derive(workers: int = 9, pairs_dir: Path = PAIRS, sha_file: Path = SHA_FILE, meta_sha_file: Path = META_SHA_OUT,
           root: Path = ROOT, force: bool = False, meta_dir: Path = META_DIR) -> dict:
    if meta_sha_file.exists() and not force:
        raise FileExistsError(f"{meta_sha_file} exists: the test metadata is frozen (use --check)")
    tables = rebuild_tables(workers)
    metas, reps = {}, {}
    for arm in ("R2", "R3b", "R4b"):
        metas[arm], reps[arm] = verify_and_select(arm, tables[arm], pairs_dir, sha_file)
    lines = []
    for arm, meta in metas.items():
        meta_dir.mkdir(parents=True, exist_ok=True)
        p = meta_path(arm, meta_dir)
        b = tsv_bytes(meta)
        p.write_bytes(b)
        reps[arm]["meta"]["sha256"] = sha256_bytes(b)
        lines.append(f"{sha256_bytes(b)}  {os.path.relpath(p, root)}")
    (meta_dir / "test_meta_v1.verify.json").write_text(json.dumps(reps, indent=1))
    meta_sha_file.parent.mkdir(parents=True, exist_ok=True)
    meta_sha_file.write_text("\n".join(lines) + "\n")
    return reps


def check(meta_sha_file: Path = META_SHA_FILE, root: Path = ROOT) -> list[str]:
    from data.real.freeze_test_ids import check as _check
    return _check(meta_sha_file, root)


def load_meta(arm: str, meta_dir: Path = META_DIR, meta_sha_file: Path = META_SHA_FILE, root: Path = ROOT) -> pd.DataFrame:
    """The verified metadata of the frozen sub list of `arm`, read exactly like `<arm>_candidates.tsv` is read
    by extract.run_arm (dtype chr = str), after checking its sha256 against datasets/configs/real_test_meta_v1.sha256."""
    p = meta_path(arm, meta_dir)
    want = {Path(l.split(None, 1)[1].strip()).name: l.split(None, 1)[0]
            for l in meta_sha_file.read_text().splitlines() if l.strip()}
    if want.get(p.name) != sha256_bytes(p.read_bytes()):
        raise ValueError(f"{p} does not match its sha256 in {meta_sha_file}")
    return pd.read_csv(p, sep="\t", dtype={"chr": str})


def main() -> None:
    from data.real.config import assert_env
    assert_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.check:
        bad = check()
        print("real test metadata sha256:", "OK" if not bad else f"MISMATCH {bad}")
        raise SystemExit(1 if bad else 0)
    reps = derive(a.workers, force=a.force)
    print(json.dumps(reps, indent=1))


if __name__ == "__main__":
    main()
