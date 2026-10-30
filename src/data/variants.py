"""Variant QC and per-population allele frequencies (DESIGN §2.1).

Per chromosome, from data/processed/1kg/chrN.{pgen,pvar,psam} (scripts/prep_1kg_pgen.sh):
  keep  biallelic SNVs (REF and ALT a single A/C/G/T base);
  drop  positions with > 1 record: either in the pgen itself, or in the raw 1000G VCF among
        SNV/indel records polymorphic in our EUR+AFR samples (data/processed/variants/multirec/chrN.pos,
        src/data/multirecord_positions.sh);
  drop  MHC chr6:28,500,000-33,400,000 (1-based, inclusive).
No MAF filter here: MAF filters are population-specific and applied downstream with the stored AFs.

Output data/processed/variants/chrN.parquet, one row per retained variant, sorted by pos:
  chr, pos (1-based), variant_id (chr:pos:ref:alt), ref, alt, pgen_idx (row in chrN.pvar),
  af_<pool> = ALT allele frequency in pool, for pools A, B, FIN, ASW_ACB, L6c, NFE (= A u B),
  ac_missing = number of missing genotype calls among the 1164 samples (expected 0).
plus data/processed/variants/chrN.summary.json with the QC counts.

Usage: python -m data.variants --chroms 1 2 ... [--jobs 22]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PGEN_DIR = ROOT / "data/processed/1kg"
OUT_DIR = ROOT / "data/processed/variants"
MULTIREC_DIR = OUT_DIR / "multirec"
POOLS_FILE = ROOT / "datasets/splits/pools_v1.tsv"
MHC = ("chr6", 28_500_000, 33_400_000)
AF_POOLS = ("A", "B", "FIN", "ASW_ACB", "L6c", "NFE")
CHUNK = 20_000


def read_pvar(path: Path) -> pd.DataFrame:
    with open(path) as fh:
        skip = 0
        for line in fh:
            if line.startswith("##"):
                skip += 1
            else:
                break
    pv = pd.read_csv(path, sep="\t", skiprows=skip, dtype={"#CHROM": str, "POS": np.int64, "ID": str,
                                                          "REF": str, "ALT": str}, usecols=range(5))
    pv.columns = ["chr", "pos", "variant_id", "ref", "alt"]
    pv["pgen_idx"] = np.arange(len(pv), dtype=np.int64)
    return pv


def read_psam(path: Path) -> list[str]:
    return pd.read_csv(path, sep="\t")["#IID"].astype(str).tolist()


def pool_indices(sample_ids: list[str], pools_file: Path = POOLS_FILE) -> dict[str, np.ndarray]:
    df = pd.read_csv(pools_file, sep="\t")
    pos = {s: i for i, s in enumerate(sample_ids)}
    out = {}
    for p, g in df.groupby("pool"):
        missing = set(g["sample_id"]) - set(pos)
        assert not missing, f"pool {p}: samples not in psam: {sorted(missing)[:5]}"
        out[p] = np.sort(np.array([pos[s] for s in g["sample_id"]], dtype=np.int64))
    out["NFE"] = np.union1d(out["A"], out["B"])
    return out


def qc_mask(pv: pd.DataFrame, multirec_pos: np.ndarray) -> tuple[np.ndarray, dict]:
    """Boolean keep-mask and counts. multirec_pos: positions with >1 record in the raw VCF."""
    bases = {"A", "C", "G", "T"}
    snv = pv["ref"].isin(bases).to_numpy() & pv["alt"].isin(bases).to_numpy()
    dup_pgen = pv["pos"].duplicated(keep=False).to_numpy()
    dup_raw = np.isin(pv["pos"].to_numpy(), multirec_pos)
    c, lo, hi = MHC
    mhc = (pv["chr"].to_numpy() == c) & (pv["pos"].to_numpy() >= lo) & (pv["pos"].to_numpy() <= hi)
    keep = snv & ~dup_pgen & ~dup_raw & ~mhc
    counts = {"n_pvar": int(len(pv)), "n_snv": int(snv.sum()),
              "n_snv_dup_in_pgen": int((snv & dup_pgen).sum()),
              "n_snv_dup_in_raw_only": int((snv & dup_raw & ~dup_pgen).sum()),
              "n_snv_mhc": int((snv & mhc & ~dup_pgen & ~dup_raw).sum()),
              "n_pass": int(keep.sum())}
    return keep, counts


def allele_freqs(pgen: Path, n_samples: int, idx: np.ndarray, pools: dict[str, np.ndarray]) -> pd.DataFrame:
    """ALT allele frequency per pool for the pgen variant indices idx (sorted)."""
    import pgenlib
    r = pgenlib.PgenReader(str(pgen).encode(), raw_sample_ct=n_samples)
    out = {p: np.empty(len(idx)) for p in pools}
    nmiss = np.zeros(len(idx), dtype=np.int64)
    for s in range(0, len(idx), CHUNK):
        vi = idx[s:s + CHUNK].astype(np.uint32)
        g = np.empty((len(vi), n_samples), dtype=np.int8)
        r.read_list(vi, g)
        miss = g < 0
        nmiss[s:s + CHUNK] = miss.sum(1)
        gf = np.where(miss, 0, g).astype(np.float64)
        for p, si in pools.items():
            nobs = (~miss[:, si]).sum(1)
            out[p][s:s + CHUNK] = gf[:, si].sum(1) / (2.0 * np.maximum(nobs, 1))
    r.close()
    df = pd.DataFrame({f"af_{p}": out[p] for p in AF_POOLS})
    df["n_missing"] = nmiss
    return df


def run_chrom(c: int, pgen_dir: Path = PGEN_DIR, out_dir: Path = OUT_DIR,
              multirec_dir: Path = MULTIREC_DIR, pools_file: Path = POOLS_FILE) -> dict:
    pv = read_pvar(pgen_dir / f"chr{c}.pvar")
    assert (pv["chr"] == f"chr{c}").all()
    assert pv["pos"].is_monotonic_increasing, "pvar not sorted"
    assert (pv["variant_id"] == pv["chr"] + ":" + pv["pos"].astype(str) + ":" + pv["ref"] + ":" + pv["alt"]).all()
    mr_file = multirec_dir / f"chr{c}.pos"
    if not mr_file.exists():
        raise FileNotFoundError(f"{mr_file} missing: run src/data/multirecord_positions.sh {c}")
    mr = pd.read_csv(mr_file, sep="\t", header=None, names=["pos", "n"])["pos"].to_numpy(np.int64)
    keep, counts = qc_mask(pv, mr)
    samples = read_psam(pgen_dir / f"chr{c}.psam")
    pools = pool_indices(samples, pools_file)
    out = pv.loc[keep].reset_index(drop=True)
    af = allele_freqs(pgen_dir / f"chr{c}.pgen", len(samples), out["pgen_idx"].to_numpy(), pools)
    out = pd.concat([out, af], axis=1)
    span_mb = (pv["pos"].iloc[-1] - pv["pos"].iloc[0]) / 1e6
    counts.update({"chr": f"chr{c}", "n_multirec_positions_raw": int(len(mr)),
                   "pvar_density_per_mb": float(len(pv) / span_mb),
                   "n_pass_maf01_A": int((np.minimum(out.af_A, 1 - out.af_A) >= 0.01).sum()),
                   "n_pass_maf01_B": int((np.minimum(out.af_B, 1 - out.af_B) >= 0.01).sum()),
                   "n_pass_maf01_FIN": int((np.minimum(out.af_FIN, 1 - out.af_FIN) >= 0.01).sum()),
                   "n_missing_calls": int(out["n_missing"].sum())})
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"chr{c}.parquet.tmp"
    out.to_parquet(tmp, index=False)
    tmp.rename(out_dir / f"chr{c}.parquet")
    (out_dir / f"chr{c}.summary.json").write_text(json.dumps(counts, indent=1))
    return counts


def load_variants(chrom: str | int, out_dir: Path = OUT_DIR) -> pd.DataFrame:
    c = str(chrom).removeprefix("chr")
    return pd.read_parquet(out_dir / f"chr{c}.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroms", nargs="+", type=int, default=list(range(1, 23)))
    ap.add_argument("--jobs", type=int, default=11)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.jobs) as ex:
        for res in ex.map(run_chrom, a.chroms):
            print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
