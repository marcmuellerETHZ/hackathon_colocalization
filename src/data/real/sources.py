"""Readers for eQTL Catalogue r7 and FinnGen R12 files (raw formats -> tidy pandas tables).

All coordinates are GRCh38 1-based. Contigs are returned as 'chrN'. No harmonisation happens here
(see harmonise.py); alleles are returned as in the source (eQTL Catalogue and FinnGen both report
beta for the ALT allele, REF = GRCh38 reference base).
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from data.real.config import OUT, EqtlDataset, FinnGenTrait

CACHE = OUT / "cache"


def norm_chr(c) -> str:
    c = str(c)
    return c if c.startswith("chr") else f"chr{c}"


def bare_chr(c) -> str:
    c = str(c)
    return c[3:] if c.startswith("chr") else c


def tabix_text(path: Path | str, region: str) -> str:
    """Raw tabix output for one region. Errors propagate (check=True, stderr shown)."""
    res = subprocess.run(["tabix", str(path), region], check=True, capture_output=True, text=True)
    # the .tbi is downloaded before its (larger) data file, so htslib warns that the index is older; that
    # warning is benign (index content verified by sha256 / region queries); anything else is fatal
    err = [ln for ln in res.stderr.splitlines() if ln.strip() and "index file is older than the data file" not in ln]
    if err:
        raise RuntimeError(f"tabix {path} {region}: {' | '.join(err)}")
    return res.stdout


def _read_tsv(text: str, names: list[str], dtype: dict | None = None) -> pd.DataFrame:
    if not text:
        return pd.DataFrame({n: pd.Series(dtype=object) for n in names})
    return pd.read_csv(io.StringIO(text), sep="\t", header=None, names=names, dtype=dtype, na_values=["NA", ""],
                       keep_default_na=False)


def _cached(name: str, build) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    df = build()
    tmp = f.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(f)
    return df


# --------------------------------------------------------------------------------------------------
# eQTL Catalogue
# --------------------------------------------------------------------------------------------------
EQTL_ALL_COLS = ["molecular_trait_id", "chromosome", "position", "ref", "alt", "variant", "ma_samples", "maf",
                 "pvalue", "beta", "se", "type", "ac", "an", "r2", "molecular_trait_object_id", "gene_id",
                 "median_tpm", "rsid"]


def eqtl_permuted(ds: EqtlDataset) -> pd.DataFrame:
    """One row per tested gene: gene_id, lead_variant, chr, lead_pos, lead_p, lead_beta, p_perm, p_beta."""
    df = pd.read_csv(ds.path("permuted"), sep="\t", dtype={"chromosome": str})
    assert (df["molecular_trait_id"] == df["molecular_trait_object_id"]).all(), "ge: trait id == object id"
    out = pd.DataFrame({
        "gene_id": df["molecular_trait_id"].astype(str),
        "lead_variant": df["variant"].astype(str),
        "chr": df["chromosome"].map(norm_chr),
        "lead_pos": df["position"].astype(np.int64),
        "lead_p": df["pvalue"].astype(float),
        "lead_beta": df["beta"].astype(float),
        "n_variants": df["n_variants"].astype(np.int64),
        "p_perm": df["p_perm"].astype(float),
        "p_beta": df["p_beta"].astype(float),
    })
    if out["gene_id"].duplicated().any():
        raise ValueError(f"{ds.dataset_id}: duplicated gene in permuted file")
    return out


def eqtl_cs_variants(ds: EqtlDataset) -> pd.DataFrame:
    """Credible-set membership rows: gene_id, cs_id, variant, chr, pos, pip, pvalue, z, cs_size, cs_min_r2."""
    df = pd.read_csv(ds.path("cs"), sep="\t")
    v = df["variant"].str.split("_", expand=True)
    return pd.DataFrame({
        "gene_id": df["molecular_trait_id"].astype(str),
        "cs_id": df["cs_id"].astype(str),
        "variant": df["variant"].astype(str),
        "chr": v[0].map(norm_chr),
        "pos": v[1].astype(np.int64),
        "pip": df["pip"].astype(float),
        "pvalue": df["pvalue"].astype(float),
        "z": df["z"].astype(float),
        "cs_size": df["cs_size"].astype(np.int64),
        "cs_min_r2": df["cs_min_r2"].astype(float),
    })


def cs_summary(cs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One row per credible set (grouped by `keys`): lead (max pip), lead p/z, size, purity, variant list."""
    cs = cs.sort_values(keys + ["pip"], ascending=[True] * len(keys) + [False])
    lead = cs.groupby(keys, sort=False).head(1).set_index(keys)
    members = cs.groupby(keys, sort=False)["variant"].agg(lambda s: ";".join(sorted(s)))
    out = lead[["variant", "chr", "pos", "pvalue", "z", "pip", "cs_size", "cs_min_r2"]].rename(
        columns={"variant": "lead_variant", "pos": "lead_pos", "pvalue": "lead_p", "z": "lead_z", "pip": "lead_pip"})
    out["members"] = members
    out["n_members"] = members.str.count(";") + 1
    # top-PIP SNV of the CS (used where an SNV is required, e.g. LD look-ups; NaN if the CS has no SNV)
    parts = cs["variant"].str.replace(":", "_").str.split("_")
    is_snv = parts.map(lambda x: len(x) == 4 and len(x[2]) == 1 and len(x[3]) == 1)
    snv = cs[is_snv.to_numpy()]
    snv_lead = snv.groupby(keys, sort=False).head(1).set_index(keys)["variant"]
    out["lead_snv"] = snv_lead.reindex(out.index)
    # SNV members in decreasing PIP order (cs is sorted by PIP within CS)
    out["snv_by_pip"] = snv.groupby(keys, sort=False)["variant"].agg(";".join).reindex(out.index)
    return out.reset_index()


def eqtl_gene_region(ds: EqtlDataset, gene_id: str, chrom: str, start: int, end: int) -> pd.DataFrame:
    """Nominal summary statistics of one gene in [start, end] (tabix on the local *.all.tsv.gz)."""
    txt = tabix_text(ds.path("all"), f"{bare_chr(chrom)}:{max(1, start)}-{end}")
    df = _read_tsv(txt, EQTL_ALL_COLS, dtype={"chromosome": str, "rsid": str, "molecular_trait_id": str})
    df = df[df["molecular_trait_id"] == gene_id]
    return pd.DataFrame({
        "chr": df["chromosome"].map(norm_chr),
        "pos": df["position"].astype(np.int64),
        "ref": df["ref"].astype(str),
        "alt": df["alt"].astype(str),
        "beta": df["beta"].astype(float),
        "se": df["se"].astype(float),
        "pvalue": df["pvalue"].astype(float),
        "af": (df["ac"].astype(float) / df["an"].astype(float)),
        "n": df["an"].astype(float) / 2.0,
        "rsid": df["rsid"].astype(str),
    }).reset_index(drop=True)


# --------------------------------------------------------------------------------------------------
# FinnGen
# --------------------------------------------------------------------------------------------------
FG_SS_COLS_LAB = ["chrom", "pos", "ref", "alt", "pval", "mlogp", "beta", "sebeta", "af_alt"]
FG_SS_COLS_END = ["chrom", "pos", "ref", "alt", "rsids", "nearest_genes", "pval", "mlogp", "beta", "sebeta",
                  "af_alt", "af_alt_cases", "af_alt_controls"]
FG_SNP_COLS = ["trait", "region", "v", "rsid", "chromosome", "position", "allele1", "allele2", "maf", "beta", "se",
               "p", "mean", "sd", "prob", "cs", "cs_specific_prob", "low_purity", "lead_r2"]


def finngen_region(t: FinnGenTrait, chrom: str, start: int, end: int) -> pd.DataFrame:
    txt = tabix_text(t.path("sumstats"), f"{bare_chr(chrom)}:{max(1, start)}-{end}")
    cols = FG_SS_COLS_LAB if t.kind == "lab" else FG_SS_COLS_END
    df = _read_tsv(txt, cols, dtype={"chrom": str})
    return pd.DataFrame({
        "chr": df["chrom"].map(norm_chr),
        "pos": df["pos"].astype(np.int64),
        "ref": df["ref"].astype(str),
        "alt": df["alt"].astype(str),
        "beta": df["beta"].astype(float),
        "se": df["sebeta"].astype(float),
        "pvalue": df["pval"].astype(float),
        "af": df["af_alt"].astype(float),
        "rsid": df["rsids"].astype(str) if "rsids" in df else "",
    }).reset_index(drop=True)


def finngen_cred(t: FinnGenTrait) -> pd.DataFrame:
    df = pd.read_csv(t.path("cred"), sep="\t", dtype={"trait": str, "region": str}, compression="gzip")
    df["low_purity"] = df["low_purity"].astype(str).str.upper().eq("TRUE")
    return df


def _finngen_cs_variants_build(t: FinnGenTrait) -> pd.DataFrame:
    # stream the (large) SuSiE snp file and keep only rows inside a 95% credible set (cs != -1)
    cmd = f"set -euo pipefail; zcat '{t.path('snp')}' | cut -f1-19 | awk -F'\\t' 'NR==1 || $16 != -1'"
    res = subprocess.run(["bash", "-c", cmd], check=True, capture_output=True, text=True)
    df = pd.read_csv(io.StringIO(res.stdout), sep="\t", dtype={"trait": str, "region": str, "chromosome": str})
    assert list(df.columns) == FG_SNP_COLS, df.columns
    return df


def finngen_cs_variants(t: FinnGenTrait) -> pd.DataFrame:
    """CS membership: region, cs, variant (chr_pos_ref_alt), chr, pos, pip (cs_specific_prob), pvalue, z."""
    df = _cached(f"finngen_{t.pheno}_cs_variants", lambda: _finngen_cs_variants_build(t))
    return pd.DataFrame({
        "region": df["region"].astype(str),
        "cs": df["cs"].astype(np.int64),
        "variant": df["rsid"].astype(str),  # FinnGen 'rsid' column holds chrN_pos_ref_alt
        "chr": df["chromosome"].map(norm_chr),
        "pos": df["position"].astype(np.int64),
        "pip": df["cs_specific_prob"].astype(float),
        "pvalue": df["p"].astype(float),
        "z": df["beta"].astype(float) / df["se"].astype(float),
    })


def finngen_cs_summary(t: FinnGenTrait) -> pd.DataFrame:
    """One row per FinnGen 95% CS (cred file joined with membership): region, cs, lead, purity flags."""
    cred = finngen_cred(t)
    mem = finngen_cs_variants(t)
    s = cs_summary(mem.assign(cs_size=0, cs_min_r2=np.nan), ["region", "cs"])
    s = s.drop(columns=["cs_size", "cs_min_r2"]).merge(
        cred[["region", "cs", "cs_size", "cs_min_r2", "low_purity", "cs_log10bf"]], on=["region", "cs"], how="left",
        validate="1:1")
    if s["cs_size"].isna().any():
        raise ValueError(f"{t.pheno}: CS in snp file missing from cred file")
    s["trait"] = t.pheno
    return s
