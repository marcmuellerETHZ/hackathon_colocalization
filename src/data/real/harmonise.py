"""Harmonisation of real summary statistics to the 1000G reference (DESIGN §2.5; R1-data 9).

Rules (one implementation for every real-data arm):
  1. contig naming -> 'chrN'; autosomes only (the reference panel is chr1-22);
  2. positions with more than one record in the study (any type: split multi-allelics, SNV + indel) are
     dropped; then SNV only (REF and ALT single bases in ACGT, REF != ALT); the reference table is already
     one record per position (data agent's QC);
  3. effect allele = ALT of the reference record. A study record whose (REF, ALT) equals the reference's
     is kept; a swapped record (REF/ALT exchanged) is flipped (beta, z -> -beta, -z; af -> 1 - af), except
     palindromic SNPs (A/T, C/G), which cannot be told apart from strand flips and are dropped when swapped;
     anything else (strand-complement or different alleles) is dropped as an allele mismatch;
  4. AF concordance vs the reference panel AF (pool B by default): flag |dAF| > 0.2; palindromic SNPs with
     |dAF| > 0.1 are dropped (DESIGN). Non-palindromic flagged SNPs are kept and counted;
  5. counts of every exclusion are returned in a log dict; lead-variant presence is checked with
     `lead_missingness`.
The build check `check_build` verifies GRCh38 by (a) REF-allele concordance with the GRCh38 1000G records
at shared positions and (b) known rsID positions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.real.config import AF_FLAG, AF_FLAG_PALINDROMIC

ACGT = frozenset("ACGT")
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
AUTOSOMES = frozenset(f"chr{i}" for i in range(1, 23))

# GRCh38 positions of well-known SNPs (dbSNP; GRCh37 positions in comments for the build test)
KNOWN_RSIDS_GRCH38 = {
    "rs429358": ("chr19", 44908684),   # APOE; GRCh37 45411941
    "rs7412": ("chr19", 44908822),     # APOE; GRCh37 45412079
    "rs12740374": ("chr1", 109274968), # CELSR2/SORT1; GRCh37 109817590
    "rs4149056": ("chr12", 21178615),  # SLCO1B1; GRCh37 21331549
}


def norm_chr(c) -> str:
    c = str(c)
    return c if c.startswith("chr") else f"chr{c}"


def is_snv(ref: pd.Series, alt: pd.Series) -> np.ndarray:
    r = ref.astype(str)
    a = alt.astype(str)
    return (r.isin(ACGT) & a.isin(ACGT) & (r != a)).to_numpy()


def is_palindromic(ref: pd.Series, alt: pd.Series) -> np.ndarray:
    r = ref.astype(str).to_numpy()
    a = alt.astype(str).to_numpy()
    return np.array([COMP.get(x) == y for x, y in zip(r, a)], dtype=bool)


def harmonise(study: pd.DataFrame, ref: pd.DataFrame, af_ref_col: str = "af_ref", label: str = "") -> tuple[pd.DataFrame, dict]:
    """Align a study table to reference records.

    study: columns chr, pos, ref, alt, beta, se, af (ALT frequency in the study; may be NaN), optional others.
    ref:   columns chr, pos, ref, alt, variant_id, and `af_ref_col` (ALT frequency in the reference panel);
           one record per position (QC'd SNVs).
    Returns (aligned table with columns of `study` + variant_id, af_ref, flipped, palindromic, af_flag,
    z (= beta/se after alignment)), and a log dict of counts.
    """
    log: dict = {"label": label, "n_in": int(len(study))}
    s = study.copy()
    s["chr"] = s["chr"].map(norm_chr)
    s = s[s["chr"].isin(AUTOSOMES)]
    log["n_non_autosomal"] = log["n_in"] - len(s)
    dup = s.duplicated(["chr", "pos"], keep=False).to_numpy()  # any record type (SNV, indel, multi-allelic split)
    log["n_multirecord"] = int(dup.sum())
    s = s[~dup]
    snv = is_snv(s["ref"], s["alt"])
    log["n_not_snv"] = int((~snv).sum())
    s = s[snv]
    if ref[["chr", "pos"]].duplicated().any():
        raise ValueError("reference table must have one record per position")
    m = s.merge(ref[["chr", "pos", "ref", "alt", "variant_id", af_ref_col]].rename(
        columns={"ref": "ref_ref", "alt": "ref_alt", af_ref_col: "af_ref"}), on=["chr", "pos"], how="inner")
    log["n_not_in_reference"] = int(len(s) - len(m))
    same = ((m["ref"] == m["ref_ref"]) & (m["alt"] == m["ref_alt"])).to_numpy()
    swap = ((m["ref"] == m["ref_alt"]) & (m["alt"] == m["ref_ref"])).to_numpy()
    pal = is_palindromic(m["ref_ref"], m["ref_alt"])
    log["n_allele_mismatch"] = int((~same & ~swap).sum())
    log["n_palindromic_swapped_dropped"] = int((swap & pal).sum())
    keep = same | (swap & ~pal)
    m = m[keep].copy()
    flip = swap[keep]
    pal = pal[keep]
    sign = np.where(flip, -1.0, 1.0)
    m["beta"] = m["beta"].to_numpy(float) * sign
    if "af" in m:
        af = m["af"].to_numpy(float)
        m["af"] = np.where(flip, 1.0 - af, af)
    m["ref"], m["alt"] = m["ref_ref"], m["ref_alt"]
    m["flipped"] = flip
    m["palindromic"] = pal
    log["n_flipped"] = int(flip.sum())
    d = np.abs(m["af"].to_numpy(float) - m["af_ref"].to_numpy(float)) if "af" in m else np.full(len(m), np.nan)
    m["daf"] = d
    with np.errstate(invalid="ignore"):
        pal_bad = pal & (d > AF_FLAG_PALINDROMIC)
        m["af_flag"] = d > AF_FLAG
    log["n_palindromic_af_dropped"] = int(pal_bad.sum())
    m = m[~pal_bad]
    log["n_af_flag_kept"] = int(m["af_flag"].sum())
    log["n_af_missing"] = int(np.isnan(m["daf"].to_numpy(float)).sum())
    m["z"] = m["beta"].to_numpy(float) / m["se"].to_numpy(float)
    bad = ~np.isfinite(m["z"].to_numpy(float))
    log["n_nonfinite_z"] = int(bad.sum())
    m = m[~bad]
    log["n_out"] = int(len(m))
    return m.drop(columns=["ref_ref", "ref_alt"]).sort_values("pos").reset_index(drop=True), log


def variant_key(v: str) -> str:
    """Normalise 'chr1_123_A_G', '1:123:A:G', 'chr1:123:A:G' -> 'chr1:123:A:G'."""
    parts = v.replace("_", ":").split(":")
    if len(parts) != 4:
        raise ValueError(f"bad variant id {v!r}")
    return f"{norm_chr(parts[0])}:{parts[1]}:{parts[2]}:{parts[3]}"


def same_site_any_orientation(a: str, b: str) -> bool:
    pa, pb = variant_key(a).split(":"), variant_key(b).split(":")
    return pa[:2] == pb[:2] and {pa[2], pa[3]} == {pb[2], pb[3]}


def lead_missingness(lead_ids: dict[str, str], harmonised_ids: set[str]) -> dict:
    """For each trait's lead variant: present in the final harmonised variant set (either allele orientation)."""
    keys = {":".join(variant_key(v).split(":")[:2]) + ":" + "".join(sorted(variant_key(v).split(":")[2:]))
            for v in harmonised_ids}
    out = {}
    for name, v in lead_ids.items():
        if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
            out[f"lead_present_{name}"] = False
            continue
        p = variant_key(v).split(":")
        out[f"lead_present_{name}"] = f"{p[0]}:{p[1]}:" + "".join(sorted(p[2:])) in keys
    return out


def check_build(study: pd.DataFrame, ref: pd.DataFrame, rsid_col: str | None = "rsid") -> dict:
    """GRCh38 check. (a) Among study SNVs at positions present in the GRCh38 reference table, the fraction
    whose (REF, ALT) equals the reference record, or its swap (a GRCh37 file would match at chance level);
    (b) known rsIDs found in the study at their GRCh38 positions."""
    s = study.copy()
    s["chr"] = s["chr"].map(norm_chr)
    s = s[is_snv(s["ref"], s["alt"])]
    m = s.merge(ref[["chr", "pos", "ref", "alt"]].rename(columns={"ref": "r_ref", "alt": "r_alt"}), on=["chr", "pos"])
    exact = ((m["ref"] == m["r_ref"]) & (m["alt"] == m["r_alt"])).to_numpy()
    swapped = ((m["ref"] == m["r_alt"]) & (m["alt"] == m["r_ref"])).to_numpy()
    ref_match = (m["ref"] == m["r_ref"]).to_numpy()
    out = {"n_shared_positions": int(len(m)),
           "frac_alleles_match": float((exact | swapped).mean()) if len(m) else float("nan"),
           "frac_ref_is_grch38_ref": float(ref_match.mean()) if len(m) else float("nan")}
    if rsid_col and rsid_col in s.columns:
        hits = {}
        for rs, (c, p) in KNOWN_RSIDS_GRCH38.items():
            r = s[s[rsid_col].astype(str).str.split(",").apply(lambda xs: rs in xs)]
            if len(r):
                hits[rs] = bool(((r["chr"] == c) & (r["pos"] == p)).all())
        out["known_rsids_at_grch38"] = hits
    return out
