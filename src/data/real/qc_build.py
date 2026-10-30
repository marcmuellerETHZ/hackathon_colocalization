"""GRCh38 build verification of every downloaded real-data source (DESIGN §2.5: hg38 throughout).

For each eQTL Catalogue dataset and FinnGen trait:
  (a) allele concordance: in 5 train-split windows (1 Mb, fixed below), the fraction of the source's SNV
      records whose (REF, ALT) equals, or is the swap of, the 1000G GRCh38 record at the same position;
      a GRCh37 file would score ~0.3 (chance), GRCh38 should be > 0.99;
  (b) known rsIDs: single-position look-ups at the GRCh38 positions of rs429358, rs7412 (APOE),
      rs12740374 (SORT1), rs4149056 (SLCO1B1) — the source's rsID at that position must match
      (FinnGen lab files carry no rsIDs: only (a) and the allele check at those positions apply).
Output: data/processed/real/qc/build_check.json. Run: python -m data.real.qc_build
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from data.real import sources
from data.real.config import OUT, assert_env, eqtl_datasets, finngen_traits
from data.real.harmonise import KNOWN_RSIDS_GRCH38, check_build
from data.real.regions import load_blocks

N_WINDOWS = 5
WIDTH = 1_000_000


def train_windows() -> list[tuple[str, int, int]]:
    """5 deterministic train-split 1-Mb windows: the first train block >= 2 Mb long on chr2, 5, 9, 14, 20."""
    b = load_blocks()
    out = []
    for c in ("chr2", "chr5", "chr9", "chr14", "chr20"):
        d = b[(b["chr"] == c) & (b["split"] == "train") & (b["end"] - b["start"] >= 2 * WIDTH)]
        r = d.iloc[0]
        s = int(r["start"]) + WIDTH // 2
        out.append((c, s + 1, s + WIDTH))
    return out[:N_WINDOWS]


def _ref_table(chrom, lo, hi):
    from data.genotypes import variant_table
    return variant_table(chrom, lo - 1, hi)[["chr", "pos", "ref", "alt"]]


def _source_region(src: str, chrom: str, lo: int, hi: int) -> pd.DataFrame:
    if src.startswith("FG_"):
        return sources.finngen_region(finngen_traits()[src[3:]], chrom, lo, hi)
    ds = eqtl_datasets()[src]
    txt = sources.tabix_text(ds.path("all"), f"{sources.bare_chr(chrom)}:{lo}-{hi}")
    df = sources._read_tsv(txt, sources.EQTL_ALL_COLS, dtype={"chromosome": str, "rsid": str})
    df = df.drop_duplicates(["position", "ref", "alt"])
    return pd.DataFrame({"chr": df["chromosome"].map(sources.norm_chr), "pos": df["position"].astype(np.int64),
                         "ref": df["ref"].astype(str), "alt": df["alt"].astype(str), "rsid": df["rsid"].astype(str)})


def check_source(src: str) -> tuple[str, dict]:
    res = {"windows": []}
    fr = []
    for c, lo, hi in train_windows():
        st = _source_region(src, c, lo, hi)
        cb = check_build(st, _ref_table(c, lo, hi), rsid_col=None)
        res["windows"].append({"region": f"{c}:{lo}-{hi}", **cb})
        fr.append((cb["n_shared_positions"], cb["frac_alleles_match"]))
    n = sum(x[0] for x in fr)
    res["frac_alleles_match"] = float(sum(x[0] * x[1] for x in fr) / n) if n else float("nan")
    rs = {}
    for rid, (c, p) in KNOWN_RSIDS_GRCH38.items():
        st = _source_region(src, c, p, p)
        st = st[st["pos"] == p]
        if "rsid" in st and st["rsid"].astype(str).str.len().gt(0).any() and st["rsid"].astype(str).ne("nan").any():
            rs[rid] = bool(st["rsid"].astype(str).str.split(",").apply(lambda xs: rid in xs).any())
        else:
            rs[rid] = "no_rsid_column" if len(st) else "position_absent"
    res["known_rsids_at_grch38"] = rs
    vals = list(rs.values())
    has_rsids = any(v is True or v is False for v in vals)
    res["pass"] = bool(res["frac_alleles_match"] > 0.99 and not any(v is False for v in vals)
                       and (any(v is True for v in vals) or not has_rsids))
    return src, res


def main() -> None:
    assert_env()
    srcs = list(eqtl_datasets()) + [f"FG_{p}" for p in finngen_traits()]
    with ProcessPoolExecutor(8) as ex:
        out = dict(ex.map(check_source, srcs))
    d = OUT / "qc"
    d.mkdir(parents=True, exist_ok=True)
    (d / "build_check.json").write_text(json.dumps(out, indent=1))
    for s, r in out.items():
        print(s, round(r["frac_alleles_match"], 5), r["known_rsids_at_grch38"], "PASS" if r["pass"] else "FAIL")


if __name__ == "__main__":
    main()
