"""LD-block tiling, block -> split assignment, gene table and clipped windows (DESIGN §3).

Coordinates: blocks and windows use BED convention (0-based start, exclusive end). A 1-based
position x (VCF/pvar POS, GTF TSS) lies in [start, end) iff start < x <= end.

Outputs (committed):
  datasets/splits/blocks_v1.tsv  block_id chr start end split n_pc_genes n_genes source
  datasets/splits/genes_v1.tsv.gz  (gzip, mtime 0: deterministic bytes; plain TSV would exceed the 5 MB commit limit)
                        gene_id gene_name gene_type chr tss strand block_id split window_start window_end
  datasets/splits/SHA256SUMS

split in {train, valid, test, excluded}; "excluded" = blocks overlapping the MHC (and genes whose
TSS falls in them; their window is empty, window_start = window_end = -1).

Usage: python -m data.make_splits [--outdir splits]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common.seeds import SPLIT_SEED, rng

ROOT = Path(__file__).resolve().parents[2]
BLOCKS_BED = ROOT / "data/raw/annot/pyrho_EUR_LD_blocks.bed"
GTF = ROOT / "data/raw/annot/gencode.v39.basic.annotation.gtf.gz"
RAW_VCF = ROOT / "data/raw/1kg30x/chr{c}.vcf.gz"
CHROMS = [f"chr{i}" for i in range(1, 23)]
MHC = ("chr6", 28_500_000, 33_400_000)  # 1-based closed region, DESIGN §2.1
GAP_OWN_BLOCK = 1_000_000  # gaps >= 1 Mb become their own block
SPLITS = ("train", "valid", "test")
FRACS = (0.6, 0.2, 0.2)
N_CANDIDATES = 5000  # random stratified assignments searched for protein-coding balance
HALF_WINDOW = 500_000
VERSION = "splits_v1"


# ----------------------------------------------------------------------------- inputs
def contig_lengths(chroms=CHROMS) -> dict[str, int]:
    """Contig lengths from the (checksummed) raw 1000G VCF headers."""
    out = {}
    pat = re.compile(r"##contig=<ID=(chr\w+),length=(\d+)")
    for c in chroms:
        with gzip.open(str(RAW_VCF).format(c=c[3:]), "rt") as fh:
            for line in fh:
                if not line.startswith("##"):
                    break
                m = pat.match(line)
                if m and m.group(1) == c:
                    out[c] = int(m.group(2))
        assert c in out, f"no contig length for {c}"
    return out


def read_genes(gtf: Path = GTF) -> pd.DataFrame:
    rows = []
    with gzip.open(gtf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] != "gene" or f[0] not in CHROMS:
                continue
            attr = dict(re.findall(r'(\S+) "([^"]*)"', f[8]))
            start, end, strand = int(f[3]), int(f[4]), f[6]
            rows.append((attr["gene_id"], attr["gene_name"], attr["gene_type"], f[0],
                         start if strand == "+" else end, strand))
    g = pd.DataFrame(rows, columns=["gene_id", "gene_name", "gene_type", "chr", "tss", "strand"])
    assert g["gene_id"].is_unique
    return g


# ----------------------------------------------------------------------------- tiling
def tile_blocks(raw: pd.DataFrame, lengths: dict[str, int], gap_own: int = GAP_OWN_BLOCK) -> pd.DataFrame:
    """Make blocks tile [0, contig length) of each chromosome.
    Inter-block gap >= gap_own -> own block; smaller gap -> split at its midpoint (integer floor).
    Leading/trailing stretches (before first / after last block) follow the same rule: >= gap_own
    -> own block, otherwise absorbed into the first/last block. Overlapping raw blocks are an error."""
    out = []
    for c in [c for c in CHROMS if c in set(raw["chr"])]:
        d = raw[raw["chr"] == c].sort_values("start")
        s, e = d["start"].to_numpy().copy(), d["end"].to_numpy().copy()
        assert (s < e).all() and (s[1:] >= e[:-1]).all(), f"overlapping raw blocks on {c}"
        src = ["pyrho"] * len(s)
        starts, ends, srcs = [], [], []
        L = lengths[c]
        assert e[-1] <= L
        if s[0] > 0:
            if s[0] >= gap_own:
                starts.append(0); ends.append(int(s[0])); srcs.append("gap_start")
            else:
                s[0] = 0
        for i in range(len(s)):
            if i > 0 and s[i] > e[i - 1]:
                gap = s[i] - e[i - 1]
                if gap >= gap_own:
                    starts.append(int(e[i - 1])); ends.append(int(s[i])); srcs.append("gap")
                else:
                    mid = int(e[i - 1] + gap // 2)
                    ends[-1] = mid
                    s[i] = mid
            starts.append(int(s[i])); ends.append(int(e[i])); srcs.append(src[i])
        if ends[-1] < L:
            if L - ends[-1] >= gap_own:
                starts.append(ends[-1]); ends.append(L); srcs.append("gap_end")
            else:
                ends[-1] = L
        out.append(pd.DataFrame({"chr": c, "start": starts, "end": ends, "source": srcs}))
    t = pd.concat(out, ignore_index=True)
    t.insert(0, "block_id", [f"{c}_{i:04d}" for c, n in t.groupby("chr", sort=False).size().items()
                             for i in range(n)])
    return t


def overlaps_mhc(blocks: pd.DataFrame) -> np.ndarray:
    c, lo, hi = MHC  # 1-based closed [lo, hi]  <->  BED [lo-1, hi)
    return ((blocks["chr"] == c) & (blocks["start"] < hi) & (blocks["end"] > lo - 1)).to_numpy()


def locate(blocks: pd.DataFrame, chrom: str, pos) -> np.ndarray:
    """Row index (into `blocks`) of the block containing 1-based position(s) pos on chrom; -1 if none."""
    d = blocks[blocks["chr"] == chrom]
    pos = np.atleast_1d(np.asarray(pos, dtype=np.int64))
    k = np.searchsorted(d["end"].to_numpy(), pos, side="left")  # first block with end >= pos
    ok = (k < len(d))
    kk = np.where(ok, k, 0)
    ok &= d["start"].to_numpy()[kk] < pos
    return np.where(ok, d.index.to_numpy()[kk], -1)


# ----------------------------------------------------------------------------- assignment
def largest_remainder_counts(n: int, fracs, r: np.random.Generator) -> np.ndarray:
    q = np.array(fracs) * n
    base = np.floor(q).astype(int)
    rest = n - base.sum()
    frac = q - base
    # random tie-break: sort by (-remainder, random key)
    order = np.lexsort((r.random(len(q)), -np.round(frac, 12)))
    base[order[:rest]] += 1
    return base


def assign_splits(blocks: pd.DataFrame, n_cand: int = N_CANDIDATES, seed: int = SPLIT_SEED) -> tuple[np.ndarray, dict]:
    """Seeded random assignment of eligible blocks to train/valid/test, stratified by chromosome
    (per-chromosome counts by largest remainder of 60/20/20), choosing among n_cand random
    candidates the one whose global protein-coding gene shares are closest to 60/20/20
    (min sum of squared deviations of pc-gene counts; ties -> earliest candidate)."""
    elig = blocks["split"].ne("excluded").to_numpy()
    pc = blocks["n_pc_genes"].to_numpy()
    r = rng(seed, VERSION, "counts")
    per_chr = {}
    for c in CHROMS:
        idx = np.flatnonzero(elig & (blocks["chr"] == c).to_numpy())
        if len(idx):
            per_chr[c] = (idx, largest_remainder_counts(len(idx), FRACS, r))
    target = np.array(FRACS) * pc[elig].sum()
    r = rng(seed, VERSION, "candidates")
    best, best_loss, losses = None, np.inf, []
    for _ in range(n_cand):
        lab = np.full(len(blocks), -1)
        for c, (idx, cnt) in per_chr.items():
            lab[idx] = np.repeat(np.arange(3), cnt)[r.permutation(len(idx))]
        got = np.array([pc[lab == k].sum() for k in range(3)])
        loss = float(((got - target) ** 2).sum())
        losses.append(loss)
        if loss < best_loss:
            best, best_loss = lab, loss
    info = {"n_candidates": n_cand, "best_loss": best_loss, "median_loss": float(np.median(losses)),
            "target_pc": target.tolist()}
    return best, info


# ----------------------------------------------------------------------------- windows
def split_runs(blocks: pd.DataFrame) -> np.ndarray:
    """Run id per block: maximal runs of consecutive same-split blocks within a chromosome."""
    new = (blocks["chr"] != blocks["chr"].shift()) | (blocks["split"] != blocks["split"].shift())
    return np.cumsum(new.to_numpy()) - 1


def clip_window(blocks: pd.DataFrame, chrom: str, pos: int, half: int = HALF_WINDOW) -> tuple[int, int, str, str]:
    """Window around 1-based `pos` (TSS or lead): BED [pos-1-half, pos+half) (i.e. 1-based
    pos-half..pos+half), clipped to the maximal run of same-split blocks containing pos.
    Returns (start, end, block_id, split); (-1, -1, block_id, 'excluded') for excluded blocks."""
    i = int(locate(blocks, chrom, pos)[0])
    if i < 0:
        raise ValueError(f"{chrom}:{pos} not in any block")
    b = blocks.loc[i]
    if b["split"] == "excluded":
        return -1, -1, b["block_id"], "excluded"
    run = blocks.loc[blocks["run"] == b["run"]] if "run" in blocks else None
    if run is None:
        rr = split_runs(blocks)
        run = blocks[rr == rr[blocks.index.get_loc(i)]]
    lo, hi = int(run["start"].min()), int(run["end"].max())
    return max(pos - 1 - half, lo), min(pos + half, hi), b["block_id"], b["split"]


# ----------------------------------------------------------------------------- main
def build(n_cand: int = N_CANDIDATES, seed: int = SPLIT_SEED) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(BLOCKS_BED, sep="\t")
    raw.columns = ["chr", "start", "end"]
    blocks = tile_blocks(raw, contig_lengths())
    genes = read_genes()
    bi = np.concatenate([locate(blocks, c, d["tss"].to_numpy()) for c, d in genes.groupby("chr", sort=False)])
    genes = pd.concat([d for _, d in genes.groupby("chr", sort=False)], ignore_index=True)
    assert (bi >= 0).all(), "gene TSS outside the tiling"
    genes["block_id"] = blocks["block_id"].to_numpy()[bi]
    pc_mask = genes["gene_type"].eq("protein_coding")
    blocks["n_pc_genes"] = blocks["block_id"].map(genes.loc[pc_mask, "block_id"].value_counts()).fillna(0).astype(int)
    blocks["n_genes"] = blocks["block_id"].map(genes["block_id"].value_counts()).fillna(0).astype(int)
    blocks["split"] = np.where(overlaps_mhc(blocks), "excluded", "")
    lab, info = assign_splits(blocks, n_cand, seed)
    blocks.loc[lab >= 0, "split"] = np.array(SPLITS)[lab[lab >= 0]]
    assert blocks["split"].isin(SPLITS + ("excluded",)).all()
    blocks["run"] = split_runs(blocks)
    win = [clip_window(blocks, c, int(t)) for c, t in zip(genes["chr"], genes["tss"])]
    genes["split"] = [w[3] for w in win]
    assert (genes["block_id"].to_numpy() == np.array([w[2] for w in win])).all()
    genes["window_start"] = [w[0] for w in win]
    genes["window_end"] = [w[1] for w in win]
    chrom_order = {c: i for i, c in enumerate(CHROMS)}
    genes = genes.sort_values(["chr", "tss", "gene_id"], key=lambda s: s.map(chrom_order) if s.name == "chr" else s,
                              kind="mergesort").reset_index(drop=True)
    blocks_out = blocks[["block_id", "chr", "start", "end", "split", "n_pc_genes", "n_genes", "source"]]
    return blocks_out, genes, info


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(outdir: Path, n_cand: int = N_CANDIDATES) -> dict:
    blocks, genes, info = build(n_cand)
    outdir.mkdir(parents=True, exist_ok=True)
    blocks.to_csv(outdir / "blocks_v1.tsv", sep="\t", index=False)
    with open(outdir / "genes_v1.tsv.gz", "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
        gz.write(genes.to_csv(sep="\t", index=False).encode())
    files = ["blocks_v1.tsv", "genes_v1.tsv.gz", "pools_v1.tsv"]
    with open(outdir / "SHA256SUMS", "w") as fh:
        for f in files:
            if (outdir / f).exists():
                fh.write(f"{sha256(outdir / f)}  {f}\n")
    return info


def audit_windows(blocks: pd.DataFrame, genes: pd.DataFrame) -> dict[str, int]:
    """Count violations (all should be 0): window not containing its TSS, window overlapping a block
    of another split or of another run, gene split != split of its TSS block."""
    b = blocks.copy()
    b["run"] = split_runs(b)
    g = genes[genes["split"] != "excluded"]
    bad = {"tss_outside": int((~((g.window_start < g.tss) & (g.tss <= g.window_end))).sum()),
           "cross_split": 0, "cross_run": 0,
           "split_mismatch": int((genes["split"].to_numpy() !=
                                  b.set_index("block_id").loc[genes["block_id"], "split"].to_numpy()).sum())}
    for c, d in g.groupby("chr"):
        bc = b[b.chr == c]
        s, e, sp, rn = (bc[k].to_numpy() for k in ("start", "end", "split", "run"))
        for ws, we, gs in zip(d.window_start, d.window_end, d.split):
            ov = (s < we) & (e > ws)
            bad["cross_split"] += int((sp[ov] != gs).any())
            bad["cross_run"] += int(len(set(rn[ov])) > 1)
    return bad


def load_blocks(path: Path | str = ROOT / "datasets/splits/blocks_v1.tsv") -> pd.DataFrame:
    b = pd.read_csv(path, sep="\t")
    b["run"] = split_runs(b)
    return b


def load_genes(path: Path | str = ROOT / "datasets/splits/genes_v1.tsv.gz") -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=ROOT / "datasets" / "splits", type=Path)
    ap.add_argument("--n-candidates", default=N_CANDIDATES, type=int)
    a = ap.parse_args()
    info = write(a.outdir, a.n_candidates)
    print(info)


if __name__ == "__main__":
    main()
