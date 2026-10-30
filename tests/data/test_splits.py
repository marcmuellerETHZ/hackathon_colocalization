"""Audit of LD-block splits (DESIGN §3): toy known answers + full audit of the committed split files."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data.make_splits as M
from data.variants import OUT_DIR

ROOT = Path(__file__).resolve().parents[2]
SPL = ROOT / "datasets" / "splits"


# ----------------------------------------------------------------------------- toy known answers
def test_tile_toy():
    raw = pd.DataFrame({"chr": ["chr1"] * 4, "start": [500_000, 2_000_000, 2_400_000, 5_000_000],
                        "end": [2_000_000, 2_300_000, 3_000_000, 6_000_000]})
    t = M.tile_blocks(raw, {"chr1": 6_800_000})
    # leading 0.5 Mb absorbed; 0.1 Mb gap split at midpoint 2.35 Mb; 2 Mb gap -> own block;
    # trailing 0.8 Mb absorbed into the last block
    assert t[["start", "end"]].values.tolist() == [[0, 2_000_000], [2_000_000, 2_350_000],
                                                   [2_350_000, 3_000_000], [3_000_000, 5_000_000],
                                                   [5_000_000, 6_800_000]]
    assert t["source"].tolist() == ["pyrho", "pyrho", "pyrho", "gap", "pyrho"]
    t2 = M.tile_blocks(raw, {"chr1": 8_000_000})
    assert t2.iloc[-1][["start", "end", "source"]].tolist() == [6_000_000, 8_000_000, "gap_end"]
    raw3 = raw.copy(); raw3.loc[0, "start"] = 1_500_000; raw3.loc[0, "end"] = 2_000_000
    assert M.tile_blocks(raw3, {"chr1": 6_800_000}).iloc[0][["start", "end", "source"]].tolist() == [0, 1_500_000, "gap_start"]
    bad = pd.DataFrame({"chr": ["chr1"] * 2, "start": [0, 900], "end": [1000, 2000]})
    with pytest.raises(AssertionError):
        M.tile_blocks(bad, {"chr1": 5000})


def _toy_blocks(splits):
    n = len(splits)
    b = pd.DataFrame({"block_id": [f"chr1_{i:04d}" for i in range(n)], "chr": "chr1",
                      "start": np.arange(n) * 1_000_000, "end": (np.arange(n) + 1) * 1_000_000,
                      "split": splits})
    b["run"] = M.split_runs(b)
    return b


def test_clip_window_toy():
    b = _toy_blocks(["train", "test", "test", "valid", "excluded", "train"])
    # TSS at 2.0 Mb (block 1 = (1, 2] Mb; run = blocks 1-2 = (1, 3] Mb) -> unclipped [1.5 Mb, 2.5 Mb]
    assert M.clip_window(b, "chr1", 2_000_000) == (1_499_999, 2_500_000, "chr1_0001", "test")
    # TSS at 2.8 Mb -> upper clip at run end 3.0 Mb
    assert M.clip_window(b, "chr1", 2_800_000) == (2_299_999, 3_000_000, "chr1_0002", "test")
    # TSS at 1.1 Mb -> lower clip at run start 1.0 Mb
    assert M.clip_window(b, "chr1", 1_100_000)[:2] == (1_000_000, 1_600_000)
    # TSS at 3.9 Mb (block 3 valid, single-block run [3, 4) Mb)
    assert M.clip_window(b, "chr1", 3_900_000)[:2] == (3_399_999, 4_000_000)
    # a position equal to a block end belongs to that block (start < pos <= end)
    assert M.clip_window(b, "chr1", 1_000_000)[2] == "chr1_0000"
    assert M.clip_window(b, "chr1", 4_500_000) == (-1, -1, "chr1_0004", "excluded")
    assert list(M.split_runs(b)) == [0, 1, 1, 2, 3, 4]


def test_largest_remainder_counts():
    r = np.random.default_rng(0)
    for n in range(1, 60):
        c = M.largest_remainder_counts(n, M.FRACS, r)
        assert c.sum() == n and np.all(np.abs(c - np.array(M.FRACS) * n) < 1)


def test_assign_balances_toy():
    rng = np.random.default_rng(1)
    b = pd.DataFrame({"chr": np.repeat(["chr1", "chr2"], 50), "split": "",
                      "n_pc_genes": rng.poisson(10, 100)})
    lab, info = M.assign_splits(b, n_cand=300, seed=20261026)
    for c in ["chr1", "chr2"]:
        assert np.bincount(lab[(b.chr == c).to_numpy()], minlength=3).tolist() == [30, 10, 10]
    got = np.array([b.n_pc_genes[lab == k].sum() for k in range(3)])
    assert np.abs(got - np.array(M.FRACS) * b.n_pc_genes.sum()).max() <= info["median_loss"] ** 0.5
    lab2, _ = M.assign_splits(b, n_cand=300, seed=20261026)
    assert (lab == lab2).all()


# ----------------------------------------------------------------------------- audit of committed files
HAVE = (SPL / "blocks_v1.tsv").exists() and (SPL / "genes_v1.tsv.gz").exists()
pytestmark_real = pytest.mark.skipif(not HAVE, reason="split files not built")


@pytest.fixture(scope="module")
def blocks():
    return M.load_blocks(SPL / "blocks_v1.tsv")


@pytest.fixture(scope="module")
def genes():
    return M.load_genes(SPL / "genes_v1.tsv.gz")


@pytestmark_real
def test_tiling_no_gaps_no_overlaps(blocks):
    lengths = M.contig_lengths()
    assert blocks["block_id"].is_unique
    for c, d in blocks.groupby("chr", sort=False):
        assert d["start"].iloc[0] == 0 and d["end"].iloc[-1] == lengths[c]
        assert (d["start"].to_numpy()[1:] == d["end"].to_numpy()[:-1]).all()
        assert (d["start"] < d["end"]).all()
    assert set(blocks["chr"]) == set(M.CHROMS)


@pytestmark_real
def test_raw_blocks_preserved(blocks):
    raw = pd.read_csv(M.BLOCKS_BED, sep="\t"); raw.columns = ["chr", "start", "end"]
    assert (blocks["source"] == "pyrho").sum() == len(raw)
    # every raw block lies within exactly one tiled pyrho block (only gap edges can be moved outwards)
    for c, d in raw.groupby("chr"):
        t = blocks[(blocks.chr == c) & (blocks.source == "pyrho")]
        assert (t["start"].to_numpy() <= d["start"].to_numpy()).all()
        assert (t["end"].to_numpy() >= d["end"].to_numpy()).all()


@pytestmark_real
def test_mhc_excluded(blocks, genes):
    mhc = M.overlaps_mhc(blocks)
    assert mhc.sum() > 0 and (blocks.loc[mhc, "split"] == "excluded").all()
    assert (blocks.loc[~mhc, "split"].isin(M.SPLITS)).all()
    ex = genes[genes.split == "excluded"]
    assert ((ex.window_start == -1) & (ex.window_end == -1)).all()
    assert ex["block_id"].isin(blocks.loc[mhc, "block_id"]).all()


@pytestmark_real
def test_genes_unique_and_in_tss_block(blocks, genes):
    assert genes["gene_id"].is_unique
    bmap = blocks.set_index("block_id")
    for c, d in genes.groupby("chr"):
        idx = M.locate(blocks, c, d["tss"].to_numpy())
        assert (idx >= 0).all()
        assert (blocks["block_id"].to_numpy()[idx] == d["block_id"].to_numpy()).all()
    assert (genes["split"].to_numpy() == bmap.loc[genes["block_id"], "split"].to_numpy()).all()
    # each gene is in exactly one split by construction (one row per gene); and protein-coding counts
    pc = blocks.groupby("split")["n_pc_genes"].sum()
    gpc = genes[genes.gene_type == "protein_coding"].groupby("split").size()
    assert (pc == gpc.reindex(pc.index)).all()


@pytestmark_real
def test_windows_contain_tss_and_lie_in_one_split_run(blocks, genes):
    g = genes[genes.split != "excluded"]
    assert ((g.window_start < g.tss) & (g.tss <= g.window_end)).all(), "window must contain its TSS"
    assert ((g.tss - g.window_start) <= M.HALF_WINDOW + 1).all() and ((g.window_end - g.tss) <= M.HALF_WINDOW).all()
    bl = blocks.set_index("block_id")
    for c, d in g.groupby("chr"):
        bc = blocks[blocks.chr == c]
        s, e = bc["start"].to_numpy(), bc["end"].to_numpy()
        runs, splits = bc["run"].to_numpy(), bc["split"].to_numpy()
        for ws, we, bid, sp in zip(d.window_start, d.window_end, d.block_id, d.split):
            ov = (s < we) & (e > ws)  # blocks overlapping BED window [ws, we)
            assert (splits[ov] == sp).all() and len(set(runs[ov])) == 1
            assert runs[ov][0] == bl.loc[bid, "run"]


@pytestmark_real
def test_windows_maximal(genes, blocks):
    g = genes[genes.split != "excluded"].copy()
    lo = g.tss - 1 - M.HALF_WINDOW
    hi = g.tss + M.HALF_WINDOW
    runs = blocks.groupby("run").agg(rs=("start", "min"), re=("end", "max"))
    rid = blocks.set_index("block_id").loc[g.block_id, "run"].to_numpy()
    rs, re_ = runs.loc[rid, "rs"].to_numpy(), runs.loc[rid, "re"].to_numpy()
    assert (g.window_start.to_numpy() == np.maximum(lo, rs)).all()
    assert (g.window_end.to_numpy() == np.minimum(hi, re_)).all()


@pytestmark_real
def test_split_balance_and_counts(blocks, genes):
    el = blocks[blocks.split != "excluded"]
    n = el.groupby("split").size()
    frac = n / n.sum()
    assert np.allclose(frac[list(M.SPLITS)].to_numpy(), M.FRACS, atol=0.02)
    # stratified by chromosome: per chromosome, counts within 1 of 60/20/20
    for c, d in el.groupby("chr"):
        k = d.groupby("split").size().reindex(M.SPLITS, fill_value=0).to_numpy()
        assert np.all(np.abs(k - np.array(M.FRACS) * len(d)) < 1), c
    pc = el.groupby("split")["n_pc_genes"].sum()
    pcf = pc / pc.sum()
    assert np.allclose(pcf[list(M.SPLITS)].to_numpy(), M.FRACS, atol=0.005)
    print("\nblocks per split:", n.to_dict(), "pc genes:", pc.to_dict(),
          "all genes:", genes.groupby("split").size().to_dict())


VARS = sorted(OUT_DIR.glob("chr*.parquet"))


@pytestmark_real
@pytest.mark.skipif(not VARS, reason="no variant files")
@pytest.mark.parametrize("path", VARS, ids=[p.stem for p in VARS])
def test_every_variant_in_exactly_one_block_and_windows_single_split(path, blocks, genes):
    v = pd.read_parquet(path, columns=["chr", "pos"])
    c = path.stem
    bc = blocks[blocks.chr == c]
    idx = M.locate(blocks, c, v["pos"].to_numpy())
    assert (idx >= 0).all()
    # exactly one: blocks are a partition (no overlaps), so count containing blocks directly on a sample
    sample = v["pos"].to_numpy()[:: max(1, len(v) // 2000)]
    cnt = ((bc["start"].to_numpy()[None, :] < sample[:, None]) & (sample[:, None] <= bc["end"].to_numpy()[None, :])).sum(1)
    assert (cnt == 1).all()
    vsplit = blocks["split"].to_numpy()[idx]
    pos = v["pos"].to_numpy()
    g = genes[(genes.chr == c) & (genes.split != "excluded")]
    for ws, we, sp in zip(g.window_start, g.window_end, g.split):
        a, b = np.searchsorted(pos, ws, "right"), np.searchsorted(pos, we, "right")
        assert (vsplit[a:b] == sp).all()


@pytestmark_real
def test_sha256sums_match_files():
    for line in (SPL / "SHA256SUMS").read_text().splitlines():
        h, f = line.split()
        assert hashlib.sha256((SPL / f).read_bytes()).hexdigest() == h, f


@pytestmark_real
def test_reproducible(tmp_path):
    """Re-running make_splits from the raw inputs gives byte-identical files."""
    M.write(tmp_path)
    for f in ["blocks_v1.tsv", "genes_v1.tsv.gz"]:
        assert hashlib.sha256((tmp_path / f).read_bytes()).hexdigest() == \
            hashlib.sha256((SPL / f).read_bytes()).hexdigest(), f


@pytestmark_real
def test_audit_clean_and_negative_controls(blocks, genes):
    assert M.audit_windows(blocks, genes) == {"tss_outside": 0, "cross_split": 0, "cross_run": 0,
                                              "split_mismatch": 0}
    # negative control 1: flip the split of one block inside a multi-block run -> windows now leak
    run_sizes = blocks.groupby("run").size()
    big = run_sizes[run_sizes >= 3].index[0]
    i = blocks.index[blocks.run == big][1]
    bad = blocks.copy()
    bad.loc[i, "split"] = "test" if bad.loc[i, "split"] != "test" else "train"
    a = M.audit_windows(bad, genes)
    assert a["cross_split"] > 0 and a["split_mismatch"] > 0
    # negative control 2: widen one window by 1 Mb -> crosses into another split
    g2 = genes.copy()
    j = g2.index[(g2.split != "excluded") & (g2.window_end - g2.window_start < 900_000)][0]
    g2.loc[j, "window_end"] += 1_000_000
    assert M.audit_windows(blocks, g2)["cross_split"] == 1
    # negative control 3: window shifted off its TSS
    g3 = genes.copy()
    g3.loc[j, ["window_start", "window_end"]] = [g3.loc[j, "tss"] + 10, g3.loc[j, "tss"] + 20]
    assert M.audit_windows(blocks, g3)["tss_outside"] == 1
