"""Tests for src/data/genotypes.py: toy known-answer phase/dosage test, real data vs raw VCF."""
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import data.genotypes as G
from data.variants import OUT_DIR, run_chrom
from conftest import ROOT, TOY, SAMPLES


@pytest.fixture()
def toy_loader(toy_pgen, monkeypatch):
    t = toy_pgen
    run_chrom(6, t["pgen_dir"], t["out_dir"], t["multirec_dir"], t["pools"])
    monkeypatch.setattr(G, "PGEN_DIR", t["pgen_dir"])
    monkeypatch.setattr(G, "OUT_DIR", t["out_dir"])
    G._variants.cache_clear(); G._psam.cache_clear()
    yield t
    G._variants.cache_clear(); G._psam.cache_clear()


def _expected_haps(samples, positions):
    gts = {p: g for p, _, _, g, keep in TOY if keep}
    H = np.zeros((2 * len(samples), len(positions)), dtype=np.int8)
    for j, p in enumerate(positions):
        for i, s in enumerate(samples):
            a, b = gts[p][SAMPLES.index(s)].split("|")
            H[2 * i, j], H[2 * i + 1, j] = int(a), int(b)
    return H


def test_toy_haplotypes_and_dosages_known_answer(toy_loader):
    samples = ["S4", "S1", "S6", "S2"]  # deliberately not in psam order
    H, v = G.load_haplotypes("chr6", 0, 10_000, samples)
    assert v["pos"].tolist() == [1000, 6000, 7000]
    np.testing.assert_array_equal(H, _expected_haps(samples, [1000, 6000, 7000]))
    D, v2 = G.load_genotypes("chr6", 0, 10_000, samples)
    assert D.dtype == np.int8 and D.shape == (4, 3) and v2.equals(v)
    np.testing.assert_array_equal(D, H[0::2] + H[1::2])


def test_toy_region_convention_is_bed(toy_loader):
    # start < pos <= end
    assert G.variant_table("chr6", 1000, 7000)["pos"].tolist() == [6000, 7000]
    assert G.variant_table("chr6", 999, 6000)["pos"].tolist() == [1000, 6000]
    assert len(G.variant_table("chr6", 28_000_000, 33_400_000)) == 1  # 28,499,999 only
    D, v = G.load_genotypes("chr6", 100, 200, ["S1"])
    assert D.shape == (1, 0) and len(v) == 0


def test_toy_maf_filter_and_errors(toy_loader):
    # pos 33,400,001: S1 1|1, S2 0|1 -> af_A = 3/4 (MAF .25); af_FIN = 0
    v = G.variant_table("chr6", 0, 40_000_000, maf_pool="FIN", min_maf=0.01)
    assert 33_400_001 not in v["pos"].tolist() and 6000 in v["pos"].tolist()
    with pytest.raises(KeyError):
        G.load_genotypes("chr6", 0, 10_000, ["S1", "NOPE"])
    with pytest.raises(ValueError):
        G.load_genotypes("chr6", 0, 10_000, ["S1", "S1"])
    with pytest.raises(ValueError):
        G.variant_table("chr6", 0, 10_000, min_maf=0.01)


BCFTOOLS = shutil.which("bcftools") or str(ROOT / "env/tools/bin/bcftools")
RAW22 = ROOT / "data/raw/1kg30x/chr22.vcf.gz"


@pytest.mark.skipif(not ((OUT_DIR / "chr22.parquet").exists() and RAW22.exists()), reason="real chr22 missing")
def test_real_haplotypes_match_raw_vcf():
    pools = pd.read_csv(ROOT / "datasets/splits/pools_v1.tsv", sep="\t")
    samples = pools.loc[pools.pool == "A", "sample_id"].tolist()[::-1][:60] + \
        pools.loc[pools.pool == "ASW_ACB", "sample_id"].tolist()[:20]
    start, end = 30_000_000, 30_100_000
    H, v = G.load_haplotypes("chr22", start, end, samples)
    D, _ = G.load_genotypes("chr22", start, end, samples)
    assert len(v) > 200
    out = subprocess.run([BCFTOOLS, "query", "-r", f"chr22:{start + 1}-{end}", "-s", ",".join(samples),
                          "-f", "%CHROM:%POS:%REF:%ALT[\t%GT]\n", str(RAW22)], check=True, capture_output=True, text=True).stdout
    raw = {}
    for line in out.splitlines():
        f = line.split("\t")
        raw[f[0]] = f[1:]
    # bcftools -s keeps the requested order
    for j, vid in enumerate(v["variant_id"]):
        gts = raw[vid]
        exp = np.array([[int(a) for a in g.split("|")] for g in gts], dtype=np.int8).reshape(-1)
        np.testing.assert_array_equal(H[:, j], exp, err_msg=vid)
    np.testing.assert_array_equal(D, H[0::2] + H[1::2])
    # phase is informative: haplotype pairs are not sorted (some 1|0 and some 0|1 hets)
    het = H[0::2] != H[1::2]
    assert (H[0::2][het] == 1).any() and (H[1::2][het] == 1).any()


@pytest.mark.skipif(not (OUT_DIR / "chr22.parquet").exists(), reason="real chr22 missing")
def test_real_af_consistent_with_loader():
    pools = pd.read_csv(ROOT / "datasets/splits/pools_v1.tsv", sep="\t")
    s = pools.loc[pools.pool == "B", "sample_id"].tolist()
    D, v = G.load_genotypes("chr22", 25_000_000, 26_000_000, s)
    np.testing.assert_allclose(D.mean(0) / 2, v["af_B"].to_numpy(), atol=1e-9)
