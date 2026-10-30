"""Toy and small real-data tests for ld.py, geuvadis.py and extract.py helpers."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.real import extract, geuvadis, ld

ROOT = Path(__file__).resolve().parents[2]
HAVE_1KG = (ROOT / "data/processed/variants/chr22.parquet").exists() and (ROOT / "datasets/splits/pools_v1.tsv").exists()


def test_finngen_r2_parsing(monkeypatch):
    rows = pd.DataFrame({"chr": "22", "pos": 100, "variant1": ["chr22_100_A_G"] * 2,
                         "variant2": ["chr22_200_C_T", "chr22_300_G_A"], "r": [0.5, -0.2], "r2": [0.25, 0.04]})
    monkeypatch.setattr(ld, "finngen_rows", lambda v: rows)
    assert ld.finngen_r2("chr22:100:A:G", "chr22:200:C:T") == (0.25, "found")
    assert ld.finngen_r2("chr22_100_G_A", "chr22_300_A_G") == (0.04, "found")   # orientation-insensitive
    assert ld.finngen_r2("chr22:100:A:G", "chr22:400:A:G") == (0.0, "below_0.01")
    r, s = ld.finngen_r2("chr22:100:A:T", "chr22:200:C:T")                      # other allele at same pos
    assert np.isnan(r) and s == "v1_absent"


def test_standardize_gives_correlation():
    rng = np.random.default_rng(20261026)
    G = rng.binomial(2, [0.1, 0.3, 0.5, 0.0], size=(200, 4))
    X, af, sd = ld.standardize(G)
    R = X[:, :3].T @ X[:, :3] / 200
    np.testing.assert_allclose(R, np.corrcoef(G[:, :3].T), atol=1e-12)
    assert sd[3] == 0 and np.all(X[:, 3] == 0)
    np.testing.assert_allclose(af[:3], G[:, :3].mean(0) / 2)


def test_std_beta_se_known_answer():
    b, s = extract._std_beta_se(np.array([2.0]), np.array([1000.0]), np.array([0.5]))
    assert s[0] == pytest.approx(1 / np.sqrt(500.0)) and b[0] == pytest.approx(2 / np.sqrt(500.0))
    # z is preserved exactly
    assert (b / s)[0] == pytest.approx(2.0)


def test_residualise_orthogonal():
    rng = np.random.default_rng(20261027)
    C = rng.normal(size=(100, 3))
    G = rng.binomial(2, 0.3, size=(100, 5)) + C[:, :1] * 0.5
    E = geuvadis.residualise(G, C)
    A = np.column_stack([np.ones(100), C])
    np.testing.assert_allclose(A.T @ E, 0, atol=1e-9)


def test_extract_refuses_test_split():
    with pytest.raises(PermissionError):
        extract.extract_pair({"locus_id": "x", "split": "test", "chr": "chr1", "win_lo": 1, "win_hi": 2}, ["poolB"])
    with pytest.raises(PermissionError):
        extract.run_arm("R2", "test")


def test_io_guard_refuses_test_locus(tmp_path):
    from eval.io import LocusData, TestSplitAccessError, write_locus_set
    p = 3
    loc = LocusData(locus_id="t", variant_id=np.array(["a", "b", "c"], dtype=object), pos=np.arange(p),
                    z1=np.zeros(p), z2=np.zeros(p), beta1=np.zeros(p), se1=np.ones(p), beta2=np.zeros(p),
                    se2=np.ones(p), n1=np.ones(p), n2=np.ones(p), maf1=np.ones(p) * .1, maf2=np.ones(p) * .1,
                    maf_ref=np.ones(p) * .1, ld_type="factor", n_ref=2, X_ref=np.ones((2, p), np.float32), split="test")
    with pytest.raises(TestSplitAccessError):
        write_locus_set(tmp_path / "x.h5", [loc])


@pytest.mark.skipif(not HAVE_1KG, reason="1000G processed data not available")
def test_panel_r2_real_chr22():
    from data.genotypes import variant_table
    v = variant_table("chr22", 20_000_000, 20_200_000)
    af = v["af_B"].to_numpy()
    v = v[(af > 0.2) & (af < 0.8)].head(2)
    a, b = v["variant_id"].tolist()
    r = ld.panel_r2("chr22", [(a, a), (a, b), (b, a)], ld.pools()["B"])
    assert r[0] == pytest.approx(1.0)
    assert r[1] == pytest.approx(r[2]) and 0 <= r[1] <= 1
    assert np.isnan(ld.panel_r2("chr22", [(a, "chr22:20000001:A:C")], ld.pools()["B"])[0])


@pytest.mark.skipif(not HAVE_1KG, reason="1000G processed data not available")
def test_geuvadis_samples_match_eqtl_catalogue_counts():
    s = pd.read_csv(ROOT / "data/processed/real/geuvadis/samples.tsv", sep="\t") \
        if (ROOT / "data/processed/real/geuvadis/samples.tsv").exists() else geuvadis.derive_samples()
    assert len(s) == 445
    assert (s["super_pop"] == "EUR").sum() == 358 and (s["super_pop"] == "AFR").sum() == 87
    ref = set(geuvadis.nfe_minus_geuvadis())
    assert not ref & set(s["sample_id"])  # R4b reference excludes every GEUVADIS individual


@pytest.mark.skipif(not HAVE_1KG, reason="1000G processed data not available")
def test_ld_lead_falls_back_to_next_pip_snv_in_panel():
    from data.genotypes import variant_table
    from data.real.pairs import _poolb_lead_r2
    v = variant_table("chr22", 20_000_000, 20_200_000)
    af = v["af_B"].to_numpy()
    v = v[(af > 0.2) & (af < 0.8)].head(2)
    a, b = v["variant_id"].tolist()
    absent = "chr22:20000001:A:C"
    r2, l1, l2 = _poolb_lead_r2(("chr22", f"{absent};{a}", b))
    assert l1 == a and l2 == b
    assert r2 == pytest.approx(ld.panel_r2("chr22", [(a, b)], ld.pools()["B"])[0])
    r2, l1, l2 = _poolb_lead_r2(("chr22", absent, b))
    assert np.isnan(r2) and l1 == ""
