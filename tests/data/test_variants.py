"""Tests for src/data/variants.py: toy known-answer QC/AF test and invariants on the real outputs."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.variants import MHC, OUT_DIR, PGEN_DIR, POOLS_FILE, load_variants, run_chrom
from conftest import TOY, ROOT, PLINK2


def test_toy_qc_and_af(toy_pgen):
    t = toy_pgen
    counts = run_chrom(6, t["pgen_dir"], t["out_dir"], t["multirec_dir"], t["pools"])
    v = pd.read_parquet(t["out_dir"] / "chr6.parquet")
    expected = [p for p, *_, keep in TOY if keep]
    assert v["pos"].tolist() == expected
    assert counts["n_pass"] == len(expected) and counts["n_pvar"] == len(TOY)
    assert counts["n_snv_dup_in_pgen"] == 2 and counts["n_snv_dup_in_raw_only"] == 1
    assert counts["n_snv_mhc"] == 3
    r = v.set_index("pos")
    # pos 1000: S1 0|1, S2 1|1, S3 0|0, S4 1|0, S5 0|0, S6 1|1
    assert r.loc[1000, "af_A"] == pytest.approx(3 / 4)
    assert r.loc[1000, "af_B"] == pytest.approx(1 / 4)
    assert r.loc[1000, "af_FIN"] == 0 and r.loc[1000, "af_ASW_ACB"] == 1
    assert r.loc[1000, "af_L6c"] == pytest.approx(1 / 2)
    assert r.loc[1000, "af_NFE"] == pytest.approx(4 / 8)
    # pos 7000: S1 0|0, S2 0|1, S3 1|0, S4 1|1, S5 0|1, S6 1|0
    assert r.loc[7000, ["af_A", "af_B", "af_FIN", "af_ASW_ACB"]].tolist() == pytest.approx([1 / 4, 3 / 4, 1 / 2, 1 / 2])
    assert (v["variant_id"] == "chr6:" + v["pos"].astype(str) + ":" + v["ref"] + ":" + v["alt"]).all()


def test_missing_multirec_file_fails(toy_pgen, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_chrom(6, toy_pgen["pgen_dir"], tmp_path / "o", tmp_path, toy_pgen["pools"])


REAL = sorted(OUT_DIR.glob("chr*.parquet"))


@pytest.mark.skipif(not REAL, reason="no real variant parquet files")
@pytest.mark.parametrize("path", REAL, ids=[p.stem for p in REAL])
def test_real_invariants(path):
    v = pd.read_parquet(path)
    c = path.stem
    assert (v["chr"] == c).all()
    assert v["ref"].str.fullmatch("[ACGT]").all() and v["alt"].str.fullmatch("[ACGT]").all()
    assert v["pos"].is_monotonic_increasing and v["pos"].is_unique and v["pgen_idx"].is_unique
    if c == MHC[0]:
        assert not v["pos"].between(MHC[1], MHC[2]).any()
    mr = pd.read_csv(OUT_DIR / "multirec" / f"{c}.pos", sep="\t", header=None)[0]
    assert not v["pos"].isin(mr).any()
    for col in [c_ for c_ in v.columns if c_.startswith("af_")]:
        assert v[col].between(0, 1).all()
    assert (v["n_missing"] == 0).all()
    # QC'd SNV density sanity (1000G EUR MAF>=0.005): > 2,500 per Mb over the covered span
    span = (v["pos"].iloc[-1] - v["pos"].iloc[0]) / 1e6
    assert len(v) / span > 2500, f"{c}: only {len(v) / span:.0f} QC'd SNVs per Mb"
    # pgen_idx points to the right pvar row
    s = json.loads((OUT_DIR / f"{c}.summary.json").read_text())
    assert s["n_pass"] == len(v)


@pytest.mark.skipif(not (OUT_DIR / "chr22.parquet").exists(), reason="chr22 not processed")
def test_real_af_matches_plink2_freq(tmp_path):
    """Independent check: pool-B and ASW_ACB AFs vs plink2 --freq on chr22."""
    v = load_variants("chr22")
    pools = pd.read_csv(POOLS_FILE, sep="\t")
    for pool in ["B", "ASW_ACB"]:
        keep = tmp_path / f"{pool}.keep"
        pools.loc[pools.pool == pool, ["sample_id"]].to_csv(keep, index=False, header=False)
        subprocess.run([PLINK2, "--pfile", str(PGEN_DIR / "chr22"), "--keep", str(keep), "--freq",
                        "--out", str(tmp_path / pool), "--threads", "2", "--memory", "2000"],
                       check=True, capture_output=True)
        f = pd.read_csv(tmp_path / f"{pool}.afreq", sep="\t")
        f = f.set_index("ID").loc[v["variant_id"]]
        assert (f["ALT"].to_numpy() == v["alt"].to_numpy()).all()
        np.testing.assert_allclose(v[f"af_{pool}"].to_numpy(), f["ALT_FREQS"].to_numpy(), atol=1e-6)
