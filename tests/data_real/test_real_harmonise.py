"""Toy tests with known answers for src/data/real/harmonise.py (positive and negative controls)."""
import numpy as np
import pandas as pd
import pytest

from data.real.harmonise import check_build, harmonise, is_palindromic, lead_missingness, variant_key


def _ref():
    # chr1 reference records, ALT frequency in the reference panel
    return pd.DataFrame({
        "chr": ["chr1"] * 8,
        "pos": [100, 200, 300, 400, 500, 600, 700, 800],
        "ref": ["A", "C", "A", "G", "C", "A", "T", "G"],
        "alt": ["G", "T", "T", "C", "A", "C", "C", "A"],
        "variant_id": [f"chr1:{p}:{r}:{a}" for p, r, a in zip([100, 200, 300, 400, 500, 600, 700, 800],
                                                                "ACAGCATG", "GTTCACCA")],
        "af_ref": [0.30, 0.40, 0.20, 0.25, 0.50, 0.10, 0.30, 0.45],
    })


def _study():
    return pd.DataFrame({
        "chr": ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "X"],
        "pos": [100, 200, 300, 400, 500, 600, 700, 700, 800, 900, 100],
        "ref": ["A", "T", "A", "C", "C", "T", "T", "T", "G", "A", "A"],
        "alt": ["G", "C", "T", "G", "A", "G", "C", "TC", "A", "G", "G"],
        "beta": [0.5, 0.2, 0.1, 0.3, -0.4, 0.2, 0.1, 0.1, 0.3, 0.1, 0.1],
        "se": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "af": [0.31, 0.62, 0.45, 0.74, 0.20, 0.10, 0.30, 0.01, 0.46, 0.2, 0.3],
    })


def test_harmonise_known_answer():
    out, log = harmonise(_study(), _ref())
    ids = out["variant_id"].tolist()
    # 100: identical -> kept, unflipped
    r = out.set_index("variant_id")
    assert "chr1:100:A:G" in ids and not r.at["chr1:100:A:G", "flipped"]
    assert r.at["chr1:100:A:G", "beta"] == pytest.approx(0.5)
    # 200: study T/C vs reference C/T -> swapped, non-palindromic -> flipped: beta -> -0.2, af -> 0.38
    assert r.at["chr1:200:C:T", "flipped"]
    assert r.at["chr1:200:C:T", "beta"] == pytest.approx(-0.2)
    assert r.at["chr1:200:C:T", "af"] == pytest.approx(0.38)
    assert r.at["chr1:200:C:T", "z"] == pytest.approx(-2.0)
    # 300: palindromic A/T, same orientation, |dAF| = 0.25 > 0.1 -> dropped
    assert "chr1:300:A:T" not in ids
    # 400: palindromic G/C reported swapped (C/G) -> dropped (ambiguous with strand flip)
    assert "chr1:400:G:C" not in ids
    # 500: non-palindromic, |dAF| = 0.30 > 0.2 -> kept but flagged
    assert r.at["chr1:500:C:A", "af_flag"]
    # 600: T/G vs A/C = strand complement -> allele mismatch, dropped
    assert "chr1:600:A:C" not in ids
    # 700: two records at one position (SNV + indel) -> multi-record, dropped
    assert "chr1:700:T:C" not in ids
    # 800: palindromic? G/A no -> kept
    assert "chr1:800:G:A" in ids
    assert log["n_non_autosomal"] == 1
    assert log["n_multirecord"] == 2          # SNV + indel at 700
    assert log["n_not_snv"] == 0
    assert log["n_not_in_reference"] == 1     # 900
    assert log["n_allele_mismatch"] == 1      # 600
    assert log["n_palindromic_swapped_dropped"] == 1
    assert log["n_palindromic_af_dropped"] == 1
    assert log["n_af_flag_kept"] == 1
    assert log["n_out"] == len(out) == 4      # 100, 200, 500, 800


def test_harmonise_negative_control_identity():
    """A study identical to the reference passes untouched (no flips, no drops)."""
    ref = _ref()
    study = ref.rename(columns={"af_ref": "af"}).drop(columns="variant_id").assign(beta=0.1, se=0.05)
    study = study[~is_palindromic(study["ref"], study["alt"])]
    out, log = harmonise(study, ref)
    assert log["n_out"] == len(study) and log["n_flipped"] == 0 and not out["af_flag"].any()


def test_harmonise_flip_roundtrip():
    """Swapping every record's alleles and negating beta gives the same aligned output."""
    ref = _ref()
    rng = np.random.default_rng(20261026)
    study = ref.rename(columns={"af_ref": "af"}).drop(columns="variant_id")
    study = study[~is_palindromic(study["ref"], study["alt"])].copy()
    study["beta"] = rng.normal(size=len(study))
    study["se"] = 0.1
    a, _ = harmonise(study, ref)
    sw = study.rename(columns={"ref": "alt", "alt": "ref"}).assign(beta=-study["beta"], af=1 - study["af"])
    b, lb = harmonise(sw, ref)
    assert lb["n_flipped"] == len(study)
    np.testing.assert_allclose(a["beta"], b["beta"])
    np.testing.assert_allclose(a["af"], b["af"])


def test_variant_key_and_lead_missingness():
    assert variant_key("chr1_123_A_G") == variant_key("1:123:A:G") == "chr1:123:A:G"
    m = lead_missingness({"t1": "chr1_100_G_A", "t2": "chr1_999_A_G", "t3": None}, {"chr1:100:A:G"})
    assert m == {"lead_present_t1": True, "lead_present_t2": False, "lead_present_t3": False}


def test_check_build_detects_wrong_build():
    """GRCh38-consistent alleles match; positions shifted (as from another build) do not."""
    rng = np.random.default_rng(20261027)
    n = 2000
    pos = np.sort(rng.choice(10**7, n, replace=False)) + 1
    bases = np.array(list("ACGT"))
    r = bases[rng.integers(0, 4, n)]
    a = np.array([rng.choice([b for b in "ACGT" if b != x]) for x in r])
    ref = pd.DataFrame({"chr": "chr1", "pos": pos, "ref": r, "alt": a})
    good = check_build(ref.assign(chr="1"), ref, rsid_col=None)
    assert good["frac_alleles_match"] == 1.0
    shifted = ref.assign(pos=np.roll(pos, 1))  # alleles land on other reference positions
    bad = check_build(shifted, ref, rsid_col=None)
    assert bad["frac_alleles_match"] < 0.3


def test_check_build_known_rsids():
    st = pd.DataFrame({"chr": ["19", "19"], "pos": [44908684, 45412079], "ref": ["T", "C"], "alt": ["C", "T"],
                       "rsid": ["rs429358", "rs7412"]})
    out = check_build(st, pd.DataFrame(columns=["chr", "pos", "ref", "alt"]))
    assert out["known_rsids_at_grch38"] == {"rs429358": True, "rs7412": False}  # rs7412 given at GRCh37
