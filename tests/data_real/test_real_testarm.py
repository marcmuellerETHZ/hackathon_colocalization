"""Tests of the real-data test-arm builder (data.real.extract.build_test_arm[s]) and of data.real.test_meta.

No test-split data is read: guard tests stop before any data access, ID-list tests use toy lists, and the
end-to-end equality test uses VALIDATION loci (real_locus_sets@2)."""
import hashlib
import inspect
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eval.io as io
from data.real import extract, test_meta
from data.real.freeze_test_ids import ID_COLS

ROOT = Path(__file__).resolve().parents[2]
LOCUS_DIR = ROOT / "data/processed/real/locus_sets"
HAVE_REAL = (LOCUS_DIR / "R3b_valid.h5").exists() and (LOCUS_DIR / "rho_v2.parquet").exists()


def _toy_ids(path: Path, arm="R2", split="test", n=3) -> Path:
    pd.DataFrame({"locus_id": [f"{arm}__a__b__G{i}" for i in range(n)], "arm": arm, "dataset1": "a",
                  "dataset2": "b", "gene_id": [f"G{i}" for i in range(n)], "split": split, "chr": "chr1",
                  "win_lo": 1, "win_hi": 2}).to_csv(path, sep="\t", index=False)
    return path


# ------------------------------------------------------------------------------------------- guards
def test_build_test_arm_refuses_without_access(tmp_path, monkeypatch):
    monkeypatch.delenv(io.ACCESS_ENV, raising=False)
    with pytest.raises(io.TestSplitAccessError):
        extract.build_test_arm("R2", _toy_ids(tmp_path / "ids.tsv"), tmp_path / "out")
    with pytest.raises(io.TestSplitAccessError):   # entry-point form used by src/eval/testrun.py
        extract.build_test_arm({"R2": {"n": 800}}, tmp_path / "out", workers=2)
    with pytest.raises(io.TestSplitAccessError):
        extract.build_test_arms({}, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_parent_check_needs_eval_test_on_the_stack(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "has_test_access", lambda repo=None: True)
    with pytest.raises(io.TestSplitAccessError):        # access granted, but not called from scripts/eval_test.py
        extract.build_test_arm("R2", _toy_ids(tmp_path / "ids.tsv"), tmp_path / "out")
    extract._check_test_access(in_worker=True)          # a worker only needs the open nonce
    # known answer: the same call from a file named scripts/eval_test.py passes the parent check
    fake = tmp_path / "scripts" / "eval_test.py"
    fake.parent.mkdir()
    fake.write_text("from data.real.extract import _check_test_access\n_check_test_access(in_worker=False)\n")
    runpy.run_path(str(fake))


def test_worker_check_refuses_without_access(monkeypatch):
    monkeypatch.setattr(io, "has_test_access", lambda repo=None: False)
    row = {"locus_id": "R2__a__b__G", "split": "test", "chr": "chr1", "win_lo": 1, "win_hi": 2}
    with pytest.raises(io.TestSplitAccessError):
        extract._check_test_access(in_worker=True)
    with pytest.raises(io.TestSplitAccessError):
        extract.extract_pair(row, ["poolB"], allow_test=True)
    with pytest.raises(io.TestSplitAccessError):        # not swallowed into the per-locus error log
        extract._job((row, ["poolB"], True))
    with pytest.raises(PermissionError):                 # without allow_test a test row is refused regardless
        extract.extract_pair(row, ["poolB"])


def test_test_mode_refuses_unfrozen_id_list(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "_check_test_access", lambda in_worker=False: None)
    with pytest.raises(extract.TestArmError, match="frozen"):
        extract.build_test_arm("R2", _toy_ids(tmp_path / "ids.tsv"), tmp_path / "out")


def test_valid_mode_refuses_other_splits_and_bad_format(tmp_path):
    with pytest.raises(extract.TestArmError):
        extract.build_test_arm("R2", _toy_ids(tmp_path / "a.tsv", split="test"), tmp_path / "o", mode="valid")
    with pytest.raises(extract.TestArmError):
        extract.build_test_arm("R2", _toy_ids(tmp_path / "b.tsv", split="train"), tmp_path / "o", mode="valid")
    pd.DataFrame({"locus_id": ["x"], "split": ["valid"]}).to_csv(tmp_path / "c.tsv", sep="\t", index=False)
    with pytest.raises(extract.TestArmError):
        extract.build_test_arm("R2", tmp_path / "c.tsv", tmp_path / "o", mode="valid")
    with pytest.raises(extract.TestArmError):           # registered @2 directory is never an output
        extract.build_test_arm("R2", _toy_ids(tmp_path / "d.tsv", split="valid"), extract.LOCUS_DIR, mode="valid")
    for bad in (extract.OUT, extract.OUT / "pairs", extract.OUT / "locus_sets" / "new"):   # whole @2 tree
        with pytest.raises(extract.TestArmError):
            extract.build_test_arm("R2", _toy_ids(tmp_path / "d.tsv", split="valid"), bad, mode="valid")
    assert not (extract.OUT / "locus_sets" / "new").exists()


def test_spec_size_must_match_frozen_list(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "_check_test_access", lambda in_worker=False: None)
    lists = {a: _toy_ids(tmp_path / f"{a}.tsv", arm=a) for a in ("R2", "R3b", "R4b")}
    with pytest.raises(extract.TestArmError, match="n=800"):
        extract.build_test_arms({"R2": {"n": 800}}, tmp_path / "o", id_lists=lists)


def test_entry_point_dispatch_and_keys(tmp_path, monkeypatch):
    """testrun._build_real_test calls builder(real_spec, out_dir, workers=n) and needs real_R2, real_R3b,
    real_R4b_insample, real_R4b_reference with "path" and "ids"."""
    monkeypatch.setattr(extract, "_check_test_access", lambda in_worker=False: None)
    calls = []
    orig = extract.build_test_arm

    def fake(arm, lst, out_dir, workers=16, mode="test", **kw):
        calls.append((arm, Path(lst).name, workers, mode))
        ms = extract.ARM_MODES[arm]
        return {"ids": [f"{arm}_1", f"{arm}_2"], "n_requested": 2, "failed_ids": [],
                "files": {m: str(Path(out_dir) / f"{arm}.{m}.h5") for m in ms}}
    monkeypatch.setattr(extract, "build_test_arm", fake)
    lists = {a: _toy_ids(tmp_path / f"{a}.tsv", arm=a, n=2) for a in ("R2", "R3b", "R4b")}
    spec = {"R2": {"n": 2}, "R3b": {"n": 2}, "R4b": {"n": 2, "ld_conditions": ["in_sample", "reference"]}}
    res = orig(spec, tmp_path / "out", workers=7, id_lists=lists)
    assert [c[0] for c in calls] == ["R2", "R3b", "R4b"] and all(c[2] == 7 and c[3] == "test" for c in calls)
    need = {"real_R2", "real_R3b", "real_R4b_insample", "real_R4b_reference"}
    assert need <= set(res)
    assert res["real_R4b_insample"]["path"].endswith("R4b.geuv_insample_pcadj.h5")
    assert res["real_R4b_reference"]["path"].endswith("R4b.nfe_minus_geuv.h5")
    assert res["real_R4b_insample"]["ids"] == res["real_R4b_reference"]["ids"]
    assert all(isinstance(res[k]["ids"], list) and res[k]["path"] for k in need)
    # default lists: the frozen sub lists
    calls.clear()
    monkeypatch.setattr(extract, "PAIRS", tmp_path)
    for a in ("R2", "R3b", "R4b"):
        _toy_ids(tmp_path / extract.FROZEN_SUB_LIST[a], arm=a, n=2)
    orig({}, tmp_path / "out2", workers=3)
    assert [c[1] for c in calls] == [extract.FROZEN_SUB_LIST[a] for a in ("R2", "R3b", "R4b")]


def test_signature_matches_testrun_call():
    sig = inspect.signature(extract.build_test_arm)
    sig.bind({"R2": {}}, Path("x"), workers=4)          # builder(self.real_spec, out_dir, workers=...)
    sig.bind("R2", "ids.tsv", "out", workers=4, mode="valid")


def test_donor_disjoint_known_answers():
    assert not extract.donor_disjoint("QTD000356", "QTD000266")   # GTEx blood / liver
    assert not extract.donor_disjoint("QTD000549", "QTD000539")   # TwinsUK blood / LCL
    assert not extract.donor_disjoint("QTD000110", "QTD000110")
    assert extract.donor_disjoint("QTD000356", "QTD000373")       # GTEx x EstBB
    assert extract.donor_disjoint("QTD000539", "QTD000110")       # TwinsUK LCL x GEUVADIS (R4b eQTL pair)
    assert extract.donor_disjoint("FG_T2D", "QTD000110")


# ------------------------------------------------------------------------------------------- test_meta
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_test_meta_verify_toy(tmp_path):
    """Rebuilt table -> candidates byte-identical, full/sub hashes equal the frozen ones, metadata in frozen order;
    any tampering is detected."""
    def row(i, split):
        return {"locus_id": f"R4b__a__b__G{i}", "arm": "R4b", "dataset1": "a", "dataset2": "b", "gene_id": f"G{i}",
                "gene_name": f"N{i}", "split": split, "chr": "chr1", "tss": 100 + i, "block_id": f"blk{i % 3}",
                "win_lo": 1 + i, "win_hi": 1000 + i, "lead1": f"chr1_{i}_A_G", "lead1_p": 1e-9, "lead1_z": 6.1,
                "lead2": f"chr1_{i}_C_T", "lead2_p": 1e-7, "lead2_z": -5.3, "n_fg_cs_in_window": 1}
    table = pd.DataFrame([row(i, s) for i, s in enumerate(["train", "test", "valid", "test", "test", "train"])])
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    tv = table[table["split"].isin(["train", "valid"])]
    te = table[table["split"] == "test"]
    tv.to_csv(pairs / "R4b_candidates.tsv", sep="\t", index=False)
    te[ID_COLS].to_csv(pairs / "R4b_test_ids.tsv", sep="\t", index=False)
    te[ID_COLS].to_csv(pairs / "R4b_test_ids_sub_v1.tsv", sep="\t", index=False)   # n < 600: sub = all
    sha = tmp_path / "ids.sha256"
    sha.write_text("".join(f"{_sha((pairs / f).read_bytes())}  x/{f}\n"
                           for f in ("R4b_test_ids.tsv", "R4b_test_ids_sub_v1.tsv")))
    meta, rep = test_meta.verify_and_select("R4b", table, pairs, sha)
    assert rep["trainvalid_check"]["candidates_file_byte_identical"]
    assert rep["full"]["sha256"] == rep["full"]["frozen"] and rep["sub"]["n"] == 3
    assert meta["locus_id"].tolist() == te["locus_id"].tolist()
    assert meta["lead1"].tolist() == te["lead1"].tolist() and meta["block_id"].tolist() == te["block_id"].tolist()
    assert "lead1_p" not in meta.columns and "lead1_z" not in meta.columns      # no statistics in the metadata
    bad = table.copy()
    bad.loc[bad["locus_id"] == "R4b__a__b__G3", "win_hi"] += 1                   # a test window changed
    with pytest.raises(AssertionError, match="hash"):
        test_meta.verify_and_select("R4b", bad, pairs, sha)
    bad = table.copy()
    bad.loc[0, "lead1"] = "chr1_0_A_T"                                           # train row differs
    with pytest.raises(AssertionError, match="byte-identical"):
        test_meta.verify_and_select("R4b", bad, pairs, sha)


def test_test_meta_frozen_files_check():
    if not (ROOT / "datasets/configs/real_test_meta_v1.sha256").exists() or not HAVE_REAL:
        pytest.skip("frozen metadata not available")
    assert test_meta.check() == []                       # sha256 of the files only; contents are not inspected


# ------------------------------------------------------------------------------------------- end to end (valid)
@pytest.mark.skipif(not HAVE_REAL, reason="real_locus_sets@2 not available")
@pytest.mark.parametrize("arm,n", [("R3b", 2), ("R4b", 1)])
def test_valid_mode_equals_real_locus_sets_v2(tmp_path, arm, n):
    from eval.io import LocusSetReader, read_locus_attrs
    from data.real.rho_update import locus_file
    cand = pd.read_csv(ROOT / f"data/processed/real/pairs/{arm}_candidates.tsv", sep="\t", usecols=ID_COLS)
    cand = cand[cand["split"] == "valid"].iloc[[0, 7, 11][:n]]
    cand.to_csv(tmp_path / "ids.tsv", sep="\t", index=False)
    res = extract.build_test_arm(arm, tmp_path / "ids.tsv", tmp_path / "out", workers=2, mode="valid")
    assert res["ids"] == cand["locus_id"].tolist() and res["failed_ids"] == []
    for m, path in res["files"].items():
        with LocusSetReader(path) as a, LocusSetReader(locus_file(arm, "valid", m)) as b:
            for lid in res["ids"]:
                A, B = a.read(lid), b.read(lid)
                for f in ("pos", "z1", "z2", "beta1", "se1", "beta2", "se2", "n1", "n2", "maf1", "maf2", "maf_ref",
                          "X_ref"):
                    np.testing.assert_array_equal(getattr(A, f), getattr(B, f), err_msg=f"{lid} {m} {f}")
                assert list(A.variant_id) == list(B.variant_id)
                assert (A.rho_hat == B.rho_hat or np.isnan(A.rho_hat) and np.isnan(B.rho_hat))
                assert A.rho_se == B.rho_se or np.isnan(A.rho_se) and np.isnan(B.rho_se)
                for k, v in B.extra_attrs.items():
                    w = A.extra_attrs[k]
                    assert w == v or (isinstance(v, float) and np.isnan(v) and np.isnan(w)), (lid, m, k, w, v)
                assert set(A.extra_attrs) - set(B.extra_attrs) == {"donor_disjoint", "rho_sens_zero"}
                assert A.extra_attrs["donor_disjoint"] and A.extra_attrs["rho_sens_zero"] == 0.0
        at = read_locus_attrs(path)
        assert (at["split"] == "valid").all()
