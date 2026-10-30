import subprocess

import h5py
import numpy as np
import pandas as pd
import pytest

import eval.io as eio
from eval.io import (LocusSetReader, TestSplitAccessError, list_loci, read_locus, read_locus_attrs,
                     truth_path_for, write_locus_set)
from eval.truth import read_truth, write_truth
from eval_toy import make_locus, make_locus_set


@pytest.fixture
def no_splits(monkeypatch):
    monkeypatch.setattr(eio, "_splits_file", lambda: None)


def test_roundtrip_dense_and_factor(tmp_path, no_splits):
    rng = np.random.default_rng(0)
    a, _ = make_locus("A", "S", rng, ld_type="dense")
    b, _ = make_locus("B", "D", rng, ld_type="factor", n_ref=200)
    f = tmp_path / "x.h5"
    assert write_locus_set(f, [a, b]) == 2
    assert list_loci(f) == ["A", "B"]
    ra, rb = read_locus(f, "A"), read_locus(f, "B")
    for orig, got in ((a, ra), (b, rb)):
        for k in ("z1", "z2", "beta1", "se1", "beta2", "se2", "n1", "n2", "maf1", "maf2", "maf_ref", "pos"):
            np.testing.assert_array_equal(getattr(orig, k), getattr(got, k))
        assert list(got.variant_id) == list(orig.variant_id)
        assert got.block_ids == orig.block_ids and got.cell_id == orig.cell_id and got.split == orig.split
        assert got.window_start == orig.window_start and got.rho_se == orig.rho_se
    np.testing.assert_array_equal(ra.R, a.R)
    assert rb.ld_type == "factor" and rb.n_ref == 200 and rb.R is None
    # factor LD = X'X / n_ref, ld_rows consistent with ld()
    full = rb.ld()
    np.testing.assert_allclose(full, rb.X_ref.astype(float).T @ rb.X_ref.astype(float) / 200)
    np.testing.assert_allclose(rb.ld_rows([3, 7]), full[[3, 7]])
    np.testing.assert_allclose(np.diag(full), 1.0, atol=1e-5)
    attrs = read_locus_attrs(f)
    assert list(attrs["locus_id"]) == ["A", "B"] and list(attrs["p"]) == [40, 40]


def test_shared_ld_hardlink(tmp_path, no_splits):
    rng = np.random.default_rng(1)
    a, _ = make_locus("A", "S", rng)
    b, _ = make_locus("B", "S", rng)
    a.ld_key = b.ld_key = "w1"
    f = tmp_path / "x.h5"
    write_locus_set(f, [a, b])
    with h5py.File(f, "r") as h:
        assert h["A/R"].id == h["B/R"].id  # same object stored once
        assert "_ld" in h
    assert list_loci(f) == ["A", "B"]  # reserved group skipped
    np.testing.assert_array_equal(read_locus(f, "B").R, a.R)


def test_validation_errors(tmp_path, no_splits):
    rng = np.random.default_rng(2)
    a, _ = make_locus("A", "S", rng)
    a.z2 = a.z2[:-1]
    with pytest.raises(ValueError):
        write_locus_set(tmp_path / "x.h5", [a])
    b, _ = make_locus("_bad", "S", rng)
    with pytest.raises(ValueError):
        write_locus_set(tmp_path / "y.h5", [b])
    c, _ = make_locus("C", "S", rng)
    with pytest.raises(ValueError):
        write_locus_set(tmp_path / "z.h5", [c, c])


def test_guard_split_attribute(tmp_path, no_splits, monkeypatch):
    monkeypatch.delenv(eio.ACCESS_ENV, raising=False)
    rng = np.random.default_rng(3)
    t, _ = make_locus("T", "S", rng, split="test")
    with pytest.raises(TestSplitAccessError):  # writer refuses (positive control)
        write_locus_set(tmp_path / "t.h5", [t])
    write_locus_set(tmp_path / "t.h5", [t], guard=False)  # simulate a file produced elsewhere
    with pytest.raises(TestSplitAccessError):
        read_locus(tmp_path / "t.h5", "T")
    with pytest.raises(TestSplitAccessError):
        read_locus_attrs(tmp_path / "t.h5")
    with pytest.raises(TestSplitAccessError):
        list(LocusSetReader(tmp_path / "t.h5"))
    v, _ = make_locus("V", "S", rng, split="validation")  # negative control
    write_locus_set(tmp_path / "v.h5", [v])
    read_locus(tmp_path / "v.h5", "V")


def test_guard_blocks_file(tmp_path, monkeypatch):
    monkeypatch.delenv(eio.ACCESS_ENV, raising=False)
    sp = tmp_path / "blocks_v1.tsv"
    pd.DataFrame({"block_id": ["b1", "b2", "b3"], "chr": ["chr1", "chr1", "chr2"], "start": [0, 5000, 0],
                  "end": [5000, 9000, 9000], "split": ["train", "test", "validation"]}).to_csv(sp, sep="\t", index=False)
    monkeypatch.setattr(eio, "_splits_file", lambda: sp)
    rng = np.random.default_rng(4)
    # attribute says validation, but a block id is in the test split
    a, _ = make_locus("A", "S", rng, block="b2")
    with pytest.raises(TestSplitAccessError):
        write_locus_set(tmp_path / "a.h5", [a])
    # block id fine but the window overlaps a test block by coordinates
    b, _ = make_locus("B", "S", rng, block="b1")
    b.window_chr, b.window_start, b.window_end = "chr1", 4000, 6000
    with pytest.raises(TestSplitAccessError):
        write_locus_set(tmp_path / "b.h5", [b])
    c, _ = make_locus("C", "S", rng, block="b1")
    c.window_chr, c.window_start, c.window_end = "1", 100, 4999
    write_locus_set(tmp_path / "c.h5", [c])  # negative control
    assert eio.is_test_locus(block_ids=["b2"]) and not eio.is_test_locus(block_ids=["b1", "b3"])
    # truth reader is guarded too
    df = pd.DataFrame([dict(locus_id="X", oracle_class="S", mu1=1.0, mu2=1.0, r2=np.nan, d=np.nan, rho=0.0,
                            block_id="b2", cell_id="c", split="validation")])
    with pytest.raises(TestSplitAccessError):
        write_truth(df, tmp_path / "t.truth.parquet")
    write_truth(df, tmp_path / "t.truth.parquet", guard=False)
    with pytest.raises(TestSplitAccessError):
        read_truth(tmp_path / "t.truth.parquet")


def _git(repo, *a):
    return subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def test_access_grant_requires_frozen_tag_and_open_nonce(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "f").write_text("x")
    _git(repo, "add", "f")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    head = _git(repo, "rev-parse", "HEAD")
    (repo / "results").mkdir()
    log = repo / "results" / "test_access.log"
    log.write_text(f"2026\tstarted\t{head}\tNONCE1\tfrozen-x\t\n")
    monkeypatch.setenv(eio.ACCESS_ENV, "NONCE1")
    assert not eio.has_test_access(repo)  # no frozen tag yet
    _git(repo, "tag", "frozen-x")
    assert eio.has_test_access(repo)
    monkeypatch.setenv(eio.ACCESS_ENV, "WRONG")
    assert not eio.has_test_access(repo)
    monkeypatch.setenv(eio.ACCESS_ENV, "NONCE1")
    with open(log, "a") as fh:
        fh.write(f"2026\tcompleted\t{head}\tNONCE1\tfrozen-x\t\n")
    assert not eio.has_test_access(repo)  # closed nonce
    # guard integration: grant lets a test locus through
    log.write_text(f"2026\tstarted\t{head}\tNONCE2\tfrozen-x\t\n")
    monkeypatch.setenv(eio.ACCESS_ENV, "NONCE2")
    monkeypatch.setattr(eio, "PROJECT_ROOT", repo)
    monkeypatch.setattr(eio, "_splits_file", lambda: None)
    t, _ = make_locus("T", "S", np.random.default_rng(5), split="test")
    write_locus_set(tmp_path / "t.h5", [t])
    assert read_locus(tmp_path / "t.h5", "T").split == "test"


def test_truth_roundtrip_and_schema(tmp_path, no_splits):
    _, truth, tp = make_locus_set(tmp_path / "s.h5", n_per_class=3)
    assert truth_path_for(tmp_path / "s.h5") == tmp_path / "s.truth.parquet"
    got = read_truth(tp)
    pd.testing.assert_frame_equal(got[truth.columns].reset_index(drop=True), truth, check_dtype=False)
    bad = truth.drop(columns=["block_id"])
    with pytest.raises(ValueError):
        write_truth(bad, tmp_path / "bad.parquet")
    bad2 = truth.copy()
    bad2.loc[0, "oracle_class"] = "X"
    with pytest.raises(ValueError):
        write_truth(bad2, tmp_path / "bad2.parquet")
