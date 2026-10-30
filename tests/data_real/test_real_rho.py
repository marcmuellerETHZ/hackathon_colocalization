"""Known-answer tests for the P3-1 real-data rho-hat update (rho_update.py) and the test-ID freeze."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from data.real import freeze_test_ids as fz
from data.real import rho_update as ru
from stats.rho import rho_hat_pooled, rho_hat_window, signal_mask


def _null_pair(g, p, rho):
    e1 = g.standard_normal(p)
    return e1, rho * e1 + np.sqrt(1 - rho * rho) * g.standard_normal(p)


def test_trait_type_matches_simulator_convention():
    from sim.params import ALPHA
    assert ru.trait_type("FG_3001308") == "gwas" and ru.trait_type("QTD000110") == "mol"
    assert {ru.trait_type("FG_T2D"), ru.trait_type("QTD000356")} == set(ALPHA)


def test_window_rho_equals_rho_hat_window_and_recovers_rho():
    g = np.random.default_rng(20261029)
    p, rho = 4000, 0.4
    z1, z2 = _null_pair(g, p, rho)
    z1[100] = z2[100] = 8.0            # one strong signal: masked
    R = np.eye(p)
    out = ru.window_rho(z1, z2, R, "toy_locus")
    ref = rho_hat_window(z1, z2, R, 0.1, 2.0, ru.prune_order("toy_locus", p), residualize_signal=False)
    assert out["rho_hat"] == ref["rho_hat"] and out["rho_se"] == ref["rho_se"]
    assert abs(out["rho_hat"] - rho) < 4 * out["rho_se"] and out["rho_se"] < 0.05
    assert out["rho_n_masked"] == int((np.maximum(abs(z1), abs(z2)) > 4).sum()) >= 1   # R = I: signals only
    # deterministic pruning order per locus id, shared across LD modes
    assert np.array_equal(ru.prune_order("a", 50), ru.prune_order("a", 50))
    assert not np.array_equal(ru.prune_order("a", 50), ru.prune_order("b", 50))


def test_pooled_from_stats_equals_rho_hat_pooled():
    g = np.random.default_rng(20261027)
    wins, stats = [], []
    for k in range(14):
        p = 300
        z1, z2 = _null_pair(g, p, -0.3)
        z1[5] = 6.0
        A = g.standard_normal((80, p)) / np.sqrt(80)
        R = A.T @ A
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
        wins.append((z1, z2, R))
        m = signal_mask(z1, z2, R)
        use = ~m & (np.abs(z1) < 2) & (np.abs(z2) < 2)
        stats.append((use.sum(), np.sum(z1[use] ** 2 + z2[use] ** 2), np.sum(z1[use] * z2[use])))
    n, sqq, s12 = map(np.array, zip(*stats))
    ref = rho_hat_pooled(wins)
    got = ru.pooled_from_stats(n, sqq, s12)
    assert got["rho_hat"] == pytest.approx(ref["rho_hat"], abs=1e-9)
    assert got["rho_se"] == pytest.approx(ref["rho_se"], abs=1e-9)        # singleton clusters = windows
    assert got["rho_se_win"] == pytest.approx(ref["rho_se"], abs=1e-9)
    # clustering windows (pairs) changes only the SE
    got2 = ru.pooled_from_stats(n, sqq, s12, labels=np.arange(14) // 2)
    assert got2["rho_hat"] == got["rho_hat"] and got2["n_clusters"] == 7 and got2["rho_se"] != got["rho_se"]


def test_ld_clusters_union_find():
    lab = ru.ld_clusters([["b1"], ["b1", "b2"], ["b2"], ["b3"], [], ["b3", "b4"]])
    assert lab[0] == lab[1] == lab[2] and lab[3] == lab[5] and len({lab[0], lab[3], lab[4]}) == 3


def _toy_locus(lid, split, d1, d2, g, rho, p=400, block="chr1_0001"):
    from eval.io import LocusData
    z1, z2 = _null_pair(g, p, rho)
    X = g.standard_normal((150, p)).astype(np.float32)
    X = (X - X.mean(0)) / X.std(0)
    one = np.ones(p)
    return LocusData(locus_id=lid, variant_id=np.array([f"v{i}" for i in range(p)], dtype=object),
                     pos=np.arange(p, dtype=np.int64) + 1, z1=z1, z2=z2, beta1=z1, se1=one, beta2=z2, se2=one,
                     n1=one * 1e3, n2=one * 1e3, maf1=one * .2, maf2=one * .2, maf_ref=one * .2, ld_type="factor",
                     n_ref=150, X_ref=X, rho_hat=0.123, rho_se=0.05, split=split, block_ids=(block,),
                     mismatch_level="real_ref_poolB", pairing_type="real_eqtl_x_eqtl", cell_id="R2",
                     window_chr="chr1", window_start=-1, window_end=-1,
                     extra_attrs={"dataset1": d1, "dataset2": d2, "trait1_type": "molecular", "trait2_type": "molecular"})


def test_end_to_end_toy_locus_sets(tmp_path, monkeypatch):
    """Two cohort pairs, train + valid: attrs rewritten, valid loci get the TRAIN pooled value, v1 kept."""
    from eval.io import read_locus_attrs, write_locus_set
    import eval.io as io
    monkeypatch.setattr(io, "get_test_blocks", lambda: (frozenset(), None))
    g = np.random.default_rng(20261026)
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    cand, logs = [], {"train": [], "valid": []}
    for split, nloc in (("train", 12), ("valid", 4)):
        loci = []
        for k in range(nloc):
            for d1, d2, rho in (("QTD_A", "QTD_B", 0.0), ("QTD_C", "QTD_D", 0.5)):
                lid = f"R2__{d1}__{d2}__{split}{k}"
                loci.append(_toy_locus(lid, split, d1, d2, g, rho, block=f"chr1_{k:04d}"))
                cand.append({"locus_id": lid, "dataset1": d1, "block_id": f"chr1_{k:04d}", "split": split})
                logs[split].append({"locus_id": lid, "status": "ok", "rho_hat": 0.123, "rho_se": 0.05})
        write_locus_set(tmp_path / f"R2_{split}.h5", loci)
        pd.DataFrame(logs[split]).to_csv(tmp_path / f"R2_{split}.extract_log.tsv", sep="\t", index=False)
    pd.DataFrame(cand).to_csv(pairs / "R2_candidates.tsv", sep="\t", index=False)
    s = ru.run(arms=("R2",), workers=2, locus_dir=tmp_path, pairs_dir=pairs)
    assert s["n_written"] == {"R2_train.h5": 24, "R2_valid.h5": 8} and s["v1_file_vs_log_mismatch"] == 0
    a = pd.concat([read_locus_attrs(tmp_path / f"R2_{x}.h5") for x in ("train", "valid")])
    assert set(a["trait1_type"]) == {"mol"} and set(a["ld_role"]) == {"reference"}
    assert (a["rho_hat"] != 0.123).all() and a["rho_hat"].notna().all()
    for d1, rho in (("QTD_A", 0.0), ("QTD_C", 0.5)):
        h = a[a["dataset1"] == d1]
        assert h["rho_hat_pair"].nunique() == 1 and h["rho_pair_n_windows"].iloc[0] == 12
        assert abs(h["rho_hat_pair"].iloc[0] - rho) < 4 * h["rho_se_pair"].iloc[0]
    assert (a["boot_unit"].str.startswith("chr1_")).all()
    side = pd.read_parquet(tmp_path / "rho_v2.parquet")
    assert (side["rho_hat_v1"] == 0.123).all() and len(side) == 32
    # idempotent: a second run reads v1 from the logs and gives identical attributes
    s2 = ru.run(arms=("R2",), workers=2, locus_dir=tmp_path, pairs_dir=pairs)
    b = pd.concat([read_locus_attrs(tmp_path / f"R2_{x}.h5") for x in ("train", "valid")])
    np.testing.assert_array_equal(a["rho_hat"].to_numpy(), b["rho_hat"].to_numpy())
    assert s2["n_fresh"] == 0


def test_rho_update_refuses_test_split(tmp_path, monkeypatch):
    from eval.io import write_locus_set
    import eval.io as io
    monkeypatch.setattr(io, "get_test_blocks", lambda: (frozenset(), None))
    g = np.random.default_rng(1)
    write_locus_set(tmp_path / "R2_train.h5", [_toy_locus("R2__x__y__t", "test", "x", "y", g, 0.0)], guard=False)
    with pytest.raises(PermissionError):
        ru._chunk_job((str(tmp_path / "R2_train.h5"), ["R2__x__y__t"]))
    with pytest.raises(PermissionError):
        ru.run(arms=("R2",), splits=("test",), locus_dir=tmp_path)


def test_freeze_test_ids_subsample(tmp_path):
    pairs = tmp_path / "data/processed/real/pairs"
    pairs.mkdir(parents=True)
    for arm, n in (("R2", 1000), ("R3b", 50), ("R4b", 700)):
        pd.DataFrame({"locus_id": [f"{arm}__x__{i}" for i in range(n)], "arm": arm, "dataset1": "a", "dataset2": "b",
                      "gene_id": "g", "split": "test", "chr": "chr1", "win_lo": 1, "win_hi": 2}).to_csv(
            pairs / f"{arm}_test_ids.tsv", sep="\t", index=False)
    sha = tmp_path / "configs/real_test_ids_v1.sha256"
    out = fz.freeze(pairs, sha, tmp_path)
    assert {k: v["n_sub"] for k, v in out.items()} == {"R2": 800, "R3b": 50, "R4b": 600}
    sub = pd.read_csv(pairs / "R2_test_ids_sub_v1.tsv", sep="\t")
    full = pd.read_csv(pairs / "R2_test_ids.tsv", sep="\t")
    assert sub["locus_id"].is_unique and set(sub["locus_id"]) <= set(full["locus_id"])
    assert list(sub["locus_id"]) == [x for x in full["locus_id"] if x in set(sub["locus_id"])]  # order kept
    assert fz.check(sha, tmp_path) == [] and len(sha.read_text().splitlines()) == 6
    assert sha.read_text().splitlines()[0].endswith("  data/processed/real/pairs/R2_test_ids.tsv")
    with pytest.raises(FileExistsError):
        fz.freeze(pairs, sha, tmp_path)
    first = sub["locus_id"].tolist()
    fz.freeze(pairs, sha, tmp_path, force=True)                       # deterministic (seeded)
    assert pd.read_csv(pairs / "R2_test_ids_sub_v1.tsv", sep="\t")["locus_id"].tolist() == first
    (pairs / "R4b_test_ids_sub_v1.tsv").write_text("tampered\n")
    assert fz.check(sha, tmp_path) == ["data/processed/real/pairs/R4b_test_ids_sub_v1.tsv"]
    bad = pd.DataFrame({"locus_id": ["a"], "arm": "R2", "split": "valid"})
    with pytest.raises(ValueError):
        fz.subsample(bad, "R2", 1)
