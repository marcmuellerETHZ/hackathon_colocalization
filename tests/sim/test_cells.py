"""Cell builder / locus-set tests: format, truth isolation (DESIGN §7.1 item 8), determinism, rejection
buckets, L4 corruption, PVE cap. Builds a tiny screening-like set on chr22 validation windows (GPU)."""
import hashlib

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from eval.io import iter_loci, read_locus_attrs
from eval.truth import read_truth
from sim import params
from sim.cells import SCREEN_BUCKETS, build_set, link_set, screening_cells
from sim.generator import WindowContext
from sim.windows import split_windows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
ALLOWED_DS = {"variant_id", "pos", "z1", "z2", "beta1", "se1", "beta2", "se2", "n1", "n2", "maf1", "maf2", "maf_ref",
              "R", "X_ref"}
L0_EXTRA_DS = {"R2", "X_ref2"}
ALLOWED_ATTRS = {"ld_type", "n_ref", "rho_hat", "rho_se", "split", "block_ids", "mismatch_level", "pairing_type",
                 "cell_id", "window_chr", "window_start", "window_end", "trait1_type", "trait2_type",
                 "ld_type2", "n_ref2"}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("sets")
    stats = build_set("tiny", screening_cells(), SCREEN_BUCKETS, 2, "valid", 20261027, out, K=20, M=2,
                      chroms=["chr22"], log=lambda s: None)
    link_set("tiny", out)
    return out, stats


def _digest(path):
    h = hashlib.sha256()
    for L in iter_loci(path):
        h.update(L.locus_id.encode())
        h.update(np.ascontiguousarray(L.z1).tobytes())
        h.update(np.ascontiguousarray(L.z2).tobytes())
    return h.hexdigest()


def test_test_split_refused():
    with pytest.raises(PermissionError):
        split_windows("test")


def test_format_and_buckets(built):
    out, stats = built
    t = read_truth(out / "tiny.truth.parquet")
    assert len(t) == 6 * 4 * 2
    assert all(c["complete"] for c in stats["cells"].values())
    for b in SCREEN_BUCKETS:
        sub = t[t["bucket"] == b.name]
        assert (sub["oracle_class"] == b.cls).all()
        if b.r2_lo is not None:
            assert ((sub["r2"] >= b.r2_lo) & (sub["r2"] < b.r2_hi)).all()
    assert (t["split"] == "valid").all()
    assert (t["pve1"] <= params.MAX_TOTAL_PVE + 1e-9).all() and (t["pve2"] <= params.MAX_TOTAL_PVE + 1e-9).all()
    a = read_locus_attrs(out / "tiny.h5")
    assert set(a["locus_id"]) == set(t["locus_id"])
    n = 0
    for L in iter_loci(out / "tiny.h5"):
        L.validate()
        assert np.all(np.isfinite(L.z1)) and np.all(np.isfinite(L.z2))
        assert np.all(L.maf1 >= params.MIN_MAF_STUDY - 1e-12) and np.all(L.maf2 >= params.MIN_MAF_STUDY - 1e-12)
        assert np.allclose(L.beta1 / L.se1, L.z1) and np.allclose(L.beta2 / L.se2, L.z2)
        n += 1
    assert n == len(t)


def test_truth_isolation(built):
    """Method inputs never contain truth: only the INTERFACES datasets/attributes, no true rho, and at
    levels >= L1 the LD given is not R_true (DESIGN §7.1.8)."""
    out, _ = built
    t = read_truth(out / "tiny.truth.parquet").set_index("locus_id")
    with h5py.File(out / "tiny.h5", "r") as f:
        for lid in f.keys():
            g = f[lid]
            ds = set(g.keys())
            lvl = g.attrs["mismatch_level"]
            assert ds <= ALLOWED_DS | (L0_EXTRA_DS if lvl == "L0" else set()), ds
            assert set(g.attrs.keys()) <= ALLOWED_ATTRS, set(g.attrs.keys()) - ALLOWED_ATTRS
            assert g.attrs["rho_hat"] != t.loc[lid, "rho"]
    # R_true comparison at >= L1 (rebuild the target population LD for a few loci)
    checked = 0
    for L in iter_loci(out / "tiny.h5"):
        if L.mismatch_level == "L0" or checked >= 6:
            continue
        ctx = WindowContext(L.window_chr, L.window_start, L.window_end)
        P = ctx.pop("NFE")
        idx = np.searchsorted(ctx.v["pos"].to_numpy(), L.pos)
        assert np.array_equal(ctx.v["pos"].to_numpy()[idx], L.pos)
        Z = P.std[:, torch.as_tensor(idx, device=ctx.device)].double()
        Rtrue = (Z.T @ Z / P.N).cpu().numpy()
        Rgiven = L.ld()
        assert np.abs(Rgiven - Rtrue).max() > 0.05
        assert L.n_ref == 150
        ctx.release()
        checked += 1
    assert checked > 0


def test_l4_corruption(built):
    out, _ = built
    t = read_truth(out / "tiny.truth.parquet")
    t4 = t[t["mismatch_level"] == "L4"]
    assert len(t4)
    for _, r in t4.iterrows():
        assert r["p"] == r["p_study"] - round(params.L4_MISSING * r["p_study"])
        assert len(r["flipped"]) == round(params.L4_FLIP * r["p"])
    t2 = t[t["mismatch_level"] != "L4"]
    assert (t2["p"] == t2["p_study"]).all()


def test_deterministic_rebuild(built, tmp_path):
    out, _ = built
    build_set("tiny", screening_cells()[:2], SCREEN_BUCKETS, 2, "valid", 20261027, tmp_path, K=20, M=2,
              chroms=["chr22"], log=lambda s: None)
    link_set("tiny", tmp_path)
    a = read_truth(out / "tiny.truth.parquet")
    b = read_truth(tmp_path / "tiny.truth.parquet")
    a = a[a["cell_id"].isin(b["cell_id"])].sort_values("locus_id").reset_index(drop=True)
    b = b.sort_values("locus_id").reset_index(drop=True)
    assert list(a["locus_id"]) == list(b["locus_id"])
    ids = list(b["locus_id"])
    za = {L.locus_id: L.z1 for L in iter_loci(out / "tiny.h5", ids)}
    zb = {L.locus_id: L.z1 for L in iter_loci(tmp_path / "tiny.h5", ids)}
    for k in ids:
        assert np.array_equal(za[k], zb[k])
