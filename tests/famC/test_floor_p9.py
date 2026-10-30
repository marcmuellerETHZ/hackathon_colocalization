"""Family C floor (phase 2): CPU feature path equals the GPU path (the floor's training features are computed on CPU
workers, its harness Method may run on either), and FloorColoc scores real validation loci end to end with valid
probabilities and the same features as the training-record path."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "data" / "processed" / "famC" / "train_dev_snap1"
VG = ROOT / "data" / "processed" / "sets" / "valid_grid_v1.h5"
FLOOR_DIR = ROOT / "results" / "famC" / "floor_snap1"


@pytest.fixture(scope="module")
def recs():
    f = sorted(SNAP.glob("shard*.pt"))[0]
    return torch.load(f, weights_only=False)["recs"][:48]


def test_features_cpu_equals_gpu(recs):
    from famC.floor import FEATS, features_batch
    Xc = features_batch(recs, susie=True, device="cpu")
    Xg = features_batch(recs, susie=True, device="cuda")
    assert Xc.shape == Xg.shape == (len(recs), len(FEATS))
    assert np.isfinite(Xc).all()
    # counts exact; continuous features equal up to rounding of the iterative SuSiE fits (observed max |diff|
    # 1.3e-6 on susie_PP4 in the first run, job famC_p9_tests_r1)
    cnt = [FEATS.index(k) for k in ("ncs1", "ncs2", "lead_same", "nsig1", "nsig2", "susie_fallback")]
    np.testing.assert_array_equal(Xc[:, cnt], Xg[:, cnt])
    np.testing.assert_allclose(Xc, Xg, rtol=1e-4, atol=1e-5)


def test_floor_method_end_to_end():
    from eval.io import iter_loci, list_loci
    from famC.floor import features_batch
    from famC.floor_method import FloorColoc
    from famC.method import locus_batch_dict
    from famC.features import extract
    from famC.simadapter import compact
    ids = [i for i in list_loci(VG) if "__L2_" in i or "__L0_" in i]
    rng = np.random.default_rng(0)
    ids = list(rng.choice(ids, 24, replace=False))
    loci = list(iter_loci(VG, ids))
    m = FloorColoc(FLOOR_DIR, device="cpu", batch=10)
    rows = m.fit_batch(loci)
    assert [r["locus_id"] for r in rows] == [lc.locus_id for lc in loci]
    assert all(r["status"] == "ok" for r in rows)
    P = np.array([[r["P_S"], r["P_D"], r["P_W"]] for r in rows])
    assert ((P >= 0) & (P <= 1)).all()
    np.testing.assert_allclose(P.sum(1), 1.0, atol=1e-9)
    # the same probabilities from the explicit record path (training-time features of the same loci)
    recs = []
    for lc in loci:
        b, ld = locus_batch_dict(lc, "cpu")
        recs.append(compact(extract(b, ld, M=128)))
    X = features_batch(recs, device="cpu")
    Q = m.bst.predict(X, num_iteration=m.best_iter) / m.cw[None]
    Q = Q / Q.sum(1, keepdims=True)
    np.testing.assert_allclose(P[:, 0], Q[:, 0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(P[:, 1], Q[:, 1], rtol=1e-9, atol=1e-12)
    for r in rows:
        assert set(json.loads(r["extra"])["P5"]) == {"S", "D", "W", "G", "U"}
