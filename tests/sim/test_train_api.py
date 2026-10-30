"""Family C training-data API (src/sim/train_api.py): train split only, held-out mechanism, shapes, LD block
equals the LD given to methods, determinism."""
import numpy as np
import pytest
import torch

from sim.train_api import LEVELS, TrainSampler, collate, default_screen

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_train_split_only():
    with pytest.raises(ValueError):
        TrainSampler(split="valid")


def test_default_screen_union():
    g = np.random.default_rng(0)
    p = 500
    z1, z2 = g.standard_normal(p), g.standard_normal(p)
    z1[10], z2[400] = 12.0, -9.0
    rows = lambda idx: torch.eye(p, dtype=torch.float64)[idx]   # noqa: E731  (identity LD)
    idx = default_screen(z1, z2, rows, 32)
    assert len(idx) == 32 and len(set(idx.tolist())) == 32 and np.all(np.diff(idx) > 0)
    assert 10 in idx and 400 in idx
    assert np.array_equal(default_screen(z1[:20], z2[:20], rows, 32), np.arange(20))


def test_batches_shapes_holdout_and_determinism():
    ts = TrainSampler(M=64, loci_per_window=12, buffer_windows=1)
    loci = ts.window_loci(0)
    assert len(loci) >= 10
    assert all(LEVELS[d["level"]] != "L6" for d in loci)          # held-out mechanism
    b = collate(loci)
    B = len(loci)
    assert b["z1"].shape == (B, 64) and b["R"].shape == (B, 64, 64) and b["y_class"].shape == (B,)
    m = b["mask"]
    assert torch.all(b["z1"][~m] == 0) and torch.all(b["R"][0][~m[0]] == 0)
    # LD block: symmetric, unit diagonal on valid entries
    for i in range(B):
        k = int(m[i].sum())
        R = b["R"][i, :k, :k]
        assert torch.allclose(R, R.T, atol=1e-5) and torch.allclose(torch.diagonal(R), torch.ones(k, device=R.device), atol=1e-4)
    assert torch.all(b["shared"] == (b["causal1"] & b["causal2"]))
    assert torch.all(b["proxy1"][b["causal1"]] > 0.999)
    # deterministic given (seed, step)
    again = TrainSampler(M=64, loci_per_window=12).window_loci(0)
    assert len(again) == len(loci)
    for x, y in zip(loci, again):
        assert torch.equal(x["z1"], y["z1"]) and x["y_class"] == y["y_class"] and x["rho_hat"] == y["rho_hat"]
    # held-out flag can be lifted
    ts2 = TrainSampler(M=32, loci_per_window=40, holdout=())
    assert "L6" in ts2.levels
