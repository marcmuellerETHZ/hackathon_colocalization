"""Family C, C1: the L0* training stream gives the network exactly the inputs an L0* locus set gives a method; the
stream uses the l0star generator settings; the true rho is never a network input; the logit correction of the
Method API is switchable (C1 models are trained with unweighted CE and need none)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def stream_items():
    from famC.c1 import L0StarStream
    st = L0StarStream("pytest", per_seg=4, segs_per_window=3)
    items = st.segment_loci()
    return st, items


def _locusdata(seg, loc, k):
    """The LocusData sim.l0star.build_set writes for (seg, loc) (copied field by field from build_set)."""
    from eval.io import LocusData
    from sim.l0star import ETA
    n_nominal = (100_000, 1_000)
    p = len(seg.pos)
    se1 = 1 / np.sqrt(2 * n_nominal[0] * seg.maf * (1 - seg.maf))
    se2 = 1 / np.sqrt(2 * n_nominal[1] * seg.maf * (1 - seg.maf))
    rho = loc["rho"]
    return LocusData(
        locus_id=f"t__{k}", variant_id=seg.variant_id.astype(object), pos=seg.pos.astype(np.int64),
        z1=loc["z1"], z2=loc["z2"], beta1=loc["z1"] * se1, se1=se1, beta2=loc["z2"] * se2, se2=se2,
        n1=np.full(p, float(n_nominal[0])), n2=np.full(p, float(n_nominal[1])), maf1=seg.maf, maf2=seg.maf,
        maf_ref=seg.maf, ld_type="dense", n_ref=int(1e9), R=seg.Rt.astype(np.float32),
        rho_hat=rho, rho_se=0.0, split="train", block_ids=("x",), mismatch_level="L0star", pairing_type="L0star",
        cell_id=f"L0star|rho={rho}", window_chr=seg.chrom, window_start=int(seg.window[1]),
        window_end=int(seg.window[2]), extra_attrs={"eta": ETA, "trait1_type": "gwas", "trait2_type": "mol"})


def test_stream_uses_l0star_settings(stream_items):
    from sim import l0star
    st, items = stream_items
    assert st.rho_values == tuple(l0star.L0STAR_RHO)
    assert len(items) == 3 * 4 or len(items) % 4 == 0
    for seg, loc in items:
        assert len(seg.pos) == l0star.P_DEFAULT == 20
        assert np.all(seg.maf >= 0.05)
        assert np.allclose(seg.Rt, (1 - l0star.ETA) * seg.R + l0star.ETA * np.eye(20), atol=1e-6)
        assert loc["rho"] in l0star.L0STAR_RHO
        assert loc["oracle"].cls in ("S", "D", "W", "G", "U")
    assert (st.wins["split"].astype(str) == "train").all()


def test_stream_inputs_equal_set_inputs(stream_items):
    """extract() on a stream batch == extract() on the build_set LocusData of the same loci via the Method path."""
    from famC.c1 import l0star_batch
    from famC.features import extract
    from famC.method import locus_batch_dict
    _, items = stream_items
    segs = [x[0] for x in items]
    loci = [x[1] for x in items]
    b, ld = l0star_batch(segs, loci)
    rs = extract(b, ld, M=128)
    for i, (seg, loc) in enumerate(items):
        lc = _locusdata(seg, loc, i)
        lc.validate()
        b1, ld1 = locus_batch_dict(lc, "cuda")
        r1 = extract(b1, ld1, M=128)
        for k in ("tok_raw", "all_raw", "glob_raw", "Rmm", "sel", "valid", "lead1", "lead2"):
            a, c = rs[k][i], r1[k][0]
            if a.dtype.is_floating_point:
                assert torch.allclose(a.double(), c.double(), rtol=1e-5, atol=1e-6), k
            else:
                assert torch.equal(a, c), k


def test_true_rho_never_an_input():
    """extract refuses a batch that carries the truth field `rho` (sim.train_api batches do); rho_hat is read."""
    from famC.features import GLOB_RAW_RHO, DenseLD, extract
    B, p = 2, 6
    dev = "cuda"
    ones = torch.ones(B, p, dtype=torch.float64, device=dev)
    b = dict(z1=torch.randn(B, p, dtype=torch.float64, device=dev), z2=torch.randn(B, p, dtype=torch.float64, device=dev),
             se1=0.1 * ones, se2=0.1 * ones, n1=1e4 * ones, n2=1e3 * ones, maf1=0.3 * ones, maf2=0.3 * ones,
             maf_ref=0.3 * ones, mask=torch.ones(B, p, dtype=torch.bool, device=dev),
             n_ref=torch.full((B,), 1e9, dtype=torch.float64, device=dev),
             rho_hat=torch.tensor([0.25, -0.5], dtype=torch.float64, device=dev),
             rho_se=torch.zeros(B, dtype=torch.float64, device=dev),
             type1=torch.zeros(B, device=dev), type2=torch.ones(B, device=dev))
    ld = DenseLD(torch.eye(p, device=dev)[None].expand(B, p, p).contiguous())
    r = extract(b, ld, M=8)
    assert torch.allclose(r["glob_raw"][:, GLOB_RAW_RHO], b["rho_hat"])
    with pytest.raises(AssertionError, match="true rho"):
        extract(dict(b, rho=torch.tensor([0.9, 0.9], dtype=torch.float64, device=dev)), ld, M=8)


def test_method_logit_correction_switch(tmp_path, stream_items):
    """class_w=None returns softmax(logits); class_w='main' returns softmax(logits - log CLASS_W)."""
    from famC.method import NeuralColoc as Meth
    from famC.model import NeuralColoc as Net
    from famC.train_main import CLASS_W
    torch.manual_seed(0)
    ck = tmp_path / "m.pt"
    torch.save(Net(d=32, layers=1, K=4).state_dict(), ck)
    _, items = stream_items
    loci = [_locusdata(s, l, i) for i, (s, l) in enumerate(items[:4])]
    p_none = np.array([list(json.loads(r["extra"])["P5"].values())
                       for r in Meth(ck, d=32, layers=1, K=4, class_w=None).fit_batch(loci)])
    p_main = np.array([list(json.loads(r["extra"])["P5"].values())
                       for r in Meth(ck, d=32, layers=1, K=4, class_w="main").fit_batch(loci)])
    q = p_none / CLASS_W.numpy()[None]
    q /= q.sum(1, keepdims=True)
    assert np.allclose(q, p_main, atol=1e-6)
    assert not np.allclose(p_none, p_main, atol=1e-3)
