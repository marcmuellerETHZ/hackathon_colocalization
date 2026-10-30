"""Family C: record compaction round-trip, Method API on LocusData (dense == factor LD), output contract."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from eval.io import LocusData
from famC.c0world import sample_c0
from famC.features import FactorLD, build, extract
from famC.ldbank import LDBank, torch_gen
from famC.model import NeuralColoc as Net
from famC.simadapter import collate_records, compact

SEED = 20261030
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def bank():
    return LDBank("train", 3, SEED)


def test_compact_collate_roundtrip(bank):
    """build(collate(compact(extract(locus)))) == build(extract(batch)) up to the fp16 storage of non-z columns."""
    g = torch_gen(SEED, "test", "rt", device="cuda")
    gc = torch_gen(SEED, "test", "rtc")
    b = sample_c0(bank, g, gc, n_win=1, per_win=4, widx=[1])
    full = build(extract(b, FactorLD(b["X"], 4), M=64))
    recs = []
    for i in range(4):
        bi = {k: (v[i:i + 1] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 4 else v) for k, v in b.items()}
        recs.append(compact(extract(bi, FactorLD(b["X"], 1), M=64)))
    rt = build(collate_records(recs))
    assert torch.equal(full["sel"], rt["sel"])
    for k in ("tok", "q_tok", "q_all", "glob"):
        assert torch.allclose(full[k], rt[k], rtol=1e-3, atol=2e-3), k
    assert torch.allclose(full["Rmm"], rt["Rmm"], atol=1e-3)


def _locus(bank, i, ld_type):
    g = torch_gen(SEED, "test", "loc", device="cuda")
    gc = torch_gen(SEED, "test", "locc")
    b = sample_c0(bank, g, gc, n_win=1, per_win=1, widx=[i], hyp_probs=[0, 0, 0, 0, 1])
    X = b["X"][0].cpu().numpy()
    n = X.shape[0]
    m = b["mask"][0].cpu().numpy()
    X = X[:, m]
    p = X.shape[1]
    f = lambda k: b[k][0].cpu().numpy()[m]
    R = (X.T.astype(np.float64) @ X / n).astype(np.float32) if ld_type == "dense" else None
    return LocusData(locus_id=f"toy{i}", variant_id=np.array([f"v{j}" for j in range(p)], dtype=object),
                     pos=np.arange(p, dtype=np.int64), z1=f("z1"), z2=f("z2"), beta1=f("z1") * f("se1"), se1=f("se1"),
                     beta2=f("z2") * f("se2"), se2=f("se2"), n1=f("n1"), n2=f("n2"), maf1=f("maf1"), maf2=f("maf2"),
                     maf_ref=f("maf_ref"), ld_type=ld_type, n_ref=n, R=R, X_ref=X if ld_type == "factor" else None,
                     rho_hat=0.0, rho_se=0.05, split="train", extra_attrs={"trait1_type": "gwas", "trait2_type": "mol"})


def test_method_dense_equals_factor(bank, tmp_path):
    from famC.method import NeuralColoc
    torch.manual_seed(0)
    net = Net(d=32, layers=1, H=4, K=4)
    ck = tmp_path / "m.pt"
    torch.save(net.state_dict(), ck)
    m = NeuralColoc(ck, d=32, layers=1, K=4)
    rows_d = m.fit_batch([_locus(bank, 0, "dense"), _locus(bank, 2, "dense")])
    rows_f = m.fit_batch([_locus(bank, 0, "factor"), _locus(bank, 2, "factor")])
    for a, b in zip(rows_d, rows_f):
        assert a["status"] == b["status"] == "ok"
        for k in ("s", "P_S", "P_D", "P_W"):
            assert abs(a[k] - b[k]) < 1e-4, (k, a[k], b[k])
        assert set(a) == {"locus_id", "method", "s", "P_S", "P_D", "P_W", "E1", "E2", "call", "status", "runtime_s",
                          "extra"}
        p5 = json.loads(a["extra"])["P5"]
        assert abs(sum(p5.values()) - 1) < 1e-6
        assert 0 <= a["P_W"] <= 1 and np.isnan(a["E1"])


def test_method_reports_failure(bank, tmp_path):
    from famC.method import NeuralColoc
    net = Net(d=32, layers=1, H=4, K=4)
    ck = tmp_path / "m.pt"
    torch.save(net.state_dict(), ck)
    m = NeuralColoc(ck, d=32, layers=1, K=4)
    bad = _locus(bank, 0, "dense")
    bad.R = bad.R[:-1, :-1]          # inconsistent LD -> must be reported, not swallowed
    rows = m.fit_batch([bad, _locus(bank, 2, "dense")])
    assert rows[0]["status"] == "failed" and "error" in json.loads(rows[0]["extra"])
    assert rows[1]["status"] == "ok"


def test_method_batched_equals_single(bank, tmp_path):
    """Batched extract over loci of different p and LD types == one locus at a time."""
    from famC.method import NeuralColoc
    torch.manual_seed(0)
    net = Net(d=32, layers=1, H=4, K=4)
    ck = tmp_path / "m.pt"
    torch.save(net.state_dict(), ck)
    m = NeuralColoc(ck, d=32, layers=1, K=4)
    loci = [_locus(bank, 0, "dense"), _locus(bank, 1, "factor"), _locus(bank, 2, "factor")]
    together = m.fit_batch(loci)
    single = [m.fit_batch([lc])[0] for lc in loci]
    for a, b in zip(together, single):
        for k in ("s", "P_S", "P_D"):
            assert abs(a[k] - b[k]) < 1e-5, (k, a[k], b[k])
