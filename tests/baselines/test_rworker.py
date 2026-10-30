"""Persistent R worker: library isolation (two susieR versions), array layout, errors, timeouts, respawn."""
import numpy as np
import pytest

from baselines.rworker import RWorker, RWorkerError, RWorkerTimeout


@pytest.mark.parametrize("lib,susie,mvsusie", [("main", "0.14.2", None), ("susie016", "0.16.6", None),
                                               ("mvsusie", "0.16.6", "0.3.0")])
def test_library_isolation(lib, susie, mvsusie):
    w = RWorker(lib)
    try:
        v = w.call("info")["result"]["versions"]
        assert v["susieR"] == susie
        assert v["mvsusieR"] == mvsusie
        assert v["coloc"] == "5.2.3"
        if lib == "main":
            assert v["colocboost"] == "1.0.10" and v["prop.coloc"] == "2.0.0" and v["colocPropTest"] == "0.9.3"
    finally:
        w.close()


def test_array_roundtrip_column_major():
    w = RWorker("main")
    try:
        M = np.arange(12, dtype=float).reshape(3, 4)
        v = np.array([1.5, -2.0, np.pi])
        r = w.call("echo", {}, {"M": M, "v": v})["result"]
        assert r["M"]["dim"] == [3, 4]
        np.testing.assert_array_equal(np.array(r["M"]["values"]).reshape(4, 3).T, M)  # R column-major
        np.testing.assert_array_equal(np.array(r["v"]["values"]), v)
    finally:
        w.close()


def test_error_timeout_respawn():
    w = RWorker("main")
    try:
        pid1 = w.call("info")["result"]["pid"]
        with pytest.raises(RWorkerError, match="deliberate failure"):
            w.call("fail")
        assert w.call("info")["result"]["pid"] == pid1  # an R error does not kill the worker
        with pytest.raises(RWorkerTimeout):
            w.call("sleep", {"seconds": 30}, timeout=1)
        assert not w.alive()
        assert w.call("info")["result"]["pid"] != pid1  # respawned
        assert w.peak_rss_kb() > 0
    finally:
        w.close()


def test_coloc_susie_wrapper_reproduces_coloc_example():
    """ColocSusie wrapper == coloc::coloc.susie(runsusie(D3), runsusie(D4)) on coloc's documented test data."""
    from baselines.locus import make_locus
    from baselines.methods import ColocSusie

    w = RWorker("main")
    try:
        ref = w.call("coloc_example", timeout=300)["result"]
    finally:
        w.close()
    assert ref["LD4_equal"]
    p = len(ref["beta3"])
    b1, v1, b2, v2 = (np.asarray(ref[k], float) for k in ("beta3", "varbeta3", "beta4", "varbeta4"))
    R = np.asarray(ref["LD"], float).reshape(p, p)
    L = make_locus(locus_id="coloc_example", variant_id=np.array([f"s{i+1}" for i in range(p)], dtype=object),
                   pos=np.arange(p), z1=b1 / np.sqrt(v1), z2=b2 / np.sqrt(v2), beta1=b1, se1=np.sqrt(v1), beta2=b2,
                   se2=np.sqrt(v2), n1=np.full(p, float(ref["N3"])), n2=np.full(p, float(ref["N4"])),
                   maf1=np.asarray(ref["MAF3"], float), maf2=np.asarray(ref["MAF4"], float),
                   maf_ref=np.asarray(ref["MAF3"], float), ld_type="dense", n_ref=1000, R=R.astype(np.float32))
    import json

    m = ColocSusie(mismatch=False)  # p12 = 5e-6 = coloc.bf_bf default used by coloc.susie

    row = m.fit_one(L)
    ex = json.loads(row["extra"])
    assert row["status"] == "ok" and not ex["fallback"]
    got = np.asarray(ex["pairs"]["PP"], float)
    exp = np.asarray(ref["PP"], float).reshape(-1, 5)
    # R reference uses LD in float64; the wrapper receives float32 LD (INTERFACES) -> small differences allowed
    order_g = np.lexsort((ex["pairs"]["hit2"], ex["pairs"]["hit1"]))
    order_e = np.lexsort((np.atleast_1d(ref["hit2"]), np.atleast_1d(ref["hit1"])))
    assert got.shape == exp.shape
    print(f"\n[coloc example] {len(got)} CS pairs, max |dPP| = {np.max(np.abs(got[order_g] - exp[order_e])):.2e}")
    np.testing.assert_allclose(got[order_g], exp[order_e], atol=1e-4)


def test_mvsusie_preadjusted_equals_native_N():
    """MvSusie wrapper's per-trait z adjustment + no N == mvsusie_rss(Z, R, N) when N1 = N2."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from baselines.methods import MvSusie
    from baselines.toy import load_window, make_toy_locus
    from common.seeds import VALID_SEED, rng

    X, maf, ids, pos = load_window(300)
    g = rng(VALID_SEED, "tests/baselines/mvsusie_native")
    L = make_toy_locus(X, maf, ids, pos, {100: 12.0, 200: 8.0}, {100: 10.0}, g, N1=2000, N2=2000, rho=0.3)
    R = X.T @ X / X.shape[0]
    V = np.array([[1, 0.3], [0.3, 1]])
    w = RWorker("mvsusie")
    try:
        Za = np.column_stack([MvSusie.adjust_z(L.z1, 2000), MvSusie.adjust_z(L.z2, 2000)])
        a = w.call("mvsusie", {"L": 10}, {"Z": Za, "R": R, "V": V})["result"]
        b = w.call("mvsusie", {"L": 10, "N": 2000}, {"Z": np.column_stack([L.z1, L.z2]), "R": R, "V": V})["result"]
    finally:
        w.close()
    assert np.max(np.abs(np.asarray(a["pip"]) - np.asarray(b["pip"]))) < 1e-6
    np.testing.assert_allclose(np.asarray(a["single_effect_lfsr"]), np.asarray(b["single_effect_lfsr"]), atol=1e-6)


def test_colocboost_patch_only_touches_unnamed_placeholder():
    """README decision 18: the get_integrated_weight patch changes only the crashing unnamed placeholder case."""
    from baselines.rworker import RWorker

    w = RWorker("main")
    try:
        r = w.call("cb_patch_check", {}, {}, timeout=120)["result"]
    finally:
        w.close()
    assert r["named_equal"] is True
    assert r["orig_fails"] is True  # the package bug reproduces
    assert r["placeholder"] == r["placeholder_nullrc"]
