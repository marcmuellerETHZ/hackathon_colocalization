"""Family G Method (INTERFACES §4) on LocusData-like inputs: output contract and known answers."""
import json

import numpy as np
import pytest

from famag_testutils import ar_genotypes, toy_locus
from famG.locus import LocusData
from famG.method import GPUColocSusie


def test_method_contract_and_known_answers(cuda):
    rng = np.random.default_rng(4)
    p, n = 250, 1500
    X = ar_genotypes(rng, n, p)
    R = (X.T @ X / n).astype(np.float32)
    maf = np.full(p, 0.3)
    cases = {"shared": ((60,), (60,)), "distinct": ((30,), (220,)), "null": ((), ())}
    loci = []
    for name, (c1, c2) in cases.items():
        _, _, z1, z2 = toy_locus(rng, n, p, c1, [12.0] * len(c1), c2, [12.0] * len(c2), X=X)
        loci.append(LocusData(locus_id=name, z1=z1, z2=z2, n1=np.full(p, 5e4), n2=np.full(p, 5e4), ld_type="dense", R=R,
                              maf1=maf, maf2=maf))
    out = {o["locus_id"]: o for o in GPUColocSusie().fit_batch(loci)}
    for o in out.values():
        assert {"locus_id", "method", "s", "P_S", "P_D", "P_W", "E1", "E2", "status", "runtime_s", "extra"} <= set(o)
        assert o["status"] == "ok"
    assert out["shared"]["P_S"] > 0.9 and out["shared"]["s"] > 0.9
    assert out["distinct"]["P_D"] > 0.9 and out["distinct"]["s"] < 0.1
    assert out["null"]["P_W"] > 0.9            # coloc.abf fallback (no CS) -> H0


def test_abf_fallback_matches_tier1_baseline_convention(cuda):
    """M3.5: G's coloc.abf fallback = the tier-1 coloc-SuSiE wrapper's fallback (src/baselines/methods.py:
    _abf_fallback, W = 0.15, the same p12 as bf_bf) incl. coloc.abf's adjust_prior (p * nsnps >= 1)."""
    from baselines.abf import coloc_abf as base_abf
    from eval import adapters
    from famG.coloc import coloc_abf
    rng = np.random.default_rng(8)
    for p in (300, 12000):                                   # 12,000 SNPs: p1 = p2 = 1e-4 are adjusted
        z1, z2 = rng.normal(0, 2, p), rng.normal(0, 2, p)
        z1[5] = 7.0; z2[5] = 6.0
        se1, se2 = np.full(p, 0.01), np.full(p, 0.05)
        for p12 in (5e-6, 1e-5):
            ours = coloc_abf(z1 * se1, se1, z2 * se2, se2, p12=p12, W=0.15)
            ref = base_abf(z1, se1, z2, se2, W=0.15, p12=p12)
            assert np.abs(ours - ref).max() < 1e-12
    # end-to-end: a null locus (no CS) takes the fallback with the wrapper's p12
    X = ar_genotypes(rng, 1500, 250)
    _, R, z1, z2 = toy_locus(rng, 1500, 250, X=X)
    maf = np.full(250, 0.3)
    lc = LocusData(locus_id="null", z1=z1, z2=z2, n1=np.full(250, 5e4), n2=np.full(250, 5e4), ld_type="dense",
                   R=R.astype(np.float32), maf1=maf, maf2=maf)
    for p12 in (5e-6, 1e-5):
        o = GPUColocSusie(p12=p12).fit_batch([lc])[0]
        se = 1.0 / np.sqrt(2 * 5e4 * maf * (1 - maf))
        ref = adapters.coloc_abf(base_abf(z1, se, z2, se, W=0.15, p12=p12))
        ex = json.loads(o["extra"])
        assert ex["fallback_abf"] and ex["p12_abf"] == p12
        for k in ("s", "P_S", "P_D"):
            assert abs(o[k] - ref[k]) < 1e-12
