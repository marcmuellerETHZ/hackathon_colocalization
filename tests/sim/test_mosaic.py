"""Tests for src/sim/mosaic.py (DESIGN §2.1). Needs the GPU and data/processed (chr22)."""
import numpy as np
import pytest
import torch

from sim import params
from sim.mosaic import Mosaic
from sim.world import source_haplotypes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CH, A, B = "chr22", 30_000_000, 30_400_000


@pytest.fixture(scope="module")
def src():
    return source_haplotypes("A", CH, A, B)


def test_deterministic_and_window_consistent(src):
    Hs, v = src
    m1 = Mosaic("A", 2000, "test_key")
    m2 = Mosaic("A", 2000, "test_key")
    h1 = m1.haplotypes_from_source(CH, Hs, v)
    h2 = m2.haplotypes_from_source(CH, Hs, v)
    assert torch.equal(h1, h2)
    sub = v["pos"].between(30_100_000, 30_200_000).to_numpy()
    h3 = m1.haplotypes_from_source(CH, Hs[:, sub], v[sub].reset_index(drop=True))
    assert torch.equal(h3, h1[:, torch.as_tensor(np.where(sub)[0], device=h1.device)])
    h4 = Mosaic("A", 2000, "other_key").haplotypes_from_source(CH, Hs, v)
    assert not torch.equal(h1, h4)


def test_allele_frequencies_and_copying(src):
    """Mosaic AF tracks pool AF (copying model); every mosaic haplotype segment equals a pool haplotype
    up to miscopy flips."""
    Hs, v = src
    m = Mosaic("A", 20000, "test_af")
    h = m.haplotypes_from_source(CH, Hs, v).cpu().numpy()
    af_m, af_s = h.mean(0), Hs.mean(0)
    assert np.corrcoef(af_m, af_s)[0, 1] > 0.999
    assert np.abs(af_m - af_s).max() < 0.02
    # haplotype 0: every variant equals the allele of some source haplotype path with <= theta mismatches
    base, h_e, j_e, srcv = m._source_index(CH, v["cm"].to_numpy())
    seg = np.zeros(len(v), int)
    for j in j_e[h_e == 0]:
        seg[j:] += 1
    path = srcv[base[0] + seg]
    copied = Hs[path, np.arange(len(v))]
    mism = (copied != h[0]).mean()
    assert mism < 20 * params.THETA + 1e-3


def test_switch_rate(src):
    m = Mosaic("A", 5000, "test_rate", ne=4000)
    offs, s, srcv, L = m.switches(CH)
    n_h = np.diff(offs)
    expected = 4 * 4000 / (m.K * 100) * L
    assert abs(n_h.mean() - expected) < 4 * np.sqrt(expected / len(n_h))
    # sorted within haplotype
    for h in range(5):
        seg = s[offs[h]:offs[h + 1]]
        assert np.all(np.diff(seg) >= 0)
