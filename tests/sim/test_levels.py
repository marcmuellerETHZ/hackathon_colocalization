"""Smoke + property tests for every mismatch level and pairing type not covered by the screening cells
(L1, L3, L6, L6c, pool-swap levels, P2) on one chr22 validation window (GPU)."""
import math

import numpy as np
import pytest
import torch

from common.seeds import rng
from sim import params
from sim.generator import CellSpec, WindowContext, draw_candidate, draw_pair_spec, finalize, level_pops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def ctx():
    c = WindowContext("chr22", 30_000_000, 30_600_000, split="valid", block_ids=("x",), anchor_block="x")
    yield c
    c.release()


CASES = [("L1", "P1", 0.6), ("L1", "P2", 0.3), ("L1", "P3", 0.8), ("L2", "P2", 0.0), ("L6", "P1", 0.0),
         ("L6", "P3", 0.0), ("L6c", "P1", 0.0), ("L6c", "P2", 0.0), ("L3", "P1", 0.0), ("L3", "P3", 0.0),
         ("L3", "P2", 0.0), ("L1swap", "P1", 0.6), ("L2swap", "P1", 0.0)]


@pytest.mark.parametrize("level,pairing,ov", CASES)
def test_level_generates_valid_loci(ctx, level, pairing, ov):
    cell = CellSpec(f"{level}|{pairing}|{ov}", level, pairing, ov)
    n_ok = 0
    for k in range(6):
        c = draw_candidate(ctx, cell, rng(20261029, "test_levels", cell.cell_id, k))
        if c is None:
            continue
        loc, tr, ex = finalize(ctx, c, rng(20261029, "test_levels_fin", cell.cell_id, k), f"t{k}")
        loc.validate()
        assert np.all(np.isfinite(loc.z1)) and np.all(np.isfinite(loc.z2))
        exp_nref = {"L1": params.N_L1_REF, "L1swap": params.N_L1_REF, "L2swap": 254}.get(level, 150)
        assert loc.n_ref == exp_nref
        target, pop1, pop2 = level_pops(level)
        assert (tr["pop1"], tr["pop2"]) == (pop1, pop2)
        if level in ("L3", "L6", "L6c"):
            assert tr["rho"] == 0.0
        if pairing == "P2":
            assert (tr["N1"], tr["N2"]) == (100_000, 50_000)
            assert tr["rho_mode"] == "genomewide"
        n_ok += 1
    assert n_ok >= 3


def test_overlap_only_with_rho0_for_population_levels():
    for lvl in ("L3", "L6", "L6c"):
        with pytest.raises(ValueError):
            draw_pair_spec(CellSpec("x", lvl, "P1", 0.6), np.random.default_rng(0))


def test_p2_reachability_assertion():
    with pytest.raises(ValueError):   # rho = 0.7 needs r_p = 0.99 > 0.9 at N1 = 100k, N2 = 50k
        draw_pair_spec(CellSpec("x", "L1", "P2", 0.7), np.random.default_rng(0))
    ps = draw_pair_spec(CellSpec("x", "L1", "P2", 0.3), np.random.default_rng(0))
    assert abs(ps.r_p - 0.3 * math.sqrt(2)) < 1e-12


def test_l6_allelic_scale_conversion(ctx):
    """L6: trait-1 standardized effects are converted by FIN vs target allele frequency."""
    cell = CellSpec("L6|P1|0.0", "L6", "P1", 0.0)
    fin, nfe = ctx.pop("FIN"), ctx.pop("NFE")
    for k in range(20):
        c = draw_candidate(ctx, cell, rng(20261029, "test_l6", k))
        if c is None:
            continue
        for j, spec in c.arch.eff1.items():
            if spec[0] != "pve":
                continue
            expect = spec[2] * math.sqrt(spec[1]) * math.sqrt(fin.af[j] * (1 - fin.af[j]) / (nfe.af[j] * (1 - nfe.af[j])))
            assert abs(float(c.beta1[j]) - expect) < 1e-5 * max(1, abs(expect)) or c.arch.bg


def test_l3_pcs_separate_ancestry(ctx):
    """The top PC of an admixed study sample separates the ASW/ACB-mosaic rows from the NFE-mosaic rows."""
    from sim.generator import TraitSpec, admix_pcs, make_indiv_study
    st = make_indiv_study(ctx, TraitSpec("mol", 1000, "indiv", "ADMIX"), np.random.default_rng(1))
    pcs = admix_pcs(ctx, st.rows).cpu().numpy()
    grp = np.r_[np.zeros(len(st.rows[0][1])), np.ones(len(st.rows[1][1]))]
    assert abs(np.corrcoef(pcs[:, 0], grp)[0, 1]) > 0.9
    # PC-residualised genotypes are orthogonal to the PCs
    assert torch.abs(st.Q.T @ st.Xr).max() < 1e-2
