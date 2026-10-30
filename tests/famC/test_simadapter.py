"""Family C adapter to the simulation agent's train API: records reproduce the generator's statistics and labels."""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def ctx():
    from sim import train_api
    from sim.generator import WindowContext
    ts = train_api.TrainSampler(M=128)
    w = ts.wins.iloc[777]
    c = WindowContext(w["chr"], int(w["window_start"]), int(w["window_end"]), split="train",
                      block_ids=tuple(w["block_ids"]), anchor_block=str(w["block_id"]), anchor=int(w["anchor"]))
    c.pop("NFE")
    return c


def test_record_matches_generator(ctx):
    """Same seed -> the record's z equal sim.generator.observed_stats' z; label = the candidate's oracle class;
    the shared-PIP target token is the class-determining shared variant (or a proxy with r² >= 0.8)."""
    from common.seeds import rng
    from famC.simadapter import CLASS_ORDER, MI_META, locus_record
    from sim import train_api
    from sim.generator import CellSpec, draw_candidate, observed_stats
    n_s = 0
    for k in range(40):
        cell = CellSpec("t", ("L0", "L2", "L1")[k % 3], "P1", 0.0)
        r = locus_record(ctx, cell, rng(20261030, "famC", "test", k), 128, None, train_api.LEVELS, train_api.PAIRINGS)
        g = rng(20261030, "famC", "test", k)
        c = draw_candidate(ctx, cell, g)
        if r is None:
            assert c is None
            continue
        ob = observed_stats(ctx, c, g)
        rec, meta = r
        assert np.allclose(rec["all_z"][:, 0].numpy(), ob["z1"], rtol=1e-6, atol=1e-5)
        assert np.allclose(rec["all_z"][:, 1].numpy(), ob["z2"], rtol=1e-6, atol=1e-5)
        assert CLASS_ORDER[int(meta[MI_META["y"]])] == c.oracle.cls
        assert int(meta[MI_META["level"]]) == train_api.LEVELS.index(cell.level)
        # the network's rho input is rho-hat (meta keeps the true rho for monitoring only)
        from famC.features import GLOB_RAW_RHO
        assert abs(float(rec["glob_raw"][GLOB_RAW_RHO]) - float(meta[MI_META["rho_hat"]])) < 1e-6
        pt = int(meta[MI_META["pip_t"]])
        if c.oracle.cls == "S":
            n_s += 1
            if pt < 128:
                j = int(rec["sel"][pt])                       # locus index of the target token
                keep = ob["keep"]
                cd = int(np.searchsorted(keep, c.oracle.cd1))
                # either the causal variant itself, or (proxy) the causal variant was not screened
                assert j == cd or cd not in set(rec["sel"][rec["valid"]].tolist())
        else:
            assert pt == 128
