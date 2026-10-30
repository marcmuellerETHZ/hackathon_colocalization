"""Tests for src/data/pools.py (DESIGN §2.1)."""
from pathlib import Path

import pandas as pd
import pytest

from data.pools import (ROOT, largest_remainder, load_pools, make_pools, read_panel,
                        read_psam_ids)

PANEL = ROOT / "data/processed/1kg/panel.tsv"
PSAM = ROOT / "data/processed/1kg/chr22.psam"
COMMITTED = ROOT / "datasets/splits/pools_v1.tsv"


def test_largest_remainder_known_answer():
    # 150 of (CEU 99, GBR 91, IBS 107, TSI 107): quotas 36.76, 33.79, 39.73, 39.73
    alloc = largest_remainder({"CEU": 99, "GBR": 91, "IBS": 107, "TSI": 107}, 150)
    assert alloc == {"CEU": 37, "GBR": 34, "IBS": 40, "TSI": 39}
    assert largest_remainder({"a": 1, "b": 1}, 1) == {"a": 1, "b": 0}  # tie -> name order
    assert largest_remainder({"a": 5}, 0) == {"a": 0}


def _toy_panel():
    rows = []
    for pop, sup, n in [("CEU", "EUR", 10), ("GBR", "EUR", 10), ("IBS", "EUR", 10), ("TSI", "EUR", 10),
                        ("FIN", "EUR", 5), ("ASW", "AFR", 3), ("ACB", "AFR", 4), ("YRI", "AFR", 6)]:
        rows += [(f"{pop}{i:03d}", pop, sup, "female") for i in range(n)]
    return pd.DataFrame(rows, columns=["sample_id", "pop", "superpop", "sex"])


def test_toy_pools(monkeypatch):
    import data.pools as P
    monkeypatch.setattr(P, "N_POOL_B", 12)
    monkeypatch.setattr(P, "N_L6C", 8)
    panel = _toy_panel()
    pools = P.make_pools(panel, panel["sample_id"].tolist())
    g = {p: set(d["sample_id"]) for p, d in pools.groupby("pool")}
    assert len(g["B"]) == 12 and len(g["A"]) == 28 and len(g["L6c"]) == 8
    assert g["A"].isdisjoint(g["B"]) and g["L6c"] <= g["A"]
    assert g["FIN"] == {f"FIN{i:03d}" for i in range(5)}
    assert len(g["ASW_ACB"]) == 7 and not any(s.startswith("YRI") for s in pools["sample_id"])
    # each NFE subpop contributes exactly 3 to B (12 * 10/40)
    b = pools[pools.pool == "B"].groupby("pop").size()
    assert (b == 3).all()
    # determinism; and a sample absent from psam is never drawn
    assert pools.equals(P.make_pools(panel, panel["sample_id"].tolist()))
    pools2 = P.make_pools(panel, [s for s in panel["sample_id"] if s != "CEU000"])
    assert "CEU000" not in set(pools2["sample_id"])


@pytest.mark.skipif(not PANEL.exists(), reason="real panel not available")
def test_real_pools_match_design_and_committed_file():
    pools = make_pools(read_panel(PANEL), read_psam_ids(PSAM))
    g = {p: set(d["sample_id"]) for p, d in pools.groupby("pool")}
    sizes = {p: len(s) for p, s in g.items()}
    assert sizes == {"A": 254, "B": 150, "FIN": 99, "ASW_ACB": 157, "L6c": 99}
    nfe = set(pools[pools["pop"].isin(["CEU", "GBR", "IBS", "TSI"])]["sample_id"])
    assert g["A"] | g["B"] == nfe and len(nfe) == 404 and g["A"].isdisjoint(g["B"])
    assert g["L6c"] <= g["A"] and g["L6c"].isdisjoint(g["B"])
    # stratification: each subpop's B share within 1 of proportional
    for pop, d in pools[pools.pool.isin(["A", "B"])].groupby("pop"):
        nb = (d.pool == "B").sum()
        assert abs(nb - 150 * len(d) / 404) < 1
    committed = pd.read_csv(COMMITTED, sep="\t")
    pd.testing.assert_frame_equal(committed, pools.reset_index(drop=True))
    assert {k: len(v) for k, v in load_pools(COMMITTED).items()} == sizes
