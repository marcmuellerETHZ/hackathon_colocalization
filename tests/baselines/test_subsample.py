"""Known-answer tests of the p4 subsample draws (src/baselines/subsample.py)."""
import numpy as np
import pandas as pd
import pytest

from baselines import subsample as SS
from common.seeds import rng


def _m4_table():
    rows = []
    sizes = {"S": 300, "D": 300, "W": 300, "G": 100, "U": 17}
    for cell in ("L0|P1|r_p=0.0", "L2|P3|rho=0.5"):
        for cls, n in sizes.items():
            rows += [(f"{cell}__{cls}__{i:05d}", cell, cls) for i in range(n)]
    return pd.DataFrame(rows, columns=["locus_id", "cell_id", "oracle_class"])


def test_m4_counts_nesting_and_determinism():
    df = _m4_table()
    out = SS.m4_subsample(df)
    cls = df.set_index("locus_id")["oracle_class"]
    cell = df.set_index("locus_id")["cell_id"]
    for name, k in (("main", 100), ("slow", 40), ("mvsusie", 20)):
        ids = out[name]
        assert len(ids) == len(set(ids))
        c = pd.crosstab(cell.loc[ids], cls.loc[ids])
        for x in ("S", "D", "W"):
            assert (c[x] == k).all()
        if name == "main":
            assert (c["G"] == 30).all() and (c["U"] == 17).all()  # G capped at 30, U (17 < 30) all kept
        else:
            assert "G" not in c and "U" not in c
    assert set(out["mvsusie"]) <= set(out["slow"]) <= set(out["main"])
    # independent of input row order
    out2 = SS.m4_subsample(df.sample(frac=1.0, random_state=1))
    assert out2 == out


def test_m4_draw_is_the_documented_permutation():
    df = _m4_table()
    cell, k = "L2|P3|rho=0.5", "D"
    ids = sorted(df.loc[(df.cell_id == cell) & (df.oracle_class == k), "locus_id"])
    perm = rng(20261029, "m4_subsample", cell, k).permutation(len(ids))
    expect = sorted(ids[i] for i in perm[:20])
    got = [i for i in SS.m4_subsample(df)["mvsusie"] if i.startswith(cell) and "__D__" in i]
    assert got == expect


def test_screen_slow_subsample():
    rows = []
    for cell in ("L0|P1|r_p=0.0", "L2|P1|r_p=0.0", "L4|P3|rho=0.5"):
        for b, n in (("S", 500), ("D_r2lt0.5", 500), ("D_r2_0.5-0.8", 500), ("W", 500)):
            rows += [(f"{cell}__{b}__{i:05d}", cell, b) for i in range(n)]
    df = pd.DataFrame(rows, columns=["locus_id", "cell_id", "bucket"])
    ids = SS.screen_slow_subsample(df)
    d = df.set_index("locus_id").loc[ids]
    n = d.groupby(["cell_id", "bucket"]).size()
    assert (n.loc["L0|P1|r_p=0.0"] == 500).all()  # the A1 cell: all loci
    for c in ("L2|P1|r_p=0.0", "L4|P3|rho=0.5"):
        assert (n.loc[c] == 50).all()
    b_ids = sorted(df.loc[(df.cell_id == "L2|P1|r_p=0.0") & (df.bucket == "W"), "locus_id"])
    perm = rng(20261027, "screen_slow_subsample", "L2|P1|r_p=0.0", "W").permutation(500)
    assert sorted(b_ids[i] for i in perm[:50]) == sorted(i for i in ids if i.startswith("L2|P1|r_p=0.0__W__"))
    assert SS.screen_slow_subsample(df.iloc[::-1]) == ids


def test_rejects_duplicates_and_bad_nesting():
    df = _m4_table()
    with pytest.raises(ValueError):
        SS.m4_subsample(pd.concat([df, df.iloc[:1]]))
    with pytest.raises(ValueError):
        SS.m4_subsample(df, slow=10, mvsusie=20)
