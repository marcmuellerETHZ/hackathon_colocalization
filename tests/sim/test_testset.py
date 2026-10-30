"""Evaluation-grid builder (src/sim/testset.py, GridPlan) and the test lock (DESIGN §7.3)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from eval.io import TestSplitAccessError, iter_loci
from eval.truth import read_truth
from sim.cells import GridPlan, d_stratum
from sim.testset import generate, main_grid_cells, shifted_cells

ROOT = Path(__file__).resolve().parents[2]

CFG = {"version": 1, "seed": 20261028,
       "main_grid": {"mismatch_levels": ["L0", "L1", "L2", "L3", "L4", "L6"],
                     "pairing_overlap": [{"pairing": "P1", "param": "r_p", "value": 0.0, "nesting": "nested"},
                                         {"pairing": "P1", "param": "r_p", "value": 0.6, "nesting": "nested"},
                                         {"pairing": "P2", "param": "rho", "value": 0.0},
                                         {"pairing": "P2", "param": "rho", "value": 0.3},
                                         {"pairing": "P3", "param": "rho", "value": 0.0},
                                         {"pairing": "P3", "param": "rho", "value": 0.5},
                                         {"pairing": "P3", "param": "rho", "value": 0.8}],
                     "n_cells": 34},
       "per_cell": {"S": 400, "D": 400, "W": 400, "G_U": {"max": 200}},
       "shifted_arm": {"n_loci": 6000, "class_mix": {"S": 0.15, "D": 0.15, "W": 0.5, "M": 0.2},
                       "biobank_strong_share": 0.5, "max_causal_per_trait": 4, "polygenic_background_share": 0.5}}


def test_34_cells_with_population_level_exclusion():
    cells = main_grid_cells(CFG, "test")
    assert len(cells) == 34
    for c in cells:
        if c.level in ("L3", "L6"):
            assert c.overlap == 0.0
        if c.pairing == "P3":
            assert c.both_signs is False          # test: positive rho
        if c.pairing == "P1":
            assert c.p1_mode == "nested"
    assert all(c.both_signs for c in main_grid_cells(CFG, "valid") if c.pairing == "P3")
    bad = dict(CFG, main_grid=dict(CFG["main_grid"], n_cells=42))
    with pytest.raises(ValueError):
        main_grid_cells(bad, "test")
    sh = shifted_cells(CFG)
    assert sum(n for _, n in sh) == 6000 and {c.level for c, _ in sh} == {"L1swap", "L2swap"}
    assert {c.overlap for c, _ in sh} == {0.0, 0.6}


def _orc(cls, r2=np.nan, d=np.nan):
    return SimpleNamespace(cls=cls, r2=r2, d=d)


def test_gridplan_staged_relaxation():
    plan = GridPlan(n_S=2, n_D=40, n_W=2, n_G=1, n_U=1, k1=100, k2=200, by="r2xd")
    assert sum(plan.q.values()) == 40 and set(plan.q.values()) == {2}
    filled = {}

    def put(o, k):
        b = plan.assign(o, k, filled)
        if b is not None:
            filled[b] = filled.get(b, 0) + 1
        return b

    assert d_stratum(0.05, 5.0) == (0, 1) and d_stratum(0.97, 3.0) == ("ge0.95", 0)
    assert put(_orc("D", 0.05, 5.0), 0) == "D_r0_d1"
    assert put(_orc("D", 0.05, 5.0), 1) == "D_r0_d1"
    assert put(_orc("D", 0.05, 5.0), 2) is None              # stratum full in stage 0
    assert put(_orc("D", 0.05, 5.0), 100) == "D_r0_d1"      # stage 1: r² bin 0 quota (8) not full
    assert put(_orc("D", 0.97, 5.0), 150) is None            # r² >= 0.95 only in stage 2
    assert put(_orc("D", 0.97, 5.0), 200) == "D_rge0.95_d1"
    assert put(_orc("G"), 0) == "G" and put(_orc("G"), 1) is None
    assert put(_orc("S"), 0) == "S" and put(_orc("S"), 0) == "S" and put(_orc("S"), 0) is None
    assert not plan.done(filled)
    for k in range(60):
        put(_orc("D", 0.4, 9.0), 300 + k)
    put(_orc("W"), 0)
    put(_orc("W"), 0)
    assert plan.done(filled)
    assert sum(v for k, v in filled.items() if k.startswith("D_")) == 40
    p2 = GridPlan(n_S=1, n_D=10, n_W=1, n_G=0, n_U=0, k1=5, k2=10, by="r2")
    assert set(p2.q.values()) == {2} and len(p2.q) == 5
    f2 = {}
    assert p2.assign(_orc("D", 0.6, 3.0), 0, f2) == "D_r3_dall"


def test_config_file_parses():
    """The eval agent's datasets/configs/testset_v1.yaml (v4) gives 34 cells and the 4 shifted cells."""
    cfg = yaml.safe_load((ROOT / "datasets/configs/testset_v1.yaml").read_text())
    cells = main_grid_cells(cfg, "test")
    assert len(cells) == 34
    sh = shifted_cells(cfg)
    assert sum(n for _, n in sh) == int(cfg["shifted_arm"]["n_loci"])


def test_test_split_refused_without_eval_test():
    with pytest.raises(TestSplitAccessError):
        generate(CFG, "/tmp/claude-should-not-exist", split="test")
    with pytest.raises(TestSplitAccessError):
        generate(CFG, "/tmp/claude-should-not-exist", split="test", dry_run=True)
    with pytest.raises(ValueError):
        generate(CFG, "/tmp/claude-should-not-exist", split="valid", dry_run=False)
    from sim.windows import split_windows
    with pytest.raises(PermissionError):
        split_windows("test")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_validation_dry_run(tmp_path):
    """Tiny dry run on chr22 validation windows: two cells + shifted arm; format, split, buckets, seeds."""
    cells = ["L2|P1|r_p=0.6", "L3|P3|rho=0.0"]
    paths = generate(CFG, tmp_path, 20261029, split="valid", dry_run=True, set_name="dry", n_per_class=2,
                     n_gu=1, workers=1, cells_filter=cells, chroms=["chr22"], n_shifted=8, k1_per_d=10,
                     k2_per_d=20, log=lambda s: None)
    assert [p.name for p in paths] == ["dry.h5", "dry_shifted.h5"]
    t = read_truth(tmp_path / "dry.truth.parquet")
    assert set(t["cell_id"]) == set(cells)
    for cid, sub in t.groupby("cell_id"):
        cnt = sub["oracle_class"].value_counts()
        assert cnt.get("S", 0) == 2 and cnt.get("W", 0) == 2 and cnt.get("D", 0) == 2
        assert cnt.get("G", 0) <= 1 and cnt.get("U", 0) <= 1
    assert (t["split"] == "valid").all()
    assert (t.loc[t["mismatch_level"] == "L3", "rho"] == 0).all()
    ts = read_truth(tmp_path / "dry_shifted.truth.parquet")
    assert len(ts) == 8 and set(ts["mismatch_level"]) == {"L1swap", "L2swap"}
    n = 0
    for L in iter_loci(tmp_path / "dry.h5"):
        assert np.allclose(L.beta1 / L.se1, L.z1) and np.isfinite(L.rho_hat)
        n += 1
    assert n == len(t)


def test_worker_access_check_does_not_need_eval_test_on_stack(monkeypatch, tmp_path):
    """Review p2.review_sim defect 1: generate() runs cells in spawned pool workers (workers=4 is what
    scripts/eval_test.py gets); a spawned worker's stack never contains scripts/eval_test.py, so the worker check
    must rely on has_test_access() alone (the parent already checked the caller). Never generates anything."""
    import json
    import subprocess
    import sys
    import textwrap

    import eval.io as eio
    from sim.testset import _check_access

    monkeypatch.setattr(eio, "has_test_access", lambda *a, **k: True)
    with pytest.raises(TestSplitAccessError):
        _check_access("test", False)                       # caller is not scripts/eval_test.py
    _check_access("test", False, in_worker=True)           # worker of an authorised run: allowed
    monkeypatch.setattr(eio, "has_test_access", lambda *a, **k: False)
    with pytest.raises(TestSplitAccessError):
        _check_access("test", False, in_worker=True)       # no open eval_test.py nonce: refused everywhere
    with pytest.raises(TestSplitAccessError):
        _check_access("test", True, in_worker=True)        # dry runs never use the test split
    # why: the stack check is True in the eval_test.py process but False in its spawned workers
    d = tmp_path / "scripts"
    d.mkdir()
    # the pool target lives in a module (as sim.testset._worker does), not in the eval_test.py script itself
    (tmp_path / "p2rs_helper.py").write_text(textwrap.dedent('''
        def child(_):
            from sim.testset import _called_from_eval_test
            return _called_from_eval_test()
    '''))
    (d / "eval_test.py").write_text(textwrap.dedent(f'''
        import sys, json, multiprocessing as mp
        sys.path[:0] = [{str(ROOT / "src")!r}, {str(tmp_path)!r}]
        from p2rs_helper import child

        if __name__ == "__main__":
            from sim.testset import _called_from_eval_test
            with mp.get_context("spawn").Pool(1) as pool:
                kids = pool.map(child, [0])
            print(json.dumps([_called_from_eval_test(), kids[0]]))
    '''))
    out = subprocess.run([sys.executable, str(d / "eval_test.py")], capture_output=True, text=True, timeout=600)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == [True, False], out.stderr[-2000:]
