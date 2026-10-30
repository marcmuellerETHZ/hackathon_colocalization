"""Test set v2 (DESIGN_P2 §6, M8): per-testset lock and generation-key disjointness, known-answer tests.

  * configs/testset_v2.yaml equals configs/testset_v1.yaml except the pre-declared keys;
  * its generation keys (seed 20261101, set testset_v2) are disjoint from v1's (20261028, testset_v1);
  * eval.testset_lock: v1 defaults, lock problems, check_disjoint on identical / different sets;
  * end to end in a throw-away repo (toy generator that writes generate()'s stats format): after a completed and
    committed v1 run, v1 still refuses a second run (also at a frozen-v2 tag); v2 refuses without its own
    `frozen-v2*` tag; with it, the v2 run completes under results/test_access_v2.log with seed 20261101, the
    in-run disjointness check against v1 passes, the v1 log is byte-identical, and a second v2 run is refused;
    a v2 nonce never grants access at a commit without a frozen-v2 tag.
"""
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

import check_frozen
import eval.io as eio
import eval_test
from eval import testset_lock as TL
from sim.testset import main_grid_cells, shifted_cells

from test_eval_testrun import OPTS, prepo  # noqa: F401  (fixture: throw-away repo with a toy v1 test set)

ROOT = Path(__file__).resolve().parents[2]

ALLOWED_V2_DIFF = {"version", "testset_id", "design_ref", "seed", "lock", "disjoint_from", "methods_file"}


def git(repo, *a):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()


# ----------------------------------------------------------------------------------------------------------
# the real configs
# ----------------------------------------------------------------------------------------------------------
def test_v2_config_identical_to_v1_except_declared_keys():
    v1 = yaml.safe_load((ROOT / "datasets" / "configs" / "testset_v1.yaml").read_text())
    v2 = yaml.safe_load((ROOT / "datasets" / "configs" / "testset_v2.yaml").read_text())
    assert v2["version"] == 2 and v2["testset_id"] == "v2" and v2["seed"] == 20261101
    a, b = copy.deepcopy(v1), copy.deepcopy(v2)
    for k in ALLOWED_V2_DIFF:
        a.pop(k, None)
        b.pop(k, None)
    for k in ("seed", "draw"):          # seed labels of the slow subsample (v1 20261028 -> v2 20261101)
        a["slow_method_subsample"].pop(k)
        b["slow_method_subsample"].pop(k)
    a["shifted_arm"].pop("seed")
    b["shifted_arm"].pop("seed")
    a["resolved_items"].pop(5)
    b["resolved_items"].pop(5)
    assert a == b
    assert [c.cell_id for c in main_grid_cells(v1, "test")] == [c.cell_id for c in main_grid_cells(v2, "test")]
    assert [(c.cell_id, n) for c, n in shifted_cells(v1)] == [(c.cell_id, n) for c, n in shifted_cells(v2)]
    lk1 = TL.lock_for(v1, "configs/testset_v1.yaml")
    lk2 = TL.lock_for(v2, "configs/testset_v2.yaml")
    assert (lk1.access_log, lk1.frozen_tag_prefix, lk1.frozen_hashes) == (
        "results/test_access.log", "frozen-", "configs/FROZEN_SHA256")      # v1 = phase-1 behaviour
    assert (lk2.access_log, lk2.frozen_tag_prefix, lk2.frozen_hashes, lk2.set_name) == (
        "results/test_access_v2.log", "frozen-v2", "configs/FROZEN_P2_SHA256", "testset_v2")
    assert TL.lock_problems(lk2) == [] and TL.lock_problems(lk1) == []


def test_real_v1_v2_generation_keys_disjoint():
    """Key tuples and derived seeds of v1 and v2 (all 38 cells, 2,000 candidate indices each) never coincide."""
    v1 = yaml.safe_load((ROOT / "datasets" / "configs" / "testset_v1.yaml").read_text())
    v2 = yaml.safe_load((ROOT / "datasets" / "configs" / "testset_v2.yaml").read_text())

    def st(cfg, shifted):
        cells = [c.cell_id for c, _ in shifted_cells(cfg)] if shifted else [c.cell_id for c in main_grid_cells(cfg, "test")]
        name = f"testset_v{cfg['version']}" + ("_shifted" if shifted else "")
        return {"seed": cfg["seed"], "set": name, "key_prefix": ["shifted"] if shifted else [],
                "cells": {c: 2000 for c in cells}}
    for sh in (False, True):
        r = TL.check_disjoint(st(v2, sh), st(v1, sh))
        assert r["ok"] and r["key_overlap"] == 0 and r["derived_seed_overlap"] == 0
        assert r["derived_seed_collisions_within_new"] == 0
        assert r["new"]["n_keys"] == TL.N_WINDOW_PASSES + 2 * 2000 * len(st(v2, sh)["cells"])


# ----------------------------------------------------------------------------------------------------------
# eval.testset_lock units
# ----------------------------------------------------------------------------------------------------------
def test_lock_problems_and_disjoint_known_answers(tmp_path):
    bad = TL.lock_for({"seed": 20261101, "version": 3, "lock": {"testset_id": "v3", "access_log": "results/x.log",
                                                                 "frozen_tag_prefix": "frozen-", "frozen_hashes": "f",
                                                                 "default_out": "d"}}, "c")
    probs = " | ".join(TL.lock_problems(bad))
    assert "not registered" in probs and "own frozen tag prefix" in probs
    with pytest.raises(ValueError):
        TL.lock_for({"seed": 1, "lock": {"testset_id": "v3"}}, "c")
    a = {"seed": 20261029, "set": "A", "key_prefix": [], "cells": {"c1": 5, "c2": 3}}
    same = TL.check_disjoint(a, dict(a), ["x", "y"], ["y"])
    assert not same["ok"] and same["key_overlap"] == TL.N_WINDOW_PASSES + 2 * 8
    assert same["derived_seed_overlap"] == same["key_overlap"] and same["locus_id_overlap"] == 1
    other = TL.check_disjoint(dict(a, set="B"), a, ["x"], ["y"])
    assert other["ok"] and other["key_overlap"] == 0 and other["derived_seed_overlap"] == 0
    # a prefix alone separates the main grid from the shifted arm
    assert TL.check_disjoint(dict(a, key_prefix=["shifted"]), a)["ok"]
    # partial overlap: a set that re-uses the same seed/name but more candidates overlaps on the common range
    part = TL.check_disjoint(dict(a, cells={"c1": 7}), a)
    assert part["key_overlap"] == TL.N_WINDOW_PASSES + 2 * 5 and not part["ok"]
    # stats files: segments merged (max n_candidates), mixed sets refused
    for i, n in enumerate((4, 9)):
        (tmp_path / f"S.stats.seg{i}.json").write_text(json.dumps(
            {"seed": 20261029, "cells": {"c1": {"n_candidates": n}}, "workers": [{"set": "S", "seed": 20261029,
                                                                                   "key_prefix": []}]}))
    st = TL.read_stats(str(tmp_path / "S.stats*.json"))
    assert st["cells"] == {"c1": 9} and st["set"] == "S" and len(st["files"]) == 2
    (tmp_path / "S.stats.seg2.json").write_text(json.dumps(
        {"seed": 20261029, "cells": {"c1": {"n_candidates": 1}}, "workers": [{"set": "T", "seed": 20261029,
                                                                               "key_prefix": []}]}))
    with pytest.raises(ValueError):
        TL.read_stats(str(tmp_path / "S.stats*.json"))


# ----------------------------------------------------------------------------------------------------------
# end to end: v1 lock unchanged, v2 lock of its own
# ----------------------------------------------------------------------------------------------------------
FAKE_GEN_V2 = '''
import json
from pathlib import Path
import numpy as np, pandas as pd
from common.seeds import rng as seeded_rng
from eval.io import write_locus_set
from eval.truth import write_truth
from eval_toy import make_locus
from sim.cells import _safe, link_set

CALLS = Path(__file__).with_name("gen2_calls.txt")

def _cell(out, base, prefix, cid, n, seed, split):
    g = seeded_rng(seed, *prefix, base, cid, "cand", 0)      # the generator's key layout (toy draws)
    rng = np.random.default_rng(int(g.integers(2**32)))
    lvl, pr, ov = cid.split("|")
    locs, truths = [], []
    for k in range(n):
        lid = f"{base}__{_safe(cid)}__{k:07d}"
        loc, t = make_locus(lid, "SDW"[k % 3], rng, p=30, cell=cid, level=lvl, pairing=pr, block=f"tb{k % 5}",
                            split=split)
        t["overlap"] = float(ov.split("=")[1])
        if pr == "P3":
            t["rho"] = float(t["overlap"])
        locs.append(loc); truths.append(t)
    d = Path(out) / base
    d.mkdir(parents=True, exist_ok=True)
    write_locus_set(d / f"{_safe(cid)}.h5", locs)
    write_truth(pd.DataFrame(truths), d / f"{_safe(cid)}.truth.parquet")
    return {"n_candidates": 3 * n}

def generate(config, out_dir, seed=None, *, main=True, shifted=True, cells_filter=None, stats_suffix="",
             split="test", dry_run=False, set_name=None, n_per_class=None, n_gu=None, n_shifted=None, **kw):
    seed = int(config["seed"] if seed is None else seed)
    with open(CALLS, "a") as fh:
        fh.write(json.dumps({"seed": seed, "main": main, "split": split}) + "\\n")
    base = set_name or f"testset_v{config.get('version', 1)}"
    def stats(name, prefix, cells):
        return {"seed": seed, "cells": cells, "workers": [{"set": name, "seed": seed, "key_prefix": list(prefix)}]}
    if main:
        cells = {cid: _cell(out_dir, base, (), cid, 18, seed, split) for cid in (cells_filter or config["main_grid"]["cells"])}
        Path(out_dir, f"{base}.stats{stats_suffix}.json").write_text(json.dumps(stats(base, (), cells)))
        m, _ = link_set(base, out_dir)
        return [m]
    sname = base + "_shifted"
    cells = {c["cell_id"]: _cell(out_dir, sname, ("shifted",), c["cell_id"], c["n"], seed, split)
             for c in config["shifted_arm"]["cells"]}
    Path(out_dir, f"{sname}.stats.json").write_text(json.dumps(stats(sname, ("shifted",), cells)))
    m, _ = link_set(sname, out_dir)
    return [m]
'''


def _commit(repo, msg):
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", msg)


def test_v1_lock_unchanged_and_v2_lock_end_to_end(prepo):  # noqa: F811
    r = prepo
    (r.parent / "genmod" / "fake_gen_v2.py").write_text(FAKE_GEN_V2)
    v1p = r / "configs" / "testset_v1.yaml"
    ts1 = yaml.safe_load(v1p.read_text())
    ts1["generator"]["entry"] = "fake_gen_v2:generate"
    v1p.write_text(yaml.safe_dump(ts1))
    check_frozen.update(r)
    _commit(r, "toy v1 with stats-writing generator")
    mf = r / "configs" / "test_methods_v1.yaml"
    git(r, "tag", "frozen-v1toy")
    # ---- the v1 test run (phase-1 behaviour, default testset)
    assert eval_test.check_preconditions(r, mf) == []
    res1 = eval_test.run_test(r, mf, r / "data" / "testsets", opts=OPTS)
    assert res1["verify"]["ok"]
    log1 = r / "results" / "test_access.log"
    assert [e.split("\t")[1] for e in log1.read_text().splitlines()[1:]] == ["started", "completed"]
    v1_state = next((r / "results" / "eval_test").glob("test_frozen-v1toy_*/state/sim.json"))
    git(r, "add", "-f", "results/test_access.log")
    git(r, "commit", "-qm", "commit v1 log")
    v1_log_bytes = log1.read_bytes()
    # ---- test set v2 config (toy: same cells), its hash list, the v2 methods file
    ts2 = copy.deepcopy(ts1)
    real2 = yaml.safe_load((ROOT / "datasets" / "configs" / "testset_v2.yaml").read_text())
    ts2.update(version=2, testset_id="v2", seed=20261101, lock=real2["lock"],
               disjoint_from=[{"testset_id": "v1", "config": "configs/testset_v1.yaml",
                               "sim_state": str(v1_state.relative_to(r))}])
    (r / "configs" / "testset_v2.yaml").write_text(yaml.safe_dump(ts2))
    (r / "configs" / "FROZEN_P2_SHA256").write_text("PENDING  configs/testset_v2.yaml\nPENDING  src/eval/metrics.py\n")
    check_frozen.update(r, r / "configs" / "FROZEN_P2_SHA256")
    mf2 = r / "configs" / "test_methods_v2.yaml"
    shutil.copy(mf, mf2)
    _commit(r, "test set v2")
    V2 = "configs/testset_v2.yaml"
    # v1 still refuses a second run, also at a commit carrying a frozen-v2 tag
    git(r, "tag", "frozen-v1toy-again")
    assert any("one test run only" in x for x in eval_test.check_preconditions(r, mf))
    # v2 refuses without its own frozen-v2 tag (a v1-style frozen-* tag is not enough)
    ref = eval_test.check_preconditions(r, mf2, testset_rel=V2)
    assert any("no frozen-v2* tag" in x for x in ref), ref
    git(r, "tag", "frozen-v2toy")
    assert any("one test run only" in x for x in eval_test.check_preconditions(r, mf))   # v1: unchanged
    assert eval_test.check_preconditions(r, mf2, testset_rel=V2) == []
    # a v2-protected file changed -> v2 refuses (FROZEN_P2_SHA256), v1's list is not consulted for v2
    (r / "configs" / "testset_v2.yaml").write_text((r / "configs" / "testset_v2.yaml").read_text() + "\n# drift\n")
    assert any("check_frozen failed" in x for x in eval_test.check_preconditions(r, mf2, testset_rel=V2))
    git(r, "checkout", "--", "configs/testset_v2.yaml")
    # ---- the v2 run
    calls = r.parent / "genmod" / "gen2_calls.txt"
    calls.unlink()
    res2 = eval_test.run_test(r, mf2, r / "data" / "testsets_v2", opts=OPTS, testset_rel=V2)
    assert res2["verify"]["ok"], res2["verify"]["problems"]
    assert log1.read_bytes() == v1_log_bytes                                  # v1 log untouched
    log2 = r / "results" / "test_access_v2.log"
    ev2 = [e.split("\t") for e in log2.read_text().splitlines()[1:]]
    assert [e[1] for e in ev2] == ["started", "completed"] and ev2[0][4] == "frozen-v2toy"
    assert "testset=v2" in ev2[0][5]
    seeds = {json.loads(l)["seed"] for l in calls.read_text().splitlines()}
    assert seeds == {20261101}
    bundle = Path(res2["evaluate"]["benchmark"]["dir"]).parents[1]
    dis = json.loads((bundle / "state" / "disjoint.json").read_text())
    assert dis["ok"] and len(dis["checks"]) == 2
    for c in dis["checks"]:
        assert c["key_overlap"] == 0 and c["derived_seed_overlap"] == 0 and c["locus_id_overlap"] == 0
        assert c["new"]["seed"] == 20261101 and c["ref"]["seed"] == 20261028
        assert c["new"]["set"].startswith("testset_v2") and c["ref"]["set"].startswith("testset_v1")
    sub = json.loads((bundle / "state" / "subsample.json").read_text())
    assert sub["seed"] == [20261101, "slow_subsample"]
    ids = json.loads((bundle / "state" / "sim.json").read_text())["main"]
    assert "testset_v2" in ids
    # after completion: no access; committing the v2 log blocks a second v2 run; v1 still refused
    assert not eio.has_test_access(r)
    git(r, "add", "-f", "results/test_access_v2.log")
    git(r, "commit", "-qm", "commit v2 log")
    git(r, "tag", "frozen-v2toy-again")
    assert any("one test run only" in x for x in eval_test.check_preconditions(r, mf2, testset_rel=V2))
    assert any("one test run only" in x for x in eval_test.check_preconditions(r, mf))


def test_v2_nonce_needs_frozen_v2_tag(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "f").write_text("x")
    git(repo, "add", "f")
    git(repo, "commit", "-qm", "init")
    head = git(repo, "rev-parse", "HEAD")
    for k in ("ORCH_CODE", "ORCH_COMMIT", "ORCH_HOME"):
        monkeypatch.delenv(k, raising=False)
    (repo / "results").mkdir()
    (repo / "results" / "test_access_v2.log").write_text(f"t\tstarted\t{head}\tNV2\tfrozen-v2x\t\n")
    monkeypatch.setenv(eio.ACCESS_ENV, "NV2")
    git(repo, "tag", "frozen-v1x")                  # a v1-style tag only
    assert not eio.has_test_access(repo)
    git(repo, "tag", "frozen-v2x")
    assert eio.has_test_access(repo)
    # the v2 nonce is not a v1 nonce: an open v1 log does not know it
    (repo / "results" / "test_access.log").write_text(f"t\tstarted\t{head}\tNV1\tfrozen-v1x\t\n")
    monkeypatch.setenv(eio.ACCESS_ENV, "NV1")
    assert eio.has_test_access(repo)                # v1 semantics unchanged (any frozen-* tag)
    with open(repo / "results" / "test_access_v2.log", "a") as fh:
        fh.write(f"t\tcompleted\t{head}\tNV2\tfrozen-v2x\t\n")
    monkeypatch.setenv(eio.ACCESS_ENV, "NV2")
    assert not eio.has_test_access(repo)            # closed in its own log
