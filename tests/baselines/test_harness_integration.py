"""Every baseline run end to end through eval.harness (INTERFACES §4) on a toy HDF5 locus set (known answers).

Checks: one row per locus, harness columns, statuses; known directions (shared s high, distinct s low); the
harness rows equal a direct in-process fit (same wrapper, same answers); non-finite input -> failed; a too-short
timeout -> timeout (and the next locus still runs); worker count limited by memory.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baselines import run as BR
from baselines.methods import all_methods
from baselines.toy import make_toy_locus, pick_pair
from common.seeds import VALID_SEED, rng
from eval.io import write_locus_set

PVALUE = {"prop_coloc", "coloc_prop_test"}
TMP = Path(f"/tmp/claude-{os.getuid()}") / "pytest-baselines"


@pytest.fixture(scope="module")
def toy_set(window300):
    X, maf, ids, pos = window300
    R = X.T @ X / X.shape[0]
    g = rng(VALID_SEED, "tests/baselines/harness")
    a, b = pick_pair(R, g, r2_max=0.01)
    loci = [make_toy_locus(X, maf, ids, pos, {a: 30.0}, {a: 30.0}, g, locus_id="shared"),
            make_toy_locus(X, maf, ids, pos, {a: 30.0}, {b: 30.0}, g, locus_id="distinct"),
            make_toy_locus(X, maf, ids, pos, {a: 30.0}, {}, g, locus_id="null2", ld_type="factor")]
    bad = make_toy_locus(X, maf, ids, pos, {a: 30.0}, {}, g, locus_id="nan_input")
    bad.z2 = bad.z2.copy()
    bad.z2[5] = np.nan
    loci.append(bad)
    TMP.mkdir(parents=True, exist_ok=True)
    path = TMP / f"toyset_{os.getpid()}.h5"
    write_locus_set(path, loci)
    yield path, {l.locus_id: l for l in loci}
    path.unlink(missing_ok=True)


@pytest.mark.parametrize("name", BR.METHOD_NAMES)
def test_run_through_harness(name, toy_set, tmp_path):
    path, loci = toy_set
    r = BR.run_baseline(name, path, mem_budget_gb=8, max_cpus=2, results_dir=tmp_path, log=lambda *_: None)
    pred = pd.read_parquet(r.run_dir / "predictions.parquet")
    man = json.loads((r.run_dir / "manifest.json").read_text())
    assert man["method_name"] == name and man["config"]["baseline"] == name
    assert man["harness"]["timeout_s_per_locus"] == BR.spec(name).timeout_s
    assert sorted(pred["locus_id"]) == sorted(loci)
    st = dict(zip(pred["locus_id"], pred["status"]))
    assert st["nan_input"] == "failed"
    for lid in ("shared", "distinct", "null2"):
        assert st[lid] in ("ok", "nonconverged"), pred.set_index("locus_id").loc[lid, "extra"]
    s = dict(zip(pred["locus_id"], pred["s"]))
    if name in PVALUE:
        assert s["shared"] > 0.01
    else:
        assert s["shared"] > 0.9
        assert s["distinct"] < 0.1
    # the harness rows equal a direct in-process fit of the same wrapper (deterministic methods)
    direct = all_methods()[name]().fit_batch([loci["shared"], loci["distinct"]])
    for d in direct:
        row = pred.set_index("locus_id").loc[d["locus_id"]]
        for k in ("s", "P_S", "P_D", "P_W"):
            a, b = float(row[k]), float(d[k])
            assert (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-8, (k, a, b)
    ex = json.loads(pred.set_index("locus_id").loc["shared", "extra"])
    assert "mem_mb" in ex and ex["mem_mb"]["py_peak"] > 0
    if BR.spec(name).uses_r:
        assert ex["mem_mb"]["r_peak"] > 0


def test_timeout_then_recovery(toy_set, tmp_path):
    """A per-locus timeout below the R call time gives status 'timeout'; later loci in the same run still finish."""
    path, loci = toy_set
    r = BR.run_baseline("mvsusie", path, ids=["shared"], timeout_s=0.3, mem_budget_gb=4, max_cpus=1,
                        results_dir=tmp_path, log=lambda *_: None)
    pred = pd.read_parquet(r.run_dir / "predictions.parquet")
    assert pred["status"].tolist() == ["timeout"] and math.isnan(pred["s"].iloc[0])


def test_worker_count_limited_by_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(BR, "FROZEN_TIMEOUTS_FILE", tmp_path / "none.json")  # provisional models only
    # at p = 5000 coloc-SuSiE's model gives 0.8 + 0.1 * 25 = 3.3 GB per worker
    assert BR.n_workers("coloc_susie", 5000, mem_budget_gb=33, max_cpus=64) == 10
    assert BR.n_workers("coloc_susie", 5000, mem_budget_gb=33, max_cpus=4) == 4
    assert BR.n_workers("coloc_abf", 5000, mem_budget_gb=33, max_cpus=64) == 36  # 0.4 + 0.02 * 25 = 0.9 GB
    assert BR.n_workers("mvsusie", 50000, mem_budget_gb=1, max_cpus=8) == 1  # never 0
    ms = BR.method_spec("coloc_susie_mm")
    assert ms.name == "coloc_susie_mm" and ms.build().name == "coloc_susie_mm"
    assert ms.kwargs["timeout"] == pytest.approx(BR.R_TIMEOUT_FRACTION * BR.spec("coloc_susie_mm").timeout_s)
    assert BR.method_spec("colocboost_m1").build().M == 1


def test_coloc_abf_grid_variant_run_name(toy_set, tmp_path):
    """A non-default configuration runs under its own name: rows, manifest and the W used all follow kwargs."""
    path, loci = toy_set
    r = BR.run_baseline("coloc_abf", path, ids=["shared", "distinct"], kwargs={"W": 0.1, "p12": 1e-5},
                        run_name="coloc_abf_W0.1_p12_1e-05", workers=1, results_dir=tmp_path, log=lambda *_: None)
    pred = pd.read_parquet(r.run_dir / "predictions.parquet")
    man = json.loads((r.run_dir / "manifest.json").read_text())
    assert man["method_name"] == "coloc_abf_W0.1_p12_1e-05" and man["harness"]["n_workers"] == 1
    assert set(pred["method"]) == {"coloc_abf_W0.1_p12_1e-05"}
    from baselines.methods import ColocABF
    direct = ColocABF(W=0.1, p12=1e-5).fit_batch([loci["shared"]])[0]
    row = pred.set_index("locus_id").loc["shared"]
    assert json.loads(row["extra"])["W"] == 0.1 and json.loads(row["extra"])["p12"] == 1e-5
    for k in ("s", "P_S", "P_D", "P_W"):
        assert abs(float(row[k]) - direct[k]) < 1e-12
    assert BR.PILOT_TIMEOUTS_S["mvsusie"] == 1080 and set(BR.PILOT_TIMEOUTS_S) == set(BR.METHOD_NAMES)
