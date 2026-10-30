"""Release smoke tests: the code imports, the recompute script reproduces the shipped benchmark on a subset, and
the headline numbers quoted in README.md are the ones in the shipped benchmark. Needs numpy, scipy, pandas,
pyarrow (and h5py/pyyaml for the import test of the full code base). Runs in about 30 s."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

MODULES = ["common.seeds", "eval.metrics", "eval.adapters", "eval.truth", "eval.io", "baselines.abf",
           "baselines.locus", "baselines.methods", "sim.params", "sim.oracle", "sim.cells", "sim.generator",
           "famA.method", "famB.method", "famG.method", "famG.method_mm", "famC.method", "stats.rho"]


@pytest.mark.parametrize("mod", MODULES)
def test_import(mod):
    try:
        importlib.import_module(mod)
    except ModuleNotFoundError as e:  # optional heavy dependency (torch, h5py, ...) missing: not a code error
        if e.name and e.name.split(".")[0] not in {m.split(".")[0] for m in MODULES}:
            pytest.skip(f"optional dependency missing: {e.name}")
        raise


def test_recompute_subset_reproduces_benchmark(tmp_path):
    """Least-favourable metrics of coloc.abf and mismatch-aware coloc-SuSiE (main + ColocBoost subsample) must
    reproduce benchmark.json, descriptive.json and summary.md (exit status 0 = every value within 1e-6)."""
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "recompute_benchmark.py"),
                        "--methods", "coloc_abf,coloc_susie_mm", "--modes", "least_favourable",
                        "--out", str(tmp_path)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REPRODUCED" in r.stdout
    rep = json.loads((tmp_path / "recompute_test_v1.json").read_text())["report"]
    assert rep["n_mismatch"] == 0 and rep["n_compared"] > 1000 and rep["summary_md_checked"] > 0


def test_headline_numbers():
    """README key result: realistic AUPRC S-vs-D 0.900 (coloc_susie_mm), 0.860 (coloc.abf), 0.717 (coloc-SuSiE);
    validated thresholds t_S 0.840, t_D 0.732."""
    b = json.loads((REPO / "results/test_v1/benchmarks/sim/benchmark.json").read_text())
    est = b["bootstrap"]["least_favourable"]["estimates"]
    got = {m: round(est[m]["realistic"]["auprc_sd"]["est"], 3) for m in ("coloc_susie_mm", "coloc_abf", "coloc_susie")}
    assert got == {"coloc_susie_mm": 0.900, "coloc_abf": 0.860, "coloc_susie": 0.717}
    op = json.loads((REPO / "results/phase2_validation/m4_valid_baselines_v3/operating_points_p2.json").read_text())
    mm = op["methods"]["coloc_susie_mm"]["OP2"]
    assert (mm["t_S"], mm["t_D"]) == (0.8402818292911564, 0.7322585039086077)
    src = (REPO / "mvp" / "run_coloc_mm.R").read_text()
    assert "T_S <- 0.8402818292911564" in src and "T_D <- 0.7322585039086077" in src
