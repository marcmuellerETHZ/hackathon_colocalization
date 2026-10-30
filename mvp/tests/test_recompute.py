"""The benchmark recompute through the mvp wrapper (mvp/recompute_benchmark.py), and the thresholds of the
recommended pipeline against the phase-2 operating points shipped in results/. Needs numpy, scipy, pandas, pyarrow."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

MVP = Path(__file__).resolve().parents[1]
REPO = MVP.parent


def test_wrapper_reproduces_benchmark(tmp_path):
    """Least-favourable metrics of the recommended method and coloc.abf (main benchmark and ColocBoost subsample)
    reproduce benchmark.json, descriptive.json and summary.md exactly (exit status 0, no mismatches)."""
    r = subprocess.run([sys.executable, str(MVP / "recompute_benchmark.py"), "--methods", "coloc_abf,coloc_susie_mm",
                        "--modes", "least_favourable", "--out", str(tmp_path)],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REPRODUCED" in r.stdout
    rep = json.loads((tmp_path / "recompute_test_v1.json").read_text())["report"]
    assert rep["n_mismatch"] == 0 and rep["n_compared"] > 1000 and rep["summary_md_checked"] > 0


def test_pipeline_thresholds_match_validated_operating_point():
    op = json.loads((REPO / "results/phase2_validation/m4_valid_baselines_v3/operating_points_p2.json").read_text())
    mm = op["methods"]["coloc_susie_mm"]["OP2"]
    src = (MVP / "run_coloc_mm.R").read_text()
    t_s = float(re.search(r"^T_S <- ([0-9.]+)", src, re.M).group(1))
    t_d = float(re.search(r"^T_D <- ([0-9.]+)", src, re.M).group(1))
    assert (t_s, t_d) == (mm["t_S"], mm["t_D"]) == (0.8402818292911564, 0.7322585039086077)


def test_headline_auprc():
    b = json.loads((REPO / "results/test_v1/benchmarks/sim/benchmark.json").read_text())
    est = b["bootstrap"]["least_favourable"]["estimates"]
    got = {m: round(est[m]["realistic"]["auprc_sd"]["est"], 3) for m in ("coloc_susie_mm", "coloc_abf", "coloc_susie")}
    assert got == {"coloc_susie_mm": 0.900, "coloc_abf": 0.860, "coloc_susie": 0.717}
