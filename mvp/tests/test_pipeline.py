"""Smoke test of the recommended pipeline (mvp/run_coloc_mm.R) on the shipped validation-locus example.

The expected values are the benchmark's own stored coloc_susie_mm predictions for these loci
(mvp/example/expected_output.json). Needs Rscript with susieR 0.16.6 and coloc 5.2.3:
  COLOC_BENCH_RSCRIPT        Rscript to use (default: "Rscript" on PATH)
  COLOC_BENCH_RLIB_SUSIE016  library holding susieR 0.16.6, prepended to R_LIBS (optional)
The test is skipped when these are not available.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

MVP = Path(__file__).resolve().parents[1]
PIPE = MVP / "run_coloc_mm.R"
EX = MVP / "example"
EXPECTED = json.loads((EX / "expected_output.json").read_text())


def _r_env() -> tuple[str, dict]:
    rscript = os.environ.get("COLOC_BENCH_RSCRIPT") or shutil.which("Rscript")
    if not rscript or not Path(rscript).exists() and not shutil.which(rscript):
        pytest.skip("Rscript not available (set COLOC_BENCH_RSCRIPT)")
    env = dict(os.environ, R_PROFILE_USER="/dev/null")
    lib = os.environ.get("COLOC_BENCH_RLIB_SUSIE016")
    if lib:
        env["R_LIBS"] = lib + (":" + env["R_LIBS"] if env.get("R_LIBS") else "")
    chk = subprocess.run([rscript, "-e", 'cat(as.character(packageVersion("susieR")), '
                                         'as.character(packageVersion("coloc")))'],
                         env=env, capture_output=True, text=True)
    if chk.returncode != 0 or chk.stdout.split() != ["0.16.6", "5.2.3"]:
        pytest.skip(f"susieR 0.16.6 / coloc 5.2.3 not available (got {chk.stdout.strip()!r}; "
                    "set COLOC_BENCH_RLIB_SUSIE016)")
    return rscript, env


@pytest.mark.parametrize("name", sorted(EXPECTED["loci"]))
def test_pipeline_reproduces_benchmark(name, tmp_path):
    rscript, env = _r_env()
    exp = EXPECTED["loci"][name]
    out = tmp_path / name
    r = subprocess.run([rscript, str(PIPE), "--sumstats", str(EX / f"{name}.sumstats.tsv.gz"),
                        "--ld-factor", str(EX / f"{name}.ld_factor.txt.gz"), "--n-ref", str(exp["n_ref"]),
                        "--rho-hat", repr(exp["rho_hat"]), "--out", str(out)],
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    got = json.loads(out.with_suffix(".json").read_text())
    tol = EXPECTED["tolerance"]
    for k in ("P_S", "P_D", "P_W", "s"):
        assert abs(got[k] - exp[k]) < tol, (k, got[k], exp[k])
    assert got["call"] == exp["call"]
    assert got["fallback"] == exp["fallback"]
    assert got["versions"]["susieR"] == "0.16.6" and got["versions"]["coloc"] == "5.2.3"
    assert (out.with_suffix(".tsv")).exists()


def test_pipeline_rejects_misaligned_ld(tmp_path):
    rscript, env = _r_env()
    r = subprocess.run([rscript, str(PIPE), "--sumstats", str(EX / "L2_S_locus.sumstats.tsv.gz"),
                        "--ld-factor", str(EX / "L2_W_fallback_locus.ld_factor.txt.gz"), "--n-ref", "150"],
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "columns" in r.stderr
