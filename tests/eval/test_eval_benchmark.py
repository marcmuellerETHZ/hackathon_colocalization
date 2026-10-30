"""End-to-end test of scripts/run_benchmark.py on a tiny synthetic multi-cell locus set (not the project generator):
methods x locus set -> harness runs -> benchmark.json / summary.md / operating points -> --verify; reuse of runs
and given operating points; tamper detection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import eval.io as eio
from eval.io import write_locus_set
from eval.truth import write_truth
from eval_toy import make_locus

import run_benchmark

CELLS = [("L0", "P1", "r_p", 0.0), ("L1", "P1", "r_p", 0.0), ("L2", "P3", "rho", 0.5)]


@pytest.fixture(autouse=True)
def no_splits(monkeypatch):
    monkeypatch.setattr(eio, "_splits_file", lambda: None)
    monkeypatch.delenv(eio.ACCESS_ENV, raising=False)


def build_set(path: Path, n_per_class=14, seed=7, n_blocks=8):
    rng = np.random.default_rng(seed)
    locs, truths, k = [], [], 0
    for L, P, par, v in CELLS:
        cid = f"{L}|{P}|{par}={v}"
        for cls in ("S", "D", "W"):
            for _ in range(n_per_class):
                lam = rng.uniform(6, 14)
                loc, t = make_locus(f"toy__{L}_{P}__{k:04d}", cls, rng, p=40, lam=lam, cell=cid,
                                    block=f"vb{k % n_blocks}", level=L, pairing=P)
                t.update(overlap=v, rho=v if par == "rho" else 0.0)
                locs.append(loc)
                truths.append(t)
                k += 1
    write_locus_set(path, locs)
    write_truth(pd.DataFrame(truths), str(path)[:-3] + ".truth.parquet")
    return pd.DataFrame(truths)


def test_run_benchmark_end_to_end(tmp_path):
    ls = tmp_path / "toybench.h5"
    truth = build_set(ls)
    res_dir = tmp_path / "results"
    mfile = tmp_path / "methods.yaml"
    mfile.write_text(yaml.safe_dump({"methods": [
        {"name": "toy_coloc", "target": "eval_dummy_methods:ToyColoc", "kwargs": {}, "n_workers": 2, "timeout_s": 60},
        {"name": "toy_noisy", "target": "eval_dummy_methods:ToyColoc", "kwargs": {"noise": 0.35, "name": "toy_noisy"},
         "n_workers": 2, "timeout_s": 60},
        {"name": "lead_snp", "target": "eval.reference_methods:LeadSNPRule", "kwargs": {}, "n_workers": 2,
         "timeout_s": 60}]}))
    rc = run_benchmark.main(["--locus-set", str(ls), "--methods", str(mfile), "--results-dir", str(res_dir),
                             "--label", "unit", "--n-boot", "60", "--n-jobs", "1", "--unit", "block",
                             "--candidate", "toy_coloc", "--strength-folds", "3", "--no-hash-inputs"])
    assert rc == 0
    bdirs = sorted((res_dir / "benchmarks").iterdir())
    assert len(bdirs) == 1
    bdir = bdirs[0]
    for f in ("benchmark.json", "summary.md", "operating_points.json", "manifest.json", "expected_ids.txt",
              "strength_table_crossfit.json", "strength_table_full.json"):
        assert (bdir / f).exists(), f
    res = json.loads((bdir / "benchmark.json").read_text())
    assert res["n_loci"] == len(truth) == 126
    assert set(res["methods"]) == {"toy_coloc", "toy_noisy", "lead_snp", "strength_only"}
    assert res["settings"]["least_favourable"] == ["realistic", "ideal"]
    b = res["bootstrap"]["least_favourable"]
    est = b["estimates"]
    # known answers on toy loci: toy_coloc separates S from D (D causal variants far apart) and D from W
    assert est["toy_coloc"]["realistic"]["auprc_sd"]["est"] > 0.9
    assert est["toy_coloc"]["realistic"]["auroc_dw"]["est"] > 0.9
    for m in est:
        for st in ("realistic", "ideal"):
            e = est[m][st]["auprc_sd"]
            assert e["ci_low"] <= e["est"] <= e["ci_high"]
    assert est["lead_snp"]["realistic"]["auroc_dw"]["est"] is None  # no P_D
    # paired contrasts vs the strength-only control and vs the candidate
    assert set(b["vs_control"]) == {"toy_coloc", "toy_noisy", "lead_snp"}
    vc = b["vs_control"]["toy_coloc"]["realistic"]["auprc_sd"]
    assert vc["diff"] == pytest.approx(est["toy_coloc"]["realistic"]["auprc_sd"]["est"]
                                       - est["strength_only"]["realistic"]["auprc_sd"]["est"], abs=1e-12)
    assert set(b["vs_candidate"]) == {"toy_noisy", "lead_snp", "strength_only"}
    # pooled point estimates equal the metrics functions on the same frame (single implementation)
    lf = res["methods"]["toy_coloc"]["least_favourable"]
    assert lf["realistic"]["auprc_sd"] == pytest.approx(est["toy_coloc"]["realistic"]["auprc_sd"]["est"], abs=1e-12)
    assert set(lf["per_cell"]) == {f"{L}|{P}|{par}={v}" for L, P, par, v in CELLS}
    # C3 operating points: confident calls on W <= 2.5% at the fitted thresholds
    op = json.loads((bdir / "operating_points.json").read_text())
    assert set(op) == set(res["methods"])
    r = lf["at_operating_points"]
    assert r["W_called_S"] <= 0.025 + 1e-12 and r["W_called_D"] <= 0.025 + 1e-12
    assert op["lead_snp"]["t_S"] == "inf" and op["lead_snp"]["t_D"] == "inf"
    # calibration raw and recalibrated (intercept fitted on this validation set)
    cal = lf["calibration"]
    assert "ece_S" in cal and "ece_S_recal" in cal and "ece_D_recal" in cal
    # strata reports per v4 edges
    assert list(lf["by_r2_stratum_config_thresholds"]) == ["[0,0.1)", "[0.1,0.3)", "[0.3,0.5)", "[0.5,0.8)",
                                                           "[0.8,0.95)", "out"]
    # C3 dry run: candidate vs eligible baselines (strength control and lead-SNP excluded)
    c3 = res["c3_dry_run"]["least_favourable"]
    assert c3["candidate"] == "toy_coloc" and c3["eligible"] == ["toy_noisy"]
    # runtime and failures are reported per method
    assert res["runtime"]["toy_coloc"]["toybench.h5"]["n"] == 126
    assert lf["failures"]["n_failed"] == 0
    md = (bdir / "summary.md").read_text()
    assert "strength_only" in md and "C3 operating points" in md and "Per-cell AUPRC" in md
    # the strength-only run went through the harness (manifest + ledger) with the cross-fitted table
    man = json.loads((bdir / "manifest.json").read_text())
    assert man["strength_control"]["mode"] == "crossfit_k3"
    led = pd.read_csv(res_dir / "LEDGER.tsv", sep="\t")
    assert len(led) == 5 and led["run_id"].iloc[-1].startswith("benchmark:")

    # --verify recomputes everything (runs via verify_run, benchmark.json via evaluate) and agrees
    assert run_benchmark.main(["--verify", str(bdir)]) == 0
    # tampering is detected
    d = json.loads((bdir / "benchmark.json").read_text())
    d["bootstrap"]["least_favourable"]["estimates"]["toy_coloc"]["realistic"]["auprc_sd"]["est"] = 0.5
    (bdir / "benchmark.json").write_text(json.dumps(d))
    assert run_benchmark.main(["--verify", str(bdir)]) == 1

    # second benchmark: reuse the toy_coloc run, given operating points, fixed strength table, per-cell subsample
    run_dir = man["runs"]["toy_coloc"][str(ls.resolve())]
    rc = run_benchmark.main(["--locus-set", str(ls), "--reuse", f"toy_coloc={run_dir}", "--results-dir",
                             str(res_dir), "--label", "reuse", "--n-boot", "40", "--n-jobs", "1", "--unit", "locus",
                             "--operating-points", str(bdir / "operating_points.json"),
                             "--strength-table", str(bdir / "strength_table_full.json"), "--per-cell", "30",
                             "--no-hash-inputs"])
    assert rc == 0
    b2 = [p for p in sorted((res_dir / "benchmarks").iterdir()) if p.name.endswith("_reuse")][0]
    man2 = json.loads((b2 / "manifest.json").read_text())
    assert man2["runs"]["toy_coloc"][str(ls.resolve())] == run_dir  # no re-run
    assert man2["strength_control"]["mode"] == "given"
    res2 = json.loads((b2 / "benchmark.json").read_text())
    assert res2["n_loci"] == 90  # 30 per cell
    used = res2["methods"]["toy_coloc"]["least_favourable"]["operating_points"]
    assert used["source"] == "given" and used["used"]["t_S"] == pytest.approx(float(op["toy_coloc"]["t_S"]))
    assert res2["bootstrap"]["least_favourable"]["unit"] == "locus"
    # the reused run covers more loci than expected: only the expected ids are evaluated
    assert res2["runtime"]["toy_coloc"]["toybench.h5"]["n"] == 90
    assert run_benchmark.main(["--verify", str(b2)]) == 0


def test_run_benchmark_refuses_test_split(tmp_path, monkeypatch):
    ls = tmp_path / "t.h5"
    rng = np.random.default_rng(1)
    loc, t = make_locus("x0", "S", rng, split="test")
    write_locus_set(ls, [loc], guard=False)
    with pytest.raises(eio.TestSplitAccessError):
        run_benchmark.expected_ids(ls, None)
