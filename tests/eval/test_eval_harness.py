"""End-to-end harness tests on a tiny synthetic MVN locus set (not the project generator)."""
import fcntl
import json
import os

import numpy as np
import pandas as pd
import pytest

import eval.io as eio
from eval.harness import MethodSpec, machine_lock, run_method
from eval.io import TestSplitAccessError, write_locus_set
from eval.reference_methods import LeadSNPRule
from eval_toy import make_locus, make_locus_set

import verify_run


@pytest.fixture(autouse=True)
def no_splits(monkeypatch):
    monkeypatch.setattr(eio, "_splits_file", lambda: None)
    monkeypatch.delenv(eio.ACCESS_ENV, raising=False)


@pytest.fixture
def toyset(tmp_path):
    f = tmp_path / "toy.h5"
    locs, truth, tp = make_locus_set(f, n_per_class=10, seed=3)
    return f, truth


def test_lead_snp_end_to_end_and_verify(tmp_path, toyset):
    f, truth = toyset
    res_dir = tmp_path / "results"
    r = run_method(MethodSpec("eval.reference_methods:LeadSNPRule", {"r2_min": 0.8}), f, results_dir=res_dir,
                   n_workers=4, timeout_s=60, label="unit", seeds_used={"toy": 3})
    assert r.counts == {"ok": 30, "nonconverged": 0, "failed": 0, "timeout": 0, "expected": 30}
    pred = pd.read_parquet(r.run_dir / "predictions.parquet")
    assert list(pred.columns) == ["locus_id", "method", "s", "P_S", "P_D", "P_W", "E1", "E2", "call", "status",
                                  "runtime_s", "extra"]
    assert list(pred["locus_id"]) == list(truth["locus_id"])
    assert (pred["call"] == "NA").all()  # lead-SNP rule has no probabilities
    assert (pred["runtime_s"] > 0).all()
    # known answers: D causal variants are >= 24 SNPs apart in AR(1) LD (r² ~ 0.8^48): never 'shared'
    m = pred.merge(truth, on="locus_id")
    assert (m.loc[m.oracle_class == "D", "s"] == 0).all()
    assert m.loc[m.oracle_class == "S", "s"].mean() >= 0.5
    # same rule computed in-process gives identical scores
    from eval.io import iter_loci
    local = {o["locus_id"]: o["s"] for o in LeadSNPRule(0.8).fit_batch(list(iter_loci(f)))}
    assert all(local[l] == s for l, s in zip(pred["locus_id"], pred["s"]))
    man = json.loads((r.run_dir / "manifest.json").read_text())
    for k in ("git", "env_lock_sha256", "config", "seeds", "start_utc", "end_utc", "counts", "locus_set", "truth",
              "predictions_sha256", "metrics_sha256", "harness", "method"):
        assert k in man
    assert man["git"]["commit"] and isinstance(man["git"]["clean"], bool)
    assert man["seeds"]["used"] == {"toy": 3} and man["harness"]["lock"] == "shared"
    met = json.loads((r.run_dir / "metrics.json").read_text())
    lf = met["methods"]["lead_snp"]["least_favourable"]
    assert lf["all"]["auprc_sd"] > 0.5
    led = pd.read_csv(res_dir / "LEDGER.tsv", sep="\t")
    assert list(led["run_id"]) == [r.run_id] and led["n_ok"].iloc[0] == 30
    assert verify_run.verify(r.run_dir) == []
    # a second run appends (append-only ledger)
    r2 = run_method(LeadSNPRule(0.5), f, results_dir=res_dir, n_workers=2, timeout_s=60)
    led = pd.read_csv(res_dir / "LEDGER.tsv", sep="\t")
    assert list(led["run_id"]) == [r.run_id, r2.run_id]
    # tampering is detected (positive controls for verify_run)
    mj = r.run_dir / "metrics.json"
    d = json.loads(mj.read_text())
    d["methods"]["lead_snp"]["least_favourable"]["all"]["auprc_sd"] = 0.99
    mj.write_text(json.dumps(d))
    probs = verify_run.verify(r.run_dir)
    assert any("metrics.json sha256" in p for p in probs) and any("auprc_sd" in p for p in probs)
    pp = r2.run_dir / "predictions.parquet"
    p2 = pd.read_parquet(pp)
    p2.loc[0, "s"] = 1 - p2.loc[0, "s"]
    p2.to_parquet(pp, index=False)
    assert any("predictions.parquet sha256" in p for p in verify_run.verify(r2.run_dir))


def test_factor_ld_locus_set(tmp_path):
    f = tmp_path / "fac.h5"
    make_locus_set(f, n_per_class=4, seed=5, ld_type="factor")
    r = run_method(LeadSNPRule(), f, results_dir=tmp_path / "res", n_workers=2, timeout_s=60, timing=True)
    assert r.counts["ok"] == 12
    assert verify_run.verify(r.run_dir) == []
    man = json.loads((r.run_dir / "manifest.json").read_text())
    assert man["harness"]["lock"] == "exclusive" and "machine_end" in man and "loadavg" in man["machine_start"]


def test_timeout_kills_process_group(tmp_path, toyset):
    f, _ = toyset
    pid_file = tmp_path / "grandchild.pid"
    spec = MethodSpec("eval_dummy_methods:SlowMethod", {"slow_ids": ["loc0003"], "pid_file": str(pid_file),
                                                        "sleep_s": 120})
    r = run_method(spec, f, results_dir=tmp_path / "res", n_workers=3, timeout_s=3)
    pred = pd.read_parquet(r.run_dir / "predictions.parquet").set_index("locus_id")
    assert pred.loc["loc0003", "status"] == "timeout"
    assert (pred.drop(index="loc0003")["status"] == "ok").all()
    gpid = int(pid_file.read_text())
    try:
        os.kill(gpid, 0)
        with open(f"/proc/{gpid}/stat") as fh:
            alive = fh.read().split()[2] != "Z"
    except ProcessLookupError:
        alive = False
    assert not alive, "grandchild process of a timed-out worker survived"


def test_failures_crashes_and_outer_join(tmp_path, toyset):
    f, truth = toyset
    spec = MethodSpec("eval_dummy_methods:FlakyMethod", {"fail_ids": ["loc0001"], "nc_ids": ["loc0002"],
                                                         "crash_ids": ["loc0011"], "missing_ids": ["loc0012"],
                                                         "bad_status_ids": ["loc0013"]})
    r = run_method(spec, f, results_dir=tmp_path / "res", n_workers=3, timeout_s=30)
    pred = pd.read_parquet(r.run_dir / "predictions.parquet").set_index("locus_id")
    assert pred.loc["loc0001", "status"] == "failed" and "boom" in pred.loc["loc0001", "extra"]
    assert pred.loc["loc0002", "status"] == "nonconverged"
    assert pred.loc["loc0011", "status"] == "failed" and "worker died" in pred.loc["loc0011", "extra"]
    assert pred.loc["loc0012", "status"] == "failed" and "no output row" in pred.loc["loc0012", "extra"]
    assert pred.loc["loc0013", "status"] == "failed" and "invalid status" in pred.loc["loc0013", "extra"]
    assert r.counts == {"ok": 25, "nonconverged": 1, "failed": 4, "timeout": 0, "expected": 30}
    ok = pred.loc["loc0000"]
    assert json.loads(ok["extra"]) == {"k": 1, "method_runtime_s": 0.001}
    assert ok["call"] == "insufficient"  # P_S = P_D = 0.3 < 0.8
    assert pred.loc["loc0001", "call"] == "NA"
    met = r.metrics["methods"]["flaky"]
    assert met["failures"]["n_failed"] == 4
    assert verify_run.verify(r.run_dir) == []


def test_batch_mode_groups_by_p(tmp_path):
    rng = np.random.default_rng(7)
    locs = [make_locus(f"x{i}", "S", rng, p=(30 if i % 2 else 50))[0] for i in range(20)]
    f = tmp_path / "b.h5"
    write_locus_set(f, locs)
    r = run_method(MethodSpec("eval_dummy_methods:BatchMethod", {"use_gpu": True}), f, results_dir=tmp_path / "res",
                   mode="batch", n_workers=1, batch_size=5, timeout_s=60, truth_path=None)
    pred = pd.read_parquet(r.run_dir / "predictions.parquet")
    ex = [json.loads(e) for e in pred["extra"]]
    assert all(e["batch_size"] == 5 and len(e["ps"]) == 1 for e in ex)  # p-homogeneous batches
    assert list(pred["locus_id"]) == [f"x{i}" for i in range(20)]  # output in expected order
    assert not (r.run_dir / "metrics.json").exists()
    print("device used:", {e["device"] for e in ex})


def test_harness_refuses_test_split(tmp_path):
    t, _ = make_locus("T", "S", np.random.default_rng(1), split="test")
    f = tmp_path / "t.h5"
    write_locus_set(f, [t], guard=False)
    with pytest.raises(TestSplitAccessError):
        run_method(LeadSNPRule(), f, results_dir=tmp_path / "res", n_workers=1)
    assert not (tmp_path / "res" / "LEDGER.tsv").exists()


def test_machine_lock_semantics(tmp_path):
    lock = tmp_path / ".machine.lock"
    with machine_lock(tmp_path, exclusive=False):
        with open(lock, "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_SH | fcntl.LOCK_NB)  # other shared holders allowed
            fcntl.flock(fh, fcntl.LOCK_UN)
        with open(lock, "a") as fh:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)  # timing run must wait
    with machine_lock(tmp_path, exclusive=True):
        with open(lock, "a") as fh:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fh, fcntl.LOCK_SH | fcntl.LOCK_NB)  # nobody else may run during timing
        with open(tmp_path / ".timing.lock", "a") as fh:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
