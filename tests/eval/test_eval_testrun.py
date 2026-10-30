"""p6.dryrun_testpipeline: the single-test-run pipeline (src/eval/testrun.py via scripts/eval_test.py) and the
fixes of review p5_verify_infra in the eval area. Known-answer tests on toy loci (eval_toy; not the project
generator) in throw-away git repos:

  * defect 1  : git_info / code_identity / has_test_access inside an orch snapshot use $ORCH_COMMIT, not the main
                checkout's (moved, dirty) HEAD;
  * defect 4  : the access cache is keyed on the log content (a `completed` line with an unchanged mtime closes
                access);
  * defect 2  : the whole pipeline (slow-method subsamples, real arms, strength control, C1/C3 evaluation,
                verify) runs inside the open access window; an interrupted run is resumed under the same nonce;
                truth is unreadable after `completed`; thresholds are required;
  * defect 6  : run_benchmark --method-ids evaluates a subsampled method only on its own ids;
  * harness   : per-locus checkpoint resume, batch retry, LPT order.
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import check_frozen
import eval.io as eio
import eval_test
import run_benchmark
from eval import harness as H
from eval import strength_control as SC
from eval.io import write_locus_set
from eval.truth import write_truth
from eval_toy import make_locus

ROOT = Path(__file__).resolve().parents[2]


def git(repo, *a):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()


def _events(repo):
    return [l.split("\t") for l in (repo / "results" / "test_access.log").read_text().splitlines()
            if l.strip() and not l.startswith("#")]


# ----------------------------------------------------------------------------------------------------------
# defect 1: orch snapshot identity
# ----------------------------------------------------------------------------------------------------------
def test_git_info_and_access_inside_orch_snapshot(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    git(home, "init", "-q")
    (home / "f.txt").write_text("v1")
    git(home, "add", "f.txt")
    git(home, "commit", "-qm", "c1")
    c1 = git(home, "rev-parse", "HEAD")
    git(home, "tag", "frozen-snap")
    # the main checkout moves on and gets dirty while the job runs
    (home / "f.txt").write_text("v2")
    git(home, "commit", "-qam", "c2")
    (home / "f.txt").write_text("dirty")
    snap = home / "runs" / "job1" / "code"   # orch: git archive of c1 inside the main checkout (no .git)
    snap.mkdir(parents=True)
    (snap / "f.txt").write_text("v1")
    (home / "results").mkdir()
    (snap / "results").symlink_to(home / "results")
    monkeypatch.setenv("ORCH_CODE", str(snap))
    monkeypatch.setenv("ORCH_COMMIT", c1)
    monkeypatch.setenv("ORCH_HOME", str(home))
    monkeypatch.setenv("ORCH_JOB_ID", "job1")
    gi = H.git_info(snap)
    assert gi["commit"] == c1 and gi["clean"] is True and "frozen-snap" in gi["tags"]
    # plain git from the snapshot sees the main checkout (the defect): moved HEAD, dirty tree
    assert git(snap, "rev-parse", "HEAD") != c1
    assert eio.code_identity(snap) == (home, c1)
    # access: started at the snapshot commit (tagged) -> granted although main HEAD moved
    log = home / "results" / "test_access.log"
    log.write_text(f"t\tstarted\t{c1}\tN1\tfrozen-snap\t\n")
    monkeypatch.setenv(eio.ACCESS_ENV, "N1")
    assert eio.has_test_access(snap)
    # a nonce started at main HEAD (not the code that runs) is refused
    log.write_text(f"t\tstarted\t{git(home, 'rev-parse', 'HEAD')}\tN2\tfrozen-snap\t\n")
    monkeypatch.setenv(eio.ACCESS_ENV, "N2")
    assert not eio.has_test_access(snap)
    # outside a snapshot (ORCH_CODE elsewhere) the repo HEAD is used as before
    monkeypatch.setenv("ORCH_CODE", str(tmp_path / "elsewhere"))
    assert eio.code_identity(home)[1] == git(home, "rev-parse", "HEAD")
    # eval_test preconditions read the committed log and the tag at ORCH_COMMIT
    monkeypatch.setenv("ORCH_CODE", str(snap))
    assert eval_test._frozen_tags(*eval_test._identity(snap)[:2]) == ["frozen-snap"]


# ----------------------------------------------------------------------------------------------------------
# defect 4: cache keyed on content
# ----------------------------------------------------------------------------------------------------------
def test_access_cache_closes_with_unchanged_mtime(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "f").write_text("x")
    git(repo, "add", "f")
    git(repo, "commit", "-qm", "init")
    git(repo, "tag", "frozen-x")
    head = git(repo, "rev-parse", "HEAD")
    (repo / "results").mkdir()
    log = repo / "results" / "test_access.log"
    log.write_text(f"t\tstarted\t{head}\tNC\tfrozen-x\t\n")
    st = os.stat(log)
    monkeypatch.setenv(eio.ACCESS_ENV, "NC")
    for k in ("ORCH_CODE", "ORCH_COMMIT", "ORCH_HOME"):
        monkeypatch.delenv(k, raising=False)
    assert eio.has_test_access(repo)                      # grant cached
    with open(log, "a") as fh:
        fh.write(f"t\tcompleted\t{head}\tNC\tfrozen-x\t\n")
    os.utime(log, ns=(st.st_atime_ns, st.st_mtime_ns))    # same mtime (1-s resolution on the cluster scratch filesystem)
    assert os.stat(log).st_mtime_ns == st.st_mtime_ns
    assert not eio.has_test_access(repo)                  # the old mtime key kept granting here
    # interrupted / resumed keep the nonce open
    log.write_text(f"t\tstarted\t{head}\tNR\tfrozen-x\t\nt\tinterrupted\t{head}\tNR\tfrozen-x\t\n"
                   f"t\tresumed\t{head}\tNR\tfrozen-x\t\n")
    monkeypatch.setenv(eio.ACCESS_ENV, "NR")
    assert eio.has_test_access(repo)


# ----------------------------------------------------------------------------------------------------------
# harness: checkpoint resume, batch retry, LPT
# ----------------------------------------------------------------------------------------------------------
@pytest.fixture
def toyset(tmp_path, monkeypatch):
    monkeypatch.setattr(eio, "_splits_file", lambda: None)
    monkeypatch.delenv(eio.ACCESS_ENV, raising=False)
    rng = np.random.default_rng(3)
    locs, truths = [], []
    for k in range(24):
        cls = "SDW"[k % 3]
        loc, t = make_locus(f"loc{k:03d}", cls, rng, p=20 + k, block=f"b{k % 4}")
        locs.append(loc)
        truths.append(t)
    path = tmp_path / "toy.h5"
    write_locus_set(path, locs)
    write_truth(pd.DataFrame(truths), tmp_path / "toy.truth.parquet")
    return path


def test_checkpoint_resume_runs_only_missing_loci(toyset, tmp_path):
    import verify_run
    ids = sorted(eio.read_locus_attrs(toyset)["locus_id"].astype(str))
    rec = tmp_path / "fitted.txt"
    spec = H.MethodSpec("eval_dummy_methods:RecordingColoc", {"log_file": str(rec)}, "recording")
    res = tmp_path / "res"
    ck = tmp_path / "ck"
    # a first segment that "crashed" after 10 loci: its checkpoint rows exist, its run dir does not
    H.run_method(spec, toyset, results_dir=res, run_id="R_part", ids=ids[:10], n_workers=2, timeout_s=60,
                 checkpoint_dir=ck, hash_inputs=False, log=lambda s: None)
    shutil.copy(ck / "R_part.rows.jsonl", ck / "R.rows.jsonl")
    with open(ck / "R.rows.jsonl", "a") as fh:
        fh.write('{"locus_id": "loc0')   # torn last line of a crash mid-write
    rec.write_text("")
    r = H.run_method(spec, toyset, results_dir=res, run_id="R", ids=ids, n_workers=2, timeout_s=60,
                     checkpoint_dir=ck, hash_inputs=False, log=lambda s: None, order="lpt")
    fitted = rec.read_text().split()
    assert sorted(fitted) == ids[10:]                    # only the missing loci were run
    man = json.loads((r.run_dir / "manifest.json").read_text())
    assert man["checkpoint"]["rows_from_checkpoint"] == 10 and man["checkpoint"]["resumed"]
    assert r.counts["ok"] == 24
    pred = pd.read_parquet(r.run_dir / "predictions.parquet")
    ref = pd.read_parquet(res / "R_part" / "predictions.parquet")
    a = pred.set_index("locus_id").loc[ids[:10], ["s", "P_S", "P_D"]].to_numpy()
    assert np.array_equal(a, ref.set_index("locus_id").loc[ids[:10], ["s", "P_S", "P_D"]].to_numpy())
    assert verify_run.verify(r.run_dir) == []


def test_lpt_order_runs_largest_p_first(toyset, tmp_path):
    rec = tmp_path / "order.txt"
    spec = H.MethodSpec("eval_dummy_methods:RecordingColoc", {"log_file": str(rec)}, "recording")
    attrs = eio.read_locus_attrs(toyset)
    p = dict(zip(attrs["locus_id"], attrs["p"]))
    H.execute(spec, toyset, list(p), "recording", n_workers=1, timeout_s=60, sort_keys=p, order="lpt",
              log=lambda s: None)
    got = rec.read_text().split()
    assert got == sorted(p, key=lambda i: -p[i])


def test_batch_failure_is_retried_as_single_loci(toyset):
    attrs = eio.read_locus_attrs(toyset)
    ids = attrs["locus_id"].astype(str).tolist()
    p = dict(zip(ids, attrs["p"]))
    spec = H.MethodSpec("eval_dummy_methods:BatchOOM", {"fail_ids": ["loc005"]}, "batch_oom")
    rows = H.execute(spec, toyset, ids, "batch_oom", mode="batch", n_workers=1, batch_size=8, timeout_s=60,
                     sort_keys=p, log=lambda s: None)
    st = {r["locus_id"]: r["status"] for r in rows}
    assert st.pop("loc005") == "failed" and set(st.values()) == {"ok"}
    assert all("harness_batch_retry" in json.loads(r["extra"]) for r in rows)
    rows = H.execute(spec, toyset, ids, "batch_oom", mode="batch", n_workers=1, batch_size=8, timeout_s=60,
                     sort_keys=p, log=lambda s: None, retry_batches=False)
    assert {r["status"] for r in rows} == {"failed"}   # without the retry every locus of every batch fails


# ----------------------------------------------------------------------------------------------------------
# slow-method subsample: configured rule, nesting
# ----------------------------------------------------------------------------------------------------------
def test_slow_subsample_matches_config_rule_and_nests():
    from eval.testrun import MethodEntry, Pipeline, resolve_subsamples
    from common.seeds import rng
    ts = yaml.safe_load((ROOT / "datasets" / "configs" / "testset_v1.yaml").read_text())
    ss = ts["slow_method_subsample"]
    assert ss["per_method"]["mvsusie"] == {"S": 20, "D": 20, "W": 20}          # P3-17
    assert ss["per_cell"] == {"S": 40, "D": 40, "W": 40}                       # P3-12 (ColocBoost)
    ms = [MethodEntry("mvsusie", "x:y", {}, timeout_s=1, subsample_per_class=-1),
          MethodEntry("colocboost", "x:y", {}, timeout_s=1, subsample_per_class=-1),
          MethodEntry("coloc_abf", "x:y", {}, timeout_s=1)]
    assert resolve_subsamples(ms, ts) == []
    assert [m.subsample_per_class for m in ms] == [20, 40, None]
    bad = [MethodEntry("colocboost_m1", "x:y", {}, timeout_s=1)]
    assert "without its subsample" in resolve_subsamples(bad, ts)[0]
    # toy truth: 2 main cells + 1 shifted cell, 50 loci per class
    cells = ts["main_grid"]["cells"][:2] + [ts["shifted_arm"]["cells"][0]["cell_id"]]
    rows = [{"locus_id": f"{c}#{cls}{i:03d}", "cell_id": c, "oracle_class": cls}
            for c in cells for cls in "SDWG" for i in range(50 if cls != "G" else 5)]
    tr = pd.DataFrame(rows)
    pipe = Pipeline.__new__(Pipeline)
    pipe.methods, pipe.cfg = ms, type("C", (), {"mode": "test", "scale": 1.0, "seed": 20261028})()
    pipe.testset = dict(ts, main_grid=dict(ts["main_grid"], cells=cells[:2]),
                        shifted_arm=dict(ts["shifted_arm"], cells=[ts["shifted_arm"]["cells"][0]]))
    pipe.sim_truth = lambda: tr
    pipe._stage_done = lambda name: None
    pipe._mark = lambda name, payload: None
    sub = Pipeline.slow_subsamples(pipe)
    # literal implementation of datasets/configs/testset_v1.yaml slow_method_subsample.draw
    g = rng(20261028, "slow_subsample")
    want40, want20 = [], []
    for c in cells:
        for cls in "SDW":
            ids = sorted(tr.loc[(tr.cell_id == c) & (tr.oracle_class == cls), "locus_id"])
            idx = g.choice(len(ids), size=min(40, len(ids)), replace=False)
            want40 += [ids[j] for j in idx]
            want20 += [ids[j] for j in idx[:20]]
    assert sub[40] == sorted(want40) and sub[20] == sorted(want20)
    assert set(sub[20]) < set(sub[40]) and len(sub[40]) == 3 * 3 * 40 and len(sub[20]) == 3 * 3 * 20
    assert not any("G" in i.split("#")[1][:1] for i in sub[40])                 # G/U not included


# ----------------------------------------------------------------------------------------------------------
# defect 6: run_benchmark per-method id lists
# ----------------------------------------------------------------------------------------------------------
def test_run_benchmark_method_ids(tmp_path, monkeypatch):
    from test_eval_benchmark import build_set
    monkeypatch.setattr(eio, "_splits_file", lambda: None)
    monkeypatch.delenv(eio.ACCESS_ENV, raising=False)
    ls = tmp_path / "toybench.h5"
    truth = build_set(ls, n_per_class=8)
    sub = sorted(truth.groupby(["cell_id", "oracle_class"]).head(3)["locus_id"])
    idf = tmp_path / "ids.txt"
    idf.write_text("\n".join(sub) + "\n")
    mfile = tmp_path / "m.yaml"
    mfile.write_text(yaml.safe_dump({"methods": [
        {"name": "toy_coloc", "target": "eval_dummy_methods:ToyColoc", "kwargs": {}, "n_workers": 2, "timeout_s": 60},
        {"name": "toy_slow", "target": "eval_dummy_methods:ToyColoc", "kwargs": {"noise": 0.2, "name": "toy_slow"},
         "n_workers": 2, "timeout_s": 60}]}))
    res_dir = tmp_path / "res"
    rc = run_benchmark.main(["--locus-set", str(ls), "--methods", str(mfile), "--results-dir", str(res_dir),
                             "--label", "mids", "--n-boot", "40", "--n-jobs", "1", "--candidate", "toy_coloc",
                             "--strength-folds", "3", "--no-hash-inputs", "--method-ids", f"toy_slow={idf}"])
    assert rc == 0
    bdir = next((res_dir / "benchmarks").iterdir())
    main = json.loads((bdir / "benchmark.json").read_text())
    assert "toy_slow" not in main["methods"] and main["n_loci"] == len(truth)
    sdir = next(d for d in bdir.iterdir() if d.name.startswith("subsample_"))
    s = json.loads((sdir / "benchmark.json").read_text())
    assert s["n_loci"] == len(sub) and {"toy_slow", "toy_coloc", "strength_only"} <= set(s["methods"])
    assert s["methods"]["toy_slow"]["least_favourable"]["failures"]["n_failed"] == 0   # not scored off-subsample
    man = json.loads((bdir / "manifest.json").read_text())
    slow_run = Path(man["runs"]["toy_slow"][str(ls.resolve())])
    assert json.loads((slow_run / "manifest.json").read_text())["counts"]["expected"] == len(sub)
    assert run_benchmark.verify(bdir) == []
    # tamper: a changed sub-benchmark is detected
    s["n_loci"] = 1
    (sdir / "benchmark.json").write_text(json.dumps(s))
    assert any("subsample_" in p for p in run_benchmark.verify(bdir))


# ----------------------------------------------------------------------------------------------------------
# defect 2: the whole pipeline inside the window; interrupt + resume under the same nonce
# ----------------------------------------------------------------------------------------------------------
FAKE_GEN = '''
from pathlib import Path
import numpy as np, pandas as pd
from eval.io import write_locus_set
from eval.truth import write_truth
from eval_toy import make_locus
from sim.cells import _safe, link_set

CALLS = Path(__file__).with_name("gen_calls.txt")

def _cell(out, base, cid, n, seed, split):
    rng = np.random.default_rng(seed)
    lvl, pr, ov = cid.split("|")
    locs, truths = [], []
    for k in range(n):
        cls = "SDW"[k % 3]
        lid = f"{base}_{_safe(cid)}_{k:03d}"
        loc, t = make_locus(lid, cls, rng, p=30, cell=cid, level=lvl, pairing=pr, block=f"tb{k % 5}", split=split)
        t["overlap"] = float(ov.split("=")[1])
        if pr == "P3":
            t["rho"] = float(t["overlap"] + rng.uniform(-0.1, 0.1))
        locs.append(loc); truths.append(t)
    d = Path(out) / base
    d.mkdir(parents=True, exist_ok=True)
    write_locus_set(d / f"{_safe(cid)}.h5", locs)
    write_truth(pd.DataFrame(truths), d / f"{_safe(cid)}.truth.parquet")

def generate(config, out_dir, seed=None, *, main=True, shifted=True, cells_filter=None, stats_suffix="",
             split="test", dry_run=False, set_name=None, n_per_class=None, n_gu=None, n_shifted=None, **kw):
    with open(CALLS, "a") as fh:
        fh.write(f"main={main} shifted={shifted} cells={cells_filter}\\n")
    base = set_name or "testset_v1"
    if main:
        for i, cid in enumerate(cells_filter or config["main_grid"]["cells"]):
            _cell(out_dir, base, cid, 18, i, split)
        m, _ = link_set(base, out_dir)
        return [m]
    if Path(out_dir).joinpath("FAIL_SHIFTED").exists():
        Path(out_dir).joinpath("FAIL_SHIFTED").unlink()
        raise RuntimeError("simulated crash during the shifted arm")
    for c in config["shifted_arm"]["cells"]:
        _cell(out_dir, base + "_shifted", c["cell_id"], c["n"], 99, split)
    m, _ = link_set(base + "_shifted", out_dir)
    return [m]
'''

FAKE_REAL = '''
from pathlib import Path
import numpy as np, pandas as pd
from eval.io import write_locus_set
from eval_toy import make_locus

def build(spec, out_dir, workers=4):
    out = {}
    rng = np.random.default_rng(5)
    for key in ("real_R2", "real_R3b", "real_R4b_insample", "real_R4b_reference"):
        cls = {"real_R2": "S", "real_R3b": "D"}.get(key, "S")
        r = np.random.default_rng(7) if key.startswith("real_R4b") else rng
        locs = []
        for k in range(10):
            loc, _ = make_locus(f"{key[:8]}_{k:02d}", cls if k % 2 == 0 else "W", r, p=30, cell="real",
                                level="real", pairing="real_gwas_x_eqtl", block=f"rb{k % 4}", split="test")
            loc.rho_hat = 0.1
            loc.extra_attrs = {"rho_hat_pair": 0.3}
            locs.append(loc)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        p = Path(out_dir) / f"{key}.h5"
        write_locus_set(p, locs)
        out[key] = {"path": str(p), "ids": [l.locus_id for l in locs]}
    return out
'''


@pytest.fixture
def prepo(tmp_path, monkeypatch):
    """Throw-away repo with a 2-cell + 1 shifted-cell test-set config, a frozen methods file with a slow method,
    a primary candidate, real arms and the strength-only control."""
    r = tmp_path / "repo"
    (r / "configs").mkdir(parents=True)
    (r / "src" / "eval").mkdir(parents=True)
    shutil.copy(ROOT / "src" / "eval" / "metrics.py", r / "src" / "eval" / "metrics.py")
    ts = yaml.safe_load((ROOT / "datasets" / "configs" / "testset_v1.yaml").read_text())
    ts["generator"]["entry"] = "fake_gen:generate"
    po = ts["main_grid"]["pairing_overlap"]
    ts["main_grid"].update(mismatch_levels=["L0", "L1"], pairing_overlap=[po[0], po[5]],
                           n_cells=4, cells=["L0|P1|r_p=0.0", "L0|P3|rho=0.5", "L1|P1|r_p=0.0", "L1|P3|rho=0.5"])
    ts["shifted_arm"].update(n_loci=12, cells=[{"cell_id": "L1swap|P1|r_p=0.0", "n": 12}])
    ts["slow_method_subsample"].update(methods=["toy_slow", "toy_slow2"], per_cell={"S": 4, "D": 4, "W": 4},
                                       per_method={"toy_slow2": {"S": 2, "D": 2, "W": 2}})
    (r / "configs" / "testset_v1.yaml").write_text(yaml.safe_dump(ts))
    (r / "configs" / "FROZEN_SHA256").write_text("# test\nPENDING  src/eval/metrics.py\nPENDING  configs/testset_v1.yaml\n")
    check_frozen.update(r)
    # frozen strength table fitted on separate toy (validation) loci
    rng = np.random.default_rng(11)
    vl, vt = [], []
    for k in range(60):
        loc, t = make_locus(f"v{k}", "SDW"[k % 3], rng, p=30)
        vl.append(loc)
        vt.append(t)
    write_locus_set(tmp_path / "val.h5", vl)
    monkeypatch.setattr(eio, "_splits_file", lambda: None)
    feats = SC.read_features(tmp_path / "val.h5")
    SC.dump(SC.StrengthTable.fit(feats, np.array([t["oracle_class"] for t in vt])).to_dict(),
            r / "configs" / "strength_toy.json")
    mc = {"t_S": 0.5, "t_D": 0.5, "recal_a_S": 0.0, "recal_a_D": 0.0}
    methods = {
        "version": 1, "status": "frozen",
        "candidates": {"primary": "toy_coloc", "secondary": []},
        "c3_eligible_baselines": ["toy_noisy", "toy_slow"],
        "methods": [
            {"name": "toy_coloc", "target": "eval_dummy_methods:ToyColoc", "kwargs": {}, "n_workers": 2,
             "timeout_s": 60, "metric_config": mc},
            {"name": "toy_noisy", "target": "eval_dummy_methods:ToyColoc", "kwargs": {"noise": 0.35, "name": "toy_noisy"},
             "n_workers": 2, "timeout_s": 60, "metric_config": mc},
            {"name": "toy_slow", "target": "eval_dummy_methods:ToyColoc", "kwargs": {"noise": 0.2, "name": "toy_slow"},
             "n_workers": 1, "timeout_s": 60, "metric_config": mc, "subsample": "slow_method_subsample"},
            {"name": "toy_slow2", "target": "eval_dummy_methods:ToyColoc", "kwargs": {"noise": 0.1, "name": "toy_slow2"},
             "mode": "batch", "gpu": True, "batch_size": 4, "timeout_s": 60, "metric_config": mc,
             "subsample": "slow_method_subsample"},
            {"name": "lead_snp", "target": "eval.reference_methods:LeadSNPRule", "kwargs": {}, "n_workers": 1,
             "timeout_s": 60, "metric_config": {"t_S": None, "t_D": None}},
            {"name": "strength_only", "target": "eval.reference_methods:StrengthOnly",
             "kwargs": {"table": "configs/strength_toy.json"}, "n_workers": 1, "timeout_s": 60, "metric_config": mc}],
        "real_data_arms": {"builder": "fake_real:build", "R2": {"n": 10}, "R3b": {"n": 10}, "R4b": {"n": 10},
                           "methods": ["primary_candidate", "toy_noisy", "strength_only"]},
    }
    (r / "configs" / "test_methods_v1.yaml").write_text(yaml.safe_dump(methods, sort_keys=False))
    (r / ".gitignore").write_text("data/\nresults/eval_test/\nresults/LEDGER.tsv\n")
    git(r, "init", "-q")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "init")
    gen = tmp_path / "genmod"
    gen.mkdir()
    (gen / "fake_gen.py").write_text(FAKE_GEN)
    (gen / "fake_real.py").write_text(FAKE_REAL)
    monkeypatch.syspath_prepend(str(gen))
    monkeypatch.setenv("PYTHONPATH", f"{gen}:{os.environ.get('PYTHONPATH', '')}")   # spawned harness workers
    monkeypatch.setattr(eio, "PROJECT_ROOT", r)
    monkeypatch.delenv(eio.ACCESS_ENV, raising=False)
    for k in ("ORCH_CODE", "ORCH_COMMIT", "ORCH_HOME", "ORCH_JOB_ID"):
        monkeypatch.delenv(k, raising=False)
    return r


OPTS = {"cores": 4, "ram_gb": 16, "gen_workers": 1, "n_boot": 40, "n_jobs": 1}


def test_thresholds_required_and_plumbing_checked(prepo):
    mf = prepo / "configs" / "test_methods_v1.yaml"
    git(prepo, "tag", "frozen-p1")
    assert eval_test.check_preconditions(prepo, mf) == []
    d = yaml.safe_load(mf.read_text())
    d["methods"][0]["metric_config"] = {"t_S": None, "t_D": 0.5}
    d["methods"][1]["target"] = "eval_dummy_methods:DoesNotExist"
    d["methods"][2].pop("subsample")
    d["real_data_arms"]["builder"] = "fake_real:nope"
    bad = prepo / "bad.yaml"
    bad.write_text(yaml.safe_dump(d))
    ref = " | ".join(eval_test.check_preconditions(prepo, bad))
    for want in ("toy_coloc: metric_config lacks", "toy_noisy: target", "toy_slow is a slow method",
                 "real_data_arms.builder fake_real:nope not importable"):
        assert want in ref, want


def test_pipeline_in_window_interrupt_resume(prepo):
    mf = prepo / "configs" / "test_methods_v1.yaml"
    git(prepo, "tag", "frozen-p1")
    out = prepo / "data" / "testsets"
    label_dir = None
    # first attempt: crash during the shifted arm (after the main grid and the real arms)
    assert eval_test.check_preconditions(prepo, mf) == []
    (out).mkdir(parents=True)
    import glob
    with pytest.raises(BaseException):
        # FAIL_SHIFTED is looked up in the sets directory of the bundle's data dir: create it lazily
        orig = eval_test._pipeline

        def wrapped(*a, **k):
            pipe, ts = orig(*a, **k)
            (pipe.cfg.data_dir / "sets").mkdir(parents=True, exist_ok=True)
            (pipe.cfg.data_dir / "sets" / "FAIL_SHIFTED").write_text("x")
            return pipe, ts
        eval_test._pipeline = wrapped
        try:
            eval_test.run_test(prepo, mf, out, opts=OPTS)
        finally:
            eval_test._pipeline = orig
    ev = _events(prepo)
    assert [e[1] for e in ev] == ["started", "interrupted"]
    nonce = ev[0][3]
    assert not eio.has_test_access(prepo)                     # env popped
    # a fresh start is refused (uncommitted entries); resume of that nonce is allowed
    assert any("uncommitted" in x for x in eval_test.check_preconditions(prepo, mf))
    assert eval_test.check_preconditions(prepo, mf, resume=nonce) == []
    calls = (Path(glob.glob(str(prepo.parent / "genmod" / "gen_calls.txt"))[0])).read_text().splitlines()
    assert calls[0].startswith("main=True") and calls[1].startswith("main=False")
    res = eval_test.run_test(prepo, mf, out, resume=nonce, opts=OPTS)
    ev = _events(prepo)
    assert [e[1] for e in ev] == ["started", "interrupted", "resumed", "completed"] and {e[3] for e in ev} == {nonce}
    calls = (prepo.parent / "genmod" / "gen_calls.txt").read_text().splitlines()
    assert len(calls) == 3 and calls[2].startswith("main=False")   # main grid NOT regenerated on resume
    assert res["verify"]["ok"], res["verify"]["problems"]
    bundle = Path(res["evaluate"]["benchmark"]["dir"]).parents[1]
    b = json.loads((bundle / "bundle.json").read_text())
    assert b["mode"] == "test" and b["extra"]["nonce"] == nonce
    # every unit ran, slow methods on their subsamples (main + shifted), real arms for the real methods only
    units = b["units"]
    assert all(v["status"] == "done" for v in units.values())
    n_sub = {k: v["n"] for k, v in units.items()}
    lab = f"test_frozen-p1_{nonce[:8]}"
    assert n_sub[f"{lab}__toy_slow__sim_main"] == 4 * 3 * 4 and n_sub[f"{lab}__toy_slow2__sim_main"] == 4 * 3 * 2
    assert n_sub[f"{lab}__toy_slow__sim_shifted"] == 3 * 4 and n_sub[f"{lab}__toy_coloc__sim_main"] == 72
    assert f"{lab}__toy_coloc__real_R2" in units and f"{lab}__lead_snp__real_R2" not in units
    assert f"{lab}__strength_only__real_R3b" in units
    # evaluation: main benchmark without the slow methods; one sub-benchmark per subsample with C3 dry run
    parts = res["evaluate"]["benchmark"]["parts"]
    assert set(parts) == {"main", *[k for k in parts if k.startswith("subsample_")]} and len(parts) == 3
    main = json.loads((Path(parts["main"]["dir"]) / "benchmark.json").read_text())
    assert "toy_slow" not in main["methods"] and "c3_dry_run" in main
    assert main["methods"]["toy_coloc"]["least_favourable"]["operating_points"]["source"] == "given"
    assert "shifted" in main["settings"]["least_favourable"]
    for k, v in parts.items():
        if k != "main":
            bj = json.loads((Path(v["dir"]) / "benchmark.json").read_text())
            assert "toy_slow" in bj["methods"] and "toy_coloc" in bj["methods"]
            assert bj["methods"]["toy_slow"]["least_favourable"]["failures"]["n_failed"] == 0
    c4 = json.loads((Path(parts["main"]["dir"]) / "c4.json").read_text())
    sl = c4["modes"]["least_favourable"]["slopes"]["toy_coloc"]["P3"]["D"]
    assert c4["modes"]["least_favourable"]["n_loci"] == 18 and sl["ci_low"] <= sl["slope_per_0.1"] <= sl["ci_high"]
    real = json.loads((bundle / "real_arms.json").read_text())
    assert set(real["methods"]) == {"toy_coloc", "toy_noisy", "strength_only", "toy_coloc_rhopooled",
                                    "toy_coloc_rhozero"}
    assert real["methods"]["toy_coloc_rhozero"]["P3_23_sensitivity_of"] == "toy_coloc"
    assert real["candidate_vs_best_baseline"]["recall_S_R2_at_5pct_R3b"]["best_baseline"] in ("toy_noisy", "strength_only")
    for mode, want in (("pooled", 0.3), ("zero", 0.0)):
        pr = pd.read_parquet(bundle / "runs" / f"{lab}__toy_coloc_rho{mode}__real_R2" / "predictions.parquet")
        assert {json.loads(e)["rho_used"] for e in pr["extra"]} == {want}
    # after `completed` the test truth and loci are unreadable (C1/C3 and verify ran inside the window)
    tr = Path(json.loads((bundle / "state" / "sim.json").read_text())["main"])
    with pytest.raises(eio.TestSplitAccessError):
        eio.read_locus_attrs(tr)
    # one test run only: after committing the log, every new start is refused
    git(prepo, "add", "-f", "results/test_access.log")
    git(prepo, "commit", "-qm", "log")
    git(prepo, "tag", "frozen-p2")
    assert any("one test run only" in x for x in eval_test.check_preconditions(prepo, mf))


# ----------------------------------------------------------------------------------------------------------
# famA request: opt-in trait-2 LD (R2 / X_ref2) through the shared reader and the harness
# ----------------------------------------------------------------------------------------------------------
def test_trait2_ld_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr(eio, "_splits_file", lambda: None)
    monkeypatch.delenv(eio.ACCESS_ENV, raising=False)
    rng = np.random.default_rng(1)
    a, _ = make_locus("withR2", "S", rng, p=12)
    b, _ = make_locus("noR2", "S", rng, p=12)
    a.R2 = (0.5 * a.R).astype(np.float32)
    a.extra_attrs = {"ld_type2": "dense", "n_ref2": 300}
    path = tmp_path / "ld2.h5"
    write_locus_set(path, [a, b])
    with eio.LocusSetReader(path) as r:
        assert r.read("withR2").R2 is None                       # default: not read
        got = r.read("withR2", load_ld2=True)
        assert np.array_equal(got.R2, a.R2) and got.X_ref2 is None and got.extra_attrs["ld_type2"] == "dense"
        assert r.read("noR2", load_ld2=True).R2 is None           # locus without trait-2 LD
    for flag, want in ((True, {"withR2": True, "noR2": False}), (False, {"withR2": False, "noR2": False})):
        rows = H.execute(H.MethodSpec("eval_dummy_methods:LD2Check", {"needs_ld2": flag}, "ld2_check"), path,
                         ["withR2", "noR2"], "ld2_check", n_workers=1, timeout_s=60, log=lambda s: None)
        assert {r["locus_id"]: json.loads(r["extra"])["has_R2"] for r in rows} == want
    rows = H.execute(H.MethodSpec("eval_dummy_methods:LD2Check", {"needs_ld2": True}, "ld2_check"), path,
                     ["withR2"], "ld2_check", n_workers=1, timeout_s=60, log=lambda s: None)
    assert abs(json.loads(rows[0]["extra"])["R2_sum"] - float(a.R2.astype(float).sum())) < 1e-3
    from famA.method import JointSusieMethod, ladder_config
    m = JointSusieMethod()
    assert hasattr(m, "needs_ld2")                                # famA declares it (True when per_trait_ld)


# ----------------------------------------------------------------------------------------------------------
# PI decisions after the handover: (d) slowest cells first, (e) P3-23 rho sensitivities
# ----------------------------------------------------------------------------------------------------------
def test_cells_ordered_slowest_first(tmp_path):
    from eval.testrun import order_cells_by_cost
    cells = ["L0|P1|r_p=0.0", "L3|P2|rho=0.0", "L4|P1|r_p=0.0", "L3|P1|r_p=0.0", "L1|P3|rho=0.5"]
    by_level = order_cells_by_cost(cells, tmp_path)               # no timing file: level rank
    assert by_level[:2] == ["L3|P1|r_p=0.0", "L3|P2|rho=0.0"] and by_level[-1] == "L0|P1|r_p=0.0"
    st = tmp_path / "data/processed/sets"
    st.mkdir(parents=True)
    t = {c: {"t_candidates_s": v, "t_finalize_write_s": 1.0} for c, v in zip(cells, [5, 50, 400, 900, 10])}
    (st / "valid_grid_v1.stats.json").write_text(json.dumps({"cells": t}))
    assert order_cells_by_cost(cells, tmp_path) == ["L3|P1|r_p=0.0", "L4|P1|r_p=0.0", "L3|P2|rho=0.0",
                                                    "L1|P3|rho=0.5", "L0|P1|r_p=0.0"]


def test_sim_generate_keeps_filter_order(monkeypatch, tmp_path):
    import sim.testset as T
    seen = []
    monkeypatch.setattr(T, "_run_groups", lambda name, groups, *a, **k: seen.extend(g[0][0].cell_id for g in groups) or [])
    monkeypatch.setattr(T, "link_set", lambda base, out: (tmp_path / "m.h5", tmp_path / "t.parquet"))
    monkeypatch.setattr(T, "_merge_stats", lambda *a, **k: {})
    ts = yaml.safe_load((ROOT / "datasets" / "configs" / "testset_v1.yaml").read_text())
    order = ["L3|P1|r_p=0.0", "L0|P1|r_p=0.0", "L4|P3|rho=0.8"]
    T.generate(ts, tmp_path, 20261029, split="valid", dry_run=True, shifted=False, cells_filter=order,
               log=lambda s: None)
    assert seen == order


def test_rho_override_wrapper(tmp_path, monkeypatch):
    from eval.reference_methods import RhoOverride
    rng = np.random.default_rng(2)
    locs = []
    for k, pair in enumerate([0.25, float("nan"), None]):
        loc, _ = make_locus(f"r{k}", "S", rng, p=20)
        loc.rho_hat = 0.7
        loc.extra_attrs = {} if pair is None else {"rho_hat_pair": pair}
        locs.append(loc)
    for mode, want in (("pooled", [0.25, 0.0, 0.0]), ("zero", [0.0, 0.0, 0.0])):
        m = RhoOverride("eval_dummy_methods:RhoEcho", {}, mode=mode)
        rows = m.fit_batch([__import__("copy").deepcopy(l) for l in locs])
        assert [json.loads(r["extra"])["rho_seen"] for r in rows] == want
        assert [json.loads(r["extra"])["rho_used"] for r in rows] == want
        assert m.name == f"rho_echo_rho{mode}" and m.needs_ld is False


def test_rho_override_zero_uses_donor_disjoint_attribute():
    """P3-23(b) via data.real.extract.build_test_arm's rho_sens_zero: 0.0 (donor-disjoint) -> 0; NaN -> primary."""
    import copy
    from eval.reference_methods import RhoOverride
    rng = np.random.default_rng(4)
    locs = []
    for k, z in enumerate([0.0, float("nan")]):
        loc, _ = make_locus(f"z{k}", "S", rng, p=20)
        loc.rho_hat = 0.4
        loc.extra_attrs = {"rho_sens_zero": z, "rho_hat_pair": 0.2}
        locs.append(loc)
    rows = RhoOverride("eval_dummy_methods:RhoEcho", {}, mode="zero").fit_batch(copy.deepcopy(locs))
    assert [json.loads(r["extra"])["rho_seen"] for r in rows] == [0.0, 0.4]


# ----------------------------------------------------------------------------------------------------------
# D16: no primary candidate (MVR route) + pre-declared descriptive entries
# ----------------------------------------------------------------------------------------------------------
def test_mvr_route_no_candidate_and_descriptive(prepo):
    mf = prepo / "configs" / "test_methods_v1.yaml"
    d = yaml.safe_load(mf.read_text())
    d["candidates"] = {"primary": None, "secondary": []}
    d["methods"][1]["descriptive"] = True                      # toy_noisy = descriptive entry
    d["c3_eligible_baselines"] = ["toy_slow"]
    d["real_data_arms"] = None
    mf.write_text(yaml.safe_dump(d, sort_keys=False))
    git(prepo, "commit", "-qam", "mvr")
    git(prepo, "tag", "frozen-mvr0")
    assert any("no decision note" in r for r in eval_test.check_preconditions(prepo, mf))   # unfilled template
    d["candidates"]["decision"] = "D16"
    bad = dict(d, c3_eligible_baselines=["toy_noisy"])
    (prepo / "bad.yaml").write_text(yaml.safe_dump(bad, sort_keys=False))
    assert any("descriptive entry" in r for r in eval_test.check_preconditions(prepo, prepo / "bad.yaml"))
    (prepo / "bad.yaml").unlink()
    mf.write_text(yaml.safe_dump(d, sort_keys=False))
    git(prepo, "commit", "-qam", "mvr decision")
    git(prepo, "tag", "frozen-mvr1")
    assert eval_test.check_preconditions(prepo, mf) == []
    res = eval_test.run_test(prepo, mf, prepo / "data" / "t", opts=OPTS)
    assert res["verify"]["ok"], res["verify"]["problems"]
    bundle = Path(res["evaluate"]["benchmark"]["dir"]).parents[1]
    claims = json.loads((bundle / "claims.json").read_text())
    assert claims["primary"] is None and claims["C1"] == claims["C3"] == claims["C4"] == "no candidate (D16)"
    assert claims["descriptive_entries"] == ["toy_noisy"]
    main = json.loads((Path(res["evaluate"]["benchmark"]["parts"]["main"]["dir"]) / "benchmark.json").read_text())
    assert "c3_dry_run" not in main and not main["bootstrap"]["least_favourable"]["vs_candidate"]
    assert "toy_noisy" in main["methods"]                     # descriptive entries are benchmarked (C5)
    desc = json.loads((bundle / "descriptive.json").read_text())
    nz = desc["methods"]["toy_noisy"]
    assert nz["descriptive"] and nz["fixed_0.8"]["t_S"] == 0.8 and nz["matched_w"]["t_S"] == 0.5
    assert sum(sum(v.values()) for v in nz["fixed_0.8"]["confusion"].values()) == 84
    assert [e[1] for e in _events(prepo)] == ["started", "completed"]


def test_c4_with_a_single_bootstrap_unit_is_not_estimable(monkeypatch):
    """p6dr_mvr_v1 (scale 0.01): C4's realistic P2/P3 subset fell into one LD block -> BCa jackknife impossible;
    the C4 block must report 'not estimable' instead of crashing the evaluation."""
    from eval import metrics as M
    from eval.testrun import Pipeline
    rng = np.random.default_rng(0)
    rows, preds = [], []
    for k in range(12):
        cls = "DW"[k % 2]
        rows.append({"locus_id": f"c{k}", "oracle_class": cls, "mu1": 7.0, "mu2": 7.0, "r2": 0.1, "d": 5.0,
                     "rho": rng.uniform(0, 0.5), "block_id": "only_block", "cell_id": "L1|P3|rho=0.5",
                     "mismatch_level": "L1", "pairing_type": "P3", "overlap": 0.5, "control": ""})
        preds.append({"locus_id": f"c{k}", "method": "m", "s": 0.3, "P_S": rng.uniform(), "P_D": 0.2,
                      "status": "ok"})
    truth, pr = pd.DataFrame(rows), pd.DataFrame(preds)
    pipe = Pipeline.__new__(Pipeline)
    pipe.cfg = type("C", (), {"unit": "block", "n_boot": 20, "n_jobs": 4, "mode": "dry"})()
    pipe.repo = ROOT
    pipe.testset = yaml.safe_load((ROOT / "datasets" / "configs" / "testset_v1.yaml").read_text())
    import run_benchmark as RB
    monkeypatch.setattr(RB, "_load_truth_and_preds", lambda spec, bdir: (truth, {"m": pr}, {}))
    out = Pipeline.c4_block(pipe, Path("."), {"bench_id": "x"}, {"m": {"t_S": 0.5}})
    assert "not estimable" in out["modes"]["least_favourable"]["note"]


def _tiny(cell="L1|P3|rho=0.5", blocks=("only_block",), n=12, seed=0):
    rng = np.random.default_rng(seed)
    rows, preds = [], []
    for k in range(n):
        cls = "SDW"[k % 3]
        rows.append({"locus_id": f"c{k}", "oracle_class": cls, "mu1": 7.0, "mu2": 7.0, "r2": 0.1, "d": 5.0,
                     "rho": rng.uniform(0, 0.5), "block_id": blocks[k % len(blocks)], "cell_id": cell,
                     "mismatch_level": cell.split("|")[0], "pairing_type": cell.split("|")[1], "overlap": 0.5,
                     "control": ""})
        for m in ("a", "b"):
            preds.append({"locus_id": f"c{k}", "method": m, "s": rng.uniform(), "P_S": rng.uniform(),
                          "P_D": rng.uniform(), "status": "ok"})
    pr = pd.DataFrame(preds)
    return pd.DataFrame(rows), {m: pr[pr.method == m].reset_index(drop=True) for m in ("a", "b")}


def test_bootstrap_blocks_never_raise_on_empty_or_single_unit_subsets():
    """PI request after p6dr_mvr_v1: every bootstrap / metric block records NaN + a reason for an empty or too-small
    subset instead of raising (run_benchmark bootstrap and C3; the C4 / real / descriptive guards are tested
    separately)."""
    from eval import metrics as M
    import run_benchmark as RB
    spec = {"bench_id": "t", "unit": "block", "n_boot": 20, "n_jobs": 2}
    truth, preds = _tiny()
    fr = M.build_eval_frame(truth, preds, "least_favourable")
    out = RB._bootstrap_block(fr, ["a", "b"], ["realistic"], spec, "lf", "a")      # one LD block
    assert "fewer than 2 bootstrap units" in out["not_estimable"] and out["estimates"]["a"]["realistic"]["auprc_sd"]["ci_low"] is None
    empty = fr.mask(np.zeros(len(fr), bool))                                      # empty cell / subset
    out = RB._bootstrap_block(empty, ["a", "b"], ["realistic"], spec, "lf", "a")
    assert "n_loci=0" in out["not_estimable"] and out["estimates"] == {}
    truth2, preds2 = _tiny(blocks=("b1", "b2", "b3"))                              # >= 2 units: normal CIs
    out = RB._bootstrap_block(M.build_eval_frame(truth2, preds2, "least_favourable"), ["a", "b"], ["realistic"],
                              spec, "lf", "a")
    assert "not_estimable" not in out and out["n_clusters"] == 3


def test_c4_with_no_p2_p3_loci_is_not_estimable(monkeypatch):
    from eval.testrun import Pipeline
    import run_benchmark as RB
    truth, preds = _tiny(cell="L1|P1|r_p=0.0", blocks=("b1", "b2"))
    pipe = Pipeline.__new__(Pipeline)
    pipe.cfg = type("C", (), {"unit": "block", "n_boot": 20, "n_jobs": 2, "mode": "dry"})()
    pipe.repo = ROOT
    pipe.testset = yaml.safe_load((ROOT / "datasets" / "configs" / "testset_v1.yaml").read_text())
    monkeypatch.setattr(RB, "_load_truth_and_preds", lambda spec, bdir: (truth, preds, {}))
    out = Pipeline.c4_block(pipe, Path("."), {"bench_id": "x"}, {"a": {"t_S": 0.5}, "b": {"t_S": 0.5}})
    assert out["modes"]["least_favourable"]["n_loci"] == 0 and "not estimable" in out["modes"]["least_favourable"]["note"]
