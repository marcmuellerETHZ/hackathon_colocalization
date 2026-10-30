"""End-to-end tests of scripts/orch with toy jobs through the real scheduler daemon.

Each test module run builds a throw-away git repo under /tmp/claude-orchtest-*, points
ORCH_HOME at it (so production registries are never touched) and starts a scheduler daemon
with a fast poll; teardown stops it and removes the repo. Run:
    source env/activate.sh && python -m pytest -q tests/orch
"""
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
ORCH = REPO / "scripts" / "orch" / "orch"
ENV_SCRIPT = Path(os.environ.get("COLOC_BENCH_ROOT", Path(__file__).resolve().parents[2])) / "env" / "activate.sh"
GITC = ["git", "-c", "user.name=orch-test", "-c", "user.email=orch@test"]


def sh(args, cwd, env=None, check=True):
    r = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
    if check and r.returncode:
        raise AssertionError(f"{args} -> {r.returncode}\n{r.stdout}\n{r.stderr}")
    return r


class Home:
    def __init__(self, repo, env):
        self.repo, self.env = repo, env

    def orch(self, *args, check=True):
        return sh([str(ORCH), *args], cwd=self.repo, env=self.env, check=check)

    def submit(self, jid, cmd, gpu="no", vram=0, cores=1, extra=(), check=True):
        a = ["submit", "--id", jid, "--owner", "test", "--gpu", gpu, "--cores", str(cores),
             "--expected-min", "1", *extra]
        if gpu == "yes":
            a += ["--vram-gb", str(vram)]
        return self.orch(*a, "--", cmd, check=check)

    def jobs(self):
        d = yaml.safe_load((self.repo / "JOBS.yaml").read_text())
        return {j["id"]: j for j in d["jobs"]}

    def wait(self, *ids):
        return self.orch("wait", *ids, "--timeout-min", "3", check=False)

    def commit(self, msg):
        sh(["git", "add", "-A"], self.repo)
        sh([*GITC, "commit", "-q", "-m", msg], self.repo)


def t(s):
    import datetime as dt
    return dt.datetime.fromisoformat(s)


@pytest.fixture(scope="module")
def home():
    base = Path(tempfile.mkdtemp(prefix="claude-orchtest-", dir="/tmp"))
    repo = base / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "results").mkdir()
    (repo / "results" / "README").write_text("tracked results file\n")
    (repo / "src" / "toy.py").write_text("VALUE = 1\n")
    (repo / ".gitignore").write_text("runs/\n.orch/\nlogs/jobs/\n/data\n")
    sh(["git", "init", "-q", "-b", "master"], repo)
    h = Home(repo, None)
    h.commit("init")
    env = os.environ.copy()
    env.update(ORCH_HOME=str(repo), ORCH_ENV_SCRIPT=str(ENV_SCRIPT), ORCH_POLL_S="0.3",
               ORCH_SAMPLE_S="0.5", ORCH_WAIT_POLL_S="0.3")
    env.pop("PYTHONPATH", None)
    h.env = env
    h.orch("scheduler")
    yield h
    h.orch("scheduler", "--stop", check=False)
    for j in (h.jobs().values() if (repo / "JOBS.yaml").exists() else []):
        if j["status"] in ("running", "queued"):
            h.orch("cancel", j["id"], check=False)
    shutil.rmtree(base, ignore_errors=True)


def test_scheduler_idempotent(home):
    r = home.orch("scheduler")
    assert "already running" in r.stdout
    assert home.orch("scheduler", "--status").returncode == 0


def test_a_cpu_concurrency_and_core_budget(home):
    home.submit("a1", "sleep 3", cores=2)
    home.submit("a2", "sleep 3", cores=2)
    assert home.wait("a1", "a2").returncode == 0
    j = home.jobs()
    assert t(j["a2"]["start"]) < t(j["a1"]["end"]), "2+2 cores should run concurrently"
    # 40 + 40 cores > 64: must be serialised
    home.submit("a3", "sleep 2", cores=40)
    home.submit("a4", "sleep 2", cores=40)
    assert home.wait("a3", "a4").returncode == 0
    j = home.jobs()
    assert t(j["a4"]["start"]) >= t(j["a3"]["end"]), "40+40 cores must not be co-scheduled"
    assert j["a3"]["ram_gb"] == 80  # default 2 GB/core


@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="no GPU")
def test_b_gpu_oom_recorded_and_queue_moves_on(home):
    oom = ("python -c \"import torch; x = torch.empty(3 * 2**30, dtype=torch.uint8, device='cuda'); "
           "print('allocated 3GB?!')\"")
    ok = ("python -c \"import os, time, torch; x = torch.empty(2**29, dtype=torch.uint8, device='cuda'); "
          "print('cap', os.environ.get('ORCH_VRAM_FRACTION_SET'), 'alloc', torch.cuda.memory_allocated()); time.sleep(2)\"")
    home.submit("b1_oom", oom, gpu="yes", vram=1)
    home.submit("b2_ok", ok, gpu="yes", vram=1)
    home.submit("b3_cpu", "python -c \"import torch; print(torch.cuda.is_available())\"")
    r = home.wait("b1_oom", "b2_ok", "b3_cpu")
    assert r.returncode == 1  # b1 failed
    j = home.jobs()
    assert j["b1_oom"]["status"] == "failed"
    assert j["b1_oom"]["failure_reason"].startswith("CUDA OOM"), j["b1_oom"]["failure_reason"]
    assert j["b2_ok"]["status"] == "done", j["b2_ok"]
    log2 = (home.repo / j["b2_ok"]["log"]).read_text()
    assert "cap 1 alloc 536870912" in log2
    assert j["b2_ok"]["peak_gpu_gb"] and j["b2_ok"]["peak_gpu_gb"] > 0.4
    assert j["b3_cpu"]["status"] == "done"
    assert "False" in (home.repo / j["b3_cpu"]["log"]).read_text()  # CUDA hidden for gpu=no


def test_c_killed_and_traceback_reasons(home):
    home.submit("c1_kill9", "python -c \"import os; os.kill(os.getpid(), 9)\"")
    home.submit("c2_exc", "python -c \"raise ValueError('boom 42')\"")
    home.submit("c3_hog", "python -c \"import time; x = bytearray(600 * 2**20); time.sleep(2)\"")
    home.wait("c1_kill9", "c2_exc", "c3_hog")
    j = home.jobs()
    assert j["c1_kill9"]["status"] == "failed" and j["c1_kill9"]["exit_code"] == 137
    assert "Killed" in j["c1_kill9"]["failure_reason"]
    assert j["c2_exc"]["failure_reason"] == "Python exception: ValueError: boom 42"
    assert j["c3_hog"]["status"] == "done" and j["c3_hog"]["peak_rss_gb"] >= 0.5


def test_d_gpu_vram_packing(home):
    home.submit("d1", "sleep 3", gpu="yes", vram=12)
    home.submit("d2", "sleep 3", gpu="yes", vram=12)
    assert home.wait("d1", "d2").returncode == 0
    j = home.jobs()
    assert t(j["d2"]["start"]) >= t(j["d1"]["end"]), "12+12 GB must not be co-scheduled"
    home.submit("d3", "sleep 3", gpu="yes", vram=8)
    home.submit("d4", "sleep 3", gpu="yes", vram=8)
    assert home.wait("d3", "d4").returncode == 0
    j = home.jobs()
    assert t(j["d4"]["start"]) < t(j["d3"]["end"]), "8+8 GB should be co-scheduled"


def test_e_dirty_tree_refused(home):
    toy = home.repo / "src" / "toy.py"
    toy.write_text("VALUE = 99\n")
    r = home.submit("e1", "true", check=False)
    assert r.returncode == 3 and "REFUSED" in r.stderr
    sh(["git", "checkout", "src/toy.py"], home.repo)
    (home.repo / "src" / "new.py").write_text("x = 1\n")
    r = home.submit("e2", "true", check=False)
    assert r.returncode == 3 and "src/new.py" in r.stderr
    (home.repo / "src" / "new.py").unlink()
    (home.repo / "logs").mkdir(exist_ok=True)
    (home.repo / "logs" / "noise.log").write_text("noise\n")
    home.submit("e3", "true")  # logs noise is fine
    assert home.wait("e3").returncode == 0
    assert "e1" not in home.jobs()


def test_f_snapshot_isolation(home):
    cmd = ("sleep 2; python -c \"import toy, os; print('VALUE', toy.VALUE); "
           "open(os.environ['ORCH_OUT'] + '/o.txt', 'w').write('x')\"; ls -l data results")
    home.submit("f1", cmd)
    (home.repo / "src" / "toy.py").write_text("VALUE = 2\n")
    home.commit("change toy")
    home.submit("f2", "python -c \"import toy; print('VALUE', toy.VALUE)\"")
    assert home.wait("f1", "f2").returncode == 0
    j = home.jobs()
    assert "VALUE 1" in (home.repo / j["f1"]["log"]).read_text()
    assert "VALUE 2" in (home.repo / j["f2"]["log"]).read_text()
    assert j["f1"]["commit"] != j["f2"]["commit"]
    assert (home.repo / "runs" / "f1" / "out" / "o.txt").exists()
    code = home.repo / "runs" / "f1" / "code"
    assert (code / "data").resolve() == (home.repo / "data").resolve()
    assert (code / "results").resolve() == (home.repo / "results").resolve()


def test_cancel_and_timeout(home):
    home.submit("g_cancel", "sleep 60")
    home.submit("g_timeout", "sleep 60", extra=("--timeout-min", "0.05"))
    for _ in range(50):
        if home.jobs()["g_cancel"]["status"] == "running":
            break
        time.sleep(0.2)
    home.orch("cancel", "g_cancel")
    home.wait("g_cancel", "g_timeout")
    j = home.jobs()
    assert j["g_cancel"]["status"] == "killed"
    assert j["g_timeout"]["status"] == "failed" and "timeout" in j["g_timeout"]["failure_reason"]
    time.sleep(1)
    for jid in ("g_cancel", "g_timeout"):
        r = subprocess.run(["pgrep", "-g", str(j[jid]["pgid"])], capture_output=True)
        assert r.returncode == 1, f"{jid} process group still alive"


def test_tasks_and_artifacts(home):
    home.orch("task", "add", "T1", "--owner", "sim", "--outputs", "toyset@v1", "--done-criterion", "file exists")
    home.orch("task", "add", "T2", "--owner", "famC", "--depends-on", "T1", "--inputs", "toyset@v1")
    assert "READY T1" in home.orch("task", "ready").stdout
    assert "T2" not in home.orch("task", "ready").stdout
    home.submit("h1", "mkdir -p $ORCH_DATA/processed/toyset && echo hi > $ORCH_DATA/processed/toyset/a.txt",
                extra=("--task", "T1"))
    assert home.wait("h1").returncode == 0
    home.orch("artifact", "register", "--name", "toyset", "--version", "v1", "--path", "data/processed/toyset",
              "--job", "h1", "--schema", "n_loci=10", "rho_grid=[0, 0.5]")
    home.orch("task", "set", "T1", "--status", "done")
    assert "READY T2" in home.orch("task", "ready").stdout
    assert home.orch("task", "show", "T1").stdout.count("h1") == 1
    assert home.orch("artifact", "verify", "toyset@v1").returncode == 0
    (home.repo / "data" / "processed" / "toyset" / "a.txt").write_text("changed!\n")
    assert home.orch("artifact", "verify", "toyset", check=False).returncode == 1
    home.orch("artifact", "stale", "toyset@v1", "--reason", "test")
    assert "T2" not in home.orch("task", "ready").stdout


def test_g_reconcile_lost_job_and_placeholder(home):
    # clean-up of earlier deliberate failures so only the new flags matter
    home.orch("ack", *[k for k, v in home.jobs().items() if v["status"] in ("failed", "lost")])
    home.submit("r1", "sleep 120")
    for _ in range(50):
        j = home.jobs()["r1"]
        if j["status"] == "running":
            break
        time.sleep(0.2)
    assert j["status"] == "running"
    home.orch("scheduler", "--stop")  # reconcile, not the scheduler, must catch it
    os.killpg(int(j["pgid"]), signal.SIGKILL)
    time.sleep(0.5)
    (home.repo / "reports").mkdir(exist_ok=True)
    (home.repo / "reports" / "x.md").write_text("result: __C1L8__\nok line\nVALID_GRID_COMMIT here\n")
    r = home.orch("reconcile", "--no-du", check=False)
    assert r.returncode == 1
    assert "job r1 was 'running' but PID" in r.stdout and "-> lost" in r.stdout, r.stdout
    assert "reports/x.md has placeholders" in r.stdout and "__C1L8__" in r.stdout and "VALID_GRID_COMMIT" in r.stdout
    assert home.jobs()["r1"]["status"] == "lost"
    home.orch("scheduler")  # restore for teardown symmetry
