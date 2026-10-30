#!/usr/bin/env python3
"""orch -- job queue/scheduler, task and artifact registries for this project.

See scripts/orch/README.md. Single source of truth lives in ORCH_HOME (the main
checkout, auto-detected from git's common dir; override with $ORCH_HOME):
  JOBS.yaml, TASKS.yaml, ARTIFACTS.yaml   registries (all writes under flock)
  .orch/                                  lock, scheduler pid/heartbeat
  runs/<job_id>/{code,site,out,run.sh,exit_code}   per-job snapshot + outputs
  logs/jobs/<job_id>.log                  job stdout+stderr
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

MAX_CORES = int(os.environ.get("ORCH_MAX_CORES", 64))
MAX_RAM_GB = float(os.environ.get("ORCH_MAX_RAM_GB", 120))
GPU_VRAM_LIMIT = float(os.environ.get("ORCH_GPU_VRAM_LIMIT", 22))  # sum of co-scheduled GPU jobs must be < this
POLL_S = float(os.environ.get("ORCH_POLL_S", 2))
SAMPLE_S = float(os.environ.get("ORCH_SAMPLE_S", 5))
BACKFILL_MAX_WAIT_MIN = float(os.environ.get("ORCH_BACKFILL_MAX_WAIT_MIN", 30))
MAIN_BRANCH = os.environ.get("ORCH_MAIN_BRANCH", "master")

TERMINAL = ("done", "failed", "killed", "lost")
# Tracked changes under these prefixes (or root-level *.md / registries) do not block a submit.
NOISE_PREFIXES = ("logs/", "data/", "results/", "runs/", "reports/", "reviews/", "lit/")
CODE_DIRS = ("src/", "scripts/", "datasets/configs/", "tests/", "env/")
REGISTRY_FILES = ("JOBS.yaml", "TASKS.yaml", "ARTIFACTS.yaml")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


# ----------------------------------------------------------------------------- basics
def now() -> str:
    return dt.datetime.now().astimezone().replace(microsecond=0).isoformat()


def parse_t(s):
    return dt.datetime.fromisoformat(s) if s else None


def die(msg, code=2):
    print(f"orch: {msg}", file=sys.stderr)
    sys.exit(code)


def git(args, cwd, check=True):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed in {cwd}: {r.stderr.strip()}")
    return r.stdout.strip()


class Paths:
    def __init__(self, home: Path):
        self.home = home
        self.state = home / ".orch"
        self.jobs = home / "JOBS.yaml"
        self.tasks = home / "TASKS.yaml"
        self.arts = home / "ARTIFACTS.yaml"
        self.runs = home / "runs"
        self.logs = home / "logs" / "jobs"
        self.lock = self.state / "registry.lock"
        self.schedlock = self.state / "scheduler.lock"
        self.pidfile = self.state / "scheduler.pid"
        self.heartbeat = self.state / "scheduler.heartbeat"


def paths() -> Paths:
    h = os.environ.get("ORCH_HOME")
    if h:
        return Paths(Path(h).resolve())
    common = git(["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=os.getcwd())
    return Paths(Path(common).parent)


def env_script(p: Paths) -> Path:
    return Path(os.environ.get("ORCH_ENV_SCRIPT", p.home / "env" / "activate.sh"))


@contextmanager
def locked(p: Paths):
    p.state.mkdir(parents=True, exist_ok=True)
    with open(p.lock, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


HEADERS = {
    "jobs": "# JOBS.yaml -- written by scripts/orch (do not hand-edit while the scheduler runs)\n",
    "tasks": "# TASKS.yaml -- edit via `orch task ...` (flock-protected)\n",
    "artifacts": "# ARTIFACTS.yaml -- edit via `orch artifact ...`; consumers read this, not reports\n",
}


def load(path: Path, key: str) -> list:
    if not path.exists():
        return []
    d = yaml.safe_load(path.read_text()) or {}
    return d.get(key) or []


def save(path: Path, key: str, items: list):
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(HEADERS[key] + yaml.safe_dump({key: items}, sort_keys=False, width=1000,
                                                 default_flow_style=False))
    os.replace(tmp, path)


def find(items, key, val):
    for it in items:
        if it.get(key) == val:
            return it
    return None


# ----------------------------------------------------------------------------- git checks
def tree_root(cwd=None) -> Path:
    return Path(git(["rev-parse", "--show-toplevel"], cwd=cwd or os.getcwd()))


def dirty_files(tree: Path) -> list[str]:
    """Code-relevant uncommitted changes: tracked modifications outside noise dirs, and
    untracked (non-ignored) files under src/ scripts/ datasets/configs/ tests/ env/."""
    out = subprocess.run(["git", "status", "--porcelain=v1"], cwd=tree, capture_output=True,
                         text=True, check=True).stdout
    bad = []
    for line in out.splitlines():
        xy, path = line[:2], line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip('"')
        if xy == "??":
            if path.startswith(CODE_DIRS):
                bad.append(line)
        else:
            if path.startswith(NOISE_PREFIXES) or path in REGISTRY_FILES:
                continue
            if "/" not in path and path.endswith(".md"):
                continue
            bad.append(line)
    return bad


# ----------------------------------------------------------------------------- process utils
def pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        st = Path(f"/proc/{pid}/stat").read_text()
        return st[st.rindex(")") + 2] != "Z"
    except (OSError, ValueError):
        return False


def proc_table():
    """pid -> dict(pgrp, uid, rss_bytes, cmd, cwd) for all readable processes."""
    page = os.sysconf("SC_PAGE_SIZE")
    tab = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            st = Path(f"/proc/{d}/stat").read_text()
            rest = st[st.rindex(")") + 2:].split()
            state, pgrp = rest[0], int(rest[2])
            if state == "Z":
                continue
            rss = int(Path(f"/proc/{d}/statm").read_text().split()[1]) * page
            uid = os.stat(f"/proc/{d}").st_uid
            cmd = Path(f"/proc/{d}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            try:
                cwd = os.readlink(f"/proc/{d}/cwd")
            except OSError:
                cwd = ""
            tab[int(d)] = dict(pgrp=pgrp, uid=uid, rss=rss, cmd=cmd, cwd=cwd)
        except (OSError, ValueError, IndexError):
            continue
    return tab


def gpu_apps():
    """pid -> used MiB (compute apps), or None if nvidia-smi unavailable."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode:
        return None
    out = {}
    for line in r.stdout.strip().splitlines():
        try:
            pid, mem = [x.strip() for x in line.split(",")]
            out[int(pid)] = out.get(int(pid), 0) + float(mem)
        except ValueError:
            continue
    return out


def job_pids(job_id):
    """PIDs of our processes whose environment carries ORCH_JOB_ID=job_id (catches workers that left the
    job's process group, e.g. multiprocessing/R workers started with their own session or pgid)."""
    out, uid, key = [], os.getuid(), f"ORCH_JOB_ID={job_id}".encode()
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            if os.stat(f"/proc/{d}").st_uid != uid:
                continue
            with open(f"/proc/{d}/environ", "rb") as f:
                if key in f.read().split(b"\0"):
                    out.append(int(d))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return out


def killpg(pgid, sig, job_id=None):
    ok = False
    try:
        os.killpg(int(pgid), sig)
        ok = True
    except (ProcessLookupError, PermissionError, TypeError):
        pass
    if job_id:
        for pid in job_pids(job_id):
            try:
                os.kill(pid, sig)
                ok = True
            except (ProcessLookupError, PermissionError):
                pass
    return ok


def pg_alive(pgid, tab=None) -> bool:
    tab = tab if tab is not None else proc_table()
    return any(v["pgrp"] == int(pgid) for v in tab.values())


# ----------------------------------------------------------------------------- failure classification
CUDA_OOM_RE = re.compile(r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|"
                         r"cudaErrorMemoryAllocation|CUDA error: out of memory")
EXC_RE = re.compile(r"^([A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt|Failure)|KeyboardInterrupt)\b.*")


def log_tail(log: Path, nbytes=65536) -> str:
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - nbytes))
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def classify(ec, log: Path) -> str | None:
    if ec == 0:
        return None
    tail = log_tail(log)
    lines = [l.rstrip() for l in tail.splitlines() if l.strip()]
    oom = [l for l in lines if CUDA_OOM_RE.search(l)]
    if oom:
        return "CUDA OOM: " + oom[-1].strip()[:240]
    if any("MemoryError" in l for l in lines[-30:]):
        return "host OOM: MemoryError"
    if ec in (137, -9):
        return f"Killed (SIGKILL, exit {ec}): host OOM-killer or external kill -9"
    if ec in (143, -15):
        return f"terminated (SIGTERM, exit {ec})"
    if ec == 97:
        return "orch: job cwd does not exist in snapshot"
    if "Traceback (most recent call last)" in tail:
        after = tail.rsplit("Traceback (most recent call last)", 1)[1].splitlines()
        exc = [l for l in after if EXC_RE.match(l)]
        return "Python exception: " + (exc[-1] if exc else after[-1] if after else "?").strip()[:240]
    rerr = [l for l in lines if l.startswith("Error in ") or l.startswith("Error:")]
    if rerr:
        return "R error: " + rerr[-1][:240]
    return f"nonzero exit {ec}"


# ----------------------------------------------------------------------------- submit
def cmd_submit(a):
    p = paths()
    if not ID_RE.match(a.id):
        die(f"bad job id {a.id!r} (allowed: letters, digits, _ . + -)")
    if not a.command:
        die("no command given (put it after --)")
    if a.gpu == "no" and a.vram_gb:
        die("--vram-gb only makes sense with --gpu yes")
    if a.gpu == "yes" and not a.vram_gb:
        die("GPU jobs must declare --vram-gb (peak)")
    ram = a.ram_gb if a.ram_gb is not None else 2.0 * a.cores
    if a.cores < 1 or a.cores > MAX_CORES or ram > MAX_RAM_GB or (a.vram_gb or 0) > 24:
        die(f"request exceeds node budget (cores<= {MAX_CORES}, ram<= {MAX_RAM_GB} GB, vram<=24 GB)")
    tree = tree_root()
    bad = dirty_files(tree)
    if bad:
        die("REFUSED: tree %s has uncommitted code changes; commit them first:\n  %s"
            % (tree, "\n  ".join(bad[:20])), 3)
    commit = git(["rev-parse", "HEAD"], cwd=tree)
    branch = git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=tree)
    p.runs.mkdir(parents=True, exist_ok=True)
    p.logs.mkdir(parents=True, exist_ok=True)
    run = p.runs / a.id
    with locked(p):
        if find(load(p.jobs, "jobs"), "id", a.id):
            die(f"job id {a.id} already exists in JOBS.yaml")
        try:
            run.mkdir()
        except FileExistsError:
            die(f"{run} already exists (stale run dir?); pick another id")
    # snapshot committed code
    code = run / "code"
    code.mkdir()
    arch = subprocess.Popen(["git", "archive", "--format=tar", commit], cwd=tree, stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(code)], stdin=arch.stdout, check=True)
    if arch.wait():
        shutil.rmtree(run)
        die("git archive failed")
    for d in ("data", "results", "logs"):
        tgt = code / d
        if tgt.is_symlink() or tgt.is_file():
            tgt.unlink()
        elif tgt.exists():
            shutil.rmtree(tgt)
        (p.home / d).mkdir(exist_ok=True)
        tgt.symlink_to(p.home / d)
    if (p.home / ".venv").exists() and not (code / ".venv").exists():
        (code / ".venv").symlink_to(p.home / ".venv")
    # untracked installs under env/ (R libraries, SharePro source, tools, micromamba envs) are not in git:
    # link each missing entry so snapshot jobs see the same environment as the main checkout
    if (p.home / "env").is_dir():
        (code / "env").mkdir(exist_ok=True)
        for e in (p.home / "env").iterdir():
            t = code / "env" / e.name
            if not t.exists() and not t.is_symlink():
                t.symlink_to(e)
    if not (code / a.cwd).is_dir():
        shutil.rmtree(run)
        die(f"--cwd {a.cwd} does not exist at commit {commit[:10]}")
    (run / "out").mkdir()
    (run / "site").mkdir()
    shutil.copy(HERE / "sitecustomize_template.py", run / "site" / "sitecustomize.py")
    cmd = a.command[0] if len(a.command) == 1 else shlex.join(a.command)
    rec = dict(
        id=a.id, owner=a.owner, task=a.task, status="queued", gpu=a.gpu == "yes",
        vram_gb=a.vram_gb or 0, cores=a.cores, ram_gb=ram, expected_min=a.expected_min, priority=int(getattr(a, "priority", 0) or 0),
        timeout_min=a.timeout_min, commit=commit, branch=branch, tree=str(tree), cwd=a.cwd,
        command=cmd, log=str((p.logs / f"{a.id}.log").relative_to(p.home)),
        run_dir=str(run.relative_to(p.home)), submitted=now(), pid=None, pgid=None,
        start=None, expected_end=None, end=None, exit_code=None, failure_reason=None,
        peak_rss_gb=None, peak_gpu_gb=None,
    )
    with locked(p):
        jobs = load(p.jobs, "jobs")
        jobs.append(rec)
        save(p.jobs, "jobs", jobs)
        if a.task:
            tasks = load(p.tasks, "tasks")
            t = find(tasks, "id", a.task)
            if t is not None:
                t.setdefault("jobs", []).append(a.id)
                t["updated"] = now()
                save(p.tasks, "tasks", tasks)
            else:
                print(f"orch: warning: task {a.task} not in TASKS.yaml", file=sys.stderr)
    print(f"queued {a.id} @ {commit[:10]} ({branch}); log: {p.home / rec['log']}")
    if not scheduler_running(p):
        print("orch: WARNING: scheduler is not running (start: scripts/orch/orch scheduler)", file=sys.stderr)


# ----------------------------------------------------------------------------- scheduler
def scheduler_running(p: Paths):
    """Return pid if a scheduler holds the scheduler lock, else None."""
    if not p.schedlock.exists():
        return None
    with open(p.schedlock, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                return int(p.pidfile.read_text().strip())
            except (OSError, ValueError):
                return -1
        fcntl.flock(f, fcntl.LOCK_UN)
        return None


def write_runsh(p: Paths, j) -> Path:
    run = p.home / j["run_dir"]
    code = run / "code"
    es = env_script(p)
    ex = {
        "ORCH_JOB_ID": j["id"], "ORCH_HOME": str(p.home), "ORCH_RUN_DIR": str(run),
        "ORCH_OUT": str(run / "out"), "ORCH_CODE": str(code), "ORCH_DATA": str(p.home / "data"),
        "ORCH_RESULTS": str(p.home / "results"), "ORCH_CORES": str(j["cores"]),
        "ORCH_RAM_GB": str(j["ram_gb"]), "ORCH_GPU": "1" if j["gpu"] else "0",
        "ORCH_COMMIT": j["commit"],
        "PYTHONPATH": f"{run / 'site'}:{code / 'src'}",
    }
    if j["gpu"]:
        ex["ORCH_VRAM_GB"] = str(j["vram_gb"])
    else:
        ex["CUDA_VISIBLE_DEVICES"] = ""
    lines = ["#!/bin/bash", f"# generated by orch for job {j['id']} at {now()}"]
    if es.exists():
        lines.append(f"source {shlex.quote(str(es))}")
    lines += [f"export {k}={shlex.quote(v)}" for k, v in ex.items()]
    if j["gpu"]:
        lines.append('export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"')
    lines += [
        f"cd {shlex.quote(str(code / j['cwd']))} || {{ echo 97 > {shlex.quote(str(run / 'exit_code'))}; exit 97; }}",
        f'echo "[orch] job {j["id"]} start $(date -Is) host $(hostname) commit {j["commit"][:10]} pid $$"',
        f"echo {shlex.quote('[orch] cmd: ' + j['command'])}",  # single-quoted: $(...)/backticks are not evaluated
        j["command"],
        "ec=$?",
        'echo "[orch] exit $ec at $(date -Is)"',
        f"echo $ec > {shlex.quote(str(run / 'exit_code.tmp'))} && mv {shlex.quote(str(run / 'exit_code.tmp'))} {shlex.quote(str(run / 'exit_code'))}",
        "exit $ec", "",
    ]
    rs = run / "run.sh"
    rs.write_text("\n".join(lines))
    rs.chmod(0o755)
    return rs


class Scheduler:
    def __init__(self, p: Paths):
        self.p = p
        self.children = {}      # job id -> Popen
        self.kill_at = {}       # pgid -> time to SIGKILL
        self.last_sample = 0.0
        self.stop = False

    def log(self, msg):
        print(f"[{now()}] {msg}", flush=True)

    def finalize(self, j, ec):
        j["end"] = now()
        j["exit_code"] = ec
        if ec is None:
            j["status"] = "lost"
            j["failure_reason"] = "process vanished without exit code (killed externally / scheduler restart / node issue)"
        elif ec == 0:
            j["status"] = "done"
        else:
            j["status"] = "failed"
            j["failure_reason"] = classify(ec, self.p.home / j["log"])
        self.log(f"job {j['id']} -> {j['status']} (exit {ec}) {j.get('failure_reason') or ''}")

    def launch(self, j):
        rs = write_runsh(self.p, j)
        logf = open(self.p.home / j["log"], "ab")
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
        proc = subprocess.Popen(["bash", str(rs)], stdout=logf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True,
                                cwd=str(self.p.home / j["run_dir"]), env=env)
        logf.close()
        self.children[j["id"]] = proc
        t = dt.datetime.now().astimezone().replace(microsecond=0)
        j.update(status="running", pid=proc.pid, pgid=proc.pid, start=t.isoformat(),
                 expected_end=(t + dt.timedelta(minutes=float(j["expected_min"]))).isoformat())
        self.log(f"launched {j['id']} pid {proc.pid} gpu={j['gpu']} vram={j['vram_gb']} cores={j['cores']}")

    def tick(self):
        p = self.p
        tab = gpu = None  # sample outside the lock (nvidia-smi can be slow)
        if time.time() - self.last_sample >= SAMPLE_S and any(
                j["status"] == "running" for j in load(p.jobs, "jobs")):
            tab, gpu = proc_table(), gpu_apps()
            self.last_sample = time.time()
        with locked(p):
            jobs = load(p.jobs, "jobs")
            changed = False
            running = [j for j in jobs if j["status"] == "running"]
            for j in running:
                ch = self.children.get(j["id"])
                rc = ch.poll() if ch is not None else None
                done = (rc is not None) if ch is not None else not pid_alive(j["pid"])
                if done:
                    ecf = p.home / j["run_dir"] / "exit_code"
                    ec = int(ecf.read_text().strip()) if ecf.exists() else rc
                    self.finalize(j, ec)
                    self.children.pop(j["id"], None)
                    changed = True
                    continue
                if tab is not None:
                    pg = int(j["pgid"])
                    pids = [pid for pid, v in tab.items() if v["pgrp"] == pg]
                    rss = round(sum(tab[x]["rss"] for x in pids) / 2**30, 2)
                    if rss > (j.get("peak_rss_gb") or 0):
                        j["peak_rss_gb"] = rss
                        changed = True
                    if gpu:
                        g = round(sum(gpu.get(x, 0) for x in pids) / 1024, 2)
                        if g > (j.get("peak_gpu_gb") or 0):
                            j["peak_gpu_gb"] = g
                            changed = True
                if j.get("timeout_min"):
                    el = (dt.datetime.now().astimezone() - parse_t(j["start"])).total_seconds() / 60
                    if el > float(j["timeout_min"]):
                        killpg(j["pgid"], signal.SIGTERM, j["id"])
                        self.kill_at[int(j["pgid"])] = time.time() + 15
                        j.update(status="failed", end=now(), failure_reason=f"timeout after {j['timeout_min']} min")
                        self.log(f"job {j['id']} timed out; SIGTERM pgid {j['pgid']}")
                        changed = True
            # reap children of jobs cancelled/killed by others
            for jid in list(self.children):
                if self.children[jid].poll() is not None:
                    jj = find(jobs, "id", jid)
                    if jj is None or jj["status"] != "running":
                        self.children.pop(jid)
            for pg, t in list(self.kill_at.items()):
                if time.time() >= t:
                    killpg(pg, signal.SIGKILL)
                    del self.kill_at[pg]
            # launch queued jobs (FIFO with backfill; head-of-queue reservation after a long wait)
            running = [j for j in jobs if j["status"] == "running"]
            cores = sum(j["cores"] for j in running)
            ram = sum(j["ram_gb"] for j in running)
            gpu_v = [j["vram_gb"] for j in running if j["gpu"]]
            queued = [j for j in jobs if j["status"] == "queued"]
            # higher priority first, then FIFO by submission order (stable sort keeps registry order)
            queued.sort(key=lambda j: -int(j.get("priority") or 0))
            reserve_gpu_only = False  # a long-waiting head blocked only by VRAM reserves the GPU (and its own CPU/RAM share)
            res_cores = res_ram = 0
            for k, j in enumerate(queued):
                cpu_fits = (cores + res_cores + j["cores"] <= MAX_CORES
                            and ram + res_ram + j["ram_gb"] <= MAX_RAM_GB)
                gpu_fits = (not j["gpu"]) or (not gpu_v or sum(gpu_v) + j["vram_gb"] < GPU_VRAM_LIMIT)
                fits = cpu_fits and gpu_fits and not (reserve_gpu_only and j["gpu"])
                if fits:
                    try:
                        self.launch(j)
                    except Exception as e:  # noqa: BLE001
                        j.update(status="failed", end=now(), failure_reason=f"launch error: {e}")
                        self.log(f"launch of {j['id']} failed: {e}")
                    changed = True
                    if j["status"] == "running":
                        cores += j["cores"]
                        ram += j["ram_gb"]
                        if j["gpu"]:
                            gpu_v.append(j["vram_gb"])
                elif k == 0:
                    waited = (dt.datetime.now().astimezone() - parse_t(j["submitted"])).total_seconds() / 60
                    if waited > BACKFILL_MAX_WAIT_MIN:
                        if cpu_fits and not gpu_fits:
                            reserve_gpu_only = True  # only GPU jobs may not overtake it ...
                            res_cores, res_ram = j["cores"], j["ram_gb"]  # ... and backfill must leave its CPU/RAM free
                        else:
                            break  # reserve: nothing may overtake a head job that has waited too long
            if changed:
                save(p.jobs, "jobs", jobs)
        p.heartbeat.write_text(now() + "\n")

    def run(self):
        p = self.p
        p.state.mkdir(parents=True, exist_ok=True)
        lockf = open(p.schedlock, "a")
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("orch: scheduler already running", file=sys.stderr)
            return 0
        p.pidfile.write_text(f"{os.getpid()}\n")
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop", True))
        self.log(f"scheduler start pid {os.getpid()} home {p.home} cores<={MAX_CORES} ram<={MAX_RAM_GB} "
                 f"gpu sum vram<{GPU_VRAM_LIMIT}")
        while not self.stop:
            try:
                self.tick()
            except Exception:  # noqa: BLE001 -- never die on one bad tick
                self.log("tick error:\n" + traceback.format_exc())
            t_end = time.time() + POLL_S
            while not self.stop and time.time() < t_end:
                time.sleep(0.2)
        self.log("scheduler stop (running jobs keep running; they are re-adopted on restart)")
        try:
            p.pidfile.unlink()
        except OSError:
            pass
        return 0


def cmd_scheduler(a):
    p = paths()
    pid = scheduler_running(p)
    if a.status:
        hb = p.heartbeat.read_text().strip() if p.heartbeat.exists() else "never"
        print(f"scheduler {'RUNNING pid %s' % pid if pid else 'NOT running'}; home {p.home}; last heartbeat {hb}")
        return 0 if pid else 1
    if a.stop:
        if not pid:
            print("scheduler not running")
            return 0
        os.kill(pid, signal.SIGTERM)
        for _ in range(100):
            if not scheduler_running(p):
                print(f"scheduler pid {pid} stopped")
                return 0
            time.sleep(0.1)
        die(f"scheduler pid {pid} did not stop")
    if a.foreground:
        return Scheduler(p).run()
    if pid:
        print(f"scheduler already running (pid {pid})")
        return 0
    p.logs.mkdir(parents=True, exist_ok=True)
    log = open(p.logs / "scheduler.log", "ab")
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "scheduler", "--foreground"],
                     stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     start_new_session=True, cwd=str(p.home))
    for _ in range(50):
        pid = scheduler_running(p)
        if pid and pid > 0:
            print(f"scheduler started (pid {pid}); log {p.logs / 'scheduler.log'}")
            return 0
        time.sleep(0.1)
    die("scheduler did not start; see " + str(p.logs / "scheduler.log"))


# ----------------------------------------------------------------------------- job commands
def cmd_cancel(a):
    p = paths()
    rc = 0
    for jid in a.ids:
        with locked(p):
            jobs = load(p.jobs, "jobs")
            j = find(jobs, "id", jid)
            if j is None:
                print(f"{jid}: unknown")
                rc = 1
                continue
            if j["status"] not in ("queued", "running"):
                print(f"{jid}: already {j['status']}")
                continue
            was = j["status"]
            j.update(status="killed", end=now(), failure_reason=f"cancelled by user ({a.reason})")
            save(p.jobs, "jobs", jobs)
        if was == "running":
            killpg(j["pgid"], signal.SIGTERM, j["id"])
            for _ in range(100):
                if not pg_alive(j["pgid"]):
                    break
                time.sleep(0.1)
            else:
                killpg(j["pgid"], signal.SIGKILL, j["id"])
        print(f"{jid}: killed (was {was})")
    return rc


def cmd_prio(a):
    p = paths()
    with locked(p):
        jobs = load(p.jobs, "jobs")
        for jid in a.ids:
            j = find(jobs, "id", jid)
            if j is None:
                die(f"unknown job {jid}")
            j["priority"] = int(a.priority)
        save(p.jobs, "jobs", jobs)
    print(f"priority {a.priority}: {' '.join(a.ids)}")


def cmd_ack(a):
    p = paths()
    with locked(p):
        jobs = load(p.jobs, "jobs")
        for jid in a.ids:
            j = find(jobs, "id", jid)
            if j is None:
                die(f"unknown job {jid}")
            j["acked"] = a.note or True
        save(p.jobs, "jobs", jobs)


def fmt_job(j):
    el = ""
    if j.get("start"):
        end = parse_t(j["end"]) if j.get("end") else dt.datetime.now().astimezone()
        el = f"{(end - parse_t(j['start'])).total_seconds() / 60:.1f}/{j['expected_min']}m"
    res = f"c{j['cores']} {j['ram_gb']}G" + (f" gpu{j['vram_gb']}G" if j["gpu"] else "")
    peaks = f"rss {j.get('peak_rss_gb') or '-'}G" + (f" vram {j.get('peak_gpu_gb') or '-'}G" if j["gpu"] else "")
    s = f"{j['id']:<28} {j['status']:<8} {j['owner'] or '-':<12} {res:<20} {el:<12} {peaks:<22} {j['commit'][:8]}"
    if j.get("failure_reason"):
        s += f"\n    reason: {j['failure_reason']}"
    return s


def cmd_status(a):
    p = paths()
    jobs = load(p.jobs, "jobs")
    if a.running:
        jobs = [j for j in jobs if j["status"] in ("running", "queued")]
    if a.ids:
        jobs = [j for j in jobs if j["id"] in a.ids]
    if a.yaml:
        print(yaml.safe_dump(jobs, sort_keys=False))
        return 0
    pid = scheduler_running(p)
    print(f"# scheduler: {'running pid %s' % pid if pid else 'NOT RUNNING'}; home {p.home}")
    for j in jobs[-a.last:] if a.last else jobs:
        print(fmt_job(j))
    return 0


def cmd_wait(a):
    p = paths()
    poll = float(os.environ.get("ORCH_WAIT_POLL_S", 10))
    t0 = time.time()
    while True:
        jobs = {j["id"]: j for j in load(p.jobs, "jobs")}
        missing = [x for x in a.ids if x not in jobs]
        if missing:
            die(f"unknown job(s): {missing}")
        if all(jobs[x]["status"] in TERMINAL for x in a.ids):
            break
        if a.timeout_min and time.time() - t0 > a.timeout_min * 60:
            print("orch wait: timeout")
            return 2
        time.sleep(poll)
    bad = 0
    for x in a.ids:
        print(fmt_job(jobs[x]))
        bad += jobs[x]["status"] != "done"
    return 1 if bad else 0


# ----------------------------------------------------------------------------- tasks
TASK_STATUS = ("planned", "blocked", "running", "done", "failed", "abandoned")


def csv(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def cmd_task(a):
    p = paths()
    if a.tcmd in ("add", "set"):
        with locked(p):
            tasks = load(p.tasks, "tasks")
            t = find(tasks, "id", a.id)
            if a.tcmd == "add":
                if t is not None:
                    die(f"task {a.id} exists; use `orch task set`")
                if not ID_RE.match(a.id):
                    die("bad task id")
                t = dict(id=a.id, owner=a.owner, status=a.status or "planned", depends_on=csv(a.depends_on),
                         inputs=csv(a.inputs), outputs=csv(a.outputs), done_criterion=a.done_criterion,
                         jobs=[], notes=a.notes, created=now(), updated=now())
                tasks.append(t)
            else:
                if t is None:
                    die(f"unknown task {a.id}")
                for k in ("owner", "status", "done_criterion", "notes"):
                    v = getattr(a, k)
                    if v is not None:
                        t[k] = v
                for k in ("depends_on", "inputs", "outputs"):
                    v = getattr(a, k)
                    if v is not None:
                        t[k] = csv(v)
                if a.append_notes:
                    t["notes"] = ((t.get("notes") or "") + f"\n[{now()}] {a.append_notes}").strip()
                if a.add_job:
                    t.setdefault("jobs", []).append(a.add_job)
                t["updated"] = now()
            if t["status"] not in TASK_STATUS:
                die(f"status must be one of {TASK_STATUS}")
            save(p.tasks, "tasks", tasks)
        print(yaml.safe_dump(t, sort_keys=False).rstrip())
        return 0
    tasks = load(p.tasks, "tasks")
    if a.tcmd == "show":
        sel = [t for t in tasks if (not a.id or t["id"] == a.id) and (not a.status or t["status"] == a.status)]
        if a.id:
            print(yaml.safe_dump(sel, sort_keys=False).rstrip())
        else:
            for t in sel:
                print(f"{t['id']:<32} {t['status']:<10} {t.get('owner') or '-':<14} deps={','.join(t.get('depends_on') or []) or '-'}"
                      f" jobs={','.join(t.get('jobs') or []) or '-'}")
        return 0
    if a.tcmd == "ready":
        arts = load(p.arts, "artifacts")
        for t in tasks:
            if t["status"] not in ("planned", "blocked"):
                continue
            why = []
            for d in t.get("depends_on") or []:
                dt_ = find(tasks, "id", d)
                if dt_ is None or dt_["status"] != "done":
                    why.append(f"dep {d}={dt_['status'] if dt_ else 'missing'}")
            for inp in t.get("inputs") or []:
                st = artifact_status(arts, inp)
                if st not in ("ok", "provisional"):
                    why.append(f"input {inp}={st}")
            if not why:
                prov = [i for i in t.get("inputs") or [] if artifact_status(arts, i) == "provisional"]
                print(f"READY {t['id']}" + (f"  (provisional inputs: {', '.join(prov)})" if prov else ""))
            elif a.all:
                print(f"wait  {t['id']}: {'; '.join(why)}")
        return 0


# ----------------------------------------------------------------------------- artifacts
def split_nv(s):
    if "@" in s:
        n, v = s.rsplit("@", 1)
        return n, v
    return s, None


def artifact_status(arts, nv):
    n, v = split_nv(nv)
    cands = [x for x in arts if x["name"] == n and (v is None or str(x["version"]) == v)]
    return cands[-1]["status"] if cands else "unregistered"


def resolve_path(p: Paths, path: str) -> Path:
    q = Path(path)
    return q if q.is_absolute() else p.home / q


def file_digest(f: Path, fast: bool) -> str:
    h = hashlib.sha256()
    st = f.stat()
    if fast:
        h.update(f"{st.st_size}:{int(st.st_mtime)}".encode())
        with open(f, "rb") as fh:
            h.update(fh.read(1 << 20))
            if st.st_size > 2 << 20:
                fh.seek(-(1 << 20), 2)
                h.update(fh.read(1 << 20))
    else:
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(8 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def list_files(root: Path):
    if root.is_file():
        return [root]
    out = []
    for dp, dns, fns in os.walk(root, followlinks=True):
        dns.sort()
        out += [Path(dp) / f for f in sorted(fns)]
    return out


def manifest_checksum(root: Path, fast: bool):
    files = list_files(root)
    with ThreadPoolExecutor(8) as ex:
        digs = list(ex.map(lambda f: file_digest(f, fast), files))
    size = sum(f.stat().st_size for f in files)
    if root.is_file():
        return digs[0], size, 1
    h = hashlib.sha256()
    for f, d in zip(files, digs):
        h.update(f"{f.relative_to(root)}\t{d}\n".encode())
    return h.hexdigest(), size, len(files)


def parse_schema(items, js):
    sch = json.loads(js) if js else {}
    for kv in items or []:
        k, v = kv.split("=", 1)
        try:
            sch[k] = yaml.safe_load(v)
        except yaml.YAMLError:
            sch[k] = v
    return sch


def cmd_artifact(a):
    p = paths()
    if a.acmd == "register":
        path = resolve_path(p, a.path).resolve()
        if not path.exists():
            die(f"path {path} does not exist")
        rel = str(path.relative_to(p.home)) if path.is_relative_to(p.home) else str(path)
        if a.job:
            j = find(load(p.jobs, "jobs"), "id", a.job)
            if j is None:
                die(f"unknown job {a.job}")
            if j["status"] != "done":
                die(f"job {a.job} is {j['status']}, not done")
            commit, dirty = j["commit"], False
        else:
            tree = tree_root()
            commit = a.commit or git(["rev-parse", "HEAD"], cwd=tree)
            dirty = bool(dirty_files(tree)) if not a.commit else False
        size = sum(f.stat().st_size for f in list_files(path))
        fast = a.fast or (size > 10 * 2**30 and not a.full)
        print(f"hashing {path} ({size / 2**30:.2f} GB, mode {'fast' if fast else 'full'}) ...", file=sys.stderr)
        ck, size, nfiles = manifest_checksum(path, fast)
        fck = ck if fast else manifest_checksum(path, True)[0]
        rec = dict(name=a.name, version=str(a.version), status=a.status, path=rel, producing_commit=commit,
                   producing_job=a.job, dirty_tree=dirty, checksum=ck, checksum_mode="fast" if fast else "full",
                   fast_checksum=fck, size_gb=round(size / 2**30, 3), n_files=nfiles,
                   schema=parse_schema(a.schema, a.schema_json), notes=a.notes, created=now())
        with locked(p):
            arts = load(p.arts, "artifacts")
            old = [x for x in arts if x["name"] == a.name and str(x["version"]) == str(a.version)]
            if old and not a.force:
                die(f"{a.name}@{a.version} already registered (use a new version, or --force)")
            arts = [x for x in arts if x not in old] + [rec]
            save(p.arts, "artifacts", arts)
        print(yaml.safe_dump(rec, sort_keys=False).rstrip())
        if dirty:
            print("orch: WARNING: registered from a dirty tree (flagged by reconcile)", file=sys.stderr)
        return 0
    if a.acmd == "stale":
        with locked(p):
            arts = load(p.arts, "artifacts")
            n, v = split_nv(a.name)
            hit = [x for x in arts if x["name"] == n and (v is None or str(x["version"]) == v)]
            if not hit:
                die(f"no artifact {a.name}")
            for x in hit:
                x["status"] = "stale"
                x["stale_reason"] = f"[{now()}] {a.reason}"
            save(p.arts, "artifacts", arts)
        print(f"marked stale: {', '.join(x['name'] + '@' + x['version'] for x in hit)}")
        return 0
    if a.acmd == "set-status":
        with locked(p):
            arts = load(p.arts, "artifacts")
            n, v = split_nv(a.name)
            hit = [x for x in arts if x["name"] == n and str(x["version"]) == v]
            if not hit:
                die(f"no artifact {a.name} (give name@version)")
            hit[0]["status"] = a.status
            save(p.arts, "artifacts", arts)
        return 0
    arts = load(p.arts, "artifacts")
    if a.name:
        n, v = split_nv(a.name)
        arts = [x for x in arts if x["name"] == n and (v is None or str(x["version"]) == v)]
    if a.acmd == "show":
        if a.name:
            print(yaml.safe_dump(arts, sort_keys=False).rstrip())
        else:
            for x in arts:
                print(f"{x['name'] + '@' + x['version']:<36} {x['status']:<11} {x['size_gb']:>9} GB  "
                      f"{x['producing_commit'][:8]}  {x['checksum_mode']}:{x['checksum'][:12]}  {x['path']}")
        return 0
    if a.acmd == "verify":
        bad = 0
        for x in arts:
            path = resolve_path(p, x["path"])
            if not path.exists():
                print(f"MISSING  {x['name']}@{x['version']}  {path}")
                bad += 1
                continue
            fast = a.fast or x["checksum_mode"] == "fast"
            ref = x["fast_checksum"] if fast else x["checksum"]
            ck = manifest_checksum(path, fast)[0]
            ok = ck == ref
            bad += not ok
            print(f"{'OK      ' if ok else 'MISMATCH'} {x['name']}@{x['version']} ({'fast' if fast else 'full'})")
        return 1 if bad else 0


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="orch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("submit", help="queue a job (snapshot of the committed HEAD of the current tree)")
    s.add_argument("--id", required=True)
    s.add_argument("--owner", required=True)
    s.add_argument("--task")
    s.add_argument("--gpu", choices=("yes", "no"), required=True)
    s.add_argument("--vram-gb", type=float, default=0)
    s.add_argument("--cores", type=int, required=True)
    s.add_argument("--ram-gb", type=float, help="peak host RAM (default 2 GB/core)")
    s.add_argument("--expected-min", type=float, required=True)
    s.add_argument("--priority", type=int, default=0, help="higher runs first (critical path); default 0")
    s.add_argument("--timeout-min", type=float)
    s.add_argument("--cwd", default=".", help="repo-relative working dir inside the snapshot")
    s.add_argument("command", nargs=argparse.REMAINDER)

    s = sp.add_parser("scheduler", help="start the scheduler daemon (idempotent)")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--status", action="store_true")
    g.add_argument("--stop", action="store_true")
    g.add_argument("--foreground", action="store_true")

    s = sp.add_parser("cancel", help="kill job(s) by PGID / dequeue")
    s.add_argument("ids", nargs="+")
    s.add_argument("--reason", default="manual")

    s = sp.add_parser("prio", help="set the priority of queued job(s) (higher runs first; default 0)")
    s.add_argument("priority", type=int)
    s.add_argument("ids", nargs="+")

    s = sp.add_parser("ack", help="acknowledge failed/lost job(s) so reconcile stops flagging them")
    s.add_argument("ids", nargs="+")
    s.add_argument("--note")

    s = sp.add_parser("status", help="list jobs")
    s.add_argument("ids", nargs="*")
    s.add_argument("--running", action="store_true", help="only running/queued")
    s.add_argument("--last", type=int, default=0)
    s.add_argument("--yaml", action="store_true")

    s = sp.add_parser("wait", help="block until job(s) finish; exit 1 if any not done")
    s.add_argument("ids", nargs="+")
    s.add_argument("--timeout-min", type=float)

    s = sp.add_parser("task", help="TASKS.yaml")
    tsp = s.add_subparsers(dest="tcmd", required=True)
    for name in ("add", "set"):
        t = tsp.add_parser(name)
        t.add_argument("id")
        t.add_argument("--owner")
        t.add_argument("--status", choices=TASK_STATUS)
        t.add_argument("--depends-on", help="comma-separated task ids")
        t.add_argument("--inputs", help="comma-separated name@version")
        t.add_argument("--outputs", help="comma-separated name@version")
        t.add_argument("--done-criterion")
        t.add_argument("--notes")
        if name == "set":
            t.add_argument("--append-notes")
            t.add_argument("--add-job")
    t = tsp.add_parser("show")
    t.add_argument("id", nargs="?")
    t.add_argument("--status")
    t = tsp.add_parser("ready")
    t.add_argument("--all", action="store_true", help="also list waiting tasks and why")

    s = sp.add_parser("artifact", help="ARTIFACTS.yaml")
    asp = s.add_subparsers(dest="acmd", required=True)
    r = asp.add_parser("register")
    r.add_argument("--name", required=True)
    r.add_argument("--version", required=True)
    r.add_argument("--path", required=True, help="file or dir (repo-relative to ORCH_HOME, or absolute)")
    r.add_argument("--job", help="producing job id (commit taken from it)")
    r.add_argument("--commit", help="producing commit if not produced by an orch job")
    r.add_argument("--status", default="ok", choices=("ok", "provisional", "stale"))
    r.add_argument("--schema", nargs="*", help="key=value (YAML-parsed values)")
    r.add_argument("--schema-json")
    r.add_argument("--notes")
    r.add_argument("--fast", action="store_true", help="fast checksum (sizes+mtimes+first/last MB)")
    r.add_argument("--full", action="store_true", help="force full sha256 even if > 10 GB")
    r.add_argument("--force", action="store_true", help="overwrite an existing name@version")
    r = asp.add_parser("stale")
    r.add_argument("name", help="name or name@version")
    r.add_argument("--reason", required=True)
    r = asp.add_parser("set-status")
    r.add_argument("name", help="name@version")
    r.add_argument("status", choices=("ok", "provisional", "stale"))
    r = asp.add_parser("verify")
    r.add_argument("name", nargs="?")
    r.add_argument("--fast", action="store_true")
    r = asp.add_parser("show")
    r.add_argument("name", nargs="?")

    s = sp.add_parser("reconcile", help="consistency check; exit 1 if flags")
    s.add_argument("--no-checksum", action="store_true", help="skip fast-checksum verification of artifacts")
    s.add_argument("--no-du", action="store_true", help="skip the unregistered-big-output scan")

    a = ap.parse_args(argv)
    if a.cmd == "submit":
        if a.command and a.command[0] == "--":
            a.command = a.command[1:]
        return cmd_submit(a) or 0
    if a.cmd == "reconcile":
        sys.path.insert(0, str(HERE))
        import reconcile
        return reconcile.run(a)
    return {"scheduler": cmd_scheduler, "cancel": cmd_cancel, "ack": cmd_ack, "prio": cmd_prio, "status": cmd_status,
            "wait": cmd_wait, "task": cmd_task, "artifact": cmd_artifact}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
