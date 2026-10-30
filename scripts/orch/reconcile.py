"""orch reconcile: compare registries with reality. FLAG lines -> exit 1; NOTE lines are informational."""
from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
from pathlib import Path

import orch as O

PLACEHOLDER_RE = re.compile(r"__[A-Z0-9_]+__|\b[A-Z_]{3,}_(?:COMMIT|LOCI|SIZE|TIME|NOTES)\b|\bTODO\b|\bTBD\b")


class Report:
    def __init__(self):
        self.flags, self.notes = [], []

    def flag(self, area, msg):
        self.flags.append(f"FLAG [{area}] {msg}")

    def note(self, area, msg):
        self.notes.append(f"NOTE [{area}] {msg}")


def check_git(p, R):
    out = O.git(["worktree", "list", "--porcelain"], cwd=p.home)
    trees = [Path(l.split(" ", 1)[1]) for l in out.splitlines() if l.startswith("worktree ")]
    for t in trees:
        if not t.exists():
            R.flag("git", f"worktree {t} missing on disk (git worktree prune?)")
            continue
        bad = O.dirty_files(t)
        if bad:
            R.flag("git", f"dirty tree {t}: {len(bad)} uncommitted code change(s), e.g. {bad[0].strip()}")
    try:
        br = O.git(["branch", "--no-merged", O.MAIN_BRANCH, "--format=%(refname:short)"], cwd=p.home).split()
    except RuntimeError as e:
        R.flag("git", str(e))
        br = []
    for b in br:
        n = O.git(["rev-list", "--count", f"{O.MAIN_BRANCH}..{b}"], cwd=p.home)
        R.note("git", f"branch {b} has {n} commit(s) not merged into {O.MAIN_BRANCH}")


def check_jobs(p, R, tab):
    sched = O.scheduler_running(p)
    with O.locked(p):
        jobs = O.load(p.jobs, "jobs")
        changed = False
        for j in jobs:
            if j["status"] == "running" and not O.pid_alive(j["pid"]):
                ecf = p.home / j["run_dir"] / "exit_code"
                if ecf.exists():
                    ec = int(ecf.read_text().strip())
                    j.update(end=O.now(), exit_code=ec, status="done" if ec == 0 else "failed",
                             failure_reason=O.classify(ec, p.home / j["log"]))
                else:
                    j.update(end=O.now(), status="lost", exit_code=None,
                             failure_reason="reconcile: PID dead, no exit code (killed externally?)")
                changed = True
                R.flag("jobs", f"job {j['id']} was 'running' but PID {j['pid']} is dead -> {j['status']}")
        if changed:
            O.save(p.jobs, "jobs", jobs)
    for j in jobs:
        if j["status"] in ("failed", "lost") and not j.get("acked"):
            R.flag("jobs", f"job {j['id']} {j['status']} (owner {j['owner']}): {j.get('failure_reason')} "
                           f"-> fix/requeue, then `orch ack {j['id']}`")
        if j["status"] == "running" and j.get("expected_end"):
            over = (dt.datetime.now().astimezone() - O.parse_t(j["expected_end"])).total_seconds() / 60
            if over > max(10, 0.5 * float(j["expected_min"])):
                R.note("jobs", f"job {j['id']} is {over:.0f} min past its expected end")
        if j["status"] == "running" and j.get("peak_rss_gb") and j["peak_rss_gb"] > 1.2 * j["ram_gb"]:
            R.note("jobs", f"job {j['id']} peak RSS {j['peak_rss_gb']} GB > declared {j['ram_gb']} GB")
    if not sched:
        q = [j["id"] for j in jobs if j["status"] in ("queued", "running")]
        R.flag("sched", "scheduler NOT running" + (f"; queued/running jobs: {q}" if q else "")) if q else \
            R.note("sched", "scheduler not running (nothing queued)")
    live_pg = {int(j["pgid"]) for j in jobs if j["status"] == "running" and j.get("pgid")}
    runs = str(p.runs) + "/"
    me, uid = os.getpid(), os.getuid()
    live_ids = {j["id"] for j in jobs if j["status"] == "running"}
    for pid, v in tab.items():
        if v["uid"] != uid or pid == me or v["pgrp"] in live_pg:
            continue
        try:  # workers that left the job's pgid still carry ORCH_JOB_ID in their environment
            with open(f"/proc/{pid}/environ", "rb") as f:
                env_ids = [e.split(b"=", 1)[1].decode() for e in f.read().split(b"\0") if e.startswith(b"ORCH_JOB_ID=")]
            if env_ids and env_ids[0] in live_ids:
                continue
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
        if runs in v["cmd"] or v["cwd"].startswith(runs):
            R.flag("procs", f"orphan pid {pid} (pgid {v['pgrp']}) from runs/ not in a running job: {v['cmd'][:120]}")
        elif (v["cmd"].split(" ")[0].rsplit("/", 1)[-1].startswith(("python", "R", "Rscript"))
              and v["cwd"].startswith(str(p.home)) and "scripts/orch" not in v["cmd"]
              and pid != sched):
            R.note("procs", f"unmanaged process pid {pid} (not via orch): {v['cmd'][:120]}")
    return jobs


def check_gpu(p, R, jobs, tab):
    apps = O.gpu_apps()
    if apps is None:
        R.note("gpu", "nvidia-smi unavailable")
        return
    pg_job = {int(j["pgid"]): j for j in jobs if j["status"] == "running" and j.get("pgid")}
    use = {}
    for pid, mib in apps.items():
        v = tab.get(pid)
        j = pg_job.get(v["pgrp"]) if v else None
        if j is None and v:  # worker outside the job's pgid: match by ORCH_JOB_ID in its environment
            try:
                with open(f"/proc/{pid}/environ", "rb") as f:
                    ids = [e.split(b"=", 1)[1].decode() for e in f.read().split(b"\0") if e.startswith(b"ORCH_JOB_ID=")]
                j = next((jj for jj in pg_job.values() if ids and jj["id"] == ids[0]), None)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
        if j is None:
            who = f"uid {v['uid']}: {v['cmd'][:80]}" if v else "not visible"
            R.flag("gpu", f"GPU process pid {pid} ({mib / 1024:.1f} GB) not an orch GPU job ({who})")
        else:
            use[j["id"]] = use.get(j["id"], 0) + mib / 1024
    for jid, g in use.items():
        j = O.find(jobs, "id", jid)
        if not j["gpu"]:
            R.flag("gpu", f"job {jid} declared gpu=no but uses {g:.1f} GB")
        elif g > j["vram_gb"] + 0.75:  # CUDA context is outside the allocator cap
            R.flag("gpu", f"job {jid} uses {g:.1f} GB > declared {j['vram_gb']} GB")


def check_artifacts(p, R, a):
    arts = O.load(p.arts, "artifacts")
    for x in arts:
        nv = f"{x['name']}@{x['version']}"
        path = O.resolve_path(p, x["path"])
        if x["status"] == "stale":
            R.note("art", f"{nv} is stale: {x.get('stale_reason', '')}")
        elif x["status"] == "provisional":
            R.note("art", f"{nv} is provisional")
        if not path.exists():
            R.flag("art", f"{nv} path missing: {path}")
            continue
        if not a.no_checksum and O.manifest_checksum(path, True)[0] != x["fast_checksum"]:
            R.flag("art", f"{nv} fast checksum mismatch (modified after registration?)")
        c = x.get("producing_commit")
        r = subprocess.run(["git", "merge-base", "--is-ancestor", str(c), O.MAIN_BRANCH], cwd=p.home,
                           capture_output=True)
        if r.returncode:
            R.flag("art", f"{nv} producing commit {str(c)[:10]} is not in {O.MAIN_BRANCH}")
        if x.get("dirty_tree"):
            R.flag("art", f"{nv} was registered from a dirty tree")
    if a.no_du:
        return arts
    reg = [O.resolve_path(p, x["path"]).resolve() for x in arts]
    root = p.home / "data" / "processed"
    if root.exists():
        sizes = {}
        r = subprocess.run(["du", "-b", "--max-depth=3", str(root)], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            s, q = line.split("\t", 1)
            sizes[Path(q)] = int(s)

        def walk(d: Path):
            for c in sorted(d.iterdir()):
                cr = c.resolve()
                if any(cr == r_ or cr.is_relative_to(r_) for r_ in reg):
                    continue
                sz = sizes.get(c, c.stat().st_size if c.is_file() else 0)
                if sz <= 2**30:
                    continue
                if c.is_dir() and any(r_.is_relative_to(cr) for r_ in reg) and len(c.relative_to(root).parts) < 3:
                    walk(c)
                else:
                    R.flag("art", f"unregistered output {c.relative_to(p.home)} ({sz / 2**30:.1f} GB)")
        walk(root)
    return arts


def check_tasks(p, R, jobs, arts):
    tasks = O.load(p.tasks, "tasks")
    jd = {j["id"]: j for j in jobs}
    for t in tasks:
        tid = t["id"]
        for d in t.get("depends_on") or []:
            if O.find(tasks, "id", d) is None:
                R.flag("task", f"{tid} depends on unknown task {d}")
        if t["status"] == "running":
            live = [x for x in t.get("jobs") or [] if jd.get(x, {}).get("status") in ("queued", "running")]
            if not live:
                R.flag("task", f"{tid} is running but has no live job (jobs: {t.get('jobs') or []})")
        if t["status"] in ("running", "planned", "blocked", "done"):
            for inp in t.get("inputs") or []:
                st = O.artifact_status(arts, inp)
                if st in ("stale", "unregistered") and t["status"] in ("running", "done"):
                    R.flag("task", f"{tid} ({t['status']}) input {inp} is {st}")
                n, v = O.split_nv(inp)
                newer = [x["version"] for x in arts if x["name"] == n and x["status"] == "ok" and v and str(x["version"]) != v
                         and x["created"] > next((y["created"] for y in arts if y["name"] == n and str(y["version"]) == v), "")]
                if newer:
                    R.note("task", f"{tid} uses {inp}; newer version(s) registered: {newer}")
        if t["status"] == "done":
            for out in t.get("outputs") or []:
                st = O.artifact_status(arts, out)
                if st != "ok":
                    R.flag("task", f"{tid} is done but output {out} is {st}")


def check_reports(p, R):
    for f in sorted((p.home / "reports").glob("*.md")) if (p.home / "reports").exists() else []:
        hits = []
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            for m in PLACEHOLDER_RE.finditer(line):
                hits.append(f"L{i}:{m.group(0)}")
        if hits:
            R.flag("report", f"{f.relative_to(p.home)} has placeholders: {', '.join(hits[:5])}"
                             + (f" (+{len(hits) - 5})" if len(hits) > 5 else ""))


def run(a):
    p = O.paths()
    R = Report()
    tab = O.proc_table()
    check_git(p, R)
    jobs = check_jobs(p, R, tab)
    check_gpu(p, R, jobs, tab)
    arts = check_artifacts(p, R, a)
    check_tasks(p, R, jobs, arts)
    check_reports(p, R)
    nrun = sum(j["status"] == "running" for j in jobs)
    nq = sum(j["status"] == "queued" for j in jobs)
    print(f"# reconcile {O.now()} home={p.home}: {len(R.flags)} flag(s), {len(R.notes)} note(s); "
          f"jobs running={nrun} queued={nq}")
    for line in R.flags + R.notes:
        print(line)
    return 1 if R.flags else 0
