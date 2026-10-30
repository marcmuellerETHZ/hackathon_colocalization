"""Run harness (DESIGN §7.2, §7.5, §7.7; INTERFACES §3-4).

run_method(method, locus_set, ...) runs a Method over a locus set and writes
    results/<run_id>/manifest.json, predictions.parquet, metrics.json
and appends one line to results/LEDGER.tsv (append-only, flock-protected).

Execution modes
  * "cpu":   persistent worker processes (spawn), one locus per task, per-locus timeout. A timed-out or crashed
             worker is killed with its whole process group (R child processes included) and replaced.
  * "batch": GPU/batched methods: tasks of `batch_size` loci (sorted by p so batches are p-homogeneous),
             usually n_workers=1; timeout = timeout_s * len(batch); runtime_s = batch wall / len(batch).
Every task's wall time includes reading the locus (and its LD) from the HDF5 file inside the worker.

Machine lock: shared flock on results/.machine.lock for every run; timing runs take it EXCLUSIVELY (waiting for
all other runs to finish) plus results/.timing.lock, and record machine load at start.

Method: an object with `name` and `fit_batch(list[LocusData]) -> list[dict]` (INTERFACES §4), or a MethodSpec
("module:Class", kwargs) built inside each worker. Instances are pickled to workers (class must be importable).
"""
from __future__ import annotations

import collections
import contextlib
import fcntl
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from multiprocessing.connection import wait as mp_wait
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from common.seeds import SEEDS
from eval import metrics as M
from eval.io import PROJECT_ROOT, LocusSetReader, has_test_access, read_locus_attrs, truth_path_for

PRED_COLUMNS = ["locus_id", "method", "s", "P_S", "P_D", "P_W", "E1", "E2", "call", "status", "runtime_s", "extra"]
STATUSES = ("ok", "nonconverged", "failed", "timeout")
COMMIT_LIMIT_BYTES = 5 * 1024 * 1024  # D6(h): predictions.parquet is committed only below 5 MB
LEDGER_COLUMNS = ["run_id", "start_utc", "end_utc", "method", "locus_set", "n_expected", "n_ok", "n_nonconverged",
                  "n_failed", "n_timeout", "git_commit", "git_clean", "timing", "label", "run_dir"]


# --------------------------------------------------------------------------------------------------
# Method specs
# --------------------------------------------------------------------------------------------------
@dataclass
class MethodSpec:
    target: str  # "package.module:ClassName"
    kwargs: dict = field(default_factory=dict)
    name: str | None = None  # default: the class attribute `name` (read without instantiating)

    def cls(self):
        mod, _, cls = self.target.partition(":")
        return getattr(importlib.import_module(mod), cls)

    def build(self):
        return self.cls()(**self.kwargs)

    def method_name(self) -> str:
        return self.name or self.cls().name


def _describe(method) -> dict:
    if isinstance(method, MethodSpec):
        return {"target": method.target, "kwargs": method.kwargs, "name": method.method_name()}
    cls = type(method)
    params = {k: v for k, v in vars(method).items() if not k.startswith("_")} if hasattr(method, "__dict__") else {}
    return {"target": f"{cls.__module__}:{cls.__qualname__}", "params": json.loads(json.dumps(params, default=repr))}


# --------------------------------------------------------------------------------------------------
# Worker process
# --------------------------------------------------------------------------------------------------
def _worker_main(conn, method, locus_set: str, project_root: str):
    os.setsid()  # own process group: the parent can kill the worker together with any R children
    import eval.io as _eio
    _eio.PROJECT_ROOT = Path(project_root)  # same root (splits file, access log) as the parent
    try:
        m = method.build() if isinstance(method, MethodSpec) else method
    except Exception:
        conn.send(("init_error", None, traceback.format_exc(), 0.0))
        return
    reader = None
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:
            break
        task_id, ids = msg
        t0 = time.perf_counter()
        try:
            if reader is None:
                reader = LocusSetReader(locus_set)
            loci = [reader.read(i, load_ld=bool(getattr(m, "needs_ld", True)),
                                load_ld2=bool(getattr(m, "needs_ld2", False))) for i in ids]
            outs = m.fit_batch(loci)
            if "torch" in sys.modules:
                torch = sys.modules["torch"]
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    torch.cuda.synchronize()
            err = None
        except Exception:
            outs, err = None, traceback.format_exc()
        conn.send((task_id, outs, err, time.perf_counter() - t0))
    if reader is not None:
        reader.close()


class _Worker:
    def __init__(self, ctx, method, locus_set):
        self.parent, child = ctx.Pipe(duplex=True)
        import eval.io as _eio
        self.proc = ctx.Process(target=_worker_main, args=(child, method, str(locus_set), str(_eio.PROJECT_ROOT)),
                                daemon=True)
        self.proc.start()
        child.close()
        self.task = None  # (task_id, ids)
        self.deadline = None

    def send(self, task, timeout):
        self.task = task
        self.deadline = time.monotonic() + timeout
        self.parent.send(task)

    def kill(self):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.proc.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            self.proc.kill()
        self.proc.join(timeout=10)
        self.parent.close()

    def stop(self):
        with contextlib.suppress(Exception):
            self.parent.send(None)
        self.proc.join(timeout=30)
        if self.proc.is_alive():
            self.kill()


def _row(lid, method_name, status, runtime, extra) -> dict:
    return {"locus_id": lid, "method": method_name, "s": np.nan, "P_S": np.nan, "P_D": np.nan, "P_W": np.nan,
            "E1": np.nan, "E2": np.nan, "call": "NA", "status": status, "runtime_s": runtime,
            "extra": json.dumps(extra)}


def _normalise_outputs(ids, outs, method_name, per_locus_rt) -> list[dict]:
    rows = {}
    if not isinstance(outs, list):
        return [_row(i, method_name, "failed", per_locus_rt, {"harness_error": "fit_batch did not return a list"})
                for i in ids]
    for k, o in enumerate(outs):
        o = dict(o)
        lid = str(o.get("locus_id", ids[k] if k < len(ids) else ""))
        if lid not in ids:
            continue
        r = _row(lid, method_name, str(o.get("status", "ok")), per_locus_rt, {})
        for c in ("s", "P_S", "P_D", "P_W", "E1", "E2"):
            if o.get(c) is not None:
                r[c] = float(o[c])
        if r["status"] not in STATUSES:
            r["extra"] = json.dumps({"harness_error": f"invalid status {r['status']!r}"})
            r["status"] = "failed"
        ex = o.get("extra", "")
        if r["extra"] == "{}":
            r["extra"] = ex if isinstance(ex, str) else json.dumps(ex, default=repr)
        if "runtime_s" in o and o["runtime_s"] is not None:
            # keep the harness wall time (includes LD loading); method-reported time is kept in extra
            try:
                exd = json.loads(r["extra"]) if r["extra"] else {}
                if isinstance(exd, dict):
                    exd["method_runtime_s"] = float(o["runtime_s"])
                    r["extra"] = json.dumps(exd)
            except json.JSONDecodeError:
                pass
        rows[lid] = r
    for lid in ids:
        if lid not in rows:
            rows[lid] = _row(lid, method_name, "failed", per_locus_rt, {"harness_error": "no output row"})
    return [rows[i] for i in ids]


def _add_extra(row: dict, key: str, value) -> dict:
    try:
        ex = json.loads(row["extra"]) if row.get("extra") else {}
    except (json.JSONDecodeError, TypeError):
        ex = {"method_extra": row.get("extra")}
    if not isinstance(ex, dict):
        ex = {"method_extra": ex}
    ex[key] = value
    row["extra"] = json.dumps(ex, default=repr)
    return row


def execute(method, locus_set: str | Path, ids: list[str], method_name: str, mode: str = "cpu",
            n_workers: int = 8, batch_size: int = 64, timeout_s: float = 600.0, sort_keys: dict | None = None,
            log: Callable[[str], None] = print, order: str = "given", retry_batches: bool = True,
            on_rows: Callable[[list[dict]], None] | None = None) -> list[dict]:
    """Run `method` over loci `ids`; returns one prediction row per id (order of `ids`).

    order: "given" (cpu tasks in the order of `ids`) or "lpt" (cpu tasks by descending sort key, i.e. largest p
    first: longest-processing-time-first list scheduling, which shortens the tail of a run). Batch mode always
    sorts by ascending sort key (p-homogeneous batches).
    retry_batches: in batch mode, a task of more than one locus that crashes, times out, kills its worker or
    returns only failed rows is re-queued once as single-locus tasks (a batch-level failure such as a GPU OOM
    must not be scored as a failure of every locus in the batch); the rows of the retry carry
    extra.harness_batch_retry = the batch failure. A single locus is never retried.
    on_rows(rows): called with the final rows of every finished task (checkpointing)."""
    if mode not in ("cpu", "batch"):
        raise ValueError(mode)
    sk = sort_keys or {}
    if mode == "cpu":
        seq = sorted(ids, key=lambda i: -float(sk.get(i, 0))) if order == "lpt" else list(ids)
        tasks = [[i] for i in seq]
    else:
        srt = sorted(ids, key=lambda i: sk.get(i, 0))
        tasks = [srt[k:k + batch_size] for k in range(0, len(srt), batch_size)]
    if not tasks:
        return []
    ctx = mp.get_context("spawn")
    pending = collections.deque(enumerate(tasks))
    next_tid = len(tasks)
    retry_reason: dict[str, str] = {}
    n_workers = max(1, min(n_workers, len(tasks)))
    workers = [_Worker(ctx, method, locus_set) for _ in range(n_workers)]
    results: dict[str, dict] = {}
    done_tasks, last_log = 0, time.monotonic()
    init_errors = 0

    def finish(rows: list[dict]):
        for r in rows:
            if r["locus_id"] in retry_reason:
                _add_extra(r, "harness_batch_retry", retry_reason[r["locus_id"]])
            results[r["locus_id"]] = r
        if on_rows is not None:
            on_rows(rows)

    def maybe_retry(tids, reason) -> bool:
        nonlocal next_tid
        if not (retry_batches and mode == "batch" and len(tids) > 1):
            return False
        for lid in tids:
            retry_reason[lid] = reason[:500]
            pending.append((next_tid, [lid]))
            next_tid += 1
        log(f"[harness] batch of {len(tids)} failed ({reason[:120]!r}); re-queued as single-locus tasks")
        return True
    try:
        while pending or any(w.task is not None for w in workers):
            for w in workers:
                if w.task is None and pending:
                    tid, tids = pending.popleft()
                    w.send((tid, tids), timeout_s * len(tids))
            busy = [w for w in workers if w.task is not None]
            now = time.monotonic()
            wait_t = max(0.0, min(w.deadline for w in busy) - now) if busy else 0.0
            ready = mp_wait([w.parent for w in busy], timeout=wait_t)
            for i, w in enumerate(workers):
                if w.task is None:
                    continue
                tid, tids = w.task
                replace = False
                if w.parent in ready:
                    try:
                        msg = w.parent.recv()
                    except (EOFError, OSError):
                        msg = None
                    if msg is None:  # worker died (segfault, OOM kill, ...)
                        if not maybe_retry(tids, "worker died"):
                            finish([_row(lid, method_name, "failed", np.nan, {"harness_error": "worker died"})
                                    for lid in tids])
                        done_tasks += 1
                        replace = True
                    elif msg[0] == "init_error":
                        init_errors += 1
                        if init_errors > 3:
                            raise RuntimeError(f"method failed to initialise in worker:\n{msg[2]}")
                        pending.appendleft((tid, tids))
                        replace = True
                    else:
                        _, outs, err, dt = msg
                        per = dt / len(tids)
                        if err is not None:
                            if not maybe_retry(tids, f"fit_batch raised: {err[-300:]}"):
                                finish([_row(lid, method_name, "failed", per, {"error": err[-2000:]}) for lid in tids])
                        else:
                            rows = _normalise_outputs(tids, outs, method_name, per)
                            if all(r["status"] == "failed" for r in rows) and maybe_retry(
                                    tids, "every row of the batch failed: " + str(rows[0].get("extra", ""))[:300]):
                                pass
                            else:
                                finish(rows)
                        done_tasks += 1
                        w.task = None
                elif time.monotonic() > w.deadline:
                    if not maybe_retry(tids, f"batch timeout ({timeout_s:g} s x {len(tids)})"):
                        finish([_row(lid, method_name, "timeout", timeout_s, {"timeout_s": timeout_s}) for lid in tids])
                    done_tasks += 1
                    replace = True
                if replace:
                    w.task = None
                    w.kill()
                    workers[i] = _Worker(ctx, method, locus_set)
            if time.monotonic() - last_log > 60:
                log(f"[harness] {method_name}: {len(results)}/{len(ids)} loci done")
                last_log = time.monotonic()
    finally:
        for w in workers:
            w.stop()
    return [results[i] for i in ids]


# --------------------------------------------------------------------------------------------------
# Provenance helpers
# --------------------------------------------------------------------------------------------------
def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def git_info(repo: Path = PROJECT_ROOT) -> dict:
    from eval.io import code_identity, in_orch_snapshot
    if in_orch_snapshot(repo):
        # orch job: the code is a `git archive` of a committed HEAD (clean by construction); the main
        # checkout's HEAD/status (what plain git commands would see from runs/<id>/code) is not the code that runs
        home, commit = code_identity(repo)
        tags = subprocess.run(["git", "tag", "--points-at", commit], cwd=home, capture_output=True, text=True,
                              check=True).stdout.split()
        return {"commit": commit, "clean": True, "clean_basis": "orch snapshot of a committed tree",
                "snapshot": str(repo), "orch_job": os.environ.get("ORCH_JOB_ID"), "modified_tracked": [],
                "untracked_code": [], "tags": tags}

    def g(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True).stdout
    commit = g("rev-parse", "HEAD").strip()
    tracked = [l for l in g("status", "--porcelain", "--untracked-files=no").splitlines() if l.strip()]
    untracked_code = [l[3:] for l in g("status", "--porcelain", "--untracked-files=all", "--", "src", "scripts",
                                        "datasets/configs").splitlines() if l.startswith("??")]
    tags = g("tag", "--points-at", "HEAD").split()
    return {"commit": commit, "clean": not tracked and not untracked_code, "modified_tracked": tracked[:50],
            "untracked_code": untracked_code[:50], "tags": tags}


def env_lock_hash(repo: Path = PROJECT_ROOT) -> dict:
    files = ["env/requirements.lock", "env/renv.lock", "env/R_packages_pinned.tsv"]
    out, h = {}, hashlib.sha256()
    for f in files:
        p = repo / f
        if p.exists():
            d = sha256_file(p)
            out[f] = d
            h.update(f"{f}:{d}\n".encode())
        else:
            out[f] = None
    out["combined"] = h.hexdigest()
    return out


def machine_state() -> dict:
    st = {"hostname": socket.gethostname(), "loadavg": list(os.getloadavg()), "cpu_affinity": len(os.sched_getaffinity(0))}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        st["gpu"] = q.stdout.strip() if q.returncode == 0 else f"nvidia-smi rc={q.returncode}"
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        st["gpu"] = f"unavailable: {e}"
    return st


@contextlib.contextmanager
def machine_lock(results_dir: Path, exclusive: bool, log: Callable[[str], None] = print):
    results_dir.mkdir(parents=True, exist_ok=True)
    files = [results_dir / ".machine.lock"] + ([results_dir / ".timing.lock"] if exclusive else [])
    handles = []
    try:
        for f in (reversed(files) if exclusive else files):  # timing lock first, then machine lock
            fh = open(f, "a")
            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            try:
                fcntl.flock(fh, mode | fcntl.LOCK_NB)
            except BlockingIOError:
                log(f"[harness] waiting for {'exclusive' if exclusive else 'shared'} lock on {f}")
                fcntl.flock(fh, mode)
            handles.append(fh)
        yield
    finally:
        for fh in handles:
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()


def append_ledger(results_dir: Path, row: dict) -> None:
    ledger = results_dir / "LEDGER.tsv"
    with open(ledger, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            empty = fh.read(1) == ""
            if empty:
                fh.write("\t".join(LEDGER_COLUMNS) + "\n")
            vals = [str(row.get(c, "")).replace("\t", " ").replace("\n", " ") for c in LEDGER_COLUMNS]
            fh.write("\t".join(vals) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def results_policy(run_dir: Path, limit: int = COMMIT_LIMIT_BYTES) -> dict:
    """Decision D6(h): manifest.json and metrics.json are always committed; predictions.parquet only when smaller
    than 5 MB. A larger file stays under results/ but is git-ignored by a run-local .gitignore (committed with the
    manifest); its sha256 is in the manifest, so scripts/verify_run.py can check a restored copy."""
    size = (run_dir / "predictions.parquet").stat().st_size
    commit = size < limit
    if not commit:
        (run_dir / ".gitignore").write_text("# D6(h): predictions >= 5 MB are not committed (sha256 in manifest.json)\n"
                                            "predictions.parquet\n")
    return {"rule": "D6(h)", "predictions_size_bytes": int(size), "predictions_committed": bool(commit),
            "commit_limit_bytes": int(limit)}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def fill_calls(pred: pd.DataFrame, t_S: float, t_D: float, e_gate: float = M.E_GATE) -> pd.Series:
    return pd.Series(M.calls_from_arrays(pred["P_S"].to_numpy(float), pred["P_D"].to_numpy(float),
                                         pred["E1"].to_numpy(float), pred["E2"].to_numpy(float), t_S, t_D, e_gate,
                                         failed=~pred["status"].isin(["ok", "nonconverged"]).to_numpy()),
                     index=pred.index)


# --------------------------------------------------------------------------------------------------
# Checkpoints (resumable runs)
# --------------------------------------------------------------------------------------------------
def load_checkpoint(path: Path, ids: set) -> dict[str, dict]:
    """Rows of a checkpoint file (last row per locus wins); a torn last line (crash mid-write) is ignored."""
    out: dict[str, dict] = {}
    if path is None or not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(r, dict) or str(r.get("locus_id")) not in ids:
            continue
        r = {c: r.get(c) for c in PRED_COLUMNS}
        for c in ("s", "P_S", "P_D", "P_W", "E1", "E2", "runtime_s"):
            r[c] = np.nan if r[c] is None else float(r[c])
        r["locus_id"] = str(r["locus_id"])
        out[r["locus_id"]] = r
    return out


class _CheckpointWriter:
    def __init__(self, path: Path):
        self.fh = open(path, "a")
        self.last_sync = time.monotonic()

    def write(self, rows: list[dict]) -> None:
        for r in rows:
            rr = {c: r.get(c) for c in PRED_COLUMNS}
            for c in ("s", "P_S", "P_D", "P_W", "E1", "E2", "runtime_s"):
                v = rr[c]
                rr[c] = None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)
            self.fh.write(json.dumps(rr, default=str) + "\n")
        self.fh.flush()
        if time.monotonic() - self.last_sync > 5:
            os.fsync(self.fh.fileno())
            self.last_sync = time.monotonic()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.fh.flush()
            os.fsync(self.fh.fileno())
        self.fh.close()


# --------------------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------------------
@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    counts: dict
    metrics: dict | None


def run_method(method: Any, locus_set: str | Path, *, results_dir: str | Path | None = None, run_id: str | None = None,
               mode: str = "cpu", n_workers: int = 8, batch_size: int = 64, timeout_s: float = 600.0,
               timing: bool = False, ids: list[str] | None = None, truth_path: str | Path | None | str = "auto",
               config: dict | None = None, metric_config: dict | None = None, seeds_used: dict | None = None,
               label: str = "", hash_inputs: bool = True, log: Callable[[str], None] = print,
               checkpoint_dir: str | Path | None = None, order: str = "given",
               retry_batches: bool = True) -> RunResult:
    """Run a Method over a locus set and record everything (manifest, predictions, metrics, ledger).

    checkpoint_dir (resumable runs; give a fixed run_id): every finished task's rows are appended to
    <checkpoint_dir>/<run_id>.rows.jsonl (flushed per task, fsync at most every 5 s). A later call with the same
    run_id (after a crash/kill of the caller) loads those rows and executes only the remaining ids; the manifest
    lists every segment (<run_id>.segments.jsonl) and the number of rows taken from the checkpoint. The run
    directory itself is written only when every id has a row, so "run_dir/manifest.json exists" = finished.
    order / retry_batches: see execute()."""
    if os.environ.get("COLOC_ENV_ACTIVE") != "1":
        raise RuntimeError("source env/activate.sh first (DESIGN §7.5)")
    locus_set = Path(locus_set).resolve()
    results_dir = Path(results_dir) if results_dir is not None else PROJECT_ROOT / "results"
    method_name = method.method_name() if isinstance(method, MethodSpec) else str(getattr(method, "name"))
    attrs = read_locus_attrs(locus_set)  # applies the test-split guard to every locus
    all_ids = attrs["locus_id"].astype(str).tolist()
    if ids is None:
        ids = all_ids
    else:
        missing = sorted(set(ids) - set(all_ids))
        if missing:
            raise ValueError(f"{len(missing)} requested ids not in locus set, e.g. {missing[:3]}")
        ids = list(ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids")
    p_by_id = dict(zip(attrs["locus_id"].astype(str), attrs["p"]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_id or f"{stamp}_{method_name}_{locus_set.stem}_{uuid.uuid4().hex[:6]}"
    run_dir = results_dir / run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    if truth_path == "auto":
        tp = truth_path_for(locus_set)
        truth_path = tp if tp.exists() else None
    mcfg = M.MetricConfig.from_dict(metric_config)
    ck_rows: dict[str, dict] = {}
    ck_file = seg_file = None
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ck_file = checkpoint_dir / f"{run_id}.rows.jsonl"
        seg_file = checkpoint_dir / f"{run_id}.segments.jsonl"
        ck_rows = load_checkpoint(ck_file, set(ids))
    todo = [i for i in ids if i not in ck_rows]

    with machine_lock(results_dir, exclusive=timing, log=log):
        start = _now()
        t0 = time.perf_counter()
        segments = []
        if seg_file is not None:
            seg = {"start_utc": start, "pid": os.getpid(), "host": socket.gethostname(),
                   "rows_from_checkpoint": len(ck_rows), "n_todo": len(todo), "git": git_info()["commit"]}
            with open(seg_file, "a") as fh:
                fh.write(json.dumps(seg) + "\n")
            segments = [json.loads(l) for l in seg_file.read_text().splitlines() if l.strip()]
            if ck_rows:
                log(f"[harness] {run_id}: resuming, {len(ck_rows)} rows from checkpoint, {len(todo)} to run")
        manifest = {
            "run_id": run_id, "label": label, "method_name": method_name, "method": _describe(method),
            "locus_set": {"path": str(locus_set), "sha256": sha256_file(locus_set) if hash_inputs else None,
                          "n_loci_in_file": len(all_ids), "n_expected": len(ids),
                          "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()},
            "truth": None, "git": git_info(), "env_lock_sha256": env_lock_hash(),
            "config": config or {}, "metric_config": mcfg.to_dict(),
            "seeds": {"SEEDS": SEEDS, "used": seeds_used or {}},
            "harness": {"mode": mode, "n_workers": n_workers, "batch_size": batch_size if mode == "batch" else 1,
                        "timeout_s_per_locus": timeout_s, "timing": timing,
                        "lock": "exclusive" if timing else "shared"},
            "machine_start": machine_state(), "python": sys.version, "platform": platform.platform(),
            "test_access": has_test_access(), "start_utc": start,
        }
        if truth_path is not None:
            manifest["truth"] = {"path": str(Path(truth_path).resolve()),
                                 "sha256": sha256_file(truth_path) if hash_inputs else None}
        log(f"[harness] run {run_id}: {len(ids)} loci ({len(todo)} to run), mode={mode}, workers={n_workers}")
        writer = _CheckpointWriter(ck_file) if ck_file is not None else None
        try:
            new_rows = execute(method, locus_set, todo, method_name, mode=mode, n_workers=n_workers,
                               batch_size=batch_size, timeout_s=timeout_s, sort_keys=p_by_id, log=log, order=order,
                               retry_batches=retry_batches, on_rows=writer.write if writer else None)
        finally:
            if writer is not None:
                writer.close()
        by_id = dict(ck_rows)
        by_id.update({r["locus_id"]: r for r in new_rows})
        rows = [by_id[i] for i in ids]
        pred = pd.DataFrame(rows, columns=PRED_COLUMNS)
        pred["call"] = fill_calls(pred, mcfg.t_S, mcfg.t_D, mcfg.e_gate)
        wall = time.perf_counter() - t0
        end = _now()
        counts = {s: int((pred["status"] == s).sum()) for s in STATUSES}
        counts["expected"] = len(ids)

        run_dir.mkdir(parents=True)
        pred_path = run_dir / "predictions.parquet"
        pred.to_parquet(pred_path, index=False)
        policy = results_policy(run_dir)
        metrics = None
        if truth_path is not None:
            from eval.truth import read_truth  # harness-only import; methods never see truth
            truth = read_truth(truth_path)
            truth = truth[truth["locus_id"].astype(str).isin(set(ids))]
            metrics = M.compute_run_metrics(truth, pred, mcfg)
            (run_dir / "metrics.json").write_text(M.dumps(metrics))
        if ck_file is not None:
            manifest["checkpoint"] = {"rows_file": str(ck_file), "rows_from_checkpoint": len(ck_rows),
                                      "rows_this_segment": len(todo), "segments": segments,
                                      "resumed": len(segments) > 1 or bool(ck_rows)}
            manifest["start_utc"] = segments[0]["start_utc"] if segments else start
        manifest.update({
            "end_utc": end, "wall_s": wall, "counts": counts,
            "runtime_s": {"sum": float(np.nansum(pred["runtime_s"])),
                          "median": float(np.nanmedian(pred["runtime_s"])) if len(pred) else None,
                          "p95": float(np.nanpercentile(pred["runtime_s"], 95)) if len(pred) else None},
            "predictions_sha256": sha256_file(pred_path), "results_policy": policy,
            "metrics_sha256": sha256_file(run_dir / "metrics.json") if metrics is not None else None,
        })
        if timing:
            manifest["machine_end"] = machine_state()
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str))
        append_ledger(results_dir, {
            "run_id": run_id, "start_utc": start, "end_utc": end, "method": method_name, "locus_set": str(locus_set),
            "n_expected": len(ids), "n_ok": counts["ok"], "n_nonconverged": counts["nonconverged"],
            "n_failed": counts["failed"], "n_timeout": counts["timeout"], "git_commit": manifest["git"]["commit"],
            "git_clean": manifest["git"]["clean"], "timing": timing, "label": label, "run_dir": str(run_dir)})
        log(f"[harness] done {run_id}: {counts} in {wall:.1f}s")
    return RunResult(run_id=run_id, run_dir=run_dir, counts=counts, metrics=metrics)
