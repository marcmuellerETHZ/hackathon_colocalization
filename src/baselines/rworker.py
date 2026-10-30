"""Persistent R worker processes (one long-lived Rscript per library setting per Python process).

Avoids per-locus R start-up (loading coloc/susieR/mvsusieR costs 1-3 s). Requests are JSON lines on stdin;
numeric arrays travel as raw float64 files under a private temp dir (/tmp/claude-<uid>/baselines_ipc/...).
Each worker runs with single-threaded BLAS. A request that exceeds its timeout kills the worker (process
group) and the next request respawns it.

Library settings (env/BASELINES_INSTALL.md; never load both susieR versions in one R session):
    "main"     susieR 0.14.2, coloc 5.2.3, colocboost, prop.coloc, colocPropTest
    "susie016" susieR 0.16.6 (env/rlib_susie016) + coloc from the main library
    "mvsusie"  susieR 0.16.6 + mvsusieR 0.3.0
"""
from __future__ import annotations

import atexit
import json
import os
import select
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
# Paths are configurable for use outside the original project layout (release note):
#   COLOC_BENCH_RSCRIPT       Rscript with coloc 5.2.3 + susieR 0.14.2 in its default library
#                             (default: <repo>/env/renv/bin/Rscript if present, else "Rscript" on PATH)
#   COLOC_BENCH_RLIB_SUSIE016 library holding susieR 0.16.6 (default: <repo>/env/rlib_susie016)
#   COLOC_BENCH_RLIB_MVSUSIE  library holding mvsusieR 0.3.0 (default: <repo>/env/rlib_mvsusie)
#   COLOC_BENCH_TMP           parent dir of the R IPC temp files (default: the system temp dir)
_RS_DEFAULT = PROJECT / "env" / "renv" / "bin" / "Rscript"
RSCRIPT = Path(os.environ.get("COLOC_BENCH_RSCRIPT") or (_RS_DEFAULT if _RS_DEFAULT.exists() else "Rscript"))
WORKER_R = Path(__file__).resolve().parent / "r" / "worker.R"
LOG_DIR = PROJECT / "logs" / "baselines"

_LIB016 = os.environ.get("COLOC_BENCH_RLIB_SUSIE016") or str(PROJECT / "env" / "rlib_susie016")
_LIBMV = os.environ.get("COLOC_BENCH_RLIB_MVSUSIE") or str(PROJECT / "env" / "rlib_mvsusie")
LIBS = {
    "main": "",
    "susie016": _LIB016,
    "mvsusie": f"{_LIB016}:{_LIBMV}",
}


class RWorkerError(RuntimeError):
    """The R handler raised an error (message from R)."""


class RWorkerTimeout(RuntimeError):
    pass


class RWorkerDied(RuntimeError):
    pass


def _ipc_root() -> Path:
    base = os.environ.get("COLOC_BENCH_TMP") or os.path.join(tempfile.gettempdir(), f"coloc-bench-{os.getuid()}")
    root = Path(base) / "baselines_ipc"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _env(lib: str) -> dict:
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "RCPP_PARALLEL_NUM_THREADS"):
        env[k] = "1"
    env["R_PROFILE_USER"] = "/dev/null"  # BASELINES_INSTALL.md: no user profile
    env["R_LIBS"] = LIBS[lib]
    if not LIBS[lib]:
        env.pop("R_LIBS", None)
    return env


class RWorker:
    def __init__(self, lib: str = "main", max_calls: int = 2000):
        if lib not in LIBS:
            raise ValueError(f"unknown lib {lib!r}")
        self.lib = lib
        self.max_calls = max_calls
        self.proc: subprocess.Popen | None = None
        self._n = 0
        self._calls = 0
        self._tmp = Path(tempfile.mkdtemp(prefix=f"{lib}_{os.getpid()}_", dir=_ipc_root()))
        self.peak_rss_kb_seen = 0  # max VmHWM over all R processes this object spawned
        self.log_path = None

    # ------------------------------------------------------------------ process management
    def _spawn(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"rworker_{self.lib}_{os.getpid()}.log"
        logf = open(self.log_path, "ab")
        self.proc = subprocess.Popen([str(RSCRIPT), "--vanilla", str(WORKER_R)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=logf, env=_env(self.lib), cwd=str(self._tmp),
                                     start_new_session=True, bufsize=0)
        logf.close()
        self._calls = 0
        self._buf = b""

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def peak_rss_kb(self) -> int:
        """VmHWM (peak resident set) of the current R process in kB (0 if not running)."""
        if not self.alive():
            return 0
        try:
            for line in Path(f"/proc/{self.proc.pid}/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    v = int(line.split()[1])
                    self.peak_rss_kb_seen = max(self.peak_rss_kb_seen, v)
                    return v
        except FileNotFoundError:
            return 0
        return 0

    def kill(self):
        if self.proc is not None:
            self.peak_rss_kb()
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass
            for f in (self.proc.stdin, self.proc.stdout):
                try:
                    f.close()
                except Exception:
                    pass
        self.proc = None

    def close(self):
        self.close_proc()
        shutil.rmtree(self._tmp, ignore_errors=True)

    # ------------------------------------------------------------------ requests
    def _readline(self, deadline: float) -> bytes:
        while b"\n" not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RWorkerTimeout()
            r, _, _ = select.select([self.proc.stdout], [], [], remaining)
            if not r:
                raise RWorkerTimeout()
            chunk = os.read(self.proc.stdout.fileno(), 1 << 20)
            if not chunk:
                raise RWorkerDied(f"R worker ({self.lib}) exited with code {self.proc.poll()}; see {self.log_path}")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line

    def call(self, fn: str, args: dict | None = None, arrays: dict | None = None, timeout: float = 600.0) -> dict:
        """Run handler `fn`; returns the R result dict. Raises RWorkerError / RWorkerTimeout / RWorkerDied."""
        if not self.alive() or self._calls >= self.max_calls:
            if self.proc is not None:
                self.close_proc()
            self._spawn()
        self._n += 1
        self._calls += 1
        req_dir = self._tmp / f"r{self._n}"
        req_dir.mkdir()
        spec = {}
        try:
            for name, a in (arrays or {}).items():
                a = np.asarray(a, dtype="<f8")
                if a.ndim > 2:
                    raise ValueError("arrays must be 1-D or 2-D")
                f = req_dir / f"{name}.f64"
                # R reads column-major: write a.T in C order (= a in Fortran order)
                np.ascontiguousarray(a.T if a.ndim == 2 else a).tofile(f)
                spec[name] = {"file": str(f), "dim": list(a.shape)}
            req = {"id": self._n, "fn": fn, "args": args or {}, "arrays": spec}
            try:
                self.proc.stdin.write((json.dumps(req, allow_nan=False) + "\n").encode())
                self.proc.stdin.flush()
            except BrokenPipeError as e:
                self.kill()
                raise RWorkerDied(f"R worker ({self.lib}) pipe closed; see {self.log_path}") from e
            deadline = time.monotonic() + timeout
            try:
                while True:
                    line = self._readline(deadline)
                    if line.startswith(b"@@RESP "):
                        break
            except RWorkerTimeout:
                self.kill()
                raise
            except RWorkerDied:
                self.kill()
                raise
            resp = json.loads(line[7:].decode())
            if resp.get("id") != self._n:
                self.kill()
                raise RWorkerDied(f"protocol desync: expected id {self._n}, got {resp.get('id')}")
            self.peak_rss_kb()
            if not resp.get("ok"):
                raise RWorkerError(resp.get("error", "unknown R error"))
            resp["result"] = resp.get("result") or {}
            return resp
        finally:
            shutil.rmtree(req_dir, ignore_errors=True)

    def close_proc(self):
        if self.alive():
            try:
                self.proc.stdin.write(b'{"id": -1, "fn": "quit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                pass
        self.kill()
        try:  # do not leave empty logs behind
            if self.log_path is not None and self.log_path.exists() and self.log_path.stat().st_size == 0:
                self.log_path.unlink()
        except OSError:
            pass


_POOL: dict[tuple[int, str], RWorker] = {}


def get_worker(lib: str) -> RWorker:
    """Per-process persistent worker for a library setting (fork-safe: keyed by pid)."""
    key = (os.getpid(), lib)
    w = _POOL.get(key)
    if w is None:
        w = RWorker(lib)
        _POOL[key] = w
    return w


def shutdown_all():
    for (pid, _), w in list(_POOL.items()):
        if pid == os.getpid():
            w.close()
            _POOL.pop((pid, _), None)


atexit.register(shutdown_all)
