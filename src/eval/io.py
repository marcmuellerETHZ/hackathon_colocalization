"""Locus-set I/O (INTERFACES §2) and the test-split guard (DESIGN §7.3).

A locus set is ONE HDF5 file with one group per locus (`/<locus_id>/`). Groups whose name starts with
"_" are reserved (e.g. `/_ld/<key>` holds LD shared by several loci through HDF5 hard links; readers see
the linked dataset as the locus' own `R` / `X_ref`, so the on-disk format is still INTERFACES §2).

Test-split guard: every reader and writer here refuses loci whose `split` attribute is "test", or whose
`block_ids` (or window) touch a test-split LD block of `datasets/splits/blocks_v1.tsv` (read lazily; falls back
to `datasets/splits/split_v1.tsv`). Access is granted only inside a test run started by `scripts/eval_test.py`,
which writes a nonce to `results/test_access.log` and exports it as $COLOC_TEST_ACCESS (see
`has_test_access`).
"""
from __future__ import annotations

import functools
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARRAY_FIELDS_F64 = ("z1", "z2", "beta1", "se1", "beta2", "se2", "n1", "n2", "maf1", "maf2", "maf_ref")
LOCUS_ATTRS = ("rho_hat", "rho_se", "split", "block_ids", "mismatch_level", "pairing_type", "cell_id",
               "window_chr", "window_start", "window_end")
TEST_SPLIT = "test"
ACCESS_ENV = "COLOC_TEST_ACCESS"
# Per-testset access logs (DESIGN_P2 §6, M8): (log path relative to the repo, required frozen-tag prefix).
# v1 = DESIGN §7.3, unchanged (its log, its `frozen-*` rule). v2 has its own log and needs a `frozen-v2*` tag.
# A nonce grants access only through the log it was started in, and only at a commit carrying that log's tag.
ACCESS_LOGS = (("results/test_access.log", "frozen-"),
               ("results/test_access_v2.log", "frozen-v2"))


class TestSplitAccessError(PermissionError):
    """Raised when a loader touches test-split data outside an authorised test run."""

    __test__ = False  # not a pytest test class


# --------------------------------------------------------------------------------------------------
# LocusData
# --------------------------------------------------------------------------------------------------
@dataclass
class LocusData:
    """One locus pair, mirroring INTERFACES §2. Arrays have length p (number of variants)."""

    locus_id: str
    variant_id: np.ndarray
    pos: np.ndarray
    z1: np.ndarray
    z2: np.ndarray
    beta1: np.ndarray
    se1: np.ndarray
    beta2: np.ndarray
    se2: np.ndarray
    n1: np.ndarray
    n2: np.ndarray
    maf1: np.ndarray
    maf2: np.ndarray
    maf_ref: np.ndarray
    ld_type: str  # "dense" | "factor"
    n_ref: int
    R: np.ndarray | None = None  # float32 [p, p] if dense
    X_ref: np.ndarray | None = None  # float32 [n_ref, p] standardized, if factor (R = X'X / n_ref)
    rho_hat: float = float("nan")
    rho_se: float = float("nan")
    split: str = ""
    block_ids: tuple = ()
    mismatch_level: str = ""
    pairing_type: str = ""
    cell_id: str = ""
    window_chr: str = ""
    window_start: int = -1
    window_end: int = -1
    extra_attrs: dict = field(default_factory=dict)  # any further locus attributes, passed through
    ld_key: str | None = None  # writer only: if set, LD is stored once under /_ld/<ld_key> and hard-linked
    # trait-2 in-sample LD where the input provides one (sim L0: datasets R2 / X_ref2, attrs ld_type2 / n_ref2 in
    # extra_attrs; P3-5 item 2). Read only on request (LocusSetReader.read(load_ld2=True); the harness asks for
    # it when the method sets `needs_ld2 = True`). None otherwise.
    R2: np.ndarray | None = None
    X_ref2: np.ndarray | None = None

    @property
    def p(self) -> int:
        return int(len(self.z1))

    def ld(self, dtype=np.float64) -> np.ndarray:
        """Dense inference LD (p x p). For factor form computes X'X / n_ref."""
        if self.ld_type == "dense":
            return np.asarray(self.R, dtype=dtype)
        X = np.asarray(self.X_ref, dtype=dtype)
        return X.T @ X / float(self.n_ref)

    def ld_rows(self, idx, dtype=np.float64) -> np.ndarray:
        """Rows `idx` of the inference LD (len(idx) x p) without forming the full matrix for factor form."""
        idx = np.atleast_1d(np.asarray(idx))
        if self.ld_type == "dense":
            return np.asarray(self.R[idx], dtype=dtype)
        X = np.asarray(self.X_ref, dtype=dtype)
        return X[:, idx].T @ X / float(self.n_ref)

    def validate(self) -> None:
        p = self.p
        for name in ("variant_id", "pos") + ARRAY_FIELDS_F64:
            a = getattr(self, name)
            if a is None or len(a) != p:
                raise ValueError(f"{self.locus_id}: field {name} has length {None if a is None else len(a)} != p={p}")
        if self.ld_type == "dense":
            if self.R is None or self.R.shape != (p, p):
                raise ValueError(f"{self.locus_id}: dense LD must be [p,p]")
        elif self.ld_type == "factor":
            if self.X_ref is None or self.X_ref.ndim != 2 or self.X_ref.shape[1] != p:
                raise ValueError(f"{self.locus_id}: factor LD X_ref must be [n_ref,p]")
            if int(self.n_ref) != self.X_ref.shape[0]:
                raise ValueError(f"{self.locus_id}: n_ref={self.n_ref} != X_ref rows {self.X_ref.shape[0]}")
        else:
            raise ValueError(f"{self.locus_id}: ld_type must be 'dense' or 'factor', got {self.ld_type!r}")
        if self.locus_id.startswith("_") or "/" in self.locus_id:
            raise ValueError(f"invalid locus_id {self.locus_id!r} (no leading '_' and no '/')")


# --------------------------------------------------------------------------------------------------
# Test-split guard
# --------------------------------------------------------------------------------------------------
def _splits_file() -> Path | None:
    for name in ("blocks_v1.tsv", "split_v1.tsv"):
        f = PROJECT_ROOT / "datasets" / "splits" / name
        if f.exists():
            return f
    return None


@functools.lru_cache(maxsize=4)
def _test_blocks_cached(path: str, mtime: float) -> tuple[frozenset, pd.DataFrame | None]:
    df = pd.read_csv(path, sep="\t", dtype=str)
    bcol = next((c for c in ("block_id", "block") if c in df.columns), None)
    if bcol is None or "split" not in df.columns:
        raise ValueError(f"{path}: expected columns block_id (or block) and split; got {list(df.columns)}")
    test = df[df["split"].str.lower() == TEST_SPLIT]
    ids = frozenset(test[bcol].astype(str))
    coords = None
    if {"chr", "start", "end"}.issubset(df.columns):
        coords = test[["chr", "start", "end"]].copy()
        coords["start"] = coords["start"].astype(np.int64)
        coords["end"] = coords["end"].astype(np.int64)
    return ids, coords


def get_test_blocks() -> tuple[frozenset, pd.DataFrame | None]:
    """(test block ids, test block coordinates or None). Empty if the splits file does not exist yet."""
    f = _splits_file()
    if f is None:
        return frozenset(), None
    return _test_blocks_cached(str(f), f.stat().st_mtime)


def _norm_chr(c) -> str:
    c = str(c)
    return c if c.startswith("chr") else f"chr{c}"


def is_test_locus(split: str = "", block_ids: Iterable = (), window_chr: str = "", window_start: int = -1,
                  window_end: int = -1) -> bool:
    if str(split).lower() == TEST_SPLIT:
        return True
    ids, coords = get_test_blocks()
    if ids and any(str(b) in ids for b in block_ids):
        return True
    if coords is not None and window_chr and window_start >= 0 and window_end >= 0:
        c = coords[coords["chr"].map(_norm_chr) == _norm_chr(window_chr)]
        # half-open intervals [start, end)
        if ((c["start"] < window_end) & (c["end"] > window_start)).any():
            return True
    return False


def _git(*args, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def has_test_access(repo: Path | None = None) -> bool:
    """True only inside an authorised test run.

    Conditions: $COLOC_TEST_ACCESS holds a nonce that appears in a `started` entry of
    `results/test_access.log` (working tree) with no matching `completed`/`aborted` entry, and that entry's
    commit equals the commit of the running code (HEAD; inside an orch job the snapshot commit $ORCH_COMMIT,
    see code_identity), which carries a `frozen-*` tag. (`interrupted`/`resumed` entries keep the nonce open.) Written by scripts/eval_test.py only.
    Positive answers are cached per (repo, nonce, sha256 of the log's CONTENT) so per-locus checks do not call
    git each time; negative answers are never cached (a later tag or log entry is seen immediately). The key is
    the content, not the mtime: the cluster scratch filesystem has 1-second mtime resolution, so an mtime key kept a grant
    alive after a `completed` line appended within the same second (review p5_verify_infra, defect 4).
    Per testset (DESIGN_P2 §6): the same rule is applied to each log of ACCESS_LOGS with that log's tag prefix
    (v1: results/test_access.log, `frozen-*`, unchanged; v2: results/test_access_v2.log, `frozen-v2*`).
    """
    repo = PROJECT_ROOT if repo is None else Path(repo)
    nonce = os.environ.get(ACCESS_ENV, "")
    if not nonce:
        return False
    import hashlib
    for rel, prefix in ACCESS_LOGS:
        log = repo / rel
        if not log.exists():
            continue
        key = (str(repo), rel, nonce, hashlib.sha256(log.read_bytes()).hexdigest())
        if key in _ACCESS_OK:
            return True
        if _check_access(repo, nonce, rel, prefix):
            _ACCESS_OK.add(key)
            return True
    return False


_ACCESS_OK: set = set()


def _check_access(repo: Path, nonce: str, log_rel: str = "results/test_access.log", tag_prefix: str = "frozen-") -> bool:
    log = repo / log_rel
    started, closed = {}, set()
    for line in log.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        _, event, commit, n = parts[:4]
        if event == "started":
            started[n] = commit
        elif event in ("completed", "aborted"):
            closed.add(n)
    if nonce not in started or nonce in closed:
        return False
    try:
        git_repo, head = code_identity(repo)
        tags = _git("tag", "--points-at", head, cwd=git_repo).split()
    except subprocess.CalledProcessError:
        return False
    return started[nonce] == head and any(t.startswith(tag_prefix) for t in tags)


def in_orch_snapshot(repo: Path | None = None) -> bool:
    """True when `repo` is the code snapshot of an orch job (scripts/orch: runs/<id>/code = `git archive` of the
    submitting tree's committed HEAD; $ORCH_CODE names it, $ORCH_COMMIT is its commit, $ORCH_HOME the checkout)."""
    repo = PROJECT_ROOT if repo is None else Path(repo)
    code, commit, home = (os.environ.get(k, "") for k in ("ORCH_CODE", "ORCH_COMMIT", "ORCH_HOME"))
    if not (code and commit and home):
        return False
    try:
        return Path(code).resolve() == Path(repo).resolve()
    except OSError:
        return False


def code_identity(repo: Path | None = None) -> tuple[Path, str]:
    """(git repository, commit) of the code being run. Outside orch: (repo, HEAD of repo). Inside an orch job
    snapshot: ($ORCH_HOME, $ORCH_COMMIT): the snapshot is not a git work tree, and the main checkout's HEAD may
    move during a long run (merges) without changing the code that runs, so the commit of the snapshot is the
    identity that the test lock checks (p6.dryrun_testpipeline)."""
    repo = PROJECT_ROOT if repo is None else Path(repo)
    if in_orch_snapshot(repo):
        home = Path(os.environ["ORCH_HOME"])
        commit = _git("rev-parse", "--verify", os.environ["ORCH_COMMIT"] + "^{commit}", cwd=home)
        return home, commit
    return repo, _git("rev-parse", "HEAD", cwd=repo)


def guard_locus(locus_id: str, split: str = "", block_ids: Iterable = (), window_chr: str = "",
                window_start: int = -1, window_end: int = -1) -> None:
    if is_test_locus(split, block_ids, window_chr, window_start, window_end) and not has_test_access():
        raise TestSplitAccessError(
            f"locus {locus_id!r} belongs to the test split; test data may only be produced/read by "
            f"scripts/eval_test.py (DESIGN §7.3)")


# --------------------------------------------------------------------------------------------------
# HDF5 reading
# --------------------------------------------------------------------------------------------------
def _attr_value(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        if v.dtype.kind in ("S", "O"):
            return tuple(x.decode() if isinstance(x, bytes) else str(x) for x in v.ravel())
        return tuple(v.ravel().tolist())
    if isinstance(v, np.generic):
        return v.item()
    return v


def list_loci(path: str | Path) -> list[str]:
    """Locus ids in file order (reserved groups starting with '_' skipped). No data is read."""
    with h5py.File(path, "r") as f:
        return [k for k in f.keys() if not k.startswith("_")]


def _group_attrs(g: h5py.Group) -> dict:
    return {k: _attr_value(v) for k, v in g.attrs.items()}


def _guard_from_attrs(lid: str, a: dict) -> None:
    guard_locus(lid, a.get("split", ""), a.get("block_ids", ()) or (), a.get("window_chr", ""),
                int(a.get("window_start", -1)), int(a.get("window_end", -1)))


_ATTRS_CACHE: dict = {}
_ATTRS_LOCK = __import__("threading").Lock()


def read_locus_attrs(path: str | Path, guard: bool = True) -> pd.DataFrame:
    """One row per locus with its attributes (no arrays). Refuses test-split loci (guard).

    The attribute rows are cached in-process per (file, inode, size, mtime): reading the attributes of a master
    file with external links costs ~8 ms per locus (35k loci: 287 s), and one test-run driver reads each set many
    times (one harness run per method). The guard is re-applied to every row on every call, cached or not."""
    st = os.stat(path)
    key = (str(Path(path).resolve()), st.st_ino, st.st_size, st.st_mtime_ns)
    with _ATTRS_LOCK:
        cached = _ATTRS_CACHE.get(key)
    if cached is None:
        cached = []
        with h5py.File(path, "r") as f:
            for lid in f.keys():
                if lid.startswith("_"):
                    continue
                g = f[lid]
                a = _group_attrs(g)
                if guard:
                    _guard_from_attrs(lid, a)
                a["locus_id"] = lid
                a["p"] = int(g["z1"].shape[0])
                a["ld_type"] = a.get("ld_type", "")
                cached.append(a)
        with _ATTRS_LOCK:
            if len(_ATTRS_CACHE) > 64:
                _ATTRS_CACHE.clear()
            _ATTRS_CACHE[key] = cached
    elif guard:
        for a in cached:
            _guard_from_attrs(a["locus_id"], a)
    return pd.DataFrame([dict(a) for a in cached])


def _read_group(lid: str, g: h5py.Group, guard: bool, load_ld: bool = True, load_ld2: bool = False) -> LocusData:
    a = _group_attrs(g)
    if guard:
        _guard_from_attrs(lid, a)
    kw = {k: np.asarray(g[k][()], dtype=np.float64) for k in ARRAY_FIELDS_F64}
    kw["variant_id"] = np.asarray(g["variant_id"].asstr()[()], dtype=object)
    kw["pos"] = np.asarray(g["pos"][()], dtype=np.int64)
    ld_type = a.pop("ld_type")
    n_ref = int(a.pop("n_ref"))
    # load_ld=False (methods declaring needs_ld = False, e.g. the strength-only control): no LD is read
    R = np.asarray(g["R"][()], dtype=np.float32) if (load_ld and ld_type == "dense") else None
    X = np.asarray(g["X_ref"][()], dtype=np.float32) if (load_ld and ld_type == "factor") else None
    # load_ld2=True (methods declaring needs_ld2 = True): trait-2 LD where the locus stores one (sim L0)
    R2 = np.asarray(g["R2"][()], dtype=np.float32) if (load_ld2 and "R2" in g) else None
    X2 = np.asarray(g["X_ref2"][()], dtype=np.float32) if (load_ld2 and "X_ref2" in g) else None
    known = {}
    for k in LOCUS_ATTRS:
        if k in a:
            known[k] = a.pop(k)
    if "block_ids" in known:
        b = known["block_ids"]
        known["block_ids"] = tuple(str(x) for x in (b if isinstance(b, tuple) else (b,)))
    for k in ("window_start", "window_end"):
        if k in known:
            known[k] = int(known[k])
    for k in ("rho_hat", "rho_se"):
        if k in known:
            known[k] = float(known[k])
    return LocusData(locus_id=lid, ld_type=ld_type, n_ref=n_ref, R=R, X_ref=X, R2=R2, X_ref2=X2, extra_attrs=a,
                     **known, **kw)


class LocusSetReader:
    """Keeps the HDF5 file open; use as a context manager. Not shareable across processes (open one per process)."""

    def __init__(self, path: str | Path, guard: bool = True):
        self.path = Path(path)
        self.guard = guard
        self._f = h5py.File(self.path, "r")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def ids(self) -> list[str]:
        return [k for k in self._f.keys() if not k.startswith("_")]

    def read(self, locus_id: str, load_ld: bool = True, load_ld2: bool = False) -> LocusData:
        return _read_group(locus_id, self._f[locus_id], self.guard, load_ld, load_ld2)

    def __iter__(self) -> Iterator[LocusData]:
        for lid in self.ids():
            yield self.read(lid)


def read_locus(path: str | Path, locus_id: str, guard: bool = True) -> LocusData:
    with LocusSetReader(path, guard=guard) as r:
        return r.read(locus_id)


def iter_loci(path: str | Path, ids: Iterable[str] | None = None, guard: bool = True) -> Iterator[LocusData]:
    with LocusSetReader(path, guard=guard) as r:
        for lid in (r.ids() if ids is None else ids):
            yield r.read(lid)


# --------------------------------------------------------------------------------------------------
# HDF5 writing
# --------------------------------------------------------------------------------------------------
def write_locus_set(path: str | Path, loci: Iterable[LocusData], mode: str = "w", compression: str | None = None,
                    guard: bool = True) -> int:
    """Write loci into one HDF5 file (mode 'w' creates/overwrites, 'a' appends). Returns number written.

    If `locus.ld_key` is set, its LD is stored once under `/_ld/<ld_key>/{R|X_ref}` and hard-linked into the
    locus group (identical content is NOT checked beyond shape: the caller guarantees one key = one matrix).
    """
    n = 0
    str_dt = h5py.string_dtype("utf-8")
    with h5py.File(path, mode, track_order=True) as f:  # list_loci returns loci in insertion order
        for loc in loci:
            loc.validate()
            if guard:
                guard_locus(loc.locus_id, loc.split, loc.block_ids, loc.window_chr, loc.window_start, loc.window_end)
            if loc.locus_id in f:
                raise ValueError(f"duplicate locus_id {loc.locus_id!r} in {path}")
            g = f.create_group(loc.locus_id, track_order=True)
            g.create_dataset("variant_id", data=np.asarray(loc.variant_id, dtype=object), dtype=str_dt)
            g.create_dataset("pos", data=np.asarray(loc.pos, dtype=np.int64))
            for k in ARRAY_FIELDS_F64:
                g.create_dataset(k, data=np.asarray(getattr(loc, k), dtype=np.float64), compression=compression)
            ld_name = "R" if loc.ld_type == "dense" else "X_ref"
            ld_arr = np.asarray(loc.R if loc.ld_type == "dense" else loc.X_ref, dtype=np.float32)
            if loc.ld_key is not None:
                key = f"_ld/{loc.ld_key}/{ld_name}"
                if key not in f:
                    f.create_dataset(key, data=ld_arr, compression=compression)
                elif f[key].shape != ld_arr.shape:
                    raise ValueError(f"ld_key {loc.ld_key!r} reused with a different shape")
                g[ld_name] = f[key]  # hard link
            else:
                g.create_dataset(ld_name, data=ld_arr, compression=compression)
            for name in ("R2", "X_ref2"):   # optional trait-2 LD (round trip of LocusData.R2 / X_ref2)
                if getattr(loc, name, None) is not None:
                    g.create_dataset(name, data=np.asarray(getattr(loc, name), dtype=np.float32),
                                     compression=compression)
            g.attrs["ld_type"] = loc.ld_type
            g.attrs["n_ref"] = int(loc.n_ref)
            g.attrs["rho_hat"] = float(loc.rho_hat)
            g.attrs["rho_se"] = float(loc.rho_se)
            for k in ("split", "mismatch_level", "pairing_type", "cell_id", "window_chr"):
                g.attrs[k] = str(getattr(loc, k))
            g.attrs["window_start"] = int(loc.window_start)
            g.attrs["window_end"] = int(loc.window_end)
            g.attrs.create("block_ids", data=np.asarray([str(b) for b in loc.block_ids], dtype=object), dtype=str_dt)
            for k, v in loc.extra_attrs.items():
                if k in g.attrs:
                    raise ValueError(f"extra attr {k!r} clashes with a standard attribute")
                g.attrs[k] = v
            n += 1
    return n


def truth_path_for(locus_set_path: str | Path) -> Path:
    """Convention: `<dir>/<stem>.h5` -> `<dir>/<stem>.truth.parquet` (INTERFACES §2)."""
    p = Path(locus_set_path)
    stem = p.name[:-3] if p.name.endswith(".h5") else (p.name[:-5] if p.name.endswith(".hdf5") else p.name)
    return p.with_name(stem + ".truth.parquet")

