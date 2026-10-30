"""The single test run (DESIGN §7.3, §8, §11 M7; DECISIONS P3-12, P3-14, P3-17) and its validation dry run.

Driven ONLY by scripts/eval_test.py (the test lock lives there and in eval.io / sim.testset). One pipeline,
two modes that differ only in their inputs:

  mode "test": test-split loci generated with seed 20261028 (configs/testset_v1.yaml), real-data test arms
               from the frozen ID lists (builder entry point named in the methods file), operating points,
               recalibration intercepts and the strength-only table frozen in the methods file.
  mode "dry":  identical code path on VALIDATION blocks (split "valid", seed 20261029), every per-cell size
               multiplied by `scale`, real arms = seeded subsamples of the validation real locus sets
               (real_locus_sets@2) with the P3-14 sizes multiplied by `scale`; operating points,
               recalibration intercepts and the strength table are fitted on the dry set itself and labelled
               DRY (placeholders that exercise the frozen-value code path; never used for any claim).

Stages (every stage is idempotent; a restart resumes, see `resume` below):
  1. real arms   -> <bundle>/state/real.json
  2. sim sets    -> per-cell part files + master files; a cell is done when its part truth parquet exists
                   (sim.cells.build_set writes it last); generation runs in the MAIN thread (sim.testset checks
                   that scripts/eval_test.py is on the call stack)
  3. method units: one harness run per (method x locus set) with a fixed run_id and a per-locus checkpoint
                   (eval.harness checkpoint_dir), so a crash loses at most the loci in flight. CPU methods run
                   concurrently (core / RAM budget, longest expected unit first, loci largest-p first); GPU
                   methods run one at a time on the GPU after generation has released it. Slow methods run on
                   the pre-registered nested subsample (P3-12/P3-17: 40/40/40 per cell, mvSuSiE the first 20/20/20
                   of the same draw), main grid and shifted cells.
  4. evaluation  -> benchmark directories (scripts/run_benchmark.evaluate, operating points GIVEN):
                   sim_full (non-slow methods, all loci), sim_slow<k> (every method on the k-subsample), C4
                   slopes, real-data summary (DESIGN §2.5 criterion), runtime table.
  5. verification-> run_benchmark.verify on every benchmark (re-verifies every harness run with verify_run and
                   recomputes benchmark.json), verify_run on the real-arm runs, sha256 of every input file.
The bundle manifest (<bundle>/bundle.json) lists all of it; scripts/eval_test.py --verify-bundle re-checks it.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from common.seeds import rng as seeded_rng
from eval.harness import MethodSpec, append_ledger, git_info, run_method, sha256_file
from eval.io import PROJECT_ROOT, read_locus_attrs, truth_path_for

TEST_SEED = 20261028
DRY_SEED = 20261029           # DESIGN §9: validation draws
CONTROL = "strength_only"
NO_THRESHOLD_METHODS = ("lead_snp", "prop_coloc", "coloc_prop_test")   # no P_S / P_D (DESIGN §4.3)
EXCLUDED_FROM_C3 = NO_THRESHOLD_METHODS + (CONTROL,)
REAL_SETS = {  # arm key -> (arm, LD condition)
    "real_R2": ("R2", None), "real_R3b": ("R3b", None),
    "real_R4b_insample": ("R4b", "in_sample"), "real_R4b_reference": ("R4b", "reference"),
}
# validation real locus sets (real_locus_sets@2) used by the dry run; R4b in-sample = PC-adjusted GEUVADIS
# (D6(i) primary), reference = NFE minus GEUVADIS
DRY_REAL_FILES = {
    "real_R2": "R2_valid.h5", "real_R3b": "R3b_valid.h5",
    "real_R4b_insample": "R4b_valid.geuv_insample_pcadj.h5", "real_R4b_reference": "R4b_valid.nfe_minus_geuv.h5",
}
DRY_REAL_DIR = Path("data/processed/real/locus_sets")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


def _ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


# ==================================================================================================
# Methods file
# ==================================================================================================
@dataclass
class MethodEntry:
    name: str
    target: str
    kwargs: dict
    mode: str = "cpu"
    n_workers: int = 8
    batch_size: int = 64
    timeout_s: float = 600.0
    gpu: bool = False
    mem_gb_per_worker: float = 2.0
    expected_s_per_locus: float = 1.0
    subsample_per_class: int | None = None
    metric_config: dict = field(default_factory=dict)
    real: bool = False
    descriptive: bool = False   # D16: pre-declared descriptive entry (no claims; never candidate or C3 baseline)

    @classmethod
    def from_dict(cls, d: dict, real_names: set) -> "MethodEntry":
        name = d.get("name") or MethodSpec(d["target"], d.get("kwargs") or {}).method_name()
        sub = d.get("subsample")
        per = None
        if isinstance(sub, dict):
            per = int(sub["per_class"])
        elif sub == "slow_method_subsample":
            per = -1   # resolved from configs/testset_v1.yaml slow_method_subsample (resolve_subsamples)
        elif sub not in (None, False):
            raise ValueError(f"{name}: subsample must be null or {{per_class: N}}")
        mode = d.get("mode", "cpu")
        return cls(name=name, target=d["target"], kwargs=dict(d.get("kwargs") or {}), mode=mode,
                   n_workers=int(d.get("n_workers", 1 if mode == "batch" else 8)),
                   batch_size=int(d.get("batch_size", 64)), timeout_s=float(d["timeout_s"]),
                   gpu=bool(d.get("gpu", mode == "batch")), mem_gb_per_worker=float(d.get("mem_gb_per_worker", 2.0)),
                   expected_s_per_locus=float(d.get("expected_s_per_locus", 1.0)), subsample_per_class=per,
                   metric_config=dict(d.get("metric_config") or {}), real=name in real_names,
                   descriptive=bool(d.get("descriptive", False)))


def parse_methods(doc: dict) -> tuple[list[MethodEntry], dict]:
    """Method entries and the real-arm spec of a methods file (configs/test_methods_v1.yaml layout)."""
    cand = doc.get("candidates") or {}
    ra = doc.get("real_data_arms") or None
    real_names = set()
    if ra:
        for n in ra.get("methods") or []:
            nm = cand.get("primary") if n == "primary_candidate" else n
            if nm:                                    # primary: null (D16 MVR route) -> no candidate on real arms
                real_names.add(nm)
    ms = [MethodEntry.from_dict(m, real_names) for m in doc.get("methods") or []]
    names = [m.name for m in ms]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate method names in the methods file: {names}")
    return ms, (ra or {})


def resolve_subsamples(methods: list[MethodEntry], testset: dict) -> list[str]:
    """Set each slow method's per-class subsample size from configs/testset_v1.yaml slow_method_subsample
    (per_method overrides per_cell; P3-12 / P3-17) and check the methods file against it. Returns problems."""
    ss = testset.get("slow_method_subsample") or {}
    listed = set(ss.get("methods") or [])
    out = []
    for m in methods:
        cfg_n = None
        if m.name in listed:
            pc = (ss.get("per_method") or {}).get(m.name) or ss.get("per_cell") or {}
            vals = {int(pc[c]) for c in ("S", "D", "W") if c in pc}
            if len(vals) != 1:
                out.append(f"slow_method_subsample for {m.name}: per-class sizes must be equal ({pc})")
                continue
            cfg_n = vals.pop()
        if m.subsample_per_class == -1:
            if cfg_n is None:
                out.append(f"{m.name}: subsample slow_method_subsample, but it is not in slow_method_subsample.methods")
            m.subsample_per_class = cfg_n
        elif m.subsample_per_class is not None and m.subsample_per_class != cfg_n:
            out.append(f"{m.name}: methods-file subsample {m.subsample_per_class} != testset config {cfg_n}")
        elif m.subsample_per_class is None and cfg_n is not None:
            out.append(f"{m.name} is a slow method (testset slow_method_subsample) but runs without its subsample")
    return out


def check_methods_runnable(doc: dict, repo: Path, mode: str, testset: dict | None = None) -> list[str]:
    """Plumbing checks that can be made before anything runs (both modes): targets importable, required
    fields, frozen values present (test mode), strength table present, real-arm builder importable."""
    out = []
    try:
        ms, ra = parse_methods(doc)
    except Exception as e:  # noqa: BLE001
        return [f"methods file does not parse: {type(e).__name__}: {e}"]
    names = {m.name for m in ms}
    if testset is not None:
        out += resolve_subsamples(ms, testset)
    for m in ms:
        try:
            MethodSpec(m.target, m.kwargs, m.name).cls()
        except Exception as e:  # noqa: BLE001
            out.append(f"{m.name}: target {m.target} not importable ({type(e).__name__}: {e})")
        if m.mode not in ("cpu", "batch"):
            out.append(f"{m.name}: mode {m.mode!r}")
        if mode == "test" and m.name not in NO_THRESHOLD_METHODS:
            mc = m.metric_config
            if mc.get("t_S") is None or mc.get("t_D") is None:
                out.append(f"{m.name}: metric_config lacks the validation-fixed t_S / t_D (C3)")
    if CONTROL not in names:
        out.append(f"{CONTROL} (D6(e) control) is not in the methods list")
    elif mode == "test":
        tab = next(m for m in ms if m.name == CONTROL).kwargs.get("table")
        if not tab or not (repo / tab).exists():
            out.append(f"{CONTROL}: frozen strength table {tab!r} missing")
        else:
            want = (doc.get("validation_source") or {}).get("strength_table_sha256")
            if want and sha256_file(repo / tab) != want:
                out.append(f"{CONTROL}: strength table {tab} sha256 differs from validation_source.strength_table_sha256")
    prim = (doc.get("candidates") or {}).get("primary")
    if prim and prim not in names:
        out.append(f"primary candidate {prim!r} not in the methods list")
    desc = {m.name for m in ms if m.descriptive}
    if prim and prim in desc:
        out.append(f"primary candidate {prim!r} is marked descriptive")
    for b in doc.get("c3_eligible_baselines") or []:
        if b in desc:
            out.append(f"c3_eligible_baselines: {b!r} is a descriptive entry (D16: no claims)")
    for b in doc.get("c3_eligible_baselines") or []:
        if b not in names:
            out.append(f"c3_eligible_baselines: {b!r} not in the methods list")
    if ra:
        missing = [n for n in ra.get("methods") or [] if n != "primary_candidate" and n not in names]
        if missing:
            out.append(f"real_data_arms.methods not in the methods list: {missing}")
        if mode == "test":
            b = ra.get("builder")
            if not b:
                out.append("real_data_arms.builder (entry point building the test arms from the frozen ID lists) "
                           "is not set")
            else:
                try:
                    _entry(b)
                except Exception as e:  # noqa: BLE001
                    out.append(f"real_data_arms.builder {b} not importable ({type(e).__name__}: {e})")
    return out


# expected generation cost rank per mismatch level when no validation timing is available (valid_grid_v1:
# L3 slowest (L3|P1 5,136 s of candidate time), then L4, L6, L2, L1, L0)
_LEVEL_COST = {"L3": 6, "L4": 5, "L6": 4, "L2": 3, "L1": 2, "L0": 1}


def order_cells_by_cost(cells: list[str], repo: Path) -> list[str]:
    """Slowest cells first (PI decision after p6 handover, item d): candidate + write seconds per cell of the
    validation grid valid_grid_v1 (same cells, same generator; validation metadata only), else the level rank.
    With a pool of a few generator workers this puts the critical-path cell (L3|P1) at the start, not 22nd."""
    cost = {}
    stats = Path(repo) / "data/processed/sets/valid_grid_v1.stats.json"
    try:
        st = json.loads(stats.read_text())["cells"]
        cost = {c: float(v["t_candidates_s"]) + float(v["t_finalize_write_s"]) for c, v in st.items()}
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return sorted(cells, key=lambda c: (-cost.get(c, 0.0), -_LEVEL_COST.get(c.split("|")[0], 0), c))


def _entry(spec: str):
    import importlib
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


# ==================================================================================================
# Pipeline configuration
# ==================================================================================================
@dataclass
class RunConfig:
    mode: str                    # "test" | "dry"
    bundle: Path                 # results bundle root (manifests, metrics, benchmarks)
    data_dir: Path               # generated locus sets (large; git-ignored)
    label: str
    scale: float = 1.0
    cores: int = 64
    ram_gb: float = 120.0
    gen_workers: int = 4
    n_boot: int = 2000
    n_jobs: int = 16
    unit: str = "block"
    retry_failed_units: int = 1
    accept_failed_units: bool = False
    stages: tuple = ("real", "sim", "methods", "evaluate", "verify")
    methods_only: tuple | None = None   # dry-run convenience: restrict to these method names
    dry_real_builder: bool = False      # dry run: build the real arms with the test-run builder in mode "valid"

    @property
    def seed(self) -> int:
        return TEST_SEED if self.mode == "test" else DRY_SEED


@dataclass
class Unit:
    method: MethodEntry
    set_key: str
    set_path: Path
    ids: list
    sim: bool
    run_id: str

    @property
    def cost(self) -> float:
        k = 1 if self.method.gpu else max(1, self.method.n_workers)
        return len(self.ids) * self.method.expected_s_per_locus / k


class PipelineIncomplete(RuntimeError):
    pass


# ==================================================================================================
# The pipeline
# ==================================================================================================
class Pipeline:
    def __init__(self, cfg: RunConfig, testset: dict, methods_doc: dict, repo: Path = PROJECT_ROOT,
                 log: Callable[[str], None] | None = None, extra_manifest: dict | None = None):
        self.cfg, self.testset, self.doc, self.repo = cfg, testset, methods_doc, Path(repo)
        self.methods, self.real_spec = parse_methods(methods_doc)
        probs = resolve_subsamples(self.methods, testset)
        if probs:
            raise ValueError("slow-method subsample: " + "; ".join(probs))
        if cfg.methods_only:
            keep = set(cfg.methods_only) | {CONTROL}
            self.methods = [m for m in self.methods if m.name in keep]
        self.bundle = Path(cfg.bundle)
        self.runs_dir = self.bundle / "runs"
        self.state_dir = self.bundle / "state"
        self.ckpt_dir = self.bundle / "checkpoints"
        self.log_dir = self.bundle / "logs"
        for d in (self.runs_dir, self.state_dir, self.ckpt_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.bundle / "pipeline.log", "a")
        self._lock = threading.Lock()
        self._print = log or (lambda s: print(s, flush=True))
        self.extra_manifest = extra_manifest or {}
        self.sets: dict[str, dict] = {}
        self.ready = {k: threading.Event() for k in ["sim_main", "sim_shifted", *REAL_SETS]}
        self.gen_running = threading.Event()
        self.unit_status: dict[str, dict] = self._load_json(self.state_dir / "units.json", {})
        self.sens_base: dict[str, str] = {}   # P3-23 sensitivity unit name -> base method

    # ---------------------------------------------------------------- utilities
    def log(self, msg: str) -> None:
        line = f"[eval_test {datetime.now().strftime('%H:%M:%S')}] {msg}"
        with self._lock:
            self._log_fh.write(line + "\n")
            self._log_fh.flush()
        self._print(line)

    @staticmethod
    def _load_json(path: Path, default):
        try:
            return json.loads(Path(path).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    def _stage_done(self, name: str) -> dict | None:
        return self._load_json(self.state_dir / f"{name}.json", None)

    def _mark(self, name: str, payload: dict) -> None:
        _atomic_json(self.state_dir / f"{name}.json", dict(payload, done_utc=_now()))

    def _set_unit(self, run_id: str, **kw) -> None:
        with self._lock:
            self.unit_status.setdefault(run_id, {}).update(kw)
            _atomic_json(self.state_dir / "units.json", self.unit_status)

    @property
    def run_seed(self) -> int:
        """Seed of the run: in test mode the test-set config's `seed` (v1: 20261028 = TEST_SEED, as before;
        v2: 20261101, DESIGN_P2 §6), else DRY_SEED (= RunConfig.seed)."""
        return int(self.testset.get("seed", TEST_SEED)) if self.cfg.mode == "test" else DRY_SEED

    def scaled(self, n: int, minimum: int = 1) -> int:
        return int(n) if self.cfg.mode == "test" else max(minimum, int(math.ceil(n * self.cfg.scale)))

    # ---------------------------------------------------------------- stage 1: real arms
    def stage_real(self) -> None:
        if not self.real_spec:
            self.log("real-data arms: none configured")
            self._mark("real", {"sets": {}})
            return
        st = self._stage_done("real")
        if st is None:
            t0 = time.time()
            if self.cfg.mode == "test":
                sets = self._build_real_test()
            elif self.cfg.dry_real_builder and self.real_spec.get("builder"):
                sets = self._build_real_dry_builder()
            else:
                sets = self._build_real_dry()
            st = {"sets": sets, "wall_s": time.time() - t0}
            self._mark("real", st)
        for k, v in st["sets"].items():
            self.sets[k] = {"path": Path(v["path"]), "ids": v["ids"], "sim": False}
            self.ready[k].set()
        self.log(f"real-data arms ready: " + ", ".join(f"{k} n={len(v['ids'])}" for k, v in st["sets"].items()))

    def _real_n(self, arm: str) -> int | None:
        spec = (self.real_spec.get(arm) or {})
        n = spec.get("n")
        return None if n is None else int(n)

    def _build_real_dry(self) -> dict:
        out = {}
        g_ids = {}
        for key, fname in DRY_REAL_FILES.items():
            arm, cond = REAL_SETS[key]
            path = (self.repo / DRY_REAL_DIR / fname).resolve()
            ids = sorted(read_locus_attrs(path)["locus_id"].astype(str).tolist())
            if arm not in g_ids:
                n = self._real_n(arm)
                k = len(ids) if n is None else min(len(ids), self.scaled(n))
                pick = seeded_rng(DRY_SEED, "dryrun_real_subsample", arm).choice(len(ids), size=k, replace=False)
                g_ids[arm] = sorted(ids[j] for j in pick)
            sel = g_ids[arm]
            missing = set(sel) - set(ids)
            if missing:   # R4b: the two LD conditions share one locus list (data.real.extract, same variant set)
                raise ValueError(f"{key}: {len(missing)} ids of the {arm} subsample are not in {path.name}")
            out[key] = {"path": str(path), "ids": sel, "sha256": sha256_file(path), "source": "real_locus_sets@2 (valid)",
                        "condition": cond}
        return out

    def _build_real_dry_builder(self) -> dict:
        """Dry run through the TEST-RUN builder (real_data_arms.builder, e.g. data.real.extract:build_test_arm) in
        its validation mode: seeded validation ID lists in the frozen ID-list format (P3-14 sizes x scale; R4b one
        list for both LD conditions) -> builder(real_spec, out_dir, workers, mode="valid", id_lists=...)."""
        from data.real.freeze_test_ids import ID_COLS
        builder = _entry(self.real_spec["builder"])
        pairs = self.repo / "data/processed/real/pairs"
        lists = {}
        ldir = self.cfg.data_dir / "real_id_lists"
        ldir.mkdir(parents=True, exist_ok=True)
        for arm in ("R2", "R3b", "R4b"):
            cand = pd.read_csv(pairs / f"{arm}_candidates.tsv", sep="\t", dtype=str)
            cand = cand[cand["split"] == "valid"].sort_values("locus_id")
            n = self._real_n(arm)
            k = len(cand) if n is None else min(len(cand), self.scaled(n))
            pick = np.sort(seeded_rng(DRY_SEED, "dryrun_real_subsample", arm).choice(len(cand), size=k, replace=False))
            f = ldir / f"{arm}_valid_ids_dry.tsv"
            cand.iloc[pick][ID_COLS].to_csv(f, sep="\t", index=False)
            lists[arm] = str(f)
        res = builder(self.real_spec, self.cfg.data_dir / "real", workers=max(2, self.cfg.cores // 2), mode="valid",
                      id_lists=lists)
        out = {}
        for key in REAL_SETS:
            if key not in res:
                raise ValueError(f"real-arm builder returned no {key}")
            path = Path(res[key]["path"]).resolve()
            out[key] = {"path": str(path), "ids": [str(i) for i in res[key]["ids"]], "sha256": sha256_file(path),
                        "source": f"{self.real_spec['builder']} (mode valid)", "condition": REAL_SETS[key][1],
                        "failed_ids": res[key].get("failed_ids")}
        return out

    def _build_real_test(self) -> dict:  # pragma: no cover - only inside the authorised test run
        builder = _entry(self.real_spec["builder"])
        res = builder(self.real_spec, self.cfg.data_dir / "real", workers=max(4, self.cfg.cores // 2))
        out = {}
        for key in REAL_SETS:
            if key not in res:
                raise ValueError(f"real-arm builder returned no {key}")
            path = Path(res[key]["path"]).resolve()
            ids = [str(i) for i in res[key]["ids"]]
            out[key] = {"path": str(path), "ids": ids, "sha256": sha256_file(path), "source": self.real_spec["builder"],
                        "condition": REAL_SETS[key][1]}
        return out

    # ---------------------------------------------------------------- stage 2: simulated sets
    def _gen_kwargs(self) -> dict:
        tsc = self.testset
        kw = {"K": int(tsc["generator"].get("K", 200)), "M": int(tsc["generator"].get("M", 3)),
              "workers": self.cfg.gen_workers, "log": self.log, "log_dir": self.log_dir}
        if self.cfg.mode == "test":
            kw.update(seed=self.run_seed)
        else:
            pc = tsc["per_cell"]
            gu = pc["G_U"]["max"] if isinstance(pc.get("G_U"), dict) else pc.get("G_U", 30)
            assert int(pc["S"]) == int(pc["D"]) == int(pc["W"]), "generator takes one n_per_class"
            kw.update(seed=DRY_SEED, split="valid", dry_run=True, set_name=self.cfg.label,
                      n_per_class=self.scaled(pc["S"]), n_gu=self.scaled(gu),
                      n_shifted=self.scaled(int(tsc["shifted_arm"]["n_loci"]), minimum=4))
        return kw

    def _set_base(self) -> str:
        return f"testset_v{self.testset.get('version', 1)}" if self.cfg.mode == "test" else self.cfg.label

    def stage_sim(self, generate: Callable) -> None:
        """Generate (or resume) the main grid and the shifted arm. MUST run in the main thread of
        scripts/eval_test.py (sim.testset refuses test-split generation otherwise)."""
        st = self._stage_done("sim")
        if st is None:
            from sim.cells import _safe
            from sim.testset import main_grid_cells
            kw = self._gen_kwargs()
            out = self.cfg.data_dir / "sets"
            out.mkdir(parents=True, exist_ok=True)
            base = self._set_base()
            cells = [c.cell_id for c in main_grid_cells(self.testset, "test" if self.cfg.mode == "test" else "valid")]
            part = out / base

            def cell_done(cid):
                t = part / f"{_safe(cid)}.truth.parquet"
                try:
                    return t.exists() and len(pd.read_parquet(t)) > 0
                except Exception:  # noqa: BLE001 - torn file from a crash: regenerate the cell
                    return False
            todo = order_cells_by_cost([c for c in cells if not cell_done(c)], self.repo)
            t0 = time.time()
            self.gen_running.set()
            segs = self._load_json(self.state_dir / "sim_segments.json", [])
            try:
                if todo:
                    seg = len(segs)
                    self.log(f"generating main grid: {len(todo)}/{len(cells)} cells to do (segment {seg})")
                    generate(self.testset, out, main=True, shifted=False, cells_filter=todo,
                             stats_suffix=f".seg{seg}", **kw)
                    segs.append({"segment": seg, "cells": todo, "wall_s": time.time() - t0, "end_utc": _now()})
                    _atomic_json(self.state_dir / "sim_segments.json", segs)
                from sim.cells import link_set
                master, mtruth = link_set(base, out)
                missing = [c for c in cells if not cell_done(c)]
                if missing:
                    raise RuntimeError(f"generation finished but cells are incomplete: {missing}")
                t1 = time.time()
                self.log("generating shifted arm")
                spaths = generate(self.testset, out, main=False, shifted=True, **kw)
                t_shift = time.time() - t1
            finally:
                self.gen_running.clear()
            st = {"main": str(Path(master).resolve()), "shifted": str(Path(spaths[0]).resolve()),
                  "main_truth": str(Path(mtruth).resolve()), "wall_s_main_this_segment": t1 - t0,
                  "wall_s_shifted": t_shift, "segments": segs, "generator_kwargs": {
                      k: v for k, v in kw.items() if k not in ("log", "log_dir")}}
            self._mark("sim", st)
        if (self.testset.get("disjoint_from") or []) and self._stage_done("disjoint") is None:
            self._mark("disjoint", self.disjointness_check(st))
        for key, p in (("sim_main", st["main"]), ("sim_shifted", st["shifted"])):
            ids = read_locus_attrs(p)["locus_id"].astype(str).tolist()
            self.sets[key] = {"path": Path(p), "ids": ids, "sim": True}
            self.ready[key].set()
        self.log(f"simulated sets ready: main n={len(self.sets['sim_main']['ids'])}, "
                 f"shifted n={len(self.sets['sim_shifted']['ids'])}")

    def disjointness_check(self, st: dict) -> dict:
        """DESIGN_P2 §6 / M8: no generation key, derived seed or locus id of this run's simulated sets overlaps a
        reference test set named in the config's `disjoint_from` (eval.testset_lock). Test mode: the reference is
        that test set's own run (its sim stage state: stats files and truth ids; read inside this run's access
        window). Dry mode: the reference is the entry's `dry_reference_stats` (validation sets), so the same code
        runs without test data. Any overlap aborts the run before a method sees a locus."""
        from eval import testset_lock as TL
        from eval.truth import read_truth
        out = {"definition": self.disjointness_check.__doc__.strip(), "mode": self.cfg.mode, "checks": []}
        for ref in self.testset.get("disjoint_from") or []:
            for arm in ("main", "shifted"):
                new_set = Path(st[arm])
                new = TL.read_stats(str(new_set.with_suffix("")) + ".stats*.json")
                new_ids = read_truth(truth_path_for(new_set), guard=True)["locus_id"].astype(str).tolist()
                if self.cfg.mode == "test":
                    rst = json.loads((self.repo / ref["sim_state"]).read_text())
                    rset = Path(rst[arm])
                    rs = TL.read_stats(str(rset.with_suffix("")) + ".stats*.json")
                    rids = read_truth(truth_path_for(rset), guard=True)["locus_id"].astype(str).tolist()
                    src = str(ref["sim_state"])
                else:
                    pat = (ref.get("dry_reference_stats") or {}).get(arm)
                    if not pat:
                        out["checks"].append({"reference": ref.get("testset_id"), "arm": arm, "skipped":
                                              "no dry_reference_stats for this arm"})
                        continue
                    rs, rids, src = TL.read_stats(str(self.repo / pat)), None, pat
                res = TL.check_disjoint(new, rs, new_ids, rids)
                res.update(reference=ref.get("testset_id"), arm=arm, reference_source=src, new_stats=new["files"],
                           reference_stats=rs["files"])
                out["checks"].append(res)
                self.log(f"disjointness vs {ref.get('testset_id')} ({arm}): keys {res['new']['n_keys']} vs "
                         f"{res['ref']['n_keys']}, key overlap {res['key_overlap']}, derived-seed overlap "
                         f"{res['derived_seed_overlap']}, locus-id overlap {res.get('locus_id_overlap', 'n/a')}")
        out["ok"] = all(c.get("ok", True) for c in out["checks"])
        if not out["ok"]:
            _atomic_json(self.state_dir / "disjoint_FAILED.json", out)
            raise RuntimeError("generation keys overlap a reference test set (state/disjoint_FAILED.json)")
        return out

    # ---------------------------------------------------------------- slow-method subsample
    def sim_truth(self) -> pd.DataFrame:
        from eval.truth import read_truth
        parts = [read_truth(truth_path_for(self.sets[k]["path"])) for k in ("sim_main", "sim_shifted")]
        return pd.concat(parts, ignore_index=True)

    def slow_subsamples(self) -> dict[int, list[str]]:
        """configs/testset_v1.yaml slow_method_subsample rule, nested over the per-class sizes in use:
        g = rng(seed, 'slow_subsample'); cells in main_grid.cells order then shifted_arm.cells; classes S, D, W;
        ids of the cell and class sorted as strings; idx = g.choice(n, size=min(k_max, n), replace=False);
        the k-subsample = sorted(ids[idx[:min(k, n)]]). For k = k_max this is exactly the configured draw;
        smaller k (mvSuSiE, P3-17) are the first k of the same draw, so they are nested in it."""
        sizes = sorted({m.subsample_per_class for m in self.methods if m.subsample_per_class})
        if not sizes:
            return {}
        sizes_used = {k: self.scaled(k) for k in sizes}
        kmax = max(sizes_used.values())
        cached = self._stage_done("subsample")
        if cached is not None and cached.get("sizes_used") == {str(k): v for k, v in sizes_used.items()}:
            return {int(k): v for k, v in cached["ids"].items()}
        tr = self.sim_truth()
        cells = list(self.testset["main_grid"]["cells"]) + [c["cell_id"] for c in self.testset["shifted_arm"]["cells"]]
        present = set(tr["cell_id"].astype(str))
        g = seeded_rng(self.run_seed, "slow_subsample")
        out = {k: [] for k in sizes}
        realised = {}
        for c in cells:
            if c not in present:
                raise ValueError(f"cell {c} has no loci")
            for cls in ("S", "D", "W"):
                ids = sorted(tr.loc[(tr["cell_id"].astype(str) == c) & (tr["oracle_class"].astype(str) == cls),
                                    "locus_id"].astype(str).tolist())
                if not ids:
                    continue
                idx = g.choice(len(ids), size=min(kmax, len(ids)), replace=False)
                for k in sizes:
                    sel = [ids[j] for j in idx[:min(sizes_used[k], len(ids))]]
                    out[k] += sel
                    realised[f"{k}|{c}|{cls}"] = len(sel)
        out = {k: sorted(v) for k, v in out.items()}
        self._mark("subsample", {"sizes_used": {str(k): v for k, v in sizes_used.items()},
                                 "seed": [self.run_seed, "slow_subsample"], "ids": {str(k): v for k, v in out.items()},
                                 "realised_per_cell_class": realised})
        return out

    # ---------------------------------------------------------------- dry placeholders
    def dry_strength_table(self) -> Path:
        """DRY placeholder: the strength-only table fitted on the dry sim set itself (in-sample)."""
        from eval import strength_control as SC
        path = self.bundle / "DRY_strength_table_in_sample.json"
        if path.exists():
            return path
        tr = self.sim_truth()
        feats = pd.concat([SC.read_features(self.sets[k]["path"], self.sets[k]["ids"]) for k in ("sim_main", "sim_shifted")],
                          ignore_index=True)
        cls = feats[["locus_id"]].merge(tr[["locus_id", "oracle_class"]], on="locus_id", how="left")["oracle_class"]
        tab = SC.StrengthTable.fit(feats, cls.to_numpy()).to_dict()
        SC.dump(tab, path)
        return path

    # ---------------------------------------------------------------- stage 3: units
    def plan_units(self) -> list[Unit]:
        subs = self.slow_subsamples()
        units = []
        for m in self.methods:
            for key in ("sim_main", "sim_shifted"):
                ids = self.sets[key]["ids"]
                if m.subsample_per_class:
                    keep = set(subs[m.subsample_per_class])
                    ids = [i for i in ids if i in keep]
                units.append(Unit(m, key, self.sets[key]["path"], ids, True, f"{self.cfg.label}__{m.name}__{key}"))
            if m.real:
                for key in REAL_SETS:
                    if key in self.sets:
                        units.append(Unit(m, key, self.sets[key]["path"], self.sets[key]["ids"], False,
                                          f"{self.cfg.label}__{m.name}__{key}"))
        for sm in self.rho_sensitivity_entries():
            for key in REAL_SETS:
                if key in self.sets:
                    units.append(Unit(sm, key, self.sets[key]["path"], self.sets[key]["ids"], False,
                                      f"{self.cfg.label}__{sm.name}__{key}"))
        return units

    def rho_sensitivity_entries(self) -> list[MethodEntry]:
        """P3-23 (PI decision after the p6 handover, item e): when real arms run, the methods listed in
        real_data_arms.rho_sensitivity.methods (default: the primary candidate) also run with the pooled
        per-cohort-pair rho (`rho_hat_pair`) and with rho = 0 (eval.reference_methods.RhoOverride). Reported as
        sensitivities in real_arms.json, never as separate methods."""
        if not self.real_spec:
            return []
        cfg = self.real_spec.get("rho_sensitivity", {}) or {}
        if cfg is False or cfg.get("enabled", True) is False:
            return []
        prim = (self.doc.get("candidates") or {}).get("primary")
        names = [prim if n == "primary_candidate" else n for n in cfg.get("methods", ["primary_candidate"])]
        names = [n for n in names if n]          # primary: null -> no sensitivity runs of a candidate
        modes = list(cfg.get("modes", ["pooled", "zero"]))
        out = []
        by = {m.name: m for m in self.methods}
        for n in names:
            m = by.get(n)
            if m is None:
                continue
            for mode in modes:
                e = MethodEntry(name=f"{m.name}_rho{mode}", target="eval.reference_methods:RhoOverride",
                                kwargs={"target": m.target, "kwargs": dict(m.kwargs), "mode": mode,
                                        "name": f"{m.name}_rho{mode}"},
                                mode=m.mode, n_workers=m.n_workers, batch_size=m.batch_size, timeout_s=m.timeout_s,
                                gpu=m.gpu, mem_gb_per_worker=m.mem_gb_per_worker,
                                expected_s_per_locus=m.expected_s_per_locus, metric_config=dict(m.metric_config),
                                real=True)
                self.sens_base[e.name] = m.name
                out.append(e)
        return out

    def _run_unit(self, u: Unit, n_workers: int | None = None) -> None:
        run_dir = self.runs_dir / u.run_id
        if (run_dir / "manifest.json").exists():
            self._set_unit(u.run_id, status="done", run_dir=str(run_dir))
            return
        if run_dir.exists():  # crash while the run directory was being written: keep it aside, redo from checkpoint
            run_dir.rename(run_dir.with_name(run_dir.name + f".incomplete_{int(time.time())}"))
        m = u.method
        kwargs = dict(m.kwargs)
        if m.name == CONTROL and self.cfg.mode == "dry":
            kwargs["table"] = str(self.dry_strength_table())
        elif m.name == CONTROL:
            kwargs["table"] = str((self.repo / kwargs["table"]).resolve())
        logf = open(self.log_dir / f"{u.run_id}.log", "a")

        def ulog(s):
            logf.write(f"{datetime.now().strftime('%H:%M:%S')} {s}\n")
            logf.flush()
        t0 = time.time()
        self._set_unit(u.run_id, status="running", start_utc=_now(), n=len(u.ids), method=m.name, set=u.set_key)
        try:
            r = run_method(MethodSpec(m.target, kwargs, m.name), u.set_path, results_dir=self.runs_dir,
                           run_id=u.run_id, mode=m.mode,
                           n_workers=(1 if m.gpu else int(n_workers or m.n_workers)),
                           batch_size=m.batch_size, timeout_s=m.timeout_s, ids=u.ids,
                           truth_path="auto" if u.sim else None,
                           metric_config={k: v for k, v in m.metric_config.items() if v is not None} or None,
                           label=f"{'TEST' if self.cfg.mode == 'test' else 'DRY'} {self.cfg.label}",
                           config=dict(self.extra_manifest, unit=u.run_id, set_key=u.set_key, mode=self.cfg.mode,
                                       subsample_per_class=m.subsample_per_class),
                           log=ulog, checkpoint_dir=self.ckpt_dir, order="given" if m.mode == "batch" else "lpt")
            self._set_unit(u.run_id, status="done", end_utc=_now(), wall_s=time.time() - t0, counts=r.counts,
                           run_dir=str(r.run_dir), error=None)
            self.log(f"unit done {u.run_id}: {r.counts} in {time.time() - t0:.0f}s")
        except Exception as e:  # noqa: BLE001 - recorded; retried by the scheduler; reported in the bundle
            tb = traceback.format_exc()
            ulog(tb)
            prev = self.unit_status.get(u.run_id, {}).get("attempts", 0)
            self._set_unit(u.run_id, status="failed", end_utc=_now(), error=f"{type(e).__name__}: {e}"[:2000],
                           attempts=prev + 1)
            self.log(f"unit FAILED {u.run_id}: {type(e).__name__}: {str(e)[:300]}")
        finally:
            logf.close()

    def run_units(self, source: "queue.Queue[Unit]", planning_done: threading.Event, stop: threading.Event) -> None:
        """Scheduler. Units arrive on `source` (real-arm units first, sim units once the sets exist) until
        `planning_done`. CPU units run in parallel under the core / RAM budget, largest expected cost first; a unit
        whose full worker count does not fit starts with fewer workers once at least half of them are free (so
        the longest units are not starved by back-filling). GPU units run one at a time and never while the
        generator holds the GPU. A failed unit is retried `retry_failed_units` times after the others."""
        cores = max(1, self.cfg.cores - 1)        # one core for the driver
        ram = float(self.cfg.ram_gb)
        pending: list[Unit] = []
        seen: set[str] = set()
        attempts: dict[str, int] = {}
        running: dict[str, tuple[threading.Thread, Unit, int, float]] = {}
        gpu_busy = False
        last_status = 0.0
        while not stop.is_set():
            while True:
                try:
                    u = source.get_nowait()
                except queue.Empty:
                    break
                if u.run_id in seen:
                    continue
                seen.add(u.run_id)
                if (self.runs_dir / u.run_id / "manifest.json").exists():
                    self._set_unit(u.run_id, status="done", run_dir=str(self.runs_dir / u.run_id))
                    continue
                attempts[u.run_id] = 0
                pending.append(u)
                pending.sort(key=lambda x: -x.cost)
            for rid, (th, u, k, mem) in list(running.items()):
                if not th.is_alive():
                    del running[rid]
                    if u.method.gpu:
                        gpu_busy = False
                    if self.unit_status.get(rid, {}).get("status") == "failed" and \
                            attempts[rid] <= self.cfg.retry_failed_units:
                        pending.append(u)   # retried after the units already queued
            if not pending and not running and planning_done.is_set() and source.empty():
                break
            used_c = sum(k for _, _, k, _ in running.values())
            used_m = sum(mm for _, _, _, mm in running.values())
            gen_c = self.cfg.gen_workers if self.gen_running.is_set() else 0
            for u in list(pending):
                if not self.ready[u.set_key].is_set():
                    continue
                m = u.method
                if m.gpu:
                    if gpu_busy or self.gen_running.is_set():
                        continue
                    k, mem = 1, m.mem_gb_per_worker
                    if (used_c + gen_c + 1 > cores or used_m + mem > ram) and (running or gen_c):
                        continue
                else:
                    want = min(m.n_workers, cores)
                    free_c = cores - used_c - gen_c
                    free_by_mem = int((ram - used_m) // max(m.mem_gb_per_worker, 1e-9))
                    k = min(want, free_c, free_by_mem)
                    if k < want and (k < max(1, want // 2)) and (running or gen_c):
                        continue
                    k = max(1, k)
                    mem = k * m.mem_gb_per_worker
                attempts[u.run_id] += 1
                th = threading.Thread(target=self._run_unit, args=(u, k), name=u.run_id, daemon=True)
                th.start()
                running[u.run_id] = (th, u, k, mem)
                used_c += k
                used_m += mem
                if m.gpu:
                    gpu_busy = True
                pending.remove(u)
                self.log(f"start {u.run_id}: {len(u.ids)} loci, {'GPU' if m.gpu else f'{k} workers'} "
                         f"(busy {used_c + gen_c}/{cores} cores, {used_m:.0f}/{ram:.0f} GB; {len(pending)} pending)")
            if time.time() - last_status > 600:
                last_status = time.time()
                self.log(f"status: running {sorted(running)}; pending {[u.run_id for u in pending]}")
            time.sleep(2)
        for th, *_ in running.values():
            th.join()

    # ---------------------------------------------------------------- stage 4: evaluation
    def operating_points(self) -> tuple[dict, str]:
        """Per-method {t_S, t_D, recal_a_S, recal_a_D}. Test: the frozen metric_config of the methods file.
        Dry: fitted on the dry sim set (C3 rule, realistic, least favourable) -- DRY placeholders."""
        if self.cfg.mode == "test":
            ops = {}
            for m in self.methods:
                mc = m.metric_config
                ops[m.name] = {k: mc.get(k) for k in ("t_S", "t_D", "recal_a_S", "recal_a_D")}
                ops[m.name]["setting"] = "realistic"
            return ops, "frozen (methods file metric_config)"
        from eval import metrics as M
        sys.path.insert(0, str(self.repo / "scripts"))
        import run_benchmark as RB
        truth = self.sim_truth()
        preds = self._sim_preds([m.name for m in self.methods if not m.subsample_per_class])
        fr = M.build_eval_frame(truth, preds, "least_favourable")
        ops = {k: v["fitted_here"] for k, v in RB._op_points(fr, sorted(preds), "realistic", None).items()}
        # slow methods: fitted on their subsample frame
        for m in self.methods:
            if m.subsample_per_class:
                p = self._sim_preds([m.name])
                t = truth[truth["locus_id"].astype(str).isin(set(p[m.name]["locus_id"].astype(str)))]
                f2 = M.build_eval_frame(t, p, "least_favourable")
                ops[m.name] = RB._op_points(f2, [m.name], "realistic", None)[m.name]["fitted_here"]
        for v in ops.values():
            v["DRY"] = True
        return M._clean(ops), "DRY placeholders fitted in-sample on the dry-run set (not validation values)"

    def _sim_preds(self, names: list[str]) -> dict[str, pd.DataFrame]:
        out = {}
        for n in names:
            parts = []
            for key in ("sim_main", "sim_shifted"):
                d = self.runs_dir / f"{self.cfg.label}__{n}__{key}"
                if (d / "manifest.json").exists():
                    parts.append(pd.read_parquet(d / "predictions.parquet"))
            if parts:
                out[n] = pd.concat(parts, ignore_index=True)
        return out

    def _run_dirs(self, name: str, keys) -> dict[str, str]:
        out = {}
        for key in keys:
            d = self.runs_dir / f"{self.cfg.label}__{name}__{key}"
            if (d / "manifest.json").exists():
                out[str(self.sets[key]["path"])] = str(d)
        return out

    def stage_evaluate(self) -> dict:
        sys.path.insert(0, str(self.repo / "scripts"))
        import run_benchmark as RB
        ops, ops_source = self.operating_points()
        ops_path = self.bundle / ("operating_points_used.json" if self.cfg.mode == "test"
                                  else "DRY_operating_points_in_sample.json")
        _atomic_json(ops_path, ops)
        if self.cfg.mode == "test":
            tab = next(m for m in self.methods if m.name == CONTROL).kwargs["table"]
            table = (self.repo / tab).resolve()
            table_mode = "given (frozen)"
        else:
            table = self.dry_strength_table()
            table_mode = "DRY in-sample placeholder"
        subs = self.slow_subsamples()
        sim_keys = ("sim_main", "sim_shifted")
        set_paths = [str(self.sets[k]["path"]) for k in sim_keys]
        prim = (self.doc.get("candidates") or {}).get("primary")
        c3b = list(self.doc.get("c3_eligible_baselines") or [])
        desc = {m.name for m in self.methods if m.descriptive}
        bdir = self.bundle / "benchmarks" / "sim"
        bdir.mkdir(parents=True, exist_ok=True)
        names = [m.name for m in self.methods]
        runs = {n: r for n, r in ((n, self._run_dirs(n, sim_keys)) for n in names) if r}
        not_run = sorted(set(names) - set(runs))
        ids, ids_sha = [], {}
        for k in sim_keys:
            ids += self.sets[k]["ids"]
            ids_sha[str(self.sets[k]["path"])] = _ids_hash(self.sets[k]["ids"])
        (bdir / "expected_ids.txt").write_text("\n".join(ids) + "\n")
        method_ids = {}
        for m in self.methods:   # per-method id lists = the slow-method subsamples (run_benchmark --method-ids)
            if m.subsample_per_class and m.name in runs:
                f = bdir / f"ids_subsample{m.subsample_per_class}.txt"
                f.write_text("\n".join(subs[m.subsample_per_class]) + "\n")
                method_ids[m.name] = {"file": str(f.resolve()), "sha256": _ids_hash(sorted(subs[m.subsample_per_class]))}
        spec = {"bench_id": f"{self.cfg.label}__sim", "label": f"{self.cfg.mode} sim",
                "locus_sets": set_paths, "ids_sha256": ids_sha,
                "n_expected": {str(self.sets[k]["path"]): len(self.sets[k]["ids"]) for k in sim_keys},
                "per_cell": None, "runs": runs, "method_ids": method_ids, "methods_not_run": not_run,
                "strength_control": {"table": str(table), "mode": table_mode, "sha256": sha256_file(table),
                                     "folds": None},
                "operating_points_in": str(ops_path.resolve()), "operating_points_in_sha256": sha256_file(ops_path),
                "operating_points_source": ops_source,
                "candidate": prim if (prim in runs and prim not in desc) else None,
                "c3_baselines": [b for b in c3b if b in runs and b not in desc] or None,
                "n_boot": self.cfg.n_boot, "unit": self.cfg.unit,
                "boot_seed": [RB.BOOT_SEED, "run_benchmark", "bootstrap", f"{self.cfg.label}__sim"],
                "n_jobs": self.cfg.n_jobs, "git": git_info(), "argv": sys.argv, "start_utc": _now(),
                "mode": self.cfg.mode}
        _atomic_json(bdir / "manifest.json", spec)
        t0 = time.time()
        self.log(f"evaluate sim benchmark: {len(ids)} loci, {len(runs)} methods, per-method id lists {sorted(method_ids)}")
        RB.evaluate_all(spec, bdir)
        spec["evaluate_wall_s"] = time.time() - t0
        c4s, benches = {}, {}
        for rel, sp, keep in RB._sub_benchmarks(spec):
            d = bdir / rel if rel else bdir
            c4 = self.c4_block(d, sp, ops)
            _atomic_json(d / "c4.json", c4)
            c4s[rel or "."] = sha256_file(d / "c4.json")
            benches[rel or "main"] = {"dir": str(d), "n_loci": len((d / "expected_ids.txt").read_text().split()),
                                      "methods": sorted(sp["runs"]), "benchmark_sha256": sha256_file(d / "benchmark.json")}
        spec["c4_sha256"] = c4s
        _atomic_json(bdir / "manifest.json", spec)
        benches = {"dir": str(bdir), "not_run": not_run, "evaluate_wall_s": spec["evaluate_wall_s"], "parts": benches}
        claims = self.claims_block(prim, spec)
        _atomic_json(self.bundle / "claims.json", claims)
        descr = self.descriptive_block(bdir, spec, ops, desc)
        _atomic_json(self.bundle / "descriptive.json", descr)
        real = self.real_summary(ops) if any(k in self.sets for k in REAL_SETS) else None
        if real is not None:
            _atomic_json(self.bundle / "real_arms.json", real)
        rt = self.runtime_table()
        _atomic_json(self.bundle / "runtime.json", rt)
        out = {"claims": str(self.bundle / "claims.json"), "descriptive": str(self.bundle / "descriptive.json"),
               "benchmark": benches, "operating_points": str(ops_path), "operating_points_source": ops_source,
               "strength_table": str(table), "real_arms": str(self.bundle / "real_arms.json") if real else None,
               "runtime": str(self.bundle / "runtime.json")}
        self._mark("evaluate", out)
        return out

    def claims_block(self, prim, spec: dict) -> dict:
        """C1-C4 status. With `candidates.primary: null` (D16, MVR route) every candidate claim is 'no candidate':
        the run is the C5 benchmark (per-cell / pooled metrics of the baselines, false-S by mismatch level, r², d
        and rho, runtimes) plus the pre-declared descriptive entries."""
        dec = (self.doc.get("candidates") or {}).get("decision")
        if not prim:
            msg = f"no candidate ({dec or 'candidates.primary is null'})"
            return {"primary": None, "decision": dec, "C1": msg, "C2": msg, "C3": msg, "C4": msg,
                    "C5": "benchmark of every listed method (benchmarks/sim, c4.json slopes per method, runtime.json)",
                    "descriptive_entries": sorted(m.name for m in self.methods if m.descriptive)}
        return {"primary": prim, "decision": dec, "C1": "benchmarks/sim (vs_candidate, every baseline)",
                "C3": "benchmarks/sim c3_dry_run (given thresholds)", "C4": "c4.json (candidate slopes)",
                "descriptive_entries": sorted(m.name for m in self.methods if m.descriptive)}

    def descriptive_block(self, bdir: Path, spec: dict, ops: dict, desc: set) -> dict:
        """D16: pre-declared descriptive entries (no claims). Three-way calls (unweighted confusion, true class x
        call) and weighted recalls at BOTH the matched-W operating point in use and the fixed 0.8 / 0.8
        thresholds, least-favourable failures, on the main benchmark (realistic setting) -- and the same for every
        other method in the benchmark for comparison."""
        from eval import metrics as M
        sys.path.insert(0, str(self.repo / "scripts"))
        import run_benchmark as RB
        main = dict(spec, runs={m: r for m, r in spec["runs"].items() if m not in (spec.get("method_ids") or {})})
        truth, preds, _ = RB._load_truth_and_preds(main, bdir)
        fr = M.build_eval_frame(truth, preds, "least_favourable")
        if len(fr) == 0:
            return {"definition": self.descriptive_block.__doc__.strip(), "descriptive": sorted(desc), "methods": {},
                    "note": "not estimable: n=0 loci in the main benchmark"}
        st = "realistic" if M.cell_pooling_weights(fr, "realistic") else "all"
        out = {"definition": self.descriptive_block.__doc__.strip(), "descriptive": sorted(desc), "setting": st,
               "methods": {}}
        for m in sorted(preds):
            if np.isnan(fr.scores[m]["P_S"]).all() and np.isnan(fr.scores[m]["P_D"]).all():
                continue
            o = ops.get(m, {})
            tS = np.inf if o.get("t_S") is None else float(o["t_S"])
            tD = np.inf if o.get("t_D") is None else float(o["t_D"])
            out["methods"][m] = {
                "descriptive": m in desc,
                "matched_w": {"t_S": tS, "t_D": tD, "confusion": M.confusion(fr, m, tS, tD),
                              "recall": M.recall_at(fr, m, tS, tD, st)},
                "fixed_0.8": {"t_S": 0.8, "t_D": 0.8, "confusion": M.confusion(fr, m, 0.8, 0.8),
                              "recall": M.recall_at(fr, m, 0.8, 0.8, st)}}
        return M._clean(out)

    def c4_block(self, bdir: Path, spec: dict, ops: dict) -> dict:
        """C4 (configs/testset_v1.yaml endpoints.C4): slope of the false-S rate (P_S >= t_S) on the realised rho,
        per 0.1, in P2 and P3 cells of the realistic setting (levels L1-L6), among D and among W loci; BCa 95% CI
        from one cluster bootstrap (test LD block); margin 0.02 (|slope| CI within +-0.02 = no overlap effect)."""
        from eval import metrics as M
        sys.path.insert(0, str(self.repo / "scripts"))
        import run_benchmark as RB
        truth, preds, _ = RB._load_truth_and_preds(spec, bdir)
        out = {"definition": self.c4_block.__doc__.strip(), "modes": {}}
        realistic = set(self.testset["pooling"]["realistic"]["levels"])
        for mode in ("least_favourable", "excluded"):
            fr = M.build_eval_frame(truth, preds, mode)
            lv = np.array([c.split("|")[0] for c in fr.cell])
            pt = np.array([c.split("|")[1] if "|" in c else "" for c in fr.cell])
            labels, cols = [], []
            for m in sorted(preds):
                if np.isnan(fr.scores[m]["P_S"]).all():
                    continue
                t_S = ops.get(m, {}).get("t_S")
                t_S = np.inf if t_S is None else float(t_S)
                for P in ("P2", "P3"):
                    for cls in ("D", "W"):
                        labels.append((m, P, cls, t_S))
            if not labels:
                continue
            sel = np.nonzero(np.isin(lv, list(realistic)) & np.isin(pt, ["P2", "P3"]))[0]
            if len(sel) == 0:
                out["modes"][mode] = {"n_loci": 0, "note": "not estimable: n=0 realistic P2/P3 loci in this (sub-)benchmark"}
                continue
            sub = fr.take(sel)
            pt_sub = pt[sel]
            n_units = len(np.unique(M.resampling_units(sub, self.cfg.unit)))
            if n_units < 2:   # the leave-one-unit-out jackknife (BCa) needs at least 2 bootstrap units
                out["modes"][mode] = {"n_loci": int(len(sel)), "n_units": int(n_units),
                                      "note": "not estimable: fewer than 2 bootstrap units among realistic P2/P3 loci"}
                continue

            def stat(idx, sub=sub, pt_sub=pt_sub):
                f = sub.take(idx)
                p = pt_sub[idx]
                return np.array([M.false_s_slope(f.mask(p == P), m, cls, t) for m, P, cls, t in labels])
            g = seeded_rng(RB.BOOT_SEED if self.cfg.mode == "dry" else self.run_seed, "eval_test", "c4", spec["bench_id"], mode)
            r = M.cluster_bootstrap(stat, M.resampling_units(sub, self.cfg.unit), n_boot=self.cfg.n_boot, rng=g,
                                    n_jobs=self.cfg.n_jobs)
            est, lo, hi = (np.atleast_1d(r[k]) for k in ("estimate", "ci_low", "ci_high"))
            rows = {}
            for i, (m, P, cls, t) in enumerate(labels):
                rows.setdefault(m, {}).setdefault(P, {})[cls] = {
                    "slope_per_0.1": est[i], "ci_low": lo[i], "ci_high": hi[i], "t_S": t,
                    "within_margin_0.02": bool(np.isfinite(lo[i]) and lo[i] > -0.02 and hi[i] < 0.02)}
            out["modes"][mode] = {"n_loci": int(len(sel)), "n_boot_nan": r.get("n_boot_nan"), "slopes": rows}
        return M._clean(out)

    def real_summary(self, ops: dict) -> dict:
        """DESIGN §2.5 pre-registered real-data criterion (descriptive in the dry run): per method, t_R3b = the
        threshold giving a 5% confident-S rate on R3b (matched_threshold on P_S, alpha 0.05); S recall on R2 at
        t_R3b; R4b agreement = share of loci whose three-way call (frozen t_S/t_D, E gate) is the same with
        reference LD as with in-sample LD (calls recomputed here from the operating points in use). Failures are least favourable (R3b: P_S = 1, R2: P_S = 0, R4b:
        disagreement). CIs: cluster bootstrap (2,000) over each locus set's boot_unit attribute (R2/R4b block,
        R3b FinnGen CS; D6(i)), arms resampled independently; candidate minus best baseline (by point estimate)."""
        from eval import metrics as M
        self.rho_sensitivity_entries()   # fills self.sens_base also when this invocation planned no units
        listed = {m.name for m in self.methods} | set(self.sens_base)
        names = sorted({u.split("__")[1] for u in os.listdir(self.runs_dir) if "__real_" in u and
                        u.split("__")[1] in listed and
                        (self.runs_dir / u / "manifest.json").exists() and u.startswith(self.cfg.label + "__")})
        data = {}
        for key in REAL_SETS:
            if key not in self.sets:
                continue
            attrs = read_locus_attrs(self.sets[key]["path"])
            attrs = attrs[attrs["locus_id"].astype(str).isin(set(self.sets[key]["ids"]))]
            unit_col = next((c for c in ("boot_unit", "gene_id") if c in attrs), "locus_id")
            clus = dict(zip(attrs["locus_id"].astype(str), attrs[unit_col].astype(str)))
            ids = list(self.sets[key]["ids"])
            per = {}
            for n in names:
                d = self.runs_dir / f"{self.cfg.label}__{n}__{key}"
                if not (d / "manifest.json").exists():
                    continue
                p = pd.read_parquet(d / "predictions.parquet").set_index("locus_id").reindex(ids)
                ok = p["status"].isin(["ok", "nonconverged"]).to_numpy() & np.isfinite(p["P_S"].to_numpy(float))
                op = ops.get(n) or ops.get(self.sens_base.get(n, ""), {})
                t_S = np.inf if op.get("t_S") is None else float(op["t_S"])
                t_D = np.inf if op.get("t_D") is None else float(op["t_D"])
                call = M.calls_from_arrays(*(p[c].to_numpy(float) for c in ("P_S", "P_D", "E1", "E2")), t_S, t_D,
                                           failed=~ok)
                per[n] = {"P_S": p["P_S"].to_numpy(float), "call": np.asarray(call).astype(str), "ok": ok}
            data[key] = {"ids": ids, "clusters": np.array([clus.get(i, i) for i in ids]), "per": per}
        methods = sorted(set.intersection(*[set(v["per"]) for v in data.values()])) if data else []
        methods = [m for m in methods if not all(np.isnan(data["real_R3b"]["per"][m]["P_S"]))] if "real_R3b" in data else []

        def stats(i2, i3, i4):
            out = []
            for m in methods:
                r3, r2 = data["real_R3b"]["per"][m], data["real_R2"]["per"][m]
                s3 = np.where(r3["ok"], r3["P_S"], 1.0)[i3]
                t = M.matched_threshold(s3, alpha=0.05)["t"]
                s2 = np.where(r2["ok"], r2["P_S"], 0.0)[i2]
                out.append(float(np.mean(s2 >= t)) if len(s2) else np.nan)
                a, b = data["real_R4b_insample"]["per"][m], data["real_R4b_reference"]["per"][m]
                agree = (a["call"] == b["call"]) & a["ok"] & b["ok"]
                out.append(float(np.mean(agree[i4])) if len(i4) else np.nan)
            return np.array(out)
        need = ("real_R2", "real_R3b", "real_R4b_insample", "real_R4b_reference")
        if not methods or any(k not in data for k in need):
            return {"definition": self.real_summary.__doc__.strip(), "methods": methods,
                    "note": "not all four real-data sets present"}
        empty = [k for k in need if len(data[k]["ids"]) == 0]
        if empty:
            return {"definition": self.real_summary.__doc__.strip(), "methods": methods,
                    "note": f"not estimable: n=0 loci in {empty}"}
        idx = {k: np.arange(len(data[k]["ids"])) for k in need}
        est = stats(idx["real_R2"], idx["real_R3b"], idx["real_R4b_insample"])

        def groups(key):
            c = data[key]["clusters"]
            u, inv = np.unique(c, return_inverse=True)
            return [np.nonzero(inv == j)[0] for j in range(len(u))]
        G = {k: groups(k) for k in ("real_R2", "real_R3b", "real_R4b_insample")}
        g = seeded_rng(self.run_seed, "eval_test", "real_bootstrap")
        B = self.cfg.n_boot if min(len(v) for v in G.values()) >= 2 else 0   # < 2 clusters: no CI (NaN, reason)
        reps = np.full((max(B, 1), len(est)), np.nan)
        for b in range(B):
            ii = {k: np.concatenate([G[k][j] for j in g.integers(0, len(G[k]), len(G[k]))]) for k in G}
            reps[b] = stats(ii["real_R2"], ii["real_R3b"], ii["real_R4b_insample"])
        prim = (self.doc.get("candidates") or {}).get("primary")
        res = {"definition": self.real_summary.__doc__.strip(), "n": {k: len(data[k]["ids"]) for k in need},
               "n_clusters": {k: len(v) for k, v in G.items()}, "n_boot": B, "methods": {}}
        if B == 0:
            res["ci_note"] = "CIs not estimable: fewer than 2 bootstrap units in an arm"
        for j, m in enumerate(methods):
            fails = {k: int((~data[k]["per"][m]["ok"]).sum()) for k in need}
            res["methods"][m] = {
                "recall_S_R2_at_5pct_R3b": {"est": est[2 * j], "ci_low": float(np.nanpercentile(reps[:, 2 * j], 2.5)),
                                            "ci_high": float(np.nanpercentile(reps[:, 2 * j], 97.5))},
                "R4b_call_agreement": {"est": est[2 * j + 1],
                                       "ci_low": float(np.nanpercentile(reps[:, 2 * j + 1], 2.5)),
                                       "ci_high": float(np.nanpercentile(reps[:, 2 * j + 1], 97.5))},
                "failures": fails}
        for m in methods:
            if m in self.sens_base:
                res["methods"][m]["P3_23_sensitivity_of"] = self.sens_base[m]
                res["methods"][m]["rho_mode"] = m.rsplit("_rho", 1)[1]
        if prim in methods:
            jp = methods.index(prim)
            base = [m for m in methods if m != prim and m != CONTROL and m not in self.sens_base]
            for off, key in ((0, "recall_S_R2_at_5pct_R3b"), (1, "R4b_call_agreement")):
                if not base:
                    break
                best = max(base, key=lambda m: est[2 * methods.index(m) + off])
                jb = methods.index(best)
                diff = reps[:, 2 * jp + off] - reps[:, 2 * jb + off]
                res.setdefault("candidate_vs_best_baseline", {})[key] = {
                    "candidate": prim, "best_baseline": best, "diff": est[2 * jp + off] - est[2 * jb + off],
                    "ci_low": float(np.nanpercentile(diff, 2.5)), "ci_high": float(np.nanpercentile(diff, 97.5))}
        return M._clean(res)

    def runtime_table(self) -> dict:
        """Per (method, set): loci, workers, harness wall, worker-seconds (sum of per-locus runtime_s incl. LD
        read), per-1,000-loci figures, failures, resumed segments."""
        rows = []
        for d in sorted(self.runs_dir.iterdir()):
            if not (d / "manifest.json").exists() or not d.name.startswith(self.cfg.label + "__"):
                continue
            man = json.loads((d / "manifest.json").read_text())
            _, m, key = d.name.split("__")
            n = man["counts"]["expected"]
            wsum = man["runtime_s"]["sum"]
            nw = man["harness"]["n_workers"]
            segs = (man.get("checkpoint") or {}).get("segments") or []
            rows.append({"method": m, "set": key, "n": n, "mode": man["harness"]["mode"], "n_workers": nw,
                         "wall_s": man["wall_s"], "worker_s": wsum,
                         "worker_s_per_1000": 1000 * wsum / n if n else None,
                         "wall_s_per_1000": 1000 * man["wall_s"] / n if n else None,
                         "median_s": man["runtime_s"]["median"], "p95_s": man["runtime_s"]["p95"],
                         "counts": man["counts"], "segments": len(segs),
                         "rows_from_checkpoint": (man.get("checkpoint") or {}).get("rows_from_checkpoint", 0)})
        return {"rows": rows, "stages": {k: self._stage_done(k) and {kk: vv for kk, vv in self._stage_done(k).items()
                                                                       if "wall" in kk or kk == "segments"}
                                         for k in ("real", "sim")}}

    # ---------------------------------------------------------------- stage 5: verification
    def stage_verify(self) -> dict:
        sys.path.insert(0, str(self.repo / "scripts"))
        import run_benchmark as RB
        import verify_run
        ev = self._stage_done("evaluate")
        problems = {}
        bdir = Path(ev["benchmark"]["dir"])
        self.log("verify sim benchmark (every harness run re-verified, every (sub-)benchmark recomputed)")
        p = RB.verify(bdir)
        spec = json.loads((bdir / "manifest.json").read_text())
        for rel, h in (spec.get("c4_sha256") or {}).items():
            if h != sha256_file((bdir / rel) / "c4.json"):
                p.append(f"{rel}/c4.json sha256 differs from the benchmark manifest")
        problems["benchmark"] = p
        real_runs = [d for d in self.runs_dir.iterdir() if "__real_" in d.name and (d / "manifest.json").exists()]
        problems["real_runs"] = [f"{d.name}: {x}" for d in sorted(real_runs) for x in verify_run.verify(d)]
        for f in ("claims.json", "descriptive.json"):
            if not (self.bundle / f).exists():
                problems.setdefault("bundle_files", []).append(f"{f} missing")
        # every sim run is covered by a benchmark verify; check none is orphaned
        covered = set()
        for by in spec["runs"].values():
            covered |= {Path(x).name for x in by.values()}
        listed = {m.name for m in self.methods}
        orphan = [d.name for d in self.runs_dir.iterdir() if (d / "manifest.json").exists() and "__sim_" in d.name
                  and d.name.split("__")[1] in listed and d.name not in covered]
        problems["orphan_sim_runs"] = orphan
        listed_units = {u for u in self.unit_status if u.split("__")[1] in listed | set(self.sens_base)}
        failed_units = {k: v for k, v in self.unit_status.items() if k in listed_units and v.get("status") != "done"}
        problems["units_not_done"] = sorted(failed_units)
        out = {"problems": problems, "ok": not any(problems.values())}
        self._mark("verify", out)
        return out

    def input_hashes(self) -> dict:
        """sha256 of every generated / real input file (part files hashed in parallel)."""
        files = []
        for k, s in self.sets.items():
            files.append(Path(s["path"]))
            tp = truth_path_for(s["path"])
            if tp.exists():
                files.append(tp)
            if s["sim"]:
                part = Path(s["path"]).with_suffix("")
                files += sorted(part.glob("*.h5")) + sorted(part.glob("*.parquet"))
        files = sorted(set(files))
        with cf.ProcessPoolExecutor(min(16, max(1, len(files)))) as ex:
            hs = list(ex.map(sha256_file, files))
        return {str(f): h for f, h in zip(files, hs)}

    # ---------------------------------------------------------------- driver
    def run(self, generate: Callable) -> dict:
        """Stages in order; the scheduler thread runs method units while the main thread generates (real-arm
        units of CPU methods start during generation)."""
        t0 = time.time()
        stop = threading.Event()
        if "real" in self.cfg.stages:
            self.stage_real()
        else:
            st = self._stage_done("real") or {"sets": {}}
            for k, v in st["sets"].items():
                self.sets[k] = {"path": Path(v["path"]), "ids": v["ids"], "sim": False}
                self.ready[k].set()
        if set(self.cfg.stages) == {"real"}:
            self.log("stages = real only: done")
            return {"sets": {k: {"path": str(v["path"]), "n": len(v["ids"])} for k, v in self.sets.items()},
                    "verify": {"ok": True, "problems": {}}, "evaluate": {"benchmark": {"parts": {}}}}
        q: "queue.Queue[Unit]" = queue.Queue()
        planning_done = threading.Event()
        sched_err: list = []

        def scheduler():
            try:
                self.run_units(q, planning_done, stop)
            except BaseException as e:  # noqa: BLE001
                sched_err.append(e)
                self.log(f"scheduler error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        sch = None
        if "methods" in self.cfg.stages:
            sch = threading.Thread(target=scheduler, name="scheduler", daemon=True)
            if self._stage_done("sim") is None and "sim" in self.cfg.stages:
                self.gen_running.set()   # reserve the generator's cores / the GPU before anything starts
            sch.start()
            # real-arm units of CPU methods can run during generation (the control's dry table needs the sim set)
            for m in self.methods:
                if m.real and not m.gpu and not (m.name == CONTROL and self.cfg.mode == "dry"):
                    for k in REAL_SETS:
                        if k in self.sets:
                            q.put(Unit(m, k, self.sets[k]["path"], self.sets[k]["ids"], False,
                                       f"{self.cfg.label}__{m.name}__{k}"))
        try:
            if "sim" in self.cfg.stages:
                self.stage_sim(generate)
            else:
                st = self._stage_done("sim")
                for key, p in (("sim_main", st["main"]), ("sim_shifted", st["shifted"])):
                    self.sets[key] = {"path": Path(p), "ids": read_locus_attrs(p)["locus_id"].astype(str).tolist(),
                                      "sim": True}
                    self.ready[key].set()
            self.gen_running.clear()
            if sch is not None:
                units = self.plan_units()
                _atomic_json(self.state_dir / "plan.json", [
                    {"run_id": u.run_id, "method": u.method.name, "set": u.set_key, "n": len(u.ids),
                     "gpu": u.method.gpu, "ids_sha256": _ids_hash(u.ids), "expected_cost_s": u.cost} for u in units])
                for u in units:
                    q.put(u)
                planning_done.set()
                while sch.is_alive():
                    sch.join(timeout=5)
        except BaseException:
            stop.set()
            self.gen_running.clear()
            raise
        if sched_err:
            raise sched_err[0]
        listed = {m.name for m in self.methods} | set(self.sens_base)
        not_done = {k: v for k, v in self.unit_status.items()
                    if k.split("__")[1] in listed and v.get("status") != "done"}
        if not_done and not self.cfg.accept_failed_units:
            raise PipelineIncomplete(f"{len(not_done)} unit(s) not done: " + "; ".join(
                f"{k}: {v.get('error')}" for k, v in sorted(not_done.items()))[:3000])
        ev = self.stage_evaluate() if "evaluate" in self.cfg.stages else self._stage_done("evaluate")
        ver = self.stage_verify() if "verify" in self.cfg.stages else None
        bundle = {"mode": self.cfg.mode, "label": self.cfg.label, "scale": self.cfg.scale,
                  "DRY": self.cfg.mode != "test", "config": {k: (str(v) if isinstance(v, Path) else v)
                                                             for k, v in self.cfg.__dict__.items()},
                  "extra": self.extra_manifest, "git": git_info(), "sets": {
                      k: {"path": str(v["path"]), "n": len(v["ids"]), "ids_sha256": _ids_hash(v["ids"])}
                      for k, v in self.sets.items()},
                  "units": self.unit_status, "evaluate": ev, "verify": ver,
                  "input_sha256": self.input_hashes(), "wall_s_this_invocation": time.time() - t0,
                  "end_utc": _now()}
        _atomic_json(self.bundle / "bundle.json", bundle)
        append_ledger(self.repo / "results", {
            "run_id": f"eval_test:{self.cfg.label}", "start_utc": "", "end_utc": _now(), "method": "eval_test bundle",
            "locus_set": ";".join(str(v["path"]) for v in self.sets.values()),
            "n_expected": sum(len(v["ids"]) for v in self.sets.values()), "git_commit": bundle["git"]["commit"],
            "git_clean": bundle["git"]["clean"], "timing": False, "label": f"{self.cfg.mode} {self.cfg.label}",
            "run_dir": str(self.bundle)})
        self.log(f"bundle written: {self.bundle / 'bundle.json'} (verify ok: {None if ver is None else ver['ok']})")
        return bundle
