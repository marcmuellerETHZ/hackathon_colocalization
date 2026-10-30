"""Harness integration of the tier-1 baselines (INTERFACES §4; DESIGN §4.3, §7.7; decision D6(f)).

Every baseline is exposed as an `eval.harness.MethodSpec` (built inside each harness worker process, so each worker
owns its own persistent R process) together with its per-locus timeout and a memory model that limits the number
of parallel workers (DESIGN §7.7: worker count limited by memory, <= 2 GB/core budget; susieR/mvSuSiE fewer).

    from baselines.run import run_baseline
    r = run_baseline("coloc_susie", "data/processed/sets/screen_v2.h5", ids=ids, mem_budget_gb=48, max_cpus=24)

CLI:  python -m baselines.run --method coloc_susie --set <locus_set.h5> [--ids-file f] [--mem-budget-gb 48]
      [--max-cpus 24] [--timeout-s T] [--label L] [--results-dir results]

Timeouts: `TIMEOUT_S` is the per-locus wall-clock limit enforced by the harness (it kills the worker and its R
process group). The wrapper's own R-call timeout is set to R_TIMEOUT_FRACTION of it, so a slow R call normally
ends as a clean `timeout` row without the cost of respawning a harness worker.
Values are PROVISIONAL (M3 toy benchmark) until frozen from the 200-locus validation pilot (pilot.py); the frozen
values are written to FROZEN_TIMEOUTS_FILE and, when that file exists, they replace the provisional ones.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
FROZEN_TIMEOUTS_FILE = Path(__file__).resolve().parent / "benchmarks" / "frozen_timeouts.json"
R_TIMEOUT_FRACTION = 0.9


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    target: str                 # "module:Class" for eval.harness.MethodSpec
    kwargs: dict = field(default_factory=dict)
    timeout_s: float = 600.0    # provisional per-locus timeout (harness)
    # per-worker peak memory model (GB) = mem_a + mem_b * (p / 1000)^2, Python worker + its R process
    mem_a: float = 0.6
    mem_b: float = 0.05
    uses_r: bool = True


# Provisional timeouts = methods.DEFAULT_TIMEOUT (M3 toy benchmark: max observed x ~4 for p up to 10k x >= 3).
# Memory models (provisional): M3 runtime_M3.tsv peak R RSS at p = 5000 (susieR 1.9-2.5 GB, mvSuSiE 1.4-1.7,
# ColocBoost 1.0-1.3, prop tests 1.0) plus the Python worker (float32 LD as read + float64 dense copy, ~0.4 GB at
# p = 5000), rounded up. Refitted from the pilot's measured per-row peaks (extra.mem_mb) by pilot.py.
SPECS: dict[str, BaselineSpec] = {s.name: s for s in [
    BaselineSpec("coloc_abf", "baselines.methods:ColocABF", timeout_s=60, mem_a=0.4, mem_b=0.02, uses_r=False),
    BaselineSpec("lead_snp", "baselines.methods:LeadSNP", timeout_s=60, mem_a=0.4, mem_b=0.02, uses_r=False),
    BaselineSpec("sharepro", "baselines.methods:SharePro", timeout_s=300, mem_a=0.5, mem_b=0.05, uses_r=False),
    BaselineSpec("coloc_susie", "baselines.methods:ColocSusie", timeout_s=600, mem_a=0.8, mem_b=0.10),
    BaselineSpec("coloc_susie_mm", "baselines.methods:ColocSusieMM", timeout_s=900, mem_a=0.8, mem_b=0.10),
    BaselineSpec("colocboost", "baselines.methods:ColocBoost", timeout_s=900, mem_a=0.8, mem_b=0.06),
    BaselineSpec("colocboost_m1", "baselines.methods:ColocBoostM1", timeout_s=120, mem_a=0.8, mem_b=0.06),
    BaselineSpec("mvsusie", "baselines.methods:MvSusie", timeout_s=1800, mem_a=0.8, mem_b=0.08),
    BaselineSpec("prop_coloc", "baselines.methods:PropColoc", timeout_s=300, mem_a=0.8, mem_b=0.05),
    BaselineSpec("coloc_prop_test", "baselines.methods:ColocPropTest", timeout_s=300, mem_a=0.8, mem_b=0.05),
]}
METHOD_NAMES = list(SPECS)


def frozen_overrides() -> dict:
    """Per-method frozen values from the pilot (timeout_s, mem_a, mem_b), if the frozen file exists."""
    if not FROZEN_TIMEOUTS_FILE.exists():
        return {}
    d = json.loads(FROZEN_TIMEOUTS_FILE.read_text())
    return d.get("methods", {})


def spec(name: str) -> BaselineSpec:
    s = SPECS[name]
    ov = frozen_overrides().get(name)
    if ov:
        s = BaselineSpec(s.name, s.target, dict(s.kwargs), float(ov.get("timeout_s", s.timeout_s)),
                         float(ov.get("mem_a", s.mem_a)), float(ov.get("mem_b", s.mem_b)), s.uses_r)
    return s


# Provisional per-locus timeouts (s) frozen by the pre-registered pilot rule (max(60, 3 x max observed runtime),
# ceiled to 30 s) on the 200-locus screen_v2 validation pilot; = datasets/configs/test_methods_v1.yaml
# provisional_timeouts_s (reviews/p2_review_wip.md). Used by the p4 validation runs (screen_v2, M4).
PILOT_TIMEOUTS_S = {"coloc_abf": 60, "lead_snp": 60, "sharepro": 60, "coloc_susie": 360, "coloc_susie_mm": 240,
                    "colocboost": 120, "colocboost_m1": 60, "mvsusie": 1080, "prop_coloc": 60,
                    "coloc_prop_test": 60}


def method_spec(name: str, timeout_s: float | None = None, run_name: str | None = None, **kwargs):
    """eval.harness.MethodSpec for baseline `name` (the wrapper's R-call timeout = R_TIMEOUT_FRACTION x timeout).

    run_name: name of the run / prediction rows for a non-default configuration of the same wrapper (e.g. the
    coloc.abf W grid: run_name="coloc_abf_W0.1", kwargs W=0.1). For ColocABF it is also passed to the wrapper."""
    from eval.harness import MethodSpec

    s = spec(name)
    t = float(timeout_s if timeout_s is not None else s.timeout_s)
    kw = dict(s.kwargs)
    kw.update(kwargs)
    kw.setdefault("timeout", R_TIMEOUT_FRACTION * t)
    if run_name and name == "coloc_abf":
        kw.setdefault("name", run_name)
    return MethodSpec(s.target, kw, name=run_name or s.name)


def mem_gb(name: str, p: int) -> float:
    s = spec(name)
    return s.mem_a + s.mem_b * (float(p) / 1000.0) ** 2


def n_workers(name: str, p_max: int, mem_budget_gb: float, max_cpus: int) -> int:
    """Parallel workers for a run: limited by cores and by the memory budget at the largest locus (p_max)."""
    by_mem = int(math.floor(mem_budget_gb / mem_gb(name, p_max)))
    return max(1, min(int(max_cpus), by_mem))


def run_baseline(name: str, locus_set: str | Path, *, ids: list[str] | None = None, mem_budget_gb: float = 48.0,
                 max_cpus: int = 24, timeout_s: float | None = None, label: str = "", results_dir=None,
                 config: dict | None = None, metric_config: dict | None = None, timing: bool = False, log=print,
                 kwargs: dict | None = None, run_name: str | None = None, workers: int | None = None):
    """Run one baseline through eval.harness.run_method with its timeout and memory-limited worker count.

    Writes predictions + manifest + LEDGER line only: the harness is told not to compute metrics here, because
    metrics need the oracle labels, which method code (this package) never touches (the eval label reader refuses callers
    from `baselines.*`). Metrics are computed afterwards by the evaluation side from predictions.parquet.
    kwargs / run_name: non-default wrapper configuration under its own run name (see method_spec).
    workers: explicit worker count (overrides the memory model; the caller sized the job's RAM for it).
    """
    from eval.harness import run_method
    from eval.io import read_locus_attrs

    s = spec(name)
    t = float(timeout_s if timeout_s is not None else s.timeout_s)
    attrs = read_locus_attrs(locus_set)
    if ids is not None:
        attrs = attrs[attrs["locus_id"].astype(str).isin(set(map(str, ids)))]
    p_max = int(attrs["p"].max())
    nw = int(workers) if workers else n_workers(name, p_max, mem_budget_gb, max_cpus)
    cfg = {"baseline": name, "run_name": run_name or name, "kwargs": dict(kwargs or {}), "timeout_s": t, "r_timeout_s": R_TIMEOUT_FRACTION * t, "p_max": p_max,
           "mem_model_gb": {"a": s.mem_a, "b": s.mem_b, "at_p_max": mem_gb(name, p_max)},
           "mem_budget_gb": mem_budget_gb, "max_cpus": max_cpus, "n_workers": nw,
           "frozen_timeouts_file": str(FROZEN_TIMEOUTS_FILE) if FROZEN_TIMEOUTS_FILE.exists() else None}
    cfg.update(config or {})
    log(f"[baselines.run] {run_name or name}: {len(attrs)} loci, p_max={p_max}, workers={nw}, timeout={t:g}s")
    return run_method(method_spec(name, t, run_name=run_name, **(kwargs or {})), locus_set, ids=ids, mode="cpu", n_workers=nw, timeout_s=t,
                      label=label or f"baseline {name}", results_dir=results_dir, config=cfg,
                      metric_config=metric_config, timing=timing, truth_path=None, log=log)


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", required=True, choices=METHOD_NAMES)
    ap.add_argument("--set", required=True)
    ap.add_argument("--ids-file", default=None, help="one locus id per line")
    ap.add_argument("--mem-budget-gb", type=float, default=48.0)
    ap.add_argument("--max-cpus", type=int, default=24)
    ap.add_argument("--timeout-s", type=float, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--kwargs", default=None, help="JSON dict of wrapper kwargs (non-default configuration)")
    ap.add_argument("--run-name", default=None, help="run / method name for a non-default configuration")
    ap.add_argument("--workers", type=int, default=None, help="explicit worker count (else memory model)")
    ap.add_argument("--pilot-timeout", action="store_true", help="use PILOT_TIMEOUTS_S[method] as the timeout")
    a = ap.parse_args(argv)
    if os.environ.get("COLOC_ENV_ACTIVE") != "1":
        raise SystemExit("source env/activate.sh first")
    ids = None
    if a.ids_file:
        ids = [l.strip() for l in Path(a.ids_file).read_text().splitlines() if l.strip()]
    t = a.timeout_s if a.timeout_s is not None else (PILOT_TIMEOUTS_S[a.method] if a.pilot_timeout else None)
    r = run_baseline(a.method, a.set, ids=ids, mem_budget_gb=a.mem_budget_gb, max_cpus=a.max_cpus,
                     timeout_s=t, label=a.label, results_dir=a.results_dir,
                     kwargs=json.loads(a.kwargs) if a.kwargs else None, run_name=a.run_name, workers=a.workers)
    print(json.dumps({"run_id": r.run_id, "run_dir": str(r.run_dir), "counts": r.counts}))


if __name__ == "__main__":
    _main()
