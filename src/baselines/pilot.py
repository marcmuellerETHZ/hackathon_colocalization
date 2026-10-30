"""200-locus timing / failure pilot of the tier-1 baselines on validation loci (DESIGN §4.3 "timeouts and
subsampling are frozen per method from a pilot of 200 validation loci"; R2-alg 5).

Steps (all runs go through eval.harness via baselines.run, so each has a manifest and a LEDGER line):
  sample   200 locus ids of a validation locus set, stratified equally over its cells (locus attributes only),
           seed derive(20261029, "baselines/pilot200", <set name>) (DESIGN §9: 20261029 = validation draws).
  run      every baseline on the same ids with its provisional timeout and memory-limited worker count, one method
           after another (so runtimes are not inflated by our own concurrent runs). Loci that time out are re-run
           once with RERUN_FACTOR x the timeout to measure their actual runtime (censoring check).
  summarise  per method: status counts, runtime median / p95 / max, peak memory per worker (extra.mem_mb), runtime
           and memory vs p fits; frozen timeouts and slow-method subsample decisions by the rules below.

Pre-registered freezing rules (written before the pilot ran):
  * timeout_s = max(60, 3 x max observed runtime over the pilot loci incl. the censoring re-runs), rounded up to a
    multiple of 30 s. A locus that also exceeds RERUN_FACTOR x provisional timeout counts as a timeout failure.
  * memory model: peak (Python + R) per worker = a + b (p/1000)^2, b from a least-squares fit on the pilot's
    running peaks with a 25% safety margin (never below the provisional model's b/2).
  * slow-method rule: projected wall-clock of a method on the M4 workload (WORKLOAD_LOCI loci at the pilot's mean
    runtime per locus, with the memory-limited worker count under MACHINE_MEM_GB / MACHINE_CPUS) above
    SLOW_WALL_H hours -> that method runs on the pre-registered 400-loci-per-cell subsample (DESIGN §4.3), which is
    identical for all methods (subsample_ids).
Choices are documented in src/baselines/README.md ("Decisions / assumptions").

CLI:
  python -m baselines.pilot sample --set S.h5 --out ids.txt [--n 200]
  python -m baselines.pilot run --set S.h5 --ids-file ids.txt --tag pilot200 [--methods a,b] [--mem-budget-gb 48]
         [--max-cpus 24] [--results-dir results]  -> writes src/baselines/benchmarks/<tag>_runs.json
  python -m baselines.pilot summarise --runs src/baselines/benchmarks/<tag>_runs.json [--freeze]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common.seeds import VALID_SEED, rng

from . import run as BR

BENCH = Path(__file__).resolve().parent / "benchmarks"
RERUN_FACTOR = 3.0
WORKLOAD_LOCI = 60_000      # M4: screening set (~12k) + validation grid (34 cells x ~1.4k), DESIGN §10
MACHINE_MEM_GB = 110.0      # 128 GB job limit minus headroom for the harness parent and other processes
MACHINE_CPUS = 60
SLOW_WALL_H = 12.0
SUBSAMPLE_PER_CELL = 400


# ------------------------------------------------------------------------------------------------ sampling
def sample_ids(locus_set: str | Path, n: int = 200, key: str = "baselines/pilot200") -> list[str]:
    """n ids stratified equally over cell_id (remainder to randomly chosen cells), validation split only."""
    from eval.io import read_locus_attrs

    a = read_locus_attrs(locus_set)
    bad = a[a["split"].astype(str).str.lower() != "valid"]
    if len(bad):
        raise ValueError(f"{len(bad)} loci are not in the validation split (pilot uses validation loci only)")
    g = rng(VALID_SEED, key, Path(locus_set).stem)
    cells = sorted(a["cell_id"].astype(str).unique())
    base, rem = divmod(n, len(cells))
    extra = set(g.choice(len(cells), size=rem, replace=False).tolist()) if rem else set()
    out = []
    for i, c in enumerate(cells):
        ids = sorted(a.loc[a["cell_id"].astype(str) == c, "locus_id"].astype(str))
        k = base + (1 if i in extra else 0)
        out += [ids[j] for j in sorted(g.choice(len(ids), size=min(k, len(ids)), replace=False))]
    return out


def subsample_ids(frame: pd.DataFrame, n_per_cell: int = SUBSAMPLE_PER_CELL, stratum_col: str | None = None,
                  key: str = "baselines/subsample400") -> list[str]:
    """Pre-registered slow-method subsample: n_per_cell ids per cell_id, identical for all methods.

    frame: columns locus_id, cell_id (+ optional `stratum_col`, supplied by the evaluation side, for allocation
    proportional to strata). Seed derive(20261029, key, cell_id). Deterministic given the frame's id set.
    """
    out = []
    for c, d in frame.groupby("cell_id", sort=True):
        g = rng(VALID_SEED, key, str(c))
        d = d.sort_values("locus_id")
        if len(d) <= n_per_cell:
            out += d["locus_id"].astype(str).tolist()
            continue
        if stratum_col is None:
            idx = g.choice(len(d), size=n_per_cell, replace=False)
            out += d["locus_id"].astype(str).to_numpy()[np.sort(idx)].tolist()
            continue
        sizes = d.groupby(stratum_col).size()
        alloc = np.floor(sizes / sizes.sum() * n_per_cell).astype(int)
        short = n_per_cell - int(alloc.sum())
        if short:
            frac = (sizes / sizes.sum() * n_per_cell - alloc).sort_values(ascending=False, kind="stable")
            for s in frac.index[:short]:
                alloc[s] += 1
        for s, ds in d.groupby(stratum_col, sort=True):
            idx = g.choice(len(ds), size=int(alloc[s]), replace=False)
            out += ds["locus_id"].astype(str).to_numpy()[np.sort(idx)].tolist()
    return out


# ------------------------------------------------------------------------------------------------ running
def run_pilot(locus_set, ids, methods, tag, mem_budget_gb=48.0, max_cpus=24, results_dir=None, log=print) -> dict:
    """Run each method on ids (+ censoring re-run of its timeouts). Records run dirs in benchmarks/<tag>_runs.json."""
    BENCH.mkdir(parents=True, exist_ok=True)
    rec_path = BENCH / f"{tag}_runs.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {"locus_set": str(Path(locus_set).resolve()),
                                                                       "n_ids": len(ids), "runs": {}}
    for m in methods:
        if m in rec["runs"] and rec["runs"][m].get("done"):
            log(f"[pilot] {m}: already done, skipping")
            continue
        t0 = time.time()
        r = BR.run_baseline(m, locus_set, ids=ids, mem_budget_gb=mem_budget_gb, max_cpus=max_cpus,
                            label=f"M3.5 {tag} {m}", results_dir=results_dir, log=log)
        entry = {"run_dir": str(r.run_dir), "counts": r.counts, "wall_s": time.time() - t0, "rerun": None}
        pred = pd.read_parquet(r.run_dir / "predictions.parquet")
        tout = pred.loc[pred["status"] == "timeout", "locus_id"].astype(str).tolist()
        if tout:
            t_re = RERUN_FACTOR * BR.spec(m).timeout_s
            log(f"[pilot] {m}: {len(tout)} timeouts; re-running them with timeout {t_re:g}s")
            r2 = BR.run_baseline(m, locus_set, ids=tout, mem_budget_gb=mem_budget_gb, max_cpus=max_cpus,
                                 timeout_s=t_re, label=f"M3.5 {tag} {m} timeout-rerun", results_dir=results_dir,
                                 log=log)
            entry["rerun"] = {"run_dir": str(r2.run_dir), "counts": r2.counts, "timeout_s": t_re}
        entry["done"] = True
        rec["runs"][m] = entry
        rec_path.write_text(json.dumps(rec, indent=1))
        log(f"[pilot] {m}: {r.counts} in {entry['wall_s']:.0f}s")
    return rec


# ------------------------------------------------------------------------------------------------ summary
def _round_up(x: float, step: float = 30.0) -> float:
    return float(step * math.ceil(x / step))


def load_predictions(rec: dict, m: str) -> pd.DataFrame:
    """Pilot predictions of method m with censoring re-runs merged in (column `rerun` = True for those rows)."""
    e = rec["runs"][m]
    pred = pd.read_parquet(Path(e["run_dir"]) / "predictions.parquet")
    pred["rerun"] = False
    if e.get("rerun"):
        re = pd.read_parquet(Path(e["rerun"]["run_dir"]) / "predictions.parquet")
        re["rerun"] = True
        pred = pd.concat([pred[~pred["locus_id"].isin(re["locus_id"])], re], ignore_index=True)
    ex = pred["extra"].map(lambda s: json.loads(s) if isinstance(s, str) and s else {})
    pred["py_peak_mb"] = ex.map(lambda d: (d.get("mem_mb") or {}).get("py_peak", np.nan))
    pred["r_peak_mb"] = ex.map(lambda d: (d.get("mem_mb") or {}).get("r_peak", np.nan))
    return pred


def summarise(rec: dict, attrs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-method table + frozen values (timeouts, memory model, slow-method flag) by the pre-registered rules."""
    attrs = attrs.set_index(attrs["locus_id"].astype(str))
    rows, frozen = [], {}
    for m, e in rec["runs"].items():
        pred = load_predictions(rec, m)
        pred = pred.join(attrs[["p", "cell_id", "mismatch_level"]], on="locus_id")
        ok = pred["status"].isin(["ok", "nonconverged"])
        rt = pred.loc[ok, "runtime_s"].astype(float)
        n = len(pred)
        s = BR.spec(m)
        # memory: running peaks (MB) of Python + R per row, fit a + b (p/1000)^2 on the upper envelope
        mem = (pred["py_peak_mb"].fillna(0) + pred["r_peak_mb"].fillna(0)) / 1024.0
        pp = (pred["p"].astype(float) / 1000.0) ** 2
        okm = mem > 0
        b_fit = np.nan
        if okm.sum() >= 5:
            A = np.column_stack([np.ones(okm.sum()), pp[okm]])
            coef, *_ = np.linalg.lstsq(A, mem[okm].to_numpy(), rcond=None)
            b_fit = float(coef[1])
        mem_max = float(mem.max()) if okm.any() else np.nan
        p_max = float(pred["p"].max())
        b_new = max(s.mem_b / 2, 1.25 * b_fit) if np.isfinite(b_fit) else s.mem_b
        a_new = max(0.3, 1.25 * mem_max - b_new * (p_max / 1000) ** 2) if np.isfinite(mem_max) else s.mem_a
        max_rt = float(rt.max()) if len(rt) else np.nan
        t_new = max(60.0, _round_up(3 * max_rt)) if np.isfinite(max_rt) else s.timeout_s
        # projected M4 wall-clock at whole-machine budget (memory-limited workers at the pilot's p_max)
        nw = max(1, min(MACHINE_CPUS, int(MACHINE_MEM_GB // (a_new + b_new * (p_max / 1000) ** 2))))
        mean_rt = float(pred["runtime_s"].astype(float).clip(upper=t_new).mean())
        wall_h = WORKLOAD_LOCI * mean_rt / nw / 3600
        n_final_timeout = int((pred["status"] == "timeout").sum())
        rows.append({
            "method": m, "n": n, "ok": int((pred["status"] == "ok").sum()),
            "nonconverged": int((pred["status"] == "nonconverged").sum()),
            "failed": int((pred["status"] == "failed").sum()), "timeout_final": n_final_timeout,
            "timeout_first_pass": int(e["counts"].get("timeout", 0)),
            "fail_rate": float((~ok).mean()), "nonconv_rate": float((pred["status"] == "nonconverged").mean()),
            "rt_median": float(rt.median()) if len(rt) else np.nan,
            "rt_p95": float(rt.quantile(0.95)) if len(rt) else np.nan, "rt_max": max_rt,
            "rt_mean_all": mean_rt, "mem_peak_gb": mem_max, "mem_b_fit": b_fit,
            "timeout_provisional": s.timeout_s, "timeout_frozen": t_new,
            "mem_a_frozen": a_new, "mem_b_frozen": b_new, "workers_at_machine": nw, "m4_wall_h": wall_h,
            "slow_subsample": bool(wall_h > SLOW_WALL_H),
        })
        frozen[m] = {"timeout_s": t_new, "mem_a": round(a_new, 3), "mem_b": round(b_new, 4),
                     "slow_subsample": bool(wall_h > SLOW_WALL_H), "m4_wall_h_projected": round(wall_h, 2)}
    return pd.DataFrame(rows), frozen


def by_level(rec: dict, attrs: pd.DataFrame) -> pd.DataFrame:
    """Per method x mismatch level x pairing: status counts and runtime median / p95 / max (DESIGN §7.7 reporting)."""
    attrs = attrs.set_index(attrs["locus_id"].astype(str))
    out = []
    for m in rec["runs"]:
        pred = load_predictions(rec, m).join(attrs[["mismatch_level", "pairing_type", "p"]], on="locus_id")
        for (lev, pt), d in pred.groupby(["mismatch_level", "pairing_type"]):
            rt = d.loc[d["status"].isin(["ok", "nonconverged"]), "runtime_s"].astype(float)
            out.append({"method": m, "level": lev, "pairing": pt, "n": len(d),
                        "nonconverged": int((d["status"] == "nonconverged").sum()),
                        "failed": int((d["status"] == "failed").sum()), "timeout": int((d["status"] == "timeout").sum()),
                        "rt_median": float(rt.median()) if len(rt) else np.nan,
                        "rt_p95": float(rt.quantile(0.95)) if len(rt) else np.nan,
                        "rt_max": float(rt.max()) if len(rt) else np.nan, "p_median": float(d["p"].median())})
    return pd.DataFrame(out)


def _main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("sample")
    a1.add_argument("--set", required=True)
    a1.add_argument("--out", required=True)
    a1.add_argument("--n", type=int, default=200)
    a2 = sub.add_parser("run")
    a2.add_argument("--set", required=True)
    a2.add_argument("--ids-file", required=True)
    a2.add_argument("--tag", required=True)
    a2.add_argument("--methods", default=",".join(BR.METHOD_NAMES))
    a2.add_argument("--mem-budget-gb", type=float, default=48.0)
    a2.add_argument("--max-cpus", type=int, default=24)
    a2.add_argument("--results-dir", default=None)
    a3 = sub.add_parser("summarise")
    a3.add_argument("--runs", required=True)
    a3.add_argument("--freeze", action="store_true", help=f"write {BR.FROZEN_TIMEOUTS_FILE.name}")
    a = ap.parse_args(argv)
    if os.environ.get("COLOC_ENV_ACTIVE") != "1":
        raise SystemExit("source env/activate.sh first")
    if a.cmd == "sample":
        ids = sample_ids(a.set, a.n)
        Path(a.out).write_text("\n".join(ids) + "\n")
        print(f"{len(ids)} ids -> {a.out}")
    elif a.cmd == "run":
        ids = [l.strip() for l in Path(a.ids_file).read_text().splitlines() if l.strip()]
        run_pilot(a.set, ids, [m for m in a.methods.split(",") if m], a.tag, a.mem_budget_gb, a.max_cpus,
                  a.results_dir)
    else:
        from eval.io import read_locus_attrs

        rec = json.loads(Path(a.runs).read_text())
        attrs = read_locus_attrs(rec["locus_set"])
        tab, frozen = summarise(rec, attrs)
        out = Path(a.runs).with_name(Path(a.runs).name.replace("_runs.json", "_summary.tsv"))
        tab.to_csv(out, sep="\t", index=False, float_format="%.4g")
        by_level(rec, attrs).to_csv(out.with_name(out.name.replace("_summary.tsv", "_by_level.tsv")), sep="\t",
                                    index=False, float_format="%.4g")
        pd.set_option("display.width", 250)
        print(tab.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        if a.freeze:
            doc = {"source": a.runs, "locus_set": rec["locus_set"], "n_ids": rec["n_ids"],
                   "rules": {"timeout": "max(60, 3 x max observed runtime incl. censoring re-runs), ceil to 30 s",
                             "memory": "a + b (p/1000)^2 GB; b = 1.25 x LSQ fit (>= provisional b/2); a from 1.25 x max",
                             "slow_subsample": f"projected M4 wall ({WORKLOAD_LOCI} loci, {MACHINE_MEM_GB} GB, "
                                               f"{MACHINE_CPUS} cpus) > {SLOW_WALL_H} h -> {SUBSAMPLE_PER_CELL}/cell"},
                   "methods": frozen}
            BR.FROZEN_TIMEOUTS_FILE.write_text(json.dumps(doc, indent=1))
            print(f"frozen -> {BR.FROZEN_TIMEOUTS_FILE}")
        print(f"table -> {out}")


if __name__ == "__main__":
    _main()
