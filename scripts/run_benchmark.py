#!/usr/bin/env python
"""Validation benchmark runner (DESIGN v4 §6, §8; decisions D6(e), D6(h)).

method(s) x locus set(s) -> harness runs (src/eval/harness.py) -> one benchmark directory with
  benchmark.json        all metrics (both failure modes: least_favourable and excluded):
                        * pooled co-primaries (realistic / ideal / shifted, whichever cells exist) with BCa CIs from
                          ONE joint cluster bootstrap (all methods on the same resamples, so contrasts are paired);
                        * per-cell, per-level and macro tables; D strata (r² and d) reports;
                        * D-vs-W AUROC; C3 operating points (t_S, then t_D with S precedence; W rates <= 2.5%) and
                          recall at them; calibration raw and recalibrated (logit intercept); confusion; FDR;
                        * the strength-only control (D6(e)) and every method's paired difference to it;
                        * optional candidate: paired differences to every other method and a C3 dry run;
                        * failures and runtime per method.
  summary.md            human-readable tables of the above
  operating_points.json t_S, t_D, recal_a_S, recal_a_D per method (to freeze into configs/test_methods_v1.yaml)
  strength_table_crossfit.json / strength_table_full.json   the control's tables (cross-fitted / full-set)
  expected_ids.txt      the evaluated locus ids (one per line, per set in order)
  manifest.json         the evaluation spec (runs, seeds, options, sha256s, git) used by --verify
Refuses test-split loci (the harness and the feature reader apply the io guard to every locus).

Usage
  python scripts/run_benchmark.py --locus-set data/processed/sets/screen_v1.h5 --methods bench_methods.yaml \\
      --label M4_screen --per-cell 200 --n-boot 2000 --unit locus --candidate famA_joint_susie_rho
  python scripts/run_benchmark.py ... --reuse coloc_abf=results/<run_id>      # reuse an existing harness run
  python scripts/run_benchmark.py --operating-points <validation bench>/operating_points.json ...  # fixed thresholds
  python scripts/run_benchmark.py ... --method-ids mvsusie=ids20.txt --method-ids colocboost=ids40.txt  # slow subsamples
  python scripts/run_benchmark.py --verify results/benchmarks/<id>            # recompute and compare
The methods YAML has a list `methods` of {name, target, kwargs, mode, n_workers, batch_size, timeout_s}
(same keys as configs/test_methods_v1.yaml). Bootstrap unit: "block" for claims (§3, §8.3), "locus" for the
screening kill rules (§3).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.seeds import rng as seeded_rng  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval import strength_control as SC  # noqa: E402
from eval.harness import MethodSpec, git_info, run_method, sha256_file, append_ledger  # noqa: E402
from eval.io import read_locus_attrs, truth_path_for  # noqa: E402
from eval.truth import read_truth  # noqa: E402

CONTROL = "strength_only"
BOOT_SEED = 20261029  # DESIGN §9: validation draws
SETTINGS = ("realistic", "ideal", "shifted")
EXCLUDED_FROM_C3 = ("prop_coloc", "coloc_prop_test", "lead_snp", CONTROL)


def log(msg: str) -> None:
    print(f"[benchmark {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ==================================================================================================
# Inputs
# ==================================================================================================
def load_methods(path: Path | None) -> list[dict]:
    if path is None:
        return []
    import yaml
    ms = yaml.safe_load(Path(path).read_text()).get("methods") or []
    for m in ms:
        if "target" not in m:
            raise ValueError(f"method entry without target: {m}")
        m.setdefault("name", MethodSpec(m["target"], m.get("kwargs", {})).method_name())
    names = [m["name"] for m in ms]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate method names {names}")
    if CONTROL in names:
        raise ValueError(f"{CONTROL!r} is added by the runner; do not list it")
    return ms


def expected_ids(locus_set: Path, per_cell: int | None) -> list[str]:
    """All loci of the set (file order), or a deterministic per-cell subsample (seed 20261029; identical for
    every method, §4.3 'pre-registered random subsample')."""
    attrs = read_locus_attrs(locus_set)  # test-split guard on every locus
    ids = attrs["locus_id"].astype(str).tolist()
    if not per_cell:
        return ids
    cells = attrs["cell_id"].astype(str) if "cell_id" in attrs else pd.Series([""] * len(attrs))
    keep = set()
    for c in sorted(cells.unique()):
        cid = sorted(np.asarray(ids)[(cells == c).to_numpy()].tolist())
        if len(cid) > per_cell:
            g = seeded_rng(BOOT_SEED, "run_benchmark", "subsample", locus_set.stem, c)
            cid = sorted(g.choice(cid, size=per_cell, replace=False).tolist())
        keep.update(cid)
    return [i for i in ids if i in keep]


def _ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def load_method_ids(items: list[str]) -> dict:
    """--method-ids NAME=FILE (one locus id per line): the method runs only on those ids (slow-method
    subsamples). Returns {name: {file, sha256 (of the sorted id list), ids}}."""
    out = {}
    for it in items:
        name, _, f = it.partition("=")
        if not f:
            raise ValueError(f"--method-ids expects NAME=FILE, got {it!r}")
        ids = sorted({l.strip() for l in Path(f).read_text().splitlines() if l.strip()})
        out[name] = {"file": str(Path(f).resolve()), "sha256": _ids_hash(ids), "ids": ids}
    return out


def _sub_benchmarks(spec: dict) -> list[tuple[str, dict, set | None]]:
    """(relative dir, spec, id filter) of the evaluations of a benchmark with per-method id lists (review
    p5_verify_infra defect 6): the main evaluation holds every method WITHOUT an own id list, on all expected
    ids; each distinct id list gets a sub-benchmark `subsample_<sha8>` holding every method whose id list
    covers it (methods without a list included), on the expected ids within that list. A slow method is thus
    never scored on loci it did not run (no spurious least-favourable failures), and paired comparisons with it
    use its subsample only."""
    mids = spec.get("method_ids") or {}
    if not mids:
        return [("", spec, None)]
    lists = {n: set(Path(v["file"]).read_text().split()) for n, v in mids.items()}
    for n, v in mids.items():
        if _ids_hash(sorted(lists[n])) != v["sha256"]:
            raise ValueError(f"id list of {n} ({v['file']}) differs from the manifest")
    main = dict(spec, runs={m: r for m, r in spec["runs"].items() if m not in mids}, method_ids={})
    out = [("", main, None)]
    for h in sorted({v["sha256"] for v in mids.values()}):
        keep = next(lists[n] for n, v in mids.items() if v["sha256"] == h)
        members = [m for m in spec["runs"] if m not in mids or keep <= lists[m]]
        sub = dict(spec, runs={m: spec["runs"][m] for m in members}, method_ids={},
                   bench_id=f"{spec['bench_id']}__subsample_{h[:8]}",
                   c3_baselines=[b for b in (spec.get("c3_baselines") or []) if b in members] or None,
                   candidate=spec.get("candidate") if spec.get("candidate") in members else None)
        out.append((f"subsample_{h[:8]}", sub, keep))
    return out


def evaluate_all(spec: dict, bdir: Path) -> dict:
    """evaluate() for the main benchmark and every per-method-id-list sub-benchmark (written to bdir/<sub>/)."""
    ids = Path(bdir / "expected_ids.txt").read_text().split()
    res = None
    for rel, sp, keep in _sub_benchmarks(spec):
        d = bdir / rel if rel else bdir
        d.mkdir(parents=True, exist_ok=True)
        if keep is not None:
            (d / "expected_ids.txt").write_text("\n".join(i for i in ids if i in keep) + "\n")
        r = evaluate(sp, d)
        if not rel:
            res = r
    return res


# ==================================================================================================
# Run stage
# ==================================================================================================
def run_stage(a) -> Path:
    if os.environ.get("COLOC_ENV_ACTIVE") != "1":
        raise SystemExit("source env/activate.sh first")
    results_dir = Path(a.results_dir).resolve()
    sets = [Path(s).resolve() for s in a.locus_set]
    methods = load_methods(a.methods)
    reuse: dict[str, list[Path]] = {}
    for r in a.reuse or []:
        name, _, d = r.partition("=")
        reuse.setdefault(name, []).append(Path(d).resolve())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bench_id = f"{stamp}_{a.label}"
    bdir = results_dir / "benchmarks" / bench_id
    bdir.mkdir(parents=True)
    log(f"benchmark {bench_id}: {len(sets)} locus set(s), {len(methods)} method(s) + reuse {sorted(reuse)}")

    ids = {str(s): expected_ids(s, a.per_cell) for s in sets}
    method_ids = load_method_ids(a.method_ids or [])
    with open(bdir / "expected_ids.txt", "w") as fh:
        for s in sets:
            fh.write("\n".join(ids[str(s)]) + "\n")
    runs: dict[str, dict[str, str]] = {}

    # reused runs: match by the manifest's locus-set path
    for name, dirs in reuse.items():
        for d in dirs:
            man = json.loads((d / "manifest.json").read_text())
            ls = str(Path(man["locus_set"]["path"]).resolve())
            if ls not in ids:
                raise ValueError(f"reused run {d} is on {ls}, not one of the benchmark's locus sets")
            runs.setdefault(name, {})[ls] = str(d)

    # strength-only control (D6(e)): features -> tables -> harness run with the table
    truth_all = []
    feats = []
    for s in sets:
        tr = read_truth(truth_path_for(s))
        tr = tr[tr["locus_id"].astype(str).isin(set(ids[str(s)]))]
        truth_all.append(tr)
        feats.append(SC.read_features(s, ids[str(s)]))
    truth = pd.concat(truth_all, ignore_index=True)
    feats = pd.concat(feats, ignore_index=True)
    cls = feats[["locus_id"]].merge(truth[["locus_id", "oracle_class"]], on="locus_id", how="left")["oracle_class"]
    if cls.isna().any():
        raise ValueError(f"{int(cls.isna().sum())} loci without truth rows")
    if a.strength_table:
        table_path = Path(a.strength_table).resolve()
        table_mode = "given"
    else:
        table_path = bdir / "strength_table_crossfit.json"
        SC.dump(SC.fit_crossfit(feats, cls.to_numpy(), k=a.strength_folds), table_path)
        table_mode = f"crossfit_k{a.strength_folds}"
    SC.dump(SC.StrengthTable.fit(feats, cls.to_numpy()).to_dict(), bdir / "strength_table_full.json")
    all_methods = methods + [{"name": CONTROL, "target": "eval.reference_methods:StrengthOnly",
                              "kwargs": {"table": str(table_path)}, "mode": "cpu",
                              "n_workers": a.control_workers, "timeout_s": 120}]
    for m in all_methods:
        for s in sets:
            if str(s) in runs.get(m["name"], {}):
                continue
            run_ids = ids[str(s)]
            if m["name"] in method_ids:   # slow method: its own pre-registered id list (P3-12/P3-13/P3-17)
                keep = set(method_ids[m["name"]]["ids"])
                run_ids = [i for i in run_ids if i in keep]
            log(f"run {m['name']} on {s.name} ({len(run_ids)} loci)")
            r = run_method(MethodSpec(m["target"], m.get("kwargs", {}), m["name"]), s, results_dir=results_dir,
                           mode=m.get("mode", "cpu"), n_workers=int(m.get("n_workers", 8)),
                           batch_size=int(m.get("batch_size", 64)), timeout_s=float(m.get("timeout_s", 600)),
                           ids=run_ids, label=f"benchmark {bench_id}", hash_inputs=not a.no_hash_inputs,
                           config={"benchmark": bench_id, "strength_table_mode": table_mode} if m["name"] == CONTROL
                           else {"benchmark": bench_id},
                           seeds_used={"subsample": BOOT_SEED} if a.per_cell else None, log=log)
            runs.setdefault(m["name"], {})[str(s)] = str(r.run_dir)

    spec = {
        "bench_id": bench_id, "label": a.label, "locus_sets": [str(s) for s in sets],
        "ids_sha256": {str(s): _ids_hash(ids[str(s)]) for s in sets}, "n_expected": {str(s): len(ids[str(s)]) for s in sets},
        "per_cell": a.per_cell, "runs": runs,
        "strength_control": {"table": str(table_path), "mode": table_mode,
                             "sha256": sha256_file(table_path), "folds": a.strength_folds},
        "operating_points_in": str(Path(a.operating_points).resolve()) if a.operating_points else None,
        "operating_points_in_sha256": sha256_file(a.operating_points) if a.operating_points else None,
        "method_ids": {n: {"file": v["file"], "sha256": v["sha256"]} for n, v in method_ids.items()},
        "candidate": a.candidate, "c3_baselines": a.c3_baselines, "n_boot": a.n_boot, "unit": a.unit,
        "boot_seed": [BOOT_SEED, "run_benchmark", "bootstrap", bench_id], "n_jobs": a.n_jobs,
        "git": git_info(), "argv": sys.argv, "start_utc": stamp,
    }
    (bdir / "manifest.json").write_text(json.dumps(spec, indent=1, sort_keys=True))
    t0 = time.perf_counter()
    res = evaluate_all(spec, bdir)
    spec["evaluate_wall_s"] = time.perf_counter() - t0
    (bdir / "manifest.json").write_text(json.dumps(spec, indent=1, sort_keys=True))
    append_ledger(results_dir, {"run_id": f"benchmark:{bench_id}", "start_utc": stamp,
                                "end_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                                "method": "benchmark:" + ",".join(sorted(runs)), "locus_set": ";".join(spec["locus_sets"]),
                                "n_expected": sum(spec["n_expected"].values()), "git_commit": spec["git"]["commit"],
                                "git_clean": spec["git"]["clean"], "timing": False, "label": a.label,
                                "run_dir": str(bdir)})
    log(f"done: {bdir} ({res['n_loci']} loci, {len(res['methods'])} methods)")
    return bdir


# ==================================================================================================
# Evaluation stage (deterministic given the spec; re-run by --verify)
# ==================================================================================================
def _load_truth_and_preds(spec: dict, bdir: Path):
    ids = set(Path(bdir / "expected_ids.txt").read_text().split())
    truth = []
    for s in spec["locus_sets"]:
        tr = read_truth(truth_path_for(s))
        truth.append(tr[tr["locus_id"].astype(str).isin(ids)])
    truth = pd.concat(truth, ignore_index=True)
    if len(truth) != len(ids):
        raise ValueError(f"truth rows {len(truth)} != expected ids {len(ids)}")
    preds, runtime = {}, {}
    for m, by_set in sorted(spec["runs"].items()):
        parts = []
        runtime[m] = {}
        for s, d in by_set.items():
            d = Path(d)
            man = json.loads((d / "manifest.json").read_text())
            p = pd.read_parquet(d / "predictions.parquet")
            if sha256_file(d / "predictions.parquet") != man["predictions_sha256"]:
                raise ValueError(f"{d}: predictions sha256 differs from its manifest")
            p = p[p["locus_id"].astype(str).isin(ids)].copy()
            p["method"] = m
            parts.append(p)
            rt = p["runtime_s"].to_numpy(float)
            runtime[m][Path(s).name] = {"run_id": man["run_id"], "n": int(len(p)), "wall_s": man.get("wall_s"),
                                        "median_s": float(np.nanmedian(rt)) if len(rt) else None,
                                        "p95_s": float(np.nanpercentile(rt, 95)) if len(rt) else None,
                                        "sum_s": float(np.nansum(rt)), "harness": man.get("harness")}
        preds[m] = pd.concat(parts, ignore_index=True)
    return truth, preds, runtime


def _settings(frame: M.EvalFrame) -> list[str]:
    st = [s for s in SETTINGS if M.cell_pooling_weights(frame, s)]
    return st or ["all"]


def _op_points(frame: M.EvalFrame, methods: list[str], main: str, given: dict | None) -> dict:
    """C3 thresholds and recalibration intercepts fitted HERE, and the ones USED (given file, else fitted here)."""
    out = {}
    for m in methods:
        mw = M.matched_w_thresholds(frame, m, main)
        cal = M.calibration(frame, m)
        here = {"t_S": mw.get("t_S", np.inf), "t_D": mw.get("t_D", np.inf),
                "rate_S_on_W": mw.get("rate_S_on_W"), "rate_D_on_W": mw.get("rate_D_on_W"),
                "recal_a_S": cal.get("recal_a_S_fitted_here"), "recal_a_D": cal.get("recal_a_D_fitted_here"),
                "setting": main}
        used = dict(here)
        if given is not None:
            if m not in given:
                raise ValueError(f"operating points file has no entry for {m}")
            g = given[m]
            used = {k: (float(g[k]) if isinstance(g.get(k), str) else g.get(k))
                    for k in ("t_S", "t_D", "recal_a_S", "recal_a_D")}
            used["setting"] = g.get("setting", main)
        for k in ("t_S", "t_D"):
            used[k] = np.inf if used.get(k) is None else float(used[k])
        out[m] = {"fitted_here": here, "used": used, "source": "given" if given is not None else "fitted_here"}
    return out


def _cfg(op: dict) -> M.MetricConfig:
    u = op["used"]
    return M.MetricConfig(t_S=float(u["t_S"]), t_D=float(u["t_D"]),
                          recal_a_S=None if u.get("recal_a_S") is None else float(u["recal_a_S"]),
                          recal_a_D=None if u.get("recal_a_D") is None else float(u["recal_a_D"]))


def n_boot_units(frame: M.EvalFrame, unit: str) -> int:
    return int(len(np.unique(M.resampling_units(frame, unit)))) if len(frame) else 0


def _bootstrap_block(frame: M.EvalFrame, methods: list[str], settings: list[str], spec: dict, tag: str,
                     candidate: str | None) -> dict:
    g = seeded_rng(BOOT_SEED, "run_benchmark", "bootstrap", spec["bench_id"], tag)
    K = n_boot_units(frame, spec["unit"])
    if K < 2:   # BCa needs the leave-one-unit-out jackknife: record point estimates and the reason, never raise
        out = {"unit": spec["unit"], "n_boot": spec["n_boot"], "n_clusters": K, "n_boot_nan": None,
               "not_estimable": f"fewer than 2 bootstrap units (n_loci={len(frame)}, n_units={K})",
               "estimates": {}, "vs_control": {}, "vs_candidate": {}}
        if len(frame):
            est, labels = M.endpoint_vector(frame, methods, settings)
            for (m, st, e), v in zip(labels, np.atleast_1d(est)):
                out["estimates"].setdefault(m, {}).setdefault(st, {})[e] = {"est": v, "ci_low": None, "ci_high": None}
        return out
    _, labels = M.endpoint_vector(frame, methods, settings)

    def stat(idx):
        return M.endpoint_vector(frame.take(idx), methods, settings)[0]

    res = M.cluster_bootstrap(stat, M.resampling_units(frame, spec["unit"]), n_boot=spec["n_boot"], rng=g,
                              n_jobs=spec["n_jobs"], return_replicates=True)
    pos = {lab: i for i, lab in enumerate(labels)}
    est = np.atleast_1d(res["estimate"])
    lo, hi = np.atleast_1d(res["ci_low"]), np.atleast_1d(res["ci_high"])
    out = {"unit": spec["unit"], "n_boot": spec["n_boot"], "n_clusters": res["n_clusters"],
           "n_boot_nan": res["n_boot_nan"], "estimates": {}, "vs_control": {}, "vs_candidate": {}}
    for (m, st, e), i in pos.items():
        out["estimates"].setdefault(m, {}).setdefault(st, {})[e] = {"est": est[i], "ci_low": lo[i], "ci_high": hi[i]}

    def ctr(a, b, st, e):
        c = M.contrast(res, pos[(a, st, e)], pos[(b, st, e)])
        return {"diff": c["estimate"], "ci_low": c["ci_low"], "ci_high": c["ci_high"],
                "p_greater": M.bootstrap_p_value(c, 0.0, alternative="greater"),
                "p_less": M.bootstrap_p_value(c, 0.0, alternative="less")}

    for m in methods:
        for st in settings:
            for e in ("auprc_sd", "auroc_sd", "auroc_dw"):
                if CONTROL in methods and m != CONTROL:
                    out["vs_control"].setdefault(m, {}).setdefault(st, {})[e] = ctr(m, CONTROL, st, e)
                if candidate and m != candidate and candidate in methods:
                    out["vs_candidate"].setdefault(m, {}).setdefault(st, {})[e] = ctr(candidate, m, st, e)
    return out


def evaluate(spec: dict, bdir: Path) -> dict:
    truth, preds, runtime = _load_truth_and_preds(spec, bdir)
    given = json.loads(Path(spec["operating_points_in"]).read_text()) if spec.get("operating_points_in") else None
    methods = sorted(preds)
    out = {"bench_id": spec["bench_id"], "n_loci": int(len(truth)),
           "class_counts": {k: int(v) for k, v in truth["oracle_class"].value_counts().sort_index().items()},
           "cells": sorted(truth["cell_id"].astype(str).unique().tolist()), "methods": {}, "runtime": runtime,
           "strength_control": spec["strength_control"]}
    for mode in ("least_favourable", "excluded"):
        fr = M.build_eval_frame(truth, preds, mode)
        out.setdefault("n_loci_by_mode", {})[mode] = int(len(fr))
        if len(fr) == 0:
            continue
        settings = _settings(fr)
        main = settings[0]
        ops = _op_points(fr, methods, main, given)
        for m in methods:
            cfg = _cfg(ops[m])
            mm = M.method_metrics(fr, m, cfg)
            mm["operating_points"] = ops[m]
            mm["at_operating_points"] = M.recall_at(fr, m, cfg.t_S, cfg.t_D, main, cfg)
            if mode == "least_favourable":
                lf_only = M.build_eval_frame(truth, {m: preds[m]}, "least_favourable")
                mm["failures"] = M.failure_summary(lf_only, m)
            out["methods"].setdefault(m, {})[mode] = mm
        log(f"[{mode}] bootstrap ({spec['n_boot']} x {spec['unit']}, {len(methods)} methods, settings {settings})")
        out.setdefault("bootstrap", {})[mode] = _bootstrap_block(fr, methods, settings, spec, mode, spec.get("candidate"))
        cand = spec.get("candidate")
        if cand and cand in methods and not np.isnan(fr.scores[cand]["P_D"]).all():
            bl = spec.get("c3_baselines") or [m for m in methods if m != cand and m not in EXCLUDED_FROM_C3]
            bl = [b for b in bl if b in methods and not np.isnan(fr.scores[b]["P_D"]).all()]
            if bl and n_boot_units(fr, spec["unit"]) < 2:
                out.setdefault("c3_dry_run", {})[mode] = {
                    "candidate": cand, "eligible": bl, "noninferior_all": None, "superior_one": None, "pass": None,
                    "not_estimable": f"fewer than 2 bootstrap units (n_loci={len(fr)})"}
            elif bl:
                thr = {m: {"t_S": ops[m]["used"]["t_S"], "t_D": ops[m]["used"]["t_D"]} for m in [cand] + bl}
                g = seeded_rng(BOOT_SEED, "run_benchmark", "c3", spec["bench_id"], mode)
                out.setdefault("c3_dry_run", {})[mode] = M.c3_test(
                    fr, cand, bl, thr, main, M.MetricConfig(), n_boot=spec["n_boot"], rng=g, n_jobs=spec["n_jobs"],
                    unit=spec["unit"])
        out.setdefault("settings", {})[mode] = settings
    ops_lf = {m: out["methods"][m]["least_favourable"]["operating_points"]["used"] for m in methods}
    out = M._clean(out)
    (bdir / "benchmark.json").write_text(M.dumps(out))
    (bdir / "operating_points.json").write_text(json.dumps(M._clean(ops_lf), indent=1, sort_keys=True))
    (bdir / "summary.md").write_text(summary_md(out, spec))
    return out


# ==================================================================================================
# Summary
# ==================================================================================================
def _f(x, nd=3) -> str:
    if x is None:
        return "NA"
    if isinstance(x, str):
        return x
    return f"{x:.{nd}f}"


def _ci(d: dict | None) -> str:
    if not d:
        return "NA"
    return f"{_f(d.get('est'))} [{_f(d.get('ci_low'))}, {_f(d.get('ci_high'))}]"


def summary_md(res: dict, spec: dict) -> str:
    L = [f"# Benchmark {res['bench_id']}", "",
         f"- loci: {res['n_loci']} ({', '.join(f'{k} {v}' for k, v in res['class_counts'].items())}); "
         f"cells: {len(res['cells'])}; per-cell subsample: {spec.get('per_cell')}",
         f"- bootstrap: {spec['n_boot']} replicates, resampling unit {spec['unit']} (BCa 95%); seed {spec['boot_seed']}",
         f"- strength-only control (D6(e)): table {spec['strength_control']['mode']}",
         f"- operating points: {'given ' + str(spec['operating_points_in']) if spec.get('operating_points_in') else 'fitted on this set (C3 rule, in-sample)'}",
         f"- git {spec['git']['commit'][:10]} clean={spec['git']['clean']}", ""]
    for mode in ("least_favourable", "excluded"):
        b = res.get("bootstrap", {}).get(mode)
        if not b:
            continue
        sts = res["settings"][mode]
        L += [f"## Co-primary endpoints ({mode}; n = {res['n_loci_by_mode'][mode]})", "",
              "| method | " + " | ".join(f"{st} AUPRC S-vs-D | {st} AUROC D-vs-W" for st in sts)
              + " | Δ AUPRC vs control (" + sts[0] + ") |",
              "|---|" + "---|---|" * len(sts) + "---|"]
        for m in sorted(b["estimates"]):
            row = [m]
            for st in sts:
                e = b["estimates"][m].get(st, {})
                row += [_ci(e.get("auprc_sd")), _ci(e.get("auroc_dw"))]
            vc = b["vs_control"].get(m, {}).get(sts[0], {}).get("auprc_sd")
            row.append(f"{_f(vc['diff'])} [{_f(vc['ci_low'])}, {_f(vc['ci_high'])}]" if vc else "control")
            L.append("| " + " | ".join(row) + " |")
        L.append("")
        if b.get("vs_candidate"):
            L += [f"### Candidate {spec['candidate']} minus method ({sts[0]}, {mode})", "",
                  "| method | Δ AUPRC [CI] | p (cand > m) | Δ AUROC D-vs-W [CI] | p |", "|---|---|---|---|---|"]
            for m, d in sorted(b["vs_candidate"].items()):
                a, w = d[sts[0]]["auprc_sd"], d[sts[0]]["auroc_dw"]
                L.append(f"| {m} | {_f(a['diff'])} [{_f(a['ci_low'])}, {_f(a['ci_high'])}] | {_f(a['p_greater'], 4)} | "
                         f"{_f(w['diff'])} [{_f(w['ci_low'])}, {_f(w['ci_high'])}] | {_f(w['p_greater'], 4)} |")
            L.append("")
    lf = {m: v["least_favourable"] for m, v in res["methods"].items() if "least_favourable" in v}
    L += ["## C3 operating points and calibration (least_favourable)", "",
          "| method | t_S | t_D | W→S | W→D | recall S | recall D | ECE_S raw / recal | ECE_D raw / recal | failures | median s/locus |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, v in sorted(lf.items()):
        u, r, c = v["operating_points"]["used"], v["at_operating_points"], v["calibration"]
        fl = v.get("failures", {})
        rt = [x["median_s"] for x in res["runtime"].get(m, {}).values() if x.get("median_s") is not None]
        L.append(f"| {m} | {_f(u.get('t_S'), 4)} | {_f(u.get('t_D'), 4)} | {_f(r.get('W_called_S'))} | "
                 f"{_f(r.get('W_called_D'))} | {_f(r.get('recall_S'))} | {_f(r.get('recall_D'))} | "
                 f"{_f(c.get('ece_S'))} / {_f(c.get('ece_S_recal'))} | {_f(c.get('ece_D'))} / {_f(c.get('ece_D_recal'))} | "
                 f"{fl.get('n_failed', 'NA')} (max cell {_f(fl.get('max_cell_failure_share'))}) | "
                 f"{_f(float(np.median(rt)) if rt else None, 4)} |")
    L.append("")
    cells = res["cells"]
    for key, title in (("auprc_sd", "AUPRC S vs D"), ("auroc_dw", "AUROC D vs W")):
        L += [f"## Per-cell {title} (least_favourable)", "", "| method | " + " | ".join(cells) + " |",
              "|---|" + "---|" * len(cells)]
        for m, v in sorted(lf.items()):
            L.append(f"| {m} | " + " | ".join(_f(v["per_cell"].get(c, {}).get(key)) for c in cells) + " |")
        L.append("")
    for by in ("r2", "d"):
        key = f"by_{by}_stratum_config_thresholds"
        labs = None
        for v in lf.values():
            labs = list(v.get(key, {}).keys())
            break
        if not labs:
            continue
        L += [f"## D strata by {by} (least_favourable; AUPRC S vs D_b / false-S rate at t_S)", "",
              "| method | " + " | ".join(labs) + " |", "|---|" + "---|" * len(labs)]
        for m, v in sorted(lf.items()):
            cells_ = [f"{_f(v[key][b].get('auprc_sd'))} / {_f(v[key][b].get('false_S'))} (n={v[key][b]['n']})"
                      for b in labs]
            L.append(f"| {m} | " + " | ".join(cells_) + " |")
        L.append("")
    for mode, c3 in (res.get("c3_dry_run") or {}).items():
        L += [f"## C3 dry run on this set ({mode}; thresholds {'given' if spec.get('operating_points_in') else 'fitted here — in-sample'})", "",
              f"- candidate {c3['candidate']}; eligible {c3['eligible']}; non-inferior to all: {c3['noninferior_all']}; "
              f"superior on one (Holm): {c3['superior_one']}; **pass: {c3['pass']}**", ""]
    return "\n".join(L) + "\n"


# ==================================================================================================
# Verify
# ==================================================================================================
def verify(bdir: Path, tol: float = 1e-9) -> list[str]:
    import verify_run
    spec = json.loads((bdir / "manifest.json").read_text())
    problems = []
    for m, by_set in spec["runs"].items():
        for s, d in by_set.items():
            problems += [f"run {m} on {Path(s).name}: {p}" for p in verify_run.verify(d, tol)]
    if spec["strength_control"]["sha256"] != sha256_file(spec["strength_control"]["table"]):
        problems.append("strength-control table sha256 differs from the manifest")
    ids = Path(bdir / "expected_ids.txt").read_text().split()
    for s in spec["locus_sets"]:
        in_set = set(read_locus_attrs(s)["locus_id"].astype(str))
        sub = [i for i in ids if i in in_set]
        if _ids_hash(sub) != spec["ids_sha256"][s]:
            problems.append(f"expected ids of {Path(s).name} differ from the manifest")
    for rel, sp, keep in _sub_benchmarks(spec):
        d = bdir / rel if rel else bdir
        if not (d / "benchmark.json").exists():
            problems.append(f"{rel or '.'}/benchmark.json missing")
            continue
        reported = json.loads((d / "benchmark.json").read_text())
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
            tdir = Path(td)
            exp = (bdir / "expected_ids.txt").read_text().split()
            (tdir / "expected_ids.txt").write_text("\n".join(i for i in exp if keep is None or i in keep) + "\n")
            recomputed = evaluate(sp, tdir)
        diffs = verify_run._compare(json.loads(M.dumps(recomputed)), reported, tol)
        problems += [f"{rel + '/' if rel else ''}benchmark.json mismatch {d}" for d in diffs[:50]]
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--locus-set", action="append", help="HDF5 locus set (repeatable); truth must sit alongside")
    ap.add_argument("--methods", type=Path, help="methods YAML (list `methods`)")
    ap.add_argument("--reuse", action="append", help="NAME=RUN_DIR: reuse an existing harness run (repeatable)")
    ap.add_argument("--label", default="bench")
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--per-cell", type=int, default=None, help="deterministic subsample of loci per cell")
    ap.add_argument("--method-ids", action="append", default=None, metavar="NAME=FILE",
                    help="run method NAME only on the ids in FILE (slow-method subsample; repeatable); it is "
                         "evaluated in a sub-benchmark restricted to those ids, together with every other method")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--unit", choices=("block", "locus"), default="block")
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--c3-baselines", nargs="*", default=None)
    ap.add_argument("--operating-points", default=None, help="operating_points.json of a validation benchmark")
    ap.add_argument("--strength-table", default=None, help="fixed strength-control table (else K-fold cross-fit)")
    ap.add_argument("--strength-folds", type=int, default=5)
    ap.add_argument("--control-workers", type=int, default=8)
    ap.add_argument("--no-hash-inputs", action="store_true")
    ap.add_argument("--verify", type=Path, default=None, help="recompute a benchmark directory and compare")
    a = ap.parse_args(argv)
    if os.environ.get("COLOC_ENV_ACTIVE") != "1":
        print("source env/activate.sh first", file=sys.stderr)
        return 2
    if a.verify:
        probs = verify(a.verify.resolve())
        for p in probs:
            print(f"  - {p}")
        print("BENCHMARK OK" if not probs else f"BENCHMARK VERIFY FAILED ({len(probs)} problems)")
        return 1 if probs else 0
    if not a.locus_set:
        ap.error("--locus-set is required")
    if not a.methods and not a.reuse:
        ap.error("give --methods and/or --reuse")
    run_stage(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
