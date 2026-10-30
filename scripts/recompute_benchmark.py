#!/usr/bin/env python
"""Recompute the headline simulated test-set numbers of the single test run (frozen-v1, nonce ab94c5a1) from the
shipped predictions and truth tables, and compare them with the numbers the pipeline wrote at the time.

Inputs (all in this repository):
  results/test_v1/runs/*__sim_{main,shifted}/predictions.parquet       per-locus method outputs (sha256-checked
                                                                         against each run's manifest.json)
  results/truth/testset_v1{,_shifted}.truth.parquet                     simulation truth (metric columns only)
  results/test_v1/benchmarks/sim/{expected_ids.txt, ids_subsample40.txt, manifest.json}
  results/test_v1/operating_points_used.json                           frozen validation thresholds (P3-30)

It recomputes, with the shipped src/eval/metrics.py (the code that produced the original numbers):
  * AUPRC S-vs-D (power-matched) and AUROC D-vs-W in the realistic (L1-L6), ideal (L0) and shifted settings;
  * every other entry of the per-method metrics block (per cell, per level, strata, calibration, ...);
  * the C3 three-way calls at the frozen operating points (weighted recalls / W rates, confusion matrices) and at
    fixed 0.8 thresholds (descriptive.json);
for the main benchmark (13,207 loci, every method without its own id list) and the ColocBoost subsample
(4,467 loci, 40/40/40 per cell), in both failure modes (least-favourable, excluded).

Comparison: every numeric leaf is compared with benchmarks/sim[/subsample_8685f328]/benchmark.json and
descriptive.json (tolerance 1e-6, absolute); the co-primary table is also compared, at its printed 3 decimals,
with summary.md. Bootstrap CIs are not recomputed (2,000 block-bootstrap replicates; see scripts/run_benchmark.py).

Usage:
  python scripts/recompute_benchmark.py                       # full recomputation, writes results/recompute/
  python scripts/recompute_benchmark.py --methods coloc_abf,coloc_susie_mm --modes least_favourable --no-write
      (a subset: least-favourable metrics of a method do not depend on the other methods, so the subset still
       reproduces the reference exactly; the excluded mode depends on the method set and needs all methods)
Exit status 0 iff every compared value agrees within the tolerance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eval import metrics as M  # noqa: E402

BUNDLE = ROOT / "results" / "test_v1"
TRUTH_DIR = ROOT / "results" / "truth"
BENCH = BUNDLE / "benchmarks" / "sim"
SUB = "subsample_8685f328"
SETTINGS = ("realistic", "ideal", "shifted")  # scripts/run_benchmark.py SETTINGS
MODES = ("least_favourable", "excluded")
TOL = 1e-6
SKIP_KEYS = {"operating_points", "failures"}  # added by run_benchmark around method_metrics; compared separately


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_truth() -> pd.DataFrame:
    parts = [pd.read_parquet(TRUTH_DIR / f"{n}.truth.parquet") for n in ("testset_v1", "testset_v1_shifted")]
    t = pd.concat(parts, ignore_index=True)
    for c in ("mu1", "mu2", "r2", "d", "rho"):
        t[c] = t[c].astype(np.float64)
    return t


def load_preds(manifest: dict, methods: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for m in methods:
        parts = []
        for _, run_dir in sorted(manifest["runs"][m].items()):
            d = BUNDLE / "runs" / Path(run_dir).name
            man = json.loads((d / "manifest.json").read_text())
            f = d / "predictions.parquet"
            if sha256_file(f) != man["predictions_sha256"]:
                raise SystemExit(f"{f}: sha256 differs from its manifest")
            p = pd.read_parquet(f)
            p["method"] = m
            parts.append(p)
        out[m] = pd.concat(parts, ignore_index=True)
    return out


def used_ops(m: str) -> M.MetricConfig:
    """scripts/run_benchmark.py _op_points/_cfg with a given operating-points file."""
    g = json.loads((BUNDLE / "operating_points_used.json").read_text())[m]
    u = {k: (float(g[k]) if isinstance(g.get(k), str) else g.get(k)) for k in ("t_S", "t_D", "recal_a_S", "recal_a_D")}
    for k in ("t_S", "t_D"):
        u[k] = np.inf if u.get(k) is None else float(u[k])
    return M.MetricConfig(t_S=float(u["t_S"]), t_D=float(u["t_D"]),
                          recal_a_S=None if u.get("recal_a_S") is None else float(u["recal_a_S"]),
                          recal_a_D=None if u.get("recal_a_D") is None else float(u["recal_a_D"]))


def flatten(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flatten(v, f"{pre}/{k}")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from flatten(v, f"{pre}[{i}]")
    else:
        yield pre, o


def _num(x):
    if x is None:
        return math.nan
    if isinstance(x, str):
        return {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}.get(x, x)
    if isinstance(x, (bool, np.bool_)):
        return float(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    return x


def compare(ours, ref, label: str, report: dict) -> None:
    """Every leaf of `ref` must exist in `ours` and agree within TOL (NaN/None equal)."""
    o = dict(flatten(M._clean(ours)))
    n = bad = 0
    worst = (0.0, "")
    for k, rv in flatten(ref):
        if any(f"/{s}" in k for s in SKIP_KEYS):
            continue
        n += 1
        if k not in o:
            bad += 1
            report["mismatches"].append(f"{label}{k}: missing in recomputation")
            continue
        a, b = _num(o[k]), _num(rv)
        if isinstance(a, float) and isinstance(b, float):
            if math.isnan(a) and math.isnan(b):
                continue
            if math.isinf(a) or math.isinf(b):
                ok = a == b
                diff = 0.0 if ok else math.inf
            else:
                diff = abs(a - b)
                ok = diff <= TOL
            if diff > worst[0]:
                worst = (diff, label + k)
            if not ok:
                bad += 1
                report["mismatches"].append(f"{label}{k}: {a!r} vs reference {b!r}")
        elif a != b:
            bad += 1
            report["mismatches"].append(f"{label}{k}: {a!r} vs reference {b!r}")
    report["n_compared"] += n
    report["n_mismatch"] += bad
    if worst[0] > report["max_abs_diff"]:
        report["max_abs_diff"], report["max_abs_diff_at"] = worst


def evaluate_part(truth: pd.DataFrame, preds: dict, methods: list[str], modes: list[str]) -> dict:
    out = {}
    for mode in modes:
        fr = M.build_eval_frame(truth, {m: preds[m] for m in methods}, mode)
        settings = [s for s in SETTINGS if M.cell_pooling_weights(fr, s)] or ["all"]
        main = settings[0]
        res = {"n_loci": int(len(fr)), "settings": settings, "methods": {}}
        for m in methods:
            cfg = used_ops(m)
            mm = M.method_metrics(fr, m, cfg)
            mm["at_operating_points"] = M.recall_at(fr, m, cfg.t_S, cfg.t_D, main, cfg)
            res["methods"][m] = mm
        if mode == "least_favourable":
            res["descriptive"] = {}
            for m in methods:
                if np.isnan(fr.scores[m]["P_S"]).all() and np.isnan(fr.scores[m]["P_D"]).all():
                    continue
                cfg = used_ops(m)
                res["descriptive"][m] = {
                    "matched_w": {"t_S": cfg.t_S, "t_D": cfg.t_D, "confusion": M.confusion(fr, m, cfg.t_S, cfg.t_D),
                                  "recall": M.recall_at(fr, m, cfg.t_S, cfg.t_D, main)},
                    "fixed_0.8": {"t_S": 0.8, "t_D": 0.8, "confusion": M.confusion(fr, m, 0.8, 0.8),
                                  "recall": M.recall_at(fr, m, 0.8, 0.8, main)}}
        out[mode] = res
    return out


def summary_table(path: Path, mode: str) -> dict:
    """Co-primary table of summary.md for one failure mode -> {method: [cells...]}."""
    txt = path.read_text().splitlines()
    head = f"## Co-primary endpoints ({mode};"
    i = next(i for i, l in enumerate(txt) if l.startswith(head))
    rows = {}
    for l in txt[i + 4:]:
        if not l.startswith("|"):
            break
        cells = [c.strip() for c in l.strip("|").split("|")]
        rows[cells[0]] = cells[1:]
    return rows


def check_summary(part: dict, sdir: Path, report: dict, methods: list[str]) -> None:
    for mode, res in part.items():
        try:
            rows = summary_table(sdir / "summary.md", mode)
        except StopIteration:
            continue
        for m in methods:
            if m not in rows:
                continue
            got = []
            for st in res["settings"]:
                for e in ("auprc_sd", "auroc_dw"):
                    v = res["methods"][m].get(st, {}).get(e)
                    got.append("NA" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}")
            want = [re.split(r"\s", c)[0] for c in rows[m][:len(got)]]
            report["summary_md_checked"] += len(got)
            for g, w, lab in zip(got, want, [f"{st} {e}" for st in res["settings"] for e in ("auprc_sd", "auroc_dw")]):
                if g != w:
                    report["n_mismatch"] += 1
                    report["mismatches"].append(f"summary.md [{mode}] {m} {lab}: {g} vs printed {w}")


def headline_md(parts: dict, methods_main: list[str], report: dict) -> str:
    L = ["# Recomputed test-set benchmark (frozen-v1, nonce ab94c5a1)", "",
         "Generated by `scripts/recompute_benchmark.py` from `results/test_v1/` and `results/truth/` (point estimates; the CIs in "
         "`results/test_v1/benchmarks/sim/summary.md` come from 2,000 block-bootstrap replicates and are not "
         "recomputed here).", "",
         f"- compared numeric values: {report['n_compared']} (benchmark.json + descriptive.json) and "
         f"{report['summary_md_checked']} printed summary.md cells",
         f"- mismatches (> {TOL:g} absolute): {report['n_mismatch']}; max |difference| {report['max_abs_diff']:.3g}",
         ""]
    for name, part in parts.items():
        lf = part.get("least_favourable")
        if not lf:
            continue
        L += [f"## {name}: co-primary endpoints (failures least-favourable; n = {lf['n_loci']})", "",
              "| method | realistic AUPRC S-vs-D | ideal AUPRC S-vs-D | shifted AUPRC S-vs-D | realistic AUROC D-vs-W |"
              " ideal AUROC D-vs-W | shifted AUROC D-vs-W |", "|---|---|---|---|---|---|---|"]
        for m, mm in lf["methods"].items():
            f = lambda st, e: ("NA" if mm.get(st, {}).get(e) is None or math.isnan(mm[st][e]) else f"{mm[st][e]:.4f}")  # noqa: E731
            L.append(f"| {m} | {f('realistic', 'auprc_sd')} | {f('ideal', 'auprc_sd')} | {f('shifted', 'auprc_sd')} | "
                     f"{f('realistic', 'auroc_dw')} | {f('ideal', 'auroc_dw')} | {f('shifted', 'auroc_dw')} |")
        L += ["", f"## {name}: C3 three-way calls at the frozen operating points (realistic, least-favourable)", "",
              "| method | t_S | t_D | recall S (weighted) | recall D (weighted) | W called S | W called D | "
              "D loci called S (unweighted) |", "|---|---|---|---|---|---|---|---|"]
        for m, d in lf.get("descriptive", {}).items():
            mw = d["matched_w"]
            r, c = mw["recall"], mw["confusion"]
            nD = sum(c.get("D", {}).values())
            fs = c.get("D", {}).get("S", 0) / nD if nD else float("nan")
            L.append(f"| {m} | {mw['t_S']:.4f} | {mw['t_D']:.4f} | {r['recall_S']:.3f} | {r['recall_D']:.3f} | "
                     f"{r['W_called_S']:.3f} | {r['W_called_D']:.3f} | {fs:.3f} |")
        L.append("")
    if report["mismatches"]:
        L += ["## Mismatches", ""] + [f"- {x}" for x in report["mismatches"][:200]]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--methods", help="comma-separated subset (default: all)")
    ap.add_argument("--modes", default=",".join(MODES), help="least_favourable,excluded")
    ap.add_argument("--parts", default=f"main,{SUB}", help=f"main,{SUB}")
    ap.add_argument("--out", default=str(ROOT / "results" / "recompute"))
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    man = json.loads((BENCH / "manifest.json").read_text())
    mids = man.get("method_ids") or {}
    want = set(a.methods.split(",")) if a.methods else None
    modes = [m for m in a.modes.split(",") if m]
    if want is not None and "excluded" in modes and want != set(man["runs"]):
        print("note: the excluded mode depends on the full method set; with --methods only least_favourable is "
              "compared", file=sys.stderr)
        modes = [m for m in modes if m != "excluded"]
    truth_all = load_truth()
    report = {"n_compared": 0, "n_mismatch": 0, "max_abs_diff": 0.0, "max_abs_diff_at": "", "summary_md_checked": 0,
              "mismatches": []}
    parts, methods_main = {}, []
    ref_desc = json.loads((BUNDLE / "descriptive.json").read_text())["methods"]
    for part in a.parts.split(","):
        if part == "main":
            sdir = BENCH
            ids = (BENCH / "expected_ids.txt").read_text().split()
            methods = sorted(m for m in man["runs"] if m not in mids)
        else:
            sdir = BENCH / part
            keep = set((BENCH / "ids_subsample40.txt").read_text().split())
            ids = [i for i in (BENCH / "expected_ids.txt").read_text().split() if i in keep]
            if (sdir / "expected_ids.txt").read_text().split() != ids:
                raise SystemExit(f"{part}: expected ids differ from ids_subsample40 within expected_ids")
            methods = sorted(man["runs"])
        if want is not None:
            methods = [m for m in methods if m in want]
        if not methods:
            continue
        if part == "main":
            methods_main = methods
        idset = set(ids)
        truth = truth_all[truth_all["locus_id"].astype(str).isin(idset)].reset_index(drop=True)
        if len(truth) != len(ids):
            raise SystemExit(f"{part}: truth rows {len(truth)} != expected ids {len(ids)}")
        preds = {m: p[p["locus_id"].astype(str).isin(idset)].copy() for m, p in load_preds(man, methods).items()}
        res = evaluate_part(truth, preds, methods, modes)
        ref = json.loads((sdir / "benchmark.json").read_text())
        for mode, r in res.items():
            if r["n_loci"] != ref["n_loci_by_mode"][mode]:
                report["n_mismatch"] += 1
                report["mismatches"].append(f"{part}/{mode}: n_loci {r['n_loci']} vs {ref['n_loci_by_mode'][mode]}")
            for m in methods:
                compare(r["methods"][m], ref["methods"][m][mode], f"{part}/{mode}/{m}", report)
                if mode in ref.get("bootstrap", {}):  # point estimates behind the printed CIs
                    est = ref["bootstrap"][mode]["estimates"].get(m, {})
                    for st, d in est.items():
                        for e, v in d.items():
                            compare({"x": r["methods"][m].get(st, {}).get(e)}, {"x": v["est"]},
                                    f"{part}/{mode}/{m}/bootstrap_est/{st}/{e}", report)
            if part == "main" and "descriptive" in r:
                for m, d in r["descriptive"].items():
                    if m in ref_desc:
                        compare(d, {k: ref_desc[m][k] for k in ("matched_w", "fixed_0.8")}, f"descriptive/{m}", report)
        check_summary(res, sdir, report, methods)
        parts[part] = res
    report["wall_s"] = round(time.time() - t0, 1)
    report["methods"] = sorted({m for p in parts.values() for r in p.values() for m in r["methods"]})
    report["modes"] = modes
    ok = report["n_mismatch"] == 0 and report["n_compared"] > 0
    report["verdict"] = "REPRODUCED" if ok else "MISMATCH"
    print(f"{report['verdict']}: {report['n_compared']} values compared with benchmark.json/descriptive.json and "
          f"{report['summary_md_checked']} summary.md cells; mismatches {report['n_mismatch']}; max |diff| "
          f"{report['max_abs_diff']:.3g} ({report['max_abs_diff_at']}); {report['wall_s']} s")
    for x in report["mismatches"][:20]:
        print("  ", x)
    if not a.no_write:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "recompute_test_v1.json").write_text(json.dumps(M._clean({"report": report, "parts": parts}),
                                                               indent=1, sort_keys=True, allow_nan=False))
        (out / "recompute_test_v1.md").write_text(headline_md(parts, methods_main, report))
        print(f"wrote {out / 'recompute_test_v1.json'} and {out / 'recompute_test_v1.md'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
