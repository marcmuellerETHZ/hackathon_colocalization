#!/usr/bin/env python
"""p4 validation baseline runs: subsample id lists, benchmarks on a fixed id list, and the M4 summary.

Evaluation side (reads oracle labels; lives outside src/baselines, which never touches truth).

  subsample screen --set data/processed/sets/screen_v2.h5 --out results/baselines_screen_v2/ids
      -> L0P1_all.txt (2,000), rest_all.txt (10,000), rest_slow.txt (5 x 200, by bucket), subsample.json
  subsample m4 --set data/processed/sets/valid_grid_v1.h5 --out results/baselines_m4/m4_subsample_v1
      -> main.txt, slow.txt, mvsusie.txt (nested; DECISIONS P3-13, P3-17), subsample.json
  bench --ids-file F [--reuse-from RUNS_DIR ...] <scripts/run_benchmark.py arguments>
      -> scripts/run_benchmark.py with its expected ids = the given list (instead of the whole set or its
         --per-cell draw); every reused run is filtered to these ids (missing ids = failures, outer join).
  collect --runs-dir D --group NAME=IDS_FILE ... --out DIR   (one predictions parquet + index.json)
  m4summary --main B1 --slow B2 --mvsusie B3 --out reports/... (json + markdown tables for the report)
  op2 --set S [--runs-dir D --ids-file F] [--pred P] --out J   (OP1 + OP2 per method, DESIGN_P2 §3; phase 2)

The subsample draws are in src/baselines/subsample.py (pure functions, tested in tests/baselines).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _write_ids(path: Path, ids: list[str]) -> str:
    path.write_text("\n".join(ids) + "\n")
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def cmd_subsample(a) -> int:
    from baselines import subsample as SS
    from eval.io import read_locus_attrs, truth_path_for
    from eval.truth import read_truth

    s = Path(a.set)
    attrs = read_locus_attrs(s)  # test-split guard on every locus
    tr = read_truth(truth_path_for(s))
    if set(tr["locus_id"].astype(str)) != set(attrs["locus_id"].astype(str)):
        raise SystemExit("truth ids differ from the locus set ids")
    if (tr["split"].astype(str) != "valid").any():
        raise SystemExit("non-validation loci in the set")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = tr[["locus_id", "cell_id", "oracle_class", "bucket"]].copy()
    df["locus_id"] = df["locus_id"].astype(str)
    man = {"locus_set": str(s), "n_loci_in_set": int(len(df)), "files": {}}
    if a.which == "screen":
        slow = SS.screen_slow_subsample(df)
        full = set(SS.SCREEN_FULL_CELLS)
        l0 = sorted(df.loc[df["cell_id"].isin(full), "locus_id"])
        rest = sorted(df.loc[~df["cell_id"].isin(full), "locus_id"])
        rest_slow = sorted(set(slow) - set(l0))
        lists = {"L0P1_all": l0, "rest_all": rest, "rest_slow": rest_slow}
        man["rule"] = ("slow methods (mvsusie, colocboost, colocboost_m1): all loci of L0|P1|r_p=0.0 + 200 per "
                       "other cell, equal over generation buckets (50 each), seed derive(20261027, "
                       "'screen_slow_subsample'), draw in src/baselines/subsample.py; fast methods: all loci")
    else:
        lists = SS.m4_subsample(df)
        man["rule"] = ("DECISIONS P3-13/P3-17: per cell 100 S/100 D/100 W + G, U up to 30 each (main, all methods); "
                       "40/40/40 (slow: colocboost, colocboost_m1); 20/20/20 (mvsusie); nested; seed "
                       "derive(20261029, 'm4_subsample'); draw in src/baselines/subsample.py")
    sub = df.set_index("locus_id")
    for name, ids in lists.items():
        sha = _write_ids(out / f"{name}.txt", ids)
        d = sub.loc[ids]
        man["files"][f"{name}.txt"] = {
            "n": len(ids), "ids_sha256": sha,
            "by_cell_class": {f"{c}|{k}": int(v) for (c, k), v in d.groupby(["cell_id", "oracle_class"]).size().items()},
            "by_class": {k: int(v) for k, v in d["oracle_class"].value_counts().sort_index().items()}}
    (out / "subsample.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    print(json.dumps({k: v["n"] for k, v in man["files"].items()}))
    return 0


def cmd_bench(a, rest: list[str]) -> int:
    import run_benchmark as RB

    want = [l.strip() for l in Path(a.ids_file).read_text().splitlines() if l.strip()]
    want_set = set(want)
    orig = RB.expected_ids

    def expected_ids(locus_set, per_cell):
        if per_cell:
            raise SystemExit("--per-cell cannot be combined with --ids-file")
        all_ids = orig(locus_set, None)  # applies the test-split guard
        missing = want_set - set(all_ids)
        if missing:
            raise SystemExit(f"{len(missing)} ids of {a.ids_file} are not in {locus_set}")
        return [i for i in all_ids if i in want_set]

    RB.expected_ids = expected_ids
    full: dict[str, str] = {}
    for d in a.reuse_from or []:
        full.update(find_runs(Path(d), want_set, a.exclude or []))
    # methods run in several disjoint parts (e.g. two half-jobs to fit the RAM budget): the first part is passed
    # as the reused run, the others are merged into its predictions at evaluation (recorded in the benchmark
    # manifest as "extra_runs"; scripts/run_benchmark.py --verify does not know them).
    extra: dict[str, list[str]] = {}
    for name, parts in find_partial_runs([Path(d) for d in a.reuse_from or []], want_set,
                                         (a.exclude or []) + list(full)).items():
        full[name] = parts[0]
        extra[name] = parts[1:]
    for name, run_dir in sorted(full.items()):
        rest += ["--reuse", f"{name}={run_dir}"]
    print("[baselines_p4 bench] " + " ".join(rest) + f" extra_runs={extra}", flush=True)
    if extra:
        orig_load, orig_eval = RB._load_truth_and_preds, RB.evaluate

        def load(spec, bdir):
            truth, preds, runtime = orig_load(spec, bdir)
            ids = set(Path(bdir / "expected_ids.txt").read_text().split())
            for m, dirs in (spec.get("extra_runs") or {}).items():
                parts = [preds[m]]
                for d in dirs:
                    d = Path(d)
                    man = json.loads((d / "manifest.json").read_text())
                    if RB.sha256_file(d / "predictions.parquet") != man["predictions_sha256"]:
                        raise ValueError(f"{d}: predictions sha256 differs from its manifest")
                    p = pd.read_parquet(d / "predictions.parquet")
                    p = p[p["locus_id"].astype(str).isin(ids)].copy()
                    p["method"] = m
                    parts.append(p)
                allp = pd.concat(parts, ignore_index=True)
                if allp["locus_id"].duplicated().any():
                    raise ValueError(f"{m}: overlapping parts")
                preds[m] = allp
                rt = allp["runtime_s"].to_numpy(float)
                for k in runtime[m]:
                    runtime[m][k].update(n=int(len(allp)), median_s=float(np.nanmedian(rt)),
                                         p95_s=float(np.nanpercentile(rt, 95)), sum_s=float(np.nansum(rt)),
                                         extra_runs=[str(x) for x in dirs])
            return truth, preds, runtime

        def evaluate(spec, bdir):
            spec["extra_runs"] = extra
            return orig_eval(spec, bdir)

        RB._load_truth_and_preds, RB.evaluate = load, evaluate
    return RB.main(rest)


def find_partial_runs(runs_dirs: list[Path], ids: set[str], exclude: list[str]) -> dict[str, list[str]]:
    """Per method (not in exclude): runs that are pairwise disjoint on `ids` and together cover every id of `ids`
    (newest first when choosing; ids outside `ids` are filtered out at evaluation). Methods without such a cover
    are omitted."""
    cand: dict[str, list[tuple[str, set]]] = {}
    for rd in runs_dirs:
        for man_path in sorted(rd.glob("*/manifest.json")):
            d = man_path.parent
            if not (d / "predictions.parquet").exists():
                continue
            name = json.loads(man_path.read_text()).get("method_name")
            if not name or name in exclude:
                continue
            got = set(pd.read_parquet(d / "predictions.parquet", columns=["locus_id"])["locus_id"].astype(str))
            got &= ids
            if got and got != ids:
                cand.setdefault(name, []).append((str(d), got))
    out = {}
    for name, runs in cand.items():
        chosen, cov = [], set()
        for d, got in sorted(runs, key=lambda x: Path(x[0]).name, reverse=True):
            if not (got & cov):
                chosen.append(d)
                cov |= got
        if cov == ids and len(chosen) > 1:
            out[name] = sorted(chosen)
    return out


def find_runs(runs_dir: Path, ids: set[str], exclude: list[str]) -> dict[str, str]:
    """Harness run directories under runs_dir whose predictions cover every id (one per method; the newest by
    run id if several qualify). Runs that do not cover the ids (other cells / smaller samples) are skipped."""
    out: dict[str, str] = {}
    for man_path in sorted(runs_dir.glob("*/manifest.json")):
        d = man_path.parent
        if not (d / "predictions.parquet").exists():
            continue
        man = json.loads(man_path.read_text())
        name = man.get("method_name")
        if not name or name in exclude:
            continue
        got = set(pd.read_parquet(d / "predictions.parquet", columns=["locus_id"])["locus_id"].astype(str))
        if ids <= got:
            if name in out:
                print(f"[baselines_p4 bench] {name}: several runs cover the ids; using the newer {d.name}", flush=True)
            out[name] = str(d)
    return out


# ------------------------------------------------------------------------------------------------ collect
def cmd_collect(a) -> int:
    """Combine harness runs into one predictions parquet + index.json.

    --group NAME=IDS_FILE (repeatable): an id group (e.g. a cell group or a subsample). For every method, the newest
    run under --runs-dir covering the group's ids is used for that group (a method with no covering run for a group
    is listed as not run there). A method's rows are the union over its groups; a locus may appear in only one group
    per method (groups must be disjoint for a method)."""
    from eval.harness import sha256_file

    runs_dirs = [Path(d) for d in a.runs_dir]
    groups = {}
    for g in a.group:
        name, _, f = g.partition("=")
        groups[name] = [l.strip() for l in Path(f).read_text().splitlines() if l.strip()]
    index = {"runs_dirs": [str(d) for d in runs_dirs], "groups": {g: {"ids_file": dict(x.partition("=")[::2] for x in a.group)[g],
                                                        "n": len(v)} for g, v in groups.items()},
             "methods": {}}
    parts = []
    have: dict[str, set] = {}
    for gname, ids in groups.items():
        found = {}
        for rd in runs_dirs:
            found.update(find_runs(rd, set(ids), a.exclude or []))
        for m, d in sorted(found.items()):
            if set(ids) <= have.get(m, set()):
                index["methods"].setdefault(m, {})[gname] = {"covered_by_earlier_group": True}
                continue  # e.g. a fast method's rest_all run already covers the slow subsample
            have.setdefault(m, set()).update(ids)
            man = json.loads((Path(d) / "manifest.json").read_text())
            if sha256_file(Path(d) / "predictions.parquet") != man["predictions_sha256"]:
                raise SystemExit(f"{d}: predictions sha256 differs from manifest")
            p = pd.read_parquet(Path(d) / "predictions.parquet")
            p = p[p["locus_id"].astype(str).isin(set(ids))].copy()
            p["run_id"] = man["run_id"]
            p["group"] = gname
            parts.append(p)
            st = p["status"].value_counts().to_dict()
            rt = p["runtime_s"].to_numpy(float)
            index["methods"].setdefault(m, {})[gname] = {
                "run_dir": str(d), "run_id": man["run_id"], "commit": man["git"]["commit"], "n": int(len(p)),
                "status": {k: int(v) for k, v in st.items()}, "timeout_s": man["harness"]["timeout_s_per_locus"],
                "n_workers": man["harness"]["n_workers"], "wall_s": man.get("wall_s"),
                "runtime_median_s": float(np.median(rt)), "runtime_p95_s": float(np.percentile(rt, 95))}
    allp = pd.concat(parts, ignore_index=True)
    dup = allp.duplicated(["locus_id", "method"])
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} duplicate (locus, method) rows: groups overlap for a method")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    allp.to_parquet(out / "predictions.parquet", index=False)
    index["n_rows"] = int(len(allp))
    index["n_by_method"] = {k: int(v) for k, v in allp["method"].value_counts().sort_index().items()}
    index["predictions_sha256"] = sha256_file(out / "predictions.parquet")
    (out / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    print(json.dumps(index["n_by_method"], indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ operating points
def cmd_oppoints(a) -> int:
    """C3 operating points (t_S, t_D) and logit-intercept recalibration per method, fitted by the SAME code as
    scripts/run_benchmark.py (_op_points on a least-favourable frame, main setting = first pooled setting present),
    on each method's own predicted loci (no bootstrap). Used for (i) M4 methods whose runs finished after the
    benchmark and (ii) the P3-27 screen_v2-derived fallback for methods without finished M4 runs."""
    import run_benchmark as RB
    from eval import metrics as M
    from eval.io import truth_path_for
    from eval.truth import read_truth

    truth = read_truth(truth_path_for(Path(a.set)))
    preds = pd.concat([pd.read_parquet(f) for f in a.pred], ignore_index=True)
    out = {"set": a.set, "pred_files": a.pred, "label": a.label, "methods": {}}
    for m in a.methods:
        p = preds[preds["method"] == m].copy()
        if p.empty:
            out["methods"][m] = {"error": "no predictions"}
            continue
        if p["locus_id"].duplicated().any():
            raise SystemExit(f"{m}: duplicate locus rows")
        tr = truth[truth["locus_id"].astype(str).isin(set(p["locus_id"].astype(str)))]
        fr = M.build_eval_frame(tr, {m: p}, "least_favourable")
        sts = RB._settings(fr)
        ops = RB._op_points(fr, [m], sts[0], None)[m]["fitted_here"]
        st = p["status"].value_counts().to_dict()
        out["methods"][m] = {**{k: ops.get(k) for k in ("t_S", "t_D", "recal_a_S", "recal_a_D", "rate_S_on_W",
                                                          "rate_D_on_W", "setting")},
                             "n_loci": int(len(p)), "status": {k: int(v) for k, v in st.items()},
                             "cells": int(tr["cell_id"].nunique()),
                             "class_counts": {k: int(v) for k, v in tr["oracle_class"].value_counts().items()}}
    Path(a.out).write_text(json.dumps(M._clean(out), indent=1, sort_keys=True, default=float))
    for m, v in out["methods"].items():
        print(m, {k: v.get(k) for k in ("t_S", "t_D", "recal_a_S", "recal_a_D", "setting", "n_loci")})
    return 0


# ------------------------------------------------------------------------------------------------ OP1 + OP2 (phase 2)
def _gather_preds(runs_dirs: list[Path], ids: set[str] | None, pred_files: list[str],
                  methods: list[str] | None) -> tuple[dict[str, pd.DataFrame], dict]:
    """Per method: predictions from parquet files (method column) and/or harness run dirs. With `ids`, a run dir
    method must cover the ids with one run or with disjoint partial runs (find_runs / find_partial_runs), and its
    rows are restricted to the ids; sha256 of every run's predictions is checked against its manifest."""
    from eval.harness import sha256_file

    out: dict[str, list] = {}
    src: dict[str, list] = {}
    for f in pred_files:
        df = pd.read_parquet(f)
        for m, g in df.groupby("method"):
            out.setdefault(str(m), []).append(g)
            src.setdefault(str(m), []).append(str(f))
    if runs_dirs:
        if ids is None:
            raise SystemExit("--runs-dir needs --ids-file")
        full: dict[str, list[str]] = {}
        for d in runs_dirs:
            for m, rd in find_runs(d, ids, []).items():
                full[m] = [rd]
        for m, parts in find_partial_runs(runs_dirs, ids, list(full)).items():
            full[m] = parts
        for m, dirs in full.items():
            if m in out:
                continue  # an explicit --pred file wins
            for d in dirs:
                man = json.loads((Path(d) / "manifest.json").read_text())
                if sha256_file(Path(d) / "predictions.parquet") != man["predictions_sha256"]:
                    raise SystemExit(f"{d}: predictions sha256 differs from manifest")
                out.setdefault(m, []).append(pd.read_parquet(Path(d) / "predictions.parquet"))
                src.setdefault(m, []).append(str(d))
    preds = {}
    for m, parts in out.items():
        if methods and m not in methods:
            continue
        p = pd.concat(parts, ignore_index=True)
        p["method"] = m
        if ids is not None:
            p = p[p["locus_id"].astype(str).isin(ids)]
            if set(p["locus_id"].astype(str)) != ids:
                raise SystemExit(f"{m}: predictions do not cover the ids ({p['locus_id'].nunique()} of {len(ids)})")
        if p["locus_id"].duplicated().any():
            raise SystemExit(f"{m}: duplicate locus rows")
        preds[m] = p
    return preds, src


def _at_op(fr, m, t_S, t_D, setting) -> dict:
    """Three-way calls at (t_S, t_D): weighted recalls / W rates (metrics.recall_at) and the unweighted false-S
    share on power-matched D loci (the OP2 D-side population) and on all D loci of the frame."""
    from eval import metrics as M

    r = M.recall_at(fr, m, t_S, t_D, setting)
    calls = M.three_way_calls(fr, m, t_S, t_D)
    isD = fr.cls == "D"
    matched = isD & (M.sd_weights(fr, setting) > 0)
    insetting = isD & (M.dw_weights(fr, setting) > 0)
    r["false_S_on_D_matched_unweighted"] = float((calls[matched] == "S").mean()) if matched.any() else float("nan")
    r["false_S_on_D_setting_unweighted"] = float((calls[insetting] == "S").mean()) if insetting.any() else float("nan")
    r["false_S_on_D_all_unweighted"] = float((calls[isD] == "S").mean()) if isD.any() else float("nan")
    return r


def cmd_op2(a) -> int:
    """OP1 (matched-W C3, the SAME code path as run_benchmark/_op_points and `oppoints`) and OP2 (DESIGN_P2 §3,
    eval.op2.op2_thresholds) per method, each method on its own predicted loci (least-favourable failure
    scoring), in the fit setting (default: run_benchmark's main setting = first pooled setting present, i.e.
    "realistic": M4 cells L1-L6; screen_v2 cells L2 + L4). No bootstrap. Calls at both operating points are
    summarised (weighted recalls, W rates, unweighted false-S on D)."""
    import run_benchmark as RB
    from eval import metrics as M
    from eval import op2 as O2
    from eval.io import truth_path_for
    from eval.truth import read_truth

    truth = pd.concat([read_truth(truth_path_for(Path(s))) for s in a.set], ignore_index=True)
    if (truth["split"].astype(str).isin(["test"])).any():
        raise SystemExit("test-split truth rows: refused")
    ids = set(Path(a.ids_file).read_text().split()) if a.ids_file else None
    preds, src = _gather_preds([Path(d) for d in a.runs_dir or []], ids, a.pred or [], a.methods)
    out = {"sets": a.set, "ids_file": a.ids_file, "label": a.label, "sources": src, "d_limit": a.d_limit,
           "definition": {"OP1": M.matched_w_thresholds.__doc__.strip(), "OP2": O2.op2_thresholds.__doc__.strip()},
           "methods": {}}
    for m in sorted(preds):
        if m in (a.exclude or []):
            continue
        p = preds[m]
        tr = truth[truth["locus_id"].astype(str).isin(set(p["locus_id"].astype(str)))]
        fr = M.build_eval_frame(tr, {m: p}, "least_favourable")
        setting = a.setting or RB._settings(fr)[0]
        op1 = RB._op_points(fr, [m], setting, None)[m]["fitted_here"]
        op2 = O2.op2_thresholds(fr, m, setting, d_limit=a.d_limit)
        has_ps = not np.isnan(fr.scores[m]["P_S"]).all()
        has_pd = not np.isnan(fr.scores[m]["P_D"]).all()
        rec = {"n_loci": int(len(p)), "status": {k: int(v) for k, v in p["status"].value_counts().items()},
               "n_failed": int(fr.failed[m].sum()), "setting": setting, "has_P_S": has_ps, "has_P_D": has_pd,
               "class_counts": {k: int(v) for k, v in tr["oracle_class"].value_counts().items()},
               "OP1": op1, "OP2": op2}
        if has_ps or has_pd:
            rec["at_OP1"] = _at_op(fr, m, float(op1["t_S"]), float(op1["t_D"]), setting)
            rec["at_OP2"] = _at_op(fr, m, float(op2.get("t_S_call", np.inf)), float(op2.get("t_D", np.inf)), setting)
        out["methods"][m] = rec
        print(f"{m:28s} n={len(p):6d} OP1 t_S={op1['t_S']:.6g} t_D={op1['t_D']:.6g} | OP2 t_S={op2.get('t_S', np.inf):.6g}"
              f" feasible={op2.get('op2_feasible')} t_D={op2.get('t_D', np.inf):.6g} "
              f"S_on_W={op2.get('rate_S_on_W', np.nan):.4f} S_on_D={op2.get('rate_S_on_D', np.nan):.4f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(M._clean(out), indent=1, sort_keys=True, default=float))
    return 0


def cmd_opfinal(a) -> int:
    """Phase-2 operating points per method (DESIGN_P2 §3, §5): OP1 = the frozen phase-1 value (P3-30, --frozen-op1)
    except for the methods in --refit-op1 (coloc_susie_mm: its M4 OP1 replaces the P3-27 screen fallback); OP2 =
    fitted on M4 (--op2 files of `op2`). Every frozen OP1 is compared with the OP1 refitted by `op2` on the same
    runs (reported as op1_refit_matches_frozen)."""
    from eval import metrics as M

    frozen = json.loads(Path(a.frozen_op1).read_text())["methods"]
    fits = {}
    for f in a.op2:
        d = json.loads(Path(f).read_text())
        for m, v in d["methods"].items():
            fits[m] = dict(v, fit_file=str(f), fit_label=d.get("label"))
    def fl(x):
        return np.inf if x is None or (isinstance(x, str) and x == "inf") else float(x)

    out = {"definition": cmd_opfinal.__doc__.strip(), "frozen_op1_file": a.frozen_op1, "op2_files": a.op2,
           "excluded": a.exclude or [], "methods": {}}
    for m in sorted(set(frozen) | set(fits)):
        if m in (a.exclude or []):
            continue
        fz, ft = frozen.get(m), fits.get(m)
        rec = {}
        if m in (a.refit_op1 or []):
            o = ft["OP1"]
            rec["OP1"] = {"t_S": fl(o["t_S"]), "t_D": fl(o["t_D"]), "recal_a_S": o.get("recal_a_S"),
                          "recal_a_D": o.get("recal_a_D"),
                          "source": f"M4 refit (DESIGN_P2 §5; replaces {fz.get('source') if fz else 'none'}): "
                                    f"{ft['fit_label']}"}
        elif fz is not None:
            rec["OP1"] = {k: fz.get(k) for k in ("t_S", "t_D", "recal_a_S", "recal_a_D")}
            rec["OP1"].update(t_S=fl(fz["t_S"]), t_D=fl(fz["t_D"]), source=f"frozen phase-1 (P3-30): {fz.get('source')}")
        if ft is not None:
            if fz is not None:
                rec["op1_refit_matches_frozen"] = bool(fl(ft["OP1"]["t_S"]) == fl(fz["t_S"])
                                                       and fl(ft["OP1"]["t_D"]) == fl(fz["t_D"]))
            o2 = ft["OP2"]
            rec["OP2"] = {k: o2.get(k) for k in ("t_S", "t_S_call", "op2_feasible", "t_S_W_side", "t_S_D_side",
                                                  "t_D", "rate_S_on_W", "rate_S_on_D", "rate_D_on_W", "n_W",
                                                  "n_D_matched", "setting", "d_limit", "w_limit")}
            rec["OP2"]["source"] = f"M4 fit: {ft['fit_label']}"
            rec["at_OP1_M4"] = ft.get("at_OP1")
            rec["at_OP2_M4"] = ft.get("at_OP2")
            rec["n_loci_M4"] = ft["n_loci"]
        out["methods"][m] = rec
    Path(a.out).write_text(json.dumps(M._clean(out), indent=1, sort_keys=True, default=float))
    L = ["| method | OP1 t_S | OP1 t_D | OP1 source | OP2 t_S | feasible | OP2 t_D | W→S at OP2 | false-S on D at OP2 | "
         "OP1 refit = frozen |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in out["methods"].items():
        o1, o2 = r.get("OP1", {}), r.get("OP2", {})
        L.append(f"| {m} | {o1.get('t_S')} | {o1.get('t_D')} | {str(o1.get('source', ''))[:40]} | {o2.get('t_S')} | "
                 f"{o2.get('op2_feasible')} | {o2.get('t_D')} | {o2.get('rate_S_on_W')} | {o2.get('rate_S_on_D')} | "
                 f"{r.get('op1_refit_matches_frozen')} |")
    Path(a.out).with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


# ------------------------------------------------------------------------------------------------ quick co-primaries
def cmd_quickeval(a) -> int:
    """Co-primary endpoints (realistic / ideal; AUPRC S-vs-D, AUROC D-vs-W) with a joint 2,000-replicate BCa
    LD-block bootstrap for the given methods on the loci all of them predicted (least-favourable failures), without
    the strength-control refit of run_benchmark (used for M4 runs that finished after the benchmark; the
    strength_only rows of the main benchmark can be passed as one of the prediction files)."""
    from common.seeds import rng as seeded_rng
    from eval import metrics as M
    from eval.io import truth_path_for
    from eval.truth import read_truth

    truth = read_truth(truth_path_for(Path(a.set)))
    preds = pd.concat([pd.read_parquet(f) for f in a.pred], ignore_index=True)
    ids = set(Path(a.ids_file).read_text().split())
    by = {m: preds[(preds["method"] == m) & preds["locus_id"].astype(str).isin(ids)].copy() for m in a.methods}
    for m, p in by.items():
        if set(p["locus_id"].astype(str)) != ids:
            raise SystemExit(f"{m}: predictions do not cover the ids ({len(p)} of {len(ids)})")
    tr = truth[truth["locus_id"].astype(str).isin(ids)]
    out = {"set": a.set, "ids_file": a.ids_file, "n_loci": int(len(tr)), "label": a.label, "n_boot": a.n_boot,
           "unit": "block", "seed": [20261029, "baselines_p4", "quickeval", a.label], "modes": {}}
    for mode in ("least_favourable", "excluded"):
        fr = M.build_eval_frame(tr, by, mode)
        sts = [s for s in ("realistic", "ideal") if M.cell_pooling_weights(fr, s)]
        g = seeded_rng(20261029, "baselines_p4", "quickeval", a.label, mode)
        _, labels = M.endpoint_vector(fr, a.methods, sts)
        res = M.cluster_bootstrap(lambda idx: M.endpoint_vector(fr.take(idx), a.methods, sts)[0],
                                  M.resampling_units(fr, "block"), n_boot=a.n_boot, rng=g, n_jobs=a.n_jobs)
        est, lo, hi = (np.atleast_1d(res[k]) for k in ("estimate", "ci_low", "ci_high"))
        out["modes"][mode] = {"n": int(len(fr)), "estimates": {f"{m}|{st}|{e}": {"est": float(est[i]), "ci_low":
                              float(lo[i]), "ci_high": float(hi[i])} for i, (m, st, e) in enumerate(labels)}}
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    for k, v in out["modes"]["least_favourable"]["estimates"].items():
        print(k, f"{v['est']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]")
    return 0


# ------------------------------------------------------------------------------------------------ M4 summary
def _est(b, m, st, e):
    try:
        return b["estimates"][m][st][e]
    except KeyError:
        return None


def _fmt(d, nd=3):
    if not d or d.get("est") is None:
        return "NA"
    return f"{d['est']:.{nd}f} [{d['ci_low']:.{nd}f}, {d['ci_high']:.{nd}f}]"


def _runtime(res, m):
    rt = [x for x in res["runtime"].get(m, {}).values()]
    return (rt[0]["median_s"], rt[0]["p95_s"]) if rt else (None, None)


def mvsusie_soft_sensitivity(bdir: Path) -> dict:
    """Point estimates of the co-primaries for mvSuSiE with the soft rule (q = 1 - lfsr, D4; P3-2 sensitivity),
    rebuilt from extra["soft"] of the hard-rule run, next to the hard rule, on the same frame."""
    import run_benchmark as RB
    from eval import metrics as M

    spec = json.loads((bdir / "manifest.json").read_text())
    truth, preds, _ = RB._load_truth_and_preds(spec, bdir)
    if "mvsusie" not in preds:
        return {}
    hard = preds["mvsusie"].copy()
    soft = hard.copy()
    ex = [json.loads(e) if isinstance(e, str) else {} for e in hard["extra"]]
    for k in ("s", "P_S", "P_D", "P_W"):
        soft[k] = [float(x["soft"][k]) if x.get("soft") and x["soft"].get(k) is not None else np.nan for x in ex]
    bad = hard["status"].isin(["failed", "timeout"])
    soft.loc[bad, ["s", "P_S", "P_D", "P_W"]] = np.nan
    soft["method"] = "mvsusie_soft"
    out = {}
    for mode in ("least_favourable", "excluded"):
        fr = M.build_eval_frame(truth, {"mvsusie": hard, "mvsusie_soft": soft}, mode)
        sts = [s for s in ("realistic", "ideal") if M.cell_pooling_weights(fr, s)]
        vals, labels = M.endpoint_vector(fr, ["mvsusie", "mvsusie_soft"], sts)
        out[mode] = {f"{m}|{st}|{e}": float(v) for (m, st, e), v in zip(labels, np.atleast_1d(vals))}
    return out


def cmd_m4summary(a) -> int:
    rows, out = [], {"benchmarks": {}, "coloc_abf_W_selection": None, "mvsusie_soft": None}
    sources = [(t, Path(d)) for t, d in (("main", a.main), ("slow", a.slow), ("mvsusie", a.mvsusie)) if d]
    out["missing_benchmarks"] = [t for t, d in (("slow", a.slow), ("mvsusie", a.mvsusie)) if not d]
    slow_methods = {"main": None, "slow": ("colocboost", "colocboost_m1"), "mvsusie": ("mvsusie",)}
    for tag, bdir in sources:
        res = json.loads((bdir / "benchmark.json").read_text())
        spec = json.loads((bdir / "manifest.json").read_text())
        out["benchmarks"][tag] = {"dir": str(bdir), "bench_id": res["bench_id"], "n_loci": res["n_loci"],
                                  "class_counts": res["class_counts"]}
        b = res["bootstrap"]["least_favourable"]
        for m in sorted(b["estimates"]):
            own = slow_methods[tag]
            if own is not None and m not in own:
                continue  # fast methods are reported from the main benchmark
            if own is None and m in ("colocboost", "colocboost_m1", "mvsusie"):
                continue
            lf = res["methods"][m]["least_favourable"]
            ex = res["methods"][m].get("excluded", {})
            med, p95 = _runtime(res, m)
            fl = lf.get("failures", {})
            rows.append({
                "method": m, "sample": tag, "n_loci": res["n_loci"],
                "realistic_auprc_sd": _est(b, m, "realistic", "auprc_sd"),
                "ideal_auprc_sd": _est(b, m, "ideal", "auprc_sd"),
                "realistic_auroc_dw": _est(b, m, "realistic", "auroc_dw"),
                "ideal_auroc_dw": _est(b, m, "ideal", "auroc_dw"),
                "realistic_auprc_sd_excluded": _est(res["bootstrap"].get("excluded", {}), m, "realistic", "auprc_sd")
                if res["bootstrap"].get("excluded") else None,
                "vs_control_realistic_auprc": b["vs_control"].get(m, {}).get("realistic", {}).get("auprc_sd"),
                "op": lf["operating_points"]["used"], "at_op": lf["at_operating_points"],
                "n_failed": fl.get("n_failed"), "max_cell_failure_share": fl.get("max_cell_failure_share"),
                "runtime_median_s": med, "runtime_p95_s": p95,
                "excluded_available": bool(ex)})
        if tag == "main":
            w = {m: (_est(b, m, "realistic", "auprc_sd") or {}).get("est") for m in b["estimates"]
                 if m.startswith("coloc_abf_W") and m.endswith("p12_5e-06")}
            if w:
                best = max(w, key=lambda k: (w[k], -abs(float(k.split("W")[1].split("_")[0]) - 0.15)))
                out["coloc_abf_W_selection"] = {"rule": "maximise validation realistic AUPRC S-vs-D "
                                                "(least_favourable, p12 = 5e-6; DESIGN §4.1)",
                                                "realistic_auprc": w, "selected": best}
        if tag == "mvsusie":
            out["mvsusie_soft"] = mvsusie_soft_sensitivity(bdir)
    out["rows"] = rows
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    def g(x, nd=3):
        if x is None:
            return "NA"
        try:
            x = float(x)
        except (TypeError, ValueError):
            return str(x)
        return f"{x:.{nd}g}" if np.isfinite(x) or np.isinf(x) else "NA"

    L = ["| method | sample (n) | realistic AUPRC S-vs-D [95% CI] | ideal AUPRC S-vs-D [95% CI] | "
         "realistic AUROC D-vs-W [95% CI] | ideal AUROC D-vs-W [95% CI] | failures (max cell share) | "
         "runtime median / p95 (s) |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['method']} | {r['sample']} ({r['n_loci']}) | {_fmt(r['realistic_auprc_sd'])} | "
                 f"{_fmt(r['ideal_auprc_sd'])} | {_fmt(r['realistic_auroc_dw'])} | {_fmt(r['ideal_auroc_dw'])} | "
                 f"{r['n_failed']} ({g(r['max_cell_failure_share'])}) | "
                 f"{g(r['runtime_median_s'])} / {g(r['runtime_p95_s'])} |")
    L += ["", "| method | t_S | t_D | recal a_S | recal a_D | W called S | W called D | recall S | recall D |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        op, at = r["op"], r["at_op"]
        L.append(f"| {r['method']} | {g(op.get('t_S'), 4)} | {g(op.get('t_D'), 4)} | {g(op.get('recal_a_S'))} | "
                 f"{g(op.get('recal_a_D'))} | {g(at.get('W_called_S'))} | {g(at.get('W_called_D'))} | "
                 f"{g(at.get('recall_S'))} | {g(at.get('recall_D'))} |")
    Path(a.out).with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subsample")
    s.add_argument("which", choices=("screen", "m4"))
    s.add_argument("--set", required=True)
    s.add_argument("--out", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--ids-file", required=True)
    b.add_argument("--reuse-from", action="append", help="directory of harness runs; every run covering the ids "
                   "is reused (repeatable)")
    b.add_argument("--exclude", action="append", help="method name not to reuse (repeatable)")
    c = sub.add_parser("collect")
    c.add_argument("--runs-dir", action="append", required=True, help="repeatable; later dirs win per method")
    c.add_argument("--group", action="append", required=True, help="NAME=IDS_FILE (repeatable)")
    c.add_argument("--exclude", action="append")
    c.add_argument("--out", required=True)
    o = sub.add_parser("oppoints")
    o.add_argument("--set", required=True)
    o.add_argument("--pred", action="append", required=True, help="predictions parquet (repeatable)")
    o.add_argument("--methods", nargs="+", required=True)
    o.add_argument("--label", default="")
    o.add_argument("--out", required=True)
    o2 = sub.add_parser("op2", help="OP1 + OP2 per method (DESIGN_P2 §3)")
    o2.add_argument("--set", action="append", required=True, help="locus set (truth alongside; repeatable)")
    o2.add_argument("--runs-dir", action="append", help="harness runs dir (needs --ids-file; repeatable)")
    o2.add_argument("--ids-file", default=None)
    o2.add_argument("--pred", action="append", help="predictions parquet with a method column (repeatable)")
    o2.add_argument("--methods", nargs="*", default=None)
    o2.add_argument("--exclude", action="append")
    o2.add_argument("--setting", default=None, help="fit setting (default: run_benchmark's main setting)")
    o2.add_argument("--d-limit", type=float, default=0.10)
    o2.add_argument("--label", default="")
    o2.add_argument("--out", required=True)
    of = sub.add_parser("opfinal", help="phase-2 OP1 (frozen / refit) + OP2 per method")
    of.add_argument("--frozen-op1", required=True, help="operating_points_final.json of m4_valid_baselines@2")
    of.add_argument("--op2", action="append", required=True, help="json written by `op2` (repeatable)")
    of.add_argument("--refit-op1", nargs="*", default=None, help="methods whose OP1 is taken from the M4 refit")
    of.add_argument("--exclude", action="append")
    of.add_argument("--out", required=True)
    q = sub.add_parser("quickeval")
    q.add_argument("--set", required=True)
    q.add_argument("--ids-file", required=True)
    q.add_argument("--pred", action="append", required=True)
    q.add_argument("--methods", nargs="+", required=True)
    q.add_argument("--label", required=True)
    q.add_argument("--n-boot", type=int, default=2000)
    q.add_argument("--n-jobs", type=int, default=4)
    q.add_argument("--out", required=True)
    m = sub.add_parser("m4summary")
    m.add_argument("--main", required=True)
    m.add_argument("--slow", default=None, help="optional (benchmark not available yet)")
    m.add_argument("--mvsusie", default=None, help="optional (benchmark not available yet)")
    m.add_argument("--out", required=True)
    a, rest = ap.parse_known_args(argv)
    if a.cmd == "bench":
        return cmd_bench(a, rest)
    if rest:
        ap.error(f"unrecognised arguments {rest}")
    if a.cmd == "collect":
        return cmd_collect(a)
    if a.cmd == "quickeval":
        return cmd_quickeval(a)
    if a.cmd == "oppoints":
        return cmd_oppoints(a)
    if a.cmd == "op2":
        return cmd_op2(a)
    if a.cmd == "opfinal":
        return cmd_opfinal(a)
    return cmd_subsample(a) if a.cmd == "subsample" else cmd_m4summary(a)


if __name__ == "__main__":
    sys.exit(main())
