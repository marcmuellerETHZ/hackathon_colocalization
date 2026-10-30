#!/usr/bin/env python
"""Build the prediction folders of results/ (test_v1/, truth/, phase2_validation/) of this release from the original project tree (run once, by the release agent).

Usage: python scripts/release_build/build_predictions.py <source_project_root> <release_root>

What it does (documented in results/README.md):
  * test v1 bundle (results/eval_test/test_frozen-v1_ab94c5a1): every runs/*/{predictions.parquet, manifest.json,
    metrics.json}, runs/LEDGER.tsv, benchmarks/sim (benchmark.json, summary.md, c4.json, manifest.json,
    operating_points.json, expected_ids.txt, ids_subsample40.txt, subsample_8685f328/), the top-level JSONs and
    results/test_access.log. The harness checkpoints (checkpoints/*.rows.jsonl, 152 MB, duplicates of the
    predictions), logs/ and state/ are not copied.
  * real-data arms (R2/R3b/R4b; derived from FinnGen R12 and the eQTL Catalogue): every column is kept except
    raw summary-statistic or LD values found in the JSON `extra` column, which are removed:
      - strength_only: extra.lead_abs_z1, extra.lead_abs_z2 (lead |z| of each trait = raw summary statistics);
      - coloc_susie / coloc_susie_mm: extra.t1.purity_min, extra.t2.purity_min (min |LD r| within each credible
        set = LD values);
      - any extra.traceback (none present; defensive).
    The stripped files therefore no longer match `predictions_sha256` in their manifest.json; the new sha256 is
    recorded in results/test_v1/REAL_ARM_STRIPPING.tsv.
  * truth tables of the simulated test set and of the M4 validation subsample, reduced to the columns the
    metrics read.
  * phase-2 validation predictions (coloc_susie_mm M4, B-mm, phase-1 B on M4, famC C2), each < 50 MB.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

SRC = Path(sys.argv[1]).resolve()
DST = Path(sys.argv[2]).resolve()
BUNDLE = "results/eval_test/test_frozen-v1_ab94c5a1"
TRUTH_COLS = ["locus_id", "oracle_class", "mu1", "mu2", "r2", "d", "rho", "block_id", "cell_id", "split",
              "mismatch_level", "pairing_type", "overlap", "control"]
MAX_BYTES = 50 * 1024 * 1024


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cp(src: Path, dst: Path) -> None:
    if src.stat().st_size >= MAX_BYTES:
        raise SystemExit(f"{src} is >= 50 MB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def strip_extra(method: str, extra: str) -> tuple[str, list[str]]:
    d = json.loads(extra)
    removed = []
    if method == "strength_only":
        for k in ("lead_abs_z1", "lead_abs_z2"):
            if k in d:
                d.pop(k)
                removed.append(f"extra.{k}")
    if method in ("coloc_susie", "coloc_susie_mm"):
        for t in ("t1", "t2"):
            if isinstance(d.get(t), dict) and "purity_min" in d[t]:
                d[t].pop("purity_min")
                removed.append(f"extra.{t}.purity_min")
    if "traceback" in d:
        d.pop("traceback")
        removed.append("extra.traceback")
    return json.dumps(d, allow_nan=False), removed


def main() -> None:
    b = SRC / BUNDLE
    out = DST / "results" / "test_v1"
    # ---------------------------------------------------------------- runs
    strip_rows = ["run\tmethod\tremoved_keys\tn_rows_changed\toriginal_sha256\treleased_sha256"]
    for d in sorted((b / "runs").iterdir()):
        if d.is_file():
            cp(d, out / "runs" / d.name)
            continue
        for f in ("manifest.json", "metrics.json"):
            if (d / f).exists():
                cp(d / f, out / "runs" / d.name / f)
        src_p = d / "predictions.parquet"
        dst_p = out / "runs" / d.name / "predictions.parquet"
        if "__real_" in d.name:
            p = pd.read_parquet(src_p)
            method = d.name.split("__")[1]
            removed_all, n_changed, new = set(), 0, []
            for e in p["extra"]:
                if e is None:
                    new.append(e)
                    continue
                s, rem = strip_extra(method, e)
                removed_all.update(rem)
                n_changed += bool(rem)
                new.append(s)
            p["extra"] = new
            # no top-level raw-statistic columns exist in the prediction schema; assert it
            bad = [c for c in p.columns if c.lower() in ("z", "z1", "z2", "beta", "beta1", "beta2", "se", "se1", "se2",
                                                         "maf", "maf1", "maf2", "af", "r", "ld")]
            assert not bad, bad
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            p.to_parquet(dst_p, index=False)
            strip_rows.append(f"{d.name}\t{method}\t{','.join(sorted(removed_all)) or '-'}\t{n_changed}\t"
                              f"{sha256(src_p)}\t{sha256(dst_p)}")
        else:
            cp(src_p, dst_p)
    (out / "REAL_ARM_STRIPPING.tsv").write_text("\n".join(strip_rows) + "\n")
    # ---------------------------------------------------------------- benchmarks/sim
    sim = b / "benchmarks" / "sim"
    for f in ("benchmark.json", "summary.md", "c4.json", "manifest.json", "operating_points.json",
              "expected_ids.txt", "ids_subsample40.txt"):
        cp(sim / f, out / "benchmarks" / "sim" / f)
    for f in sorted((sim / "subsample_8685f328").iterdir()):
        cp(f, out / "benchmarks" / "sim" / "subsample_8685f328" / f.name)
    for f in ("claims.json", "descriptive.json", "real_arms.json", "runtime.json", "operating_points_used.json",
              "bundle.json"):
        cp(b / f, out / f)
    cp(SRC / "results" / "test_access.log", out / "test_access.log")
    # ---------------------------------------------------------------- truth
    sets = SRC / "data/processed/testset_v1/test_frozen-v1_ab94c5a1/sets"
    for name in ("testset_v1", "testset_v1_shifted"):
        t = pd.read_parquet(sets / f"{name}.truth.parquet")[TRUTH_COLS]
        dst = DST / "results" / "truth" / f"{name}.truth.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        t.to_parquet(dst, index=False)
    # ---------------------------------------------------------------- phase-2 validation (M4 subsample)
    p2 = DST / "results" / "phase2_validation"
    runs = SRC / "results/baselines_m4/runs"
    for r in ("20261027T192144Z_coloc_susie_mm_valid_grid_v1_977229", "20261027T192145Z_coloc_susie_mm_valid_grid_v1_9ee9eb"):
        for f in ("predictions.parquet", "manifest.json"):
            cp(runs / r / f, p2 / "coloc_susie_mm_M4" / r / f)
    v3 = SRC / "results/baselines_m4/m4_valid_v3"
    for f in ("operating_points_p2.json", "operating_points_p2.md", "m4summary.json", "m4summary.md",
              "op2_m4_main.json", "op2_m4_slow.json"):
        cp(v3 / f, p2 / "m4_valid_baselines_v3" / f)
    for part in ("main", "slow"):
        for f in sorted((v3 / part).iterdir()):
            cp(f, p2 / "m4_valid_baselines_v3" / part / f.name)
    cp(SRC / "results/baselines_m4/m4_subsample_v1/main.txt", p2 / "m4_subsample_v1_main_ids.txt")
    # validation truth of the M4 main subsample (12,170 loci, validation split), metric columns only
    ids = set((p2 / "m4_subsample_v1_main_ids.txt").read_text().split())
    tv = pd.read_parquet(SRC / "data/processed/sets/valid_grid_v1.truth.parquet")
    tv = tv[tv["locus_id"].astype(str).isin(ids)][TRUTH_COLS]
    assert len(tv) == len(ids) and set(tv["split"]) == {"valid"}
    tv.to_parquet(DST / "results" / "truth" / "valid_grid_v1_M4main.truth.parquet", index=False)
    for name, rel in (("famB_mm_M4", "results/famB/p9_m4_valid/Bmm_M50"), ("famB_M4", "results/famB/p9_m4_valid/B1_M50")):
        for f in ("predictions.parquet", "runs.json"):
            cp(SRC / rel / f, p2 / name / f)
    c2 = SRC / "results/famC/c2_v1"
    cp(c2 / "c2.json", p2 / "famC_C2" / "c2.json")
    for sub in ("net_m4", "nold_m4", "floor_m4", "net_trainregion"):
        for f in ("predictions.parquet", "runs.json"):
            cp(c2 / sub / f, p2 / "famC_C2" / sub / f)
    print("done")


if __name__ == "__main__":
    main()
