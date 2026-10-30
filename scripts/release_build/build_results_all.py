#!/usr/bin/env python
"""Copy the remaining results of the original project into results/ of this release (run once, by the release agent).

Usage: python scripts/release_build/build_results_all.py <source_project_root> <release_root>

Run after build_predictions.py. Rules (documented in results/README.md and results/EXCLUDED.tsv):
  * every file under <source>/results/ is copied to results/<same relative path>, except:
      - files already in this release: same sha256 as a file under results/ (test_v1/, phase2_validation/,
        models/famC/, truth/), or a real-arm prediction whose original sha256 is listed in
        results/test_v1/REAL_ARM_STRIPPING.tsv; a content match counts only inside the folders build_predictions.py
        released from (RELEASED_FROM) or for files >= 4 KB, so small coincidentally identical files stay in place;
      - results/eval_test/*/checkpoints/ and results/eval_test/*/state/ (row-by-row duplicates of runs/*/predictions
        and pipeline state);
      - files >= 50 MB;
      - harness bookkeeping: .gitignore files inside run folders, empty *.lock files.
    Each skipped item is listed in results/EXCLUDED.tsv with its size and the reason.
  * real-data arms (any path containing "real": R2, R3b, R4b, derived from FinnGen R12 and the eQTL Catalogue):
    predictions.parquet and checkpoint *.rows.jsonl files are stripped with strip_extra() of build_predictions.py
    (strength_only lead |z|; coloc_susie / coloc_susie_mm credible-set purity, i.e. LD values; any traceback).
    After stripping, every key of every `extra` JSON is checked against a list of raw-statistic names and the build
    fails if one is left. Old and new sha256 go to results/REAL_ARM_STRIPPING_ALL.tsv.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_predictions import strip_extra  # noqa: E402  (same stripping rules as the test_v1 release)

SRC = Path(sys.argv[1]).resolve()
DST = Path(sys.argv[2]).resolve()
MAX_BYTES = 50 * 1024 * 1024
# source folders that build_predictions.py already released (into test_v1/, phase2_validation/, models/famC/)
RELEASED_FROM = ("eval_test/test_frozen-v1_ab94c5a1/", "test_access.log",
                 "baselines_m4/runs/20261027T192144Z_coloc_susie_mm_valid_grid_v1_977229/",
                 "baselines_m4/runs/20261027T192145Z_coloc_susie_mm_valid_grid_v1_9ee9eb/",
                 "baselines_m4/m4_valid_v3/", "baselines_m4/m4_subsample_v1/main.txt",
                 "famB/p9_m4_valid/Bmm_M50/", "famB/p9_m4_valid/B1_M50/", "famC/c2_v1/",
                 "famC/p9_main_600k/", "famC/p9_nold_600k/", "famC/p9_floor_600k/")
FORBIDDEN = {"z", "z1", "z2", "zscore", "beta", "beta1", "beta2", "se", "se1", "se2", "af", "af1", "af2", "maf",
             "maf1", "maf2", "eaf", "lead_abs_z1", "lead_abs_z2", "purity_min", "purity", "ld", "r", "r2", "R",
             "traceback"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def keys(d, pre=""):
    out = set()
    if isinstance(d, dict):
        for k, v in d.items():
            out.add(k)
            out |= keys(v)
    elif isinstance(d, list):
        for v in d:
            out |= keys(v)
    return out


def check_clean(extra: str | None, where: str) -> None:
    if extra is None:
        return
    bad = keys(json.loads(extra)) & FORBIDDEN
    if bad:
        raise SystemExit(f"raw-statistic key(s) {sorted(bad)} left in {where}")


def is_real(rel: str) -> bool:
    return "real" in rel.lower()


def main() -> None:
    res_src = SRC / "results"
    res_dst = DST / "results"
    # ------------------------------------------------ what this release already holds
    have: dict[str, str] = {}
    for p in sorted(res_dst.rglob("*")):
        if p.is_file():
            have.setdefault(sha256(p), str(p.relative_to(DST)))
    strip_tsv = res_dst / "test_v1" / "REAL_ARM_STRIPPING.tsv"
    for line in strip_tsv.read_text().splitlines()[1:]:
        run, _m, _k, _n, orig, _new = line.split("\t")
        have.setdefault(orig, f"results/test_v1/runs/{run}/predictions.parquet (stripped)")

    excluded = ["source_path\tsize_bytes\treason"]
    stripping = ["source_path\tmethod\tremoved_keys\tn_rows_changed\toriginal_sha256\treleased_sha256"]
    n_copied = bytes_copied = 0

    # whole directories skipped in results/eval_test
    skip_dirs = []
    for run in sorted((res_src / "eval_test").iterdir()):
        for sub in ("checkpoints", "state"):
            d = run / sub
            if d.is_dir():
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                skip_dirs.append(d)
                excluded.append(f"results/{d.relative_to(res_src)}/\t{size}\tharness {sub} of the test run; "
                                f"{'row-by-row duplicates of runs/*/predictions.parquet' if sub == 'checkpoints' else 'pipeline state files'} "
                                f"(released as results/test_v1/runs/)")

    for p in sorted(res_src.rglob("*")):
        if not p.is_file() or any(d in p.parents for d in skip_dirs):
            continue
        rel = str(p.relative_to(res_src))
        size = p.stat().st_size
        srel = f"results/{rel}"
        if p.name == ".gitignore":
            excluded.append(f"{srel}\t{size}\tharness .gitignore of a run folder (would hide its predictions from git)")
            continue
        if p.suffix == ".lock" and size == 0:
            excluded.append(f"{srel}\t{size}\tempty harness lock file")
            continue
        if size >= MAX_BYTES:
            excluded.append(f"{srel}\t{size}\tfile >= 50 MB (GitHub limit; no LFS)")
            continue
        h = sha256(p)
        # a file counts as already released if its content matches a released file and it is either one of the
        # folders build_predictions.py released from, or large enough (>= 4 KB) not to be a coincidental match
        if h in have and (rel.startswith(RELEASED_FROM) or size >= 4096):
            excluded.append(f"{srel}\t{size}\tduplicate: already released as {have[h]}")
            continue
        out = res_dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if is_real(rel) and p.name == "predictions.parquet":
            df = pd.read_parquet(p)
            method = p.parent.name.split("__")[1]
            removed, n_changed, new = set(), 0, []
            for e in df["extra"]:
                if e is None:
                    new.append(e)
                    continue
                s, rem = strip_extra(method, e)
                check_clean(s, srel)
                removed.update(rem)
                n_changed += bool(rem)
                new.append(s)
            df["extra"] = new
            bad = [c for c in df.columns if c in FORBIDDEN or c.lower() in FORBIDDEN]
            assert not bad, (srel, bad)
            df.to_parquet(out, index=False)
            stripping.append(f"{srel}\t{method}\t{','.join(sorted(removed)) or '-'}\t{n_changed}\t{h}\t{sha256(out)}")
        elif is_real(rel) and p.name.endswith(".rows.jsonl"):
            method = p.name.split("__")[1]
            removed, n_changed, lines = set(), 0, []
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("extra") is not None:
                    s, rem = strip_extra(row.get("method", method), row["extra"])
                    check_clean(s, srel)
                    removed.update(rem)
                    n_changed += bool(rem)
                    row["extra"] = s
                bad = set(row) & FORBIDDEN
                assert not bad, (srel, bad)
                lines.append(json.dumps(row, allow_nan=True))
            out.write_text("\n".join(lines) + ("\n" if lines else ""))
            stripping.append(f"{srel}\t{method}\t{','.join(sorted(removed)) or '-'}\t{n_changed}\t{h}\t{sha256(out)}")
        else:
            shutil.copy2(p, out)
        n_copied += 1
        bytes_copied += out.stat().st_size

    (res_dst / "EXCLUDED.tsv").write_text("\n".join(excluded) + "\n")
    (res_dst / "REAL_ARM_STRIPPING_ALL.tsv").write_text("\n".join(stripping) + "\n")
    print(f"copied {n_copied} files, {bytes_copied / 1e6:.1f} MB; excluded {len(excluded) - 1}; "
          f"stripped {len(stripping) - 1} real-arm files")


if __name__ == "__main__":
    main()
