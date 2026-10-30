"""End-to-end check of data.real.extract.build_test_arm(s) on VALIDATION ID lists (no test data).

  python scripts/real/check_testarm_valid.py --out data/processed/real_testarm_valid_check_v1 --workers 16

1. Writes validation ID lists in the frozen test-list format (data.real.freeze_test_ids.ID_COLS, serialised like
   freeze_test_ids) with the P3-14 sizes (R2 800, R3b all, R4b 600), drawn from the valid rows of
   pairs/<arm>_candidates.tsv with rng(VALID_SEED, "real_testarm_valid_check", arm), file order kept.
2. Builds them with build_test_arms(..., mode="valid") -- the same code path as the test run except the access
   guard and the metadata source (candidates.tsv instead of the verified test metadata).
3. Compares every built locus with real_locus_sets@2 (data/processed/real/locus_sets/<arm>_valid[.<mode>].h5):
   variant ids, positions, all per-variant arrays, X_ref (exact equality, NaN == NaN), every standard and extra
   attribute of the @2 locus (exact; float differences are reported), the set of written ids, and lists the
   attributes the builder adds (expected: donor_disjoint, rho_sens_zero; P3-23).
Writes <out>/check.json; exit code 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common.seeds import VALID_SEED, rng
from data.real.config import OUT
from data.real.extract import REAL_SET_MODES, build_test_arms
from data.real.freeze_test_ids import ID_COLS
from data.real.rho_update import locus_file

SIZES = {"R2": 800, "R3b": None, "R4b": 600}
ARRAYS = ("variant_id", "pos", "z1", "z2", "beta1", "se1", "beta2", "se2", "n1", "n2", "maf1", "maf2", "maf_ref",
          "X_ref")
SCALARS = ("ld_type", "n_ref", "rho_hat", "rho_se", "split", "block_ids", "mismatch_level", "pairing_type", "cell_id",
           "window_chr", "window_start", "window_end")
ADDED = {"donor_disjoint", "rho_sens_zero"}


def write_valid_lists(out: Path, sizes: dict = SIZES) -> dict:
    d = out / "ids"
    d.mkdir(parents=True, exist_ok=True)
    paths = {}
    for arm, k in sizes.items():
        c = pd.read_csv(OUT / "pairs" / f"{arm}_candidates.tsv", sep="\t", usecols=ID_COLS)[ID_COLS]
        c = c[c["split"] == "valid"].reset_index(drop=True)
        tmp = d / f"{arm}_valid_ids_all.tsv"
        c.to_csv(tmp, sep="\t", index=False)
        ids = pd.read_csv(tmp, sep="\t", dtype=str, usecols=ID_COLS)[ID_COLS]
        if k is not None and k < len(ids):
            ids = ids.iloc[np.sort(rng(VALID_SEED, "real_testarm_valid_check", arm).choice(len(ids), k, replace=False))]
        p = d / f"{arm}_valid_ids_sub.tsv"
        ids.to_csv(p, sep="\t", index=False)
        paths[arm] = p
    return paths


def _eq(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return False
        return (np.isnan(fa) and np.isnan(fb)) or fa == fb
    if isinstance(a, (tuple, list, np.ndarray)) or isinstance(b, (tuple, list, np.ndarray)):
        return list(np.atleast_1d(a)) == list(np.atleast_1d(b))
    return a == b


def _arr_eq(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a.astype(str), b.astype(str)))


def compare(key: str, new_path: str, new_ids: list[str], ref_path: Path, req_ids: list[str]) -> dict:
    from eval.io import LocusSetReader, list_loci
    ref_all = set(list_loci(ref_path))
    exp_ids = [i for i in req_ids if i in ref_all]
    res = {"set": key, "new": str(new_path), "ref": str(ref_path), "n_requested": len(req_ids),
           "n_written": len(new_ids), "n_expected_from_ref": len(exp_ids), "ids_equal": new_ids == exp_ids,
           "n_compared": 0, "mismatches": [], "added_attrs": set(), "max_abs_float_diff": 0.0}
    with LocusSetReader(new_path) as rn, LocusSetReader(ref_path) as rr:
        for lid in new_ids:
            if lid not in ref_all:
                res["mismatches"].append(f"{lid}: not in @2")
                continue
            A, B = rn.read(lid), rr.read(lid)
            res["n_compared"] += 1
            for f in ARRAYS:
                if not _arr_eq(getattr(A, f), getattr(B, f)):
                    res["mismatches"].append(f"{lid}: array {f}")
            for f in SCALARS:
                if not _eq(getattr(A, f), getattr(B, f)):
                    res["mismatches"].append(f"{lid}: {f} {getattr(A, f)!r} != {getattr(B, f)!r}")
            ea, eb = A.extra_attrs, B.extra_attrs
            for k, v in eb.items():
                if k not in ea:
                    res["mismatches"].append(f"{lid}: attr {k} missing")
                elif not _eq(ea[k], v):
                    try:
                        res["max_abs_float_diff"] = max(res["max_abs_float_diff"], abs(float(ea[k]) - float(v)))
                    except (TypeError, ValueError):
                        pass
                    res["mismatches"].append(f"{lid}: attr {k} {ea[k]!r} != {v!r}")
            res["added_attrs"] |= set(ea) - set(eb)
    res["added_attrs"] = sorted(res["added_attrs"])
    res["n_mismatches"] = len(res["mismatches"])
    res["mismatches"] = res["mismatches"][:50]
    res["ok"] = bool(res["ids_equal"] and not res["n_mismatches"] and set(res["added_attrs"]) <= ADDED)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT.parent / "real_testarm_valid_check_v1"))  # never inside data/processed/real
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--small", type=int, default=None, help="per-arm size override (quick test)")
    a = ap.parse_args()
    out = Path(a.out)
    sizes = SIZES if a.small is None else {k: a.small for k in SIZES}
    lists = write_valid_lists(out, sizes)
    t0 = time.time()
    built = build_test_arms({}, out / "sets", workers=a.workers, mode="valid", id_lists=lists)
    wall = time.time() - t0
    report = {"wall_s_build": round(wall, 1), "sizes": {k: int(sum(1 for _ in open(p)) - 1) for k, p in lists.items()},
              "sets": {}}
    for key, (arm, mode) in REAL_SET_MODES.items():
        req = pd.read_csv(lists[arm], sep="\t", dtype=str)["locus_id"].tolist()
        report["sets"][key] = compare(key, built[key]["path"], built[key]["ids"], locus_file(arm, "valid", mode), req)
        # P3-23 attributes: every pair of these arms is donor-disjoint
        from eval.io import read_locus_attrs
        at = read_locus_attrs(built[key]["path"])
        report["sets"][key]["donor_disjoint_all"] = bool(at["donor_disjoint"].astype(bool).all()) if len(at) else None
        report["sets"][key]["rho_sens_zero_all0"] = bool((at["rho_sens_zero"] == 0.0).all()) if len(at) else None
    report["ok"] = all(v["ok"] for v in report["sets"].values())
    (out / "check.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: {kk: v[kk] for kk in ("n_requested", "n_written", "n_compared", "ids_equal", "n_mismatches",
                                                 "added_attrs", "ok")} for k, v in report["sets"].items()}, indent=1))
    print("wall_s_build", report["wall_s_build"], "OK" if report["ok"] else "MISMATCH")
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
