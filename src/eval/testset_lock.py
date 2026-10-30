"""Per-testset test lock and the generation-key disjointness check (DESIGN_P2 §6, review must-change M8).

Lock. A test-set config (configs/testset_v<N>.yaml) names its own lock in an optional `lock:` block:
    testset_id, access_log, frozen_tag_prefix, frozen_hashes, default_out.
A config without that block is v1 and gets exactly the phase-1 values (results/test_access.log, any `frozen-*`
tag, configs/FROZEN_SHA256, data/processed/testset_v1), so v1 behaviour is unchanged. scripts/eval_test.py reads
the lock of the config it is given (--testset); eval.io.has_test_access accepts a nonce only through the log of
eval.io.ACCESS_LOGS it was started in, at a commit carrying that log's tag prefix. A v2 run therefore needs a
`frozen-v2*` tag, writes only results/test_access_v2.log (the v1 log is never written), and refuses a second v2
run once its own log has a committed `completed` entry, exactly as v1 does with its log.

Disjointness (DESIGN_P2 §6 "No locus or noise draw is shared with v1"). Every random draw of the frozen generator
(src/sim/cells.py) comes from common.seeds.rng(seed, *key_prefix, set_name, ...):
    windows order    (seed, *prefix, set, "windows", pass)
    candidate k      (seed, *prefix, set, cell_id, "cand", k)
    finalisation k   (seed, *prefix, set, cell_id, "fin", k)
for k < n_candidates of the cell (stats.json), prefix () for the main grid and ("shifted",) for the shifted arm.
`generation_keys` enumerates these key tuples from a set's stats; `check_disjoint` compares two sets' key tuples
AND the 64-bit derived seeds (common.seeds.derive = sha256 of the key tuple), plus their locus ids. Inside the v2
test run (eval.testrun, test mode) the reference is the v1 run named in the config (`disjoint_from`); in a dry run
the reference is a validation set (`dry_reference_stats`), which exercises the same code without test data.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

V1_LOCK = {"testset_id": "v1", "access_log": "results/test_access.log", "frozen_tag_prefix": "frozen-",
           "frozen_hashes": "configs/FROZEN_SHA256", "default_out": "data/processed/testset_v1"}
N_WINDOW_PASSES = 1000   # windows-order keys enumerated per set (a run uses a few passes)


@dataclass(frozen=True)
class TestsetLock:
    testset_id: str
    config_rel: str
    access_log: str
    frozen_tag_prefix: str
    frozen_hashes: str
    default_out: str
    seed: int
    set_name: str

    __test__ = False


def lock_for(config: dict, config_rel: str) -> TestsetLock:
    """The lock of a test-set config (v1 defaults when the config has no `lock:` block)."""
    lk = dict(V1_LOCK)
    if config.get("lock"):
        lk = dict(config["lock"])     # a lock block never inherits v1's values (a missing key is an error)
        for k in ("testset_id", "access_log", "frozen_tag_prefix", "frozen_hashes", "default_out"):
            if not lk.get(k):
                raise ValueError(f"{config_rel}: lock.{k} missing")
    return TestsetLock(testset_id=str(lk["testset_id"]), config_rel=str(config_rel),
                       access_log=str(lk["access_log"]), frozen_tag_prefix=str(lk["frozen_tag_prefix"]),
                       frozen_hashes=str(lk["frozen_hashes"]), default_out=str(lk["default_out"]),
                       seed=int(config["seed"]), set_name=f"testset_v{config.get('version', 1)}")


def lock_problems(lock: TestsetLock) -> list[str]:
    """Static checks of a lock against eval.io.ACCESS_LOGS (the logs the data guard honours)."""
    from eval.io import ACCESS_LOGS
    out = []
    reg = dict(ACCESS_LOGS)
    if lock.access_log not in reg:
        out.append(f"access log {lock.access_log} is not registered in eval.io.ACCESS_LOGS")
    elif reg[lock.access_log] != lock.frozen_tag_prefix:
        out.append(f"access log {lock.access_log}: tag prefix {lock.frozen_tag_prefix!r} differs from eval.io's "
                   f"{reg[lock.access_log]!r}")
    if lock.testset_id != "v1" and lock.frozen_tag_prefix == "frozen-":
        out.append(f"testset {lock.testset_id} must have its own frozen tag prefix (not the v1 'frozen-')")
    if lock.testset_id != "v1" and lock.access_log == V1_LOCK["access_log"]:
        out.append(f"testset {lock.testset_id} must not use the v1 access log")
    return out


# --------------------------------------------------------------------------------------------- disjointness
def read_stats(paths) -> dict:
    """Merge generate()'s stats files of one set (segments .seg<n> and the shifted arm are separate sets):
    {"seed", "set", "key_prefix", "cells": {cell: n_candidates}}. All files must agree on seed / set / prefix."""
    files = sorted({str(p) for x in ([paths] if isinstance(paths, (str, Path)) else paths)
                    for p in (glob.glob(str(x)) or [x])})
    if not files:
        raise FileNotFoundError(f"no stats files: {paths}")
    seed = set_name = prefix = None
    cells: dict[str, int] = {}
    for f in files:
        d = json.loads(Path(f).read_text())
        ws = d.get("workers") or []
        s = {int(d["seed"])} | {int(w["seed"]) for w in ws if "seed" in w}
        sets = {w["set"] for w in ws if "set" in w}
        pfx = {tuple(w.get("key_prefix", ())) for w in ws}
        if len(s) != 1 or len(sets) != 1 or len(pfx) != 1:
            raise ValueError(f"{f}: ambiguous seed/set/key_prefix {s} {sets} {pfx}")
        if seed is not None and (seed, set_name, prefix) != (s.pop(), sets.pop(), pfx.pop()):
            raise ValueError(f"{f}: stats files of different sets mixed")
        if seed is None:
            seed, set_name, prefix = s.pop(), sets.pop(), pfx.pop()
        for c, v in d["cells"].items():
            cells[c] = max(cells.get(c, 0), int(v["n_candidates"]))
    return {"seed": seed, "set": set_name, "key_prefix": list(prefix), "cells": cells, "files": files}


def generation_keys(st: dict):
    """Every rng key tuple (seed, *prefix, set, ...) the generator may have used for this set (superset: fin keys
    for every candidate index, N_WINDOW_PASSES window passes)."""
    base = (int(st["seed"]), *st["key_prefix"], st["set"])
    for p in range(N_WINDOW_PASSES):
        yield base + ("windows", p)
    for cell, n in sorted(st["cells"].items()):
        for k in range(int(n)):
            yield base + (cell, "cand", k)
            yield base + (cell, "fin", k)


def derived_seeds(keys) -> set[int]:
    from common.seeds import derive
    return {derive(k[0], *k[1:]) for k in keys}


def check_disjoint(new: dict, ref: dict, new_ids=None, ref_ids=None) -> dict:
    """Overlap of key tuples, of derived 64-bit seeds and (if given) of locus ids between two sets' stats."""
    kn, kr = set(generation_keys(new)), set(generation_keys(ref))
    dn, dr = derived_seeds(kn), derived_seeds(kr)
    out = {"new": {"seed": new["seed"], "set": new["set"], "key_prefix": new["key_prefix"], "n_keys": len(kn),
                   "n_cells": len(new["cells"])},
           "ref": {"seed": ref["seed"], "set": ref["set"], "key_prefix": ref["key_prefix"], "n_keys": len(kr),
                   "n_cells": len(ref["cells"])},
           "key_overlap": len(kn & kr), "derived_seed_overlap": len(dn & dr),
           "derived_seed_collisions_within_new": len(kn) - len(dn)}
    if new_ids is not None and ref_ids is not None:
        out["locus_id_overlap"] = len(set(map(str, new_ids)) & set(map(str, ref_ids)))
    out["ok"] = out["key_overlap"] == 0 and out["derived_seed_overlap"] == 0 and out.get("locus_id_overlap", 0) == 0
    return out
