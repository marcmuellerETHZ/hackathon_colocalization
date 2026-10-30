"""Rejection-sampling cell builder (DESIGN v4 §2.4, §6.1) writing locus sets in the INTERFACES §2 format.

Algorithm (deterministic given the seed, independent of how cells are grouped into processes):
  * windows (anchor genes of the split) are visited in the order rng(seed, set, "windows").permutation,
    cycling with a fresh permutation per pass;
  * at each visit, every cell that still needs loci draws K candidates from the locus mixture (§2.4),
    candidate k of a cell uses rng(seed, set, cell_id, "cand", k);
  * a candidate is accepted iff its oracle class falls in a bucket of the cell that is not yet full and
    that has accepted < M loci at this visit (spreads loci over windows); accepted candidates are finalised
    with rng(seed, set, cell_id, "fin", k).
Because acceptance depends only on the oracle class, accepted loci follow the mixture conditional on the
bucket (exact rejection sampling); candidate class counts are recorded to report the mixture's class shares.

Output layout (one HDF5 per cell keeps writers independent; the set file links them):
  <out>/<set>.h5                    external links /<locus_id> -> <set>/<cell>.h5:/<locus_id>
  <out>/<set>/<cell>.h5             locus groups (INTERFACES §2) [+ R2/X_ref2 at L0, README D11]
  <out>/<set>/<cell>.truth.parquet  truth (eval.truth schema + extra columns)
  <out>/<set>.truth.parquet         all truth rows
  <out>/<set>.stats.json            per-cell candidate class counts, runtime, windows visited
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS (set before CUDA init)

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from common.seeds import rng
from eval.io import write_locus_set
from eval.truth import write_truth
from sim import params
from sim.generator import CellSpec, WindowContext, assert_no_tf32, draw_candidate, finalize
from sim.windows import split_windows


@dataclass(frozen=True)
class Bucket:
    name: str
    cls: str
    r2_lo: float | None = None
    r2_hi: float | None = None

    def match(self, orc) -> bool:
        if orc.cls != self.cls:
            return False
        if self.r2_lo is not None and not (self.r2_lo <= orc.r2 < self.r2_hi):
            return False
        return True


SCREEN_BUCKETS = (Bucket("S", "S"), Bucket("D_r2lt0.5", "D", 0.0, 0.5), Bucket("D_r2_0.5-0.8", "D", 0.5, 0.8),
                  Bucket("W", "W"))


class FixedPlan:
    """Screening-type plan: a candidate fills the first matching bucket; every bucket has the same target."""

    def __init__(self, buckets, per_bucket: int):
        self.buckets = tuple(buckets)
        self.targets = {b.name: int(per_bucket) for b in self.buckets}

    def assign(self, orc, k: int, filled: dict) -> str | None:
        b = next((b for b in self.buckets if b.match(orc)), None)
        if b is None or filled.get(b.name, 0) >= self.targets[b.name]:
            return None
        return b.name

    def bucket_of(self, orc) -> str | None:
        b = next((b for b in self.buckets if b.match(orc)), None)
        return None if b is None else b.name

    def done(self, filled: dict) -> bool:
        return all(filled.get(k, 0) >= t for k, t in self.targets.items())

    def describe(self) -> dict:
        return {"type": "fixed", "targets": self.targets}


def d_stratum(r2: float, d: float) -> tuple:
    """(r² bin, d bin) of a D/U class-determining pair with the DESIGN v4 §8.6 edges; r² bin 'ge0.95' above."""
    rb = None
    for k, (lo, hi) in enumerate(params.R2_BINS):
        if lo <= r2 < hi:
            rb = k
    db = None
    for k, (lo, hi) in enumerate(params.D_BINS):
        if lo <= d < hi:
            db = k
    return ("ge0.95" if rb is None else rb), db


class GridPlan:
    """Evaluation-grid plan (DESIGN v4 §8.6; README D17): n_S S, n_W W, n_D D stratified uniformly over strata
    as feasible, G and U kept as they arise up to n_G / n_U each (optional buckets).

    Strata: by="r2" (datasets/configs/testset_v1.yaml D_generation_strata, D6(e)): the 5 r² bins of §8.6, quota n_D/5;
    by="r2xd": the 5 r² x 4 d bins, quota n_D/20 (remainders to the lowest strata).
    Staged relaxation, deterministic in the cell's candidate index k: stage 0 (k < k1): a D locus is accepted
    only into its own stratum while that stratum is below quota; stage 1 (k1 <= k < k2): into its r² bin while the
    bin's total is below the bin quota (any d); stage 2 (k >= k2): any D (incl. r² >= 0.95) while the D total is
    below n_D. Bucket keys: D_r<r² bin>_d<d bin> (d bin = "all" with by="r2"; r² bin "ge0.95" above 0.95)."""

    def __init__(self, n_S: int, n_D: int, n_W: int, n_G: int, n_U: int, k1: int, k2: int, by: str = "r2"):
        if by not in ("r2", "r2xd"):
            raise ValueError(by)
        self.n_S, self.n_D, self.n_W, self.n_G, self.n_U, self.by = n_S, n_D, n_W, n_G, n_U, by
        self.k1, self.k2 = int(k1), int(k2)
        dbs = list(range(len(params.D_BINS))) if by == "r2xd" else ["all"]
        strata = [(rb, db) for rb in range(len(params.R2_BINS)) for db in dbs]
        base, rem = divmod(n_D, len(strata))
        self.q = {st: base + (1 if i < rem else 0) for i, st in enumerate(strata)}
        self.qr = {rb: sum(v for (r, _), v in self.q.items() if r == rb) for rb in range(len(params.R2_BINS))}
        self.targets = {"S": n_S, "W": n_W, "D": n_D}

    @staticmethod
    def key(rb, db) -> str:
        return f"D_r{rb}_d{db}"

    def _stratum(self, orc):
        rb, db = d_stratum(orc.r2, orc.d)
        return rb, (db if self.by == "r2xd" else "all")

    def bucket_of(self, orc) -> str | None:
        if orc.cls == "D":
            return self.key(*self._stratum(orc))
        return orc.cls

    def _dsum(self, filled, rb=None):
        return sum(v for k, v in filled.items() if k.startswith("D_") and (rb is None or k.startswith(f"D_r{rb}_")))

    def assign(self, orc, k: int, filled: dict) -> str | None:
        c = orc.cls
        if c in ("S", "W"):
            return c if filled.get(c, 0) < self.targets[c] else None
        if c in ("G", "U"):
            return c if filled.get(c, 0) < (self.n_G if c == "G" else self.n_U) else None
        if c != "D":
            return None
        if self._dsum(filled) >= self.n_D:
            return None
        rb, db = self._stratum(orc)
        key = self.key(rb, db)
        if rb != "ge0.95" and db is not None and filled.get(key, 0) < self.q[(rb, db)]:
            return key
        if k >= self.k1 and rb != "ge0.95" and self._dsum(filled, rb) < self.qr[rb]:
            return key
        if k >= self.k2:
            return key
        return None

    def done(self, filled: dict) -> bool:
        return (filled.get("S", 0) >= self.n_S and filled.get("W", 0) >= self.n_W
                and self._dsum(filled) >= self.n_D)

    def describe(self) -> dict:
        return {"type": "grid", "by": self.by, "n_S": self.n_S, "n_D": self.n_D, "n_W": self.n_W, "n_G": self.n_G,
                "n_U": self.n_U, "k1": self.k1, "k2": self.k2,
                "d_quota": {self.key(*k): v for k, v in self.q.items()},
                "r2_edges": [b[0] for b in params.R2_BINS] + [params.R2_BINS[-1][1]],
                "d_edges": [b[0] for b in params.D_BINS] + ["inf"]}


def cell_id(level: str, pairing: str, overlap: float) -> str:
    param = "r_p" if pairing == "P1" else "rho"
    return f"{level}|{pairing}|{param}={overlap}"


def screening_cells() -> list[CellSpec]:
    """DESIGN §6.1: {L0, L2, L4} x {P1 r_p = 0, P3 rho = 0.5}. P1 nested/disjoint 50:50 (README D12);
    P3: eQTL x eQTL vs lab x lab 50:50, both signs of rho (validation, DESIGN v4 §2.4)."""
    out = []
    for lvl in ("L0", "L2", "L4"):
        out.append(CellSpec(cell_id(lvl, "P1", 0.0), lvl, "P1", 0.0))
        out.append(CellSpec(cell_id(lvl, "P3", 0.5), lvl, "P3", 0.5))
    return out


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", s)


@dataclass
class CellState:
    spec: CellSpec
    plan: object
    filled: dict = field(default_factory=dict)
    n_cand: int = 0
    n_none: int = 0
    class_counts: dict = field(default_factory=dict)
    bucket_counts_all: dict = field(default_factory=dict)
    truth: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    pending_extra: list = field(default_factory=list)
    t_cand: float = 0.0
    t_fin: float = 0.0

    def done(self) -> bool:
        return self.plan.done(self.filled)


def _write_extras(path: Path, extras: list[tuple[str, dict]]):
    with h5py.File(path, "a") as f:
        for lid, ex in extras:
            if "ld2" in ex:
                lt, R2, X2, n2 = ex["ld2"]
                g = f[lid]
                if lt == "dense":
                    g.create_dataset("R2", data=R2)
                else:
                    g.create_dataset("X_ref2", data=X2)
                g.attrs["ld_type2"] = lt
                g.attrs["n_ref2"] = int(n2)


def _snapshot(st: "CellState") -> dict:
    return {"filled": dict(st.filled), "n_cand": st.n_cand, "n_none": st.n_none,
            "class_counts": dict(st.class_counts), "bucket_counts_all": dict(st.bucket_counts_all),
            "n_truth": len(st.truth)}


def _restore(st: "CellState", sn: dict) -> None:
    st.filled, st.n_cand, st.n_none = dict(sn["filled"]), sn["n_cand"], sn["n_none"]
    st.class_counts, st.bucket_counts_all = dict(sn["class_counts"]), dict(sn["bucket_counts_all"])
    del st.truth[sn["n_truth"]:]
    st.pending, st.pending_extra = [], []


def _visit(ctx, states: dict, set_name: str, seed: int, K: int, M: int, ld_share: bool = True,
           key_prefix: tuple = ()) -> None:
    for st in states.values():
        if st.done():
            continue
        acc_here: dict[str, int] = {}
        for _ in range(K):
            k = st.n_cand
            st.n_cand += 1
            ta = time.time()
            cand = draw_candidate(ctx, st.spec, rng(seed, *key_prefix, set_name, st.spec.cell_id, "cand", k),
                                  seed_key=(seed, *key_prefix, set_name, st.spec.cell_id, k))
            st.t_cand += time.time() - ta
            if cand is None:
                st.n_none += 1
                continue
            cls = cand.oracle.cls
            st.class_counts[cls] = st.class_counts.get(cls, 0) + 1
            bname = st.plan.bucket_of(cand.oracle)
            if bname is not None:
                st.bucket_counts_all[bname] = st.bucket_counts_all.get(bname, 0) + 1
            bname = st.plan.assign(cand.oracle, k, st.filled)
            if bname is None or acc_here.get(bname, 0) >= M:
                continue
            tb = time.time()
            lid = f"{set_name}__{_safe(st.spec.cell_id)}__{bname}__{k:07d}"
            loc, tr, ex = finalize(ctx, cand, rng(seed, *key_prefix, set_name, st.spec.cell_id, "fin", k), lid)
            if not ld_share:
                loc.ld_key = None
            tr["bucket"] = bname
            tr["cand_index"] = k
            tr["set"] = set_name
            st.truth.append(tr)
            st.pending.append(loc)
            if ex:
                st.pending_extra.append((lid, ex))
            st.filled[bname] = st.filled.get(bname, 0) + 1
            acc_here[bname] = acc_here.get(bname, 0) + 1
            st.t_fin += time.time() - tb
            del cand


def build_set(set_name: str, cells: list[CellSpec], buckets, per_bucket: int, split: str, seed: int,
              out_dir: Path, K: int = 40, M: int = 2, max_visits: int | None = None, chroms=None,
              device: str = "cuda", log=print, plan=None, ld_share: bool = True, key_prefix: tuple = ()) -> dict:
    """Build the cells of a locus set. `plan` (FixedPlan / GridPlan, shared by all cells) overrides
    (buckets, per_bucket). ld_share: loci of one window with identical LD (same variant set and reference)
    store it once (HDF5 hard link; INTERFACES format unchanged for readers). key_prefix: extra rng keys
    after the seed (e.g. ("shifted",) for the shifted arm: rng(seed, "shifted", set, ...))."""
    if plan is None:
        plan = FixedPlan(buckets, per_bucket)
    assert_no_tf32()
    torch.use_deterministic_algorithms(True, warn_only=True)
    out_dir = Path(out_dir)
    part_dir = out_dir / set_name
    part_dir.mkdir(parents=True, exist_ok=True)
    wins = split_windows(split)
    if chroms is not None:
        wins = wins[wins["chr"].isin(chroms)].reset_index(drop=True)
    states = {c.cell_id: CellState(c, plan) for c in cells}
    for st in states.values():
        p = part_dir / f"{_safe(st.spec.cell_id)}.h5"
        if p.exists():
            p.unlink()
    t0 = time.time()
    visits, skipped = 0, 0
    pass_no = 0
    order = []
    while not all(s.done() for s in states.values()):
        if not order:
            order = list(rng(seed, *key_prefix, set_name, "windows", pass_no).permutation(len(wins)))
            pass_no += 1
        wi = order.pop(0)
        if max_visits is not None and visits >= max_visits:
            break
        w = wins.iloc[wi]
        ctx = WindowContext(w["chr"], int(w["window_start"]), int(w["window_end"]), split=str(w["split"]),
                            block_ids=tuple(w["block_ids"]), anchor_block=str(w["block_id"]), anchor=int(w["anchor"]),
                            device=device)
        if ctx.p < 200:
            skipped += 1
            continue
        visits += 1
        snap = {cid: _snapshot(st) for cid, st in states.items()}
        for attempt in range(1, 31):
            try:
                _visit(ctx, states, set_name, seed, K, M, ld_share, key_prefix)
                break
            except torch.OutOfMemoryError:
                # GPU shared with other agents: restore the cell states (deterministic retry), wait, retry
                for cid, st in states.items():
                    _restore(st, snap[cid])
                ctx.release()
                torch.cuda.empty_cache()
                log(f"CUDA OOM at visit {visits} (attempt {attempt}); retrying in 30 s")
                time.sleep(30)
        else:
            raise RuntimeError("repeated CUDA OOM")
        for st in states.values():   # flush only after the whole visit succeeded
            if st.pending:
                tb = time.time()
                path = part_dir / f"{_safe(st.spec.cell_id)}.h5"
                write_locus_set(path, st.pending, mode="a")
                if st.pending_extra:
                    _write_extras(path, st.pending_extra)
                st.pending, st.pending_extra = [], []
                st.t_fin += time.time() - tb
        ctx.release()
        del ctx
        if visits % 10 == 0:
            log(f"[{time.time()-t0:7.0f}s] visit {visits} " + " | ".join(
                f"{cid}: " + _fill_str(st) for cid, st in states.items()))
    runtime = time.time() - t0
    stats = {"set": set_name, "seed": seed, "key_prefix": list(key_prefix), "split": split, "K": K, "M": M,
             "plan": plan.describe(),
             "windows_available": int(len(wins)), "visits": visits, "skipped_small_windows": skipped,
             "runtime_s": runtime, "world_version": params.WORLD_VERSION, "cells": {}}
    for cid, st in states.items():
        tdf = pd.DataFrame(st.truth)
        if len(tdf):
            write_truth(tdf, part_dir / f"{_safe(cid)}.truth.parquet")
        stats["cells"][cid] = {"filled": st.filled, "n_candidates": st.n_cand, "n_none": st.n_none,
                               "candidate_class_counts": st.class_counts,
                               "candidate_bucket_counts": st.bucket_counts_all,
                               "t_candidates_s": st.t_cand, "t_finalize_write_s": st.t_fin,
                               "complete": st.done()}
    return stats


def _fill_str(st: "CellState") -> str:
    f = st.filled
    d = sum(v for k, v in f.items() if k.startswith("D"))
    return ",".join(f"{k}={f.get(k, 0)}" for k in ("S", "W", "G", "U")) + f",D={d},k={st.n_cand}"


def link_set(set_name: str, out_dir: Path) -> tuple[Path, Path]:
    """Create <set>.h5 with external links to all part files, and the combined truth parquet."""
    out_dir = Path(out_dir)
    part_dir = out_dir / set_name
    master = out_dir / f"{set_name}.h5"
    truths = []
    with h5py.File(master, "w") as f:
        for p in sorted(part_dir.glob("*.h5")):
            with h5py.File(p, "r") as g:
                ids = [k for k in g.keys() if not k.startswith("_")]
            for lid in ids:
                f[lid] = h5py.ExternalLink(f"{set_name}/{p.name}", f"/{lid}")
        f.attrs["set"] = set_name
        f.attrs["layout"] = "external links to per-cell part files (src/sim/cells.py)"
    for p in sorted(part_dir.glob("*.truth.parquet")):
        truths.append(pd.read_parquet(p))
    tdf = pd.concat(truths, ignore_index=True)
    tpath = out_dir / f"{set_name}.truth.parquet"
    write_truth(tdf, tpath)
    return master, tpath


def main():
    ap = argparse.ArgumentParser(description="Build the frozen screening set (DESIGN §6.1)")
    ap.add_argument("--set", default="screen_v1")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--per-bucket", type=int, default=500)
    ap.add_argument("--cells", nargs="*", default=None, help="subset of cell ids (for parallel processes)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "data/processed/sets"))
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--max-visits", type=int, default=None)
    ap.add_argument("--chroms", nargs="*", default=None)
    ap.add_argument("--stats-suffix", default="")
    ap.add_argument("--link-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if a.link_only:
        m, t = link_set(a.set, out)
        print(m, t)
        return
    cells = screening_cells()
    if a.cells:
        cells = [c for c in cells if c.cell_id in set(a.cells)]
        assert cells, "no matching cells"
    stats = build_set(a.set, cells, SCREEN_BUCKETS, a.per_bucket, a.split, a.seed, out, a.K, a.M,
                      a.max_visits, a.chroms, log=lambda s: print(s, flush=True))
    (out / f"{a.set}.stats{a.stats_suffix}.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "cells"}))


if __name__ == "__main__":
    main()
