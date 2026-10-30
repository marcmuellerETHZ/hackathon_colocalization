"""On-the-fly training data for Family C (NeuralColoc; DESIGN v4 §5 C "Training").

Loci are drawn from TRAIN-split windows only, over the full v4 grid: mismatch levels L0-L4, L6 (one held-out
mechanism, default L6, excluded by a flag), all pairing types with their overlap grids (P1 r_p in {0, .3, .6}
nested/disjoint; P2 rho in {0, .1, .2, .3}; P3 rho in {0, .3, .5, .8}, both signs; L3/L6 only with zero
overlap), from the §2.4 locus mixture (the explicit training prior; no rejection on the oracle class). Inputs
are what methods see (OLS z, MAFs, N, LD given at that level, rho-hat from the D6(d) estimator, never rho).

Per window visit `step` (seed derived as rng(20261030, "train", step, ...), DESIGN §9), the window's synthetic
populations are materialised once on the GPU and `loci_per_window` loci are drawn from it; a shuffle buffer of
`buffer_windows` windows mixes loci across windows before batching.

Each locus is reduced to a top-M screen (default: per-trait union of marginal |z| and lead-removed conditional
|z| as in Family B's screen, without duplicate collapse; or a user `screen_fn`). A batch is a dict of tensors
on the GPU (B = batch size, M = screen size; padded entries have mask = False and zeros):

  z1, z2, maf1, maf2, maf_ref         [B, M] float32
  R                                   [B, M, M] float32  LD given to methods on the screened SNPs
  mask                                [B, M] bool
  pos                                 [B, M] int64  (bp; 0 = padding)
  n1, n2, n_ref, rho_hat, rho_se      [B] float32  (n_ref = reference size; L0 factor-form: population 20,000)
  level, pairing                      [B] int64  (indices into LEVELS / PAIRINGS)
  y_class                             [B] int64  oracle class index in sim.oracle.CLASSES = (S, U, D, W, G)
  causal1, causal2, shared            [B, M] bool   screened SNP is a causal variant of trait 1 / 2 / both
  proxy1, proxy2                      [B, M] float32  max r² (R_true) of the screened SNP with a causal of trait t
  mu1, mu2, r2, d, rho, p             [B] float32  truth (class-determining |mu|, r², d, true rho, locus size)
  cd_in_screen                        [B] bool   class-determining variants (or r² > 0.95 proxies) in the screen

  from sim.train_api import TrainSampler
  ts = TrainSampler(M=128)                       # holdout=("L6",) by default
  for batch in ts.batches(256, n_batches=100):   # tensors on cuda
      ...
  python -m sim.train_api --bench                # throughput benchmark
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from common.seeds import REPLICATE_SEED, rng
from sim import params
from sim.generator import (CellSpec, WindowContext, assert_no_tf32, draw_candidate, ld_source, observed_stats,
                           rho_hat_for)
from sim.oracle import CLASSES
from sim.windows import split_windows

LEVELS = ("L0", "L1", "L2", "L3", "L4", "L6")
PAIRINGS = ("P1", "P2", "P3")
PAIRING_WEIGHTS = {"P1": 0.5, "P2": 0.25, "P3": 0.25}         # DESIGN §8.6 headline weights
OVERLAPS = {"P1": (0.0, 0.3, 0.6), "P2": (0.0, 0.1, 0.2, 0.3), "P3": (0.0, 0.3, 0.5, 0.8)}   # DESIGN §2.4
POP_LEVELS = ("L3", "L6")
FIELDS_F = ("z1", "z2", "maf1", "maf2", "maf_ref", "proxy1", "proxy2")
FIELDS_B = ("mask", "causal1", "causal2", "shared")
SCAL_F = ("n1", "n2", "n_ref", "rho_hat", "rho_se", "mu1", "mu2", "r2", "d", "rho", "p")
SCAL_I = ("level", "pairing", "y_class")


def draw_cell(g: np.random.Generator, levels) -> CellSpec:
    lvl = levels[int(g.integers(len(levels)))]
    pr = list(PAIRING_WEIGHTS)
    pairing = pr[int(g.choice(len(pr), p=np.array([PAIRING_WEIGHTS[k] for k in pr])))]
    ov = 0.0 if lvl in POP_LEVELS else float(OVERLAPS[pairing][int(g.integers(len(OVERLAPS[pairing])))])
    return CellSpec(f"train|{lvl}|{pairing}|{ov}", lvl, pairing, ov, both_signs=True)


def default_screen(z1: np.ndarray, z2: np.ndarray, rows, M: int) -> np.ndarray:
    """Per-trait union screen (DESIGN §5 B): per trait, M/2 - M/8 by marginal |z| and M/8 by the lead-removed
    conditional |z| (z_j - r_jl z_l)/sqrt(1 - r_jl²); round-robin m1, m2, c1, c2 without duplicates; filled
    from max(|z1|, |z2|). z on the CPU (numpy); rows(idx) -> LD rows (torch). Returns sorted indices (<= M)."""
    p = len(z1)
    if p <= M:
        return np.arange(p)
    kc = max(1, M // 8)
    km = M // 2 - kc
    lists = [np.argsort(-np.abs(z1), kind="stable")[:M], np.argsort(-np.abs(z2), kind="stable")[:M]]
    leads = [int(np.argmax(np.abs(z1))), int(np.argmax(np.abs(z2)))]
    rr = rows(leads).double().cpu().numpy()
    for t, z in enumerate((z1, z2)):
        r = np.clip(rr[t], -0.999999, 0.999999)
        zc = (z - r * z[leads[t]]) / np.sqrt(1 - r * r)
        zc[leads[t]] = 0.0
        lists.append(np.argsort(-np.abs(zc), kind="stable")[:M])
    quota = [km, km, kc, kc]
    chosen: list[int] = []
    seen = set()
    ptr = [0, 0, 0, 0]
    taken = [0, 0, 0, 0]
    while len(chosen) < M and any(taken[i] < quota[i] and ptr[i] < len(lists[i]) for i in range(4)):
        for i in range(4):
            while taken[i] < quota[i] and ptr[i] < len(lists[i]):
                j = int(lists[i][ptr[i]])
                ptr[i] += 1
                if j not in seen:
                    seen.add(j)
                    chosen.append(j)
                    taken[i] += 1
                    break
    if len(chosen) < M:
        for j in np.argsort(-np.maximum(np.abs(z1), np.abs(z2)), kind="stable"):
            j = int(j)
            if j not in seen:
                seen.add(j)
                chosen.append(j)
                if len(chosen) >= M:
                    break
    return np.sort(np.asarray(chosen[:M], dtype=np.int64))


class TrainSampler:
    def __init__(self, M: int = 128, seed: int = REPLICATE_SEED, holdout=("L6",), levels=LEVELS,
                 loci_per_window: int = 64, buffer_windows: int = 4, split: str = "train", device: str = "cuda",
                 screen_fn=None, ltype_weights=None, start_step: int = 0):
        if split != "train":
            raise ValueError("Family C trains on the train split only (DESIGN §5 C)")
        assert_no_tf32()
        self.M, self.seed = int(M), int(seed)
        self.levels = tuple(l for l in levels if l not in set(holdout or ()))
        assert self.levels, "no training levels left"
        self.holdout = tuple(holdout or ())
        self.lpw, self.buffer_windows = int(loci_per_window), int(buffer_windows)
        self.device = torch.device(device)
        self.screen_fn = screen_fn or default_screen
        self.ltype_weights = ltype_weights
        self.wins = split_windows(split)
        self.step = int(start_step)
        self.stats = {"windows": 0, "loci": 0, "none": 0, "t_window": 0.0, "t_loci": 0.0}

    # ------------------------------------------------------------------ one locus
    def _locus(self, ctx: WindowContext, cell: CellSpec, g: np.random.Generator) -> dict | None:
        c = draw_candidate(ctx, cell, g)
        if c is None:
            return None
        ob = observed_stats(ctx, c, g)
        keep = ob["keep"]
        src = ld_source(ctx, c, keep)
        rh = rho_hat_for(c.ps, ob["z1"], ob["z2"], src, g, cell.level)
        dev = self.device
        idx_np = np.asarray(self.screen_fn(ob["z1"], ob["z2"], src.rows, self.M), dtype=np.int64)
        k = len(idx_np)
        M = self.M
        win_idx = keep[idx_np]
        arr = np.zeros((len(FIELDS_F), M), np.float32)
        arr[0, :k] = ob["z1"][idx_np]
        arr[1, :k] = ob["z2"][idx_np]
        arr[2, :k] = np.minimum(c.s1.af, 1 - c.s1.af)[win_idx]
        arr[3, :k] = np.minimum(c.s2.af, 1 - c.s2.af)[win_idx]
        arr[4, :k] = src.maf_ref[idx_np]
        boo = np.zeros((len(FIELDS_B), M), bool)
        boo[0, :k] = True
        boo[1, :k] = np.isin(win_idx, c.arch.c1)
        boo[2, :k] = np.isin(win_idx, c.arch.c2)
        boo[3, :k] = boo[1, :k] & boo[2, :k]
        pos = np.zeros(M, np.int64)
        pos[:k] = ctx.v["pos"].to_numpy()[win_idx]
        out = {}
        Rb = torch.zeros(M, M, device=dev)
        Rb[:k, :k] = src.block(idx_np).float()
        out["R"] = Rb
        # truth on the screen: max r² (R_true = target population LD) with a causal variant of each trait
        target = ctx.pop("SWAP" if cell.level in ("L1swap", "L2swap") else "NFE")
        Zt = target.std
        wi = torch.as_tensor(win_idx, device=dev)
        allc = list(c.arch.c1) + list(c.arch.c2)
        o = c.oracle
        cds = [int(x) for x in (o.cd1, o.cd2) if x is not None and o.cls != "W"]
        cols = allc + cds
        if cols:
            r = ((Zt[:, wi].T @ Zt[:, torch.as_tensor(cols, device=dev)]) / target.N).double().cpu().numpy()
            r2 = r * r
            n1 = len(c.arch.c1)
            if n1:
                arr[5, :k] = r2[:, :n1].max(1)
            if len(c.arch.c2):
                arr[6, :k] = r2[:, n1:len(allc)].max(1)
            cd_ok = bool(all(r2[:, len(allc) + i].max() > 0.95 for i in range(len(cds))))
        else:
            cd_ok = True
        t_arr = torch.as_tensor(arr, device=dev)
        t_boo = torch.as_tensor(boo, device=dev)
        for i, f in enumerate(FIELDS_F):
            out[f] = t_arr[i]
        for i, f in enumerate(FIELDS_B):
            out[f] = t_boo[i]
        out["pos"] = torch.as_tensor(pos, device=dev)
        out.update({"n1": float(c.ps.t1.N), "n2": float(c.ps.t2.N), "n_ref": float(src.n_ref_attr),
                    "rho_hat": float(rh["rho_hat"]) if np.isfinite(rh["rho_hat"]) else 0.0,
                    "rho_se": float(rh["rho_se"]) if np.isfinite(rh["rho_se"]) else 1.0,
                    "mu1": float(o.mu1), "mu2": float(o.mu2),
                    "r2": float(o.r2) if np.isfinite(o.r2) else -1.0, "d": float(o.d) if np.isfinite(o.d) else -1.0,
                    "rho": float(c.ps.rho), "p": float(len(keep)),
                    "level": LEVELS.index(cell.level), "pairing": PAIRINGS.index(cell.pairing),
                    "y_class": CLASSES.index(o.cls), "cd_in_screen": cd_ok})
        return out

    # ------------------------------------------------------------------ one window
    def window_loci(self, step: int) -> list[dict]:
        g = rng(self.seed, "train", step, "window")
        t0 = time.time()
        while True:
            w = self.wins.iloc[int(g.integers(len(self.wins)))]
            ctx = WindowContext(w["chr"], int(w["window_start"]), int(w["window_end"]), split="train",
                                block_ids=tuple(w["block_ids"]), anchor_block=str(w["block_id"]),
                                anchor=int(w["anchor"]), device=str(self.device))
            if ctx.p >= 200:
                break
        for attempt in range(60):          # the GPU is shared: wait for memory instead of failing
            try:
                ctx.pop("NFE")
                break
            except torch.OutOfMemoryError:
                ctx.release()
                time.sleep(10)
        else:
            raise RuntimeError("repeated CUDA OOM while materialising a window")
        self.stats["t_window"] += time.time() - t0
        t1 = time.time()
        out = []
        for k in range(self.lpw):
            gk = rng(self.seed, "train", step, k)
            cell = draw_cell(gk, self.levels)
            if self.ltype_weights is not None:
                cell = CellSpec(cell.cell_id, cell.level, cell.pairing, cell.overlap, both_signs=True,
                                ltype_weights=self.ltype_weights)
            for attempt in range(30):
                try:
                    d = self._locus(ctx, cell, rng(self.seed, "train", step, k, "locus"))
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    time.sleep(10)
            else:
                raise RuntimeError("repeated CUDA OOM")
            if d is None:
                self.stats["none"] += 1
                continue
            out.append(d)
        ctx.release()
        self.stats["t_loci"] += time.time() - t1
        self.stats["windows"] += 1
        self.stats["loci"] += len(out)
        return out

    # ------------------------------------------------------------------ batches
    def batches(self, batch_size: int, n_batches: int | None = None):
        buf: list[dict] = []
        made = 0
        gshuf = rng(self.seed, "train", "shuffle", self.step)
        while n_batches is None or made < n_batches:
            while len(buf) < max(batch_size, self.buffer_windows * self.lpw // 2):
                buf.extend(self.window_loci(self.step))
                self.step += 1
            perm = gshuf.permutation(len(buf))
            take = [buf[i] for i in perm[:batch_size]]
            buf = [buf[i] for i in perm[batch_size:]]
            yield collate(take)
            made += 1


def collate(items: list[dict]) -> dict:
    out = {}
    for f in FIELDS_F + FIELDS_B + ("pos", "R"):
        out[f] = torch.stack([x[f] for x in items])
    dev = items[0]["z1"].device
    for f in SCAL_F:
        out[f] = torch.tensor([x[f] for x in items], dtype=torch.float32, device=dev)
    for f in SCAL_I:
        out[f] = torch.tensor([x[f] for x in items], dtype=torch.long, device=dev)
    out["cd_in_screen"] = torch.tensor([x["cd_in_screen"] for x in items], dtype=torch.bool, device=dev)
    return out


def bench(n_windows: int = 6, lpw: int = 64, M: int = 128, out: Path | None = None) -> dict:
    ts = TrainSampler(M=M, loci_per_window=lpw)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n = 0
    cls = np.zeros(len(CLASSES), int)
    lv = np.zeros(len(LEVELS), int)
    for s in range(n_windows):
        loci = ts.window_loci(s)
        n += len(loci)
        for d in loci:
            cls[d["y_class"]] += 1
            lv[d["level"]] += 1
    torch.cuda.synchronize()
    dt = time.time() - t0
    res = {"windows": n_windows, "loci_per_window": lpw, "M": M, "loci": n, "seconds": dt,
           "loci_per_s": n / dt, "t_window_materialise_s": ts.stats["t_window"], "t_loci_s": ts.stats["t_loci"],
           "loci_per_s_excl_window": n / max(ts.stats["t_loci"], 1e-9),
           "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
           "class_counts": dict(zip(CLASSES, cls.tolist())), "level_counts": dict(zip(LEVELS, lv.tolist())),
           "holdout": list(ts.holdout)}
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return res


def _bench_proc(args):
    i, n_windows, lpw, M = args
    import torch as _t
    ts = TrainSampler(M=M, loci_per_window=lpw)
    t0 = time.time()
    n = 0
    for s in range(n_windows):
        n += len(ts.window_loci(1_000_000 * (i + 1) + s))
    _t.cuda.synchronize()
    return n, time.time() - t0, _t.cuda.max_memory_allocated() / 1e9


def bench_multi(procs: int, n_windows: int, lpw: int, M: int) -> dict:
    """Aggregate throughput of `procs` independent sampler processes sharing the GPU."""
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("spawn").Pool(procs) as pool:
        res = pool.map(_bench_proc, [(i, n_windows, lpw, M) for i in range(procs)])
    wall = time.time() - t0
    n = sum(r[0] for r in res)
    return {"procs": procs, "windows_per_proc": n_windows, "loci_per_window": lpw, "M": M, "loci": n,
            "wall_s_incl_startup": wall, "loci_per_s_aggregate": n / max(r[1] for r in res),
            "per_proc_loci_per_s": [r[0] / r[1] for r in res], "peak_gpu_mem_gb_per_proc": [r[2] for r in res]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--lpw", type=int, default=64)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.bench and a.procs > 1:
        r = bench_multi(a.procs, a.windows, a.lpw, a.M)
        if a.out:
            Path(a.out).write_text(json.dumps(r, indent=1))
        print(json.dumps(r, indent=1))
    elif a.bench:
        print(json.dumps(bench(a.windows, a.lpw, a.M, a.out), indent=1))


if __name__ == "__main__":
    main()
